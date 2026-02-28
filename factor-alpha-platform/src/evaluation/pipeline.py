"""
Alpha Evaluation Pipeline — local scoring for alpha research.

Evaluates alpha expressions using the FastExpression engine + vectorized
simulation engine. Computes metrics (Sharpe, fitness, turnover, max drawdown).
Integrates with the alpha database for persistence and LLM learning.

Usage:
    from src.evaluation.pipeline import AlphaEvaluator
    evaluator = AlphaEvaluator.from_synthetic(n_stocks=200, n_days=500)
    result = evaluator.evaluate("rank(ts_delta(divide(close, volume), 120))")
    print(result.sharpe, result.fitness)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.alpha_database import AlphaDatabase
from src.data.context_research import InMemoryDataContext
from src.data.synthetic import SyntheticDataGenerator
from src.operators.fastexpression import FastExpressionEngine, create_engine_from_context
from src.simulation.vectorized_sim import simulate_vectorized, VectorizedSimResult
from src.agent.gp_engine import _build_classifications

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluating an alpha expression."""
    expression: str
    success: bool
    error: str | None = None

    # Core metrics
    sharpe: float | None = None
    fitness: float | None = None
    turnover: float | None = None
    max_drawdown: float | None = None
    returns_ann: float | None = None
    profit_dollar: float | None = None
    margin_bps: float | None = None

    # Detailed
    sim_result: VectorizedSimResult | None = None
    alpha_df: pd.DataFrame | None = dc_field(default=None, repr=False)

    # Quality gates
    passed_checks: int = 0

    def to_dict(self) -> dict:
        return {
            "expression": self.expression,
            "success": self.success,
            "error": self.error,
            "sharpe": self.sharpe,
            "fitness": self.fitness,
            "turnover": self.turnover,
            "max_drawdown": self.max_drawdown,
            "returns_ann": self.returns_ann,
            "profit_dollar": self.profit_dollar,
            "margin_bps": self.margin_bps,
            "passed_checks": self.passed_checks,
        }

    def compute_quality_checks(self, min_sharpe: float = 1.25,
                                min_fitness: float = 1.0) -> int:
        """Run quality gate checks and return count of passed checks."""
        if not self.success or self.sharpe is None or self.fitness is None:
            self.passed_checks = 0
            return 0
        checks = 0
        if self.sharpe >= min_sharpe:
            checks += 1
        if self.sharpe >= 0.5:
            checks += 1
        if self.fitness >= min_fitness:
            checks += 1
        if self.fitness >= 0.5:
            checks += 1
        if self.turnover is not None and 0.01 < self.turnover < 0.7:
            checks += 1
        if self.max_drawdown is not None and self.max_drawdown > -0.3:
            checks += 1
        if self.returns_ann is not None and self.returns_ann > 0:
            checks += 1
        if self.margin_bps is not None and abs(self.margin_bps) > 0.1:
            checks += 1
        self.passed_checks = checks
        return checks


class AlphaEvaluator:
    """Evaluate alpha expressions locally using synthetic or real data."""

    def __init__(
        self,
        engine: FastExpressionEngine,
        ctx: InMemoryDataContext,
        db: AlphaDatabase | None = None,
        sim_settings: dict | None = None,
    ):
        self.engine = engine
        self.ctx = ctx
        self.db = db
        self.sim_settings = sim_settings or {
            "booksize": 10_000_000,
            "decay": 0,
            "delay": 1,
            "neutralization": "subindustry",
            "max_stock_weight": 0.01,
            "fees_bps": 0.0,
        }

        # Pre-build returns and close matrices (use _price_matrices for full DF)
        self._returns_df = ctx._price_matrices.get("returns")
        self._close_df = ctx._price_matrices.get("close")
        self._open_df = ctx._price_matrices.get("open")

        # Build forward returns: tomorrow's close-to-close return (fallback)
        if self._close_df is not None:
            self._forward_returns = self._close_df.pct_change().shift(-1)
        elif self._returns_df is not None:
            self._forward_returns = self._returns_df.shift(-1)
        else:
            self._forward_returns = None

        # Build GICS classifications dict
        self._classifications = _build_classifications(ctx._classifications)

    @classmethod
    def from_synthetic(
        cls,
        n_stocks: int = 200,
        n_days: int = 500,
        seed: int = 42,
        db_path: str | None = None,
        **kwargs,
    ) -> "AlphaEvaluator":
        """Create an evaluator backed by synthetic data."""
        ds = SyntheticDataGenerator().generate(
            n_stocks=n_stocks,
            n_days=n_days,
            seed=seed,
            start_date="2020-01-02",
        )
        ctx = InMemoryDataContext(ds)
        engine = create_engine_from_context(ctx)
        db = AlphaDatabase(db_path) if db_path else AlphaDatabase()
        return cls(engine=engine, ctx=ctx, db=db, **kwargs)

    def evaluate(
        self,
        expression: str,
        params: dict | None = None,
        run_id: int | None = None,
        trial_index: int | None = None,
        reasoning: str = "",
        store: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate a single alpha expression.

        Steps:
        1. Parse and compute alpha DataFrame via FastExpression engine
        2. Run vectorized simulation to get PnL and metrics
        3. Store in database (if enabled)
        """
        result = EvaluationResult(expression=expression, success=False)

        # Step 1: Parse and compute alpha
        try:
            alpha_df = self.engine.evaluate(expression)
        except Exception as e:
            result.error = f"Parse/compute error: {e}"
            logger.warning(f"Failed to evaluate '{expression}': {e}")
            if store and self.db:
                self._store_result(result, params, run_id, trial_index, reasoning)
            return result

        result.alpha_df = alpha_df

        if self._forward_returns is None:
            result.error = "No returns data available for simulation"
            if store and self.db:
                self._store_result(result, params, run_id, trial_index, reasoning)
            return result

        # Step 2: Simulate
        try:
            merge_params = dict(self.sim_settings)
            if params:
                merge_params.update(params)

            sim = simulate_vectorized(
                alpha_df=alpha_df,
                returns_df=self._forward_returns,
                close_df=self._close_df,
                open_df=self._open_df,
                classifications=self._classifications,
                booksize=merge_params.get("booksize", 10_000_000),
                max_stock_weight=merge_params.get("max_stock_weight", 0.01),
                decay=merge_params.get("decay", 0),
                delay=merge_params.get("delay", 1),
                neutralization=merge_params.get("neutralization", "subindustry"),
                fees_bps=merge_params.get("fees_bps", 0.0),
            )

            result.sim_result = sim
            result.sharpe = sim.sharpe
            result.fitness = sim.fitness
            result.turnover = sim.turnover
            result.max_drawdown = sim.max_drawdown
            result.returns_ann = sim.returns_ann
            result.margin_bps = sim.margin_bps
            result.profit_dollar = sim.total_pnl
            result.success = True
            result.compute_quality_checks()

        except Exception as e:
            result.error = f"Simulation error: {e}"
            logger.warning(f"Simulation failed for '{expression}': {e}")

        # Step 3: Store
        if store and self.db:
            self._store_result(result, params, run_id, trial_index, reasoning)

        return result

    def batch_evaluate(
        self,
        expressions: List[str],
        params: dict | None = None,
        run_id: int | None = None,
    ) -> List[EvaluationResult]:
        """Evaluate multiple expressions in batch."""
        results = []
        for i, expr in enumerate(expressions):
            result = self.evaluate(
                expr, params=params, run_id=run_id, trial_index=i + 1
            )
            results.append(result)
            logger.info(
                f"[{i + 1}/{len(expressions)}] "
                f"{'✓' if result.success else '✗'} "
                f"Sharpe={result.sharpe or 0:.3f} Fitness={result.fitness or 0:.3f} "
                f"| {expr[:60]}..."
            )
        return results

    def _store_result(self, result: EvaluationResult, params: dict | None,
                      run_id: int | None, trial_index: int | None,
                      reasoning: str) -> None:
        """Persist to database."""
        try:
            alpha_id = self.db.insert_alpha(
                expression=result.expression,
                params=params,
                reasoning=reasoning,
                run_id=run_id,
                trial_index=trial_index,
            )
            if result.success:
                self.db.update_alpha_status(alpha_id, "evaluated")
            else:
                self.db.update_alpha_status(alpha_id, "failed")

            self.db.insert_evaluation(
                alpha_id=alpha_id,
                source="local",
                sharpe=result.sharpe,
                fitness=result.fitness,
                turnover=result.turnover,
                max_drawdown=result.max_drawdown,
                returns_ann=result.returns_ann,
                profit_dollar=result.profit_dollar,
                margin_bps=result.margin_bps,
                passed_checks=result.passed_checks,
                error=result.error,
                metrics=result.to_dict(),
            )
        except Exception as e:
            logger.error(f"Failed to store evaluation result: {e}")

    def get_history(self, limit: int = 20) -> List[Dict]:
        """Get recent history in LLM-compatible format."""
        if self.db:
            return self.db.get_history_for_prompt(limit=limit)
        return []

    def get_stats(self) -> Dict:
        """Get aggregate stats."""
        if self.db:
            return self.db.get_stats()
        return {}
