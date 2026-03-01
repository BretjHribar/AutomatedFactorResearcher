"""
Genetic Programming Alpha Engine — DEAP-based evolutionary search.

Breeds alpha expressions via crossover and mutation of typed expression
trees, evaluating fitness through the vectorized simulation engine.

Replaces the original AWSgenProgWorker.py with a clean, configurable,
self-contained implementation that uses our vectorized operators and
simulation pipeline.

Usage:
    gp_engine = GPAlphaEngine.from_synthetic(n_stocks=200, n_days=500)
    results = gp_engine.run(n_generations=100, population_size=500)
"""

from __future__ import annotations

import logging
import math
import operator
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from deap import algorithms, base, creator, gp, tools

from src.operators import vectorized as ops
from src.data.alpha_database import AlphaDatabase
from src.data.context_research import InMemoryDataContext
from src.data.synthetic import SyntheticDataGenerator
from src.simulation.vectorized_sim import simulate_vectorized, VectorizedSimResult

logger = logging.getLogger(__name__)


def _build_classifications(raw_cls: dict) -> dict | None:
    """Build GICS classification Series dict from raw classifications.

    Args:
        raw_cls: Dict of {ticker: {"sector": code, "industry": code,
                 "subindustry": code, ...}}

    Returns:
        Dict of {"sector": pd.Series, "industry": pd.Series,
                 "subindustry": pd.Series} or None.
    """
    if not raw_cls:
        return None
    try:
        sector_dict = {}
        industry_dict = {}
        subindustry_dict = {}
        for ticker, info in raw_cls.items():
            if "sector" in info:
                sector_dict[ticker] = info["sector"]
            if "industry" in info:
                industry_dict[ticker] = info["industry"]
            if "subindustry" in info:
                subindustry_dict[ticker] = info["subindustry"]
        result = {}
        if sector_dict:
            result["sector"] = pd.Series(sector_dict)
        if industry_dict:
            result["industry"] = pd.Series(industry_dict)
        if subindustry_dict:
            result["subindustry"] = pd.Series(subindustry_dict)
        return result if result else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# DEAP Type System
# ---------------------------------------------------------------------------

DF = pd.core.frame.DataFrame


def _ensure_creator():
    """Create DEAP fitness and individual classes (safe for re-imports)."""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


# ---------------------------------------------------------------------------
# Primitive Set Builder
# ---------------------------------------------------------------------------

def build_primitive_set(
    feature_names: List[str],
    lookback_range: int = 40,
    include_advanced: bool = True,
) -> gp.PrimitiveSetTyped:
    """
    Build a DEAP PrimitiveSetTyped with all vectorized operators.

    Args:
        feature_names: List of input feature names (e.g., ['open', 'high', ...]).
        lookback_range: Range of integer constants for lookback params.
        include_advanced: Include advanced operators (kurtosis, skewness, etc.)

    Returns:
        Configured PrimitiveSetTyped ready for GP.
    """
    n_inputs = len(feature_names)
    input_types = [DF] * n_inputs
    pset = gp.PrimitiveSetTyped("alpha", input_types, DF)

    # --- Integer terminals for lookback windows ---
    for x in range(1, lookback_range + 1):
        pset.addTerminal(x, int)

    # --- Ephemeral random constants ---
    pset.addEphemeralConstant("rand_wide", lambda: random.uniform(-10, 10), float)
    pset.addEphemeralConstant("rand_narrow", lambda: random.uniform(-1, 1), float)
    pset.addEphemeralConstant("rand_positive", lambda: random.uniform(0, 1), float)

    # --- Identity / type conversion ---
    pset.addPrimitive(ops.extend, [int], int)
    pset.addPrimitive(ops.extend, [float], float)

    # --- Arithmetic (DF, DF) → DF ---
    pset.addPrimitive(ops.add, [DF, DF], DF, name="add")
    pset.addPrimitive(ops.subtract, [DF, DF], DF, name="subtract")
    pset.addPrimitive(ops.multiply, [DF, DF], DF, name="multiply")
    pset.addPrimitive(ops.divide, [DF, DF], DF, name="divide")
    pset.addPrimitive(ops.true_divide, [DF, DF], DF, name="true_divide")
    pset.addPrimitive(ops.df_max, [DF, DF], DF, name="df_max")
    pset.addPrimitive(ops.df_min, [DF, DF], DF, name="df_min")

    # --- Arithmetic (DF, scalar) → DF ---
    pset.addPrimitive(ops.npfadd, [DF, float], DF, name="npfadd")
    pset.addPrimitive(ops.npfsub, [DF, float], DF, name="npfsub")
    pset.addPrimitive(ops.npfmul, [DF, float], DF, name="npfmul")
    pset.addPrimitive(ops.npfdiv, [DF, float], DF, name="npfdiv")
    pset.addPrimitive(ops.SignedPower, [DF, float], DF, name="SignedPower")

    # --- Unary (DF) → DF ---
    pset.addPrimitive(ops.negative, [DF], DF, name="negative")
    pset.addPrimitive(ops.Abs, [DF], DF, name="Abs")
    pset.addPrimitive(ops.Sign, [DF], DF, name="Sign")
    pset.addPrimitive(ops.Inverse, [DF], DF, name="Inverse")
    pset.addPrimitive(ops.rank, [DF], DF, name="rank")
    pset.addPrimitive(ops.log, [DF], DF, name="log")
    pset.addPrimitive(ops.log10, [DF], DF, name="log10")
    pset.addPrimitive(ops.sqrt, [DF], DF, name="sqrt")
    pset.addPrimitive(ops.square, [DF], DF, name="square")
    pset.addPrimitive(ops.log_diff, [DF], DF, name="log_diff")
    pset.addPrimitive(ops.s_log_1p, [DF], DF, name="s_log_1p")

    # --- Time-series (DF, int) → DF ---
    pset.addPrimitive(ops.ts_sum, [DF, int], DF, name="ts_sum")
    pset.addPrimitive(ops.sma, [DF, int], DF, name="sma")
    pset.addPrimitive(ops.ts_rank, [DF, int], DF, name="ts_rank")
    pset.addPrimitive(ops.ts_min, [DF, int], DF, name="ts_min")
    pset.addPrimitive(ops.ts_max, [DF, int], DF, name="ts_max")
    pset.addPrimitive(ops.delta, [DF, int], DF, name="delta")
    pset.addPrimitive(ops.stddev, [DF, int], DF, name="stddev")
    pset.addPrimitive(ops.delay, [DF, int], DF, name="delay")
    pset.addPrimitive(ops.ArgMax, [DF, int], DF, name="ArgMax")
    pset.addPrimitive(ops.ArgMin, [DF, int], DF, name="ArgMin")
    pset.addPrimitive(ops.Product, [DF, int], DF, name="Product")
    pset.addPrimitive(ops.Decay_lin, [DF, int], DF, name="Decay_lin")
    pset.addPrimitive(ops.ts_zscore, [DF, int], DF, name="ts_zscore")

    # --- Exponential decay (DF, float) → DF ---
    pset.addPrimitive(ops.Decay_exp, [DF, float], DF, name="Decay_exp")

    # --- Two-input time-series (DF, DF, int) → DF ---
    pset.addPrimitive(ops.correlation, [DF, DF, int], DF, name="correlation")
    pset.addPrimitive(ops.covariance, [DF, DF, int], DF, name="covariance")

    if include_advanced:
        pset.addPrimitive(ops.ts_skewness, [DF, int], DF, name="ts_skewness")
        pset.addPrimitive(ops.ts_kurtosis, [DF, int], DF, name="ts_kurtosis")
        pset.addPrimitive(ops.ts_entropy, [DF, int], DF, name="ts_entropy")
        pset.addPrimitive(ops.normalize, [DF], DF, name="normalize")

    # Rename arguments to field names
    for i, name in enumerate(feature_names):
        pset.renameArguments(**{f"ARG{i}": name})

    return pset


# ---------------------------------------------------------------------------
# GP Configuration
# ---------------------------------------------------------------------------

@dataclass
class GPConfig:
    """Configuration for the GP engine."""
    population_size: int = 500
    n_generations: int = 100
    max_tree_depth: int = 6
    tournament_size: int = 7
    crossover_prob: float = 0.7
    mutation_prob: float = 0.1
    hall_of_fame_size: int = 10

    # Simulation parameters
    booksize: float = 20_000_000.0
    max_stock_weight: float = 0.01
    decay: int = 0
    delay: int = 1
    neutralization: str = "subindustry"
    fees_bps: float = 0.0

    # Fitness filters
    min_turnover: float = 0.001
    max_turnover: float = 1.0
    fitness_cutoff: float = 0.0
    corr_cutoff: float = 0.7

    # Feature configuration
    lookback_range: int = 40
    include_advanced: bool = True


# ---------------------------------------------------------------------------
# GPAlphaEngine
# ---------------------------------------------------------------------------

class GPAlphaEngine:
    """
    DEAP-powered Genetic Programming engine for alpha discovery.

    Evolves expression trees that map market data DataFrames → alpha signals,
    evaluating each individual via vectorized simulation.
    """

    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        returns_df: pd.DataFrame,
        close_df: pd.DataFrame | None = None,
        open_df: pd.DataFrame | None = None,
        classifications: dict | None = None,
        config: GPConfig | None = None,
        db: AlphaDatabase | None = None,
    ):
        """
        Args:
            data: Dict of field_name → DataFrame (dates × tickers).
                  Must include the feature_names used in the primitive set.
            returns_df: Forward returns matrix for PnL computation.
            close_df: Close prices for price-based filtering.
            open_df: Open prices for open-to-open returns (delay >= 1).
            classifications: Dict of GICS classification Series keyed by level:
                             {"sector": Series, "industry": Series,
                              "subindustry": Series}
            config: GP configuration.
            db: Optional alpha database for persistence.
        """
        self.config = config or GPConfig()
        self.data = data
        self.returns_df = returns_df
        self.close_df = close_df
        self.open_df = open_df
        self.classifications = classifications
        self.db = db

        # Feature names = ordered keys of data dict
        self.feature_names = list(data.keys())
        self.feature_dfs = [data[k] for k in self.feature_names]

        # Build DEAP primitives
        _ensure_creator()
        self.pset = build_primitive_set(
            self.feature_names,
            lookback_range=self.config.lookback_range,
            include_advanced=self.config.include_advanced,
        )

        # Build toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=2)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("evaluate", self._evaluate)
        self.toolbox.register("select", tools.selTournament, tournsize=self.config.tournament_size)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genGrow, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)

        # Depth limits
        self.toolbox.decorate("mate", gp.staticLimit(
            key=operator.attrgetter("height"), max_value=self.config.max_tree_depth))
        self.toolbox.decorate("mutate", gp.staticLimit(
            key=operator.attrgetter("height"), max_value=self.config.max_tree_depth))

        # Tracking
        self.trial_counter = 0
        self.best_alphas: List[Dict[str, Any]] = []
        self.run_id: int | None = None

    @classmethod
    def from_synthetic(
        cls,
        n_stocks: int = 200,
        n_days: int = 500,
        seed: int = 42,
        config: GPConfig | None = None,
        db_path: str | None = None,
    ) -> "GPAlphaEngine":
        """Create engine backed by synthetic data for development/testing."""
        ds = SyntheticDataGenerator().generate(
            n_stocks=n_stocks, n_days=n_days, seed=seed, start_date="2020-01-02"
        )
        ctx = InMemoryDataContext(ds)

        # Build feature DataFrames
        data = {}
        for field in ["open", "high", "low", "close", "volume"]:
            data[field] = ctx._price_matrices[field]

        close = data["close"]
        open_prices = data["open"]
        volume = data["volume"]
        data["dollars_traded"] = close * volume
        data["adv20"] = (close * volume).rolling(20).mean()
        data["returns"] = close.pct_change()

        for w in [10, 20, 30, 90]:
            data[f"hist_vol_{w}"] = close.pct_change().rolling(w).std() * (252 ** 0.5)

        returns_df = close.pct_change().shift(-1)  # fallback forward returns
        db = AlphaDatabase(db_path) if db_path else None

        # Build GICS classification Series for each level
        classifications = _build_classifications(ctx._classifications)

        return cls(
            data=data,
            returns_df=returns_df,
            close_df=close,
            open_df=open_prices,
            classifications=classifications,
            config=config,
            db=db,
        )

    # Full feature set for GP — curated, high-impact datasets.
    # Aliases (e.g. 'sales'→'revenue', 'assets'→'total_assets') and
    # internal classification matrices (_sector_groups, sector, etc.) are excluded.
    GP_FEATURE_SET = [
        # Price & Volume (6)
        "open", "high", "low", "close", "volume", "vwap",
        # Liquidity & Market (5)
        "returns", "log_returns", "dollars_traded", "adv20", "adv60",
        # Market Cap (2)
        "cap", "market_cap",
        # Income Statement (10)
        "revenue", "cost_of_revenue", "gross_profit", "sga_expense",
        "operating_income", "ebitda", "net_income", "eps", "eps_diluted",
        "interest_expense",
        # Income Statement — NEW (3)
        "income_before_tax", "interest_income", "stock_based_compensation",
        # Tax & R&D (2)
        "income_tax", "rd_expense",
        # Balance Sheet (14)
        "total_assets", "assets_curr", "total_liabilities", "liabilities_curr",
        "total_equity", "total_debt", "cash", "inventory", "receivables",
        "payables", "goodwill", "intangibles", "ppe_net", "retained_earnings",
        # Balance Sheet — NEW (6)
        "short_term_debt", "long_term_debt", "deferred_revenue",
        "accrued_expenses", "treasury_stock", "long_term_investments",
        # Derived Balance Sheet (4)
        "net_debt", "working_capital", "invested_capital", "shares_out",
        # Cash Flow (5)
        "cashflow_op", "capex", "free_cashflow", "depreciation",
        "dividends_paid",
        # Cash Flow — NEW (5)
        "change_in_working_capital", "acquisitions_net",
        "net_debt_issuance", "net_stock_issuance", "deferred_income_tax",
        # Corporate Actions (2)
        "stock_repurchase", "debt_repayment",
        # Valuation Ratios (7)
        "pe_ratio", "pb_ratio", "ev_to_ebitda", "debt_to_equity",
        "current_ratio", "roe", "roa",
        # Profitability & Quality — NEW (7)
        "roic", "roce", "income_quality", "sbc_to_revenue",
        "net_debt_to_ebitda", "capex_to_depreciation",
        "gross_margin",
        # Cash Cycle — NEW (5)
        "days_sales_outstanding", "days_payables_outstanding",
        "days_inventory_outstanding", "operating_cycle", "cash_conversion_cycle",
        # Yield (1)
        "dividend_yield",
        # Per-Share (4)
        "book_value_per_share", "revenue_per_share",
        "tangible_book_per_share", "fcf_per_share",
        # Derived Valuation (2)
        "enterprise_value", "inventory_turnover",
        # Volatility — trimmed to key windows (6)
        "historical_volatility_20", "historical_volatility_60",
        "historical_volatility_120",
        "parkinson_volatility_20", "parkinson_volatility_60",
        "parkinson_volatility_120",
    ]

    @classmethod
    def from_context(
        cls,
        ctx: InMemoryDataContext,
        feature_names: List[str] | None = None,
        config: GPConfig | None = None,
        db: AlphaDatabase | None = None,
    ) -> "GPAlphaEngine":
        """Create engine from an InMemoryDataContext using all available fields."""
        available = set(k for k in ctx._price_matrices.keys()
                        if not k.startswith("_"))  # skip internal fields

        if feature_names is None:
            # Use curated set, filtered to what's actually available
            feature_names = [f for f in cls.GP_FEATURE_SET if f in available]

        logger.info(f"GP features: {len(feature_names)} fields from {len(available)} available")

        data = {}
        for field in feature_names:
            if field in ctx._price_matrices:
                data[field] = ctx._price_matrices[field]

        close = data.get("close", ctx._price_matrices["close"])
        open_prices = data.get("open", ctx._price_matrices.get("open"))
        data.setdefault("returns", close.pct_change())
        data.setdefault("dollars_traded", close * data.get("volume", ctx._price_matrices["volume"]))
        data.setdefault("adv20", data["dollars_traded"].rolling(20).mean())

        returns_df = close.pct_change().shift(-1)  # fallback (close-to-close)

        classifications = _build_classifications(ctx._classifications)

        return cls(data=data, returns_df=returns_df, close_df=close,
                   open_df=open_prices, classifications=classifications,
                   config=config, db=db)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, individual) -> Tuple[float]:
        """Evaluate a single GP individual. Returns (fitness,) tuple."""
        self.trial_counter += 1
        expr_str = str(individual)

        try:
            func = self.toolbox.compile(expr=individual)
            alpha_df = func(*self.feature_dfs)
        except Exception as e:
            logger.debug(f"Compile/eval error for '{expr_str[:60]}': {e}")
            return (0.0,)

        if not isinstance(alpha_df, pd.DataFrame):
            return (0.0,)

        try:
            result = simulate_vectorized(
                alpha_df=alpha_df,
                returns_df=self.returns_df,
                close_df=self.close_df,
                open_df=self.open_df,
                classifications=self.classifications,
                booksize=self.config.booksize,
                max_stock_weight=self.config.max_stock_weight,
                decay=self.config.decay,
                delay=self.config.delay,
                neutralization=self.config.neutralization,
                fees_bps=self.config.fees_bps,
            )
        except Exception as e:
            logger.debug(f"Simulation error for '{expr_str[:60]}': {e}")
            return (0.0,)

        fitness = result.fitness  # BRAIN fitness: sharpe * sqrt(|returns| / max(turnover, 0.125))

        # Filter: turnover bounds
        if result.turnover < self.config.min_turnover:
            fitness = 0.0
        if result.turnover > self.config.max_turnover:
            # Apply decay sweep like the original AWSgenProgWorker
            fitness = self._decay_sweep(individual, result)

        if math.isnan(fitness):
            fitness = 0.0

        # Store good alphas (with correlation check)
        if fitness > self.config.fitness_cutoff and result.turnover > self.config.min_turnover:
            # Check correlation against existing recorded alphas
            pnl_vec = result.daily_pnl.values if hasattr(result.daily_pnl, 'values') else np.array(result.daily_pnl)
            is_diverse = self._check_diversity(pnl_vec)
            if is_diverse:
                self._record_alpha(expr_str, result, fitness, pnl_vec)

        return (fitness,)

    def _check_diversity(self, pnl_vec: np.ndarray) -> bool:
        """Check if new alpha PnL is sufficiently uncorrelated with stored alphas."""
        if not hasattr(self, '_stored_pnl_vectors') or len(self._stored_pnl_vectors) == 0:
            return True

        for existing_pnl in self._stored_pnl_vectors:
            try:
                # Align lengths
                min_len = min(len(pnl_vec), len(existing_pnl))
                if min_len < 20:
                    continue
                corr = np.corrcoef(pnl_vec[:min_len], existing_pnl[:min_len])[0, 1]
                if not np.isnan(corr) and abs(corr) > self.config.corr_cutoff:
                    return False  # Too correlated with an existing alpha
            except Exception:
                continue
        return True

    def _decay_sweep(self, individual, initial_result: VectorizedSimResult) -> float:
        """
        If turnover is too high, try wrapping with exponential decay.
        Matches original AWSgenProgWorker.py pattern.
        """
        original_str = str(individual)
        best_fitness = 0.0

        for alpha_exp in [x * 0.05 for x in range(19, 1, -1)]:
            try:
                func = self.toolbox.compile(expr=individual)
                alpha_df = func(*self.feature_dfs)
                alpha_df = ops.Decay_exp(alpha_df, alpha_exp)

                result = simulate_vectorized(
                    alpha_df=alpha_df,
                    returns_df=self.returns_df,
                    close_df=self.close_df,
                    open_df=self.open_df,
                    classifications=self.classifications,
                    booksize=self.config.booksize,
                    max_stock_weight=self.config.max_stock_weight,
                    decay=0,
                    delay=self.config.delay,
                    neutralization=self.config.neutralization,
                    fees_bps=self.config.fees_bps,
                )

                if (result.turnover < self.config.max_turnover and
                        result.sharpe > self.config.fitness_cutoff):
                    if result.sharpe > best_fitness:
                        best_fitness = result.sharpe
                        self._record_alpha(
                            f"Decay_exp({original_str}, {alpha_exp})",
                            result, best_fitness
                        )
                    break
            except Exception:
                continue

        return best_fitness

    def _record_alpha(self, expression: str, result: VectorizedSimResult,
                      fitness: float, pnl_vec: np.ndarray | None = None) -> None:
        """Store a successful alpha."""
        record = {
            "expression": expression,
            "sharpe": result.sharpe,
            "fitness": fitness,
            "turnover": result.turnover,
            "returns_ann": result.returns_ann,
            "max_drawdown": result.max_drawdown,
            "margin_bps": result.margin_bps,
            "trial": self.trial_counter,
        }
        self.best_alphas.append(record)

        # Store PnL vector for correlation checking
        if pnl_vec is not None:
            if not hasattr(self, '_stored_pnl_vectors'):
                self._stored_pnl_vectors = []
            self._stored_pnl_vectors.append(pnl_vec.copy())

        if self.db:
            try:
                alpha_params = {
                    "delay": self.config.delay,
                    "neutralization": self.config.neutralization,
                    "universe": f"TOP{len(self.close_df.columns)}",
                    "decay": self.config.decay,
                    "booksize": self.config.booksize,
                    "fees_bps": self.config.fees_bps,
                }
                alpha_id = self.db.insert_alpha(
                    expression=expression,
                    params=alpha_params,
                    source="gp",
                    run_id=self.run_id,
                    trial_index=self.trial_counter,
                )
                self.db.insert_evaluation(
                    alpha_id=alpha_id,
                    source="gp_local",
                    sharpe=result.sharpe,
                    fitness=fitness,
                    turnover=result.turnover,
                    max_drawdown=result.max_drawdown,
                    returns_ann=result.returns_ann,
                    margin_bps=result.margin_bps,
                    passed_checks=0,
                )
            except Exception as e:
                logger.debug(f"DB store error: {e}")

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(
        self,
        n_generations: int | None = None,
        population_size: int | None = None,
        seed: int | None = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the GP evolution.

        Returns dict with: best_expression, best_fitness, hall_of_fame,
        logbook, best_alphas.
        """
        n_gen = n_generations or self.config.n_generations
        pop_size = population_size or self.config.population_size

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Create run in database
        if self.db:
            self.run_id = self.db.create_run(
                strategy="GP evolution",
                llm_model="DEAP",
                config={
                    "n_generations": n_gen,
                    "population_size": pop_size,
                    "max_tree_depth": self.config.max_tree_depth,
                    "features": self.feature_names,
                },
            )

        print(f"\n🧬 GP Alpha Engine — {n_gen} generations × {pop_size} individuals")
        print(f"   Features: {', '.join(self.feature_names)}")
        print(f"   Max depth: {self.config.max_tree_depth}")
        print("=" * 60)

        start_time = time.time()

        pop = self.toolbox.population(n=pop_size)
        hof = tools.HallOfFame(self.config.hall_of_fame_size)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        pop, logbook = algorithms.eaSimple(
            pop, self.toolbox,
            cxpb=self.config.crossover_prob,
            mutpb=self.config.mutation_prob,
            ngen=n_gen,
            stats=mstats,
            halloffame=hof,
            verbose=verbose,
        )

        elapsed = time.time() - start_time

        # Finalize
        if self.db and self.run_id:
            self.db.finish_run(self.run_id)

        # Results
        best_expr = str(hof[0]) if len(hof) > 0 else ""
        best_fitness = hof[0].fitness.values[0] if len(hof) > 0 else 0.0

        print(f"\n{'='*60}")
        print(f"🎯 GP Complete in {elapsed:.1f}s")
        print(f"   Trials evaluated: {self.trial_counter}")
        print(f"   Best fitness: {best_fitness:.4f}")
        if best_expr:
            print(f"   Best expression: {best_expr[:100]}")
        print(f"   Alphas recorded: {len(self.best_alphas)}")

        if self.best_alphas:
            top = sorted(self.best_alphas, key=lambda x: x['sharpe'], reverse=True)[:5]
            print(f"\n🏆 Top 5 alphas:")
            for i, a in enumerate(top, 1):
                print(f"   {i}. Sharpe={a['sharpe']:.3f} T/O={a['turnover']:.3f} | {a['expression'][:80]}")

        return {
            "best_expression": best_expr,
            "best_fitness": best_fitness,
            "hall_of_fame": [str(h) for h in hof],
            "logbook": logbook,
            "best_alphas": self.best_alphas,
            "elapsed_seconds": elapsed,
            "trials_evaluated": self.trial_counter,
        }

    def evaluate_expression(self, expression: str) -> VectorizedSimResult | None:
        """Evaluate a single expression string (useful for testing discovered alphas)."""
        try:
            func = self.toolbox.compile(expr=expression)
            alpha_df = func(*self.feature_dfs)
            return simulate_vectorized(
                alpha_df=alpha_df,
                returns_df=self.returns_df,
                close_df=self.close_df,
                open_df=self.open_df,
                classifications=self.classifications,
                booksize=self.config.booksize,
                max_stock_weight=self.config.max_stock_weight,
                delay=self.config.delay,
                neutralization=self.config.neutralization,
            )
        except Exception as e:
            logger.error(f"Failed to evaluate '{expression[:60]}': {e}")
            return None
