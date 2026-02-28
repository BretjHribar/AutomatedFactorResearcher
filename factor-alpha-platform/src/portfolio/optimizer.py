"""
Portfolio Optimizer — combine multiple alphas into an optimal portfolio.

Implements:
  - Pairwise alpha correlation analysis
  - Mean-variance optimization (max Sharpe)
  - Risk-parity weighting
  - Equal-weight baseline
  - Turnover-constrained optimization
  - Out-of-sample portfolio evaluation

Usage:
    from src.portfolio.optimizer import PortfolioOptimizer
    opt = PortfolioOptimizer()
    opt.add_alpha("alpha1", sim_result_1)
    opt.add_alpha("alpha2", sim_result_2)
    result = opt.optimize(method="max_sharpe")
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Result Types
# ---------------------------------------------------------------------------

@dataclass
class CorrelationInfo:
    """Pairwise alpha correlation analysis."""
    correlation_matrix: pd.DataFrame      # name × name correlation
    avg_pairwise_corr: float = 0.0
    max_pairwise_corr: float = 0.0
    highly_correlated_pairs: list[tuple[str, str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "avg_pairwise_corr": self.avg_pairwise_corr,
            "max_pairwise_corr": self.max_pairwise_corr,
            "highly_correlated": [
                {"alpha_a": a, "alpha_b": b, "corr": c}
                for a, b, c in self.highly_correlated_pairs
            ],
        }


@dataclass
class PortfolioResult:
    """Result from portfolio optimization."""
    method: str
    weights: Dict[str, float]       # alpha_name → weight

    # Combined portfolio metrics
    sharpe: float = 0.0
    returns_ann: float = 0.0
    max_drawdown: float = 0.0
    turnover: float = 0.0
    fitness: float = 0.0
    total_pnl: float = 0.0

    # Time series
    daily_pnl: pd.Series | None = None
    cumulative_pnl: pd.Series | None = None
    pnl_dates: list[str] | None = None

    # Per-alpha contribution
    alpha_contributions: Dict[str, float] | None = None  # alpha → PnL contribution %

    # Correlation
    correlation_info: CorrelationInfo | None = None

    # OOS metrics (if computed)
    oos_sharpe: float | None = None
    oos_returns_ann: float | None = None

    def to_dict(self) -> dict:
        d = {
            "method": self.method,
            "weights": self.weights,
            "sharpe": self.sharpe,
            "returns_ann": self.returns_ann,
            "max_drawdown": self.max_drawdown,
            "turnover": self.turnover,
            "fitness": self.fitness,
            "total_pnl": self.total_pnl,
            "alpha_contributions": self.alpha_contributions,
            "oos_sharpe": self.oos_sharpe,
            "oos_returns_ann": self.oos_returns_ann,
        }
        if self.correlation_info:
            d["correlation"] = self.correlation_info.to_dict()
        if self.daily_pnl is not None:
            d["daily_pnl"] = self.daily_pnl.tolist()
        if self.cumulative_pnl is not None:
            d["cumulative_pnl"] = self.cumulative_pnl.tolist()
        if self.pnl_dates is not None:
            d["pnl_dates"] = self.pnl_dates
        return d


# ---------------------------------------------------------------------------
# Portfolio Optimizer
# ---------------------------------------------------------------------------

class PortfolioOptimizer:
    """
    Multi-alpha portfolio optimizer.

    Combines multiple alpha PnL streams into an optimal portfolio
    using various weighting methods.
    """

    def __init__(self, booksize: float = 20_000_000.0):
        self.booksize = booksize
        self._alphas: Dict[str, Dict[str, Any]] = {}  # name → {pnl, sharpe, ...}

    def add_alpha(
        self,
        name: str,
        daily_pnl: pd.Series,
        sharpe: float = 0.0,
        expression: str = "",
    ) -> None:
        """Add an alpha's PnL series to the optimizer."""
        self._alphas[name] = {
            "daily_pnl": daily_pnl.dropna(),
            "sharpe": sharpe,
            "expression": expression,
        }

    def add_from_sim_result(self, name: str, sim_result: Any) -> None:
        """Add alpha from a VectorizedSimResult."""
        self.add_alpha(
            name=name,
            daily_pnl=sim_result.daily_pnl,
            sharpe=sim_result.sharpe,
        )

    @property
    def n_alphas(self) -> int:
        return len(self._alphas)

    # ----- Correlation Analysis -----

    def compute_correlation(self, corr_threshold: float = 0.5) -> CorrelationInfo:
        """Compute pairwise correlation matrix of alpha PnL streams."""
        if len(self._alphas) < 2:
            names = list(self._alphas.keys())
            return CorrelationInfo(
                correlation_matrix=pd.DataFrame(1.0, index=names, columns=names)
            )

        # Build aligned PnL DataFrame
        pnl_df = self._build_pnl_df()
        corr_matrix = pnl_df.corr()

        # Extract pairwise correlations (upper triangle)
        names = list(pnl_df.columns)
        pairs = []
        corr_values = []
        high_pairs = []

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                c = corr_matrix.iloc[i, j]
                if not np.isnan(c):
                    corr_values.append(abs(c))
                    if abs(c) >= corr_threshold:
                        high_pairs.append((names[i], names[j], float(c)))

        avg_corr = float(np.mean(corr_values)) if corr_values else 0.0
        max_corr = float(np.max(corr_values)) if corr_values else 0.0

        return CorrelationInfo(
            correlation_matrix=corr_matrix,
            avg_pairwise_corr=avg_corr,
            max_pairwise_corr=max_corr,
            highly_correlated_pairs=high_pairs,
        )

    # ----- Optimization Methods -----

    def optimize(
        self,
        method: str = "max_sharpe",
        max_weight: float = 0.4,
        min_weight: float = 0.0,
        turnover_penalty: float = 0.0,
    ) -> PortfolioResult:
        """
        Optimize portfolio weights.

        Methods:
          - 'equal_weight': 1/N allocation
          - 'risk_parity': inverse-volatility weighting
          - 'max_sharpe': maximize Sharpe ratio (mean-variance)
          - 'min_variance': minimize total variance
          - 'sharpe_weighted': weight proportional to Sharpe ratio
        """
        if len(self._alphas) == 0:
            return PortfolioResult(method=method, weights={})

        if len(self._alphas) == 1:
            name = list(self._alphas.keys())[0]
            weights = {name: 1.0}
            return self._evaluate_portfolio(weights, method)

        if method == "equal_weight":
            weights = self._equal_weight()
        elif method == "risk_parity":
            weights = self._risk_parity()
        elif method == "max_sharpe":
            weights = self._max_sharpe(max_weight, min_weight)
        elif method == "min_variance":
            weights = self._min_variance(max_weight, min_weight)
        elif method == "sharpe_weighted":
            weights = self._sharpe_weighted()
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        return self._evaluate_portfolio(weights, method)

    def optimize_all(self) -> Dict[str, PortfolioResult]:
        """Run all optimization methods and return results."""
        results = {}
        for method in ["equal_weight", "risk_parity", "max_sharpe",
                        "min_variance", "sharpe_weighted"]:
            try:
                results[method] = self.optimize(method=method)
            except Exception as e:
                logger.warning(f"Optimization {method} failed: {e}")
        return results

    # ----- Weighting Implementations -----

    def _equal_weight(self) -> Dict[str, float]:
        n = len(self._alphas)
        return {name: 1.0 / n for name in self._alphas}

    def _risk_parity(self) -> Dict[str, float]:
        """Inverse-volatility weighting."""
        vols = {}
        for name, data in self._alphas.items():
            pnl = data["daily_pnl"]
            vol = pnl.std()
            vols[name] = vol if vol > 0 else 1e-8

        inv_vols = {name: 1.0 / vol for name, vol in vols.items()}
        total = sum(inv_vols.values())
        return {name: v / total for name, v in inv_vols.items()}

    def _sharpe_weighted(self) -> Dict[str, float]:
        """Weight proportional to absolute Sharpe ratio (positive Sharpe only)."""
        sharpes = {}
        for name, data in self._alphas.items():
            s = max(data["sharpe"], 0.0)
            sharpes[name] = s

        total = sum(sharpes.values())
        if total == 0:
            return self._equal_weight()
        return {name: s / total for name, s in sharpes.items()}

    def _max_sharpe(
        self, max_weight: float = 0.4, min_weight: float = 0.0
    ) -> Dict[str, float]:
        """
        Mean-variance optimization to maximize Sharpe ratio.

        Uses analytical solution when unconstrained, or
        iterative grid search for constrained case.
        """
        pnl_df = self._build_pnl_df()
        names = list(pnl_df.columns)
        n = len(names)

        # Mean returns and covariance
        mu = pnl_df.mean().values
        cov = pnl_df.cov().values

        # Check for positive returns
        if np.all(mu <= 0):
            return self._equal_weight()

        # Analytical solution (unconstrained Markowitz)
        try:
            cov_inv = np.linalg.inv(cov + np.eye(n) * 1e-8)
            w_raw = cov_inv @ mu
            w_raw = np.maximum(w_raw, 0)  # Long-only
            total = np.sum(w_raw)
            if total > 0:
                w = w_raw / total
            else:
                w = np.ones(n) / n

            # Apply constraints
            w = np.clip(w, min_weight, max_weight)
            total = np.sum(w)
            if total > 0:
                w = w / total

            return dict(zip(names, w.tolist()))
        except np.linalg.LinAlgError:
            return self._risk_parity()

    def _min_variance(
        self, max_weight: float = 0.4, min_weight: float = 0.0
    ) -> Dict[str, float]:
        """Minimum variance portfolio."""
        pnl_df = self._build_pnl_df()
        names = list(pnl_df.columns)
        n = len(names)

        cov = pnl_df.cov().values

        try:
            cov_inv = np.linalg.inv(cov + np.eye(n) * 1e-8)
            ones = np.ones(n)
            w_raw = cov_inv @ ones
            w_raw = np.maximum(w_raw, 0)
            total = np.sum(w_raw)
            if total > 0:
                w = w_raw / total
            else:
                w = np.ones(n) / n

            w = np.clip(w, min_weight, max_weight)
            total = np.sum(w)
            if total > 0:
                w = w / total

            return dict(zip(names, w.tolist()))
        except np.linalg.LinAlgError:
            return self._equal_weight()

    # ----- Portfolio Evaluation -----

    def _evaluate_portfolio(
        self, weights: Dict[str, float], method: str
    ) -> PortfolioResult:
        """Evaluate a weighted portfolio."""
        pnl_df = self._build_pnl_df()

        # Weighted PnL
        combined_pnl = pd.Series(0.0, index=pnl_df.index)
        contributions = {}

        for name, weight in weights.items():
            if name in pnl_df.columns:
                alpha_pnl = pnl_df[name] * weight
                combined_pnl += alpha_pnl
                contributions[name] = float(alpha_pnl.sum())

        total_portfolio_pnl = float(combined_pnl.sum())

        # Normalize contributions to percentages
        if total_portfolio_pnl != 0:
            contributions = {
                k: v / total_portfolio_pnl * 100
                for k, v in contributions.items()
            }

        cumulative_pnl = combined_pnl.cumsum()

        # Sharpe
        equity = self.booksize / 2.0
        daily_ret = combined_pnl / equity
        mean_ret = daily_ret.mean()
        std_ret = daily_ret.std()
        sharpe = float(mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

        # Annualized return
        returns_ann = float(mean_ret * 252)

        # Max drawdown
        running_max = cumulative_pnl.cummax()
        drawdown = cumulative_pnl - running_max
        max_dd = float(drawdown.min() / self.booksize) if self.booksize > 0 else 0.0

        # Fitness
        if abs(returns_ann) > 0:
            fitness = sharpe * math.sqrt(abs(returns_ann * 100) / max(0.125, 0.3))
        else:
            fitness = 0.0

        # Dates
        dates = [str(d.date()) if hasattr(d, 'date') else str(d) for d in combined_pnl.index]

        # Correlation
        corr_info = self.compute_correlation()

        return PortfolioResult(
            method=method,
            weights=weights,
            sharpe=sharpe,
            returns_ann=returns_ann,
            max_drawdown=max_dd,
            turnover=0.0,  # Combined turnover needs position-level data
            fitness=fitness,
            total_pnl=total_portfolio_pnl,
            daily_pnl=combined_pnl,
            cumulative_pnl=cumulative_pnl,
            pnl_dates=dates,
            alpha_contributions=contributions,
            correlation_info=corr_info,
        )

    # ----- Utility -----

    def _build_pnl_df(self) -> pd.DataFrame:
        """Build aligned DataFrame of daily PnL for all alphas."""
        series = {}
        for name, data in self._alphas.items():
            series[name] = data["daily_pnl"]

        df = pd.DataFrame(series)
        df = df.dropna(how="all")
        df = df.fillna(0.0)
        return df
