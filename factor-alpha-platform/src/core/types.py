"""
Core type definitions for the Factor Alpha Platform.

All core data classes: FactorDef, SimSettings, SimResult, AnnualMetrics,
QuintileResult, SubsampleMetrics, FactorEvaluation, DailyResult, etc.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Simulation Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimSettings:
    """Simulation configuration matching BRAIN parameters."""

    booksize: float = 20_000_000.0
    delay: int = 1
    duration_years: int = 5
    neutralization: str = "subindustry"
    max_instrument_weight: float = 0.0  # 0 = unconstrained
    decay: int = 0  # 0 = no decay
    universe: str = "TOP3000"
    transaction_cost_model: str = "none"  # none, basic, realistic
    start_date: dt.date | None = None
    end_date: dt.date | None = None


# ---------------------------------------------------------------------------
# Factor Definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FactorDef:
    """Immutable factor definition. Frozen at promotion time."""

    factor_id: str
    version: int
    name: str
    description: str
    category: str  # momentum, value, quality, growth, sentiment, etc.

    # Core computation — a pure function: (date, DataContext) -> dict[str, float]
    compute_fn: Callable[..., dict[str, float]]

    # Metadata
    data_fields: tuple[str, ...] = ()
    frequency: str = "daily"
    lookback_days: int = 256
    universe: str = "TOP3000"
    delay: int = 1
    decay: int = 0
    neutralization: str = "subindustry"

    # Research metrics (frozen at promotion)
    is_sharpe: float | None = None
    is_return: float | None = None
    is_turnover: float | None = None
    is_max_drawdown: float | None = None
    is_fitness: float | None = None
    is_ic_mean: float | None = None
    is_ic_ir: float | None = None
    oos_sharpe: float | None = None
    correlation_cluster: str | None = None
    approved_date: dt.date | None = None
    approved_by: str | None = None


# ---------------------------------------------------------------------------
# Daily Result
# ---------------------------------------------------------------------------

@dataclass
class DailyResult:
    """Single day of simulation output."""

    date: dt.date
    pnl: float
    positions: dict[str, float]  # ticker -> dollar position
    turnover: float  # as fraction of booksize
    alpha_values: dict[str, float] | None = None  # raw alpha values


# ---------------------------------------------------------------------------
# Annual Metrics
# ---------------------------------------------------------------------------

@dataclass
class AnnualMetrics:
    """Per-year (or total) performance metrics — BRAIN-compatible."""

    year: int | str  # year number or "Total"
    booksize: float
    pnl: float
    annual_return: float  # pnl / (booksize / 2)
    sharpe: float  # annualized
    max_drawdown: float  # as % of equity capital (booksize/2)
    pct_profitable_days: float
    daily_turnover: float  # mean daily turnover
    profit_per_dollar: float  # in cents
    fitness: float
    margin_bps: float
    n_trading_days: int = 0


# ---------------------------------------------------------------------------
# Quintile Analysis
# ---------------------------------------------------------------------------

@dataclass
class QuintileResult:
    """Quintile analysis results."""

    quintile_returns: list[float]  # Q1 (top) to Q5 (bottom) mean returns
    quintile_sharpes: list[float]
    quintile_cumulative: list[list[float]]  # cumulative return curves per quintile
    monotonicity: float  # Spearman corr of quintile rank vs return
    spread: float  # Q1 return - Q5 return


# ---------------------------------------------------------------------------
# Subsample Metrics
# ---------------------------------------------------------------------------

@dataclass
class SubsampleMetrics:
    """Metrics for a sub-period of the backtest."""

    period_start: dt.date
    period_end: dt.date
    sharpe: float
    annual_return: float
    max_drawdown: float


# ---------------------------------------------------------------------------
# Simulation Result
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    """Complete simulation output."""

    # Daily time series
    dates: list[dt.date]
    daily_pnl: list[float]
    daily_positions: list[dict[str, float]]
    daily_turnover: list[float]

    # Annual metrics (per year + total)
    annual_metrics: list[AnnualMetrics]
    total_metrics: AnnualMetrics

    # Deep evaluation (may be None if not computed)
    quintile_analysis: QuintileResult | None = None
    ic_series: list[float] | None = None
    subsample_results: list[SubsampleMetrics] | None = None

    # Factor metadata
    factor_id: str = ""
    settings: SimSettings | None = None

    @property
    def daily_returns(self) -> np.ndarray:
        """Daily PnL as returns on equity capital."""
        if self.settings is None:
            equity = 10_000_000.0
        else:
            equity = self.settings.booksize / 2.0
        return np.array(self.daily_pnl) / equity


# ---------------------------------------------------------------------------
# Factor Evaluation
# ---------------------------------------------------------------------------

@dataclass
class FactorEvaluation:
    """Comprehensive factor evaluation — all screening metrics."""

    factor_id: str
    evaluation_date: dt.date

    # BRAIN-compatible metrics
    sharpe: float
    annual_return: float
    max_drawdown: float
    daily_turnover: float
    pct_profitable_days: float
    profit_per_dollar: float
    fitness: float
    sharpe_rating: str  # Spectacular/Excellent/Good/Average/Inferior/Poor

    # IC analysis
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ic_ir: float = 0.0
    ic_hit_rate: float = 0.0

    # Robustness
    oos_sharpe: float = 0.0
    is_to_oos_decay: float = 0.0
    subsample_stability: float = 0.0
    regime_sensitivity: dict[str, float] = field(default_factory=dict)

    # Quintile analysis
    quintile_returns: list[float] = field(default_factory=list)
    quintile_monotonicity: float = 0.0
    quintile_spread: float = 0.0

    # Correlation
    max_correlation_existing: float = 0.0
    correlation_cluster: str = ""

    # Portfolio contribution
    marginal_sharpe_contribution: float = 0.0
    marginal_return_contribution: float = 0.0

    # Implementability
    avg_holding_period_days: float = 0.0
    capacity_estimate_usd: float = 0.0
    alpha_decay_halflife_days: float = 0.0

    # Screening
    pass_screening: bool = False
    screening_failures: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Factor Health Check
# ---------------------------------------------------------------------------

@dataclass
class FactorHealthCheck:
    """Per-factor daily health check result."""

    factor_id: str
    date: dt.date
    status: str  # "OK", "WARN", "CRITICAL"
    alerts: list[str]

    coverage: float = 1.0
    distribution_shift: float = 1.0  # KS test p-value
    rolling_ic_60d: float = 0.0
    rolling_ic_252d: float = 0.0
    correlation_drift: float = 0.0
    data_freshness_days: int = 0
    compute_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Sharpe Rating
# ---------------------------------------------------------------------------

def get_sharpe_rating(sharpe: float, delay: int = 1) -> str:
    """Return BRAIN-compatible Sharpe rating label."""
    if delay == 0:
        if sharpe > 6.0:
            return "Spectacular"
        elif sharpe > 5.25:
            return "Excellent"
        elif sharpe > 4.50:
            return "Good"
        elif sharpe > 3.95:
            return "Average"
        elif sharpe >= 1.0:
            return "Inferior"
        else:
            return "Poor"
    else:  # delay >= 1
        if sharpe > 4.5:
            return "Spectacular"
        elif sharpe > 3.5:
            return "Excellent"
        elif sharpe > 3.0:
            return "Good"
        elif sharpe > 2.5:
            return "Average"
        elif sharpe >= 1.0:
            return "Inferior"
        else:
            return "Poor"


# ---------------------------------------------------------------------------
# Optimization Types
# ---------------------------------------------------------------------------

@dataclass
class OptimizationConstraints:
    """Constraints for portfolio optimization."""

    max_position_pct: float = 0.02
    max_sector_pct: float = 0.20
    max_adv_pct: float = 0.05
    max_turnover_pct: float = 0.20
    dollar_neutral: bool = True
    beta_neutral: bool = True
    max_tracking_error: float = 0.05
    max_factor_exposure: dict[str, float] = field(default_factory=dict)


@dataclass
class OptimizationProblem:
    """Input to the portfolio optimizer."""

    alpha_scores: pd.Series  # ticker -> combined alpha score
    risk_model: Any  # RiskModel (covariance + factor exposures)
    current_positions: pd.Series  # ticker -> current dollar position
    constraints: OptimizationConstraints = field(default_factory=OptimizationConstraints)
