"""
Out-of-Sample (OOS) Backtesting Framework.

Provides walk-forward and fixed-split OOS evaluation for alphas.
The key principle: alphas are designed/discovered on IS data, and
their true performance is measured on OOS data only.

This module provides:
  - fixed_split: simple IS/OOS split at a date boundary
  - walk_forward: rolling window IS→OOS evaluation
  - oos_evaluate: full OOS evaluation pipeline with detailed stats
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.simulation.vectorized_sim import simulate_vectorized, VectorizedSimResult


# ---------------------------------------------------------------------------
# OOS Result Types
# ---------------------------------------------------------------------------

@dataclass
class OOSResult:
    """Out-of-sample evaluation result with IS and OOS metrics."""

    # In-sample metrics
    is_sharpe: float = 0.0
    is_fitness: float = 0.0
    is_turnover: float = 0.0
    is_returns_ann: float = 0.0
    is_max_drawdown: float = 0.0
    is_margin_bps: float = 0.0
    is_total_pnl: float = 0.0
    is_start: str = ""
    is_end: str = ""
    is_days: int = 0

    # Out-of-sample metrics
    oos_sharpe: float = 0.0
    oos_fitness: float = 0.0
    oos_turnover: float = 0.0
    oos_returns_ann: float = 0.0
    oos_max_drawdown: float = 0.0
    oos_margin_bps: float = 0.0
    oos_total_pnl: float = 0.0
    oos_start: str = ""
    oos_end: str = ""
    oos_days: int = 0

    # Degradation metrics
    sharpe_decay: float = 0.0       # OOS sharpe / IS sharpe
    fitness_decay: float = 0.0
    return_decay: float = 0.0

    # Stability
    is_consistent: bool = False     # OOS sharpe > 0 if IS sharpe > 0

    # Full sim results
    is_sim: VectorizedSimResult | None = None
    oos_sim: VectorizedSimResult | None = None

    # Walk-forward folds
    walk_forward_folds: list["WalkForwardFold"] | None = None

    def to_dict(self) -> dict:
        d = {
            "is_sharpe": self.is_sharpe,
            "is_fitness": self.is_fitness,
            "is_turnover": self.is_turnover,
            "is_returns_ann": self.is_returns_ann,
            "is_max_drawdown": self.is_max_drawdown,
            "is_margin_bps": self.is_margin_bps,
            "is_total_pnl": self.is_total_pnl,
            "is_start": self.is_start,
            "is_end": self.is_end,
            "is_days": self.is_days,
            "oos_sharpe": self.oos_sharpe,
            "oos_fitness": self.oos_fitness,
            "oos_turnover": self.oos_turnover,
            "oos_returns_ann": self.oos_returns_ann,
            "oos_max_drawdown": self.oos_max_drawdown,
            "oos_margin_bps": self.oos_margin_bps,
            "oos_total_pnl": self.oos_total_pnl,
            "oos_start": self.oos_start,
            "oos_end": self.oos_end,
            "oos_days": self.oos_days,
            "sharpe_decay": self.sharpe_decay,
            "fitness_decay": self.fitness_decay,
            "return_decay": self.return_decay,
            "is_consistent": self.is_consistent,
        }
        if self.walk_forward_folds:
            d["walk_forward"] = [f.to_dict() for f in self.walk_forward_folds]
        return d


@dataclass
class WalkForwardFold:
    """One fold in a walk-forward analysis."""
    fold_idx: int
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    is_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    is_fitness: float = 0.0
    oos_fitness: float = 0.0
    is_returns_ann: float = 0.0
    oos_returns_ann: float = 0.0

    def to_dict(self) -> dict:
        return {
            "fold": self.fold_idx,
            "is_start": self.is_start, "is_end": self.is_end,
            "oos_start": self.oos_start, "oos_end": self.oos_end,
            "is_sharpe": self.is_sharpe, "oos_sharpe": self.oos_sharpe,
            "is_fitness": self.is_fitness, "oos_fitness": self.oos_fitness,
            "is_returns_ann": self.is_returns_ann,
            "oos_returns_ann": self.oos_returns_ann,
        }


# ---------------------------------------------------------------------------
# Fixed Split OOS
# ---------------------------------------------------------------------------

def fixed_split_oos(
    alpha_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    close_df: pd.DataFrame | None = None,
    open_df: pd.DataFrame | None = None,
    universe_df: pd.DataFrame | None = None,
    classifications: dict | None = None,
    split_date: str | pd.Timestamp | None = None,
    is_ratio: float = 0.7,
    booksize: float = 20_000_000.0,
    max_stock_weight: float = 0.01,
    decay: int = 0,
    delay: int = 1,
    neutralization: str = "market",
    fees_bps: float = 0.0,
) -> OOSResult:
    """
    Fixed date split: IS before split_date, OOS after.

    Args:
        alpha_df: Alpha signal (dates × tickers)
        returns_df: Forward returns
        split_date: Date to split at. If None, uses is_ratio.
        is_ratio: Fraction of dates for IS (if split_date is None)
    """
    common_idx = alpha_df.index.intersection(returns_df.index).sort_values()

    if split_date is None:
        split_pos = int(len(common_idx) * is_ratio)
        split_date = common_idx[split_pos]
    else:
        split_date = pd.Timestamp(split_date)

    is_dates = common_idx[common_idx < split_date]
    oos_dates = common_idx[common_idx >= split_date]

    if len(is_dates) < 60 or len(oos_dates) < 60:
        return OOSResult()  # Not enough data

    sim_kwargs = dict(
        close_df=close_df,
        open_df=open_df,
        universe_df=universe_df,
        classifications=classifications,
        booksize=booksize,
        max_stock_weight=max_stock_weight,
        decay=decay,
        delay=delay,
        neutralization=neutralization,
        fees_bps=fees_bps,
    )

    # IS simulation
    is_alpha = alpha_df.loc[is_dates]
    is_returns = returns_df.loc[is_dates]
    is_sim = simulate_vectorized(
        alpha_df=is_alpha, returns_df=is_returns, **sim_kwargs
    )

    # OOS simulation
    oos_alpha = alpha_df.loc[oos_dates]
    oos_returns = returns_df.loc[oos_dates]
    oos_sim = simulate_vectorized(
        alpha_df=oos_alpha, returns_df=oos_returns, **sim_kwargs
    )

    return _build_oos_result(is_sim, oos_sim, is_dates, oos_dates)


# ---------------------------------------------------------------------------
# Walk-Forward OOS
# ---------------------------------------------------------------------------

def walk_forward_oos(
    alpha_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    close_df: pd.DataFrame | None = None,
    open_df: pd.DataFrame | None = None,
    universe_df: pd.DataFrame | None = None,
    classifications: dict | None = None,
    n_folds: int = 5,
    is_window_days: int = 504,  # 2 years
    oos_window_days: int = 252,  # 1 year
    booksize: float = 20_000_000.0,
    max_stock_weight: float = 0.01,
    decay: int = 0,
    delay: int = 1,
    neutralization: str = "market",
    fees_bps: float = 0.0,
) -> OOSResult:
    """
    Walk-forward validation: rolling IS→OOS windows.

    Each fold uses `is_window_days` of IS data followed by
    `oos_window_days` of OOS data. Folds advance by `oos_window_days`.
    """
    common_idx = alpha_df.index.intersection(returns_df.index).sort_values()
    total_days = len(common_idx)

    min_required = is_window_days + oos_window_days
    if total_days < min_required:
        # Fall back to fixed split
        return fixed_split_oos(
            alpha_df, returns_df, close_df, open_df, universe_df, classifications,
            is_ratio=0.7, booksize=booksize, max_stock_weight=max_stock_weight,
            decay=decay, delay=delay, neutralization=neutralization,
            fees_bps=fees_bps,
        )

    sim_kwargs = dict(
        close_df=close_df,
        open_df=open_df,
        universe_df=universe_df,
        classifications=classifications,
        booksize=booksize,
        max_stock_weight=max_stock_weight,
        decay=decay,
        delay=delay,
        neutralization=neutralization,
        fees_bps=fees_bps,
    )

    folds: List[WalkForwardFold] = []
    oos_pnl_series = []

    # Compute actual folds
    max_folds = (total_days - is_window_days) // oos_window_days
    actual_folds = min(n_folds, max_folds)

    # Start from the end and work backward to use most recent data
    for i in range(actual_folds):
        oos_end_idx = total_days - i * oos_window_days
        oos_start_idx = oos_end_idx - oos_window_days
        is_start_idx = max(0, oos_start_idx - is_window_days)
        is_end_idx = oos_start_idx

        if is_end_idx - is_start_idx < 60:
            continue

        is_dates = common_idx[is_start_idx:is_end_idx]
        oos_dates = common_idx[oos_start_idx:oos_end_idx]

        is_sim = simulate_vectorized(
            alpha_df=alpha_df.loc[is_dates],
            returns_df=returns_df.loc[is_dates],
            **sim_kwargs,
        )
        oos_sim = simulate_vectorized(
            alpha_df=alpha_df.loc[oos_dates],
            returns_df=returns_df.loc[oos_dates],
            **sim_kwargs,
        )

        fold = WalkForwardFold(
            fold_idx=actual_folds - i,
            is_start=str(is_dates[0].date() if hasattr(is_dates[0], 'date') else is_dates[0]),
            is_end=str(is_dates[-1].date() if hasattr(is_dates[-1], 'date') else is_dates[-1]),
            oos_start=str(oos_dates[0].date() if hasattr(oos_dates[0], 'date') else oos_dates[0]),
            oos_end=str(oos_dates[-1].date() if hasattr(oos_dates[-1], 'date') else oos_dates[-1]),
            is_sharpe=is_sim.sharpe,
            oos_sharpe=oos_sim.sharpe,
            is_fitness=is_sim.fitness,
            oos_fitness=oos_sim.fitness,
            is_returns_ann=is_sim.returns_ann,
            oos_returns_ann=oos_sim.returns_ann,
        )
        folds.append(fold)
        oos_pnl_series.append(oos_sim.daily_pnl)

    folds.reverse()  # Chronological order

    # Aggregate: use first 70% as IS, last 30% as OOS for the summary
    split_pos = int(len(common_idx) * 0.7)
    is_dates_all = common_idx[:split_pos]
    oos_dates_all = common_idx[split_pos:]

    is_sim_all = simulate_vectorized(
        alpha_df=alpha_df.loc[is_dates_all],
        returns_df=returns_df.loc[is_dates_all],
        **sim_kwargs,
    )
    oos_sim_all = simulate_vectorized(
        alpha_df=alpha_df.loc[oos_dates_all],
        returns_df=returns_df.loc[oos_dates_all],
        **sim_kwargs,
    )

    result = _build_oos_result(is_sim_all, oos_sim_all, is_dates_all, oos_dates_all)
    result.walk_forward_folds = folds
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_oos_result(
    is_sim: VectorizedSimResult,
    oos_sim: VectorizedSimResult,
    is_dates: pd.DatetimeIndex,
    oos_dates: pd.DatetimeIndex,
) -> OOSResult:
    """Build an OOSResult from IS and OOS simulation results."""

    def _date_str(d):
        return str(d.date()) if hasattr(d, 'date') else str(d)

    # Decay ratios (how much does OOS degrade vs IS)
    sharpe_decay = (oos_sim.sharpe / is_sim.sharpe) if is_sim.sharpe != 0 else 0.0
    fitness_decay = (oos_sim.fitness / is_sim.fitness) if is_sim.fitness != 0 else 0.0
    return_decay = (oos_sim.returns_ann / is_sim.returns_ann) if is_sim.returns_ann != 0 else 0.0

    # Cap decay ratios
    for v in [sharpe_decay, fitness_decay, return_decay]:
        if math.isnan(v) or math.isinf(v):
            v = 0.0

    # Consistency: OOS Sharpe > 0 when IS Sharpe > 0
    is_consistent = (oos_sim.sharpe > 0) if is_sim.sharpe > 0 else True

    return OOSResult(
        is_sharpe=is_sim.sharpe,
        is_fitness=is_sim.fitness,
        is_turnover=is_sim.turnover,
        is_returns_ann=is_sim.returns_ann,
        is_max_drawdown=is_sim.max_drawdown,
        is_margin_bps=is_sim.margin_bps,
        is_total_pnl=is_sim.total_pnl,
        is_start=_date_str(is_dates[0]),
        is_end=_date_str(is_dates[-1]),
        is_days=len(is_dates),
        oos_sharpe=oos_sim.sharpe,
        oos_fitness=oos_sim.fitness,
        oos_turnover=oos_sim.turnover,
        oos_returns_ann=oos_sim.returns_ann,
        oos_max_drawdown=oos_sim.max_drawdown,
        oos_margin_bps=oos_sim.margin_bps,
        oos_total_pnl=oos_sim.total_pnl,
        oos_start=_date_str(oos_dates[0]),
        oos_end=_date_str(oos_dates[-1]),
        oos_days=len(oos_dates),
        sharpe_decay=sharpe_decay,
        fitness_decay=fitness_decay,
        return_decay=return_decay,
        is_consistent=is_consistent,
        is_sim=is_sim,
        oos_sim=oos_sim,
    )
