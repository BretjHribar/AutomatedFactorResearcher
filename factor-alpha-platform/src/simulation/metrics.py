"""
Metrics computation — all BRAIN-compatible performance metrics.

Every metric matches BRAIN/WebSim definitions exactly (Section 2.2 of spec).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.core.types import AnnualMetrics, SimSettings, get_sharpe_rating

TRADING_DAYS_PER_YEAR = 252


def compute_sharpe(daily_returns: np.ndarray) -> float:
    """
    Annualized Sharpe ratio.

    sharpe = mean(daily_return) / std(daily_return) * sqrt(252)
    """
    if len(daily_returns) < 2:
        return 0.0
    mean_ret = np.nanmean(daily_returns)
    std_ret = np.nanstd(daily_returns, ddof=1)
    if std_ret == 0 or np.isnan(std_ret):
        return 0.0
    ir = mean_ret / std_ret
    return float(ir * math.sqrt(TRADING_DAYS_PER_YEAR))


def compute_max_drawdown(cumulative_pnl: np.ndarray, equity: float) -> float:
    """
    Max drawdown as percentage of equity capital.

    cumulative_pnl[t] = sum(daily_pnl[0:t])
    running_max[t] = max(cumulative_pnl[0:t])
    drawdown[t] = running_max[t] - cumulative_pnl[t]
    max_drawdown = max(drawdown) / equity
    """
    if len(cumulative_pnl) == 0:
        return 0.0
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdown = running_max - cumulative_pnl
    max_dd = np.max(drawdown)
    if equity == 0:
        return 0.0
    return float(max_dd / equity)


def compute_pct_profitable(daily_pnl: np.ndarray) -> float:
    """Percent of days with positive PnL."""
    total = len(daily_pnl)
    if total == 0:
        return 0.0
    positive = np.sum(daily_pnl > 0)
    return float(positive / total)


def compute_daily_turnover(
    daily_positions: list[dict[str, float]],
    booksize: float,
) -> np.ndarray:
    """
    Compute daily turnover as fraction of booksize.

    daily_turnover[t] = sum_abs(position_change[t]) / booksize
    """
    turnovers = []
    for i in range(1, len(daily_positions)):
        old_pos = daily_positions[i - 1]
        new_pos = daily_positions[i]

        # All tickers in either old or new
        all_tickers = set(old_pos.keys()) | set(new_pos.keys())
        total_change = sum(
            abs(new_pos.get(t, 0.0) - old_pos.get(t, 0.0))
            for t in all_tickers
        )
        turnovers.append(total_change / booksize if booksize > 0 else 0.0)

    return np.array(turnovers)


def compute_fitness(sharpe: float, annual_return_pct: float, turnover: float) -> float:
    """
    BRAIN Fitness metric.

    fitness = sharpe * sqrt(abs(returnsPerc) / max(turnover, 0.125))

    Where returnsPerc is annual return in % (e.g., 10.39),
    turnover is daily turnover as decimal (e.g., 0.42).
    """
    turnover_capped = max(turnover, 0.125)
    return float(sharpe * math.sqrt(abs(annual_return_pct) / turnover_capped))


def compute_profit_per_dollar(total_pnl: float, total_dollars_traded: float) -> float:
    """
    Profit per dollar traded (in cents).

    profit_per_dollar = (total_pnl / total_dollars_traded) * 100  [cents]
    """
    if total_dollars_traded == 0:
        return 0.0
    return float((total_pnl / total_dollars_traded) * 100.0)


def compute_margin_bps(total_pnl: float, total_dollars_traded: float) -> float:
    """
    Margin in basis points.

    margin_bps = (total_pnl / total_dollars_traded) * 10000
    """
    if total_dollars_traded == 0:
        return 0.0
    return float((total_pnl / total_dollars_traded) * 10000.0)


def compute_total_dollars_traded(daily_positions: list[dict[str, float]]) -> float:
    """Total absolute position changes across all days and instruments."""
    total = 0.0
    for i in range(1, len(daily_positions)):
        old_pos = daily_positions[i - 1]
        new_pos = daily_positions[i]
        all_tickers = set(old_pos.keys()) | set(new_pos.keys())
        for t in all_tickers:
            total += abs(new_pos.get(t, 0.0) - old_pos.get(t, 0.0))
    return total


def compute_annual_metrics(
    dates: list,
    daily_pnl: np.ndarray,
    daily_positions: list[dict[str, float]],
    settings: SimSettings,
) -> tuple[list[AnnualMetrics], AnnualMetrics]:
    """
    Compute per-year and total metrics.

    Returns: (list of per-year AnnualMetrics, total AnnualMetrics)
    """
    equity = settings.booksize / 2.0
    booksize = settings.booksize

    # Group by year
    years: dict[int, list[int]] = {}
    for i, d in enumerate(dates):
        yr = d.year if hasattr(d, 'year') else pd.Timestamp(d).year
        if yr not in years:
            years[yr] = []
        years[yr].append(i)

    annual_results: list[AnnualMetrics] = []

    for year in sorted(years.keys()):
        indices = years[year]
        yr_pnl = daily_pnl[indices]
        yr_returns = yr_pnl / equity

        yr_total_pnl = float(np.sum(yr_pnl))
        yr_ann_return = yr_total_pnl / equity

        yr_sharpe = compute_sharpe(yr_returns)
        yr_cum_pnl = np.cumsum(yr_pnl)
        yr_max_dd = compute_max_drawdown(yr_cum_pnl, equity)
        yr_pct_profitable = compute_pct_profitable(yr_pnl)

        # Turnover for this year
        yr_positions = [daily_positions[i] for i in indices]
        yr_turnover_arr = compute_daily_turnover(yr_positions, booksize)
        yr_daily_turnover = float(np.mean(yr_turnover_arr)) if len(yr_turnover_arr) > 0 else 0.0

        # Total dollars traded this year
        yr_dollars_traded = compute_total_dollars_traded(yr_positions)
        yr_profit_per_dollar = compute_profit_per_dollar(yr_total_pnl, yr_dollars_traded)
        yr_margin_bps = compute_margin_bps(yr_total_pnl, yr_dollars_traded)

        yr_ann_return_pct = yr_ann_return * 100
        yr_fitness = compute_fitness(yr_sharpe, yr_ann_return_pct, yr_daily_turnover)

        annual_results.append(AnnualMetrics(
            year=year,
            booksize=booksize,
            pnl=yr_total_pnl,
            annual_return=yr_ann_return,
            sharpe=yr_sharpe,
            max_drawdown=yr_max_dd,
            pct_profitable_days=yr_pct_profitable,
            daily_turnover=yr_daily_turnover,
            profit_per_dollar=yr_profit_per_dollar,
            fitness=yr_fitness,
            margin_bps=yr_margin_bps,
            n_trading_days=len(indices),
        ))

    # --- Total metrics ---
    total_pnl = float(np.sum(daily_pnl))
    total_returns = daily_pnl / equity
    total_sharpe = compute_sharpe(total_returns)
    total_cum_pnl = np.cumsum(daily_pnl)
    total_max_dd = compute_max_drawdown(total_cum_pnl, equity)
    total_pct_profitable = compute_pct_profitable(daily_pnl)

    all_turnover = compute_daily_turnover(daily_positions, booksize)
    total_daily_turnover = float(np.mean(all_turnover)) if len(all_turnover) > 0 else 0.0

    total_dollars_traded = compute_total_dollars_traded(daily_positions)
    total_profit_per_dollar = compute_profit_per_dollar(total_pnl, total_dollars_traded)
    total_margin_bps = compute_margin_bps(total_pnl, total_dollars_traded)

    n_years = len(set(d.year if hasattr(d, 'year') else pd.Timestamp(d).year for d in dates))
    ann_return = total_pnl / equity / max(n_years, 1)
    ann_return_pct = ann_return * 100
    total_fitness = compute_fitness(total_sharpe, ann_return_pct, total_daily_turnover)

    total_metrics = AnnualMetrics(
        year="Total",
        booksize=booksize,
        pnl=total_pnl,
        annual_return=ann_return,
        sharpe=total_sharpe,
        max_drawdown=total_max_dd,
        pct_profitable_days=total_pct_profitable,
        daily_turnover=total_daily_turnover,
        profit_per_dollar=total_profit_per_dollar,
        fitness=total_fitness,
        margin_bps=total_margin_bps,
        n_trading_days=len(dates),
    )

    return annual_results, total_metrics
