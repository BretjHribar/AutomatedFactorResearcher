"""
Core Simulation Engine — the backtester.

Implements the exact BRAIN/WebSim simulation loop:
1. Compute raw alpha values
2. Pasteurize (remove INF, filter to universe)
3. Apply decay (if configured)
4. Apply neutralization
5. Apply max instrument weight constraint
6. Scale to booksize
7. Compute daily PnL from position changes
"""

from __future__ import annotations

import datetime as dt
import math
from collections import deque
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.core.types import (
    DailyResult,
    FactorDef,
    SimResult,
    SimSettings,
)
from src.operators.element_wise import pasteurize
from src.simulation.metrics import compute_annual_metrics
from src.simulation.neutralization import neutralize


def simulate(
    factor: FactorDef,
    ctx: Any,
    settings: SimSettings | None = None,
) -> SimResult:
    """
    Core simulation loop. Matches WebSim behavior exactly.

    Args:
        factor: Factor definition with compute_fn.
        ctx: DataContext implementation.
        settings: Simulation parameters (booksize, delay, etc.)

    Returns:
        SimResult with all BRAIN-compatible metrics.
    """
    if settings is None:
        settings = SimSettings()

    # Determine date range
    if settings.start_date and settings.end_date:
        start, end = settings.start_date, settings.end_date
    else:
        trading_days = ctx.get_trading_days(
            dt.date(2000, 1, 1), dt.date(2099, 12, 31)
        )
        if not trading_days:
            return _empty_result(factor.factor_id, settings)
        # Use last `duration_years` of data
        end = trading_days[-1]
        target_start = dt.date(end.year - settings.duration_years, end.month, end.day)
        start = max(trading_days[0], target_start)

    trading_days = ctx.get_trading_days(start, end)
    if len(trading_days) < 2:
        return _empty_result(factor.factor_id, settings)

    # Need lookback buffer for time-series operators
    buffer_start = start - dt.timedelta(days=int(factor.lookback_days * 1.5))
    all_days = ctx.get_trading_days(buffer_start, end)

    # --- Simulation Loop ---
    positions: dict[str, float] = {}
    results: list[DailyResult] = []
    alpha_history: deque[dict[str, float]] = deque(maxlen=max(factor.decay, 1) + 1)

    for date in trading_days:
        # 1. Compute signal date (apply delay)
        signal_date = _get_signal_date(date, factor.delay, all_days)
        if signal_date is None:
            continue

        # 2. Compute raw alpha values
        try:
            raw_alpha = factor.compute_fn(signal_date, ctx)
        except Exception:
            raw_alpha = {}

        if not raw_alpha:
            # No signal — maintain previous positions
            daily_pnl = _compute_daily_pnl(positions, ctx, date)
            results.append(DailyResult(
                date=date, pnl=daily_pnl, positions=dict(positions),
                turnover=0.0, alpha_values=raw_alpha,
            ))
            continue

        # Convert to Series for vectorized operations
        alpha = pd.Series(raw_alpha)

        # 3. Pasteurize — remove INF, filter to universe
        universe = ctx.get_universe(date, settings.universe)
        alpha = pasteurize(alpha, universe)
        alpha = alpha.dropna()

        if alpha.empty:
            daily_pnl = _compute_daily_pnl(positions, ctx, date)
            results.append(DailyResult(
                date=date, pnl=daily_pnl, positions=dict(positions),
                turnover=0.0, alpha_values={},
            ))
            continue

        # 4. Apply decay (linear blend with historical alpha)
        alpha_history.append(alpha.to_dict())
        if factor.decay > 0 and len(alpha_history) > 1:
            alpha = _apply_decay(alpha_history, factor.decay)

        # 5. Neutralize
        if settings.neutralization != "none":
            alpha = neutralize(alpha, settings.neutralization, ctx, date)

        # 6. Apply max instrument weight constraint
        if settings.max_instrument_weight > 0:
            alpha = _apply_weight_cap(alpha, settings.max_instrument_weight)

        # 7. Scale to booksize
        new_positions = _scale_to_booksize(alpha, settings.booksize)

        # 8. Compute daily PnL (from holding PREVIOUS positions through today's returns)
        daily_pnl = _compute_daily_pnl(positions, ctx, date)

        # 9. Compute turnover
        turnover = _compute_turnover(positions, new_positions, settings.booksize)

        # 10. Record
        results.append(DailyResult(
            date=date,
            pnl=daily_pnl,
            positions=new_positions,
            turnover=turnover,
            alpha_values=alpha.to_dict(),
        ))

        positions = new_positions

    # --- Build SimResult ---
    if not results:
        return _empty_result(factor.factor_id, settings)

    dates = [r.date for r in results]
    daily_pnl_arr = np.array([r.pnl for r in results])
    daily_positions_list = [r.positions for r in results]
    daily_turnover_list = [r.turnover for r in results]

    annual_metrics, total_metrics = compute_annual_metrics(
        dates, daily_pnl_arr, daily_positions_list, settings
    )

    return SimResult(
        dates=dates,
        daily_pnl=daily_pnl_arr.tolist(),
        daily_positions=daily_positions_list,
        daily_turnover=daily_turnover_list,
        annual_metrics=annual_metrics,
        total_metrics=total_metrics,
        factor_id=factor.factor_id,
        settings=settings,
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_signal_date(
    date: dt.date, delay: int, all_days: list[dt.date]
) -> dt.date | None:
    """Find the signal date by looking back `delay` trading days."""
    try:
        idx = all_days.index(date)
    except ValueError:
        return None
    target_idx = idx - delay
    if target_idx < 0:
        return None
    return all_days[target_idx]


def _scale_to_booksize(alpha: pd.Series, booksize: float) -> dict[str, float]:
    """Scale weights so sum(abs(positions)) = booksize."""
    total_abs = alpha.abs().sum()
    if total_abs == 0 or np.isnan(total_abs):
        return {}
    scale_factor = booksize / total_abs
    scaled = alpha * scale_factor
    return {k: float(v) for k, v in scaled.items() if not np.isnan(v)}


def _compute_daily_pnl(
    positions: dict[str, float], ctx: Any, date: dt.date
) -> float:
    """
    PnL from holding positions through today's return.

    PnL = sum(position * daily_return) for all positions.
    """
    if not positions:
        return 0.0

    pnl = 0.0
    for ticker, pos in positions.items():
        daily_ret = ctx.get_price(ticker, "returns", date)
        if not math.isnan(daily_ret) and not math.isnan(pos):
            pnl += pos * daily_ret
    return pnl


def _compute_turnover(
    old_pos: dict[str, float],
    new_pos: dict[str, float],
    booksize: float,
) -> float:
    """Compute turnover as fraction of booksize."""
    if booksize == 0:
        return 0.0

    all_tickers = set(old_pos.keys()) | set(new_pos.keys())
    total_change = sum(
        abs(new_pos.get(t, 0.0) - old_pos.get(t, 0.0))
        for t in all_tickers
    )
    return total_change / booksize


def _apply_decay(
    alpha_history: deque[dict[str, float]], decay: int
) -> pd.Series:
    """
    Apply linear decay blending over recent alpha values.

    Blend current alpha with past alphas using linearly declining weights.
    """
    n = min(len(alpha_history), decay + 1)
    weights = np.arange(1, n + 1, dtype=float)  # oldest=1, newest=n
    weight_sum = weights.sum()

    # Collect all tickers
    all_tickers: set[str] = set()
    for h in alpha_history:
        all_tickers.update(h.keys())

    result: dict[str, float] = {t: 0.0 for t in all_tickers}
    history_list = list(alpha_history)[-n:]

    for i, alpha_dict in enumerate(history_list):
        w = weights[i]
        for ticker, val in alpha_dict.items():
            if not math.isnan(val):
                result[ticker] += val * w

    # Normalize
    for ticker in result:
        result[ticker] /= weight_sum

    return pd.Series(result)


def _apply_weight_cap(alpha: pd.Series, max_weight: float) -> pd.Series:
    """
    Cap maximum instrument weight.

    Clips abs(alpha) so no single instrument exceeds max_weight of total.
    """
    total_abs = alpha.abs().sum()
    if total_abs == 0:
        return alpha

    weights = alpha / total_abs
    capped = weights.clip(lower=-max_weight, upper=max_weight)
    return capped * total_abs


def _empty_result(factor_id: str, settings: SimSettings) -> SimResult:
    """Return an empty SimResult."""
    from src.core.types import AnnualMetrics

    empty_metrics = AnnualMetrics(
        year="Total", booksize=settings.booksize, pnl=0.0,
        annual_return=0.0, sharpe=0.0, max_drawdown=0.0,
        pct_profitable_days=0.0, daily_turnover=0.0,
        profit_per_dollar=0.0, fitness=0.0, margin_bps=0.0,
    )
    return SimResult(
        dates=[], daily_pnl=[], daily_positions=[], daily_turnover=[],
        annual_metrics=[], total_metrics=empty_metrics,
        factor_id=factor_id, settings=settings,
    )
