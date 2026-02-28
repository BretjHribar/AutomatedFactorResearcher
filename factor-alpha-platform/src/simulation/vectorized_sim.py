"""
Vectorized Simulation Engine — DataFrame-based backtester.

Operates on full DataFrames (dates × tickers) rather than per-date loops.
This is the fast path for evaluating alphas from the FastExpression engine
and GP engine, matching the original AWSgenProgWorker.py pattern exactly.

Usage:
    result = simulate_vectorized(
        alpha_df=engine.evaluate("rank(ts_delta(close, 60))"),
        returns_df=returns_matrix,
        booksize=20_000_000,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class VectorizedSimResult:
    """Result from vectorized simulation."""
    # Time series
    daily_pnl: pd.Series          # date → PnL
    daily_turnover: pd.Series     # date → turnover fraction
    daily_returns: pd.Series      # date → return as fraction of book
    positions: pd.DataFrame       # normalized positions (dates × tickers)
    cumulative_pnl: pd.Series     # cumulative PnL

    # Aggregate metrics
    sharpe: float = 0.0
    fitness: float = 0.0
    turnover: float = 0.0
    returns_ann: float = 0.0
    max_drawdown: float = 0.0
    margin_bps: float = 0.0
    psr: float = 0.0              # Probabilistic Sharpe Ratio
    total_pnl: float = 0.0


def simulate_vectorized(
    alpha_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    close_df: pd.DataFrame | None = None,
    universe_df: pd.DataFrame | None = None,
    groups: pd.Series | None = None,
    booksize: float = 20_000_000.0,
    max_stock_weight: float = 0.01,
    decay: int = 0,
    delay: int = 1,
    neutralization: str = "market",
    fees_bps: float = 0.0,
    min_price: float = -1.0,
    max_price: float = 1e7,
) -> VectorizedSimResult:
    """
    Vectorized simulation matching AWSgenProgWorker.py logic.

    Pipeline:
    1. Clean alpha (replace inf → 0)
    2. Apply delay (shift alpha forward)
    3. Apply optional linear decay
    4. Apply universe / price filters
    5. Apply neutralization (market demean or group demean)
    6. Clip to max stock weight
    7. Scale to booksize
    8. Compute PnL = positions * forward_returns
    9. Compute metrics

    Args:
        alpha_df: Raw alpha signal (dates × tickers)
        returns_df: Forward returns matrix (dates × tickers)
        close_df: Close prices for price-based filtering
        universe_df: Boolean DataFrame mask for universe membership
        groups: Series (ticker → group) for industry neutralization
        booksize: Dollar amount to allocate
        max_stock_weight: Max position weight per stock
        decay: Linear decay period (0 = no decay)
        delay: Signal delay in trading days
        neutralization: 'market', 'group', or 'none'
        fees_bps: Transaction fees in basis points
        min_price: Minimum stock price filter
        max_price: Maximum stock price filter
    """
    # Align columns
    common_cols = alpha_df.columns.intersection(returns_df.columns)
    common_idx = alpha_df.index.intersection(returns_df.index)
    alpha = alpha_df[common_cols].reindex(common_idx).copy()
    returns = returns_df[common_cols].reindex(common_idx)

    # Step 1: Clean
    alpha = alpha.replace([np.inf, -np.inf], 0.0)

    # Step 2: Delay
    if delay > 0:
        alpha = alpha.shift(delay)

    # Step 3: Linear decay
    if decay > 0:
        weights = np.arange(1, decay + 1, dtype=float)
        weights = weights / weights.sum()
        # Rolling weighted average
        alpha_decayed = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns)
        for i, w in enumerate(weights):
            alpha_decayed += alpha.shift(i).fillna(0) * w
        alpha = alpha_decayed

    # Step 4: Universe / price filters
    if close_df is not None:
        close_aligned = close_df[common_cols].reindex(common_idx)
        price_mask = (close_aligned < min_price) | (close_aligned > max_price)
        alpha[price_mask] = np.nan

    if universe_df is not None:
        uni_aligned = universe_df.reindex(index=common_idx, columns=common_cols)
        alpha[~uni_aligned.fillna(False).astype(bool)] = np.nan

    # Step 5: Neutralization
    if neutralization == "market":
        row_mean = alpha.mean(axis=1)
        alpha = alpha.sub(row_mean, axis=0)
    elif neutralization == "group" and groups is not None:
        for grp in groups.unique():
            tickers = groups[groups == grp].index.tolist()
            cols = [t for t in tickers if t in alpha.columns]
            if cols:
                grp_mean = alpha[cols].mean(axis=1)
                alpha[cols] = alpha[cols].sub(grp_mean, axis=0)

    # Step 6: Clip to max stock weight
    alpha = alpha.clip(lower=-max_stock_weight, upper=max_stock_weight)

    # Step 7: Scale to booksize
    abs_sum = alpha.abs().sum(axis=1).replace(0, np.nan)
    positions_normalized = alpha.div(abs_sum, axis=0)
    positions_money = positions_normalized * booksize

    # Step 8: Compute PnL
    daily_pnl_df = positions_money * returns
    turnover_adj = positions_money.diff().abs().sum(axis=1)

    daily_pnl = daily_pnl_df.sum(axis=1) - (turnover_adj * fees_bps / 10_000)
    daily_turnover = turnover_adj / booksize

    # Step 9: Metrics
    daily_returns = daily_pnl / (booksize * 0.5)
    cumulative_pnl = daily_pnl.cumsum()
    total_pnl = daily_pnl.sum()

    # Sharpe
    pnl_mean = daily_pnl.mean()
    pnl_std = daily_pnl.std()
    if pnl_std > 0 and not np.isnan(pnl_std):
        sharpe = (pnl_mean / pnl_std) * math.sqrt(252)
    else:
        sharpe = 0.0

    # Average turnover
    avg_turnover = daily_turnover.mean()

    # Annualized returns
    returns_ann = (pnl_mean * 252) / (booksize * 0.5) if booksize > 0 else 0.0

    # Fitness = sharpe * sqrt(|returns| / max(turnover, 0.125))
    if avg_turnover > 0.001:
        fitness = sharpe * math.sqrt(abs(returns_ann) / max(avg_turnover, 0.125))
    else:
        fitness = 0.0

    # Max drawdown (as fraction of booksize)
    running_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - running_max
    if booksize > 0:
        max_drawdown = (drawdown.min() / booksize) if len(drawdown) > 0 else 0.0
    else:
        max_drawdown = 0.0

    # Margin (bps) = total_pnl / total_shares_traded * 100
    if close_df is not None:
        close_aligned = close_df[common_cols].reindex(common_idx)
        shares_traded = (positions_money.diff().abs() / close_aligned.replace(0, np.nan)).sum().sum()
        margin_bps = (total_pnl / shares_traded * 100) if shares_traded > 0 else 0.0
    else:
        margin_bps = 0.0

    # Probabilistic Sharpe Ratio
    psr = _probabilistic_sharpe_ratio(daily_pnl)

    if math.isnan(sharpe):
        sharpe = 0.0
    if math.isnan(fitness):
        fitness = 0.0

    return VectorizedSimResult(
        daily_pnl=daily_pnl,
        daily_turnover=daily_turnover,
        daily_returns=daily_returns,
        positions=positions_normalized,
        cumulative_pnl=cumulative_pnl,
        sharpe=sharpe,
        fitness=fitness,
        turnover=avg_turnover,
        returns_ann=returns_ann,
        max_drawdown=max_drawdown,
        margin_bps=margin_bps,
        psr=psr,
        total_pnl=total_pnl,
    )


def _probabilistic_sharpe_ratio(daily_pnl: pd.Series, benchmark_sharpe: float = 0.0) -> float:
    """
    Probabilistic Sharpe Ratio (Bailey & López de Prado, 2012).

    Tests whether the observed Sharpe ratio is significantly greater than 
    a benchmark Sharpe ratio, accounting for skewness and kurtosis.
    """
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        return 0.5  # Can't compute without scipy

    n = len(daily_pnl.dropna())
    if n < 20:
        return 0.5

    sr = daily_pnl.mean() / daily_pnl.std() if daily_pnl.std() > 0 else 0.0
    sr_ann = sr * math.sqrt(252)
    bench = benchmark_sharpe / math.sqrt(252)

    skew = daily_pnl.skew()
    kurt = daily_pnl.kurtosis()

    # Standard error of Sharpe ratio
    se = math.sqrt((1 + 0.5 * sr**2 - skew * sr + (kurt / 4) * sr**2) / n)
    if se <= 0:
        return 0.5

    z = (sr - bench) / se
    return float(scipy_stats.norm.cdf(z))
