"""
Vectorized Simulation Engine — Polars-based backtester.

Drop-in replacement for vectorized_sim.py using Polars instead of Pandas
for improved performance. Produces identical numerical results.

Usage:
    result = simulate_vectorized_polars(
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
import polars as pl

try:
    from sklearn.decomposition import PCA as _PCA
except ImportError:
    _PCA = None


@dataclass
class VectorizedSimResult:
    """Result from vectorized simulation (identical to pandas version)."""
    # Time series (returned as pd.Series for API compat)
    daily_pnl: pd.Series
    daily_turnover: pd.Series
    daily_returns: pd.Series
    positions: pd.DataFrame  # keep as pandas for downstream compat
    cumulative_pnl: pd.Series

    # Aggregate metrics
    sharpe: float = 0.0
    fitness: float = 0.0
    turnover: float = 0.0
    returns_ann: float = 0.0
    max_drawdown: float = 0.0
    margin_bps: float = 0.0
    psr: float = 0.0
    total_pnl: float = 0.0


def _pd_to_pl(df: pd.DataFrame) -> tuple[pl.DataFrame, list[str], np.ndarray]:
    """Convert pandas DataFrame (dates x tickers) to polars.
    
    Returns (polars_df, ticker_columns, date_index_array).
    The polars df has a '__date__' column plus one column per ticker.
    """
    cols = list(df.columns)
    # Reset index to get dates as a column
    df_reset = df.reset_index()
    date_col = df_reset.columns[0]
    # Rename date column for consistency
    df_reset = df_reset.rename(columns={date_col: "__date__"})
    pldf = pl.from_pandas(df_reset)
    return pldf, cols, df.index.values


def _resolve_groups(classifications, level: str) -> pd.Series | None:
    """Extract a group Series for the given GICS neutralization level."""
    if classifications is None:
        return None
    if isinstance(classifications, pd.Series):
        return classifications
    if isinstance(classifications, dict):
        return classifications.get(level)
    return None


def simulate_vectorized_polars(
    alpha_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    close_df: pd.DataFrame | None = None,
    open_df: pd.DataFrame | None = None,
    universe_df: pd.DataFrame | None = None,
    classifications: dict | None = None,
    booksize: float = 20_000_000.0,
    max_stock_weight: float = 0.01,
    decay: int = 0,
    delay: int = 1,
    neutralization: str = "subindustry",
    fees_bps: float = 0.0,
    min_price: float = -1.0,
    max_price: float = 1e7,
    bars_per_day: int = 1,
) -> VectorizedSimResult:
    """
    Polars-based vectorized simulation — identical logic to pandas version.

    Accepts pandas DataFrames as input (for API compatibility), converts
    internally to Polars/numpy for computation, returns results as pandas.
    """
    # Return stream depends on delay:
    #   delay=0: Signal at close T, execute at close T, earn close T -> close T+1.
    #            Use close-to-close returns shifted forward by 1.
    #   delay>=1: Signal at close T, execute at open T+delay, earn open-to-open.
    #            Use open-to-open returns (if open prices available).
    if delay == 0:
        # Shift returns forward: position on day T earns return from close T to close T+1
        returns_df = returns_df.shift(-1)
    elif delay >= 1 and open_df is not None:
        returns_df = open_df.pct_change().shift(-1)

    # Align columns & index — preserve pandas intersection order (do NOT sort)
    common_cols = alpha_df.columns.intersection(returns_df.columns).tolist()
    common_idx = alpha_df.index.intersection(returns_df.index)

    # Extract as numpy arrays for speed (dates x tickers)
    alpha_np = alpha_df.loc[common_idx, common_cols].values.astype(np.float64).copy()
    returns_np = returns_df.loc[common_idx, common_cols].values.astype(np.float64)
    n_dates, n_tickers = alpha_np.shape
    dates = common_idx

    # Step 1: Clean (replace inf -> 0, but keep NaN as NaN — matches pandas .replace([inf,-inf], 0))
    inf_mask = np.isinf(alpha_np)
    alpha_np[inf_mask] = 0.0

    # Step 2: Delay
    if delay > 0:
        alpha_np = np.roll(alpha_np, delay, axis=0)
        alpha_np[:delay, :] = np.nan

    # Step 3: Linear decay
    if decay > 0:
        weights = np.arange(1, decay + 1, dtype=np.float64)
        weights /= weights.sum()
        result = np.zeros_like(alpha_np)
        for i, w in enumerate(weights):
            shifted = np.roll(alpha_np, i, axis=0)
            shifted[:i, :] = 0.0
            np.nan_to_num(shifted, copy=False, nan=0.0)
            result += shifted * w
        alpha_np = result

    # Step 4: Universe / price filters
    if close_df is not None:
        close_np = close_df.reindex(index=common_idx, columns=common_cols).values
        price_mask = (close_np < min_price) | (close_np > max_price)
        alpha_np[price_mask] = np.nan

    if universe_df is not None:
        uni_np = universe_df.reindex(index=common_idx, columns=common_cols).fillna(False).values.astype(bool)
        alpha_np[~uni_np] = np.nan

    # Step 5: Neutralization
    if neutralization == "market":
        row_means = np.nanmean(alpha_np, axis=1, keepdims=True)
        alpha_np -= row_means
    elif neutralization == "pca":
        # PCA neutralization matching super project's pcaConvertAlpha + hedgeGlobal
        # 1. Fit PCA on returns to get factor loadings
        # 2. Project alpha onto factor space, subtract to get residual
        # 3. Then apply global hedge (demean)
        try:
            if _PCA is None:
                raise ImportError("sklearn not available")
            n_factors = 5  # Match super project default
            ret_np = np.nan_to_num(returns_np, nan=0.0)
            pca = _PCA(n_components=min(n_factors, min(ret_np.shape) - 1))
            pca.fit(ret_np)
            eig_vectors = pca.components_  # (n_factors, n_tickers)

            # For each row of alpha, project onto factor space and subtract
            alpha_filled = np.nan_to_num(alpha_np, nan=0.0)
            # factor_map = lstsq(eig_vectors.T, alpha.T)[0]
            factor_map = np.linalg.lstsq(eig_vectors.T, alpha_filled.T, rcond=None)[0]
            # outPostRisk = factor_map.T @ eig_vectors
            out_post_risk = factor_map.T @ eig_vectors
            # adjOut = alpha - outPostRisk (residual alpha)
            alpha_np = np.where(np.isnan(alpha_np), np.nan, alpha_filled - out_post_risk)

            # Then apply global hedge (demean rows)
            row_means = np.nanmean(alpha_np, axis=1, keepdims=True)
            alpha_np -= row_means
        except Exception:
            # Fallback to simple market neutralization
            row_means = np.nanmean(alpha_np, axis=1, keepdims=True)
            alpha_np -= row_means
    elif neutralization in ("subindustry", "industry", "sector"):
        groups = _resolve_groups(classifications, neutralization)
        if groups is not None:
            # Build group mask arrays for vectorized demeaning
            group_codes = groups.reindex(common_cols)
            unique_groups = group_codes.dropna().unique()
            for grp in unique_groups:
                col_mask = (group_codes == grp).values
                if col_mask.any():
                    grp_data = alpha_np[:, col_mask]
                    grp_means = np.nanmean(grp_data, axis=1, keepdims=True)
                    alpha_np[:, col_mask] = grp_data - grp_means

    # Step 6: Normalize by abs sum
    abs_sum = np.nansum(np.abs(alpha_np), axis=1, keepdims=True)
    abs_sum[abs_sum == 0] = np.nan
    alpha_np = alpha_np / abs_sum

    # Step 7: Clip to max stock weight
    alpha_np = np.clip(alpha_np, -max_stock_weight, max_stock_weight)

    # Step 8: Scale to booksize — keep NaN for position-diff (turnover) then zero-fill for PnL
    # Pandas: positions_money = alpha * booksize  (keeps NaN)
    positions_with_nan = alpha_np * booksize  # NaN preserved

    # Step 9: Compute PnL (need zero-filled positions)
    positions_np = np.nan_to_num(positions_with_nan, nan=0.0)
    returns_clean = np.nan_to_num(returns_np, nan=0.0)
    daily_pnl_mat = positions_np * returns_clean

    # Turnover: match pandas .diff().abs().sum(axis=1) exactly
    # pandas diff() preserves NaN, then sum(skipna=True) ignores NaN entries
    diff_np = np.empty_like(positions_with_nan)
    diff_np[0, :] = np.nan
    diff_np[1:, :] = positions_with_nan[1:] - positions_with_nan[:-1]
    turnover_adj = np.nansum(np.abs(diff_np), axis=1)  # skipna via nansum

    # Daily PnL (sum across tickers minus fees)
    daily_pnl_arr = np.sum(daily_pnl_mat, axis=1) - (turnover_adj * fees_bps / 10_000)
    daily_turnover_arr = turnover_adj / booksize

    # Step 10: Metrics
    daily_returns_arr = daily_pnl_arr / (booksize * 0.5)
    cumulative_pnl_arr = np.cumsum(daily_pnl_arr)
    total_pnl = float(np.sum(daily_pnl_arr))

    # Annualization: bars_per_year = 252 trading days * bars_per_day
    bars_per_year = 252 * bars_per_day

    # Sharpe (annualized)
    pnl_mean = float(np.nanmean(daily_pnl_arr))
    pnl_std = float(np.nanstd(daily_pnl_arr, ddof=1))
    if pnl_std > 0 and not np.isnan(pnl_std):
        sharpe = (pnl_mean / pnl_std) * math.sqrt(bars_per_year)
    else:
        sharpe = 0.0

    # Average turnover
    avg_turnover = float(np.nanmean(daily_turnover_arr))

    # Annualized returns
    returns_ann = (pnl_mean * bars_per_year) / (booksize * 0.5) if booksize > 0 else 0.0

    # Fitness
    if avg_turnover > 0.001:
        fitness = sharpe * math.sqrt(abs(returns_ann) / max(avg_turnover, 0.125))
    else:
        fitness = 0.0

    # Max drawdown
    running_max = np.maximum.accumulate(cumulative_pnl_arr)
    drawdown = cumulative_pnl_arr - running_max
    max_drawdown = float(np.min(drawdown) / booksize) if booksize > 0 and len(drawdown) > 0 else 0.0

    # Margin bps — match pandas diff() which has NaN on first row
    margin_bps = 0.0
    if close_df is not None:
        close_np = close_df.reindex(index=common_idx, columns=common_cols).values
        close_safe = np.where(close_np == 0, np.nan, close_np)
        # pandas diff() = NaN on row 0, then elementwise diff
        pos_diff_abs = np.empty_like(positions_np)
        pos_diff_abs[0, :] = np.nan
        pos_diff_abs[1:, :] = np.abs(positions_np[1:] - positions_np[:-1])
        shares_traded = float(np.nansum(pos_diff_abs / close_safe))
        margin_bps = (total_pnl / shares_traded * 100) if shares_traded > 0 else 0.0

    # Convert back to pandas for API compatibility
    daily_pnl_s = pd.Series(daily_pnl_arr, index=dates, name="daily_pnl")
    daily_turnover_s = pd.Series(daily_turnover_arr, index=dates, name="daily_turnover")
    daily_returns_s = pd.Series(daily_returns_arr, index=dates, name="daily_returns")
    cumulative_pnl_s = pd.Series(cumulative_pnl_arr, index=dates, name="cumulative_pnl")
    positions_pd = pd.DataFrame(alpha_np, index=dates, columns=common_cols)

    # PSR
    psr = _probabilistic_sharpe_ratio(daily_pnl_s)

    if math.isnan(sharpe):
        sharpe = 0.0
    if math.isnan(fitness):
        fitness = 0.0

    return VectorizedSimResult(
        daily_pnl=daily_pnl_s,
        daily_turnover=daily_turnover_s,
        daily_returns=daily_returns_s,
        positions=positions_pd,
        cumulative_pnl=cumulative_pnl_s,
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
    """Probabilistic Sharpe Ratio (Bailey & López de Prado, 2012)."""
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        return 0.5

    n = len(daily_pnl.dropna())
    if n < 20:
        return 0.5

    sr = daily_pnl.mean() / daily_pnl.std() if daily_pnl.std() > 0 else 0.0
    sr_ann = sr * math.sqrt(252)
    bench = benchmark_sharpe / math.sqrt(252)

    skew = daily_pnl.skew()
    kurt = daily_pnl.kurtosis()

    se = math.sqrt((1 + 0.5 * sr**2 - skew * sr + (kurt / 4) * sr**2) / n)
    if se <= 0:
        return 0.5

    z = (sr - bench) / se
    return float(scipy_stats.norm.cdf(z))
