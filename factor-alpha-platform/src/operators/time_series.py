"""
Time-series operators — operate across time for each instrument.

Input: DataFrame (dates × tickers) — a "matrix" of historical values.
Output: Series (ticker → value) — the computed value for the most recent date.

All operators use the full matrix but return only the latest cross-section.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis as scipy_kurtosis


def delay(x: pd.DataFrame, n: int) -> pd.Series:
    """Value of x at n days ago. n < 256."""
    if len(x) <= n:
        return pd.Series(np.nan, index=x.columns)
    return x.iloc[-(n + 1)]


def delta(x: pd.DataFrame, n: int) -> pd.Series:
    """x[today] - x[today - n]."""
    if len(x) <= n:
        return pd.Series(np.nan, index=x.columns)
    return x.iloc[-1] - x.iloc[-(n + 1)]


def ts_sum(x: pd.DataFrame, n: int) -> pd.Series:
    """Sum of x over past n days (including today)."""
    return x.iloc[-n:].sum()


def ts_mean(x: pd.DataFrame, n: int) -> pd.Series:
    """Mean of x over past n days."""
    return x.iloc[-n:].mean()


def ts_std(x: pd.DataFrame, n: int) -> pd.Series:
    """Standard deviation of x over past n days."""
    return x.iloc[-n:].std()


def ts_min(x: pd.DataFrame, n: int) -> pd.Series:
    """Min of x over past n days."""
    return x.iloc[-n:].min()


def ts_max(x: pd.DataFrame, n: int) -> pd.Series:
    """Max of x over past n days."""
    return x.iloc[-n:].max()


def ts_rank(x: pd.DataFrame, n: int) -> pd.Series:
    """
    Rank current value within past n days.

    Returns value in [0.0, 1.0] for each instrument.
    1.0 = current value is the maximum over the window.
    0.0 = current value is the minimum over the window.
    """
    window = x.iloc[-n:]
    current = x.iloc[-1]

    result = pd.Series(index=x.columns, dtype=float)
    for col in x.columns:
        vals = window[col].dropna()
        if len(vals) < 2:
            result[col] = np.nan
        else:
            curr_val = current[col]
            if np.isnan(curr_val):
                result[col] = np.nan
            else:
                rank_val = (vals < curr_val).sum()
                result[col] = rank_val / (len(vals) - 1) if len(vals) > 1 else 0.5
    return result


def ts_skewness(x: pd.DataFrame, n: int) -> pd.Series:
    """Skewness of x over past n days."""
    window = x.iloc[-n:]
    return window.apply(lambda col: skew(col.dropna()) if col.dropna().shape[0] >= 3 else np.nan)


def ts_kurtosis(x: pd.DataFrame, n: int) -> pd.Series:
    """Kurtosis of x over past n days."""
    window = x.iloc[-n:]
    return window.apply(
        lambda col: scipy_kurtosis(col.dropna(), fisher=True)
        if col.dropna().shape[0] >= 4
        else np.nan
    )


def ts_moment(x: pd.DataFrame, k: int, n: int) -> pd.Series:
    """kth central moment of x over past n days."""
    window = x.iloc[-n:]
    mean_vals = window.mean()
    return ((window - mean_vals) ** k).mean()


def correlation(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.Series:
    """Correlation of x and y over past n days, per instrument."""
    x_win = x.iloc[-n:]
    y_win = y.iloc[-n:]

    # Align columns
    common = x_win.columns.intersection(y_win.columns)
    result = pd.Series(index=common, dtype=float)

    for col in common:
        xv = x_win[col].dropna()
        yv = y_win[col].dropna()
        # Align on common indices
        common_idx = xv.index.intersection(yv.index)
        if len(common_idx) < 3:
            result[col] = np.nan
        else:
            result[col] = xv.loc[common_idx].corr(yv.loc[common_idx])

    return result


def covariance(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.Series:
    """Covariance of x and y over past n days, per instrument."""
    x_win = x.iloc[-n:]
    y_win = y.iloc[-n:]

    common = x_win.columns.intersection(y_win.columns)
    result = pd.Series(index=common, dtype=float)

    for col in common:
        xv = x_win[col].dropna()
        yv = y_win[col].dropna()
        common_idx = xv.index.intersection(yv.index)
        if len(common_idx) < 2:
            result[col] = np.nan
        else:
            result[col] = xv.loc[common_idx].cov(yv.loc[common_idx])

    return result


def decay_linear(x: pd.DataFrame, n: int) -> pd.Series:
    """
    Linear decay weighted average over past n days.

    Weight for i days ago = (n - i).
    decay_linear(x, 3) = (x[t]*3 + x[t-1]*2 + x[t-2]*1) / (3+2+1)
    """
    window = x.iloc[-n:]
    weights = np.arange(1, n + 1, dtype=float)  # [1, 2, ..., n]
    weight_sum = weights.sum()

    result = pd.Series(0.0, index=x.columns)
    for i, (_, row) in enumerate(window.iterrows()):
        result += row.fillna(0) * weights[i]
    return result / weight_sum


def decay_exp(x: pd.DataFrame, f: float, n: int) -> pd.Series:
    """
    Exponential decay over past n days with factor f.

    Weight for i days ago = f^i. Most recent day has weight f^0 = 1.
    """
    window = x.iloc[-n:]
    weights = np.array([f ** (n - 1 - i) for i in range(n)])
    weight_sum = weights.sum()

    result = pd.Series(0.0, index=x.columns)
    for i, (_, row) in enumerate(window.iterrows()):
        result += row.fillna(0) * weights[i]
    return result / weight_sum


def product(x: pd.DataFrame, n: int) -> pd.Series:
    """Product of x over past n days."""
    window = x.iloc[-n:]
    return window.prod()


def count_nans(x: pd.DataFrame, n: int) -> pd.Series:
    """Count of NaN values in x over past n days."""
    window = x.iloc[-n:]
    return window.isna().sum()
