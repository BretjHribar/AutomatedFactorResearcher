"""
Cross-sectional operators — operate across instruments on a single date.

All operators accept and return pandas Series (ticker -> value).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rank(x: pd.Series) -> pd.Series:
    """
    Rank values across instruments.

    Returns floats equally distributed in [0.0, 1.0].
    NaN values are preserved as NaN in output.
    Uses average ranking method for ties.
    """
    return x.rank(pct=True, method="average", na_option="keep")


def scale(x: pd.Series) -> pd.Series:
    """
    Scale so sum(abs(x)) = 1 (booksize = 1).

    NaN values are ignored in the denominator but preserved.
    """
    total_abs = x.abs().sum()
    if total_abs == 0 or np.isnan(total_abs):
        return x * 0.0
    return x / total_abs


def ind_neutralize(x: pd.Series, groups: pd.Series) -> pd.Series:
    """
    Demean x within each group (industry, subindustry, sector).

    After neutralization, the sum of alpha within each group ≈ 0.

    Args:
        x: Alpha values indexed by ticker.
        groups: Group labels indexed by ticker (same index as x).
    """
    return x.groupby(groups).transform(lambda g: g - g.mean())


def zscore(x: pd.Series) -> pd.Series:
    """
    Cross-sectional z-score: (x - mean) / std.

    NaN values are ignored in mean/std computation but preserved.
    """
    mean = x.mean()
    std = x.std()
    if std == 0 or np.isnan(std):
        return x * 0.0
    return (x - mean) / std


def winsorize(x: pd.Series, limits: tuple[float, float] = (0.01, 0.99)) -> pd.Series:
    """
    Clip values to percentile limits.

    Default: clip to [1st percentile, 99th percentile].
    """
    lower = x.quantile(limits[0])
    upper = x.quantile(limits[1])
    return x.clip(lower=lower, upper=upper)


def truncate(x: pd.Series, max_weight: float) -> pd.Series:
    """
    Truncate maximum absolute weight.

    Clips abs(x) to max_weight while preserving sign. Then re-scales.
    """
    return x.clip(lower=-max_weight, upper=max_weight)
