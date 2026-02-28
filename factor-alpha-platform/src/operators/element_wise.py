"""
Element-wise operators — operate on individual values.

Accept and return pandas Series or scalar values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def op_abs(x: pd.Series) -> pd.Series:
    """Absolute value."""
    return x.abs()


def sign(x: pd.Series) -> pd.Series:
    """Returns 1 if x > 0, -1 if x < 0, 0 if x == 0."""
    return np.sign(x)


def log(x: pd.Series) -> pd.Series:
    """Natural logarithm. Returns NaN for non-positive values."""
    return np.log(x.where(x > 0, np.nan))


def signed_power(x: pd.Series, e: float) -> pd.Series:
    """sign(x) * abs(x)^e"""
    return np.sign(x) * np.abs(x) ** e


def pasteurize(x: pd.Series, universe: list[str] | None = None) -> pd.Series:
    """
    Set to NaN if INF or if instrument not in universe.

    Removes infinite values and filters to universe membership.
    """
    result = x.replace([np.inf, -np.inf], np.nan)
    if universe is not None:
        # Zero out anything not in universe
        mask = result.index.isin(universe)
        result = result.where(mask, np.nan)
    return result


def op_min(x: pd.Series, y: pd.Series) -> pd.Series:
    """Parallel (element-wise) minimum of two vectors."""
    return pd.concat([x, y], axis=1).min(axis=1)


def op_max(x: pd.Series, y: pd.Series) -> pd.Series:
    """Parallel (element-wise) maximum of two vectors."""
    return pd.concat([x, y], axis=1).max(axis=1)


def tail(x: pd.Series, lower: float, upper: float, newval: float = 0.0) -> pd.Series:
    """Set x to newval if x is between lower and upper (inclusive)."""
    mask = (x >= lower) & (x <= upper)
    return x.where(~mask, newval)


def clamp(x: pd.Series, lower: float, upper: float) -> pd.Series:
    """Clip values to [lower, upper]."""
    return x.clip(lower=lower, upper=upper)


def step(n: float) -> float:
    """Return 1 if n >= 0, else 0. Used as a constant in expressions."""
    return 1.0 if n >= 0 else 0.0
