"""
Neutralization methods — market, industry, subindustry, risk factor.

After neutralization, the sum of alpha within each group ≈ 0,
ensuring no net position in the group direction.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd

from src.operators.cross_sectional import ind_neutralize


def neutralize(
    alpha: pd.Series,
    method: str,
    ctx: Any,
    date: dt.date,
) -> pd.Series:
    """
    Neutralize alpha values by the specified method.

    Args:
        alpha: Alpha values indexed by ticker.
        method: 'none', 'market', 'industry', 'subindustry'
        ctx: DataContext for looking up classifications.
        date: Current date (for time-varying classifications).

    Returns:
        Neutralized alpha values.
    """
    if method == "none":
        return alpha

    if method == "market":
        return _market_neutralize(alpha)

    if method in ("industry", "subindustry", "sector"):
        return _group_neutralize(alpha, method, ctx, date)

    raise ValueError(f"Unknown neutralization method: {method}")


def _market_neutralize(alpha: pd.Series) -> pd.Series:
    """Demean across all instruments: alpha - mean(alpha)."""
    mean_val = alpha.mean()
    if np.isnan(mean_val):
        return alpha
    return alpha - mean_val


def _group_neutralize(
    alpha: pd.Series,
    level: str,
    ctx: Any,
    date: dt.date,
) -> pd.Series:
    """Demean within each industry/subindustry/sector group."""
    # Build group assignments
    groups = pd.Series(
        {ticker: ctx.get_industry(ticker, date, level=level) for ticker in alpha.index},
        dtype=str,
    )
    return ind_neutralize(alpha, groups)


def risk_neutralize(
    alpha: pd.Series,
    risk_factors: pd.DataFrame,
) -> pd.Series:
    """
    Regression-based neutralization against known risk factors.

    Regress alpha on risk factors and return residuals.

    Args:
        alpha: Alpha values indexed by ticker.
        risk_factors: DataFrame with tickers as index, risk factor columns.

    Returns:
        Residual alpha after removing risk factor exposures.
    """
    # Align indices
    common = alpha.index.intersection(risk_factors.index)
    if len(common) < 10:
        return alpha

    y = alpha.loc[common].values
    X = risk_factors.loc[common].values

    # Remove NaN
    valid = ~(np.isnan(y) | np.isnan(X).any(axis=1))
    if valid.sum() < 10:
        return alpha

    y_clean = y[valid]
    X_clean = X[valid]

    # Add intercept
    X_with_intercept = np.column_stack([np.ones(len(X_clean)), X_clean])

    # OLS
    try:
        beta, _, _, _ = np.linalg.lstsq(X_with_intercept, y_clean, rcond=None)
        predicted = X_with_intercept @ beta
        residuals = y_clean - predicted
    except np.linalg.LinAlgError:
        return alpha

    # Map back
    result = alpha.copy()
    valid_tickers = common[valid]
    result.loc[valid_tickers] = residuals
    return result
