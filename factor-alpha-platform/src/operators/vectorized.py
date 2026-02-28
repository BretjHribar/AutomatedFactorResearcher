"""
Vectorized DataFrame operators — the core operator library.

Every operator takes and returns DataFrames (dates × tickers), matching
the WorldQuant BRAIN / WebSim fastexpression semantics AND the existing
GPfunctions.py from the original codebase exactly.

Design principle: ALL operations are vectorized pandas operations on full
DataFrames. No per-date loops. This is the key performance optimization
that makes the platform usable at scale.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


# ==========================================================================
# Time-Series Operators (operate down each column = per-instrument over time)
# ==========================================================================

def ts_sum(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Rolling sum over past `window` days."""
    return df.rolling(window, min_periods=1).sum()


def sma(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Simple moving average over past `window` days."""
    return df.rolling(window, min_periods=1).mean()


def ts_mean(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Alias for sma."""
    return sma(df, window)


def ts_rank(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Rolling rank as percentile within the past `window` days."""
    return df.rolling(window, min_periods=1).rank(pct=True)


def ts_min(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Rolling minimum over past `window` days."""
    return df.rolling(window, min_periods=1).min()


def ts_max(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Rolling maximum over past `window` days."""
    return df.rolling(window, min_periods=1).max()


def delta(df: pd.DataFrame, period: int = 2) -> pd.DataFrame:
    """x[t] - x[t - period]."""
    return df.diff(period)


def ts_delta(df: pd.DataFrame, period: int = 2) -> pd.DataFrame:
    """Alias for delta. Used in BRAIN fastexpression."""
    return delta(df, period)


def stddev(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Rolling standard deviation over past `window` days."""
    return df.rolling(window, min_periods=2).std()


def ts_std_dev(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Alias for stddev. Used in BRAIN fastexpression."""
    return stddev(df, window)


def correlation(x: pd.DataFrame, y: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Rolling correlation of x and y over past `window` days, per instrument."""
    if x is y or (isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame) and x.equals(y)):
        return pd.DataFrame(0, columns=x.columns, index=x.index)
    return x.rolling(window, min_periods=3).corr(y)


def ts_corr(x: pd.DataFrame, y: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Alias for correlation. Used in BRAIN fastexpression."""
    return correlation(x, y, window)


def covariance(x: pd.DataFrame, y: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Rolling covariance of x and y over past `window` days."""
    if x is y or (isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame) and x.equals(y)):
        return pd.DataFrame(0, columns=x.columns, index=x.index)
    return x.rolling(window, min_periods=3).cov(y)


def ts_cov(x: pd.DataFrame, y: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Alias for covariance."""
    return covariance(x, y, window)


def Product(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Rolling product over past `window` days."""
    return df.rolling(window, min_periods=1).apply(lambda na: np.prod(na), raw=True)


def delay(df: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    """Shift data by `period` days (positive = look back)."""
    return df.shift(period)


def ts_delay(df: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    """Alias for delay. Used in BRAIN fastexpression."""
    return delay(df, period)


def ArgMax(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Index of the maximum value within the past `window` days (1-indexed)."""
    return df.rolling(window, min_periods=1).apply(np.argmax, raw=True) + 1


def ts_argmax(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Alias for ArgMax."""
    return ArgMax(df, window)


def ArgMin(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Index of the minimum value within the past `window` days (1-indexed)."""
    return df.rolling(window, min_periods=1).apply(np.argmin, raw=True) + 1


def ts_argmin(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Alias for ArgMin."""
    return ArgMin(df, window)


def ts_skewness(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Rolling skewness over past `window` days."""
    return df.rolling(window, min_periods=3).skew()


def ts_kurtosis(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Rolling kurtosis over past `window` days."""
    return df.rolling(window, min_periods=4).kurt()


def ts_zscore(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Time-series z-score: (x - rolling_mean) / rolling_std.

    Uses shifted mean/std (look-back only, no look-ahead bias).
    Matches GPfunctions.ts_zscore exactly.
    """
    r = df.rolling(window=window, min_periods=2)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    return (df - m) / s


def ts_av_diff(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Time-series deviation from moving average: x - sma(x, window)."""
    return df - sma(df, window)


def ts_moment(df: pd.DataFrame, window: int = 10, k: int = 2) -> pd.DataFrame:
    """kth central moment over past `window` days."""
    m = df.rolling(window, min_periods=2).mean()
    return ((df - m) ** k).rolling(window, min_periods=2).mean()


def ts_entropy(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Rolling entropy approximation over past `window` days."""
    def _entropy(x):
        x = x[~np.isnan(x)]
        if len(x) < 2:
            return np.nan
        x = np.abs(x)
        total = np.sum(x)
        if total == 0:
            return 0.0
        p = x / total
        p = p[p > 0]
        return -np.sum(p * np.log(p))
    return df.rolling(window, min_periods=2).apply(_entropy, raw=True)


def ts_regression(y: pd.DataFrame, x: pd.DataFrame, window: int = 252,
                  lag: int = 0, rettype: int = 0) -> pd.DataFrame:
    """
    Time-series regression: y(t) = a + b * x(t - lag) over `window` days.

    rettype:
        0 = residual (error)
        1 = intercept (a)
        2 = slope (b)
        3 = fitted value (a + b * x)

    Matches BRAIN fastexpression ts_regression semantics.
    """
    if lag > 0:
        x = x.shift(lag)

    result = pd.DataFrame(np.nan, index=y.index, columns=y.columns)

    for col in y.columns:
        if col not in x.columns:
            continue
        yc = y[col]
        xc = x[col]

        for i in range(window, len(y)):
            y_win = yc.iloc[i - window:i].values
            x_win = xc.iloc[i - window:i].values

            # Remove NaN
            valid = ~(np.isnan(y_win) | np.isnan(x_win))
            if valid.sum() < 3:
                continue

            y_v = y_win[valid]
            x_v = x_win[valid]

            # OLS: y = a + b*x
            x_mean = np.mean(x_v)
            y_mean = np.mean(y_v)
            ss_xx = np.sum((x_v - x_mean) ** 2)
            if ss_xx == 0:
                continue
            b = np.sum((x_v - x_mean) * (y_v - y_mean)) / ss_xx
            a = y_mean - b * x_mean

            if rettype == 0:
                result.iat[i, result.columns.get_loc(col)] = yc.iloc[i] - (a + b * xc.iloc[i])
            elif rettype == 1:
                result.iat[i, result.columns.get_loc(col)] = a
            elif rettype == 2:
                result.iat[i, result.columns.get_loc(col)] = b
            elif rettype == 3:
                result.iat[i, result.columns.get_loc(col)] = a + b * xc.iloc[i]

    return result


def hump(df: pd.DataFrame, hump_val: float = 0.01) -> pd.DataFrame:
    """
    Smooth extreme values: attenuate values beyond hump_val standard deviations.
    """
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    upper = mean + hump_val * std
    lower = mean - hump_val * std
    return df.clip(lower=lower, upper=upper, axis=0)


# ==========================================================================
# Linear / Exponential Decay
# ==========================================================================

def _rolling_decay_lin(na: np.ndarray) -> float:
    """Linear decay weights: [1, 2, ..., n] applied to window."""
    weights = np.arange(1, na.size + 1)
    return np.sum(np.multiply(weights, na) / sum(weights))


def Decay_lin(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Linear decay weighted moving average.

    Matches GPfunctions.Decay_lin exactly.
    """
    return df.rolling(window, min_periods=1).apply(_rolling_decay_lin, raw=True)


def decay_linear(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Alias for Decay_lin."""
    return Decay_lin(df, window)


def Decay_exp(df: pd.DataFrame, alpha_exp: float = 0.99) -> pd.DataFrame:
    """
    Exponential decay (EWM) moving average.

    Matches GPfunctions.Decay_exp exactly.
    """
    if alpha_exp > 1 or alpha_exp < -1:
        alpha_exp = 0.99
    return pd.DataFrame(df).ewm(alpha=abs(alpha_exp)).mean()


def decay_exp(df: pd.DataFrame, alpha_exp: float = 0.99) -> pd.DataFrame:
    """Alias for Decay_exp."""
    return Decay_exp(df, alpha_exp)


# ==========================================================================
# Cross-Sectional Operators (operate across columns = across instruments)
# ==========================================================================

def rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional rank as percentile [0, 1].

    Matches GPfunctions.rank exactly: rank across instruments (axis=1).
    """
    df = pd.DataFrame(df)
    return df.rank(axis=1, pct=True)


def scale(df: pd.DataFrame, k: float = 1.0) -> pd.DataFrame:
    """Scale so sum(abs(x)) = k across instruments."""
    return df.mul(k).div(np.abs(df).sum(axis=1), axis=0)


def zscore_cs(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score: (x - mean) / std across instruments."""
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    return df.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)


def group_rank(df: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """
    Rank within each group (industry, subindustry, sector).

    groups: Series mapping ticker -> group_label.
    Returns percentile rank within each group.
    """
    result = df.copy()
    for group_name in groups.unique():
        tickers = groups[groups == group_name].index.tolist()
        cols_in_df = [t for t in tickers if t in df.columns]
        if len(cols_in_df) > 1:
            result[cols_in_df] = df[cols_in_df].rank(axis=1, pct=True)
    return result


def group_neutralize(df: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """
    Neutralize (demean) within each group.

    After neutralization, the mean within each group ≈ 0.

    groups: Series mapping ticker -> group_label (subindustry, industry, sector).
    """
    result = df.copy()
    for group_name in groups.unique():
        tickers = groups[groups == group_name].index.tolist()
        cols_in_df = [t for t in tickers if t in df.columns]
        if cols_in_df:
            group_mean = df[cols_in_df].mean(axis=1)
            result[cols_in_df] = df[cols_in_df].sub(group_mean, axis=0)
    return result


def market_neutralize(df: pd.DataFrame) -> pd.DataFrame:
    """Demean across all instruments (axis=1)."""
    return df.sub(df.mean(axis=1), axis=0)


def industry_neutralize(df: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Alias for group_neutralize."""
    return group_neutralize(df, groups)


# ==========================================================================
# Element-Wise / Arithmetic Operators
# ==========================================================================

def add(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Element-wise addition. Handles DataFrame + DataFrame and DataFrame + scalar."""
    return np.add(left, right)


def subtract(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Element-wise subtraction."""
    return np.subtract(left, right)


def multiply(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Element-wise multiplication."""
    return np.multiply(left, right)


def divide(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Element-wise division. Returns inf/nan for division by zero."""
    return np.divide(left, right)


def true_divide(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Element-wise true division."""
    return np.true_divide(left, right)


def protectedDiv(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Protected division: returns 0 where right == 0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(right != 0, np.divide(left, right), 0.0)
    if isinstance(left, pd.DataFrame):
        return pd.DataFrame(result, index=left.index, columns=left.columns)
    return result


def negative(df: pd.DataFrame) -> pd.DataFrame:
    """Element-wise negation."""
    return np.negative(df)


def Abs(df: pd.DataFrame) -> pd.DataFrame:
    """Element-wise absolute value. Matches GPfunctions.Abs."""
    return df.abs()


def abs_op(df: pd.DataFrame) -> pd.DataFrame:
    """Alias for Abs."""
    return Abs(df)


def Sign(df: pd.DataFrame) -> pd.DataFrame:
    """Element-wise sign (-1, 0, +1). Matches GPfunctions.Sign."""
    return np.sign(df)


def sign(df: pd.DataFrame) -> pd.DataFrame:
    """Alias for Sign."""
    return Sign(df)


def SignedPower(df: pd.DataFrame, y: float) -> pd.DataFrame:
    """abs(x)^y preserving sign. Matches GPfunctions.SignedPower."""
    return df.pow(abs(y))


def signed_power(df: pd.DataFrame, y: float) -> pd.DataFrame:
    """sign(x) * abs(x)^y — BRAIN semantics."""
    return np.sign(df) * np.abs(df) ** y


def Inverse(df: pd.DataFrame) -> pd.DataFrame:
    """1/x. Matches GPfunctions.Inverse."""
    return 1.0 / df


def inverse(df: pd.DataFrame) -> pd.DataFrame:
    """Alias for Inverse."""
    return Inverse(df)


def log(df: pd.DataFrame) -> pd.DataFrame:
    """Natural logarithm. Returns NaN for non-positive values."""
    return np.log(df.where(df > 0, np.nan))


def log10(df: pd.DataFrame) -> pd.DataFrame:
    """Base-10 logarithm."""
    return np.log10(df.where(df > 0, np.nan))


def sqrt(df: pd.DataFrame) -> pd.DataFrame:
    """Square root."""
    return np.sqrt(df.where(df >= 0, np.nan))


def square(df: pd.DataFrame) -> pd.DataFrame:
    """Element-wise square."""
    return np.square(df)


def log_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Log difference: x - x.shift(1). Matches GPfunctions.log_diff."""
    return df - df.shift(1)


def s_log_1p(df: pd.DataFrame) -> pd.DataFrame:
    """sign(x) * log(1 + |x|). Matches GPfunctions.s_log_1p."""
    return np.sign(df) * np.log1p(np.abs(df))


def Tail(df: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """Set values within [-cutoff, cutoff] to 0. Matches GPfunctions.Tail."""
    result = df.copy()
    result[(result > -cutoff) & (result < cutoff)] = 0
    return result


def tail(df: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """Alias for Tail."""
    return Tail(df, cutoff)


def df_max(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Element-wise maximum. Matches GPfunctions.df_max."""
    return left.where(left > right, right)


def df_min(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Element-wise minimum. Matches GPfunctions.df_min."""
    return left.where(left < right, right)


def if_else(cond: pd.DataFrame, true_val: pd.DataFrame, false_val: pd.DataFrame) -> pd.DataFrame:
    """Conditional: where cond is True use true_val, else false_val."""
    return true_val.where(cond.astype(bool), false_val)


def power(df: pd.DataFrame, exp: float) -> pd.DataFrame:
    """x ** exp."""
    return df ** exp


# Scalar-DataFrame arithmetic (matching the npf* functions from GPfunctions)
def npfadd(df: pd.DataFrame, f: float) -> pd.DataFrame:
    """DataFrame + scalar."""
    return np.add(df, f)


def npfsub(df: pd.DataFrame, f: float) -> pd.DataFrame:
    """DataFrame - scalar."""
    return np.subtract(df, f)


def npfmul(df: pd.DataFrame, f: float) -> pd.DataFrame:
    """DataFrame * scalar."""
    return np.multiply(df, f)


def npfdiv(df: pd.DataFrame, f: float) -> pd.DataFrame:
    """DataFrame / scalar."""
    return np.divide(df, f)


# ==========================================================================
# Misc / Identity
# ==========================================================================

def extend(i):
    """Identity function for type coercion in DEAP. Returns input unchanged."""
    return i


def pasteurize(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf/-inf with NaN."""
    return df.replace([np.inf, -np.inf], np.nan)


def winsorize(df: pd.DataFrame, limits: tuple[float, float] = (0.01, 0.99)) -> pd.DataFrame:
    """Winsorize each row to percentile limits."""
    lower = df.quantile(limits[0], axis=1)
    upper = df.quantile(limits[1], axis=1)
    return df.clip(lower=lower, upper=upper, axis=0)


def truncate(df: pd.DataFrame, max_weight: float) -> pd.DataFrame:
    """Clip absolute values to max_weight."""
    return df.clip(lower=-max_weight, upper=max_weight)


def ts_count_nans(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Count NaN values in rolling window."""
    return df.isna().rolling(window, min_periods=1).sum()


# Cross-sectional zscore that was mislabeled in original GPfunctions
def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Note: In the original GPfunctions.py, zscore was implemented as abs().
    This is the CORRECT cross-sectional z-score implementation.
    """
    return zscore_cs(df)


# ==========================================================================
# Additional BRAIN Operators (from the WorldQuant fastexpression catalog)
# ==========================================================================

def normalize(df: pd.DataFrame, use_std: bool = False, limit: float = 0.0) -> pd.DataFrame:
    """Cross-sectional normalize to [-1, 1] (or by std if use_std=True)."""
    if use_std:
        return zscore_cs(df)
    row_min = df.min(axis=1)
    row_max = df.max(axis=1)
    span = (row_max - row_min).replace(0, np.nan)
    result = 2 * (df.sub(row_min, axis=0)).div(span, axis=0) - 1
    if limit > 0:
        result = result.clip(-limit, limit)
    return result


def quantile(df: pd.DataFrame, driver: str = "gaussian", sigma: float = 1.0) -> pd.DataFrame:
    """Cross-sectional quantile transformation."""
    return rank(df)


def ts_backfill(df: pd.DataFrame, window: int = 30, k: int = 1) -> pd.DataFrame:
    """Forward-fill NaN values within a rolling window."""
    return df.ffill(limit=window)


def ts_quantile(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Rolling quantile (position as fraction of window)."""
    return ts_rank(df, window)


def ts_scale(df: pd.DataFrame, window: int = 10, constant: float = 0.0) -> pd.DataFrame:
    """Time-series scale: (x - ts_min) / (ts_max - ts_min) over window."""
    ts_lo = ts_min(df, window)
    ts_hi = ts_max(df, window)
    span = (ts_hi - ts_lo).replace(0, np.nan)
    return (df - ts_lo) / span + constant


def ts_product(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Alias for Product — BRAIN name."""
    return Product(df, window)


def kth_element(df: pd.DataFrame, window: int = 10, k: int = 1) -> pd.DataFrame:
    """Return the k-th element looking back in the window."""
    if k < 0 or k >= window:
        return pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    return df.shift(window - k - 1)


def last_diff_value(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Return the most recent value that differs from the current value within window."""
    result = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    for col in df.columns:
        series = df[col]
        for i in range(1, min(window + 1, len(series))):
            shifted = series.shift(i)
            mask = (series != shifted) & result[col].isna()
            result.loc[mask, col] = shifted[mask]
    return result


def ts_step(constant: float = 1.0) -> float:
    """Return a scalar constant — used in BRAIN as ts_step(1)."""
    return constant


def days_from_last_change(df: pd.DataFrame) -> pd.DataFrame:
    """Count the number of days since the last value change."""
    changed = df.diff().ne(0).astype(int)
    result = pd.DataFrame(0, index=df.index, columns=df.columns)
    for col in df.columns:
        counter = 0
        vals = []
        for c in changed[col]:
            if c == 1:
                counter = 0
            else:
                counter += 1
            vals.append(counter)
        result[col] = vals
    return result


def reverse(df: pd.DataFrame) -> pd.DataFrame:
    """Negate: -x. BRAIN name for negative()."""
    return -df


def is_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Return 1 where NaN, 0 otherwise."""
    return df.isna().astype(float)


def trade_when(cond: pd.DataFrame, alpha: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    """Conditional trade mask: use alpha when cond is True, else fallback."""
    return alpha.where(cond.astype(bool), fallback)


def bucket(df: pd.DataFrame, buckets: int = 10) -> pd.DataFrame:
    """Cross-sectional bucketing into N equal-frequency bins."""
    def _bucket_row(row):
        valid = row.dropna()
        if len(valid) < 2:
            return row
        try:
            return pd.qcut(row, q=buckets, labels=False, duplicates='drop') + 1
        except Exception:
            return row.rank(pct=True) * buckets
    return df.apply(_bucket_row, axis=1)


def group_mean(df: pd.DataFrame, groups: pd.Series, weight: pd.DataFrame = None) -> pd.DataFrame:
    """Weighted mean within each group."""
    result = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    for group_name in groups.unique():
        tickers = groups[groups == group_name].index.tolist()
        cols = [t for t in tickers if t in df.columns]
        if cols:
            if weight is not None:
                w_cols = [t for t in cols if t in weight.columns]
                if w_cols:
                    w = weight[w_cols]
                    wsum = w.sum(axis=1).replace(0, np.nan)
                    g_mean = (df[w_cols] * w).sum(axis=1) / wsum
                else:
                    g_mean = df[cols].mean(axis=1)
            else:
                g_mean = df[cols].mean(axis=1)
            for c in cols:
                result[c] = g_mean
    return result


def group_scale(df: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Scale within each group so sum(abs) = 1."""
    result = df.copy()
    for group_name in groups.unique():
        tickers = groups[groups == group_name].index.tolist()
        cols = [t for t in tickers if t in df.columns]
        if cols:
            abs_sum = df[cols].abs().sum(axis=1).replace(0, np.nan)
            result[cols] = df[cols].div(abs_sum, axis=0)
    return result


def group_zscore(df: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Z-score within each group."""
    result = df.copy()
    for group_name in groups.unique():
        tickers = groups[groups == group_name].index.tolist()
        cols = [t for t in tickers if t in df.columns]
        if len(cols) > 1:
            g_mean = df[cols].mean(axis=1)
            g_std = df[cols].std(axis=1).replace(0, np.nan)
            result[cols] = df[cols].sub(g_mean, axis=0).div(g_std, axis=0)
    return result


def group_backfill(df: pd.DataFrame, groups: pd.Series, window: int = 30) -> pd.DataFrame:
    """Forward-fill NaN values within each group."""
    result = df.copy()
    for group_name in groups.unique():
        tickers = groups[groups == group_name].index.tolist()
        cols = [t for t in tickers if t in df.columns]
        if cols:
            result[cols] = df[cols].ffill(limit=window)
    return result


def vec_avg(df: pd.DataFrame) -> pd.DataFrame:
    """Mean across all instruments (returns scalar per date, broadcast)."""
    m = df.mean(axis=1)
    return pd.DataFrame({c: m for c in df.columns}, index=df.index)


def vec_sum(df: pd.DataFrame) -> pd.DataFrame:
    """Sum across all instruments (returns scalar per date, broadcast)."""
    s = df.sum(axis=1)
    return pd.DataFrame({c: s for c in df.columns}, index=df.index)


def ts_decay_linear(df: pd.DataFrame, window: int = 10, dense: bool = False) -> pd.DataFrame:
    """BRAIN alias for Decay_lin / decay_linear."""
    return Decay_lin(df, window)


def ts_arg_max(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """BRAIN alias for ts_argmax."""
    return ts_argmax(df, window)


def ts_arg_min(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """BRAIN alias for ts_argmin."""
    return ts_argmin(df, window)


def ts_covariance(y: pd.DataFrame, x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """BRAIN alias for ts_cov (note: BRAIN uses y, x ordering)."""
    return ts_cov(x, y, window)
