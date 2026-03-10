"""
Crypto Alpha Operators — vectorized implementations matching CryptoRL.

These are the reference implementations for evaluating alpha expressions
on DataFrames. All operators are lookahead-bias free.

Critical matching details vs CryptoRL (Operators class):
  - sma: min_periods=1 (not window/2)
  - stddev: min_periods=2
  - div: protected_div returning 0 for inf/NaN
  - ts_zscore: std.replace(0, nan) not std + epsilon
  - correlation: min_periods=2
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════
# Cross-Sectional Operators
# ═══════════════════════════════════════════════════════════════════

def rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank normalized to [0, 1]."""
    return df.rank(axis=1, pct=True)


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score: (x - mean) / std."""
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    return df.sub(mean, axis=0).div(std, axis=0)


def demean(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional demean: x - mean(x)."""
    return df.sub(df.mean(axis=1), axis=0)


# ═══════════════════════════════════════════════════════════════════
# Time-Series Operators
# ═══════════════════════════════════════════════════════════════════

def sma(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Simple moving average. min_periods=1 to match CryptoRL."""
    return df.rolling(window, min_periods=1).mean()


def stddev(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling standard deviation. min_periods=2 to match CryptoRL."""
    return df.rolling(window, min_periods=2).std()


def ts_min(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling minimum."""
    return df.rolling(window, min_periods=1).min()


def ts_max(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling maximum."""
    return df.rolling(window, min_periods=1).max()


def ts_sum(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling sum."""
    return df.rolling(window, min_periods=1).sum()


def ts_zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Time-series z-score. Uses std.replace(0, nan) to match CryptoRL."""
    mean = df.rolling(window, min_periods=2).mean()
    std = df.rolling(window, min_periods=2).std()
    return (df - mean) / std.replace(0, np.nan)


def correlation(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling correlation. min_periods=2 to match CryptoRL."""
    return x.rolling(window, min_periods=2).corr(y)


def delta(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Difference from period steps ago."""
    return df - df.shift(period)


def delay(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Lagged value from period steps ago."""
    return df.shift(period)


def ts_argmax(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Bars since rolling max."""
    return df.rolling(window, min_periods=1).apply(
        lambda x: window - 1 - np.argmax(x), raw=True
    )


def ts_argmin(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Bars since rolling min."""
    return df.rolling(window, min_periods=1).apply(
        lambda x: window - 1 - np.argmin(x), raw=True
    )


def ts_rank(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Time-series percentile rank of current value vs history."""
    def rank_last(x):
        if len(x) < 2:
            return 0.5
        return (x[:-1] < x[-1]).sum() / (len(x) - 1)
    return df.rolling(window, min_periods=2).apply(rank_last, raw=True)


# ═══════════════════════════════════════════════════════════════════
# Arithmetic Operators
# ═══════════════════════════════════════════════════════════════════

def sign(df):
    """Element-wise sign."""
    return np.sign(df)


def abs(df):
    """Element-wise absolute value."""
    return np.abs(df)


def div(a, b):
    """Protected division matching CryptoRL. Returns 0 for inf/NaN."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = a / b
        if isinstance(result, (pd.DataFrame, pd.Series)):
            result = result.replace([np.inf, -np.inf], 0.0)
            return result.fillna(0.0)
        return result


def mul(a, b):
    """Element-wise multiplication."""
    return a * b


# ═══════════════════════════════════════════════════════════════════
# Expression Evaluator
# ═══════════════════════════════════════════════════════════════════

def build_context(features: dict) -> dict:
    """Build evaluation context for alpha expressions."""
    return {
        "rank": rank, "sma": sma, "stddev": stddev,
        "ts_min": ts_min, "ts_max": ts_max, "ts_sum": ts_sum,
        "ts_zscore": ts_zscore, "correlation": correlation,
        "sign": sign, "abs": abs, "div": div, "mul": mul,
        "delta": delta, "delay": delay,
        "ts_argmax": ts_argmax, "ts_argmin": ts_argmin,
        "ts_rank": ts_rank, "zscore": zscore, "demean": demean,
        **features,
    }


def evaluate_expression(expr_str: str, features: dict) -> pd.DataFrame | None:
    """Evaluate an alpha expression string against feature DataFrames."""
    ctx = build_context(features)
    try:
        result = eval(expr_str, {"__builtins__": {}}, ctx)
        if isinstance(result, pd.DataFrame):
            return result
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════
# Alpha Definitions (56 proven expressions from CryptoRL)
# ═══════════════════════════════════════════════════════════════════

CRYPTO_ALPHA_DEFINITIONS = [
    # Momentum / Trend (long lookback)
    ("double_smooth_30_90",   "sma(sma(returns, 30), 90)"),
    ("double_smooth_45_120",  "sma(sma(returns, 45), 120)"),
    ("mom_90bar",             "sma(returns, 90)"),
    ("mom_120bar",            "sma(returns, 120)"),
    ("mom_180bar",            "sma(returns, 180)"),
    ("mom_240bar",            "sma(returns, 240)"),
    # Donchian Channels
    ("donchian_90",  "div(close - ts_min(close, 90), ts_max(close, 90) - ts_min(close, 90) + 1e-10)"),
    ("donchian_120", "div(close - ts_min(close, 120), ts_max(close, 120) - ts_min(close, 120) + 1e-10)"),
    ("donchian_180", "div(close - ts_min(close, 180), ts_max(close, 180) - ts_min(close, 180) + 1e-10)"),
    ("donchian_240", "div(close - ts_min(close, 240), ts_max(close, 240) - ts_min(close, 240) + 1e-10)"),
    # Trend Strength
    ("trend_strength_60",  "div(sma(returns, 60), sma(abs(returns), 60) + 1e-10)"),
    ("trend_strength_90",  "div(sma(returns, 90), sma(abs(returns), 90) + 1e-10)"),
    ("trend_strength_120", "div(sma(returns, 120), sma(abs(returns), 120) + 1e-10)"),
    # Momentum Diff (MACD-like)
    ("mom_diff_30_90",  "sma(returns, 30) - sma(returns, 90)"),
    ("mom_diff_60_180", "sma(returns, 60) - sma(returns, 180)"),
    ("mom_diff_90_270", "sma(returns, 90) - sma(returns, 270)"),
    # Risk-Adjusted Momentum (Sharpe-style)
    ("sharpe_60",  "div(sma(returns, 60), stddev(returns, 60) + 1e-10)"),
    ("sharpe_90",  "div(sma(returns, 90), stddev(returns, 90) + 1e-10)"),
    ("sharpe_120", "div(sma(returns, 120), stddev(returns, 120) + 1e-10)"),
    # Trend Vol-Adjusted
    ("trend_vol_adj_90",  "mul(sma(returns, 90), div(1, stddev(returns, 90) + 1e-10))"),
    ("trend_vol_adj_120", "mul(sma(returns, 120), div(1, stddev(returns, 120) + 1e-10))"),
    # Volume-Weighted Momentum
    ("vol_weighted_mom_60",  "sma(mul(returns, volume), 60)"),
    ("vol_weighted_mom_90",  "sma(mul(returns, volume), 90)"),
    ("vol_weighted_mom_120", "sma(mul(returns, volume), 120)"),
    # Short-Term Reversal
    ("reversal_1day", "-sma(returns, 6)"),
    ("reversal_2day", "-sma(returns, 12)"),
    ("reversal_3day", "-sma(returns, 18)"),
    ("reversal_4day", "-sma(returns, 24)"),
    ("reversal_5day", "-sma(returns, 30)"),
    # Z-Score Reversal
    ("zscore_rev_30", "-ts_zscore(close, 30)"),
    ("zscore_rev_45", "-ts_zscore(close, 45)"),
    ("zscore_rev_60", "-ts_zscore(close, 60)"),
    ("zscore_rev_90", "-ts_zscore(close, 90)"),
    # Buy the Dip
    ("buy_dip_30", "div(ts_max(high, 30) - close, ts_max(high, 30) + 1e-10)"),
    ("buy_dip_45", "div(ts_max(high, 45) - close, ts_max(high, 45) + 1e-10)"),
    ("buy_dip_60", "div(ts_max(high, 60) - close, ts_max(high, 60) + 1e-10)"),
    # Fade Big Moves
    ("fade_moves_1d", "-ts_max(abs(returns), 6)"),
    ("fade_moves_2d", "-ts_max(abs(returns), 12)"),
    ("fade_moves_3d", "-ts_max(abs(returns), 18)"),
    # Low Volatility Premium
    ("low_vol_60",  "-stddev(returns, 60)"),
    ("low_vol_90",  "-stddev(returns, 90)"),
    ("low_vol_120", "-stddev(returns, 120)"),
    # On-Balance Volume
    ("obv_60",  "ts_sum(mul(sign(returns), volume), 60)"),
    ("obv_90",  "ts_sum(mul(sign(returns), volume), 90)"),
    ("obv_120", "ts_sum(mul(sign(returns), volume), 120)"),
    # Correlation Based
    ("corr_ret_vol_60",  "correlation(returns, volume, 60)"),
    ("corr_ret_vol_90",  "correlation(returns, volume, 90)"),
    ("hl_corr_90",       "correlation(high, low, 90)"),
    ("hl_corr_120",      "correlation(high, low, 120)"),
    # Price vs Moving Average
    ("price_vs_ma120", "div(close, sma(close, 120))"),
    ("price_vs_ma180", "div(close, sma(close, 180))"),
    ("ma30_vs_ma90",   "div(sma(close, 30), sma(close, 90))"),
    ("ma60_vs_ma180",  "div(sma(close, 60), sma(close, 180))"),
    # Range / Candle Based
    ("close_location_60", "sma(div(close - low, high - low + 1e-10), 60)"),
    ("close_location_90", "sma(div(close - low, high - low + 1e-10), 90)"),
    ("oc_position_60",    "sma(div(close - open, high - low + 1e-10), 60)"),
]
