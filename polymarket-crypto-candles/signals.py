"""
signals.py — Signal computation library for Polymarket crypto candle prediction.

Each signal function takes a DataFrame of kline data and returns a Series
of signal values, where positive = bullish (predict UP) and negative = bearish (predict DOWN).

All signals are computed using ONLY data available BEFORE the candle they predict.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Callable


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_div(a, b, fill=0.0):
    """Safe division avoiding div-by-zero."""
    result = np.where(np.abs(b) > 1e-10, a / b, fill)
    return result


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score."""
    mean = series.rolling(window, min_periods=max(window // 2, 2)).mean()
    std = series.rolling(window, min_periods=max(window // 2, 2)).std()
    return (series - mean) / std.replace(0, np.nan)


# ============================================================================
# PRICE-BASED SIGNALS
# ============================================================================

def sig_momentum(df: pd.DataFrame, lookback: int = 12) -> pd.Series:
    """Price momentum: log(close / close[lookback])."""
    return np.log(df["close"] / df["close"].shift(lookback))


def sig_momentum_smooth(df: pd.DataFrame, lookback: int = 12, smooth: int = 3) -> pd.Series:
    """Smoothed price momentum."""
    mom = np.log(df["close"] / df["close"].shift(lookback))
    return mom.rolling(smooth, min_periods=1).mean()


def sig_return_sma(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Simple moving average of returns."""
    returns = df["close"].pct_change()
    return returns.rolling(window, min_periods=max(window // 2, 2)).mean()


def sig_return_zscore(df: pd.DataFrame, window: int = 30) -> pd.Series:
    """Z-score of close price."""
    return rolling_zscore(df["close"], window)


def sig_close_position_in_range(df: pd.DataFrame, window: int = 1) -> pd.Series:
    """Where the close sits in the high-low range (0-1). >0.5 = bullish."""
    if window == 1:
        hl_range = df["high"] - df["low"]
        pos = safe_div((df["close"] - df["low"]).values, hl_range.values, 0.5)
        return pd.Series(pos - 0.5, index=df.index)  # Center at 0
    else:
        hl_range = df["high"] - df["low"]
        pos = safe_div((df["close"] - df["low"]).values, hl_range.values, 0.5)
        return pd.Series(pos - 0.5, index=df.index).rolling(window, min_periods=1).mean()


def sig_candle_body_ratio(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Signed candle body / total range, smoothed. Positive = bullish bodies."""
    body = df["close"] - df["open"]
    hl_range = df["high"] - df["low"]
    ratio = safe_div(body.values, hl_range.values, 0.0)
    return pd.Series(ratio, index=df.index).rolling(window, min_periods=1).mean()


def sig_upper_shadow_ratio(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Upper shadow relative to range, smoothed. High = rejection of highs (bearish)."""
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    hl_range = df["high"] - df["low"]
    ratio = safe_div(upper.values, hl_range.values, 0.0)
    return -pd.Series(ratio, index=df.index).rolling(window, min_periods=1).mean()


def sig_lower_shadow_ratio(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Lower shadow relative to range, smoothed. High = rejection of lows (bullish)."""
    lower = df[["open", "close"]].min(axis=1) - df["low"]
    hl_range = df["high"] - df["low"]
    ratio = safe_div(lower.values, hl_range.values, 0.0)
    return pd.Series(ratio, index=df.index).rolling(window, min_periods=1).mean()


def sig_price_vs_sma(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Close price relative to SMA. >0 = above SMA (bullish)."""
    sma = df["close"].rolling(window, min_periods=max(window // 2, 2)).mean()
    return (df["close"] - sma) / sma


def sig_donchian_position(df: pd.DataFrame, window: int = 30) -> pd.Series:
    """Position in Donchian channel (0-1). Center at 0."""
    high_max = df["high"].rolling(window, min_periods=max(window // 2, 2)).max()
    low_min = df["low"].rolling(window, min_periods=max(window // 2, 2)).min()
    ch_range = high_max - low_min
    pos = safe_div((df["close"] - low_min).values, ch_range.values, 0.5)
    return pd.Series(pos - 0.5, index=df.index)


def sig_trend_strength(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Trend strength: mean(returns) / mean(abs(returns)). [-1, 1]."""
    returns = df["close"].pct_change()
    mean_ret = returns.rolling(window, min_periods=max(window // 2, 2)).mean()
    mean_abs = returns.abs().rolling(window, min_periods=max(window // 2, 2)).mean()
    return safe_div(mean_ret.values, mean_abs.values, 0.0)


def sig_macd_signal(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    """MACD-style: fast SMA of returns - slow SMA of returns."""
    returns = df["close"].pct_change()
    fast_sma = returns.rolling(fast, min_periods=max(fast // 2, 2)).mean()
    slow_sma = returns.rolling(slow, min_periods=max(slow // 2, 2)).mean()
    return fast_sma - slow_sma


# ============================================================================
# VOLUME-BASED SIGNALS
# ============================================================================

def sig_volume_ratio(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Current volume relative to rolling average."""
    avg_vol = df["volume"].rolling(window, min_periods=max(window // 2, 2)).mean()
    ratio = safe_div(df["volume"].values, avg_vol.values, 1.0)
    return pd.Series(ratio - 1.0, index=df.index)  # Center at 0


def sig_taker_buy_ratio(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Taker buy volume as ratio of total. >0.5 = net buying."""
    ratio = safe_div(df["taker_buy_base"].values, df["volume"].values, 0.5)
    smoothed = pd.Series(ratio, index=df.index).rolling(window, min_periods=1).mean()
    return smoothed - 0.5  # Center at 0


def sig_taker_buy_delta(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Change in taker buy ratio."""
    ratio = safe_div(df["taker_buy_base"].values, df["volume"].values, 0.5)
    ratio_series = pd.Series(ratio, index=df.index)
    sma = ratio_series.rolling(window, min_periods=max(window // 2, 2)).mean()
    return ratio_series - sma


def sig_volume_price_confirm(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Volume confirms price: momentum × volume_ratio. High = confirmed move."""
    mom = df["close"].pct_change()
    avg_vol = df["volume"].rolling(window, min_periods=max(window // 2, 2)).mean()
    vol_ratio = safe_div(df["volume"].values, avg_vol.values, 1.0)
    return mom * pd.Series(vol_ratio, index=df.index)


def sig_obv_momentum(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """On-balance volume momentum: rate of change of cumulative OBV."""
    direction = np.sign(df["close"].diff())
    obv = (direction * df["volume"]).cumsum()
    obv_sma = obv.rolling(window, min_periods=max(window // 2, 2)).mean()
    return (obv - obv_sma) / obv_sma.abs().replace(0, np.nan)


def sig_trades_intensity(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Trade count intensity: current trades vs rolling average."""
    avg_trades = df["trades"].rolling(window, min_periods=max(window // 2, 2)).mean()
    ratio = safe_div(df["trades"].values, avg_trades.values, 1.0)
    return pd.Series(ratio - 1.0, index=df.index)


# ============================================================================
# VOLATILITY-BASED SIGNALS
# ============================================================================

def sig_volatility_regime(df: pd.DataFrame, fast: int = 5, slow: int = 30) -> pd.Series:
    """Volatility regime: fast vol / slow vol. High = expanding vol."""
    returns = df["close"].pct_change()
    fast_vol = returns.rolling(fast, min_periods=max(fast // 2, 2)).std()
    slow_vol = returns.rolling(slow, min_periods=max(slow // 2, 2)).std()
    return safe_div(fast_vol.values, slow_vol.values, 1.0) - 1.0


def sig_range_expansion(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """High-low range relative to average. Positive = expanding range."""
    hl_range = df["high"] - df["low"]
    avg_range = hl_range.rolling(window, min_periods=max(window // 2, 2)).mean()
    return (hl_range - avg_range) / avg_range.replace(0, np.nan)


def sig_parkinson_vol_ratio(df: pd.DataFrame, fast: int = 5, slow: int = 30) -> pd.Series:
    """Parkinson volatility ratio (fast/slow)."""
    log_hl = np.log(df["high"] / df["low"]) ** 2
    fast_pv = np.sqrt(log_hl.rolling(fast, min_periods=max(fast // 2, 2)).mean() / (4 * np.log(2)))
    slow_pv = np.sqrt(log_hl.rolling(slow, min_periods=max(slow // 2, 2)).mean() / (4 * np.log(2)))
    return safe_div(fast_pv.values, slow_pv.values, 1.0) - 1.0


# ============================================================================
# PATTERN/AUTOCORRELATION SIGNALS
# ============================================================================

def sig_serial_correlation(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling autocorrelation of returns (lag 1)."""
    returns = df["close"].pct_change()
    return returns.rolling(window, min_periods=max(window // 2, 5)).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 else 0,
        raw=True
    )


def sig_streak(df: pd.DataFrame) -> pd.Series:
    """Count of consecutive UP or DOWN candles. Positive = UP streak."""
    direction = np.sign(df["close"] - df["open"]).fillna(0)
    streaks = []
    current_streak = 0
    for d in direction:
        if d == 0:
            current_streak = 0
        elif d > 0:
            current_streak = max(current_streak, 0) + 1
        else:
            current_streak = min(current_streak, 0) - 1
        streaks.append(current_streak)
    return pd.Series(streaks, index=df.index)


def sig_mean_reversion_score(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Mean reversion: negative of z-scored close (extreme highs → sell, extreme lows → buy)."""
    return -rolling_zscore(df["close"], window)


def sig_bollinger_reversal(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Bollinger band z-score reversal. Negative = expect bounce up from lower band."""
    sma = df["close"].rolling(window, min_periods=max(window // 2, 2)).mean()
    std = df["close"].rolling(window, min_periods=max(window // 2, 2)).std()
    bb_z = (df["close"] - sma) / std.replace(0, np.nan)
    return -bb_z


def sig_rsi_reversal(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """RSI reversal signal. Negative = overbought (expect down), Positive = oversold (expect up)."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=max(window // 2, 2)).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=max(window // 2, 2)).mean()
    rs = safe_div(gain.values, loss.values, 1.0)
    rsi = 100 - 100 / (1 + pd.Series(rs, index=df.index))
    return -(rsi - 50) / 50  # Center at 0, negative = overbought


def sig_mr_vol_interaction(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Mean reversion × volume confirmation."""
    z = rolling_zscore(df["close"], window)
    avg_vol = df["volume"].rolling(window, min_periods=max(window // 2, 2)).mean()
    vol_ratio = safe_div(df["volume"].values, avg_vol.values, 1.0)
    return -z * pd.Series(vol_ratio, index=df.index)


def sig_mr_speed(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Mean reversion speed: z-score × rate of change of z-score."""
    z = rolling_zscore(df["close"], window)
    z_change = z.diff(3)
    return -z * z_change.abs()


# ============================================================================
# COMPOSITE / INTERACTION SIGNALS
# ============================================================================

def sig_momentum_volume_confirm(df: pd.DataFrame, mom_lb: int = 6, vol_lb: int = 20) -> pd.Series:
    """Momentum confirmed by volume surge."""
    mom = df["close"].pct_change(mom_lb)
    avg_vol = df["volume"].rolling(vol_lb, min_periods=max(vol_lb // 2, 2)).mean()
    vol_ratio = safe_div(df["volume"].values, avg_vol.values, 1.0)
    return mom * pd.Series(vol_ratio, index=df.index)


def sig_momentum_taker_confirm(df: pd.DataFrame, mom_lb: int = 6) -> pd.Series:
    """Momentum confirmed by taker buy ratio."""
    mom = df["close"].pct_change(mom_lb)
    taker_ratio = safe_div(df["taker_buy_base"].values, df["volume"].values, 0.5)
    return mom * pd.Series(taker_ratio, index=df.index)


def sig_vwap_deviation(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Deviation from rolling VWAP."""
    vwap = (df["quote_volume"].rolling(window, min_periods=2).sum() /
            df["volume"].rolling(window, min_periods=2).sum())
    return (df["close"] - vwap) / vwap


def sig_momentum_vol_scaled(df: pd.DataFrame, mom_lb: int = 12, vol_lb: int = 30) -> pd.Series:
    """Momentum scaled inversely by volatility (Sharpe-like)."""
    returns = df["close"].pct_change()
    mom = returns.rolling(mom_lb, min_periods=max(mom_lb // 2, 2)).mean()
    vol = returns.rolling(vol_lb, min_periods=max(vol_lb // 2, 2)).std()
    return safe_div(mom.values, vol.values, 0.0)


# ============================================================================
# SIGNAL REGISTRY
# ============================================================================

def get_all_signals() -> Dict[str, Dict]:
    """
    Returns a registry of all available signals with their factory functions
    and default parameters. Each signal can be parameterized.
    """
    registry = {}

    # Momentum family
    for lb in [3, 6, 12, 24, 48]:
        registry[f"momentum_{lb}"] = {"fn": sig_momentum, "params": {"lookback": lb}}
        registry[f"momentum_smooth_{lb}"] = {"fn": sig_momentum_smooth, "params": {"lookback": lb, "smooth": 3}}

    # Return SMA family
    for w in [5, 10, 20, 40]:
        registry[f"return_sma_{w}"] = {"fn": sig_return_sma, "params": {"window": w}}

    # Z-score
    for w in [10, 20, 30, 60]:
        registry[f"return_zscore_{w}"] = {"fn": sig_return_zscore, "params": {"window": w}}

    # Candle structure
    for w in [1, 3, 5]:
        registry[f"close_pos_range_{w}"] = {"fn": sig_close_position_in_range, "params": {"window": w}}
    for w in [3, 5, 10]:
        registry[f"candle_body_{w}"] = {"fn": sig_candle_body_ratio, "params": {"window": w}}
        registry[f"upper_shadow_{w}"] = {"fn": sig_upper_shadow_ratio, "params": {"window": w}}
        registry[f"lower_shadow_{w}"] = {"fn": sig_lower_shadow_ratio, "params": {"window": w}}

    # Price relative to SMA
    for w in [10, 20, 50]:
        registry[f"price_vs_sma_{w}"] = {"fn": sig_price_vs_sma, "params": {"window": w}}

    # Donchian
    for w in [20, 30, 60]:
        registry[f"donchian_{w}"] = {"fn": sig_donchian_position, "params": {"window": w}}

    # Trend strength
    for w in [10, 20, 40]:
        registry[f"trend_strength_{w}"] = {"fn": sig_trend_strength, "params": {"window": w}}

    # MACD variants
    registry["macd_6_12"] = {"fn": sig_macd_signal, "params": {"fast": 6, "slow": 12}}
    registry["macd_12_26"] = {"fn": sig_macd_signal, "params": {"fast": 12, "slow": 26}}
    registry["macd_5_20"] = {"fn": sig_macd_signal, "params": {"fast": 5, "slow": 20}}

    # Volume
    for w in [10, 20]:
        registry[f"volume_ratio_{w}"] = {"fn": sig_volume_ratio, "params": {"window": w}}
    for w in [5, 10, 20]:
        registry[f"taker_buy_{w}"] = {"fn": sig_taker_buy_ratio, "params": {"window": w}}
    for w in [5, 10]:
        registry[f"taker_delta_{w}"] = {"fn": sig_taker_buy_delta, "params": {"window": w}}
    for w in [5, 10, 20]:
        registry[f"vol_price_confirm_{w}"] = {"fn": sig_volume_price_confirm, "params": {"window": w}}
    for w in [10, 20]:
        registry[f"obv_mom_{w}"] = {"fn": sig_obv_momentum, "params": {"window": w}}
    for w in [5, 10]:
        registry[f"trades_intensity_{w}"] = {"fn": sig_trades_intensity, "params": {"window": w}}

    # Volatility
    registry["vol_regime_5_30"] = {"fn": sig_volatility_regime, "params": {"fast": 5, "slow": 30}}
    registry["vol_regime_3_20"] = {"fn": sig_volatility_regime, "params": {"fast": 3, "slow": 20}}
    for w in [10, 20]:
        registry[f"range_expansion_{w}"] = {"fn": sig_range_expansion, "params": {"window": w}}
    registry["parkinson_ratio"] = {"fn": sig_parkinson_vol_ratio, "params": {"fast": 5, "slow": 30}}

    # Pattern
    registry["streak"] = {"fn": sig_streak, "params": {}}

    # Mean reversion (extended with v3 proven windows)
    for w in [5, 8, 10, 15, 20, 30]:
        registry[f"mean_rev_{w}"] = {"fn": sig_mean_reversion_score, "params": {"window": w}}
        # Also register as mr_X for v3 compatibility
        registry[f"mr_{w}"] = {"fn": sig_mean_reversion_score, "params": {"window": w}}

    # Composites
    for lb in [3, 6, 12]:
        registry[f"mom_vol_confirm_{lb}"] = {"fn": sig_momentum_volume_confirm, "params": {"mom_lb": lb}}
        registry[f"mom_taker_confirm_{lb}"] = {"fn": sig_momentum_taker_confirm, "params": {"mom_lb": lb}}

    # VWAP
    for w in [10, 20]:
        registry[f"vwap_dev_{w}"] = {"fn": sig_vwap_deviation, "params": {"window": w}}

    # Sharpe-like momentum
    for lb in [6, 12, 24]:
        registry[f"mom_vol_scaled_{lb}"] = {"fn": sig_momentum_vol_scaled, "params": {"mom_lb": lb}}

    # ========== V3 PROVEN SIGNALS ==========

    # Bollinger band reversal
    for w in [10, 20]:
        registry[f"bb_{w}"] = {"fn": sig_bollinger_reversal, "params": {"window": w}}

    # RSI reversal
    for w in [7, 14]:
        registry[f"rsi_{w}"] = {"fn": sig_rsi_reversal, "params": {"window": w}}

    # MR × volume interaction
    for w in [10, 20]:
        registry[f"mr_vol_{w}"] = {"fn": sig_mr_vol_interaction, "params": {"window": w}}

    # MR speed
    for w in [10, 20]:
        registry[f"mr_speed_{w}"] = {"fn": sig_mr_speed, "params": {"window": w}}

    return registry


def compute_signals(df: pd.DataFrame, signal_names: List[str] = None) -> pd.DataFrame:
    """
    Compute all (or specified) signals for a kline DataFrame.
    Returns a DataFrame where each column is a signal value.
    Signal values are shifted by 1 to avoid look-ahead bias
    (the signal at time t uses data up to t-1 to predict candle t).
    """
    registry = get_all_signals()
    if signal_names is None:
        signal_names = list(registry.keys())

    results = {}
    for name in signal_names:
        if name not in registry:
            continue
        entry = registry[name]
        try:
            sig = entry["fn"](df, **entry["params"])
            # CRITICAL: Shift by 1 to avoid look-ahead bias
            # Signal at time t predicts candle t, computed from data up to t-1
            results[name] = sig.shift(1)
        except Exception as e:
            pass  # Skip broken signals

    return pd.DataFrame(results, index=df.index)


def compute_target(df: pd.DataFrame) -> pd.Series:
    """
    Compute the binary target: 1 if candle closed UP (close >= open), 0 otherwise.
    This is the actual outcome we're trying to predict.
    """
    return (df["close"] >= df["open"]).astype(int)

