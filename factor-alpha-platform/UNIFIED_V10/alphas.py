"""
Alpha computation functions — shared between streaming and vectorized.

These are the EXACT SAME functions used by the backtest.
The streaming engine calls them on its internal buffer DataFrame.

NO LOOKAHEAD IS POSSIBLE because:
  1. The buffer only contains bars up to the current time
  2. Alpha functions use rolling/shifted operations that only look BACKWARD
  3. The engine only reads the LAST value from the output
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd


# ===========================================================================
# ALPHA PRIMITIVES
# ===========================================================================

def sma(s, w): return s.rolling(w, min_periods=max(w // 2, 2)).mean()
def ema(s, w): return s.ewm(halflife=w, min_periods=1).mean()
def stddev(s, w): return s.rolling(w, min_periods=2).std()

def ts_zscore(s, w):
    m = s.rolling(w, min_periods=max(w // 2, 2)).mean()
    sd = s.rolling(w, min_periods=max(w // 2, 2)).std()
    return (s - m) / sd.replace(0, np.nan)

def delta(s, p): return s - s.shift(p)
def ts_sum(s, w): return s.rolling(w, min_periods=1).sum()
def ts_min(s, w): return s.rolling(w, min_periods=1).min()
def ts_max(s, w): return s.rolling(w, min_periods=1).max()
def safe_div(a, b): r = a / b; return r.replace([np.inf, -np.inf], 0).fillna(0)
def correlation(x, y, w): return x.rolling(w, min_periods=2).corr(y)

def decay_exp(s, hl):
    if 0 < hl < 1:
        ah = -np.log(2) / np.log(hl)
    else:
        ah = hl
    return s.ewm(halflife=max(ah, 0.5), min_periods=1).mean()


# ===========================================================================
# HIGHER-TIMEFRAME SIGNALS
# ===========================================================================

def build_htf_signals(df_htf, df_1h, prefix, shift_n=1):
    """Build signals from a higher-timeframe DataFrame, safely shifted.

    CRITICAL: shift_n=1 means we use the PREVIOUS HTF bar only,
    preventing any lookahead from the current incomplete bar.
    """
    c = df_htf['close'].shift(shift_n)
    h = df_htf['high'].shift(shift_n)
    l = df_htf['low'].shift(shift_n)
    v = df_htf['volume'].shift(shift_n)
    lr = np.log(c / c.shift(1))
    cp = safe_div(c - l, h - l)
    alphas = {}

    # Mean reversion
    for w in [3, 5, 8, 12, 20, 30]:
        sig = (-ts_zscore(c, w)).reindex(df_1h.index, method='ffill')
        alphas[f'{prefix}_mr_{w}'] = sig

    # Momentum
    for w in [3, 5, 8, 12, 20]:
        sig = ts_sum(lr, w).reindex(df_1h.index, method='ffill')
        alphas[f'{prefix}_mom_{w}'] = sig

    # Breakout
    for w in [12, 24, 48, 72, 120]:
        h_ = ts_max(h, w); l_ = ts_min(l, w); rng = h_ - l_
        sig = (safe_div(c - l_, rng) * 2 - 1).reindex(df_1h.index, method='ffill')
        alphas[f'{prefix}_brk_{w}'] = sig

    # Decay momentum
    for mw in [3, 6, 12]:
        for dc in [0.9, 0.95, 0.98]:
            sig = decay_exp((c - c.shift(mw)) * cp, dc).reindex(df_1h.index, method='ffill')
            alphas[f'{prefix}_dec{dc}_d{mw}'] = sig

    # Volume z-score
    vz = ts_zscore(v, 20)
    if vz is not None:
        sig = vz.reindex(df_1h.index, method='ffill')
        alphas[f'{prefix}_vol_z20'] = sig

    return alphas


# ===========================================================================
# 1H ALPHA GENERATION (~220 signals)
# ===========================================================================

def build_1h_alphas(df_1h):
    """Core 1H alpha library across 10 signal categories.

    ALL operations are backward-looking (rolling, shift, ewm).
    NO lookahead is possible.
    """
    close = df_1h['close']; volume = df_1h['volume']
    high = df_1h['high']; low = df_1h['low']; opn = df_1h['open']
    qv = df_1h.get('quote_volume', pd.Series(0.0, index=df_1h.index))
    tbv = df_1h.get('taker_buy_volume', pd.Series(0.0, index=df_1h.index))

    ret = close.pct_change()
    log_ret = np.log(close / close.shift(1))
    vwap = safe_div(qv, volume)
    taker_ratio = safe_div(tbv, volume)
    rvol_short = ret.rolling(6, min_periods=2).std()
    rvol_long = ret.rolling(72, min_periods=2).std()
    close_pos = safe_div(close - low, high - low)
    body = close - opn
    candle_range = high - low

    alphas = {}

    # === Mean Reversion ===
    for w in [3,4,5,6,7,8,9,10,11,12,14,16,18,20,24,28,30,36,42,48,60,72,96]:
        alphas[f'mr_{w}'] = -ts_zscore(close, w)
    for w in [3,5,8,10,12,15,20,24,30]:
        alphas[f'logrev_{w}'] = -ts_sum(log_ret, w)
    for w in [3,5,8,10,12,15,20]:
        alphas[f'dstd_{w}'] = -safe_div(delta(close, w), stddev(close, w))
    for w in [5,10,15,20,30,48]:
        alphas[f'vwap_mr_{w}'] = -ts_zscore(vwap, w)
    for w in [5,10,15,20,30]:
        alphas[f'ema_mr_{w}'] = -(close - ema(close, w)) / stddev(close, w * 2)
    for w in [5,10,20,30]:
        alphas[f'ema_ret_{w}'] = -ts_zscore(ema(ret, w), w * 2)

    # === Momentum ===
    for w in [3,5,8,12,18,24,36,48,72]:
        alphas[f'mom_{w}'] = ts_sum(log_ret, w)
    for fast, slow in [(3,12),(5,15),(5,20),(8,24),(12,36),(12,48),(24,72),(36,120)]:
        alphas[f'emax_{fast}_{slow}'] = (ema(close, fast) - ema(close, slow)) / stddev(close, slow)
    for w in [12,24,36,48,72,96,120,168,240,360]:
        h = ts_max(high, w); l = ts_min(low, w); rng = h - l
        alphas[f'breakout_{w}'] = safe_div(close - l, rng) * 2 - 1

    # === Trend Direction ===
    for w in [6,12,24,48]:
        up = high - high.shift(1); dn = low.shift(1) - low
        plus_dm = up.where((up > dn) & (up > 0), 0)
        minus_dm = dn.where((dn > up) & (dn > 0), 0)
        atr = (high - low).rolling(w, min_periods=1).mean()
        plus_di = safe_div(plus_dm.rolling(w).mean(), atr)
        minus_di = safe_div(minus_dm.rolling(w).mean(), atr)
        alphas[f'trend_{w}'] = plus_di - minus_di

    # === Decay Momentum ===
    for mw in [3,6,12,24,48]:
        for dc in [0.8,0.9,0.95,0.98,0.99]:
            alphas[f'dec{dc}_d{mw}_cp'] = decay_exp(
                (close - close.shift(mw)) * close_pos, dc)
            alphas[f'dec{dc}_d{mw}_tbr'] = decay_exp(
                (close - close.shift(mw)) * taker_ratio, dc)

    # === Vol-Conditioned ===
    vol_ratio = safe_div(rvol_short, rvol_long)
    for w in [8,10,15,20,30]:
        alphas[f'lovol_mr_{w}'] = -ts_zscore(close, w) * (vol_ratio < 1.0).astype(float)
    for w in [5,8,12,24]:
        alphas[f'hivol_mom_{w}'] = ts_sum(log_ret, w) * (vol_ratio > 1.0).astype(float)
    for w in [10,20,30]:
        alphas[f'vs_mr_{w}'] = -ts_zscore(close, w) * safe_div(
            rvol_long, rvol_short).clip(0.2, 5.0)

    # === Volume / Microstructure ===
    obv = (np.sign(ret) * volume).cumsum()
    for w in [10,20,30,48]:
        alphas[f'obv_{w}'] = -ts_zscore(obv, w)
    for w in [5,10,20,30]:
        alphas[f'tbr_{w}'] = ts_zscore(taker_ratio, w)
    timb = safe_div(tbv - (volume - tbv), volume)
    for w in [5,10,20,30]:
        alphas[f'timb_{w}'] = ts_zscore(timb, w)
    for w in [5,10,20]:
        vw = (ret * volume).rolling(w).sum()
        vs = volume.rolling(w).sum()
        alphas[f'vwret_{w}'] = safe_div(vw, vs)

    # === Candle ===
    for w in [5,10,20]:
        alphas[f'body_{w}'] = ts_zscore(safe_div(body, candle_range), w)
    uwick = high - np.maximum(close, opn)
    lwick = np.minimum(close, opn) - low
    rejection = safe_div(uwick - lwick, candle_range)
    for w in [5,10]:
        alphas[f'reject_{w}'] = -ts_zscore(rejection, w)
    atr = (high - low).rolling(14, min_periods=2).mean()
    for w in [5,10,20]:
        alphas[f'atr_mr_{w}'] = -safe_div(delta(close, w), atr)

    # === Regime ===
    for w in [12,24,48,72]:
        ret_lag = ret.shift(1)
        ac = correlation(ret, ret_lag, w)
        alphas[f'regime_mom_{w}'] = ts_sum(log_ret, 5) * ac.clip(lower=0)
        alphas[f'regime_mr_{w}'] = -ts_zscore(close, 10) * (-ac).clip(lower=0)

    # === Cross-Timeframe ===
    m5 = ts_sum(log_ret, 5); m20 = ts_sum(log_ret, 20); m60 = ts_sum(log_ret, 60)
    alphas['mtf_agree'] = np.sign(m5) * np.sign(m20) * np.sign(m60) * abs(m5)
    alphas['trend_pullback'] = -ts_zscore(close, 5) * np.sign(m60)
    alphas['trend_pullback_20'] = -ts_zscore(close, 10) * np.sign(m20)

    # === Technical Indicators ===
    for w in [8,10,15,20,30]:
        s = close.rolling(w).mean(); sd = close.rolling(w).std()
        alphas[f'bb_{w}'] = -(close - s) / sd.replace(0, np.nan)
    for w in [7,10,14,21]:
        d = close.diff()
        gain = d.clip(lower=0).rolling(w).mean()
        loss = (-d.clip(upper=0)).rolling(w).mean()
        rs = safe_div(gain, loss)
        rsi = 100 - 100 / (1 + rs)
        alphas[f'rsi_{w}'] = -(rsi - 50) / 50
    for w in [14,21]:
        lowest = ts_min(low, w); highest = ts_max(high, w)
        alphas[f'stoch_{w}'] = -(safe_div(close - lowest, highest - lowest) * 100 - 50) / 50
    for w in [14,20]:
        tp = (high + low + close) / 3; tp_sma = sma(tp, w)
        md = tp.rolling(w).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        alphas[f'cci_{w}'] = -safe_div(tp - tp_sma, 0.015 * md)

    # === Acceleration ===
    for w in [3,5,8,12]:
        mom_w = ts_sum(log_ret, w)
        alphas[f'accel_{w}'] = mom_w - mom_w.shift(w)

    # === Volume acceleration ===
    vol_ma5 = volume.rolling(5).mean()
    vol_ma20 = volume.rolling(20).mean()
    alphas['vol_accel'] = safe_div(vol_ma5, vol_ma20) - 1

    # === Intrabar volatility ===
    ib_vol = safe_div(high - low, close.shift(1))
    for w in [10, 20]:
        alphas[f'ibvol_z_{w}'] = -ts_zscore(ib_vol, w)

    # === Consecutive direction ===
    direction = np.sign(ret)
    for w in [3, 5, 8]:
        alphas[f'consec_{w}'] = -direction.rolling(w).sum() / w

    return alphas


# ===========================================================================
# CROSS-ASSET SIGNALS
# ===========================================================================

def build_cross_asset_signals(all_1h, sym, df_1h):
    """Cross-asset signals using BTC + ETH + SOL as market factors."""
    alphas = {}
    for factor_sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
        if factor_sym == sym or factor_sym not in all_1h:
            continue
        fac = all_1h[factor_sym]
        common = df_1h.index.intersection(fac.index)
        if len(common) < 500:
            continue
        fac_ret = fac['close'].pct_change().loc[common]
        fac_lr = np.log(fac['close'] / fac['close'].shift(1)).loc[common]
        sym_ret = df_1h['close'].pct_change().loc[common]
        pfx = factor_sym[:3].lower()

        # Factor momentum
        for w in [3, 5, 8, 12, 24]:
            alphas[f'{pfx}_mom_{w}'] = ts_sum(fac_lr, w).reindex(
                df_1h.index, method='ffill')
        # Factor mean reversion
        for w in [5, 10, 20]:
            alphas[f'{pfx}_mr_{w}'] = (-ts_zscore(
                fac['close'].loc[common], w)).reindex(df_1h.index, method='ffill')
        # Relative strength
        for w in [5, 12, 24]:
            rel = ts_sum(sym_ret, w) - ts_sum(fac_ret, w)
            alphas[f'{pfx}_relstr_{w}'] = (-rel).reindex(
                df_1h.index, method='ffill')
    return alphas
