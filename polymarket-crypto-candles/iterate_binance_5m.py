"""
iterate_binance_5m.py — Binance Futures 5m directional trading.

PROPER ML METHODOLOGY:
  Walk-forward with rolling train window.
  Within each fold:
    1. TRAIN (first 80%): alpha discovery + parameter selection
    2. VAL (last 20%): verify params, no re-tuning
    3. TEST (next month): truly out-of-sample
  
  NO architecture/alpha decisions based on test performance.

Key difference from Polymarket model:
  - PnL is PROPORTIONAL to move size (not binary)
  - MR signals alone fail because avg_win < avg_loss
  - Need: momentum, vol-scaling, magnitude prediction
  
Target: Annualized Sharpe > 5 on walk-forward test.
"""
import sys, os, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SYMBOLS, SYMBOL_NAMES, DATA_DIR

INTERVAL = "5m"
BARS_PER_DAY = 288
FEE_BPS = 3  # 3bps per trade
FEE_FRAC = FEE_BPS / 10000.0

# ============================================================================
# ALPHA PRIMITIVES
# ============================================================================

def sma(s, w): return s.rolling(w, min_periods=1).mean()
def ema(s, w): return s.ewm(halflife=w, min_periods=1).mean()
def stddev(s, w): return s.rolling(w, min_periods=2).std()
def ts_zscore(s, w):
    m = s.rolling(w, min_periods=2).mean()
    sd = s.rolling(w, min_periods=2).std()
    return (s - m) / sd.replace(0, np.nan)
def delta(s, p): return s - s.shift(p)
def ts_sum(s, w): return s.rolling(w, min_periods=1).sum()
def ts_min(s, w): return s.rolling(w, min_periods=1).min()
def ts_max(s, w): return s.rolling(w, min_periods=1).max()
def ts_rank(s, w): return s.rolling(w, min_periods=2).rank(pct=True)
def safe_div(a, b):
    r = a / b; return r.replace([np.inf, -np.inf], 0).fillna(0)
def correlation(x, y, w): return x.rolling(w, min_periods=2).corr(y)

# ============================================================================
# ALPHA GENERATION — BINANCE-OPTIMIZED
# ============================================================================

def build_alpha_matrix(df):
    """Generate alpha signals optimized for proportional-return trading.
    
    Key design principles:
    1. Include MOMENTUM alphas (not just MR) — catch trending moves
    2. Include VOLATILITY-CONDITIONED signals — reduce exposure in high-vol
    3. Include MAGNITUDE signals — predict size, not just direction
    4. All shifted by 1 to prevent lookahead
    """
    close = df["close"]
    volume = df["volume"]
    high = df["high"]
    low = df["low"]
    opn = df["open"]
    qv = df["quote_volume"]
    taker_buy = df["taker_buy_base"]
    trades = df["trades"].astype(float)

    ret = close.pct_change()
    log_ret = np.log(close / close.shift(1))
    vwap = safe_div(qv, volume)
    taker_ratio = safe_div(taker_buy, volume)
    
    # Realized volatility (for scaling)
    rvol_short = ret.rolling(12, min_periods=2).std()  # 1h
    rvol_med = ret.rolling(48, min_periods=2).std()    # 4h
    rvol_long = ret.rolling(288, min_periods=2).std()  # 1d
    
    alphas = {}

    # ===== MEAN REVERSION (works in range-bound) =====
    for w in [5, 8, 10, 12, 15, 20, 24, 30, 36, 48]:
        alphas[f"mr_{w}"] = -ts_zscore(close, w)
    
    # Log return reversal
    for w in [3, 5, 8, 10, 12, 15, 20]:
        alphas[f"logrev_{w}"] = -ts_sum(log_ret, w)
    
    # Normalized delta (price change / vol)
    for w in [3, 5, 8, 10, 12, 15, 20]:
        alphas[f"dstd_{w}"] = -safe_div(delta(close, w), stddev(close, w))
    
    # VWAP z-score
    for w in [5, 10, 15, 20, 30]:
        alphas[f"vwap_mr_{w}"] = -ts_zscore(vwap, w)
    
    # EMA MR
    for w in [5, 10, 20]:
        alphas[f"ema_mr_{w}"] = -(close - ema(close, w)) / stddev(close, w * 2)

    # ===== MOMENTUM (catch trending moves — the key Binance addition) =====
    # Short-term momentum
    for w in [3, 5, 8, 12]:
        alphas[f"mom_{w}"] = ts_sum(log_ret, w)  # positive = continue trend
    
    # EMA crossover momentum
    for fast, slow in [(3, 12), (5, 20), (8, 24), (12, 48)]:
        alphas[f"emax_{fast}_{slow}"] = (ema(close, fast) - ema(close, slow)) / stddev(close, slow)
    
    # Breakout: close near high of range
    for w in [12, 24, 48]:
        h = ts_max(high, w)
        l = ts_min(low, w)
        rng = h - l
        alphas[f"breakout_{w}"] = safe_div(close - l, rng) * 2 - 1  # [-1, +1]
    
    # Trend strength (ADX-like)
    for w in [12, 24]:
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        atr = (high - low).rolling(w, min_periods=1).mean()
        plus_di = safe_div(plus_dm.rolling(w).mean(), atr)
        minus_di = safe_div(minus_dm.rolling(w).mean(), atr)
        dx = safe_div(abs(plus_di - minus_di), plus_di + minus_di)
        alphas[f"trend_{w}"] = (plus_di - minus_di)  # positive = uptrend

    # ===== VOLATILITY-CONDITIONED =====
    # MR only during LOW volatility (when MR works, scale down during trends)
    vol_ratio = safe_div(rvol_short, rvol_long)
    for w in [10, 15, 20]:
        low_vol = (vol_ratio < 1.0).astype(float)
        alphas[f"lovol_mr_{w}"] = -ts_zscore(close, w) * low_vol
    
    # Momentum only during HIGH volatility (when trends exist)  
    for w in [5, 12]:
        high_vol = (vol_ratio > 1.0).astype(float)
        alphas[f"hivol_mom_{w}"] = ts_sum(log_ret, w) * high_vol
    
    # Vol-scaled MR: bet size inversely proportional to vol
    for w in [10, 20]:
        vol_scale = safe_div(rvol_long, rvol_short).clip(0.2, 5.0)
        alphas[f"volscale_mr_{w}"] = -ts_zscore(close, w) * vol_scale

    # ===== VOLUME / MICROSTRUCTURE =====
    # OBV momentum  
    obv = (np.sign(ret) * volume).cumsum()
    for w in [10, 20, 30]:
        alphas[f"obv_{w}"] = -ts_zscore(obv, w)
    
    # Volume-price divergence (smart money)
    for w in [10, 20]:
        price_change = ts_zscore(close, w)
        vol_change = ts_zscore(volume, w)
        alphas[f"vp_div_{w}"] = vol_change - price_change  # high vol + falling price = reversal
    
    # Taker buy pressure
    for w in [5, 10, 20]:
        alphas[f"tbr_{w}"] = ts_zscore(taker_ratio, w)
    
    # Aggressive taker imbalance
    taker_sell = volume - taker_buy
    taker_imbalance = safe_div(taker_buy - taker_sell, volume)
    for w in [5, 10, 20]:
        alphas[f"timb_{w}"] = ts_zscore(taker_imbalance, w)

    # ===== CANDLE PATTERN =====
    body = close - opn
    upper_wick = high - pd.concat([close, opn], axis=1).max(axis=1)
    lower_wick = pd.concat([close, opn], axis=1).min(axis=1) - low
    candle_range = high - low
    
    # Body ratio (bullish/bearish pressure)
    for w in [5, 10]:
        alphas[f"body_{w}"] = ts_zscore(safe_div(body, candle_range), w)
    
    # Rejection candles (long wick = reversal signal)
    rejection = safe_div(upper_wick - lower_wick, candle_range)
    for w in [5, 10]:
        alphas[f"reject_{w}"] = -ts_zscore(rejection, w)

    # ===== REGIME / ADAPTIVE (vectorized, no slow apply) =====
    # Use rolling correlation of ret[t] vs ret[t-1] as trend/MR indicator
    for w in [24, 48]:
        ret_lag = ret.shift(1)
        ac = correlation(ret, ret_lag, w)
        # In trending regime: use momentum; in MR regime: use MR
        alphas[f"regime_mom_{w}"] = ts_sum(log_ret, 5) * ac.clip(lower=0)
        alphas[f"regime_mr_{w}"] = -ts_zscore(close, 10) * (-ac).clip(lower=0)

    # ===== CROSS-TIMEFRAME =====
    mom_5 = ts_sum(log_ret, 5)
    mom_20 = ts_sum(log_ret, 20)
    mom_60 = ts_sum(log_ret, 60)
    alphas["mtf_agree"] = np.sign(mom_5) * np.sign(mom_20) * np.sign(mom_60) * abs(mom_5)
    
    # Short-term reversal in direction of longer trend
    alphas["trend_pullback"] = -ts_zscore(close, 5) * np.sign(mom_60)

    # Shift ALL by 1 bar (lookahead prevention)
    alpha_df = pd.DataFrame(alphas, index=df.index)
    alpha_df = alpha_df.shift(1)
    return alpha_df


# ============================================================================
# EVALUATION — BINANCE PnL (not binary accuracy)
# ============================================================================

def evaluate_alpha_binance(signal, returns, min_bars=500):
    """Evaluate alpha by actual Binance PnL (proportional returns).
    
    This is the KEY difference from Polymarket evaluation:
    We care about direction * magnitude, not just direction accuracy.
    """
    common = signal.dropna().index.intersection(returns.dropna().index)
    if len(common) < min_bars:
        return None
    
    sig = signal.loc[common]
    ret = returns.loc[common]
    
    # Direction signal
    direction = np.sign(sig)
    
    # Raw PnL = direction * return (no fees)
    raw_pnl = direction * ret
    
    # Daily aggregation
    daily_pnl = raw_pnl.resample("1D").sum()
    daily_pnl = daily_pnl[daily_pnl != 0]
    
    if len(daily_pnl) < 10 or daily_pnl.std() == 0:
        return None
    
    nofee_sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(365)
    
    # With fees
    fee_pnl = raw_pnl - FEE_FRAC  # deduct fee per trade
    daily_fee = fee_pnl.resample("1D").sum()
    daily_fee = daily_fee[daily_fee != 0]
    fee_sharpe = daily_fee.mean() / daily_fee.std() * np.sqrt(365) if len(daily_fee) > 10 and daily_fee.std() > 0 else 0
    
    # Win rate (by PnL, not direction)
    win_rate = (raw_pnl > 0).mean()
    
    # Average win vs loss (key for Binance!)
    wins = raw_pnl[raw_pnl > 0]
    losses = raw_pnl[raw_pnl < 0]
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    # IC (information coefficient)
    ic = sig.corr(ret)
    
    # Sub-period stability
    n = len(common)
    q1_pnl = raw_pnl.iloc[:n//3].mean()
    q2_pnl = raw_pnl.iloc[n//3:2*n//3].mean()
    q3_pnl = raw_pnl.iloc[2*n//3:].mean()
    
    return {
        "nofee_sharpe": nofee_sharpe,
        "fee_sharpe": fee_sharpe,
        "win_rate": win_rate,
        "avg_win_bps": avg_win * 10000,
        "avg_loss_bps": avg_loss * 10000,
        "win_loss_ratio": win_loss_ratio,
        "ic": ic,
        "pnl_bps": raw_pnl.mean() * 10000,
        "q1": q1_pnl * 10000,
        "q2": q2_pnl * 10000,
        "q3": q3_pnl * 10000,
        "n_bars": len(common),
    }


def select_orthogonal(results, alpha_matrix, max_n=15, corr_cutoff=0.70):
    """Select top orthogonal alphas by Binance Sharpe."""
    selected = []
    for r in results:
        if r["name"] not in alpha_matrix.columns:
            continue
        sig = alpha_matrix[r["name"]]
        too_corr = False
        for sel in selected:
            if abs(sig.corr(alpha_matrix[sel["name"]])) > corr_cutoff:
                too_corr = True
                break
        if not too_corr:
            selected.append(r)
        if len(selected) >= max_n:
            break
    return selected


# ============================================================================
# PORTFOLIO STRATEGIES — BINANCE
# ============================================================================

def strategy_adaptive_net_binance(alpha_matrix, returns, selected,
                                  lookback=2880, phl=1):
    """Adaptive weighting by rolling net factor returns.
    CRITICAL: Only charge fee on position CHANGES (Binance futures).
    If you hold the same direction, no new fee."""
    cols = [s["name"] for s in selected if s["name"] in alpha_matrix.columns]
    if len(cols) < 2:
        return None
    
    X = alpha_matrix[cols].copy()
    ret = returns.copy()
    
    valid = X.dropna().index.intersection(ret.dropna().index)
    X, ret = X.loc[valid], ret.loc[valid]
    n = len(valid)
    
    if n < lookback + BARS_PER_DAY:
        return None
    
    # Per-alpha returns (NO fee here — fee applied at portfolio level)
    factor_returns = pd.DataFrame(index=valid, columns=cols, dtype=float)
    for col in cols:
        d = np.sign(X[col].values)
        factor_returns[col] = d * ret.values
    
    # Rolling expected return
    rolling_er = factor_returns.rolling(lookback, min_periods=min(200, lookback)).mean()
    
    # Only positive ER alphas get weight
    weights = rolling_er.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)
    
    # Smooth weights
    if phl > 1:
        weights_smooth = weights_norm.ewm(halflife=phl, min_periods=1).mean()
        wsum2 = weights_smooth.sum(axis=1).replace(0, np.nan)
        weights_smooth = weights_smooth.div(wsum2, axis=0).fillna(0)
    else:
        weights_smooth = weights_norm
    
    # Combined signal
    combined = (X * weights_smooth).sum(axis=1)
    direction = np.sign(combined.values)
    
    # Fee only on position CHANGES (open/close/flip)
    pos_changes = np.abs(np.diff(np.concatenate([[0], direction])))
    # Each change costs FEE_FRAC (if flip from +1 to -1, cost is 2x)
    
    pnl = direction * ret.values - FEE_FRAC * pos_changes
    pnl_nofee = direction * ret.values
    
    return compute_metrics_binance(pnl, pnl_nofee, valid)


def strategy_volscaled_adaptive(alpha_matrix, returns, selected,
                                 lookback=2880, phl=1, vol_target=0.001):
    """Adaptive net with volatility-targeted position sizing.
    Scale position inversely to recent vol + fee only on changes."""
    cols = [s["name"] for s in selected if s["name"] in alpha_matrix.columns]
    if len(cols) < 2:
        return None
    
    X = alpha_matrix[cols].copy()
    ret = returns.copy()
    
    valid = X.dropna().index.intersection(ret.dropna().index)
    X, ret = X.loc[valid], ret.loc[valid]
    n = len(valid)
    
    if n < lookback + BARS_PER_DAY:
        return None
    
    factor_returns = pd.DataFrame(index=valid, columns=cols, dtype=float)
    for col in cols:
        d = np.sign(X[col].values)
        factor_returns[col] = d * ret.values
    
    rolling_er = factor_returns.rolling(lookback, min_periods=200).mean()
    weights = rolling_er.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)
    if phl > 1:
        weights_smooth = weights_norm.ewm(halflife=phl, min_periods=1).mean()
        wsum2 = weights_smooth.sum(axis=1).replace(0, np.nan)
        weights_smooth = weights_smooth.div(wsum2, axis=0).fillna(0)
    else:
        weights_smooth = weights_norm
    
    combined = (X * weights_smooth).sum(axis=1)
    direction = np.sign(combined.values)
    
    # Vol scaling
    recent_vol = ret.rolling(48, min_periods=10).std()
    vol_scalar = (vol_target / recent_vol.replace(0, np.nan)).clip(0.1, 5.0).fillna(1.0)
    position = direction * vol_scalar.values
    
    # Fee only on position changes
    pos_changes = np.abs(np.diff(np.concatenate([[0], position])))
    
    pnl = position * ret.values - FEE_FRAC * pos_changes
    pnl_nofee = position * ret.values
    
    return compute_metrics_binance(pnl, pnl_nofee, valid)


def strategy_magnitude_weighted(alpha_matrix, returns, selected,
                                 lookback=2880, phl=1):
    """Weight by MAGNITUDE of alpha signal, not just sign.
    
    Stronger signals → bigger positions. Weak signals → smaller or skip.
    This helps because strong MR signals tend to have better win/loss ratio.
    """
    cols = [s["name"] for s in selected if s["name"] in alpha_matrix.columns]
    if len(cols) < 2:
        return None
    
    X = alpha_matrix[cols].copy()
    ret = returns.copy()
    
    valid = X.dropna().index.intersection(ret.dropna().index)
    X, ret = X.loc[valid], ret.loc[valid]
    n = len(valid)
    
    if n < lookback + BARS_PER_DAY:
        return None
    
    # Adaptive weights
    factor_returns = pd.DataFrame(index=valid, columns=cols, dtype=float)
    for col in cols:
        d = np.sign(X[col].values)
        factor_returns[col] = d * ret.values - FEE_FRAC
    
    rolling_er = factor_returns.rolling(lookback, min_periods=200).mean()
    weights = rolling_er.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)
    if phl > 1:
        weights_smooth = weights_norm.ewm(halflife=phl, min_periods=1).mean()
        wsum2 = weights_smooth.sum(axis=1).replace(0, np.nan)
        weights_smooth = weights_smooth.div(wsum2, axis=0).fillna(0)
    else:
        weights_smooth = weights_norm
    
    # Combined signal — USE MAGNITUDE not just sign
    combined = (X * weights_smooth).sum(axis=1)
    
    # Normalize to z-score so magnitude is comparable across time
    combined_z = ts_zscore(combined, 288)  # daily z-score
    
    # Position = sign * min(|z|, 3) / 3  → scales 0 to 1
    position = np.sign(combined_z) * np.minimum(np.abs(combined_z), 3.0) / 3.0
    position = position.fillna(0).values
    
    # PnL with magnitude-weighted position
    pnl = position * ret.values - FEE_FRAC * np.abs(position)
    pnl_nofee = position * ret.values
    
    return compute_metrics_binance(pnl, pnl_nofee, valid)


def strategy_filtered(alpha_matrix, returns, selected,
                       lookback=2880, phl=1, min_strength=0.5):
    """Only trade when signal strength exceeds threshold.
    
    Skip weak signals entirely. This reduces trade count but
    should improve avg trade quality (higher win/loss ratio).
    """
    cols = [s["name"] for s in selected if s["name"] in alpha_matrix.columns]
    if len(cols) < 2:
        return None
    
    X = alpha_matrix[cols].copy()
    ret = returns.copy()
    
    valid = X.dropna().index.intersection(ret.dropna().index)
    X, ret = X.loc[valid], ret.loc[valid]
    n = len(valid)
    
    if n < lookback + BARS_PER_DAY:
        return None
    
    # Adaptive weights
    factor_returns = pd.DataFrame(index=valid, columns=cols, dtype=float)
    for col in cols:
        d = np.sign(X[col].values)
        factor_returns[col] = d * ret.values - FEE_FRAC
    
    rolling_er = factor_returns.rolling(lookback, min_periods=200).mean()
    weights = rolling_er.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)
    if phl > 1:
        weights_smooth = weights_norm.ewm(halflife=phl, min_periods=1).mean()
        wsum2 = weights_smooth.sum(axis=1).replace(0, np.nan)
        weights_smooth = weights_smooth.div(wsum2, axis=0).fillna(0)
    else:
        weights_smooth = weights_norm
    
    combined = (X * weights_smooth).sum(axis=1)
    combined_z = ts_zscore(combined, 288).fillna(0)
    
    # Only trade when |z| > threshold
    trade_mask = np.abs(combined_z.values) > min_strength
    direction = np.sign(combined_z.values) * trade_mask.astype(float)
    
    pnl = direction * ret.values - FEE_FRAC * np.abs(direction)
    pnl_nofee = direction * ret.values
    
    return compute_metrics_binance(pnl, pnl_nofee, valid)


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics_binance(pnl, pnl_nofee, index):
    """Compute strategy metrics for Binance."""
    pnl_series = pd.Series(pnl, index=index)
    daily = pnl_series.resample("1D").sum()
    daily = daily[daily != 0]
    
    if len(daily) < 10 or daily.std() == 0:
        return None
    
    sharpe = daily.mean() / daily.std() * np.sqrt(365)
    
    nofee_series = pd.Series(pnl_nofee, index=index)
    daily_nofee = nofee_series.resample("1D").sum()
    daily_nofee = daily_nofee[daily_nofee != 0]
    nofee_sharpe = daily_nofee.mean() / daily_nofee.std() * np.sqrt(365) if len(daily_nofee) > 10 and daily_nofee.std() > 0 else 0
    
    cum = np.cumsum(pnl)
    max_dd = (cum - np.maximum.accumulate(cum)).min()
    
    active = pnl != 0
    n_trades = active.sum()
    wins = (pnl[active] > 0).sum() if n_trades > 0 else 0
    
    avg_win = pnl[pnl > 0].mean() * 10000 if (pnl > 0).any() else 0  # in bps
    avg_loss = pnl[pnl < 0].mean() * 10000 if (pnl < 0).any() else 0
    
    return {
        "sharpe": sharpe,
        "nofee_sharpe": nofee_sharpe,
        "win_rate": wins / max(n_trades, 1),
        "total_trades": int(n_trades),
        "net_pnl_bps": pnl.sum() * 10000,
        "max_drawdown_bps": max_dd * 10000,
        "avg_win_bps": avg_win,
        "avg_loss_bps": avg_loss,
        "daily_pnl_bps": daily.mean() * 10000,
    }


# ============================================================================
# WALK-FORWARD ENGINE
# ============================================================================

def walk_forward(sym, alpha_builder=build_alpha_matrix, 
                 strategy_fn=strategy_adaptive_net_binance,
                 strategy_kwargs={},
                 train_months=6, test_months=1,
                 corr_cutoffs=[0.70, 0.80], max_ns=[10, 15],
                 lookbacks=[2880, 5760], verbose=True):
    """Proper walk-forward backtest.
    
    For each fold:
      1. Train window: alpha discovery + param selection
      2. Test window: evaluate (never used for decisions)
    """
    name = SYMBOL_NAMES[sym]
    df = pd.read_parquet(DATA_DIR / f"{sym}_5m.parquet")
    returns = df["close"].pct_change()
    
    months = pd.date_range(start="2024-09-01", end="2026-03-01", freq="MS")
    fold_results = []
    
    for i in range(len(months) - 1):
        test_start = months[i]
        test_end = months[i + 1]
        train_start = test_start - pd.DateOffset(months=train_months)
        
        train_df = df.loc[str(train_start):str(test_start)]
        test_df = df.loc[str(test_start):str(test_end)]
        
        if len(train_df) < 50000 or len(test_df) < 5000:
            continue
        
        train_ret = returns.loc[train_df.index]
        test_ret = returns.loc[test_df.index]
        
        # Phase 1: Alpha discovery on TRAIN
        alpha_tr = alpha_builder(train_df)
        results = []
        for col in alpha_tr.columns:
            metric = evaluate_alpha_binance(alpha_tr[col], train_ret)
            if metric and metric["nofee_sharpe"] > 0:
                results.append({"name": col, **metric})
        
        # Sort by no-fee Sharpe (on train only!)
        results.sort(key=lambda x: x["nofee_sharpe"], reverse=True)
        
        if len(results) < 3:
            continue
        
        # Phase 2: Param selection on TRAIN
        best_train_sr = -999
        best_cfg = None
        
        for cc in corr_cutoffs:
            for mn in max_ns:
                sel = select_orthogonal(results, alpha_tr, max_n=mn, corr_cutoff=cc)
                if len(sel) < 3:
                    continue
                for lb in lookbacks:
                    mt = strategy_fn(alpha_tr, train_ret, sel, 
                                     lookback=lb, **strategy_kwargs)
                    if mt and mt["sharpe"] > best_train_sr:
                        best_train_sr = mt["sharpe"]
                        best_cfg = {"corr": cc, "n": len(sel), "lb": lb}
                        best_selected = sel
        
        if best_cfg is None:
            continue
        
        # Phase 3: Evaluate on TEST (never seen)
        alpha_te = alpha_builder(test_df)
        mte = strategy_fn(alpha_te, test_ret, best_selected,
                          lookback=best_cfg["lb"], **strategy_kwargs)
        
        if mte:
            fold_results.append({
                "fold": test_start.strftime("%Y-%m"),
                **mte,
                "cfg": best_cfg,
            })
    
    if verbose and fold_results:
        print(f"\n{name}:")
        print(f"  {'Fold':<10} {'SR':>8} {'NF_SR':>8} {'WR':>8} {'Trades':>8} "
              f"{'PnL(bps)':>10} {'AvgW':>8} {'AvgL':>8}")
        print(f"  {'-'*75}")
        for fr in fold_results:
            print(f"  {fr['fold']:<10} {fr['sharpe']:>8.2f} {fr['nofee_sharpe']:>8.2f} "
                  f"{fr['win_rate']:>7.1%} {fr['total_trades']:>8} "
                  f"{fr['net_pnl_bps']:>10.1f} {fr['avg_win_bps']:>8.1f} {fr['avg_loss_bps']:>8.1f}")
        
        srs = [f["sharpe"] for f in fold_results]
        avg_sr = np.mean(srs)
        total_pnl = sum(f["net_pnl_bps"] for f in fold_results)
        print(f"  {'AVG':<10} {avg_sr:>8.2f}            "
              f"          {total_pnl:>10.1f}")
    
    return fold_results


# ============================================================================
# MAIN — ITERATIVE IMPROVEMENT
# ============================================================================

if __name__ == "__main__":
    print("#" * 80)
    print("# BINANCE FUTURES 5m — WALK-FORWARD BACKTEST")
    print("# Proper ML: alpha discovery + param selection on TRAIN only")
    print("# 3bps fees, proportional returns")
    print("#" * 80)
    
    for sym in SYMBOLS:
        # Strategy 1: Basic adaptive net
        print(f"\n{'='*60}")
        print(f"  Strategy: Adaptive Net (basic)")
        walk_forward(sym, strategy_fn=strategy_adaptive_net_binance)
        
        # Strategy 2: Vol-scaled
        print(f"\n{'='*60}")
        print(f"  Strategy: Vol-Scaled Adaptive")
        walk_forward(sym, strategy_fn=strategy_volscaled_adaptive,
                     strategy_kwargs={"vol_target": 0.001})
        
        # Strategy 3: Magnitude-weighted
        print(f"\n{'='*60}")
        print(f"  Strategy: Magnitude-Weighted")
        walk_forward(sym, strategy_fn=strategy_magnitude_weighted)
        
        # Strategy 4: Filtered (only strong signals)
        print(f"\n{'='*60}")
        print(f"  Strategy: Filtered (min_strength=1.0)")
        walk_forward(sym, strategy_fn=strategy_filtered,
                     strategy_kwargs={"min_strength": 1.0})
