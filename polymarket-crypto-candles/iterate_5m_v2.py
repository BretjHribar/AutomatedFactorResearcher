"""
iterate_5m_v2.py — Advanced portfolio construction for Polymarket 5m candles.

Architecture:
  Phase 1: Alpha discovery (no fees) — IC + no-fee Sharpe evaluation
  Phase 2: Portfolio construction with multiple strategies:
    a) Rolling Ridge regression (baseline)
    b) Adaptive weighting (rolling net-of-fee factor returns)
    c) Ridge + regime scaling (vol-adjusted exposure)
    d) Expanding window ridge (uses all available history)
  Phase 3: Holdout validation

Target: Sharpe > 10 after fees out-of-sample.
"""
import sys, os, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, LinearRegression
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (SYMBOLS, SYMBOL_NAMES, DATA_DIR, TRAIN_START, TRAIN_END,
                     HOLDOUT_START, HOLDOUT_END, BASE_TRADE_SIZE)

INTERVAL = "5m"
BARS_PER_DAY = 288

# Quality gates
MIN_IC = 0.005
CORR_CUTOFF = 0.70

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
def safe_div(a, b):
    r = a / b; return r.replace([np.inf, -np.inf], 0).fillna(0)
def correlation(x, y, w): return x.rolling(w, min_periods=2).corr(y)

# ============================================================================
# ALPHA GENERATION — EXPANDED LIBRARY
# ============================================================================

def build_alpha_matrix(df):
    """Generate all alpha signals from raw OHLCV."""
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
    
    alphas = {}

    # Mean reversion: -ts_zscore(close, N)
    for w in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 24, 30, 36, 48]:
        alphas[f"mr_{w}"] = -ts_zscore(close, w)

    # Cumulative return reversal
    for w in [3, 4, 5, 6, 8, 10, 12, 15, 20, 24]:
        alphas[f"logrev_{w}"] = -ts_sum(log_ret, w)

    # Normalized delta
    for w in [3, 5, 8, 10, 12, 15, 20]:
        alphas[f"dstd_{w}"] = -safe_div(delta(close, w), stddev(close, w))

    # VWAP z-score
    for w in [5, 10, 15, 20, 30]:
        alphas[f"vwap_mr_{w}"] = -ts_zscore(vwap, w)

    # Close-VWAP deviation
    dev = close - vwap
    for w in [5, 10, 20]:
        alphas[f"cvdev_{w}"] = -ts_zscore(dev, w)

    # OBV momentum
    obv = (np.sign(ret) * volume).cumsum()
    for w in [10, 20, 30]:
        alphas[f"obv_{w}"] = -ts_zscore(obv, w)

    # Acceleration
    for w in [5, 8, 12]:
        alphas[f"accel_{w}"] = -ts_zscore(delta(sma(ret, w), w), w * 2)

    # Volume z-score (signed by return)
    for w in [5, 10, 20]:
        alphas[f"vol_signed_{w}"] = ts_zscore(volume, w) * np.sign(ret)

    # Taker buy ratio
    for w in [5, 10, 20]:
        alphas[f"tbr_{w}"] = ts_zscore(taker_ratio, w)

    # Delta taker
    for w in [3, 5, 10]:
        alphas[f"dtaker_{w}"] = delta(taker_ratio, w)

    # Close position in range
    close_pos = safe_div(close - low, high - low)
    for w in [5, 10, 20]:
        alphas[f"cpos_{w}"] = -ts_zscore(close_pos, w)

    # Body
    body = close - opn
    for w in [5, 10]:
        alphas[f"body_{w}"] = -ts_zscore(body, w)

    # Range expansion
    hl = safe_div(high - low, close)
    for w in [10, 20]:
        alphas[f"rng_{w}"] = -ts_zscore(hl, w)

    # Trade intensity
    intensity = safe_div(trades, volume)
    for w in [10, 20]:
        alphas[f"intens_{w}"] = ts_zscore(intensity, w)

    # EMA-based mean reversion (faster signal)
    for w in [5, 10, 20]:
        alphas[f"ema_mr_{w}"] = -(close - ema(close, w)) / stddev(close, w * 2)

    # High-low midpoint reversion
    mid = (high + low) / 2
    for w in [10, 20]:
        alphas[f"mid_mr_{w}"] = -ts_zscore(close - mid, w)

    # Volume-weighted price reversion
    for w in [10, 20]:
        alphas[f"vw_mr_{w}"] = -ts_zscore(close - vwap, w) * ts_zscore(volume, w).clip(-2, 2)

    # Shift all by 1 (lookahead prevention)
    alpha_df = pd.DataFrame(alphas, index=df.index)
    alpha_df = alpha_df.shift(1)
    return alpha_df


# ============================================================================
# ALPHA EVALUATION (NO FEES)
# ============================================================================

def evaluate_alpha_nofee(signal, target):
    binary_return = 2.0 * (target.astype(float) - 0.5)
    common = signal.dropna().index.intersection(binary_return.dropna().index)
    if len(common) < 500:
        return None

    sig = signal.loc[common]
    ret = binary_return.loc[common]

    ic_series = sig.rolling(BARS_PER_DAY, min_periods=100).corr(ret)
    ic_mean = ic_series.dropna().mean()

    direction = np.sign(sig)
    hit_rate = (direction * ret > 0).mean()

    nofee_pnl = direction * ret
    daily_pnl = nofee_pnl.resample("1D").sum()
    daily_pnl = daily_pnl[daily_pnl != 0]
    nofee_sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(365)
                    if len(daily_pnl) > 10 and daily_pnl.std() > 0 else 0.0)

    n = len(common)
    q1 = (direction.iloc[:n//3] * ret.iloc[:n//3] > 0).mean()
    q2 = (direction.iloc[n//3:2*n//3] * ret.iloc[n//3:2*n//3] > 0).mean()
    q3 = (direction.iloc[2*n//3:] * ret.iloc[2*n//3:] > 0).mean()

    return {
        "ic_mean": ic_mean, "hit_rate": hit_rate, "nofee_sharpe": nofee_sharpe,
        "edge_bps": (hit_rate - 0.5) * 10000,
        "q1_hr": q1, "q2_hr": q2, "q3_hr": q3, "n_bars": len(common),
    }


def discover_alphas(df, target):
    alpha_matrix = build_alpha_matrix(df)
    results = []
    for col in alpha_matrix.columns:
        m = evaluate_alpha_nofee(alpha_matrix[col], target)
        if m is None:
            continue
        results.append({"name": col, **m})

    results.sort(key=lambda x: x["nofee_sharpe"], reverse=True)
    return results, alpha_matrix


def select_orthogonal(results, alpha_matrix, max_n=15, corr_cutoff=CORR_CUTOFF):
    """Select top orthogonal alphas."""
    selected = []
    for r in results:
        if r["name"] not in alpha_matrix.columns:
            continue
        if r["ic_mean"] < MIN_IC:
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
# PORTFOLIO SIMULATION
# ============================================================================

def simulate_portfolio(direction, outcomes, trade_size=BASE_TRADE_SIZE, 
                       entry_price=0.50):
    """Compute PnL from direction array and binary outcomes.
    Includes dynamic Polymarket fees."""
    n = len(direction)
    pnl = np.zeros(n)
    nofee_pnl = np.zeros(n)
    fees_total = 0.0
    n_trades = 0

    for i in range(n):
        d = direction[i]
        if d == 0:
            continue
        n_trades += 1
        fee = 0.02 * entry_price * (1 - entry_price) * trade_size
        fees_total += fee

        outcome = outcomes.iloc[i]
        if d > 0:  # Bet UP
            gross = ((1.0 - entry_price) if outcome == 1 else -entry_price) * trade_size
        else:  # Bet DOWN
            gross = (entry_price if outcome == 0 else -(1.0 - entry_price)) * trade_size

        pnl[i] = gross - fee
        nofee_pnl[i] = gross + fee  # No-fee is gross

    return pnl, nofee_pnl, n_trades, fees_total


def compute_sharpe(pnl_array, index):
    """Compute annualized Sharpe from per-bar PnL."""
    pnl_series = pd.Series(pnl_array, index=index)
    daily = pnl_series.resample("1D").sum()
    daily = daily[daily != 0]
    if len(daily) > 10 and daily.std() > 0:
        return daily.mean() / daily.std() * np.sqrt(365)
    return 0.0


def compute_metrics(pnl_array, nofee_array, index, n_trades, fees_total):
    """Full metrics from simulation."""
    sharpe = compute_sharpe(pnl_array, index)
    nofee_sharpe = compute_sharpe(nofee_array, index)
    cum = np.cumsum(pnl_array)
    max_dd = (cum - np.maximum.accumulate(cum)).min()
    wins = (pnl_array[pnl_array != 0] > 0).sum()
    total = (pnl_array != 0).sum()
    return {
        "sharpe": sharpe, "nofee_sharpe": nofee_sharpe,
        "win_rate": wins / max(total, 1), "total_trades": int(n_trades),
        "net_pnl": pnl_array.sum(), "total_fees": fees_total,
        "max_drawdown": max_dd,
    }


# ============================================================================
# PORTFOLIO STRATEGIES
# ============================================================================

def strategy_rolling_ridge(alpha_matrix, target, selected, 
                           ridge_alpha=1.0, window=5760, retrain=288, phl=36):
    """Rolling ridge regression with position smoothing."""
    cols = [s["name"] for s in selected if s["name"] in alpha_matrix.columns]
    X = alpha_matrix[cols].copy()
    y = 2.0 * (target.astype(float) - 0.5)
    outcomes = target.copy()
    
    valid = X.dropna().index.intersection(y.dropna().index)
    X, y, outcomes = X.loc[valid], y.loc[valid], outcomes.loc[valid]
    n = len(valid)
    
    if n < window + BARS_PER_DAY:
        return None
    
    raw_signal = np.zeros(n)
    model = Ridge(alpha=ridge_alpha, fit_intercept=True)
    
    for i in range(window, n, retrain):
        s, e = max(0, i - window), i
        Xt, yt = X.iloc[s:e].values, y.iloc[s:e].values
        mask = ~(np.isnan(Xt).any(axis=1) | np.isnan(yt))
        if mask.sum() < 100:
            continue
        model.fit(Xt[mask], yt[mask])
        pe = min(i + retrain, n)
        raw_signal[i:pe] = model.predict(np.nan_to_num(X.iloc[i:pe].values, 0))
    
    # Position smoothing
    decay = np.exp(-np.log(2) / phl)
    smoothed = np.zeros(n)
    smoothed[window] = raw_signal[window]
    for i in range(window + 1, n):
        smoothed[i] = decay * smoothed[i-1] + (1 - decay) * raw_signal[i]
    
    direction = np.sign(smoothed)
    pnl, nofee, nt, fees = simulate_portfolio(direction, outcomes)
    metrics = compute_metrics(pnl, nofee, valid, nt, fees)
    metrics["model_coefs"] = dict(zip(cols, model.coef_)) if hasattr(model, 'coef_') else {}
    return metrics


def strategy_expanding_ridge(alpha_matrix, target, selected,
                              ridge_alpha=10.0, min_window=2880, retrain=288, phl=24):
    """Expanding window ridge — uses ALL available history."""
    cols = [s["name"] for s in selected if s["name"] in alpha_matrix.columns]
    X = alpha_matrix[cols].copy()
    y = 2.0 * (target.astype(float) - 0.5)
    outcomes = target.copy()
    
    valid = X.dropna().index.intersection(y.dropna().index)
    X, y, outcomes = X.loc[valid], y.loc[valid], outcomes.loc[valid]
    n = len(valid)
    
    if n < min_window + BARS_PER_DAY:
        return None
    
    raw_signal = np.zeros(n)
    model = Ridge(alpha=ridge_alpha, fit_intercept=True)
    
    for i in range(min_window, n, retrain):
        Xt, yt = X.iloc[:i].values, y.iloc[:i].values  # EXPANDING: start from 0
        mask = ~(np.isnan(Xt).any(axis=1) | np.isnan(yt))
        if mask.sum() < 100:
            continue
        model.fit(Xt[mask], yt[mask])
        pe = min(i + retrain, n)
        raw_signal[i:pe] = model.predict(np.nan_to_num(X.iloc[i:pe].values, 0))
    
    decay = np.exp(-np.log(2) / phl)
    smoothed = np.zeros(n)
    smoothed[min_window] = raw_signal[min_window]
    for i in range(min_window + 1, n):
        smoothed[i] = decay * smoothed[i-1] + (1 - decay) * raw_signal[i]
    
    direction = np.sign(smoothed)
    pnl, nofee, nt, fees = simulate_portfolio(direction, outcomes)
    return compute_metrics(pnl, nofee, valid, nt, fees)


def strategy_adaptive_net(alpha_matrix, target, selected,
                           lookback=2880, phl=36, fee_per_trade_bps=50):
    """Adaptive weighting by rolling net factor returns (Isichenko-style)."""
    cols = [s["name"] for s in selected if s["name"] in alpha_matrix.columns]
    X = alpha_matrix[cols].copy()
    y = 2.0 * (target.astype(float) - 0.5)
    outcomes = target.copy()
    
    valid = X.dropna().index.intersection(y.dropna().index)
    X, y, outcomes = X.loc[valid], y.loc[valid], outcomes.loc[valid]
    n = len(valid)
    
    if n < lookback + BARS_PER_DAY:
        return None
    
    # Compute per-alpha directional returns with fees
    factor_returns = pd.DataFrame(index=valid, columns=cols, dtype=float)
    fee_per = fee_per_trade_bps / 10000.0
    
    for col in cols:
        sig = X[col]
        d = np.sign(sig.values)
        fr = d * y.values - fee_per  # Net of fees
        factor_returns[col] = fr
    
    # Rolling expected return per factor
    rolling_er = factor_returns.rolling(lookback, min_periods=200).mean()
    
    # Only positive ER factors get weight  
    weights = rolling_er.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)
    
    # Smooth weights
    weights_smooth = weights_norm.ewm(halflife=phl, min_periods=1).mean()
    wsum2 = weights_smooth.sum(axis=1).replace(0, np.nan)
    weights_smooth = weights_smooth.div(wsum2, axis=0).fillna(0)
    
    # Combined signal
    combined = (X * weights_smooth).sum(axis=1)
    direction = np.sign(combined.values)
    
    pnl, nofee, nt, fees = simulate_portfolio(direction, outcomes)
    return compute_metrics(pnl, nofee, valid, nt, fees)


def strategy_ridge_regime(alpha_matrix, target, df, selected,
                           ridge_alpha=10.0, window=5760, retrain=288, phl=36):
    """Ridge regression + volatility regime scaling."""
    cols = [s["name"] for s in selected if s["name"] in alpha_matrix.columns]
    X = alpha_matrix[cols].copy()
    y = 2.0 * (target.astype(float) - 0.5)
    outcomes = target.copy()
    
    valid = X.dropna().index.intersection(y.dropna().index)
    X, y, outcomes = X.loc[valid], y.loc[valid], outcomes.loc[valid]
    df_valid = df.loc[valid]
    n = len(valid)
    
    if n < window + BARS_PER_DAY:
        return None
    
    raw_signal = np.zeros(n)
    model = Ridge(alpha=ridge_alpha, fit_intercept=True)
    
    for i in range(window, n, retrain):
        s, e = max(0, i - window), i
        Xt, yt = X.iloc[s:e].values, y.iloc[s:e].values
        mask = ~(np.isnan(Xt).any(axis=1) | np.isnan(yt))
        if mask.sum() < 100:
            continue
        model.fit(Xt[mask], yt[mask])
        pe = min(i + retrain, n)
        raw_signal[i:pe] = model.predict(np.nan_to_num(X.iloc[i:pe].values, 0))
    
    # Position smoothing
    decay = np.exp(-np.log(2) / phl)
    smoothed = np.zeros(n)
    smoothed[window] = raw_signal[window]
    for i in range(window + 1, n):
        smoothed[i] = decay * smoothed[i-1] + (1 - decay) * raw_signal[i]
    
    # Regime scaling: inverse market volatility
    market_ret = df_valid["close"].pct_change()
    fast_vol = market_ret.rolling(BARS_PER_DAY // 6, min_periods=10).std()  # 2h vol
    slow_vol = market_ret.rolling(BARS_PER_DAY * 5, min_periods=100).std()  # 5d vol
    vol_scalar = (slow_vol / fast_vol.replace(0, np.nan)).clip(0.3, 3.0)
    vol_scalar = vol_scalar.ewm(halflife=BARS_PER_DAY // 2).mean().fillna(1.0)
    
    regime_signal = smoothed * vol_scalar.values
    direction = np.sign(regime_signal)
    
    pnl, nofee, nt, fees = simulate_portfolio(direction, outcomes)
    return compute_metrics(pnl, nofee, valid, nt, fees)


def strategy_ensemble(alpha_matrix, target, df, selected, 
                       ridge_alpha=10.0, window=5760, retrain=288, phl=36):
    """Ensemble: average of ridge, expanding ridge, and adaptive signals.
    Uses diversity across model types for robustness."""
    cols = [s["name"] for s in selected if s["name"] in alpha_matrix.columns]
    X = alpha_matrix[cols].copy()
    y = 2.0 * (target.astype(float) - 0.5)
    outcomes = target.copy()
    
    valid = X.dropna().index.intersection(y.dropna().index)
    X, y, outcomes = X.loc[valid], y.loc[valid], outcomes.loc[valid]
    n = len(valid)
    
    if n < window + BARS_PER_DAY:
        return None
    
    # Model 1: Rolling Ridge
    model1 = Ridge(alpha=ridge_alpha, fit_intercept=True)
    sig1 = np.zeros(n)
    for i in range(window, n, retrain):
        s, e = max(0, i - window), i
        Xt, yt = X.iloc[s:e].values, y.iloc[s:e].values
        mask = ~(np.isnan(Xt).any(axis=1) | np.isnan(yt))
        if mask.sum() < 100: continue
        model1.fit(Xt[mask], yt[mask])
        pe = min(i + retrain, n)
        sig1[i:pe] = model1.predict(np.nan_to_num(X.iloc[i:pe].values, 0))
    
    # Model 2: Expanding Ridge
    model2 = Ridge(alpha=ridge_alpha * 10, fit_intercept=True)
    sig2 = np.zeros(n)
    for i in range(window, n, retrain):
        Xt, yt = X.iloc[:i].values, y.iloc[:i].values
        mask = ~(np.isnan(Xt).any(axis=1) | np.isnan(yt))
        if mask.sum() < 100: continue
        model2.fit(Xt[mask], yt[mask])
        pe = min(i + retrain, n)
        sig2[i:pe] = model2.predict(np.nan_to_num(X.iloc[i:pe].values, 0))
    
    # Model 3: Adaptive net
    fee_per = 50 / 10000.0
    factor_returns = pd.DataFrame(index=valid, columns=cols, dtype=float)
    for col in cols:
        d = np.sign(X[col].values)
        factor_returns[col] = d * y.values - fee_per
    rolling_er = factor_returns.rolling(window, min_periods=200).mean()
    weights = rolling_er.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)
    sig3 = (X * weights_norm).sum(axis=1).values
    
    # Ensemble: average of all three (each normalized by their std)
    for s in [sig1, sig2, sig3]:
        std = np.std(s[s != 0]) if (s != 0).sum() > 0 else 1.0
        if std > 0:
            s[:] = s / std
    
    combined_raw = (sig1 + sig2 + sig3) / 3.0
    
    # Position smoothing
    decay = np.exp(-np.log(2) / phl)
    smoothed = np.zeros(n)
    smoothed[window] = combined_raw[window]
    for i in range(window + 1, n):
        smoothed[i] = decay * smoothed[i-1] + (1 - decay) * combined_raw[i]
    
    direction = np.sign(smoothed)
    pnl, nofee, nt, fees = simulate_portfolio(direction, outcomes)
    return compute_metrics(pnl, nofee, valid, nt, fees)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_full_pipeline(symbol):
    name = SYMBOL_NAMES[symbol]
    path = DATA_DIR / f"{symbol}_{INTERVAL}.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    train_df = df.loc[TRAIN_START:TRAIN_END]
    holdout_df = df.loc[HOLDOUT_START:HOLDOUT_END]
    target_train = (train_df["close"] >= train_df["open"]).astype(int)
    target_holdout = (holdout_df["close"] >= holdout_df["open"]).astype(int)

    # ---- PHASE 1: Alpha Discovery ----
    print(f"\n{'='*80}")
    print(f"  {name} {INTERVAL} — PHASE 1: ALPHA DISCOVERY (NO FEES)")
    print(f"  Train: {len(train_df)} bars | Holdout: {len(holdout_df)} bars")
    print(f"{'='*80}")

    all_results, train_alphas = discover_alphas(train_df, target_train)
    holdout_alphas = build_alpha_matrix(holdout_df)

    print(f"\n  {'Name':<18} {'IC':>7} {'HitRate':>8} {'Edge':>7} {'NoFee SR':>9}")
    print(f"  {'-'*55}")
    for r in all_results[:20]:
        print(f"  {r['name']:<18} {r['ic_mean']:>7.4f} {r['hit_rate']:>7.1%} "
              f"{r['edge_bps']:>5.0f}bp {r['nofee_sharpe']:>9.2f}")

    # Select orthogonal alphas at varying correlation cutoffs
    best_selected = None
    for cutoff in [0.50, 0.60, 0.70, 0.80]:
        for max_n in [8, 12, 15, 20]:
            sel = select_orthogonal(all_results, train_alphas, max_n=max_n, corr_cutoff=cutoff)
            if best_selected is None or len(sel) > len(best_selected):
                best_selected = sel

    selected = best_selected or all_results[:10]
    print(f"\n  Selected {len(selected)} orthogonal alphas:")
    for s in selected:
        print(f"    {s['name']:<20} IC={s['ic_mean']:.4f}  NoFee SR={s['nofee_sharpe']:.2f}")

    # ---- PHASE 2: Portfolio Construction ----
    print(f"\n{'='*80}")
    print(f"  {name} {INTERVAL} — PHASE 2: PORTFOLIO CONSTRUCTION (WITH FEES)")
    print(f"{'='*80}")

    strategies = {}
    
    # Strategy sweeps
    print(f"\n  A) Rolling Ridge Regression:")
    print(f"  {'alpha':>8} {'window':>7} {'phl':>5} {'Sharpe':>8} {'NoFee':>8} {'WR':>7} {'PnL':>12}")
    print(f"  {'-'*60}")
    best_ridge_sr = -999
    for ra in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        for tw in [2880, 5760, 8640, 14400]:
            for phl in [12, 24, 36, 72]:
                m = strategy_rolling_ridge(train_alphas, target_train, selected,
                                           ridge_alpha=ra, window=tw, phl=phl)
                if m and m["sharpe"] > best_ridge_sr:
                    best_ridge_sr = m["sharpe"]
                    strategies["rolling_ridge"] = {**m, "params": {"alpha": ra, "window": tw, "phl": phl}}
                    print(f"  {ra:>8.1f} {tw:>7} {phl:>5} {m['sharpe']:>8.2f} "
                          f"{m['nofee_sharpe']:>8.2f} {m['win_rate']:>6.1%} ${m['net_pnl']:>11,.0f} <-- BEST")

    print(f"\n  B) Expanding Ridge:")
    best_exp_sr = -999
    for ra in [1.0, 10.0, 100.0, 1000.0]:
        for phl in [12, 24, 36]:
            m = strategy_expanding_ridge(train_alphas, target_train, selected,
                                          ridge_alpha=ra, phl=phl)
            if m and m["sharpe"] > best_exp_sr:
                best_exp_sr = m["sharpe"]
                strategies["expanding_ridge"] = {**m, "params": {"alpha": ra, "phl": phl}}
                print(f"  alpha={ra:<6.1f} phl={phl:<4} SR={m['sharpe']:.2f} "
                      f"NoFee={m['nofee_sharpe']:.2f} WR={m['win_rate']:.1%} "
                      f"PnL=${m['net_pnl']:,.0f} <-- BEST")

    print(f"\n  C) Adaptive Net Factor Returns:")
    best_adapt_sr = -999
    for lb in [1440, 2880, 5760, 8640]:
        for phl in [12, 24, 36, 72]:
            m = strategy_adaptive_net(train_alphas, target_train, selected,
                                      lookback=lb, phl=phl)
            if m and m["sharpe"] > best_adapt_sr:
                best_adapt_sr = m["sharpe"]
                strategies["adaptive_net"] = {**m, "params": {"lookback": lb, "phl": phl}}
                print(f"  lb={lb:<6} phl={phl:<4} SR={m['sharpe']:.2f} "
                      f"NoFee={m['nofee_sharpe']:.2f} WR={m['win_rate']:.1%} "
                      f"PnL=${m['net_pnl']:,.0f} <-- BEST")

    print(f"\n  D) Ridge + Regime Scaling:")
    best_regime_sr = -999
    for ra in [1.0, 10.0, 100.0]:
        for tw in [2880, 5760, 8640]:
            for phl in [12, 24, 36]:
                m = strategy_ridge_regime(train_alphas, target_train, train_df, selected,
                                           ridge_alpha=ra, window=tw, phl=phl)
                if m and m["sharpe"] > best_regime_sr:
                    best_regime_sr = m["sharpe"]
                    strategies["ridge_regime"] = {**m, "params": {"alpha": ra, "window": tw, "phl": phl}}
                    print(f"  alpha={ra:<6.1f} w={tw:<6} phl={phl:<4} SR={m['sharpe']:.2f} "
                          f"NoFee={m['nofee_sharpe']:.2f} WR={m['win_rate']:.1%} "
                          f"PnL=${m['net_pnl']:,.0f} <-- BEST")

    print(f"\n  E) Ensemble (Ridge+Expanding+Adaptive):")
    for ra in [1.0, 10.0, 100.0]:
        for phl in [12, 24, 36]:
            m = strategy_ensemble(train_alphas, target_train, train_df, selected,
                                   ridge_alpha=ra, phl=phl)
            if m:
                key = "ensemble"
                if key not in strategies or m["sharpe"] > strategies[key]["sharpe"]:
                    strategies[key] = {**m, "params": {"alpha": ra, "phl": phl}}
                    print(f"  alpha={ra:<6.1f} phl={phl:<4} SR={m['sharpe']:.2f} "
                          f"NoFee={m['nofee_sharpe']:.2f} WR={m['win_rate']:.1%} "
                          f"PnL=${m['net_pnl']:,.0f} <-- BEST")

    # ---- TRAIN SUMMARY ----
    print(f"\n  --- TRAIN RESULTS SUMMARY ---")
    print(f"  {'Strategy':<22} {'Sharpe':>8} {'NoFee SR':>9} {'WR':>7} {'PnL':>12} {'MaxDD':>10}")
    print(f"  {'-'*72}")
    best_strat = None
    best_sr = -999
    for sname, m in sorted(strategies.items(), key=lambda x: x[1]["sharpe"], reverse=True):
        print(f"  {sname:<22} {m['sharpe']:>8.2f} {m['nofee_sharpe']:>9.2f} "
              f"{m['win_rate']:>6.1%} ${m['net_pnl']:>11,.0f} ${m['max_drawdown']:>9,.0f}")
        if m["sharpe"] > best_sr:
            best_sr = m["sharpe"]
            best_strat = sname

    # ---- PHASE 3: HOLDOUT ----
    print(f"\n{'='*80}")
    print(f"  {name} {INTERVAL} — PHASE 3: HOLDOUT (best strategy: {best_strat})")
    print(f"{'='*80}")

    p = strategies[best_strat]["params"]
    ho_results = {}

    # Run ALL strategies on holdout for comparison
    for sname in strategies:
        sp = strategies[sname]["params"]
        if sname == "rolling_ridge":
            ho_m = strategy_rolling_ridge(holdout_alphas, target_holdout, selected,
                                           ridge_alpha=sp["alpha"], window=sp["window"], phl=sp["phl"])
        elif sname == "expanding_ridge":
            ho_m = strategy_expanding_ridge(holdout_alphas, target_holdout, selected,
                                             ridge_alpha=sp["alpha"], phl=sp["phl"])
        elif sname == "adaptive_net":
            ho_m = strategy_adaptive_net(holdout_alphas, target_holdout, selected,
                                          lookback=sp["lookback"], phl=sp["phl"])
        elif sname == "ridge_regime":
            ho_m = strategy_ridge_regime(holdout_alphas, target_holdout, holdout_df, selected,
                                          ridge_alpha=sp["alpha"], window=sp["window"], phl=sp["phl"])
        elif sname == "ensemble":
            ho_m = strategy_ensemble(holdout_alphas, target_holdout, holdout_df, selected,
                                      ridge_alpha=sp["alpha"], phl=sp["phl"])
        else:
            ho_m = None

        if ho_m:
            ho_results[sname] = ho_m

    print(f"\n  {'Strategy':<22} {'HO Sharpe':>10} {'HO NoFee':>9} {'HO WR':>7} {'HO PnL':>12}")
    print(f"  {'-'*65}")
    for sname, m in sorted(ho_results.items(), key=lambda x: x[1]["sharpe"], reverse=True):
        flag = " <-- BEST" if sname == best_strat else ""
        print(f"  {sname:<22} {m['sharpe']:>10.2f} {m['nofee_sharpe']:>9.2f} "
              f"{m['win_rate']:>6.1%} ${m['net_pnl']:>11,.0f}{flag}")

    return {
        "name": name, "selected": [s["name"] for s in selected],
        "train": strategies, "holdout": ho_results, "best_strat": best_strat,
    }


def main():
    print(f"{'#'*80}")
    print(f"# POLYMARKET 5m — INSTITUTIONAL PORTFOLIO PIPELINE v2")
    print(f"# Phase 1: No-fee alpha discovery (IC + Sharpe)")
    print(f"# Phase 2: Multi-strategy portfolio (Ridge, Expanding, Adaptive, Regime, Ensemble)")
    print(f"# Phase 3: Out-of-sample holdout validation")
    print(f"{'#'*80}")

    all_results = {}
    for symbol in SYMBOLS:
        result = run_full_pipeline(symbol)
        if result:
            all_results[result["name"]] = result

    print(f"\n\n{'='*80}")
    print(f"  FINAL CROSS-ASSET SUMMARY")
    print(f"{'='*80}")
    for name, r in all_results.items():
        bs = r["best_strat"]
        tr = r["train"][bs]
        ho = r["holdout"].get(bs, {})
        print(f"\n  {name} 5m ({bs}):")
        print(f"    Alphas:  {r['selected'][:5]}... ({len(r['selected'])} total)")
        print(f"    Train:   SR={tr['sharpe']:.2f}  NoFee={tr['nofee_sharpe']:.2f}  "
              f"WR={tr['win_rate']:.1%}  PnL=${tr['net_pnl']:,.0f}")
        if ho:
            print(f"    Holdout: SR={ho['sharpe']:.2f}  NoFee={ho['nofee_sharpe']:.2f}  "
                  f"WR={ho['win_rate']:.1%}  PnL=${ho['net_pnl']:,.0f}")


if __name__ == "__main__":
    main()
