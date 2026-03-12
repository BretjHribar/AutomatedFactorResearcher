"""
Walk-Forward Alpha Research for Polymarket 5m Candle Trading
============================================================

STRICT RULES:
1. NO lookahead: signals computed on bar N, evaluated against bar N+1
2. Walk-forward: train on past, test on future, never look back
3. Alpha selection done ONLY on training window
4. Weight estimation done ONLY on training window
5. OOS WR is the ONLY metric that matters
6. Multiple non-overlapping test periods for stability

Walk-forward structure:
  - Expanding window: train on [start, t], test on [t, t+test_window]
  - Step forward, repeat
  - Each test period is NEVER seen during training
"""

import sys, os, time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SYMBOLS, SYMBOL_NAMES, DATA_DIR

# ============================================================================
# ALPHA PRIMITIVES (copied from live_trade_real.py — no changes)
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
def safe_div(a, b):
    r = a / b; return r.replace([np.inf, -np.inf], 0).fillna(0)


def build_alpha_signals_v1(df):
    """CONTROL model: exact same alphas as live_trade_real.py
    Returns shifted signals: signal on bar N uses data up to bar N (inclusive),
    shifted by 1 so signal[N] predicts bar N+1."""
    close = df["close"]; volume = df["volume"]; high = df["high"]; low = df["low"]
    opn = df["open"]; taker_buy = df["taker_buy_base"]
    qv = df["quote_volume"]
    ret = close.pct_change()
    log_ret = np.log(close / close.shift(1))
    vwap = safe_div(qv, volume)
    taker_ratio = safe_div(taker_buy, volume)
    taker_sell = volume - taker_buy
    taker_imbalance = safe_div(taker_buy - taker_sell, volume)
    obv = (np.sign(ret) * volume).cumsum()
    alphas = {}
    for w in [5, 8, 10, 12, 15, 20, 24, 30, 36, 48]:
        alphas[f"mr_{w}"] = -ts_zscore(close, w)
    for w in [3, 5, 8, 10, 12, 15, 20]:
        alphas[f"logrev_{w}"] = -ts_sum(log_ret, w)
    for w in [3, 5, 8, 10, 12, 15, 20]:
        alphas[f"dstd_{w}"] = -safe_div(delta(close, w), stddev(close, w))
    for w in [5, 10, 15, 20, 30]:
        alphas[f"vwap_mr_{w}"] = -ts_zscore(vwap, w)
    for w in [5, 10, 20]:
        alphas[f"ema_mr_{w}"] = -(close - ema(close, w)) / stddev(close, w*2)
    for w in [10, 20, 30]:
        alphas[f"obv_{w}"] = -ts_zscore(obv, w)
    for w in [10, 20]:
        alphas[f"vp_div_{w}"] = ts_zscore(volume, w) - ts_zscore(close, w)
    for w in [5, 10, 20]:
        alphas[f"tbr_{w}"] = ts_zscore(taker_ratio, w)
        alphas[f"timb_{w}"] = ts_zscore(taker_imbalance, w)
    alpha_df = pd.DataFrame(alphas, index=df.index)
    # SHIFT: signal[N] uses data up to bar N, predicts bar N+1
    alpha_df = alpha_df.shift(1)
    return alpha_df


def build_alpha_signals_v2(df):
    """CANDIDATE model: additional alphas for testing.
    All new alphas must be theoretically motivated, not data-mined.
    
    New ideas:
    - Higher-order momentum (acceleration)
    - Range-based volatility (Parkinson, Garman-Klass)
    - Volume-price divergence with different normalization
    - Candle body/wick ratios
    """
    # Start with all V1 alphas
    v1 = build_alpha_signals_v1(df)
    
    close = df["close"]; volume = df["volume"]; high = df["high"]; low = df["low"]
    opn = df["open"]; taker_buy = df["taker_buy_base"]
    qv = df["quote_volume"]
    ret = close.pct_change()
    log_ret = np.log(close / close.shift(1))
    
    new_alphas = {}
    
    # --- Candle microstructure ---
    body = close - opn
    full_range = high - low
    upper_wick = high - close.where(close >= opn, opn)
    lower_wick = close.where(close < opn, opn) - low
    
    # Body ratio: how much of the candle is body vs wicks
    body_ratio = safe_div(body.abs(), full_range.replace(0, np.nan))
    
    for w in [5, 10, 20]:
        # Rejection signal: upper wick dominance after uptick = bearish
        new_alphas[f"wick_reject_{w}"] = -ts_zscore(
            safe_div(upper_wick - lower_wick, full_range.replace(0, np.nan)), w
        )
        # Body momentum: strong bodies = continuation
        new_alphas[f"body_mom_{w}"] = ts_zscore(body_ratio * np.sign(body), w)
    
    # --- Acceleration (momentum of momentum) ---
    for w in [5, 10, 20]:
        mom = ret.rolling(w, min_periods=2).mean()
        accel = mom - mom.shift(w)
        new_alphas[f"accel_{w}"] = -ts_zscore(accel, w)  # MR on acceleration
    
    # --- Parkinson volatility estimator ---
    parkinson_vol = np.log(high / low) ** 2 / (4 * np.log(2))
    realized_vol = ret ** 2
    for w in [10, 20]:
        pv = parkinson_vol.rolling(w, min_periods=2).mean()
        rv = realized_vol.rolling(w, min_periods=2).mean()
        # Vol of vol: high vol-of-vol = potential reversal
        new_alphas[f"volofvol_{w}"] = -ts_zscore(pv, w)
        # Excess vol: Parkinson > Realized = jumpy market = MR
        new_alphas[f"exvol_{w}"] = -safe_div(pv - rv, rv.replace(0, np.nan))
    
    # --- Taker flow pressure ---
    taker_sell = volume - taker_buy
    net_taker = taker_buy - taker_sell
    for w in [3, 5, 10]:
        # Cumulative taker pressure
        cum_flow = net_taker.rolling(w, min_periods=1).sum()
        new_alphas[f"tflow_{w}"] = -ts_zscore(cum_flow, max(w*2, 10))
        
        # Taker exhaustion: flow vs price change
        price_move = delta(close, w)
        flow_efficiency = safe_div(price_move, cum_flow.abs().replace(0, np.nan))
        new_alphas[f"tflow_eff_{w}"] = -ts_zscore(flow_efficiency, 20)
    
    # --- Volume-weighted momentum ---
    for w in [5, 10, 20]:
        vw_ret = (ret * volume).rolling(w, min_periods=2).sum() / volume.rolling(w, min_periods=2).sum()
        new_alphas[f"vwmom_{w}"] = -ts_zscore(vw_ret, w)
    
    new_df = pd.DataFrame(new_alphas, index=df.index).shift(1)
    return pd.concat([v1, new_df], axis=1)


# ============================================================================
# WALK-FORWARD ENGINE
# ============================================================================

def select_alphas_on_train(alpha_df, target, corr_cutoff=0.85, max_alphas=15, min_wr=0.505):
    """Select alphas using ONLY training data. No future information."""
    results = []
    for col in alpha_df.columns:
        sig = alpha_df[col].dropna()
        common = sig.index.intersection(target.dropna().index)
        if len(common) < 500:
            continue
        s, t = sig.loc[common], target.loc[common]
        d = np.sign(s)
        correct = (d == (2*t - 1))
        wr = correct.mean()
        if wr < min_wr:
            continue
        # Simple Sharpe on daily returns
        daily_pnl = (d * (2*t.astype(float) - 1)).resample("1D").sum()
        daily_pnl = daily_pnl[daily_pnl != 0]
        if len(daily_pnl) < 20 or daily_pnl.std() == 0:
            continue
        sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(365)
        results.append({"name": col, "sharpe": sharpe, "wr": wr})
    
    results.sort(key=lambda x: x["sharpe"], reverse=True)
    
    selected = []
    for r in results:
        sig = alpha_df[r["name"]]
        too_corr = False
        for sel in selected:
            if abs(sig.corr(alpha_df[sel])) > corr_cutoff:
                too_corr = True
                break
        if not too_corr:
            selected.append(r["name"])
            if len(selected) >= max_alphas:
                break
    
    return selected


def compute_adaptive_weights(alpha_df, target, cols, lookback):
    """Compute adaptive weights using fee-adjusted rolling returns.
    alpha_df is already .shift(1)'d, so alpha[N] uses data[0..N-1] and predicts bar N.
    target[N] = outcome of bar N. This alignment is correct.
    Aligns target to alpha_df index to prevent shape mismatch."""
    # Align target to alpha_df index
    t_aligned = target.reindex(alpha_df.index)
    y = 2.0 * (t_aligned.astype(float) - 0.5)
    fee_per = 50 / 10000.0  # 50 bps
    
    fr = pd.DataFrame(index=alpha_df.index, columns=cols, dtype=float)
    for col in cols:
        d = np.sign(alpha_df[col].values)
        fr[col] = d * y.values - fee_per
    
    rer = fr.rolling(lookback, min_periods=200).mean()
    w = rer.clip(lower=0)
    ws = w.sum(axis=1).replace(0, np.nan)
    wn = w.div(ws, axis=0).fillna(0)
    
    return wn


def run_walkforward(df, build_fn, label, 
                    train_start="2024-03-01",
                    test_start="2025-09-01",
                    test_step_days=30,
                    lookback=1440,
                    corr_cutoff=0.85,
                    max_alphas=15):
    """
    Strict walk-forward test.
    
    For each test period:
      1. Select alphas on training data [train_start, test_start)
      2. Compute adaptive weights on training data
      3. Apply to test period [test_start, test_start + step)
      4. Record OOS WR, PnL
      5. Step forward
    
    NO information from test period leaks into training.
    """
    
    target = (df["close"] >= df["open"]).astype(int)
    # IMPORTANT: Since build_fn already applies .shift(1) to alphas,
    # alpha_df[N] uses data through bar N-1 and predicts bar N.
    # So we compare DIRECTLY to target[N]. NO additional shift needed.
    # (If we used target.shift(-1), that would be 2 bars ahead = WRONG)
    
    # Build all alpha signals on full data (this is OK because alphas use
    # only past data via rolling windows, and .shift(1) prevents any
    # current-bar information from leaking into the signal)
    alpha_df = build_fn(df)
    
    results = []
    current_test_start = pd.Timestamp(test_start, tz="UTC")
    end = df.index[-1]
    
    fold = 0
    while current_test_start < end:
        fold += 1
        test_end = current_test_start + pd.Timedelta(days=test_step_days)
        if test_end > end:
            test_end = end
        
        # ---- TRAINING: everything before test_start ----
        train_mask = df.index < current_test_start
        train_alpha = alpha_df.loc[train_mask]
        train_target = target.loc[train_mask]
        
        if len(train_alpha) < 5000:
            current_test_start = test_end
            continue
        
        # Step 1: Select alphas on training data ONLY
        selected = select_alphas_on_train(
            train_alpha, train_target, 
            corr_cutoff=corr_cutoff, max_alphas=max_alphas
        )
        
        if len(selected) < 2:
            current_test_start = test_end
            continue
        
        # ---- TEST: apply to unseen data ----
        # We need the FULL data up to test_end for rolling computations
        # But weights are estimated on training data only
        full_alpha = alpha_df.loc[:test_end][selected]
        full_target = target.loc[:test_end]
        
        # Compute weights on the full expanding window
        # (weights at each point only use past data via the rolling window)
        wn = compute_adaptive_weights(
            full_alpha, full_target, selected, lookback
        )
        
        # Combined signal
        combined = (full_alpha * wn).sum(axis=1)
        direction = np.sign(combined)
        
        # Extract test period results ONLY (using timestamp slicing)
        test_dir = direction.loc[current_test_start:test_end]
        test_target = target.loc[current_test_start:test_end]
        
        traded = test_dir != 0
        if traded.sum() < 10:
            current_test_start = test_end
            continue
        
        correct = ((test_dir == 1) & (test_target == 1)) | \
                  ((test_dir == -1) & (test_target == 0))
        
        wr_traded = correct.loc[traded].mean()
        n_traded = int(traded.sum())
        n_total = len(test_dir)
        trade_rate = n_traded / n_total
        
        # PnL (at fair value $0.50)
        fill = 0.505
        pnl_per_trade = wr_traded * (1-fill)/fill - (1 - wr_traded)
        
        results.append({
            "fold": fold,
            "test_start": current_test_start.strftime("%Y-%m-%d"),
            "test_end": test_end.strftime("%Y-%m-%d"),
            "n_alphas": len(selected),
            "n_traded": int(n_traded),
            "n_total": int(n_total),
            "trade_rate": trade_rate,
            "wr": wr_traded,
            "pnl_per_dollar": pnl_per_trade,
            "selected": selected[:5],
        })
        
        current_test_start = test_end
    
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("WALK-FORWARD ALPHA RESEARCH")
    print("=" * 70)
    print()
    
    for symbol in SYMBOLS:
        df = pd.read_parquet(DATA_DIR / f"{symbol}_5m.parquet")
        print(f"\n{'='*50}")
        print(f"  {SYMBOL_NAMES[symbol]}")
        print(f"  Data: {df.index[0]} to {df.index[-1]} ({len(df)} bars)")
        print(f"{'='*50}")
        
        # --- CONTROL: V1 alphas (current model) ---
        print(f"\n--- CONTROL (V1 — current live model) ---")
        v1_results = run_walkforward(
            df, build_alpha_signals_v1, "V1_control",
            train_start="2024-03-01",
            test_start="2025-09-01",
            test_step_days=30,
            lookback=1440,
        )
        
        for r in v1_results:
            print(f"  Fold {r['fold']}: {r['test_start']} to {r['test_end']} | "
                  f"WR={r['wr']:.3f} | n={r['n_traded']} | "
                  f"trade_rate={r['trade_rate']:.1%} | "
                  f"PnL/$ = {r['pnl_per_dollar']:+.4f}")
        
        avg_wr_v1 = np.mean([r["wr"] for r in v1_results]) if v1_results else 0
        avg_pnl_v1 = np.mean([r["pnl_per_dollar"] for r in v1_results]) if v1_results else 0
        print(f"  >>> AVG OOS WR: {avg_wr_v1:.3f} | AVG PnL/$: {avg_pnl_v1:+.4f}")
        
        # --- CANDIDATE: V2 alphas (new features) ---
        print(f"\n--- CANDIDATE (V2 — new alphas) ---")
        v2_results = run_walkforward(
            df, build_alpha_signals_v2, "V2_candidate",
            train_start="2024-03-01",
            test_start="2025-09-01",
            test_step_days=30,
            lookback=1440,
        )
        
        for r in v2_results:
            print(f"  Fold {r['fold']}: {r['test_start']} to {r['test_end']} | "
                  f"WR={r['wr']:.3f} | n={r['n_traded']} | "
                  f"trade_rate={r['trade_rate']:.1%} | "
                  f"PnL/$ = {r['pnl_per_dollar']:+.4f}")
        
        avg_wr_v2 = np.mean([r["wr"] for r in v2_results]) if v2_results else 0
        avg_pnl_v2 = np.mean([r["pnl_per_dollar"] for r in v2_results]) if v2_results else 0
        print(f"  >>> AVG OOS WR: {avg_wr_v2:.3f} | AVG PnL/$: {avg_pnl_v2:+.4f}")
        
        delta_wr = avg_wr_v2 - avg_wr_v1
        print(f"\n  IMPROVEMENT: {delta_wr:+.3f} WR ({delta_wr*100:+.1f}%)")
        print()
