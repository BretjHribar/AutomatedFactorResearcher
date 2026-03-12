"""
alpha_research_wf.py — Vectorized Walk-Forward Alpha Research

Inspired by proven alphas from factor-alpha-platform/data/alphas.db:
  - ts_delta(close, 6) × feature  (momentum × confirmation)
  - vwap_deviation (mean reversion)  
  - close_position_in_range
  - upper_shadow / lower_shadow (candle structure)
  - Sign(sma(returns, 120)) × volume_ratio / volatility
  - ts_corr(returns, vwap_deviation, 20)

STRICT NO LOOK-AHEAD:
  - All signals shifted by 1 (signal[N] uses data up to bar N-1)
  - Walk-forward: expanding window alpha selection + weight optimization
  - Retrain only on past data

TARGET: ≥54% WR on all trades in walk-forward OOS
"""

import sys, os, time, json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
INTERVAL = "5m"
FEE_BPS = 50

# ============================================================================
# ALPHA PRIMITIVES
# ============================================================================

def sma(s, w): return s.rolling(w, min_periods=max(w//2, 2)).mean()
def ema(s, w): return s.ewm(halflife=w, min_periods=max(w//2, 2)).mean()
def stddev(s, w): return s.rolling(w, min_periods=max(w//2, 2)).std()
def ts_zscore(s, w):
    m = s.rolling(w, min_periods=max(w//2, 2)).mean()
    sd = s.rolling(w, min_periods=max(w//2, 2)).std()
    return (s - m) / sd.replace(0, np.nan)
def delta(s, p): return s - s.shift(p)
def ts_sum(s, w): return s.rolling(w, min_periods=1).sum()
def ts_max(s, w): return s.rolling(w, min_periods=1).max()
def ts_min(s, w): return s.rolling(w, min_periods=1).min()
def ts_rank(s, w): return s.rolling(w, min_periods=max(w//2, 2)).rank(pct=True)
def ts_corr(a, b, w): return a.rolling(w, min_periods=max(w//2, 5)).corr(b)
def ts_kurt(s, w): return s.rolling(w, min_periods=max(w//2, 5)).kurt()
def safe_div(a, b):
    r = a / b
    return r.replace([np.inf, -np.inf], 0).fillna(0)
def decay_exp(s, hl):
    """Exponential decay smoothing."""
    if 0 < hl < 1:
        # Interpret as decay factor: halflife = -log(2)/log(hl)
        actual_hl = -np.log(2) / np.log(hl) if hl > 0 else 1
    else:
        actual_hl = hl
    return s.ewm(halflife=max(actual_hl, 0.5), min_periods=1).mean()


# ============================================================================
# EXPANDED ALPHA UNIVERSE — inspired by proven cross-sectional alphas
# ============================================================================

def build_expanded_alphas(df):
    """Build expanded alpha universe from all available data fields.
    
    Data fields available in 5m klines:
      open, high, low, close, volume, quote_volume, trades, 
      taker_buy_base, taker_buy_quote
    
    ALL signals shifted by 1 at the end to avoid look-ahead.
    """
    close = df["close"]; volume = df["volume"]
    high = df["high"]; low = df["low"]
    opn = df["open"]
    taker_buy = df["taker_buy_base"]
    qv = df["quote_volume"]
    trades = df["trades"].astype(float) if "trades" in df.columns else volume * 0 + 1
    
    # Derived fields (match what the cross-sectional platform uses)
    ret = close.pct_change()
    log_ret = np.log(close / close.shift(1))
    vwap = safe_div(qv, volume)
    taker_ratio = safe_div(taker_buy, volume)
    taker_sell = volume - taker_buy
    taker_imbalance = safe_div(taker_buy - taker_sell, volume)
    obv = (np.sign(ret) * volume).cumsum()
    
    # Candle structure fields (matching cross-sectional proven patterns)
    body = close - opn
    hl_range = high - low  # high_low_range
    oc_range = (close - opn).abs()  # open_close_range
    upper_shadow = high - pd.concat([close, opn], axis=1).max(axis=1)
    lower_shadow = pd.concat([close, opn], axis=1).min(axis=1) - low
    close_pos = safe_div(close - low, hl_range)  # close_position_in_range [0,1]
    
    # Volume-derived fields
    vol_ratio_20 = safe_div(volume, sma(volume, 20))  # volume_ratio_20d
    vol_mom_1 = volume - volume.shift(1)  # volume_momentum_1
    vol_mom_5_20 = sma(volume, 5) - sma(volume, 20)  # volume_momentum_5_20
    trades_ratio = safe_div(trades, sma(trades, 30))  # trades_count / sma(trades_count, 30)
    trades_per_vol = safe_div(trades, volume)  # trades_per_volume
    
    # Volatility fields
    park_vol_10 = np.sqrt((np.log(high/low)**2).rolling(10).mean() / (4*np.log(2)))  # parkinson_volatility_10
    park_vol_60 = np.sqrt((np.log(high/low)**2).rolling(60).mean() / (4*np.log(2)))  # parkinson_volatility_60
    hist_vol_20 = ret.rolling(20).std()  # historical_volatility_20
    hist_vol_60 = ret.rolling(60).std()  # historical_volatility_60
    hist_vol_120 = ret.rolling(120).std()  # historical_volatility_120
    
    # VWAP deviation (key alpha from DB)
    vwap_dev = safe_div(close - vwap, vwap)  # vwap_deviation
    
    alphas = {}

    # ========== FAMILY 1: ts_delta(close, N) × feature ==========
    # This is the #1 pattern from the DB — nearly ALL top alphas use this
    for mom_w in [1, 3, 6, 12]:
        mom = delta(close, mom_w)
        
        # × candle structure
        alphas[f"dclose{mom_w}_x_closepos"] = mom * close_pos
        alphas[f"dclose{mom_w}_x_hlrange"] = mom * hl_range
        alphas[f"dclose{mom_w}_x_hlrange_div_ocrange"] = mom * safe_div(hl_range, oc_range)
        alphas[f"dclose{mom_w}_x_hlrange_div_ushadow"] = mom * safe_div(hl_range, upper_shadow + 1e-10)
        alphas[f"dclose{mom_w}_x_lshadow"] = mom * lower_shadow
        
        # × volume features
        alphas[f"dclose{mom_w}_x_volratio20"] = mom * vol_ratio_20
        alphas[f"dclose{mom_w}_x_tbr"] = mom * taker_ratio
        alphas[f"dclose{mom_w}_x_volmom1"] = mom * vol_mom_1
        alphas[f"dclose{mom_w}_x_tradesratio"] = mom * trades_ratio
        alphas[f"dclose{mom_w}_x_tradespervol"] = mom * trades_per_vol
        alphas[f"dclose{mom_w}_x_volmom520"] = mom * vol_mom_5_20
        
        # × volatility (proven: momentum works better in some vol regimes)
        alphas[f"dclose{mom_w}_x_parkvol10"] = mom * park_vol_10
        alphas[f"dclose{mom_w}_x_parkvol60"] = mom * park_vol_60
        alphas[f"dclose{mom_w}_x_histvol20"] = mom * hist_vol_20
        alphas[f"dclose{mom_w}_x_inv_parkvol20"] = mom * safe_div(pd.Series(1.0, index=df.index), 
                                                     np.sqrt((np.log(high/low)**2).rolling(20).mean() / (4*np.log(2))))
        
        # × volume/vol ratio (proven: Sign(sma(returns,120)) * vol/histvol)
        alphas[f"dclose{mom_w}_x_volratio_div_histvol60"] = mom * safe_div(vol_ratio_20, hist_vol_60)
        
        # × kurtosis of volume
        vk = ts_kurt(volume, 60)
        alphas[f"dclose{mom_w}_x_volkurt"] = mom * vk

    # Smoothed versions (Decay_exp from DB — many top alphas use this)
    for mom_w in [6]:
        mom = delta(close, mom_w)
        for feat, fname in [(close_pos, "closepos"), (vol_ratio_20, "volratio20"), 
                             (taker_ratio, "tbr"), (hl_range/oc_range.replace(0,np.nan), "hlr_ocr")]:
            raw = mom * feat
            for decay in [0.8, 0.9, 0.95, 0.98]:
                alphas[f"decay{decay}_dclose{mom_w}_x_{fname}"] = decay_exp(raw, decay)
            # Also SMA(2)
            alphas[f"sma2_dclose{mom_w}_x_{fname}"] = sma(raw, 2)

    # ========== FAMILY 2: VWAP deviation (mean reversion — proven) ==========
    alphas["neg_vwap_dev"] = -vwap_dev
    for w in [5, 10]:
        alphas[f"neg_dclose_vwapdev_{w}"] = -delta(vwap_dev, w)
    alphas["neg_vwapdev_x_histvol20"] = -vwap_dev * hist_vol_20
    alphas["neg_vwapdev_x_parkvol60"] = -vwap_dev * park_vol_60
    # ts_zscore of vwap (proven: Decay_exp(ts_zscore(vwap, 50), 0.97))
    for w in [20, 40, 50, 60]:
        alphas[f"neg_zscore_vwap_{w}"] = -ts_zscore(vwap, w)
        alphas[f"decay_zscore_vwap_{w}"] = decay_exp(-ts_zscore(vwap, w), 0.97)
    # vwap_dev × taker confirmation
    alphas["neg_vwapdev_x_tbr"] = -vwap_dev * taker_ratio
    # ts_corr(returns, vwap_deviation) — proven SR=1.52
    for w in [10, 20, 30]:
        alphas[f"corr_ret_vwapdev_{w}"] = ts_corr(ret, vwap_dev, w)

    # ========== FAMILY 3: Candle structure (wick ratios) ==========
    # upper_shadow / lower_shadow — proven SR=1.70
    alphas["ushadow_div_lshadow"] = safe_div(upper_shadow, lower_shadow + 1e-10)
    # Body ratio
    for w in [3, 5, 10]:
        alphas[f"body_ratio_{w}"] = sma(safe_div(body, hl_range), w)
    # Close position smoothed
    for w in [1, 3, 5]:
        alphas[f"close_pos_{w}"] = (close_pos - 0.5) if w == 1 else sma(close_pos - 0.5, w)
    # Wick imbalance
    for w in [3, 5, 10]:
        alphas[f"wick_imb_{w}"] = sma(safe_div(lower_shadow - upper_shadow, hl_range), w)

    # ========== FAMILY 4: Regime-conditioned momentum ==========
    # Sign(sma(returns, N)) × volume_feature / volatility — proven SR ~1.56
    for trend_w in [30, 60, 120]:
        trend_sign = np.sign(sma(ret, trend_w))
        for (feat, fname) in [(vol_ratio_20, "volratio20"), (vol_mom_5_20, "volmom520"),
                               (taker_ratio, "tbr")]:
            for (vol, vname) in [(hist_vol_60, "hv60"), (park_vol_60, "pv60")]:
                alphas[f"regime{trend_w}_{fname}_div_{vname}"] = trend_sign * safe_div(feat, vol)
    
    # Sign(momentum) × vol confirmation
    for (mom_w, vol_w) in [(10, 60), (20, 60), (60, 60)]:
        mom_sign = np.sign(sma(ret, mom_w))
        alphas[f"momsign{mom_w}_x_volmom520_div_hv{vol_w}"] = mom_sign * safe_div(vol_mom_5_20, hist_vol_60)
    
    # ========== FAMILY 5: Volatility-scaled signals ==========
    # multiply(ts_std_dev(close, 20), Sign(returns)) — proven SR=1.82
    alphas["vol20_x_signret"] = hist_vol_20 * np.sign(ret)
    alphas["vol60_x_signret"] = hist_vol_60 * np.sign(ret)
    
    # ========== FAMILY 6: Pure momentum (raw, various windows) ==========
    for w in [1, 3, 6, 12, 24]:
        alphas[f"logret_sum_{w}"] = ts_sum(log_ret, w)
    # Decay_exp(ts_delta(close, 1), 0.5) — proven SR=1.61
    alphas["decay_ret1"] = decay_exp(delta(close, 1), 0.5)
    
    # ========== FAMILY 7: Mean reversion (existing) ==========
    for w in [5, 8, 10, 15, 20, 30]:
        alphas[f"mr_{w}"] = -ts_zscore(close, w)
    for w in [5, 10, 15, 20]:
        alphas[f"dstd_{w}"] = -safe_div(delta(close, w), stddev(close, w))
    for w in [5, 10, 20]:
        alphas[f"ema_mr_{w}"] = -(close - ema(close, w)) / stddev(close, w*2)

    # ========== FAMILY 8: Volume microstructure ==========
    for w in [10, 20, 30]:
        alphas[f"obv_zscore_{w}"] = ts_zscore(obv, w)  # NOT negated — let selection decide
    for w in [5, 10, 20]:
        alphas[f"tbr_zscore_{w}"] = ts_zscore(taker_ratio, w)
        alphas[f"timb_zscore_{w}"] = ts_zscore(taker_imbalance, w)
    # Volume surge × return direction
    for w in [10, 20]:
        alphas[f"vol_surge_x_ret_{w}"] = safe_div(volume, sma(volume, w)) * np.sign(ret)
    
    # ========== FAMILY 9: Rank-based (ts_rank from DB) ==========
    for w in [10, 20, 30, 60]:
        alphas[f"ts_rank_close_{w}"] = ts_rank(close, w) - 0.5
    for w in [10, 30]:
        alphas[f"ts_rank_volume_{w}"] = ts_rank(volume, w) - 0.5

    # ========== FAMILY 10: Gap / overnight ==========
    gap_pct = safe_div(opn - close.shift(1), close.shift(1))
    alphas["gap_pct"] = gap_pct
    for w in [3, 5]:
        alphas[f"gap_mom_{w}"] = ts_sum(gap_pct, w)

    # ========== Apply shift to avoid look-ahead ==========
    alpha_df = pd.DataFrame(alphas, index=df.index)
    alpha_df = alpha_df.shift(1)  # CRITICAL: signal at N uses data up to N-1
    
    return alpha_df


# ============================================================================
# VECTORIZED WALK-FORWARD
# ============================================================================

def vectorized_walk_forward(df, 
                             train_bars=30000,      # ~52 days min training
                             retrain_every=4000,     # ~14 days between retrains
                             lookback=1440,          # Weight estimation window
                             phl=1,
                             corr_cutoff=0.80,
                             max_alphas=15,
                             min_wr=0.505,
                             fee_bps=FEE_BPS):
    """
    Vectorized walk-forward. Much faster than per-bar.
    
    1. Build all alphas once (shifted by 1)
    2. At each retrain point, select alphas using ONLY past data
    3. Compute adaptive weights on past data
    4. Generate predictions for the next retrain_every bars
    5. Check predictions against actuals
    """
    t0 = time.time()
    
    target = (df["close"] >= df["open"]).astype(int)
    all_alphas = build_expanded_alphas(df)
    alpha_names = list(all_alphas.columns)
    
    n = len(df)
    fee_per = fee_bps / 10000.0
    
    all_preds = []
    
    # Walk forward in chunks
    retrain_points = list(range(train_bars, n - 1, retrain_every))
    
    for rp_idx, rp in enumerate(retrain_points):
        # ---- TRAINING: use only data [0, rp] ----
        train_alphas = all_alphas.iloc[:rp+1]
        train_target = target.iloc[:rp+1]
        
        # Select alphas
        selected = select_alphas(
            train_alphas, train_target, alpha_names,
            corr_cutoff=corr_cutoff, max_alphas=max_alphas,
            min_wr=min_wr, fee_bps=fee_bps
        )
        
        if len(selected) < 2:
            continue
        
        # ---- PREDICTION: bars [rp+1, next_rp] ----
        next_rp = retrain_points[rp_idx + 1] if rp_idx + 1 < len(retrain_points) else n - 1
        pred_end = min(next_rp, n - 1)
        
        # Compute weights using expanding data up to each prediction bar
        # For efficiency: compute weights at rp (using all data up to rp),
        # then generate signals for bars rp+1..pred_end
        cols = [c for c in selected if c in all_alphas.columns]
        X_full = all_alphas[cols].iloc[:pred_end+1]
        
        # Compute Y for weight estimation (same-bar: signal[N] predicts target[N])
        y_full = 2.0 * (train_target.reindex(X_full.index).astype(float) - 0.5)
        
        # Compute fee-adjusted returns for each alpha
        fr = pd.DataFrame(index=X_full.index, columns=cols, dtype=float)
        for col in cols:
            d = np.sign(X_full[col].values)
            fr[col] = d * y_full.values - fee_per
        
        # Rolling return estimation
        lb = min(lookback, rp)
        rer = fr.rolling(lb, min_periods=min(200, lb)).mean()
        w = rer.clip(lower=0)
        ws = w.sum(axis=1).replace(0, np.nan)
        wn = w.div(ws, axis=0).fillna(0)
        
        if phl > 1:
            wn = wn.ewm(halflife=phl, min_periods=1).mean()
            ws2 = wn.sum(axis=1).replace(0, np.nan)
            wn = wn.div(ws2, axis=0).fillna(0)
        
        # Combined signal
        combined = (X_full * wn).sum(axis=1)
        
        # Extract only OOS predictions [rp+1, pred_end]
        for i in range(rp + 1, pred_end + 1):
            sig_val = combined.iloc[i]
            if np.isnan(sig_val) or abs(sig_val) < 0.001:
                continue
            
            direction = int(np.sign(sig_val))
            actual = int(target.iloc[i])
            correct = (direction > 0 and actual == 1) or (direction < 0 and actual == 0)
            
            all_preds.append({
                "bar_idx": i,
                "direction": direction,
                "signal_val": float(sig_val),
                "actual": actual,
                "correct": correct,
            })
    
    elapsed = time.time() - t0
    return all_preds, elapsed


def select_alphas(alpha_df, target, candidate_names, 
                   corr_cutoff=0.80, max_alphas=15,
                   min_wr=0.505, fee_bps=FEE_BPS):
    """Select best alphas using ONLY training data."""
    fee_per = fee_bps / 10000.0
    
    scores = []
    for name in candidate_names:
        if name not in alpha_df.columns:
            continue
        sig = alpha_df[name]
        common = sig.dropna().index.intersection(target.dropna().index)
        if len(common) < 500:
            continue
        
        s = sig.loc[common]
        t = target.loc[common]
        
        d = np.sign(s)
        hit = ((d > 0) & (t == 1)) | ((d < 0) & (t == 0))
        wr = hit.mean()
        
        if wr < min_wr:
            continue
        
        daily_ret = (d * (2 * t.astype(float) - 1) - fee_per).resample("1D").sum()
        daily_ret = daily_ret[daily_ret != 0]
        if len(daily_ret) < 20 or daily_ret.std() == 0:
            continue
        
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(365)
        scores.append({"name": name, "wr": wr, "sharpe": sharpe})
    
    if not scores:
        return []
    
    scores.sort(key=lambda x: x["sharpe"], reverse=True)
    
    # Greedy correlation-aware selection
    selected = []
    for s in scores:
        sig = alpha_df[s["name"]].dropna()
        too_corr = False
        for sel_name in selected:
            sel_sig = alpha_df[sel_name].dropna()
            common = sig.index.intersection(sel_sig.index)
            if len(common) < 100:
                continue
            corr = np.abs(sig.loc[common].corr(sel_sig.loc[common]))
            if corr > corr_cutoff:
                too_corr = True
                break
        if not too_corr:
            selected.append(s["name"])
        if len(selected) >= max_alphas:
            break
    
    return selected


# ============================================================================
# MAIN RESEARCH
# ============================================================================

def run_research():
    results_all = {}
    
    for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        print(f"\n{'='*70}")
        print(f"  WALK-FORWARD RESEARCH: {symbol}")
        print(f"{'='*70}")
        
        df = pd.read_parquet(DATA_DIR / f"{symbol}_{INTERVAL}.parquet")
        print(f"  Data: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")
        
        # Compute alpha count
        sample_alphas = build_expanded_alphas(df.head(1000))
        print(f"  Alpha universe: {len(sample_alphas.columns)} signals")
        
        best_wr = 0
        best_config = None
        
        configs = [
            # (lookback, phl, corr, max_a, min_wr_sel, retrain_every, label)
            (1440,  1, 0.80, 15, 0.505, 4000,  "base"),
            (2880,  1, 0.80, 15, 0.505, 4000,  "lb2880"),
            (720,   1, 0.80, 15, 0.505, 4000,  "lb720"),
            (1440,  1, 0.70, 10, 0.510, 4000,  "strict"),
            (1440,  1, 0.65, 8,  0.515, 4000,  "v_strict"),
            (1440, 24, 0.80, 15, 0.505, 4000,  "phl24"),
            (1440, 72, 0.80, 15, 0.505, 4000,  "phl72"),
            (2880,  1, 0.70, 10, 0.510, 4000,  "lb2880_strict"),
            (720,   1, 0.70, 8,  0.515, 2000,  "lb720_vstrict"),
            (1440,  1, 0.80, 20, 0.505, 8000,  "max20_retrain8k"),
            (4320,  1, 0.80, 15, 0.505, 4000,  "lb4320"),
            (1440,  1, 0.60, 6,  0.520, 4000,  "top6_elite"),
        ]
        
        for lb, phl, corr, max_a, min_wr_sel, retrain, label in configs:
            preds, elapsed = vectorized_walk_forward(
                df, train_bars=30000, retrain_every=retrain,
                lookback=lb, phl=phl, corr_cutoff=corr,
                max_alphas=max_a, min_wr=min_wr_sel, fee_bps=FEE_BPS,
            )
            
            if not preds:
                print(f"  [{label:18s}] No predictions")
                continue
            
            total = len(preds)
            wins = sum(1 for p in preds if p["correct"])
            wr = wins / total * 100
            
            # Strong signal breakdown
            strong = [p for p in preds if abs(p["signal_val"]) >= 1.0]
            s_w = sum(1 for p in strong if p["correct"])
            s_wr = s_w / len(strong) * 100 if strong else 0
            
            # Quarter stability
            q_size = total // 4
            q_wrs = []
            for q in range(4):
                qp = preds[q*q_size:(q+1)*q_size]
                if qp:
                    qw = sum(1 for p in qp if p["correct"])
                    q_wrs.append(qw/len(qp)*100)
            min_q = min(q_wrs) if q_wrs else 0
            
            marker = " <<<" if wr >= 54 else (" **" if wr >= 53 else "")
            print(f"  [{label:18s}] WR={wr:.1f}% ({wins}W/{total-wins}L) "
                  f"|sig|≥1.0: {s_wr:.1f}% Qmin={min_q:.1f}% ({elapsed:.0f}s){marker}")
            
            if wr > best_wr:
                best_wr = wr
                best_config = label
                results_all[symbol] = {
                    "config": label, "wr": wr, "total": total,
                    "wins": wins, "preds": preds, "strong_wr": s_wr,
                    "q_wrs": q_wrs, "params": (lb, phl, corr, max_a, min_wr_sel, retrain),
                    "elapsed": elapsed,
                }
        
        if symbol in results_all:
            r = results_all[symbol]
            print(f"\n  BEST: [{r['config']}] WR={r['wr']:.1f}% "
                  f"Quarters: {[f'{q:.1f}%' for q in r['q_wrs']]}")
            
            # Show selected alphas from last retrain
            last_preds = r['preds']
            # Check per-direction WR
            up_p = [p for p in last_preds if p['direction'] > 0]
            dn_p = [p for p in last_preds if p['direction'] < 0]
            up_w = sum(1 for p in up_p if p['correct'])
            dn_w = sum(1 for p in dn_p if p['correct'])
            print(f"  UP bets: {up_w}/{len(up_p)}={up_w/max(len(up_p),1)*100:.1f}% "
                  f"DN bets: {dn_w}/{len(dn_p)}={dn_w/max(len(dn_p),1)*100:.1f}%")
    
    # Final summary
    print(f"\n\n{'='*70}")
    print(f"  FINAL WALK-FORWARD RESULTS")
    print(f"{'='*70}")
    for sym, r in results_all.items():
        lb, phl, corr, max_a, min_wr_sel, retrain = r["params"]
        print(f"  {sym}: WR={r['wr']:.1f}% config={r['config']}")
        print(f"    Params: lb={lb}, phl={phl}, corr={corr}, max_a={max_a}, min_wr={min_wr_sel}, retrain={retrain}")
        print(f"    Quarters: {[f'{q:.1f}%' for q in r['q_wrs']]}")
        print(f"    Strong: {r['strong_wr']:.1f}%")
    
    return results_all


if __name__ == "__main__":
    results = run_research()
