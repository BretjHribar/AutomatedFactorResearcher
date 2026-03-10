"""
iterate_5m.py — Institutional-grade alpha discovery and portfolio construction
for 5-minute Polymarket crypto candle contracts.

Architecture (matching factor-alpha-platform):
  1. ALPHA DISCOVERY — No fees. Evaluate raw predictive power via IC and 
     no-fee Sharpe. Store all passing alphas in DB.
  2. PORTFOLIO CONSTRUCTION — Rolling ridge regression to combine alphas.
     Walk-forward: train on expanding window, predict next bar.
     Fees applied only here. Position smoothing for turnover control.

Primitives from crypto_ops.py: ts_zscore, delta, sma, stddev, etc.
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, LinearRegression
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (SYMBOLS, SYMBOL_NAMES, DATA_DIR, TRAIN_START, TRAIN_END,
                     HOLDOUT_START, HOLDOUT_END, BASE_TRADE_SIZE, BLENDED_TAKER_FEE)

INTERVAL = "5m"
BARS_PER_DAY = 288

# Quality gates (alpha discovery)
MIN_IC = 0.01          # Minimum mean IC to pass
MIN_NOFEE_SHARPE = 3.0 # Minimum no-fee Sharpe to pass
CORR_CUTOFF = 0.70     # Max pairwise correlation for inclusion

# Portfolio construction
RIDGE_ALPHA = 1.0          # Ridge regularization
RIDGE_TRAIN_WINDOW = 5760  # Rolling window = 20 days of 5m bars
RIDGE_RETRAIN_EVERY = 288  # Retrain daily
POS_HALFLIFE = 36          # Position smoothing half-life (bars)
MIN_EDGE_TO_TRADE = 0.0    # Minimum predicted probability edge to trade

# ============================================================================
# ALPHA PRIMITIVES (from crypto_ops.py)
# ============================================================================

def sma(s, w): return s.rolling(w, min_periods=1).mean()
def stddev(s, w): return s.rolling(w, min_periods=2).std()
def ts_zscore(s, w):
    m = s.rolling(w, min_periods=2).mean()
    sd = s.rolling(w, min_periods=2).std()
    return (s - m) / sd.replace(0, np.nan)
def delta(s, p): return s - s.shift(p)
def delay(s, p): return s.shift(p)
def ts_sum(s, w): return s.rolling(w, min_periods=1).sum()
def ts_min(s, w): return s.rolling(w, min_periods=1).min()
def ts_max(s, w): return s.rolling(w, min_periods=1).max()
def safe_div(a, b):
    r = a / b; return r.replace([np.inf, -np.inf], 0).fillna(0)
def correlation(x, y, w): return x.rolling(w, min_periods=2).corr(y)
def ts_rank(s, w): return s.rolling(w, min_periods=1).apply(
    lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False)

# ============================================================================
# ALPHA GENERATION
# ============================================================================

def build_alpha_matrix(df):
    """Generate all alpha signals from raw OHLCV. Returns DataFrame of signals.
    All signals are SHIFTED by 1 bar to prevent lookahead."""
    
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
    
    # --- Mean reversion: -ts_zscore(close, N) ---
    for w in [5, 8, 10, 12, 15, 20, 24, 30, 36, 48]:
        alphas[f"mr_close_{w}"] = -ts_zscore(close, w)
    
    # --- Cumulative log return reversal ---
    for w in [3, 5, 8, 10, 12, 15, 20, 24]:
        alphas[f"logret_rev_{w}"] = -ts_sum(log_ret, w)
    
    # --- Normalized delta: -delta(close,N)/stddev(close,N) ---
    for w in [5, 8, 12, 20]:
        alphas[f"delta_std_{w}"] = -safe_div(delta(close, w), stddev(close, w))
    
    # --- VWAP z-score reversal ---
    for w in [5, 10, 20, 30]:
        alphas[f"mr_vwap_{w}"] = -ts_zscore(vwap, w)
    
    # --- Close-VWAP deviation ---
    dev = close - vwap
    for w in [5, 10, 20]:
        alphas[f"close_vwap_dev_{w}"] = -ts_zscore(dev, w)
    
    # --- Volume zscore (information arrival) ---
    for w in [5, 10, 20]:
        alphas[f"vol_z_{w}"] = ts_zscore(volume, w)
    
    # --- Taker buy ratio z-score (order flow) ---
    for w in [5, 10, 20]:
        alphas[f"taker_z_{w}"] = ts_zscore(taker_ratio, w)
    
    # --- Delta taker ratio ---
    for w in [3, 5, 10]:
        alphas[f"delta_taker_{w}"] = delta(taker_ratio, w)
    
    # --- Close position in range ---
    close_pos = safe_div(close - low, high - low)
    for w in [5, 10, 20]:
        alphas[f"close_pos_z_{w}"] = -ts_zscore(close_pos, w)
    
    # --- Body z-score (candle strength) ---
    body = close - opn
    for w in [5, 10]:
        alphas[f"body_z_{w}"] = -ts_zscore(body, w)
    
    # --- Range expansion ---
    hl_range = safe_div(high - low, close)
    for w in [10, 20]:
        alphas[f"range_z_{w}"] = -ts_zscore(hl_range, w)
    
    # --- OBV momentum ---
    signed_vol_cum = (np.sign(ret) * volume).cumsum()
    for w in [10, 20]:
        alphas[f"obv_z_{w}"] = -ts_zscore(signed_vol_cum, w)
    
    # --- Acceleration ---
    for w in [5, 8, 12]:
        alphas[f"accel_{w}"] = -ts_zscore(delta(sma(ret, w), w), w * 2)
    
    # --- Trade intensity ---
    intensity = safe_div(trades, volume)
    for w in [10, 20]:
        alphas[f"intensity_z_{w}"] = ts_zscore(intensity, w)
    
    # Build DataFrame, shift by 1 to avoid lookahead
    alpha_df = pd.DataFrame(alphas, index=df.index)
    alpha_df = alpha_df.shift(1)  # Signal at bar t uses data up to t-1
    
    return alpha_df

# ============================================================================
# PHASE 1: ALPHA EVALUATION (NO FEES)
# ============================================================================

def evaluate_alpha_nofee(signal, target):
    """Evaluate a single alpha signal WITHOUT fees.
    
    Returns dict with:
      ic_mean: mean information coefficient (correlation with binary return)
      hit_rate: fraction of correct directional calls
      nofee_sharpe: annualized Sharpe of $1 directional bets (no fees)
      stability: sub-period hit-rate stability
    """
    binary_return = 2.0 * (target.astype(float) - 0.5)  # +1 or -1
    
    common = signal.dropna().index.intersection(binary_return.dropna().index)
    if len(common) < 500:
        return None
    
    sig = signal.loc[common]
    ret = binary_return.loc[common]
    outcome = target.loc[common]
    
    # IC: rolling correlation with binary return
    ic_series = sig.rolling(BARS_PER_DAY, min_periods=100).corr(ret)
    ic_mean = ic_series.dropna().mean()
    
    # Hit rate
    direction = np.sign(sig)
    correct = (direction * ret > 0)
    hit_rate = correct.mean()
    
    # No-fee PnL and Sharpe
    nofee_pnl = direction * ret  # +1 or -1 per bar
    daily_pnl = nofee_pnl.resample("1D").sum()
    daily_pnl = daily_pnl[daily_pnl != 0]
    
    if len(daily_pnl) > 10 and daily_pnl.std() > 0:
        nofee_sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(365)
    else:
        nofee_sharpe = 0.0
    
    # Sub-period stability
    n = len(common)
    q1 = (direction.iloc[:n//3] * ret.iloc[:n//3] > 0).mean()
    q2 = (direction.iloc[n//3:2*n//3] * ret.iloc[n//3:2*n//3] > 0).mean()
    q3 = (direction.iloc[2*n//3:] * ret.iloc[2*n//3:] > 0).mean()
    stability = 1.0 - np.std([q1, q2, q3])  # Higher = more stable
    
    return {
        "ic_mean": ic_mean,
        "hit_rate": hit_rate,
        "nofee_sharpe": nofee_sharpe,
        "edge_bps": (hit_rate - 0.5) * 10000,
        "stability": stability,
        "q1_hr": q1,
        "q2_hr": q2,
        "q3_hr": q3,
        "n_bars": len(common),
    }


def discover_alphas(df, target, min_ic=MIN_IC, min_sharpe=MIN_NOFEE_SHARPE):
    """Phase 1: Discover and rank all alpha signals (NO FEES).
    Returns sorted list of passing alphas."""
    
    alpha_matrix = build_alpha_matrix(df)
    
    results = []
    for col in alpha_matrix.columns:
        m = evaluate_alpha_nofee(alpha_matrix[col], target)
        if m is None:
            continue
        results.append({"name": col, **m})
    
    # Sort by no-fee Sharpe
    results.sort(key=lambda x: x["nofee_sharpe"], reverse=True)
    
    # Quality gate
    passing = [r for r in results if r["ic_mean"] > min_ic and r["nofee_sharpe"] > min_sharpe]
    
    return results, passing, alpha_matrix


# ============================================================================
# PHASE 2: PORTFOLIO CONSTRUCTION — ROLLING RIDGE REGRESSION
# ============================================================================

def rolling_regression_portfolio(alpha_matrix, target, df,
                             selected_alphas,
                             model_type="ridge",
                             ridge_alpha=RIDGE_ALPHA,
                             train_window=RIDGE_TRAIN_WINDOW,
                             retrain_every=RIDGE_RETRAIN_EVERY,
                             pos_halflife=POS_HALFLIFE,
                             fee_rate=BLENDED_TAKER_FEE,
                             trade_size=BASE_TRADE_SIZE):
    """
    Walk-forward rolling regression portfolio construction.
    
    Supports: 'ridge' (Ridge regression) or 'ols' (OLS linear regression).
    
    1. At each rebalance point, fit model on the trailing window:
         y = binary_return (+1/-1), X = alpha signals
    2. Predict next bar's direction using fitted coefficients
    3. Apply position smoothing (EMA) to reduce turnover
    4. Compute PnL with Polymarket fees applied
    """
    cols = [a["name"] for a in selected_alphas if a["name"] in alpha_matrix.columns]
    X_full = alpha_matrix[cols].copy()
    y_full = 2.0 * (target.astype(float) - 0.5)  # +1 or -1
    outcomes = target.copy()
    
    # Align
    valid = X_full.dropna().index.intersection(y_full.dropna().index)
    X_full = X_full.loc[valid]
    y_full = y_full.loc[valid]
    outcomes = outcomes.loc[valid]
    
    n = len(valid)
    if n < train_window + BARS_PER_DAY:
        print(f"    Insufficient data: {n} bars, need {train_window + BARS_PER_DAY}")
        return None
    
    # Walk-forward
    raw_signal = np.zeros(n)
    predictions = np.zeros(n)
    coefficients = []
    
    if model_type == "ols":
        model = LinearRegression(fit_intercept=True)
    else:
        model = Ridge(alpha=ridge_alpha, fit_intercept=True)
    last_coefs = None
    
    for i in range(train_window, n, retrain_every):
        # Train on trailing window
        train_end = i
        train_start = max(0, i - train_window)
        
        X_train = X_full.iloc[train_start:train_end].values
        y_train = y_full.iloc[train_start:train_end].values
        
        # Remove NaN rows
        mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        if mask.sum() < 100:
            continue
        
        model.fit(X_train[mask], y_train[mask])
        last_coefs = model.coef_.copy()
        coefficients.append({
            "bar": i,
            "coefs": dict(zip(cols, model.coef_)),
            "intercept": model.intercept_,
        })
        
        # Predict next retrain_every bars
        pred_end = min(i + retrain_every, n)
        X_pred = X_full.iloc[i:pred_end].values
        
        # Handle NaN in prediction features
        X_pred = np.nan_to_num(X_pred, 0)
        preds = model.predict(X_pred)
        raw_signal[i:pred_end] = preds
    
    # Position smoothing: EMA of raw signal
    decay = np.exp(-np.log(2) / pos_halflife)
    smoothed = np.zeros(n)
    smoothed[train_window] = raw_signal[train_window]
    for i in range(train_window + 1, n):
        smoothed[i] = decay * smoothed[i-1] + (1 - decay) * raw_signal[i]
    
    # Direction: sign of smoothed signal
    direction = np.sign(smoothed)
    
    # PnL with fees
    pnl = np.zeros(n)
    traded = np.zeros(n, dtype=bool)
    fees_total = 0.0
    
    for i in range(train_window, n):
        if direction[i] == 0:
            continue
        
        traded[i] = True
        entry_price = 0.50
        
        # Dynamic fee
        fee = 0.02 * entry_price * (1 - entry_price) * trade_size  # ~0.5% at p=0.50
        fees_total += fee
        
        if direction[i] > 0:  # Bet UP
            if outcomes.iloc[i] == 1:
                pnl[i] = (1.0 - entry_price) * trade_size - fee
            else:
                pnl[i] = -entry_price * trade_size - fee
        else:  # Bet DOWN
            if outcomes.iloc[i] == 0:
                pnl[i] = entry_price * trade_size - fee
            else:
                pnl[i] = -(1.0 - entry_price) * trade_size - fee
    
    # Build results
    pnl_series = pd.Series(pnl, index=valid)
    trade_mask = traded
    traded_pnl = pnl[trade_mask]
    
    if trade_mask.sum() == 0:
        return None
    
    daily_pnl = pnl_series.resample("1D").sum()
    daily_pnl = daily_pnl[daily_pnl != 0]
    
    if len(daily_pnl) > 10 and daily_pnl.std() > 0:
        sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(365)
    else:
        sharpe = 0.0
    
    cumulative = pnl_series.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_dd = drawdown.min()
    
    wins = (traded_pnl > 0).sum()
    total_trades = trade_mask.sum()
    
    # No-fee comparison
    nofee_pnl = np.zeros(n)
    for i in range(train_window, n):
        if direction[i] != 0:
            if direction[i] > 0:
                nofee_pnl[i] = (1.0 - 0.5) * trade_size if outcomes.iloc[i] == 1 else -0.5 * trade_size
            else:
                nofee_pnl[i] = 0.5 * trade_size if outcomes.iloc[i] == 0 else -0.5 * trade_size
    nofee_daily = pd.Series(nofee_pnl, index=valid).resample("1D").sum()
    nofee_daily = nofee_daily[nofee_daily != 0]
    nofee_sharpe = nofee_daily.mean() / nofee_daily.std() * np.sqrt(365) if len(nofee_daily) > 10 and nofee_daily.std() > 0 else 0
    
    return {
        "sharpe": sharpe,
        "nofee_sharpe": nofee_sharpe,
        "win_rate": wins / max(total_trades, 1),
        "total_trades": int(total_trades),
        "net_pnl": traded_pnl.sum(),
        "total_fees": fees_total,
        "max_drawdown": max_dd,
        "trades_per_day": total_trades / max(len(daily_pnl), 1),
        "pnl_series": pnl_series,
        "cumulative_pnl": cumulative,
        "daily_pnl": daily_pnl,
        "coefficients": coefficients,
        "n_alphas": len(cols),
        "alpha_names": cols,
        "model_type": model_type,
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(symbol):
    """Full alpha discovery + portfolio construction pipeline for one asset."""
    name = SYMBOL_NAMES[symbol]
    path = DATA_DIR / f"{symbol}_{INTERVAL}.parquet"
    if not path.exists():
        print(f"  SKIP {symbol}: no data")
        return None
    
    df = pd.read_parquet(path)
    train_df = df.loc[TRAIN_START:TRAIN_END]
    holdout_df = df.loc[HOLDOUT_START:HOLDOUT_END]
    target_train = (train_df["close"] >= train_df["open"]).astype(int)
    target_holdout = (holdout_df["close"] >= holdout_df["open"]).astype(int)
    
    # =================================================================
    # PHASE 1: ALPHA DISCOVERY (NO FEES)
    # =================================================================
    print(f"\n{'='*80}")
    print(f"  {name} {INTERVAL} — PHASE 1: ALPHA DISCOVERY (NO FEES)")
    print(f"  Train: {TRAIN_START} → {TRAIN_END} ({len(train_df)} bars)")
    print(f"{'='*80}")
    
    all_results, passing, train_alphas = discover_alphas(
        train_df, target_train, min_ic=MIN_IC, min_sharpe=MIN_NOFEE_SHARPE)
    
    # Print top alphas
    print(f"\n  {'Name':<22} {'IC':>7} {'HitRate':>8} {'Edge(bp)':>9} "
          f"{'NoFee SR':>9} {'Q1 HR':>7} {'Q2 HR':>7} {'Q3 HR':>7}")
    print(f"  {'-'*85}")
    for r in all_results[:25]:
        stable = "*" if r["stability"] > 0.995 else " "
        gate = "PASS" if r in passing else "    "
        print(f"  {r['name']:<22} {r['ic_mean']:>7.4f} {r['hit_rate']:>7.1%} "
              f"{r['edge_bps']:>8.0f}bp {r['nofee_sharpe']:>9.2f} "
              f"{r['q1_hr']:>6.1%} {r['q2_hr']:>6.1%} {r['q3_hr']:>6.1%} {gate}{stable}")
    
    print(f"\n  Total alphas: {len(all_results)} | Passing: {len(passing)}")
    
    if not passing:
        # Relax gate if nothing passes
        print(f"  No alphas passed MIN_SHARPE={MIN_NOFEE_SHARPE}. Relaxing to top 10.")
        passing = [r for r in all_results 
                   if r["ic_mean"] > 0.005 and r["nofee_sharpe"] > 1.0][:10]
        if not passing:
            passing = all_results[:10]
    
    # Correlation-based filtering
    selected = []
    for candidate in passing:
        if candidate["name"] not in train_alphas.columns:
            continue
        sig = train_alphas[candidate["name"]]
        too_corr = False
        for sel in selected:
            corr_val = sig.corr(train_alphas[sel["name"]])
            if abs(corr_val) > CORR_CUTOFF:
                too_corr = True
                break
        if not too_corr:
            selected.append(candidate)
    
    print(f"  After correlation filter (corr < {CORR_CUTOFF}): {len(selected)} alphas")
    for s in selected:
        print(f"    {s['name']:<25} IC={s['ic_mean']:.4f}  NoFee SR={s['nofee_sharpe']:.2f}")
    
    # =================================================================
    # PHASE 2: PORTFOLIO CONSTRUCTION (Rolling Ridge, WITH FEES)
    # =================================================================
    print(f"\n{'='*80}")
    print(f"  {name} {INTERVAL} — PHASE 2: ROLLING RIDGE PORTFOLIO CONSTRUCTION")
    print(f"{'='*80}")
    
    # Grid search over ridge parameters
    best_result = None
    best_params = None
    best_sharpe = -999
    
    param_grid = []
    # Ridge variants
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        for phl in [12, 24, 36, 72]:
            for tw in [2880, 5760, 8640]:  # 10, 20, 30 days
                param_grid.append({"model_type": "ridge", "ridge_alpha": alpha,
                                   "pos_halflife": phl, "train_window": tw})
    # OLS variants
    for phl in [12, 24, 36, 72]:
        for tw in [2880, 5760, 8640]:
            param_grid.append({"model_type": "ols", "ridge_alpha": 0,
                               "pos_halflife": phl, "train_window": tw})
    
    print(f"  Sweeping {len(param_grid)} parameter combinations (Ridge + OLS)...")
    print(f"  {'Model':>8} {'alpha':>6} {'phl':>5} {'window':>7} {'Sharpe':>8} {'NoFee SR':>9} "
          f"{'WR':>7} {'PnL':>10} {'MaxDD':>10} {'Trades':>8}")
    print(f"  {'-'*85}")
    
    for params in param_grid:
        result = rolling_regression_portfolio(
            train_alphas, target_train, train_df,
            selected,
            model_type=params["model_type"],
            ridge_alpha=params["ridge_alpha"],
            train_window=params["train_window"],
            pos_halflife=params["pos_halflife"],
        )
        if result is None:
            continue
        
        if result["sharpe"] > best_sharpe:
            best_sharpe = result["sharpe"]
            best_result = result
            best_params = params
            mt = params['model_type'].upper()
            print(f"  {mt:>8} {params['ridge_alpha']:>6.1f} {params['pos_halflife']:>5} "
                  f"{params['train_window']:>7} {result['sharpe']:>8.2f} "
                  f"{result['nofee_sharpe']:>9.2f} {result['win_rate']:>6.1%} "
                  f"${result['net_pnl']:>9,.0f} ${result['max_drawdown']:>9,.0f} "
                  f"{result['total_trades']:>8} <-- BEST")
    
    if best_result is None:
        print(f"  No valid portfolio found.")
        return None
    
    print(f"\n  BEST TRAIN RESULT:")
    print(f"    Model:    {best_params['model_type'].upper()}")
    print(f"    Params:   ridge_alpha={best_params['ridge_alpha']}, "
          f"pos_halflife={best_params['pos_halflife']}, "
          f"train_window={best_params['train_window']}")
    print(f"    Sharpe:   {best_result['sharpe']:.2f} (with fees)")
    print(f"    NoFee SR: {best_result['nofee_sharpe']:.2f}")
    print(f"    Win Rate: {best_result['win_rate']:.1%}")
    print(f"    Net PnL:  ${best_result['net_pnl']:,.0f}")
    print(f"    Fees:     ${best_result['total_fees']:,.0f}")
    print(f"    Max DD:   ${best_result['max_drawdown']:,.0f}")
    print(f"    Ridge coefs (last window):")
    if best_result["coefficients"]:
        last_coefs = best_result["coefficients"][-1]["coefs"]
        for cname, cval in sorted(last_coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
            print(f"      {cname:<25} {cval:>+.4f}")
    
    # =================================================================
    # PHASE 3: HOLDOUT VALIDATION
    # =================================================================
    print(f"\n{'='*80}")
    print(f"  {name} {INTERVAL} — PHASE 3: HOLDOUT VALIDATION")
    print(f"  Holdout: {HOLDOUT_START} → {HOLDOUT_END} ({len(holdout_df)} bars)")
    print(f"{'='*80}")
    
    holdout_alphas = build_alpha_matrix(holdout_df)
    
    ho_result = rolling_regression_portfolio(
        holdout_alphas, target_holdout, holdout_df,
        selected,
        model_type=best_params["model_type"],
        ridge_alpha=best_params["ridge_alpha"],
        train_window=best_params["train_window"],
        pos_halflife=best_params["pos_halflife"],
    )
    
    if ho_result:
        print(f"    Sharpe:   {ho_result['sharpe']:.2f} (with fees)")
        print(f"    NoFee SR: {ho_result['nofee_sharpe']:.2f}")
        print(f"    Win Rate: {ho_result['win_rate']:.1%}")
        print(f"    Net PnL:  ${ho_result['net_pnl']:,.0f}")
        print(f"    Max DD:   ${ho_result['max_drawdown']:,.0f}")
    else:
        print(f"    Holdout evaluation failed.")
    
    return {
        "symbol": symbol,
        "name": name,
        "selected_alphas": [s["name"] for s in selected],
        "best_params": best_params,
        "train": best_result,
        "holdout": ho_result,
    }


def main():
    print(f"{'#'*80}")
    print(f"# POLYMARKET 5m — INSTITUTIONAL ALPHA + PORTFOLIO PIPELINE")
    print(f"# Phase 1: Alpha discovery (no fees, IC + no-fee Sharpe)")
    print(f"# Phase 2: Rolling ridge regression portfolio (with fees)")
    print(f"# Phase 3: Holdout validation")
    print(f"{'#'*80}")
    
    all_results = {}
    
    for symbol in SYMBOLS:
        result = run_pipeline(symbol)
        if result:
            all_results[SYMBOL_NAMES[symbol]] = result
    
    # Final summary
    print(f"\n\n{'='*80}")
    print(f"  FINAL PORTFOLIO SUMMARY — 5m Contracts")
    print(f"{'='*80}")
    for name, r in all_results.items():
        tr = r["train"]
        ho = r["holdout"]
        print(f"\n  {name} 5m:")
        print(f"    Alphas:          {r['selected_alphas']}")
        print(f"    Ridge params:    alpha={r['best_params']['ridge_alpha']}, "
              f"phl={r['best_params']['pos_halflife']}, "
              f"window={r['best_params']['train_window']}")
        print(f"    Train:           Sharpe={tr['sharpe']:.2f}  NoFee={tr['nofee_sharpe']:.2f}  "
              f"WR={tr['win_rate']:.1%}  PnL=${tr['net_pnl']:,.0f}")
        if ho:
            print(f"    Holdout:         Sharpe={ho['sharpe']:.2f}  NoFee={ho['nofee_sharpe']:.2f}  "
                  f"WR={ho['win_rate']:.1%}  PnL=${ho['net_pnl']:,.0f}")


if __name__ == "__main__":
    main()
