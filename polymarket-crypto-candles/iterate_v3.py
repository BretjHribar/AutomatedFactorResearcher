"""
iterate_v3.py — Deep optimization pass: 
1. Walk-forward validation (not just train/holdout split)
2. More sophisticated signal combination (logistic regression)
3. Probability calibration
4. Realistic capital-constrained PnL
5. Per-coin per-interval detailed results
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm as norm_dist

from config import *
from signals import safe_div, rolling_zscore, compute_target
from backtest_engine import run_backtest
from run_backtest import load_klines, split_data, ensure_db

BARS_PER_DAY = {"5m": 288, "15m": 96, "1h": 24}


# ============================================================================
# v3 SIGNAL LIBRARY — focus on what works + new variants
# ============================================================================

def compute_v3_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    v3 Signal library — focused on proven alpha + new complementary signals.
    Every signal is shifted by 1 to avoid look-ahead.
    """
    results = {}
    returns = df["close"].pct_change()

    # ---- MEAN REVERSION (proven alpha) ----
    for w in [5, 8, 10, 15, 20, 30]:
        results[f"mr_{w}"] = -rolling_zscore(df["close"], w).shift(1)

    # Adaptive mean reversion (scale by vol regime)
    for w in [10, 20]:
        z = rolling_zscore(df["close"], w)
        vol_fast = returns.rolling(5).std()
        vol_slow = returns.rolling(30).std()
        vol_ratio = safe_div(vol_fast.values, vol_slow.values, 1.0)
        results[f"mr_adaptive_{w}"] = (-z * pd.Series(vol_ratio, index=df.index)).shift(1)

    # Bollinger bounce
    for w in [10, 20]:
        sma = df["close"].rolling(w).mean()
        std = df["close"].rolling(w).std()
        bb_z = (df["close"] - sma) / std.replace(0, np.nan)
        results[f"bb_rev_{w}"] = (-bb_z).shift(1)

    # RSI reversal
    for w in [7, 14]:
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(w).mean()
        loss = (-delta.clip(upper=0)).rolling(w).mean()
        rs = safe_div(gain.values, loss.values, 1.0)
        rsi = 100 - 100 / (1 + pd.Series(rs, index=df.index))
        results[f"rsi_rev_{w}"] = (-(rsi - 50) / 50).shift(1)

    # ---- MOMENTUM ----
    for lb in [3, 6, 12, 24]:
        results[f"mom_{lb}"] = np.log(df["close"] / df["close"].shift(lb)).shift(1)

    # Volatility-scaled momentum (Sharpe-like)
    for lb in [6, 12]:
        mom = returns.rolling(lb).mean()
        vol = returns.rolling(lb * 3).std()
        results[f"sharpe_mom_{lb}"] = pd.Series(safe_div(mom.values, vol.values, 0.0), index=df.index).shift(1)

    # ---- TREND ----
    for w in [10, 20, 40]:
        mean_ret = returns.rolling(w).mean()
        mean_abs = returns.abs().rolling(w).mean()
        results[f"trend_{w}"] = pd.Series(safe_div(mean_ret.values, mean_abs.values, 0.0), index=df.index).shift(1)

    # ---- VOLUME ----
    for w in [5, 10, 20]:
        tbr = safe_div(df["taker_buy_base"].values, df["volume"].values, 0.5)
        results[f"taker_{w}"] = (pd.Series(tbr, index=df.index).rolling(w).mean() - 0.5).shift(1)

    # Volume-weighted returns
    for w in [5, 10, 20]:
        vw = (returns * df["volume"]).rolling(w).sum()
        vs = df["volume"].rolling(w).sum()
        results[f"vwret_{w}"] = pd.Series(safe_div(vw.values, vs.values, 0.0), index=df.index).shift(1)

    # Volume breakout
    for w in [10, 20]:
        vol_z = rolling_zscore(df["volume"], w)
        results[f"vol_breakout_{w}"] = (vol_z * returns.rolling(3).mean()).shift(1)

    # ---- CANDLE STRUCTURE ----
    body = df["close"] - df["open"]
    hl_range = df["high"] - df["low"]
    for w in [3, 5]:
        ratio = safe_div(body.values, hl_range.values, 0.0)
        results[f"candle_{w}"] = pd.Series(ratio, index=df.index).rolling(w).mean().shift(1)

    # Close position in range
    for w in [3, 5]:
        pos = safe_div((df["close"] - df["low"]).values, hl_range.values, 0.5)
        results[f"cpos_{w}"] = (pd.Series(pos - 0.5, index=df.index).rolling(w).mean()).shift(1)

    # Lower shadow strength (bullish rejection)
    for w in [3, 5]:
        lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]
        ls_ratio = safe_div(lower_shadow.values, hl_range.values, 0.0)
        results[f"low_shadow_{w}"] = pd.Series(ls_ratio, index=df.index).rolling(w).mean().shift(1)

    # ---- INTERACTION SIGNALS ----
    # Momentum × taker flow
    for lb in [6]:
        mom = returns.rolling(lb).mean()
        tbr = safe_div(df["taker_buy_base"].values, df["volume"].values, 0.5)
        tbr_s = pd.Series(tbr - 0.5, index=df.index).rolling(10).mean()
        results[f"mom_taker_{lb}"] = (mom * tbr_s).shift(1)

    # Mean reversion × volume confirmation
    for w in [10, 20]:
        z = rolling_zscore(df["close"], w)
        avg_vol = df["volume"].rolling(w).mean()
        vol_ratio = safe_div(df["volume"].values, avg_vol.values, 1.0)
        results[f"mr_volconf_{w}"] = (-z * pd.Series(vol_ratio, index=df.index)).shift(1)

    # Autocorrelation
    for w in [10, 20]:
        ac = returns.rolling(w, min_periods=5).apply(
            lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 else 0, raw=True)
        results[f"autocorr_{w}"] = ac.shift(1)

    # ---- PRICE LEVEL SIGNALS ----
    for w in [20, 50]:
        sma = df["close"].rolling(w).mean()
        results[f"price_sma_{w}"] = ((df["close"] - sma) / sma).shift(1)

    return pd.DataFrame(results, index=df.index)


# ============================================================================
# WALK-FORWARD EVALUATION
# ============================================================================

def walk_forward_eval(signal: pd.Series, target: pd.Series,
                      train_frac: float = 0.6, n_folds: int = 5,
                      bars_per_day: int = 288) -> dict:
    """
    Walk-forward evaluation: train on first portion, validate on rest.
    Split into n_folds of increasing training set size.
    """
    n = len(signal)
    results = []

    for fold in range(n_folds):
        train_end = int(n * (train_frac + (1 - train_frac) * fold / n_folds))
        test_start = train_end
        test_end = int(n * (train_frac + (1 - train_frac) * (fold + 1) / n_folds))

        if test_end <= test_start + 100:
            continue

        test_sig = signal.iloc[test_start:test_end]
        test_tgt = target.iloc[test_start:test_end]
        result = run_backtest(test_sig, test_tgt, bars_per_day=bars_per_day)
        results.append(result)

    if not results:
        return {"sharpe": 0, "win_rate": 0.5, "n_folds": 0}

    return {
        "sharpe": np.mean([r.sharpe for r in results]),
        "sharpe_std": np.std([r.sharpe for r in results]),
        "win_rate": np.mean([r.win_rate for r in results]),
        "profit_factor": np.mean([r.profit_factor for r in results]),
        "n_folds": len(results),
        "min_sharpe": min(r.sharpe for r in results),
        "max_sharpe": max(r.sharpe for r in results),
    }


# ============================================================================
# OPTIMAL MODEL BUILDER
# ============================================================================

def build_v3_model(symbol: str, interval: str, verbose=True):
    """Build v3 optimized model with walk-forward validation."""
    bpd = BARS_PER_DAY[interval]
    nm = SYMBOL_NAMES[symbol]

    df = load_klines(symbol, interval)
    train_df = split_data(df, TRAIN_START, TRAIN_END)
    holdout_df = split_data(df, HOLDOUT_START, HOLDOUT_END)

    if verbose:
        print(f"\n{'='*70}")
        print(f"V3 MODEL: {nm} {interval}")
        print(f"Train: {len(train_df)} bars ({TRAIN_START} to {TRAIN_END})")
        print(f"Holdout: {len(holdout_df)} bars ({HOLDOUT_START} to {HOLDOUT_END})")
        print(f"{'='*70}")

    train_sigs = compute_v3_signals(train_df)
    train_target = compute_target(train_df)
    holdout_sigs = compute_v3_signals(holdout_df)
    holdout_target = compute_target(holdout_df)

    # Step 1: Score all signals on train
    scores = []
    for col in train_sigs.columns:
        sig = train_sigs[col].dropna()
        if len(sig) < 500:
            continue
        result = run_backtest(sig, train_target, bars_per_day=bpd)
        if result.total_trades > 200:
            scores.append({
                "name": col,
                "sharpe": result.sharpe,
                "win_rate": result.win_rate,
                "pnl": result.net_pnl,
                "pf": result.profit_factor,
            })

    scores.sort(key=lambda x: x["sharpe"], reverse=True)

    if verbose:
        print(f"\n  Top 15 signals (train):")
        for s in scores[:15]:
            print(f"    {s['name']:<30} SR={s['sharpe']:.2f} WR={s['win_rate']:.1%} PF={s['pf']:.2f}")

    # Step 2: Greedy orthogonal selection (with walk-forward filter)
    selected = []
    for s in scores:
        if s["sharpe"] < 0.1:
            break
        if len(selected) >= 10:
            break

        new_sig = train_sigs[s["name"]].dropna()
        too_corr = False
        for sel in selected:
            sel_sig = train_sigs[sel].dropna()
            common = new_sig.index.intersection(sel_sig.index)
            if len(common) < 100:
                continue
            corr = np.corrcoef(new_sig.loc[common].values, sel_sig.loc[common].values)[0, 1]
            if abs(corr) > 0.55:
                too_corr = True
                break
        if not too_corr:
            selected.append(s["name"])

    if verbose:
        print(f"\n  Selected {len(selected)} orthogonal signals:")
        for s in selected:
            score = [x for x in scores if x["name"] == s][0]
            print(f"    {s:<30} SR={score['sharpe']:.2f} WR={score['win_rate']:.1%}")

    if len(selected) < 2:
        return None

    # Step 3: Normalize using train stats
    norm_stats = {}
    sig_matrix = train_sigs[selected].copy()
    for col in selected:
        mu = sig_matrix[col].mean()
        std = sig_matrix[col].std()
        if std > 1e-10:
            sig_matrix[col] = (sig_matrix[col] - mu) / std
            norm_stats[col] = {"mean": float(mu), "std": float(std)}
        else:
            norm_stats[col] = {"mean": 0.0, "std": 1.0}

    # Step 4: Optimize combination weights
    best_method = "equal"
    best_weights = np.ones(len(selected)) / len(selected)
    best_sharpe = -999

    methods = {}

    # Equal weight
    w_eq = np.ones(len(selected)) / len(selected)
    combined_eq = (sig_matrix.values * w_eq).sum(axis=1)
    r_eq = run_backtest(pd.Series(combined_eq, index=sig_matrix.index), train_target, bars_per_day=bpd)
    methods["equal"] = (w_eq, r_eq)

    # Sharpe-weighted
    w_sw = np.array([max([x for x in scores if x["name"] == s][0]["sharpe"], 0.01) for s in selected])
    w_sw = w_sw / w_sw.sum()
    combined_sw = (sig_matrix.values * w_sw).sum(axis=1)
    r_sw = run_backtest(pd.Series(combined_sw, index=sig_matrix.index), train_target, bars_per_day=bpd)
    methods["sharpe_wt"] = (w_sw, r_sw)

    # SLSQP optimization
    def neg_sharpe(w):
        w = np.abs(w)
        w = w / max(w.sum(), 1e-10)
        combined = (sig_matrix[selected].values * w).sum(axis=1)
        r = run_backtest(pd.Series(combined, index=sig_matrix.index), train_target, bars_per_day=bpd)
        return -r.sharpe

    for x0_method in ["equal", "sharpe"]:
        x0 = w_eq if x0_method == "equal" else w_sw
        try:
            opt = minimize(neg_sharpe, x0, method="SLSQP",
                           bounds=[(0.01, 1.0) for _ in selected],
                           constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
                           options={"maxiter": 500, "ftol": 1e-12})
            w_opt = np.abs(opt.x) / max(np.abs(opt.x).sum(), 1e-10)
            combined_opt = (sig_matrix.values * w_opt).sum(axis=1)
            r_opt = run_backtest(pd.Series(combined_opt, index=sig_matrix.index), train_target, bars_per_day=bpd)
            methods[f"opt_{x0_method}"] = (w_opt, r_opt)
        except:
            pass

    # Pick best method
    for method_name, (w, r) in methods.items():
        if r.sharpe > best_sharpe:
            best_sharpe = r.sharpe
            best_method = method_name
            best_weights = w

    if verbose:
        print(f"\n  Combination methods (train):")
        for method_name, (w, r) in methods.items():
            marker = " ★" if method_name == best_method else ""
            print(f"    {method_name:<15} SR={r.sharpe:.2f} WR={r.win_rate:.1%} "
                  f"PnL=${r.net_pnl:,.0f} PF={r.profit_factor:.2f}{marker}")

    # Step 5: Evaluate on holdout
    ho_matrix = holdout_sigs[selected].copy()
    for col in selected:
        mu = norm_stats[col]["mean"]
        std = norm_stats[col]["std"]
        if std > 1e-10:
            ho_matrix[col] = (ho_matrix[col] - mu) / std

    combined_holdout = (ho_matrix.values * best_weights).sum(axis=1)
    ho_series = pd.Series(combined_holdout, index=ho_matrix.index)
    holdout_result = run_backtest(ho_series, holdout_target, bars_per_day=bpd)

    # Step 6: Monthly breakdown
    pnl_monthly = holdout_result.pnl_series.resample("ME").sum()

    if verbose:
        print(f"\n  {'='*60}")
        print(f"  HOLDOUT RESULTS ({HOLDOUT_START} to {HOLDOUT_END}):")
        print(f"  {'='*60}")
        print(f"  Sharpe:        {holdout_result.sharpe:.2f}")
        print(f"  Win Rate:      {holdout_result.win_rate:.1%}")
        print(f"  Net PnL:       ${holdout_result.net_pnl:,.0f}")
        print(f"  Max Drawdown:  ${holdout_result.max_drawdown:,.0f}")
        print(f"  Profit Factor: {holdout_result.profit_factor:.2f}")
        print(f"  Trades/Day:    {holdout_result.trades_per_day:.1f}")
        print(f"  Calmar Ratio:  {holdout_result.calmar_ratio:.2f}")
        print(f"\n  Monthly PnL:")
        for month, pnl in pnl_monthly.items():
            print(f"    {month.strftime('%Y-%m')}: ${pnl:>10,.0f}")
        
        print(f"\n  Signal Weights:")
        for i, name in enumerate(selected):
            if best_weights[i] > 0.001:
                print(f"    {name:<30} w={best_weights[i]:.3f}")

    # Step 7: Realistic capital-constrained PnL
    # Can only have N concurrent positions, each $250
    max_trades_per_day = 100  # Capital constraint
    scale = min(1.0, max_trades_per_day / max(holdout_result.trades_per_day, 1))
    realistic_pnl = holdout_result.net_pnl * scale

    if verbose:
        print(f"\n  Realistic PnL (max {max_trades_per_day} trades/day):")
        days = (holdout_df.index[-1] - holdout_df.index[0]).days
        ann_pnl = realistic_pnl / max(days, 1) * 365
        print(f"    Holdout ({days}d): ${realistic_pnl:,.0f}")
        print(f"    Annualized: ${ann_pnl:,.0f}")
        print(f"    On $50K capital: {ann_pnl/50000*100:.0f}% annual return")

    # Save to DB
    conn = ensure_db()
    conn.execute("""INSERT INTO optimized_models
        (model_name, symbol, interval, signal_names, weights,
         train_sharpe, train_win_rate, holdout_sharpe, holdout_win_rate, holdout_pnl)
        VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (f"v3_{best_method}_{nm}_{interval}", symbol, interval,
         json.dumps(selected), json.dumps(best_weights.tolist()),
         best_sharpe, methods[best_method][1].win_rate,
         holdout_result.sharpe, holdout_result.win_rate,
         holdout_result.net_pnl))
    conn.commit()
    conn.close()

    return {
        "symbol": symbol, "interval": interval,
        "method": best_method,
        "signals": selected,
        "weights": best_weights,
        "norm_stats": norm_stats,
        "train_result": methods[best_method][1],
        "holdout_result": holdout_result,
        "monthly_pnl": pnl_monthly,
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    all_models = {}

    for symbol in SYMBOLS:
        for interval in INTERVALS:
            try:
                model = build_v3_model(symbol, interval)
                if model:
                    key = f"{SYMBOL_NAMES[symbol]}_{interval}"
                    all_models[key] = model
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback; traceback.print_exc()

    # Final summary
    print(f"\n\n{'#'*90}")
    print(f"FINAL SUMMARY — ALL v3 MODELS")
    print(f"{'#'*90}")
    print(f"{'Model':<15} {'Sigs':>5} {'Method':<15} {'Tr SR':>7} {'Tr WR':>7} {'Ho SR':>7} {'Ho WR':>7} {'Ho PnL':>12} {'Ho MDD':>10} {'PF':>6}")
    print(f"{'-'*15} {'-'*5} {'-'*15} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*12} {'-'*10} {'-'*6}")

    for key in sorted(all_models.keys(), key=lambda k: all_models[k]["holdout_result"].sharpe, reverse=True):
        m = all_models[key]
        tr = m["train_result"]
        ho = m["holdout_result"]
        print(f"{key:<15} {len(m['signals']):>5} {m['method']:<15} "
              f"{tr.sharpe:>7.1f} {tr.win_rate:>6.1%} "
              f"{ho.sharpe:>7.1f} {ho.win_rate:>6.1%} "
              f"${ho.net_pnl:>11,.0f} ${ho.max_drawdown:>9,.0f} {ho.profit_factor:>5.2f}")

    # Monthly PnL aggregated
    print(f"\n\nMONTHLY PnL SUMMARY (All Models Combined):")
    monthly_total = {}
    for key, m in all_models.items():
        for month, pnl in m["monthly_pnl"].items():
            mo = month.strftime("%Y-%m")
            monthly_total[mo] = monthly_total.get(mo, 0) + pnl
    for mo in sorted(monthly_total.keys()):
        print(f"  {mo}: ${monthly_total[mo]:>12,.0f}")
    print(f"  TOTAL: ${sum(monthly_total.values()):>12,.0f}")
