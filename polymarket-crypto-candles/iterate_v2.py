"""
iterate_v2.py — Iteration 2: Expanded signal library + proper combination + holdout evaluation.

Focuses on:
1. Adding more mean reversion variants (the dominant alpha)
2. Adding support/resistance signals
3. Finding complementary signals to combine with mean reversion
4. Building a combined model with proper normalization
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm as norm_dist

from config import *
from signals import compute_signals, compute_target, safe_div, rolling_zscore
from backtest_engine import run_backtest, BacktestResult
from run_backtest import load_klines, split_data, ensure_db

BARS_PER_DAY = {"5m": 288, "15m": 96, "1h": 24}


# ============================================================================
# EXPANDED SIGNAL LIBRARY (v2)
# ============================================================================

def compute_v2_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Expanded signal library with emphasis on mean reversion and complementary signals."""
    results = {}
    returns = df["close"].pct_change()

    # ---- MEAN REVERSION FAMILY (the alpha king) ----
    for w in [5, 8, 10, 15, 20, 30, 45, 60]:
        results[f"mr_zscore_{w}"] = -rolling_zscore(df["close"], w).shift(1)

    # Mean reversion with different anchors
    for w in [10, 20, 30]:
        sma = df["close"].rolling(w, min_periods=max(w//2, 2)).mean()
        results[f"mr_sma_dev_{w}"] = (-(df["close"] - sma) / sma).shift(1)

    # Mean reversion with volatility scaling
    for w in [10, 20, 30]:
        z = rolling_zscore(df["close"], w)
        vol = returns.rolling(w, min_periods=max(w//2, 2)).std()
        avg_vol = returns.rolling(w*3, min_periods=w).std()
        vol_ratio = safe_div(vol.values, avg_vol.values, 1.0)
        # In high vol, mean reversion is stronger
        results[f"mr_vol_scaled_{w}"] = (-z * pd.Series(vol_ratio, index=df.index)).shift(1)

    # Bollinger band bounce
    for w in [10, 20, 30]:
        sma = df["close"].rolling(w, min_periods=max(w//2, 2)).mean()
        std = df["close"].rolling(w, min_periods=max(w//2, 2)).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        bb_pos = safe_div((df["close"] - lower).values, (upper - lower).values, 0.5)
        results[f"mr_bb_{w}"] = pd.Series(-(bb_pos - 0.5), index=df.index).shift(1)

    # RSI-style mean reversion
    for w in [7, 14, 21]:
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(w, min_periods=max(w//2, 2)).mean()
        loss = (-delta.clip(upper=0)).rolling(w, min_periods=max(w//2, 2)).mean()
        rs = safe_div(gain.values, loss.values, 1.0)
        rsi = 100 - 100 / (1 + pd.Series(rs, index=df.index))
        results[f"mr_rsi_{w}"] = (-(rsi - 50) / 50).shift(1)

    # ---- MOMENTUM (to complement mean reversion) ----
    for lb in [3, 6, 12, 24, 48]:
        results[f"mom_{lb}"] = np.log(df["close"] / df["close"].shift(lb)).shift(1)

    # Smoothed momentum
    for lb in [6, 12, 24]:
        mom = np.log(df["close"] / df["close"].shift(lb))
        results[f"mom_smooth_{lb}"] = mom.rolling(3, min_periods=1).mean().shift(1)

    # ---- VOLATILITY-SCALED MOMENTUM ----
    for lb in [6, 12]:
        mom = returns.rolling(lb, min_periods=max(lb//2, 2)).mean()
        vol = returns.rolling(lb*3, min_periods=lb).std()
        results[f"mom_vs_{lb}"] = safe_div(mom.values, vol.values, 0.0)

    # ---- VOLUME SIGNALS ----
    for w in [5, 10, 20]:
        tbr = safe_div(df["taker_buy_base"].values, df["volume"].values, 0.5)
        results[f"taker_{w}"] = (pd.Series(tbr, index=df.index).rolling(w, min_periods=1).mean() - 0.5).shift(1)

    # Volume-weighted return (OBV direction)
    for w in [10, 20]:
        vw_ret = (returns * df["volume"]).rolling(w, min_periods=max(w//2, 2)).sum()
        vol_sum = df["volume"].rolling(w, min_periods=max(w//2, 2)).sum()
        results[f"vwret_{w}"] = safe_div(vw_ret.values, vol_sum.values, 0.0)

    # ---- CANDLE STRUCTURE ----
    body = df["close"] - df["open"]
    hl_range = df["high"] - df["low"]
    for w in [3, 5, 10]:
        body_ratio = safe_div(body.values, hl_range.values, 0.0)
        results[f"candle_body_{w}"] = pd.Series(body_ratio, index=df.index).rolling(w, min_periods=1).mean().shift(1)

    # Close position in range
    for w in [3, 5]:
        pos = safe_div((df["close"] - df["low"]).values, hl_range.values, 0.5)
        results[f"close_pos_{w}"] = (pd.Series(pos - 0.5, index=df.index).rolling(w, min_periods=1).mean()).shift(1)

    # ---- TREND STRENGTH ----
    for w in [10, 20, 40]:
        mean_ret = returns.rolling(w, min_periods=max(w//2, 2)).mean()
        mean_abs = returns.abs().rolling(w, min_periods=max(w//2, 2)).mean()
        results[f"trend_{w}"] = safe_div(mean_ret.values, mean_abs.values, 0.0)

    # ---- PRICE RELATIVE TO LEVELS ----
    for w in [20, 50]:
        sma = df["close"].rolling(w, min_periods=max(w//2, 2)).mean()
        results[f"price_vs_sma_{w}"] = ((df["close"] - sma) / sma).shift(1)

    # ---- AUTOCORRELATION ----
    for w in [10, 20]:
        ac = returns.rolling(w, min_periods=max(w//2, 5)).apply(
            lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 else 0, raw=True)
        results[f"autocorr_{w}"] = ac.shift(1)

    # ---- MEAN REVERSION × VOLUME INTERACTION ----
    for w in [10, 20]:
        z = rolling_zscore(df["close"], w)
        avg_vol = df["volume"].rolling(w, min_periods=max(w//2, 2)).mean()
        vol_ratio = safe_div(df["volume"].values, avg_vol.values, 1.0)
        results[f"mr_vol_interact_{w}"] = (-z * pd.Series(vol_ratio, index=df.index)).shift(1)

    # ---- DOUBLE MEAN REVERSION (reversion after reversal) ----
    for w in [10, 20]:
        z = rolling_zscore(df["close"], w)
        z_change = z.diff(3)
        # If z-score is extreme AND has moved fast, expect reversion
        results[f"mr_speed_{w}"] = (-z * z_change.abs()).shift(1)

    return pd.DataFrame(results, index=df.index)


# ============================================================================
# COMBINED MODEL WITH PORTFOLIO-STYLE OPTIMIZATION
# ============================================================================

def build_optimal_model(symbol: str, interval: str, max_signals: int = 12):
    """Build an optimized combined model for a symbol/interval."""
    bpd = BARS_PER_DAY[interval]
    nm = SYMBOL_NAMES[symbol]

    # Load data
    df = load_klines(symbol, interval)
    train_df = split_data(df, TRAIN_START, TRAIN_END)
    holdout_df = split_data(df, HOLDOUT_START, HOLDOUT_END)

    print(f"\n{'='*70}")
    print(f"BUILDING MODEL: {nm} {interval}")
    print(f"Train: {len(train_df)} bars, Holdout: {len(holdout_df)} bars")
    print(f"{'='*70}")

    # Compute signals
    train_signals = compute_v2_signals(train_df)
    train_target = compute_target(train_df)

    # Step 1: Score all signals
    signal_scores = []
    for col in train_signals.columns:
        sig = train_signals[col].dropna()
        if len(sig) < 200:
            continue
        result = run_backtest(sig, train_target, bars_per_day=bpd)
        if result.total_trades > 100:
            signal_scores.append({
                "name": col,
                "sharpe": result.sharpe,
                "win_rate": result.win_rate,
                "pnl": result.net_pnl,
                "pf": result.profit_factor,
            })

    signal_scores.sort(key=lambda x: x["sharpe"], reverse=True)
    print(f"\n  {len(signal_scores)} signals evaluated on train set")
    print(f"  Top 10:")
    for s in signal_scores[:10]:
        print(f"    {s['name']:<35} SR={s['sharpe']:.2f} WR={s['win_rate']:.1%} PnL=${s['pnl']:,.0f}")

    # Step 2: Greedy orthogonal selection
    selected = []
    for s in signal_scores:
        if s["sharpe"] < 0.1:
            break
        if len(selected) >= max_signals:
            break

        # Check correlation with already selected
        new_sig = train_signals[s["name"]].dropna()
        too_corr = False
        for sel in selected:
            sel_sig = train_signals[sel].dropna()
            common = new_sig.index.intersection(sel_sig.index)
            if len(common) < 100:
                continue
            corr = np.corrcoef(new_sig.loc[common].values, sel_sig.loc[common].values)[0, 1]
            if abs(corr) > 0.60:
                too_corr = True
                break
        if not too_corr:
            selected.append(s["name"])

    print(f"\n  Selected {len(selected)} orthogonal signals:")
    for s in selected:
        score = [x for x in signal_scores if x["name"] == s][0]
        print(f"    {s:<35} SR={score['sharpe']:.2f}")

    if len(selected) < 2:
        print("  Too few signals, aborting")
        return None

    # Step 3: Normalize signals
    norm_stats = {}
    sig_matrix = train_signals[selected].copy()
    for col in selected:
        mu = sig_matrix[col].mean()
        std = sig_matrix[col].std()
        if std > 0:
            sig_matrix[col] = (sig_matrix[col] - mu) / std
            norm_stats[col] = {"mean": float(mu), "std": float(std)}
        else:
            norm_stats[col] = {"mean": 0.0, "std": 1.0}

    # Step 4: Try multiple combination methods
    results = {}

    # Equal weight
    w_equal = np.ones(len(selected)) / len(selected)
    combined_eq = (sig_matrix.values * w_equal).sum(axis=1)
    r_eq = run_backtest(pd.Series(combined_eq, index=sig_matrix.index), train_target, bars_per_day=bpd)
    results["equal"] = (w_equal, r_eq)

    # Sharpe-weighted
    w_sharpe = []
    for s in selected:
        score = [x for x in signal_scores if x["name"] == s][0]
        w_sharpe.append(max(score["sharpe"], 0.01))
    w_sharpe = np.array(w_sharpe)
    w_sharpe = w_sharpe / w_sharpe.sum()
    combined_sw = (sig_matrix.values * w_sharpe).sum(axis=1)
    r_sw = run_backtest(pd.Series(combined_sw, index=sig_matrix.index), train_target, bars_per_day=bpd)
    results["sharpe_weighted"] = (w_sharpe, r_sw)

    # Optimized (maximize Sharpe)
    def neg_sharpe(w):
        w = w / max(w.sum(), 1e-10)
        combined = (sig_matrix[selected].values * w).sum(axis=1)
        result = run_backtest(pd.Series(combined, index=sig_matrix.index), train_target, bars_per_day=bpd)
        return -result.sharpe

    x0 = w_sharpe.copy()
    bounds = [(0, 1) for _ in selected]
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}

    try:
        opt = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"maxiter": 300, "ftol": 1e-10})
        w_opt = opt.x / max(opt.x.sum(), 1e-10)
        combined_opt = (sig_matrix.values * w_opt).sum(axis=1)
        r_opt = run_backtest(pd.Series(combined_opt, index=sig_matrix.index), train_target, bars_per_day=bpd)
        results["optimized"] = (w_opt, r_opt)
    except Exception as e:
        print(f"  Optimization failed: {e}")

    # Print comparison
    print(f"\n  TRAIN RESULTS:")
    print(f"  {'Method':<20} {'Sharpe':>8} {'WR':>8} {'PnL':>12} {'MDD':>10} {'PF':>6}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*12} {'-'*10} {'-'*6}")
    best_method = None
    best_sharpe = -999
    for method, (w, r) in results.items():
        print(f"  {method:<20} {r.sharpe:>8.2f} {r.win_rate:>7.1%} ${r.net_pnl:>11,.0f} "
              f"${r.max_drawdown:>9,.0f} {r.profit_factor:>5.2f}")
        if r.sharpe > best_sharpe:
            best_sharpe = r.sharpe
            best_method = method

    best_weights, best_train = results[best_method]
    print(f"\n  Best method: {best_method}")

    # Step 5: Evaluate on holdout
    holdout_signals = compute_v2_signals(holdout_df)
    holdout_target = compute_target(holdout_df)

    ho_matrix = holdout_signals[selected].copy()
    for col in selected:
        mu = norm_stats[col]["mean"]
        std = norm_stats[col]["std"]
        if std > 0:
            ho_matrix[col] = (ho_matrix[col] - mu) / std

    combined_holdout = (ho_matrix.values * best_weights).sum(axis=1)
    holdout_result = run_backtest(pd.Series(combined_holdout, index=ho_matrix.index),
                                  holdout_target, bars_per_day=bpd)

    print(f"\n  HOLDOUT RESULTS ({HOLDOUT_START} to {HOLDOUT_END}):")
    print(f"  Sharpe:        {holdout_result.sharpe:.2f}")
    print(f"  Win Rate:      {holdout_result.win_rate:.1%}")
    print(f"  Net PnL:       ${holdout_result.net_pnl:,.0f}")
    print(f"  Max Drawdown:  ${holdout_result.max_drawdown:,.0f}")
    print(f"  Profit Factor: {holdout_result.profit_factor:.2f}")
    print(f"  Trades/Day:    {holdout_result.trades_per_day:.1f}")
    print(f"  Calmar Ratio:  {holdout_result.calmar_ratio:.2f}")

    # Per-signal weight breakdown
    print(f"\n  Signal Weights:")
    for i, name in enumerate(selected):
        if best_weights[i] > 0.01:
            print(f"    {name:<35} w={best_weights[i]:.3f}")

    # Save model
    conn = ensure_db()
    conn.execute("""INSERT INTO optimized_models
        (model_name, symbol, interval, signal_names, weights,
         train_sharpe, train_win_rate, holdout_sharpe, holdout_win_rate, holdout_pnl)
        VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (f"v2_{best_method}_{nm}_{interval}", symbol, interval,
         json.dumps(selected), json.dumps(best_weights.tolist()),
         best_train.sharpe, best_train.win_rate,
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
        "train": best_train,
        "holdout": holdout_result,
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    all_models = {}
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            try:
                model = build_optimal_model(symbol, interval)
                if model:
                    key = f"{SYMBOL_NAMES[symbol]}_{interval}"
                    all_models[key] = model
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback; traceback.print_exc()

    # Final summary
    print(f"\n\n{'='*90}")
    print(f"FINAL SUMMARY — ALL OPTIMIZED MODELS (v2)")
    print(f"{'='*90}")
    print(f"{'Model':<15} {'Signals':>8} {'Train SR':>10} {'Train WR':>10} {'Hold SR':>10} {'Hold WR':>10} {'Hold PnL':>12} {'MDD':>10} {'Calmar':>8}")
    print(f"{'-'*15} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*8}")

    for key, m in sorted(all_models.items(), key=lambda x: x[1]["holdout"].sharpe, reverse=True):
        print(f"{key:<15} {len(m['signals']):>8} {m['train'].sharpe:>10.2f} "
              f"{m['train'].win_rate:>9.1%} {m['holdout'].sharpe:>10.2f} "
              f"{m['holdout'].win_rate:>9.1%} ${m['holdout'].net_pnl:>11,.0f} "
              f"${m['holdout'].max_drawdown:>9,.0f} {m['holdout'].calmar_ratio:>7.2f}")
