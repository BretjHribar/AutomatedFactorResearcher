"""
Compare two pipeline approaches using GP-discovered alphas + original 56 alphas:
  1. Simple averaging of all alpha signals (equal weight)
  2. Walk-forward with QP top selector

Uses cached 4h matrices with TOP50 universe.
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
import sqlite3
import math
import time
import matplotlib.pyplot as plt
from pathlib import Path

# ── Load GP alphas from DB ──
def load_gp_alphas():
    db = sqlite3.connect('data/alpha_gp_crypto_v2_4h.db')
    cur = db.cursor()
    rows = cur.execute('SELECT expression, alpha_id FROM alphas').fetchall()
    db.close()
    return [(f"gp_{r[1]}", r[0]) for r in rows]

# ── Load original 56 alphas ──
from run_crypto_walkforward import ALPHA_DEFINITIONS, evaluate_expression, load_data

# ── Load data (all matrices for GP compatibility) ──
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

# Load via load_data for the basic set
features = load_data(n_symbols=50, n_days=9999)
returns = features["returns"]
close = features["close"]
adv20 = features["adv20"]

# Load ALL 42 matrices so GP expressions using funding_rate etc. work
from pathlib import Path
matrices_dir = Path("data/binance_cache/matrices/4h")
all_features = dict(features)  # Start with basic features
for fpath in sorted(matrices_dir.glob("*.parquet")):
    name = fpath.stem
    if name not in all_features:
        df = pd.read_parquet(fpath)
        # Filter to same columns as close for consistency
        common = [c for c in close.columns if c in df.columns]
        if common:
            all_features[name] = df[common].reindex(close.index)

n_bars = len(close)
print(f"  {n_bars} bars, {len(close.columns)} tickers")
print(f"  {len(all_features)} feature matrices loaded")

# ── Compute all alpha signals ──
print(f"\nComputing alpha signals...")

# Original 56
alpha_signals = {}
for name, expr in ALPHA_DEFINITIONS:
    sig = evaluate_expression(expr, features)
    if sig is not None:
        alpha_signals[name] = sig
print(f"  {len(alpha_signals)} original alphas computed")

# GP alphas - need the full operator set from vectorized ops
from src.operators import vectorized as ops

def eval_gp_expression(expr_str, features):
    """Evaluate a GP expression using the same ops as the GP engine."""
    ctx = {
        # Arithmetic (DF, DF)
        "add": ops.add, "subtract": ops.subtract, "multiply": ops.multiply,
        "divide": ops.divide, "true_divide": ops.true_divide,
        "df_max": ops.df_max, "df_min": ops.df_min,
        # Arithmetic (DF, float)
        "npfadd": ops.npfadd, "npfsub": ops.npfsub, "npfmul": ops.npfmul,
        "npfdiv": ops.npfdiv, "SignedPower": ops.SignedPower,
        # Unary
        "negative": ops.negative, "Abs": ops.Abs, "Sign": ops.Sign,
        "Inverse": ops.Inverse, "rank": ops.rank, "log": ops.log,
        "log10": ops.log10, "sqrt": ops.sqrt, "square": ops.square,
        "log_diff": ops.log_diff, "s_log_1p": ops.s_log_1p, "normalize": ops.normalize,
        # Time-series (DF, int)
        "ts_sum": ops.ts_sum, "sma": ops.sma, "ts_rank": ops.ts_rank,
        "ts_min": ops.ts_min, "ts_max": ops.ts_max, "delta": ops.delta,
        "stddev": ops.stddev, "delay": ops.delay, "ArgMax": ops.ArgMax,
        "ArgMin": ops.ArgMin, "Product": ops.Product, "Decay_lin": ops.Decay_lin,
        "ts_zscore": ops.ts_zscore, "ts_skewness": ops.ts_skewness,
        "ts_kurtosis": ops.ts_kurtosis, "ts_entropy": ops.ts_entropy,
        # Decay (DF, float)
        "Decay_exp": ops.Decay_exp,
        # Two-input (DF, DF, int)
        "correlation": ops.correlation, "covariance": ops.covariance,
        # Identity
        "extend": lambda x: x,
        # Features
        **all_features,
    }
    try:
        result = eval(expr_str, {"__builtins__": {}}, ctx)
        if isinstance(result, pd.DataFrame):
            return result
    except Exception:
        pass
    return None

gp_alphas = load_gp_alphas()
gp_count = 0
for name, expr in gp_alphas:
    sig = eval_gp_expression(expr, all_features)
    if sig is not None:
        alpha_signals[name] = sig
        gp_count += 1
    else:
        print(f"  WARN: failed to eval {name}: {expr[:60]}")
print(f"  {gp_count} GP alphas computed")
print(f"  TOTAL: {len(alpha_signals)} alphas")

# ── Constants ──
ANN_FACTOR = math.sqrt(252 * 6)  # 4h bars annualized
REBAL_BARS = 120  # 20-day universe rebalancing

# ═══════════════════════════════════════════════════════════════
# APPROACH 1: Simple Equal-Weight Average of ALL alpha signals
# ═══════════════════════════════════════════════════════════════
def run_equal_weight(features, alpha_signals, fee_bps=5.0):
    """Average all alpha signals with equal weight, then optimize."""
    from run_crypto_walkforward import optimize_portfolio
    
    returns = features["returns"]
    close = features["close"]
    adv20 = features["adv20"]
    n_bars = len(close)
    
    start_bar = 1280  # Same as walkforward
    
    results = []
    prev_holdings = None
    equity = 1.0
    peak_equity = 1.0
    universe = None
    
    print(f"\n{'='*70}")
    print(f"APPROACH 1: EQUAL-WEIGHT AVERAGE ({len(alpha_signals)} alphas)")
    print(f"{'='*70}")
    print(f"  Simulating {n_bars - start_bar} bars...", flush=True)
    
    for t in range(start_bar, n_bars):
        # Universe rebalancing every 120 bars
        if universe is None or (t - start_bar) % REBAL_BARS == 0:
            trailing = adv20.iloc[max(0, t - 60):t].mean()
            valid = trailing.dropna()
            if len(valid) < 20:
                continue
            universe = valid.nlargest(50).index.tolist()
        
        # Average ALL alpha signals at bar t
        forecasts = []
        for name, sig in alpha_signals.items():
            if t < len(sig):
                row = sig.iloc[t].reindex(universe, fill_value=0)
                row = row.sub(row.mean())  # demean
                norm = row.abs().sum()
                if norm > 0:
                    row = row / norm
                forecasts.append(row)
        
        if len(forecasts) < 3:
            continue
        
        forecast = pd.concat(forecasts, axis=1).mean(axis=1)
        forecast = forecast.sub(forecast.mean())
        
        # Optimize
        ret_slice = returns[universe].iloc[max(0, t-120):t]
        holdings, converged = optimize_portfolio(
            forecast, ret_slice, prev_holdings,
        )
        
        # PnL
        actual_ret = returns.iloc[t].reindex(universe, fill_value=0)
        pnl_gross = float((holdings * actual_ret).sum())
        turnover = float((holdings - (prev_holdings.reindex(universe, fill_value=0) 
                         if prev_holdings is not None else 0)).abs().sum())
        pnl_net = pnl_gross - turnover * fee_bps * 1e-4
        
        equity *= (1 + pnl_net)
        peak_equity = max(peak_equity, equity)
        
        results.append({
            "date": close.index[t],
            "pnl_gross": pnl_gross,
            "pnl_net": pnl_net,
            "turnover": turnover,
            "equity": equity,
        })
        prev_holdings = holdings
        
        if (t - start_bar) % 200 == 0 and len(results) > 10:
            df_tmp = pd.DataFrame(results)
            sr = float((df_tmp["pnl_net"].mean() / (df_tmp["pnl_net"].std() + 1e-12)) * ANN_FACTOR)
            print(f"  Bar {t}/{n_bars}: equity={equity:.3f}, SR={sr:+.2f}", flush=True)
    
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# APPROACH 2: Walk-Forward QP Top Selector (existing pipeline)
# ═══════════════════════════════════════════════════════════════
def run_walkforward_qp(features, alpha_signals, fee_bps=5.0,
                       train_bars=720, val_bars=360, reeval_interval=6):
    """Walk-forward with alpha selection + QP combiner."""
    from run_crypto_walkforward import (
        select_alphas, QPCombiner, optimize_portfolio
    )
    
    returns = features["returns"]
    close = features["close"]
    adv20 = features["adv20"]
    n_bars = len(close)
    
    start_bar = locals().get("start_bar_override", train_bars + val_bars + 200)
    
    results = []
    prev_holdings = None
    current_selected = []
    combiner = None
    equity = 1.0
    peak_equity = 1.0
    universe = None
    
    print(f"\n{'='*70}")
    print(f"APPROACH 2: WALK-FORWARD QP SELECTOR ({len(alpha_signals)} alphas)")
    print(f"{'='*70}")
    print(f"  Train: {train_bars} bars | Val: {val_bars} bars | Re-eval: every {reeval_interval} bars")
    print(f"  Simulating {n_bars - start_bar} bars...", flush=True)
    
    for t in range(start_bar, n_bars):
        # Alpha selection
        if (t - start_bar) % reeval_interval == 0 or not current_selected:
            new_sel = select_alphas(alpha_signals, returns, t,
                                    train_bars, val_bars, max_corr=0.65, max_alphas=15)
            if new_sel:
                current_selected = new_sel
                combiner = QPCombiner(
                    {n: alpha_signals[n] for n in current_selected},
                    current_selected, returns, lookback=45
                )
        
        if not current_selected or combiner is None:
            continue
        
        # Universe rebalancing every 120 bars
        if universe is None or (t - start_bar) % REBAL_BARS == 0:
            trailing = adv20.iloc[max(0, t - 60):t].mean()
            valid = trailing.dropna()
            if len(valid) < 20:
                continue
            universe = valid.nlargest(50).index.tolist()
        
        # Combine alphas
        forecast, n_active = combiner.combine(t)
        if forecast is None:
            continue
        forecast = forecast.reindex(universe).fillna(0)
        forecast = forecast.sub(forecast.mean())
        
        # Optimize
        ret_slice = returns[universe].iloc[max(0, t-120):t]
        holdings, converged = optimize_portfolio(
            forecast, ret_slice, prev_holdings,
        )
        
        # PnL
        actual_ret = returns.iloc[t].reindex(universe, fill_value=0)
        pnl_gross = float((holdings * actual_ret).sum())
        turnover = float((holdings - (prev_holdings.reindex(universe, fill_value=0) 
                         if prev_holdings is not None else 0)).abs().sum())
        pnl_net = pnl_gross - turnover * fee_bps * 1e-4
        
        equity *= (1 + pnl_net)
        peak_equity = max(peak_equity, equity)
        
        results.append({
            "date": close.index[t],
            "pnl_gross": pnl_gross,
            "pnl_net": pnl_net,
            "turnover": turnover,
            "equity": equity,
            "n_alphas": n_active,
        })
        prev_holdings = holdings
        
        if (t - start_bar) % 200 == 0 and len(results) > 10:
            df_tmp = pd.DataFrame(results)
            sr = float((df_tmp["pnl_net"].mean() / (df_tmp["pnl_net"].std() + 1e-12)) * ANN_FACTOR)
            n_alp = n_active
            print(f"  Bar {t}/{n_bars}: equity={equity:.3f}, SR={sr:+.2f}, alphas={n_alp}", flush=True)
    
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# RUN BOTH
# ═══════════════════════════════════════════════════════════════
fee_bps = 0.0

t0 = time.time()
results_avg = run_equal_weight(features, alpha_signals, fee_bps=fee_bps)
t1 = time.time()
print(f"\n  Equal-weight completed in {t1-t0:.0f}s")

results_qp = run_walkforward_qp(features, alpha_signals, fee_bps=fee_bps)
t2 = time.time()
print(f"\n  QP walkforward completed in {t2-t1:.0f}s")

# ═══════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════
def compute_metrics(df, name, fee_bps):
    if df.empty:
        print(f"\n{name}: NO RESULTS")
        return {}
    
    sr_net = float((df["pnl_net"].mean() / (df["pnl_net"].std() + 1e-12)) * ANN_FACTOR)
    sr_gross = float((df["pnl_gross"].mean() / (df["pnl_gross"].std() + 1e-12)) * ANN_FACTOR)
    total_ret = float((1 + df["pnl_net"]).prod() - 1)
    
    cum = (1 + df["pnl_net"]).cumprod()
    max_dd = float((cum / cum.cummax() - 1).min())
    avg_to = float(df["turnover"].mean())
    
    print(f"\n{'='*70}")
    print(f"{name} (fee={fee_bps} bps)")
    print(f"{'='*70}")
    print(f"  Sharpe (net):    {sr_net:+.2f}")
    print(f"  Sharpe (gross):  {sr_gross:+.2f}")
    print(f"  Total Return:    {total_ret:+.1%}")
    print(f"  Max Drawdown:    {max_dd:+.1%}")
    print(f"  Avg Turnover:    {avg_to:.3f}")
    print(f"  Bars:            {len(df)}")
    print(f"  Date Range:      {df['date'].iloc[0].strftime('%Y-%m-%d')} → {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    
    # Fee sensitivity
    print(f"\n  Fee Sensitivity:")
    for fee in [0, 3, 5, 10]:
        pnl_adj = df["pnl_gross"] - df["turnover"] * fee * 1e-4
        sr = float((pnl_adj.mean() / (pnl_adj.std() + 1e-12)) * ANN_FACTOR)
        ret = float((1 + pnl_adj).prod() - 1)
        print(f"    {fee:2d} bps: Sharpe={sr:+.2f}, Return={ret:+.1%}")
    
    return {"sr_net": sr_net, "sr_gross": sr_gross, "total_ret": total_ret, 
            "max_dd": max_dd, "avg_to": avg_to}

m1 = compute_metrics(results_avg, "EQUAL-WEIGHT AVERAGE", fee_bps)
m2 = compute_metrics(results_qp, "WALK-FORWARD QP SELECTOR", fee_bps)

# ── Chart ──
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

if not results_avg.empty and m1:
    eq_avg = (1 + results_avg["pnl_net"]).cumprod()
    axes[0].plot(results_avg["date"], eq_avg, label=f"EW Average (SR={m1['sr_net']:+.2f})", color="#4FC3F7", linewidth=1.5)
if not results_qp.empty and m2:
    eq_qp = (1 + results_qp["pnl_net"]).cumprod()
    axes[0].plot(results_qp["date"], eq_qp, label=f"QP Selector (SR={m2['sr_net']:+.2f})", color="#FF7043", linewidth=1.5)

axes[0].set_ylabel("Equity")
axes[0].set_title(f"Pipeline Comparison: {len(alpha_signals)} Alphas (56 original + {gp_count} GP), TOP50, 5bps fees")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Drawdown
if not results_avg.empty and m1:
    dd_avg = eq_avg / eq_avg.cummax() - 1
    axes[1].fill_between(results_avg["date"], dd_avg, 0, alpha=0.3, color="#4FC3F7", label="EW Average")
if not results_qp.empty and m2:
    dd_qp = eq_qp / eq_qp.cummax() - 1
    axes[1].fill_between(results_qp["date"], dd_qp, 0, alpha=0.3, color="#FF7043", label="QP Selector")

axes[1].set_ylabel("Drawdown")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pipeline_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Chart saved: pipeline_comparison.png")
