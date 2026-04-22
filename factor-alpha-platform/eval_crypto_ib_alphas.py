"""
Crypto Portfolio Evaluation with Realistic Transaction Costs
Uses the SAME PortfolioOptimizer from src/portfolio/optimizer.py
that IB equities uses, with all combiner methods compared.

- Binance VIP9 Taker: 1.7 bps/side
- 1 tick slippage per trade
- Both TOP50 and TOP100 universes
- All 5 combiner methods from PortfolioOptimizer
- Equity curve plots
"""
import sys, time, os
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from src.operators.fastexpression import FastExpressionEngine
from src.portfolio.optimizer import PortfolioOptimizer
import sqlite3

# ─── Config ───────────────────────────────────────────────────────────
EXCHANGE = os.environ.get("EXCHANGE", "binance")
if EXCHANGE == "binance":
    MATRICES_DIR = Path("data/binance_cache/matrices/4h")
    TICK_FILE = "data/binance_tick_sizes.json"
    TAKER_BPS = 1.7
    OUT_DIR = Path("data/crypto_results")
    TITLE_PREFIX = "Binance"
elif EXCHANGE == "kucoin":
    MATRICES_DIR = Path("data/kucoin_cache/matrices/4h")
    TICK_FILE = "data/kucoin_cache/tick_sizes.json"
    TAKER_BPS = 1.5
    OUT_DIR = Path("data/kucoin_results")
    TITLE_PREFIX = "KuCoin"
else:
    raise ValueError(f"Unknown exchange: {EXCHANGE}")

bars_per_year = 6 * 365
OUT_DIR.mkdir(exist_ok=True)

# ─── Load data ────────────────────────────────────────────────────────
print(f"Loading {EXCHANGE} matrices from {MATRICES_DIR}...", flush=True)
matrices = {}
for f in MATRICES_DIR.glob("*.parquet"):
    matrices[f.stem] = pd.read_parquet(f)
close = matrices["close"]
returns = matrices["returns"]

# Load tick sizes
with open(TICK_FILE) as f:
    tick_sizes_raw = json.load(f)

tick_bps_matrix = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
for sym in close.columns:
    tick = tick_sizes_raw.get(sym, None)
    if tick is not None and tick > 0:
        tick_bps_matrix[sym] = tick / close[sym] * 10000
    else:
        tick_bps_matrix[sym] = 2.8

print(f"  {len(matrices)} matrices, {close.shape[1]} tickers, {len(close)} bars", flush=True)
print(f"  Median tick bps: {tick_bps_matrix.iloc[-1].median():.1f}", flush=True)

# ─── Universes ────────────────────────────────────────────────────────
qv = matrices.get("quote_volume", matrices.get("turnover", matrices["volume"]))
adv20 = qv.rolling(120, min_periods=60).mean()
rank = adv20.rank(axis=1, ascending=False)
top50 = rank <= 50
top100 = rank <= 100

# ─── Load alphas from DBs ────────────────────────────────────────────
conn = sqlite3.connect("data/alphas.db")
crypto_alphas = conn.execute("SELECT id, expression FROM alphas WHERE archived=0").fetchall()
conn.close()

all_alphas = [(f"C{aid}", expr) for aid, expr in crypto_alphas]
print(f"  {len(all_alphas)} alphas loaded", flush=True)

engine = FastExpressionEngine(data_fields=matrices)

# ─── Cost model ───────────────────────────────────────────────────────
def eval_alpha_with_costs(sig_norm, ret_df, tick_bps_mat, taker_bps):
    """Compute PnL net of taker fee + 1 tick slippage.
    
    Uses CAUSAL evaluation: position at bar t is determined by signal[t],
    held through bar t+1, earning return[t+1].  Implemented as:
        pnl[t] = signal[t-1] * return[t]   (shift signal by 1 bar)
    This avoids concurrent lookahead where signal[t] and return[t] both
    depend on close[t].
    """
    lagged_sig = sig_norm.shift(1)
    pnl_gross = (lagged_sig * ret_df).sum(axis=1)
    turnover = lagged_sig.diff().abs()
    cost_per_bar = turnover * (taker_bps + tick_bps_mat.reindex(
        index=turnover.index, columns=turnover.columns, method="ffill").fillna(2.8)) / 10000
    total_cost = cost_per_bar.sum(axis=1)
    pnl_net = pnl_gross - total_cost
    return pnl_gross, pnl_net, turnover.sum(axis=1), total_cost

# ─── Evaluate all alphas ─────────────────────────────────────────────
print(f"\n--- Evaluating alphas on {EXCHANGE} ---", flush=True)

results = []
alpha_signals = {"T50": {}, "T100": {}}

for alpha_name, expr in all_alphas:
    t0 = time.time()
    print(f"  {alpha_name}...", end=" ", flush=True)
    
    try:
        raw = engine.evaluate(expr)
        elapsed = time.time() - t0
        
        if raw is None or raw.empty:
            print(f"EMPTY ({elapsed:.1f}s)", flush=True)
            continue
        
        for uni_name, mask in [("T50", top50), ("T100", top100)]:
            sig = raw.where(mask.reindex(raw.index, method="ffill"), np.nan)
            sig = sig.sub(sig.mean(axis=1), axis=0)
            sig_abs = sig.abs().sum(axis=1).replace(0, np.nan)
            sig_norm = sig.div(sig_abs, axis=0)
            
            pnl_g, pnl_n, to_series, cost_series = eval_alpha_with_costs(
                sig_norm, returns, tick_bps_matrix, TAKER_BPS)
            
            pnl_g = pnl_g.dropna()
            pnl_n = pnl_n.reindex(pnl_g.index)
            
            if len(pnl_g) < 500:
                continue
            
            sr_g = pnl_g.mean() / pnl_g.std() * np.sqrt(bars_per_year) if pnl_g.std() > 0 else 0
            sr_n = pnl_n.mean() / pnl_n.std() * np.sqrt(bars_per_year) if pnl_n.std() > 0 else 0
            ar_n = pnl_n.mean() * bars_per_year
            mean_to = to_series.mean()
            mean_cost = cost_series.mean() * bars_per_year
            cum = (1 + pnl_n).cumprod()
            dd = (cum / cum.cummax() - 1).min()
            
            results.append({
                "alpha": alpha_name, "uni": uni_name, "sr_g": sr_g, "sr_n": sr_n,
                "ann_net": ar_n, "to": mean_to, "dd": dd, 
                "cost_ann": mean_cost,
                "taker_drag": mean_to * TAKER_BPS / 10000 * bars_per_year,
                "tick_drag": mean_cost - mean_to * TAKER_BPS / 10000 * bars_per_year,
                "expr": expr[:70],
            })
            
            if sr_g > 0:
                alpha_signals[uni_name][alpha_name] = (sig_norm, pnl_n)
        
        last_r = [r for r in results if r["alpha"] == alpha_name]
        if last_r:
            print(f"SR_net={last_r[-1]['sr_n']:.2f} ({elapsed:.1f}s)", flush=True)
        else:
            print(f"SKIP ({elapsed:.1f}s)", flush=True)
    except Exception as e:
        err = str(e)
        if "Unknown identifier" in err:
            field = err.split("'")[1] if "'" in err else "?"
            print(f"SKIP (missing: {field}) ({time.time()-t0:.1f}s)", flush=True)
        else:
            print(f"ERR: {err[:50]} ({time.time()-t0:.1f}s)", flush=True)

# ─── Print individual alpha results ──────────────────────────────────
df = pd.DataFrame(results)

for uni in ["T50", "T100"]:
    sub = df[df["uni"] == uni].sort_values("sr_n", ascending=False)
    print(f"\n{'='*120}", flush=True)
    print(f"  {TITLE_PREFIX} {uni} -- taker={TAKER_BPS}bps + 1 tick", flush=True)
    print(f"{'='*120}", flush=True)
    print(f"{'Alpha':<8} {'SR_G':>7} {'SR_N':>7} {'Net%':>7} {'TO':>6} {'DD%':>7} {'Taker%':>7} {'Tick%':>6} {'Cost%':>7}  Expr", flush=True)
    print("-" * 120, flush=True)
    for _, r in sub.iterrows():
        print(f"{r['alpha']:<8} {r['sr_g']:>7.2f} {r['sr_n']:>7.2f} {r['ann_net']*100:>6.1f}% "
              f"{r['to']:>6.3f} {r['dd']*100:>6.1f}% "
              f"{r['taker_drag']*100:>6.1f}% {r['tick_drag']*100:>5.1f}% {r['cost_ann']*100:>6.1f}%  "
              f"{r['expr'][:55]}", flush=True)
    
    print(f"\n  Passing SR_net > 1.0: {(sub['sr_n'] > 1.0).sum()}/{len(sub)}", flush=True)


# ─── Also store RAW signals for signal-level combiners ────────────────
raw_alpha_signals = {"T50": {}, "T100": {}}

for alpha_name, expr in all_alphas:
    try:
        raw = engine.evaluate(expr)
        if raw is None or raw.empty:
            continue
        for uni_name, mask in [("T50", top50), ("T100", top100)]:
            if alpha_name in alpha_signals[uni_name]:
                # Only include alphas that had positive gross SR
                raw_masked = raw.where(mask.reindex(raw.index, method="ffill"), np.nan)
                raw_alpha_signals[uni_name][alpha_name] = raw_masked
    except Exception:
        pass

print(f"\n  Raw signals stored: T50={len(raw_alpha_signals['T50'])}, "
      f"T100={len(raw_alpha_signals['T100'])}", flush=True)


# ─── Portfolio Construction using shared combiners ────────────────────
# Uses the SAME combiner library as IB equities (src/portfolio/combiners.py)
from src.portfolio.combiners import (
    combiner_equal, combiner_adaptive, combiner_risk_parity, combiner_billions
)

CRYPTO_MAX_WT = 0.10  # 10% per name (crypto has fewer tickers than equities)

COMBINER_METHODS = {
    "Equal Weight":  lambda sigs, m, u, r: combiner_equal(sigs, m, u, r, max_wt=CRYPTO_MAX_WT),
    "Adaptive":      lambda sigs, m, u, r: combiner_adaptive(sigs, m, u, r, lookback=504, max_wt=CRYPTO_MAX_WT),
    "Risk Parity":   lambda sigs, m, u, r: combiner_risk_parity(sigs, m, u, r, lookback=504, max_wt=CRYPTO_MAX_WT),
    "Billions":      lambda sigs, m, u, r: combiner_billions(sigs, m, u, r, optim_lookback=60, max_wt=CRYPTO_MAX_WT),
}

COMBINER_COLORS = {
    "Equal Weight":  "#2196F3",
    "Adaptive":      "#FF9800",
    "Risk Parity":   "#4CAF50",
    "Billions":      "#F44336",
}

for uni_name, uni_mask in [("T50", top50), ("T100", top100)]:
    sigs_raw = raw_alpha_signals[uni_name]
    if len(sigs_raw) < 2:
        print(f"\n  {uni_name}: only {len(sigs_raw)} alphas, skipping portfolio", flush=True)
        continue

    print(f"\n{'='*120}", flush=True)
    print(f"  PORTFOLIO: {TITLE_PREFIX} {uni_name} -- {len(sigs_raw)} alphas", flush=True)
    print(f"  Combiners from src/portfolio/combiners.py (same as IB)", flush=True)
    print(f"{'='*120}", flush=True)

    # Build matrices/universe for combiner calls
    uni_df = uni_mask.astype(bool)

    combiner_results = {}
    best_method = None
    best_sr = -999

    print(f"\n  --- Combiner Methods ---", flush=True)
    for method_name, method_fn in COMBINER_METHODS.items():
        t0 = time.time()
        try:
            composite = method_fn(sigs_raw, matrices, uni_df, returns)
            elapsed = time.time() - t0

            # Evaluate composite with cost model
            sig_abs = composite.abs().sum(axis=1).replace(0, np.nan)
            cn = composite.div(sig_abs, axis=0)

            pnl_g, pnl_n, to_series, cost_series = eval_alpha_with_costs(
                cn, returns, tick_bps_matrix, TAKER_BPS)

            pnl_g = pnl_g.dropna()
            pnl_n = pnl_n.reindex(pnl_g.index).dropna()

            if len(pnl_n) < 500:
                print(f"    {method_name:<18s}: SKIP (too few bars)", flush=True)
                continue

            sr_g = pnl_g.mean() / pnl_g.std() * np.sqrt(bars_per_year) if pnl_g.std() > 0 else 0
            sr_n = pnl_n.mean() / pnl_n.std() * np.sqrt(bars_per_year) if pnl_n.std() > 0 else 0
            ar_n = pnl_n.mean() * bars_per_year
            mean_to = to_series.mean()
            cum = (1 + pnl_n).cumprod()
            dd = (cum / cum.cummax() - 1).min()

            combiner_results[method_name] = {
                "pnl_g": pnl_g, "pnl_n": pnl_n, "sr_g": sr_g, "sr_n": sr_n,
                "ar_n": ar_n, "dd": dd, "to": mean_to, "cum": cum,
            }

            print(f"    {method_name:<18s}: SR_g={sr_g:>6.2f} SR_n={sr_n:>6.2f} "
                  f"Ret={ar_n*100:>7.1f}%/yr DD={dd*100:>5.1f}% TO={mean_to:.3f} "
                  f"({elapsed:.1f}s)", flush=True)

            if sr_n > best_sr:
                best_sr = sr_n
                best_method = method_name

        except Exception as e:
            elapsed = time.time() - t0
            print(f"    {method_name:<18s}: ERR: {str(e)[:60]} ({elapsed:.1f}s)", flush=True)

    if not combiner_results:
        continue

    print(f"\n  BEST METHOD: {best_method} (SR_net={best_sr:.2f})", flush=True)

    # ── Equity curve plot ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"{TITLE_PREFIX} {uni_name} -- All Combiners | {len(sigs_raw)} Alphas | "
                 f"Taker={TAKER_BPS}bps + 1 Tick",
                 fontsize=14, fontweight="bold")

    for method_name, r in combiner_results.items():
        color = COMBINER_COLORS.get(method_name, "gray")
        is_best = method_name == best_method
        axes[0].semilogy(r["cum"].index, r["cum"],
                        label=f"{method_name} (SR={r['sr_n']:.1f})",
                        color=color, linewidth=2 if is_best else 1,
                        alpha=1.0 if is_best else 0.5)
        dd_series = r["cum"] / r["cum"].cummax() - 1
        axes[1].plot(dd_series.index, dd_series * 100, color=color,
                    linewidth=1, alpha=0.6)

    # Walk-forward lines
    first = list(combiner_results.values())[0]
    n = len(first["pnl_n"])
    idx = first["pnl_n"].index
    if n > 100:
        axes[0].axvline(idx[int(n*.6)], color="orange", linestyle="--", alpha=0.4, label="Train/Val")
        axes[0].axvline(idx[int(n*.8)], color="red", linestyle="--", alpha=0.4, label="Val/Test")

    axes[0].set_ylabel("Cumulative Return (log)")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("Drawdown %")
    axes[1].set_xlabel("Date")
    axes[1].set_ylim(-20, 0)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = OUT_DIR / f"equity_all_combiners_{uni_name.lower()}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Equity curves: {fig_path}", flush=True)

    # Walk-forward for best
    best_pnl = combiner_results[best_method]["pnl_n"]
    n = len(best_pnl)
    print(f"\n  Walk-forward for {best_method}:", flush=True)
    for name, sl in [("Train(60%)", slice(0,int(n*.6))),
                      ("Val(20%)", slice(int(n*.6),int(n*.8))),
                      ("Test(20%)", slice(int(n*.8),None))]:
        p = best_pnl.iloc[sl]
        sr = p.mean() / p.std() * np.sqrt(bars_per_year) if p.std() > 0 else 0
        ar = p.mean() * bars_per_year
        cum_s = (1 + p).cumprod()
        dd_s = (cum_s / cum_s.cummax() - 1).min()
        print(f"    {name}: SR={sr:.2f} Ret={ar*100:.1f}% DD={dd_s*100:.1f}%", flush=True)

    # GMV scaling
    ar_best = best_pnl.mean() * bars_per_year
    print(f"\n  --- GMV Scaling ({best_method}) ---", flush=True)
    for gmv in [50_000, 100_000, 500_000, 1_000_000]:
        ann_pnl = gmv * ar_best
        print(f"    ${gmv:>10,}: PnL=${ann_pnl:>10,.0f}/yr", flush=True)

print(f"\n{'='*80}", flush=True)
print(f"  DONE: {EXCHANGE} evaluation complete", flush=True)
print(f"  Results in: {OUT_DIR}", flush=True)
print(f"{'='*80}", flush=True)

