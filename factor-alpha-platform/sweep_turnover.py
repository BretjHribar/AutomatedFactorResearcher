"""
Turnover penalty sweep for the Isichenko pipeline.
Tests trade_aversion values to find optimal turnover/alpha tradeoff.
"""
import sys, os, json, time, sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from src.pipeline.isichenko import PipelineConfig, IsichenkoPipeline
from src.operators.fastexpression import FastExpressionEngine

def main():
    print("Loading data...")
    t0 = time.time()
    
    mdir = "data/fmp_cache/matrices_clean"
    if not os.path.isdir(mdir):
        mdir = "data/fmp_cache/matrices"
    
    universe_df = pd.read_parquet("data/fmp_cache/universes/TOP3000.parquet")
    ticker_coverage = universe_df.sum(axis=0) / len(universe_df)
    eligible = ticker_coverage[ticker_coverage > 0.3].index.tolist()
    
    matrices = {}
    for fname in sorted(os.listdir(mdir)):
        if not fname.endswith(".parquet") or fname.startswith("_"):
            continue
        field = fname.replace(".parquet", "")
        matrices[field] = pd.read_parquet(os.path.join(mdir, fname))
    
    close = matrices["close"]
    adv_key = "adv20" if "adv20" in matrices else "dollars_traded"
    adv_df = matrices.get(adv_key, pd.DataFrame())
    
    last_date = close.index[-1]
    if not adv_df.empty and last_date in adv_df.index:
        adv_last = adv_df.loc[last_date]
    else:
        adv_last = (close * matrices.get("volume", pd.DataFrame())).rolling(20).mean().iloc[-1]
    
    adv_eligible = adv_last.reindex(eligible).dropna().sort_values(ascending=False)
    top_tickers = sorted(adv_eligible.head(500).index.tolist())
    
    for field in list(matrices.keys()):
        valid = [c for c in top_tickers if c in matrices[field].columns]
        matrices[field] = matrices[field][valid]
    
    with open("data/fmp_cache/classifications.json") as f:
        all_cls = json.load(f)
    classifications = {k: v for k, v in all_cls.items() if k in top_tickers}
    
    # Load alphas
    conn = sqlite3.connect("data/alpha_gp_pipeline.db")
    cur = conn.cursor()
    cur.execute("""
        SELECT a.expression, e.sharpe 
        FROM alphas a JOIN evaluations e ON a.alpha_id = e.alpha_id 
        WHERE e.sharpe >= 0.5 
        ORDER BY e.sharpe DESC
    """)
    alpha_rows = cur.fetchall()
    conn.close()
    alpha_expressions = [r[0] for r in alpha_rows]
    print(f"  {len(alpha_expressions)} alphas, {len(top_tickers)} tickers")
    print(f"  Data load: {time.time()-t0:.1f}s")
    
    # Build expression engine (shared)
    expr_engine = FastExpressionEngine(data_fields=matrices)
    cs = {}
    for lev in ["sector", "industry", "subindustry"]:
        mp = {s: cd.get(lev, "Unk") for s, cd in classifications.items() if isinstance(cd, dict)}
        if mp:
            cs[lev] = pd.Series(mp)
    for gn, gs in cs.items():
        expr_engine.add_group(gn, gs)
    
    # Sweep trade_aversion values
    sweep_values = [0.0, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6]
    results = []
    
    for ta in sweep_values:
        print(f"\n{'='*60}")
        print(f"  trade_aversion = {ta:.1e}")
        print(f"{'='*60}")
        
        cfg = PipelineConfig(
            is_start="2020-01-01",
            oos_start="2024-01-01",
            warmup_days=120,
            booksize=20_000_000.0,
            risk_aversion=1e-6,
            trade_aversion=ta,
            slippage_bps=1.0,
            impact_coeff=0.1,
            borrow_cost_bps=1.0,
        )
        
        pipeline = IsichenkoPipeline(cfg)
        t1 = time.time()
        
        stats = pipeline.run(
            alpha_expressions=alpha_expressions,
            matrices=matrices,
            classifications=classifications,
            universe_df=universe_df,
            expr_engine=expr_engine,
        )
        
        elapsed = time.time() - t1
        
        if stats and "oos" in stats:
            oos = stats["oos"]
            full = stats["full"]
            results.append({
                "trade_aversion": ta,
                "oos_net_sharpe": oos.get("sharpe", 0),
                "oos_gross_sharpe": oos.get("gross_sharpe", 0),
                "oos_return_net": oos.get("ann_return", 0) * 100,
                "oos_return_gross": oos.get("ann_gross_return", 0) * 100,
                "oos_turnover": oos.get("avg_turnover", 0),
                "oos_tcost": oos.get("total_tcost", 0),
                "oos_pnl": oos.get("cum_pnl", 0),
                "oos_max_dd": oos.get("max_drawdown", 0) * 100,
                "full_net_sharpe": full.get("sharpe", 0),
                "full_pnl": full.get("cum_pnl", 0),
                "runtime": elapsed,
            })
            
            r = results[-1]
            print(f"\n  OOS: Sharpe={r['oos_net_sharpe']:+.2f} (gross={r['oos_gross_sharpe']:+.2f})")
            print(f"  Return: net={r['oos_return_net']:.1f}% gross={r['oos_return_gross']:.1f}%")
            print(f"  Turnover: {r['oos_turnover']:.1%}, TCost: ${r['oos_tcost']:,.0f}")
            print(f"  Runtime: {elapsed:.0f}s")
    
    # Print comparison table
    print(f"\n\n{'='*100}")
    print(f"  TURNOVER PENALTY SWEEP RESULTS")
    print(f"{'='*100}")
    print(f"{'trade_aversion':>16s} | {'OOS Net SR':>10s} | {'Gross SR':>8s} | {'Net Ret':>7s} | {'T/O':>5s} | {'TCost':>10s} | {'OOS PnL':>12s} | {'MaxDD':>6s}")
    print(f"{'-'*16}-+-{'-'*10}-+-{'-'*8}-+-{'-'*7}-+-{'-'*5}-+-{'-'*10}-+-{'-'*12}-+-{'-'*6}")
    
    best_sharpe = -999
    best_ta = 0
    for r in results:
        is_best = r["oos_net_sharpe"] > best_sharpe
        if is_best:
            best_sharpe = r["oos_net_sharpe"]
            best_ta = r["trade_aversion"]
        marker = " ★" if is_best else ""
        print(f"{r['trade_aversion']:>16.1e} | {r['oos_net_sharpe']:>+10.3f} | {r['oos_gross_sharpe']:>+8.2f} | {r['oos_return_net']:>6.1f}% | {r['oos_turnover']:>4.1%} | ${r['oos_tcost']:>9,.0f} | ${r['oos_pnl']:>11,.0f} | {r['oos_max_dd']:>5.1f}%{marker}")
    
    print(f"\n  ★ BEST: trade_aversion={best_ta:.1e} → OOS Net Sharpe={best_sharpe:+.3f}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Turnover Penalty Sweep — Isichenko Pipeline", fontsize=16, fontweight="bold")
    
    ta_labels = [f"{r['trade_aversion']:.0e}" for r in results]
    x = range(len(results))
    
    colors = ['gold' if r['trade_aversion'] == best_ta else 'steelblue' for r in results]
    
    ax = axes[0, 0]
    ax.bar(x, [r["oos_net_sharpe"] for r in results], color=colors, alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(ta_labels, rotation=45, fontsize=8)
    ax.set_ylabel("OOS Net Sharpe")
    ax.set_title("OOS Net Sharpe vs Trade Aversion")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    
    ax = axes[0, 1]
    ax.bar(x, [r["oos_turnover"]*100 for r in results], color=colors, alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(ta_labels, rotation=45, fontsize=8)
    ax.set_ylabel("OOS Avg Turnover (%)")
    ax.set_title("Turnover vs Trade Aversion")
    
    ax = axes[1, 0]
    ax.bar(x, [r["oos_pnl"] for r in results], color=colors, alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(ta_labels, rotation=45, fontsize=8)
    ax.set_ylabel("OOS PnL ($)")
    ax.set_title("OOS PnL vs Trade Aversion")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e6:.1f}M"))
    
    ax = axes[1, 1]
    net = [r["oos_return_net"] for r in results]
    gross = [r["oos_return_gross"] for r in results]
    w = 0.35
    ax.bar([i-w/2 for i in x], gross, w, color="lightblue", alpha=0.7, label="Gross")
    ax.bar([i+w/2 for i in x], net, w, color="steelblue", alpha=0.9, label="Net")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ta_labels, rotation=45, fontsize=8)
    ax.set_ylabel("Ann Return (%)")
    ax.set_title("Gross vs Net Return")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("data/turnover_sweep.png", dpi=150)
    print(f"\n  Saved: data/turnover_sweep.png")
    
    # Save results
    with open("data/turnover_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: data/turnover_sweep.json")


if __name__ == "__main__":
    main()
