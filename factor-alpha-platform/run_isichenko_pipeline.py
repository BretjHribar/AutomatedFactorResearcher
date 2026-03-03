"""
Run the Isichenko Stat-Arb Pipeline
====================================
Loads GP-discovered alphas from the database, runs the full 
Isichenko pipeline (alpha scaling, risk model, QP optimizer, 
sequential backtest), and produces comprehensive statistics + graphs.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.isichenko import PipelineConfig, IsichenkoPipeline
from src.operators.fastexpression import FastExpressionEngine
import sqlite3

def main():
    t0 = time.time()
    print("=" * 70)
    print("  ISICHENKO STAT-ARB PIPELINE")
    print("  Following: Quantitative Portfolio Management (2021)")
    print("=" * 70)
    
    # ── Config ──
    config = PipelineConfig(
        is_start="2020-01-01",
        oos_start="2024-01-01",
        warmup_days=120,
        booksize=20_000_000.0,
        risk_aversion=1e-6,
        slippage_bps=1.0,
        impact_coeff=0.1,
        ema_halflife_risk=60,
        ema_halflife_alpha=120,
        dollar_neutral=True,
        sector_neutral=True,
        max_position_pct_gmv=0.02,
        max_position_pct_adv=0.05,
        delay=1,
    )
    
    print(f"\n Config:")
    print(f"   IS period:    {config.is_start} → {config.oos_start}")
    print(f"   OOS period:   {config.oos_start} → present")
    print(f"   Book size:    ${config.booksize:,.0f}")
    print(f"   Risk aversion: {config.risk_aversion}")
    print(f"   Slippage:     {config.slippage_bps} bps")
    print(f"   Impact coeff: {config.impact_coeff}")
    print(f"   Neutralization: dollar={config.dollar_neutral}, sector={config.sector_neutral}")
    
    # ── Load Data ──
    print(f"\n📊 Loading data...")
    
    # Universe — use TOP1000 (same as GP alpha discovery)
    universe_df = pd.read_parquet("data/fmp_cache/universes/TOP1000.parquet")
    
    # Filter tickers with good coverage
    ui = universe_df.loc[config.is_start:config.oos_start]
    tc = ui.sum(axis=0) / len(ui)
    tickers = sorted(tc[tc > config.min_coverage].index.tolist())
    print(f"   Universe: TOP1000, {len(tickers)} tickers with >{config.min_coverage*100:.0f}% coverage")
    
    # Load matrices (use cleaned data if available)
    matrices = {}
    mdir_clean = "data/fmp_cache/matrices_clean"
    mdir_raw = "data/fmp_cache/matrices"
    mdir = mdir_clean if os.path.isdir(mdir_clean) else mdir_raw
    print(f"   Using data from: {mdir}")
    for fn in sorted(os.listdir(mdir)):
        if not fn.endswith(".parquet") or fn.startswith("_"):
            continue
        df = pd.read_parquet(f"{mdir}/{fn}")
        vc = [c for c in tickers if c in df.columns]
        if vc:
            matrices[fn.replace(".parquet", "")] = df[vc]
    print(f"   Loaded {len(matrices)} data fields")
    
    # Apply universe mask
    for f, m in matrices.items():
        if isinstance(m, pd.DataFrame) and m.shape[1] > 1:
            cc = m.columns.intersection(universe_df.columns)
            ci = m.index.intersection(universe_df.index)
            if len(cc) > 0 and len(ci) > 0:
                matrices[f] = m.loc[ci, cc].where(universe_df.loc[ci, cc])
    
    # Classifications
    with open("data/fmp_cache/classifications.json") as f:
        all_cls = json.load(f)
    classifications = {k: v for k, v in all_cls.items() if k in tickers}
    
    # ── Curated Alpha Set ──
    print(f"\n🧬 Loading curated alphas...")
    alpha_expressions = [
        "Inverse(ArgMax(total_debt, 20))",
        "rank(change_in_working_capital)",
        "sqrt(ts_rank(working_capital, 18))",
        "log10(npfadd(debt_to_equity, -0.6120160888134647))",
        "ts_entropy(stock_based_compensation, 38)",
        "ts_min(volume, 35)",
    ]
    print(f"   {len(alpha_expressions)} curated alphas:")
    for i, expr in enumerate(alpha_expressions):
        print(f"     {i+1}. {expr}")
    
    # ── Build Expression Engine ──
    print(f"\n⚙️  Setting up expression engine...")
    engine = FastExpressionEngine(data_fields=matrices)
    
    cs = {}
    for lev in ["sector", "industry", "subindustry"]:
        mp = {s: cd.get(lev, "Unk") for s, cd in classifications.items() if isinstance(cd, dict)}
        if mp:
            cs[lev] = pd.Series(mp)
    for gn, gs in cs.items():
        engine.add_group(gn, gs)
    
    # ── Run Pipeline ──
    print(f"\n🚀 Running Isichenko pipeline...")
    print(f"   This will process ~{len(pd.bdate_range(config.is_start, '2026-02-26'))} trading days sequentially.\n")
    
    pipeline = IsichenkoPipeline(config)
    results = pipeline.run(
        alpha_expressions=alpha_expressions,
        matrices=matrices,
        classifications=classifications,
        universe_df=universe_df,
        expr_engine=engine,
    )
    
    if not results:
        print("Pipeline returned no results!")
        return
    
    elapsed = time.time() - t0
    
    # ── Print Results ──
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    
    for period in ["full", "is", "oos"]:
        s = results.get(period, {})
        if not s:
            continue
        print(f"\n  ── {s['label']} ({s['start_date']} → {s['end_date']}, {s['n_days']} days) ──")
        print(f"    Net Sharpe:        {s['sharpe']:+.2f}")
        print(f"    Gross Sharpe:      {s['gross_sharpe']:+.2f}")
        print(f"    Ann Return (net):  {s['ann_return']:+.1%}")
        print(f"    Ann Return (gross):{s['ann_gross_return']:+.1%}")
        print(f"    Max Drawdown:      {s['max_drawdown']:.1%}")
        print(f"    Calmar Ratio:      {s['calmar']:.2f}")
        print(f"    Win Rate:          {s['win_rate']:.1%}")
        print(f"    Profit Factor:     {s['profit_factor']:.2f}")
        print(f"    Cumulative PnL:    ${s['cum_pnl']:+,.0f}")
        print(f"    Avg Daily PnL:     ${s['avg_daily_pnl']:+,.0f}")
        print(f"    Std Daily PnL:     ${s['std_daily_pnl']:,.0f}")
        print(f"    Avg Daily Gross:   ${s['avg_daily_gross']:+,.0f}")
        print(f"    Avg Daily TCost:   ${s['avg_daily_tcost']:,.0f}")
        print(f"    Total TCost:       ${s['total_tcost']:,.0f}")
        print(f"    Avg GMV:           ${s['avg_gmv']:,.0f}")
        print(f"    Avg Turnover:      {s['avg_turnover']:.1%}")
        print(f"    Avg Longs:         {s['avg_n_long']:.0f}")
        print(f"    Avg Shorts:        {s['avg_n_short']:.0f}")
    
    # ── IS vs OOS Comparison ──
    is_s = results.get("is", {})
    oos_s = results.get("oos", {})
    if is_s and oos_s:
        print(f"\n  ── IS → OOS Decay Analysis ──")
        sharpe_decay = (1 - oos_s["sharpe"] / is_s["sharpe"]) * 100 if is_s["sharpe"] != 0 else 0
        ret_decay = (1 - oos_s["ann_return"] / is_s["ann_return"]) * 100 if is_s["ann_return"] != 0 else 0
        print(f"    Sharpe Decay:      {sharpe_decay:+.0f}% ({is_s['sharpe']:+.2f} → {oos_s['sharpe']:+.2f})")
        print(f"    Return Decay:      {ret_decay:+.0f}% ({is_s['ann_return']:+.1%} → {oos_s['ann_return']:+.1%})")
    
    print(f"\n  Total runtime: {elapsed:.1f}s")
    
    # ── Generate Graphs ──
    print(f"\n📈 Generating performance graphs...")
    
    full = results["full"]
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in full["dates"]]
    pnls = np.array(full["daily_pnls"])
    cum_pnl = np.cumsum(pnls)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), gridspec_kw={"height_ratios": [3, 1.5, 1, 1]})
    fig.suptitle("Isichenko Stat-Arb Pipeline — Full Performance", fontsize=16, fontweight="bold")
    
    # 1. Cumulative PnL with IS/OOS split
    ax1 = axes[0]
    oos_date = datetime.strptime(config.oos_start, "%Y-%m-%d")
    is_mask = np.array([d < oos_date for d in dates])
    oos_mask = ~is_mask
    
    ax1.fill_between(dates, 0, cum_pnl, where=is_mask, alpha=0.3, color="steelblue", label="IS")
    ax1.fill_between(dates, 0, cum_pnl, where=oos_mask, alpha=0.3, color="darkorange", label="OOS")
    ax1.plot(dates, cum_pnl, color="black", linewidth=1.2)
    ax1.axvline(x=oos_date, color="red", linestyle="--", alpha=0.7, label="OOS Start")
    ax1.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    
    if is_s:
        ax1.text(0.02, 0.95, f"IS Sharpe: {is_s['sharpe']:+.2f}", transform=ax1.transAxes,
                fontsize=11, verticalalignment="top", fontweight="bold", color="steelblue")
    if oos_s:
        ax1.text(0.02, 0.88, f"OOS Sharpe: {oos_s['sharpe']:+.2f}", transform=ax1.transAxes,
                fontsize=11, verticalalignment="top", fontweight="bold", color="darkorange")
    
    ax1.set_ylabel("Cumulative PnL ($)")
    ax1.set_title("Cumulative Net PnL (After Transaction Costs)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    
    # 2. Drawdown
    ax2 = axes[1]
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - running_max
    ax2.fill_between(dates, drawdown, 0, alpha=0.4, color="crimson")
    ax2.plot(dates, drawdown, color="crimson", linewidth=0.8)
    ax2.set_ylabel("Drawdown ($)")
    ax2.set_title("Drawdown")
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    
    # 3. Rolling 60-day Sharpe
    ax3 = axes[2]
    window = 60
    if len(pnls) >= window:
        rolling_mean = pd.Series(pnls).rolling(window).mean()
        rolling_std = pd.Series(pnls).rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std * np.sqrt(252)).values
        ax3.plot(dates, rolling_sharpe, color="purple", linewidth=0.8)
        ax3.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax3.axhline(y=1, color="green", linestyle="--", alpha=0.3, label="Sharpe=1")
        ax3.axhline(y=2, color="blue", linestyle="--", alpha=0.3, label="Sharpe=2")
        ax3.axvline(x=oos_date, color="red", linestyle="--", alpha=0.7)
    ax3.set_ylabel("Rolling Sharpe")
    ax3.set_title(f"Rolling {window}-Day Sharpe Ratio")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Daily PnL bar chart
    ax4 = axes[3]
    colors = ["green" if p > 0 else "red" for p in pnls]
    ax4.bar(dates, pnls, color=colors, alpha=0.5, width=1.5)
    ax4.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax4.axvline(x=oos_date, color="red", linestyle="--", alpha=0.7)
    ax4.set_ylabel("Daily PnL ($)")
    ax4.set_title("Daily Net PnL")
    ax4.grid(True, alpha=0.3)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    fig.savefig("data/isichenko_pipeline_performance.png", dpi=150, bbox_inches="tight")
    print(f"   Saved: data/isichenko_pipeline_performance.png")
    
    # ── Save results JSON ──
    output = {
        "config": {
            "is_start": config.is_start,
            "oos_start": config.oos_start,
            "booksize": config.booksize,
            "risk_aversion": config.risk_aversion,
            "slippage_bps": config.slippage_bps,
            "impact_coeff": config.impact_coeff,
            "n_alphas": len(alpha_expressions),
        },
        "full": {k: v for k, v in full.items() if k not in ["daily_pnls", "dates"]},
        "is": {k: v for k, v in is_s.items() if k not in ["daily_pnls", "dates"]},
        "oos": {k: v for k, v in oos_s.items() if k not in ["daily_pnls", "dates"]},
    }
    with open("data/isichenko_pipeline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"   Saved: data/isichenko_pipeline_results.json")
    
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
