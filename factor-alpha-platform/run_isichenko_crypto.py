"""
Run the Isichenko Stat-Arb Pipeline — CRYPTO VERSION
=====================================================
Same pipeline as run_isichenko_pipeline.py (used for IB equities)
but adapted for Binance/KuCoin 4h bar crypto data.

Key differences from IB:
  - No sector/industry classifications → sector_neutral=False
  - 4h bars → bars_per_year = 6*365 = 2190
  - No fundamental data → style factors are price-derived only
  - Fees: Binance VIP9 taker (1.7 bps) + tick slippage

Usage:
  python run_isichenko_crypto.py                    # Binance (default)
  python run_isichenko_crypto.py --exchange kucoin   # KuCoin
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.isichenko import PipelineConfig, IsichenkoPipeline
from src.operators.fastexpression import FastExpressionEngine
import sqlite3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", default="binance", choices=["binance", "kucoin"])
    parser.add_argument("--universe", default="TOP100", choices=["TOP50", "TOP100"])
    args = parser.parse_args()

    t0 = time.time()
    exchange = args.exchange
    uni_size = int(args.universe.replace("TOP", ""))

    print("=" * 70)
    print(f"  ISICHENKO STAT-ARB PIPELINE — {exchange.upper()} CRYPTO")
    print(f"  Universe: {args.universe} | Same pipeline as IB equities")
    print("=" * 70)

    # ── Config — adapted for crypto ──
    if exchange == "binance":
        matrices_dir = "data/binance_cache/matrices/4h"
        tick_file = "data/binance_tick_sizes.json"
        taker_bps = 1.7
        out_prefix = "binance"
    else:
        matrices_dir = "data/kucoin_cache/matrices/4h"
        tick_file = "data/kucoin_cache/tick_sizes.json"
        taker_bps = 1.5
        out_prefix = "kucoin"

    config = PipelineConfig(
        is_start="2020-06-01",        # Start training (crypto has less history)
        oos_start="2024-01-01",       # OOS start
        warmup_days=120,              # ~20 days in daily terms (120 4h bars)
        booksize=100_000.0,           # $100k crypto book
        risk_aversion=1e-6,           # Same as IB
        slippage_bps=taker_bps,       # Exchange taker fee as slippage
        commission_bps=0.0,           # Already in slippage
        impact_coeff=0.05,            # Lower impact — crypto is liquid 24/7
        ema_halflife_risk=360,        # 60 "days" × 6 bars/day = 360 bars
        ema_halflife_alpha=720,       # 120 "days" × 6 bars = 720 bars
        dollar_neutral=True,
        sector_neutral=False,         # NO sectors in crypto
        max_position_pct_gmv=0.03,    # 3% max per name (crypto is concentrated)
        max_position_pct_adv=0.05,
        delay=1,                      # Delay-1 (conservative)
        raw_signal_mode=False,        # IC-weighted (the "Billions" combiner)
    )

    bars_per_year = 6 * 365

    print(f"\n  Config:")
    print(f"    IS period:     {config.is_start} -> {config.oos_start}")
    print(f"    OOS period:    {config.oos_start} -> present")
    print(f"    Book size:     ${config.booksize:,.0f}")
    print(f"    Risk aversion: {config.risk_aversion}")
    print(f"    Slippage:      {config.slippage_bps} bps (taker fee)")
    print(f"    Impact coeff:  {config.impact_coeff}")
    print(f"    IC-weighted:   {not config.raw_signal_mode} (Billions combiner)")

    # ── Load Data ──
    print(f"\n  Loading {exchange} data from {matrices_dir}...")
    matrices = {}
    for fn in sorted(os.listdir(matrices_dir)):
        if not fn.endswith(".parquet") or fn.startswith("_"):
            continue
        df = pd.read_parquet(f"{matrices_dir}/{fn}")
        matrices[fn.replace(".parquet", "")] = df
    print(f"    {len(matrices)} matrices loaded")

    close = matrices["close"]
    print(f"    {close.shape[1]} tickers, {len(close)} bars")
    print(f"    Date range: {close.index[0]} -> {close.index[-1]}")

    # ── Build universe ──
    qv = matrices.get("quote_volume", matrices.get("turnover", matrices["volume"]))
    adv20 = qv.rolling(120, min_periods=60).mean()
    rank = adv20.rank(axis=1, ascending=False)
    universe_df = (rank <= uni_size).astype(bool)
    active = universe_df.iloc[-1].sum()
    print(f"    Universe: {args.universe} ({active} active tickers)")

    # Filter matrices to universe tickers (all tickers that ever appear)
    ever_in_uni = universe_df.any(axis=0)
    tickers = sorted(ever_in_uni[ever_in_uni].index.tolist())
    print(f"    Tickers ever in universe: {len(tickers)}")

    for name in list(matrices.keys()):
        cols = [c for c in tickers if c in matrices[name].columns]
        if cols:
            matrices[name] = matrices[name][cols]
        else:
            del matrices[name]

    universe_df = universe_df[[c for c in tickers if c in universe_df.columns]]

    # ── Load alphas ──
    print(f"\n  Loading alphas...")
    
    # Same alphas as the eval script
    conn = sqlite3.connect("data/alphas.db")
    crypto_alphas = conn.execute("SELECT id, expression FROM alphas WHERE archived=0").fetchall()
    conn.close()
    conn2 = sqlite3.connect("data/ib_alphas.db")
    ib12 = conn2.execute("SELECT expression FROM alphas WHERE id=12").fetchone()
    conn2.close()

    alpha_expressions = [expr for _, expr in crypto_alphas]
    alpha_expressions.append(ib12[0])
    print(f"    {len(alpha_expressions)} alpha expressions")

    # ── Build Expression Engine ──
    engine = FastExpressionEngine(data_fields=matrices)

    # No sector/industry groups for crypto — just empty classifications
    classifications = {}

    # ── Run Pipeline ──
    print(f"\n  Running Isichenko pipeline ({exchange} {args.universe})...")
    n_bars_approx = len(close.loc[config.is_start:])
    print(f"    ~{n_bars_approx} bars to process sequentially\n")

    pipeline = IsichenkoPipeline(config)
    results = pipeline.run(
        alpha_expressions=alpha_expressions,
        matrices=matrices,
        classifications=classifications,
        universe_df=universe_df,
        expr_engine=engine,
    )

    if not results:
        print("  Pipeline returned no results!")
        return

    elapsed = time.time() - t0

    # ── Print Results ──
    print("\n" + "=" * 70)
    print(f"  RESULTS — {exchange.upper()} {args.universe} (Isichenko Pipeline)")
    print("=" * 70)

    for period in ["full", "is", "oos"]:
        s = results.get(period, {})
        if not s:
            continue
        print(f"\n  -- {s['label']} ({s['start_date']} -> {s['end_date']}, {s['n_days']} bars) --")
        
        # Annualize using bars_per_year for 4h data
        avg_pnl = s['avg_daily_pnl']
        std_pnl = s['std_daily_pnl']
        sr_4h = avg_pnl / std_pnl * np.sqrt(bars_per_year) if std_pnl > 0 else 0
        
        print(f"    Net Sharpe (4h ann): {sr_4h:+.2f}")
        print(f"    Gross Sharpe:        {s['gross_sharpe']:+.2f}")
        print(f"    Cum PnL:             ${s['cum_pnl']:+,.0f}")
        print(f"    Max Drawdown:        {s['max_drawdown']:.1%}")
        print(f"    Win Rate:            {s['win_rate']:.1%}")
        print(f"    Profit Factor:       {s['profit_factor']:.2f}")
        print(f"    Avg Bar PnL:         ${avg_pnl:+,.2f}")
        print(f"    Total TCost:         ${s['total_tcost']:,.0f}")
        print(f"    Avg GMV:             ${s['avg_gmv']:,.0f}")
        print(f"    Avg Turnover:        {s['avg_turnover']:.1%}")
        print(f"    Avg L/S:             {s['avg_n_long']:.0f}L / {s['avg_n_short']:.0f}S")

    # IS vs OOS decay
    is_s = results.get("is", {})
    oos_s = results.get("oos", {})
    if is_s and oos_s:
        print(f"\n  -- IS -> OOS Decay --")
        print(f"    Sharpe: {is_s['sharpe']:+.2f} -> {oos_s['sharpe']:+.2f}")

    print(f"\n  Runtime: {elapsed:.1f}s")

    # ── Graphs ──
    print(f"\n  Generating graphs...")
    
    full = results["full"]
    dates = [datetime.strptime(d, "%Y-%m-%d") if len(d) == 10 else 
             datetime.fromisoformat(d.replace("T", " ").split("+")[0][:19]) 
             for d in full["dates"]]
    pnls = np.array(full["daily_pnls"])
    cum_pnl = np.cumsum(pnls)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle(f"Isichenko Pipeline — {exchange.upper()} {args.universe} | "
                 f"{len(alpha_expressions)} Alphas | IC-Weighted (Billions)",
                 fontsize=14, fontweight="bold")

    # 1. Cumulative PnL
    ax1 = axes[0]
    try:
        oos_date = datetime.strptime(config.oos_start, "%Y-%m-%d")
        is_mask = np.array([d < oos_date for d in dates])
        oos_mask = ~is_mask
        ax1.fill_between(dates, 0, cum_pnl, where=is_mask, alpha=0.3, color="steelblue", label="IS")
        ax1.fill_between(dates, 0, cum_pnl, where=oos_mask, alpha=0.3, color="darkorange", label="OOS")
        ax1.axvline(x=oos_date, color="red", linestyle="--", alpha=0.7, label="OOS Start")
    except:
        pass
    ax1.plot(dates, cum_pnl, color="black", linewidth=1.2)
    ax1.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

    if is_s:
        ax1.text(0.02, 0.95, f"IS Sharpe: {is_s['sharpe']:+.2f}", transform=ax1.transAxes,
                fontsize=11, verticalalignment="top", fontweight="bold", color="steelblue")
    if oos_s:
        ax1.text(0.02, 0.88, f"OOS Sharpe: {oos_s['sharpe']:+.2f}", transform=ax1.transAxes,
                fontsize=11, verticalalignment="top", fontweight="bold", color="darkorange")

    ax1.set_ylabel("Cumulative PnL ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # 2. Drawdown
    ax2 = axes[1]
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - running_max
    ax2.fill_between(dates, drawdown, 0, alpha=0.4, color="crimson")
    ax2.set_ylabel("Drawdown ($)")
    ax2.grid(True, alpha=0.3)

    # 3. Rolling Sharpe
    ax3 = axes[2]
    window = 360  # ~60 days at 6 bars/day
    if len(pnls) >= window:
        rm = pd.Series(pnls).rolling(window).mean()
        rs = pd.Series(pnls).rolling(window).std()
        rolling_sr = (rm / rs * np.sqrt(bars_per_year)).values
        ax3.plot(dates, rolling_sr, color="purple", linewidth=0.8)
        ax3.axhline(y=0, color="gray", alpha=0.3)
        ax3.axhline(y=1, color="green", linestyle="--", alpha=0.3)
    ax3.set_ylabel("Rolling Sharpe")
    ax3.set_xlabel("Date")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"data/{out_prefix}_isichenko_{args.universe.lower()}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")

    # Save results JSON
    output = {
        "config": {
            "exchange": exchange,
            "universe": args.universe,
            "is_start": config.is_start,
            "oos_start": config.oos_start,
            "booksize": config.booksize,
            "taker_bps": taker_bps,
            "n_alphas": len(alpha_expressions),
            "combiner": "IC-weighted (Billions)",
        },
        "full": {k: v for k, v in full.items() if k not in ["daily_pnls", "dates"]},
        "is": {k: v for k, v in is_s.items() if k not in ["daily_pnls", "dates"]} if is_s else {},
        "oos": {k: v for k, v in oos_s.items() if k not in ["daily_pnls", "dates"]} if oos_s else {},
    }
    results_path = f"data/{out_prefix}_isichenko_{args.universe.lower()}_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"    Saved: {results_path}")

    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
