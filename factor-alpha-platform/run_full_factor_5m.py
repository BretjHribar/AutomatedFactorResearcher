"""
run_full_factor_5m.py — Full-factor continuous portfolio (eval_alpha_5m style)

Uses the EXACT same data loading, universe, process_signal, and simulate
pipeline as eval_alpha_5m.py. The 2 active alphas are equally combined,
processed, and simulated across train, val, and test splits.

This is a standard cross-sectional alpha: hold every symbol, weight
proportional to signal, rebalance every 5m bar. No thresholds, no hold timers.
"""

import sys, os, time, sqlite3
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Use eval_alpha_5m's exact infrastructure
import eval_alpha_5m as ea

# ============================================================================
# CONFIGURATION — match eval_alpha_5m.py exactly
# ============================================================================

FEES_BPS = 7.0        # 7 bps per side
UNIVERSE = "BINANCE_TOP50"   # Alphas were discovered here
INTERVAL = "5m"
BOOKSIZE = ea.BOOKSIZE       # 2M
BARS_PER_DAY = ea.BARS_PER_DAY  # 288
NEUTRALIZE = ea.NEUTRALIZE      # "market"
COVERAGE_CUTOFF = ea.COVERAGE_CUTOFF  # 0.3
DB_PATH = ea.DB_PATH

# Override eval_alpha_5m's global UNIVERSE so its load_data uses correct file
ea.UNIVERSE = UNIVERSE

# Universe-specific max weight
UNIVERSE_CONFIG = ea.UNIVERSE_CONFIG
ucfg = UNIVERSE_CONFIG.get(UNIVERSE, UNIVERSE_CONFIG["BINANCE_TOP100"])
MAX_WEIGHT = ucfg["max_weight"]
ea.MAX_WEIGHT = MAX_WEIGHT

# Splits: full range from train through test
TRAIN_START = "2025-02-01"
TRAIN_END   = "2026-02-01"
VAL_START   = "2026-02-01"
VAL_END     = "2026-03-01"
TEST_START  = "2026-03-01"
TEST_END    = "2026-03-27"

SPLITS = {
    "train": (TRAIN_START, TRAIN_END),
    "val":   (VAL_START, VAL_END),
    "test":  (TEST_START, TEST_END),
}


# ============================================================================
# DATA LOADING — extends eval_alpha_5m to support val/test splits
# ============================================================================

def load_data_extended(split):
    """
    Load data for any split using eval_alpha_5m's exact universe + matrix loading,
    extended to support val and test periods.
    """
    mat_dir = Path(f"data/binance_cache/matrices/{INTERVAL}")
    
    # Load universe using eval_alpha_5m's pattern
    uni_path = Path(f"data/binance_cache/universes/{UNIVERSE}_{INTERVAL}.parquet")
    if not uni_path.exists():
        raise FileNotFoundError(f"Universe file not found: {uni_path}")
    
    if '_full_matrices' not in ea._DATA_CACHE:
        universe_df = pd.read_parquet(uni_path)
        coverage = universe_df.sum(axis=0) / len(universe_df)
        valid_tickers = sorted(coverage[coverage > COVERAGE_CUTOFF].index.tolist())
        ea._DATA_CACHE['_tickers'] = valid_tickers
        ea._DATA_CACHE['_universe_df'] = universe_df
        
        print(f"  Loading 5m matrices ({len(valid_tickers)} tickers)...", flush=True)
        t0 = time.time()
        matrices = {}
        for fp in sorted(mat_dir.glob("*.parquet")):
            df = pd.read_parquet(fp)
            cols = [c for c in valid_tickers if c in df.columns]
            if cols:
                matrices[fp.stem] = df[cols]
        ea._DATA_CACHE['_full_matrices'] = matrices
        print(f"  Loaded {len(matrices)} fields in {time.time()-t0:.1f}s", flush=True)
    else:
        valid_tickers = ea._DATA_CACHE['_tickers']
        universe_df = ea._DATA_CACHE['_universe_df']
        matrices = ea._DATA_CACHE['_full_matrices']
    
    start, end = SPLITS[split]
    split_matrices = {name: df.loc[start:end] for name, df in matrices.items()}
    split_universe = universe_df[valid_tickers].loc[start:end]
    
    if "close" in split_matrices:
        close = split_matrices["close"]
        split_matrices["returns"] = close.pct_change()
    
    return split_matrices, split_universe


# ============================================================================
# LOAD ALPHAS FROM DB
# ============================================================================

def load_alphas():
    """Load the 2 active alphas from the DB (BINANCE_TOP50)."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT a.id, a.expression, COALESCE(e.ic_mean, 0), COALESCE(e.sharpe_is, 0)
        FROM alphas a LEFT JOIN evaluations e ON e.alpha_id = a.id
        WHERE a.archived = 0 AND a.universe = ?
        ORDER BY COALESCE(e.ic_mean, 0) DESC
    """, (UNIVERSE,)).fetchall()
    conn.close()
    return rows


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0_total = time.time()
    
    # Load alphas
    alphas = load_alphas()
    print(f"\n{'='*80}")
    print(f"  FULL-FACTOR CONTINUOUS PORTFOLIO — eval_alpha_5m style")
    print(f"  Universe: {UNIVERSE} | Fees: {FEES_BPS} bps | Max Wt: {MAX_WEIGHT}")
    print(f"  Booksize: ${BOOKSIZE:,.0f} | Neutralization: {NEUTRALIZE}")
    print(f"  Alphas: {len(alphas)}")
    print(f"{'='*80}")
    for a in alphas:
        print(f"  #{a[0]:2d} IC={a[2]:+.05f} SR={a[3]:+.2f} | {a[1][:70]}")
    
    if not alphas:
        print("  No alphas found! Exiting.")
        return
    
    # Run each split
    results = {}
    
    for split_name in ["train", "val", "test"]:
        start, end = SPLITS[split_name]
        print(f"\n  {'='*60}")
        print(f"  {split_name.upper()} ({start} to {end})")
        print(f"  {'='*60}")
        
        # Load data for this split
        matrices, universe = load_data_extended(split_name)
        close = matrices.get("close")
        returns = close.pct_change() if close is not None else matrices.get("returns")
        
        print(f"  Data: {close.shape[0]} bars x {close.shape[1]} tickers")
        
        # Evaluate each alpha and combine (equal-weight average of raw signals)
        raw_signals = []
        for alpha_id, expression, ic_mean, sharpe_is in alphas:
            try:
                raw = ea.evaluate_expression(expression, matrices)
                if raw is not None and not raw.empty:
                    raw_signals.append(raw)
                    print(f"    Alpha #{alpha_id}: OK ({raw.shape})")
                else:
                    print(f"    Alpha #{alpha_id}: SKIP (None/empty)")
            except Exception as e:
                print(f"    Alpha #{alpha_id}: ERROR ({e})")
        
        if not raw_signals:
            print(f"  No valid signals for {split_name}!")
            continue
        
        # Equal-weight average of raw signals
        combined_raw = raw_signals[0].copy()
        for sig in raw_signals[1:]:
            combined_raw = combined_raw.add(sig, fill_value=0)
        combined_raw = combined_raw / len(raw_signals)
        
        # Process signal using eval_alpha_5m's exact pipeline:
        # demean → scale by abs sum → clip to max_weight → fillna(0)
        alpha_processed = ea.process_signal(combined_raw, universe_df=universe, max_wt=MAX_WEIGHT)
        
        print(f"  Combined signal: {alpha_processed.shape}")
        print(f"  Non-zero weights per bar: {(alpha_processed.abs() > 1e-8).sum(axis=1).mean():.1f} avg")
        
        # Simulate using eval_alpha_5m's exact simulate function
        # This is the full-factor version: hold every symbol, rebalance every 5m
        sim_result = ea.simulate(
            alpha_df=alpha_processed,
            returns_df=returns,
            close_df=close,
            universe_df=universe,
            fees_bps=FEES_BPS,
        )
        
        # Store result
        results[split_name] = {
            "sharpe": sim_result.sharpe,
            "fitness": sim_result.fitness,
            "turnover": sim_result.turnover,
            "returns_ann": sim_result.returns_ann,
            "max_drawdown": sim_result.max_drawdown,
            "total_pnl": sim_result.total_pnl,
            "psr": sim_result.psr,
            "daily_pnl": sim_result.daily_pnl,
            "cumulative_pnl": sim_result.cumulative_pnl,
            "n_bars": len(sim_result.daily_pnl),
        }
        
        # Print results
        total_ret = sim_result.total_pnl / BOOKSIZE * 100
        print(f"\n  {split_name.upper()} Results:")
        print(f"  Sharpe:      {sim_result.sharpe:+.3f}")
        print(f"  Fitness:     {sim_result.fitness:.2f}")
        print(f"  Return:      {total_ret:+.2f}%")
        print(f"  Returns Ann: {sim_result.returns_ann*100:+.2f}%")
        print(f"  Max DD:      {sim_result.max_drawdown*100:.2f}%")
        print(f"  Turnover:    {sim_result.turnover:.4f}")
        print(f"  PSR:         {sim_result.psr:.3f}")
        print(f"  Bars:        {len(sim_result.daily_pnl)}")
    
    # ============================================================================
    # SUMMARY TABLE
    # ============================================================================
    print(f"\n\n{'='*80}")
    print(f"  FULL-FACTOR CONTINUOUS PORTFOLIO — SUMMARY")
    print(f"  ({len(alphas)} alphas, {UNIVERSE}, {FEES_BPS:.0f} bps, every-bar rebalance)")
    print(f"{'='*80}")
    print(f"  {'Split':<10} {'Sharpe':>8} {'Return%':>10} {'RetAnn%':>10} {'MaxDD%':>8} {'Turnover':>10} {'Fitness':>8} {'PSR':>6} {'Bars':>8}")
    print(f"  {'-'*90}")
    for split_name in ["train", "val", "test"]:
        if split_name not in results:
            continue
        r = results[split_name]
        total_ret = r["total_pnl"] / BOOKSIZE * 100
        print(f"  {split_name:<10} {r['sharpe']:+8.3f} {total_ret:+9.2f}% "
              f"{r['returns_ann']*100:+9.2f}% {r['max_drawdown']*100:7.2f}% "
              f"{r['turnover']:9.4f} {r['fitness']:7.2f} {r['psr']:5.3f} {r['n_bars']:8d}")
    
    # ============================================================================
    # CHART: Train / Val / Test cumulative PnL
    # ============================================================================
    print(f"\n  Generating chart...", flush=True)
    
    fig, axes = plt.subplots(3, 1, figsize=(20, 14),
                              gridspec_kw={'height_ratios': [3, 1.2, 1]})
    
    colors = {
        "train": "#2196F3",  # blue
        "val":   "#FF9800",  # orange  
        "test":  "#F44336",  # red
    }
    
    # ── Panel 1: Cumulative PnL (stitched across all splits) ──
    ax1 = axes[0]
    
    running_pnl_offset = 0.0
    bar_offset = 0
    split_info = []
    
    for split_name in ["train", "val", "test"]:
        if split_name not in results:
            continue
        r = results[split_name]
        cum = r["cumulative_pnl"]
        n = len(cum)
        
        # Convert to % of book
        cum_pct = cum.values / BOOKSIZE * 100 + running_pnl_offset
        x = np.arange(bar_offset, bar_offset + n)
        
        # Background
        ax1.axvspan(bar_offset, bar_offset + n, alpha=0.06, color=colors[split_name])
        
        # Line
        ax1.plot(x, cum_pct, color=colors[split_name], linewidth=2.2,
                label=f"{split_name.upper()} (SR={r['sharpe']:+.2f}, "
                      f"Ret={r['total_pnl']/BOOKSIZE*100:+.1f}%, "
                      f"DD={r['max_drawdown']*100:.1f}%)")
        
        # Drawdown fill
        peak = np.maximum.accumulate(cum_pct)
        ax1.fill_between(x, cum_pct, peak, alpha=0.15, color='red')
        
        # Split boundary
        if split_name != "train":
            ax1.axvline(x=bar_offset, color='gray', linestyle='--', linewidth=1.8, alpha=0.7)
            ypos = ax1.get_ylim()[1] if ax1.get_ylim()[1] != ax1.get_ylim()[0] else 1
            ax1.text(bar_offset + n*0.02, cum_pct.max() * 0.95, 
                    split_name.upper(), fontsize=12, fontweight='bold', 
                    color=colors[split_name], alpha=0.8)
        
        split_info.append((bar_offset, bar_offset + n, split_name, r))
        running_pnl_offset = cum_pct[-1]
        bar_offset += n
    
    ax1.set_title(f"Full-Factor Continuous Portfolio — Cumulative Return (% of Book)\n"
                  f"{len(alphas)} alphas, {UNIVERSE}, {FEES_BPS:.0f} bps fees, 5m rebalance, all symbols",
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel("Cumulative Return (%)", fontsize=12)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.25)
    ax1.axhline(0, color='black', linewidth=0.5)
    
    # ── Panel 2: Per-bar PnL distribution (rolling daily Sharpe) ──
    ax2 = axes[1]
    bar_offset = 0
    for split_name in ["train", "val", "test"]:
        if split_name not in results:
            continue
        r = results[split_name]
        daily = r["daily_pnl"]
        
        # Rolling daily Sharpe (30-day window)
        rolling_sr = daily.rolling(30 * BARS_PER_DAY, min_periods=5 * BARS_PER_DAY).apply(
            lambda x: x.mean() / x.std() * np.sqrt(BARS_PER_DAY * 365) if x.std() > 0 else 0
        )
        n = len(rolling_sr)
        x = np.arange(bar_offset, bar_offset + n)
        
        ax2.plot(x, rolling_sr.values, color=colors[split_name], linewidth=1.0, alpha=0.8)
        ax2.fill_between(x, 0, rolling_sr.values, 
                        where=rolling_sr.values > 0, alpha=0.2, color='green')
        ax2.fill_between(x, 0, rolling_sr.values,
                        where=rolling_sr.values < 0, alpha=0.2, color='red')
        
        if split_name != "train":
            ax2.axvline(x=bar_offset, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        
        bar_offset += n
    
    ax2.set_ylabel("Rolling Sharpe", fontsize=11)
    ax2.set_title("Rolling 30-Day Sharpe Ratio", fontsize=11)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.grid(True, alpha=0.25)
    
    # ── Panel 3: Daily PnL bars ──
    ax3 = axes[2]
    day_offset = 0
    for split_name in ["train", "val", "test"]:
        if split_name not in results:
            continue
        r = results[split_name]
        daily = r["daily_pnl"]
        
        # Aggregate to true daily
        n_bars_split = len(daily)
        daily_agg = []
        for d in range(0, n_bars_split, BARS_PER_DAY):
            daily_agg.append(daily.iloc[d:d+BARS_PER_DAY].sum())
        daily_agg = np.array(daily_agg)
        daily_pct = daily_agg / BOOKSIZE * 100
        
        n_d = len(daily_pct)
        x = np.arange(day_offset, day_offset + n_d)
        bar_colors = ['#4CAF50' if d > 0 else '#F44336' for d in daily_pct]
        ax3.bar(x, daily_pct, width=0.8, color=bar_colors, alpha=0.6, edgecolor='none')
        
        if split_name != "train":
            ax3.axvline(x=day_offset, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        
        day_offset += n_d
    
    ax3.set_ylabel("Daily Return (%)", fontsize=11)
    ax3.set_xlabel("Day", fontsize=11)
    ax3.set_title("Daily PnL (% of Book)", fontsize=11)
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.grid(True, alpha=0.25)
    
    plt.tight_layout()
    chart_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_factor_5m_results.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: {chart_path}")
    
    elapsed = time.time() - t0_total
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
