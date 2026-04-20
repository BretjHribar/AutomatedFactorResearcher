"""
run_compare_combiners.py — Compare 3 signal combination methods:
  1. PCA 3-factor hedged (ts_autonorm + signal_hedge=3)
  2. Billion Alphas (Kakushadze regression)
  3. Return-weighted (ts_autonorm + rolling_return=1440)

Uses eval_alpha_5m's exact data loading + universe + simulate pipeline.
Uses eval_portfolio_5m's sophisticated combiners for signal construction.
"""

import sys, os, time, sqlite3
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_alpha_5m as ea
import eval_portfolio_5m as ep

# ============================================================================
# CONFIGURATION
# ============================================================================

FEES_BPS    = 7.0
UNIVERSE    = "BINANCE_TOP50"
INTERVAL    = "5m"
BOOKSIZE    = ea.BOOKSIZE
BARS_PER_DAY = ea.BARS_PER_DAY
NEUTRALIZE  = ea.NEUTRALIZE
DB_PATH     = ea.DB_PATH

ea.UNIVERSE = UNIVERSE
UNIVERSE_CONFIG = ea.UNIVERSE_CONFIG
ucfg = UNIVERSE_CONFIG.get(UNIVERSE, UNIVERSE_CONFIG["BINANCE_TOP100"])
MAX_WEIGHT = ucfg["max_weight"]
ea.MAX_WEIGHT = MAX_WEIGHT

SPLITS = {
    "train": ("2025-02-01", "2026-02-01"),
    "val":   ("2026-02-01", "2026-03-01"),
    "test":  ("2026-03-01", "2026-03-27"),
}

# The 3 combiner configs to compare
COMBINERS = {
    "PCA Hedge (3 factors)": {
        "func": "ts_autonorm",
        "kwargs": {
            "gamma": 0.01,
            "ic_weighted": True,
            "signal_smooth": 12,
            "signal_hedge": 3,
            "rolling_ic": 0,
            "rolling_return": 0,
            "concordance_boost": False,
            "rank_norm": False,
            "beta_hedge": False,
        },
    },
    "Billion Alphas": {
        "func": "billion_alphas",
        "kwargs": {
            "lookback_days": 15,
            "retrain_every": 288,
            "gamma": 0.01,
            "signal_smooth": 12,
        },
    },
    "Return-Weighted": {
        "func": "ts_autonorm",
        "kwargs": {
            "gamma": 0.01,
            "ic_weighted": True,
            "signal_smooth": 12,
            "signal_hedge": 0,
            "rolling_ic": 0,
            "rolling_return": 1440,  # 5-day EMA of realized L/S returns
            "concordance_boost": False,
            "rank_norm": False,
            "beta_hedge": False,
        },
    },
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_alphas():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT a.id, a.expression, COALESCE(e.ic_mean, 0), COALESCE(e.sharpe_is, 0)
        FROM alphas a LEFT JOIN evaluations e ON e.alpha_id = a.id
        WHERE a.archived = 0 AND a.universe = ?
        ORDER BY COALESCE(e.ic_mean, 0) DESC
    """, (UNIVERSE,)).fetchall()
    conn.close()
    return rows


def load_full_data():
    """Load ALL data once, then slice per split."""
    mat_dir = Path(f"data/binance_cache/matrices/{INTERVAL}")
    uni_path = Path(f"data/binance_cache/universes/{UNIVERSE}_{INTERVAL}.parquet")

    universe_df = pd.read_parquet(uni_path)
    coverage = universe_df.sum(axis=0) / len(universe_df)
    valid_tickers = sorted(coverage[coverage > ea.COVERAGE_CUTOFF].index.tolist())

    print(f"  Loading 5m matrices ({len(valid_tickers)} tickers)...", flush=True)
    t0 = time.time()
    matrices = {}
    for fp in sorted(mat_dir.glob("*.parquet")):
        df = pd.read_parquet(fp)
        cols = [c for c in valid_tickers if c in df.columns]
        if cols:
            matrices[fp.stem] = df[cols]
    print(f"  Loaded {len(matrices)} fields in {time.time()-t0:.1f}s", flush=True)

    # Cache for eval_alpha_5m's evaluate_expression
    ea._DATA_CACHE['_full_matrices'] = matrices
    ea._DATA_CACHE['_tickers'] = valid_tickers
    ea._DATA_CACHE['_universe_df'] = universe_df

    return matrices, universe_df, valid_tickers


def slice_data(matrices, universe_df, valid_tickers, split):
    start, end = SPLITS[split]
    split_matrices = {name: df.loc[start:end] for name, df in matrices.items()}
    split_universe = universe_df[valid_tickers].loc[start:end]
    if "close" in split_matrices:
        split_matrices["returns"] = split_matrices["close"].pct_change()
    return split_matrices, split_universe


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0_total = time.time()

    alphas = load_alphas()
    print(f"\n{'='*80}")
    print(f"  COMBINER COMPARISON — Full-Factor 5m Portfolio")
    print(f"  Universe: {UNIVERSE} | Fees: {FEES_BPS} bps | Alphas: {len(alphas)}")
    print(f"{'='*80}")
    for a in alphas:
        print(f"  #{a[0]:2d} IC={a[2]:+.05f} SR={a[3]:+.2f} | {a[1][:70]}")

    if not alphas:
        print("  No alphas found!")
        return

    # Load full data once
    full_matrices, universe_df, valid_tickers = load_full_data()

    # ── Build composites for each combiner on the FULL date range ──
    # Build one composite per combiner on the full range, then slice per split
    full_start = "2025-02-01"
    full_end   = "2026-03-27"
    full_mat = {name: df.loc[full_start:full_end] for name, df in full_matrices.items()}
    full_uni = universe_df[valid_tickers].loc[full_start:full_end]
    if "close" in full_mat:
        full_mat["returns"] = full_mat["close"].pct_change()
    full_returns = full_mat.get("returns")

    composites = {}
    for cname, cfg in COMBINERS.items():
        print(f"\n  Building composite: {cname}...", flush=True)
        func_name = cfg["func"]
        kwargs = cfg["kwargs"]

        if func_name == "ts_autonorm":
            composite = ep.compute_composite_ts_autonorm(
                alphas, full_mat, full_uni, full_returns, **kwargs
            )
        elif func_name == "billion_alphas":
            composite = ep.compute_composite_billion_alphas(
                alphas, full_mat, full_uni, full_returns, **kwargs
            )
        else:
            raise ValueError(f"Unknown combiner: {func_name}")

        if composite is None:
            print(f"  FAILED: {cname}")
            continue
        composites[cname] = composite

    if not composites:
        print("  No composites built!")
        return

    # ── Run simulation for each combiner × each split ──
    all_results = {}  # {combiner_name: {split_name: VectorizedSimResult}}

    for cname, composite in composites.items():
        all_results[cname] = {}
        for split_name, (start, end) in SPLITS.items():
            # Slice composite to this split
            comp_slice = composite.loc[start:end]

            # Slice data
            split_mat, split_uni = slice_data(full_matrices, universe_df, valid_tickers, split_name)
            close = split_mat.get("close")
            returns = split_mat.get("returns")

            # Process signal using eval_alpha_5m's exact pipeline
            alpha_processed = ea.process_signal(comp_slice, universe_df=split_uni, max_wt=MAX_WEIGHT)

            # Simulate using eval_alpha_5m's exact simulate
            sim = ea.simulate(
                alpha_df=alpha_processed,
                returns_df=returns,
                close_df=close,
                universe_df=split_uni,
                fees_bps=FEES_BPS,
            )
            all_results[cname][split_name] = sim

    # ============================================================================
    # SUMMARY TABLE
    # ============================================================================
    print(f"\n\n{'='*90}")
    print(f"  COMBINER COMPARISON RESULTS")
    print(f"  ({len(alphas)} alphas, {UNIVERSE}, {FEES_BPS:.0f} bps, full-factor 5m rebalance)")
    print(f"{'='*90}")

    for cname in composites:
        print(f"\n  -- {cname} --")
        print(f"  {'Split':<8} {'Sharpe':>8} {'Ret%':>9} {'RetAnn%':>10} {'MaxDD%':>8} {'TO':>8} {'Fitness':>8} {'PSR':>6}")
        print(f"  {'-'*70}")
        for split_name in ["train", "val", "test"]:
            s = all_results[cname][split_name]
            ret_pct = s.total_pnl / BOOKSIZE * 100
            print(f"  {split_name:<8} {s.sharpe:+8.3f} {ret_pct:+8.2f}% "
                  f"{s.returns_ann*100:+9.2f}% {s.max_drawdown*100:7.2f}% "
                  f"{s.turnover:7.4f} {s.fitness:7.2f} {s.psr:5.3f}")

    # ============================================================================
    # CHART 1: Side-by-side cumulative PnL (stitched train+val+test)
    # ============================================================================
    print(f"\n  Generating charts...", flush=True)

    combiner_colors = {
        "PCA Hedge (3 factors)": "#2196F3",  # blue
        "Billion Alphas":       "#4CAF50",   # green
        "Return-Weighted":      "#FF9800",   # orange
    }
    split_bg = {
        "train": (0.92, 0.95, 1.0, 0.3),
        "val":   (1.0, 0.95, 0.88, 0.3),
        "test":  (1.0, 0.90, 0.90, 0.3),
    }

    fig, axes = plt.subplots(3, 1, figsize=(20, 16),
                              gridspec_kw={'height_ratios': [3, 1.5, 1.5]})

    # ── Panel 1: Stitched cumulative PnL across all splits ──
    ax1 = axes[0]

    # Draw split backgrounds first
    total_bars = sum(
        len(all_results[list(composites.keys())[0]][s].daily_pnl) for s in ["train", "val", "test"]
    )
    offset = 0
    for split_name in ["train", "val", "test"]:
        n = len(all_results[list(composites.keys())[0]][split_name].daily_pnl)
        ax1.axvspan(offset, offset + n, alpha=0.06,
                   color=['#2196F3', '#FF9800', '#F44336'][["train","val","test"].index(split_name)])
        if split_name != "train":
            ax1.axvline(x=offset, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        offset += n

    # Plot each combiner
    for cname in composites:
        running_offset = 0.0
        bar_off = 0
        all_x, all_y = [], []

        for split_name in ["train", "val", "test"]:
            s = all_results[cname][split_name]
            cum = s.cumulative_pnl.values / BOOKSIZE * 100 + running_offset
            n = len(cum)
            x = np.arange(bar_off, bar_off + n)
            all_x.extend(x)
            all_y.extend(cum)
            running_offset = cum[-1]
            bar_off += n

        color = combiner_colors.get(cname, '#999999')
        # Get split-level stats for legend
        sr_train = all_results[cname]["train"].sharpe
        sr_val   = all_results[cname]["val"].sharpe
        sr_test  = all_results[cname]["test"].sharpe
        ax1.plot(all_x, all_y, color=color, linewidth=2.2,
                label=f"{cname}  (Train={sr_train:+.2f} | Val={sr_val:+.2f} | Test={sr_test:+.2f})")

    # Add split labels
    offset = 0
    for split_name in ["train", "val", "test"]:
        n = len(all_results[list(composites.keys())[0]][split_name].daily_pnl)
        if split_name != "train":
            ax1.text(offset + n * 0.02, ax1.get_ylim()[1] * 0.95 if ax1.get_ylim()[1] != 0 else 5,
                    split_name.upper(), fontsize=12, fontweight='bold', alpha=0.6)
        offset += n

    ax1.set_title(f"Combiner Comparison — Full-Factor 5m Portfolio\n"
                  f"{len(alphas)} alphas, {UNIVERSE}, {FEES_BPS:.0f} bps fees, all symbols, every-bar rebalance",
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel("Cumulative Return (%)", fontsize=12)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.25)
    ax1.axhline(0, color='black', linewidth=0.5)

    # ── Panel 2: Per-combiner rolling Sharpe (30-day) ──
    ax2 = axes[1]
    for cname in composites:
        bar_off = 0
        all_x, all_sr = [], []
        for split_name in ["train", "val", "test"]:
            s = all_results[cname][split_name]
            daily = s.daily_pnl
            roll_window = min(30 * BARS_PER_DAY, len(daily) - 1)
            if roll_window < BARS_PER_DAY:
                roll_window = BARS_PER_DAY
            rolling_sr = daily.rolling(roll_window, min_periods=max(BARS_PER_DAY, 1)).apply(
                lambda x: x.mean() / x.std() * np.sqrt(BARS_PER_DAY * 365) if x.std() > 0 else 0,
                raw=True,
            )
            n = len(rolling_sr)
            x = np.arange(bar_off, bar_off + n)
            all_x.extend(x)
            all_sr.extend(rolling_sr.values)

            if split_name != "train":
                ax2.axvline(x=bar_off, color='gray', linestyle='--', linewidth=1.2, alpha=0.5)
            bar_off += n

        color = combiner_colors.get(cname, '#999999')
        ax2.plot(all_x, all_sr, color=color, linewidth=1.0, alpha=0.8, label=cname)

    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_ylabel("Rolling Sharpe", fontsize=11)
    ax2.set_title("Rolling 30-Day Sharpe Ratio", fontsize=11)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.25)

    # ── Panel 3: Per-split bar chart comparison ──
    ax3 = axes[2]
    n_combiners = len(composites)
    comb_names = list(composites.keys())
    n_splits = 3
    bar_width = 0.25
    x_pos = np.arange(n_splits)

    for i, cname in enumerate(comb_names):
        sharpes = [all_results[cname][s].sharpe for s in ["train", "val", "test"]]
        color = combiner_colors.get(cname, '#999999')
        bars = ax3.bar(x_pos + i * bar_width, sharpes, bar_width,
                      label=cname, color=color, alpha=0.8, edgecolor='white')
        # Value labels
        for j, v in enumerate(sharpes):
            ax3.text(x_pos[j] + i * bar_width, v + (0.15 if v >= 0 else -0.35),
                    f"{v:+.2f}", ha='center', va='bottom' if v >= 0 else 'top',
                    fontsize=9, fontweight='bold')

    ax3.set_xticks(x_pos + bar_width)
    ax3.set_xticklabels(["TRAIN", "VAL", "TEST"], fontsize=12, fontweight='bold')
    ax3.set_ylabel("Sharpe Ratio", fontsize=11)
    ax3.set_title("Sharpe by Split × Combiner", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.grid(True, alpha=0.25, axis='y')

    plt.tight_layout()
    chart_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "combiner_comparison_5m.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: {chart_path}")

    # ============================================================================
    # CHART 2: Per-split detailed view (3 subplots, one per split)
    # ============================================================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(22, 6), sharey=True)

    for idx, split_name in enumerate(["train", "val", "test"]):
        ax = axes2[idx]
        start, end = SPLITS[split_name]
        for cname in composites:
            s = all_results[cname][split_name]
            cum = s.cumulative_pnl.values / BOOKSIZE * 100
            x = np.arange(len(cum))
            color = combiner_colors.get(cname, '#999999')
            ret_pct = s.total_pnl / BOOKSIZE * 100
            ax.plot(x, cum, color=color, linewidth=1.8,
                   label=f"{cname}\nSR={s.sharpe:+.2f} Ret={ret_pct:+.1f}%")
        ax.set_title(f"{split_name.upper()}\n({start} → {end})", fontsize=12, fontweight='bold')
        ax.set_xlabel("Bar", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Cumulative Return (%)", fontsize=11)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.25)
        ax.axhline(0, color='black', linewidth=0.5)

    fig2.suptitle(f"Combiner Comparison — Per-Split Detail  ({len(alphas)} alphas, {FEES_BPS:.0f} bps)",
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    chart2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "combiner_comparison_splits.png")
    plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: {chart2_path}")

    elapsed = time.time() - t0_total
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
