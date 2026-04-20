#!/usr/bin/env python
"""
plot_equity_curves_rb1.py -- Equity curves for top rb=1 VoC + QP configs (K=58)
Plots:
  1. FM h=24 + QP(tc=0.005, rb=1, to=0.05) -- best val+test balance (Val SR@5=+9.5, Test SR@5=+7.4)
  2. FM h=12 + QP(tc=0.005, rb=1, to=0.10) -- best raw test SR@5 = +8.06
Both K=58 (34 raw + 24 DB alphas), at 5bps fees.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys, gc, time
sys.path.insert(0, str(Path(__file__).parent))

from run_voc_complexity import (
    load_full_data, log, SPLITS, BOOKSIZE, MAX_WEIGHT,
    RIDGE_WARMUP, simulate, process_signal, augment_matrices_with_db_alphas
)
from run_voc_sdf import (
    prepare_panel, cross_sectional_standardize,
    generate_rff_params, compute_rff_features as compute_rff_sdf,
)
from run_voc_qp import run_fm_multihorizon, qp_optimize_aggressive

FEE_BPS = 5.0
SEEDS   = 5
OUT_FILE = Path(__file__).parent / "voc_equity_curves_rb1.png"

plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d", "axes.labelcolor": "#e6edf3",
    "text.color": "#e6edf3", "xtick.color": "#8b949e",
    "ytick.color": "#8b949e", "grid.color": "#21262d",
    "grid.linewidth": 0.6, "font.family": "sans-serif",
    "font.size": 11,
})

SPLIT_COLORS = {"train": "#1f6feb", "val": "#f0883e", "test": "#3fb950"}
CURVE_COLORS = ["#58a6ff", "#d2a8ff"]


def build_fm(std_panel, ret_np, valid_mask, horizon):
    T, N, K = std_panel.shape
    alpha_sum = np.zeros((T, N))
    for s in range(SEEDS):
        a, _ = run_fm_multihorizon(
            std_panel, ret_np, valid_mask, 500, 42 + s * 137,
            z=1e-5, fm_window=1, retrain_every=6,
            horizon=horizon, tau=0.0)
        alpha_sum += a
    return alpha_sum / SEEDS


def full_pnl(weight_df, returns_df, close_df, universe_df, fee_bps):
    all_daily, all_cum = [], []
    offset = 0.0
    for split_name, (start, end) in SPLITS.items():
        w = weight_df.loc[start:end]
        r = returns_df.loc[start:end]
        c = close_df.loc[start:end]
        u = universe_df.loc[start:end]
        sim = simulate(w, r, c, u, fees_bps=fee_bps)
        daily = sim.daily_pnl
        cum = daily.cumsum() + offset
        offset = float(cum.iloc[-1]) if len(cum) > 0 else offset
        all_daily.append(daily)
        all_cum.append(cum)
    return pd.concat(all_daily).sort_index(), pd.concat(all_cum).sort_index()


def main():
    t0 = time.time()

    # Load augmented data (K=58)
    log("Loading augmented data (K=58)...")
    matrices, universe_df, valid_tickers = load_full_data()
    aug_matrices, n_alphas = augment_matrices_with_db_alphas(matrices, universe_df)
    panel, ret_np, valid_mask, dates, tickers, _ = prepare_panel(aug_matrices, valid_tickers)
    std = cross_sectional_standardize(panel)
    std = np.nan_to_num(std, nan=0.0)
    del panel; gc.collect()
    log(f"K=58 panel ready ({time.time()-t0:.0f}s), building signals...")

    returns_df = aug_matrices["returns"]
    close_df   = aug_matrices["close"]

    configs = [
        dict(horizon=24, tcost=0.005, rebal=1, max_to=0.05,
             label="FM h=24 + QP(tc=0.005, rb=1, to=0.05)  |  K=58  |  5bps",
             color=CURVE_COLORS[0]),
        dict(horizon=12, tcost=0.005, rebal=1, max_to=0.10,
             label="FM h=12 + QP(tc=0.005, rb=1, to=0.10)  |  K=58  |  5bps",
             color=CURVE_COLORS[1]),
    ]

    results = []
    for cfg in configs:
        log(f"\nBuilding: {cfg['label']}")
        alpha = build_fm(std, ret_np, valid_mask, cfg["horizon"])
        alpha_df = pd.DataFrame(alpha, index=dates, columns=tickers)
        alpha_df = process_signal(alpha_df, universe_df=universe_df, max_wt=MAX_WEIGHT)
        qp = qp_optimize_aggressive(
            alpha_df, aug_matrices, universe_df, returns_df,
            risk_aversion=1.0, tcost_lambda=cfg["tcost"],
            lookback_bars=120, rebal_every=cfg["rebal"],
            max_turnover=cfg["max_to"])
        pnl, cum = full_pnl(qp, returns_df, close_df, universe_df, FEE_BPS)
        results.append((pnl, cum, cfg["label"], cfg["color"]))
        log(f"  Done ({time.time()-t0:.0f}s)")
        del alpha, alpha_df; gc.collect()

    # Plot
    log("\nPlotting...")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                             gridspec_kw={"hspace": 0.08})

    split_bounds = {name: (pd.Timestamp(s), pd.Timestamp(e))
                    for name, (s, e) in SPLITS.items()}

    for ax, (pnl_s, cum, label, color) in zip(axes, results):
        # Shade splits
        for sn, (s, e) in split_bounds.items():
            ax.axvspan(s, e, alpha=0.07, color=SPLIT_COLORS[sn], zorder=0)
            ax.axvline(s, color=SPLIT_COLORS[sn], lw=0.8, ls="--", alpha=0.5)

        # Equity curve per split (colored by split)
        for sn, (s, e) in split_bounds.items():
            seg = cum.loc[s:e]
            if len(seg):
                ax.plot(seg.index, seg.values / BOOKSIZE * 100,
                        color=SPLIT_COLORS[sn], lw=1.5, alpha=0.95)

        ax.axhline(0, color="#484f58", lw=0.8)

        # Split boundary labels at bottom of chart (x-axis transform = data x, axes y)
        for sn, (s, e) in split_bounds.items():
            mid = s + (e - s) / 2
            ax.text(mid, 0.02, sn.upper(),
                    ha="center", va="bottom",
                    transform=ax.get_xaxis_transform(),
                    color=SPLIT_COLORS[sn], fontsize=9,
                    fontweight="bold", alpha=0.70)

        # Stats: one line per split, placed in a box outside the right edge
        stats_lines = []
        for sn, (s, e) in split_bounds.items():
            seg_pnl = pnl_s.loc[s:e]
            seg_cum = cum.loc[s:e]
            if len(seg_pnl) < 2:
                continue
            n_days = len(seg_pnl) / 6  # 6 bars per day
            total_ret = (seg_pnl.sum() / BOOKSIZE) * 100
            sr = (seg_pnl.mean() / seg_pnl.std() * np.sqrt(2190)
                  if seg_pnl.std() > 0 else 0)
            stats_lines.append(
                (sn, f"{sn.capitalize()}: cumret={total_ret:+.1f}%  SR(ann)={sr:+.2f}  ({n_days:.0f} days)")
            )

        # Place stats below the subplot using figure text
        ax._stats_lines = stats_lines  # store for later placement

        ax.set_ylabel("Cumulative Return % (not annualized)", fontsize=9)
        ax.set_title(label, fontsize=12, color=color, loc="left", pad=6)
        ax.grid(True, axis="y", alpha=0.4)
        ax.grid(True, axis="x", alpha=0.2)

    axes[-1].set_xlabel("Date", fontsize=10)
    fig.suptitle(
        "Virtue of Complexity + QP  |  4H Crypto  |  @5bps fees  |  K=58 (every-bar rebalancing)\n"
        "FM Ridge P=500 z=1e-5, 5 seeds, QP rb=1 (every 4h bar), tcost_lambda penalizes trades",
        fontsize=12, y=1.01, color="#e6edf3"
    )

    legend_elements = [
        mpatches.Patch(color=SPLIT_COLORS["train"], alpha=0.7, label="Train (2021–Jan 2025)"),
        mpatches.Patch(color=SPLIT_COLORS["val"],   alpha=0.7, label="Val (Jan–Sep 2025)  ← primary OOS"),
        mpatches.Patch(color=SPLIT_COLORS["test"],  alpha=0.7, label="Test (Sep 2025–Mar 2026)"),
    ]
    fig.legend(handles=legend_elements, loc="upper right",
               framealpha=0.15, fontsize=9, ncol=3)

    plt.tight_layout(rect=[0, 0.06, 1, 1])  # leave bottom space for stats

    # Draw per-subplot stats below each axes
    for i, (ax, (pnl_s, cum, label, color)) in enumerate(zip(axes, results)):
        lines = getattr(ax, "_stats_lines", [])
        # Get axes position in figure coords
        pos = ax.get_position()
        x_start = pos.x0 + 0.01
        y_base  = pos.y0 - 0.005  # just below the subplot
        for j, (sn, txt) in enumerate(lines):
            fig.text(x_start + j * 0.28, y_base, txt,
                     color=SPLIT_COLORS[sn], fontsize=8.5,
                     ha="left", va="top",
                     transform=fig.transFigure)

    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    log(f"\nSaved: {OUT_FILE}")
    log(f"Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

