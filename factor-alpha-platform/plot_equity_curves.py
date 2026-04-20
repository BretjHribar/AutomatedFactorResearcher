#!/usr/bin/env python
"""
plot_equity_curves.py -- Equity curves for top VoC + QP configs
Plots:
  1. FM h=6 + QP(tc=0.005, rb=12, to=0.10) -- K=34 winner (test SR@5 = +3.77)
  2. FM h=12 + QP(tc=0.005, rb=12, to=0.10) -- K=58 winner (test SR@5 = +4.45)
Both at 5bps fees. Includes train/val/test split markers.
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
from run_voc_qp import (
    run_fm_multihorizon, qp_optimize_aggressive, REPORT_FILE
)

FEE_BPS    = 5.0
SEEDS      = 5
OUT_FILE   = Path(__file__).parent / "voc_equity_curves.png"

# ── plot style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d", "axes.labelcolor": "#e6edf3",
    "text.color": "#e6edf3", "xtick.color": "#8b949e",
    "ytick.color": "#8b949e", "grid.color": "#21262d",
    "grid.linewidth": 0.6, "font.family": "sans-serif",
    "font.size": 11,
})

SPLIT_COLORS = {"train": "#1f6feb", "val": "#f0883e", "test": "#3fb950"}
CURVE_COLORS = ["#58a6ff", "#d2a8ff"]  # blue for K=34, purple for K=58


def build_fm_signal(std_panel, ret_np, valid_mask, horizon, seeds, z=1e-5):
    T, N, K = std_panel.shape
    alpha_sum = np.zeros((T, N))
    for s in range(seeds):
        seed = 42 + s * 137
        a, _ = run_fm_multihorizon(
            std_panel, ret_np, valid_mask, 500, seed,
            z=z, fm_window=1, retrain_every=6, horizon=horizon, tau=0.0)
        alpha_sum += a
    return alpha_sum / seeds


def apply_qp(alpha_df, matrices, universe_df, returns_df,
             tcost=0.005, rebal=12, max_to=0.10):
    return qp_optimize_aggressive(
        alpha_df, matrices, universe_df, returns_df,
        risk_aversion=1.0, tcost_lambda=tcost,
        lookback_bars=120, rebal_every=rebal, max_turnover=max_to)


def full_pnl_series(weight_df, returns_df, close_df, universe_df, fee_bps):
    """Return daily PnL and cumulative PnL across ALL splits concatenated."""
    all_daily = []
    all_cumulative = []
    offset = 0.0
    for split_name, (start, end) in SPLITS.items():
        w = weight_df.loc[start:end]
        r = returns_df.loc[start:end]
        u = universe_df.loc[start:end]
        c = close_df.loc[start:end]
        sim = simulate(w, r, c, u, fees_bps=fee_bps)
        daily = sim.daily_pnl
        # Offset cumulative so it's continuous across splits
        cum = sim.daily_pnl.cumsum() + offset
        offset = float(cum.iloc[-1]) if len(cum) > 0 else offset
        all_daily.append(daily)
        all_cumulative.append(cum)
    daily_all  = pd.concat(all_daily).sort_index()
    cum_all    = pd.concat(all_cumulative).sort_index()
    return daily_all, cum_all


def cumulative_return(pnl_series, booksize=BOOKSIZE):
    return (pnl_series / booksize).cumsum() * 100  # percent


def main():
    t0 = time.time()
    log("Loading data (K=34)...")
    matrices, universe_df, valid_tickers = load_full_data()
    panel_raw, ret_np, valid_mask, dates, tickers, _ = prepare_panel(matrices, valid_tickers)
    std_raw = cross_sectional_standardize(panel_raw)
    std_raw = np.nan_to_num(std_raw, nan=0.0)
    del panel_raw; gc.collect()
    log(f"  K=34 panel ready ({time.time()-t0:.0f}s)")

    returns_df = matrices["returns"]
    close_df   = matrices["close"]

    # ── Config 1: K=34, FM h=6 + QP ─────────────────────────────────────────
    log("\nBuilding Config 1: K=34, FM h=6 + QP(tc=0.005, rb=12, to=0.10)...")
    alpha1 = build_fm_signal(std_raw, ret_np, valid_mask, horizon=6, seeds=SEEDS)
    alpha1_df = pd.DataFrame(alpha1, index=dates, columns=tickers)
    alpha1_df = process_signal(alpha1_df, universe_df=universe_df, max_wt=MAX_WEIGHT)
    qp1 = apply_qp(alpha1_df, matrices, universe_df, returns_df,
                   tcost=0.005, rebal=12, max_to=0.10)
    pnl1, cum1 = full_pnl_series(qp1, returns_df, close_df, universe_df, FEE_BPS)
    log(f"  Config 1 done ({time.time()-t0:.0f}s)")
    del alpha1, std_raw; gc.collect()

    # ── Load augmented data (K=58) ────────────────────────────────────────────
    log("\nLoading augmented data (K=58)...")
    aug_matrices, n_alphas = augment_matrices_with_db_alphas(matrices, universe_df)
    panel_aug, ret_np2, valid_mask2, dates2, tickers2, _ = prepare_panel(aug_matrices, valid_tickers)
    std_aug = cross_sectional_standardize(panel_aug)
    std_aug = np.nan_to_num(std_aug, nan=0.0)
    del panel_aug; gc.collect()
    log(f"  K=58 panel ready ({time.time()-t0:.0f}s)")

    # ── Config 2: K=58, FM h=12 + QP ─────────────────────────────────────────
    log("\nBuilding Config 2: K=58, FM h=12 + QP(tc=0.005, rb=12, to=0.10)...")
    alpha2 = build_fm_signal(std_aug, ret_np2, valid_mask2, horizon=12, seeds=SEEDS)
    alpha2_df = pd.DataFrame(alpha2, index=dates2, columns=tickers2)
    alpha2_df = process_signal(alpha2_df, universe_df=universe_df, max_wt=MAX_WEIGHT)
    qp2 = apply_qp(alpha2_df, aug_matrices, universe_df,
                   aug_matrices["returns"], tcost=0.005, rebal=12, max_to=0.10)
    pnl2, cum2 = full_pnl_series(qp2, aug_matrices["returns"], aug_matrices["close"],
                            universe_df, FEE_BPS)
    log(f"  Config 2 done ({time.time()-t0:.0f}s)")

    # ── Plot ──────────────────────────────────────────────────────────────────
    log("\nPlotting...")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                             gridspec_kw={"hspace": 0.08})

    configs = [
        (cum1, pnl1, "FM h=6 + QP  |  K=34  |  5bps", CURVE_COLORS[0]),
        (cum2, pnl2, "FM h=12 + QP  |  K=58 (+24 alphas)  |  5bps", CURVE_COLORS[1]),
    ]

    # Split boundary dates
    split_bounds = {}
    for name, (start, end) in SPLITS.items():
        split_bounds[name] = (pd.Timestamp(start), pd.Timestamp(end))

    for ax, (cum, pnl_s, label, color) in zip(axes, configs):
        # Shade splits and add labels (after plotting)
        for split_name, (s, e) in split_bounds.items():
            ax.axvspan(s, e, alpha=0.07, color=SPLIT_COLORS[split_name], zorder=0)
            ax.axvline(s, color=SPLIT_COLORS[split_name], lw=0.8, ls="--", alpha=0.5)

        # Equity curve — color by split
        for split_name, (s, e) in split_bounds.items():
            seg = cum.loc[s:e]
            if len(seg) == 0:
                continue
            ax.plot(seg.index, seg.values,
                    color=SPLIT_COLORS[split_name], lw=1.5, alpha=0.9)

        # Zero line
        ax.axhline(0, color="#484f58", lw=0.8)

        # Stats annotation per split
        y_pos = 0.97
        for split_name, (s, e) in split_bounds.items():
            seg_cum = cum.loc[s:e]
            seg_pnl = pnl_s.loc[s:e]
            if len(seg_cum) < 2:
                continue
            total_ret = (seg_pnl.sum() / BOOKSIZE) * 100
            n_bars = len(seg_pnl)
            ann_factor = 2190 / n_bars  # 6 bars/day * 365 = 2190 bars/year
            sharpe_seg = (seg_pnl.mean() / seg_pnl.std() * np.sqrt(2190)) if seg_pnl.std() > 0 else 0
            ax.annotate(
                f"{split_name}: {total_ret:+.1f}%  SR={sharpe_seg:+.2f}",
                xy=(1.0, y_pos), xycoords="axes fraction",
                ha="right", va="top", fontsize=8.5,
                color=SPLIT_COLORS[split_name],
            )
            y_pos -= 0.07

        ax.set_ylabel("Cumulative Return (%)", fontsize=10)
        ax.set_title(label, fontsize=12, color=color, loc="left", pad=6)
        ax.grid(True, axis="y", alpha=0.4)
        ax.grid(True, axis="x", alpha=0.2)

        # Split labels after ylim is known
        yrange = ax.get_ylim()
        ymax_label = yrange[0] + (yrange[1] - yrange[0]) * 0.97
        for split_name, (s, e) in split_bounds.items():
            mid = s + (e - s) / 2
            ax.text(mid, ymax_label, split_name.upper(),
                    ha="center", va="top",
                    color=SPLIT_COLORS[split_name], fontsize=9, alpha=0.65,
                    fontweight="bold")

    axes[-1].set_xlabel("Date", fontsize=10)
    fig.suptitle(
        "Virtue of Complexity + QP  |  4H Crypto  |  @5bps fees\n"
        "Fama-MacBeth Ridge (P=500, z=1e-5) + QP(tcost=0.005, rb=12, max_to=0.10)",
        fontsize=13, y=1.01, color="#e6edf3"
    )

    # Legend
    legend_elements = [
        mpatches.Patch(color=SPLIT_COLORS["train"], alpha=0.7, label="Train"),
        mpatches.Patch(color=SPLIT_COLORS["val"],   alpha=0.7, label="Val (1st OOS)"),
        mpatches.Patch(color=SPLIT_COLORS["test"],  alpha=0.7, label="Test (2nd OOS)"),
    ]
    fig.legend(handles=legend_elements, loc="upper right",
               framealpha=0.15, fontsize=9, ncol=3)

    plt.tight_layout()
    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    log(f"\nSaved to: {OUT_FILE}")
    log(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
