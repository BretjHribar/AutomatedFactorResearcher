"""
compare_rebal.py — Compare AIPT configs across P and REBAL_EVERY.

Configurations:
  1. P=100,  rebal=12  (current production)
  2. P=100,  rebal=1   (every bar)
  3. P=1000, rebal=12
  4. P=1000, rebal=1

Usage:
    python compare_rebal.py
    python compare_rebal.py --fees 5   # custom fee bps
"""

import argparse
import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aipt_kucoin import (
    load_data,
    build_characteristics_matrix,
    build_characteristics_panel,
    generate_rff_params,
    compute_rff_signals,
    compute_factor_returns,
    estimate_ridge_markowitz,
    OOS_START,
    TRAIN_BARS,
    MIN_TRAIN_BARS,
    BARS_PER_YEAR,
    RESULTS_DIR,
)

CONFIGS = [
    dict(P=100,  rebal=12, label="P=100  rebal/12",  color="#2196F3"),
    dict(P=100,  rebal=1,  label="P=100  rebal/1",   color="#64B5F6"),
    dict(P=1000, rebal=12, label="P=1000 rebal/12",  color="#FF5722"),
    dict(P=1000, rebal=1,  label="P=1000 rebal/1",   color="#FFAB91"),
]

FEE_BPS = 3.0


def run_config(P: int, rebal_every: int, z: float = 1e-3, seed: int = 42,
               fee_bps: float = FEE_BPS) -> dict:
    """
    Run the AIPT normalized L/S backtest for a single (P, rebal_every) config.
    Returns a dict with performance arrays and summary stats.
    """
    t0 = time.time()
    print(f"\n  Running P={P}, rebal={rebal_every}...", flush=True)

    matrices, universe_df, tickers = load_data()
    close = matrices["close"]
    returns_pct = matrices["returns_pct"]
    dates = close.index
    T_total = len(dates)
    N = len(tickers)

    # OOS start index
    oos_start_idx = next(
        (i for i, dt in enumerate(dates) if str(dt) >= OOS_START), None
    )
    if oos_start_idx is None:
        raise ValueError(f"OOS start {OOS_START} not found in data")

    _, char_names = build_characteristics_matrix(matrices, tickers, oos_start_idx)
    D = len(char_names)
    theta, gamma = generate_rff_params(D, P, seed=seed)

    # Build characteristics panel
    panel_start = max(0, oos_start_idx - TRAIN_BARS - 10)
    Z_panel, _ = build_characteristics_panel(matrices, tickers, panel_start, T_total)

    returns_np = returns_pct.reindex(columns=tickers).values.astype(np.float64)

    # Compute factor returns + store RFF signals for position tracking
    factor_returns = {}   # bar_idx (return date) -> F vector (P,)
    rff_signals = {}      # bar_idx (signal date) -> (S_t, valid_mask, N_t)

    for t in range(panel_start, T_total - 1):
        if t not in Z_panel:
            continue
        Z_t = Z_panel[t]
        r_t1 = returns_np[t + 1, :]
        valid = ~np.isnan(r_t1) & ~np.isnan(Z_t).any(axis=1)
        N_t = valid.sum()
        if N_t < 5:
            continue
        S_t = compute_rff_signals(Z_t[valid], theta, gamma)
        r_valid = np.nan_to_num(r_t1[valid], nan=0.0)
        factor_returns[t + 1] = compute_factor_returns(S_t, r_valid, N_t)
        rff_signals[t] = (S_t, valid, N_t)

    print(f"    Factor returns: {len(factor_returns)} bars", flush=True)

    # Rolling SDF with normalized L/S weights
    all_fr_indices = sorted(factor_returns.keys())
    oos_bar_indices = [t for t in all_fr_indices if t >= oos_start_idx]

    port_returns = []
    turnovers = []
    bar_dates = []

    lambda_hat = None
    bars_since_rebal = rebal_every  # force immediate first estimation
    prev_weights = None

    for oos_t in oos_bar_indices:
        if bars_since_rebal >= rebal_every or lambda_hat is None:
            train_indices = [
                t for t in all_fr_indices
                if t < oos_t and t >= oos_t - TRAIN_BARS
            ]
            if len(train_indices) < MIN_TRAIN_BARS:
                bars_since_rebal += 1
                continue
            F_train = np.vstack([factor_returns[t] for t in train_indices])
            lambda_hat = estimate_ridge_markowitz(F_train, z)
            bars_since_rebal = 0

        sig_bar = oos_t - 1
        if sig_bar not in rff_signals:
            bars_since_rebal += 1
            continue

        S_t, valid_mask, N_t = rff_signals[sig_bar]
        raw_w = np.zeros(N)
        raw_w[valid_mask] = (1.0 / np.sqrt(N_t)) * (S_t @ lambda_hat)
        abs_sum = np.abs(raw_w).sum()
        if abs_sum < 1e-12:
            bars_since_rebal += 1
            continue
        w_norm = raw_w / abs_sum

        r_t1 = np.nan_to_num(returns_np[oos_t, :], nan=0.0)
        port_returns.append(float(w_norm @ r_t1))
        bar_dates.append(dates[oos_t])

        if prev_weights is not None:
            turnovers.append(np.abs(w_norm - prev_weights).sum())
        else:
            turnovers.append(0.0)
        prev_weights = w_norm.copy()
        bars_since_rebal += 1

    port_arr = np.array(port_returns)
    to_arr = np.array(turnovers)

    mean_bar = port_arr.mean()
    std_bar = port_arr.std(ddof=1)
    sr_0 = (mean_bar / std_bar * np.sqrt(BARS_PER_YEAR)) if std_bar > 1e-12 else 0.0

    fee_per_bar = to_arr * fee_bps / 10_000 * 2
    net_arr = port_arr - fee_per_bar
    mean_net = net_arr.mean()
    std_net = net_arr.std(ddof=1)
    sr_net = (mean_net / std_net * np.sqrt(BARS_PER_YEAR)) if std_net > 1e-12 else 0.0

    elapsed = time.time() - t0
    print(f"    SR(0bps)={sr_0:+.3f}  SR({fee_bps:.0f}bps)={sr_net:+.3f}  "
          f"TO={to_arr.mean():.3f}  [{elapsed:.1f}s]", flush=True)

    return dict(
        P=P, rebal=rebal_every,
        bar_dates=bar_dates,
        gross=port_arr,
        net=net_arr,
        turnover=to_arr,
        sr_gross=sr_0,
        sr_net=sr_net,
        avg_to=float(to_arr.mean()),
        n_bars=len(port_arr),
    )


def plot_comparison(results: list, fee_bps: float, save_dir: Path = None):
    """Plot cumulative returns + turnover for all configs on one figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if save_dir is None:
        save_dir = RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2, 1, figsize=(15, 10), height_ratios=[3, 1], sharex=True
    )
    fig.patch.set_facecolor("#16213e")

    ax_ret, ax_to = axes
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#333")

    # Find OOS boundary from first result
    res0 = results[0]
    oos_start_idx = next(
        (i for i, d in enumerate(res0["bar_dates"]) if str(d) >= OOS_START), 0
    )
    oos_date = res0["bar_dates"][oos_start_idx] if oos_start_idx < len(res0["bar_dates"]) else res0["bar_dates"][0]

    all_dates_flat = [d for r in results for d in r["bar_dates"]]
    date_min = min(all_dates_flat)
    date_max = max(all_dates_flat)

    for cfg, res in zip(CONFIGS, results):
        dates = res["bar_dates"]
        cum_gross = np.cumsum(res["gross"])
        cum_net   = np.cumsum(res["net"])

        label = (
            f"{cfg['label']}  "
            f"SR={res['sr_gross']:+.2f}/{res['sr_net']:+.2f}  "
            f"TO={res['avg_to']:.3f}"
        )
        ax_ret.plot(dates, cum_gross, linewidth=1.8, color=cfg["color"],
                    label=label, alpha=0.9)
        ax_ret.plot(dates, cum_net, linewidth=1.0, color=cfg["color"],
                    linestyle="--", alpha=0.55)

        # Rolling 100-bar turnover
        to_series = pd.Series(res["turnover"], index=dates)
        to_roll = to_series.rolling(100, min_periods=1).mean()
        ax_to.plot(dates, to_roll.values, linewidth=1.2,
                   color=cfg["color"], alpha=0.8)

    # Train / test shading
    ax_ret.axvline(x=oos_date, color="white", linewidth=1.2,
                   linestyle="--", alpha=0.6)
    ax_ret.axvspan(date_min, oos_date, alpha=0.06, color="yellow")
    ax_ret.axvspan(oos_date, date_max, alpha=0.06, color="#00e676")

    ax_ret.set_title(
        f"AIPT Comparison: P × REBAL_EVERY  |  solid=gross  dashed={fee_bps:.0f}bps net\n"
        f"Legend: SR gross/net  |  avg turnover/bar  |  OOS from {OOS_START}",
        fontsize=12, fontweight="bold", color="white",
    )
    ax_ret.set_ylabel("Cumulative Return (sum|w|=1)", color="white")
    ax_ret.legend(
        facecolor="#1a1a2e", edgecolor="#555", labelcolor="white",
        loc="upper left", fontsize=9,
    )
    ax_ret.grid(True, alpha=0.15)

    ax_to.set_ylabel("Turnover\n(100-bar avg)", color="white")
    ax_to.set_xlabel("Date", color="white")
    ax_to.grid(True, alpha=0.15)

    # Small turnover legend
    handles = [
        matplotlib.lines.Line2D([0], [0], color=c["color"], linewidth=1.5, label=c["label"])
        for c in CONFIGS
    ]
    ax_to.legend(handles=handles, facecolor="#1a1a2e", edgecolor="#555",
                 labelcolor="white", fontsize=8, loc="upper right")

    plt.tight_layout()
    path = save_dir / "compare_rebal.png"
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    print(f"\n  Saved to {path}", flush=True)
    plt.close()

    # Also save a CSV summary
    rows = []
    for cfg, res in zip(CONFIGS, results):
        rows.append({
            "label":    cfg["label"],
            "P":        res["P"],
            "rebal":    res["rebal"],
            "sr_gross": round(res["sr_gross"], 4),
            f"sr_{fee_bps:.0f}bps": round(res["sr_net"], 4),
            "avg_turnover": round(res["avg_to"], 5),
            "n_oos_bars":   res["n_bars"],
        })
    pd.DataFrame(rows).to_csv(save_dir / "compare_rebal.csv", index=False)
    print(f"  CSV saved to {save_dir / 'compare_rebal.csv'}", flush=True)


def print_summary(results: list, fee_bps: float):
    col = f"SR({fee_bps:.0f}bps)"
    header = f"  {'Config':<22} {'SR(0bps)':>9} {col:>10} {'Avg TO':>8} {'Bars':>6}"
    print(f"\n{'='*65}")
    print(f"  COMPARISON SUMMARY — KuCoin TOP100 (4h)  OOS: {OOS_START}+")
    print(f"{'='*65}")
    print(header)
    print(f"  {'-'*58}")
    for cfg, res in zip(CONFIGS, results):
        print(f"  {cfg['label']:<22} {res['sr_gross']:>+9.3f} {res['sr_net']:>+10.3f} "
              f"{res['avg_to']:>8.4f} {res['n_bars']:>6}")
    print(f"{'='*65}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fees", type=float, default=FEE_BPS,
                        help="One-way fee in bps (default 3)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  AIPT REBAL COMPARISON  P={{100,1000}} × rebal={{1,12}}")
    print(f"  Fee: {args.fees:.1f}bps  |  OOS: {OOS_START}+")
    print(f"{'='*65}")

    results = []
    for cfg in CONFIGS:
        res = run_config(
            P=cfg["P"],
            rebal_every=cfg["rebal"],
            fee_bps=args.fees,
            seed=args.seed,
        )
        results.append(res)

    print_summary(results, args.fees)
    plot_comparison(results, args.fees)


if __name__ == "__main__":
    main()
