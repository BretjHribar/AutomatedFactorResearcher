"""Re-run the best equities config (TOP2000 P=2000, default ridge, no smoothing)
and plot equity curve."""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from backtest_voc_equities import (
    BARS_PER_YEAR, TRAIN_BARS, OOS_START, RESULTS_DIR, build_Z_panel,
)
from backtest_voc_equities_sweep import load_data_universe
from backtest_voc_equities_top2000 import run_full

# ── Best config from sweep ───────────────────────────────────────────────────
UNIVERSE = "TOP2000"
P        = 2000
ALPHA    = 1.0       # no EWMA smoothing
RIDGE    = 1e-3
SEED     = 42
TAKER_BPS = 1.0


def main():
    print(f"Loading {UNIVERSE}...", flush=True)
    matrices, tickers, dates, close_vals, chars = load_data_universe(UNIVERSE)
    T_total = len(dates)
    oos_start_idx = next(i for i, d in enumerate(dates) if str(d) >= OOS_START)
    start_bar = max(1, oos_start_idx - TRAIN_BARS - 10)
    print(f"  N={len(tickers)} T={T_total} OOS_start={oos_start_idx} chars={len(chars)}", flush=True)

    print("Building Z panel...", flush=True)
    t1 = time.time()
    Z_panel, D = build_Z_panel(matrices, tickers, chars, start_bar, T_total, delay=1)
    print(f"  D={D} built in {time.time()-t1:.1f}s", flush=True)

    print(f"Running AIPT P={P} (no smoothing, ρ={RIDGE}, seed={SEED})...", flush=True)
    t2 = time.time()
    df = run_full(P, Z_panel, close_vals, start_bar, T_total, oos_start_idx, D,
                  alpha=ALPHA, ridge=RIDGE, seed=SEED)
    df["date"] = [dates[i] for i in df["bar_idx"]]
    print(f"  Done in {time.time()-t2:.1f}s, {len(df)} bars", flush=True)

    # Stats — full OOS
    df_oos = df[df["bar_idx"] >= oos_start_idx].reset_index(drop=True)
    df_train = df[df["bar_idx"] < oos_start_idx].reset_index(drop=True)
    n_oos = len(df_oos)
    split = n_oos // 2
    df_val = df_oos.iloc[:split]
    df_test = df_oos.iloc[split:]

    ann = np.sqrt(BARS_PER_YEAR)
    def stats_label(sub, tag):
        if len(sub) < 30: return ""
        g, nn = sub["gross"].values, sub["net_1bps"].values
        sr_g = g.mean() / g.std(ddof=1) * ann
        sr_n = nn.mean() / nn.std(ddof=1) * ann
        ic = sub["ic_p"].mean()
        return (f"{tag}: gSR={sr_g:+.2f}  nSR={sr_n:+.2f}  "
                f"IC={ic:+.4f}  TO={sub['turnover'].mean()*100:.1f}%  ncum={nn.sum()*100:+.1f}%")

    print(f"\n{stats_label(df, 'FULL  ')}", flush=True)
    print(stats_label(df_train, 'TRAIN '), flush=True)
    print(stats_label(df_val,   'VAL   '), flush=True)
    print(stats_label(df_test,  'TEST  '), flush=True)

    # Save CSV
    out_csv = RESULTS_DIR / "voc_equities_top2000_P2000_best.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nCSV: {out_csv}", flush=True)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))

    train_mask = df["bar_idx"] < oos_start_idx
    oos_mask = df["bar_idx"] >= oos_start_idx
    val_end_bar = df_oos["bar_idx"].iloc[split]

    # Cumulative net (lighter color for train, darker for OOS)
    cum_gross = df["gross"].cumsum() * 100
    cum_net = df["net_1bps"].cumsum() * 100

    # TRAIN (rolling-window walk-forward, but still pre-OOS-cut)
    ax.plot(df.loc[train_mask, "date"], cum_gross[train_mask],
            color="tab:blue", alpha=0.4, linewidth=1.4, label="TRAIN gross")
    ax.plot(df.loc[train_mask, "date"], cum_net[train_mask],
            color="tab:orange", alpha=0.4, linewidth=1.4, label="TRAIN net (1 bp)")

    # OOS — VAL
    val_dates = df["date"][df["bar_idx"].between(oos_start_idx, val_end_bar - 1)]
    val_gross = cum_gross[df["bar_idx"].between(oos_start_idx, val_end_bar - 1)]
    val_net = cum_net[df["bar_idx"].between(oos_start_idx, val_end_bar - 1)]
    ax.plot(val_dates, val_gross, color="tab:blue", linewidth=2, label="VAL gross")
    ax.plot(val_dates, val_net, color="tab:orange", linewidth=2, label="VAL net (1 bp)")

    # OOS — TEST
    test_mask = df["bar_idx"] >= val_end_bar
    ax.plot(df.loc[test_mask, "date"], cum_gross[test_mask],
            color="tab:green", linewidth=2.2, label="TEST gross")
    ax.plot(df.loc[test_mask, "date"], cum_net[test_mask],
            color="tab:red", linewidth=2.2, label="TEST net (1 bp)")

    ax.axvline(dates[oos_start_idx], color="black", linestyle="--", alpha=0.5,
               label=f"OOS start ({OOS_START})")
    ax.axvline(dates[val_end_bar], color="black", linestyle=":", alpha=0.5,
               label="VAL/TEST split")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.grid(alpha=0.3)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title(
        f"Best equities config: {UNIVERSE} D={D} chars + RFF P={P}, "
        f"ρ={RIDGE}, no smoothing  |  fees={TAKER_BPS}bp aggregate",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out_png = RESULTS_DIR / "voc_equities_top2000_best_curve.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"PNG: {out_png}")


if __name__ == "__main__":
    main()
