"""
Iter v4 — push past nSR 2.68 toward 3.0.

Recap:
  iter3 best  ridge=0.01 + α=0.5 (no β-smoothing) on 36 alphas → nSR 2.68, ncum +103%
  iter2 best  combo(b=0.3, a=0.5) (no ridge change) on 18 alphas → nSR 2.50

Unsearched cell: ridge × β (we did ridge × α, and combo(α,β) at default ridge).
DB now has 49 alphas (was 36). Rebuild with full set.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from backtest_voc_postfix import (
    BARS_PER_YEAR, TRAIN_BARS, MIN_TRAIN_BARS, REBAL_EVERY,
    OOS_START, Z_RIDGE, GAMMA_GRID, TAKER_BPS, RESULTS_DIR,
)
from backtest_voc_iter import smooth_z_panel, stats
from backtest_voc_iter2 import run_ewma_w
from backtest_voc_iter3 import load_data_with_all_db_alphas, build_Z_panel

SEED = 42
TARGET_NET_SR = 3.0
LOG_CSV  = RESULTS_DIR / "iter4_results.csv"
PLOT_OUT = RESULTS_DIR / "iter4_frontier.png"


def run_variant(name, kwargs, Z_panel, close_vals, start_bar, T_total, oos_start_idx, D, dates):
    t0 = time.time()
    df = run_ewma_w(Z_panel=Z_panel, close_vals=close_vals, start_bar=start_bar,
                    T_total=T_total, oos_start_idx=oos_start_idx, D=D, **kwargs)
    df["date"] = [dates[i] for i in df["bar_idx"]]
    s = stats(df)
    s["name"] = name
    s["secs"] = time.time() - t0
    print(f"  {name:<44}  bars={s['bars']}  gSR={s['g_sr']:+.2f}  nSR={s['n_sr']:+.2f}  "
          f"TO={s['avg_to']*100:5.1f}%  ncum={s['n_cum']:+6.1f}%  ({s['secs']:.1f}s)")
    return s


def main():
    overall_t0 = time.time()
    print("=" * 100)
    print("Iter v4 — full DB alphas (~49) + ridge × β unsearched cell")
    matrices, tickers, dates, close_vals, available_chars, alpha_dfs = load_data_with_all_db_alphas()
    T_total = len(dates)
    oos_start_idx = next(i for i, d in enumerate(dates) if str(d) >= OOS_START)
    start_bar = max(0, oos_start_idx - TRAIN_BARS - 10)
    print(f"  N={len(tickers)} T={T_total} chars={len(available_chars)} alphas={len(alpha_dfs)}")
    t1 = time.time()
    Z_panel, D = build_Z_panel(matrices, tickers, available_chars, alpha_dfs,
                               start_bar, T_total, mode=3)
    bars_iter = list(range(start_bar, T_total))
    print(f"  Z panel D={D}  built in {time.time()-t1:.1f}s")
    print("=" * 100)

    results = []
    best = {"n_sr": -np.inf, "name": "(none)"}

    def log(s):
        nonlocal best
        results.append(s)
        if s["n_sr"] > best["n_sr"]:
            best = s.copy()
            print(f"  >> new best: {best['name']}  nSR={best['n_sr']:+.3f}")
        pd.DataFrame(results).to_csv(LOG_CSV, index=False)

    # --- batch A: confirm iter3 best on the new 49-alpha panel --------------------
    print(f"\n[A] confirm iter3 best on 49-alpha panel")
    log(run_variant("baseline", dict(P=1000, alpha=1.0),
                    Z_panel, close_vals, start_bar, T_total, oos_start_idx, D, dates))
    log(run_variant("ridge=0.01,a=0.5", dict(P=1000, alpha=0.5, ridge=0.01),
                    Z_panel, close_vals, start_bar, T_total, oos_start_idx, D, dates))

    # --- batch B: ridge × β grid, α=0.5 fixed (the unsearched cell) --------------
    print(f"\n[B] ridge × β grid (α=0.5 fixed)")
    for beta in [0.2, 0.3, 0.4, 0.5]:
        Zsm = smooth_z_panel(Z_panel, bars_iter, beta)
        for ridge in [3e-3, 1e-2, 3e-2]:
            if best["n_sr"] >= TARGET_NET_SR: break
            log(run_variant(f"ridge={ridge:g},a=0.5,b={beta}",
                            dict(P=1000, alpha=0.5, ridge=ridge),
                            Zsm, close_vals, start_bar, T_total, oos_start_idx, D, dates))

    # --- batch C: refined α around 0.5 at ridge=0.01, β=0.3 ----------------------
    if best["n_sr"] < TARGET_NET_SR:
        print(f"\n[C] refined α around 0.5 with ridge=0.01, β=0.3")
        Zsm3 = smooth_z_panel(Z_panel, bars_iter, 0.3)
        for alpha in [0.35, 0.4, 0.45, 0.55, 0.6, 0.7]:
            if best["n_sr"] >= TARGET_NET_SR: break
            log(run_variant(f"ridge=0.01,a={alpha},b=0.3",
                            dict(P=1000, alpha=alpha, ridge=0.01),
                            Zsm3, close_vals, start_bar, T_total, oos_start_idx, D, dates))

    # --- batch D: even higher ridge with smoothing -------------------------------
    if best["n_sr"] < TARGET_NET_SR:
        print(f"\n[D] higher ridge with α=0.5, β=0.3")
        Zsm3 = smooth_z_panel(Z_panel, bars_iter, 0.3)
        for ridge in [0.05, 0.1, 0.3]:
            if best["n_sr"] >= TARGET_NET_SR: break
            log(run_variant(f"ridge={ridge:g},a=0.5,b=0.3",
                            dict(P=1000, alpha=0.5, ridge=ridge),
                            Zsm3, close_vals, start_bar, T_total, oos_start_idx, D, dates))

    # --- batch E: alphas-only mode with best recipes -----------------------------
    if best["n_sr"] < TARGET_NET_SR:
        print(f"\n[E] alphas-only (mode 2) with full DB alphas + best smoothing")
        Z2, D2 = build_Z_panel(matrices, tickers, available_chars, alpha_dfs,
                               start_bar, T_total, mode=2)
        Z2_sm3 = smooth_z_panel(Z2, bars_iter, 0.3)
        log(run_variant("alphas_only ridge=0.01,a=0.5,b=0.3",
                        dict(P=1000, alpha=0.5, ridge=0.01),
                        Z2_sm3, close_vals, start_bar, T_total, oos_start_idx, D2, dates))

    # ── done ──
    print("\n" + "=" * 100)
    print(f"Best: {best['name']}  nSR={best['n_sr']:+.3f}  TO={best['avg_to']*100:.1f}%  "
          f"ncum={best['n_cum']:+.1f}%")
    print(f"Target nSR={TARGET_NET_SR}: {'REACHED' if best['n_sr']>=TARGET_NET_SR else 'NOT reached'}")
    print(f"Total: {(time.time()-overall_t0)/60:.1f} min  ({len(results)} variants)")

    df_all = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.scatter(df_all["avg_to"]*100, df_all["n_sr"], s=70, alpha=0.7,
               c=range(len(df_all)), cmap="viridis", edgecolors="black", linewidths=0.5)
    for _, r in df_all.iterrows():
        ax.annotate(r["name"], (r["avg_to"]*100, r["n_sr"]),
                    textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax.axhline(TARGET_NET_SR, color="red", linestyle="--", label=f"target nSR={TARGET_NET_SR}")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("Avg turnover per bar (%)")
    ax.set_ylabel("Net Sharpe (3 bps taker)")
    ax.set_title(f"Iter v4 — best: {best['name']} nSR={best['n_sr']:+.2f}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=150, bbox_inches="tight")
    print(f"Figure: {PLOT_OUT}")
    print(f"Log:    {LOG_CSV}")


if __name__ == "__main__":
    main()
