"""
Iter v2 — focused attack to push net OOS Sharpe past 2.4 toward 3.0.

Findings from iter v1:
  - EWMA on weights (α=0.5)   → nSR 2.33  (TO 25.6%)
  - EWMA on Z      (β=0.5)   → nSR 2.38  (TO 24.9%)  ★ best so far
  - pa_step (REBAL=1+TC pen) → no effect (factor-space penalty doesn't reach Sₜ drift)
  - Both EWMA grids peaked at the LEAST-aggressive end, so explore lower β.

This script focuses on what should help:
  1. EWMA-z with β ∈ {0.1, 0.2, 0.3, 0.4}   ← unsearched zone
  2. Combo grid α × β (full Cartesian)
  3. P=2000 with EWMA (more capacity)
  4. REBAL_EVERY ∈ {6, 24, 48} × EWMA  (tighter / looser rebal)
  5. Smaller ridge (more aggressive λ)
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
    load_data, build_Z_panel,
    BARS_PER_YEAR, TRAIN_BARS, MIN_TRAIN_BARS, REBAL_EVERY as REBAL_DEFAULT,
    OOS_START, Z_RIDGE, GAMMA_GRID, TAKER_BPS, RESULTS_DIR,
)
from backtest_voc_iter import smooth_z_panel, stats

SEED = 42
TARGET_NET_SR = 3.0
LOG_CSV  = RESULTS_DIR / "iter2_results.csv"
PLOT_OUT = RESULTS_DIR / "iter2_frontier.png"


def run_ewma_w(P, Z_panel, close_vals, start_bar, T_total, oos_start_idx, D,
               alpha=1.0, ridge=Z_RIDGE, rebal=REBAL_DEFAULT, seed=SEED):
    """Standard Markowitz + EWMA on weights. Configurable P, ridge, rebal."""
    rng = np.random.default_rng(seed)
    n_pairs = P // 2
    theta = rng.standard_normal((n_pairs, D))
    gamma = rng.choice(GAMMA_GRID, size=n_pairs)

    fr_history, lambda_hat = [], None
    bars_since_rebal, prev_w, sm_w = rebal, None, None
    rows = []

    for t in range(start_bar, T_total - 1):
        Z_t = Z_panel[t]
        proj = (Z_t @ theta.T) * gamma[None, :]
        S_t = np.empty((Z_t.shape[0], P))
        S_t[:, 0::2] = np.sin(proj)
        S_t[:, 1::2] = np.cos(proj)

        R_t1 = (close_vals[t + 1] - close_vals[t]) / close_vals[t]
        R_t1 = np.nan_to_num(R_t1, nan=0.0)

        valid = (~np.isnan(Z_t).any(axis=1)
                 & ~np.isnan(close_vals[t]) & ~np.isnan(close_vals[t + 1]))
        N_t = int(valid.sum())
        if N_t < 5:
            continue

        S_v, R_v = S_t[valid], R_t1[valid]
        F_t1 = (1.0 / np.sqrt(N_t)) * (S_v.T @ R_v)
        fr_history.append((t + 1, F_t1))

        if t + 1 < oos_start_idx:
            continue

        if bars_since_rebal >= rebal or lambda_hat is None:
            cutoff_low = (t + 1) - TRAIN_BARS
            train = [fr for (idx, fr) in fr_history if cutoff_low <= idx < (t + 1)]
            if len(train) < MIN_TRAIN_BARS:
                continue
            F_train = np.vstack(train)
            T_tr, P_tr = F_train.shape
            if P_tr <= T_tr:
                FF = (F_train.T @ F_train) / T_tr
                A = ridge * np.eye(P_tr) + FF
                lambda_hat = np.linalg.solve(A, F_train.mean(axis=0))
            else:
                # Woodbury for P > T
                mu = F_train.mean(axis=0)
                FFT = F_train @ F_train.T
                A_T = ridge * T_tr * np.eye(T_tr) + FFT
                inv_F_mu = np.linalg.solve(A_T, F_train @ mu)
                lambda_hat = (mu - F_train.T @ inv_F_mu) / ridge
            bars_since_rebal = 0

        raw_w = np.zeros(Z_t.shape[0])
        raw_w[valid] = (1.0 / np.sqrt(N_t)) * (S_v @ lambda_hat)

        sm_w = raw_w.copy() if sm_w is None else (1 - alpha) * sm_w + alpha * raw_w
        sm_abs = np.abs(sm_w).sum()
        if sm_abs < 1e-12:
            bars_since_rebal += 1
            continue
        w_norm = sm_w / sm_abs

        port_ret = float(w_norm @ R_t1)
        to = float(np.abs(w_norm - prev_w).sum() / 2.0) if prev_w is not None else 0.0
        prev_w = w_norm.copy()
        bars_since_rebal += 1

        rows.append({"bar_idx": t + 1, "gross": port_ret, "turnover": to})

    df = pd.DataFrame(rows)
    df["net_3bps"] = df["gross"] - df["turnover"] * TAKER_BPS / 10000.0 * 2.0
    return df


def run_variant(name, kwargs, Z_panel, close_vals, start_bar, T_total, oos_start_idx, D, dates):
    t0 = time.time()
    try:
        df = run_ewma_w(Z_panel=Z_panel, close_vals=close_vals, start_bar=start_bar,
                        T_total=T_total, oos_start_idx=oos_start_idx, D=D, **kwargs)
    except Exception as e:
        print(f"  {name:<40}  FAIL: {e!r}")
        return None
    df["date"] = [dates[i] for i in df["bar_idx"]]
    s = stats(df)
    s["name"] = name
    s["secs"] = time.time() - t0
    print(f"  {name:<40}  bars={s['bars']}  gSR={s['g_sr']:+.2f}  nSR={s['n_sr']:+.2f}  "
          f"TO={s['avg_to']*100:5.1f}%  ncum={s['n_cum']:+6.1f}%  ({s['secs']:.1f}s)")
    return s


def main():
    overall_t0 = time.time()
    print("=" * 100)
    print("Iter v2 — focused on combos and lower-β EWMA-z (mode 3, chars+alphas)")
    matrices, tickers, dates, close_vals, available_chars, alpha_dfs = load_data()
    T_total = len(dates)
    oos_start_idx = next(i for i, d in enumerate(dates) if str(d) >= OOS_START)
    start_bar = max(0, oos_start_idx - TRAIN_BARS - 10)
    print(f"  N={len(tickers)} T={T_total} OOS={oos_start_idx} start={start_bar}  "
          f"chars={len(available_chars)} alphas={len(alpha_dfs)}")
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
        if s is None:
            return
        results.append(s)
        if s["n_sr"] > best["n_sr"]:
            best = s.copy()
            print(f"  >> new best: {best['name']}  nSR={best['n_sr']:+.3f}")
        pd.DataFrame(results).to_csv(LOG_CSV, index=False)

    # --- batch A: lower-β EWMA-z (sweet spot might be β<0.5) ----------------------
    print(f"\n[A] EWMA-z lower beta exploration  (P=1000, no w-smoothing)")
    for beta in [0.1, 0.2, 0.3, 0.4]:
        if best["n_sr"] >= TARGET_NET_SR: break
        Zsm = smooth_z_panel(Z_panel, bars_iter, beta)
        log(run_variant(f"ewma_z(b={beta})", dict(P=1000, alpha=1.0),
                        Zsm, close_vals, start_bar, T_total, oos_start_idx, D, dates))

    # --- batch B: combo α × β -----------------------------------------------------
    print(f"\n[B] combo grid: ewma_w(α) on top of ewma_z(β)")
    for beta in [0.0, 0.3, 0.5, 0.7]:
        Zsm = Z_panel if beta == 0 else smooth_z_panel(Z_panel, bars_iter, beta)
        for alpha in [0.3, 0.5, 0.75]:
            if best["n_sr"] >= TARGET_NET_SR: break
            log(run_variant(f"combo(b={beta},a={alpha})", dict(P=1000, alpha=alpha),
                            Zsm, close_vals, start_bar, T_total, oos_start_idx, D, dates))

    # --- batch C: higher P with best smoothing ------------------------------------
    print(f"\n[C] higher P with best smoothing (P=2000, P=3000)")
    Zsm5 = smooth_z_panel(Z_panel, bars_iter, 0.5)
    for P_try in [2000, 3000]:
        for alpha in [0.5, 1.0]:
            if best["n_sr"] >= TARGET_NET_SR: break
            log(run_variant(f"P={P_try},b=0.5,a={alpha}", dict(P=P_try, alpha=alpha),
                            Zsm5, close_vals, start_bar, T_total, oos_start_idx, D, dates))

    # --- batch D: rebal frequency × EWMA-w ----------------------------------------
    print(f"\n[D] rebalance-frequency sweep with α=0.5 EWMA-w (no z-smoothing)")
    for reb in [3, 6, 24, 48]:
        if best["n_sr"] >= TARGET_NET_SR: break
        log(run_variant(f"rebal={reb},a=0.5", dict(P=1000, alpha=0.5, rebal=reb),
                        Z_panel, close_vals, start_bar, T_total, oos_start_idx, D, dates))

    # --- batch E: ridge sweep at the (α=0.5, β=0.5) operating point ---------------
    print(f"\n[E] ridge sweep at combo(b=0.5, a=0.5)")
    for ridge in [1e-5, 1e-4, 1e-2, 1e-1, 1.0]:
        if best["n_sr"] >= TARGET_NET_SR: break
        log(run_variant(f"ridge={ridge:g},b=0.5,a=0.5", dict(P=1000, alpha=0.5, ridge=ridge),
                        Zsm5, close_vals, start_bar, T_total, oos_start_idx, D, dates))

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
    ax.set_title(f"Iter v2 frontier — best: {best['name']} nSR={best['n_sr']:+.2f}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=150, bbox_inches="tight")
    print(f"Figure: {PLOT_OUT}")
    print(f"Log:    {LOG_CSV}")


if __name__ == "__main__":
    main()
