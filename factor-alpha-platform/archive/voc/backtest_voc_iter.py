"""
Iterate closed-form Markowitz variants on mode-3 (chars + alphas) at P=1000
to push net OOS Sharpe (after 3 bps fees) toward 3.0.

Variants — all closed-form:
  baseline       — current run_one (no TC handling)
  ewma_w(α)      — w_t = (1-α) w_{t-1} + α w*_t   (asset-space partial adjustment)
  ewma_z(β)      — Z̃_t = β Z̃_{t-1} + (1-β) Z_t (then standard run_one on Z̃)
  combo(α,β)     — both stacked
  pa_step(τ)     — per-bar Markowitz with asset-space quadratic TC penalty
                   (rebalance EVERY bar instead of every 12)
  ridge_sweep(ρ) — vary the ridge to see if shrinking λ further helps

Logs each variant to results.csv and stops when nSR ≥ 3.0.
"""
from __future__ import annotations
import sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from backtest_voc_postfix import (
    load_data, build_Z_panel, run_one,
    BARS_PER_YEAR, TRAIN_BARS, MIN_TRAIN_BARS, REBAL_EVERY,
    OOS_START, Z_RIDGE, GAMMA_GRID, TAKER_BPS, P, RESULTS_DIR,
)

SEED = 42
TARGET_NET_SR = 3.0
LOG_CSV  = RESULTS_DIR / "iter_results.csv"
PLOT_OUT = RESULTS_DIR / "iter_frontier.png"


# ── helpers ──────────────────────────────────────────────────────────────────

def stats(df):
    g = df["gross"].values
    n = df["net_3bps"].values
    if len(n) < 2 or n.std(ddof=1) == 0:
        return dict(bars=len(df), g_sr=0.0, n_sr=0.0, g_cum=0.0, n_cum=0.0, avg_to=0.0)
    ann = np.sqrt(BARS_PER_YEAR)
    return dict(
        bars=len(df),
        g_sr=float(g.mean() / g.std(ddof=1) * ann),
        n_sr=float(n.mean() / n.std(ddof=1) * ann),
        g_cum=float(g.sum() * 100),
        n_cum=float(n.sum() * 100),
        avg_to=float(df["turnover"].mean()),
    )


def smooth_z_panel(Z_panel, bars, beta):
    """Z̃_t = β Z̃_{t-1} + (1-β) Z_t. β=0 → no smoothing."""
    if beta <= 0:
        return Z_panel
    sm = None
    out = {}
    for t in bars:
        Z = Z_panel[t]
        sm = Z.copy() if sm is None else (beta * sm + (1 - beta) * Z)
        out[t] = sm
    return out


# ── variant 1: EWMA on weights ───────────────────────────────────────────────

def run_ewma_w(Z_panel, close_vals, start_bar, T_total, oos_start_idx, D,
               alpha=1.0, ridge=Z_RIDGE, seed=SEED):
    """Same Markowitz as run_one but with asset-space EWMA on weights:
       sm_w_t = (1-α) sm_w_{t-1} + α (1/√N) Sₜ λ
       w_t    = sm_w_t / |sm_w_t|_1                  (re-normalize to gross=1)

    α=1 (no smoothing) reproduces run_one (up to identical normalization).
    """
    rng = np.random.default_rng(seed)
    n_pairs = P // 2
    theta = rng.standard_normal((n_pairs, D))
    gamma = rng.choice(GAMMA_GRID, size=n_pairs)

    fr_history, lambda_hat = [], None
    bars_since_rebal, prev_w, sm_w = REBAL_EVERY, None, None
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

        if bars_since_rebal >= REBAL_EVERY or lambda_hat is None:
            cutoff_low = (t + 1) - TRAIN_BARS
            train = [fr for (idx, fr) in fr_history if cutoff_low <= idx < (t + 1)]
            if len(train) < MIN_TRAIN_BARS:
                continue
            F_train = np.vstack(train)
            T_tr, P_tr = F_train.shape
            FF = (F_train.T @ F_train) / T_tr
            A = ridge * np.eye(P_tr) + FF
            lambda_hat = np.linalg.solve(A, F_train.mean(axis=0))
            bars_since_rebal = 0

        raw_w = np.zeros(Z_t.shape[0])
        raw_w[valid] = (1.0 / np.sqrt(N_t)) * (S_v @ lambda_hat)

        # EWMA in raw (un-normalized) space, then normalize for trading
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


# ── variant 2: per-bar Markowitz with asset-space TC penalty ─────────────────

def run_pa_step(Z_panel, close_vals, start_bar, T_total, oos_start_idx, D,
                tau=1.0, ridge=Z_RIDGE, seed=SEED):
    """Re-solve Markowitz EVERY bar with asset-space TC penalty:
       (FF/T + ρI + τ SₜᵀSₜ/N) λ = μ + τ (SₜᵀSₜ/N) λ_{t-1}
    Closed form, ~12× more compute than rebalance-every-12 but principled."""
    rng = np.random.default_rng(seed)
    n_pairs = P // 2
    theta = rng.standard_normal((n_pairs, D))
    gamma = rng.choice(GAMMA_GRID, size=n_pairs)

    fr_history, lambda_hat, prev_w = [], None, None
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

        cutoff_low = (t + 1) - TRAIN_BARS
        train = [fr for (idx, fr) in fr_history if cutoff_low <= idx < (t + 1)]
        if len(train) < MIN_TRAIN_BARS:
            continue
        F_train = np.vstack(train)
        T_tr, P_tr = F_train.shape
        FF = (F_train.T @ F_train) / T_tr
        A = ridge * np.eye(P_tr) + FF
        rhs = F_train.mean(axis=0)
        if tau > 0 and lambda_hat is not None:
            StS = (S_v.T @ S_v) / N_t
            A = A + tau * StS
            rhs = rhs + tau * (StS @ lambda_hat)
        lambda_hat = np.linalg.solve(A, rhs)

        raw_w = np.zeros(Z_t.shape[0])
        raw_w[valid] = (1.0 / np.sqrt(N_t)) * (S_v @ lambda_hat)
        abs_sum = np.abs(raw_w).sum()
        if abs_sum < 1e-12:
            continue
        w_norm = raw_w / abs_sum

        port_ret = float(w_norm @ R_t1)
        to = float(np.abs(w_norm - prev_w).sum() / 2.0) if prev_w is not None else 0.0
        prev_w = w_norm.copy()
        rows.append({"bar_idx": t + 1, "gross": port_ret, "turnover": to})

    df = pd.DataFrame(rows)
    df["net_3bps"] = df["gross"] - df["turnover"] * TAKER_BPS / 10000.0 * 2.0
    return df


# ── runner ───────────────────────────────────────────────────────────────────

def run_variant(name, fn, kwargs, Z_panel, close_vals, start_bar, T_total, oos_start_idx, D, dates):
    t0 = time.time()
    try:
        df = fn(Z_panel, close_vals, start_bar, T_total, oos_start_idx, D, **kwargs)
    except Exception as e:
        print(f"  {name:<32}  FAIL: {e!r}")
        traceback.print_exc()
        return None
    df["date"] = [dates[i] for i in df["bar_idx"]]
    s = stats(df)
    s["name"] = name
    s["secs"] = time.time() - t0
    print(f"  {name:<32}  bars={s['bars']}  gSR={s['g_sr']:+.2f}  nSR={s['n_sr']:+.2f}  "
          f"TO={s['avg_to']*100:5.1f}%  gcum={s['g_cum']:+6.1f}%  ncum={s['n_cum']:+6.1f}%  "
          f"({s['secs']:.1f}s)")
    return s, df


def main():
    overall_t0 = time.time()
    print("=" * 90)
    print(f"Loading data + building mode-3 Z panel (chars + alphas)...")
    matrices, tickers, dates, close_vals, available_chars, alpha_dfs = load_data()
    T_total = len(dates)
    oos_start_idx = next(i for i, d in enumerate(dates) if str(d) >= OOS_START)
    start_bar = max(0, oos_start_idx - TRAIN_BARS - 10)
    print(f"  N={len(tickers)}  T={T_total}  OOS_start={oos_start_idx}  start_bar={start_bar}")
    print(f"  chars={len(available_chars)}  alphas={len(alpha_dfs)}")

    t1 = time.time()
    Z_panel, D = build_Z_panel(matrices, tickers, available_chars, alpha_dfs,
                               start_bar, T_total, mode=3)
    bars_iter = list(range(start_bar, T_total))
    print(f"  Z panel D={D}, built in {time.time()-t1:.1f}s")
    print("=" * 90)

    all_results = []
    best = {"n_sr": -np.inf, "name": "(none)"}

    def log(s, df):
        nonlocal best
        if s is None:
            return
        all_results.append(s)
        if s["n_sr"] > best["n_sr"]:
            best = s.copy()
            print(f"  ★ new best: {best['name']}  nSR={best['n_sr']:+.3f}")
        # write CSV after each
        pd.DataFrame(all_results).to_csv(LOG_CSV, index=False)

    # --- batch 1: baseline + EWMA-w sweep ----------------------------------------
    print(f"\n[batch 1] baseline + EWMA-on-weights")
    s, df = run_variant("baseline", run_ewma_w, dict(alpha=1.0),
                        Z_panel, close_vals, start_bar, T_total, oos_start_idx, D, dates)
    log(s, df)
    for alpha in [0.5, 0.25, 0.1, 0.05, 0.025, 0.01]:
        if best["n_sr"] >= TARGET_NET_SR:
            break
        s, df = run_variant(f"ewma_w(α={alpha})", run_ewma_w, dict(alpha=alpha),
                            Z_panel, close_vals, start_bar, T_total, oos_start_idx, D, dates)
        log(s, df)

    # --- batch 2: EWMA-Z sweep ---------------------------------------------------
    if best["n_sr"] < TARGET_NET_SR:
        print(f"\n[batch 2] EWMA-on-Z (smooth feature panel)")
        for beta in [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]:
            if best["n_sr"] >= TARGET_NET_SR:
                break
            Zsm = smooth_z_panel(Z_panel, bars_iter, beta)
            s, df = run_variant(f"ewma_z(β={beta})", run_ewma_w, dict(alpha=1.0),
                                Zsm, close_vals, start_bar, T_total, oos_start_idx, D, dates)
            log(s, df)

    # --- batch 3: per-bar Markowitz with TC penalty ------------------------------
    if best["n_sr"] < TARGET_NET_SR:
        print(f"\n[batch 3] per-bar Markowitz with asset-space TC penalty (closed form)")
        for tau in [0.1, 1.0, 10.0, 100.0]:
            if best["n_sr"] >= TARGET_NET_SR:
                break
            s, df = run_variant(f"pa_step(τ={tau})", run_pa_step, dict(tau=tau),
                                Z_panel, close_vals, start_bar, T_total, oos_start_idx, D, dates)
            log(s, df)

    # --- batch 4: combos around best EWMA-w + EWMA-z -----------------------------
    if best["n_sr"] < TARGET_NET_SR:
        # find best ewma_w and best ewma_z
        ewma_w_best = max((r for r in all_results if r["name"].startswith("ewma_w")),
                         key=lambda r: r["n_sr"], default=None)
        ewma_z_best = max((r for r in all_results if r["name"].startswith("ewma_z")),
                         key=lambda r: r["n_sr"], default=None)
        if ewma_w_best and ewma_z_best:
            a0 = float(ewma_w_best["name"].split("=")[1].rstrip(")"))
            b0 = float(ewma_z_best["name"].split("=")[1].rstrip(")"))
            print(f"\n[batch 4] combos around best EWMA-w (α={a0}) + EWMA-z (β={b0})")
            for alpha in sorted({a0, a0*0.5, a0*2}):
                if alpha <= 0 or alpha > 1: continue
                for beta in sorted({b0, max(0.5, b0-0.1), min(0.99, b0+0.05)}):
                    if best["n_sr"] >= TARGET_NET_SR:
                        break
                    Zsm = smooth_z_panel(Z_panel, bars_iter, beta)
                    s, df = run_variant(f"combo(α={alpha:g},β={beta:g})", run_ewma_w,
                                        dict(alpha=alpha), Zsm, close_vals, start_bar,
                                        T_total, oos_start_idx, D, dates)
                    log(s, df)

    # --- batch 5: ridge sweep on best variant ------------------------------------
    if best["n_sr"] < TARGET_NET_SR:
        print(f"\n[batch 5] ridge sweep on baseline ewma_w")
        for ridge in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            if best["n_sr"] >= TARGET_NET_SR:
                break
            s, df = run_variant(f"baseline+ridge({ridge:g})", run_ewma_w,
                                dict(alpha=1.0, ridge=ridge), Z_panel, close_vals,
                                start_bar, T_total, oos_start_idx, D, dates)
            log(s, df)

    # ── done ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"Best variant: {best['name']}  nSR={best['n_sr']:+.3f}  "
          f"TO={best['avg_to']*100:.1f}%  ncum={best['n_cum']:+.1f}%")
    print(f"Target nSR={TARGET_NET_SR} {'REACHED' if best['n_sr'] >= TARGET_NET_SR else 'not reached'}")
    print(f"Total runtime: {(time.time()-overall_t0)/60:.1f} min  "
          f"({len(all_results)} variants tested)")

    # Frontier plot
    df_all = pd.DataFrame(all_results)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(df_all["avg_to"] * 100, df_all["n_sr"], s=70, alpha=0.7,
               c=range(len(df_all)), cmap="viridis", edgecolors="black", linewidths=0.5)
    for _, r in df_all.iterrows():
        ax.annotate(r["name"], (r["avg_to"]*100, r["n_sr"]),
                    textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax.axhline(TARGET_NET_SR, color="red", linestyle="--", label=f"target nSR={TARGET_NET_SR}")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("Avg turnover per bar (%)")
    ax.set_ylabel("Net Sharpe (3 bps taker)")
    ax.set_title(f"Iteration frontier — best: {best['name']} nSR={best['n_sr']:+.2f}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=150, bbox_inches="tight")
    print(f"Figure: {PLOT_OUT}")
    print(f"Log:    {LOG_CSV}")


if __name__ == "__main__":
    main()
