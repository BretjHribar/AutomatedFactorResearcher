"""
Compute Sharpe + IC + R² for the three baseline AIPT variants at P=1000
(no smoothing, no TC penalty) using the original 18 KuCoin 4h alphas.

  mode 1 — random projections (24 chars only)
  mode 2 — alphas only (18)
  mode 3 — both (24 + 18)

For each mode and bar t (OOS):
  pred_t   = (1/√N) Sₜ λ            (cross-sectional prediction at valid assets)
  realized = R_{t+1}                 (next-bar returns at valid assets)
  IC_t     = cross-sectional correlation(pred_t, realized)   (Pearson + Spearman)
  R²_t     = (Pearson IC_t)²         (univariate cross-sectional R²)
Reports mean / std and IR = mean(IC) / std(IC) * sqrt(BARS_PER_YEAR).
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from backtest_voc_postfix import (
    load_data, build_Z_panel,
    BARS_PER_YEAR, TRAIN_BARS, MIN_TRAIN_BARS, REBAL_EVERY,
    OOS_START, Z_RIDGE, GAMMA_GRID, TAKER_BPS,
)
from backtest_voc_iter3 import load_data_with_all_db_alphas, build_Z_panel as build_Z_panel_v3

SEED = 42
P = 1000


def run_with_ic(Z_panel, close_vals, start_bar, T_total, oos_start_idx, D, seed=SEED):
    rng = np.random.default_rng(seed)
    n_pairs = P // 2
    theta = rng.standard_normal((n_pairs, D))
    gamma = rng.choice(GAMMA_GRID, size=n_pairs)

    fr_history, lambda_hat = [], None
    bars_since_rebal = REBAL_EVERY
    prev_w = None
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
            A = Z_RIDGE * np.eye(P_tr) + FF
            lambda_hat = np.linalg.solve(A, F_train.mean(axis=0))
            bars_since_rebal = 0

        pred_v = (1.0 / np.sqrt(N_t)) * (S_v @ lambda_hat)  # cross-sectional prediction

        # IC and R² (cross-sectional)
        if pred_v.std() > 1e-12 and R_v.std() > 1e-12:
            ic_p = float(np.corrcoef(pred_v, R_v)[0, 1])
            ic_s = float(np.corrcoef(rankdata(pred_v), rankdata(R_v))[0, 1])
            r2 = ic_p ** 2
        else:
            ic_p = ic_s = r2 = 0.0

        raw_w = np.zeros(Z_t.shape[0])
        raw_w[valid] = pred_v
        abs_sum = np.abs(raw_w).sum()
        if abs_sum < 1e-12:
            bars_since_rebal += 1
            continue
        w_norm = raw_w / abs_sum
        port_ret = float(w_norm @ R_t1)
        to = float(np.abs(w_norm - prev_w).sum() / 2.0) if prev_w is not None else 0.0
        prev_w = w_norm.copy()
        bars_since_rebal += 1

        rows.append({
            "bar_idx": t + 1, "gross": port_ret, "turnover": to,
            "ic_pearson": ic_p, "ic_spearman": ic_s, "r2": r2,
        })

    df = pd.DataFrame(rows)
    df["net_3bps"] = df["gross"] - df["turnover"] * TAKER_BPS / 10000.0 * 2.0
    return df


def main():
    print("Loading data and evaluating 18 alphas...")
    matrices, tickers, dates, close_vals, available_chars, alpha_dfs = load_data()
    T_total = len(dates)
    oos_start_idx = next(i for i, d in enumerate(dates) if str(d) >= OOS_START)
    start_bar = max(0, oos_start_idx - TRAIN_BARS - 10)
    print(f"  N={len(tickers)} chars={len(available_chars)} alphas={len(alpha_dfs)} bars={T_total}\n")

    modes = {
        1: ("RP (chars)", "Random projections - chars only"),
        2: ("Alphas",     "Alphas only"),
        3: ("Both",       "Chars + alphas"),
    }

    summary = []
    for mode, (short, label) in modes.items():
        t0 = time.time()
        Z_panel, D = build_Z_panel(matrices, tickers, available_chars, alpha_dfs,
                                   start_bar, T_total, mode)
        df = run_with_ic(Z_panel, close_vals, start_bar, T_total, oos_start_idx, D)
        ann = np.sqrt(BARS_PER_YEAR)
        gross_sr = df["gross"].mean() / df["gross"].std(ddof=1) * ann
        net_sr = df["net_3bps"].mean() / df["net_3bps"].std(ddof=1) * ann
        ic_p = df["ic_pearson"].mean()
        ic_s = df["ic_spearman"].mean()
        ir_p = df["ic_pearson"].mean() / df["ic_pearson"].std(ddof=1) * ann
        ir_s = df["ic_spearman"].mean() / df["ic_spearman"].std(ddof=1) * ann
        r2_mean = df["r2"].mean()
        r2_med  = df["r2"].median()
        avg_to = df["turnover"].mean()
        elapsed = time.time() - t0

        summary.append({
            "mode": short, "D": D, "bars": len(df),
            "gross_SR": gross_sr, "net_SR": net_sr,
            "IC_Pearson": ic_p, "IR_Pearson": ir_p,
            "IC_Spearman": ic_s, "IR_Spearman": ir_s,
            "R2_mean": r2_mean, "R2_median": r2_med,
            "avg_TO": avg_to,
        })
        print(f"[mode {mode}] {label}  D={D}  ({elapsed:.1f}s)")
        print(f"  gross_SR  = {gross_sr:+.3f}    net_SR = {net_sr:+.3f}    avg_TO = {avg_to*100:.1f}%")
        print(f"  IC_Pear   = {ic_p:+.5f}    IR_Pear   = {ir_p:+.3f}")
        print(f"  IC_Spear  = {ic_s:+.5f}    IR_Spear  = {ir_s:+.3f}")
        print(f"  R²_mean   = {r2_mean:.5f}     R²_median = {r2_med:.5f}\n")

    # ── 4th row: chars + ALL DB alphas (current count) ─────────────────────────
    print("Loading ALL crypto 4h KUCOIN_TOP100 alphas from DB...")
    matrices2, tickers2, dates2, close_vals2, available_chars2, alpha_dfs_all = \
        load_data_with_all_db_alphas()
    T_total2 = len(dates2)
    oos2 = next(i for i, d in enumerate(dates2) if str(d) >= OOS_START)
    start2 = max(0, oos2 - TRAIN_BARS - 10)
    Z_panel_full, D_full = build_Z_panel_v3(matrices2, tickers2, available_chars2,
                                            alpha_dfs_all, start2, T_total2, mode=3)
    df = run_with_ic(Z_panel_full, close_vals2, start2, T_total2, oos2, D_full)
    ann = np.sqrt(BARS_PER_YEAR)
    g_sr = df["gross"].mean() / df["gross"].std(ddof=1) * ann
    n_sr = df["net_3bps"].mean() / df["net_3bps"].std(ddof=1) * ann
    ic_p = df["ic_pearson"].mean()
    ic_s = df["ic_spearman"].mean()
    ir_p = df["ic_pearson"].mean() / df["ic_pearson"].std(ddof=1) * ann
    ir_s = df["ic_spearman"].mean() / df["ic_spearman"].std(ddof=1) * ann
    r2_mean = df["r2"].mean()
    r2_med = df["r2"].median()
    avg_to = df["turnover"].mean()
    summary.append({
        "mode": f"Both+DB({len(alpha_dfs_all)})", "D": D_full, "bars": len(df),
        "gross_SR": g_sr, "net_SR": n_sr,
        "IC_Pearson": ic_p, "IR_Pearson": ir_p,
        "IC_Spearman": ic_s, "IR_Spearman": ir_s,
        "R2_mean": r2_mean, "R2_median": r2_med,
        "avg_TO": avg_to,
    })
    print(f"\n[mode 3+] Chars + ALL {len(alpha_dfs_all)} DB alphas  D={D_full}")
    print(f"  gross_SR  = {g_sr:+.3f}    net_SR = {n_sr:+.3f}    avg_TO = {avg_to*100:.1f}%")
    print(f"  IC_Pear   = {ic_p:+.5f}    IR_Pear   = {ir_p:+.3f}")
    print(f"  IC_Spear  = {ic_s:+.5f}    IR_Spear  = {ir_s:+.3f}")
    print(f"  R²_mean   = {r2_mean:.5f}     R²_median = {r2_med:.5f}\n")

    print("=" * 100)
    print(pd.DataFrame(summary).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()
