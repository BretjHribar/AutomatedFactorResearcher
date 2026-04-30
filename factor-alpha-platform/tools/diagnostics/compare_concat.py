"""
compare_concat.py — Dual-stream AIPT:

  PROD-RFF + TREES : two parallel signal streams merged at the ridge stage.
    Stream A : 24 prod chars  ->  RFF expansion -> P_rff features
    Stream B : 480 random trees (raw, no RFF)   -> K_trees features
    Combined factor returns: F_t = [F_prod_t, F_trees_t]  (length P_rff + K_trees)
    Ridge-Markowitz on the combined factor return matrix.
    Portfolio weights: w_t = S_combined_t @ lambda / sqrt(N_t)

This is the "best of both worlds" config — keeps each stream in its proven
representation (RFF for simple chars, raw for rich trees) and lets ridge
decide the relative weighting.

Usage:
    python compare_concat.py
    python compare_concat.py --prod-p 5000 --rebuild
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aipt_kucoin import (
    load_data,
    build_characteristics_panel,
    generate_rff_params,
    compute_rff_signals,
    estimate_ridge_markowitz,
    OOS_START,
    TRAIN_BARS,
    MIN_TRAIN_BARS,
    BARS_PER_YEAR,
    RESULTS_DIR,
)

from compare_random_trees import (
    generate_random_trees,
    evaluate_trees,
    rank_standardize_panel,
    print_summary,
    plot_comparison,
    VAL_END,
    RIDGE_Z,
    REBAL_EVERY,
    N_TREES,
)

PANEL_CACHE = RESULTS_DIR / f"trees_panel_n{N_TREES}_seed42.npz"


# ─────────────────────────────────────────────────────────────────────────────
# DUAL-STREAM AIPT
# ─────────────────────────────────────────────────────────────────────────────

def run_aipt_dual_stream(Z_prod_3d: np.ndarray, Z_trees_3d: np.ndarray,
                          returns_np: np.ndarray, dates,
                          P_rff_prod: int, z: float,
                          rebal_every: int, train_bars: int, min_train_bars: int,
                          seed: int = 42, label: str = "") -> dict:
    """
    Two-stream AIPT.
      Stream A: Z_prod_3d (T,N,K_prod) -> RFF -> S_prod (N, P_rff_prod)
      Stream B: Z_trees_3d (T,N,K_trees) -> raw  -> S_trees (N, K_trees)
    Concatenated S_combined feeds factor returns + ridge-Markowitz.
    """
    T, N, K_prod  = Z_prod_3d.shape
    _, _, K_trees = Z_trees_3d.shape
    P_eff = P_rff_prod + K_trees

    print(f"\n  [{label}] T={T}, N={N}", flush=True)
    print(f"    Stream A: K_prod={K_prod}  -> RFF -> P_rff={P_rff_prod}", flush=True)
    print(f"    Stream B: K_trees={K_trees} (no RFF)", flush=True)
    print(f"    Combined factor dim P_eff = {P_eff}", flush=True)

    theta, gamma = generate_rff_params(K_prod, P_rff_prod, seed=seed)

    factor_returns = {}
    sig_cache = {}

    for t in range(T - 1):
        Z_p  = Z_prod_3d[t]    # (N, K_prod)
        Z_tr = Z_trees_3d[t]   # (N, K_trees)
        r_t1 = returns_np[t + 1, :]
        valid = ((~np.isnan(r_t1))
                 & (~np.isnan(Z_p).any(axis=1))
                 & (~np.isnan(Z_tr).any(axis=1)))
        N_t = valid.sum()
        if N_t < 5:
            continue

        S_prod  = compute_rff_signals(Z_p[valid], theta, gamma)   # (N_t, P_rff_prod)
        S_trees = Z_tr[valid]                                      # (N_t, K_trees)
        S_combined = np.concatenate([S_prod, S_trees], axis=1)     # (N_t, P_eff)

        r_clean = np.nan_to_num(r_t1[valid], nan=0.0)
        F_t1 = (1.0 / np.sqrt(N_t)) * (S_combined.T @ r_clean)     # (P_eff,)
        factor_returns[t + 1] = F_t1
        sig_cache[t] = (S_combined, valid, N_t)

    # ── Rolling SDF
    all_idx = sorted(factor_returns.keys())
    port_returns, turnovers, bar_dates = [], [], []
    lambda_hat = None
    bars_since = rebal_every
    prev_w = None

    for oos_t in range(1, T):
        if bars_since >= rebal_every or lambda_hat is None:
            train_idx = [i for i in all_idx
                         if i < oos_t and i >= max(0, oos_t - train_bars)]
            if len(train_idx) < min_train_bars:
                bars_since += 1
                continue
            F_train = np.vstack([factor_returns[i] for i in train_idx])
            lambda_hat = estimate_ridge_markowitz(F_train, z)
            bars_since = 0

        sig_bar = oos_t - 1
        if sig_bar not in sig_cache or lambda_hat is None:
            bars_since += 1
            continue

        S_t, vmask, N_t = sig_cache[sig_bar]
        raw_w = np.zeros(N)
        raw_w[vmask] = (1.0 / np.sqrt(N_t)) * (S_t @ lambda_hat)
        abs_sum = np.abs(raw_w).sum()
        if abs_sum < 1e-12:
            bars_since += 1
            continue
        w_norm = raw_w / abs_sum

        r_t1 = np.nan_to_num(returns_np[oos_t, :], nan=0.0)
        port_returns.append(float(w_norm @ r_t1))
        bar_dates.append(dates[oos_t])
        if prev_w is not None:
            turnovers.append(np.abs(w_norm - prev_w).sum() / 2.0)
        else:
            turnovers.append(0.0)
        prev_w = w_norm.copy()
        bars_since += 1

    return dict(
        label=label, P=P_eff, K=K_prod + K_trees,
        bar_dates=bar_dates,
        gross=np.array(port_returns),
        turnover=np.array(turnovers),
    )


def get_or_build_trees_panel(matrices, tickers, n_trees: int, seed: int,
                              rebuild: bool = False) -> np.ndarray:
    if PANEL_CACHE.exists() and not rebuild:
        print(f"\n  Loading cached trees panel from {PANEL_CACHE}", flush=True)
        Z = np.load(PANEL_CACHE)["Z"]
        print(f"  Loaded shape {Z.shape}", flush=True)
        return Z

    print(f"\n  Generating {n_trees} random DSL trees (seed={seed})...", flush=True)
    trees = generate_random_trees(n_trees, seed=seed)
    print(f"  Evaluating on {len(tickers)} tickers x "
          f"{matrices['close'].shape[0]} bars...", flush=True)
    Z_raw, kept = evaluate_trees(trees, matrices, tickers, verbose=True)
    if Z_raw is None:
        raise RuntimeError("No valid trees")

    print(f"  Rank-standardizing panel ({Z_raw.shape})...", flush=True)
    t0 = time.time()
    Z = rank_standardize_panel(Z_raw)
    print(f"    Done in {time.time()-t0:.1f}s", flush=True)

    PANEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PANEL_CACHE, Z=Z)
    pd.Series(kept).to_csv(
        PANEL_CACHE.with_suffix(".csv"), index=False, header=["expression"]
    )
    print(f"  Cached to {PANEL_CACHE}", flush=True)
    return Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trees", type=int, default=N_TREES)
    ap.add_argument("--prod-p",  type=int, default=5000,
                    help="P for the RFF expansion of the 24 prod chars")
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--fees",    type=float, default=3.0)
    ap.add_argument("--rebuild", action="store_true",
                    help="Force regenerate the trees panel (don't use cache)")
    args = ap.parse_args()

    print("=" * 80)
    print("  AIPT DUAL-STREAM COMPARISON")
    print(f"  Stream A: 24 prod chars -> RFF -> P_rff={args.prod_p}")
    print(f"  Stream B: ~{args.n_trees} random trees (raw, no RFF)")
    print(f"  Stacked into ridge-Markowitz")
    print("=" * 80)

    # ── Load data
    print("\nLoading data...", flush=True)
    matrices, universe, tickers = load_data()
    close = matrices["close"]
    returns_pct = matrices["returns_pct"]
    dates = close.index
    T = len(dates)
    N = len(tickers)
    returns_np = returns_pct.reindex(columns=tickers).values.astype(np.float64)
    print(f"  T={T} bars, N={N} tickers", flush=True)

    # ── Build prod panel (24 chars)
    print("\nBuilding PROD panel (24 hand-picked characteristics)...", flush=True)
    Z_prod_dict, prod_chars = build_characteristics_panel(matrices, tickers, 0, T)
    K_prod = len(prod_chars)
    Z_prod_3d = np.zeros((T, N, K_prod), dtype=np.float64)
    for i in range(T):
        if i in Z_prod_dict:
            Z_prod_3d[i] = Z_prod_dict[i]
    print(f"  Built ({K_prod} characteristics)", flush=True)

    # ── Get random-tree panel (cached or build)
    Z_trees_3d = get_or_build_trees_panel(
        matrices, tickers, n_trees=args.n_trees, seed=args.seed,
        rebuild=args.rebuild,
    )
    K_trees = Z_trees_3d.shape[2]

    # ── Run dual-stream config
    print("\n" + "=" * 80)
    print("  Running dual-stream AIPT...")
    print("=" * 80)

    res = run_aipt_dual_stream(
        Z_prod_3d, Z_trees_3d, returns_np, dates,
        P_rff_prod=args.prod_p, z=RIDGE_Z,
        rebal_every=REBAL_EVERY, train_bars=TRAIN_BARS,
        min_train_bars=MIN_TRAIN_BARS,
        seed=args.seed,
        label=f"PROD-RFF + TREES (P={args.prod_p}+{K_trees}={args.prod_p+K_trees})",
    )

    # ── Report alongside prior numbers from compare_random_trees.py
    print(f"\n{'='*100}")
    print(f"  PRIOR RESULTS  (from compare_random_trees.py, fees=3bps)")
    print(f"{'='*100}")
    print("  Config         TRAIN SR_n   VAL SR_n   TEST SR_n   TEST AnnR%   TEST MaxDD")
    print("  " + "-" * 80)
    print("  PROD-RFF        +29.26       +27.15      +23.09       +260.4       -0.031")
    print("  RAND-TREES      +28.97       +26.54      +24.30       +315.8       -0.023")
    print("  TREES+RFF        -3.76        -2.56       -6.21        -88.3       -1.014")

    print_summary({"DUAL-STREAM": res}, fee_bps=args.fees)
    plot_comparison({"DUAL-STREAM": res}, fee_bps=args.fees,
                    save_dir=RESULTS_DIR)

    # rename plot to avoid clobbering prior one
    src = RESULTS_DIR / "compare_random_trees.png"
    dst = RESULTS_DIR / "compare_dual_stream.png"
    if src.exists():
        try:
            src.replace(dst)
            print(f"  Plot saved as {dst}")
        except Exception as e:
            print(f"  Could not rename plot: {e}")


if __name__ == "__main__":
    main()
