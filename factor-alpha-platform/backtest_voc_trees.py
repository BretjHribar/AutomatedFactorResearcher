"""
Overnight tree-feature AIPT sweep.

For each K in K_GRID:
  1. Generate K random GP trees (depth ≤ 3) on raw market fields.
  2. Evaluate via FastExpressionEngine to a (T, N, K) panel.
  3. Cache panel to disk so subsequent runs (different P) skip re-evaluation.
  4. For each P in P_GRID:
     - Run AIPT pipeline twice:
        (a) ridge directly on tree outputs (no RFF)   — only once per K
        (b) RFF (P) on tree outputs, then ridge       — sweep P
Reports stats per config to stdout and master CSV.
"""
from __future__ import annotations
import sys, time, os
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from compare_random_trees import (
    generate_random_trees, evaluate_trees, rank_standardize_panel,
    split_stats, VAL_END, RIDGE_Z, REBAL_EVERY,
)
from aipt_kucoin import (
    load_data, OOS_START, TRAIN_BARS, MIN_TRAIN_BARS, BARS_PER_YEAR, RESULTS_DIR,
    generate_rff_params, compute_rff_signals, estimate_ridge_markowitz,
)


GAMMA_REF_D = 24   # gamma grid in aipt_kucoin was tuned for D=24 chars


def run_aipt_low_mem(Z_panel_3d, returns_np, dates,
                     use_rff: bool, P_rff: int, z: float,
                     rebal_every: int, train_bars: int, min_train_bars: int,
                     seed: int = 42, label: str = "") -> dict:
    """Memory-efficient AIPT — no per-bar S_t cache. F_t is cached (small);
    S_t is recomputed at each OOS portfolio step. Allows P >> RAM/cache limits.

    Note: scales gamma by sqrt(GAMMA_REF_D / K) so that projection variance
    γ·Z·θ stays in the sin/cos working regime regardless of K. Without this
    rescaling, large K puts every sample into a random sin/cos phase and the
    kernel collapses to noise.
    """
    T, N, K = Z_panel_3d.shape
    if use_rff:
        theta, gamma = generate_rff_params(K, P_rff, seed=seed)
        gamma = gamma * np.sqrt(GAMMA_REF_D / K)        # K-invariant bandwidth
        P_eff = P_rff
    else:
        theta = gamma = None
        P_eff = K
    print(f"  [{label}] T={T} N={N} K={K}  P_eff={P_eff}  "
          f"({'RFF, gamma_scale=' + f'{np.sqrt(GAMMA_REF_D / K):.3f}' if use_rff else 'no-RFF'})",
          flush=True)

    def project(Z_v):
        return compute_rff_signals(Z_v, theta, gamma) if use_rff else Z_v

    # ── Pass 1: compute F_t at every bar (small per-bar storage) ──
    t0 = time.time()
    factor_returns = {}
    valid_cache = {}
    for t in range(T - 1):
        Z_t = Z_panel_3d[t]
        r_t1 = returns_np[t + 1, :]
        valid = (~np.isnan(r_t1)) & (~np.isnan(Z_t).any(axis=1))
        N_t = int(valid.sum())
        if N_t < 5:
            continue
        S_v = project(Z_t[valid])
        r_clean = np.nan_to_num(r_t1[valid], nan=0.0)
        factor_returns[t + 1] = (1.0 / np.sqrt(N_t)) * (S_v.T @ r_clean)
        valid_cache[t] = (valid, N_t)
    print(f"    F-pass: {len(factor_returns)} bars in {time.time()-t0:.1f}s", flush=True)

    # ── Pass 2: rolling SDF + portfolio (re-project S_t on demand) ──
    t0 = time.time()
    all_idx = sorted(factor_returns.keys())
    port_returns, turnovers, bar_dates = [], [], []
    lambda_hat = None
    bars_since = rebal_every
    prev_w = None
    n_solves = 0

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
            n_solves += 1

        sig_bar = oos_t - 1
        if sig_bar not in valid_cache or lambda_hat is None:
            bars_since += 1
            continue

        valid, N_t = valid_cache[sig_bar]
        S_v = project(Z_panel_3d[sig_bar][valid])
        raw_w = np.zeros(N)
        raw_w[valid] = (1.0 / np.sqrt(N_t)) * (S_v @ lambda_hat)
        abs_sum = np.abs(raw_w).sum()
        if abs_sum < 1e-12:
            bars_since += 1
            continue
        w_norm = raw_w / abs_sum

        r_t1 = np.nan_to_num(returns_np[oos_t, :], nan=0.0)
        port_returns.append(float(w_norm @ r_t1))
        bar_dates.append(dates[oos_t])
        turnovers.append(np.abs(w_norm - prev_w).sum() / 2.0 if prev_w is not None else 0.0)
        prev_w = w_norm.copy()
        bars_since += 1

    print(f"    Portfolio pass: {len(port_returns)} bars, {n_solves} ridge solves "
          f"in {time.time()-t0:.1f}s", flush=True)

    return dict(
        label=label, P=P_eff, K=K,
        bar_dates=bar_dates,
        gross=np.array(port_returns),
        turnover=np.array(turnovers),
    )

# ── Sweep grids ──────────────────────────────────────────────────────────────
K_GRID    = [500, 1000, 2000]                     # number of trees (mem-bounded; 5000 ≈ 23GB)
P_GRID    = [2000, 5000, 10000, 20000, 40000]     # RFF expansion size
SEED      = 42
FEE_BPS   = 3.0
LOG_CSV   = RESULTS_DIR / "trees_sweep_results.csv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def panel_cache_path(K: int, seed: int) -> Path:
    """Match naming used by the original compare_random_trees.py cache."""
    return RESULTS_DIR / f"trees_panel_n{K}_seed{seed}.npz"


def load_or_build_panel(K: int, seed: int, matrices, tickers):
    """Return rank-standardized (T,N,K_actual) panel. Caches Z (standardized)."""
    cache = panel_cache_path(K, seed)
    if cache.exists():
        print(f"  >>> loading cached panel {cache.name}", flush=True)
        z = np.load(cache, allow_pickle=True)
        Z_std = z["Z"]
        print(f"      cached shape={Z_std.shape}", flush=True)
        return Z_std

    print(f"  >>> generating {K} random GP trees (seed={seed})...", flush=True)
    trees = generate_random_trees(K, seed=seed)
    print(f"      {len(trees)} unique exprs. Sample:", flush=True)
    for ex in trees[:3]:
        print(f"        {ex[:110]}{'...' if len(ex)>110 else ''}", flush=True)

    print(f"  >>> evaluating trees on {len(tickers)} tickers...", flush=True)
    Z_raw, kept = evaluate_trees(trees, matrices, tickers, verbose=True)
    if Z_raw is None:
        return None

    print(f"  >>> rank-standardizing...", flush=True)
    rt0 = time.time()
    Z_std = rank_standardize_panel(Z_raw)
    print(f"      done in {time.time()-rt0:.1f}s", flush=True)
    del Z_raw

    print(f"  >>> caching to {cache.name}  ({Z_std.nbytes/1e9:.1f}GB raw)", flush=True)
    np.savez_compressed(cache, Z=Z_std)
    return Z_std


def summarize_split(res: dict, fee_bps: float) -> dict:
    """One-row summary across the OOS test split (post VAL_END)."""
    df = split_stats(res, fee_bps=fee_bps)
    test_row = df[df["split"] == "TEST  (OOS)"].iloc[0]
    val_row  = df[df["split"] == "VAL   (OOS)"].iloc[0]
    return {
        "label":       res["label"],
        "K":           res["K"],
        "P_eff":       res["P"],
        "val_sr_n":    val_row["sr_net"],
        "val_sr_g":    val_row["sr_gross"],
        "val_to":      val_row["avg_to"],
        "val_ann_ret": val_row["ann_ret_net%"],
        "test_sr_n":   test_row["sr_net"],
        "test_sr_g":   test_row["sr_gross"],
        "test_to":     test_row["avg_to"],
        "test_ann_ret":test_row["ann_ret_net%"],
        "test_max_dd": test_row["max_dd_net"],
        "n_test":      int(test_row["n_bars"]),
    }


def print_row(d: dict):
    print(f"  >>> {d['label']:<48}  "
          f"VAL: SR_n={d['val_sr_n']:+.2f} TO={d['val_to']:.2f}  "
          f"TEST: SR_n={d['test_sr_n']:+.2f} SR_g={d['test_sr_g']:+.2f} "
          f"TO={d['test_to']:.2f} AnnR={d['test_ann_ret']:+.1f}%",
          flush=True)


def append_log(rows: list[dict]):
    df = pd.DataFrame(rows)
    df.to_csv(LOG_CSV, index=False)
    print(f"  >>> log: {LOG_CSV}", flush=True)


def main():
    overall_t0 = time.time()
    print("=" * 100, flush=True)
    print(f"OVERNIGHT TREE-FEATURE AIPT SWEEP", flush=True)
    print(f"  K_GRID = {K_GRID}", flush=True)
    print(f"  P_GRID = {P_GRID}", flush=True)
    print(f"  fees = {FEE_BPS} bps,  rebal every {REBAL_EVERY} bars,  ridge = {RIDGE_Z}", flush=True)
    print("=" * 100, flush=True)

    print("\nLoading market data...", flush=True)
    matrices, universe, tickers = load_data()
    close = matrices["close"]
    returns_pct = matrices["returns_pct"]
    dates = close.index
    T, N = len(dates), len(tickers)
    returns_np = returns_pct.reindex(columns=tickers).values.astype(np.float64)
    print(f"  T={T} bars, N={N} tickers ({dates[0]} → {dates[-1]})", flush=True)

    all_rows = []

    for K_idx, K in enumerate(K_GRID):
        print("\n" + "#" * 100, flush=True)
        print(f"#  K={K}  ({K_idx+1}/{len(K_GRID)})", flush=True)
        print("#" * 100, flush=True)

        kt0 = time.time()
        Z_3d = load_or_build_panel(K, SEED, matrices, tickers)
        if Z_3d is None:
            print(f"  ! panel build failed for K={K}; skipping", flush=True)
            continue
        K_actual = Z_3d.shape[2]
        print(f"  Panel ready: {Z_3d.shape}  ({(time.time()-kt0)/60:.1f}min)", flush=True)

        # --- Mode A: ridge directly on tree outputs (NO RFF) ----------------------
        print(f"\n  [A] RIDGE ON TREES (no RFF)  K={K_actual}", flush=True)
        t0 = time.time()
        res_a = run_aipt_low_mem(
            Z_3d, returns_np, dates,
            use_rff=False, P_rff=0, z=RIDGE_Z,
            rebal_every=REBAL_EVERY, train_bars=TRAIN_BARS,
            min_train_bars=MIN_TRAIN_BARS, seed=SEED,
            label=f"TREES-only K={K_actual}",
        )
        d = summarize_split(res_a, FEE_BPS)
        d["K_actual"] = K_actual
        d["minutes"]  = (time.time()-t0)/60
        d["mode"] = "ridge_only"
        print_row(d); all_rows.append(d); append_log(all_rows)

        # --- Mode B: RFF on tree outputs, sweep P ---------------------------------
        for P in P_GRID:
            print(f"\n  [B] TREES + RFF (P={P})  K={K_actual}", flush=True)
            t0 = time.time()
            res_b = run_aipt_low_mem(
                Z_3d, returns_np, dates,
                use_rff=True, P_rff=P, z=RIDGE_Z,
                rebal_every=REBAL_EVERY, train_bars=TRAIN_BARS,
                min_train_bars=MIN_TRAIN_BARS, seed=SEED,
                label=f"TREES+RFF K={K_actual} P={P}",
            )
            d = summarize_split(res_b, FEE_BPS)
            d["K_actual"] = K_actual
            d["minutes"]  = (time.time()-t0)/60
            d["mode"] = f"rff_P{P}"
            print_row(d); all_rows.append(d); append_log(all_rows)

        del Z_3d  # free memory before next K

    print("\n" + "=" * 100, flush=True)
    print(f"  DONE in {(time.time()-overall_t0)/60:.1f}min", flush=True)
    print("=" * 100, flush=True)
    df = pd.DataFrame(all_rows)
    df = df.sort_values("test_sr_n", ascending=False)
    print(df[["label", "mode", "K_actual", "P_eff", "val_sr_n", "test_sr_n",
              "test_to", "test_ann_ret", "minutes"]].to_string(index=False))
    print(f"\nFinal log: {LOG_CSV}", flush=True)


if __name__ == "__main__":
    main()
