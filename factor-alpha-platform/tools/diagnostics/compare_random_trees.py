"""
compare_random_trees.py — AIPT with three feature spaces:

  1. PROD-RFF      : 24 hand-picked characteristics + RFF (P=5000) + ridge-Markowitz
  2. RAND-TREES    : 500 random DSL trees as characteristics + ridge-Markowitz (no RFF)
  3. TREES + RFF   : 500 random trees -> RFF (P=2000) + ridge-Markowitz

All use the same data, OOS split (2024-09-01), and ridge penalty.
Reports train/val/test Sharpe, ann return, turnover, max DD for each.

Usage:
    python compare_random_trees.py
    python compare_random_trees.py --n-trees 800 --seed 7
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
    compute_factor_returns,
    estimate_ridge_markowitz,
    OOS_START,
    TRAIN_BARS,
    MIN_TRAIN_BARS,
    BARS_PER_YEAR,
    RESULTS_DIR,
)

from src.operators.fastexpression import FastExpressionEngine

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

VAL_END    = "2025-03-01"   # split between val and test
RIDGE_Z    = 1e-3
REBAL_EVERY = 12

# Production AIPT config
PROD_P     = 5000

# Random-tree config
N_TREES        = 500
TREE_RFF_P     = 2000
TREE_DEPTH_MAX = 3

# Operators with safe semantics for random composition.
# All are causal — no future leakage by construction.
TS_OPS_UNARY = [
    ("ts_rank",     [10, 20, 60, 120]),
    ("ts_zscore",   [10, 20, 60, 120]),
    ("ts_mean",     [5, 10, 20, 60]),
    ("ts_std_dev",  [10, 20, 60]),
    ("delta",       [1, 5, 20, 60]),
    ("ts_skewness", [20, 60]),
    ("ts_min",      [10, 20, 60]),
    ("ts_max",      [10, 20, 60]),
    ("ts_argmax",   [20, 60]),
    ("ts_argmin",   [20, 60]),
    ("decay_linear",[10, 20, 60]),
]

TS_OPS_BINARY = [
    ("ts_corr", [20, 60, 120]),
    ("ts_cov",  [20, 60]),
]

EW_OPS_UNARY  = ["abs", "sign", "log", "sqrt", "negative", "square"]
EW_OPS_BINARY = ["subtract", "multiply", "divide", "df_max", "df_min"]

# Base data fields available (from matrices/4h/)
BASE_FIELDS = [
    "close", "open", "high", "low", "volume", "vwap",
    "log_returns", "returns",
    "high_low_range", "open_close_range", "close_position_in_range",
    "dollars_traded", "quote_volume", "turnover",
    "vwap_deviation",
    "momentum_5d", "momentum_20d", "momentum_60d",
    "historical_volatility_10", "historical_volatility_20",
    "historical_volatility_60", "historical_volatility_120",
    "parkinson_volatility_10", "parkinson_volatility_20",
    "parkinson_volatility_60",
    "adv20", "adv60",
    "volume_momentum_1", "volume_momentum_5_20", "volume_ratio_20d",
    "beta_to_btc",
]


# ─────────────────────────────────────────────────────────────────────────────
# RANDOM TREE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _random_field(rng: np.random.Generator) -> str:
    return rng.choice(BASE_FIELDS)


def _random_tree(rng: np.random.Generator, depth: int = 0) -> str:
    """
    Recursively build a random DSL expression. Always wraps the final
    result in `rank(...)` to ensure a consistent cross-sectional scale.
    Internal node distribution is biased toward time-series ops.
    """
    if depth == 0:
        # Force a TS aggregation at the outer layer + rank wrap
        inner = _random_subtree(rng, 1)
        return f"rank({inner})"
    return _random_subtree(rng, depth)


def _random_subtree(rng: np.random.Generator, depth: int) -> str:
    """Build a random subtree of given remaining depth."""
    if depth >= TREE_DEPTH_MAX:
        return _random_field(rng)

    # Choose op category — bias toward TS at deeper levels (more aggregation)
    cat = rng.choice(["ts_unary", "ts_binary", "ew_unary", "ew_binary", "leaf"],
                     p=[0.35, 0.20, 0.10, 0.20, 0.15])

    if cat == "leaf":
        return _random_field(rng)

    if cat == "ts_unary":
        op, windows = TS_OPS_UNARY[rng.integers(len(TS_OPS_UNARY))]
        w = int(rng.choice(windows))
        child = _random_subtree(rng, depth + 1)
        return f"{op}({child}, {w})"

    if cat == "ts_binary":
        op, windows = TS_OPS_BINARY[rng.integers(len(TS_OPS_BINARY))]
        w = int(rng.choice(windows))
        a = _random_subtree(rng, depth + 1)
        b = _random_subtree(rng, depth + 1)
        return f"{op}({a}, {b}, {w})"

    if cat == "ew_unary":
        op = EW_OPS_UNARY[rng.integers(len(EW_OPS_UNARY))]
        child = _random_subtree(rng, depth + 1)
        return f"{op}({child})"

    if cat == "ew_binary":
        op = EW_OPS_BINARY[rng.integers(len(EW_OPS_BINARY))]
        a = _random_subtree(rng, depth + 1)
        b = _random_subtree(rng, depth + 1)
        return f"{op}({a}, {b})"

    return _random_field(rng)


def generate_random_trees(n: int, seed: int = 42) -> list[str]:
    rng = np.random.default_rng(seed)
    trees = []
    seen = set()
    attempts = 0
    while len(trees) < n and attempts < n * 20:
        expr = _random_tree(rng, 0)
        if expr not in seen:
            seen.add(expr)
            trees.append(expr)
        attempts += 1
    return trees


def evaluate_trees(trees: list[str], matrices: dict, tickers: list,
                   verbose: bool = True) -> tuple[np.ndarray, list[str]]:
    """
    Evaluate each random tree expression to a (T, N) matrix.
    Filters out degenerate trees (>50% NaN, ~zero cross-sectional std,
    or evaluation failure). Returns (T, N, K_ok) array + list of kept exprs.
    """
    engine = FastExpressionEngine(data_fields=matrices)

    close = matrices["close"][tickers]
    T, N = close.shape
    dates = close.index

    panels = []
    kept_exprs = []
    n_failed = 0
    n_degen  = 0
    t0 = time.time()

    for i, expr in enumerate(trees):
        if verbose and (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"    [{i+1}/{len(trees)}] kept={len(kept_exprs)} "
                  f"fail={n_failed} degen={n_degen} ({elapsed:.0f}s)",
                  flush=True)
        try:
            df = engine.evaluate(expr)
            if df is None or df.empty:
                n_failed += 1
                continue
            arr = df.reindex(index=dates, columns=tickers).values.astype(np.float64)
            arr[~np.isfinite(arr)] = np.nan

            # Reject degenerate: too many NaN OR no cross-sectional variation
            valid_frac = np.isfinite(arr).mean()
            if valid_frac < 0.30:
                n_degen += 1
                continue
            cs_std = np.nanstd(arr, axis=1)
            if np.nanmean(cs_std) < 1e-10:
                n_degen += 1
                continue

            panels.append(arr)
            kept_exprs.append(expr)
        except Exception:
            n_failed += 1
            continue

    if verbose:
        print(f"    Done in {time.time()-t0:.1f}s. Kept {len(kept_exprs)}/{len(trees)} "
              f"(failed={n_failed}, degenerate={n_degen})", flush=True)

    if not panels:
        return None, []

    # Stack into (T, N, K)
    K = len(panels)
    Z = np.stack(panels, axis=2)
    return Z, kept_exprs


def rank_standardize_panel(Z_raw: np.ndarray) -> np.ndarray:
    """Cross-sectionally rank-standardize each (t, k) slice to [-0.5, 0.5]."""
    from scipy.stats import rankdata
    T, N, K = Z_raw.shape
    Z = np.zeros_like(Z_raw)
    for t in range(T):
        for k in range(K):
            col = Z_raw[t, :, k]
            valid = np.isfinite(col)
            n_v = valid.sum()
            if n_v < 3:
                continue
            r = rankdata(col[valid], method='average') / n_v - 0.5
            out = np.zeros(N)
            out[valid] = r
            Z[t, :, k] = out
    return Z


# ─────────────────────────────────────────────────────────────────────────────
# AIPT CORE — applied to any (T, N, K) panel with optional RFF wrapper
# ─────────────────────────────────────────────────────────────────────────────

def run_aipt_on_panel(Z_panel_3d: np.ndarray, returns_np: np.ndarray, dates,
                       use_rff: bool, P_rff: int, z: float,
                       rebal_every: int, train_bars: int, min_train_bars: int,
                       seed: int = 42, label: str = "") -> dict:
    """
    Run AIPT pipeline on a (T, N, K) characteristics panel.
    If use_rff=True: applies RFF expansion to P_rff features before ridge-Markowitz.
    If use_rff=False: ridge-Markowitz operates directly on the K characteristics.
    Returns dict with bar_dates, gross/net returns, turnover, OOS Sharpe.
    """
    T, N, K = Z_panel_3d.shape
    print(f"\n  [{label}] T={T}, N={N}, K={K}, "
          f"{'RFF P=' + str(P_rff) if use_rff else 'no RFF'}", flush=True)

    if use_rff:
        theta, gamma = generate_rff_params(K, P_rff, seed=seed)
        P_eff = P_rff
    else:
        P_eff = K

    # ── Pre-compute factor returns + signals
    factor_returns = {}
    sig_cache = {}

    for t in range(T - 1):
        Z_t = Z_panel_3d[t]
        r_t1 = returns_np[t + 1, :]
        valid = (~np.isnan(r_t1)) & (~np.isnan(Z_t).any(axis=1))
        N_t = valid.sum()
        if N_t < 5:
            continue
        if use_rff:
            S_t = compute_rff_signals(Z_t[valid], theta, gamma)
        else:
            S_t = Z_t[valid]
        r_clean = np.nan_to_num(r_t1[valid], nan=0.0)
        F_t1 = (1.0 / np.sqrt(N_t)) * (S_t.T @ r_clean)
        factor_returns[t + 1] = F_t1
        sig_cache[t] = (S_t, valid, N_t)

    # ── Rolling SDF + normalized L/S portfolio
    all_idx = sorted(factor_returns.keys())
    port_returns = []
    turnovers = []
    bar_dates = []
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

        S_t, valid_mask, N_t = sig_cache[sig_bar]
        raw_w = np.zeros(N)
        raw_w[valid_mask] = (1.0 / np.sqrt(N_t)) * (S_t @ lambda_hat)
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

    port_arr = np.array(port_returns)
    to_arr = np.array(turnovers)
    return dict(
        label=label, P=P_eff, K=K,
        bar_dates=bar_dates,
        gross=port_arr,
        turnover=to_arr,
    )


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING (per train / val / test)
# ─────────────────────────────────────────────────────────────────────────────

def split_stats(res: dict, fee_bps: float = 3.0) -> pd.DataFrame:
    """Compute Sharpe / ann ret / turnover / max DD for train / val / test."""
    dates_full = pd.DatetimeIndex(res["bar_dates"])
    gross = pd.Series(res["gross"], index=dates_full)
    to    = pd.Series(res["turnover"], index=dates_full)
    fee_per_bar = to * (fee_bps / 10_000) * 2
    net = gross - fee_per_bar

    splits = {
        "TRAIN (IS)":  (None,         OOS_START),
        "VAL   (OOS)": (OOS_START,    VAL_END),
        "TEST  (OOS)": (VAL_END,      None),
    }

    rows = []
    for split, (s, e) in splits.items():
        sl = slice(s, e)
        g_s, n_s, t_s = gross.loc[sl], net.loc[sl], to.loc[sl]
        n_bars = len(g_s)
        if n_bars < 30:
            rows.append((split, np.nan, np.nan, np.nan, np.nan, np.nan, n_bars))
            continue
        sr_g = g_s.mean() / g_s.std(ddof=1) * np.sqrt(BARS_PER_YEAR) if g_s.std() > 1e-12 else 0.0
        sr_n = n_s.mean() / n_s.std(ddof=1) * np.sqrt(BARS_PER_YEAR) if n_s.std() > 1e-12 else 0.0
        ann_ret_n = n_s.mean() * BARS_PER_YEAR * 100
        cum = n_s.cumsum()
        max_dd = float((cum - cum.cummax()).min())
        rows.append((split, sr_g, sr_n, ann_ret_n, t_s.mean(), max_dd, n_bars))

    return pd.DataFrame(rows, columns=["split", "sr_gross", "sr_net",
                                        "ann_ret_net%", "avg_to", "max_dd_net",
                                        "n_bars"])


def print_summary(all_results: dict, fee_bps: float):
    print(f"\n{'='*100}")
    print(f"  RESULTS SUMMARY  (fees={fee_bps:.1f}bps)")
    print(f"{'='*100}")
    print(f"  {'Config':<24} {'Split':<14} "
          f"{'SR_g':>7} {'SR_n':>7} {'AnnR%':>8} {'TO':>6} {'MaxDD':>8} {'Bars':>6}")
    print(f"  {'-'*95}")
    for cfg_name, res in all_results.items():
        df = split_stats(res, fee_bps=fee_bps)
        for _, row in df.iterrows():
            sg = f"{row['sr_gross']:+7.2f}" if not np.isnan(row['sr_gross']) else "    --"
            sn = f"{row['sr_net']:+7.2f}"   if not np.isnan(row['sr_net'])   else "    --"
            ar = f"{row['ann_ret_net%']:+8.1f}" if not np.isnan(row['ann_ret_net%']) else "      --"
            to = f"{row['avg_to']:6.3f}"   if not np.isnan(row['avg_to'])   else "    --"
            dd = f"{row['max_dd_net']:+8.3f}" if not np.isnan(row['max_dd_net']) else "      --"
            print(f"  {cfg_name:<24} {row['split']:<14} "
                  f"{sg} {sn} {ar} {to} {dd} {int(row['n_bars']):>6}")
        print()


def plot_comparison(all_results: dict, fee_bps: float, save_dir: Path = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if save_dir is None:
        save_dir = RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    colors = {
        "PROD-RFF":     "#2196F3",
        "RAND-TREES":   "#FF5722",
        "TREES+RFF":    "#9C27B0",
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=False)
    fig.patch.set_facecolor("#16213e")

    splits = [
        ("TRAIN (IS)",  None,        OOS_START),
        ("VAL   (OOS)", OOS_START,   VAL_END),
        ("TEST  (OOS)", VAL_END,     None),
    ]

    for ax, (split_name, s, e) in zip(axes, splits):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("#333")

        for cfg_name, res in all_results.items():
            dates = pd.DatetimeIndex(res["bar_dates"])
            gross = pd.Series(res["gross"], index=dates)
            to    = pd.Series(res["turnover"], index=dates)
            net   = gross - to * (fee_bps / 10_000) * 2
            sl    = slice(s, e)
            cum   = net.loc[sl].cumsum()
            if len(cum) < 30:
                continue
            sr_n = net.loc[sl].mean() / net.loc[sl].std(ddof=1) * np.sqrt(BARS_PER_YEAR)
            ax.plot(cum.index, cum.values, linewidth=1.6,
                    color=colors.get(cfg_name, "white"),
                    label=f"{cfg_name}  SR_net={sr_n:+.2f}")

        ax.set_title(split_name, color="white", fontsize=12, fontweight="bold")
        ax.set_xlabel("Date", color="white")
        ax.set_ylabel("Cumulative Net Return", color="white")
        ax.legend(facecolor="#1a1a2e", edgecolor="#555",
                  labelcolor="white", fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.15)

    plt.suptitle(
        f"AIPT Feature-Space Comparison  |  fees={fee_bps:.0f}bps  |  rebal/{REBAL_EVERY}",
        color="white", fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out = save_dir / "compare_random_trees.png"
    plt.savefig(out, dpi=140, facecolor=fig.get_facecolor())
    print(f"\n  Plot saved to {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trees",  type=int, default=N_TREES)
    ap.add_argument("--prod-p",   type=int, default=PROD_P)
    ap.add_argument("--tree-rff-p", type=int, default=TREE_RFF_P)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--fees",     type=float, default=3.0)
    args = ap.parse_args()

    print("=" * 80)
    print(f"  AIPT FEATURE-SPACE COMPARISON")
    print(f"  Configs: PROD-RFF (P={args.prod_p}), RAND-TREES (K={args.n_trees}),")
    print(f"           TREES+RFF (K={args.n_trees}, P={args.tree_rff_p})")
    print(f"  OOS split: train < {OOS_START} <= val < {VAL_END} <= test")
    print(f"  fees: {args.fees:.1f}bps   rebal_every: {REBAL_EVERY}")
    print("=" * 80)

    # ── Load data once
    print("\nLoading data...", flush=True)
    matrices, universe, tickers = load_data()
    close = matrices["close"]
    returns_pct = matrices["returns_pct"]
    dates = close.index
    T = len(dates)
    N = len(tickers)
    returns_np = returns_pct.reindex(columns=tickers).values.astype(np.float64)
    print(f"  T={T} bars, N={N} tickers ({dates[0]} to {dates[-1]})", flush=True)

    # ── 1. Build PROD characteristics panel
    print("\n[1/3] Building PROD-RFF panel (24 hand-picked characteristics)...",
          flush=True)
    panel_start = 0
    Z_prod_dict, prod_chars = build_characteristics_panel(matrices, tickers, panel_start, T)
    Z_prod_3d = np.zeros((T, N, len(prod_chars)), dtype=np.float64)
    for i in range(T):
        if i in Z_prod_dict:
            Z_prod_3d[i] = Z_prod_dict[i]
    print(f"  Built ({len(prod_chars)} characteristics)", flush=True)

    # ── 2. Generate + evaluate random trees
    print(f"\n[2/3] Generating {args.n_trees} random DSL trees (seed={args.seed})...",
          flush=True)
    trees = generate_random_trees(args.n_trees, seed=args.seed)
    print(f"  Generated {len(trees)} unique expressions. Sample:", flush=True)
    for ex in trees[:3]:
        print(f"    {ex[:120]}{'...' if len(ex) > 120 else ''}", flush=True)

    print(f"\n  Evaluating trees on {N} tickers x {T} bars...", flush=True)
    Z_trees_raw, kept_trees = evaluate_trees(trees, matrices, tickers, verbose=True)
    if Z_trees_raw is None:
        print("  ERROR: no valid trees produced. Exiting.")
        return
    K_trees = Z_trees_raw.shape[2]

    print(f"\n  Rank-standardizing panel ({T} bars x {N} assets x {K_trees} trees)...",
          flush=True)
    t0 = time.time()
    Z_trees_3d = rank_standardize_panel(Z_trees_raw)
    print(f"    Done in {time.time()-t0:.1f}s", flush=True)

    # ── 3. Run all three AIPT configs
    print("\n[3/3] Running AIPT pipeline on each feature space...", flush=True)
    all_results = {}

    all_results["PROD-RFF"] = run_aipt_on_panel(
        Z_prod_3d, returns_np, dates,
        use_rff=True, P_rff=args.prod_p, z=RIDGE_Z,
        rebal_every=REBAL_EVERY, train_bars=TRAIN_BARS, min_train_bars=MIN_TRAIN_BARS,
        seed=args.seed, label=f"PROD-RFF (24ch + P={args.prod_p})",
    )

    all_results["RAND-TREES"] = run_aipt_on_panel(
        Z_trees_3d, returns_np, dates,
        use_rff=False, P_rff=0, z=RIDGE_Z,
        rebal_every=REBAL_EVERY, train_bars=TRAIN_BARS, min_train_bars=MIN_TRAIN_BARS,
        seed=args.seed, label=f"RAND-TREES (K={K_trees})",
    )

    all_results["TREES+RFF"] = run_aipt_on_panel(
        Z_trees_3d, returns_np, dates,
        use_rff=True, P_rff=args.tree_rff_p, z=RIDGE_Z,
        rebal_every=REBAL_EVERY, train_bars=TRAIN_BARS, min_train_bars=MIN_TRAIN_BARS,
        seed=args.seed, label=f"TREES+RFF (K={K_trees} -> P={args.tree_rff_p})",
    )

    # ── Report
    print_summary(all_results, fee_bps=args.fees)
    plot_comparison(all_results, fee_bps=args.fees)

    # Save trees to disk for inspection
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(kept_trees, name="expression").to_csv(
        out_dir / "compare_random_trees_exprs.csv", index=False
    )
    print(f"  Tree expressions saved to {out_dir / 'compare_random_trees_exprs.csv'}")


if __name__ == "__main__":
    main()
