"""
Random-subsets-of-alphas → Kelly/AIPT (DKKM Random Fourier Features + ridge-Markowitz)
   - With and without asset-level QP execution layer (Isichenko Eq 6.4)
   - Same N grid + 8 random seeds per N as sweep_alpha_thorough.py
   - 5 bps fees (KuCoin 4h validation, 2024-09 → 2025-03)

Pipeline:
  1. Random N alphas → rank-normalized Z_panel (N × T)
  2. AIPT: random Fourier features S, ridge-Markowitz, lambda_hat → weights w_t per bar
  3. Track weights as (T, N_tickers) DataFrame
  4. Two evaluations:
        Kelly (no QP):  simulate(w, fees=5bps)
        Kelly + QP:     simulate(qp_execution(w), fees=5bps)
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

import eval_portfolio as ep

# ── AIPT/Kelly hyperparameters (matched to crypto 4h work) ─────────────────
P_RFF          = 1000          # number of random Fourier features
TRAIN_BARS     = 4380          # ~2 years of 4h
MIN_TRAIN_BARS = 1000
REBAL_EVERY    = 12            # rebalance every 12 bars (~2 days)
Z_RIDGE        = 1e-3
GAMMA_GRID     = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
OOS_START_DATE = "2024-09-01"  # matches VAL_START
SIM_FEES_BPS   = 5.0

# Sweep schedule
N_GRID  = [5, 10, 15, 20, 30, 50, 75, 100, 150]
N_SEEDS = 8
RNG_SEEDS = list(range(1000, 1000 + N_SEEDS))

OUT_DIR = ROOT / "data/aipt_results/kelly_alphas_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_full_panel():
    """Load full kucoin 4h matrices for [TRAIN..VAL_END] + raw alpha signals."""
    # Use eval_alpha-style full data load
    mat_dir = ROOT / f"data/kucoin_cache/matrices/{ep.INTERVAL}"
    matrices_full = {}
    for fp in sorted(mat_dir.glob("*.parquet")):
        df = pd.read_parquet(fp)
        matrices_full[fp.stem] = df
    close = matrices_full["close"]
    returns_pct = close.pct_change()

    universe_path = ROOT / f"data/kucoin_cache/universes/{ep.UNIVERSE}_{ep.INTERVAL}.parquet"
    universe_df = pd.read_parquet(universe_path)
    cov_frac = universe_df.sum(axis=0) / len(universe_df)
    valid_tickers = sorted(cov_frac[cov_frac > ep.COVERAGE_CUTOFF].index.tolist())
    tickers = sorted(set(close.columns) & set(valid_tickers))
    matrices_full = {k: df.reindex(columns=tickers) for k, df in matrices_full.items()}
    close = matrices_full["close"]
    returns_pct = close.pct_change()
    universe_df = universe_df.reindex(columns=tickers)

    # Load all 190 raw alpha signals (compute on full data)
    import sqlite3, os
    os.chdir(ROOT)
    from src.operators.fastexpression import FastExpressionEngine
    con = sqlite3.connect(str(ROOT / "data/alphas.db"))
    rows = con.execute(
        "SELECT id, expression FROM alphas WHERE archived=0 AND interval=? AND universe=? ORDER BY id",
        (ep.INTERVAL, ep.UNIVERSE)).fetchall()
    con.close()
    engine = FastExpressionEngine(data_fields=matrices_full)
    raw_signals = {}
    for aid, expr in rows:
        try:
            sig = engine.evaluate(expr)
            if sig is not None and not sig.empty:
                raw_signals[aid] = sig.reindex(columns=tickers)
        except Exception:
            pass
    return matrices_full, close, returns_pct, universe_df, tickers, raw_signals


def _build_z_panel(alpha_dfs_subset, tickers, dates, start_bar, end_bar):
    """Build (T_window) × (N_tickers, D_alphas) rank-normalized Z panel."""
    N = len(tickers)
    D = len(alpha_dfs_subset)
    panel = {}
    for t in range(start_bar, end_bar):
        Z = np.full((N, D), np.nan)
        for j, adf in enumerate(alpha_dfs_subset):
            try:
                Z[:, j] = adf.iloc[t].reindex(tickers).values.astype(np.float64)
            except Exception:
                pass
        for j in range(D):
            col = Z[:, j]
            ok = ~np.isnan(col)
            if ok.sum() < 3:
                Z[:, j] = 0.0
                continue
            r = rankdata(col[ok], method="average") / ok.sum() - 0.5
            Z[ok, j] = r
            Z[~ok, j] = 0.0
        panel[t] = Z
    return panel, D


def _run_aipt_weights(P, Z_panel, close_vals, start_bar, T_total, oos_start_idx, D, tickers, seed):
    """Run AIPT bar-by-bar, return (T x N) weight DataFrame for the OOS period."""
    rng = np.random.default_rng(seed)
    n_pairs = P // 2
    theta = rng.standard_normal((n_pairs, D))
    gamma = rng.choice(GAMMA_GRID, size=n_pairs)

    fr_history = []
    lambda_hat = None
    bars_since_rebal = REBAL_EVERY
    N = close_vals.shape[1]
    weights_arr = np.zeros((T_total, N))

    for t in range(start_bar, T_total - 1):
        Z_t = Z_panel[t]
        proj = (Z_t @ theta.T) * gamma[None, :]
        S_t = np.empty((Z_t.shape[0], P))
        S_t[:, 0::2] = np.sin(proj)
        S_t[:, 1::2] = np.cos(proj)

        c_t = close_vals[t]
        c_t1 = close_vals[t + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            R_t1 = np.where(c_t > 0, (c_t1 - c_t) / c_t, 0.0)
        R_t1 = np.nan_to_num(R_t1, nan=0.0, posinf=0.0, neginf=0.0)
        valid = (~np.isnan(Z_t).any(axis=1) & ~np.isnan(c_t) & ~np.isnan(c_t1) & (c_t > 0))
        N_t = int(valid.sum())
        if N_t < 5:
            continue

        S_v = S_t[valid]
        R_v = R_t1[valid]
        F_t1 = (1.0 / np.sqrt(N_t)) * (S_v.T @ R_v)
        fr_history.append((t + 1, F_t1))

        if t + 1 < oos_start_idx:
            continue
        if bars_since_rebal >= REBAL_EVERY or lambda_hat is None:
            cutoff = (t + 1) - TRAIN_BARS
            train = [fr for (idx, fr) in fr_history if cutoff <= idx < (t + 1)]
            if len(train) < MIN_TRAIN_BARS:
                continue
            F_train = np.vstack(train)
            T_tr, P_tr = F_train.shape
            if P_tr <= T_tr:
                FF = (F_train.T @ F_train) / T_tr
                A = Z_RIDGE * np.eye(P_tr) + FF
                lambda_hat = np.linalg.solve(A, F_train.mean(axis=0))
            else:
                mu = F_train.mean(axis=0)
                FFT = F_train @ F_train.T
                A_T = Z_RIDGE * T_tr * np.eye(T_tr) + FFT
                inv_F_mu = np.linalg.solve(A_T, F_train @ mu)
                lambda_hat = (mu - F_train.T @ inv_F_mu) / Z_RIDGE
            bars_since_rebal = 0

        raw_w = np.zeros(Z_t.shape[0])
        raw_w[valid] = (1.0 / np.sqrt(N_t)) * (S_v @ lambda_hat)
        abs_sum = np.abs(raw_w).sum()
        if abs_sum < 1e-12:
            bars_since_rebal += 1
            continue
        w_norm = raw_w / abs_sum
        weights_arr[t + 1] = w_norm
        bars_since_rebal += 1

    return weights_arr


def main():
    print("=" * 80)
    print("KELLY/AIPT × ALPHA-SUBSET SWEEP — with and without QP execution")
    print(f"  N grid: {N_GRID}  ×  {N_SEEDS} random seeds")
    print(f"  Fees:   {SIM_FEES_BPS} bps   |   Rebal every {REBAL_EVERY} bars   |   P_RFF={P_RFF}")
    print("=" * 80)

    print("\n[1/2] Loading full matrices + 190 raw alpha signals...")
    t0 = time.time()
    matrices, close, returns_pct, universe, tickers, raw_signals = _load_full_panel()
    aids = sorted(raw_signals.keys())
    dates = close.index
    close_vals = close.values.astype(np.float64)
    T_total = len(dates)
    oos_start_idx = next(i for i, d in enumerate(dates) if str(d) >= OOS_START_DATE)
    start_bar = max(1, oos_start_idx - TRAIN_BARS - 10)
    print(f"  T_total={T_total}, N_tickers={len(tickers)}, alphas_loaded={len(aids)}")
    print(f"  OOS start: {dates[oos_start_idx].date()}, start_bar: {dates[start_bar].date()}")
    print(f"  ({time.time()-t0:.1f}s)")

    out_path = OUT_DIR / "kelly_alphas_metrics.csv"
    if out_path.exists():
        existing = pd.read_csv(out_path)
        rows = existing.to_dict("records")
        done = set((int(r["n_alphas"]), int(r["seed"])) for _, r in existing.iterrows())
        print(f"  RESUME: {len(rows)} rows loaded, skip {len(done)} (N,seed) pairs\n")
    else:
        rows, done = [], set()

    print(f"\n[2/2] Sweep: {len(N_GRID) * N_SEEDS} configs total")
    for n in N_GRID:
        if n > len(aids): continue
        per_n = {"kelly_no_qp": [], "kelly_qp": []}
        for seed in RNG_SEEDS:
            if (n, seed) in done:
                # Pull from cache for printing
                for variant in per_n:
                    sub = [r for r in rows if r["n_alphas"]==n and r["seed"]==seed and r["variant"]==variant]
                    if sub:
                        per_n[variant].append(float(sub[0]["sharpe"]))
                continue
            rng = np.random.default_rng(seed)
            chosen = sorted(rng.choice(aids, size=n, replace=False).tolist())
            alpha_dfs_subset = [raw_signals[aid] for aid in chosen]

            t1 = time.time()
            Z_panel, D = _build_z_panel(alpha_dfs_subset, tickers, dates, start_bar, T_total)
            t2 = time.time()
            weights_arr = _run_aipt_weights(P_RFF, Z_panel, close_vals,
                                              start_bar, T_total, oos_start_idx,
                                              D, tickers, seed)
            t3 = time.time()
            weights_df = pd.DataFrame(weights_arr, index=dates, columns=tickers)

            # Variant 1: Kelly no QP — direct AIPT weights
            sim_no_qp = ep.simulate(weights_df, returns_pct, close, universe,
                                     max_wt=ep.MAX_WEIGHT, fees_bps=SIM_FEES_BPS)
            t4 = time.time()
            # Variant 2: Kelly + QP execution
            P_qp_df = ep._qp_execution_layer(
                target_df=weights_df, returns_pct=returns_pct,
                max_wt=ep.MAX_WEIGHT, rebal_every=1, cov_lookback=120,
                track_aversion=1.0, risk_aversion=0.0,
                tc_bps=SIM_FEES_BPS, verbose=False,
            )
            sim_qp = ep.simulate(P_qp_df, returns_pct, close, universe,
                                  max_wt=ep.MAX_WEIGHT, fees_bps=SIM_FEES_BPS)
            t5 = time.time()

            for variant, sim in [("kelly_no_qp", sim_no_qp), ("kelly_qp", sim_qp)]:
                rows.append({
                    "n_alphas": n, "seed": seed, "variant": variant,
                    "sharpe":   float(sim.sharpe),
                    "ann_ret":  float(sim.returns_ann),
                    "turnover": float(sim.turnover),
                    "max_dd":   float(sim.max_drawdown),
                    "elapsed_s": (t5 - t1) if variant == "kelly_qp" else (t4 - t1),
                })
                per_n[variant].append(float(sim.sharpe))
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"  N={n:>3}  seed={seed}  no_qp={sim_no_qp.sharpe:+.3f}  qp={sim_qp.sharpe:+.3f}  "
                  f"(Z {t2-t1:.0f}s, AIPT {t3-t2:.0f}s, QP {t5-t4:.0f}s)", flush=True)

        # Per-N summary
        print(f"  --- N={n:>3} summary across {N_SEEDS} seeds ---")
        for variant in ["kelly_no_qp", "kelly_qp"]:
            s = pd.Series(per_n[variant])
            if len(s) > 1:
                print(f"     {variant:<14} mean={s.mean():+.3f}  median={s.median():+.3f}  "
                      f"std={s.std():.3f}  min={s.min():+.3f}  max={s.max():+.3f}  range={s.max()-s.min():.2f}")
        print()

    # Final plot
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    colors = {"kelly_no_qp": "#1f77b4", "kelly_qp": "#d62728"}
    for ax, metric, ylabel in [(axes[0], "sharpe", "Sharpe"), (axes[1], "turnover", "Turnover")]:
        for variant in ["kelly_no_qp", "kelly_qp"]:
            sub = df[df.variant == variant]
            agg = sub.groupby("n_alphas")[metric].agg(["mean","std","median"]).reset_index()
            ax.errorbar(agg["n_alphas"], agg["mean"], yerr=agg["std"],
                         marker="o", capsize=3, label=variant, color=colors[variant], lw=1.6)
            ax.scatter(agg["n_alphas"], agg["median"], marker="x", s=40,
                        color=colors[variant], alpha=0.7)
        ax.set_xlabel("# alphas (random subset)")
        ax.set_ylabel(ylabel + (" (mean ± std, x=median)"))
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Kelly/AIPT on alpha subsets — VAL {ep.VAL_START} → {ep.VAL_END}, {SIM_FEES_BPS}bps fees",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "kelly_alphas_sweep.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n## DONE — outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
