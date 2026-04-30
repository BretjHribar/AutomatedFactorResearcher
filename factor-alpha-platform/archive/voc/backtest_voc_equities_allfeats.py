"""
TOP2000 equities AIPT using ALL 234 base features in matrices_clean (no manual
char selection, no normalization). Pipeline still rank-normalizes each column
cross-sectionally per bar.

Gamma rescaled by sqrt(24/D) so RFF projection variance stays K-invariant.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from backtest_voc_equities import (
    BARS_PER_YEAR, TRAIN_BARS, MIN_TRAIN_BARS, REBAL_EVERY,
    OOS_START, COVERAGE_CUTOFF, GAMMA_GRID, TAKER_BPS,
    MATRICES_DIR, UNIVERSES_DIR, RESULTS_DIR,
)
from backtest_voc_equities_top2000 import split_metrics, fmt_row

GAMMA_REF_D = 24

# Files we don't want as features (price level used for returns; classifications etc)
EXCLUDE_FEATURES = {
    "close",  # used for returns
}


def load_all_top2000():
    """Load TOP2000 + ALL parquet matrices in matrices_clean (excluding 'close')."""
    uni = pd.read_parquet(UNIVERSES_DIR / "TOP2000.parquet")
    cov = uni.sum(axis=0) / len(uni)
    valid_tickers = sorted(cov[cov > COVERAGE_CUTOFF].index.tolist())

    matrices = {}
    excluded = []
    close_df = pd.read_parquet(MATRICES_DIR / "close.parquet")
    cols_close = [c for c in valid_tickers if c in close_df.columns]
    matrices["close"] = close_df[cols_close]

    for fp in sorted(MATRICES_DIR.glob("*.parquet")):
        if fp.stem in EXCLUDE_FEATURES:
            continue
        try:
            df = pd.read_parquet(fp)
            cols = [c for c in valid_tickers if c in df.columns]
            if not cols:
                excluded.append((fp.stem, "no valid tickers"))
                continue
            df = df[cols]
            if df.isna().all().all():
                excluded.append((fp.stem, "all NaN"))
                continue
            matrices[fp.stem] = df
        except Exception as e:
            excluded.append((fp.stem, f"err: {e}"))

    common_idx = matrices["close"].index
    for k in list(matrices):
        matrices[k] = matrices[k].reindex(index=common_idx)

    tickers = sorted(set(matrices["close"].columns))
    for k in list(matrices):
        matrices[k] = matrices[k].reindex(columns=tickers)

    feature_names = sorted([k for k in matrices if k != "close"])

    print(f"  Loaded {len(feature_names)} feature matrices")
    if excluded:
        print(f"  Excluded {len(excluded)} (showing first 5): {[e[0] for e in excluded[:5]]}")
    return matrices, tickers, feature_names


def run_lazy_z(P, matrices, tickers, feature_names, close_vals,
               start_bar, T_total, oos_start_idx,
               alpha=1.0, ridge=1e-3, seed=42, rebal=REBAL_EVERY,
               delay=1, gamma_scale=None):
    """AIPT with lazy Z construction (no panel cache — D=234 panel would be ~7GB)."""
    D = len(feature_names)
    N = len(tickers)
    if gamma_scale is None:
        gamma_scale = float(np.sqrt(GAMMA_REF_D / D))

    rng = np.random.default_rng(seed)
    n_pairs = P // 2
    theta = rng.standard_normal((n_pairs, D))
    gamma = rng.choice(GAMMA_GRID, size=n_pairs) * gamma_scale
    print(f"    P={P} D={D} gamma_scale={gamma_scale:.3f}", flush=True)

    # Pre-extract numpy arrays for fast row access
    char_arrays = [matrices[cn].values.astype(np.float64) for cn in feature_names]

    fr_history, lambda_hat = [], None
    bars_since_rebal = rebal
    prev_w, sm_w = None, None
    rows = []

    t_loop_start = time.time()
    last_log = t_loop_start

    for t in range(start_bar, T_total - 1):
        z_idx = t - delay
        if z_idx < 0:
            continue

        # Lazy Z build + rank normalize
        Z = np.empty((N, D))
        for j, arr in enumerate(char_arrays):
            Z[:, j] = arr[z_idx]
        for j in range(D):
            col = Z[:, j]
            ok = ~np.isnan(col)
            if ok.sum() < 3:
                Z[:, j] = 0.0
                continue
            r = rankdata(col[ok], method="average") / ok.sum() - 0.5
            Z[ok, j] = r
            Z[~ok, j] = 0.0

        proj = (Z @ theta.T) * gamma[None, :]
        S_t = np.empty((N, P))
        S_t[:, 0::2] = np.sin(proj)
        S_t[:, 1::2] = np.cos(proj)

        R_t1 = (close_vals[t + 1] - close_vals[t]) / close_vals[t]
        R_t1 = np.nan_to_num(R_t1, nan=0.0)

        valid = (~np.isnan(Z).any(axis=1)
                 & ~np.isnan(close_vals[t]) & ~np.isnan(close_vals[t + 1]))
        N_t = int(valid.sum())
        if N_t < 5:
            continue

        S_v, R_v = S_t[valid], R_t1[valid]
        F_t1 = (1.0 / np.sqrt(N_t)) * (S_v.T @ R_v)
        fr_history.append((t + 1, F_t1))

        if t + 1 < oos_start_idx + 1 - TRAIN_BARS:
            continue

        if bars_since_rebal >= rebal or lambda_hat is None:
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

        pred_v = (1.0 / np.sqrt(N_t)) * (S_v @ lambda_hat)

        if pred_v.std() > 1e-12 and R_v.std() > 1e-12:
            ic_p = float(np.corrcoef(pred_v, R_v)[0, 1])
            ic_s = float(np.corrcoef(rankdata(pred_v), rankdata(R_v))[0, 1])
            r2 = ic_p ** 2
        else:
            ic_p = ic_s = r2 = 0.0

        raw_w = np.zeros(N)
        raw_w[valid] = pred_v
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

        rows.append({
            "bar_idx": t + 1, "gross": port_ret, "turnover": to,
            "ic_p": ic_p, "ic_s": ic_s, "r2": r2,
        })

        # Periodic progress log
        now = time.time()
        if now - last_log > 30:
            done = t - start_bar + 1
            total = T_total - 1 - start_bar
            print(f"    progress: {done}/{total} bars ({100*done/total:.0f}%) in {now-t_loop_start:.0f}s", flush=True)
            last_log = now

    df = pd.DataFrame(rows)
    df["net_1bps"] = df["gross"] - df["turnover"] * TAKER_BPS / 10000.0 * 2.0
    return df


def main():
    overall_t0 = time.time()
    print("=" * 100)
    print("TOP2000 EQUITIES — ALL FEATURES (no manual char selection)")
    print("=" * 100)

    print("\nLoading TOP2000 + all matrices...")
    t0 = time.time()
    matrices, tickers, feature_names = load_all_top2000()
    close_vals = matrices["close"].values
    dates = matrices["close"].index
    T_total = len(dates)
    oos_start_idx = next(i for i, d in enumerate(dates) if str(d) >= OOS_START)
    start_bar = max(1, oos_start_idx - TRAIN_BARS - 10)
    D = len(feature_names)
    N = len(tickers)
    print(f"  N={N}  T={T_total}  D={D}  OOS_start={oos_start_idx}  ({time.time()-t0:.1f}s)")

    all_rows = []
    LOG_CSV = RESULTS_DIR / "voc_equities_top2000_allfeats.csv"

    # Sweep P at D=233 — DKKM uses P/D ≈ 2700; our P=1000 has P/D=4.3 (way undersized).
    # Push P up to match the per-D coverage of our 24-char baseline (P/D ≈ 42 → need P=10000).
    configs = [
        {"name": "P=5000",  "P": 5000,  "alpha": 1.0, "ridge": 1e-3, "seed": 42},
        {"name": "P=10000", "P": 10000, "alpha": 1.0, "ridge": 1e-3, "seed": 42},
        {"name": "P=20000", "P": 20000, "alpha": 1.0, "ridge": 1e-3, "seed": 42},
        # Cross-check: same P=10000 with stronger ridge (regularize the bigger model)
        {"name": "P=10000 ridge=1e-2", "P": 10000, "alpha": 1.0, "ridge": 1e-2, "seed": 42},
    ]

    for cfg in configs:
        print(f"\n--- {cfg['name']} ---")
        t0 = time.time()
        df = run_lazy_z(cfg["P"], matrices, tickers, feature_names, close_vals,
                        start_bar, T_total, oos_start_idx,
                        alpha=cfg["alpha"], ridge=cfg["ridge"], seed=cfg["seed"])
        print(f"    finished in {(time.time()-t0)/60:.1f} min, {len(df)} bars")
        m = split_metrics(df, dates, oos_start_idx)
        m.update(cfg)
        m["D"] = D
        m["minutes"] = (time.time()-t0)/60
        all_rows.append(m)
        pd.DataFrame(all_rows).to_csv(LOG_CSV, index=False)
        print(fmt_row(cfg["name"], m))

    print(f"\n{'='*100}")
    print(f"DONE in {(time.time()-overall_t0)/60:.1f} min")
    print(f"CSV: {LOG_CSV}")


if __name__ == "__main__":
    main()
