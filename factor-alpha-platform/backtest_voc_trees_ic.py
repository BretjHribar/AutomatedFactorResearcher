"""
Re-run the tree-feature AIPT sweep, this time also capturing per-bar IC and R²
for both VAL and TEST splits. Uses cached tree panels so no regeneration.

For each (K, mode):
  - rebuild lambda via the same Markowitz pipeline used in backtest_voc_trees.py
  - at each OOS bar t:
      pred_v   = (1/√N) Sₜ λ                  (N_t-vector)
      realized = R_{t+1}                       (N_t-vector)
      ic_p_t   = pearson(pred_v, realized)
      ic_s_t   = spearman(pred_v, realized)
      r2_t     = ic_p_t²
  - Aggregate over VAL / TEST splits (mean, std, IR)

Streams a master CSV (`trees_sweep_ic_results.csv`) and prints each row as it lands.
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

from compare_random_trees import VAL_END, RIDGE_Z, REBAL_EVERY
from aipt_kucoin import (
    load_data, OOS_START, TRAIN_BARS, MIN_TRAIN_BARS, BARS_PER_YEAR, RESULTS_DIR,
    generate_rff_params, compute_rff_signals, estimate_ridge_markowitz,
)

K_GRID    = [500, 1000, 2000]                       # cached panels
P_GRID    = [2000, 5000, 10000, 20000, 40000]       # RFF size sweep
SEED      = 42
FEE_BPS   = 3.0
GAMMA_REF_D = 24
LOG_CSV   = RESULTS_DIR / "trees_sweep_ic_results.csv"


def panel_cache_path(K: int, seed: int) -> Path:
    return RESULTS_DIR / f"trees_panel_n{K}_seed{seed}.npz"


def load_cached_panel(K: int, seed: int):
    cache = panel_cache_path(K, seed)
    if not cache.exists():
        return None
    z = np.load(cache, allow_pickle=True)
    return z["Z"]


def run_with_ic(Z_panel_3d, returns_np, dates,
                use_rff: bool, P_rff: int, z: float,
                rebal_every: int, train_bars: int, min_train_bars: int,
                seed: int, label: str):
    """Same pipeline as run_aipt_low_mem but also captures per-bar IC + R²."""
    T, N, K = Z_panel_3d.shape
    if use_rff:
        theta, gamma = generate_rff_params(K, P_rff, seed=seed)
        gamma = gamma * np.sqrt(GAMMA_REF_D / K)
        P_eff = P_rff
    else:
        theta = gamma = None
        P_eff = K
    print(f"  [{label}] T={T} N={N} K={K}  P_eff={P_eff}", flush=True)

    def project(Z_v):
        return compute_rff_signals(Z_v, theta, gamma) if use_rff else Z_v

    # F-pass
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

    # Portfolio + IC capture
    t0 = time.time()
    all_idx = sorted(factor_returns.keys())
    rows = []
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
        if sig_bar not in valid_cache or lambda_hat is None:
            bars_since += 1
            continue

        valid, N_t = valid_cache[sig_bar]
        S_v = project(Z_panel_3d[sig_bar][valid])
        pred_v = (1.0 / np.sqrt(N_t)) * (S_v @ lambda_hat)

        # IC + R² (cross-sectional)
        r_t1_full = returns_np[oos_t, :]
        r_v = np.nan_to_num(r_t1_full[valid], nan=0.0)
        if pred_v.std() > 1e-12 and r_v.std() > 1e-12:
            ic_p = float(np.corrcoef(pred_v, r_v)[0, 1])
            ic_s = float(np.corrcoef(rankdata(pred_v), rankdata(r_v))[0, 1])
            r2_v = ic_p ** 2
        else:
            ic_p = ic_s = r2_v = 0.0

        raw_w = np.zeros(N)
        raw_w[valid] = pred_v
        abs_sum = np.abs(raw_w).sum()
        if abs_sum < 1e-12:
            bars_since += 1
            continue
        w_norm = raw_w / abs_sum

        port_ret = float(w_norm @ np.nan_to_num(r_t1_full, nan=0.0))
        to = float(np.abs(w_norm - prev_w).sum() / 2.0) if prev_w is not None else 0.0
        prev_w = w_norm.copy()
        bars_since += 1

        rows.append({
            "date": dates[oos_t], "gross": port_ret, "turnover": to,
            "ic_p": ic_p, "ic_s": ic_s, "r2": r2_v,
        })

    print(f"    Portfolio pass: {len(rows)} bars in {time.time()-t0:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df["net"] = df["gross"] - df["turnover"] * (FEE_BPS / 10_000) * 2
    return df


def split_metrics(df: pd.DataFrame) -> dict:
    """Compute SR, IC, IR, R² for VAL and TEST splits."""
    if df.empty:
        return {}
    dates = pd.DatetimeIndex(df["date"])
    out = {}
    splits = {
        "val":  (OOS_START, VAL_END),
        "test": (VAL_END,   None),
    }
    ann = np.sqrt(BARS_PER_YEAR)
    for tag, (s, e) in splits.items():
        mask = pd.Series(True, index=range(len(df)))
        if s is not None:
            mask &= dates >= pd.Timestamp(s)
        if e is not None:
            mask &= dates <  pd.Timestamp(e)
        sub = df[mask.values]
        n = len(sub)
        if n < 30:
            for k in ["sr_g", "sr_n", "to", "ic_p", "ic_s", "ir_p", "ir_s",
                      "r2_mean", "r2_med", "n"]:
                out[f"{tag}_{k}"] = np.nan
            continue
        g, nn = sub["gross"].values, sub["net"].values
        out[f"{tag}_sr_g"] = float(g.mean() / g.std(ddof=1) * ann) if g.std() > 1e-12 else 0.0
        out[f"{tag}_sr_n"] = float(nn.mean() / nn.std(ddof=1) * ann) if nn.std() > 1e-12 else 0.0
        out[f"{tag}_to"]   = float(sub["turnover"].mean())
        out[f"{tag}_ic_p"] = float(sub["ic_p"].mean())
        out[f"{tag}_ic_s"] = float(sub["ic_s"].mean())
        out[f"{tag}_ir_p"] = float(sub["ic_p"].mean() / sub["ic_p"].std(ddof=1) * ann) if sub["ic_p"].std() > 1e-12 else 0.0
        out[f"{tag}_ir_s"] = float(sub["ic_s"].mean() / sub["ic_s"].std(ddof=1) * ann) if sub["ic_s"].std() > 1e-12 else 0.0
        out[f"{tag}_r2_mean"] = float(sub["r2"].mean())
        out[f"{tag}_r2_med"]  = float(sub["r2"].median())
        out[f"{tag}_n"]    = int(n)
    return out


def main():
    overall_t0 = time.time()
    print("=" * 100, flush=True)
    print("Tree-feature AIPT sweep with IC + R²", flush=True)
    print(f"  K_GRID = {K_GRID}", flush=True)
    print(f"  P_GRID = {P_GRID}", flush=True)
    print("=" * 100, flush=True)

    print("\nLoading market data...", flush=True)
    matrices, universe, tickers = load_data()
    close = matrices["close"]
    returns_pct = matrices["returns_pct"]
    dates = close.index
    T, N = len(dates), len(tickers)
    returns_np = returns_pct.reindex(columns=tickers).values.astype(np.float64)
    print(f"  T={T} bars, N={N} tickers", flush=True)

    all_rows = []

    for K in K_GRID:
        Z_3d = load_cached_panel(K, SEED)
        if Z_3d is None:
            print(f"\n[K={K}] CACHE MISSING — skip", flush=True)
            continue
        K_actual = Z_3d.shape[2]
        print(f"\n#### K={K_actual} (cache n={K}) ####", flush=True)

        # Mode A: ridge-only
        t0 = time.time()
        df = run_with_ic(Z_3d, returns_np, dates,
                         use_rff=False, P_rff=0, z=RIDGE_Z,
                         rebal_every=REBAL_EVERY, train_bars=TRAIN_BARS,
                         min_train_bars=MIN_TRAIN_BARS, seed=SEED,
                         label=f"TREES-only K={K_actual}")
        m = split_metrics(df)
        m.update({"label": f"TREES-only K={K_actual}", "K": K_actual,
                  "P_eff": K_actual, "mode": "ridge_only", "minutes": (time.time()-t0)/60})
        all_rows.append(m)
        pd.DataFrame(all_rows).to_csv(LOG_CSV, index=False)
        print(f"  >>> {m['label']}", flush=True)
        print(f"      VAL : SR_n={m['val_sr_n']:+.2f}  IC_p={m['val_ic_p']:+.5f}  "
              f"IR_p={m['val_ir_p']:+.2f}  R²={m['val_r2_mean']:.5f}", flush=True)
        print(f"      TEST: SR_n={m['test_sr_n']:+.2f}  IC_p={m['test_ic_p']:+.5f}  "
              f"IR_p={m['test_ir_p']:+.2f}  R²={m['test_r2_mean']:.5f}", flush=True)

        # Mode B: RFF sweep
        for P_rff in P_GRID:
            t0 = time.time()
            df = run_with_ic(Z_3d, returns_np, dates,
                             use_rff=True, P_rff=P_rff, z=RIDGE_Z,
                             rebal_every=REBAL_EVERY, train_bars=TRAIN_BARS,
                             min_train_bars=MIN_TRAIN_BARS, seed=SEED,
                             label=f"TREES+RFF K={K_actual} P={P_rff}")
            m = split_metrics(df)
            m.update({"label": f"TREES+RFF K={K_actual} P={P_rff}",
                      "K": K_actual, "P_eff": P_rff, "mode": f"rff_P{P_rff}",
                      "minutes": (time.time()-t0)/60})
            all_rows.append(m)
            pd.DataFrame(all_rows).to_csv(LOG_CSV, index=False)
            print(f"  >>> {m['label']}", flush=True)
            print(f"      VAL : SR_n={m['val_sr_n']:+.2f}  IC_p={m['val_ic_p']:+.5f}  "
                  f"IR_p={m['val_ir_p']:+.2f}  R²={m['val_r2_mean']:.5f}", flush=True)
            print(f"      TEST: SR_n={m['test_sr_n']:+.2f}  IC_p={m['test_ic_p']:+.5f}  "
                  f"IR_p={m['test_ir_p']:+.2f}  R²={m['test_r2_mean']:.5f}", flush=True)

        del Z_3d

    print(f"\n{'='*100}", flush=True)
    print(f"DONE — {(time.time()-overall_t0)/60:.1f}min  ({len(all_rows)} configs)", flush=True)
    print(f"Log: {LOG_CSV}", flush=True)


if __name__ == "__main__":
    main()
