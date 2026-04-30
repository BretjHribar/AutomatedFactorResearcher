"""
Deep dive on TOP2000 equities AIPT.

Section A: 4-seed average of the baseline (P=1000, no smoothing) — measures seed noise.
Section B: Hyperparameter explorations on TOP2000:
  - P sweep:     {1000, 2000, 5000}
  - EWMA-w:      {1.0, 0.75, 0.5, 0.25}
  - Ridge sweep: {1e-4, 1e-3, 1e-2}
For each config: TRAIN (pre-OOS rolling), VAL (first half OOS), TEST (second half OOS).

Reuses TOP2000 Z panel built once.
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
    MATRICES_DIR, UNIVERSES_DIR, RESULTS_DIR, CHAR_NAMES,
    build_Z_panel,
)
from backtest_voc_equities_sweep import load_data_universe

LOG_CSV = RESULTS_DIR / "voc_equities_top2000_deep.csv"


def run_full(P, Z_panel, close_vals, start_bar, T_total, oos_start_idx, D,
             alpha=1.0, ridge=1e-3, seed=42, rebal=REBAL_EVERY):
    """AIPT with optional EWMA-w + ridge + rebal frequency overrides."""
    rng = np.random.default_rng(seed)
    n_pairs = P // 2
    theta = rng.standard_normal((n_pairs, D))
    gamma = rng.choice(GAMMA_GRID, size=n_pairs)

    fr_history, lambda_hat = [], None
    bars_since_rebal = rebal
    prev_w, sm_w = None, None
    rows = []

    for t in range(start_bar, T_total - 1):
        if t not in Z_panel:
            continue
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

        if t + 1 < oos_start_idx + 1 - TRAIN_BARS:
            # before any possible training window — skip
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

        raw_w = np.zeros(Z_t.shape[0])
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

    df = pd.DataFrame(rows)
    df["net_1bps"] = df["gross"] - df["turnover"] * TAKER_BPS / 10000.0 * 2.0
    return df


def split_metrics(df, dates, oos_start_idx):
    """Split df into TRAIN (before OOS), VAL (first half OOS), TEST (second half OOS)."""
    if df.empty:
        return {}
    df = df.copy()
    df["date"] = [dates[i] for i in df["bar_idx"]]
    is_oos = df["bar_idx"] >= oos_start_idx
    df_oos = df[is_oos].reset_index(drop=True)
    df_train = df[~is_oos].reset_index(drop=True)
    n = len(df_oos)
    split = n // 2
    splits = {
        "train": df_train,
        "val":   df_oos.iloc[:split],
        "test":  df_oos.iloc[split:],
    }
    out = {}
    ann = np.sqrt(BARS_PER_YEAR)
    for tag, sub in splits.items():
        nbars = len(sub)
        if nbars < 30 or sub["net_1bps"].std() < 1e-12:
            for k in ["sr_g", "sr_n", "to", "ic_p", "ir_p", "ic_s", "r2", "ncum", "nbars"]:
                out[f"{tag}_{k}"] = np.nan if k != "nbars" else nbars
            continue
        g, nn = sub["gross"].values, sub["net_1bps"].values
        out[f"{tag}_sr_g"] = float(g.mean() / g.std(ddof=1) * ann)
        out[f"{tag}_sr_n"] = float(nn.mean() / nn.std(ddof=1) * ann)
        out[f"{tag}_to"]   = float(sub["turnover"].mean())
        out[f"{tag}_ic_p"] = float(sub["ic_p"].mean())
        out[f"{tag}_ir_p"] = float(sub["ic_p"].mean() / sub["ic_p"].std(ddof=1) * ann)
        out[f"{tag}_ic_s"] = float(sub["ic_s"].mean())
        out[f"{tag}_r2"]   = float(sub["r2"].mean())
        out[f"{tag}_ncum"] = float(nn.sum() * 100)
        out[f"{tag}_nbars"] = nbars
    return out


def fmt_row(name, m):
    return (f"  {name:<28}  TRAIN nSR={m['train_sr_n']:+.2f} TO={m['train_to']*100:4.1f}% IC={m['train_ic_p']:+.4f}  "
            f"VAL nSR={m['val_sr_n']:+.2f} IC={m['val_ic_p']:+.4f}  "
            f"TEST nSR={m['test_sr_n']:+.2f} IC={m['test_ic_p']:+.4f} TO={m['test_to']*100:4.1f}%")


def main():
    overall_t0 = time.time()
    print("=" * 100)
    print("TOP2000 EQUITIES DEEP DIVE  (multi-seed + hyperparam sweep)")
    print("=" * 100)
    print("\nLoading TOP2000...")
    matrices, tickers, dates, close_vals, chars = load_data_universe("TOP2000")
    T_total = len(dates)
    oos_start_idx = next(i for i, d in enumerate(dates) if str(d) >= OOS_START)
    start_bar = max(1, oos_start_idx - TRAIN_BARS - 10)
    print(f"  N={len(tickers)}  T={T_total}  OOS_start={oos_start_idx}  chars={len(chars)}")

    print(f"\nBuilding Z panel for {T_total - start_bar} bars (DELAY=1)...")
    t1 = time.time()
    Z_panel, D = build_Z_panel(matrices, tickers, chars, start_bar, T_total, delay=1)
    print(f"  Z panel D={D} built in {time.time()-t1:.1f}s")

    all_rows = []

    # ── SECTION A: multi-seed baseline ──────────────────────────────────────
    print(f"\n{'─'*100}\n  SECTION A: 4-seed average of baseline (P=1000, no smoothing, ρ=1e-3)\n{'─'*100}")
    seeds = [42, 7, 13, 100]
    seed_rows = []
    for seed in seeds:
        t0 = time.time()
        df = run_full(1000, Z_panel, close_vals, start_bar, T_total, oos_start_idx, D,
                      alpha=1.0, ridge=1e-3, seed=seed)
        m = split_metrics(df, dates, oos_start_idx)
        m["name"] = f"seed={seed}"
        m["P"] = 1000; m["alpha"] = 1.0; m["ridge"] = 1e-3; m["seed"] = seed
        m["minutes"] = (time.time()-t0)/60
        seed_rows.append(m)
        all_rows.append(m)
        pd.DataFrame(all_rows).to_csv(LOG_CSV, index=False)
        print(fmt_row(m["name"], m))

    # Average across seeds
    sd = pd.DataFrame(seed_rows)
    print(f"\n  ── 4-seed mean (±std) ──")
    for tag in ["train", "val", "test"]:
        n_mean = sd[f"{tag}_sr_n"].mean()
        n_std  = sd[f"{tag}_sr_n"].std(ddof=1)
        ic_mean = sd[f"{tag}_ic_p"].mean()
        ic_std  = sd[f"{tag}_ic_p"].std(ddof=1)
        to_mean = sd[f"{tag}_to"].mean()
        print(f"  {tag.upper():<6}: net_SR = {n_mean:+.2f} ± {n_std:.2f}    "
              f"IC = {ic_mean:+.4f} ± {ic_std:.4f}    TO = {to_mean*100:.1f}%")

    # ── SECTION B: hyperparameter exploration (seed=42 only) ────────────────
    print(f"\n{'─'*100}\n  SECTION B: hyperparameter exploration on TOP2000 (seed=42)\n{'─'*100}")

    configs = []
    # P sweep
    for P in [2000, 5000]:
        configs.append({"name": f"P={P}", "P": P, "alpha": 1.0, "ridge": 1e-3, "seed": 42})
    # EWMA-w
    for alpha in [0.75, 0.5, 0.25]:
        configs.append({"name": f"ewma_w(α={alpha})", "P": 1000, "alpha": alpha, "ridge": 1e-3, "seed": 42})
    # Ridge sweep
    for ridge in [1e-4, 1e-2]:
        configs.append({"name": f"ridge={ridge:g}", "P": 1000, "alpha": 1.0, "ridge": ridge, "seed": 42})
    # Rebal frequency
    for reb in [1, 3]:
        configs.append({"name": f"rebal={reb}", "P": 1000, "alpha": 1.0, "ridge": 1e-3, "seed": 42, "rebal": reb})
    # Combo: best from each
    configs.append({"name": "P=2000 + ewma_w=0.5", "P": 2000, "alpha": 0.5, "ridge": 1e-3, "seed": 42})
    configs.append({"name": "ridge=1e-2 + ewma_w=0.5", "P": 1000, "alpha": 0.5, "ridge": 1e-2, "seed": 42})

    for cfg in configs:
        t0 = time.time()
        df = run_full(cfg["P"], Z_panel, close_vals, start_bar, T_total, oos_start_idx, D,
                      alpha=cfg["alpha"], ridge=cfg["ridge"], seed=cfg["seed"],
                      rebal=cfg.get("rebal", REBAL_EVERY))
        m = split_metrics(df, dates, oos_start_idx)
        m.update(cfg)
        m["minutes"] = (time.time()-t0)/60
        all_rows.append(m)
        pd.DataFrame(all_rows).to_csv(LOG_CSV, index=False)
        print(fmt_row(m["name"], m))

    print(f"\n{'='*100}")
    print(f"DONE in {(time.time()-overall_t0)/60:.1f}min  ({len(all_rows)} configs)")
    print(f"CSV: {LOG_CSV}")


if __name__ == "__main__":
    main()
