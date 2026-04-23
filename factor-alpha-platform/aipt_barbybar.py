"""
aipt_barbybar.py — Bar-by-bar sequential AIPT backtest (NO VECTORIZATION)

This script processes ONE 4h bar at a time to verify there is no lookahead
bias in the vectorized aipt_kucoin.py.  At each bar t:

  1. We observe characteristics Z_t (known at close of bar t)
  2. We compute RFF signals S_t = sin/cos(gamma * Z_t * theta)
  3. We compute portfolio weights w_t = normalize(S_t @ lambda)
  4. We WAIT for bar t+1 return R_{t+1} = close[t+1]/close[t] - 1
  5. We record portfolio return = w_t' @ R_{t+1}
  6. We record factor return F_{t+1} = S_t' @ R_{t+1} / sqrt(N)
     (used for FUTURE lambda estimation, NOT current bar)

Lambda is re-estimated every REBAL_EVERY bars using ONLY factor returns
from bars [t - TRAIN_BARS, t).  No pre-computation, no vectorized panel.

Usage:
    python aipt_barbybar.py
"""

import sys, os, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Config (same as aipt_kucoin.py) ──────────────────────────────────────
UNIVERSE      = "KUCOIN_TOP100"
INTERVAL      = "4h"
BARS_PER_YEAR = 6 * 365  # 2190
TRAIN_BARS    = 4380
MIN_TRAIN_BARS = 1000
REBAL_EVERY   = 12
OOS_START     = "2024-09-01"
COVERAGE_CUTOFF = 0.3
GAMMA_GRID    = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

P = 100
Z_RIDGE = 1e-3
SEED = 42

CHAR_NAMES = [
    "adv20", "adv60", "beta_to_btc", "close_position_in_range",
    "dollars_traded", "high_low_range",
    "historical_volatility_10", "historical_volatility_20",
    "historical_volatility_60", "historical_volatility_120",
    "log_returns", "momentum_5d", "momentum_20d", "momentum_60d",
    "open_close_range",
    "parkinson_volatility_10", "parkinson_volatility_20",
    "parkinson_volatility_60",
    "quote_volume", "turnover",
    "volume_momentum_1", "volume_momentum_5_20", "volume_ratio_20d",
    "vwap_deviation",
]

RESULTS_DIR = Path("data/aipt_results")


def main():
    t0 = time.time()
    print("=" * 70, flush=True)
    print("  AIPT BAR-BY-BAR SEQUENTIAL BACKTEST (no vectorization)", flush=True)
    print(f"  P={P}, z={Z_RIDGE:.0e}, seed={SEED}, rebal={REBAL_EVERY}", flush=True)
    print("=" * 70, flush=True)

    # ── Load raw data ────────────────────────────────────────────────────
    mat_dir  = Path(f"data/kucoin_cache/matrices/{INTERVAL}")
    uni_path = Path(f"data/kucoin_cache/universes/{UNIVERSE}_{INTERVAL}.parquet")

    universe_df = pd.read_parquet(uni_path)
    coverage = universe_df.sum(axis=0) / len(universe_df)
    valid_tickers = sorted(coverage[coverage > COVERAGE_CUTOFF].index.tolist())
    N = len(valid_tickers)

    # Load each characteristic as a raw DataFrame
    raw_matrices = {}
    for fp in sorted(mat_dir.glob("*.parquet")):
        df = pd.read_parquet(fp)
        cols = [c for c in valid_tickers if c in df.columns]
        if cols:
            raw_matrices[fp.stem] = df[cols]

    close_df = raw_matrices["close"]
    dates = close_df.index
    T_total = len(dates)

    # Available characteristics
    available_chars = [c for c in CHAR_NAMES if c in raw_matrices]
    D = len(available_chars)

    print(f"  N={N} tickers, D={D} chars, T={T_total} bars", flush=True)

    # ── Generate RFF parameters (fixed random, no data dependency) ───────
    rng = np.random.default_rng(SEED)
    n_pairs = P // 2
    theta = rng.standard_normal((n_pairs, D))
    gamma = rng.choice(GAMMA_GRID, size=n_pairs)

    # ── Find OOS start index ─────────────────────────────────────────────
    oos_start_idx = None
    for i, dt in enumerate(dates):
        if str(dt) >= OOS_START:
            oos_start_idx = i
            break
    assert oos_start_idx is not None
    print(f"  OOS start: bar {oos_start_idx} ({dates[oos_start_idx]})", flush=True)

    # ── State variables (as if running live) ─────────────────────────────
    # History of factor returns: list of (bar_index, F_vector) tuples
    factor_return_history = []

    # Current lambda (SDF weights) — None until first estimation
    lambda_hat = None
    bars_since_rebal = REBAL_EVERY  # force first estimation

    # Previous portfolio weights for turnover
    prev_weights = None

    # Results
    port_returns = []
    turnovers = []
    bar_dates_out = []

    # We need characteristics from the TRAINING period too (for factor returns)
    start_bar = max(0, oos_start_idx - TRAIN_BARS - 10)

    print(f"\n  Processing bars {start_bar} to {T_total-2} sequentially...", flush=True)
    print(f"  (OOS results start at bar {oos_start_idx})\n", flush=True)

    for t in range(start_bar, T_total - 1):
        # ════════════════════════════════════════════════════════════════
        # TIME t: Bar t just closed. We know close[t] and all chars at t.
        # We DO NOT know close[t+1] yet.
        # ════════════════════════════════════════════════════════════════

        # Step 1: Build Z_t — characteristics at bar t
        Z_t = np.full((N, D), np.nan)
        for j, char_name in enumerate(available_chars):
            df = raw_matrices[char_name]
            row_vals = df.iloc[t].reindex(valid_tickers).values.astype(np.float64)
            Z_t[:, j] = row_vals

        # Step 2: Rank-standardize Z_t cross-sectionally to [-0.5, 0.5]
        for j in range(D):
            col = Z_t[:, j]
            valid = ~np.isnan(col)
            n_valid = valid.sum()
            if n_valid < 3:
                Z_t[:, j] = 0.0
                continue
            r = rankdata(col[valid], method='average')
            r = r / n_valid - 0.5
            Z_t[valid, j] = r
            Z_t[~valid, j] = 0.0

        # Step 3: Compute RFF signals S_t (known at bar t)
        proj = Z_t @ theta.T                    # (N, P//2)
        proj_scaled = proj * gamma[np.newaxis, :]
        sin_part = np.sin(proj_scaled)
        cos_part = np.cos(proj_scaled)
        S_t = np.empty((N, P))
        S_t[:, 0::2] = sin_part
        S_t[:, 1::2] = cos_part

        # Step 4: Identify valid assets (non-NaN in characteristics)
        valid_mask = ~np.isnan(Z_t).any(axis=1)

        # ════════════════════════════════════════════════════════════════
        # TIME t+1: Bar t+1 closes. We now observe R_{t+1}.
        # ════════════════════════════════════════════════════════════════

        # Step 5: Get return at bar t+1
        close_t = close_df.iloc[t].reindex(valid_tickers).values.astype(np.float64)
        close_t1 = close_df.iloc[t + 1].reindex(valid_tickers).values.astype(np.float64)
        R_t1 = (close_t1 - close_t) / close_t  # pct_change
        R_t1 = np.nan_to_num(R_t1, nan=0.0)

        # Combined valid: have both characteristics and return
        both_valid = valid_mask & ~np.isnan(close_t) & ~np.isnan(close_t1)
        N_t = both_valid.sum()
        if N_t < 5:
            continue

        # Step 6: Compute factor return F_{t+1} = S_t[valid]' @ R_{t+1}[valid] / sqrt(N_t)
        S_valid = S_t[both_valid]
        R_valid = R_t1[both_valid]
        F_t1 = (1.0 / np.sqrt(N_t)) * (S_valid.T @ R_valid)  # (P,)

        # Store this factor return for FUTURE lambda estimation
        factor_return_history.append((t + 1, F_t1))

        # ════════════════════════════════════════════════════════════════
        # Only record OOS results
        # ════════════════════════════════════════════════════════════════
        if t + 1 < oos_start_idx:
            continue

        # Step 7: Should we re-estimate lambda?
        if bars_since_rebal >= REBAL_EVERY or lambda_hat is None:
            # Collect training factor returns: all from [t+1 - TRAIN_BARS, t+1)
            cutoff_low = (t + 1) - TRAIN_BARS
            train_data = [
                fr for (idx, fr) in factor_return_history
                if idx < (t + 1) and idx >= cutoff_low
            ]

            if len(train_data) < MIN_TRAIN_BARS:
                continue

            F_train = np.vstack(train_data)  # (T_train, P)
            T_tr, P_tr = F_train.shape

            # Ridge-Markowitz: lambda = (zI + E[FF'])^{-1} E[F]
            mu = F_train.mean(axis=0)
            FF = (F_train.T @ F_train) / T_tr
            A = Z_RIDGE * np.eye(P_tr) + FF
            lambda_hat = np.linalg.solve(A, mu)
            bars_since_rebal = 0

        # Step 8: Compute portfolio weights from signal at bar t
        #         w = S_t @ lambda / sqrt(N_t), normalized to sum|w|=1
        raw_w = np.zeros(N)
        raw_w_valid = (1.0 / np.sqrt(N_t)) * (S_valid @ lambda_hat)
        raw_w[both_valid] = raw_w_valid
        abs_sum = np.abs(raw_w).sum()
        if abs_sum < 1e-12:
            bars_since_rebal += 1
            continue
        w_norm = raw_w / abs_sum

        # Step 9: Portfolio return = w_norm' @ R_{t+1}
        port_ret = float(w_norm @ R_t1)
        port_returns.append(port_ret)
        bar_dates_out.append(dates[t + 1])

        # Step 10: Turnover
        if prev_weights is not None:
            to = np.abs(w_norm - prev_weights).sum() / 2.0
            turnovers.append(to)
        else:
            turnovers.append(0.0)
        prev_weights = w_norm.copy()

        bars_since_rebal += 1

        # Progress
        n_done = len(port_returns)
        if n_done % 500 == 0:
            arr_so_far = np.array(port_returns)
            sr_so_far = arr_so_far.mean() / arr_so_far.std() * np.sqrt(BARS_PER_YEAR)
            print(f"    bar {t+1} | {n_done} OOS bars | SR={sr_so_far:+.2f}", flush=True)

    # ── Final metrics ────────────────────────────────────────────────────
    port_arr = np.array(port_returns)
    to_arr = np.array(turnovers)

    mean_bar = port_arr.mean()
    std_bar = port_arr.std(ddof=1)
    sr_bar = mean_bar / std_bar if std_bar > 1e-12 else 0.0
    sr_ann = sr_bar * np.sqrt(BARS_PER_YEAR)
    avg_to = to_arr.mean()

    fee_3bps = port_arr - to_arr * 0.0003 * 2
    sr_3bps = (fee_3bps.mean() / fee_3bps.std() * np.sqrt(BARS_PER_YEAR)) if fee_3bps.std() > 1e-12 else 0.0

    elapsed = time.time() - t0

    print(f"\n{'='*70}", flush=True)
    print(f"  BAR-BY-BAR RESULTS  ({len(port_arr)} OOS bars)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Annualized Sharpe (0bps): {sr_ann:+.3f}", flush=True)
    print(f"  Annualized Sharpe (3bps): {sr_3bps:+.3f}", flush=True)
    print(f"  Mean turnover/bar:        {avg_to:.4f}", flush=True)
    print(f"  Mean return/bar:          {mean_bar:.6f}", flush=True)
    print(f"  Std  return/bar:          {std_bar:.6f}", flush=True)
    print(f"  Fee drag/bar (3bps):      {(to_arr * 0.0003 * 2).mean():.6f}", flush=True)
    print(f"  Fees as % of signal:      {(to_arr * 0.0003 * 2).mean() / max(mean_bar, 1e-12) * 100:.1f}%", flush=True)
    print(f"  Elapsed:                  {elapsed:.1f}s", flush=True)
    print(f"{'='*70}", flush=True)

    # ── Compare with vectorized result ───────────────────────────────────
    vec_path = RESULTS_DIR / "aipt_production_returns.csv"
    if vec_path.exists():
        vec_df = pd.read_csv(vec_path)
        vec_r = vec_df["port_return_0bps"].values
        print(f"\n  --- COMPARISON WITH VECTORIZED ---", flush=True)
        print(f"  Vectorized bars: {len(vec_r)}", flush=True)
        print(f"  Bar-by-bar bars: {len(port_arr)}", flush=True)

        # Compare element-wise if same length
        n_compare = min(len(vec_r), len(port_arr))
        if n_compare > 0:
            diff = np.abs(vec_r[:n_compare] - port_arr[:n_compare])
            print(f"  Max abs diff (first {n_compare} bars): {diff.max():.2e}", flush=True)
            print(f"  Mean abs diff: {diff.mean():.2e}", flush=True)
            corr = np.corrcoef(vec_r[:n_compare], port_arr[:n_compare])[0, 1]
            print(f"  Correlation: {corr:.6f}", flush=True)

            vec_sr = vec_r[:n_compare].mean() / vec_r[:n_compare].std() * np.sqrt(BARS_PER_YEAR)
            bbb_sr = port_arr[:n_compare].mean() / port_arr[:n_compare].std() * np.sqrt(BARS_PER_YEAR)
            print(f"  Vectorized SR (matched bars): {vec_sr:+.3f}", flush=True)
            print(f"  Bar-by-bar SR (matched bars): {bbb_sr:+.3f}", flush=True)

            if diff.max() < 1e-10:
                print(f"\n  [OK] EXACT MATCH -- no lookahead bias in vectorized version", flush=True)
            elif corr > 0.999:
                print(f"\n  [OK] NEAR-EXACT MATCH (rounding diffs) -- no lookahead bias", flush=True)
            else:
                print(f"\n  [FAIL] MISMATCH -- investigate differences!", flush=True)
    else:
        print(f"\n  (no vectorized results found at {vec_path} for comparison)", flush=True)

    # Save bar-by-bar results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bbb_df = pd.DataFrame({
        "date": bar_dates_out,
        "port_return_0bps": port_arr,
        "port_return_3bps": fee_3bps,
        "turnover": to_arr,
    })
    bbb_df.to_csv(RESULTS_DIR / "aipt_barbybar_returns.csv", index=False)
    print(f"  Saved to {RESULTS_DIR / 'aipt_barbybar_returns.csv'}", flush=True)


if __name__ == "__main__":
    main()
