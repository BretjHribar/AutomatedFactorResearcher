#!/usr/bin/env python
"""
run_voc_causal.py -- Causally-correct multi-horizon FM regression.

FIX: The original run_voc_qp.py trains on cum_ret[t] = ret[t+1:t+h],
which are FUTURE returns not yet realized at time t.

CORRECT: At time t, train on features[t-h] → cum_ret_past[t] = ret[t-h+1:t],
which is the h-bar return that's fully realized by time t.

Tests both the BUGGY (forward-looking) and FIXED (backward-looking) versions
side-by-side for direct comparison. Uses the same QP settings.
"""
import numpy as np
import pandas as pd
import gc, time, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_voc_complexity import (
    load_full_data, log, SPLITS, BOOKSIZE, MAX_WEIGHT,
    simulate, process_signal, augment_matrices_with_db_alphas
)
from run_voc_sdf import (
    prepare_panel, cross_sectional_standardize,
    generate_rff_params, compute_rff_features as compute_rff_sdf,
)
from run_voc_qp import qp_optimize_aggressive

WARMUP = 120
REPORT = []


def build_cum_ret_forward(ret_np, horizon):
    """BUGGY: cum_ret[t] = ret[t+1] + ... + ret[t+h] (FUTURE returns)."""
    T, N = ret_np.shape
    if horizon <= 1:
        return ret_np.copy()
    cum_ret = np.zeros((T, N))
    for h in range(horizon):
        shifted = np.zeros((T, N))
        if h + 1 < T:
            shifted[:T-h-1] = ret_np[h+1:]
        cum_ret += shifted
    cum_ret[T-horizon:] = 0.0
    return cum_ret


def build_cum_ret_backward(ret_np, horizon):
    """FIXED: cum_ret[t] = ret[t-h+1] + ... + ret[t] (PAST h-bar realized return).
    
    At time t, this is the cumulative return from close_{t-h} to close_t.
    Fully known at close of bar t.
    """
    T, N = ret_np.shape
    if horizon <= 1:
        return ret_np.copy()
    cum_ret = np.zeros((T, N))
    for h in range(horizon):
        shifted = np.zeros((T, N))
        if h < T:
            shifted[h:] = ret_np[h:]
        cum_ret += shifted
    # First h-1 bars don't have full history
    cum_ret[:horizon-1] = 0.0
    return cum_ret


def run_fm_causal(std_panel, ret_np, valid_mask, P_half, seed,
                  z=1e-5, retrain_every=6, horizon=1, causal=True):
    """
    FM regression with correct causal timing.
    
    causal=True (FIXED):
      - Train on features[t-h] → cum_ret_past[t] (backward h-bar return, realized)
      - Both available at close of bar t
      
    causal=False (ORIGINAL BUGGY):
      - Train on features[t-1] → cum_ret_fwd[t] (forward h-bar return, NOT realized)
      - Replicates the original run_voc_qp.py behavior
    """
    T, N, K = std_panel.shape
    P2 = 2 * P_half
    
    omega, gammas = generate_rff_params(K, P_half, seed=seed)
    features = compute_rff_sdf(std_panel, omega, gammas)
    
    if causal:
        cum_ret = build_cum_ret_backward(ret_np, horizon)
    else:
        cum_ret = build_cum_ret_forward(ret_np, horizon)
    
    alpha = np.zeros((T, N))
    beta = None
    last_refit = -retrain_every
    n_refits = 0
    
    min_start = max(WARMUP, horizon) if causal else WARMUP
    
    for t in range(min_start, T):
        if t - last_refit >= retrain_every or beta is None:
            if causal:
                # FIXED: features at t-h, return realized by t
                # Most recent usable: features[t-horizon] → cum_ret[t]
                # (cum_ret[t] = ret[t-h+1]+...+ret[t], all realized)
                feat_idx = t - horizon
                if feat_idx < 0:
                    continue
                mask = valid_mask[feat_idx] & valid_mask[t]
                n = mask.sum()
                if n < 5:
                    continue
                S = features[feat_idx, mask, :].reshape(-1, P2)
                R = cum_ret[t, mask]
            else:
                # ORIGINAL BUGGY: features[t-1] → cum_ret[t] (forward returns!)
                mask = valid_mask[t-1] & valid_mask[t]
                n = mask.sum()
                if n < 5:
                    continue
                S = features[t-1, mask, :].reshape(-1, P2)
                R = cum_ret[t, mask]
            
            n_obs = len(R)
            
            if P2 <= n_obs:
                STS = S.T @ S
                STR = S.T @ R
                beta = np.linalg.solve(z * np.eye(P2) + STS, STR)
            else:
                SST = S @ S.T
                a = np.linalg.solve(z * np.eye(n_obs) + SST, R)
                beta = S.T @ a
            
            last_refit = t
            n_refits += 1
        
        if beta is not None:
            alpha[t] = features[t] @ beta
            vm = valid_mask[t]
            if vm.sum() > 0:
                alpha[t, vm] -= np.mean(alpha[t, vm])
    
    del features
    gc.collect()
    return alpha, n_refits


def eval_split(signal_df, returns_df, close_df, universe_df, start, end, fees_bps):
    w = signal_df.loc[start:end]
    r = returns_df.loc[start:end]
    c = close_df.loc[start:end]
    u = universe_df.loc[start:end]
    if len(w) == 0:
        return None
    return simulate(w, r, c, u, fees_bps=fees_bps)


def report(msg):
    print(msg, flush=True)
    REPORT.append(msg)


def main():
    t0 = time.time()
    
    report("Loading augmented data (K=58)...")
    matrices, universe_df, valid_tickers = load_full_data()
    aug_matrices, n_alphas = augment_matrices_with_db_alphas(matrices, universe_df)
    panel, ret_np, valid_mask, dates, tickers, _ = prepare_panel(aug_matrices, valid_tickers)
    std = cross_sectional_standardize(panel)
    std = np.nan_to_num(std, nan=0.0)
    del panel; gc.collect()
    
    T, N, K = std.shape
    report(f"  Panel: T={T}, N={N}, K={K}")
    
    returns_df = aug_matrices["returns"]
    close_df = aug_matrices["close"]
    
    SEEDS = 5
    
    # Test configs: both causal and buggy for comparison
    configs = [
        # Buggy (original) for baseline comparison
        dict(horizon=1,  causal=False, label="h=1  ORIGINAL (no multi-h, no bias possible)"),
        dict(horizon=1,  causal=True,  label="h=1  CAUSAL  (should match original)"),
        dict(horizon=6,  causal=False, label="h=6  ORIGINAL (forward-looking, BIASED)"),
        dict(horizon=6,  causal=True,  label="h=6  CAUSAL  (backward-looking, CLEAN)"),
        dict(horizon=12, causal=False, label="h=12 ORIGINAL (forward-looking, BIASED)"),
        dict(horizon=12, causal=True,  label="h=12 CAUSAL  (backward-looking, CLEAN)"),
        dict(horizon=24, causal=False, label="h=24 ORIGINAL (forward-looking, BIASED)"),
        dict(horizon=24, causal=True,  label="h=24 CAUSAL  (backward-looking, CLEAN)"),
    ]
    
    report(f"\n{'='*100}")
    report(f"  LOOKAHEAD BIAS TEST: Forward vs Backward cum_ret")
    report(f"  K={K}, P=500, z=1e-5, {SEEDS} seeds, QP(tc=0.005, rb=1, to=0.05)")
    report(f"{'='*100}")
    report(f"")
    report(f"  ORIGINAL: features[t-1] -> ret[t+1:t+h]  (FUTURE - not realized at t)")
    report(f"  CAUSAL:   features[t-h] -> ret[t-h+1:t]   (PAST  - realized at t)")
    report(f"")
    
    results_table = []
    
    for cfg in configs:
        h      = cfg["horizon"]
        causal = cfg["causal"]
        label  = cfg["label"]
        
        report(f"\n  === {label} ===")
        t1 = time.time()
        
        alpha_sum = np.zeros((T, N))
        for s in range(SEEDS):
            seed = 42 + s * 137
            a, nrefits = run_fm_causal(
                std, ret_np, valid_mask, 500, seed,
                z=1e-5, retrain_every=6, horizon=h,
                causal=causal)
            alpha_sum += a
        alpha_avg = alpha_sum / SEEDS
        
        alpha_df = pd.DataFrame(alpha_avg, index=dates, columns=tickers)
        alpha_df = process_signal(alpha_df, universe_df=universe_df, max_wt=MAX_WEIGHT)
        
        # QP optimize
        qp = qp_optimize_aggressive(
            alpha_df, aug_matrices, universe_df, returns_df,
            risk_aversion=1.0, tcost_lambda=0.005,
            lookback_bars=120, rebal_every=1, max_turnover=0.05)
        
        # Evaluate
        row = {"label": label, "h": h, "causal": causal}
        for split_name, (start, end) in SPLITS.items():
            sim = eval_split(qp, returns_df, close_df, universe_df, start, end, 5.0)
            if sim:
                report(f"    QP {split_name:5s}: SR@5={sim.sharpe:+.3f} TO={sim.turnover:.4f}")
                row[f"{split_name}_sr5"] = sim.sharpe
                row[f"{split_name}_to"] = sim.turnover
        
        results_table.append(row)
        report(f"    Time: {time.time()-t1:.0f}s")
        del alpha_sum, alpha_avg, alpha_df, qp; gc.collect()
    
    # Summary table
    report(f"\n{'='*100}")
    report(f"  SUMMARY: ORIGINAL (biased) vs CAUSAL (clean)")
    report(f"{'='*100}")
    report(f"  {'Config':<50s} | {'Train':>7s} | {'Val':>7s} | {'Test':>7s} | {'Test TO':>8s}")
    report(f"  {'-'*95}")
    
    for row in results_table:
        label = row["label"]
        train = row.get("train_sr5", 0)
        val = row.get("val_sr5", 0)
        test = row.get("test_sr5", 0)
        to = row.get("test_to", 0)
        report(f"  {label:<50s} | {train:+7.2f} | {val:+7.2f} | {test:+7.2f} | {to:8.4f}")
    
    # Compute bias magnitude
    report(f"\n  BIAS MAGNITUDE (ORIGINAL - CAUSAL):")
    for h in [1, 6, 12, 24]:
        orig = [r for r in results_table if r["h"] == h and not r["causal"]]
        fixed = [r for r in results_table if r["h"] == h and r["causal"]]
        if orig and fixed:
            o, f = orig[0], fixed[0]
            for split in ["val", "test"]:
                o_sr = o.get(f"{split}_sr5", 0)
                f_sr = f.get(f"{split}_sr5", 0)
                delta = o_sr - f_sr
                report(f"    h={h:2d} {split:5s}: {o_sr:+.2f} - {f_sr:+.2f} = {delta:+.2f} bias")
    
    report(f"\n  Total time: {time.time()-t0:.0f}s")
    
    out_path = Path(__file__).parent / "voc_causal_results.md"
    with open(out_path, "w") as f:
        f.write("# Lookahead Bias Test: Forward vs Backward Returns\n\n```\n")
        for line in REPORT:
            f.write(line + "\n")
        f.write("```\n")
    report(f"  Report saved to: {out_path}")


if __name__ == "__main__":
    main()
