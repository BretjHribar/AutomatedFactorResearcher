#!/usr/bin/env python
"""
run_voc_kelly.py -- Proper Kelly expanding-window FM implementation.

Uses INCREMENTAL sufficient statistics (S'S and S'R) so expanding window
is O(N*P^2) per refit instead of O(N_obs * P^2).

Tests:
  - Expanding window (all history)
  - Rolling windows (720, 360, 180 bars)
  - Single cross-section (fm_window=1, baseline for comparison)

Compares against prior fm_window=1 results from run_voc_qp.py.
Does NOT modify any existing files.
"""
import numpy as np
import pandas as pd
import gc, time, sys
from pathlib import Path
from collections import deque

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

# Prior baselines fm_window=1, K=58, rb=1, tc=0.005, to=0.05
BASELINES = {
    24: {"val_sr5": 9.51, "test_sr5": 7.39},
    12: {"val_sr5": 9.45, "test_sr5": 6.43},
}


def run_fm_kelly(std_panel, ret_np, valid_mask, P_half, seed,
                 z=1e-5, retrain_every=6, horizon=1,
                 max_window=0):
    """
    Kelly-proper FM with incremental sufficient statistics.
    
    Maintains running STS = S'S and STR = S'R incrementally.
    For rolling window: uses a deque of per-bar (STS_bar, STR_bar) to subtract
    old bars that fall out of the window.
    
    max_window=0: truly expanding (never forget)
    max_window=N: rolling window of N bars
    """
    T, N, K = std_panel.shape
    P2 = 2 * P_half
    
    omega, gammas = generate_rff_params(K, P_half, seed=seed)
    features = compute_rff_sdf(std_panel, omega, gammas)
    
    # Build h-bar cumulative returns
    if horizon > 1:
        cum_ret = np.zeros((T, N))
        for h in range(horizon):
            shifted = np.zeros((T, N))
            if h + 1 < T:
                shifted[:T-h-1] = ret_np[h+1:]
            cum_ret += shifted
        cum_ret[T-horizon:] = 0.0
    else:
        cum_ret = ret_np
    
    alpha = np.zeros((T, N))
    beta = None
    last_refit = -retrain_every
    n_refits = 0
    complexity_log = []
    
    # Incremental sufficient statistics
    STS_total = np.zeros((P2, P2))  # S'S accumulator
    STR_total = np.zeros(P2)        # S'R accumulator
    n_obs_total = 0
    
    # For rolling window: store per-bar contributions
    if max_window > 0:
        bar_queue = deque()  # (STS_bar, STR_bar, n_bar)
    
    # Pre-add bars from WARMUP up to the first refit point
    for tw in range(WARMUP, min(WARMUP + retrain_every, T)):
        if tw < 1:
            continue
        mask = valid_mask[tw-1] & valid_mask[tw]
        n = mask.sum()
        if n < 5:
            if max_window > 0:
                bar_queue.append((None, None, 0))
            continue
        S_bar = features[tw-1, mask, :]
        R_bar = cum_ret[tw, mask]
        STS_bar = S_bar.T @ S_bar
        STR_bar = S_bar.T @ R_bar
        STS_total += STS_bar
        STR_total += STR_bar
        n_obs_total += n
        if max_window > 0:
            bar_queue.append((STS_bar, STR_bar, n))
    
    for t in range(WARMUP, T):
        # Add new bar's contribution (if we haven't already in pre-add)
        if t >= WARMUP + retrain_every:
            tw = t
            if tw < 1 or tw >= T:
                if max_window > 0:
                    bar_queue.append((None, None, 0))
            else:
                mask = valid_mask[tw-1] & valid_mask[tw]
                n = mask.sum()
                if n < 5:
                    if max_window > 0:
                        bar_queue.append((None, None, 0))
                else:
                    S_bar = features[tw-1, mask, :]
                    R_bar = cum_ret[tw, mask]
                    STS_bar = S_bar.T @ S_bar
                    STR_bar = S_bar.T @ R_bar
                    STS_total += STS_bar
                    STR_total += STR_bar
                    n_obs_total += n
                    if max_window > 0:
                        bar_queue.append((STS_bar, STR_bar, n))
            
            # Remove oldest bar if rolling window exceeded
            if max_window > 0 and len(bar_queue) > max_window:
                old_sts, old_str, old_n = bar_queue.popleft()
                if old_sts is not None:
                    STS_total -= old_sts
                    STR_total -= old_str
                    n_obs_total -= old_n
        
        # Refit beta
        if t - last_refit >= retrain_every or beta is None:
            if n_obs_total < 10:
                continue
            
            c_ratio = P2 / n_obs_total
            complexity_log.append((t, n_obs_total, c_ratio))
            
            # Solve: beta = (STS + z*I)^{-1} STR
            beta = np.linalg.solve(z * np.eye(P2) + STS_total, STR_total)
            
            last_refit = t
            n_refits += 1
        
        if beta is not None:
            alpha[t] = features[t] @ beta
            vm = valid_mask[t]
            if vm.sum() > 0:
                alpha[t, vm] -= np.mean(alpha[t, vm])
    
    del features
    gc.collect()
    return alpha, n_refits, complexity_log


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
    
    configs = [
        dict(horizon=24, max_window=1,    label="h=24 single CS (baseline)"),
        dict(horizon=24, max_window=180,  label="h=24 rolling w=180 (30d)"),
        dict(horizon=24, max_window=360,  label="h=24 rolling w=360 (60d)"),
        dict(horizon=24, max_window=720,  label="h=24 rolling w=720 (120d)"),
        dict(horizon=24, max_window=0,    label="h=24 expanding (all history)"),
        dict(horizon=12, max_window=1,    label="h=12 single CS (baseline)"),
        dict(horizon=12, max_window=360,  label="h=12 rolling w=360 (60d)"),
        dict(horizon=12, max_window=720,  label="h=12 rolling w=720 (120d)"),
        dict(horizon=12, max_window=0,    label="h=12 expanding (all history)"),
    ]
    
    SEEDS = 5
    
    report(f"\n{'='*100}")
    report(f"  Kelly Expanding-Window FM Test | K={K}, P=500, z=1e-5, {SEEDS} seeds")
    report(f"  QP: tc=0.005, rb=1, to=0.05 | Incremental sufficient statistics")
    report(f"{'='*100}")
    report(f"")
    report(f"  Prior baselines (fm_window=1):")
    for h, bl in BASELINES.items():
        report(f"    h={h}: Val SR@5={bl['val_sr5']:+.2f}, Test SR@5={bl['test_sr5']:+.2f}")
    
    results_table = []
    
    for cfg in configs:
        h     = cfg["horizon"]
        mw    = cfg["max_window"]
        label = cfg["label"]
        bl    = BASELINES.get(h, {})
        
        report(f"\n  === {label} ===")
        t1 = time.time()
        
        alpha_sum = np.zeros((T, N))
        all_complexity = []
        for s in range(SEEDS):
            seed = 42 + s * 137
            a, nrefits, clog = run_fm_kelly(
                std, ret_np, valid_mask, 500, seed,
                z=1e-5, retrain_every=6, horizon=h,
                max_window=mw)
            alpha_sum += a
            if s == 0:
                all_complexity = clog
        alpha_avg = alpha_sum / SEEDS
        
        # Complexity ratio at key points
        if all_complexity:
            early = all_complexity[min(10, len(all_complexity)-1)]
            mid = all_complexity[len(all_complexity)//2]
            late = all_complexity[-1]
            report(f"    Complexity c = P/N_obs:")
            report(f"      Early: c={early[2]:.4f} (N_obs={early[1]:,d})")
            report(f"      Mid:   c={mid[2]:.4f} (N_obs={mid[1]:,d})")
            report(f"      Late:  c={late[2]:.4f} (N_obs={late[1]:,d})")
        
        alpha_df = pd.DataFrame(alpha_avg, index=dates, columns=tickers)
        alpha_df = process_signal(alpha_df, universe_df=universe_df, max_wt=MAX_WEIGHT)
        
        # QP optimize
        qp = qp_optimize_aggressive(
            alpha_df, aug_matrices, universe_df, returns_df,
            risk_aversion=1.0, tcost_lambda=0.005,
            lookback_bars=120, rebal_every=1, max_turnover=0.05)
        
        # Evaluate
        qp_results = {}
        for split_name, (start, end) in SPLITS.items():
            sim = eval_split(qp, returns_df, close_df, universe_df, start, end, 5.0)
            if sim:
                qp_results[split_name] = sim
        
        val_sr5  = qp_results.get("val", None)
        test_sr5 = qp_results.get("test", None)
        
        for sn in ["train", "val", "test"]:
            sim = qp_results.get(sn)
            if sim:
                # Compare against baseline
                delta = ""
                if sn == "val" and bl.get("val_sr5"):
                    d = sim.sharpe - bl["val_sr5"]
                    delta = f"  (vs baseline: {d:+.2f})"
                elif sn == "test" and bl.get("test_sr5"):
                    d = sim.sharpe - bl["test_sr5"]
                    delta = f"  (vs baseline: {d:+.2f})"
                report(f"    QP {sn:5s}: SR@5={sim.sharpe:+.3f} TO={sim.turnover:.4f}{delta}")
        
        if val_sr5 and test_sr5:
            results_table.append((label, h, mw,
                                  test_sr5.sharpe, val_sr5.sharpe,
                                  test_sr5.turnover))
        
        report(f"    Time: {time.time()-t1:.0f}s")
        del alpha_sum, alpha_avg, alpha_df, qp; gc.collect()
    
    # Summary
    report(f"\n{'='*100}")
    report(f"  SUMMARY -- Sorted by Test SR@5bps")
    report(f"{'='*100}")
    report(f"  {'Config':<40s} |  h   {'win':>5s}  {'Val SR@5':>9s}  {'Test SR@5':>10s}  {'Test TO':>8s}  {'vs BL':>7s}")
    report(f"  {'-'*95}")
    
    results_table.sort(key=lambda x: -x[3])
    for label, h, mw, test_sr5, val_sr5, test_to in results_table:
        mw_str = "all" if mw == 0 else str(mw)
        bl_test = BASELINES.get(h, {}).get("test_sr5", 0)
        delta = test_sr5 - bl_test
        report(f"  {label:<40s} | {h:3d}  {mw_str:>5s}  {val_sr5:+9.3f}  {test_sr5:+10.3f}  {test_to:8.4f}  {delta:+7.2f}")
    
    report(f"\n  Total time: {time.time()-t0:.0f}s")
    
    out_path = Path(__file__).parent / "voc_kelly_results.md"
    with open(out_path, "w") as f:
        f.write("# Kelly Expanding-Window FM Results\n\n```\n")
        for line in REPORT:
            f.write(line + "\n")
        f.write("```\n")
    report(f"  Report saved to: {out_path}")


if __name__ == "__main__":
    main()
