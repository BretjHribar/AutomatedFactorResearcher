#!/usr/bin/env python
"""
run_voc_clean.py -- Clean causal sweep of the VoC pipeline (h=1, no lookahead).

The causal FM uses:
  Train: features[t-1] -> ret[t]   (characteristics at close_{t-1} predict
         return from close_{t-1} to close_t, i.e., the return of bar t)
  Signal: alpha[t] = features[t] @ beta_t
  Execution: position at close_t earns ret[t+1] (via delay=0 simulator shift)

This is standard, clean Fama-MacBeth. No multi-horizon lookahead.

Sweep:
  - P (RFF features): 100, 250, 500, 1000, 2000
  - Ridge z: 1e-3, 1e-4, 1e-5
  - K: 34 (raw only) vs 58 (+ DB alphas)
  - QP tc: 0.001, 0.003, 0.005
  - QP to: 0.05, 0.10
  - retrain_every: 1, 6
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
SEEDS = 5
REPORT = []


def run_fm_h1(std_panel, ret_np, valid_mask, P_half, seed,
              z=1e-5, retrain_every=6):
    """
    Clean h=1 FM regression.
    
    At refit time t:
      S = features[t-1, valid assets]   (known at close of bar t-1)
      R = ret_np[t, valid assets]        (return of bar t, known at close of bar t)
    
    This is the standard FM setup. The beta is then applied:
      alpha[t] = features[t] @ beta
    And positions earn ret[t+1] via the simulator's delay=0 logic.
    """
    T, N, K = std_panel.shape
    P2 = 2 * P_half
    
    omega, gammas = generate_rff_params(K, P_half, seed=seed)
    features = compute_rff_sdf(std_panel, omega, gammas)
    
    alpha = np.zeros((T, N))
    beta = None
    last_refit = -retrain_every
    n_refits = 0
    
    for t in range(WARMUP, T):
        if t - last_refit >= retrain_every or beta is None:
            # Features at t-1, return of bar t
            mask = valid_mask[t-1] & valid_mask[t]
            n = mask.sum()
            if n < 5:
                continue
            S = features[t-1, mask, :]
            R = ret_np[t, mask]
            n_obs = len(R)
            
            # Kernel trick when P > N
            if P2 <= n_obs:
                beta = np.linalg.solve(z * np.eye(P2) + S.T @ S, S.T @ R)
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


def build_signal(std, ret_np, valid_mask, T, N, dates, tickers, P_half, z, retrain):
    alpha_sum = np.zeros((T, N))
    for s in range(SEEDS):
        a, _ = run_fm_h1(std, ret_np, valid_mask, P_half, 42 + s*137,
                         z=z, retrain_every=retrain)
        alpha_sum += a
    return alpha_sum / SEEDS


def eval_qp(alpha_avg, dates, tickers, aug_matrices, universe_df, returns_df, close_df,
            tc, to_max):
    alpha_df = pd.DataFrame(alpha_avg, index=dates, columns=tickers)
    alpha_df = process_signal(alpha_df, universe_df=universe_df, max_wt=MAX_WEIGHT)
    qp = qp_optimize_aggressive(
        alpha_df, aug_matrices, universe_df, returns_df,
        risk_aversion=1.0, tcost_lambda=tc,
        lookback_bars=120, rebal_every=1, max_turnover=to_max)
    results = {}
    for split_name, (start, end) in SPLITS.items():
        w = qp.loc[start:end]
        r = returns_df.loc[start:end]
        c = close_df.loc[start:end]
        u = universe_df.loc[start:end]
        if len(w) > 0:
            sim = simulate(w, r, c, u, fees_bps=5.0)
            results[split_name] = sim
    del alpha_df, qp
    gc.collect()
    return results


def rpt(msg):
    print(msg, flush=True)
    REPORT.append(msg)


def main():
    t0 = time.time()
    
    # Load both K=34 and K=58 panels
    rpt("Loading data...")
    matrices, universe_df, valid_tickers = load_full_data()
    
    # K=34 panel
    panel34, ret_np, valid_mask, dates, tickers, _ = prepare_panel(matrices, valid_tickers)
    std34 = cross_sectional_standardize(panel34)
    std34 = np.nan_to_num(std34, nan=0.0)
    del panel34; gc.collect()
    T, N, K34 = std34.shape
    rpt(f"  K=34 panel: T={T}, N={N}, K={K34}")
    
    # K=58 panel
    aug_matrices, n_alphas = augment_matrices_with_db_alphas(matrices, universe_df)
    panel58, _, _, _, _, _ = prepare_panel(aug_matrices, valid_tickers)
    std58 = cross_sectional_standardize(panel58)
    std58 = np.nan_to_num(std58, nan=0.0)
    del panel58; gc.collect()
    K58 = std58.shape[2]
    rpt(f"  K=58 panel: T={T}, N={N}, K={K58}")
    
    returns_df = aug_matrices["returns"]
    close_df   = aug_matrices["close"]
    
    rpt(f"\n{'='*110}")
    rpt(f"  CLEAN CAUSAL VoC SWEEP | h=1,  {SEEDS} seeds, QP rb=1, fees=5bps")
    rpt(f"  Baseline (prior run): Test SR@5 = +2.02 (K=58, P=500, z=1e-5, tc=0.005, to=0.05)")
    rpt(f"{'='*110}")
    
    all_results = []
    
    # ── Phase 1: P sweep (K=58, z=1e-5, tc=0.005, to=0.05, retrain=6) ──
    rpt(f"\n--- Phase 1: P sweep (K=58, z=1e-5, tc=0.005, to=0.05) ---")
    rpt(f"  {'P':>5}  c=2P/N  {'Val SR@5':>10}  {'Test SR@5':>11}  {'Test TO':>8}  time")
    for P in [100, 250, 500, 1000, 2000]:
        t1 = time.time()
        c = 2*P / N
        alpha = build_signal(std58, ret_np, valid_mask, T, N, dates, tickers, P, 1e-5, 6)
        res = eval_qp(alpha, dates, tickers, aug_matrices, universe_df, returns_df, close_df, 0.005, 0.05)
        val  = res.get("val",  None)
        test = res.get("test", None)
        val_sr  = val.sharpe  if val  else 0.0
        test_sr = test.sharpe if test else 0.0
        test_to = test.turnover if test else 0.0
        elapsed = time.time()-t1
        rpt(f"  {P:>5}  {c:>6.1f}  {val_sr:+10.3f}  {test_sr:+11.3f}  {test_to:8.4f}  {elapsed:.0f}s")
        all_results.append(dict(phase="P_sweep", K=K58, P=P, z=1e-5, tc=0.005, to=0.05,
                                retrain=6, val_sr5=val_sr, test_sr5=test_sr, test_to=test_to))
        del alpha; gc.collect()
    
    # ── Phase 2: Ridge z sweep (K=58, P=500, tc=0.005, to=0.05, retrain=6) ──
    rpt(f"\n--- Phase 2: Ridge z sweep (K=58, P=500, tc=0.005, to=0.05) ---")
    rpt(f"  {'z':>8}  {'Val SR@5':>10}  {'Test SR@5':>11}  {'Test TO':>8}  time")
    for z in [1e-3, 1e-4, 1e-5, 1e-6]:
        t1 = time.time()
        alpha = build_signal(std58, ret_np, valid_mask, T, N, dates, tickers, 500, z, 6)
        res = eval_qp(alpha, dates, tickers, aug_matrices, universe_df, returns_df, close_df, 0.005, 0.05)
        val_sr  = res["val"].sharpe  if "val"  in res else 0.0
        test_sr = res["test"].sharpe if "test" in res else 0.0
        test_to = res["test"].turnover if "test" in res else 0.0
        elapsed = time.time()-t1
        rpt(f"  {z:>8.0e}  {val_sr:+10.3f}  {test_sr:+11.3f}  {test_to:8.4f}  {elapsed:.0f}s")
        all_results.append(dict(phase="z_sweep", K=K58, P=500, z=z, tc=0.005, to=0.05,
                                retrain=6, val_sr5=val_sr, test_sr5=test_sr, test_to=test_to))
        del alpha; gc.collect()
    
    # ── Phase 3: K=34 vs K=58 (best P from Phase 1) ──
    rpt(f"\n--- Phase 3: K comparison (P=500, z=1e-5, tc=0.005, to=0.05) ---")
    rpt(f"  {'K':>4}  {'Val SR@5':>10}  {'Test SR@5':>11}  {'Test TO':>8}  time")
    for K_label, std_use in [("K=34", std34), ("K=58", std58)]:
        t1 = time.time()
        alpha = build_signal(std_use, ret_np, valid_mask, T, N, dates, tickers, 500, 1e-5, 6)
        res = eval_qp(alpha, dates, tickers, aug_matrices, universe_df, returns_df, close_df, 0.005, 0.05)
        val_sr  = res["val"].sharpe  if "val"  in res else 0.0
        test_sr = res["test"].sharpe if "test" in res else 0.0
        test_to = res["test"].turnover if "test" in res else 0.0
        elapsed = time.time()-t1
        rpt(f"  {K_label:>4}  {val_sr:+10.3f}  {test_sr:+11.3f}  {test_to:8.4f}  {elapsed:.0f}s")
        all_results.append(dict(phase="K_sweep", K=K_label, P=500, z=1e-5, tc=0.005, to=0.05,
                                retrain=6, val_sr5=val_sr, test_sr5=test_sr, test_to=test_to))
        del alpha; gc.collect()
    
    # ── Phase 4: QP parameter sweep (K=58, P=500, z=1e-5, retrain=6) ──
    rpt(f"\n--- Phase 4: QP sweep (K=58, P=500, z=1e-5) ---")
    rpt(f"  {'tc':>6}  {'to':>5}  {'Val SR@5':>10}  {'Test SR@5':>11}  {'Test TO':>8}  time")
    alpha_cached = build_signal(std58, ret_np, valid_mask, T, N, dates, tickers, 500, 1e-5, 6)
    for tc in [0.001, 0.003, 0.005, 0.010]:
        for to in [0.05, 0.10, 0.20]:
            t1 = time.time()
            res = eval_qp(alpha_cached, dates, tickers, aug_matrices, universe_df, returns_df, close_df, tc, to)
            val_sr  = res["val"].sharpe  if "val"  in res else 0.0
            test_sr = res["test"].sharpe if "test" in res else 0.0
            test_to = res["test"].turnover if "test" in res else 0.0
            elapsed = time.time()-t1
            rpt(f"  {tc:>6.3f}  {to:>5.2f}  {val_sr:+10.3f}  {test_sr:+11.3f}  {test_to:8.4f}  {elapsed:.0f}s")
            all_results.append(dict(phase="QP_sweep", K=K58, P=500, z=1e-5, tc=tc, to=to,
                                    retrain=6, val_sr5=val_sr, test_sr5=test_sr, test_to=test_to))
    del alpha_cached; gc.collect()
    
    # ── Phase 5: retrain_every (K=58, best P, best z) ──
    rpt(f"\n--- Phase 5: Retrain frequency (K=58, P=500, z=1e-5, tc=0.005, to=0.05) ---")
    rpt(f"  {'retrain':>8}  {'Val SR@5':>10}  {'Test SR@5':>11}  {'Test TO':>8}  time")
    for retrain in [1, 3, 6, 12, 24]:
        t1 = time.time()
        alpha = build_signal(std58, ret_np, valid_mask, T, N, dates, tickers, 500, 1e-5, retrain)
        res = eval_qp(alpha, dates, tickers, aug_matrices, universe_df, returns_df, close_df, 0.005, 0.05)
        val_sr  = res["val"].sharpe  if "val"  in res else 0.0
        test_sr = res["test"].sharpe if "test" in res else 0.0
        test_to = res["test"].turnover if "test" in res else 0.0
        elapsed = time.time()-t1
        rpt(f"  {retrain:>8}  {val_sr:+10.3f}  {test_sr:+11.3f}  {test_to:8.4f}  {elapsed:.0f}s")
        all_results.append(dict(phase="retrain_sweep", K=K58, P=500, z=1e-5, tc=0.005, to=0.05,
                                retrain=retrain, val_sr5=val_sr, test_sr5=test_sr, test_to=test_to))
        del alpha; gc.collect()
    
    # ── Summary ──
    rpt(f"\n{'='*110}")
    rpt(f"  TOP 10 CONFIGS BY VAL SR@5")
    rpt(f"{'='*110}")
    rpt(f"  {'Phase':<15}  K    P     z     tc    to   ret  Val SR@5  Test SR@5  Test TO")
    rpt(f"  {'-'*100}")
    top10 = sorted(all_results, key=lambda x: -x["val_sr5"])[:10]
    for r in top10:
        rpt(f"  {r['phase']:<15}  {str(r['K']):<4}  {r['P']:>4}  {r['z']:.0e}  "
            f"{r['tc']:.3f}  {r['to']:.2f}  {r['retrain']:>3}  "
            f"{r['val_sr5']:+8.3f}  {r['test_sr5']:+9.3f}  {r['test_to']:.4f}")
    
    rpt(f"\n  Total time: {time.time()-t0:.0f}s")
    
    out_path = Path(__file__).parent / "voc_clean_results.md"
    with open(out_path, "w") as f:
        f.write("# Clean Causal VoC Sweep Results (h=1, no lookahead)\n\n```\n")
        for line in REPORT:
            f.write(line + "\n")
        f.write("```\n")
    rpt(f"  Report saved to: {out_path}")


if __name__ == "__main__":
    main()
