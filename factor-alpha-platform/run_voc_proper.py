#!/usr/bin/env python
"""
run_voc_proper.py -- Correct Kelly VoC implementation for 4H crypto.

Implements BOTH frameworks from the Kelly papers:

Option A: Univariate Time-Series VoC (Kelly, Malamud, Zhou 2024)
  - For each asset i, run time-series ridge regression:
      R_{i,t+1} = S'_{i,t} beta_i + eps
  - S_{i,t} = RFF(characteristics of asset i at time t), P-dimensional
  - Training pools T past observations of asset i's own history
  - c = P/T (complexity ratio)
  - Signal: alpha_{i,t} = S'_{i,t} * beta_hat_i
  - Combine per-asset signals into portfolio via QP

Option B: Factor Portfolio VoC / AIPT (Didisheim, Ke, Kelly, Malamud 2025)
  - Build P characteristic-managed factor portfolios:
      F_{p,t+1} = N_t^{-1/2} * sum_i S_{p,i,t} * R_{i,t+1}
  - Ridge Markowitz tangency on factor return time series:
      lambda(z) = (zI + E[F F'])^{-1} E[F]
  - SDF portfolio weights: w_t = S_t * lambda_hat
  - c = P/T

Both are strictly causal: training uses only realized returns.
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
    generate_rff_params, compute_rff_features,
)
from run_voc_qp import qp_optimize_aggressive

WARMUP = 360  # Need enough history for time-series regression
REPORT = []


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTION A: Univariate Time-Series VoC (per-asset)
# ═══════════════════════════════════════════════════════════════════════════════

def run_univariate_voc(features, ret_np, valid_mask, P2, seed,
                       z=1e-3, retrain_every=6, max_window=0):
    """
    Kelly VoC Paper 1: per-asset time-series ridge regression.
    
    For each asset i independently:
      beta_i = (zI + T^{-1} sum_{s=1}^{t} S_{i,s} S'_{i,s})^{-1}
               * (T^{-1} sum_{s=1}^{t} S_{i,s} R_{i,s+1})
    
    Using incremental sufficient statistics per asset.
    
    max_window=0: expanding (all history)
    max_window>0: rolling window of that many bars
    """
    T, N, _ = features.shape
    
    alpha = np.zeros((T, N))
    
    # Per-asset sufficient statistics
    STS = np.zeros((N, P2, P2))  # S'S accumulators per asset
    STR = np.zeros((N, P2))       # S'R accumulators per asset
    n_obs = np.zeros(N, dtype=int)
    
    # For rolling window: ring buffer of per-bar contributions
    if max_window > 0:
        # Store (bar_idx, asset_mask, STS_contributions, STR_contributions)
        from collections import deque
        bar_history = deque()
    
    beta = np.full((N, P2), np.nan)
    last_refit = -retrain_every
    n_refits = 0
    
    for t in range(WARMUP, T):
        # Add bar t's training pair: features[t-1] -> ret[t]
        # ret[t] = close[t]/close[t-1] - 1, known at close of bar t
        if t >= 1:
            for i in range(N):
                if valid_mask[t-1, i] and valid_mask[t, i]:
                    s = features[t-1, i, :]  # (P2,)
                    r = ret_np[t, i]
                    ss = np.outer(s, s)
                    sr = s * r
                    STS[i] += ss
                    STR[i] += sr
                    n_obs[i] += 1
                    
                    if max_window > 0:
                        bar_history.append((t, i, ss, sr))
            
            # Remove old bars if rolling window exceeded
            if max_window > 0:
                while len(bar_history) > 0 and bar_history[0][0] <= t - max_window:
                    old_t, old_i, old_ss, old_sr = bar_history.popleft()
                    STS[old_i] -= old_ss
                    STR[old_i] -= old_sr
                    n_obs[old_i] -= 1
        
        # Refit beta
        if t - last_refit >= retrain_every or n_refits == 0:
            for i in range(N):
                if n_obs[i] < 10:
                    continue
                T_i = n_obs[i]
                # beta_i = (zI + T_i^{-1} STS_i)^{-1} * (T_i^{-1} STR_i)
                A = z * np.eye(P2) + STS[i] / T_i
                b = STR[i] / T_i
                try:
                    beta[i] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    pass
            
            last_refit = t
            n_refits += 1
        
        # Generate signals
        for i in range(N):
            if valid_mask[t, i] and not np.isnan(beta[i, 0]):
                alpha[t, i] = features[t, i, :] @ beta[i]
        
        # Cross-sectional demean
        vm = valid_mask[t]
        if vm.sum() > 0:
            valid_alphas = alpha[t, vm]
            if len(valid_alphas) > 0:
                alpha[t, vm] -= np.mean(valid_alphas)
    
    return alpha, n_refits


def run_univariate_voc_fast(features, ret_np, valid_mask, P2,
                            z=1e-3, retrain_every=6, max_window=0):
    """
    Faster version: vectorized per-asset update using batched operations.
    Pools all assets together but maintains per-asset STS/STR.
    """
    T, N, _ = features.shape
    
    alpha = np.zeros((T, N))
    
    # Per-asset sufficient statistics
    STS = np.zeros((N, P2, P2))
    STR = np.zeros((N, P2))
    n_obs = np.zeros(N, dtype=int)
    
    beta = np.zeros((N, P2))
    last_refit = -retrain_every
    n_refits = 0
    
    # Ring buffer for rolling window
    if max_window > 0:
        from collections import deque
        bar_sts = deque()  # list of (per_asset_ss, per_asset_sr, per_asset_valid)
    
    for t in range(1, T):
        # Vectorized update: add training pair features[t-1] -> ret[t]
        mask = valid_mask[t-1] & valid_mask[t]
        if mask.any():
            S = features[t-1, mask, :]   # (n_valid, P2)
            R = ret_np[t, mask]           # (n_valid,)
            
            # Per-asset outer products
            per_asset_ss = np.einsum('ip,iq->ipq', S, S)  # (n_valid, P2, P2)
            per_asset_sr = S * R[:, np.newaxis]            # (n_valid, P2)
            
            valid_idx = np.where(mask)[0]
            for j, idx in enumerate(valid_idx):
                STS[idx] += per_asset_ss[j]
                STR[idx] += per_asset_sr[j]
                n_obs[idx] += 1
            
            if max_window > 0:
                bar_sts.append((per_asset_ss.copy(), per_asset_sr.copy(), 
                               valid_idx.copy()))
                
                # Remove old bars
                while len(bar_sts) > max_window:
                    old_ss, old_sr, old_idx = bar_sts.popleft()
                    for j, idx in enumerate(old_idx):
                        STS[idx] -= old_ss[j]
                        STR[idx] -= old_sr[j]
                        n_obs[idx] -= 1
        
        if t < WARMUP:
            continue
        
        # Refit beta
        if t - last_refit >= retrain_every or n_refits == 0:
            for i in range(N):
                if n_obs[i] < 10:
                    continue
                T_i = n_obs[i]
                try:
                    beta[i] = np.linalg.solve(
                        z * np.eye(P2) + STS[i] / T_i, STR[i] / T_i)
                except np.linalg.LinAlgError:
                    pass
            
            last_refit = t
            n_refits += 1
        
        # Generate signals
        if n_refits > 0:
            for i in range(N):
                if valid_mask[t, i]:
                    alpha[t, i] = features[t, i, :] @ beta[i]
            vm = valid_mask[t]
            if vm.sum() > 0:
                alpha[t, vm] -= np.mean(alpha[t, vm])
    
    return alpha, n_refits


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTION B: Factor Portfolio VoC / AIPT
# ═══════════════════════════════════════════════════════════════════════════════

def build_factor_returns(features, ret_np, valid_mask):
    """
    Build P characteristic-managed factor portfolios.
    
    F_{p,t+1} = N_t^{-1/2} * sum_i S_{p,i,t} * R_{i,t+1}
    
    Each factor p is a portfolio that goes long/short assets weighted by
    the p-th RFF characteristic, earning next-bar returns.
    
    Returns: (T, P) array of factor returns.
    """
    T, N, P = features.shape
    factor_returns = np.zeros((T, P))
    
    for t in range(1, T):
        mask = valid_mask[t-1] & valid_mask[t]
        n = mask.sum()
        if n < 3:
            continue
        
        # S_{p,i,t-1} are the RFF characteristics at t-1
        # R_{i,t} are the realized returns of bar t
        S = features[t-1, mask, :]   # (n, P) - characteristics at t-1
        R = ret_np[t, mask]           # (n,) - returns at t (realized)
        
        # Factor return: F_{p,t} = N^{-1/2} * S[:,p]' @ R
        scale = 1.0 / np.sqrt(n)
        factor_returns[t] = scale * (S.T @ R)  # (P,)
    
    return factor_returns


def run_aipt_sdf(features, ret_np, valid_mask, P2,
                 z=1e-3, retrain_every=6, max_window=0):
    """
    Kelly AIPT Paper 2: Factor portfolio tangency.
    
    1. Build P factor portfolios (characteristic-managed)
    2. Expanding/rolling ridge Markowitz on factor return time series:
       lambda(z) = (zI + E[F F'])^{-1} E[F]
    3. SDF portfolio weights: w_{i,t} = S_{i,t}' lambda
    
    All strictly causal: factor returns at time t use ret[t] (realized).
    Lambda is estimated from factor returns up to time t.
    Portfolio weights at time t use features[t] and lambda estimated from [1..t].
    The position earns ret[t+1] (via delay=0 simulator).
    """
    T, N, _ = features.shape
    
    # Pre-build all factor returns (causal: F[t] uses features[t-1] and ret[t])
    factor_returns = build_factor_returns(features, ret_np, valid_mask)
    
    # Incremental sufficient statistics for Markowitz
    FTF = np.zeros((P2, P2))  # sum F_t F_t'
    FT_sum = np.zeros(P2)     # sum F_t
    n_bars = 0
    
    if max_window > 0:
        from collections import deque
        bar_queue = deque()
    
    alpha = np.zeros((T, N))
    lam = None
    last_refit = -retrain_every
    n_refits = 0
    complexity_log = []
    
    for t in range(1, T):
        # Add factor returns of bar t (realized)
        f_t = factor_returns[t]
        if np.any(f_t != 0):
            ftf = np.outer(f_t, f_t)
            FTF += ftf
            FT_sum += f_t
            n_bars += 1
            
            if max_window > 0:
                bar_queue.append((ftf, f_t.copy()))
                if len(bar_queue) > max_window:
                    old_ftf, old_ft = bar_queue.popleft()
                    FTF -= old_ftf
                    FT_sum -= old_ft
                    n_bars -= 1
        
        if t < WARMUP:
            continue
        
        # Refit lambda
        if t - last_refit >= retrain_every or lam is None:
            if n_bars < 10:
                continue
            
            c_ratio = P2 / n_bars
            complexity_log.append((t, n_bars, c_ratio))
            
            # lambda = (zI + T^{-1} FTF)^{-1} * (T^{-1} FT_sum)
            A = z * np.eye(P2) + FTF / n_bars
            b = FT_sum / n_bars
            
            try:
                lam = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            
            last_refit = t
            n_refits += 1
        
        # Portfolio weights: w_{i,t} = S_{i,t}' @ lambda
        if lam is not None:
            mask = valid_mask[t]
            if mask.sum() > 0:
                S_t = features[t, mask, :]  # (n_valid, P2)
                w = S_t @ lam               # (n_valid,) - raw weights
                # These ARE the portfolio weights (not signals)
                # Demean to be dollar-neutral
                w -= np.mean(w)
                # Normalize to sum of abs = 1
                w_abs = np.abs(w).sum()
                if w_abs > 1e-10:
                    w /= w_abs
                alpha[t, mask] = w
    
    return alpha, n_refits, complexity_log


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def eval_split(signal_df, returns_df, close_df, universe_df, start, end, fees_bps):
    w = signal_df.loc[start:end]
    r = returns_df.loc[start:end]
    c = close_df.loc[start:end]
    u = universe_df.loc[start:end]
    if len(w) == 0:
        return None
    return simulate(w, r, c, u, fees_bps=fees_bps)


def rpt(msg):
    print(msg, flush=True)
    REPORT.append(msg)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    
    rpt("Loading augmented data...")
    matrices, universe_df, valid_tickers = load_full_data()
    aug_matrices, n_alphas = augment_matrices_with_db_alphas(matrices, universe_df)
    panel, ret_np, valid_mask, dates, tickers, _ = prepare_panel(aug_matrices, valid_tickers)
    std = cross_sectional_standardize(panel)
    std = np.nan_to_num(std, nan=0.0)
    del panel; gc.collect()
    
    T, N, K = std.shape
    rpt(f"  Panel: T={T}, N={N}, K={K}")
    
    returns_df = aug_matrices["returns"]
    close_df = aug_matrices["close"]
    
    SEEDS = 5
    
    rpt(f"\n{'='*110}")
    rpt(f"  PROPER KELLY VoC IMPLEMENTATION | K={K}, {SEEDS} seeds")
    rpt(f"  Option A: Univariate time-series (per-asset)")
    rpt(f"  Option B: Factor portfolio AIPT (characteristic-managed)")
    rpt(f"{'='*110}")
    
    all_results = []
    
    # ─── Option A: Univariate Time-Series VoC ─────────────────────────
    rpt(f"\n{'='*80}")
    rpt(f"  OPTION A: Univariate Time-Series VoC (per-asset)")
    rpt(f"{'='*80}")
    
    # P sweep with fixed z=1e-3 (Kelly's recommended z* = c/b*)
    rpt(f"\n  --- A1: P sweep (z=1e-3, expanding, retrain=6) ---")
    rpt(f"  {'P':>5}  {'2P':>5}  {'c=2P/T_avg':>10}  {'Val SR@5':>9}  {'Test SR@5':>10}  {'TO':>6}  time")
    
    for P_half in [25, 50, 125, 250, 500]:
        P2 = 2 * P_half
        t1 = time.time()
        
        alpha_sum = np.zeros((T, N))
        for s in range(SEEDS):
            omega, gammas = generate_rff_params(K, P_half, seed=42+s*137)
            features = compute_rff_features(std, omega, gammas)
            a, nr = run_univariate_voc_fast(
                features, ret_np, valid_mask, P2,
                z=1e-3, retrain_every=6, max_window=0)
            alpha_sum += a
            del features; gc.collect()
        alpha_avg = alpha_sum / SEEDS
        
        # QP optimize and evaluate
        alpha_df = pd.DataFrame(alpha_avg, index=dates, columns=tickers)
        alpha_df = process_signal(alpha_df, universe_df=universe_df, max_wt=MAX_WEIGHT)
        qp = qp_optimize_aggressive(
            alpha_df, aug_matrices, universe_df, returns_df,
            risk_aversion=1.0, tcost_lambda=0.005,
            lookback_bars=120, rebal_every=1, max_turnover=0.05)
        
        val_sr = test_sr = test_to = 0.0
        for sn, (start, end) in SPLITS.items():
            sim = eval_split(qp, returns_df, close_df, universe_df, start, end, 5.0)
            if sim:
                if sn == "val":  val_sr = sim.sharpe
                if sn == "test": test_sr = sim.sharpe; test_to = sim.turnover
        
        # Approximate c: T_avg ~ T/2 for expanding window
        c_ratio = P2 / (T / 2)
        rpt(f"  {P_half:>5}  {P2:>5}  {c_ratio:>10.3f}  {val_sr:+9.3f}  {test_sr:+10.3f}  {test_to:6.4f}  {time.time()-t1:.0f}s")
        all_results.append(dict(method="A_univar", P=P2, z=1e-3, window="expand",
                                val_sr5=val_sr, test_sr5=test_sr, test_to=test_to))
        del alpha_sum, alpha_avg, alpha_df, qp; gc.collect()
    
    # z sweep with P=500 (250 half)
    rpt(f"\n  --- A2: z sweep (P=500, expanding, retrain=6) ---")
    rpt(f"  {'z':>8}  {'Val SR@5':>9}  {'Test SR@5':>10}  {'TO':>6}  time")
    
    for z in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        t1 = time.time()
        alpha_sum = np.zeros((T, N))
        for s in range(SEEDS):
            omega, gammas = generate_rff_params(K, 250, seed=42+s*137)
            features = compute_rff_features(std, omega, gammas)
            a, nr = run_univariate_voc_fast(
                features, ret_np, valid_mask, 500,
                z=z, retrain_every=6, max_window=0)
            alpha_sum += a
            del features; gc.collect()
        alpha_avg = alpha_sum / SEEDS
        
        alpha_df = pd.DataFrame(alpha_avg, index=dates, columns=tickers)
        alpha_df = process_signal(alpha_df, universe_df=universe_df, max_wt=MAX_WEIGHT)
        qp = qp_optimize_aggressive(
            alpha_df, aug_matrices, universe_df, returns_df,
            risk_aversion=1.0, tcost_lambda=0.005,
            lookback_bars=120, rebal_every=1, max_turnover=0.05)
        
        val_sr = test_sr = test_to = 0.0
        for sn, (start, end) in SPLITS.items():
            sim = eval_split(qp, returns_df, close_df, universe_df, start, end, 5.0)
            if sim:
                if sn == "val":  val_sr = sim.sharpe
                if sn == "test": test_sr = sim.sharpe; test_to = sim.turnover
        
        rpt(f"  {z:>8.0e}  {val_sr:+9.3f}  {test_sr:+10.3f}  {test_to:6.4f}  {time.time()-t1:.0f}s")
        all_results.append(dict(method="A_univar", P=500, z=z, window="expand",
                                val_sr5=val_sr, test_sr5=test_sr, test_to=test_to))
        del alpha_sum, alpha_avg, alpha_df, qp; gc.collect()
    
    # ─── Option B: Factor Portfolio AIPT ──────────────────────────────
    rpt(f"\n{'='*80}")
    rpt(f"  OPTION B: Factor Portfolio VoC / AIPT")
    rpt(f"{'='*80}")
    
    rpt(f"\n  --- B1: P sweep (z=1e-3, expanding, retrain=6) ---")
    rpt(f"  {'P':>5}  {'2P':>5}  {'c_late':>8}  {'Val SR@5':>9}  {'Test SR@5':>10}  {'TO':>6}  time")
    
    for P_half in [25, 50, 125, 250, 500]:
        P2 = 2 * P_half
        t1 = time.time()
        
        alpha_sum = np.zeros((T, N))
        all_clog = []
        for s in range(SEEDS):
            omega, gammas = generate_rff_params(K, P_half, seed=42+s*137)
            features = compute_rff_features(std, omega, gammas)
            a, nr, clog = run_aipt_sdf(
                features, ret_np, valid_mask, P2,
                z=1e-3, retrain_every=6, max_window=0)
            alpha_sum += a
            if s == 0:
                all_clog = clog
            del features; gc.collect()
        alpha_avg = alpha_sum / SEEDS
        
        c_late = all_clog[-1][2] if all_clog else 0
        
        # For AIPT, the output IS portfolio weights (not a signal to optimize)
        # But we still pass through QP for fair comparison + fee control
        alpha_df = pd.DataFrame(alpha_avg, index=dates, columns=tickers)
        alpha_df = process_signal(alpha_df, universe_df=universe_df, max_wt=MAX_WEIGHT)
        qp = qp_optimize_aggressive(
            alpha_df, aug_matrices, universe_df, returns_df,
            risk_aversion=1.0, tcost_lambda=0.005,
            lookback_bars=120, rebal_every=1, max_turnover=0.05)
        
        val_sr = test_sr = test_to = 0.0
        for sn, (start, end) in SPLITS.items():
            sim = eval_split(qp, returns_df, close_df, universe_df, start, end, 5.0)
            if sim:
                if sn == "val":  val_sr = sim.sharpe
                if sn == "test": test_sr = sim.sharpe; test_to = sim.turnover
        
        rpt(f"  {P_half:>5}  {P2:>5}  {c_late:>8.3f}  {val_sr:+9.3f}  {test_sr:+10.3f}  {test_to:6.4f}  {time.time()-t1:.0f}s")
        all_results.append(dict(method="B_aipt", P=P2, z=1e-3, window="expand",
                                c_late=c_late,
                                val_sr5=val_sr, test_sr5=test_sr, test_to=test_to))
        del alpha_sum, alpha_avg, alpha_df, qp; gc.collect()
    
    # z sweep for AIPT
    rpt(f"\n  --- B2: z sweep (P=500, expanding, retrain=6) ---")
    rpt(f"  {'z':>8}  {'Val SR@5':>9}  {'Test SR@5':>10}  {'TO':>6}  time")
    
    for z in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        t1 = time.time()
        alpha_sum = np.zeros((T, N))
        for s in range(SEEDS):
            omega, gammas = generate_rff_params(K, 250, seed=42+s*137)
            features = compute_rff_features(std, omega, gammas)
            a, nr, clog = run_aipt_sdf(
                features, ret_np, valid_mask, 500,
                z=z, retrain_every=6, max_window=0)
            alpha_sum += a
            del features; gc.collect()
        alpha_avg = alpha_sum / SEEDS
        
        alpha_df = pd.DataFrame(alpha_avg, index=dates, columns=tickers)
        alpha_df = process_signal(alpha_df, universe_df=universe_df, max_wt=MAX_WEIGHT)
        qp = qp_optimize_aggressive(
            alpha_df, aug_matrices, universe_df, returns_df,
            risk_aversion=1.0, tcost_lambda=0.005,
            lookback_bars=120, rebal_every=1, max_turnover=0.05)
        
        val_sr = test_sr = test_to = 0.0
        for sn, (start, end) in SPLITS.items():
            sim = eval_split(qp, returns_df, close_df, universe_df, start, end, 5.0)
            if sim:
                if sn == "val":  val_sr = sim.sharpe
                if sn == "test": test_sr = sim.sharpe; test_to = sim.turnover
        
        rpt(f"  {z:>8.0e}  {val_sr:+9.3f}  {test_sr:+10.3f}  {test_to:6.4f}  {time.time()-t1:.0f}s")
        all_results.append(dict(method="B_aipt", P=500, z=z, window="expand",
                                val_sr5=val_sr, test_sr5=test_sr, test_to=test_to))
        del alpha_sum, alpha_avg, alpha_df, qp; gc.collect()
    
    # ─── Summary ─────────────────────────────────────────────────────
    rpt(f"\n{'='*110}")
    rpt(f"  SUMMARY -- Sorted by Test SR@5bps")
    rpt(f"{'='*110}")
    rpt(f"  {'Method':<12}  {'P':>5}  {'z':>8}  {'Window':>7}  {'Val SR@5':>9}  {'Test SR@5':>10}  {'TO':>6}")
    rpt(f"  {'-'*75}")
    
    all_results.sort(key=lambda x: -x["test_sr5"])
    for r in all_results:
        rpt(f"  {r['method']:<12}  {r['P']:>5}  {r['z']:.0e}  "
            f"{r.get('window',''):>7}  {r['val_sr5']:+9.3f}  {r['test_sr5']:+10.3f}  {r['test_to']:6.4f}")
    
    rpt(f"\n  Total time: {time.time()-t0:.0f}s")
    
    out_path = Path(__file__).parent / "voc_proper_results.md"
    with open(out_path, "w") as f:
        f.write("# Proper Kelly VoC Results (Univariate + AIPT)\n\n```\n")
        for line in REPORT:
            f.write(line + "\n")
        f.write("```\n")
    rpt(f"  Report saved to: {out_path}")


if __name__ == "__main__":
    main()
