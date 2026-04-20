#!/usr/bin/env python
"""
run_voc_proper_gpu.py -- GPU-accelerated proper Kelly VoC.

Same logic as run_voc_proper.py but all heavy computation on CUDA.

Speedup estimates vs CPU version:
  - RFF features: ~20x (batched einsum + sin/cos on GPU)
  - Per-asset STS/STR accumulation: ~30-50x (batched outer products)
  - Ridge solve: ~50x (batched linalg.solve on 49 assets simultaneously)
  - Factor returns: ~20x (batched matmul)
  - Overall: estimated 20-40x end-to-end

RTX 4060 Laptop: 8GB VRAM, 3072 CUDA cores
  Panel: T=11892 x N=49 x K=58 = ~27M floats = ~108MB fp32
  RFF features: T=11892 x N=49 x P=1000 = ~583M floats = ~2.3GB fp32
    -> P=2000 (4000 features): ~4.6GB, tight but fits in 8GB
"""
import numpy as np
import pandas as pd
import torch
import gc, time, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_voc_complexity import (
    load_full_data, log, SPLITS, BOOKSIZE, MAX_WEIGHT,
    simulate, process_signal, augment_matrices_with_db_alphas
)
from run_voc_sdf import prepare_panel, cross_sectional_standardize
from run_voc_qp import qp_optimize_aggressive

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WARMUP = 360
REPORT = []


def rpt(msg):
    print(msg, flush=True)
    REPORT.append(msg)


# ═══════════════════════════════════════════════════════════════════════════════
#  GPU RFF GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_rff_gpu(std_panel_t, K, P_half, seed=42):
    """
    Generate RFF features entirely on GPU.
    
    std_panel_t: (T, N, K) tensor on GPU
    Returns: (T, N, 2*P_half) tensor on GPU, omega, gammas
    """
    rng = torch.Generator(device='cpu').manual_seed(seed)
    omega = torch.randn(P_half, K, generator=rng, dtype=torch.float32).to(DEVICE)
    
    gamma_choices = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    gamma_idx = torch.randint(0, 6, (P_half,), generator=rng)
    gammas = gamma_choices[gamma_idx].to(DEVICE)
    
    # proj[t,n,p] = gamma_p * Z[t,n,:] @ omega[p,:]
    proj = torch.einsum('tnk,pk->tnp', std_panel_t, omega)  # (T, N, P_half)
    proj *= gammas.unsqueeze(0).unsqueeze(0)
    
    scale = 1.0 / (P_half ** 0.5)
    sin_f = torch.sin(proj) * scale
    cos_f = torch.cos(proj) * scale
    features = torch.cat([sin_f, cos_f], dim=2)  # (T, N, 2*P_half)
    
    del proj, sin_f, cos_f
    torch.cuda.empty_cache()
    
    return features, omega, gammas


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTION A: Univariate Time-Series VoC (GPU) -- Incremental
# ═══════════════════════════════════════════════════════════════════════════════

def run_univariate_voc_gpu(features_t, ret_t, mask_t, P2,
                           z=1e-3, retrain_every=6):
    """
    GPU-accelerated per-asset time-series ridge regression.
    
    Uses INCREMENTAL sufficient statistics on GPU:
      - Between refits, accumulate batches of STS/STR updates
      - At refit: solve all 49 assets in one batched linalg.solve
      - Signal generation: vectorized matmul
    
    This avoids the O(T^2) cost of recomputing from scratch at each refit.
    """
    T, N, _ = features_t.shape
    
    alpha = torch.zeros((T, N), device=DEVICE)
    zI = z * torch.eye(P2, device=DEVICE)
    
    # Per-asset sufficient statistics on GPU
    STS = torch.zeros((N, P2, P2), device=DEVICE)  # sum S_i S_i'
    STR = torch.zeros((N, P2), device=DEVICE)       # sum S_i R_i
    n_obs = torch.zeros(N, device=DEVICE)
    
    beta = torch.zeros((N, P2), device=DEVICE)
    n_refits = 0
    last_accum_t = 0  # last bar we accumulated
    refit_times = list(range(WARMUP, T, retrain_every))
    refit_set = set(refit_times)
    
    for t in range(1, T):
        # Accumulate training pair: features[t-1] -> ret[t]
        mask = mask_t[t-1] & mask_t[t]  # (N,) bool
        if mask.any():
            S = features_t[t-1, mask, :]   # (n_valid, P2)
            R = ret_t[t, mask]              # (n_valid,)
            
            # Vectorized per-asset outer product update
            # STS[valid_idx] += S[j] outer S[j]
            ss = torch.bmm(S.unsqueeze(2), S.unsqueeze(1))  # (n_valid, P2, P2)
            sr = S * R.unsqueeze(1)                           # (n_valid, P2)
            
            valid_idx = torch.where(mask)[0]
            STS[valid_idx] += ss
            STR[valid_idx] += sr
            n_obs[valid_idx] += 1
        
        if t < WARMUP:
            continue
        
        # Refit at scheduled times
        if t in refit_set:
            valid_assets = n_obs >= 10
            if valid_assets.sum() == 0:
                continue
            
            T_inv = torch.where(n_obs > 0, 1.0 / n_obs,
                               torch.zeros_like(n_obs))
            T_inv_3d = T_inv.unsqueeze(1).unsqueeze(2)  # (N, 1, 1)
            
            A = zI.unsqueeze(0) + STS * T_inv_3d  # (N, P2, P2)
            b = STR * T_inv.unsqueeze(1)            # (N, P2)
            
            try:
                beta_new = torch.linalg.solve(A, b.unsqueeze(2)).squeeze(2)
                beta[valid_assets] = beta_new[valid_assets]
            except Exception:
                pass
            
            n_refits += 1
        
        # Generate signal (every bar after warmup)
        if n_refits > 0:
            alpha[t] = (features_t[t] * beta).sum(dim=1)
            vm = mask_t[t]
            if vm.sum() > 0:
                alpha[t, vm] -= alpha[t, vm].mean()
    
    return alpha.cpu().numpy(), n_refits


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTION B: Factor Portfolio VoC / AIPT (GPU)
# ═══════════════════════════════════════════════════════════════════════════════

def run_aipt_sdf_gpu(features_t, ret_t, mask_t, P2,
                     z=1e-3, retrain_every=6):
    """
    GPU-accelerated AIPT factor portfolio tangency.
    
    Step 1: Build factor returns (fully vectorized on GPU)
    Step 2: Expanding-window ridge Markowitz
    Step 3: Portfolio weights from SDF coefficients
    """
    T, N, _ = features_t.shape
    
    # Step 1: Build ALL factor returns on GPU at once
    # F[t, p] = N_t^{-1/2} * sum_i S[t-1, i, p] * R[t, i]
    # where mask ensures both features and returns are valid
    M_both = mask_t[:-1] & mask_t[1:]  # (T-1, N)
    S_prev = features_t[:-1] * M_both.unsqueeze(2).float()  # (T-1, N, P2)
    R_next = ret_t[1:] * M_both.float()                      # (T-1, N)
    n_valid = M_both.sum(dim=1).float().clamp(min=1)          # (T-1,)
    scale = 1.0 / torch.sqrt(n_valid).unsqueeze(1)            # (T-1, 1)
    
    # Factor returns: F[t] = scale * S_prev[t]' @ R_next[t]
    factor_returns = torch.einsum('tnp,tn->tp', S_prev, R_next)  # (T-1, P2)
    factor_returns *= scale  # (T-1, P2)
    
    # Prepend a zero row for t=0 alignment
    factor_returns = torch.cat([
        torch.zeros((1, P2), device=DEVICE), factor_returns
    ], dim=0)  # (T, P2)
    
    del S_prev, R_next, M_both
    torch.cuda.empty_cache()
    
    # Step 2: Expanding-window ridge Markowitz
    alpha = torch.zeros((T, N), device=DEVICE)
    zI = z * torch.eye(P2, device=DEVICE)
    lam = None
    n_refits = 0
    complexity_log = []
    refit_times = list(range(WARMUP, T, retrain_every))
    
    for t in refit_times:
        # Use factor returns from bars 1..t
        F = factor_returns[1:t+1]  # (t, P2)
        nonzero = (F.abs().sum(dim=1) > 0)
        F_valid = F[nonzero]
        n_bars = F_valid.shape[0]
        
        if n_bars < 10:
            continue
        
        c_ratio = P2 / n_bars
        complexity_log.append((t, n_bars, c_ratio))
        
        # FTF = F' @ F / n_bars
        FTF = (F_valid.T @ F_valid) / n_bars  # (P2, P2)
        FT_mean = F_valid.mean(dim=0)           # (P2,)
        
        # lambda = (zI + FTF)^{-1} FT_mean
        A = zI + FTF
        try:
            lam = torch.linalg.solve(A, FT_mean)  # (P2,)
        except Exception:
            continue
        
        n_refits += 1
        
        # Generate portfolio weights for bars from this refit to the next
        next_t = refit_times[refit_times.index(t) + 1] if t != refit_times[-1] else T
        for tick in range(t, min(next_t, T)):
            vm = mask_t[tick]
            if vm.sum() == 0 or lam is None:
                continue
            # w_{i,tick} = S[tick, i, :] @ lambda
            w = features_t[tick, vm] @ lam  # (n_valid,)
            w -= w.mean()
            w_abs = w.abs().sum()
            if w_abs > 1e-10:
                w /= w_abs
            alpha[tick, vm] = w
    
    return alpha.cpu().numpy(), n_refits, complexity_log


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


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    
    rpt(f"GPU: {torch.cuda.get_device_name(0)}")
    rpt(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    
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
    
    # Move base data to GPU
    std_t = torch.tensor(std, dtype=torch.float32, device=DEVICE)
    ret_t = torch.tensor(ret_np, dtype=torch.float32, device=DEVICE)
    mask_t = torch.tensor(valid_mask, dtype=torch.bool, device=DEVICE)
    del std, ret_np, valid_mask; gc.collect()
    
    rpt(f"  GPU memory after data load: {torch.cuda.memory_allocated()/1e6:.0f}MB")
    
    SEEDS = 5
    
    rpt(f"\n{'='*110}")
    rpt(f"  PROPER KELLY VoC (GPU) | K={K}, {SEEDS} seeds, RTX 4060")
    rpt(f"  Option A: Univariate time-series (per-asset)")
    rpt(f"  Option B: Factor portfolio AIPT (characteristic-managed)")
    rpt(f"{'='*110}")
    
    all_results = []
    
    # ─── Option A: Univariate Time-Series VoC ─────────────────────────
    rpt(f"\n{'='*80}")
    rpt(f"  OPTION A: Univariate Time-Series VoC (per-asset)")
    rpt(f"{'='*80}")
    
    rpt(f"\n  --- A1: P sweep (z=1e-3, expanding, retrain=6) ---")
    rpt(f"  {'P':>5}  {'2P':>5}  {'c=2P/T_avg':>10}  {'Val SR@5':>9}  {'Test SR@5':>10}  {'TO':>6}  time")
    
    for P_half in [25, 50, 125, 250, 500, 1000]:
        P2 = 2 * P_half
        t1 = time.time()
        
        alpha_sum = np.zeros((T, N))
        for s in range(SEEDS):
            features, _, _ = generate_rff_gpu(std_t, K, P_half, seed=42+s*137)
            a, nr = run_univariate_voc_gpu(
                features, ret_t, mask_t, P2,
                z=1e-3, retrain_every=6)
            alpha_sum += a
            del features; torch.cuda.empty_cache()
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
        
        c_ratio = P2 / (T / 2)
        rpt(f"  {P_half:>5}  {P2:>5}  {c_ratio:>10.3f}  {val_sr:+9.3f}  {test_sr:+10.3f}  {test_to:6.4f}  {time.time()-t1:.0f}s")
        all_results.append(dict(method="A_univar", P=P2, z=1e-3, window="expand",
                                val_sr5=val_sr, test_sr5=test_sr, test_to=test_to))
        del alpha_sum, alpha_avg, alpha_df, qp; gc.collect()
    
    # z sweep
    rpt(f"\n  --- A2: z sweep (P=500, expanding, retrain=6) ---")
    rpt(f"  {'z':>8}  {'Val SR@5':>9}  {'Test SR@5':>10}  {'TO':>6}  time")
    
    for z in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        t1 = time.time()
        alpha_sum = np.zeros((T, N))
        for s in range(SEEDS):
            features, _, _ = generate_rff_gpu(std_t, K, 250, seed=42+s*137)
            a, nr = run_univariate_voc_gpu(
                features, ret_t, mask_t, 500,
                z=z, retrain_every=6)
            alpha_sum += a
            del features; torch.cuda.empty_cache()
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
    
    for P_half in [25, 50, 125, 250, 500, 1000]:
        P2 = 2 * P_half
        t1 = time.time()
        
        alpha_sum = np.zeros((T, N))
        all_clog = []
        for s in range(SEEDS):
            features, _, _ = generate_rff_gpu(std_t, K, P_half, seed=42+s*137)
            a, nr, clog = run_aipt_sdf_gpu(
                features, ret_t, mask_t, P2,
                z=1e-3, retrain_every=6)
            alpha_sum += a
            if s == 0:
                all_clog = clog
            del features; torch.cuda.empty_cache()
        alpha_avg = alpha_sum / SEEDS
        
        c_late = all_clog[-1][2] if all_clog else 0
        
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
    
    # z sweep AIPT
    rpt(f"\n  --- B2: z sweep (P=500, expanding, retrain=6) ---")
    rpt(f"  {'z':>8}  {'Val SR@5':>9}  {'Test SR@5':>10}  {'TO':>6}  time")
    
    for z in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        t1 = time.time()
        alpha_sum = np.zeros((T, N))
        for s in range(SEEDS):
            features, _, _ = generate_rff_gpu(std_t, K, 250, seed=42+s*137)
            a, nr, clog = run_aipt_sdf_gpu(
                features, ret_t, mask_t, 500,
                z=z, retrain_every=6)
            alpha_sum += a
            del features; torch.cuda.empty_cache()
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
    rpt(f"  GPU peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    
    out_path = Path(__file__).parent / "voc_proper_results.md"
    with open(out_path, "w") as f:
        f.write("# Proper Kelly VoC Results - GPU (Univariate + AIPT)\n\n```\n")
        for line in REPORT:
            f.write(line + "\n")
        f.write("```\n")
    rpt(f"  Report saved to: {out_path}")


if __name__ == "__main__":
    main()
