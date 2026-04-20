#!/usr/bin/env python
"""
run_voc_qp.py  --  VoC with QP Optimizer + Multi-Horizon + Temporal Ridge

Fixes the turnover wall in Kelly's VoC framework:
  1. Multi-horizon factor returns (h=6,12,24,48) -> smoother signal
  2. QP optimizer with transaction cost penalty -> sparse rebalancing
  3. Temporal ridge penalty on SDF weights -> stable weights over time

Usage:
  python run_voc_qp.py                    # Default: best configs
  python run_voc_qp.py --horizon 1 6 12 24 48  # Horizon sweep
  python run_voc_qp.py --no-qp            # Skip QP (signal analysis only)
"""

import numpy as np
import pandas as pd
import time
import argparse
import sys
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_voc_complexity import (
    load_full_data, log, SPLITS, FEE_LEVELS, BOOKSIZE,
    MAX_WEIGHT, BARS_PER_DAY, NEUTRALIZE, RIDGE_WARMUP,
    simulate, process_signal, compute_ic_r2, REPORT_BUFFER, EXCLUDE_FIELDS,
    augment_matrices_with_db_alphas
)
from run_voc_sdf import (
    prepare_panel, cross_sectional_standardize,
    generate_rff_params, compute_rff_features as compute_rff_sdf,
)

WARMUP = RIDGE_WARMUP
REPORT_FILE = Path(__file__).parent / "voc_qp_results.md"


# ==============================================================================
#  MULTI-HORIZON FACTOR RETURNS
# ==============================================================================

def compute_factor_returns_multihorizon(features, ret_np, valid_mask, horizon=1):
    """Build characteristic-managed portfolio returns with h-bar cumulative returns.
    
    F_{p,t} = (1/sqrt(N_{t-1})) * S_{:,t-1,p}' R^h_{:,t}
    where R^h = cumulative return over next h bars.
    
    Longer horizon -> smoother factor returns -> smoother SDF weights -> lower TO.
    """
    T, N, P2 = features.shape
    
    # Build h-bar cumulative forward returns
    if horizon > 1:
        cum_ret = np.zeros((T, N))
        for h in range(horizon):
            shifted = np.zeros((T, N))
            if h + 1 < T:
                shifted[:T-h-1] = ret_np[h+1:]
            cum_ret += shifted
        # Last h bars have no valid forward return
        cum_ret[T-horizon:] = 0.0
    else:
        cum_ret = ret_np
    
    F = np.zeros((T, P2))
    for t in range(1, T - max(horizon, 1)):
        mask = valid_mask[t-1] & valid_mask[t]
        n = mask.sum()
        if n < 5:
            continue
        S = features[t-1, mask, :]  # lagged features
        R = cum_ret[t, mask]        # h-bar forward return
        F[t] = S.T @ R / np.sqrt(n)
    return F


def sdf_markowitz_solve_temporal(F_window, z, lam_prev=None, tau=0.0):
    """Ridge Markowitz with temporal penalty on weight changes.
    
    lam = argmin ||F*lam - mu||^2 + z||lam||^2 + tau||lam - lam_prev||^2
    
    With dual form for P > T (woodbury).
    """
    T_w, P = F_window.shape
    mu = F_window.mean(axis=0)
    
    # Effective regularization
    z_eff = z
    target = mu.copy()
    
    if lam_prev is not None and tau > 0:
        # Modified objective: (z + tau) * I, target shifted by tau * lam_prev
        z_eff = z + tau
        target = mu + tau * lam_prev
    
    if P <= T_w:
        M2 = F_window.T @ F_window / T_w
        lam = np.linalg.solve(z_eff * np.eye(P) + M2, target)
    else:
        # Dual form
        G = F_window @ F_window.T
        # Need to solve (z_eff*T*I + G) alpha = ones/T, then lam = F'alpha + (tau/(z+tau))*lam_prev
        alpha = np.linalg.solve(z_eff * T_w * np.eye(T_w) + G, np.ones(T_w) / T_w)
        lam = F_window.T @ alpha
        if lam_prev is not None and tau > 0:
            lam = lam + (tau / z_eff) * lam_prev
    
    return lam


# ==============================================================================
#  SDF WITH MULTI-HORIZON + TEMPORAL PENALTY
# ==============================================================================

def run_sdf_multihorizon(std_panel, ret_np, valid_mask, P_half, seed,
                          z=1e-5, window=720, retrain_every=6,
                          horizon=1, tau=0.0):
    """SDF factor portfolio with multi-horizon returns and temporal regularization."""
    T, N, K = std_panel.shape
    P2 = 2 * P_half
    
    omega, gammas = generate_rff_params(K, P_half, seed=seed)
    features = compute_rff_sdf(std_panel, omega, gammas)
    F = compute_factor_returns_multihorizon(features, ret_np, valid_mask, horizon=horizon)
    
    alpha = np.zeros((T, N))
    lam = None
    lam_prev = None
    last_refit = -retrain_every
    n_refits = 0
    
    for t in range(WARMUP, T):
        if t - last_refit >= retrain_every or lam is None:
            t_start = max(1, t - window)
            F_window = F[t_start:t]
            
            # Remove zero rows
            nonzero = np.any(F_window != 0, axis=1)
            F_clean = F_window[nonzero]
            
            if len(F_clean) < 20:
                continue
            
            lam = sdf_markowitz_solve_temporal(F_clean, z, lam_prev=lam_prev, tau=tau)
            lam_prev = lam.copy()
            last_refit = t
            n_refits += 1
        
        if lam is not None:
            alpha[t] = features[t] @ lam
            vm = valid_mask[t]
            if vm.sum() > 0:
                alpha[t, vm] -= np.mean(alpha[t, vm])
    
    del features, F
    gc.collect()
    return alpha, n_refits


def run_fm_multihorizon(std_panel, ret_np, valid_mask, P_half, seed,
                         z=0.001, fm_window=1, retrain_every=1,
                         horizon=1, tau=0.0):
    """Fama-MacBeth with multi-horizon target and temporal regularization."""
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
    beta_prev = None
    last_refit = -retrain_every
    n_refits = 0
    
    for t in range(WARMUP, T):
        if t - last_refit >= retrain_every or beta is None:
            S_list, R_list = [], []
            for w in range(fm_window):
                tw = t - w
                if tw < 1:
                    continue
                mask = valid_mask[tw-1] & valid_mask[tw]
                n = mask.sum()
                if n < 5:
                    continue
                S_list.append(features[tw-1, mask, :])
                R_list.append(cum_ret[tw, mask])
            
            if not S_list:
                continue
            
            S = np.vstack(S_list)
            R = np.concatenate(R_list)
            n_obs = len(R)
            
            # Effective ridge with temporal penalty
            z_eff = z
            if beta_prev is not None and tau > 0:
                z_eff = z + tau
            
            if P2 <= n_obs:
                STS = S.T @ S
                STR = S.T @ R
                if beta_prev is not None and tau > 0:
                    STR = STR + tau * beta_prev
                beta = np.linalg.solve(z_eff * np.eye(P2) + STS, STR)
            else:
                SST = S @ S.T
                a = np.linalg.solve(z_eff * np.eye(n_obs) + SST, R)
                beta = S.T @ a
                if beta_prev is not None and tau > 0:
                    beta = beta + (tau / z_eff) * beta_prev
            
            beta_prev = beta.copy()
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


# ==============================================================================
#  QP OPTIMIZER (enhanced with aggressive turnover control)
# ==============================================================================

def qp_optimize_aggressive(alpha_df, matrices, universe_df, returns_df,
                            risk_aversion=1.0, tcost_lambda=0.005,
                            lookback_bars=120, rebal_every=12,
                            max_turnover=0.10):
    """QP optimizer with aggressive turnover control.
    
    Key differences from base QP:
      - Much higher tcost_lambda (0.005 vs 0.0005)
      - Less frequent rebalancing (every 12 vs 6 bars)
      - Max per-rebalance turnover constraint
      - Higher risk aversion (smaller positions)
    """
    try:
        import cvxpy as cp
    except ImportError:
        log("  QP: cvxpy not installed, returning raw signal")
        return alpha_df

    close_df = matrices["close"]
    dates = close_df.index
    tickers = close_df.columns.tolist()
    n_tickers = len(tickers)
    n_bars = len(dates)

    ret_df = returns_df.reindex(index=dates, columns=tickers).fillna(0.0).values
    signal_vals = alpha_df.reindex(index=dates, columns=tickers).fillna(0.0).values

    opt_weights = np.zeros((n_bars, n_tickers))
    prev_w = np.zeros(n_tickers)
    rebal_count = 0
    skip_count = 0
    
    for t in range(lookback_bars, n_bars):
        if t % rebal_every != 0:
            opt_weights[t] = prev_w
            continue

        ret_window = ret_df[max(0, t - lookback_bars):t]
        if ret_window.shape[0] < 30:
            opt_weights[t] = prev_w
            continue

        alpha_t = np.nan_to_num(signal_vals[t], nan=0.0)
        
        # Skip if signal is too weak
        if np.max(np.abs(alpha_t)) < 1e-10:
            opt_weights[t] = prev_w
            skip_count += 1
            continue
        
        cov = np.cov(ret_window.T)
        if cov.shape[0] != n_tickers:
            opt_weights[t] = prev_w
            continue

        # Stronger shrinkage toward diagonal
        cov = 0.3 * cov + 0.7 * np.diag(np.diag(cov)) + 1e-6 * np.eye(n_tickers)

        try:
            w = cp.Variable(n_tickers)
            obj = alpha_t @ w - risk_aversion * cp.quad_form(w, cp.psd_wrap(cov))
            
            # Transaction cost penalty (L1)
            obj -= tcost_lambda * cp.norm1(w - prev_w)
            
            constraints = [
                cp.sum(w) == 0,             # market neutral
                cp.norm1(w) <= 2.0,          # gross leverage <= 2
                w >= -MAX_WEIGHT,
                w <= MAX_WEIGHT,
            ]
            
            # Max turnover per rebalance
            if max_turnover > 0 and np.any(prev_w != 0):
                constraints.append(cp.norm1(w - prev_w) <= max_turnover * 2)
            
            prob = cp.Problem(cp.Maximize(obj), constraints)
            prob.solve(solver=cp.SCS, verbose=False, max_iters=3000)
            
            if prob.status in ["optimal", "optimal_inaccurate"]:
                opt_w = np.nan_to_num(w.value, nan=0.0)
                prev_w = opt_w
                opt_weights[t] = opt_w
                rebal_count += 1
            else:
                opt_weights[t] = prev_w
        except Exception:
            opt_weights[t] = prev_w

    # Fill non-rebalance bars
    for t in range(lookback_bars, n_bars):
        if t % rebal_every != 0:
            opt_weights[t] = opt_weights[t - 1] if t > 0 else np.zeros(n_tickers)

    opt_weights[:lookback_bars] = 0.0
    log(f"    QP: {rebal_count} rebalances, {skip_count} skipped (weak signal)")
    return pd.DataFrame(opt_weights, index=dates, columns=tickers)


# ==============================================================================
#  EVALUATION (reuse from run_voc_sdf but with proper sim engine)
# ==============================================================================

def evaluate_full(alpha_df, matrices, universe_df, name="signal", is_qp=False):
    """Full evaluation using the vectorized sim engine."""
    close = matrices["close"]
    returns = matrices["returns"]
    
    results = {}
    ic_r2 = {}
    
    for split_name, (start, end) in SPLITS.items():
        sig_slice = alpha_df.loc[start:end]
        split_close = close.loc[start:end]
        split_returns = returns.loc[start:end]
        split_universe = universe_df.loc[start:end]
        
        ic_r2[split_name] = compute_ic_r2(alpha_df, returns, universe_df, start, end)
        
        if is_qp:
            processed = sig_slice.clip(lower=-MAX_WEIGHT, upper=MAX_WEIGHT)
        else:
            processed = process_signal(sig_slice, universe_df=split_universe, max_wt=MAX_WEIGHT)
        
        for fee in FEE_LEVELS:
            try:
                sim = simulate(processed, split_returns, split_close, split_universe, fees_bps=fee)
                results[(split_name, fee)] = sim
            except Exception as e:
                log(f"    ERROR: {name}/{split_name}/{fee}bps: {e}")
    
    return results, ic_r2


def print_compact_results(results, ic_r2, label):
    """Print compact results for one method."""
    log(f"\n  {label}")
    for split in ["train", "val", "test"]:
        ic = ic_r2.get(split, {}).get("ic_mean", 0)
        parts = [f"    {split:5s}: IC={ic:+.4f}"]
        for fee in [0.0, 5.0, 7.0]:
            r = results.get((split, fee))
            if r:
                ret = r.total_pnl / BOOKSIZE * 100
                parts.append(f"{fee:.0f}bps: SR={r.sharpe:+.3f} TO={r.turnover:.4f} ({ret:+.1f}%)")
        log(", ".join(parts))


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="VoC + QP: Breaking the Turnover Wall")
    parser.add_argument("--method", type=str, default="both", choices=["sdf", "fm", "both"])
    parser.add_argument("--P", type=int, nargs="+", default=[500],
                        help="P_half values (best from prior runs)")
    parser.add_argument("--z", type=float, nargs="+", default=[1e-5],
                        help="Ridge z values")
    parser.add_argument("--horizon", type=int, nargs="+", default=[1, 6, 12, 24, 48],
                        help="Forward return horizons (bars)")
    parser.add_argument("--window", type=int, nargs="+", default=[720],
                        help="SDF training window")
    parser.add_argument("--fm-window", type=int, nargs="+", default=[1],
                        help="FM pooling window")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of random seeds")
    parser.add_argument("--retrain", type=int, default=6,
                        help="Retrain every N bars")
    parser.add_argument("--tau", type=float, nargs="+", default=[0.0, 0.01, 0.1],
                        help="Temporal ridge penalty values")
    parser.add_argument("--no-qp", action="store_true",
                        help="Skip QP optimizer")
    parser.add_argument("--qp-tcost", type=float, nargs="+", default=[0.001, 0.005, 0.01],
                        help="QP transaction cost lambda values")
    parser.add_argument("--qp-rebal", type=int, nargs="+", default=[6, 12, 24],
                        help="QP rebalance frequency (bars)")
    parser.add_argument("--qp-max-to", type=float, nargs="+", default=[0.05, 0.10, 0.20],
                        help="QP max turnover per rebalance")
    args = parser.parse_args()
    
    t0_total = time.time()
    REPORT_BUFFER.clear()
    
    # Load data
    matrices, universe_df, valid_tickers = load_full_data()
    
    # Augment matrices with DB alpha signals (evaluates expressions)
    log("  Augmenting with DB alpha signals...")
    aug_matrices, n_alphas = augment_matrices_with_db_alphas(matrices, universe_df)
    log(f"  Added {n_alphas} alpha signals to matrices")
    
    # Build panel from augmented matrices
    panel, ret_np, valid_mask, dates, tickers, char_keys = prepare_panel(aug_matrices, valid_tickers)
    
    T, N, K = panel.shape
    log(f"\n  Panel: T={T}, N={N}, K={K} ({K - n_alphas} raw + {n_alphas} alpha signals)")
    
    # Cross-sectional standardize
    log("  Cross-sectional standardizing...")
    t0 = time.time()
    std_panel = cross_sectional_standardize(panel)
    std_panel = np.nan_to_num(std_panel, nan=0.0)
    log(f"  Standardized in {time.time()-t0:.1f}s")
    
    log(f"\n{'='*100}")
    log(f"  VoC + QP: Breaking the Turnover Wall")
    log(f"  Panel: T={T}, N={N}, K={K}")
    log(f"  Horizons: {args.horizon}")
    log(f"  Temporal penalties (tau): {args.tau}")
    log(f"  QP tcost lambdas: {args.qp_tcost}")
    log(f"  QP rebal frequencies: {args.qp_rebal}")
    log(f"  QP max turnover: {args.qp_max_to}")
    log(f"{'='*100}")
    
    all_results = []
    returns_df = matrices["returns"]
    close_df = matrices["close"]
    
    # ==========================================================================
    # Phase 1: HORIZON SWEEP -- Find natural turnover reduction
    # ==========================================================================
    
    log(f"\n{'-'*80}")
    log(f"  PHASE 1: Multi-Horizon Signal Generation")
    log(f"{'-'*80}")
    
    best_signals = {}  # Store best raw signals for Phase 2 QP
    
    for horizon in args.horizon:
        for P_half in args.P:
            for z in args.z:
                for tau in args.tau:
                    tau_label = f" tau={tau:.2g}" if tau > 0 else ""
                    
                    # ── SDF ──
                    if args.method in ["sdf", "both"]:
                        for window in args.window:
                            P2 = 2 * P_half
                            c = P2 / window
                            label = f"SDF P={P_half} z={z:.0e} w={window} h={horizon}{tau_label}"
                            log(f"\n  === {label} (c={c:.2f}) ===")
                            
                            t0 = time.time()
                            alpha_sum = np.zeros((T, N))
                            
                            for s in range(args.seeds):
                                seed = 42 + s * 137
                                a, nr = run_sdf_multihorizon(
                                    std_panel, ret_np, valid_mask, P_half, seed,
                                    z=z, window=window, retrain_every=args.retrain,
                                    horizon=horizon, tau=tau)
                                alpha_sum += a
                            
                            alpha_avg = alpha_sum / args.seeds
                            elapsed = time.time() - t0
                            log(f"  {args.seeds} seeds in {elapsed:.0f}s")
                            
                            # Build DataFrame
                            alpha_df = pd.DataFrame(alpha_avg, index=dates, columns=tickers)
                            
                            # Evaluate raw signal
                            results, ic_r2 = evaluate_full(alpha_df, matrices, universe_df, name=label)
                            print_compact_results(results, ic_r2, label)
                            
                            # Track for summary
                            test_0 = results.get(("test", 0.0))
                            test_5 = results.get(("test", 5.0))
                            all_results.append({
                                "method": "SDF", "P": P_half, "z": z, "window": window,
                                "horizon": horizon, "tau": tau, "qp": "raw",
                                "test_sr0": test_0.sharpe if test_0 else 0,
                                "test_sr5": test_5.sharpe if test_5 else 0,
                                "test_to": test_0.turnover if test_0 else 0,
                                "test_ic": ic_r2.get("test", {}).get("ic_mean", 0),
                            })
                            
                            # Store for QP phase
                            key = f"SDF_P{P_half}_z{z}_w{window}_h{horizon}_tau{tau}"
                            best_signals[key] = alpha_df
                    
                    # ── FM ──
                    if args.method in ["fm", "both"]:
                        for fm_w in args.fm_window:
                            P2 = 2 * P_half
                            c = P2 / (N * fm_w)
                            label = f"FM P={P_half} z={z:.0e} w={fm_w} h={horizon}{tau_label}"
                            log(f"\n  === {label} (c={c:.2f}) ===")
                            
                            t0 = time.time()
                            alpha_sum = np.zeros((T, N))
                            
                            for s in range(args.seeds):
                                seed = 42 + s * 137
                                a, nr = run_fm_multihorizon(
                                    std_panel, ret_np, valid_mask, P_half, seed,
                                    z=z, fm_window=fm_w, retrain_every=args.retrain,
                                    horizon=horizon, tau=tau)
                                alpha_sum += a
                            
                            alpha_avg = alpha_sum / args.seeds
                            elapsed = time.time() - t0
                            log(f"  {args.seeds} seeds in {elapsed:.0f}s")
                            
                            alpha_df = pd.DataFrame(alpha_avg, index=dates, columns=tickers)
                            
                            results, ic_r2 = evaluate_full(alpha_df, matrices, universe_df, name=label)
                            print_compact_results(results, ic_r2, label)
                            
                            test_0 = results.get(("test", 0.0))
                            test_5 = results.get(("test", 5.0))
                            all_results.append({
                                "method": "FM", "P": P_half, "z": z, "window": fm_w,
                                "horizon": horizon, "tau": tau, "qp": "raw",
                                "test_sr0": test_0.sharpe if test_0 else 0,
                                "test_sr5": test_5.sharpe if test_5 else 0,
                                "test_to": test_0.turnover if test_0 else 0,
                                "test_ic": ic_r2.get("test", {}).get("ic_mean", 0),
                            })
                            
                            key = f"FM_P{P_half}_z{z}_w{fm_w}_h{horizon}_tau{tau}"
                            best_signals[key] = alpha_df
    
    # ==========================================================================
    # Phase 2: QP OPTIMIZATION on best raw signals
    # ==========================================================================
    
    if not args.no_qp:
        log(f"\n{'-'*80}")
        log(f"  PHASE 2: QP Optimization Sweep")
        log(f"{'-'*80}")
        
        # Select signals with best test SR@0 to QP-optimize
        # Take top 5 by test_sr0
        sorted_raw = sorted(all_results, key=lambda x: x["test_sr0"], reverse=True)
        top_keys = []
        for r in sorted_raw[:5]:
            tau_str = r["tau"]
            if r["method"] == "SDF":
                key = f"SDF_P{r['P']}_z{r['z']}_w{r['window']}_h{r['horizon']}_tau{tau_str}"
            else:
                key = f"FM_P{r['P']}_z{r['z']}_w{r['window']}_h{r['horizon']}_tau{tau_str}"
            if key in best_signals:
                top_keys.append((key, r))
        
        log(f"  QP-optimizing top {len(top_keys)} signals by test SR@0")
        
        for key, raw_info in top_keys:
            alpha_df = best_signals[key]
            label_base = f"{raw_info['method']} P={raw_info['P']} h={raw_info['horizon']}"
            if raw_info['tau'] > 0:
                label_base += f" tau={raw_info['tau']:.2g}"
            
            for tcost in args.qp_tcost:
                for rebal in args.qp_rebal:
                    for max_to in args.qp_max_to:
                        label = f"{label_base} +QP(tc={tcost},rb={rebal},to={max_to})"
                        log(f"\n  === {label} ===")
                        
                        t0 = time.time()
                        qp_alpha = qp_optimize_aggressive(
                            alpha_df, matrices, universe_df, returns_df,
                            risk_aversion=1.0, tcost_lambda=tcost,
                            lookback_bars=120, rebal_every=rebal,
                            max_turnover=max_to)
                        
                        results, ic_r2 = evaluate_full(
                            qp_alpha, matrices, universe_df,
                            name=label, is_qp=True)
                        print_compact_results(results, ic_r2, label)
                        log(f"    QP time: {time.time()-t0:.0f}s")
                        
                        test_0 = results.get(("test", 0.0))
                        test_5 = results.get(("test", 5.0))
                        all_results.append({
                            "method": raw_info["method"], "P": raw_info["P"],
                            "z": raw_info["z"], "window": raw_info["window"],
                            "horizon": raw_info["horizon"], "tau": raw_info["tau"],
                            "qp": f"tc={tcost}_rb={rebal}_to={max_to}",
                            "test_sr0": test_0.sharpe if test_0 else 0,
                            "test_sr5": test_5.sharpe if test_5 else 0,
                            "test_to": test_0.turnover if test_0 else 0,
                            "test_ic": ic_r2.get("test", {}).get("ic_mean", 0),
                        })
    
    # ==========================================================================
    # Summary Report
    # ==========================================================================
    
    total_time = time.time() - t0_total
    
    log(f"\n\n{'='*120}")
    log(f"  FINAL SUMMARY -- Sorted by Test SR@5bps (fee-survivability)")
    log(f"{'='*120}")
    
    sorted_all = sorted(all_results, key=lambda x: x["test_sr5"], reverse=True)
    
    log(f"\n  {'Method':<8} {'P':>4} {'h':>3} {'tau':>6} {'QP Config':<30} | "
        f"{'Test SR@0':>10} {'Test SR@5':>10} {'Test TO':>8} {'Test IC':>8}")
    log(f"  {'-'*110}")
    
    for r in sorted_all:
        tau_str = f"{r['tau']:.2g}" if r['tau'] > 0 else "0"
        log(f"  {r['method']:<8} {r['P']:>4} {r['horizon']:>3} {tau_str:>6} {r['qp']:<30} | "
            f"{r['test_sr0']:>+10.3f} {r['test_sr5']:>+10.3f} {r['test_to']:>8.4f} {r['test_ic']:>+8.4f}")
    
    log(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f}m)")
    
    # Write markdown report
    write_markdown_report(sorted_all, args, total_time)
    log(f"  Report saved to: {REPORT_FILE}")


def write_markdown_report(sorted_results, args, total_time):
    """Write comprehensive markdown report."""
    lines = [
        "# VoC + QP: Breaking the Turnover Wall",
        f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"Total time: {total_time:.0f}s ({total_time/60:.1f}m)",
        "",
        "## Configuration",
        f"- Horizons: {args.horizon}",
        f"- Temporal penalties: {args.tau}",
        f"- QP tcost: {args.qp_tcost}",
        f"- QP rebalance: {args.qp_rebal}",
        f"- QP max TO: {args.qp_max_to}",
        "",
        "## Results -- Sorted by Test SR@5bps",
        "",
        "| Method | P | h | tau | QP Config | Test SR@0 | Test SR@5 | Test TO | Test IC |",
        "|--------|---|---|-----|-----------|-----------|-----------|---------|---------|",
    ]
    
    for r in sorted_results:
        tau_str = f"{r['tau']:.2g}" if r['tau'] > 0 else "0"
        lines.append(
            f"| {r['method']} | {r['P']} | {r['horizon']} | {tau_str} | {r['qp']} | "
            f"{r['test_sr0']:+.3f} | {r['test_sr5']:+.3f} | {r['test_to']:.4f} | {r['test_ic']:+.4f} |"
        )
    
    lines.extend(["", "## Console Output", "", "```text"])
    lines.extend(REPORT_BUFFER)
    lines.extend(["```", ""])
    
    REPORT_FILE.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
