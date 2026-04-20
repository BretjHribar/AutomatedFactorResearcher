#!/usr/bin/env python
"""
run_voc_sdf.py  --  AIPT: Complex Factor Pricing for 4H Crypto

Implements the Virtue of Complexity for cross-sectional asset pricing:

Approach A: SDF Factor Portfolio (Didisheim, Ke, Kelly, Malamud 2023)
  - Builds P characteristic-managed long-short factor portfolios  
  - Optimizes Markowitz portfolio with ridge on factor return time series
  - c = P/T where T = training window (bars)
  - Best paper config: P=360k, z=1e-5, T=360

Approach B: Fama-MacBeth Cross-Sectional Regression
  - Per-bar cross-sectional ridge regression (pooled over short window)
  - c = 2P / (N * W) where N = assets (~49), W = pooling window

Usage:
  python run_voc_sdf.py --method sdf --P 100 250 500 --window 360 720 --seeds 10
  python run_voc_sdf.py --method fm  --P 50 100 250 --seeds 10
  python run_voc_sdf.py --method both --P 100 250 500 --seeds 10
"""

import numpy as np
import pandas as pd
import time
import argparse
import sys
import gc
from pathlib import Path

# ─── Import from existing pipeline ─────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from run_voc_complexity import load_full_data, log, SPLITS, FEE_LEVELS

UNIVERSE = "BINANCE_TOP50"
INTERVAL = "4h"
WARMUP = 180
REPORT_FILE = Path(__file__).parent / "voc_sdf_results.md"


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_panel(matrices, valid_tickers):
    """Build (T, N, K) panel of raw characteristics and (T, N) return matrix."""
    close = matrices["close"][valid_tickers]
    ret = close.pct_change()
    
    # Select characteristics (same order as existing pipeline)
    skip = {"close", "open", "high", "low", "volume", "quote_volume",
            "taker_buy_volume", "taker_buy_quote_volume", "number_of_trades"}
    char_keys = sorted([k for k in matrices if k not in skip and k in matrices])
    
    T, N = ret.shape
    K = len(char_keys)
    
    panel = np.full((T, N, K), np.nan)
    for k_idx, key in enumerate(char_keys):
        df = matrices[key][valid_tickers]
        panel[:, :, k_idx] = df.values
    
    ret_np = ret.values.copy()
    ret_np = np.nan_to_num(ret_np, nan=0.0)
    
    valid_mask = np.isfinite(close.values) & (close.values > 0)
    
    return panel, ret_np, valid_mask, ret.index, valid_tickers, char_keys


def augment_with_alphas(panel, matrices, valid_tickers, char_keys):
    """Add DB alpha signals as additional characteristics."""
    import sqlite3
    # Try multiple locations for alphas.db
    candidates = [
        Path(__file__).parent / "data" / "alphas.db",
        Path(__file__).parent.parent / "data" / "alphas.db",
        Path(__file__).parent / "alphas.db",
    ]
    db_path = None
    for c in candidates:
        if c.exists():
            db_path = c
            break
    if db_path is None:
        log("  No alphas.db found, skipping augmentation")
        return panel, char_keys
    log(f"  Found alphas.db at {db_path}")
    
    conn = sqlite3.connect(str(db_path))
    try:
        alpha_ids = pd.read_sql("SELECT DISTINCT alpha_id FROM alpha_values WHERE interval='4h'", conn)
        if alpha_ids.empty:
            return panel, char_keys
    except Exception:
        return panel, char_keys
    
    close_df = matrices["close"][valid_tickers]
    new_chars = []
    new_keys = list(char_keys)
    
    for alpha_id in sorted(alpha_ids["alpha_id"].unique()):
        try:
            vals = pd.read_sql(
                "SELECT timestamp, ticker, value FROM alpha_values WHERE alpha_id=? AND interval='4h'",
                conn, params=(alpha_id,))
            if vals.empty:
                continue
            vals["timestamp"] = pd.to_datetime(vals["timestamp"])
            piv = vals.pivot(index="timestamp", columns="ticker", values="value")
            piv = piv.reindex(index=close_df.index, columns=valid_tickers)
            new_chars.append(piv.values)
            new_keys.append(f"alpha_{alpha_id}")
        except Exception:
            continue
    
    conn.close()
    
    if new_chars:
        T, N, K_old = panel.shape
        extra = np.stack(new_chars, axis=2)  # (T, N, n_alphas)
        panel = np.concatenate([panel, extra], axis=2)
        log(f"  Augmented: K={panel.shape[2]} characteristics ({K_old} raw + {len(new_chars)} alphas)")
    
    return panel, new_keys


def cross_sectional_standardize(panel):
    """Cross-sectional z-score: for each (time, feature), standardize across assets."""
    T, N, K = panel.shape
    out = np.zeros_like(panel)
    for t in range(T):
        for k in range(K):
            vals = panel[t, :, k]
            valid = np.isfinite(vals)
            n_valid = valid.sum()
            if n_valid < 3:
                out[t, :, k] = 0.0
            else:
                mu = np.nanmean(vals)
                sigma = np.nanstd(vals)
                if sigma < 1e-12:
                    out[t, :, k] = 0.0
                else:
                    out[t, :, k] = np.where(valid, (vals - mu) / sigma, 0.0)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  RFF GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_rff_params(K, P_half, seed=42):
    """Generate Random Fourier Feature parameters.
    
    Following Didisheim et al. (2023): gamma randomly drawn from [0.5..1.0].
    """
    rng = np.random.RandomState(seed)
    omega = rng.randn(P_half, K)
    gammas = rng.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], size=P_half)
    return omega, gammas


def compute_rff_features(std_panel, omega, gammas):
    """Compute sin/cos RFF features for entire panel.
    
    Returns: (T, N, 2*P_half)
    """
    # proj[t,n,p] = gamma_p * Z[t,n,:] @ omega[p,:]
    proj = np.einsum('tnk,pk->tnp', std_panel, omega)  # (T, N, P_half)
    proj *= gammas[np.newaxis, np.newaxis, :]
    
    P_half = omega.shape[0]
    scale = 1.0 / np.sqrt(P_half)
    sin_f = np.sin(proj) * scale
    cos_f = np.cos(proj) * scale
    return np.concatenate([sin_f, cos_f], axis=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  APPROACH A: SDF Factor Portfolio (Didisheim et al. 2023)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_factor_returns(features, ret_np, valid_mask):
    """Build characteristic-managed portfolio returns.
    
    F_{p,t} = (1/sqrt(N_{t-1})) * S_{:,t-1,p}' R_{:,t}
    Uses LAGGED features (t-1) with CURRENT returns (t).
    
    Returns: (T, 2P) factor returns (row 0 is zero/invalid)
    """
    T, N, P2 = features.shape
    F = np.zeros((T, P2))
    for t in range(1, T):
        mask = valid_mask[t-1] & valid_mask[t]
        n = mask.sum()
        if n < 5:
            continue
        S = features[t-1, mask, :]  # (n, P2)
        R = ret_np[t, mask]         # (n,)
        F[t] = S.T @ R / np.sqrt(n)
    return F


def sdf_markowitz_solve(F_window, z):
    """Ridge Markowitz: λ = (zI + E[FF'])^{-1} E[F].
    
    Uses dual form when P > T for efficiency.
    """
    T, P = F_window.shape
    mu = F_window.mean(axis=0)
    
    if P <= T:
        M2 = F_window.T @ F_window / T
        lam = np.linalg.solve(z * np.eye(P) + M2, mu)
    else:
        # Dual: invert T×T instead of P×P
        G = F_window @ F_window.T
        alpha = np.linalg.solve(z * T * np.eye(T) + G, np.ones(T) / T)
        lam = F_window.T @ alpha
    return lam


def run_sdf_single_seed(std_panel, ret_np, valid_mask, P_half, seed,
                         z=1e-5, window=720, retrain_every=6):
    """SDF factor portfolio for one random seed. Returns (T, N) alpha."""
    T, N, K = std_panel.shape
    P2 = 2 * P_half
    c = P2 / window
    
    omega, gammas = generate_rff_params(K, P_half, seed=seed)
    features = compute_rff_features(std_panel, omega, gammas)
    F = compute_factor_returns(features, ret_np, valid_mask)
    
    alpha = np.zeros((T, N))
    lam = None
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
            
            lam = sdf_markowitz_solve(F_clean, z)
            last_refit = t
            n_refits += 1
        
        if lam is not None:
            # SDF weights for each asset
            alpha[t] = features[t] @ lam
            # Cross-sectional demean
            vm = valid_mask[t]
            if vm.sum() > 0:
                alpha[t, vm] -= np.mean(alpha[t, vm])
    
    del features, F
    gc.collect()
    return alpha, n_refits


# ═══════════════════════════════════════════════════════════════════════════════
#  APPROACH B: Fama-MacBeth Cross-Sectional Regression
# ═══════════════════════════════════════════════════════════════════════════════

def run_fm_single_seed(std_panel, ret_np, valid_mask, P_half, seed,
                        z=0.001, fm_window=1, retrain_every=1):
    """Fama-MacBeth cross-sectional ridge for one seed.
    
    fm_window: number of past cross-sections to pool per regression.
      1 = pure FM (N observations per fit, c = 2P/N)
      6 = pooled (6*N observations, c = 2P/(6*N))
    
    Returns: (T, N) alpha signal
    """
    T, N, K = std_panel.shape
    P2 = 2 * P_half
    c = P2 / (N * fm_window)
    
    omega, gammas = generate_rff_params(K, P_half, seed=seed)
    features = compute_rff_features(std_panel, omega, gammas)
    
    alpha = np.zeros((T, N))
    beta = None
    last_refit = -retrain_every
    n_refits = 0
    
    for t in range(WARMUP, T):
        if t - last_refit >= retrain_every or beta is None:
            # Pool fm_window past cross-sections
            S_list, R_list = [], []
            for w in range(fm_window):
                tw = t - w
                if tw < 1:
                    continue
                mask = valid_mask[tw-1] & valid_mask[tw]
                n = mask.sum()
                if n < 5:
                    continue
                S_list.append(features[tw-1, mask, :])  # lagged features
                R_list.append(ret_np[tw, mask])           # current returns
            
            if not S_list:
                continue
            
            S = np.vstack(S_list)  # (total_obs, P2)
            R = np.concatenate(R_list)  # (total_obs,)
            n_obs = len(R)
            
            # Ridge regression with dual trick when P2 > n_obs
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


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_signal(alpha_np, ret_np, valid_mask, dates, fee_bps_list,
                     splits=SPLITS, label="signal"):
    """Evaluate an alpha signal: IC, SR, TO at multiple fee levels and splits."""
    T, N = alpha_np.shape
    
    # Split indices
    split_masks = {}
    for name, (start, end) in splits.items():
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        mask = (dates >= s) & (dates < e)
        split_masks[name] = mask.values if hasattr(mask, 'values') else mask
    
    results = {}
    
    for split_name, t_mask in split_masks.items():
        t_indices = np.where(t_mask)[0]
        if len(t_indices) < 10:
            continue
        
        # IC: rank correlation per bar, averaged
        ics = []
        for t in t_indices:
            vm = valid_mask[t] if t < T else np.zeros(N, dtype=bool)
            if t + 1 >= T:
                continue
            vm_next = valid_mask[t+1] if t+1 < T else np.zeros(N, dtype=bool)
            vm_both = vm & vm_next
            if vm_both.sum() < 5:
                continue
            sig = alpha_np[t, vm_both]
            fwd = ret_np[t+1, vm_both]
            if np.std(sig) < 1e-15 or np.std(fwd) < 1e-15:
                continue
            from scipy.stats import spearmanr
            ic, _ = spearmanr(sig, fwd)
            if np.isfinite(ic):
                ics.append(ic)
        
        mean_ic = np.mean(ics) if ics else 0.0
        
        # Portfolio returns: normalize signal cross-sectionally to weights
        port_rets = []
        turnovers = []
        prev_weights = None
        
        for t in t_indices:
            if t + 1 >= T:
                continue
            vm = valid_mask[t]
            if vm.sum() < 5:
                continue
            
            sig = alpha_np[t].copy()
            sig[~vm] = 0.0
            
            # Normalize to unit exposure
            abs_sum = np.abs(sig).sum()
            if abs_sum < 1e-15:
                weights = np.zeros(N)
            else:
                weights = sig / abs_sum
            
            # Portfolio return
            r = np.sum(weights * ret_np[t + 1])
            port_rets.append(r)
            
            # Turnover
            if prev_weights is not None:
                to = np.sum(np.abs(weights - prev_weights))
                turnovers.append(to)
            prev_weights = weights.copy()
        
        if not port_rets:
            continue
        
        port_rets = np.array(port_rets)
        mean_to = np.mean(turnovers) if turnovers else 0.0
        
        split_results = {"IC": mean_ic, "TO": mean_to}
        
        # Sharpe at each fee level
        for fee in fee_bps_list:
            fee_frac = fee / 10000.0
            if turnovers:
                fee_costs = np.array([0.0] + [to * fee_frac for to in turnovers])
            else:
                fee_costs = np.zeros(len(port_rets))
            net_rets = port_rets - fee_costs[:len(port_rets)]
            
            mu = np.mean(net_rets) * 6 * 365  # annualize 4h bars
            sigma = np.std(net_rets) * np.sqrt(6 * 365)
            sr = mu / sigma if sigma > 1e-12 else 0.0
            cum_ret = np.sum(net_rets) * 100
            
            split_results[f"SR_{int(fee)}bps"] = sr
            split_results[f"Ret_{int(fee)}bps"] = cum_ret
        
        results[split_name] = split_results
    
    return results


def print_results(results, label, P_half, z, window, c, method, n_seeds):
    """Pretty-print evaluation results."""
    P2 = 2 * P_half
    log(f"\n  --- {label} ---")
    log(f"  Method={method}, P={P_half} (2P={P2}), z={z:.1e}, window={window}, c={c:.2f}, seeds={n_seeds}")
    
    for split in ["train", "val", "test"]:
        if split not in results:
            continue
        r = results[split]
        ic = r.get("IC", 0)
        to = r.get("TO", 0)
        parts = [f"  {split:5s}: IC={ic:+.4f}, TO={to:.4f}"]
        for fee in [0, 2, 5, 7]:
            sr = r.get(f"SR_{fee}bps", 0)
            ret = r.get(f"Ret_{fee}bps", 0)
            parts.append(f"{fee}bps SR={sr:+.3f} ({ret:+.1f}%)")
        log(", ".join(parts))


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AIPT: VoC for Cross-Sectional Crypto")
    parser.add_argument("--method", type=str, default="both",
                        choices=["sdf", "fm", "both"],
                        help="Which approach to run")
    parser.add_argument("--P", type=int, nargs="+", default=[100, 250, 500],
                        help="P_half values (total features = 2P)")
    parser.add_argument("--z", type=float, nargs="+", default=[1e-5, 1e-3, 0.1],
                        help="Ridge z values")
    parser.add_argument("--window", type=int, nargs="+", default=[360, 720],
                        help="Training window for SDF (bars)")
    parser.add_argument("--fm-window", type=int, nargs="+", default=[1, 6],
                        help="FM pooling window (1=pure FM, 6=pooled)")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of random seeds to average")
    parser.add_argument("--retrain", type=int, default=6,
                        help="Retrain every N bars")
    parser.add_argument("--no-alphas", action="store_true",
                        help="Don't augment with DB alphas")
    args = parser.parse_args()
    
    t0_total = time.time()
    
    # Load data
    matrices, universe_df, valid_tickers = load_full_data()
    panel, ret_np, valid_mask, dates, tickers, char_keys = prepare_panel(matrices, valid_tickers)
    
    T, N, K = panel.shape
    log(f"\n  Panel: T={T}, N={N}, K={K}")
    
    # Augment with alphas
    if not args.no_alphas:
        log("  Loading DB alphas...")
        panel, char_keys = augment_with_alphas(panel, matrices, valid_tickers, char_keys)
        K = panel.shape[2]
    
    # Cross-sectional standardize
    log("  Cross-sectional standardizing...")
    t0 = time.time()
    std_panel = cross_sectional_standardize(panel)
    std_panel = np.nan_to_num(std_panel, nan=0.0)
    log(f"  Standardized in {time.time()-t0:.1f}s")
    
    log(f"\n{'='*100}")
    log(f"  AIPT: Complex Factor Pricing for 4H Crypto")
    log(f"  Panel: T={T}, N={N}, K={K}")
    log(f"  Methods: {args.method}")
    log(f"  P values: {args.P}")
    log(f"  z values: {args.z}")
    log(f"  SDF windows: {args.window}")
    log(f"  FM windows: {args.fm_window}")
    log(f"  Seeds: {args.seeds}")
    log(f"  Retrain: every {args.retrain} bars")
    log(f"  Splits: train {SPLITS['train']}, val {SPLITS['val']}, test {SPLITS['test']}")
    log(f"{'='*100}")
    
    all_results = []
    
    # ─── SDF Factor Portfolio ───────────────────────
    if args.method in ["sdf", "both"]:
        log(f"\n{'-'*50}")
        log(f"  APPROACH A: SDF Factor Portfolio")
        log(f"{'-'*50}")
        
        for P_half in args.P:
            for window in args.window:
                for z in args.z:
                    P2 = 2 * P_half
                    c = P2 / window
                    label = f"SDF P={P_half} w={window} z={z:.0e}"
                    log(f"\n  === {label} (c={c:.2f}) ===")
                    
                    t0 = time.time()
                    alpha_sum = np.zeros((T, N))
                    total_refits = 0
                    
                    for s in range(args.seeds):
                        seed = 42 + s * 137  # spread seeds
                        a, nr = run_sdf_single_seed(
                            std_panel, ret_np, valid_mask, P_half, seed,
                            z=z, window=window, retrain_every=args.retrain)
                        alpha_sum += a
                        total_refits += nr
                        if (s + 1) % 5 == 0 or s == 0:
                            log(f"    Seed {s+1}/{args.seeds} done ({time.time()-t0:.0f}s)")
                    
                    alpha_avg = alpha_sum / args.seeds
                    elapsed = time.time() - t0
                    log(f"  {args.seeds} seeds, {total_refits} total refits in {elapsed:.0f}s")
                    
                    results = evaluate_signal(alpha_avg, ret_np, valid_mask, dates, FEE_LEVELS)
                    print_results(results, label, P_half, z, window, c, "SDF", args.seeds)
                    
                    all_results.append({
                        "method": "SDF", "P": P_half, "2P": P2, "z": z,
                        "window": window, "c": c, "seeds": args.seeds,
                        **{f"{split}_{k}": v for split, sr in results.items() for k, v in sr.items()}
                    })
    
    # ─── Fama-MacBeth ─────────────────────────────────
    if args.method in ["fm", "both"]:
        log(f"\n{'-'*50}")
        log(f"  APPROACH B: Fama-MacBeth Cross-Sectional")
        log(f"{'-'*50}")
        
        for P_half in args.P:
            for fm_w in args.fm_window:
                for z in args.z:
                    P2 = 2 * P_half
                    c = P2 / (N * fm_w)
                    label = f"FM P={P_half} w={fm_w} z={z:.0e}"
                    log(f"\n  === {label} (c={c:.2f}) ===")
                    
                    t0 = time.time()
                    alpha_sum = np.zeros((T, N))
                    total_refits = 0
                    
                    for s in range(args.seeds):
                        seed = 42 + s * 137
                        a, nr = run_fm_single_seed(
                            std_panel, ret_np, valid_mask, P_half, seed,
                            z=z, fm_window=fm_w, retrain_every=args.retrain)
                        alpha_sum += a
                        total_refits += nr
                        if (s + 1) % 5 == 0 or s == 0:
                            log(f"    Seed {s+1}/{args.seeds} done ({time.time()-t0:.0f}s)")
                    
                    alpha_avg = alpha_sum / args.seeds
                    elapsed = time.time() - t0
                    log(f"  {args.seeds} seeds, {total_refits} total refits in {elapsed:.0f}s")
                    
                    results = evaluate_signal(alpha_avg, ret_np, valid_mask, dates, FEE_LEVELS)
                    print_results(results, label, P_half, z, fm_w, c, "FM", args.seeds)
                    
                    all_results.append({
                        "method": "FM", "P": P_half, "2P": P2, "z": z,
                        "window": fm_w, "c": c, "seeds": args.seeds,
                        **{f"{split}_{k}": v for split, sr in results.items() for k, v in sr.items()}
                    })
    
    # ─── Summary Report ────────────────────────────────
    total_time = time.time() - t0_total
    log(f"\n{'='*100}")
    log(f"  Total time: {total_time:.0f}s ({total_time/60:.1f}m)")
    log(f"{'='*100}")
    
    # Write results to markdown
    if all_results:
        write_report(all_results, args)
    
    log(f"\n  Report saved to: {REPORT_FILE}")


def write_report(all_results, args):
    """Write comprehensive results report."""
    lines = [
        "# AIPT: Complex Factor Pricing — Results",
        f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"\nSeeds: {args.seeds} | Retrain: every {args.retrain} bars",
        "",
        "## Summary Table",
        "",
        "| Method | P | 2P | z | Window | c=P/T | Train IC | Test IC | Train SR@0 | Test SR@0 | Train SR@5 | Test SR@5 | Test TO |",
        "|--------|---|-----|---|--------|-------|----------|---------|------------|-----------|------------|-----------|---------|",
    ]
    
    for r in all_results:
        tr_ic = r.get("train_IC", 0)
        te_ic = r.get("test_IC", 0)
        tr_sr0 = r.get("train_SR_0bps", 0)
        te_sr0 = r.get("test_SR_0bps", 0)
        tr_sr5 = r.get("train_SR_5bps", 0)
        te_sr5 = r.get("test_SR_5bps", 0)
        te_to = r.get("test_TO", 0)
        
        lines.append(
            f"| {r['method']} | {r['P']} | {r['2P']} | {r['z']:.0e} | {r['window']} "
            f"| {r['c']:.2f} | {tr_ic:+.4f} | {te_ic:+.4f} "
            f"| {tr_sr0:+.3f} | {te_sr0:+.3f} | {tr_sr5:+.3f} | {te_sr5:+.3f} | {te_to:.4f} |"
        )
    
    lines.extend([
        "",
        "## Detailed Results",
        "",
    ])
    
    for r in all_results:
        lines.append(f"### {r['method']} P={r['P']} z={r['z']:.0e} w={r['window']} (c={r['c']:.2f})")
        lines.append("")
        for split in ["train", "val", "test"]:
            ic = r.get(f"{split}_IC", 0)
            to = r.get(f"{split}_TO", 0)
            parts = [f"- **{split}**: IC={ic:+.4f}, TO={to:.4f}"]
            for fee in [0, 2, 5, 7]:
                sr = r.get(f"{split}_SR_{fee}bps", 0)
                ret = r.get(f"{split}_Ret_{fee}bps", 0)
                parts.append(f"  {fee}bps: SR={sr:+.3f} ({ret:+.1f}%)")
            lines.append(", ".join(parts))
        lines.append("")
    
    REPORT_FILE.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
