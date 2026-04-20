"""
run_voc_complexity.py -- Virtue of Complexity (Kelly, Malamud & Zhou 2020)
Random Fourier Feature (RFF) pipeline for 4H Crypto Perpetual Futures.

Methodology:
  1. Load all 40+ raw data matrices (4h bars, BINANCE_TOP50)
  2. Generate P random projection vectors gamma ~ N(0, sigma^2 I), sigma = 1/sqrt(K)
  3. For each bar t, each asset i:
     - Stack K characteristics into x_{i,t} (causally standardized)
     - Compute 2P RFF features: z = [sin(gamma @ x), cos(gamma @ x)]
  4. Walk-forward ridge regression: predict next-bar return from z
  5. Predicted return y_hat_{i,t} IS the alpha signal
  6. Evaluate via existing simulation engine across splits x fees

Evaluation modes:
  - Raw (no smoothing): pure ridge signal at 0bps to show signal quality
  - Smoothed (EMA halflife=12): production-ready with lower turnover
  - QP-Optimized: CVXPY convex optimizer on the raw VoC signal (no smoothing needed)

Metrics reported:
  - Sharpe, Return%, Turnover, Fitness, MaxDD (standard)
  - Information Coefficient (IC): mean cross-sectional rank correlation(signal, next-bar return)
  - R-squared: cross-sectional R^2 of signal vs next-bar return

Anti-Lookahead Safeguards:
  - Random projections drawn ONCE at init (not fit to data)
  - Cross-sectional standardization uses expanding stats only
  - Ridge coefs fit on trailing window [t-W, t-1] only
  - Ridge penalty lambda = P*10 (fixed, not tuned on val/test)

Usage:
    python run_voc_complexity.py                     # Default P=50
    python run_voc_complexity.py --P 10              # Smoke test
    python run_voc_complexity.py --P 50 100 250 500  # Scaling sweep
    python run_voc_complexity.py --sweep             # Full VoC scaling curve
"""

import sys, os, time, sqlite3, argparse
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# CONFIGURATION
# ============================================================================

UNIVERSE     = "BINANCE_TOP50"
INTERVAL     = "4h"
BOOKSIZE     = 2_000_000.0
BARS_PER_DAY = 6
MAX_WEIGHT   = 0.10
NEUTRALIZE   = "market"
COVERAGE_CUTOFF = 0.3
DB_PATH      = "data/alphas.db"

SPLITS = {
    "train": ("2021-01-01", "2025-01-01"),   # 4 years
    "val":   ("2025-01-01", "2025-09-01"),   # 8 months
    "test":  ("2025-09-01", "2026-03-05"),   # ~6 months
}

FULL_START = "2020-10-01"   # warmup start
FULL_END   = "2026-03-05"

FEE_LEVELS = [0.0, 2.0, 5.0, 7.0]

# --- VoC Hyperparameters (FIXED -- not tuned on val/test) ---
RFF_SEED         = 42          # Random seed for projection matrix
RIDGE_WINDOW     = 720         # Trailing window for ridge fit (120 days)
RIDGE_RETRAIN    = 30          # Retrain every 30 bars (5 days)
RIDGE_WARMUP     = 180         # Minimum bars before first fit (30 days)
FORWARD_HORIZON  = 1           # Predict h-bar-ahead cumulative return (1=next bar)

# Characteristics to EXCLUDE from feature matrix
EXCLUDE_FIELDS = {"returns", "log_returns"}

# --- Reporting Buffer ---
REPORT_BUFFER = []
def log(msg="", flush=True):
    print(msg, flush=flush)
    REPORT_BUFFER.append(msg)


# ============================================================================
# DATA LOADING
# ============================================================================

_FULL_DATA = None

def load_full_data():
    global _FULL_DATA
    if _FULL_DATA is not None:
        return _FULL_DATA

    mat_dir = Path(f"data/binance_cache/matrices/{INTERVAL}")
    uni_path = Path(f"data/binance_cache/universes/{UNIVERSE}_{INTERVAL}.parquet")

    universe_df = pd.read_parquet(uni_path)
    coverage = universe_df.sum(axis=0) / len(universe_df)
    valid_tickers = sorted(coverage[coverage > COVERAGE_CUTOFF].index.tolist())

    print(f"  Loading 4h matrices ({len(valid_tickers)} tickers)...", flush=True)
    t0 = time.time()
    matrices = {}
    for fp in sorted(mat_dir.glob("*.parquet")):
        df = pd.read_parquet(fp)
        cols = [c for c in valid_tickers if c in df.columns]
        if cols:
            matrices[fp.stem] = df[cols].loc[FULL_START:FULL_END]

    # Compute returns from close
    if "close" in matrices:
        matrices["returns"] = matrices["close"].pct_change()

    universe_slice = universe_df[valid_tickers].loc[FULL_START:FULL_END]
    print(f"  Loaded {len(matrices)} fields in {time.time()-t0:.1f}s", flush=True)

    _FULL_DATA = (matrices, universe_slice, valid_tickers)
    return _FULL_DATA


def augment_matrices_with_db_alphas(matrices, universe_df):
    """
    Load all alpha expressions from DB, evaluate them, and add them as
    additional characteristic fields in the matrices dict.
    
    Each alpha becomes a field named 'alpha_XXX' (where XXX is the alpha ID).
    This lets the VoC pipeline treat pre-engineered alpha signals as
    additional characteristics to project through RFF.
    
    Returns: augmented_matrices (copy of matrices with alpha fields added),
             n_alphas_added (int)
    """
    from src.operators.fastexpression import FastExpressionEngine
    
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT a.id, a.expression
        FROM alphas a WHERE a.archived = 0
          AND a.interval = ? AND a.universe = ?
        ORDER BY a.id
    """, (INTERVAL, UNIVERSE)).fetchall()
    conn.close()
    
    if not rows:
        print("  No alphas in DB to augment with")
        return matrices, 0
    
    close_df = matrices["close"]
    aug = dict(matrices)  # shallow copy
    n_added = 0
    
    print(f"  Evaluating {len(rows)} DB alphas as additional characteristics...", flush=True)
    t0 = time.time()
    
    for aid, expr in rows:
        try:
            engine = FastExpressionEngine(data_fields=matrices)
            raw = engine.evaluate(expr)
            if raw is None or raw.empty:
                continue
            # Reindex to match close_df shape
            aligned = raw.reindex(index=close_df.index, columns=close_df.columns)
            if aligned.shape == close_df.shape:
                aug[f"alpha_{aid}"] = aligned
                n_added += 1
        except Exception:
            continue
    
    print(f"  Added {n_added}/{len(rows)} alpha signals as characteristics ({time.time()-t0:.1f}s)",
          flush=True)
    return aug, n_added


def simulate(alpha_df, returns_df, close_df, universe_df, fees_bps=0.0):
    from src.simulation.vectorized_sim_polars import simulate_vectorized_polars
    return simulate_vectorized_polars(
        alpha_df=alpha_df, returns_df=returns_df, close_df=close_df,
        universe_df=universe_df, booksize=BOOKSIZE,
        max_stock_weight=MAX_WEIGHT, decay=0, delay=0,
        neutralization=NEUTRALIZE, fees_bps=fees_bps,
        bars_per_day=BARS_PER_DAY,
    )


def process_signal(alpha_df, universe_df=None, max_wt=MAX_WEIGHT):
    """Standard signal processing: demean -> scale -> clip."""
    signal = alpha_df.copy()
    if universe_df is not None:
        uni_mask = universe_df.reindex(index=signal.index, columns=signal.columns).fillna(False)
        signal = signal.where(uni_mask, np.nan)
    mean_val = signal.mean(axis=1)
    signal = signal.sub(mean_val, axis=0)
    abs_sum = signal.abs().sum(axis=1).replace(0, np.nan)
    signal = signal.div(abs_sum, axis=0)
    signal = signal.clip(lower=-max_wt, upper=max_wt)
    return signal.fillna(0.0)


# ============================================================================
# IC AND R-SQUARED METRICS
# ============================================================================

def compute_ic_r2(signal_df, returns_df, universe_df, split_start, split_end):
    """
    Compute Information Coefficient (IC) and R-squared for a signal.
    
    IC: mean cross-sectional Spearman rank correlation between
        signal at time t and return at time t+1 (causal).
    R2: mean cross-sectional R-squared of signal vs next-bar return.
    """
    sig = signal_df.loc[split_start:split_end]
    ret = returns_df.loc[split_start:split_end]
    uni = universe_df.loc[split_start:split_end]
    
    # Align
    common_idx = sig.index.intersection(ret.index)
    common_cols = sig.columns.intersection(ret.columns)
    sig = sig.reindex(index=common_idx, columns=common_cols)
    ret = ret.reindex(index=common_idx, columns=common_cols)
    uni = uni.reindex(index=common_idx, columns=common_cols).fillna(False)
    
    ic_list = []
    r2_list = []
    
    # Signal at t predicts return at t+1 (so we lag the signal)
    sig_vals = sig.values[:-1]  # t = 0..T-2
    ret_vals = ret.values[1:]   # t = 1..T-1
    uni_vals = uni.values[:-1]  # universe at signal time
    
    for t in range(len(sig_vals)):
        mask = uni_vals[t] & np.isfinite(sig_vals[t]) & np.isfinite(ret_vals[t])
        if mask.sum() < 10:
            continue
        
        s = sig_vals[t, mask]
        r = ret_vals[t, mask]
        
        # IC: Spearman rank correlation
        try:
            corr, _ = scipy_stats.spearmanr(s, r)
            if np.isfinite(corr):
                ic_list.append(corr)
        except Exception:
            pass
        
        # R2: cross-sectional R-squared
        try:
            ss_res = np.sum((r - s * (np.dot(s, r) / max(np.dot(s, s), 1e-15)))**2)
            ss_tot = np.sum((r - np.mean(r))**2)
            if ss_tot > 1e-15:
                r2 = 1.0 - ss_res / ss_tot
                r2_list.append(r2)
        except Exception:
            pass
    
    ic_mean = np.mean(ic_list) if ic_list else 0.0
    ic_std = np.std(ic_list) if len(ic_list) > 1 else 0.0
    ic_ir = ic_mean / ic_std if ic_std > 1e-10 else 0.0
    r2_mean = np.mean(r2_list) if r2_list else 0.0
    
    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": ic_ir,       # IC / IC_std (information ratio of IC)
        "r2_mean": r2_mean,
        "n_bars": len(ic_list),
    }


# ============================================================================
# RANDOM FOURIER FEATURES -- CORE ENGINE
# ============================================================================

def generate_rff_projections(K, P, seed=RFF_SEED):
    """
    Generate P random projection vectors for K characteristics.
    gamma ~ N(0, sigma^2 I), sigma = 1/sqrt(K) (Gaussian kernel bandwidth).
    Drawn ONCE and remain fixed -- no data leakage.
    """
    rng = np.random.default_rng(seed)
    sigma = 1.0 / np.sqrt(K)
    gamma = rng.normal(0, sigma, size=(P, K))
    return gamma


def build_characteristic_panel(matrices, universe_df, field_names, dates, tickers):
    """Build 3D panel of raw characteristics: (T, N, K)."""
    T, N, K = len(dates), len(tickers), len(field_names)
    panel = np.full((T, N, K), np.nan, dtype=np.float64)
    for k, fname in enumerate(field_names):
        if fname not in matrices:
            continue
        df = matrices[fname].reindex(index=dates, columns=tickers)
        panel[:, :, k] = df.values
    # Apply universe mask
    uni_np = universe_df.reindex(index=dates, columns=tickers).fillna(False).values.astype(bool)
    for k in range(K):
        panel[:, :, k] = np.where(uni_np, panel[:, :, k], np.nan)
    return panel


def causal_standardize_panel(panel):
    """
    Cross-sectional z-score at each bar -- NO lookahead.
    Vectorized via numpy broadcasting.
    """
    T, N, K = panel.shape
    with np.errstate(all='ignore'):
        cs_mean = np.nanmean(panel, axis=1, keepdims=True)
        cs_std = np.nanstd(panel, axis=1, keepdims=True)
    cs_std = np.where(cs_std < 1e-10, np.nan, cs_std)
    out = (panel - cs_mean) / cs_std
    valid_counts = np.sum(np.isfinite(panel), axis=1, keepdims=True)
    out = np.where(valid_counts >= 5, out, np.nan)
    out = np.clip(out, -5.0, 5.0)
    out = np.nan_to_num(out, nan=0.0)
    return out


def compute_rff_features(X_std, gamma):
    """Compute RFF features: (T, N, 2P) = [sin(gamma'x), cos(gamma'x)]."""
    P = gamma.shape[0]
    projections = np.einsum('tnk,pk->tnp', X_std, gamma)
    sin_f = np.sin(projections)
    cos_f = np.cos(projections)
    rff = np.concatenate([sin_f, cos_f], axis=2)
    rff *= 1.0 / np.sqrt(P)
    return rff


def fit_ridge_closed_form(Z, y, alpha):
    """Fit ridge: beta = (Z'Z + alpha*I)^{-1} Z'y."""
    d = Z.shape[1]
    ZtZ = Z.T @ Z
    Zty = Z.T @ y
    ZtZ += alpha * np.eye(d)
    try:
        beta = np.linalg.solve(ZtZ, Zty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(ZtZ, Zty, rcond=None)[0]
    return beta


# ============================================================================
# QP EXECUTION OPTIMIZER (from run_4h_portfolio.py)
# ============================================================================

def qp_optimize_signal(composite_df, matrices, universe_df, returns_df,
                       risk_aversion=0.5, tcost_lambda=0.0005,
                       lookback_bars=120, rebal_every=6):
    """
    Convex optimization on aggregate signal.
    Maximize: alpha'w - 0.5*kappa*w'Sigma*w - lambda*||w - w_prev||_1
    Subject to: sum(w) = 0, sum(|w|) <= 2.0
    """
    try:
        import cvxpy as cp
    except ImportError:
        print("  QP: cvxpy not installed, returning raw signal", flush=True)
        return composite_df

    close_df = matrices["close"]
    dates = close_df.index
    tickers = close_df.columns.tolist()
    n_tickers = len(tickers)
    n_bars = len(dates)

    ret_df = returns_df.reindex(index=dates, columns=tickers).fillna(0.0).values
    signal_vals = composite_df.reindex(index=dates, columns=tickers).fillna(0.0).values

    opt_weights = np.zeros((n_bars, n_tickers))
    prev_w = np.zeros(n_tickers)
    rebal_count = 0

    for t in range(lookback_bars, n_bars):
        if t % rebal_every != 0:
            opt_weights[t] = prev_w
            continue

        ret_window = ret_df[max(0, t - lookback_bars):t]
        if ret_window.shape[0] < 30:
            opt_weights[t] = prev_w
            continue

        alpha_t = np.nan_to_num(signal_vals[t], nan=0.0)
        cov = np.cov(ret_window.T)
        if cov.shape[0] != n_tickers:
            opt_weights[t] = prev_w
            continue

        cov = 0.5 * cov + 0.5 * np.diag(np.diag(cov)) + 1e-8 * np.eye(n_tickers)

        try:
            w = cp.Variable(n_tickers)
            obj = alpha_t @ w - risk_aversion * cp.quad_form(w, cp.psd_wrap(cov))
            if np.any(prev_w != 0):
                obj -= tcost_lambda * cp.norm1(w - prev_w)
            constraints = [
                cp.sum(w) == 0,
                cp.norm1(w) <= 2.0,
                w >= -MAX_WEIGHT,
                w <= MAX_WEIGHT,
            ]
            prob = cp.Problem(cp.Maximize(obj), constraints)
            prob.solve(solver=cp.SCS, verbose=False, max_iters=2000)
            if prob.status in ["optimal", "optimal_inaccurate"]:
                opt_w = np.nan_to_num(w.value, nan=0.0)
                prev_w = opt_w
                opt_weights[t] = opt_w
                rebal_count += 1
            else:
                opt_weights[t] = prev_w
        except Exception:
            opt_weights[t] = prev_w

    opt_weights[:lookback_bars] = 0.0
    print(f"  QP optimizer: {rebal_count} rebalances", flush=True)
    return pd.DataFrame(opt_weights, index=dates, columns=tickers)


# ============================================================================
# VoC WALK-FORWARD PIPELINE
# ============================================================================

def run_voc_pipeline(P, matrices, universe_df, valid_tickers, verbose=True,
                      signal_halflife=0, forward_horizon=1, ridge_window=RIDGE_WINDOW,
                      retrain_every=RIDGE_RETRAIN, expanding=False,
                      ridge_lambda=None):
    """
    Main VoC pipeline: generates alpha signal from RFF + ridge regression.
    
    Parameters:
        P: number of random projections
        signal_halflife: EMA halflife for output signal smoothing (bars).
            0 = no smoothing (raw ridge signal).
        forward_horizon: number of bars ahead to predict (1=next bar, 6=next day).
            Multi-bar returns are computed as rolling sum of 1-bar returns.
        ridge_window: trailing window size for ridge fitting (ignored if expanding=True).
        retrain_every: refit ridge every N bars.
        expanding: if True, use expanding window (all past data) instead of rolling.
        ridge_lambda: explicit ridge penalty. If None, uses P*10 (legacy).
    
    Returns: alpha_df (DataFrame, same shape as close) -- predicted forward returns
    """
    close_df = matrices["close"]
    returns_df = matrices["returns"]
    dates = close_df.index
    tickers = close_df.columns.tolist()
    T, N = len(dates), len(tickers)
    
    field_names = sorted([
        name for name in matrices.keys()
        if name not in EXCLUDE_FIELDS
        and matrices[name].shape == close_df.shape
    ])
    K = len(field_names)
    
    if verbose:
        win_desc = "expanding" if expanding else f"rolling({ridge_window})"
        print(f"\n  VoC Pipeline: P={P}, K={K}, T={T}, N={N}, horizon={forward_horizon}, window={win_desc}, retrain={retrain_every}")
        print(f"  Fields: {field_names[:8]}... ({K} total)")
    
    # Step 1: Random projection matrix (FIXED)
    gamma = generate_rff_projections(K, P, seed=RFF_SEED)
    if verbose:
        print(f"  RFF: gamma ({P}, {K}), sigma={1.0/np.sqrt(K):.4f}")
    
    # Step 2: Build raw panel (T, N, K)
    t0 = time.time()
    panel = build_characteristic_panel(matrices, universe_df, field_names, dates, tickers)
    if verbose:
        print(f"  Panel: {panel.shape}, {time.time()-t0:.1f}s")
    
    # Step 3: Causal standardize
    t0 = time.time()
    X_std = causal_standardize_panel(panel)
    if verbose:
        print(f"  Standardized: {time.time()-t0:.1f}s")
    
    # Step 4: RFF features (T, N, 2P)
    t0 = time.time()
    Z = compute_rff_features(X_std, gamma)
    if verbose:
        print(f"  RFF features: {Z.shape}, {time.time()-t0:.1f}s")
    
    # Step 5: Walk-forward ridge (VECTORIZED)
    t0 = time.time()
    alpha_signal = np.zeros((T, N), dtype=np.float64)
    uni_np = universe_df.reindex(index=dates, columns=tickers).fillna(False).values.astype(bool)
    ret_np = returns_df.reindex(index=dates, columns=tickers).fillna(0.0).values
    
    # Build forward return target: h-bar cumulative return
    h = forward_horizon
    if h > 1:
        # Rolling sum of 1-bar returns over h bars
        fwd_ret = np.zeros_like(ret_np)
        for lag in range(1, h + 1):
            shifted = np.zeros_like(ret_np)
            shifted[lag:] = ret_np[:-lag] if lag < T else 0
            # Actually we want forward returns: ret[t+1] + ret[t+2] + ... + ret[t+h]
        # Correct approach: fwd_ret[t] = sum(ret[t+1:t+h+1])
        fwd_ret = np.zeros_like(ret_np)
        for t_idx in range(T - h):
            fwd_ret[t_idx] = np.sum(ret_np[t_idx+1:t_idx+h+1], axis=0)
        # Last h bars have no valid forward return
        fwd_ret[T-h:] = np.nan
        if verbose:
            print(f"  Forward returns: {h}-bar cumulative (sum of next {h} returns)")
    else:
        fwd_ret = ret_np  # 1-bar: ret[t+1] is at index t+1
    
    ridge_alpha = ridge_lambda if ridge_lambda is not None else float(P) * 10.0
    current_beta = None
    last_refit = -retrain_every
    n_refits = 0
    
    for t in range(RIDGE_WARMUP, T):
        if t - last_refit >= retrain_every or current_beta is None:
            if expanding:
                w_start = 0  # Use ALL available past data
            else:
                w_start = max(0, t - ridge_window)
            w_end = t
            n_window = w_end - w_start - 1
            if n_window < 10:
                if current_beta is not None:
                    alpha_signal[t, :] = Z[t] @ current_beta
                last_refit = t
                continue
            
            if h > 1:
                # Features at [w_start, w_end-h-1], target = fwd_ret at same indices
                Z_block = Z[w_start:w_end-h]
                ret_block = fwd_ret[w_start:w_end-h]
                uni_block = uni_np[w_start:w_end-h]
                valid_mask = uni_block & np.isfinite(ret_block)
            else:
                Z_block = Z[w_start:w_end-1]
                ret_block = ret_np[w_start+1:w_end]
                uni_feat = uni_np[w_start:w_end-1]
                uni_ret = uni_np[w_start+1:w_end]
                valid_mask = uni_feat & uni_ret & np.isfinite(ret_block)
            
            Z_flat = Z_block[valid_mask]
            y_flat = ret_block[valid_mask]
            
            if len(y_flat) < 100:
                if current_beta is not None:
                    alpha_signal[t, :] = Z[t] @ current_beta
                last_refit = t
                continue
            
            clip_val = 0.05 * h  # Scale clip with horizon
            y_flat = np.clip(y_flat, -clip_val, clip_val)
            current_beta = fit_ridge_closed_form(Z_flat, y_flat, ridge_alpha)
            n_refits += 1
            last_refit = t
        
        if current_beta is not None:
            alpha_signal[t, :] = Z[t] @ current_beta
    
    if verbose:
        print(f"  Ridge: {n_refits} refits, lambda={ridge_alpha:.0f}, {time.time()-t0:.1f}s")
        train_mask = (dates >= SPLITS["train"][0]) & (dates < SPLITS["train"][1])
        train_sig = alpha_signal[train_mask]
        nonzero = train_sig[train_sig != 0]
        if len(nonzero) > 0:
            print(f"  Signal stats (train): mean={np.mean(nonzero):.6f}, "
                  f"std={np.std(nonzero):.6f}, "
                  f"min={np.min(nonzero):.6f}, max={np.max(nonzero):.6f}")
    
    alpha_df = pd.DataFrame(alpha_signal, index=dates, columns=tickers)
    
    # Optional EMA smoothing
    if signal_halflife > 0:
        alpha_df = alpha_df.ewm(halflife=signal_halflife, min_periods=1).mean()
        if verbose:
            print(f"  Signal smoothing: EMA halflife={signal_halflife} bars")
    
    return alpha_df


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_signal(alpha_df, matrices, universe_df, name="VoC", is_qp=False):
    """Evaluate a signal across all splits x fee levels. Returns dict + IC/R2 metrics."""
    close = matrices.get("close")
    returns = matrices.get("returns")
    
    results = {}
    ic_r2 = {}
    
    for split_name, (start, end) in SPLITS.items():
        sig_slice = alpha_df.loc[start:end]
        split_close = close.loc[start:end]
        split_returns = returns.loc[start:end]
        split_universe = universe_df.loc[start:end]
        
        # IC/R2 on raw signal (before processing)
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
                print(f"    ERROR: {name}/{split_name}/{fee}bps: {e}")
    
    return results, ic_r2


def print_results_table(all_method_results, all_ic_r2, method_names):
    """Print comparison table with IC and R2."""
    # IC/R2 table
    log(f"\n  ---- SIGNAL QUALITY (IC / R2) ----")
    header = (f"  {'Method':<35s} | {'Train IC':>9} {'Train R2':>9} | "
              f"{'Val IC':>8} {'Val R2':>8} | {'Test IC':>8} {'Test R2':>8}")
    log(header)
    log(f"  {'-'*len(header)}")
    
    for mname in method_names:
        ic_data = all_ic_r2.get(mname, {})
        parts = []
        for split in ["train", "val", "test"]:
            d = ic_data.get(split, {})
            ic = d.get("ic_mean", 0)
            r2 = d.get("r2_mean", 0)
            parts.extend([f"{ic:+8.4f}", f"{r2:+8.5f}"])
        log(f"  {mname:<35s} | {parts[0]} {parts[1]} | {parts[2]} {parts[3]} | {parts[4]} {parts[5]}")
    
    # Performance tables
    for fee in FEE_LEVELS:
        log(f"\n  ---- {fee:.0f} BPS FEES ----")
        header = (f"  {'Method':<35s} | {'Train SR':>9} {'Train Ret%':>10} {'Train TO':>9} | "
                  f"{'Val SR':>8} {'Val Ret%':>9} {'Val TO':>8} | "
                  f"{'Test SR':>8} {'Test Ret%':>9} {'Test TO':>8}")
        log(header)
        log(f"  {'-'*len(header)}")
        
        for mname in method_names:
            results = all_method_results.get(mname, {})
            
            def fmt(r):
                if r is None:
                    return "  N/A", "  N/A", "  N/A"
                ret_pct = r.total_pnl / BOOKSIZE * 100
                return f"{r.sharpe:+8.3f}", f"{ret_pct:+9.2f}%", f"{r.turnover:8.4f}"
            
            tr_sr, tr_ret, tr_to = fmt(results.get(("train", fee)))
            v_sr, v_ret, v_to = fmt(results.get(("val", fee)))
            te_sr, te_ret, te_to = fmt(results.get(("test", fee)))
            
            log(f"  {mname:<35s} | {tr_sr} {tr_ret} {tr_to} | "
                f"{v_sr} {v_ret} {v_to} | {te_sr} {te_ret} {te_to}")


# ============================================================================
# BASELINE COMPARATORS
# ============================================================================

def build_equal_weight_combiner(matrices, universe_df):
    """Equal-weight combiner: average all alpha signals from DB."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT a.id, a.expression
        FROM alphas a WHERE a.archived = 0
          AND a.interval = ? AND a.universe = ?
        ORDER BY a.id
    """, (INTERVAL, UNIVERSE)).fetchall()
    conn.close()
    
    if not rows:
        print("  No alphas in DB for baseline comparison")
        return None
    
    from src.operators.fastexpression import FastExpressionEngine
    
    combined = None
    n = 0
    for aid, expr in rows:
        try:
            engine = FastExpressionEngine(data_fields=matrices)
            raw = engine.evaluate(expr)
            if raw is not None and not raw.empty:
                normed = process_signal(raw, universe_df=universe_df, max_wt=MAX_WEIGHT)
                combined = normed if combined is None else combined.add(normed, fill_value=0)
                n += 1
        except Exception:
            continue
    
    if n > 0 and combined is not None:
        combined = combined / n
        print(f"  Equal Weight: {n}/{len(rows)} alphas loaded")
    return combined


def build_billion_alphas_combiner(matrices, universe_df):
    """Billion Alphas combiner from run_4h_portfolio.py."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT a.id, a.expression, COALESCE(e.ic_mean, 0), COALESCE(e.sharpe_is, 0)
        FROM alphas a LEFT JOIN evaluations e ON e.alpha_id = a.id
        WHERE a.archived = 0
          AND a.interval = ? AND a.universe = ?
        ORDER BY COALESCE(e.sharpe_is, 0) DESC
    """, (INTERVAL, UNIVERSE)).fetchall()
    conn.close()
    
    if len(rows) < 3:
        print("  Insufficient alphas for Billion Alphas combiner")
        return None
    
    try:
        from run_4h_portfolio import combiner_billion_alphas, load_alpha_signals
        alpha_signals = load_alpha_signals(rows, matrices)
        returns_df = matrices.get("returns")
        composite = combiner_billion_alphas(alpha_signals, matrices, universe_df, returns_df)
        return composite
    except Exception as e:
        print(f"  Billion Alphas import failed: {e}")
        return None


# ============================================================================
# CHART GENERATION
# ============================================================================

def plot_voc_scaling_curve(scaling_results, scaling_ic, output_path):
    """Plot Sharpe + IC vs P."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    P_values = sorted(scaling_results.keys())
    
    # Row 1: Sharpe
    for ax_idx, split_name in enumerate(["train", "val", "test"]):
        ax = axes[0][ax_idx]
        for fee in [0.0, 5.0, 7.0]:
            sharpes = [scaling_results[p].get((split_name, fee), None) for p in P_values]
            sharpes = [r.sharpe if r else 0 for r in sharpes]
            ax.plot(P_values, sharpes, 'o-', linewidth=2, markersize=6, label=f"{fee:.0f} bps")
        ax.set_xlabel("P", fontsize=11)
        ax.set_ylabel("Annualized Sharpe", fontsize=11)
        ax.set_title(f"{split_name.upper()} Sharpe", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xscale("log")
    
    # Row 2: IC
    for ax_idx, split_name in enumerate(["train", "val", "test"]):
        ax = axes[1][ax_idx]
        ics = [scaling_ic.get(p, {}).get(split_name, {}).get("ic_mean", 0) for p in P_values]
        r2s = [scaling_ic.get(p, {}).get(split_name, {}).get("r2_mean", 0) for p in P_values]
        ax.plot(P_values, ics, 'o-', linewidth=2, markersize=6, color="#E91E63", label="IC")
        ax2 = ax.twinx()
        ax2.plot(P_values, r2s, 's--', linewidth=2, markersize=6, color="#4CAF50", label="R2")
        ax.set_xlabel("P", fontsize=11)
        ax.set_ylabel("IC (mean)", fontsize=11, color="#E91E63")
        ax2.set_ylabel("R2 (mean)", fontsize=11, color="#4CAF50")
        ax.set_title(f"{split_name.upper()} IC & R2", fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        ax2.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
    
    plt.suptitle("Virtue of Complexity: Scaling Curve",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  VoC scaling curve saved: {output_path}")


def plot_pnl_comparison(all_method_results, method_names, output_path, fee=5.0):
    """Plot cumulative PnL for all methods."""
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    colors = ["#E91E63", "#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#00BCD4",
              "#FF5722", "#607D8B", "#795548", "#3F51B5", "#CDDC39", "#009688"]
    split_colors = {"train": "#2196F3", "val": "#FF9800", "test": "#F44336"}
    
    # Split backgrounds
    offset = 0
    for split_name in ["train", "val", "test"]:
        for mname in method_names:
            r = all_method_results.get(mname, {}).get((split_name, fee))
            if r is not None:
                n = len(r.daily_pnl)
                ax.axvspan(offset, offset + n, alpha=0.06, color=split_colors[split_name])
                if split_name != "train":
                    ax.axvline(x=offset, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
                offset += n
                break
    
    for ci, mname in enumerate(method_names):
        results = all_method_results.get(mname, {})
        running_offset = 0.0
        bar_off = 0
        all_x, all_y = [], []
        for split_name in ["train", "val", "test"]:
            r = results.get((split_name, fee))
            if r is None:
                continue
            cum = r.cumulative_pnl.values / BOOKSIZE * 100 + running_offset
            n = len(cum)
            x = np.arange(bar_off, bar_off + n)
            all_x.extend(x)
            all_y.extend(cum)
            running_offset = cum[-1]
            bar_off += n
        if all_x:
            color = colors[ci % len(colors)]
            test_r = results.get(("test", fee))
            test_sr = test_r.sharpe if test_r else 0
            ax.plot(all_x, all_y, color=color, linewidth=2.0,
                    label=f"{mname} (test SR={test_sr:+.2f})")
    
    ax.set_title(f"4H Portfolio: VoC vs Baselines -- {fee:.0f} bps fees",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Cumulative Return (%)", fontsize=12)
    ax.set_xlabel("Bar index (4h bars)", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25)
    ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  PnL comparison chart saved: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="VoC: RFF Pipeline for 4H Crypto")
    parser.add_argument("--P", type=int, nargs="+", default=[50],
                        help="Number of random projections")
    parser.add_argument("--sweep", action="store_true",
                        help="Full scaling sweep: P in [10, 50, 100, 250, 500, 1000]")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip baseline comparators")
    parser.add_argument("--no-qp", action="store_true",
                        help="Skip QP optimizer (faster)")
    parser.add_argument("--with-alphas", action="store_true",
                        help="Augment characteristics with DB alpha signals")
    parser.add_argument("--alphas-only", action="store_true",
                        help="Run ONLY the alpha-augmented variant (skip sources-only)")
    parser.add_argument("--horizon", type=int, nargs="+", default=[1],
                        help="Forward return horizon(s) in bars (1=next bar, 6=next day)")
    parser.add_argument("--window", type=int, default=RIDGE_WINDOW,
                        help=f"Ridge regression trailing window in bars (default={RIDGE_WINDOW})")
    parser.add_argument("--retrain", type=int, default=RIDGE_RETRAIN,
                        help=f"Retrain ridge every N bars (default={RIDGE_RETRAIN})")
    parser.add_argument("--expanding", action="store_true",
                        help="Use expanding window (all past data) instead of rolling")
    parser.add_argument("--ridge-lambda", type=str, default="scaled",
                        help="Ridge lambda mode: 'scaled' (P*10), 'fixed' (1000), 'sqrt' (sqrt(P)*100), or a number")
    args = parser.parse_args()
    
    if args.sweep:
        P_values = [10, 50, 100, 250, 500, 1000]
    else:
        P_values = args.P
    
    t0_total = time.time()
    matrices, universe_df, valid_tickers = load_full_data()
    
    log(f"\n{'='*100}")
    log(f"  VIRTUE OF COMPLEXITY -- Random Fourier Feature Pipeline")
    log(f"  Universe: {UNIVERSE} | Interval: {INTERVAL}")
    log(f"  P values to test: {P_values}")
    log(f"  Ridge: window={'expanding' if args.expanding else args.window}, retrain={args.retrain}, warmup={RIDGE_WARMUP}")
    log(f"  Ridge lambda: {args.ridge_lambda}")
    log(f"  Horizons: {args.horizon} bars")
    log(f"  Splits: train {SPLITS['train']}, val {SPLITS['val']}, test {SPLITS['test']}")
    log(f"  Fee levels: {FEE_LEVELS} bps")
    log(f"  Mode: {'sources+alphas' if args.with_alphas or args.alphas_only else 'sources only'}")
    log(f"{'='*100}")
    
    all_method_results = {}
    all_ic_r2 = {}
    scaling_results = {}
    scaling_ic = {}
    
    # ========================================================================
    # Build matrix variants: "src" = raw sources only, "aug" = sources + DB alphas
    # ========================================================================
    
    matrix_variants = []
    
    if not args.alphas_only:
        matrix_variants.append(("src", matrices))
    
    if args.with_alphas or args.alphas_only:
        log(f"\n  === Augmenting with DB Alphas ===")
        aug_matrices, n_alphas = augment_matrices_with_db_alphas(matrices, universe_df)
        if n_alphas > 0:
            matrix_variants.append(("src+alphas", aug_matrices))
            log(f"  Augmented: K={len([k for k in aug_matrices if k not in EXCLUDE_FIELDS and aug_matrices[k].shape == matrices['close'].shape])} characteristics (was {len([k for k in matrices if k not in EXCLUDE_FIELDS and matrices[k].shape == matrices['close'].shape])})")
    
    # ========================================================================
    # For each variant x P: run raw, smoothed, and QP
    # ========================================================================
    
    for variant_name, variant_matrices in matrix_variants:
        log(f"\n  {'='*80}")
        log(f"  VARIANT: {variant_name}")
        n_chars = len([k for k in variant_matrices
                       if k not in EXCLUDE_FIELDS
                       and variant_matrices[k].shape == matrices['close'].shape])
        log(f"  Characteristics: {n_chars}")
        log(f"  {'='*80}")
        
        tag = f"[{variant_name}] " if len(matrix_variants) > 1 else ""
        
        for P in P_values:
          for fwd_h in args.horizon:
            h_tag = f"h{fwd_h}" if len(args.horizon) > 1 or fwd_h > 1 else ""
            h_label = f" h={fwd_h}" if fwd_h > 1 else ""
            log(f"\n  === {tag}VoC P={P}{h_label} (2P={2*P} features) ===")
            t0 = time.time()
            
            # Compute ridge lambda based on mode
            lam_mode = args.ridge_lambda
            if lam_mode == "scaled":
                ridge_lam = float(P) * 10.0
            elif lam_mode == "fixed":
                ridge_lam = 1000.0
            elif lam_mode == "sqrt":
                ridge_lam = np.sqrt(P) * 100.0
            else:
                ridge_lam = float(lam_mode)  # Direct numeric value
            
            # Generate RAW signal (no smoothing)
            raw_alpha = run_voc_pipeline(P, variant_matrices, universe_df, valid_tickers,
                                         signal_halflife=0, forward_horizon=fwd_h,
                                         ridge_window=args.window,
                                         retrain_every=args.retrain,
                                         expanding=args.expanding,
                                         ridge_lambda=ridge_lam)
            gen_time = time.time() - t0
            log(f"  Signal generation: {gen_time:.1f}s")
            
            # Build method name suffix for horizon
            h_sfx = f" h{fwd_h}" if fwd_h > 1 else ""
            
            # --- RAW ---
            mname_raw = f"{tag}VoC P={P}{h_sfx} (raw)"
            results_raw, ic_raw = evaluate_signal(raw_alpha, matrices, universe_df, name=mname_raw)
            all_method_results[mname_raw] = results_raw
            all_ic_r2[mname_raw] = ic_raw
            
            # Use first variant + first horizon for scaling curve
            if variant_name == matrix_variants[0][0] and fwd_h == args.horizon[0]:
                scaling_results[P] = results_raw
                scaling_ic[P] = ic_raw
            
            for split_name in ["train", "val", "test"]:
                r0 = results_raw.get((split_name, 0.0))
                ic = ic_raw.get(split_name, {}).get("ic_mean", 0)
                r2 = ic_raw.get(split_name, {}).get("r2_mean", 0)
                if r0:
                    log(f"  {split_name:5s} raw : SR={r0.sharpe:+.3f} (0bps), "
                        f"TO={r0.turnover:.4f}, IC={ic:+.4f}, R2={r2:+.5f}")
            
            # --- SMOOTHED ---
            smooth_alpha = raw_alpha.ewm(halflife=12, min_periods=1).mean()
            mname_smooth = f"{tag}VoC P={P}{h_sfx} (ema12)"
            results_smooth, ic_smooth = evaluate_signal(smooth_alpha, matrices, universe_df,
                                                         name=mname_smooth)
            all_method_results[mname_smooth] = results_smooth
            all_ic_r2[mname_smooth] = ic_smooth
            
            for split_name in ["train", "val", "test"]:
                r5 = results_smooth.get((split_name, 5.0))
                if r5:
                    log(f"  {split_name:5s} ema12: SR={r5.sharpe:+.3f} (5bps), "
                        f"TO={r5.turnover:.4f}")
            
            # --- QP ---
            if not args.no_qp:
                log(f"  Building QP for P={P}...")
                t0q = time.time()
                returns_df = matrices.get("returns")
                qp_alpha = qp_optimize_signal(raw_alpha, matrices, universe_df, returns_df)
                mname_qp = f"{tag}VoC P={P}{h_sfx} (+QP)"
                results_qp, ic_qp = evaluate_signal(qp_alpha, matrices, universe_df,
                                                     name=mname_qp, is_qp=True)
                all_method_results[mname_qp] = results_qp
                all_ic_r2[mname_qp] = ic_qp
                log(f"  QP done in {time.time()-t0q:.1f}s")
                
                for split_name in ["train", "val", "test"]:
                    r5q = results_qp.get((split_name, 5.0))
                    if r5q:
                        log(f"  {split_name:5s} +QP : SR={r5q.sharpe:+.3f} (5bps), "
                            f"TO={r5q.turnover:.4f}")
    
    # ========================================================================
    # Baselines
    # ========================================================================
    
    if not args.no_baseline:
        log(f"\n  === Building Baselines ===")
        
        log(f"\n  Building Equal Weight...")
        eq_signal = build_equal_weight_combiner(matrices, universe_df)
        if eq_signal is not None:
            r, ic = evaluate_signal(eq_signal, matrices, universe_df, name="Equal Weight")
            all_method_results["Equal Weight"] = r
            all_ic_r2["Equal Weight"] = ic
        
        log(f"\n  Building Billion Alphas...")
        ba_signal = build_billion_alphas_combiner(matrices, universe_df)
        if ba_signal is not None:
            r, ic = evaluate_signal(ba_signal, matrices, universe_df, name="Billion Alphas")
            all_method_results["Billion Alphas"] = r
            all_ic_r2["Billion Alphas"] = ic
    
    # ========================================================================
    # Build method name list for tables
    # ========================================================================
    
    # Use the actual keys we stored -- preserves insertion order (Python 3.7+)
    method_names = [k for k in all_method_results.keys()]
    
    # ========================================================================
    # Print tables
    # ========================================================================
    
    log(f"\n\n{'='*130}")
    log(f"  FULL RESULTS TABLE -- VoC vs Baselines")
    log(f"{'='*130}")
    
    print_results_table(all_method_results, all_ic_r2, method_names)
    
    # Detailed per-method breakdown
    log(f"\n\n{'='*130}")
    log(f"  DETAILED PER-METHOD BREAKDOWN")
    log(f"{'='*130}")
    
    for mname in method_names:
        results = all_method_results.get(mname, {})
        ic_data = all_ic_r2.get(mname, {})
        log(f"\n  -- {mname} --")
        
        # IC/R2 summary
        for split in ["train", "val", "test"]:
            d = ic_data.get(split, {})
            log(f"  {split:5s} IC={d.get('ic_mean',0):+.4f} (std={d.get('ic_std',0):.4f}, "
                f"IR={d.get('ic_ir',0):+.3f}), R2={d.get('r2_mean',0):+.5f}")
        
        log(f"  {'Fee':>5} | {'Split':<6} | {'Sharpe':>8} {'Ret%':>9} {'RetAnn%':>10} "
            f"{'MaxDD%':>8} {'TO':>8} {'Fitness':>8}")
        log(f"  {'-'*75}")
        for fee in FEE_LEVELS:
            for split_name in ["train", "val", "test"]:
                r = results.get((split_name, fee))
                if r is None:
                    log(f"  {fee:5.0f} | {split_name:<6} | {'N/A':>8}")
                    continue
                ret_pct = r.total_pnl / BOOKSIZE * 100
                log(f"  {fee:5.0f} | {split_name:<6} | {r.sharpe:+8.3f} {ret_pct:+8.2f}% "
                      f"{r.returns_ann*100:+9.2f}% {r.max_drawdown*100:7.2f}% "
                      f"{r.turnover:7.4f} {r.fitness:7.2f}")
    
    # ========================================================================
    # Charts
    # ========================================================================
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if len(P_values) >= 3:
        plot_voc_scaling_curve(scaling_results, scaling_ic,
                               os.path.join(base_dir, "voc_scaling_curve.png"))
    
    # PnL at 0bps (raw) and 5bps (smoothed)
    raw_names = [n for n in method_names if "(raw)" in n or n in ("Equal Weight", "Billion Alphas")]
    plot_pnl_comparison(all_method_results, raw_names,
                        os.path.join(base_dir, "voc_pnl_0bps.png"), fee=0.0)
    
    smooth_names = [n for n in method_names if "(ema12)" in n or "(+QP)" in n
                    or n in ("Equal Weight", "Billion Alphas")]
    plot_pnl_comparison(all_method_results, smooth_names,
                        os.path.join(base_dir, "voc_pnl_5bps.png"), fee=5.0)
    
    elapsed = time.time() - t0_total
    log(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    log(f"{'='*100}")
    
    # --- Write Report ---
    report_path = os.path.join(base_dir, "voc_results.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Virtue of Complexity -- 4H Crypto Results\n")
        f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Configuration\n")
        f.write(f"- P values: {P_values}\n")
        f.write(f"- Ridge: window={RIDGE_WINDOW}, retrain={RIDGE_RETRAIN}, lambda=P*10\n")
        f.write(f"- RFF seed: {RFF_SEED}\n")
        f.write(f"- Universe: {UNIVERSE}, Interval: {INTERVAL}\n")
        f.write(f"- Modes: Raw (no smoothing), EMA(12), QP-Optimized\n\n")
        f.write("## Results\n```text\n")
        f.write("\n".join(REPORT_BUFFER))
        f.write("\n```\n\n")
        f.write("## Charts\n")
        if len(P_values) >= 3:
            f.write("- [voc_scaling_curve.png](voc_scaling_curve.png)\n")
        f.write("- [voc_pnl_0bps.png](voc_pnl_0bps.png)\n")
        f.write("- [voc_pnl_5bps.png](voc_pnl_5bps.png)\n")
    
    print(f"\n  Report saved to: {report_path}")


if __name__ == "__main__":
    main()
