"""
aipt_kucoin.py — Replication of Didisheim, Ke, Kelly & Malamud (2023/2025)
          "APT or AIPT? The Surprising Dominance of Large Factor Models"
          (SSRN 4388526)
          Applied to the KuCoin TOP100 Perpetual Futures Universe (4h bars)

Paper Reference:   https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4388526
Original Setting:  Monthly US equities, 130 JKP characteristics, up to 360,000
                   Random Fourier Factors, ridge-regularized Markowitz SDF.

Adaptation Notes (Crypto):
  • We use 31 KuCoin 4h bar characteristics (momentum, volume, volatility,
    microstructure) instead of 130 monthly equity anomalies.
  • Rolling training windows are measured in 4h bars (not months).
  • The universe is KUCOIN_TOP100 perpetual futures (~100 assets) instead
    of ~3,000 NYSE/AMEX/NASDAQ stocks.
  • Bar-level returns replace monthly returns; Sharpe is annualized assuming
    6 bars/day × 365 days.
  • We evaluate VoC curves across P = {10, 50, 200, 1000, 5000, 20000}
    random Fourier factors and z ∈ {1e-5, 1e-3, 1e-1, 1, 10}.

Algorithm (Section 2 of paper):
  1. Load D characteristics Z_t ∈ R^{N_t × D} at each bar.
  2. Generate P random Fourier features:
       S_{2p-1,t} = sin(γ Z_t θ_p),  S_{2p,t} = cos(γ Z_t θ_p)
     where θ_p ~ N(0, I_D), γ ~ Uniform({0.5…1.0}).  [Eq 6]
  3. Build P characteristic-managed factor portfolios:
       F_{p,t+1} = N_t^{-1/2} S_{p,t}' R_{t+1}           [Eq 5]
  4. Estimate ridge-Markowitz SDF weights:
       λ̂(z) = (zI + Ê[F F'])^{-1} Ê[F]                   [Eq 9]
  5. Out-of-sample SDF portfolio:
       R^M_{t+1} = λ̂' F_{t+1}
  6. Compute OOS Sharpe ratio and HJ distance.

Usage:
    python aipt_kucoin.py                     # Run full VoC analysis
    python aipt_kucoin.py --P 200 --z 0.001   # Single configuration
    python aipt_kucoin.py --voc               # Full VoC curve sweep
    python aipt_kucoin.py --macro             # Macro predictability (BTC as proxy)

Output:
    Prints VoC tables + saves figures to data/aipt_results/
"""

import sys, os, argparse, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# CONFIGURATION
# ============================================================================

UNIVERSE      = "KUCOIN_TOP100"
INTERVAL      = "4h"
BARS_PER_DAY  = 6
DAYS_PER_YEAR = 365
BARS_PER_YEAR = BARS_PER_DAY * DAYS_PER_YEAR   # 2190

# Data splits — rolling window approach (paper uses 360-month rolling)
# For 4h bars: 360 months ≈ 360×30×6 = 64,800 bars, but we only have ~5,900.
# We use a 2-year rolling window ≈ 2×365×6 = 4,380 bars, then test OOS.
TRAIN_BARS     = 4380       # ~2 years of 4h bars
MIN_TRAIN_BARS = 1000       # minimum to start rolling
REBAL_EVERY    = 12         # re-estimate SDF weights every N bars (not per-bar!)

# Out-of-sample period
OOS_START = "2024-09-01"    # validation: Sep 2024 → Apr 2026

# Coverage filter
COVERAGE_CUTOFF = 0.3

# Paper's default VoC sweep parameters
P_GRID = [10, 50, 200, 1000, 5000, 20000]
Z_GRID = [1e-5, 1e-3, 1e-1, 1.0, 10.0]

# Bandwidth parameters (paper Section 2.3, footnote 12)
GAMMA_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Number of random seeds to average over (paper uses 20 for small P)
N_SEEDS = 5

# Output directory
RESULTS_DIR = Path("data/aipt_results")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SDFResult:
    """Out-of-sample SDF performance metrics."""
    P: int                          # number of factors
    z: float                        # ridge penalty
    c: float                        # complexity ratio P/T
    oos_mean: float                 # mean OOS SDF return (per bar)
    oos_std: float                  # std OOS SDF return
    sharpe_annual: float            # annualized Sharpe ratio
    hjd: float                      # Hansen-Jagannathan distance
    n_oos_bars: int                 # number of OOS observations
    cum_return: Optional[np.ndarray] = None  # cumulative return path
    oos_returns: Optional[pd.Series] = None  # raw OOS returns


# ============================================================================
# DATA LOADING
# ============================================================================

_DATA_CACHE = {}

def load_data():
    """Load all KuCoin matrices and universe, return full-period data."""
    if "full" in _DATA_CACHE:
        return _DATA_CACHE["full"]

    mat_dir  = Path(f"data/kucoin_cache/matrices/{INTERVAL}")
    uni_path = Path(f"data/kucoin_cache/universes/{UNIVERSE}_{INTERVAL}.parquet")

    universe_df = pd.read_parquet(uni_path)
    coverage = universe_df.sum(axis=0) / len(universe_df)
    valid_tickers = sorted(coverage[coverage > COVERAGE_CUTOFF].index.tolist())

    matrices = {}
    for fp in sorted(mat_dir.glob("*.parquet")):
        df = pd.read_parquet(fp)
        cols = [c for c in valid_tickers if c in df.columns]
        if cols:
            matrices[fp.stem] = df[cols]

    # Compute returns from close prices (price differences, not pct_change,
    # because the sim uses additive PnL on booksize-normalized positions)
    close = matrices["close"]
    matrices["returns_pct"] = close.pct_change()

    result = (matrices, universe_df[valid_tickers], valid_tickers)
    _DATA_CACHE["full"] = result
    return result


def build_characteristics_matrix(matrices, tickers, bar_idx):
    """
    Build the Z_t matrix (N_t × D) of cross-sectionally rank-standardized
    characteristics at bar `bar_idx`.

    Paper (Section 2.5): "We cross-sectionally rank-standardize each
    characteristic and map it to the [-0.5, 0.5] interval."

    We use all available non-price matrices as characteristics:
    momentum, volatility, volume metrics, microstructure signals.
    """
    # Characteristics to use (exclude raw OHLC and returns — those are
    # the base assets, not conditioning variables in the AIPT framework)
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

    available = [c for c in CHAR_NAMES if c in matrices]
    D = len(available)
    N = len(tickers)

    # Build N × D matrix
    Z = np.full((N, D), np.nan)
    for j, char_name in enumerate(available):
        df = matrices[char_name]
        if bar_idx < len(df.index):
            row = df.iloc[bar_idx]
            vals = row.reindex(tickers).values.astype(np.float64)
            Z[:, j] = vals

    # Rank-standardize each column to [-0.5, 0.5]  (paper Section 2.5)
    for j in range(D):
        col = Z[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < 3:
            Z[:, j] = 0.0
            continue
        # Rank, then map to [-0.5, 0.5]
        ranks = np.full(N, np.nan)
        ranks[valid] = pd.Series(col[valid]).rank(pct=True).values - 0.5
        Z[:, j] = np.nan_to_num(ranks, nan=0.0)

    return Z, available


def build_characteristics_panel(matrices, tickers, start_idx, end_idx):
    """
    Build Z for all bars in [start_idx, end_idx) efficiently.
    Returns 3D numpy array (T, N, D) of rank-standardized characteristics.
    Uses scipy.stats.rankdata for vectorized ranking (no per-bar pd.Series).
    """
    from scipy.stats import rankdata

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
    available = [c for c in CHAR_NAMES if c in matrices]
    D = len(available)
    N = len(tickers)
    actual_end = min(end_idx, matrices[available[0]].shape[0])
    T_panel = actual_end - start_idx

    # Pre-extract all characteristics into a 3D array (T, N, D)
    raw = np.full((T_panel, N, D), np.nan, dtype=np.float64)
    for j, char_name in enumerate(available):
        arr = matrices[char_name].reindex(columns=tickers).values.astype(np.float64)
        raw[:, :, j] = arr[start_idx:actual_end, :]

    # Vectorized rank-standardize: for each (t, j), rank across N assets
    panel_3d = np.zeros((T_panel, N, D), dtype=np.float64)
    for t in range(T_panel):
        for j in range(D):
            col = raw[t, :, j]
            valid = ~np.isnan(col)
            n_valid = valid.sum()
            if n_valid < 3:
                continue
            # rankdata on valid entries, then map to [-0.5, 0.5]
            r = rankdata(col[valid], method='average')
            r = r / n_valid - 0.5  # map to [-0.5, 0.5]
            panel_3d[t, valid, j] = r

    # Return as dict for backward compat, keyed by absolute bar index
    panel = {}
    for i in range(T_panel):
        panel[start_idx + i] = panel_3d[i]  # (N, D)

    return panel, available


# ============================================================================
# RANDOM FOURIER FEATURES (Section 2.3, Eq. 6)
# ============================================================================

def generate_rff_params(D, P, seed=42):
    """
    Generate random parameters for P/2 pairs of sin/cos features.

    θ_p ~ N(0, I_D)  for p = 1, ..., P/2
    γ_p ~ Uniform({0.5, 0.6, ..., 1.0})

    Returns: theta (P//2 × D), gamma (P//2,)
    """
    rng = np.random.default_rng(seed)
    n_pairs = P // 2
    theta = rng.standard_normal((n_pairs, D))        # θ_p ~ N(0, I)
    gamma = rng.choice(GAMMA_GRID, size=n_pairs)      # γ from grid
    return theta, gamma


def compute_rff_signals(Z_t, theta, gamma):
    """
    Compute Random Fourier Feature signals S_t for a single bar.

    Z_t:    (N, D) matrix of rank-standardized characteristics
    theta:  (P//2, D) random projection weights
    gamma:  (P//2,) bandwidth parameters

    Returns: S_t (N, P) matrix of nonlinear signals
               S_{2p-1} = sin(γ_p Z_t θ_p)
               S_{2p}   = cos(γ_p Z_t θ_p)
    """
    # proj = Z_t @ theta.T  → (N, P//2)
    proj = Z_t @ theta.T                              # (N, P//2)
    # Scale each column by its gamma
    proj_scaled = proj * gamma[np.newaxis, :]          # (N, P//2)

    sin_part = np.sin(proj_scaled)                     # (N, P//2)
    cos_part = np.cos(proj_scaled)                     # (N, P//2)

    # Interleave: [sin_1, cos_1, sin_2, cos_2, ...]
    N = Z_t.shape[0]
    P = 2 * theta.shape[0]
    S_t = np.empty((N, P))
    S_t[:, 0::2] = sin_part
    S_t[:, 1::2] = cos_part

    return S_t


# ============================================================================
# CHARACTERISTIC-MANAGED FACTOR PORTFOLIOS (Section 2.2, Eq. 5)
# ============================================================================

def compute_factor_returns(S_t, R_t1, N_t):
    """
    Compute factor returns: F_{p,t+1} = N_t^{-1/2} S_{p,t}' R_{t+1}  [Eq 5]

    S_t:   (N, P) signals at time t
    R_t1:  (N,) returns at time t+1
    N_t:   number of valid assets
    """
    # Weight factor returns by 1/sqrt(N) for consistent scale
    scale = 1.0 / np.sqrt(max(N_t, 1))
    F_t1 = scale * (S_t.T @ R_t1)                    # (P,)
    return F_t1


# ============================================================================
# RIDGE-MARKOWITZ SDF ESTIMATION (Section 2.4, Eq. 9)
# ============================================================================

def estimate_ridge_markowitz(F_matrix, z):
    """
    Estimate SDF weights via ridge-regularized Markowitz:
        λ̂(z) = (zI + Ê[F F'])^{-1} Ê[F]              [Eq 9]

    F_matrix: (T, P) matrix of training factor returns
    z:        ridge penalty

    Returns: lambda_hat (P,) — SDF factor weights
    """
    T, P = F_matrix.shape

    # Sample mean: Ê[F]
    mu = F_matrix.mean(axis=0)                        # (P,)

    if P <= T:
        # Standard O(P^3) inversion
        # Sample second moment: Ê[FF']
        FF = (F_matrix.T @ F_matrix) / T                  # (P, P)

        # Ridge: (zI + FF)^{-1} mu
        A = z * np.eye(P) + FF
        try:
            lambda_hat = np.linalg.solve(A, mu)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            lambda_hat = np.linalg.lstsq(A, mu, rcond=None)[0]
    else:
        # O(T^3) inversion via Woodbury matrix identity for extreme overparameterization
        # (z I_P + 1/T F^T F)^{-1} mu = 1/z * mu - 1/z * F^T (z T I_T + F F^T)^{-1} (F mu)
        FFT = F_matrix @ F_matrix.T                       # (T, T)
        A_T = z * T * np.eye(T) + FFT                     # (T, T)
        F_mu = F_matrix @ mu                              # (T,)
        try:
            inv_F_mu = np.linalg.solve(A_T, F_mu)
        except np.linalg.LinAlgError:
            inv_F_mu = np.linalg.lstsq(A_T, F_mu, rcond=None)[0]
        
        lambda_hat = (mu - F_matrix.T @ inv_F_mu) / z
        
    return lambda_hat


# ============================================================================
# HANSEN-JAGANNATHAN DISTANCE (Section 2.6, Eq. 13)
# ============================================================================

def compute_hjd(M_oos, R_test_oos):
    """
    Compute out-of-sample Hansen-Jagannathan Distance.

    HJD = Ê_OOS[M R_T]' Ê_OOS[R_T R_T']^+ Ê_OOS[M R_T]   [Eq 13]

    M_oos:       (T_oos,) SDF values 1 - R^M_t
    R_test_oos:  (T_oos, K) test asset returns

    Returns: scalar HJD
    """
    T_oos, K = R_test_oos.shape

    # Pricing errors: Ê[M R_T]
    MR = (M_oos[:, np.newaxis] * R_test_oos).mean(axis=0)  # (K,)

    # Second moment of test assets: Ê[R_T R_T']
    RR = (R_test_oos.T @ R_test_oos) / T_oos               # (K, K)

    # Moore-Penrose pseudo-inverse (handles rank deficiency when K > T)
    try:
        RR_inv = np.linalg.pinv(RR)
    except np.linalg.LinAlgError:
        return np.nan

    hjd = float(MR @ RR_inv @ MR)
    return hjd


# ============================================================================
# CORE: ROLLING WINDOW SDF EVALUATION
# ============================================================================

def run_single_config(P, z, seed=42, verbose=True):
    """
    Run the full AIPT pipeline for a single (P, z) configuration.

    1. Load data
    2. Generate RFF parameters
    3. Rolling window: for each OOS bar, train ridge-Markowitz on last
       TRAIN_BARS bars of factor returns, then record OOS SDF return.
    4. Compute performance metrics.

    Returns: SDFResult
    """
    t0 = time.time()
    matrices, universe_df, tickers = load_data()

    close = matrices["close"]
    returns_pct = matrices["returns_pct"]
    dates = close.index
    T_total = len(dates)
    N = len(tickers)

    # Find OOS start index
    oos_start_idx = None
    for i, dt in enumerate(dates):
        if str(dt) >= OOS_START:
            oos_start_idx = i
            break
    if oos_start_idx is None:
        raise ValueError(f"OOS start {OOS_START} not found in data")

    # Determine characteristic dimension
    _, char_names = build_characteristics_matrix(matrices, tickers, oos_start_idx)
    D = len(char_names)

    # Generate RFF parameters
    theta, gamma = generate_rff_params(D, P, seed=seed)

    if verbose:
        print(f"  Config: P={P:,}, z={z:.0e}, D={D}, N={N}, "
              f"T_total={T_total:,}, OOS start idx={oos_start_idx}", flush=True)

    # ── Pre-compute characteristics panel for full data range ─────────────
    # For efficiency, compute Z for all needed bars
    # We need from (oos_start_idx - TRAIN_BARS) to end
    panel_start = max(0, oos_start_idx - TRAIN_BARS - 10)
    if verbose:
        print(f"  Building characteristics panel ({panel_start} to {T_total})...", flush=True)
    Z_panel, _ = build_characteristics_panel(
        matrices, tickers, panel_start, T_total
    )
    if verbose:
        print(f"  Panel built: {len(Z_panel)} bars", flush=True)

    # ── Pre-compute ALL factor returns for the entire period ──────────────
    # F_{p,t+1} = N_t^{-1/2} S_{p,t}' R_{t+1}
    # We store factor returns for bars panel_start+1 to T_total-1
    # (bar t's factor return uses signal at t and return at t+1)

    returns_np = returns_pct.reindex(columns=tickers).values.astype(np.float64)

    # Pre-compute: for each bar t in panel, compute S_t and F_{t+1}
    if verbose:
        print(f"  Computing {len(Z_panel)} factor returns (P={P})...", flush=True)
    factor_returns = {}  # bar_idx -> F_{t+1} vector (P,)
    n_computed = 0
    for t in range(panel_start, T_total - 1):
        if t not in Z_panel:
            continue
        Z_t = Z_panel[t]

        # Mask NaN assets
        r_t1 = returns_np[t + 1, :]
        valid = ~np.isnan(r_t1) & ~np.isnan(Z_t).any(axis=1)
        N_t = valid.sum()
        if N_t < 5:
            continue

        S_t = compute_rff_signals(Z_t[valid], theta, gamma)  # (N_t, P)
        r_valid = np.nan_to_num(r_t1[valid], nan=0.0)
        F_t1 = compute_factor_returns(S_t, r_valid, N_t)     # (P,)
        factor_returns[t + 1] = F_t1  # keyed by t+1 (the return date)
        n_computed += 1
        if verbose and n_computed % 500 == 0:
            print(f"    ...{n_computed} bars done", flush=True)
    if verbose:
        print(f"  Factor returns computed: {len(factor_returns)} bars", flush=True)

    # ── Rolling Window SDF Estimation ─────────────────────────────────────
    # KEY OPTIMIZATION: Re-estimate lambda every REBAL_EVERY bars, not every
    # single bar. Paper uses monthly rebalancing; we use ~2 days (12 bars).
    oos_sdf_returns = []
    oos_dates = []

    # Sort all bar indices once
    all_fr_indices = sorted(factor_returns.keys())
    oos_bar_indices = [t for t in all_fr_indices if t >= oos_start_idx]

    if verbose:
        print(f"  Rolling SDF estimation: {len(oos_bar_indices)} OOS bars, "
              f"rebal every {REBAL_EVERY}...", flush=True)

    lambda_hat = None
    bars_since_rebal = REBAL_EVERY  # force immediate first estimation

    for oos_t in oos_bar_indices:
        # Re-estimate weights periodically
        if bars_since_rebal >= REBAL_EVERY or lambda_hat is None:
            # Collect training window factor returns
            train_indices = [
                t for t in all_fr_indices
                if t < oos_t and t >= oos_t - TRAIN_BARS
            ]

            if len(train_indices) < MIN_TRAIN_BARS:
                continue

            # Build training factor return matrix
            F_train = np.vstack([factor_returns[t] for t in train_indices])

            # Estimate ridge-Markowitz SDF weights
            lambda_hat = estimate_ridge_markowitz(F_train, z)
            bars_since_rebal = 0

        # OOS SDF portfolio return
        F_oos = factor_returns[oos_t]      # (P,)
        R_M = float(lambda_hat @ F_oos)    # scalar SDF portfolio return

        oos_sdf_returns.append(R_M)
        oos_dates.append(dates[oos_t])
        bars_since_rebal += 1

    if len(oos_sdf_returns) < 20:
        if verbose:
            print(f"  WARNING: Only {len(oos_sdf_returns)} OOS observations")
        return SDFResult(P=P, z=z, c=P/TRAIN_BARS, oos_mean=0, oos_std=1,
                         sharpe_annual=0, hjd=np.nan, n_oos_bars=len(oos_sdf_returns))

    oos_arr = np.array(oos_sdf_returns)
    oos_series = pd.Series(oos_arr, index=oos_dates)

    # ── Performance Metrics ───────────────────────────────────────────────
    oos_mean = oos_arr.mean()
    oos_std = oos_arr.std(ddof=1)
    sharpe_per_bar = oos_mean / oos_std if oos_std > 1e-12 else 0.0
    sharpe_annual = sharpe_per_bar * np.sqrt(BARS_PER_YEAR)

    # HJD: use the raw asset returns as test assets
    test_returns = returns_np[
        [dates.get_loc(d) for d in oos_dates], :
    ]
    test_returns = np.nan_to_num(test_returns, nan=0.0)
    M_oos = 1.0 - oos_arr
    hjd = compute_hjd(M_oos, test_returns)

    elapsed = time.time() - t0

    result = SDFResult(
        P=P, z=z, c=P/TRAIN_BARS,
        oos_mean=oos_mean, oos_std=oos_std,
        sharpe_annual=sharpe_annual,
        hjd=hjd,
        n_oos_bars=len(oos_sdf_returns),
        cum_return=np.cumsum(oos_arr),
        oos_returns=oos_series,
    )

    if verbose:
        print(f"  Done in {elapsed:.1f}s | c={result.c:.2f} | "
              f"SR={sharpe_annual:+.3f} | HJD={hjd:.4f} | "
              f"T_OOS={len(oos_sdf_returns)}", flush=True)

    return result


# ============================================================================
# VOC CURVES (Section 3.1, Figure 2)
# ============================================================================

def run_voc_curves(p_grid=None, z_grid=None, n_seeds=N_SEEDS):
    """
    Run the full Virtue-of-Complexity analysis.

    For each (P, z) pair, average results over n_seeds random draws.
    Returns a DataFrame with all results.
    """
    if p_grid is None:
        p_grid = P_GRID
    if z_grid is None:
        z_grid = Z_GRID

    results = []
    total = len(p_grid) * len(z_grid) * n_seeds
    done = 0

    print(f"\n{'='*80}", flush=True)
    print(f"  AIPT VoC CURVE ANALYSIS -- KuCoin TOP100 (4h)", flush=True)
    print(f"  P grid: {p_grid}", flush=True)
    print(f"  z grid: {z_grid}", flush=True)
    print(f"  Seeds:  {n_seeds}", flush=True)
    print(f"  Total configs: {total}", flush=True)
    print(f"{'='*80}\n", flush=True)

    for z in z_grid:
        for P in p_grid:
            seed_results = []
            for s in range(n_seeds):
                done += 1
                print(f"  [{done}/{total}] P={P:>6,}, z={z:.0e}, seed={s}", flush=True)
                try:
                    res = run_single_config(P, z, seed=42 + s, verbose=True)
                    seed_results.append(res)
                except Exception as e:
                    print(f"    FAILED: {e}")
                    continue

            if seed_results:
                # Average metrics across seeds
                avg_sharpe = np.mean([r.sharpe_annual for r in seed_results])
                avg_hjd = np.nanmean([r.hjd for r in seed_results])
                avg_mean = np.mean([r.oos_mean for r in seed_results])
                avg_std = np.mean([r.oos_std for r in seed_results])
                std_sharpe = np.std([r.sharpe_annual for r in seed_results])

                results.append({
                    "P": P,
                    "z": z,
                    "c": P / TRAIN_BARS,
                    "sharpe_annual": avg_sharpe,
                    "sharpe_std": std_sharpe,
                    "hjd": avg_hjd,
                    "oos_mean": avg_mean,
                    "oos_std": avg_std,
                    "n_seeds": len(seed_results),
                    "n_oos_bars": seed_results[0].n_oos_bars,
                })

    df = pd.DataFrame(results)
    return df


def print_voc_table(df):
    """Print a formatted VoC results table."""
    print(f"\n{'='*100}")
    print(f"  VIRTUE OF COMPLEXITY — KuCoin TOP100 Results")
    print(f"{'='*100}")
    print(f"  {'P':>8}  {'c':>8}  {'z':>8}  {'Sharpe':>10}  {'±':>6}  "
          f"{'HJD':>8}  {'E[R]':>10}  {'σ':>10}  {'T_OOS':>6}")
    print(f"  {'-'*90}")

    for _, row in df.sort_values(["z", "P"]).iterrows():
        print(f"  {int(row['P']):>8,}  {row['c']:>8.3f}  {row['z']:>8.0e}  "
              f"{row['sharpe_annual']:>+10.3f}  {row.get('sharpe_std', 0):>6.3f}  "
              f"{row['hjd']:>8.4f}  {row['oos_mean']:>10.6f}  "
              f"{row['oos_std']:>10.6f}  {int(row['n_oos_bars']):>6}")

    # Print key findings
    best = df.loc[df["sharpe_annual"].idxmax()]
    print(f"\n  BEST CONFIG: P={int(best['P']):,}, z={best['z']:.0e}, "
          f"c={best['c']:.3f}")
    print(f"  Best OOS Sharpe: {best['sharpe_annual']:+.3f}")
    print(f"  Best HJD: {best['hjd']:.4f}")

    # Check for VoC: does Sharpe increase with P for each z?
    print(f"\n  VoC CHECK: Does Sharpe increase with model complexity?")
    for z in sorted(df["z"].unique()):
        sub = df[df["z"] == z].sort_values("P")
        if len(sub) > 1:
            first_sr = sub.iloc[0]["sharpe_annual"]
            last_sr = sub.iloc[-1]["sharpe_annual"]
            direction = "YES ↑" if last_sr > first_sr else "NO ↓"
            print(f"    z={z:.0e}: P={int(sub.iloc[0]['P']):>6,} → "
                  f"P={int(sub.iloc[-1]['P']):>6,}:  "
                  f"SR {first_sr:+.3f} → {last_sr:+.3f}  [{direction}]")


# ============================================================================
# BENCHMARK COMPARISON (Section 3.2)
# ============================================================================

def run_benchmark_comparison():
    """
    Compare the complex SDF with simple factor model benchmarks.

    Benchmarks for crypto:
      1. Market factor (equal-weight return of universe)
      2. Momentum factor (long winners, short losers by 20d returns)
      3. Volume factor (long high volume, short low volume)
      4. Volatility factor (long low vol, short high vol)
      5. Size factor (long small, short large by dollar volume)

    These are the crypto analogues of FF6/HXZ/etc.
    """
    print(f"\n{'='*80}")
    print(f"  BENCHMARK COMPARISON — KuCoin TOP100")
    print(f"{'='*80}\n")

    matrices, universe_df, tickers = load_data()
    close = matrices["close"]
    returns_pct = matrices["returns_pct"]
    dates = close.index

    oos_mask = dates >= OOS_START
    oos_dates = dates[oos_mask]
    oos_returns = returns_pct.loc[oos_dates].reindex(columns=tickers)

    benchmarks = {}

    # 1. Market factor: equal-weight return of universe
    mkt = oos_returns.mean(axis=1)
    benchmarks["Market"] = mkt

    # 2. Momentum factor: long/short by 20d momentum
    if "momentum_20d" in matrices:
        mom = matrices["momentum_20d"].reindex(columns=tickers)
        mom_signal = mom.rank(axis=1, pct=True) - 0.5
        # Factor return = lagged signal × next return
        mom_lag = mom_signal.shift(1).loc[oos_dates]
        abs_sum = mom_lag.abs().sum(axis=1).replace(0, np.nan)
        mom_norm = mom_lag.div(abs_sum, axis=0)
        benchmarks["Momentum_20d"] = (mom_norm * oos_returns).sum(axis=1)

    # 3. Volume factor
    if "adv20" in matrices:
        vol_signal = matrices["adv20"].reindex(columns=tickers)
        vol_rank = vol_signal.rank(axis=1, pct=True) - 0.5
        vol_lag = vol_rank.shift(1).loc[oos_dates]
        abs_sum = vol_lag.abs().sum(axis=1).replace(0, np.nan)
        vol_norm = vol_lag.div(abs_sum, axis=0)
        benchmarks["Volume"] = (vol_norm * oos_returns).sum(axis=1)

    # 4. Volatility factor (low vol, long; high vol, short)
    if "historical_volatility_20" in matrices:
        hv = matrices["historical_volatility_20"].reindex(columns=tickers)
        hv_rank = -(hv.rank(axis=1, pct=True) - 0.5)  # negative = long low vol
        hv_lag = hv_rank.shift(1).loc[oos_dates]
        abs_sum = hv_lag.abs().sum(axis=1).replace(0, np.nan)
        hv_norm = hv_lag.div(abs_sum, axis=0)
        benchmarks["LowVol"] = (hv_norm * oos_returns).sum(axis=1)

    # 5. Beta factor
    if "beta_to_btc" in matrices:
        beta = matrices["beta_to_btc"].reindex(columns=tickers)
        beta_rank = -(beta.rank(axis=1, pct=True) - 0.5)  # long low beta
        beta_lag = beta_rank.shift(1).loc[oos_dates]
        abs_sum = beta_lag.abs().sum(axis=1).replace(0, np.nan)
        beta_norm = beta_lag.div(abs_sum, axis=0)
        benchmarks["LowBeta"] = (beta_norm * oos_returns).sum(axis=1)

    # Compute simple benchmark Sharpe ratios
    print(f"  {'Benchmark':<20} {'Sharpe':>10} {'E[R]':>12} {'σ':>12}")
    print(f"  {'-'*55}")
    for name, rets in benchmarks.items():
        rets_clean = rets.dropna()
        if len(rets_clean) < 20:
            continue
        mu = rets_clean.mean()
        sigma = rets_clean.std(ddof=1)
        sr = mu / sigma * np.sqrt(BARS_PER_YEAR) if sigma > 1e-12 else 0.0
        print(f"  {name:<20} {sr:>+10.3f} {mu:>12.6f} {sigma:>12.6f}")

    # Simple Markowitz on benchmark factors
    bench_df = pd.DataFrame(benchmarks).dropna()
    if len(bench_df) > 60:
        mu = bench_df.mean().values
        cov = bench_df.cov().values
        try:
            z_bench = 0.01
            A = z_bench * np.eye(len(mu)) + cov
            lam = np.linalg.solve(A, mu)
            sdf_ret = bench_df.values @ lam
            sr = sdf_ret.mean() / sdf_ret.std() * np.sqrt(BARS_PER_YEAR)
            print(f"\n  Simple Benchmark SDF (Markowitz of above): Sharpe = {sr:+.3f}")
        except Exception as e:
            print(f"\n  Benchmark Markowitz failed: {e}")

    return benchmarks


# ============================================================================
# PLOTTING
# ============================================================================

def plot_voc_curves(df, save_dir=None):
    """Create VoC plots mirroring Figure 2 of the paper."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    if save_dir is None:
        save_dir = RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("AIPT VoC Curves — KuCoin TOP100 (4h)", fontsize=14, fontweight="bold")

    z_values = sorted(df["z"].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(z_values)))

    for i, z in enumerate(z_values):
        sub = df[df["z"] == z].sort_values("c")
        c_vals = sub["c"].values
        label = f"z={z:.0e}"

        # Panel A: Expected Return
        axes[0, 0].plot(c_vals, sub["oos_mean"].values, "o-", color=colors[i],
                        label=label, markersize=4)
        axes[0, 0].set_title("Panel A: Expected Return")
        axes[0, 0].set_ylabel("E[R^M] per bar")

        # Panel B: Standard Deviation
        axes[0, 1].plot(c_vals, sub["oos_std"].values, "o-", color=colors[i],
                        label=label, markersize=4)
        axes[0, 1].set_title("Panel B: Standard Deviation")
        axes[0, 1].set_ylabel("σ(R^M) per bar")

        # Panel C: Sharpe Ratio
        axes[1, 0].plot(c_vals, sub["sharpe_annual"].values, "o-", color=colors[i],
                        label=label, markersize=4)
        axes[1, 0].set_title("Panel C: Sharpe Ratio (Annualized)")
        axes[1, 0].set_ylabel("Sharpe Ratio")

        # Panel D: Pricing Error (HJD)
        axes[1, 1].plot(c_vals, sub["hjd"].values, "o-", color=colors[i],
                        label=label, markersize=4)
        axes[1, 1].set_title("Panel D: Pricing Error (HJD)")
        axes[1, 1].set_ylabel("HJD")

    for ax in axes.flat:
        ax.set_xlabel("Complexity c = P/T")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = save_dir / "voc_curves.png"
    plt.savefig(path, dpi=150)
    print(f"\n  Saved VoC curves to {path}")
    plt.close()


def plot_cumulative_returns(results_dict, save_dir=None):
    """Plot cumulative return comparison (Panel B of Figure 3)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if save_dir is None:
        save_dir = RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    for label, result in results_dict.items():
        if result.oos_returns is not None:
            # Normalize to 10% annual vol for fair comparison
            vol_target = 0.10 / np.sqrt(BARS_PER_YEAR)
            actual_vol = result.oos_returns.std()
            scale = vol_target / actual_vol if actual_vol > 1e-12 else 1.0
            cum = (result.oos_returns * scale).cumsum()
            ax.plot(cum.index, cum.values, label=label, linewidth=1.5)

    ax.set_title("Cumulative SDF Returns (10% vol-normalized) — KuCoin TOP100",
                 fontsize=13)
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = save_dir / "cumulative_returns.png"
    plt.savefig(path, dpi=150)
    print(f"  Saved cumulative returns to {path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

# ============================================================================
# PRODUCTION BACKTEST — single config with turnover, fees, plot
# ============================================================================

def run_production(P=100, z=1e-3, seed=42):
    """
    Run a single AIPT backtest over the FULL data period with:
      - Asset-level position tracking for turnover
      - Fee modeling at 0bps and 3bps
      - Train/test split labeled on the plot
      - Annualized Sharpe at √2190

    REBAL_EVERY audit:
      At rebalance bar t, lambda is estimated from factor returns in
      [t - TRAIN_BARS, t).  All factor returns in that window are
      F_{j} = S_{j-1}' R_j  where S_{j-1} uses characteristics at bar j-1
      and R_j is the return realized at bar j — both known before bar t. ✓
      Lambda is then held constant for REBAL_EVERY bars. Each OOS bar
      uses lambda @ F_{oos} where F_{oos} uses signal at oos-1 and return
      at oos.  No future information leaks. ✓
    """
    t0 = time.time()
    print(f"\n{'='*80}", flush=True)
    print(f"  AIPT PRODUCTION BACKTEST -- KuCoin TOP100 (4h)", flush=True)
    print(f"  P={P}, z={z:.0e}, seed={seed}", flush=True)
    print(f"  REBAL_EVERY={REBAL_EVERY}, TRAIN_BARS={TRAIN_BARS}", flush=True)
    print(f"{'='*80}\n", flush=True)

    matrices, universe_df, tickers = load_data()
    close = matrices["close"]
    returns_pct = matrices["returns_pct"]
    dates = close.index
    T_total = len(dates)
    N = len(tickers)

    # OOS start
    oos_start_idx = None
    for i, dt in enumerate(dates):
        if str(dt) >= OOS_START:
            oos_start_idx = i
            break
    assert oos_start_idx is not None

    # Characteristics
    _, char_names = build_characteristics_matrix(matrices, tickers, oos_start_idx)
    D = len(char_names)

    # RFF params (fixed random — no data dependency)
    theta, gamma = generate_rff_params(D, P, seed=seed)

    print(f"  D={D} characteristics, N={N} tickers, T={T_total} bars", flush=True)
    print(f"  OOS starts at bar {oos_start_idx} ({dates[oos_start_idx]})", flush=True)

    # Build panel + factor returns (same as run_single_config)
    panel_start = max(0, oos_start_idx - TRAIN_BARS - 10)
    print(f"  Building characteristics panel...", flush=True)
    Z_panel, _ = build_characteristics_panel(matrices, tickers, panel_start, T_total)
    print(f"  Panel built: {len(Z_panel)} bars", flush=True)

    returns_np = returns_pct.reindex(columns=tickers).values.astype(np.float64)

    print(f"  Computing factor returns (P={P})...", flush=True)
    factor_returns = {}
    # Also store per-bar RFF signals for position tracking
    rff_signals = {}   # bar_idx -> (S_t, valid_mask)

    n_computed = 0
    for t in range(panel_start, T_total - 1):
        if t not in Z_panel:
            continue
        Z_t = Z_panel[t]
        r_t1 = returns_np[t + 1, :]
        valid = ~np.isnan(r_t1) & ~np.isnan(Z_t).any(axis=1)
        N_t = valid.sum()
        if N_t < 5:
            continue

        S_t = compute_rff_signals(Z_t[valid], theta, gamma)
        r_valid = np.nan_to_num(r_t1[valid], nan=0.0)
        F_t1 = compute_factor_returns(S_t, r_valid, N_t)
        factor_returns[t + 1] = F_t1
        rff_signals[t] = (S_t, valid, N_t)  # signal at bar t (used for positions)
        n_computed += 1
        if n_computed % 1000 == 0:
            print(f"    ...{n_computed} bars done", flush=True)
    print(f"  Factor returns computed: {len(factor_returns)} bars", flush=True)

    # ── Rolling SDF estimation with NORMALIZED portfolio returns ───────────
    # Key fix: compute asset-level weights w_t, normalize sum(|w|)=1,
    # then return = w_t' @ R_{t+1}.  This gives a proper L/S portfolio
    # return where fees (turnover × fee_rate) are commensurate.
    all_fr_indices = sorted(factor_returns.keys())
    oos_bar_indices = [t for t in all_fr_indices if t >= oos_start_idx]

    print(f"  Rolling SDF: {len(oos_bar_indices)} OOS bars", flush=True)

    port_returns = []      # normalized portfolio return per bar
    position_turnovers = []
    bar_dates = []

    lambda_hat = None
    bars_since_rebal = REBAL_EVERY
    prev_weights = None

    for oos_t in oos_bar_indices:
        # Re-estimate lambda periodically
        if bars_since_rebal >= REBAL_EVERY or lambda_hat is None:
            train_indices = [
                t for t in all_fr_indices
                if t < oos_t and t >= oos_t - TRAIN_BARS
            ]
            if len(train_indices) < MIN_TRAIN_BARS:
                continue
            F_train = np.vstack([factor_returns[t] for t in train_indices])
            lambda_hat = estimate_ridge_markowitz(F_train, z)
            bars_since_rebal = 0

        # Compute normalized asset weights from signal at bar oos_t-1
        sig_bar = oos_t - 1
        if sig_bar not in rff_signals:
            bars_since_rebal += 1
            continue

        S_t, valid_mask, N_t = rff_signals[sig_bar]
        # Raw position: w_i = (1/sqrt(N)) * sum_p lambda_p * S_{p,i}
        raw_w_valid = (1.0 / np.sqrt(N_t)) * (S_t @ lambda_hat)
        # Expand to full N-vector
        raw_w = np.zeros(N)
        raw_w[valid_mask] = raw_w_valid
        # Normalize: sum(|w|) = 1  →  dollar-neutral L/S portfolio
        abs_sum = np.abs(raw_w).sum()
        if abs_sum < 1e-12:
            bars_since_rebal += 1
            continue
        w_norm = raw_w / abs_sum

        # Portfolio return = w_norm' @ R_{t+1}  (using ACTUAL returns)
        r_t1 = returns_np[oos_t, :]
        r_t1_clean = np.nan_to_num(r_t1, nan=0.0)
        port_ret = float(w_norm @ r_t1_clean)
        port_returns.append(port_ret)
        bar_dates.append(dates[oos_t])

        # Turnover
        if prev_weights is not None:
            turnover = np.abs(w_norm - prev_weights).sum() / 2.0
            position_turnovers.append(turnover)
        else:
            position_turnovers.append(0.0)
        prev_weights = w_norm.copy()

        bars_since_rebal += 1

    port_arr = np.array(port_returns)
    turnover_arr = np.array(position_turnovers)

    # ── Metrics ───────────────────────────────────────────────────────────
    mean_bar = port_arr.mean()
    std_bar = port_arr.std(ddof=1)
    sr_bar = mean_bar / std_bar if std_bar > 1e-12 else 0.0
    sr_ann = sr_bar * np.sqrt(BARS_PER_YEAR)

    avg_turnover = turnover_arr.mean()

    # Fee-adjusted returns (fees = one-way fee × 2 sides × turnover)
    fee_0bps = port_arr.copy()
    fee_3bps = port_arr - turnover_arr * 0.0003 * 2

    sr_0bps = sr_ann
    mean_3 = fee_3bps.mean()
    std_3 = fee_3bps.std(ddof=1)
    sr_3bps = (mean_3 / std_3 * np.sqrt(BARS_PER_YEAR)) if std_3 > 1e-12 else 0.0

    elapsed = time.time() - t0

    print(f"\n{'='*80}", flush=True)
    print(f"  RESULTS  (P={P}, z={z:.0e}, {len(port_arr)} OOS bars)", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"  Annualized Sharpe (0bps fees): {sr_0bps:+.3f}", flush=True)
    print(f"  Annualized Sharpe (3bps fees): {sr_3bps:+.3f}", flush=True)
    print(f"  Mean turnover per bar:         {avg_turnover:.4f}", flush=True)
    print(f"  Mean port return per bar:      {mean_bar:.6f}", flush=True)
    print(f"  Std  port return per bar:      {std_bar:.6f}", flush=True)
    print(f"  Fee drag per bar (3bps):       {(turnover_arr * 0.0003 * 2).mean():.6f}", flush=True)
    print(f"  Fees as % of gross signal:     {(turnover_arr * 0.0003 * 2).mean() / max(mean_bar, 1e-12) * 100:.1f}%", flush=True)
    print(f"  Elapsed:                       {elapsed:.1f}s", flush=True)
    print(f"{'='*80}\n", flush=True)

    # ── Plot cumulative returns with train/test split ────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plot", flush=True)
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1],
                             sharex=True)

    # Top panel: Cumulative returns
    ax = axes[0]
    cum_0bps = np.cumsum(fee_0bps)
    cum_3bps = np.cumsum(fee_3bps)

    ax.plot(bar_dates, cum_0bps, label=f"0bps fees  (SR={sr_0bps:+.1f})",
            linewidth=1.5, color="#2196F3")
    ax.plot(bar_dates, cum_3bps, label=f"3bps fees  (SR={sr_3bps:+.1f})",
            linewidth=1.5, color="#FF5722")

    # Mark train/test boundary
    oos_date = dates[oos_start_idx]
    ax.axvline(x=oos_date, color="white", linewidth=1.5, linestyle="--", alpha=0.8)
    ax.axvspan(bar_dates[0], oos_date, alpha=0.08, color="yellow",
               label=f"Train (before {OOS_START})")
    ax.axvspan(oos_date, bar_dates[-1], alpha=0.08, color="green",
               label=f"Test (after {OOS_START})")

    ax.set_title(
        f"AIPT Normalized L/S Portfolio -- KuCoin TOP100 (4h)\n"
        f"P={P} RFF, z={z:.0e}, rebal/{REBAL_EVERY}bars, "
        f"TO={avg_turnover:.3f}/bar, sum|w|=1",
        fontsize=13, fontweight="bold"
    )
    ax.set_ylabel("Cumulative Return (sum|w|=1)")
    ax.grid(True, alpha=0.2)

    # Bottom panel: Rolling turnover
    ax2 = axes[1]
    # Rolling 100-bar average turnover
    to_series = pd.Series(turnover_arr, index=bar_dates)
    to_rolling = to_series.rolling(100, min_periods=1).mean()
    ax2.fill_between(bar_dates, to_rolling.values, alpha=0.5, color="#FFB74D")
    ax2.set_ylabel("Turnover (100-bar avg)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.2)

    # Style both panels
    for a in axes:
        a.set_facecolor("#1a1a2e")
        a.tick_params(colors="white")
        a.xaxis.label.set_color("white")
        a.yaxis.label.set_color("white")
        for spine in a.spines.values():
            spine.set_color("#333")
    axes[0].title.set_color("white")
    fig.patch.set_facecolor("#16213e")
    axes[0].legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white",
                   loc="upper left", fontsize=10)

    plt.tight_layout()
    path = RESULTS_DIR / "aipt_production_backtest.png"
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    print(f"  Saved plot to {path}", flush=True)
    plt.close()

    # Save CSV
    to_df = pd.DataFrame({
        "date": bar_dates,
        "port_return_0bps": fee_0bps,
        "port_return_3bps": fee_3bps,
        "turnover": turnover_arr,
    })
    to_df.to_csv(RESULTS_DIR / "aipt_production_returns.csv", index=False)
    print(f"  Saved returns CSV to {RESULTS_DIR / 'aipt_production_returns.csv'}", flush=True)




def main():
    parser = argparse.ArgumentParser(
        description="AIPT Replication on KuCoin TOP100 (Didisheim-Ke-Kelly-Malamud 2023)"
    )
    parser.add_argument("--P", type=int, default=None,
                        help="Number of RFF factors (single run)")
    parser.add_argument("--z", type=float, default=None,
                        help="Ridge penalty (single run)")
    parser.add_argument("--run", action="store_true",
                        help="Run production backtest (P=100, z=1e-3) with turnover & fees")
    parser.add_argument("--voc", action="store_true",
                        help="Run full VoC curve analysis")
    parser.add_argument("--quick-voc", action="store_true",
                        help="Quick VoC with fewer configs")
    parser.add_argument("--benchmarks", action="store_true",
                        help="Run benchmark comparison")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS,
                        help="Number of seeds to average over")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.run:
        run_production(P=100, z=1e-3, seed=args.seed)

    elif args.voc:
        df = run_voc_curves(n_seeds=args.n_seeds)
        print_voc_table(df)
        plot_voc_curves(df)
        df.to_csv(RESULTS_DIR / "voc_results.csv", index=False)
        print(f"\n  Saved results to {RESULTS_DIR / 'voc_results.csv'}")

    elif args.quick_voc:
        df = run_voc_curves(
            p_grid=[10, 100, 1000, 5000],
            z_grid=[1e-5, 1e-1, 1.0],
            n_seeds=2,
        )
        print_voc_table(df)
        plot_voc_curves(df)
        df.to_csv(RESULTS_DIR / "voc_results_quick.csv", index=False)

    elif args.benchmarks:
        run_benchmark_comparison()

    elif args.P is not None and args.z is not None:
        result = run_single_config(args.P, args.z, seed=args.seed)
        print(f"\n  Result: Sharpe={result.sharpe_annual:+.3f}, "
              f"HJD={result.hjd:.4f}")

    else:
        # Default: production run
        run_production(P=100, z=1e-3, seed=args.seed)


if __name__ == "__main__":
    main()
