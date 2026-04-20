"""
run_4h_portfolio.py -- Unified 4H Portfolio Construction & Evaluation

Evaluates ALL combination methods across ALL splits x ALL fee levels.
Combiners:
  1. Equal Weight
  2. Billion Alphas (Original Kakushadze - NO smoothing)
  3. Factor MAX (5d, 10d lookback)
  4. Adaptive (rolling expected return)
  5. Risk Parity (inverse volatility)

Each combiner is run in two modes:
  - Raw: direct signal normalization
  - QP-Optimized: CVXPY convex optimizer on aggregate signal

Fee levels: 0, 2, 5, 7 bps
Splits: train (2021-01-01 to 2025-01-01), val (2025-01-01 to 2025-09-01), test (2025-09-01 to 2026-03-05)

Usage:
    python run_4h_portfolio.py
"""

import sys, os, time, sqlite3
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

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

# Run full data from before train start (need lookback warmup)
FULL_START = "2020-10-01"   # universe first becomes viable
FULL_END   = "2026-03-05"

FEE_LEVELS = [0.0, 2.0, 5.0, 7.0]

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


def load_alphas():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT a.id, a.expression, COALESCE(e.ic_mean, 0), COALESCE(e.sharpe_is, 0)
        FROM alphas a LEFT JOIN evaluations e ON e.alpha_id = a.id
        WHERE a.archived = 0
          AND a.interval = ?
          AND a.universe = ?
        ORDER BY COALESCE(e.sharpe_is, 0) DESC
    """, (INTERVAL, UNIVERSE)).fetchall()
    conn.close()
    return rows


def evaluate_expression(expression, matrices):
    from src.operators.fastexpression import FastExpressionEngine
    engine = FastExpressionEngine(data_fields=matrices)
    return engine.evaluate(expression)


def process_signal(alpha_df, universe_df=None, max_wt=MAX_WEIGHT):
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


def simulate(alpha_df, returns_df, close_df, universe_df, fees_bps=0.0):
    from src.simulation.vectorized_sim_polars import simulate_vectorized_polars
    return simulate_vectorized_polars(
        alpha_df=alpha_df, returns_df=returns_df, close_df=close_df,
        universe_df=universe_df, booksize=BOOKSIZE,
        max_stock_weight=MAX_WEIGHT, decay=0, delay=0,
        neutralization=NEUTRALIZE, fees_bps=fees_bps,
        bars_per_day=BARS_PER_DAY,
    )


# ============================================================================
# ALPHA SIGNAL LOADING
# ============================================================================

def load_alpha_signals(alphas, matrices):
    """Load and evaluate all alpha expressions. Returns dict {alpha_id: DataFrame}."""
    signals = {}
    for aid, expr, ic, sr in alphas:
        try:
            raw = evaluate_expression(expr, matrices)
            if raw is not None and not raw.empty:
                signals[aid] = raw
        except Exception:
            continue
    return signals


# ============================================================================
# COMBINERS
# ============================================================================

def combiner_equal(alpha_signals, matrices, universe_df, returns_df):
    """Equal-weight: average all alpha signals after per-alpha normalization."""
    combined = None
    n = 0
    for aid, raw in alpha_signals.items():
        normed = process_signal(raw, universe_df=universe_df, max_wt=MAX_WEIGHT)
        if combined is None:
            combined = normed.copy()
        else:
            combined = combined.add(normed, fill_value=0)
        n += 1
    if n > 0:
        combined = combined / n
    return combined


def combiner_billion_alphas(alpha_signals, matrices, universe_df, returns_df):
    """
    Billion Alphas -- ORIGINAL Kakushadze algorithm.
    NO daily smoothing, NO EMA, NO signal_smooth.
    Pure paper implementation: Section 5.3, Appendix A R code.

    Steps:
      1. Compute per-alpha bar-level cross-sectional returns
      2. Aggregate to daily returns (sum of 6 bars per day)
      3. Rolling OLS regression of normalized daily returns
      4. Residual alpha weights = (E_tilde - Y @ beta) / s
      5. L1-normalize to unit exposure
    """
    close_df = matrices["close"]
    dates = close_df.index
    tickers = close_df.columns.tolist()
    n_bars = len(dates)

    # Step 1: Build per-alpha normalized signals and bar returns
    alpha_list = []  # list of DataFrames
    alpha_ids = []
    for aid, raw in alpha_signals.items():
        try:
            sig = raw.reindex(index=dates, columns=tickers)
            if sig.isna().all().all():
                continue
            # Cross-sectional z-score normalization
            if hasattr(sig, 'values'):
                alpha_list.append(sig)
                alpha_ids.append(aid)
        except Exception:
            continue

    N = len(alpha_list)
    if N < 3:
        print("  Billion Alphas: insufficient alphas, falling back to equal", flush=True)
        return combiner_equal(alpha_signals, matrices, universe_df, returns_df)

    # Per-alpha cross-sectional PnL
    ret_df = returns_df.reindex(index=dates, columns=tickers)
    alpha_bar_ret = np.zeros((N, n_bars), dtype=np.float64)

    for i, z in enumerate(alpha_list):
        z_lag = z.shift(1)
        uni_mask = universe_df.reindex_like(z_lag).fillna(False)
        z_masked = z_lag.where(uni_mask)
        r_masked = ret_df.where(uni_mask)
        cs_std = z_masked.std(axis=1).replace(0, np.nan)
        z_norm = z_masked.sub(z_masked.mean(axis=1), axis=0).div(cs_std, axis=0)
        alpha_bar_ret[i] = (z_norm * r_masked).sum(axis=1).fillna(0).values

    # Step 2: Aggregate to daily
    n_days = n_bars // BARS_PER_DAY
    alpha_daily_ret = np.zeros((N, n_days), dtype=np.float64)
    for d in range(n_days):
        s = d * BARS_PER_DAY
        e = s + BARS_PER_DAY
        alpha_daily_ret[:, d] = alpha_bar_ret[:, s:e].sum(axis=1)

    print(f"  Billion Alphas (original): {N} alphas, {n_days} daily obs", flush=True)

    # Step 3 & 4: Rolling Kakushadze regression
    # Use lookback = N - 3 days (paper: need M < N)
    lookback_days = min(N - 3, 15)
    lookback_days = max(lookback_days, 5)
    retrain_every = BARS_PER_DAY  # daily refit
    rm_overall = True

    weights_ts = np.full((n_bars, N), fill_value=1.0 / N)
    current_weights = np.full(N, 1.0 / N)
    last_refit = -retrain_every
    min_lookback_bars = lookback_days * BARS_PER_DAY

    for t in range(min_lookback_bars, n_bars):
        if t - last_refit >= retrain_every:
            day_idx = t // BARS_PER_DAY
            if day_idx < lookback_days:
                weights_ts[t] = current_weights
                continue

            M_plus_1 = lookback_days
            ret_window = alpha_daily_ret[:, day_idx - M_plus_1: day_idx]

            if ret_window.shape[1] < 4:
                weights_ts[t] = current_weights
                continue

            try:
                # R code: s <- apply(ret, 1, sd)
                s = ret_window.std(axis=1, ddof=1)
                s = np.where(s < 1e-15, 1e-15, s)

                # R code: x <- ret - rowMeans(ret)
                x = ret_window - ret_window.mean(axis=1, keepdims=True)

                # R code: y <- x / s
                y = x / s[:, None]

                # R code: y <- y[, -ncol(x)]  -- drop last column
                y = y[:, :-1]

                # R code: if(rm.overall) { ... }
                if rm_overall:
                    y = y - y.mean(axis=0, keepdims=True)
                    y = y[:, :-1]

                K = y.shape[1]
                if K < 2 or K >= N:
                    weights_ts[t] = current_weights
                    last_refit = t
                    continue

                # Expected returns
                E = ret_window.mean(axis=1)
                E_tilde = (E / s).reshape(-1, 1)

                # beta = (Y^T Y)^-1 @ Y^T @ E_tilde
                YtY = y.T @ y
                YtE = y.T @ E_tilde
                beta = np.linalg.solve(YtY, YtE)

                # residuals
                eps = E_tilde - y @ beta
                w = eps.ravel() / s

                # L1-normalize
                l1 = np.abs(w).sum()
                if l1 > 1e-15:
                    w = w / l1
                else:
                    w = np.full(N, 1.0 / N)

                current_weights = w

            except np.linalg.LinAlgError:
                pass
            except Exception:
                pass

            last_refit = t

        weights_ts[t] = current_weights

    # Step 5: Apply weights to alpha signals -- NO SMOOTHING
    composite = pd.DataFrame(0.0, index=dates, columns=tickers)
    for i, z in enumerate(alpha_list):
        # Per-alpha: cross-sectional z-score
        cs_mean = z.mean(axis=1)
        cs_std = z.std(axis=1).replace(0, np.nan)
        z_normed = z.sub(cs_mean, axis=0).div(cs_std, axis=0).clip(-5, 5).fillna(0)
        wi = weights_ts[:, i]
        composite = composite.add(z_normed.mul(pd.Series(wi, index=dates), axis=0))

    # Final cross-sectional z-score -- NO EMA, NO smoothing
    cs_mean = composite.mean(axis=1)
    cs_std = composite.std(axis=1).replace(0, np.nan)
    composite = composite.sub(cs_mean, axis=0).div(cs_std, axis=0).fillna(0)

    # Print weight summary
    final_w = weights_ts[-1]
    top_idx = np.argsort(np.abs(final_w))[::-1][:5]
    print(f"  Billion Alphas: final weights (top 5 by |w|):")
    for idx in top_idx:
        print(f"    alpha #{alpha_ids[idx]}: w={final_w[idx]:+.4f}")
    n_pos = np.sum(final_w > 0.01)
    n_neg = np.sum(final_w < -0.01)
    print(f"  Billion Alphas: pos={n_pos} neg={n_neg} |w|_sum={np.sum(np.abs(final_w)):.4f}")

    return composite


def combiner_factor_max(alpha_signals, matrices, universe_df, returns_df,
                        lookback_bars=60, reweight_every=6):
    """Factor MAX: weight alphas by their max single-bar PnL in lookback window."""
    close_df = matrices["close"]
    dates = close_df.index
    tickers = close_df.columns.tolist()
    n_bars = len(dates)

    alpha_ids = sorted(alpha_signals.keys())
    n_active = len(alpha_ids)

    if n_active < 2:
        return combiner_equal(alpha_signals, matrices, universe_df, returns_df)

    # Per-alpha bar-level PnL
    ret_df = returns_df.reindex(index=dates, columns=tickers).fillna(0.0)
    alpha_bar_pnl = {}

    for aid in alpha_ids:
        raw = alpha_signals[aid]
        sig = raw.reindex(index=dates, columns=tickers).fillna(0.0)
        sig_vals = sig.values
        mu = np.nanmean(sig_vals, axis=1, keepdims=True)
        sigma = np.nanstd(sig_vals, axis=1, keepdims=True)
        sigma = np.where(sigma > 1e-10, sigma, 1.0)
        sig_normed = (sig_vals - mu) / sigma
        sig_normed = np.nan_to_num(sig_normed - np.mean(sig_normed, axis=1, keepdims=True), nan=0.0)

        ret_vals = ret_df.values
        bar_pnl = np.sum(sig_normed[:-1] * ret_vals[1:], axis=1)
        bar_pnl = np.concatenate([[0.0], bar_pnl])
        alpha_bar_pnl[aid] = bar_pnl

    # Rolling MAX weights
    weight_matrix = np.zeros((n_bars, n_active))
    current_weights = np.ones(n_active) / n_active

    for t in range(n_bars):
        if t % reweight_every == 0 and t >= lookback_bars:
            max_vals = np.zeros(n_active)
            for i, aid in enumerate(alpha_ids):
                pnl_window = alpha_bar_pnl[aid][t-lookback_bars:t]
                max_vals[i] = np.max(pnl_window) if len(pnl_window) > 0 else 0.0

            ranks = np.argsort(np.argsort(max_vals)).astype(float)
            mean_rank = np.mean(ranks)
            w_raw = ranks - mean_rank
            w_sum = np.sum(np.abs(w_raw))
            if w_sum > 0:
                current_weights = w_raw / w_sum
            else:
                current_weights = np.ones(n_active) / n_active

        weight_matrix[t] = current_weights

    # Combine
    composite = np.zeros((n_bars, len(tickers)))
    for i, aid in enumerate(alpha_ids):
        raw = alpha_signals[aid]
        sig = raw.reindex(index=dates, columns=tickers).fillna(0.0).values
        cs_mean = np.nanmean(sig, axis=1, keepdims=True)
        cs_std = np.nanstd(sig, axis=1, keepdims=True)
        cs_std = np.where(cs_std > 1e-10, cs_std, 1.0)
        normed = np.clip((sig - cs_mean) / cs_std, -5, 5)
        normed = np.nan_to_num(normed, nan=0.0)
        composite += weight_matrix[:, [i]] * normed

    return pd.DataFrame(composite, index=dates, columns=tickers)


def combiner_adaptive(alpha_signals, matrices, universe_df, returns_df, lookback=120):
    """Adaptive: weight by rolling expected factor return (positive ER only)."""
    close_df = matrices["close"]
    dates = close_df.index
    tickers = close_df.columns.tolist()

    # Per-alpha normalized signals and factor returns
    normed_signals = {}
    for aid, raw in alpha_signals.items():
        normed_signals[aid] = process_signal(raw, universe_df=universe_df, max_wt=MAX_WEIGHT)

    ret_df = returns_df.reindex(index=dates, columns=tickers)
    fr_data = {}
    for aid, norm in normed_signals.items():
        lagged = norm.shift(1)
        ab = lagged.abs().sum(axis=1).replace(0, np.nan)
        n = lagged.div(ab, axis=0)
        fr_data[aid] = (n * ret_df).sum(axis=1)

    fr_df = pd.DataFrame(fr_data)
    rolling_er = fr_df.rolling(window=lookback, min_periods=20).mean()
    weights = rolling_er.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)

    combined = pd.DataFrame(0.0, index=dates, columns=tickers)
    for aid, norm in normed_signals.items():
        w = weights_norm[aid].values if aid in weights_norm.columns else np.zeros(len(dates))
        combined = combined.add(norm.mul(pd.Series(w, index=dates), axis=0))

    return combined


def combiner_risk_parity(alpha_signals, matrices, universe_df, returns_df, lookback=120):
    """Risk Parity: weight inversely by rolling factor volatility."""
    close_df = matrices["close"]
    dates = close_df.index
    tickers = close_df.columns.tolist()

    normed_signals = {}
    for aid, raw in alpha_signals.items():
        normed_signals[aid] = process_signal(raw, universe_df=universe_df, max_wt=MAX_WEIGHT)

    ret_df = returns_df.reindex(index=dates, columns=tickers)
    fr_data = {}
    for aid, norm in normed_signals.items():
        lagged = norm.shift(1)
        ab = lagged.abs().sum(axis=1).replace(0, np.nan)
        n = lagged.div(ab, axis=0)
        fr_data[aid] = (n * ret_df).sum(axis=1)

    fr_df = pd.DataFrame(fr_data)
    rolling_vol = fr_df.rolling(window=lookback, min_periods=20).std()
    rolling_er = fr_df.rolling(window=lookback, min_periods=20).mean()
    inv_vol = (1.0 / rolling_vol.replace(0, np.nan)).fillna(0)
    inv_vol = inv_vol.where(rolling_er > 0, 0)
    wsum = inv_vol.sum(axis=1).replace(0, np.nan)
    weights_norm = inv_vol.div(wsum, axis=0).fillna(0)

    combined = pd.DataFrame(0.0, index=dates, columns=tickers)
    for aid, norm in normed_signals.items():
        w = weights_norm[aid].values if aid in weights_norm.columns else np.zeros(len(dates))
        combined = combined.add(norm.mul(pd.Series(w, index=dates), axis=0))

    return combined


# ============================================================================
# QP EXECUTION OPTIMIZER
# ============================================================================

def qp_optimize_signal(composite_df, matrices, universe_df, returns_df,
                       risk_aversion=0.5, tcost_lambda=0.0005, n_factors=3,
                       lookback_bars=120, rebal_every=6):
    """
    Convex optimization on aggregate signal.
    Maximize: alpha'w - 0.5*kappa*w'Sigma*w - lambda*||w - w_prev||_1
    Subject to: sum(w) = 0, sum(|w|) <= 1
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

    # Output
    opt_weights = np.zeros((n_bars, n_tickers))
    prev_w = np.zeros(n_tickers)

    rebal_count = 0
    for t in range(lookback_bars, n_bars):
        if t % rebal_every != 0:
            opt_weights[t] = prev_w
            continue

        # Get return window for risk model
        ret_window = ret_df[max(0, t - lookback_bars):t]
        if ret_window.shape[0] < 30:
            opt_weights[t] = prev_w
            continue

        # Alpha signal at bar t
        alpha_t = signal_vals[t]
        alpha_t = np.nan_to_num(alpha_t, nan=0.0)

        # Simple covariance risk model
        cov = np.cov(ret_window.T)
        if cov.shape[0] != n_tickers:
            opt_weights[t] = prev_w
            continue

        # Regularize
        cov = 0.5 * cov + 0.5 * np.diag(np.diag(cov)) + 1e-8 * np.eye(n_tickers)

        try:
            w = cp.Variable(n_tickers)
            obj = alpha_t @ w - risk_aversion * cp.quad_form(w, cp.psd_wrap(cov))

            # Transaction cost penalty
            if np.any(prev_w != 0):
                obj -= tcost_lambda * cp.norm1(w - prev_w)

            constraints = [
                cp.sum(w) == 0,           # dollar neutral
                cp.norm1(w) <= 2.0,       # gross exposure limit
                w >= -MAX_WEIGHT,
                w <= MAX_WEIGHT,
            ]

            prob = cp.Problem(cp.Maximize(obj), constraints)
            prob.solve(solver=cp.SCS, verbose=False, max_iters=2000)

            if prob.status in ["optimal", "optimal_inaccurate"]:
                opt_w = w.value
                opt_w = np.nan_to_num(opt_w, nan=0.0)
                prev_w = opt_w
                opt_weights[t] = opt_w
                rebal_count += 1
            else:
                opt_weights[t] = prev_w
        except Exception:
            opt_weights[t] = prev_w

    # Fill initial period
    opt_weights[:lookback_bars] = 0.0

    print(f"  QP optimizer: {rebal_count} rebalances", flush=True)
    return pd.DataFrame(opt_weights, index=dates, columns=tickers)


# ============================================================================
# MAIN EVALUATION LOOP
# ============================================================================

def main():
    t0_total = time.time()

    # Load data
    alphas = load_alphas()
    full_matrices, full_universe, valid_tickers = load_full_data()
    full_returns = full_matrices.get("returns")

    log(f"\n{'='*100}")
    log(f"  4H PORTFOLIO CONSTRUCTION EVALUATION")
    log(f"  Universe: {UNIVERSE} | Interval: {INTERVAL} | Alphas: {len(alphas)}")
    log(f"  Splits: train {SPLITS['train']}, val {SPLITS['val']}, test {SPLITS['test']}")
    log(f"  Fee levels: {FEE_LEVELS} bps")
    log(f"{'='*100}")

    if not alphas:
        log("  ERROR: No alphas found in DB for interval=4h, universe=BINANCE_TOP50!")
        log("  Run discovery first: python discover_alphas_4h.py")
        return

    # Load all alpha signals once
    log(f"\n  Loading alpha signals...")
    alpha_signals = load_alpha_signals(alphas, full_matrices)
    log(f"  Loaded {len(alpha_signals)}/{len(alphas)} alpha signals")

    if len(alpha_signals) < 2:
        print("  ERROR: Need at least 2 valid alpha signals!")
        return

    # Build all composites on full data
    COMBINERS = {
        "Equal Weight":        lambda: combiner_equal(alpha_signals, full_matrices, full_universe, full_returns),
        "Billion Alphas":      lambda: combiner_billion_alphas(alpha_signals, full_matrices, full_universe, full_returns),
        "Factor MAX (5d)":     lambda: combiner_factor_max(alpha_signals, full_matrices, full_universe, full_returns,
                                                           lookback_bars=30, reweight_every=6),
        "Factor MAX (10d)":    lambda: combiner_factor_max(alpha_signals, full_matrices, full_universe, full_returns,
                                                           lookback_bars=60, reweight_every=6),
        "Adaptive":            lambda: combiner_adaptive(alpha_signals, full_matrices, full_universe, full_returns,
                                                         lookback=120),
        "Risk Parity":         lambda: combiner_risk_parity(alpha_signals, full_matrices, full_universe, full_returns,
                                                            lookback=120),
    }

    composites = {}
    for cname, builder in COMBINERS.items():
        print(f"\n  Building: {cname}...", flush=True)
        t0 = time.time()
        comp = builder()
        if comp is not None:
            composites[cname] = comp
            print(f"    Done in {time.time()-t0:.1f}s", flush=True)
        else:
            print(f"    FAILED", flush=True)

    # Build QP-optimized versions
    qp_composites = {}
    for cname, comp in composites.items():
        qp_name = f"{cname} + QP"
        print(f"\n  Building QP: {qp_name}...", flush=True)
        t0 = time.time()
        qp_comp = qp_optimize_signal(comp, full_matrices, full_universe, full_returns)
        if qp_comp is not None:
            qp_composites[qp_name] = qp_comp
            print(f"    Done in {time.time()-t0:.1f}s", flush=True)

    all_composites = {**composites, **qp_composites}

    # Evaluate all composites x splits x fees
    print(f"\n  Evaluating {len(all_composites)} methods x {len(SPLITS)} splits x {len(FEE_LEVELS)} fees...")
    all_results = {}  # {(combiner, split, fee): sim_result}

    for cname, comp in all_composites.items():
        for split_name, (start, end) in SPLITS.items():
            comp_slice = comp.loc[start:end]
            split_matrices = {name: df.loc[start:end] for name, df in full_matrices.items()}
            split_universe = full_universe.loc[start:end]
            close = split_matrices.get("close")
            returns = split_matrices.get("returns")

            # Process signal through standard pipeline
            if "+ QP" not in cname:
                alpha_processed = process_signal(comp_slice, universe_df=split_universe, max_wt=MAX_WEIGHT)
            else:
                # QP already produces weights, just clip
                alpha_processed = comp_slice.clip(lower=-MAX_WEIGHT, upper=MAX_WEIGHT)

            for fee in FEE_LEVELS:
                try:
                    sim = simulate(alpha_processed, returns, close, split_universe, fees_bps=fee)
                    all_results[(cname, split_name, fee)] = sim
                except Exception as e:
                    print(f"    ERROR: {cname}/{split_name}/{fee}bps: {e}")

    # ====================================================================
    # OUTPUT -- DISPLAY ALL DATA
    # ====================================================================

    log(f"\n\n{'='*120}")
    log(f"  FULL RESULTS TABLE -- ALL METHODS x SPLITS x FEES")
    log(f"  {len(alphas)} alphas, {UNIVERSE}, {INTERVAL}")
    log(f"{'='*120}")

    for fee in FEE_LEVELS:
        log(f"\n  ---- {fee:.0f} BPS FEES ----")
        header = f"  {'Method':<30s} | {'Train SR':>9} {'Train Ret%':>10} {'Train TO':>9} | {'Val SR':>8} {'Val Ret%':>9} {'Val TO':>8} | {'Test SR':>8} {'Test Ret%':>9} {'Test TO':>8}"
        log(header)
        log(f"  {'-'*len(header)}")

        for cname in all_composites:
            train_r = all_results.get((cname, "train", fee))
            val_r = all_results.get((cname, "val", fee))
            test_r = all_results.get((cname, "test", fee))

            def fmt(r):
                if r is None:
                    return "  N/A", "  N/A", "  N/A"
                ret_pct = r.total_pnl / BOOKSIZE * 100
                return f"{r.sharpe:+8.3f}", f"{ret_pct:+9.2f}%", f"{r.turnover:8.4f}"

            tr_sr, tr_ret, tr_to = fmt(train_r)
            v_sr, v_ret, v_to = fmt(val_r)
            te_sr, te_ret, te_to = fmt(test_r)

            log(f"  {cname:<30s} | {tr_sr} {tr_ret} {tr_to} | {v_sr} {v_ret} {v_to} | {te_sr} {te_ret} {te_to}")

    # Detailed breakdown per combiner
    log(f"\n\n{'='*120}")
    log(f"  DETAILED PER-COMBINER BREAKDOWN")
    log(f"{'='*120}")

    for cname in all_composites:
        log(f"\n  -- {cname} --")
        log(f"  {'Fee':>5} | {'Split':<6} | {'Sharpe':>8} {'Ret%':>9} {'RetAnn%':>10} {'MaxDD%':>8} {'TO':>8} {'Fitness':>8}")
        log(f"  {'-'*75}")
        for fee in FEE_LEVELS:
            for split_name in ["train", "val", "test"]:
                r = all_results.get((cname, split_name, fee))
                if r is None:
                    log(f"  {fee:5.0f} | {split_name:<6} | {'N/A':>8}")
                    continue
                ret_pct = r.total_pnl / BOOKSIZE * 100
                log(f"  {fee:5.0f} | {split_name:<6} | {r.sharpe:+8.3f} {ret_pct:+8.2f}% "
                      f"{r.returns_ann*100:+9.2f}% {r.max_drawdown*100:7.2f}% "
                      f"{r.turnover:7.4f} {r.fitness:7.2f}")

    # ====================================================================
    # CHARTS
    # ====================================================================

    print(f"\n  Generating charts...", flush=True)

    # Color palette
    base_colors = {
        "Equal Weight":     "#2196F3",
        "Billion Alphas":   "#4CAF50",
        "Factor MAX (5d)":  "#E91E63",
        "Factor MAX (10d)": "#9C27B0",
        "Adaptive":         "#FF9800",
        "Risk Parity":      "#00BCD4",
    }

    # Chart 1: Cumulative PnL per fee level
    for fee in FEE_LEVELS:
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))

        # Split backgrounds
        offset = 0
        split_colors = {"train": "#2196F3", "val": "#FF9800", "test": "#F44336"}
        split_offsets = {}

        for split_name in ["train", "val", "test"]:
            ref_key = None
            for cname in composites:  # raw only for reference
                ref_key = (cname, split_name, fee)
                if ref_key in all_results:
                    break
            if ref_key and ref_key in all_results:
                n = len(all_results[ref_key].daily_pnl)
                ax.axvspan(offset, offset + n, alpha=0.06, color=split_colors[split_name])
                if split_name != "train":
                    ax.axvline(x=offset, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
                split_offsets[split_name] = (offset, n)
                offset += n

        # Plot raw combiners only (to avoid clutter)
        for cname in composites:
            running_offset = 0.0
            bar_off = 0
            all_x, all_y = [], []

            for split_name in ["train", "val", "test"]:
                r = all_results.get((cname, split_name, fee))
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
                color = base_colors.get(cname, "#999999")
                test_r = all_results.get((cname, "test", fee))
                test_sr = test_r.sharpe if test_r else 0
                ax.plot(all_x, all_y, color=color, linewidth=2.0,
                        label=f"{cname} (test SR={test_sr:+.2f})")

        ax.set_title(f"4H Portfolio Combiners -- {len(alphas)} alphas, {UNIVERSE}, {fee:.0f} bps",
                     fontsize=14, fontweight="bold")
        ax.set_ylabel("Cumulative Return (%)", fontsize=12)
        ax.set_xlabel("Bar index (4h bars)", fontsize=11)
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(True, alpha=0.25)
        ax.axhline(0, color="black", linewidth=0.5)

        chart_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  f"4h_portfolio_{fee:.0f}bps.png")
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Chart saved: {chart_path}")

    # Chart 2: Test Sharpe bar chart across fees
    fig, axes = plt.subplots(1, len(FEE_LEVELS), figsize=(24, 8), sharey=True)

    raw_names = list(composites.keys())
    n_c = len(raw_names)

    for fi, fee in enumerate(FEE_LEVELS):
        ax = axes[fi]
        x = np.arange(n_c)

        # Raw bars
        raw_sharpes = []
        for cname in raw_names:
            r = all_results.get((cname, "test", fee))
            raw_sharpes.append(r.sharpe if r else 0)

        # QP bars
        qp_sharpes = []
        for cname in raw_names:
            qp_name = f"{cname} + QP"
            r = all_results.get((qp_name, "test", fee))
            qp_sharpes.append(r.sharpe if r else 0)

        bars1 = ax.bar(x - 0.2, raw_sharpes, 0.35, label="Raw", color="#2196F3", alpha=0.8, edgecolor="white")
        bars2 = ax.bar(x + 0.2, qp_sharpes, 0.35, label="+ QP", color="#FF5722", alpha=0.8, edgecolor="white")

        for j, v in enumerate(raw_sharpes):
            ax.text(x[j] - 0.2, v + (0.1 if v >= 0 else -0.3),
                    f"{v:+.2f}", ha="center", fontsize=7, fontweight="bold")
        for j, v in enumerate(qp_sharpes):
            ax.text(x[j] + 0.2, v + (0.1 if v >= 0 else -0.3),
                    f"{v:+.2f}", ha="center", fontsize=7, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" ", "\n") for n in raw_names], fontsize=8)
        ax.set_title(f"{fee:.0f} bps", fontsize=12, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.25, axis="y")
        if fi == 0:
            ax.set_ylabel("Test Sharpe Ratio", fontsize=11)
            ax.legend(fontsize=9)

    plt.suptitle(f"TEST Sharpe: Raw vs QP-Optimized -- {len(alphas)} 4h alphas, {UNIVERSE}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    chart_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "4h_portfolio_test_sharpe_comparison.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: {chart_path}")

    elapsed = time.time() - t0_total
    log(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    log(f"{'='*100}")

    # --- Write Report File ---
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "4h_portfolio_last_run.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 4H Alpha Portfolio Evaluation Report\n")
        f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 📜 Execution Log & Tables\n```text\n")
        f.write("\n".join(REPORT_BUFFER))
        f.write("\n```\n\n")
        f.write("## 🖼️ Generated Charts\n")
        for fee in FEE_LEVELS:
            f.write(f"- [4h_portfolio_{fee:.0f}bps.png](4h_portfolio_{fee:.0f}bps.png)\n")
        f.write("- [4h_portfolio_test_sharpe_comparison.png](4h_portfolio_test_sharpe_comparison.png)\n")
    
    print(f"\n  Report saved to: {report_path}")


if __name__ == "__main__":
    main()
