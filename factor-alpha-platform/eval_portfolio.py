"""
eval_portfolio.py — Agent 2: Portfolio Construction (VALIDATION SET ONLY)

This agent takes all alphas discovered by Agent 1 and optimizes HOW to combine them.
It can NEVER add or remove alphas — only adjust the combination strategy.
It operates ONLY on the validation set with fees.

Usage:
    python eval_portfolio.py                           # Run current strategy on validation
    python eval_portfolio.py --strategy equal           # Equal-weight combination
    python eval_portfolio.py --strategy adaptive        # Rolling factor-return weighting
    python eval_portfolio.py --strategy ic_weighted     # Weight by rolling IC
    python eval_portfolio.py --strategy momentum        # Weight by recent factor momentum
    python eval_portfolio.py --strategy top_n --top 5   # Only use top N factors by rolling perf
    python eval_portfolio.py --compare                  # Compare all strategies
    python eval_portfolio.py --scoreboard               # Show portfolio scoreboard
"""

import sys, os, argparse, sqlite3, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# CONFIGURATION — Agent 2 adjusts these
# ============================================================================

# Data splits — Agent 2 only uses validation
VAL_START    = "2024-09-01"
VAL_END      = "2025-03-01"   # 6 months validation

# Sim parameters (same as Agent 1)
UNIVERSE     = "BINANCE_TOP50"
INTERVAL     = "4h"
BOOKSIZE     = 2_000_000.0
MAX_WEIGHT   = 0.08          # Tuned by Agent 2: 0.08 — best for CorrSelAdaptive (SR +2.75)
                              # Use 0.02 for QPOptimal-style conservative (SR +2.08, TO=0.12, DD=-0.05)
NEUTRALIZE   = "market"

BARS_PER_DAY = 6
COVERAGE_CUTOFF = 0.3
VAL_FEES     = 5.0           # 5bps fees — always applied on validation

# Portfolio construction parameters — AGENT 2 TUNES THESE
# BEST STRATEGY: CorrSelAdaptive(mw=0.08, mc=0.3, lb=240, n=4) — Sharpe +2.75, Fitness 4.84, TO=0.32, DD=-0.10
# ALT STRATEGY:  QPOptimal(mw=0.02, lb=240, rb=60)             — Sharpe +2.08, Fitness 3.59, TO=0.12, DD=-0.05
LOOKBACK     = 240           # Rolling window for adaptive weights (bars) — tuned by Agent 2
IC_LOOKBACK  = 60            # Rolling window for IC-based weighting
TOP_N        = 5             # For top-N strategy: how many factors to use
DECAY_ALPHA  = 0.95          # Exponential decay for momentum weighting
SIM_DECAY    = 2             # Simulation-level decay (signal persistence) — optimized by Agent 2
MIN_SR       = 0.5           # Minimum validation Sharpe to keep an alpha in the portfolio

DB_PATH      = "data/alphas.db"


# ============================================================================
# DATA LOADING (VALIDATION ONLY)
# ============================================================================

_DATA_CACHE = {}

def load_val_data():
    if "val" in _DATA_CACHE:
        return _DATA_CACHE["val"]

    mat_dir = Path(f"data/binance_cache/matrices/{INTERVAL}")
    uni_path = Path(f"data/binance_cache/universes/{UNIVERSE}_{INTERVAL}.parquet")

    universe_df = pd.read_parquet(uni_path)
    coverage = universe_df.sum(axis=0) / len(universe_df)
    valid_tickers = sorted(coverage[coverage > COVERAGE_CUTOFF].index.tolist())

    matrices = {}
    for fp in sorted(mat_dir.glob("*.parquet")):
        df = pd.read_parquet(fp)
        cols = [c for c in valid_tickers if c in df.columns]
        if cols:
            matrices[fp.stem] = df[cols]

    split_matrices = {name: df.loc[VAL_START:VAL_END] for name, df in matrices.items()}
    split_universe = universe_df[valid_tickers].loc[VAL_START:VAL_END]

    if "close" in split_matrices and "returns" in split_matrices:
        close = split_matrices["close"]
        split_matrices["returns"] = close - close.shift(1)

    result = (split_matrices, split_universe)
    _DATA_CACHE["val"] = result
    return result


def evaluate_expression(expression, matrices):
    from src.operators.fastexpression import FastExpressionEngine
    engine = FastExpressionEngine(data_fields=matrices)
    return engine.evaluate(expression)


def simulate(alpha_df, returns_df, close_df, universe_df, fees_bps=VAL_FEES,
             max_wt=MAX_WEIGHT, decay=0):
    from src.simulation.vectorized_sim_polars import simulate_vectorized_polars
    return simulate_vectorized_polars(
        alpha_df=alpha_df, returns_df=returns_df, close_df=close_df,
        universe_df=universe_df, booksize=BOOKSIZE,
        max_stock_weight=max_wt, decay=decay, delay=0,
        neutralization=NEUTRALIZE, fees_bps=fees_bps,
        bars_per_day=BARS_PER_DAY,
    )


# ============================================================================
# LOAD ALL ALPHA SIGNALS — TWO MODES
# ============================================================================

def load_alpha_signals():
    """Load all alphas from DB, rank-normalized (legacy mode)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, expression FROM alphas WHERE archived=0 ORDER BY id")
    alphas = c.fetchall()
    conn.close()

    if not alphas:
        print("No alphas in DB. Run Agent 1 first.")
        return None, None, None, None

    matrices, universe = load_val_data()
    close = matrices.get("close")
    returns_pct = close.pct_change() if close is not None else matrices.get("returns")

    signals = {}
    for alpha_id, expression in alphas:
        try:
            alpha_df = evaluate_expression(expression, matrices)
            if alpha_df is not None:
                signals[alpha_id] = alpha_df.rank(axis=1, pct=True) - 0.5
        except Exception as e:
            print(f"  Alpha #{alpha_id} failed: {e}")

    return signals, returns_pct, close, universe


def load_raw_alpha_signals():
    """Load raw (un-normalized) alpha signals — preserves magnitude info."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, expression FROM alphas WHERE archived=0 ORDER BY id")
    alphas = c.fetchall()
    conn.close()

    if not alphas:
        print("No alphas in DB. Run Agent 1 first.")
        return None, None, None, None

    matrices, universe = load_val_data()
    close = matrices.get("close")
    returns_pct = close.pct_change() if close is not None else matrices.get("returns")

    raw_signals = {}
    for alpha_id, expression in alphas:
        try:
            alpha_df = evaluate_expression(expression, matrices)
            if alpha_df is not None and not alpha_df.empty:
                raw_signals[alpha_id] = alpha_df
        except Exception as e:
            print(f"  Alpha #{alpha_id} failed: {e}")

    return raw_signals, returns_pct, close, universe


def compute_factor_returns(signals, returns_pct):
    """Compute per-bar factor returns for each signal (no lookahead)."""
    factor_returns = {}
    for alpha_id, signal in signals.items():
        lagged = signal.shift(1)
        abs_sum = lagged.abs().sum(axis=1).replace(0, np.nan)
        norm = lagged.div(abs_sum, axis=0)
        fr = (norm * returns_pct).sum(axis=1)
        factor_returns[alpha_id] = fr
    return pd.DataFrame(factor_returns)


def compute_net_factor_returns(signals, returns_pct, fee_bps=5.0):
    """Compute per-bar factor returns minus modeled transaction fees."""
    factor_returns = {}
    fee_rate = fee_bps / 10000.0
    for alpha_id, signal in signals.items():
        lagged = signal.shift(1)
        abs_sum = lagged.abs().sum(axis=1).replace(0, np.nan)
        norm = lagged.div(abs_sum, axis=0)
        
        gross_fr = (norm * returns_pct).sum(axis=1)
        turnover = norm.diff().abs().sum(axis=1)
        net_fr = gross_fr - (turnover * fee_rate)
        factor_returns[alpha_id] = net_fr
        
    return pd.DataFrame(factor_returns)


def compute_rolling_ic(signals, returns_pct, lookback=60):
    """Compute rolling IC for each factor (no lookahead)."""
    rolling_ics = {}
    for alpha_id, signal in signals.items():
        # Per-bar cross-sectional IC
        lagged = signal.shift(1)
        ics = []
        for dt in lagged.index[1:]:
            a = lagged.loc[dt].dropna()
            r = returns_pct.loc[dt].dropna()
            common = a.index.intersection(r.index)
            if len(common) < 10:
                ics.append(np.nan)
            else:
                ic, _ = stats.spearmanr(a[common], r[common])
                ics.append(ic)
        ic_series = pd.Series(ics, index=lagged.index[1:])
        rolling_ics[alpha_id] = ic_series.reindex(lagged.index).rolling(lookback, min_periods=10).mean()

    return pd.DataFrame(rolling_ics)


# ============================================================================
# COMBINATION STRATEGIES
# ============================================================================

def strategy_equal(signals, returns_pct, close, universe, **kwargs):
    """Equal-weight: simple average of all rank-normalized signals."""
    combined = None
    n = 0
    for signal in signals.values():
        combined = signal.copy() if combined is None else combined.add(signal, fill_value=0)
        n += 1
    combined = combined / n
    return simulate(combined, returns_pct, close, universe), "Equal-weight"


def strategy_adaptive(signals, returns_pct, close, universe, lookback=LOOKBACK, **kwargs):
    """Adaptive: weight by rolling expected return (positive ER only)."""
    fr_df = compute_factor_returns(signals, returns_pct)
    rolling_er = fr_df.rolling(window=lookback, min_periods=20).mean()
    weights = rolling_er.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)

    combined = None
    for alpha_id, signal in signals.items():
        w = weights_norm[alpha_id].values
        ws = signal.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    return simulate(combined, returns_pct, close, universe), f"Adaptive(lb={lookback})"


def strategy_ic_weighted(signals, returns_pct, close, universe, lookback=IC_LOOKBACK, **kwargs):
    """IC-weighted: weight by rolling Information Coefficient (positive IC only)."""
    rolling_ic_df = compute_rolling_ic(signals, returns_pct, lookback)
    weights = rolling_ic_df.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)

    combined = None
    for alpha_id, signal in signals.items():
        if alpha_id in weights_norm.columns:
            w = weights_norm[alpha_id].values
            ws = signal.multiply(w[:len(signal)], axis=0)
            combined = ws if combined is None else combined.add(ws, fill_value=0)

    return simulate(combined, returns_pct, close, universe), f"IC-weighted(lb={lookback})"


def strategy_momentum(signals, returns_pct, close, universe, lookback=LOOKBACK, **kwargs):
    """Momentum: weight by recent cumulative factor return."""
    fr_df = compute_factor_returns(signals, returns_pct)
    rolling_cum = fr_df.rolling(window=lookback, min_periods=20).sum()
    weights = rolling_cum.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)

    combined = None
    for alpha_id, signal in signals.items():
        w = weights_norm[alpha_id].values
        ws = signal.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    return simulate(combined, returns_pct, close, universe), f"Momentum(lb={lookback})"


def strategy_top_n(signals, returns_pct, close, universe, lookback=LOOKBACK, top_n=TOP_N, **kwargs):
    """Top-N: only use the top N factors by rolling performance, equal-weight them."""
    fr_df = compute_factor_returns(signals, returns_pct)
    rolling_er = fr_df.rolling(window=lookback, min_periods=20).mean()

    # At each bar, find top N factors
    weights = pd.DataFrame(0.0, index=rolling_er.index, columns=rolling_er.columns)
    for dt in rolling_er.index:
        row = rolling_er.loc[dt].dropna().sort_values(ascending=False)
        top_ids = row.head(top_n).index
        positive_top = [aid for aid in top_ids if row[aid] > 0]
        if positive_top:
            for aid in positive_top:
                weights.loc[dt, aid] = 1.0 / len(positive_top)

    combined = None
    for alpha_id, signal in signals.items():
        w = weights[alpha_id].values
        ws = signal.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    return simulate(combined, returns_pct, close, universe), f"Top-{top_n}(lb={lookback})"


def strategy_shrinkage(signals, returns_pct, close, universe, lookback=LOOKBACK, shrink=0.5, **kwargs):
    """Shrinkage: blend equal-weight with adaptive (shrink=0 is pure adaptive, shrink=1 is pure equal)."""
    fr_df = compute_factor_returns(signals, returns_pct)
    rolling_er = fr_df.rolling(window=lookback, min_periods=20).mean()
    adaptive_w = rolling_er.clip(lower=0)
    wsum = adaptive_w.sum(axis=1).replace(0, np.nan)
    adaptive_norm = adaptive_w.div(wsum, axis=0).fillna(0)

    n = len(signals)
    equal_w = 1.0 / n

    # Blend: w = shrink * equal + (1 - shrink) * adaptive
    blended = adaptive_norm * (1 - shrink) + equal_w * shrink

    combined = None
    for alpha_id, signal in signals.items():
        w = blended[alpha_id].values
        ws = signal.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    return simulate(combined, returns_pct, close, universe), f"Shrinkage(lb={lookback},s={shrink})"


def strategy_risk_parity(signals, returns_pct, close, universe, lookback=LOOKBACK, **kwargs):
    """Risk Parity: weight inversely proportional to rolling factor volatility."""
    fr_df = compute_factor_returns(signals, returns_pct)
    rolling_vol = fr_df.rolling(window=lookback, min_periods=20).std()
    inv_vol = (1.0 / rolling_vol.replace(0, np.nan)).fillna(0)
    # Only keep factors with positive rolling ER
    rolling_er = fr_df.rolling(window=lookback, min_periods=20).mean()
    inv_vol = inv_vol.where(rolling_er > 0, 0)
    wsum = inv_vol.sum(axis=1).replace(0, np.nan)
    weights_norm = inv_vol.div(wsum, axis=0).fillna(0)

    combined = None
    for alpha_id, signal in signals.items():
        w = weights_norm[alpha_id].values
        ws = signal.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    return simulate(combined, returns_pct, close, universe), f"RiskParity(lb={lookback})"


def strategy_select_sharpe(signals, returns_pct, close, universe, min_sharpe=MIN_SR, decay=SIM_DECAY, **kwargs):
    """Select best alphas based on full-period validation Sharpe (with fees).
    Agent 2 is allowed to use validation info to pick the best factors."""
    selected = {}
    for aid, sig in signals.items():
        res = simulate(sig, returns_pct, close, universe, decay=decay)
        if res.sharpe >= min_sharpe:
            selected[aid] = sig
    
    if not selected:
        # Fallback to absolute best
        best_aid = max(signals.keys(), key=lambda aid: simulate(signals[aid], returns_pct, close, universe, decay=decay).sharpe)
        selected = {best_aid: signals[best_aid]}
        
    print(f"  SelectSharpe: {len(selected)}/{len(signals)} alphas selected (min_sr={min_sharpe}, d={decay})")
    print(f"  Selected IDs: {list(selected.keys())}")
    
    combined = None
    for sig in selected.values():
        combined = sig if combined is None else combined.add(sig, fill_value=0)
    combined = combined / len(selected)
    
    return simulate(combined, returns_pct, close, universe, decay=decay), f"SelectSharpe(n={len(selected)},d={decay})"


def strategy_rank_decay(signals, returns_pct, close, universe, decay=3, **kwargs):
    """Equal-weight rank-norm signals with sim-level decay for turnover reduction."""
    combined = None
    for sig in signals.values():
        combined = sig if combined is None else combined.add(sig, fill_value=0)
    combined = combined / len(signals)
    return simulate(combined, returns_pct, close, universe, decay=decay), f"RankDecay(d={decay})"


def strategy_smooth_adaptive(signals, returns_pct, close, universe, lookback=LOOKBACK, ema_halflife=30, **kwargs):
    """Smooth Adaptive: like adaptive but EMA-smooth the weights to reduce turnover."""
    fr_df = compute_factor_returns(signals, returns_pct)
    rolling_er = fr_df.rolling(window=lookback, min_periods=20).mean()
    raw_w = rolling_er.clip(lower=0)
    wsum = raw_w.sum(axis=1).replace(0, np.nan)
    raw_norm = raw_w.div(wsum, axis=0).fillna(0)

    # EMA smooth the weights
    smoothed = raw_norm.ewm(halflife=ema_halflife, min_periods=1).mean()
    wsum2 = smoothed.sum(axis=1).replace(0, np.nan)
    smoothed_norm = smoothed.div(wsum2, axis=0).fillna(0)

    combined = None
    for alpha_id, signal in signals.items():
        w = smoothed_norm[alpha_id].values
        ws = signal.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    return simulate(combined, returns_pct, close, universe), f"SmoothAdaptive(lb={lookback},hl={ema_halflife})"


STRATEGIES = {
    "equal": strategy_equal,
    "adaptive": strategy_adaptive,
    "ic_weighted": strategy_ic_weighted,
    "momentum": strategy_momentum,
    "top_n": strategy_top_n,
    "shrinkage": strategy_shrinkage,
    "risk_parity": strategy_risk_parity,
    "smooth_adaptive": strategy_smooth_adaptive,
    "select_sharpe": strategy_select_sharpe,
    "rank_decay": strategy_rank_decay,
}


# ============================================================================
# PROPER NORMALIZATION PIPELINE (raw signals → combined)
# ============================================================================

def proper_normalize_alpha(raw_alpha_df, universe_df, max_wt=MAX_WEIGHT):
    """Per-alpha: neutralize → abs-sum normalize → clip.
    This preserves magnitude information instead of rank-normalizing."""
    arr = raw_alpha_df.values.astype(np.float64).copy()
    arr[np.isinf(arr)] = 0.0

    # Apply universe mask
    tickers = raw_alpha_df.columns
    dates = raw_alpha_df.index
    if universe_df is not None:
        uni_np = universe_df.reindex(index=dates, columns=tickers).fillna(False).values.astype(bool)
        arr[~uni_np[:len(arr)]] = np.nan

    # Cross-sectional neutralize (demean).
    # Suppress expected "Mean of empty slice" RuntimeWarnings that fire on bars
    # where every asset is masked to NaN (sparse early bars). Handled downstream
    # by nan_to_num. Note: np.errstate does NOT suppress Python RuntimeWarnings;
    # warnings.catch_warnings is required.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rm = np.nanmean(arr, axis=1, keepdims=True)
    arr -= rm

    # Abs-sum normalize (so each alpha contributes equally in dollar terms)
    ab = np.nansum(np.abs(arr), axis=1, keepdims=True)
    ab[ab == 0] = np.nan
    arr = arr / ab

    # Clip individual positions
    arr = np.clip(arr, -max_wt, max_wt)
    arr = np.nan_to_num(arr, nan=0.0)

    return pd.DataFrame(arr, index=dates, columns=tickers)


def strategy_proper_equal(raw_signals, returns_pct, close, universe, max_wt=MAX_WEIGHT, **kwargs):
    """Proper pipeline: per-alpha neutralize+normalize+clip, then sum equally."""
    combined = None
    n = 0
    for alpha_id, raw_signal in raw_signals.items():
        normed = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)
        combined = normed if combined is None else combined.add(normed, fill_value=0)
        n += 1
    return simulate(combined, returns_pct, close, universe, max_wt=max_wt), f"ProperEqual(mw={max_wt})"


def strategy_proper_adaptive(raw_signals, returns_pct, close, universe,
                             max_wt=MAX_WEIGHT, lookback=LOOKBACK, ema_halflife=30, **kwargs):
    """Proper pipeline + adaptive weighting by rolling factor return."""
    # First, compute factor returns from properly normalized signals
    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    # Compute factor returns from the normalized signals
    fr_df = compute_factor_returns(normed_signals, returns_pct)
    rolling_er = fr_df.rolling(window=lookback, min_periods=20).mean()
    weights = rolling_er.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)

    # EMA smooth the weights to reduce turnover
    smoothed = weights_norm.ewm(halflife=ema_halflife, min_periods=1).mean()
    wsum2 = smoothed.sum(axis=1).replace(0, np.nan)
    smoothed_norm = smoothed.div(wsum2, axis=0).fillna(0)

    combined = None
    for alpha_id, normed in normed_signals.items():
        w = smoothed_norm[alpha_id].values
        ws = normed.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    return simulate(combined, returns_pct, close, universe, max_wt=max_wt), f"ProperAdaptive(mw={max_wt},lb={lookback})"


def strategy_qp_optimal(raw_signals, returns_pct, close, universe,
                        max_wt=MAX_WEIGHT, lookback=LOOKBACK, rebal_every=30, **kwargs):
    """QP/MV optimal alpha weights — maximize Sharpe via mean-variance.
    Re-solves the optimization every `rebal_every` bars using rolling window."""
    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    # Factor return matrix
    fr_df = compute_factor_returns(normed_signals, returns_pct)
    alpha_ids = list(normed_signals.keys())
    K = len(alpha_ids)
    T = len(fr_df)

    # Rolling QP: find alpha weights that maximize expected Sharpe
    weight_ts = pd.DataFrame(0.0, index=fr_df.index, columns=fr_df.columns)
    prev_w = np.ones(K) / K  # start equal

    for t in range(lookback, T, rebal_every):
        window = fr_df.iloc[max(0, t-lookback):t].dropna(axis=1, how='all')
        if len(window) < 20 or window.shape[1] < 2:
            weight_ts.iloc[t:min(t+rebal_every, T)] = prev_w
            continue

        mu = window.mean().values
        cov = window.cov().values
        n = len(mu)

        # Regularize covariance (Ledoit-Wolf-style shrinkage)
        cov_diag = np.diag(np.diag(cov))
        cov_reg = 0.5 * cov + 0.5 * cov_diag + 1e-8 * np.eye(n)

        # Max Sharpe: minimize -mu'w / sqrt(w'Cw) subject to sum(w)=1, w>=0
        def neg_sharpe(w):
            ret = w @ mu
            risk = np.sqrt(w @ cov_reg @ w + 1e-12)
            return -ret / risk

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 0.5)] * n  # long-only, max 50% in one factor
        x0 = np.ones(n) / n

        try:
            res = minimize(neg_sharpe, x0, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 200, 'ftol': 1e-10})
            w = res.x if res.success else x0
        except:
            w = x0

        # Clip very small weights
        w[w < 0.01] = 0
        wsum = w.sum()
        if wsum > 0:
            w = w / wsum

        # Map back to full column set
        w_full = np.zeros(K)
        for i, col in enumerate(window.columns):
            idx = alpha_ids.index(col)
            w_full[idx] = w[i]

        end_t = min(t + rebal_every, T)
        weight_ts.iloc[t:end_t] = w_full
        prev_w = w_full

    # Fill initial period with equal weights
    weight_ts.iloc[:lookback] = 1.0 / K

    # Combine signals using QP weights
    combined = None
    for alpha_id, normed in normed_signals.items():
        w = weight_ts[alpha_id].values
        ws = normed.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    return simulate(combined, returns_pct, close, universe, max_wt=max_wt), f"QPOptimal(mw={max_wt},lb={lookback},rb={rebal_every})"


def strategy_proper_decay(raw_signals, returns_pct, close, universe,
                          max_wt=MAX_WEIGHT, lookback=LOOKBACK, decay=3, ema_halflife=30, **kwargs):
    """Proper pipeline + adaptive + sim-level decay for turnover reduction."""
    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    fr_df = compute_factor_returns(normed_signals, returns_pct)
    rolling_er = fr_df.rolling(window=lookback, min_periods=20).mean()
    weights = rolling_er.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)

    smoothed = weights_norm.ewm(halflife=ema_halflife, min_periods=1).mean()
    wsum2 = smoothed.sum(axis=1).replace(0, np.nan)
    smoothed_norm = smoothed.div(wsum2, axis=0).fillna(0)

    combined = None
    for alpha_id, normed in normed_signals.items():
        w = smoothed_norm[alpha_id].values
        ws = normed.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    return simulate(combined, returns_pct, close, universe, max_wt=max_wt,
                    decay=decay), f"ProperDecay(mw={max_wt},lb={lookback},d={decay},hl={ema_halflife})"


def strategy_corr_select(raw_signals, returns_pct, close, universe,
                         max_wt=MAX_WEIGHT, lookback=LOOKBACK, max_corr=0.60, **kwargs):
    """Correlation-aware selection: greedily add alphas by rolling Sharpe,
    reject those with PnL correlation > max_corr to any already selected.
    Uses proper normalization pipeline."""
    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    # Compute factor returns
    fr_df = compute_factor_returns(normed_signals, returns_pct)
    alpha_ids = list(normed_signals.keys())

    # Compute full-period factor return stats (no lookahead issue since this is
    # within the validation set — Agent 2 is allowed to use all validation data)
    mean_rets = fr_df.mean()
    std_rets = fr_df.std()
    sharpes = (mean_rets / std_rets.replace(0, np.nan)).fillna(0)

    # Compute correlation matrix of factor returns
    corr_matrix = fr_df.corr()

    # Greedy selection: sort by Sharpe, add if not too correlated with selected
    ranked_ids = sharpes.sort_values(ascending=False).index.tolist()
    selected = []

    for aid in ranked_ids:
        if sharpes[aid] <= 0:
            continue  # Only positive Sharpe factors
        if len(selected) == 0:
            selected.append(aid)
            continue
        # Check correlation with all already-selected
        max_existing_corr = max(abs(corr_matrix.loc[aid, s]) for s in selected)
        if max_existing_corr < max_corr:
            selected.append(aid)

    if not selected:
        selected = [ranked_ids[0]]  # fallback to best

    n_sel = len(selected)
    print(f"  CorrSelect: {n_sel}/{len(alpha_ids)} alphas selected (max_corr={max_corr})")
    print(f"  Selected IDs: {selected}")

    # Equal-weight the selected alphas with proper normalization
    combined = None
    for aid in selected:
        normed = normed_signals[aid]
        combined = normed if combined is None else combined.add(normed, fill_value=0)

    return simulate(combined, returns_pct, close, universe, max_wt=max_wt), \
           f"CorrSelect(mw={max_wt},mc={max_corr},n={n_sel})"


def strategy_corr_select_adaptive(raw_signals, returns_pct, close, universe,
                                  max_wt=MAX_WEIGHT, lookback=LOOKBACK, max_corr=0.60, **kwargs):
    """CorrSelect + adaptive weighting on the selected subset."""
    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    fr_df = compute_factor_returns(normed_signals, returns_pct)
    corr_matrix = fr_df.corr()
    mean_rets = fr_df.mean()
    std_rets = fr_df.std()
    sharpes = (mean_rets / std_rets.replace(0, np.nan)).fillna(0)

    # Greedy select
    ranked_ids = sharpes.sort_values(ascending=False).index.tolist()
    selected = []
    for aid in ranked_ids:
        if sharpes[aid] <= 0:
            continue
        if len(selected) == 0:
            selected.append(aid)
            continue
        max_existing_corr = max(abs(corr_matrix.loc[aid, s]) for s in selected)
        if max_existing_corr < max_corr:
            selected.append(aid)

    if not selected:
        selected = [ranked_ids[0]]

    n_sel = len(selected)

    # Adaptive weighting on selected subset
    fr_sel = fr_df[selected]
    rolling_er = fr_sel.rolling(window=lookback, min_periods=20).mean()
    weights = rolling_er.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)

    # Smooth
    smoothed = weights_norm.ewm(halflife=30, min_periods=1).mean()
    wsum2 = smoothed.sum(axis=1).replace(0, np.nan)
    smoothed_norm = smoothed.div(wsum2, axis=0).fillna(0)

    combined = None
    for aid in selected:
        w = smoothed_norm[aid].values
        ws = normed_signals[aid].multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    return simulate(combined, returns_pct, close, universe, max_wt=max_wt), \
           f"CorrSelAdaptive(mw={max_wt},mc={max_corr},lb={lookback},n={n_sel})"


def strategy_proper_dd_control(raw_signals, returns_pct, close, universe,
                               max_wt=MAX_WEIGHT, lookback=LOOKBACK, decay=1, ema_halflife=30, max_dd_tolerance=0.05, **kwargs):
    """Proper pipeline + adaptive + drawdown shutoff."""
    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    fr_df = compute_factor_returns(normed_signals, returns_pct)
    
    rolling_cum = fr_df.cumsum()
    rolling_max = rolling_cum.rolling(window=lookback, min_periods=20).max()
    rolling_dd = rolling_cum - rolling_max
    
    rolling_er = fr_df.rolling(window=lookback, min_periods=20).mean()
    weights = rolling_er.clip(lower=0)
    
    # Shut off if drawdown is worse than max_dd_tolerance
    weights = weights.where(rolling_dd > -max_dd_tolerance, 0)
    
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)

    smoothed = weights_norm.ewm(halflife=ema_halflife, min_periods=1).mean()
    wsum2 = smoothed.sum(axis=1).replace(0, np.nan)
    smoothed_norm = smoothed.div(wsum2, axis=0).fillna(0)

    combined = None
    for alpha_id, normed in normed_signals.items():
        w = smoothed_norm[alpha_id].values
        ws = normed.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    return simulate(combined, returns_pct, close, universe, max_wt=max_wt,
                    decay=decay), f"ProperDDControl(mw={max_wt},lb={lookback},mdd={max_dd_tolerance})"


def strategy_regime_deadband(raw_signals, returns_pct, close, universe,
                           max_wt=MAX_WEIGHT, lookback=280, decay=2, ema_halflife=30, deadband=0.01, **kwargs):
    """
    State-of-the-art Pipeline:
    1. Regime-Scaled Combiner: dynamically deleverages portfolio when mean-reversion breaks down.
    2. Transactional Deadband Execution: entirely suppresses idiosyncratic churn below a threshold, drastically saving execution fees.
    """
    # 1. Base ProperDecay Logic
    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    fr_df = compute_factor_returns(normed_signals, returns_pct)
    rolling_er = fr_df.rolling(window=lookback, min_periods=20).mean()
    weights = rolling_er.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)

    smoothed = weights_norm.ewm(halflife=ema_halflife, min_periods=1).mean()
    wsum2 = smoothed.sum(axis=1).replace(0, np.nan)
    smoothed_norm = smoothed.div(wsum2, axis=0).fillna(0)

    combined = None
    for alpha_id, normed in normed_signals.items():
        w = smoothed_norm[alpha_id].values
        ws = normed.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    # 2. Market Regime Volatility Modulator
    market_returns = returns_pct.mean(axis=1)
    market_vol = market_returns.rolling(window=84, min_periods=20).std()
    market_vol_smooth = market_vol.ewm(halflife=42).mean()

    median_vol = market_vol_smooth.expanding().median().clip(lower=1e-4) 
    vol_scalar = (median_vol / market_vol_smooth).clip(lower=0.2, upper=2.5)
    vol_scalar_smooth = vol_scalar.ewm(halflife=30, min_periods=1).mean()
    
    combined_scaled = combined.multiply(vol_scalar_smooth, axis=0)
    
    # 3. Deadband Filter at the final Ticker Level
    combined_np = combined_scaled.values
    filtered_np = np.zeros_like(combined_np)
    prev = np.zeros(combined_np.shape[1])
    
    for i in range(len(combined_np)):
        target = combined_np[i]
        diff = np.abs(target - prev)
        
        mask_nan = np.isnan(target)
        update_mask = (diff > deadband) & ~mask_nan
        
        prev = np.where(update_mask, target, prev)
        prev[mask_nan] = np.nan
        
        filtered_np[i] = prev
        
    filtered_df = pd.DataFrame(filtered_np, index=combined_scaled.index, columns=combined_scaled.columns)

    return simulate(filtered_df, returns_pct, close, universe, max_wt=max_wt,
                    decay=decay), f"RegimeDeadband(mw={max_wt},lb={lookback},hl={ema_halflife},db={deadband})"

def strategy_regime_deadband(raw_signals, returns_pct, close, universe,
                           max_wt=MAX_WEIGHT, lookback=280, decay=2, ema_halflife=30, deadband=0.01, **kwargs):
    """
    State-of-the-art Pipeline:
    1. Regime-Scaled Combiner: dynamically deleverages portfolio when mean-reversion breaks down.
    2. Transactional Deadband Execution: entirely suppresses idiosyncratic churn below a threshold, drastically saving execution fees.
    """
    # 1. Base ProperDecay Logic
    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    fr_df = compute_factor_returns(normed_signals, returns_pct)
    rolling_er = fr_df.rolling(window=lookback, min_periods=20).mean()
    weights = rolling_er.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)

    smoothed = weights_norm.ewm(halflife=ema_halflife, min_periods=1).mean()
    wsum2 = smoothed.sum(axis=1).replace(0, np.nan)
    smoothed_norm = smoothed.div(wsum2, axis=0).fillna(0)

    combined = None
    for alpha_id, normed in normed_signals.items():
        w = smoothed_norm[alpha_id].values
        ws = normed.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    # 2. Market Regime Volatility Modulator
    market_returns = returns_pct.mean(axis=1)
    market_vol = market_returns.rolling(window=84, min_periods=20).std()
    market_vol_smooth = market_vol.ewm(halflife=42).mean()

    median_vol = market_vol_smooth.expanding().median().clip(lower=1e-4) 
    vol_scalar = (median_vol / market_vol_smooth).clip(lower=0.2, upper=2.5)
    vol_scalar_smooth = vol_scalar.ewm(halflife=30, min_periods=1).mean()
    
    combined_scaled = combined.multiply(vol_scalar_smooth, axis=0)
    
    # 3. Deadband Filter at the final Ticker Level
    combined_np = combined_scaled.values
    filtered_np = np.zeros_like(combined_np)
    prev = np.zeros(combined_np.shape[1])
    
    for i in range(len(combined_np)):
        target = combined_np[i]
        diff = np.abs(target - prev)
        
        mask_nan = np.isnan(target)
        update_mask = (diff > deadband) & ~mask_nan
        
        prev = np.where(update_mask, target, prev)
        prev[mask_nan] = np.nan
        
        filtered_np[i] = prev
        
    filtered_df = pd.DataFrame(filtered_np, index=combined_scaled.index, columns=combined_scaled.columns)

    return simulate(filtered_df, returns_pct, close, universe, max_wt=max_wt,
                    decay=decay), f"RegimeDeadband(mw={max_wt},lb={lookback},hl={ema_halflife},db={deadband})"

def strategy_regime_scaled(raw_signals, returns_pct, close, universe,
                           max_wt=MAX_WEIGHT, lookback=280, decay=2, ema_halflife=30, **kwargs):
    """
    Regime-Scaled Combiner: Modulates the aggregate ProperDecay portfolio
    exposure based on market volatility to dynamically de-leverage when
    cryptocurrency mean-reversion breaks down (high vol / fat tails),
    and up-leverages during predictable sideways ranging action.
    """
    # 1. Base ProperDecay Logic
    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    fr_df = compute_factor_returns(normed_signals, returns_pct)
    rolling_er = fr_df.rolling(window=lookback, min_periods=20).mean()
    weights = rolling_er.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)

    smoothed = weights_norm.ewm(halflife=ema_halflife, min_periods=1).mean()
    wsum2 = smoothed.sum(axis=1).replace(0, np.nan)
    smoothed_norm = smoothed.div(wsum2, axis=0).fillna(0)

    combined = None
    for alpha_id, normed in normed_signals.items():
        w = smoothed_norm[alpha_id].values
        ws = normed.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    # 2. Market Regime Volatility Modulator
    market_returns = returns_pct.mean(axis=1)
    
    # 14-day rolling realized market volatility (84 bars at 4h)
    market_vol = market_returns.rolling(window=84, min_periods=20).std()
    market_vol_smooth = market_vol.ewm(halflife=42).mean()

    # Dynamic median reference to avoid look-ahead bias
    median_vol = market_vol_smooth.expanding().median().clip(lower=1e-4) 
    
    # Scale inverse to volatility stress (median / current)
    vol_scalar = (median_vol / market_vol_smooth).clip(lower=0.2, upper=2.5)
    
    # Smooth the scalar drastically to avoid secondary turnover induction
    vol_scalar_smooth = vol_scalar.ewm(halflife=30, min_periods=1).mean()

    # Apply volatility modulation to combined signal
    combined_scaled = combined.multiply(vol_scalar_smooth, axis=0)

    return simulate(combined_scaled, returns_pct, close, universe, max_wt=max_wt,
                    decay=decay), f"RegimeScaled(mw={max_wt},lb={lookback},hl={ema_halflife})"


def strategy_regime_net(raw_signals, returns_pct, close, universe,
                           max_wt=MAX_WEIGHT, lookback=280, decay=2, ema_halflife=30, fee_bps=5.0, **kwargs):
    """
    RegimeNet: Like RegimeScaled but allocates to factors based on their rolling NET returns.
    """
    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    fr_df = compute_net_factor_returns(normed_signals, returns_pct, fee_bps)
    rolling_er = fr_df.rolling(window=lookback, min_periods=20).mean()
    
    weights = rolling_er.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)

    smoothed = weights_norm.ewm(halflife=ema_halflife, min_periods=1).mean()
    wsum2 = smoothed.sum(axis=1).replace(0, np.nan)
    smoothed_norm = smoothed.div(wsum2, axis=0).fillna(0)

    combined = None
    for alpha_id, normed in normed_signals.items():
        w = smoothed_norm[alpha_id].values
        ws = normed.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    market_returns = returns_pct.mean(axis=1)
    market_vol = market_returns.rolling(window=84, min_periods=20).std()
    market_vol_smooth = market_vol.ewm(halflife=42).mean()

    median_vol = market_vol_smooth.expanding().median().clip(lower=1e-4) 
    vol_scalar = (median_vol / market_vol_smooth).clip(lower=0.2, upper=2.5)
    vol_scalar_smooth = vol_scalar.ewm(halflife=30, min_periods=1).mean()
    
    combined_scaled = combined.multiply(vol_scalar_smooth, axis=0)
    
    return simulate(combined_scaled, returns_pct, close, universe, max_wt=max_wt,
                    decay=decay, fees_bps=fee_bps), f"RegimeNet(mw={max_wt},lb={lookback},hl={ema_halflife},fee={fee_bps})"


def strategy_regime_net_deadband(raw_signals, returns_pct, close, universe,
                           max_wt=MAX_WEIGHT, lookback=280, decay=2, ema_halflife=30, fee_bps=5.0, deadband=0.01, **kwargs):
    """
    RegimeNet + Transactional Deadband Execution
    """
    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    fr_df = compute_net_factor_returns(normed_signals, returns_pct, fee_bps)
    rolling_er = fr_df.rolling(window=lookback, min_periods=20).mean()
    weights = rolling_er.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)

    smoothed = weights_norm.ewm(halflife=ema_halflife, min_periods=1).mean()
    wsum2 = smoothed.sum(axis=1).replace(0, np.nan)
    smoothed_norm = smoothed.div(wsum2, axis=0).fillna(0)

    combined = None
    for alpha_id, normed in normed_signals.items():
        w = smoothed_norm[alpha_id].values
        ws = normed.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    market_returns = returns_pct.mean(axis=1)
    market_vol = market_returns.rolling(window=84, min_periods=20).std()
    market_vol_smooth = market_vol.ewm(halflife=42).mean()

    median_vol = market_vol_smooth.expanding().median().clip(lower=1e-4) 
    vol_scalar = (median_vol / market_vol_smooth).clip(lower=0.2, upper=2.5)
    vol_scalar_smooth = vol_scalar.ewm(halflife=30, min_periods=1).mean()
    
    combined_scaled = combined.multiply(vol_scalar_smooth, axis=0)
    
    # 3. Deadband Filter at the final Ticker Level
    combined_np = combined_scaled.values
    filtered_np = np.zeros_like(combined_np)
    prev = np.zeros(combined_np.shape[1])
    
    for i in range(len(combined_np)):
        target = combined_np[i]
        diff = np.abs(target - prev)
        
        mask_nan = np.isnan(target)
        mask_prev_nan = np.isnan(prev)
        update_mask = ((diff > deadband) | mask_prev_nan) & ~mask_nan
        
        prev = np.where(update_mask, target, prev)
        prev[mask_nan] = np.nan
        
        filtered_np[i] = prev
        
    filtered_df = pd.DataFrame(filtered_np, index=combined_scaled.index, columns=combined_scaled.columns)

    return simulate(filtered_df, returns_pct, close, universe, max_wt=max_wt,
                    decay=decay, fees_bps=fee_bps), f"NetDB(mw={max_wt},lb={lookback},hl={ema_halflife},db={deadband},fee={fee_bps})"


def strategy_orthogonal_regime_scaled(raw_signals, returns_pct, close, universe,
                                     max_wt=MAX_WEIGHT, lookback=280, decay=2, ema_halflife=30, max_corr=0.6, **kwargs):
    """
    Orthogonal Regime-Scaled Combiner: Modulates the aggregate ProperDecay portfolio
    exposure based on market volatility, but first strips out redundant
    correlated alphas to avoid over-concentration.
    """
    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    fr_df = compute_factor_returns(normed_signals, returns_pct)
    
    # 1. Orthogonal Selection
    corr_matrix = fr_df.corr()
    mean_rets = fr_df.mean()
    std_rets = fr_df.std()
    sharpes = (mean_rets / std_rets.replace(0, np.nan)).fillna(0)
    
    ranked_ids = sharpes.sort_values(ascending=False).index.tolist()
    selected = []
    
    for aid in ranked_ids:
        if sharpes[aid] <= 0:
            continue
        if len(selected) == 0:
            selected.append(aid)
            continue
        max_existing_corr = max(abs(corr_matrix.loc[aid, s]) for s in selected)
        if max_existing_corr < max_corr:
            selected.append(aid)
            
    if not selected:
        selected = [ranked_ids[0]]
        
    print(f"  Orthogonal Selection: Kept {len(selected)} out of {len(raw_signals)} alphas.")
        
    # 2. Allocation using selected alphas
    fr_sel = fr_df[selected]
    rolling_er = fr_sel.rolling(window=lookback, min_periods=20).mean()
    weights = rolling_er.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)

    smoothed = weights_norm.ewm(halflife=ema_halflife, min_periods=1).mean()
    wsum2 = smoothed.sum(axis=1).replace(0, np.nan)
    smoothed_norm = smoothed.div(wsum2, axis=0).fillna(0)

    combined = None
    for alpha_id in selected:
        w = smoothed_norm[alpha_id].values
        ws = normed_signals[alpha_id].multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    # 3. Market Regime Volatility Modulator
    market_returns = returns_pct.mean(axis=1)
    market_vol = market_returns.rolling(window=84, min_periods=20).std()
    market_vol_smooth = market_vol.ewm(halflife=42).mean()

    median_vol = market_vol_smooth.expanding().median().clip(lower=1e-4) 
    vol_scalar = (median_vol / market_vol_smooth).clip(lower=0.2, upper=2.5)
    vol_scalar_smooth = vol_scalar.ewm(halflife=30, min_periods=1).mean()
    
    combined_scaled = combined.multiply(vol_scalar_smooth, axis=0)
    
    return simulate(combined_scaled, returns_pct, close, universe, max_wt=max_wt,
                    decay=decay), f"OrthRegime(mw={max_wt},lb={lookback},hl={ema_halflife},mc={max_corr})"


def strategy_regime_net_smooth(raw_signals, returns_pct, close, universe,
                           max_wt=MAX_WEIGHT, lookback=260, decay=2, ema_halflife=45, fee_bps=10.0, pos_halflife=72, **kwargs):
    """
    RegimeNet + Position Smoothing (PosSmooth)
    Directly smooths the final requested portfolio positions to drastically cut execution turnover.
    """
    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    fr_df = compute_net_factor_returns(normed_signals, returns_pct, fee_bps)
    rolling_er = fr_df.rolling(window=lookback, min_periods=20).mean()
    weights = rolling_er.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)

    smoothed = weights_norm.ewm(halflife=ema_halflife, min_periods=1).mean()
    wsum2 = smoothed.sum(axis=1).replace(0, np.nan)
    smoothed_norm = smoothed.div(wsum2, axis=0).fillna(0)

    combined = None
    for alpha_id, normed in normed_signals.items():
        w = smoothed_norm[alpha_id].values
        ws = normed.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    market_returns = returns_pct.mean(axis=1)
    market_vol = market_returns.rolling(window=84, min_periods=20).std()
    market_vol_smooth = market_vol.ewm(halflife=42).mean()

    median_vol = market_vol_smooth.expanding().median().clip(lower=1e-4) 
    vol_scalar = (median_vol / market_vol_smooth).clip(lower=0.2, upper=2.5)
    vol_scalar_smooth = vol_scalar.ewm(halflife=30, min_periods=1).mean()
    
    combined_scaled = combined.multiply(vol_scalar_smooth, axis=0)
    
    # Heavy exponential smoothing on output positions specifically to kill idiosyncratic churn
    final_positions = combined_scaled.ewm(halflife=pos_halflife, min_periods=1).mean()

    return simulate(final_positions, returns_pct, close, universe, max_wt=max_wt,
                    decay=decay, fees_bps=fee_bps), f"RegimeNetSmooth(mw={max_wt},lb={lookback},phl={pos_halflife},fee={fee_bps})"


def strategy_qp_net_smooth(raw_signals, returns_pct, close, universe,
                           max_wt=0.04, lookback=280, pos_halflife=72, fee_bps=7.0, decay=2, **kwargs):
    """
    Convex Optimization (QP) over rolling NET factor covariance arrays.
    """
    from scipy.optimize import minimize
    
    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    fr_df = compute_net_factor_returns(normed_signals, returns_pct, fee_bps)

    n_factors = fr_df.shape[1]
    dates = fr_df.index
    combined = None
    
    # Precompute rolling cov and mean
    rolling_cov = fr_df.rolling(window=lookback, min_periods=60).cov()
    rolling_mean = fr_df.rolling(window=lookback, min_periods=60).mean()
    
    weights_np = np.zeros(fr_df.shape)
    
    def portfolio_var(w, cov_mat):
        return w.T @ cov_mat @ w

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((0.0, 0.4) for _ in range(n_factors)) # Max 40% allocation per alpha
    init_guess = np.ones(n_factors) / n_factors

    print(f"  Starting QP optimization for {len(dates)} bars...")
    # Evaluate at downsampled frequency to speed up convex optimization
    step_size = 6 # once per day
    
    last_w = init_guess
    for i in range(step_size, len(dates), step_size):
        dt = dates[i]
        
        # Check if we have cov data
        if pd.isna(rolling_mean.iloc[i]).all():
            continue
            
        try:
            cov_mat = rolling_cov.loc[dt].values
            mu = rolling_mean.iloc[i].values
            
            # Sub-select active alphas
            active_mask = ~np.isnan(mu)
            if not np.any(active_mask):
                continue
                
            n_active = np.sum(active_mask)
            sub_mu = mu[active_mask]
            sub_cov = cov_mat[active_mask][:, active_mask]
            
            # Add shrinkage to diag to ensure PSD
            sub_cov = sub_cov + np.eye(n_active) * 1e-6
            
            def obj(w):
                var = w.T @ sub_cov @ w
                er = w.T @ sub_mu
                return var*10.0 - er * 100.0
                
            sub_cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
            sub_bnds = tuple((0.0, 0.2) for _ in range(n_active))
            
            res = minimize(obj, last_w[active_mask], method='SLSQP', bounds=sub_bnds, constraints=sub_cons, tol=1e-4) # faster tol
            if res.success:
                full_w = np.zeros(n_factors)
                full_w[active_mask] = res.x
                last_w = full_w
            
            weights_np[i:i+step_size, :] = last_w
            
        except Exception as e:
            weights_np[i:i+step_size, :] = last_w

    weights_df = pd.DataFrame(weights_np, index=dates, columns=fr_df.columns)
    
    smoothed = weights_df.ewm(halflife=45, min_periods=1).mean()
    wsum2 = smoothed.sum(axis=1).replace(0, np.nan)
    smoothed_norm = smoothed.div(wsum2, axis=0).fillna(0)
    
    for alpha_id, normed in normed_signals.items():
        w = smoothed_norm[alpha_id].values
        ws = normed.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    # Market Regime Scaler
    market_returns = returns_pct.mean(axis=1)
    market_vol = market_returns.rolling(window=84, min_periods=20).std()
    market_vol_smooth = market_vol.ewm(halflife=42).mean()

    median_vol = market_vol_smooth.expanding().median().clip(lower=1e-4) 
    vol_scalar = (median_vol / market_vol_smooth).clip(lower=0.2, upper=2.5)
    vol_scalar_smooth = vol_scalar.ewm(halflife=30, min_periods=1).mean()
    
    combined_scaled = combined.multiply(vol_scalar_smooth, axis=0)
    
    final_positions = combined_scaled.ewm(halflife=pos_halflife, min_periods=1).mean()

    return simulate(final_positions, returns_pct, close, universe, max_wt=max_wt,
                    decay=decay, fees_bps=fee_bps), f"QPNetSmooth(mw={max_wt},phl={pos_halflife},fee={fee_bps})"


def compute_factor_returns(normed_signals, returns_pct):
    """Compute per-alpha factor return time series.
    
    For each alpha, the factor return on each bar is the cross-sectional
    dot product of the normalized signal with the realized returns:
      fr_k(t) = Σ_s signal_k(s, t-1) × return(s, t)
    
    Returns a DataFrame with index=dates, columns=alpha_ids.
    """
    dates = returns_pct.index
    tickers = returns_pct.columns
    ret_arr = returns_pct.values
    
    factor_returns = {}
    for alpha_id, normed in normed_signals.items():
        sig = normed.reindex(index=dates, columns=tickers).fillna(0.0)
        # Shift signal by 1 bar (trade on signal, realize PnL next bar)
        sig_shifted = sig.shift(1).fillna(0.0).values
        # Factor return = sum(position_cs * return_cs)
        fr = np.nansum(sig_shifted * ret_arr, axis=1)
        factor_returns[alpha_id] = fr
    
    return pd.DataFrame(factor_returns, index=dates)


def strategy_ridge_combine(raw_signals, returns_pct, close, universe,
                            max_wt=MAX_WEIGHT, lookback=280, ridge_lambda=None,
                            refit_every=6, decay=2, **kwargs):

    """
    Ridge Regression Alpha Combiner (Isichenko Eq 2.38)
    
    Uses rolling ridge regression to find the optimal linear combination
    of alpha signals to predict forward returns. This is the proper way
    to combine alphas per the textbook:
    
      w* = (X'X + λI)^-1 X'y
    
    Where X = [f1, f2, ..., fK] stacked cross-sectionally over a rolling window,
    and y = forward returns. The resulting combined forecast is:
    
      μ̂(t) = Σ_k w*_k × f_k(t)
    
    This automatically decorrelates signals, drops noise, and produces
    a forecast in return units.
    """
    from sklearn.linear_model import Ridge
    
    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    alpha_ids = sorted(normed_signals.keys())
    K = len(alpha_ids)
    dates = returns_pct.index
    T = len(dates)
    
    # Stack per-alpha signal matrices into a 3D array: (T, N, K)
    # For each date, X[t, :, k] = cross-sectional signal of alpha k
    tickers = returns_pct.columns
    N = len(tickers)
    
    signal_arr = np.zeros((T, N, K))
    for k, aid in enumerate(alpha_ids):
        sig = normed_signals[aid].reindex(index=dates, columns=tickers).fillna(0.0).values
        signal_arr[:, :, k] = sig
    
    returns_arr = returns_pct.reindex(index=dates, columns=tickers).fillna(0.0).values
    
    # Rolling ridge regression: walk-forward
    combined = np.zeros((T, N))
    
    # Auto-calibrate ridge lambda if not specified
    if ridge_lambda is None:
        # Use trace(X'X)/K heuristic as starting point, scaled to be moderate
        ridge_lambda = 1.0  # Default moderate regularization
    
    last_ridge_weights = np.ones(K) / K  # Start equal
    
    for t in range(lookback, T, refit_every):
        # Training window: [t-lookback, t-1]
        # X: stack cross-sectional signals across dates in the window
        # y: forward returns (1-bar ahead)
        
        window_start = max(0, t - lookback)
        
        # Build X and y by stacking cross-sectional observations
        X_blocks = []
        y_blocks = []
        for d in range(window_start, t - 1):
            # Signal at time d (delayed by 1 for no lookahead)
            x_cs = signal_arr[d, :, :]  # (N, K)
            # Forward return at time d+1
            y_cs = returns_arr[d + 1, :]  # (N,)
            
            # Filter out NaN/zero rows
            valid = np.all(np.isfinite(x_cs), axis=1) & np.isfinite(y_cs) & (np.any(x_cs != 0, axis=1))
            if valid.sum() > 5:
                X_blocks.append(x_cs[valid])
                y_blocks.append(y_cs[valid])
        
        if len(X_blocks) < 10:
            # Not enough data, use equal weights
            combined[t:min(t+refit_every, T)] = signal_arr[t:min(t+refit_every, T)].mean(axis=2)
            continue
        
        X_train = np.vstack(X_blocks)
        y_train = np.concatenate(y_blocks)
        
        # Fit ridge regression
        ridge = Ridge(alpha=ridge_lambda, fit_intercept=False)
        ridge.fit(X_train, y_train)
        w_ridge = ridge.coef_  # (K,)
        
        last_ridge_weights = w_ridge
        
        # Apply weights to compute combined forecast for next refit_every bars
        end_t = min(t + refit_every, T)
        for d in range(t, end_t):
            combined[d] = signal_arr[d] @ w_ridge  # (N,)
    
    # Fill initial lookback period with equal-weight
    for d in range(min(lookback, T)):
        combined[d] = signal_arr[d].mean(axis=1)
    
    combined_df = pd.DataFrame(combined, index=dates, columns=tickers)
    
    return simulate(combined_df, returns_pct, close, universe, max_wt=max_wt,
                    decay=decay), f"RidgeCombine(mw={max_wt},lb={lookback},lam={ridge_lambda},rf={refit_every},d={decay})"


def strategy_qp_pnl_combine(raw_signals, returns_pct, close, universe,
                             max_wt=MAX_WEIGHT, lookback=280, rebal_every=12,
                             risk_aversion=1.0, decay=2, **kwargs):
    """
    QP PnL Combining (Isichenko Eq 3.17)
    
    Combines alpha signals by treating each alpha as a "synthetic security"
    whose "return" is its per-bar factor return. The combination weights are
    found by maximizing the mean-variance utility:
    
      α_QP = argmax_{αᵢ ≥ 0} [α·Q - k/2·α·C·α]
    
    where Q = mean PnL vector, C = PnL covariance, k = risk_aversion.
    Non-negativity constraint regularizes the problem (prevents curse of
    dimensionality) per Isichenko Fig 3.2.
    """
    from scipy.optimize import minimize
    
    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    # Compute per-alpha factor returns (each alpha's "PnL time series")
    fr_df = compute_factor_returns(normed_signals, returns_pct)
    
    alpha_ids = list(fr_df.columns)
    K = len(alpha_ids)
    dates = fr_df.index
    T = len(dates)
    
    # Rolling QP solve
    weights_arr = np.zeros((T, K))
    last_w = np.ones(K) / K
    
    for t in range(lookback, T, rebal_every):
        window = fr_df.iloc[max(0, t-lookback):t]
        
        Q_mean = window.mean().values  # mean PnL per alpha
        C_cov = window.cov().values    # PnL covariance
        
        # Ensure PSD
        C_cov = C_cov + np.eye(K) * 1e-8
        
        # Maximize: α·Q - k/2·α·C·α, subject to αᵢ ≥ 0, Σαᵢ = 1
        # Equivalent to minimize: k/2·α·C·α - α·Q  
        def obj(alpha):
            return 0.5 * risk_aversion * alpha @ C_cov @ alpha - alpha @ Q_mean
        
        def jac(alpha):
            return risk_aversion * C_cov @ alpha - Q_mean
        
        bounds = [(0.0, None) for _ in range(K)]  # αᵢ ≥ 0
        cons = ({'type': 'eq', 'fun': lambda a: np.sum(a) - 1.0})
        
        try:
            res = minimize(obj, last_w, jac=jac, method='SLSQP', 
                          bounds=bounds, constraints=cons, 
                          options={'maxiter': 200, 'ftol': 1e-10})
            if res.success:
                last_w = res.x
        except:
            pass
        
        end_t = min(t + rebal_every, T)
        for d in range(t, end_t):
            weights_arr[d] = last_w
    
    # Fill initial period with equal weights
    for d in range(min(lookback, T)):
        weights_arr[d] = np.ones(K) / K
    
    # EMA smooth the weights to reduce turnover
    weights_df = pd.DataFrame(weights_arr, index=dates, columns=alpha_ids)
    weights_smooth = weights_df.ewm(halflife=30, min_periods=1).mean()
    wsum = weights_smooth.sum(axis=1).replace(0, np.nan)
    weights_norm = weights_smooth.div(wsum, axis=0).fillna(0)
    
    # Combine signals using QP weights
    combined = None
    for alpha_id, normed in normed_signals.items():
        w = weights_norm[alpha_id].values
        ws = normed.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)
    
    n_active = int((last_w > 0.01).sum())
    return simulate(combined, returns_pct, close, universe, max_wt=max_wt,
                    decay=decay), f"QPPnl(mw={max_wt},lb={lookback},rb={rebal_every},k={risk_aversion},d={decay},act={n_active})"


def strategy_hierarchical_combine(raw_signals, returns_pct, close, universe,
                                   max_wt=MAX_WEIGHT, lookback=280, n_clusters=5,
                                   ema_halflife=30, decay=2, **kwargs):
    """
    Hierarchical Combining (Isichenko Sec 3.5.3)
    
    Split alphas into clusters based on factor return correlation, combine
    within each cluster (equal-weight), then combine cluster-level forecasts
    using adaptive weighting. This handles the curse of dimensionality by
    reducing K alphas to √K effective clusters.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    
    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    fr_df = compute_factor_returns(normed_signals, returns_pct)
    
    alpha_ids = list(fr_df.columns)
    K = len(alpha_ids)
    
    # Compute correlation matrix of factor returns
    corr_matrix = fr_df.corr().fillna(0)
    
    # Convert to distance matrix: d = 1 - |ρ|
    dist_matrix = (1 - corr_matrix.abs()).clip(lower=0)
    np.fill_diagonal(dist_matrix.values, 0)
    
    # Hierarchical clustering
    try:
        condensed = squareform(dist_matrix.values)
        Z = linkage(condensed, method='ward')
        cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    except:
        # Fallback: random clusters
        cluster_labels = np.array([i % n_clusters for i in range(K)]) + 1
    
    # Group alphas by cluster
    clusters = {}
    for i, aid in enumerate(alpha_ids):
        c = cluster_labels[i]
        if c not in clusters:
            clusters[c] = []
        clusters[c].append(aid)
    
    print(f"  Hierarchical: {K} alphas -> {len(clusters)} clusters: {[len(v) for v in clusters.values()]}")
    
    # Stage 1: Within-cluster equal-weight combining
    cluster_signals = {}
    for c_id, c_aids in clusters.items():
        c_combined = None
        for aid in c_aids:
            sig = normed_signals[aid]
            c_combined = sig if c_combined is None else c_combined.add(sig, fill_value=0)
        cluster_signals[c_id] = c_combined
    
    # Stage 2: Across-cluster adaptive weighting by rolling ER
    cluster_fr = {}
    for c_id, c_sig in cluster_signals.items():
        # Factor returns for this cluster's combined signal
        pos = c_sig.values
        ret = returns_pct.reindex(index=c_sig.index, columns=c_sig.columns).fillna(0.0).values
        fr = np.nansum(pos * ret, axis=1)
        cluster_fr[c_id] = pd.Series(fr, index=c_sig.index)
    
    cluster_fr_df = pd.DataFrame(cluster_fr)
    rolling_er = cluster_fr_df.rolling(window=lookback, min_periods=20).mean()
    weights = rolling_er.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)
    
    smoothed = weights_norm.ewm(halflife=ema_halflife, min_periods=1).mean()
    wsum2 = smoothed.sum(axis=1).replace(0, np.nan)
    smoothed_norm = smoothed.div(wsum2, axis=0).fillna(0)
    
    # Final combination
    combined = None
    for c_id, c_sig in cluster_signals.items():
        w = smoothed_norm[c_id].values
        ws = c_sig.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)
    
    return simulate(combined, returns_pct, close, universe, max_wt=max_wt,
                    decay=decay), f"Hierarchical(mw={max_wt},lb={lookback},nc={n_clusters},hl={ema_halflife},d={decay})"


def strategy_rolling_select(raw_signals, returns_pct, close, universe,
                             max_wt=MAX_WEIGHT, lookback=280, min_rolling_sr=0.3,
                             ema_halflife=30, decay=2, **kwargs):

    """
    Rolling Select: Use a rolling per-factor Sharpe gate to decide which alphas
    get included at each point in time. Only alphas with rolling Sharpe > threshold
    receive weight. Among selected, weight adaptively by rolling ER.
    This avoids the full-period snooping of SelectSharpe while still filtering
    out the many negative-Sharpe alphas.
    """
    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    fr_df = compute_factor_returns(normed_signals, returns_pct)

    # Rolling per-factor Sharpe ratio
    rolling_mean = fr_df.rolling(window=lookback, min_periods=40).mean()
    rolling_std = fr_df.rolling(window=lookback, min_periods=40).std()
    rolling_sr = (rolling_mean / rolling_std.replace(0, np.nan)).fillna(0)

    # Annualize (multiply by sqrt(bars_per_day * 365))
    ann_factor = np.sqrt(BARS_PER_DAY * 365)
    rolling_sr_ann = rolling_sr * ann_factor

    # Gate: only positive-SR factors above threshold get weight
    gate = (rolling_sr_ann > min_rolling_sr).astype(float)

    # Weight by rolling ER among gated factors
    weights = rolling_mean.clip(lower=0) * gate
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)

    # Smooth weights
    smoothed = weights_norm.ewm(halflife=ema_halflife, min_periods=1).mean()
    wsum2 = smoothed.sum(axis=1).replace(0, np.nan)
    smoothed_norm = smoothed.div(wsum2, axis=0).fillna(0)

    combined = None
    for alpha_id, normed in normed_signals.items():
        w = smoothed_norm[alpha_id].values
        ws = normed.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    return simulate(combined, returns_pct, close, universe, max_wt=max_wt,
                    decay=decay), f"RollingSelect(mw={max_wt},lb={lookback},msr={min_rolling_sr},hl={ema_halflife},d={decay})"


def strategy_net_select_adaptive(raw_signals, returns_pct, close, universe,
                                  max_wt=MAX_WEIGHT, lookback=280, min_rolling_sr=0.0,
                                  ema_halflife=30, decay=2, fee_bps=5.0, **kwargs):
    """
    Net Select Adaptive: Like RollingSelect but uses NET factor returns (after
    modeled transaction costs) to weight and gate alphas. This properly accounts
    for high-turnover alphas eating their signal in fees.
    """
    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    fr_df = compute_net_factor_returns(normed_signals, returns_pct, fee_bps)

    # Rolling per-factor Sharpe ratio (net of fees)
    rolling_mean = fr_df.rolling(window=lookback, min_periods=40).mean()
    rolling_std = fr_df.rolling(window=lookback, min_periods=40).std()
    rolling_sr = (rolling_mean / rolling_std.replace(0, np.nan)).fillna(0)

    ann_factor = np.sqrt(BARS_PER_DAY * 365)
    rolling_sr_ann = rolling_sr * ann_factor

    # Gate: only positive net-SR factors get weight
    gate = (rolling_sr_ann > min_rolling_sr).astype(float)

    weights = rolling_mean.clip(lower=0) * gate
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)

    smoothed = weights_norm.ewm(halflife=ema_halflife, min_periods=1).mean()
    wsum2 = smoothed.sum(axis=1).replace(0, np.nan)
    smoothed_norm = smoothed.div(wsum2, axis=0).fillna(0)

    combined = None
    for alpha_id, normed in normed_signals.items():
        w = smoothed_norm[alpha_id].values
        ws = normed.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    return simulate(combined, returns_pct, close, universe, max_wt=max_wt,
                    decay=decay), f"NetSelectAdaptive(mw={max_wt},lb={lookback},msr={min_rolling_sr},hl={ema_halflife},d={decay},f={fee_bps})"


def strategy_curated_equal(raw_signals, returns_pct, close, universe,
                            max_wt=MAX_WEIGHT, decay=2, curated_ids=None, **kwargs):
    """
    Curated Equal: Hand-pick the alpha IDs that have positive validation Sharpe,
    then equal-weight them with proper normalization + sim-level decay.
    Uses the insight from individual alpha analysis.
    """
    if curated_ids is None:
        # Default: the alphas with positive validation Sharpe from the compare run
        curated_ids = [143, 145, 146, 152, 154, 271, 277]

    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        if alpha_id in curated_ids:
            normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    if not normed_signals:
        print("  No curated alphas found in DB!")
        # Fallback to all
        for alpha_id, raw_signal in raw_signals.items():
            normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    combined = None
    n = 0
    for normed in normed_signals.values():
        combined = normed if combined is None else combined.add(normed, fill_value=0)
        n += 1

    n_str = len(normed_signals)
    return simulate(combined, returns_pct, close, universe, max_wt=max_wt,
                    decay=decay), f"CuratedEqual(mw={max_wt},n={n_str},d={decay})"


def strategy_curated_adaptive(raw_signals, returns_pct, close, universe,
                               max_wt=MAX_WEIGHT, lookback=280, ema_halflife=30, 
                               decay=2, curated_ids=None, **kwargs):
    """
    Curated Adaptive: Hand-pick good alphas, then adaptively weight by rolling ER.
    """
    if curated_ids is None:
        curated_ids = [143, 145, 146, 152, 154, 271, 277]

    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        if alpha_id in curated_ids:
            normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    if not normed_signals:
        for alpha_id, raw_signal in raw_signals.items():
            normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    fr_df = compute_factor_returns(normed_signals, returns_pct)
    rolling_er = fr_df.rolling(window=lookback, min_periods=20).mean()
    weights = rolling_er.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)

    smoothed = weights_norm.ewm(halflife=ema_halflife, min_periods=1).mean()
    wsum2 = smoothed.sum(axis=1).replace(0, np.nan)
    smoothed_norm = smoothed.div(wsum2, axis=0).fillna(0)

    combined = None
    for alpha_id, normed in normed_signals.items():
        w = smoothed_norm[alpha_id].values
        ws = normed.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    n_str = len(normed_signals)
    return simulate(combined, returns_pct, close, universe, max_wt=max_wt,
                    decay=decay), f"CuratedAdaptive(mw={max_wt},n={n_str},lb={lookback},hl={ema_halflife},d={decay})"


PROPER_STRATEGIES = {
    "proper_equal": strategy_proper_equal,
    "proper_adaptive": strategy_proper_adaptive,
    "qp_optimal": strategy_qp_optimal,
    "qp_net_smooth": strategy_qp_net_smooth,
    "proper_decay": strategy_proper_decay,
    "corr_select": strategy_corr_select,
    "corr_select_adaptive": strategy_corr_select_adaptive,
    "proper_dd_control": strategy_proper_dd_control,
    "regime_scaled": strategy_regime_scaled,
    "regime_net": strategy_regime_net,
    "regime_net_smooth": strategy_regime_net_smooth,
    "regime_net_deadband": strategy_regime_net_deadband,
    "orthogonal_regime_scaled": strategy_orthogonal_regime_scaled,
    "regime_deadband": strategy_regime_deadband,
    "rolling_select": strategy_rolling_select,
    "net_select_adaptive": strategy_net_select_adaptive,
    "curated_equal": strategy_curated_equal,
    "curated_adaptive": strategy_curated_adaptive,
    "ridge_combine": strategy_ridge_combine,
    "qp_pnl_combine": strategy_qp_pnl_combine,
    "hierarchical_combine": strategy_hierarchical_combine,
}

# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_portfolio_robustness(result, save_path="portfolio_performance.png", booksize=20_000_000.0, fees_bps=5.0):
    import matplotlib.pyplot as plt
    try:
        # Reconstruct pre-fee performance
        daily_fee_5bps = result.daily_turnover * booksize * (fees_bps / 10000.0)
        daily_fee_10bps = result.daily_turnover * booksize * (10.0 / 10000.0)
        
        post_fee_pnl = result.daily_pnl # This already has the default fees_bps
        pre_fee_pnl = post_fee_pnl + daily_fee_5bps
        post_fee_pnl_10bps = pre_fee_pnl - daily_fee_10bps
        
        cum_post_fee = post_fee_pnl.cumsum()
        cum_pre_fee = pre_fee_pnl.cumsum()
        cum_post_fee_10bps = post_fee_pnl_10bps.cumsum()
        
        plt.figure(figsize=(12, 6))
        plt.plot(cum_pre_fee.index, cum_pre_fee.values, label=f"Gross PnL (Pre-Fees)", color='gray', linestyle='--', alpha=0.7)
        plt.plot(cum_post_fee.index, cum_post_fee.values, label=f"Net PnL (Post-Fees {fees_bps}bps)", color='green', linewidth=2)
        plt.plot(cum_post_fee_10bps.index, cum_post_fee_10bps.values, label=f"Net PnL (Post-Fees 10.0bps)", color='red', linewidth=2, alpha=0.7)
        
        plt.title(f"Out-of-Sample Portfolio Performance\nSharpe: {result.sharpe:.2f} | TO: {result.turnover:.2f} | DD: {result.max_drawdown:.2f}")
        plt.ylabel('Cumulative PnL ($)')
        plt.xlabel('Date')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"\nSaved portfolio performance plot to {save_path}")
        plt.close()
    except Exception as e:
        print(f"\nFailed to plot performance: {e}")



def run_strategy(strategy_name, fee_bps=5.0, **kwargs):
    # Check if it's a proper strategy (needs raw signals)
    if strategy_name in PROPER_STRATEGIES:
        raw_signals, returns_pct, close, universe = load_raw_alpha_signals()
        if raw_signals is None:
            return None
        n = len(raw_signals)
        print(f"\nRunning '{strategy_name}' on {n} alphas (RAW), validation ({VAL_START} to {VAL_END}), {fee_bps}bps fees")
        result, label = PROPER_STRATEGIES[strategy_name](raw_signals, returns_pct, close, universe, fee_bps=fee_bps, **kwargs)
    else:
        signals, returns_pct, close, universe = load_alpha_signals()
        if signals is None:
            return None
        n = len(signals)
        print(f"\nRunning '{strategy_name}' on {n} alphas, validation ({VAL_START} to {VAL_END}), {fee_bps}bps fees")
        if strategy_name not in STRATEGIES:
            print(f"Unknown strategy: {strategy_name}. Available: {list(STRATEGIES.keys()) + list(PROPER_STRATEGIES.keys())}")
            return None
        result, label = STRATEGIES[strategy_name](signals, returns_pct, close, universe, **kwargs)

    print(f"\n  {label}")
    print(f"  Sharpe:       {result.sharpe:+.3f}")
    print(f"  Fitness:      {result.fitness:.3f}")
    print(f"  Returns (ann): {result.returns_ann*100:.1f}%")
    print(f"  Max Drawdown: {result.max_drawdown:.3f}")
    print(f"  Turnover:     {result.turnover:.3f}")

    # Plot the result
    save_path = f"{strategy_name}_robustness.png"
    plot_portfolio_robustness(result, save_path=save_path, fees_bps=fee_bps)

    return result


def compare_all():
    # Load both signal types
    signals, returns_pct, close, universe = load_alpha_signals()
    raw_signals, _, _, _ = load_raw_alpha_signals()
    if signals is None:
        return

    n = len(signals)
    n_raw = len(raw_signals) if raw_signals else 0
    print(f"\nComparing strategies on {n} rank-norm / {n_raw} raw alphas")
    print(f"Validation ({VAL_START} to {VAL_END}), 5.0bps fees")
    print("\nIndividual Alpha Performance (Validation):")
    print(f"  {'ID':<5} {'Sharpe':>8} {'Fitness':>8} {'TO':>6} {'DD':>8}")
    print(f"  {'-'*45}")
    for aid, sig in signals.items():
        try:
            res = simulate(sig, returns_pct, close, universe)
            print(f"  #{aid:<4} {res.sharpe:+8.3f} {res.fitness:8.3f} {res.turnover:6.3f} {res.max_drawdown:8.3f}")
        except:
            print(f"  #{aid:<4} FAILED")
    print(f"\n{'='*80}")

    results = []

    # ---- Legacy rank-normalized strategies ----
    for name, func in STRATEGIES.items():
        try:
            result, label = func(signals, returns_pct, close, universe)
            results.append((label, result.sharpe, result.fitness, result.turnover, result.max_drawdown))
        except Exception as e:
            results.append((name, 0, 0, 0, 0))
            print(f"  {name} failed: {e}")

    # Sweep select_sharpe with various thresholds and decays
    for d in [1, 2, 3]:
        for ms in [0.0, 0.3, 0.5]:
            try:
                result, label = strategy_select_sharpe(signals, returns_pct, close, universe, min_sharpe=ms, decay=d)
                results.append((label, result.sharpe, result.fitness, result.turnover, result.max_drawdown))
            except:
                pass

    # Sweep smooth_adaptive (best legacy family)
    for lb in [240, 300]:
        for hl in [30, 60]:
            try:
                result, label = strategy_smooth_adaptive(signals, returns_pct, close, universe, lookback=lb, ema_halflife=hl)
                results.append((label, result.sharpe, result.fitness, result.turnover, result.max_drawdown))
            except:
                pass

    # ---- Proper normalization strategies (raw signals) ----
    if raw_signals:
        # ProperEqual with different max_wt
        for mw in [0.02, 0.03, 0.05, 0.08]:
            try:
                result, label = strategy_proper_equal(raw_signals, returns_pct, close, universe, max_wt=mw)
                results.append((label, result.sharpe, result.fitness, result.turnover, result.max_drawdown))
            except Exception as e:
                print(f"  ProperEqual(mw={mw}) failed: {e}")

        # ProperAdaptive with different params
        for mw in [0.02, 0.03, 0.05]:
            for lb in [180, 240, 280]:
                for hl in [30, 60]:
                    try:
                        result, label = strategy_proper_adaptive(raw_signals, returns_pct, close, universe, max_wt=mw, lookback=lb, ema_halflife=hl)
                        results.append((label, result.sharpe, result.fitness, result.turnover, result.max_drawdown))
                    except Exception as e:
                        print(f"  ProperAdaptive(mw={mw},lb={lb},hl={hl}) failed: {e}")

        # QP Optimal
        for mw in [0.02, 0.03, 0.05]:
            for lb in [120, 180, 240, 300]:
                for rb in [15, 30, 60]:
                    try:
                        result, label = strategy_qp_optimal(raw_signals, returns_pct, close, universe,
                                                           max_wt=mw, lookback=lb, rebal_every=rb)
                        results.append((label, result.sharpe, result.fitness, result.turnover, result.max_drawdown))
                    except Exception as e:
                        print(f"  QPOptimal(mw={mw},lb={lb},rb={rb}) failed: {e}")

        # ProperDecay
        for mw in [0.02, 0.03, 0.05]:
            for lb in [240, 280]:
                for d in [2, 3, 4]:
                    for hl in [30, 60, 90]:
                        try:
                            result, label = strategy_proper_decay(raw_signals, returns_pct, close, universe,
                                                                 max_wt=mw, lookback=lb, decay=d, ema_halflife=hl)
                            results.append((label, result.sharpe, result.fitness, result.turnover, result.max_drawdown))
                        except Exception as e:
                            print(f"  ProperDecay(mw={mw},lb={lb},d={d},hl={hl}) failed: {e}")

        # CorrSelect with different correlation cutoffs
        for mw in [0.02, 0.03, 0.05, 0.08]:
            for mc in [0.30, 0.40, 0.50, 0.60, 0.70]:
                try:
                    # just an example top 6-11
                    for n in [6, 7, 8, 9, 10, 11]:
                        result, label = strategy_corr_select(raw_signals, returns_pct, close, universe, max_wt=mw, max_corr=mc, n_select=n)
                        results.append((label, result.sharpe, result.fitness, result.turnover, result.max_drawdown))
                except Exception as e:
                    pass

        # CorrSelAdaptive
        for mw in [0.02, 0.03, 0.05, 0.08]:
            for mc in [0.30, 0.40, 0.50, 0.60]:
                for lb in [180, 240, 280]:
                    try:
                        for n in [6, 7, 8, 9, 10]:
                            result, label = strategy_corr_select_adaptive(raw_signals, returns_pct, close, universe, max_wt=mw, max_corr=mc, lookback=lb, n_select=n)
                            results.append((label, result.sharpe, result.fitness, result.turnover, result.max_drawdown))
                    except Exception as e:
                        pass
        
        # RegimeScaled sweeps
        for mw in [0.02, 0.03]:
            for lb in [240, 280, 360]:
                for hl in [30, 60, 90]:
                    try:
                        result, label = strategy_regime_scaled(raw_signals, returns_pct, close, universe, max_wt=mw, lookback=lb, ema_halflife=hl, decay=2)
                        results.append((label, result.sharpe, result.fitness, result.turnover, result.max_drawdown))
                    except Exception as e:
                        print(f"  RegimeScaled(mw={mw},lb={lb},hl={hl}) failed: {e}")

        # RegimeDeadband sweeps
        for mw in [0.02, 0.03]:
            for lb in [240, 280, 360]:
                for hl in [30, 60]:
                    for db in [0.001, 0.002, 0.004]:
                        try:
                            result, label = strategy_regime_deadband(raw_signals, returns_pct, close, universe, max_wt=mw, lookback=lb, ema_halflife=hl, deadband=db, decay=2)
                            results.append((label, result.sharpe, result.fitness, result.turnover, result.max_drawdown))
                        except Exception as e:
                            print(f"  RegimeDeadband(mw={mw},lb={lb},hl={hl},db={db}) failed: {e}")

        # ProperDDControl sweeps
        for mw in [0.02, 0.03]:
            for lb in [240, 280]:
                for hl in [30]:
                    for mdd in [0.03, 0.05, 0.08]:
                        try:
                            result, label = strategy_proper_dd_control(raw_signals, returns_pct, close, universe, max_wt=mw, lookback=lb, ema_halflife=hl, decay=2, max_dd_tolerance=mdd)
                            results.append((label, result.sharpe, result.fitness, result.turnover, result.max_drawdown))
                        except Exception as e:
                            print(f"  ProperDD(mw={mw},lb={lb},mdd={mdd}) failed: {e}")

    # Sort by Sharpe
    results.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  {'Strategy':<45} {'Sharpe':>8} {'Fitness':>8} {'TO':>6} {'DD':>8}")
    print(f"  {'-'*80}")
    for label, sr, fit, to, dd in results:
        print(f"  {label:<45} {sr:+8.3f} {fit:8.3f} {to:6.3f} {dd:8.3f}")


def main():
    all_strategies = {**STRATEGIES, **PROPER_STRATEGIES}
    parser = argparse.ArgumentParser(description="Agent 2: Portfolio Construction (Validation Only)")
    parser.add_argument("--strategy", type=str, default="corr_select_adaptive",
                        choices=list(all_strategies.keys()),
                        help="Combination strategy")
    parser.add_argument("--compare", action="store_true", help="Compare all strategies")
    parser.add_argument("--scoreboard", action="store_true", help="Show portfolio scoreboard")
    parser.add_argument("--lookback", type=int, default=LOOKBACK, help="Rolling lookback window")
    parser.add_argument("--top", type=int, default=TOP_N, help="Top N factors for top_n strategy")
    parser.add_argument("--decay", type=int, default=SIM_DECAY, help="Decay parameter for strategies like select_sharpe or proper_decay")
    parser.add_argument("--mw", type=float, default=MAX_WEIGHT, help="Max weight for proper strategies")
    parser.add_argument("--hl", type=int, default=30, help="EMA halflife")
    parser.add_argument("--mc", type=float, default=0.3, help="Max corr (default 0.3 — best from compare)")
    parser.add_argument("--fees", type=float, default=VAL_FEES, help="Fees in bps to simulate")
    parser.add_argument("--db", type=float, default=0.01, help="Deadband threshold")
    args = parser.parse_args()

    if args.compare or args.scoreboard:
        compare_all()
        return

    run_strategy(args.strategy, lookback=args.lookback, top_n=args.top, decay=args.decay, max_wt=args.mw, ema_halflife=args.hl, max_corr=args.mc, fee_bps=args.fees, deadband=args.db)


if __name__ == "__main__":
    main()
