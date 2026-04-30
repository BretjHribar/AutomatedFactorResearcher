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

# ============================================================================
# PREFERRED COMBINATION STRATEGY — 2026-04-22
# ============================================================================
# BillionsQP  (strategy_billions_qp, --strategy billions_qp)  is the preferred
# production combiner for the KUCOIN_TOP100 universe.
#
# Architecture: Two-stage pipeline
#   Stage 1: Billions regression (Kakushadze, arxiv 1603.05937)
#             Walk-forward, bar-by-bar regression that estimates E[factor return]
#             using orthogonal residuals — superior to naive rolling mean (μ).
#   Stage 2: CVXPY MV-Utility QP (Isichenko Eq 6.1)
#             max_α  α'μ  -  (k/2)·α'Cα  -  λ_tc·‖Δα‖₁
#             Explicitly optimises the 3-way tradeoff:
#               • Closeness to signal  (α'μ)
#               • Risk model           (Ledoit-Wolf shrunk factor covariance)
#               • Execution cost       (L1 TC penalty on position changes)
#
# Validated results (Sep 2024 → Mar 2025, 5bps fees, $20M book, 18 alphas):
#   BillionsQP(ol=120, k=2.0, tc=0.5):  Sharpe=+4.62, Fitness=11.31,
#                                         TO=0.195, MaxDD=-3.9%, Ann.Ret=117%
#   vs. CorrSelect (prior best):          Sharpe=+4.40
#   vs. ProperEqual (baseline):           Sharpe=+3.47
# ============================================================================
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
UNIVERSE     = "KUCOIN_TOP100"
INTERVAL     = "4h"
BOOKSIZE     = 2_000_000.0
MAX_WEIGHT   = 0.02          # KuCoin BEST: ProperEqual(mw=0.02) — Sharpe +2.97, Fitness 5.76, TO=0.19, DD=-0.05
NEUTRALIZE   = "market"

BARS_PER_DAY = 6
COVERAGE_CUTOFF = 0.3
VAL_FEES     = 5.0           # 5bps fees — always applied on validation

# Portfolio construction parameters — AGENT 2 TUNES THESE
# BEST STRATEGY [KUCOIN]: ProperEqual(mw=0.02) — Sharpe +2.97, Fitness 5.76, TO=0.19, DD=-0.05, Ann. Return 71.1%
#   Both alphas selected; corr(#2, #3) < 0.5 so diversification gain is material.
#   QPOptimal and CorrSelect collapse to ProperEqual given only 2 factors (QP has no OOS advantage with K=2).
LOOKBACK     = 240           # Rolling window for adaptive weights (bars)
IC_LOOKBACK  = 60            # Rolling window for IC-based weighting
TOP_N        = 5             # For top-N strategy: how many factors to use
DECAY_ALPHA  = 0.95          # Exponential decay for momentum weighting
SIM_DECAY    = 2             # Simulation-level decay (signal persistence)
MIN_SR       = 0.5           # Minimum validation Sharpe to keep an alpha in the portfolio

DB_PATH      = "data/alphas.db"


# ============================================================================
# DATA LOADING (VALIDATION ONLY)
# ============================================================================

_DATA_CACHE = {}

def load_val_data():
    if "val" in _DATA_CACHE:
        return _DATA_CACHE["val"]

    mat_dir = Path(f"data/kucoin_cache/matrices/{INTERVAL}")
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
    """Load raw (un-normalized) alpha signals — preserves magnitude info.

    Optional env-var filters (set before invoking eval_portfolio.py):
      ALPHA_ID_MAX=<int>      — only load alphas with id <= this value
      ALPHA_ID_MIN=<int>      — only load alphas with id >= this value
      ALPHA_LIMIT=<int>       — after id filtering, take only the first N (sorted by id)
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    sql = "SELECT id, expression FROM alphas WHERE archived=0"
    params = []
    if os.environ.get("ALPHA_ID_MIN"):
        sql += " AND id >= ?"; params.append(int(os.environ["ALPHA_ID_MIN"]))
    if os.environ.get("ALPHA_ID_MAX"):
        sql += " AND id <= ?"; params.append(int(os.environ["ALPHA_ID_MAX"]))
    sql += " ORDER BY id"
    if os.environ.get("ALPHA_LIMIT"):
        sql += " LIMIT ?"; params.append(int(os.environ["ALPHA_LIMIT"]))
    c.execute(sql, params)
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


def load_full_data():
    """Load all KuCoin matrices without date slicing (train + val + test)."""
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
    close = matrices.get("close")
    if close is not None:
        matrices["returns_pct"] = close.pct_change()
    result = (matrices, universe_df[valid_tickers])
    _DATA_CACHE["full"] = result
    return result


def load_full_alpha_signals():
    """Evaluate all alpha expressions over the full data range (no date slice)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, expression FROM alphas WHERE archived=0 ORDER BY id")
    alphas = c.fetchall()
    conn.close()
    if not alphas:
        print("No alphas in DB.")
        return None, None, None, None
    matrices, universe = load_full_data()
    close   = matrices.get("close")
    returns_pct = matrices.get("returns_pct")
    raw_signals = {}
    for alpha_id, expression in alphas:
        try:
            alpha_df = evaluate_expression(expression, matrices)
            if alpha_df is not None and not alpha_df.empty:
                raw_signals[alpha_id] = alpha_df
        except Exception as e:
            print(f"  Alpha #{alpha_id} failed: {e}")
    print(f"  Loaded {len(raw_signals)} alpha signals "
          f"({close.index[0]} to {close.index[-1]})")
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
    return simulate(combined, returns_pct, close, universe, max_wt=max_wt,
                    fees_bps=kwargs.get("fees_bps", kwargs.get("fee_bps", VAL_FEES))), f"ProperEqual(mw={max_wt})"


def strategy_proper_equal_qp(raw_signals, returns_pct, close, universe,
                              max_wt=MAX_WEIGHT, rebal_every=1, qp_lookback=120,
                              track_aversion=1.0, risk_aversion=0.0, tc_bps=None,
                              **kwargs):
    """ProperEqual aggregation + asset-level QP execution layer.

    Same target as strategy_proper_equal (uniform alpha sum), then QP solves
        max -A·||P-target||² - k·P'CP - (tc/1e4)·||P-P_prev||_1
        s.t. |P_s| ≤ max_wt
    inner tc_bps defaults to outer fees_bps for correct net-utility framing.
    """
    combined = None
    for _, raw_signal in raw_signals.items():
        normed = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)
        combined = normed if combined is None else combined.add(normed, fill_value=0)

    eff_fees = kwargs.get("fees_bps", kwargs.get("fee_bps", VAL_FEES))
    eff_tc_bps = eff_fees if tc_bps is None else tc_bps
    P_df = _qp_execution_layer(
        target_df=combined, returns_pct=returns_pct,
        max_wt=max_wt, rebal_every=rebal_every, cov_lookback=qp_lookback,
        track_aversion=track_aversion, risk_aversion=risk_aversion,
        tc_bps=eff_tc_bps, verbose=False,
    )
    return (simulate(P_df, returns_pct, close, universe, max_wt=max_wt, fees_bps=eff_fees),
            f"ProperEqualQP(mw={max_wt},A={track_aversion},k={risk_aversion},tc_bps={eff_tc_bps})")


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


# ============================================================================
# BOOK-INSPIRED CVXPY QP STRATEGIES (Isichenko — Quantitative Portfolio Mgmt)
# ============================================================================

def _ledoit_wolf_shrink(cov: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf analytical shrinkage toward scaled identity.
    Reduces noise in sample covariance for small-sample regimes.
    Formula: C* = (1-rho)*C + rho*mu_hat*I  where mu_hat = tr(C)/n
    Uses Oracle Approximating Shrinkage (OAS) approximation."""
    n = cov.shape[0]
    tr_c = np.trace(cov)
    tr_c2 = np.trace(cov @ cov)
    mu_hat = tr_c / n
    # OAS shrinkage intensity
    rho_num = (1.0 - 2.0 / n) * tr_c2 + tr_c ** 2
    rho_den = (n + 1.0 - 2.0 / n) * (tr_c2 - tr_c ** 2 / n)
    rho = np.clip(rho_num / (rho_den + 1e-16), 0.0, 1.0)
    return (1.0 - rho) * cov + rho * mu_hat * np.eye(n)


def strategy_cvxpy_mv_utility(raw_signals, returns_pct, close, universe,
                               max_wt=0.02, lookback=280, rebal_every=12,
                               risk_aversion=2.0, tc_penalty=0.5,
                               decay=2, ema_halflife=60, **kwargs):
    """
    CVXPY Mean-Variance Utility Optimizer (Isichenko Eq 6.1 / Fig 3.2)

    Implements the canonical institutional QP:
      max_alpha  alpha'*mu - (k/2)*alpha'*C*alpha - lambda_tc * ||alpha - alpha_prev||_1

    Where:
      mu    = rolling mean factor return vector (expected PnL)
      C     = rolling factor PnL covariance (Ledoit-Wolf shrunk)
      k     = risk_aversion  (controls Sharpe vs raw return tradeoff)
      lambda_tc = tc_penalty (L1 cost on changes, equivalent to bid-ask spread)

    Non-negativity enforced (long-only alpha weights per Isichenko Fig 3.2).
    Solved via CVXPY with CLARABEL (or ECOS/SCS/OSQP fallback).
    """
    import cvxpy as cp

    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    fr_df = compute_factor_returns(normed_signals, returns_pct)
    alpha_ids = list(fr_df.columns)
    K = len(alpha_ids)
    T = len(fr_df)
    dates = fr_df.index

    weights_arr = np.zeros((T, K))
    prev_w = np.ones(K) / K

    for t in range(lookback, T, rebal_every):
        window = fr_df.iloc[max(0, t - lookback):t].dropna(axis=1, how='all')
        if len(window) < 30 or window.shape[1] < 2:
            weights_arr[t:min(t + rebal_every, T)] = prev_w
            continue

        # Align active columns
        active_ids = list(window.columns)
        K_a = len(active_ids)
        mu = window.mean().values              # (K_a,)
        cov_raw = window.cov().values          # (K_a, K_a)
        cov = _ledoit_wolf_shrink(cov_raw) + 1e-8 * np.eye(K_a)

        alpha_prev_active = np.array([prev_w[alpha_ids.index(aid)] for aid in active_ids])

        # CVXPY formulation: Isichenko Eq 6.1
        alpha = cp.Variable(K_a, nonneg=True)   # long-only factor weights
        delta = cp.Variable(K_a)                 # trade vector = alpha - alpha_prev

        expected_return = mu @ alpha
        risk_term = 0.5 * risk_aversion * cp.quad_form(alpha, cov)
        tc_term = tc_penalty * cp.norm1(delta)   # L1 transaction cost

        objective = cp.Maximize(expected_return - risk_term - tc_term)
        constraints = [
            cp.sum(alpha) == 1.0,
            alpha <= 0.5,               # max 50% in one factor
            delta == alpha - alpha_prev_active,
        ]

        prob = cp.Problem(objective, constraints)
        try:
            # Try solvers in order of preference: CLARABEL > ECOS > SCS
            for solver in [cp.CLARABEL, cp.ECOS, cp.SCS]:
                try:
                    prob.solve(solver=solver, warm_start=True)
                    if prob.status in ['optimal', 'optimal_inaccurate'] and alpha.value is not None:
                        break
                except Exception:
                    continue

            w = alpha.value
            if w is None or not np.all(np.isfinite(w)):
                w = alpha_prev_active
            w = np.clip(w, 0, None)
            wsum = w.sum()
            w = w / wsum if wsum > 1e-10 else alpha_prev_active
        except Exception:
            w = alpha_prev_active

        # Map back to full factor set
        w_full = np.zeros(K)
        for i, aid in enumerate(active_ids):
            w_full[alpha_ids.index(aid)] = w[i]
        prev_w = w_full
        weights_arr[t:min(t + rebal_every, T)] = w_full

    # Fill initial lookback
    weights_arr[:lookback] = 1.0 / K

    # EMA smooth to reduce weight churn
    weights_df = pd.DataFrame(weights_arr, index=dates, columns=alpha_ids)
    weights_smooth = weights_df.ewm(halflife=ema_halflife, min_periods=1).mean()
    wsum = weights_smooth.sum(axis=1).replace(0, np.nan)
    weights_norm = weights_smooth.div(wsum, axis=0).fillna(0)

    combined = None
    for alpha_id, normed in normed_signals.items():
        w = weights_norm[alpha_id].values
        ws = normed.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    return simulate(combined, returns_pct, close, universe, max_wt=max_wt,
                    decay=decay), f"CVXPYMVUtility(mw={max_wt},lb={lookback},k={risk_aversion},tc={tc_penalty},hl={ema_halflife})"


def strategy_cvxpy_ledoit_sharpe(raw_signals, returns_pct, close, universe,
                                  max_wt=0.02, lookback=280, rebal_every=12,
                                  l2_lambda=0.01, decay=2, ema_halflife=60, **kwargs):
    """
    CVXPY Max-Sharpe with Ledoit-Wolf Covariance Shrinkage

    Maximizes the Sharpe ratio of alpha combination weights:
      max  mu'w / sqrt(w'Cw)  s.t. w >= 0, sum(w) = 1, max_wt per factor

    Reformulated as equivalent convex QP (Cornuejols & Tutuncu, 2006):
      min  y'Cy  s.t. mu'y = 1, y >= 0  then normalize: w = y / sum(y)

    Covariance C is Ledoit-Wolf shrunk for stability.
    L2 regularization lambda added to prevent degenerate concentrated bets.
    Uses OSQP/CLARABEL (suitable for large-scale per Isichenko Chapter 6).
    """
    import cvxpy as cp

    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    fr_df = compute_factor_returns(normed_signals, returns_pct)
    alpha_ids = list(fr_df.columns)
    K = len(alpha_ids)
    T = len(fr_df)
    dates = fr_df.index

    weights_arr = np.ones((T, K)) / K
    prev_w = np.ones(K) / K

    for t in range(lookback, T, rebal_every):
        window = fr_df.iloc[max(0, t - lookback):t].dropna(axis=1, how='all')
        if len(window) < 30 or window.shape[1] < 2:
            weights_arr[t:min(t + rebal_every, T)] = prev_w
            continue

        active_ids = list(window.columns)
        K_a = len(active_ids)
        mu = window.mean().values

        # Only proceed if at least some positive expected returns
        if np.all(mu <= 0):
            weights_arr[t:min(t + rebal_every, T)] = prev_w
            continue

        cov_raw = window.cov().values
        # Ledoit-Wolf shrinkage
        cov = _ledoit_wolf_shrink(cov_raw)
        # L2 regularization (prevents ill-conditioning)
        cov = cov + l2_lambda * np.eye(K_a) + 1e-8 * np.eye(K_a)

        # Max-Sharpe reformulation: min y'Cy s.t. mu'y = 1, y >= 0
        y = cp.Variable(K_a, nonneg=True)
        objective = cp.Minimize(cp.quad_form(y, cov))
        constraints = [mu @ y == 1.0]

        prob = cp.Problem(objective, constraints)
        try:
            for solver in [cp.CLARABEL, cp.OSQP, cp.ECOS, cp.SCS]:
                try:
                    prob.solve(solver=solver)
                    if prob.status in ['optimal', 'optimal_inaccurate'] and y.value is not None:
                        break
                except Exception:
                    continue

            y_val = y.value
            if y_val is None or not np.all(np.isfinite(y_val)) or y_val.sum() < 1e-10:
                raise ValueError('Solver failed')

            # Normalize to get portfolio weights
            w = np.clip(y_val, 0, None)
            w = w / w.sum()
        except Exception:
            w = np.ones(K_a) / K_a

        # Map back
        w_full = np.zeros(K)
        for i, aid in enumerate(active_ids):
            w_full[alpha_ids.index(aid)] = w[i]
        prev_w = w_full
        weights_arr[t:min(t + rebal_every, T)] = w_full

    weights_arr[:lookback] = 1.0 / K

    weights_df = pd.DataFrame(weights_arr, index=dates, columns=alpha_ids)
    weights_smooth = weights_df.ewm(halflife=ema_halflife, min_periods=1).mean()
    wsum = weights_smooth.sum(axis=1).replace(0, np.nan)
    weights_norm = weights_smooth.div(wsum, axis=0).fillna(0)

    combined = None
    for alpha_id, normed in normed_signals.items():
        w = weights_norm[alpha_id].values
        ws = normed.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    return simulate(combined, returns_pct, close, universe, max_wt=max_wt,
                    decay=decay), f"CVXPYLedoitSharpe(mw={max_wt},lb={lookback},l2={l2_lambda},hl={ema_halflife})"


def strategy_cvxpy_regime_utility(raw_signals, returns_pct, close, universe,
                                   max_wt=0.02, lookback=280, rebal_every=12,
                                   base_risk_aversion=1.5, regime_scale=3.0,
                                   decay=2, ema_halflife=60, **kwargs):
    """
    CVXPY Regime-Adaptive MV Utility (Isichenko + Regime Insight)

    Fuses the book's MV utility formulation with the project's RegimeScaled insight:
      Increases risk_aversion k proportionally to market volatility stress,
      so the optimizer naturally reduces factor bets in volatile periods.

      k(t) = base_k * (vol(t) / median_vol(t)) * regime_scale

    At each rebalance, solves:
      max_alpha  alpha'*mu - (k(t)/2)*alpha'*C*alpha
      s.t. alpha >= 0, sum(alpha) = 1

    This is equivalent to Isichenko's portfolio utility but with a time-varying
    risk aversion that acts like a dynamic leverage control — raising it during
    stress (like RegimeScaled vol_scalar) and lowering it during calm markets.
    """
    import cvxpy as cp

    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    fr_df = compute_factor_returns(normed_signals, returns_pct)
    alpha_ids = list(fr_df.columns)
    K = len(alpha_ids)
    T = len(fr_df)
    dates = fr_df.index

    # Compute regime vol scalar (same as RegimeScaled)
    market_returns = returns_pct.mean(axis=1)
    market_vol = market_returns.rolling(window=84, min_periods=20).std()
    market_vol_smooth = market_vol.ewm(halflife=42).mean()
    median_vol = market_vol_smooth.expanding().median().clip(lower=1e-4)
    # vol_ratio > 1 means high stress → increase risk aversion → reduce bets
    vol_ratio = (market_vol_smooth / median_vol).clip(lower=0.4, upper=4.0)
    vol_ratio = vol_ratio.ewm(halflife=30).mean()
    # Align to fr_df index
    vol_ratio = vol_ratio.reindex(dates).ffill().fillna(1.0)

    weights_arr = np.ones((T, K)) / K
    prev_w = np.ones(K) / K

    for t in range(lookback, T, rebal_every):
        window = fr_df.iloc[max(0, t - lookback):t].dropna(axis=1, how='all')
        if len(window) < 30 or window.shape[1] < 2:
            weights_arr[t:min(t + rebal_every, T)] = prev_w
            continue

        active_ids = list(window.columns)
        K_a = len(active_ids)
        mu = window.mean().values
        cov_raw = window.cov().values
        cov = _ledoit_wolf_shrink(cov_raw) + 1e-8 * np.eye(K_a)

        # Time-varying risk aversion: higher during vol stress
        k_t = base_risk_aversion * vol_ratio.iloc[t] * regime_scale

        alpha = cp.Variable(K_a, nonneg=True)
        objective = cp.Maximize(mu @ alpha - 0.5 * k_t * cp.quad_form(alpha, cov))
        constraints = [cp.sum(alpha) == 1.0, alpha <= 0.5]

        prob = cp.Problem(objective, constraints)
        try:
            for solver in [cp.CLARABEL, cp.ECOS, cp.SCS]:
                try:
                    prob.solve(solver=solver)
                    if prob.status in ['optimal', 'optimal_inaccurate'] and alpha.value is not None:
                        break
                except Exception:
                    continue

            w = alpha.value
            if w is None or not np.all(np.isfinite(w)):
                w = prev_w[[alpha_ids.index(aid) for aid in active_ids]]
            w = np.clip(w, 0, None)
            wsum = w.sum()
            w = w / wsum if wsum > 1e-10 else np.ones(K_a) / K_a
        except Exception:
            w = np.ones(K_a) / K_a

        w_full = np.zeros(K)
        for i, aid in enumerate(active_ids):
            w_full[alpha_ids.index(aid)] = w[i]
        prev_w = w_full
        weights_arr[t:min(t + rebal_every, T)] = w_full

    weights_arr[:lookback] = 1.0 / K

    weights_df = pd.DataFrame(weights_arr, index=dates, columns=alpha_ids)
    weights_smooth = weights_df.ewm(halflife=ema_halflife, min_periods=1).mean()
    wsum = weights_smooth.sum(axis=1).replace(0, np.nan)
    weights_norm = weights_smooth.div(wsum, axis=0).fillna(0)

    combined = None
    for alpha_id, normed in normed_signals.items():
        w = weights_norm[alpha_id].values
        ws = normed.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    return simulate(combined, returns_pct, close, universe, max_wt=max_wt,
                    decay=decay), f"CVXPYRegimeUtil(mw={max_wt},lb={lookback},k={base_risk_aversion},rs={regime_scale},hl={ema_halflife})"


# ============================================================================
# MORE BOOK-INSPIRED STRATEGIES (Isichenko Chs 3, 4, 6, 7)
# ============================================================================

def strategy_kelly_optimal(raw_signals, returns_pct, close, universe,
                            max_wt=0.02, lookback=280, rebal_every=12,
                            kelly_fraction=0.5, decay=2, ema_halflife=60, **kwargs):
    """
    Kelly-Optimal Factor Weights (Isichenko Sec 6.9)

    Unconstrained Kelly-optimal: alpha* = Sigma^{-1} * mu
    Maximises E[log(wealth)] — the long-run compound growth rate.

    Practical implementation uses 'half-Kelly' (kelly_fraction=0.5) to
    account for parameter estimation error (standard institutional practice).
    Non-negativity constraint applied; weights normalised to sum to 1.
    Covariance uses Ledoit-Wolf shrinkage for numerical stability.
    """
    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    fr_df = compute_factor_returns(normed_signals, returns_pct)
    alpha_ids = list(fr_df.columns)
    K = len(alpha_ids)
    T = len(fr_df)
    dates = fr_df.index

    weights_arr = np.ones((T, K)) / K
    prev_w = np.ones(K) / K

    for t in range(lookback, T, rebal_every):
        window = fr_df.iloc[max(0, t - lookback):t].dropna(axis=1, how='all')
        if len(window) < 30 or window.shape[1] < 2:
            weights_arr[t:min(t + rebal_every, T)] = prev_w
            continue

        active_ids = list(window.columns)
        K_a = len(active_ids)
        mu = window.mean().values
        cov_raw = window.cov().values
        cov = _ledoit_wolf_shrink(cov_raw) + 1e-8 * np.eye(K_a)

        try:
            # Kelly: alpha* = Sigma^{-1} * mu
            kelly_w = np.linalg.solve(cov, mu)      # more stable than inv(C)*mu
            kelly_w = kelly_fraction * kelly_w
            kelly_w = np.clip(kelly_w, 0, None)      # long-only projection
            wsum = kelly_w.sum()
            w = kelly_w / wsum if wsum > 1e-10 else np.ones(K_a) / K_a
        except np.linalg.LinAlgError:
            w = np.ones(K_a) / K_a

        w_full = np.zeros(K)
        for i, aid in enumerate(active_ids):
            w_full[alpha_ids.index(aid)] = w[i]
        prev_w = w_full
        weights_arr[t:min(t + rebal_every, T)] = w_full

    weights_arr[:lookback] = 1.0 / K
    weights_df = pd.DataFrame(weights_arr, index=dates, columns=alpha_ids)
    weights_smooth = weights_df.ewm(halflife=ema_halflife, min_periods=1).mean()
    wsum = weights_smooth.sum(axis=1).replace(0, np.nan)
    weights_norm = weights_smooth.div(wsum, axis=0).fillna(0)

    combined = None
    for alpha_id, normed in normed_signals.items():
        w = weights_norm[alpha_id].values
        ws = normed.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    return simulate(combined, returns_pct, close, universe, max_wt=max_wt,
                    decay=decay), f"KellyOptimal(mw={max_wt},lb={lookback},kf={kelly_fraction},hl={ema_halflife})"


def strategy_min_variance_bayesian(raw_signals, returns_pct, close, universe,
                                    max_wt=0.02, lookback=280, rebal_every=12,
                                    decay=2, ema_halflife=60, **kwargs):
    """
    Minimum-Variance Bayesian Combination (Isichenko Sec 3.3)

    Optimal Bayesian combination when expected returns are uncertain:
    weight by inverse forecast covariance (precision weighting).

      w* = Sigma^{-1} * 1 / (1' * Sigma^{-1} * 1)

    Purely risk-driven — minimises portfolio variance regardless of mu.
    Extremely robust to mean estimation error. Uses Ledoit-Wolf covariance.
    """
    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    fr_df = compute_factor_returns(normed_signals, returns_pct)
    alpha_ids = list(fr_df.columns)
    K = len(alpha_ids)
    T = len(fr_df)
    dates = fr_df.index

    weights_arr = np.ones((T, K)) / K
    prev_w = np.ones(K) / K

    for t in range(lookback, T, rebal_every):
        window = fr_df.iloc[max(0, t - lookback):t].dropna(axis=1, how='all')
        if len(window) < 30 or window.shape[1] < 2:
            weights_arr[t:min(t + rebal_every, T)] = prev_w
            continue

        active_ids = list(window.columns)
        K_a = len(active_ids)
        cov_raw = window.cov().values
        cov = _ledoit_wolf_shrink(cov_raw) + 1e-8 * np.eye(K_a)

        try:
            ones = np.ones(K_a)
            # w* = Sigma^{-1} * 1 / (1' * Sigma^{-1} * 1)
            cov_inv_ones = np.linalg.solve(cov, ones)
            w_raw = cov_inv_ones / (ones @ cov_inv_ones)
            w = np.clip(w_raw, 0, None)           # long-only projection
            wsum = w.sum()
            w = w / wsum if wsum > 1e-10 else ones / K_a
        except np.linalg.LinAlgError:
            w = np.ones(K_a) / K_a

        w_full = np.zeros(K)
        for i, aid in enumerate(active_ids):
            w_full[alpha_ids.index(aid)] = w[i]
        prev_w = w_full
        weights_arr[t:min(t + rebal_every, T)] = w_full

    weights_arr[:lookback] = 1.0 / K
    weights_df = pd.DataFrame(weights_arr, index=dates, columns=alpha_ids)
    weights_smooth = weights_df.ewm(halflife=ema_halflife, min_periods=1).mean()
    wsum = weights_smooth.sum(axis=1).replace(0, np.nan)
    weights_norm = weights_smooth.div(wsum, axis=0).fillna(0)

    combined = None
    for alpha_id, normed in normed_signals.items():
        w = weights_norm[alpha_id].values
        ws = normed.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    return simulate(combined, returns_pct, close, universe, max_wt=max_wt,
                    decay=decay), f"MinVarBayesian(mw={max_wt},lb={lookback},hl={ema_halflife})"


def strategy_hrp(raw_signals, returns_pct, close, universe,
                 max_wt=0.02, lookback=280, rebal_every=24,
                 decay=2, ema_halflife=60, **kwargs):
    """
    Hierarchical Risk Parity (de Prado 2016, Isichenko Sec 3.5.3)

    Allocates risk equal-weight across a hierarchical clustering of alphas.
    Unlike MV optimisation, HRP:
      1. Requires NO matrix inversion (numerically stable with 21 factors)
      2. Naturally handles correlated alphas via clustering
      3. Robust to covariance estimation error

    Algorithm:
      1. Build correlation-based distance matrix D_ij = sqrt(0.5*(1 - rho_ij))
      2. Hierarchical clustering (single linkage) -> seriation order
      3. Recursive bisection: at each split allocate proportional to inverse
         cluster variance (equal risk contribution)
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    fr_df = compute_factor_returns(normed_signals, returns_pct)
    alpha_ids = list(fr_df.columns)
    K = len(alpha_ids)
    T = len(fr_df)
    dates = fr_df.index

    def _hrp_weights(cov: np.ndarray, corr: np.ndarray) -> np.ndarray:
        """Compute HRP weights from cov + corr matrices."""
        n = cov.shape[0]
        # Distance matrix from correlation: D = sqrt(0.5*(1 - rho))
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
        np.fill_diagonal(dist, 0.0)

        try:
            condensed = squareform(dist, checks=False)
            Z = linkage(condensed, method='single')
            sort_ix = leaves_list(Z)       # seriation: quasi-diagonalise
        except Exception:
            sort_ix = np.arange(n)

        # Recursive bisection over the seriated index
        w = np.ones(n)
        items = [list(sort_ix)]  # start with all items as one cluster

        while items:
            items = [i[j:k] for i in items
                     for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                     if len(i) > 1]         # split each cluster in half
            for item in items:
                item_arr = list(item)
                left = item_arr[:len(item_arr) // 2]
                right = item_arr[len(item_arr) // 2:]

                # Cluster variance = w'Cw for each half
                def _cluster_var(indices):
                    sub_cov = cov[np.ix_(indices, indices)]
                    sub_w = w[indices] / (w[indices].sum() + 1e-16)
                    return float(sub_w @ sub_cov @ sub_w)

                var_l = _cluster_var(left)
                var_r = _cluster_var(right)

                # Allocate inversely proportional to cluster variance
                alpha_ = 1.0 - var_l / (var_l + var_r + 1e-16)
                w[left] *= alpha_
                w[right] *= (1.0 - alpha_)

        w = np.clip(w, 0, None)
        total = w.sum()
        return w / total if total > 1e-10 else np.ones(n) / n

    weights_arr = np.ones((T, K)) / K
    prev_w = np.ones(K) / K

    for t in range(lookback, T, rebal_every):
        window = fr_df.iloc[max(0, t - lookback):t].dropna(axis=1, how='all')
        if len(window) < 30 or window.shape[1] < 2:
            weights_arr[t:min(t + rebal_every, T)] = prev_w
            continue

        active_ids = list(window.columns)
        K_a = len(active_ids)
        cov_raw = window.cov().values
        cov = _ledoit_wolf_shrink(cov_raw) + 1e-8 * np.eye(K_a)
        # Correlation from shrunk cov
        std_diag = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std_diag, std_diag)
        corr = np.clip(corr, -1.0, 1.0)

        try:
            w = _hrp_weights(cov, corr)
        except Exception:
            w = np.ones(K_a) / K_a

        w_full = np.zeros(K)
        for i, aid in enumerate(active_ids):
            w_full[alpha_ids.index(aid)] = w[i]
        prev_w = w_full
        weights_arr[t:min(t + rebal_every, T)] = w_full

    weights_arr[:lookback] = 1.0 / K
    weights_df = pd.DataFrame(weights_arr, index=dates, columns=alpha_ids)
    weights_smooth = weights_df.ewm(halflife=ema_halflife, min_periods=1).mean()
    wsum = weights_smooth.sum(axis=1).replace(0, np.nan)
    weights_norm = weights_smooth.div(wsum, axis=0).fillna(0)

    combined = None
    for alpha_id, normed in normed_signals.items():
        w = weights_norm[alpha_id].values
        ws = normed.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    return simulate(combined, returns_pct, close, universe, max_wt=max_wt,
                    decay=decay), f"HRP(mw={max_wt},lb={lookback},rb={rebal_every},hl={ema_halflife})"


def strategy_billions(raw_signals, returns_pct, close, universe,
                      max_wt=MAX_WEIGHT, optim_lookback=60, **kwargs):
    """
    Billions Regression Combiner — Kakushadze "How to Combine a Billion Alphas"
    (arxiv 1603.05937).

    Walk-forward, bar-by-bar regression that finds the residual portfolio weights
    which are orthogonal to all reachable linear combinations of past alpha returns.

    Lookahead-bias audit
    ────────────────────
    1. Factor returns:  fr = signal.shift(1) * returns   [lagged 1 bar — no bias]
    2. Expected rets:   rolling_mean(fr).shift(1)         [only sees data up to t-1]
    3. Walk-forward:    at bar test_start, window is
                          fr[test_start : test_start + optim_lookback]
                        and result stored at optim_end + 1 (= test_start + optim_lookback + 1)
                        → at bar t, weight uses data from [t-L-1, t-1).  No bias.
    4. Sim execution:   simulate() uses the combined signal; the vectorised sim
                        lags positions by 1 bar internally before multiplying returns.
    """
    from sklearn import linear_model

    # ── 0. Normalize each alpha signal ──────────────────────────────────────
    normed_signals = {
        aid: proper_normalize_alpha(raw, universe, max_wt=max_wt)
        for aid, raw in raw_signals.items()
    }

    dates    = close.index
    tickers  = close.columns.tolist()
    n_bars   = len(dates)
    aid_list = list(normed_signals.keys())
    n_alphas = len(aid_list)

    # ── 1. Compute factor returns (lagged 1 bar — no lookahead) ─────────────
    ret_df = returns_pct.reindex(index=dates, columns=tickers)
    fr_data = {}
    for aid, norm in normed_signals.items():
        lagged = norm.shift(1)                              # lag positions by 1
        ab = lagged.abs().sum(axis=1).replace(0, np.nan)
        n  = lagged.div(ab, axis=0)
        fr_data[aid] = (n * ret_df).sum(axis=1)
    fr_df = pd.DataFrame(fr_data, index=dates)             # shape (T, N_alphas)

    # ── 2. Rolling expected returns — shifted 1 bar (no lookahead) ──────────
    min_periods = max(1, optim_lookback // 2)
    alphas_exp_ret = (
        fr_df.rolling(window=optim_lookback, min_periods=min_periods)
             .mean()
             .shift(1)                                      # crucial: no lookahead
    )
    alphas_exp_ret = alphas_exp_ret.clip(lower=0)

    # ── 3. Initialise weights to equal-weight ───────────────────────────────
    alpha_weights_ts = pd.DataFrame(
        1.0 / n_alphas, index=dates, columns=aid_list
    )

    reg = linear_model.LinearRegression(fit_intercept=False)

    # ── 4. Walk-forward regression loop ─────────────────────────────────────
    #   At test_start t, window = fr[t : t+L].
    #   Weights stored at t+L+1  →  zero lookahead.
    for test_start in range(1, n_bars - optim_lookback - 2):
        optim_end = test_start + optim_lookback
        if optim_end + 1 >= n_bars:
            break
        try:
            # a. Rolling window of realised factor returns
            bil_df = fr_df.iloc[test_start:optim_end].copy()

            # b. Demean along time axis
            demeaned = bil_df - bil_df.mean(axis=0)

            # c. Per-alpha std; skip if any alpha is constant
            sample_std = demeaned.std(axis=0).replace(0, np.nan)
            if sample_std.isna().any():
                continue

            # d. Normalize
            normalized = demeaned.divide(sample_std)

            # e. A_is = normalized returns (lookback × N_alphas)
            A_is = normalized.fillna(0.0)

            # f. Expected returns for window, normalized by same std
            sub_exp = alphas_exp_ret.iloc[test_start:optim_end].divide(sample_std)
            sub_exp = sub_exp.fillna(0.0)

            # g. Linear regression; no intercept — matches paper
            reg.fit(A_is.values, sub_exp.values)

            # h. Residuals = predicted − actual
            residuals = pd.DataFrame(
                reg.predict(A_is.values) - sub_exp.values,
                index=sub_exp.index, columns=sub_exp.columns
            )

            # i. Weights = residuals / std
            opt_w = residuals.divide(sample_std)

            # j. Normalize each row to sum = 1
            row_sums = opt_w.sum(axis=1).replace(0, np.nan)
            opt_w = opt_w.div(row_sums, axis=0)

            # k. Take last row, store ONE BAR AHEAD of window end
            alpha_weights_ts.iloc[optim_end + 1] = opt_w.iloc[-1].values

        except Exception:
            pass   # keep equal-weight for this bar

    # ── 5. Combine signals using time-varying scalar weights ─────────────────
    combined = None
    for aid in aid_list:
        w  = alpha_weights_ts[aid]
        ws = normed_signals[aid].mul(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    return (simulate(combined, returns_pct, close, universe, max_wt=max_wt,
                     fees_bps=kwargs.get("fees_bps", kwargs.get("fee_bps", VAL_FEES))),
            f"Billions(mw={max_wt},ol={optim_lookback})")


def _qp_execution_layer(target_df, returns_pct, max_wt=MAX_WEIGHT, rebal_every=1,
                         cov_lookback=120, track_aversion=1.0, risk_aversion=0.0,
                         tc_bps=5.0, dollar_neutral=False, verbose=False):
    """
    Asset-level QP execution layer — Isichenko Eq 6.4 (tracking-error formulation).

        max_P  -A·||P - target||²  -  k·P'CP  -  (tc_bps/1e4)·||P - P_prev||_1
        s.t.   |P_s| ≤ max_wt
               sum(P) = 0      (optional, dollar-neutral)

    Equivalently (Isichenko Eq 6.5: forecast f = 2A·P*):
        max_P  2A·target·P  -  P'(A·I + k·C)P  -  (tc_bps/1e4)·||P - P_prev||_1

    Limits:
        TC → 0, k → 0  ⇒  P → target  (perfect tracking)
        TC → ∞         ⇒  P → P_prev  (no trading)
        k  → ∞         ⇒  P shrinks toward 0 along low-risk directions

    Inputs
    ------
    target_df       : (T, N)  aggregate target portfolio at each bar
    returns_pct     : (T, N)  one-period asset returns (for risk model)
    max_wt          : per-name position cap
    rebal_every     : bars between QP solves; positions are held in between
    cov_lookback    : rolling window for asset return covariance
    track_aversion  : A — quadratic penalty on tracking error (default 1.0)
    risk_aversion   : k — quadratic penalty on portfolio variance (default 0.0 = off)
    tc_bps          : linear trade cost in bps (per unit traded, one-sided)

    Returns
    -------
    P_df            : (T, N) DataFrame of post-QP portfolio weights at each bar.
    """
    import cvxpy as cp
    dates = target_df.index
    tickers = target_df.columns
    T, N = len(dates), len(tickers)
    target_arr = target_df.values
    ret_arr = returns_pct.reindex(index=dates, columns=tickers).values

    P_arr = np.zeros((T, N))
    P_prev = np.zeros(N)
    n_optimal = 0; n_skip = 0; n_fallback = 0

    for t in range(cov_lookback, T):
        if (t - cov_lookback) % rebal_every != 0:
            P_arr[t] = P_prev
            continue

        ret_win = ret_arr[t - cov_lookback : t]
        target_t = target_arr[t]
        finite_t = np.isfinite(target_t)
        finite_r = np.isfinite(ret_win).mean(axis=0) >= 0.5
        active = finite_t & finite_r
        N_a = int(active.sum())
        if N_a < 5 or np.abs(target_t[active]).sum() < 1e-12:
            P_arr[t] = P_prev; n_skip += 1; continue

        target_a = np.nan_to_num(target_t[active], nan=0.0)
        Pp_a = P_prev[np.where(active)[0]]

        P = cp.Variable(N_a)
        # Tracking-error (always on): -A·||P - target||²
        obj_terms = [-track_aversion * cp.sum_squares(P - target_a)]
        # Risk model term (off by default): -k·P'CP
        if risk_aversion > 0:
            ret_win_a = np.nan_to_num(ret_win[:, active], nan=0.0)
            Sigma = np.cov(ret_win_a.T) + 1e-6 * np.eye(N_a)
            obj_terms.append(-risk_aversion * cp.quad_form(P, cp.psd_wrap(Sigma)))
        # L1 trade cost
        if tc_bps > 0:
            obj_terms.append(-(tc_bps / 10000.0) * cp.norm1(P - Pp_a))

        objective = cp.Maximize(cp.sum(obj_terms))
        constraints = [cp.abs(P) <= max_wt]
        if dollar_neutral:
            constraints.append(cp.sum(P) == 0)
        prob = cp.Problem(objective, constraints)

        solved = False
        for solver in [cp.CLARABEL, cp.ECOS, cp.SCS]:
            try:
                prob.solve(solver=solver, warm_start=True)
                if prob.status in ['optimal', 'optimal_inaccurate'] and P.value is not None \
                   and np.all(np.isfinite(P.value)):
                    solved = True; break
            except Exception:
                continue
        if solved:
            sol = P.value
            n_optimal += 1
        else:
            sol = Pp_a
            n_fallback += 1

        P_full = np.zeros(N)
        P_full[np.where(active)[0]] = sol
        P_arr[t] = P_full
        P_prev = P_full

    if verbose:
        print(f"  QP execution: {n_optimal} optimal, {n_fallback} fallback, {n_skip} skip", flush=True)
    return pd.DataFrame(P_arr, index=dates, columns=tickers)


def strategy_billions_qp(raw_signals, returns_pct, close, universe,
                          max_wt=MAX_WEIGHT, optim_lookback=120, qp_lookback=120, rebal_every=1,
                          track_aversion=1.0, risk_aversion=0.0, tc_bps=None,
                          ema_halflife=60, **kwargs):
    """If tc_bps is None, defaults to the outer fees_bps so the QP solves the same
    problem the simulator scores (correct net-utility framing per Isichenko §6.3)."""
    """
    Billions aggregator + asset-level QP execution layer (Isichenko Eq 6.6).

    Stage 1 — Billions regression on alpha factor returns (same as before),
              produces time-varying alpha-weights → aggregate target portfolio f(t).
    Stage 2 — Asset-level QP execution layer
              max_P  f·P − k·P'CP − (tc_bps/1e4)·||P-P_prev||_1
              s.t.   |P_s| ≤ max_wt
              C = rolling covariance of underlying asset (futures) returns.
    """
    import cvxpy as cp
    from sklearn import linear_model

    # ── 0. Normalize signals ─────────────────────────────────────────────────
    normed_signals = {
        aid: proper_normalize_alpha(raw, universe, max_wt=max_wt)
        for aid, raw in raw_signals.items()
    }

    dates    = close.index
    tickers  = close.columns.tolist()
    n_bars   = len(dates)
    aid_list = list(normed_signals.keys())
    n_alphas = len(aid_list)

    # ── 1. Factor returns (lagged 1 bar — no lookahead) ──────────────────────
    ret_df = returns_pct.reindex(index=dates, columns=tickers)
    fr_data = {}
    for aid, norm in normed_signals.items():
        lagged = norm.shift(1)
        ab = lagged.abs().sum(axis=1).replace(0, np.nan)
        n  = lagged.div(ab, axis=0)
        fr_data[aid] = (n * ret_df).sum(axis=1)
    fr_df = pd.DataFrame(fr_data, index=dates)

    # ── 2. Billions walk-forward → store estimated μ at each bar ─────────────
    #   At bar test_start, uses window [test_start, test_start+L).
    #   Result stored at optim_end+1 = test_start+L+1  → zero lookahead.
    min_periods = max(1, optim_lookback // 2)
    seed_exp_ret = (
        fr_df.rolling(window=optim_lookback, min_periods=min_periods)
             .mean().shift(1).clip(lower=0)
    )

    # billions_mu[t] = Billions-estimated expected factor return at bar t
    billions_mu = seed_exp_ret.copy()   # initialised to rolling mean (equal to seed)

    reg_bil = linear_model.LinearRegression(fit_intercept=False)

    for test_start in range(1, n_bars - optim_lookback - 2):
        optim_end = test_start + optim_lookback
        if optim_end + 1 >= n_bars:
            break
        try:
            bil_df   = fr_df.iloc[test_start:optim_end].copy()
            demeaned = bil_df - bil_df.mean(axis=0)
            sample_std = demeaned.std(axis=0).replace(0, np.nan)
            if sample_std.isna().any():
                continue
            normalized = demeaned.divide(sample_std)
            A_is    = normalized.fillna(0.0)
            sub_exp = seed_exp_ret.iloc[test_start:optim_end].divide(sample_std).fillna(0.0)
            reg_bil.fit(A_is.values, sub_exp.values)
            residuals = pd.DataFrame(
                reg_bil.predict(A_is.values) - sub_exp.values,
                index=sub_exp.index, columns=sub_exp.columns
            )
            opt_w    = residuals.divide(sample_std)
            row_sums = opt_w.sum(axis=1).replace(0, np.nan)
            opt_w    = opt_w.div(row_sums, axis=0)
            # The last row's weights imply a μ: w_i > 0 → factor i is favoured
            # Store the implied signal at optim_end+1 (no lookahead)
            billions_mu.iloc[optim_end + 1] = opt_w.iloc[-1].values
        except Exception:
            pass

    # ── 3. EMA-smooth Billions alpha-weights, combine into target portfolio ──
    weights_smooth = billions_mu.ewm(halflife=ema_halflife, min_periods=1).mean()
    wsum           = weights_smooth.sum(axis=1).replace(0, np.nan)
    weights_norm   = weights_smooth.div(wsum, axis=0).fillna(1.0 / n_alphas)

    target = None
    for aid, normed in normed_signals.items():
        w  = weights_norm[aid].values
        ws = normed.multiply(w, axis=0)
        target = ws if target is None else target.add(ws, fill_value=0)

    # ── 4. Asset-level QP execution (Isichenko Eq 6.4 — tracking-error) ─────
    eff_fees = kwargs.get("fees_bps", kwargs.get("fee_bps", VAL_FEES))
    eff_tc_bps = eff_fees if tc_bps is None else tc_bps
    P_df = _qp_execution_layer(
        target_df=target,
        returns_pct=returns_pct,
        max_wt=max_wt,
        rebal_every=rebal_every,
        cov_lookback=qp_lookback,
        track_aversion=track_aversion,
        risk_aversion=risk_aversion,
        tc_bps=eff_tc_bps,
        verbose=False,
    )

    return (simulate(P_df, returns_pct, close, universe, max_wt=max_wt,
                     fees_bps=eff_fees),
            f"BillionsQP(mw={max_wt},ol={optim_lookback},ql={qp_lookback},A={track_aversion},k={risk_aversion},tc_bps={eff_tc_bps})")


# ============================================================================
# AIPT-SDF ON ALPHA SIGNALS (Didisheim-Ke-Kelly-Malamud 2025)
# ============================================================================

_AIPT_GAMMA_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _aipt_rff_params(K, P, seed=42):
    rng = np.random.default_rng(seed)
    n_pairs = P // 2
    theta = rng.standard_normal((n_pairs, K))
    gamma = rng.choice(_AIPT_GAMMA_GRID, size=n_pairs)
    return theta, gamma


def _aipt_rff_signals(Z, theta, gamma, P):
    """Z: (N_valid, K) -> S: (N_valid, P)"""
    proj = Z @ theta.T * gamma[np.newaxis, :]   # (N, P//2)
    S = np.empty((Z.shape[0], P), dtype=np.float64)
    S[:, 0::2] = np.sin(proj)
    S[:, 1::2] = np.cos(proj)
    return S


def _aipt_ridge(F, z):
    """Ridge-Markowitz: lambda = (zI + E[FF'])^{-1} E[F]. Woodbury for P>T."""
    T, P = F.shape
    mu = F.mean(axis=0)
    if P <= T:
        A = z * np.eye(P) + (F.T @ F) / T
        try:    return np.linalg.solve(A, mu)
        except: return np.linalg.lstsq(A, mu, rcond=None)[0]
    FFT = F @ F.T
    A_T = z * T * np.eye(T) + FFT
    F_mu = F @ mu
    try:    inv_F_mu = np.linalg.solve(A_T, F_mu)
    except: inv_F_mu = np.linalg.lstsq(A_T, F_mu, rcond=None)[0]
    return (mu - F.T @ inv_F_mu) / z


def strategy_aipt_sdf(raw_signals, returns_pct, close, universe,
                      P=200, z=1e-3, seed=42, rebal_every=12,
                      train_bars=1000, min_train_bars=300,
                      max_wt=MAX_WEIGHT, **kwargs):
    """
    AIPT Random Fourier Feature SDF portfolio applied to alpha signal outputs.

    Treats the K alpha outputs as cross-sectional characteristics (analogous to
    price-based features in Didisheim-Ke-Kelly-Malamud 2025), applies P Random
    Fourier Features, then estimates SDF weights via ridge-Markowitz (Eq. 9).

    Parameters
    ----------
    P           : random Fourier features (default 200)
    z           : ridge penalty (default 1e-3)
    rebal_every : bars between lambda re-estimation (default 12)
    train_bars  : rolling training window length in bars (default 1000 ~ 6mo)
    min_train_bars : minimum bars before first estimation (default 300 ~ 50 days)
    """
    from scipy.stats import rankdata as _rankdata

    alpha_ids = sorted(raw_signals.keys())
    K = len(alpha_ids)
    if K == 0:
        return None, "AIPT-SDF(no alphas)"

    tickers = returns_pct.columns.tolist()
    N = len(tickers)
    dates = returns_pct.index
    T = len(dates)

    returns_np = returns_pct.reindex(columns=tickers).values.astype(np.float64)

    # ── Build Z panel (T, N, K): alpha signals as rank-standardized characteristics
    Z_panel = np.zeros((T, N, K), dtype=np.float64)
    for k, aid in enumerate(alpha_ids):
        sig = raw_signals[aid].reindex(index=dates, columns=tickers).values.astype(np.float64)
        Z_panel[:, :, k] = sig

    # Cross-sectionally rank-standardize each (t, k) to [-0.5, 0.5]
    for t in range(T):
        for k in range(K):
            col = Z_panel[t, :, k]
            valid = ~np.isnan(col)
            n_v = valid.sum()
            if n_v < 3:
                Z_panel[t, :, k] = 0.0
                continue
            r = _rankdata(col[valid], method='average') / n_v - 0.5
            out = np.zeros(N)
            out[valid] = r
            Z_panel[t, :, k] = out

    theta, gamma = _aipt_rff_params(K, P, seed)

    # ── Pre-compute factor returns F_{t+1} = S_t' R_{t+1} / sqrt(N_t)
    factor_returns  = {}   # return-bar index -> (P,) vector
    rff_cache       = {}   # signal-bar index -> (S_t, valid_mask, N_t)

    for t in range(T - 1):
        Z_t  = Z_panel[t]
        r_t1 = returns_np[t + 1, :]
        valid = (~np.isnan(r_t1)) & (~np.isnan(Z_t).any(axis=1))
        N_t = valid.sum()
        if N_t < 5:
            continue
        S_t   = _aipt_rff_signals(Z_t[valid], theta, gamma, P)
        r_cln = np.nan_to_num(r_t1[valid], nan=0.0)
        factor_returns[t + 1] = (1.0 / np.sqrt(N_t)) * (S_t.T @ r_cln)
        rff_cache[t] = (S_t, valid, N_t)

    # ── Rolling SDF estimation -> weight DataFrame
    all_fr_idx = sorted(factor_returns.keys())
    weights_np = np.zeros((T, N), dtype=np.float64)

    lambda_hat      = None
    bars_since_reb  = rebal_every   # force first estimation

    for oos_t in range(1, T):
        if bars_since_reb >= rebal_every or lambda_hat is None:
            train_idx = [i for i in all_fr_idx
                         if i < oos_t and i >= max(0, oos_t - train_bars)]
            if len(train_idx) < min_train_bars:
                bars_since_reb += 1
                continue
            F_train    = np.vstack([factor_returns[i] for i in train_idx])
            lambda_hat = _aipt_ridge(F_train, z)
            bars_since_reb = 0

        sig_bar = oos_t - 1
        if sig_bar not in rff_cache or lambda_hat is None:
            bars_since_reb += 1
            continue

        S_t, valid_mask, N_t = rff_cache[sig_bar]
        raw_w = np.zeros(N)
        raw_w[valid_mask] = (1.0 / np.sqrt(N_t)) * (S_t @ lambda_hat)
        weights_np[oos_t] = raw_w
        bars_since_reb += 1

    weight_df = pd.DataFrame(weights_np, index=dates, columns=tickers)
    result = simulate(weight_df, returns_pct, close, universe, max_wt=max_wt)
    return result, f"AIPT-SDF(K={K},P={P},z={z:.0e},rb={rebal_every},tb={train_bars})"


PROPER_STRATEGIES = {
    "billions": strategy_billions,
    "billions_qp": strategy_billions_qp,
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
    # Book-inspired CVXPY strategies
    "cvxpy_mv_utility": strategy_cvxpy_mv_utility,
    "cvxpy_ledoit_sharpe": strategy_cvxpy_ledoit_sharpe,
    "cvxpy_regime_utility": strategy_cvxpy_regime_utility,
    # Isichenko Ch 3/4/6: Kelly, MinVar Bayesian, HRP
    "kelly_optimal": strategy_kelly_optimal,
    "min_var_bayesian": strategy_min_variance_bayesian,
    "hrp": strategy_hrp,
    # AIPT SDF on alpha outputs
    "aipt_sdf": strategy_aipt_sdf,
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



# ============================================================================
# AIPT-SDF vs EQUAL WEIGHT — TRAIN / VAL / TEST COMPARISON
# ============================================================================

_SPLITS = {
    "TRAIN (IS)": (None,           "2024-09-01"),
    "VAL   (OOS)": ("2024-09-01",  "2025-03-01"),
    "TEST  (OOS)": ("2025-03-01",  None),
}


def _sim_on_slice(weight_df, returns_pct, close, universe, start, end,
                  fee_bps, max_wt):
    """Slice all DataFrames to [start:end] and simulate."""
    sl = slice(start, end)
    w   = weight_df.loc[sl]
    r   = returns_pct.loc[sl]
    c   = close.loc[sl]
    u   = universe.loc[sl]
    if len(w) < 30 or w.abs().sum().sum() < 1e-10:
        return None
    return simulate(w, r, c, u, fees_bps=fee_bps, max_wt=max_wt)


def _print_split_table(rows):
    """rows = list of (split, strategy_label, result_or_None)"""
    hdr = f"  {'Split':<14} {'Strategy':<50} {'Sharpe':>8} {'Fitness':>8} {'AnnRet%':>9} {'TO':>7} {'MaxDD':>7}"
    print(f"\n{'='*110}")
    print("  AIPT-SDF vs Equal-Weight — Train / Val / Test")
    print(f"{'='*110}")
    print(hdr)
    print(f"  {'-'*105}")
    for split, label, res in rows:
        if res is None:
            print(f"  {split:<14} {label:<50} {'—':>8}")
            continue
        print(f"  {split:<14} {label:<50} "
              f"{res.sharpe:>+8.3f} {res.fitness:>8.3f} "
              f"{res.returns_ann*100:>+9.1f}% {res.turnover:>7.3f} "
              f"{res.max_drawdown:>7.3f}")
    print(f"{'='*110}\n")


def run_aipt_train_val_test(P=200, z=1e-3, seed=42, rebal_every=12,
                             train_bars=1000, min_train_bars=300,
                             fee_bps=VAL_FEES, max_wt=MAX_WEIGHT):
    """
    Run AIPT-SDF vs ProperEqual across train / val / test periods.

    Loads the full alpha signal history (no date slice), computes:
      1. AIPT-SDF weights — rolling ridge-Markowitz on RFF-projected alpha signals
      2. ProperEqual weights — simple equal-weight with proper normalization

    Both use strictly causal (no-lookahead) rolling windows.
    Reports Sharpe, Fitness, Ann. Return, Turnover, and Max Drawdown per period.
    """
    from scipy.stats import rankdata as _rankdata

    print(f"\n{'='*70}")
    print(f"  AIPT-SDF vs Equal — Full Train/Val/Test")
    print(f"  P={P}  z={z:.0e}  rebal={rebal_every}  train_bars={train_bars}")
    print(f"  fee={fee_bps}bps  max_wt={max_wt}")
    print(f"{'='*70}")

    print("\nLoading full alpha signals...")
    raw_signals, returns_pct, close, universe = load_full_alpha_signals()
    if raw_signals is None:
        return

    alpha_ids = sorted(raw_signals.keys())
    K = len(alpha_ids)
    tickers = returns_pct.columns.tolist()
    N = len(tickers)
    dates = returns_pct.index
    T = len(dates)
    print(f"  K={K} alphas, N={N} tickers, T={T} bars  "
          f"({dates[0]} to {dates[-1]})")

    returns_np = returns_pct.reindex(columns=tickers).values.astype(np.float64)

    # ── Build Z panel (T, N, K) ──────────────────────────────────────────────
    print("Building Z panel...")
    Z_panel = np.zeros((T, N, K), dtype=np.float64)
    for k, aid in enumerate(alpha_ids):
        sig = raw_signals[aid].reindex(index=dates, columns=tickers).values.astype(np.float64)
        Z_panel[:, :, k] = sig
    for t in range(T):
        for k in range(K):
            col = Z_panel[t, :, k]
            valid = ~np.isnan(col)
            n_v = valid.sum()
            if n_v < 3:
                Z_panel[t, :, k] = 0.0
                continue
            r = _rankdata(col[valid], method='average') / n_v - 0.5
            out = np.zeros(N)
            out[valid] = r
            Z_panel[t, :, k] = out

    # ── AIPT factor returns + rolling weights ────────────────────────────────
    print(f"Computing AIPT factor returns (P={P})...")
    theta, gamma = _aipt_rff_params(K, P, seed)
    factor_returns = {}
    rff_cache      = {}
    for t in range(T - 1):
        Z_t   = Z_panel[t]
        r_t1  = returns_np[t + 1, :]
        valid = (~np.isnan(r_t1)) & (~np.isnan(Z_t).any(axis=1))
        N_t   = valid.sum()
        if N_t < 5:
            continue
        S_t = _aipt_rff_signals(Z_t[valid], theta, gamma, P)
        r_c = np.nan_to_num(r_t1[valid], nan=0.0)
        factor_returns[t + 1] = (1.0 / np.sqrt(N_t)) * (S_t.T @ r_c)
        rff_cache[t] = (S_t, valid, N_t)

    print("Rolling SDF estimation...")
    all_fr_idx  = sorted(factor_returns.keys())
    aipt_w_np   = np.zeros((T, N), dtype=np.float64)
    lambda_hat  = None
    bars_since  = rebal_every

    for oos_t in range(1, T):
        if bars_since >= rebal_every or lambda_hat is None:
            train_idx = [i for i in all_fr_idx
                         if i < oos_t and i >= max(0, oos_t - train_bars)]
            if len(train_idx) < min_train_bars:
                bars_since += 1
                continue
            F_train    = np.vstack([factor_returns[i] for i in train_idx])
            lambda_hat = _aipt_ridge(F_train, z)
            bars_since = 0
        sig_bar = oos_t - 1
        if sig_bar not in rff_cache or lambda_hat is None:
            bars_since += 1
            continue
        S_t, vmask, N_t = rff_cache[sig_bar]
        raw_w = np.zeros(N)
        raw_w[vmask] = (1.0 / np.sqrt(N_t)) * (S_t @ lambda_hat)
        aipt_w_np[oos_t] = raw_w
        bars_since += 1

    aipt_weights = pd.DataFrame(aipt_w_np, index=dates, columns=tickers)

    # ── ProperEqual combined signal (full period) ────────────────────────────
    print("Building ProperEqual signal...")
    eq_combined = None
    for aid, raw_sig in raw_signals.items():
        normed = proper_normalize_alpha(raw_sig, universe, max_wt=max_wt)
        eq_combined = normed if eq_combined is None else eq_combined.add(normed, fill_value=0)

    # ── Evaluate each split ──────────────────────────────────────────────────
    rows = []
    aipt_label = f"AIPT-SDF(K={K},P={P},z={z:.0e},rb={rebal_every},tb={train_bars})"
    eq_label   = f"ProperEqual(K={K},mw={max_wt})"

    for split_name, (start, end) in _SPLITS.items():
        for label, wdf in [(aipt_label, aipt_weights), (eq_label, eq_combined)]:
            res = _sim_on_slice(wdf, returns_pct, close, universe,
                                start, end, fee_bps, max_wt)
            rows.append((split_name, label, res))

    _print_split_table(rows)

    # ── Plot cumulative returns per split ────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
        fig.patch.set_facecolor("#16213e")

        split_items = list(_SPLITS.items())
        for ax, (split_name, (start, end)) in zip(axes, split_items):
            ax.set_facecolor("#1a1a2e")
            for sp in ax.spines.values():
                sp.set_color("#333")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")

            for label, wdf, color in [
                (aipt_label, aipt_weights, "#FF5722"),
                (eq_label,   eq_combined,  "#2196F3"),
            ]:
                res = _sim_on_slice(wdf, returns_pct, close, universe,
                                    start, end, fee_bps, max_wt)
                if res is None or res.daily_pnl is None:
                    continue
                cum = res.daily_pnl.cumsum()
                short_label = "AIPT-SDF" if "AIPT" in label else "EqualWt"
                sr = res.sharpe
                ax.plot(cum.index, cum.values, linewidth=1.5,
                        color=color, label=f"{short_label}  SR={sr:+.2f}")

            ax.set_title(split_name, fontsize=11, fontweight="bold")
            ax.set_xlabel("Date")
            ax.set_ylabel("Cum PnL ($)")
            ax.legend(facecolor="#1a1a2e", edgecolor="#555",
                      labelcolor="white", fontsize=8)
            ax.grid(True, alpha=0.15)

        plt.suptitle(
            f"AIPT-SDF vs Equal-Weight on Alpha Signals  |  "
            f"P={P}, z={z:.0e}, {fee_bps}bps fees",
            fontsize=12, fontweight="bold", color="white",
        )
        plt.tight_layout()
        out = Path("data/aipt_results/aipt_on_alphas.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
        print(f"  Plot saved to {out}")
        plt.close()
    except Exception as e:
        print(f"  Plot failed: {e}")


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

        # ── CVXPY Book-Inspired Strategies (Isichenko) ─────────────────────────
        # CVXPYMVUtility: Mean-variance utility with L1 turnover penalty
        for mw in [0.02, 0.03]:
            for lb in [240, 280]:
                for k in [1.0, 2.0, 4.0]:
                    for tc in [0.1, 0.5, 1.0]:
                        for hl in [45, 90]:
                            try:
                                result, label = strategy_cvxpy_mv_utility(
                                    raw_signals, returns_pct, close, universe,
                                    max_wt=mw, lookback=lb, rebal_every=12,
                                    risk_aversion=k, tc_penalty=tc,
                                    decay=2, ema_halflife=hl)
                                results.append((label, result.sharpe, result.fitness, result.turnover, result.max_drawdown))
                            except Exception as e:
                                print(f"  CVXPYMVUtil(mw={mw},lb={lb},k={k},tc={tc}) failed: {e}")

        # CVXPYLedoitSharpe: Max-Sharpe with Ledoit-Wolf shrinkage
        for mw in [0.02, 0.03]:
            for lb in [240, 280]:
                for l2 in [0.001, 0.01, 0.05]:
                    for hl in [45, 90]:
                        try:
                            result, label = strategy_cvxpy_ledoit_sharpe(
                                raw_signals, returns_pct, close, universe,
                                max_wt=mw, lookback=lb, rebal_every=12,
                                l2_lambda=l2, decay=2, ema_halflife=hl)
                            results.append((label, result.sharpe, result.fitness, result.turnover, result.max_drawdown))
                        except Exception as e:
                            print(f"  CVXPYLedoitSharpe(mw={mw},lb={lb},l2={l2}) failed: {e}")

        # CVXPYRegimeUtility: Regime-adaptive risk aversion (fuses MV utility + RegimeScaled)
        for mw in [0.02, 0.03]:
            for lb in [240, 280]:
                for base_k in [1.0, 2.0]:
                    for rs in [2.0, 3.0, 5.0]:
                        for hl in [45, 90]:
                            try:
                                result, label = strategy_cvxpy_regime_utility(
                                    raw_signals, returns_pct, close, universe,
                                    max_wt=mw, lookback=lb, rebal_every=12,
                                    base_risk_aversion=base_k, regime_scale=rs,
                                    decay=2, ema_halflife=hl)
                                results.append((label, result.sharpe, result.fitness, result.turnover, result.max_drawdown))
                            except Exception as e:
                                print(f"  CVXPYRegimeUtil(mw={mw},lb={lb},k={base_k},rs={rs}) failed: {e}")

        # ── Kelly, MinVar Bayesian, HRP ───────────────────────────────────────
        for mw in [0.02, 0.03]:
            for lb in [240, 280]:
                for kf in [0.3, 0.5, 0.7]:
                    for hl in [45, 90]:
                        try:
                            result, label = strategy_kelly_optimal(
                                raw_signals, returns_pct, close, universe,
                                max_wt=mw, lookback=lb, rebal_every=12,
                                kelly_fraction=kf, decay=2, ema_halflife=hl)
                            results.append((label, result.sharpe, result.fitness, result.turnover, result.max_drawdown))
                        except Exception as e:
                            print(f"  KellyOptimal(mw={mw},lb={lb},kf={kf}) failed: {e}")

        for mw in [0.02, 0.03]:
            for lb in [240, 280]:
                for hl in [45, 90]:
                    try:
                        result, label = strategy_min_variance_bayesian(
                            raw_signals, returns_pct, close, universe,
                            max_wt=mw, lookback=lb, rebal_every=12,
                            decay=2, ema_halflife=hl)
                        results.append((label, result.sharpe, result.fitness, result.turnover, result.max_drawdown))
                    except Exception as e:
                        print(f"  MinVarBayesian(mw={mw},lb={lb}) failed: {e}")

        for mw in [0.02, 0.03]:
            for lb in [240, 280]:
                for rb in [12, 24]:
                    for hl in [45, 90]:
                        try:
                            result, label = strategy_hrp(
                                raw_signals, returns_pct, close, universe,
                                max_wt=mw, lookback=lb, rebal_every=rb,
                                decay=2, ema_halflife=hl)
                            results.append((label, result.sharpe, result.fitness, result.turnover, result.max_drawdown))
                        except Exception as e:
                            print(f"  HRP(mw={mw},lb={lb},rb={rb}) failed: {e}")

    # Sort by Sharpe
    results.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  {'Strategy':<45} {'Sharpe':>8} {'Fitness':>8} {'TO':>6} {'DD':>8}")
    print(f"  {'-'*80}")
    for label, sr, fit, to, dd in results:
        print(f"  {label:<45} {sr:+8.3f} {fit:8.3f} {to:6.3f} {dd:8.3f}")

    import csv
    with open('data/portfolio_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Strategy', 'Sharpe', 'Fitness', 'TO', 'DD'])
        for row in results:
            writer.writerow([row[0], f"{row[1]:.3f}", f"{row[2]:.3f}", f"{row[3]:.3f}", f"{row[4]:.3f}"])


# ============================================================================
# VALIDATION FRAMEWORK  (no test set — Isichenko Ch.7 + de Prado CPCV)
# ============================================================================

def _bootstrap_sharpe_ci(daily_pnl: pd.Series, n_boot: int = 2000,
                          block_size: int = 5, ci: float = 0.90) -> tuple:
    """
    Circular block bootstrap 90% CI for annualised Sharpe ratio.
    Block bootstrap (not IID) preserves autocorrelation structure.
    block_size=5 days ~ 30 bars at 4h — captures weekly seasonality.
    """
    arr = daily_pnl.values
    n = len(arr)
    if n < 10:
        return float('nan'), float('nan')

    boot_srs = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        # Circular block bootstrap
        starts = rng.integers(0, n, size=(n // block_size) + 2)
        idx = np.concatenate([np.arange(s, s + block_size) % n for s in starts])[:n]
        sample = arr[idx]
        m, s = sample.mean(), sample.std(ddof=1)
        boot_srs.append((m / s * np.sqrt(252)) if s > 1e-12 else 0.0)

    boot_srs = np.array(boot_srs)
    lo = float(np.percentile(boot_srs, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_srs, (1 + ci) / 2 * 100))
    return lo, hi


def _deflated_sharpe(observed_sr: float, n_trials: int, n_obs: int,
                     pnl_skew: float = 0.0, pnl_kurt_excess: float = 0.0) -> float:
    """
    Probabilistic / Deflated Sharpe Ratio (Bailey & López de Prado 2012,
    Isichenko Eq 7.x).

    Computes P(true SR > 0 | observed SR, T, N_trials) after adjusting
    for non-normality (skew/kurtosis) and multiple testing.

    Expected max SR under H0 (all strategies noise):
      SR* ≈ sqrt(2 * log(N) / T)  [Isichenko simplified form]
    """
    from scipy.stats import norm
    if n_obs < 5 or n_trials < 1:
        return float('nan')

    # Benchmark: expected max SR from N independent noise strategies
    sr_bench = np.sqrt(2.0 * np.log(max(n_trials, 2)) / n_obs) if n_trials >= 2 else 0.0

    # SR estimator variance adjusted for non-normality (Mertens 2002)
    sr_var = (1.0
              + 0.5 * observed_sr ** 2
              - pnl_skew * observed_sr
              + (pnl_kurt_excess / 4.0) * observed_sr ** 2
              ) / max(n_obs, 1)
    sr_se = np.sqrt(max(sr_var, 1e-12))

    # P(true SR > benchmark)
    z = (observed_sr - sr_bench) / sr_se
    return float(norm.cdf(z))


def walk_forward_validate(strategy_name: str, n_folds: int = 4,
                           n_trials: int = 300, **kwargs):
    """
    Walk-forward validation within the validation set — NO test set used.

    Design (Isichenko Leave-Future-Out CV):
      The full validation period (VAL_START → VAL_END) is divided into
      n_folds equal time bands. Each band's performance is measured on its
      own PnL, giving an honest sub-period decomposition.

    Reported metrics:
      • Per-fold Sharpe ratio and its std (consistency)
      • 90% bootstrap CI for full-period Sharpe (block bootstrap)
      • Deflated SR: P(true edge > 0) after correcting for N strategy trials
      • OOS decay estimate: E[SR_out] ≈ SR_in / sqrt(1 + N/T)
        (Isichenko Ch.7 over-optimisation decay formula)

    Usage:
      python eval_portfolio.py --validate regime_scaled --n-trials 400
    """
    # ── Load signals and run strategy ──
    use_raw = strategy_name in PROPER_STRATEGIES
    if use_raw:
        raw_signals, returns_pct, close, universe = load_raw_alpha_signals()
        if raw_signals is None: return
        result, label = PROPER_STRATEGIES[strategy_name](
            raw_signals, returns_pct, close, universe, **kwargs)
    else:
        signals, returns_pct, close, universe = load_alpha_signals()
        if signals is None: return
        fn = STRATEGIES.get(strategy_name)
        if fn is None:
            print(f"Unknown strategy: {strategy_name}")
            return
        result, label = fn(signals, returns_pct, close, universe, **kwargs)

    daily_pnl = result.daily_pnl
    if daily_pnl is None or len(daily_pnl) < 20:
        print("Insufficient PnL data for validation.")
        return

    n_days = len(daily_pnl)
    fold_size = n_days // n_folds

    print(f"\n{'='*72}")
    print(f"  WALK-FORWARD VALIDATION  (no test set)")
    print(f"  Strategy : {label}")
    print(f"  Period   : {VAL_START} → {VAL_END}  ({n_days} days)")
    print(f"  Folds    : {n_folds}  ({fold_size} days each)")
    print(f"{'='*72}")

    # ── Sub-period fold analysis ──
    fold_srs = []
    print(f"\n  {'Fold':<6} {'Period':<30} {'SR':>7} {'Ann Ret':>9} {'DD':>8}")
    print(f"  {'-'*62}")
    for k in range(n_folds):
        s = k * fold_size
        e = (k + 1) * fold_size if k < n_folds - 1 else n_days
        fold = daily_pnl.iloc[s:e]
        if len(fold) < 5: continue

        m = fold.mean()
        v = fold.std(ddof=1)
        sr = (m / v * np.sqrt(252)) if v > 1e-12 else 0.0
        ann_ret = m * 252 / (result.fitness / (result.sharpe + 1e-8) if abs(result.sharpe) > 0.01 else 1.0)
        # Drawdown in this fold
        cum = fold.cumsum()
        dd = float((cum - cum.cummax()).min())

        d0 = fold.index[0].strftime('%y-%m-%d') if hasattr(fold.index[0], 'strftime') else str(fold.index[0])[:10]
        d1 = fold.index[-1].strftime('%y-%m-%d') if hasattr(fold.index[-1], 'strftime') else str(fold.index[-1])[:10]
        print(f"  Fold {k+1:<3}  {d0} → {d1}   {sr:>+7.3f}   {'N/A':>9}  {dd:>8.4f}")
        fold_srs.append(sr)

    if not fold_srs:
        print("  Not enough fold data.")
        return

    fold_arr = np.array(fold_srs)
    n_pos = int((fold_arr > 0).sum())
    sr_mu = float(fold_arr.mean())
    sr_sd = float(fold_arr.std(ddof=1)) if len(fold_arr) > 1 else 0.0
    consistency = max(0.0, 1.0 - sr_sd / (abs(sr_mu) + 1e-6))

    print(f"\n  ── Fold Consistency ──────────────────────────────────────────")
    print(f"  Mean fold SR   : {sr_mu:+.3f}")
    print(f"  Std  fold SR   : {sr_sd:.3f}  (↓ = more consistent)")
    print(f"  Positive folds : {n_pos} / {len(fold_arr)}")
    print(f"  Consistency    : {consistency:.3f}  (1.0 = perfectly flat PnL)")

    # ── Full-period summary ──
    print(f"\n  ── Full-Period Metrics ───────────────────────────────────────")
    print(f"  Sharpe         : {result.sharpe:+.3f}")
    print(f"  Fitness        : {result.fitness:.3f}")
    print(f"  Turnover       : {result.turnover:.3f}")
    print(f"  Max Drawdown   : {result.max_drawdown:.3f}")

    # ── Bootstrap CI ──
    lo, hi = _bootstrap_sharpe_ci(daily_pnl, n_boot=2000, block_size=5)
    sig_str = "✓ SIGNIFICANT" if lo > 0 else "✗ overlaps zero"
    print(f"\n  ── 90% Bootstrap CI for Sharpe (block bootstrap) ─────────")
    print(f"  [{lo:+.3f}, {hi:+.3f}]  →  {sig_str}")

    # ── Deflated SR / Multiple-Testing Adjustment ──
    pnl_skew   = float(daily_pnl.skew())
    pnl_kurt_x = float(daily_pnl.kurtosis())   # pandas returns excess kurtosis
    p_edge = _deflated_sharpe(
        observed_sr=result.sharpe,
        n_trials=n_trials,
        n_obs=n_days,
        pnl_skew=pnl_skew,
        pnl_kurt_excess=pnl_kurt_x,
    )
    sr_benchmark = np.sqrt(2.0 * np.log(max(n_trials, 2)) / n_days)
    print(f"\n  ── Multiple-Testing Adjustment  (N={n_trials} trials tested) ──")
    print(f"  SR benchmark from chance : {sr_benchmark:+.3f}  [SR* = sqrt(2*ln(N)/T)]")
    print(f"  P(true edge > 0)         : {p_edge:.3f}  ",
          end="")
    print("← STRONG" if p_edge > 0.95 else "← MODERATE" if p_edge > 0.80 else "← WEAK")

    # ── OOS decay (Isichenko Eq 7: E[SR_out] ≈ SR_in/sqrt(1+N/T)) ──
    sr_oos = result.sharpe / np.sqrt(1.0 + n_trials / n_days)
    print(f"\n  ── OOS Decay Estimate  [Isichenko Eq 7.x] ────────────────")
    print(f"  E[SR_out] ≈ {sr_oos:+.3f}   (expected true OOS Sharpe)")

    # ── Final verdict ──
    grade = ("EXCELLENT" if p_edge > 0.95 and consistency > 0.5 and lo > 0
             else "GOOD"    if p_edge > 0.80 and n_pos > len(fold_arr) // 2
             else "MARGINAL" if p_edge > 0.60
             else "WEAK")
    print(f"\n  ── VALIDATION VERDICT ────────────────────────────────────")
    print(f"  Grade : {grade}")
    print(f"  (SR={result.sharpe:.2f}, P(edge)={p_edge:.2f}, "
          f"Consistency={consistency:.2f}, 90%CI=[{lo:.2f},{hi:.2f}], "
          f"E[OOS SR]={sr_oos:.2f})")
    print(f"{'='*72}\n")

    return dict(label=label, full_sr=result.sharpe, fold_srs=fold_srs,
                sr_mean=sr_mu, sr_std=sr_sd, consistency=consistency,
                ci_lo=lo, ci_hi=hi, p_edge=p_edge, sr_oos=sr_oos, grade=grade)


def main():
    all_strategies = {**STRATEGIES, **PROPER_STRATEGIES}
    parser = argparse.ArgumentParser(description="Agent 2: Portfolio Construction (Validation Only)")
    parser.add_argument("--strategy", type=str, default="corr_select_adaptive",
                        choices=list(all_strategies.keys()),
                        help="Combination strategy")
    parser.add_argument("--compare", action="store_true", help="Compare all strategies")
    parser.add_argument("--scoreboard", action="store_true", help="Show portfolio scoreboard")
    parser.add_argument("--aipt", action="store_true",
                        help="AIPT-SDF vs Equal-Weight on alpha signals (train/val/test)")
    parser.add_argument("--validate", type=str, default=None, metavar="STRATEGY",
                        help="Walk-forward validate a strategy within the val set (no test set used)")
    parser.add_argument("--n-trials", type=int, default=300,
                        help="Number of strategy configs tried (for Deflated Sharpe). Default 300.")
    parser.add_argument("--n-folds", type=int, default=4,
                        help="Number of walk-forward folds for --validate. Default 4.")
    parser.add_argument("--lookback", type=int, default=LOOKBACK, help="Rolling lookback window")
    parser.add_argument("--top", type=int, default=TOP_N, help="Top N factors for top_n strategy")
    parser.add_argument("--decay", type=int, default=SIM_DECAY, help="Decay parameter")
    parser.add_argument("--mw", type=float, default=MAX_WEIGHT, help="Max weight for proper strategies")
    parser.add_argument("--hl", type=int, default=30, help="EMA halflife")
    parser.add_argument("--mc", type=float, default=0.3, help="Max corr (default 0.3)")
    parser.add_argument("--fees", type=float, default=VAL_FEES, help="Fees in bps")
    parser.add_argument("--db", type=float, default=0.01, help="Deadband threshold")
    args = parser.parse_args()

    strat_kwargs = dict(
        lookback=args.lookback, top_n=args.top, decay=args.decay,
        max_wt=args.mw, ema_halflife=args.hl, max_corr=args.mc,
        fee_bps=args.fees, deadband=args.db,
    )

    if args.aipt:
        run_aipt_train_val_test(fee_bps=args.fees, max_wt=args.mw)
        return

    if args.validate:
        walk_forward_validate(
            args.validate,
            n_folds=args.n_folds,
            n_trials=args.n_trials,
            **strat_kwargs,
        )
        return

    if args.compare or args.scoreboard:
        compare_all()
        return

    run_strategy(args.strategy, **strat_kwargs)


if __name__ == "__main__":
    main()
