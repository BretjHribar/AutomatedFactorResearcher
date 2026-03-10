"""
Walk-Forward Crypto Alpha Pipeline
===================================
Production pipeline based on CryptoRLQuantResearcher architecture (>2 Sharpe OOS).

Features:
  1. 56 long-lookback momentum/trend alphas (60-240 bar at 4h)
  2. Walk-forward alpha selection (train 720 / val 360 / re-eval every 6 bars)
  3. QP factor-return combination (45-bar rolling window)
  4. CVXPY portfolio optimization with PCA-3 hedging
  5. Dynamic vol targeting (20% target) and drawdown protection
  6. Proper 4h annualization: sqrt(6 * 252)

Usage:
  python run_crypto_walkforward.py [--days 480] [--fee-bps 5] [--top-n 50]
"""

import argparse
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ═════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════
ANN_FACTOR = np.sqrt(6 * 252)  # 4h bars → annual
BINANCE_URL = "https://fapi.binance.com"
CACHE_DIR = Path("data/walkforward_cache")


# ═════════════════════════════════════════════════════════════════
# 1. ALPHA DEFINITIONS — 56 proven expressions from CryptoRL
# ═════════════════════════════════════════════════════════════════
ALPHA_DEFINITIONS = [
    # ── Momentum / Trend (long lookback) ──
    ("double_smooth_30_90",   "rank(sma(sma(returns, 30), 90))"),
    ("double_smooth_45_120",  "rank(sma(sma(returns, 45), 120))"),
    ("mom_90bar",             "rank(sma(returns, 90))"),
    ("mom_120bar",            "rank(sma(returns, 120))"),
    ("mom_180bar",            "rank(sma(returns, 180))"),
    ("mom_240bar",            "rank(sma(returns, 240))"),

    # ── Donchian Channels ──
    ("donchian_90",  "rank(div(close - ts_min(close, 90), ts_max(close, 90) - ts_min(close, 90) + 1e-10))"),
    ("donchian_120", "rank(div(close - ts_min(close, 120), ts_max(close, 120) - ts_min(close, 120) + 1e-10))"),
    ("donchian_180", "rank(div(close - ts_min(close, 180), ts_max(close, 180) - ts_min(close, 180) + 1e-10))"),
    ("donchian_240", "rank(div(close - ts_min(close, 240), ts_max(close, 240) - ts_min(close, 240) + 1e-10))"),

    # ── Trend Strength ──
    ("trend_strength_60",  "rank(div(sma(returns, 60), sma(abs(returns), 60) + 1e-10))"),
    ("trend_strength_90",  "rank(div(sma(returns, 90), sma(abs(returns), 90) + 1e-10))"),
    ("trend_strength_120", "rank(div(sma(returns, 120), sma(abs(returns), 120) + 1e-10))"),

    # ── Momentum Diff (MACD-like) ──
    ("mom_diff_30_90",  "rank(sma(returns, 30) - sma(returns, 90))"),
    ("mom_diff_60_180", "rank(sma(returns, 60) - sma(returns, 180))"),
    ("mom_diff_90_270", "rank(sma(returns, 90) - sma(returns, 270))"),

    # ── Risk-Adjusted Momentum (Sharpe-style) ──
    ("sharpe_60",  "rank(div(sma(returns, 60), stddev(returns, 60) + 1e-10))"),
    ("sharpe_90",  "rank(div(sma(returns, 90), stddev(returns, 90) + 1e-10))"),
    ("sharpe_120", "rank(div(sma(returns, 120), stddev(returns, 120) + 1e-10))"),

    # ── Trend Vol-Adjusted ──
    ("trend_vol_adj_90",  "rank(mul(sma(returns, 90), div(1, stddev(returns, 90) + 1e-10)))"),
    ("trend_vol_adj_120", "rank(mul(sma(returns, 120), div(1, stddev(returns, 120) + 1e-10)))"),

    # ── Volume-Weighted Momentum ──
    ("vol_weighted_mom_60",  "rank(sma(mul(returns, volume), 60))"),
    ("vol_weighted_mom_90",  "rank(sma(mul(returns, volume), 90))"),
    ("vol_weighted_mom_120", "rank(sma(mul(returns, volume), 120))"),

    # ── Short-Term Reversal ──
    ("reversal_1day", "-rank(sma(returns, 6))"),
    ("reversal_2day", "-rank(sma(returns, 12))"),
    ("reversal_3day", "-rank(sma(returns, 18))"),
    ("reversal_4day", "-rank(sma(returns, 24))"),
    ("reversal_5day", "-rank(sma(returns, 30))"),

    # ── Z-Score Reversal ──
    ("zscore_rev_30", "-rank(ts_zscore(close, 30))"),
    ("zscore_rev_45", "-rank(ts_zscore(close, 45))"),
    ("zscore_rev_60", "-rank(ts_zscore(close, 60))"),
    ("zscore_rev_90", "-rank(ts_zscore(close, 90))"),

    # ── Buy the Dip ──
    ("buy_dip_30", "rank(div(ts_max(high, 30) - close, ts_max(high, 30) + 1e-10))"),
    ("buy_dip_45", "rank(div(ts_max(high, 45) - close, ts_max(high, 45) + 1e-10))"),
    ("buy_dip_60", "rank(div(ts_max(high, 60) - close, ts_max(high, 60) + 1e-10))"),

    # ── Fade Big Moves ──
    ("fade_moves_1d", "-rank(ts_max(abs(returns), 6))"),
    ("fade_moves_2d", "-rank(ts_max(abs(returns), 12))"),
    ("fade_moves_3d", "-rank(ts_max(abs(returns), 18))"),

    # ── Low Volatility Premium ──
    ("low_vol_60",  "-rank(stddev(returns, 60))"),
    ("low_vol_90",  "-rank(stddev(returns, 90))"),
    ("low_vol_120", "-rank(stddev(returns, 120))"),

    # ── On-Balance Volume ──
    ("obv_60",  "rank(ts_sum(mul(sign(returns), volume), 60))"),
    ("obv_90",  "rank(ts_sum(mul(sign(returns), volume), 90))"),
    ("obv_120", "rank(ts_sum(mul(sign(returns), volume), 120))"),

    # ── Correlation Based ──
    ("corr_ret_vol_60",  "rank(correlation(returns, volume, 60))"),
    ("corr_ret_vol_90",  "rank(correlation(returns, volume, 90))"),
    ("hl_corr_90",       "rank(correlation(high, low, 90))"),
    ("hl_corr_120",      "rank(correlation(high, low, 120))"),

    # ── Price vs Moving Average ──
    ("price_vs_ma120", "rank(div(close, sma(close, 120)))"),
    ("price_vs_ma180", "rank(div(close, sma(close, 180)))"),
    ("ma30_vs_ma90",   "rank(div(sma(close, 30), sma(close, 90)))"),
    ("ma60_vs_ma180",  "rank(div(sma(close, 60), sma(close, 180)))"),

    # ── Range / Candle Based ──
    ("close_location_60", "rank(sma(div(close - low, high - low + 1e-10), 60))"),
    ("close_location_90", "rank(sma(div(close - low, high - low + 1e-10), 90))"),
    ("oc_position_60",    "rank(sma(div(close - open, high - low + 1e-10), 60))"),
]


# ═════════════════════════════════════════════════════════════════
# 2. OPERATORS — vectorized pandas implementations
# ═════════════════════════════════════════════════════════════════

def op_rank(df):
    """Cross-sectional rank → [0, 1]."""
    return df.rank(axis=1, pct=True)

def op_sma(df, window):
    return df.rolling(window, min_periods=1).mean()

def op_stddev(df, window):
    return df.rolling(window, min_periods=2).std()

def op_ts_min(df, window):
    return df.rolling(window, min_periods=1).min()

def op_ts_max(df, window):
    return df.rolling(window, min_periods=1).max()

def op_ts_sum(df, window):
    return df.rolling(window, min_periods=1).sum()

def op_ts_zscore(df, window):
    mean = df.rolling(window, min_periods=2).mean()
    std = df.rolling(window, min_periods=2).std()
    return (df - mean) / std.replace(0, np.nan)

def op_correlation(a, b, window):
    return a.rolling(window, min_periods=2).corr(b)

def op_sign(df):
    return np.sign(df)

def op_abs(df):
    return df.abs()

def op_div(a, b):
    """Protected division matching CryptoRL: returns 0 for inf/NaN."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = a / b
        result = result.replace([np.inf, -np.inf], 0.0) if isinstance(result, pd.DataFrame) else result
        return result.fillna(0.0) if isinstance(result, (pd.DataFrame, pd.Series)) else result

def op_mul(a, b):
    return a * b

def op_delta(df, period):
    return df - df.shift(period)

def op_delay(df, period):
    return df.shift(period)

def op_ts_argmax(df, window):
    return df.rolling(window, min_periods=1).apply(lambda x: np.argmax(x), raw=True)

def op_ts_argmin(df, window):
    return df.rolling(window, min_periods=1).apply(lambda x: np.argmin(x), raw=True)


def evaluate_expression(expr_str: str, features: dict) -> pd.DataFrame | None:
    """Evaluate an alpha expression string against feature DataFrames."""
    ctx = {
        "rank": op_rank, "sma": op_sma, "stddev": op_stddev,
        "ts_min": op_ts_min, "ts_max": op_ts_max, "ts_sum": op_ts_sum,
        "ts_zscore": op_ts_zscore, "correlation": op_correlation,
        "sign": op_sign, "abs": op_abs, "div": op_div, "mul": op_mul,
        "delta": op_delta, "delay": op_delay,
        "ts_argmax": op_ts_argmax, "ts_argmin": op_ts_argmin,
        **features,
    }
    try:
        result = eval(expr_str, {"__builtins__": {}}, ctx)
        if isinstance(result, pd.DataFrame):
            return result
    except Exception:
        pass
    return None


# ═════════════════════════════════════════════════════════════════
# 3. DATA LOADING — from verified cached matrices
# ═════════════════════════════════════════════════════════════════

def load_data(n_symbols: int = 50, n_days: int = 480, interval: str = "4h") -> dict:
    """Load 4h data from pre-built verified matrices."""
    print(f"\n{'='*70}")
    print(f"LOADING DATA: TOP{n_symbols} universe, {interval}")
    print(f"{'='*70}")

    data_dir = Path("data/binance_cache")
    matrices_dir = data_dir / "matrices" / interval

    # Load all matrices
    matrices = {}
    for fpath in sorted(matrices_dir.glob("*.parquet")):
        matrices[fpath.stem] = pd.read_parquet(fpath)
    print(f"  {len(matrices)} matrices loaded from {matrices_dir}")

    # Load universe
    suffix = f"_{interval}" if interval != "1d" else ""
    uni_path = data_dir / f"universes/BINANCE_TOP{n_symbols}{suffix}.parquet"
    if not uni_path.exists():
        uni_path = data_dir / f"universes/BINANCE_TOP50{suffix}.parquet"
        print(f"  ⚠ TOP{n_symbols} not found, using TOP50")
    universe_df = pd.read_parquet(uni_path)

    # Filter to tickers with >10% coverage
    coverage = universe_df.sum(axis=0) / len(universe_df)
    valid_tickers = sorted(coverage[coverage > 0.1].index.tolist())
    print(f"  {len(valid_tickers)} tickers with >10% universe coverage")

    for name in list(matrices.keys()):
        cols = [c for c in valid_tickers if c in matrices[name].columns]
        if cols:
            matrices[name] = matrices[name][cols]
        else:
            del matrices[name]

    # Build features dict
    features = {}
    for name in ["open", "high", "low", "close", "volume", "quote_volume",
                  "returns", "adv20"]:
        if name in matrices:
            features[name] = matrices[name]

    if "returns" not in features and "close" in features:
        features["returns"] = features["close"].pct_change()
    if "adv20" not in features and "volume" in features:
        features["adv20"] = op_sma(features["volume"], 30)

    n_bars = len(features["close"])
    idx = features["close"].index
    valid_syms = features["close"].columns.tolist()
    print(f"  {len(valid_syms)} symbols, {n_bars} bars ({idx[0].date()} -> {idx[-1].date()})")
    return features


# ═════════════════════════════════════════════════════════════════
# 4. WALK-FORWARD ALPHA SELECTION
# ═════════════════════════════════════════════════════════════════

def compute_alpha_sharpe(signal: pd.DataFrame, returns: pd.DataFrame,
                         start: int, end: int) -> float | None:
    """Compute Sharpe for a single alpha over [start, end) bar range."""
    if end - start < 60:
        return None
    sig = signal.iloc[start:end]
    ret = returns.iloc[start:end]
    # Market-neutral normalization
    sig_n = sig.sub(sig.mean(axis=1), axis=0)
    sig_n = sig_n.div(sig_n.abs().sum(axis=1) + 1e-10, axis=0)
    pnl = (sig_n.shift(1) * ret).sum(axis=1).dropna()
    if len(pnl) < 30 or pnl.std() < 1e-12:
        return None
    return float((pnl.mean() / pnl.std()) * ANN_FACTOR)


def compute_signal_corr(s1: pd.DataFrame, s2: pd.DataFrame,
                        start: int, end: int) -> float:
    """Average cross-sectional correlation of two signals over recent bars."""
    correlations = []
    for d in range(max(start + 30, end - 50), end):
        if d >= len(s1):
            continue
        v1 = s1.iloc[d].dropna()
        v2 = s2.iloc[d].dropna()
        common = v1.index.intersection(v2.index)
        if len(common) >= 10:
            c = v1[common].corr(v2[common])
            if not np.isnan(c):
                correlations.append(abs(c))
    return np.mean(correlations) if correlations else 1.0


def select_alphas(alpha_signals: dict, returns: pd.DataFrame, t: int,
                  train_bars: int = 720, val_bars: int = 360,
                  max_corr: float = 0.65, max_alphas: int = 12) -> list[str]:
    """Walk-forward alpha selection at time t."""
    train_start = t - train_bars - val_bars
    train_end = t - val_bars
    val_end = t

    if train_start < 200:
        return []

    # Score each alpha on train + validation periods
    metrics = {}
    for name, sig in alpha_signals.items():
        train_sr = compute_alpha_sharpe(sig, returns, train_start, train_end)
        val_sr = compute_alpha_sharpe(sig, returns, train_end, val_end)
        if train_sr is not None and val_sr is not None:
            metrics[name] = {"train": train_sr, "val": val_sr}

    # Filter: positive on both, sort by validation-weighted score
    passing = [n for n, m in metrics.items() if m["train"] > 0.3 and m["val"] > 0.0]
    passing.sort(key=lambda n: metrics[n]["val"] * 0.7 + metrics[n]["train"] * 0.3,
                 reverse=True)

    # Greedy orthogonal selection
    selected = []
    for name in passing:
        is_ortho = all(
            compute_signal_corr(alpha_signals[name], alpha_signals[s], train_start, val_end) <= max_corr
            for s in selected
        )
        if is_ortho:
            selected.append(name)
        if len(selected) >= max_alphas:
            break

    return selected


# ═════════════════════════════════════════════════════════════════
# 5. QP ALPHA COMBINATION
# ═════════════════════════════════════════════════════════════════

class QPCombiner:
    """Combine alphas using Quadratic Programming on factor returns."""

    def __init__(self, alpha_signals: dict, names: list[str],
                 returns: pd.DataFrame, lookback: int = 45):
        self.signals = alpha_signals
        self.names = names
        self.returns = returns
        self.lookback = lookback

        # Pre-compute factor returns for each selected alpha
        self.factor_returns = {}
        for name in self.names:
            sig = self.signals[name]
            sig_n = sig.sub(sig.mean(axis=1), axis=0)
            sig_n = sig_n.div(sig_n.abs().sum(axis=1) + 1e-10, axis=0)
            self.factor_returns[name] = (sig_n.shift(1) * self.returns).sum(axis=1)
        self.fr_df = pd.DataFrame(self.factor_returns)

    def get_weights(self, t_idx: int) -> np.ndarray:
        """Solve QP for alpha weights at time t."""
        n = len(self.names)
        if t_idx < self.lookback + 20 or n == 0:
            return np.ones(n) / max(n, 1)

        fr = self.fr_df.iloc[max(0, t_idx - self.lookback - 1):t_idx - 1]
        mu = fr.mean().values
        cov = fr.cov().values + 0.02 * np.eye(n)

        def objective(w):
            return -np.dot(w, mu) + 0.5 * np.dot(w, cov @ w)

        bounds = [(0, 1)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        w0 = np.ones(n) / n

        try:
            result = minimize(objective, w0, method="SLSQP",
                              bounds=bounds, constraints=constraints,
                              options={"maxiter": 50})
            return result.x if result.success else w0
        except Exception:
            return w0

    def combine(self, t_idx: int) -> tuple[pd.Series | None, int]:
        """Get combined signal at time t. Returns (signal, n_active)."""
        if not self.names:
            return None, 0

        weights = self.get_weights(t_idx)
        n_active = int(np.sum(weights > 0.05))

        combined = pd.Series(0.0, index=self.signals[self.names[0]].columns)
        for i, name in enumerate(self.names):
            if weights[i] > 0.01:
                combined += weights[i] * self.signals[name].iloc[t_idx - 1]

        # Market-neutral & normalized
        combined = combined.sub(combined.mean())
        combined = combined.div(combined.abs().sum() + 1e-10)
        return combined, n_active


# ═════════════════════════════════════════════════════════════════
# 6. CVXPY PORTFOLIO OPTIMIZER WITH PCA HEDGING
# ═════════════════════════════════════════════════════════════════

def estimate_covariance(returns: pd.DataFrame, shrinkage: float = 0.9) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance estimator."""
    r = returns.fillna(0).values
    n = r.shape[1]
    sample_cov = np.cov(r, rowvar=False)
    # Shrink toward diagonal
    diag = np.diag(np.diag(sample_cov))
    return (1 - shrinkage) * sample_cov + shrinkage * diag


def compute_pca_loadings(returns: pd.DataFrame, n_factors: int = 3) -> np.ndarray | None:
    """Compute top-N PCA loadings from historical returns."""
    r = returns.fillna(0).values
    if r.shape[0] < 30 or r.shape[1] < n_factors + 3:
        return None
    r_demeaned = r - r.mean(axis=0)
    try:
        cov = np.cov(r_demeaned, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Top n_factors eigenvectors (largest eigenvalues)
        idx = np.argsort(eigvals)[::-1][:n_factors]
        return eigvecs[:, idx]
    except Exception:
        return None


# Track optimizer stats globally
_optimizer_stats = {"converged": 0, "failed": 0, "fallback": 0}


def optimize_portfolio(signal: pd.Series, returns: pd.DataFrame,
                       current_holdings: pd.Series | None,
                       max_position: float = 0.04,
                       max_gross_lev: float = 0.80,
                       max_turnover: float = 0.08,
                       tx_cost_bps: float = 10.0,
                       risk_aversion: float = 1.0,
                       n_pca_factors: int = 3) -> tuple[pd.Series, bool]:
    """CVXPY portfolio optimization with PCA constraints.
    
    Returns (holdings, converged) tuple.
    Matches CryptoRL: falls back to forecast when optimizer fails.
    """
    assets = signal.index
    n = len(assets)
    r = returns.reindex(columns=assets).fillna(0)

    alpha = signal.fillna(0).values.astype(float)
    Sigma = estimate_covariance(r)
    pca = compute_pca_loadings(r, n_pca_factors)

    is_first_bar = current_holdings is None
    h_prev = np.zeros(n)
    if not is_first_bar:
        h_prev = current_holdings.reindex(assets, fill_value=0).values

    h = cp.Variable(n)
    trade = h - h_prev
    tx_cost = (tx_cost_bps * 1e-4) * cp.norm(trade, 1)

    objective = cp.Maximize(
        alpha @ h - (risk_aversion / 2) * cp.quad_form(h, Sigma) - tx_cost
    )

    constraints = [
        cp.sum(h) == 0,                            # Market neutral
        cp.norm(h, "inf") <= max_position,          # Position limit
        cp.norm(h, 1) <= max_gross_lev,             # Gross leverage
    ]

    # Only apply turnover constraint after first bar
    if not is_first_bar:
        constraints.append(cp.norm(trade, 1) <= max_turnover)

    # PCA neutrality: zero exposure to top factors
    if pca is not None:
        for k in range(pca.shape[1]):
            constraints.append(pca[:, k] @ h == 0)

    problem = cp.Problem(objective, constraints)
    try:
        # Try CLARABEL first (best SOCP solver available), fall back to SCS
        try:
            problem.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            problem.solve(solver=cp.SCS, verbose=False, max_iters=500)
        if problem.status in ("optimal", "optimal_inaccurate") and h.value is not None:
            _optimizer_stats["converged"] += 1
            return pd.Series(h.value, index=assets), True
        else:
            _optimizer_stats["failed"] += 1
    except Exception:
        _optimizer_stats["failed"] += 1

    # Fallback: use forecast directly as holdings (matching CryptoRL)
    _optimizer_stats["fallback"] += 1
    return signal.copy(), False


# ═════════════════════════════════════════════════════════════════
# 7. MAIN WALK-FORWARD BACKTEST
# ═════════════════════════════════════════════════════════════════

def run_backtest(features: dict, alpha_signals: dict,
                 fee_bps: float = 5.0,
                 train_bars: int = 720, val_bars: int = 360,
                 reeval_interval: int = 6) -> pd.DataFrame:
    """Run walk-forward backtest."""
    returns = features["returns"]
    close = features["close"]
    adv20 = features["adv20"]
    n_bars = len(close)

    start_bar = train_bars + val_bars + 200
    rolling_vol = returns.mean(axis=1).rolling(30).std() * ANN_FACTOR

    print(f"\n{'='*70}")
    print(f"WALK-FORWARD BACKTEST")
    print(f"{'='*70}")
    print(f"  Train: {train_bars} bars | Val: {val_bars} bars | Re-eval: every {reeval_interval} bars")
    print(f"  Fee: {fee_bps} bps one-way | Start bar: {start_bar}/{n_bars}")
    print(f"  Simulating {n_bars - start_bar} bars...", flush=True)

    results = []
    prev_holdings = None
    current_selected = []
    combiner = None
    equity = 1.0
    peak_equity = 1.0

    # Universe rebalancing: every 120 bars (~20 days for 4h) per KI spec
    REBAL_BARS = 120
    universe = None

    for t in range(start_bar, n_bars):
        # ── Alpha selection (every reeval_interval bars) ──
        if (t - start_bar) % reeval_interval == 0 or not current_selected:
            new_sel = select_alphas(alpha_signals, returns, t,
                                    train_bars, val_bars, max_corr=0.65, max_alphas=15)
            if new_sel:
                current_selected = new_sel
                combiner = QPCombiner(
                    {n: alpha_signals[n] for n in current_selected},
                    current_selected, returns, lookback=45
                )

        if not current_selected or combiner is None:
            continue

        # ── Universe: top 50 by trailing ADV, rebalanced every 120 bars (~20 days) ──
        if universe is None or (t - start_bar) % REBAL_BARS == 0:
            trailing = adv20.iloc[max(0, t - 60):t].mean()
            valid = trailing.dropna()
            if len(valid) < 20:
                continue
            universe = valid.nlargest(50).index.tolist()

        # ── Combine alphas → forecast ──
        forecast, n_active = combiner.combine(t)
        if forecast is None:
            continue
        forecast = forecast.reindex(universe).fillna(0)
        forecast = forecast.sub(forecast.mean())
        forecast = forecast.div(forecast.abs().sum() + 1e-10)

        # ── Dynamic scaling: vol target + drawdown protection ──
        cur_vol = rolling_vol.iloc[t] if t < len(rolling_vol) and not np.isnan(rolling_vol.iloc[t]) else 0.3
        vol_scale = min(1.0, 0.20 / (cur_vol + 0.01))

        dd = (equity - peak_equity) / peak_equity
        dd_scale = max(0.25, 1.0 + dd * 4) if dd < -0.05 else 1.0

        forecast = forecast * vol_scale * dd_scale

        # ── CVXPY optimize ──
        hist_ret = returns.iloc[max(0, t - 120):t]
        cur = prev_holdings.reindex(forecast.index, fill_value=0) if prev_holdings is not None else None

        holdings, converged = optimize_portfolio(
            signal=forecast, returns=hist_ret, current_holdings=cur,
            max_position=0.04, max_gross_lev=0.80, max_turnover=0.08,
            tx_cost_bps=fee_bps * 2, risk_aversion=1.0, n_pca_factors=3,
        )

        # ── Realize PnL ──
        actual_ret = returns.iloc[t].reindex(universe, fill_value=0)
        pnl_gross = float((holdings * actual_ret).sum())

        turnover = float((holdings - (prev_holdings.reindex(holdings.index, fill_value=0)
                          if prev_holdings is not None else 0)).abs().sum())
        pnl_net = pnl_gross - turnover * fee_bps * 1e-4

        equity *= (1 + pnl_net)
        peak_equity = max(peak_equity, equity)

        results.append({
            "datetime": close.index[t],
            "pnl_gross": pnl_gross,
            "pnl_net": pnl_net,
            "equity": equity,
            "turnover": turnover,
            "n_alphas": len(current_selected),
            "n_active": n_active,
            "gross_lev": float(holdings.abs().sum()),
        })
        prev_holdings = holdings.copy()

        if (t - start_bar) % 200 == 0 and t > start_bar:
            sr_so_far = (pd.Series([r["pnl_net"] for r in results]).mean() /
                         (pd.Series([r["pnl_net"] for r in results]).std() + 1e-12)) * ANN_FACTOR
            conv = _optimizer_stats["converged"]
            fail = _optimizer_stats["failed"]
            total_opt = conv + fail
            conv_rate = conv / total_opt * 100 if total_opt > 0 else 0
            print(f"  Bar {t}/{n_bars}: equity={equity:.3f}, SR={sr_so_far:+.2f}, "
                  f"alphas={len(current_selected)}, optimizer={conv_rate:.0f}% converged", flush=True)

    # Print optimizer stats
    conv = _optimizer_stats["converged"]
    fail = _optimizer_stats["failed"]
    total_opt = conv + fail
    print(f"\n  Optimizer: {conv}/{total_opt} converged ({conv/total_opt*100:.0f}%), "
          f"{_optimizer_stats['fallback']} fallbacks")

    return pd.DataFrame(results).set_index("datetime")


# ═════════════════════════════════════════════════════════════════
# 8. REPORTING & VISUALIZATION
# ═════════════════════════════════════════════════════════════════

def report(df: pd.DataFrame, fee_bps: float):
    """Print and plot results."""
    total_ret = df["equity"].iloc[-1] - 1
    sharpe = (df["pnl_net"].mean() / (df["pnl_net"].std() + 1e-12)) * ANN_FACTOR
    sharpe_gross = (df["pnl_gross"].mean() / (df["pnl_gross"].std() + 1e-12)) * ANN_FACTOR
    max_dd = ((df["equity"] - df["equity"].cummax()) / df["equity"].cummax()).min()
    avg_to = df["turnover"].mean()
    avg_alphas = df["n_alphas"].mean()
    avg_lev = df["gross_lev"].mean()

    # Recent period
    if len(df) > 500:
        recent = df.iloc[-400:]
        recent_sr = (recent["pnl_net"].mean() / (recent["pnl_net"].std() + 1e-12)) * ANN_FACTOR
    else:
        recent_sr = sharpe

    print(f"\n{'='*70}")
    print(f"RESULTS (fee = {fee_bps} bps one-way)")
    print(f"{'='*70}")
    print(f"  Sharpe (net):    {sharpe:+.2f}")
    print(f"  Sharpe (gross):  {sharpe_gross:+.2f}")
    print(f"  Recent 400-bar:  {recent_sr:+.2f}")
    print(f"  Total Return:    {total_ret:+.1%}")
    print(f"  Max Drawdown:    {max_dd:.1%}")
    print(f"  Avg Turnover:    {avg_to:.3f}")
    print(f"  Avg Gross Lev:   {avg_lev:.2f}")
    print(f"  Avg Alphas:      {avg_alphas:.1f}")
    print(f"  Bars Simulated:  {len(df)}")
    print(f"  Date Range:      {df.index[0].date()} → {df.index[-1].date()}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [3, 1, 1]})

    # Equity curve
    ax = axes[0]
    ax.plot(df.index, df["equity"], color="#2196F3", linewidth=1.5, label="Equity")
    ax.fill_between(df.index, 1, df["equity"], where=df["equity"] >= 1,
                    alpha=0.15, color="#4CAF50")
    ax.fill_between(df.index, 1, df["equity"], where=df["equity"] < 1,
                    alpha=0.15, color="#F44336")
    ax.axhline(1, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(f"Walk-Forward Crypto Pipeline | Sharpe={sharpe:+.2f} | "
                 f"Return={total_ret:+.1%} | MaxDD={max_dd:.1%} | Fee={fee_bps}bps",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Equity")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.2)

    # Drawdown
    ax = axes[1]
    dd = (df["equity"] - df["equity"].cummax()) / df["equity"].cummax()
    ax.fill_between(df.index, 0, dd, color="#F44336", alpha=0.4)
    ax.set_ylabel("Drawdown")
    ax.set_ylim(min(dd.min() * 1.2, -0.01), 0.005)
    ax.grid(True, alpha=0.2)

    # Active alphas + turnover
    ax = axes[2]
    ax.bar(df.index, df["turnover"], width=0.15, color="#FF9800", alpha=0.5, label="Turnover")
    ax2 = ax.twinx()
    ax2.plot(df.index, df["n_alphas"], color="#9C27B0", linewidth=1, alpha=0.7, label="# Alphas")
    ax.set_ylabel("Turnover")
    ax2.set_ylabel("# Alphas")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = "walkforward_crypto_pipeline.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n  Chart saved: {out_path}")

    return {"sharpe": sharpe, "return": total_ret, "maxdd": max_dd}


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Crypto Pipeline")
    parser.add_argument("--days", type=int, default=480, help="Days of history")
    parser.add_argument("--fee-bps", type=float, default=5.0, help="One-way fee in bps")
    parser.add_argument("--top-n", type=int, default=50, help="Universe size")
    parser.add_argument("--train-bars", type=int, default=720, help="Training window (bars)")
    parser.add_argument("--val-bars", type=int, default=360, help="Validation window (bars)")
    args = parser.parse_args()

    # 1. Load data
    features = load_data(n_symbols=args.top_n, n_days=args.days)

    # 2. Compute all alpha signals
    print(f"\nComputing {len(ALPHA_DEFINITIONS)} alpha signals...", flush=True)
    alpha_signals = {}
    for name, expr in ALPHA_DEFINITIONS:
        sig = evaluate_expression(expr, features)
        if sig is not None:
            alpha_signals[name] = sig

    n_ok = len(alpha_signals)
    n_fail = len(ALPHA_DEFINITIONS) - n_ok
    print(f"  {n_ok} alphas computed ({n_fail} failed)")

    # 3. Run walk-forward backtest
    results = run_backtest(
        features, alpha_signals,
        fee_bps=args.fee_bps,
        train_bars=args.train_bars,
        val_bars=args.val_bars,
    )

    if results.empty:
        print("ERROR: No results generated. Need more data.")
        return

    # 4. Report
    metrics = report(results, args.fee_bps)

    # 5. Fee sensitivity sweep
    print(f"\n{'='*70}")
    print("FEE SENSITIVITY SWEEP")
    print(f"{'='*70}")
    for fee in [0, 3, 5, 10]:
        pnl_adj = results["pnl_gross"] - results["turnover"] * fee * 1e-4
        sr = float((pnl_adj.mean() / (pnl_adj.std() + 1e-12)) * ANN_FACTOR)
        ret = float((1 + pnl_adj).prod() - 1)
        print(f"  {fee:2d} bps: Sharpe={sr:+.2f}, Return={ret:+.1%}")


if __name__ == "__main__":
    main()
