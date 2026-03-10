"""
Walk-Forward Simulation Engine — bar-by-bar backtester with dynamic alpha selection.

Unlike the vectorized sim (which evaluates a single alpha on the full matrix at once),
this simulates a live trading system: at each bar, it selects alphas, combines them,
optimizes a portfolio, and trades.

Architecture sourced from CryptoRLQuantResearcher pipeline (>2 Sharpe OOS).

Key components:
  1. Walk-forward alpha selection (train/val windows, orthogonal filtering)
  2. QP factor-return alpha combination (rolling lookback)
  3. CVXPY portfolio optimization with PCA hedging + turnover control
  4. Dynamic vol targeting and drawdown protection

Usage:
    from src.simulation.walkforward import WalkForwardConfig, WalkForwardSimulator
    config = WalkForwardConfig(train_bars=720, val_bars=360)
    sim = WalkForwardSimulator(config)
    results = sim.run(features, alpha_signals)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
    _INSTALLED_SOLVERS = cp.installed_solvers()
except ImportError:
    CVXPY_AVAILABLE = False
    _INSTALLED_SOLVERS = []

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward simulation."""
    # Walk-forward windows (in bars)
    train_bars: int = 720       # Training window for alpha evaluation
    val_bars: int = 360         # Validation window for alpha evaluation
    reeval_interval: int = 6    # Re-evaluate alpha selection every N bars
    warmup_bars: int = 200      # Min bars before train_start

    # Alpha selection
    max_alphas: int = 12        # Max alphas to include
    max_corr: float = 0.65      # Max pairwise correlation
    min_train_sharpe: float = 0.3   # Min Sharpe in training period
    min_val_sharpe: float = 0.0     # Min Sharpe in validation period
    val_weight: float = 0.7     # Blend weight for validation Sharpe
    train_weight: float = 0.3   # Blend weight for training Sharpe

    # QP combiner
    qp_lookback: int = 45       # Lookback for QP alpha combination

    # Portfolio optimizer
    max_position: float = 0.04      # Max weight per asset
    max_gross_leverage: float = 0.80
    max_turnover: float = 0.08      # Max turnover per bar
    risk_aversion: float = 1.0
    cov_shrinkage: float = 0.9      # Covariance shrinkage toward scaled identity
    n_pca_factors: int = 3          # PCA factors to hedge out
    tx_cost_bps: float = 10.0       # Transaction cost in objective (bps)
    optimizer_lookback: int = 120   # Bars of returns for covariance

    # Fee for PnL computation
    fee_bps: float = 5.0        # One-way fee in bps

    # Risk management
    vol_target: float = 0.20    # Annualized vol target
    dd_threshold: float = -0.05 # Drawdown level to start cutting
    dd_floor: float = 0.25      # Min exposure multiplier at max DD

    # Universe
    universe_size: int = 50     # Top N by trailing ADV
    adv_lookback: int = 60      # Bars for trailing ADV

    # Annualization
    bars_per_day: int = 6       # 4h bars
    trading_days_per_year: int = 252

    @property
    def ann_factor(self) -> float:
        return math.sqrt(self.bars_per_day * self.trading_days_per_year)

    @property
    def start_bar(self) -> int:
        return self.train_bars + self.val_bars + self.warmup_bars


# ═══════════════════════════════════════════════════════════════════
# Result
# ═══════════════════════════════════════════════════════════════════

@dataclass
class WalkForwardResult:
    """Result from walk-forward simulation."""
    results_df: pd.DataFrame        # datetime-indexed results
    sharpe_net: float = 0.0
    sharpe_gross: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    avg_turnover: float = 0.0
    avg_alphas: float = 0.0
    avg_gross_leverage: float = 0.0
    bars_simulated: int = 0
    optimizer_convergence_rate: float = 0.0
    optimizer_converged: int = 0
    optimizer_failed: int = 0

    def fee_sensitivity(self, fee_levels: list[float] | None = None) -> dict:
        """Compute Sharpe at different fee levels."""
        if fee_levels is None:
            fee_levels = [0, 3, 5, 10]
        results = {}
        df = self.results_df
        ann = math.sqrt(6 * 252)
        for fee in fee_levels:
            pnl_adj = df["pnl_gross"] - df["turnover"] * fee * 1e-4
            sr = float((pnl_adj.mean() / (pnl_adj.std() + 1e-12)) * ann)
            ret = float((1 + pnl_adj).prod() - 1)
            results[fee] = {"sharpe": sr, "return": ret}
        return results


# ═══════════════════════════════════════════════════════════════════
# Alpha Selection
# ═══════════════════════════════════════════════════════════════════

def _compute_alpha_sharpe(signal: pd.DataFrame, returns: pd.DataFrame,
                          start: int, end: int, ann_factor: float) -> float | None:
    """Compute annualized Sharpe for a single alpha over [start, end) bar range."""
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
    return float((pnl.mean() / pnl.std()) * ann_factor)


def _compute_signal_correlation(s1: pd.DataFrame, s2: pd.DataFrame,
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


def select_alphas(alpha_signals: dict[str, pd.DataFrame],
                  returns: pd.DataFrame,
                  current_bar: int,
                  config: WalkForwardConfig) -> list[str]:
    """
    Walk-forward alpha selection at a given bar.

    Evaluates each alpha on train and validation periods,
    then greedily selects the best orthogonal subset.
    """
    train_start = current_bar - config.train_bars - config.val_bars
    train_end = current_bar - config.val_bars
    val_end = current_bar

    if train_start < config.warmup_bars:
        return []

    # Score each alpha
    metrics = {}
    for name, sig in alpha_signals.items():
        train_sr = _compute_alpha_sharpe(sig, returns, train_start, train_end, config.ann_factor)
        val_sr = _compute_alpha_sharpe(sig, returns, train_end, val_end, config.ann_factor)
        if train_sr is not None and val_sr is not None:
            metrics[name] = {"train": train_sr, "val": val_sr}

    # Filter and sort
    passing = [n for n, m in metrics.items()
               if m["train"] > config.min_train_sharpe and m["val"] > config.min_val_sharpe]
    passing.sort(
        key=lambda n: metrics[n]["val"] * config.val_weight + metrics[n]["train"] * config.train_weight,
        reverse=True,
    )

    # Greedy orthogonal selection
    selected = []
    for name in passing:
        is_ortho = all(
            _compute_signal_correlation(alpha_signals[name], alpha_signals[s], train_start, val_end)
            <= config.max_corr
            for s in selected
        )
        if is_ortho:
            selected.append(name)
        if len(selected) >= config.max_alphas:
            break

    return selected


# ═══════════════════════════════════════════════════════════════════
# QP Alpha Combination
# ═══════════════════════════════════════════════════════════════════

class QPCombiner:
    """Combine selected alphas using QP on rolling factor returns."""

    def __init__(self, alpha_signals: dict[str, pd.DataFrame],
                 names: list[str], returns: pd.DataFrame, lookback: int = 45):
        self.signals = alpha_signals
        self.names = names
        self.returns = returns
        self.lookback = lookback

        # Pre-compute factor returns
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
        """Get combined market-neutral signal at time t."""
        if not self.names:
            return None, 0

        weights = self.get_weights(t_idx)
        n_active = int(np.sum(weights > 0.05))

        combined = pd.Series(0.0, index=self.signals[self.names[0]].columns)
        for i, name in enumerate(self.names):
            if weights[i] > 0.01:
                combined += weights[i] * self.signals[name].iloc[t_idx - 1]

        combined = combined.sub(combined.mean())
        combined = combined.div(combined.abs().sum() + 1e-10)
        return combined, n_active


# ═══════════════════════════════════════════════════════════════════
# CVXPY Portfolio Optimizer
# ═══════════════════════════════════════════════════════════════════

def _estimate_covariance(returns: pd.DataFrame, shrinkage: float = 0.9) -> np.ndarray:
    """Ledoit-Wolf shrinkage toward scaled identity (matching CryptoRL)."""
    r = returns.fillna(0)
    n = r.shape[1]
    if len(r) < 10 or n < 2:
        return np.eye(n) * 0.0004
    sample_cov = r.cov().values
    sample_cov = np.nan_to_num(sample_cov, nan=0.0)
    avg_var = np.diag(sample_cov).mean()
    target = np.eye(n) * avg_var
    shrunk = shrinkage * target + (1 - shrinkage) * sample_cov
    min_eig = np.linalg.eigvalsh(shrunk).min()
    if min_eig < 0:
        shrunk -= np.eye(n) * (min_eig - 1e-8)
    return shrunk


def _compute_pca_loadings(returns: pd.DataFrame, n_factors: int = 3) -> np.ndarray | None:
    """PCA via SVD on centered returns (matching CryptoRL)."""
    r = returns.fillna(0)
    hist = r.iloc[-90:] if len(r) > 90 else r
    if len(hist) < 20 or hist.shape[1] < n_factors + 1:
        return None
    try:
        X = hist.values
        X_centered = X - X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        return Vt.T[:, :n_factors]
    except Exception:
        return None


def _choose_solver() -> str:
    """Pick the best available CVXPY solver — prefer ECOS (matching CryptoRL)."""
    for solver in ["ECOS", "CLARABEL", "SCS"]:
        if solver in _INSTALLED_SOLVERS:
            return solver
    return "SCS"


def optimize_portfolio(signal: pd.Series, returns: pd.DataFrame,
                       current_holdings: pd.Series | None,
                       config: WalkForwardConfig) -> tuple[pd.Series, bool]:
    """
    CVXPY portfolio optimization with PCA neutrality constraints.

    Returns (holdings, converged).
    Falls back to turnover-scaled forecast when optimizer fails.
    """
    if not CVXPY_AVAILABLE:
        return signal.copy(), False

    assets = signal.index
    n = len(assets)
    r = returns.reindex(columns=assets).fillna(0)

    alpha = signal.fillna(0).values.astype(float)
    Sigma = _estimate_covariance(r, config.cov_shrinkage)
    pca = _compute_pca_loadings(r, config.n_pca_factors)

    h_prev = np.zeros(n)
    if current_holdings is not None:
        h_prev = current_holdings.reindex(assets, fill_value=0).values

    h = cp.Variable(n)
    trade = h - h_prev
    tx_cost = (config.tx_cost_bps * 1e-4) * cp.norm(trade, 1)

    objective = cp.Maximize(
        alpha @ h - (config.risk_aversion / 2) * cp.quad_form(h, Sigma) - tx_cost
    )

    constraints = [
        cp.sum(h) == 0,
        cp.norm(h, "inf") <= config.max_position,
        cp.norm(h, 1) <= config.max_gross_leverage,
        cp.norm(trade, 1) <= config.max_turnover,  # Always apply (CryptoRL match)
    ]

    # PCA neutrality
    if pca is not None:
        for k in range(pca.shape[1]):
            constraints.append(pca[:, k] @ h == 0)

    problem = cp.Problem(objective, constraints)
    solver = _choose_solver()

    try:
        try:
            problem.solve(solver=cp.ECOS, verbose=False, max_iters=200)
        except Exception:
            problem.solve(solver=cp.SCS, verbose=False, max_iters=500)
        if problem.status in ("optimal", "optimal_inaccurate") and h.value is not None:
            return pd.Series(h.value, index=assets), True
    except Exception:
        pass

    # Fallback: turnover-scaled forecast to prevent cascade failures
    if current_holdings is not None:
        h_prev_s = current_holdings.reindex(assets, fill_value=0)
        trade_vec = signal - h_prev_s
        trade_norm = trade_vec.abs().sum()
        if trade_norm > config.max_turnover:
            scale = config.max_turnover / trade_norm
            return h_prev_s + trade_vec * scale, False
        return signal.copy(), False
    else:
        gross = signal.abs().sum()
        if gross > config.max_turnover:
            return signal * (config.max_turnover / gross), False
        return signal.copy(), False


# ═══════════════════════════════════════════════════════════════════
# Walk-Forward Simulator
# ═══════════════════════════════════════════════════════════════════

class WalkForwardSimulator:
    """
    Full walk-forward backtest engine.

    Orchestrates alpha selection, combination, portfolio optimization,
    and PnL computation at each bar.
    """

    def __init__(self, config: WalkForwardConfig | None = None):
        self.config = config or WalkForwardConfig()

    def run(self, features: dict[str, pd.DataFrame],
            alpha_signals: dict[str, pd.DataFrame]) -> WalkForwardResult:
        """
        Run walk-forward backtest.

        Args:
            features: Dictionary with 'returns', 'close', 'adv20', etc.
            alpha_signals: Dictionary of alpha name → signal DataFrame.

        Returns:
            WalkForwardResult with full diagnostics.
        """
        cfg = self.config
        returns = features["returns"]
        close = features["close"]
        adv20 = features.get("adv20", features.get("quote_volume",
                features["volume"]).rolling(30, min_periods=10).mean())
        n_bars = len(close)

        start_bar = cfg.start_bar
        if start_bar >= n_bars:
            logger.warning(f"Not enough data: {n_bars} bars, need {start_bar}")
            return WalkForwardResult(results_df=pd.DataFrame())

        rolling_vol = returns.mean(axis=1).rolling(30).std() * cfg.ann_factor

        logger.info(f"Walk-forward: {n_bars - start_bar} bars, {len(alpha_signals)} alphas")

        results = []
        prev_holdings = None
        current_selected: list[str] = []
        combiner: QPCombiner | None = None
        equity = 1.0
        peak_equity = 1.0
        opt_converged = 0
        opt_failed = 0

        # Universe rebalancing: every 120 4h bars (~20 days)
        REBAL_BARS = 120
        universe = None

        for t in range(start_bar, n_bars):
            # Alpha selection
            if (t - start_bar) % cfg.reeval_interval == 0 or not current_selected:
                new_sel = select_alphas(alpha_signals, returns, t, cfg)
                if new_sel:
                    current_selected = new_sel
                    combiner = QPCombiner(
                        {n: alpha_signals[n] for n in current_selected},
                        current_selected, returns, cfg.qp_lookback,
                    )

            if not current_selected or combiner is None:
                continue

            # Universe: top N by trailing ADV, rebalanced every 120 bars (~20 days)
            if universe is None or (t - start_bar) % REBAL_BARS == 0:
                trailing = adv20.iloc[max(0, t - cfg.adv_lookback):t].mean()
                valid = trailing.dropna()
                if len(valid) < 20:
                    continue
                universe = valid.nlargest(cfg.universe_size).index.tolist()

            # Combine alphas
            forecast, n_active = combiner.combine(t)
            if forecast is None:
                continue
            forecast = forecast.reindex(universe).fillna(0)
            forecast = forecast.sub(forecast.mean())
            forecast = forecast.div(forecast.abs().sum() + 1e-10)

            # Dynamic scaling
            cur_vol = rolling_vol.iloc[t] if t < len(rolling_vol) and not np.isnan(rolling_vol.iloc[t]) else 0.3
            vol_scale = min(1.0, cfg.vol_target / (cur_vol + 0.01))

            dd = (equity - peak_equity) / peak_equity
            dd_scale = max(cfg.dd_floor, 1.0 + dd * 4) if dd < cfg.dd_threshold else 1.0

            forecast = forecast * vol_scale * dd_scale

            # Portfolio optimization
            hist_ret = returns.iloc[max(0, t - cfg.optimizer_lookback):t]
            cur = prev_holdings.reindex(forecast.index, fill_value=0) if prev_holdings is not None else None

            holdings, converged = optimize_portfolio(forecast, hist_ret, cur, cfg)
            if converged:
                opt_converged += 1
            else:
                opt_failed += 1

            # PnL
            actual_ret = returns.iloc[t].reindex(universe, fill_value=0)
            pnl_gross = float((holdings * actual_ret).sum())

            turnover = float((holdings - (prev_holdings.reindex(holdings.index, fill_value=0)
                              if prev_holdings is not None else 0)).abs().sum())
            pnl_net = pnl_gross - turnover * cfg.fee_bps * 1e-4

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
                "converged": converged,
            })
            prev_holdings = holdings.copy()

        if not results:
            return WalkForwardResult(results_df=pd.DataFrame())

        df = pd.DataFrame(results).set_index("datetime")
        total_opt = opt_converged + opt_failed

        return WalkForwardResult(
            results_df=df,
            sharpe_net=float((df["pnl_net"].mean() / (df["pnl_net"].std() + 1e-12)) * cfg.ann_factor),
            sharpe_gross=float((df["pnl_gross"].mean() / (df["pnl_gross"].std() + 1e-12)) * cfg.ann_factor),
            total_return=float(df["equity"].iloc[-1] - 1),
            max_drawdown=float(((df["equity"] - df["equity"].cummax()) / df["equity"].cummax()).min()),
            avg_turnover=float(df["turnover"].mean()),
            avg_alphas=float(df["n_alphas"].mean()),
            avg_gross_leverage=float(df["gross_lev"].mean()),
            bars_simulated=len(df),
            optimizer_convergence_rate=opt_converged / total_opt if total_opt > 0 else 0.0,
            optimizer_converged=opt_converged,
            optimizer_failed=opt_failed,
        )
