"""
Isichenko-Style Stat-Arb Pipeline
==================================
Full implementation following "Quantitative Portfolio Management"
by Michael Isichenko.

Stages:
  1. Alpha Scaler   — convert raw GP signals to return forecasts
  2. Risk Model     — factor covariance (sector + style)
  3. Optimizer      — QP with transaction costs + risk (CVXPY/OSQP)
  4. Backtester     — sequential day-by-day walk-forward

Usage:
  python run_isichenko_pipeline.py
"""
import os, sys, json, time, logging, sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Mean of empty.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*overflow.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*solve.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*OSQP.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Solution may be inaccurate.*")
import pandas as pd
import cvxpy as cp
from scipy import linalg

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
@dataclass
class PipelineConfig:
    # Periods
    is_start: str = "2018-01-01"       # Start training
    oos_start: str = "2023-01-01"      # Start OOS
    warmup_days: int = 252             # 1 year warmup for risk model
    
    # Universe
    universe_name: str = "TOP3000"
    min_coverage: float = 0.3
    min_adv: float = 1e6               # $1M min ADV
    
    # Optimizer
    risk_aversion: float = 1.0         # κ — risk aversion (now in weight-space, ~1.0 is standard)
    booksize: float = 20_000_000.0     # $20M
    max_position_pct_gmv: float = 0.01 # 1% of booksize per name (matches vectorized sim max_stock_weight)
    max_position_pct_adv: float = 0.05 # 5% of ADV per name  
    max_trade_pct_adv: float = 0.10    # 10% of ADV max trade
    
    # Transaction costs (calibrated to industry benchmarks)
    # Large-cap effective spread: 1-3 bps (SEC Rule 605), IBKR Pro: 0.1-0.7 bps commission
    # Realistic for TOP1000 via DMA/algo: ~3 bps total one-way
    slippage_bps: float = 1.5          # half-spread: ~1.5 bps for large/mid cap
    commission_bps: float = 1.5        # DMA/algo institutional rate (IBKR tiered)
    impact_coeff: float = 0.1          # Almgren-Chriss: Impact ∝ coeff * σ * √(trade/ADV)
    borrow_cost_bps: float = 0.12      # ~30 bps/yr ÷ 252 for general collateral
    trade_aversion: float = 0.0        # Extra turnover penalty in QP objective
    holding_bonus: float = 0.0         # Reward for holding existing positions
    
    # Alpha combination mode
    raw_signal_mode: bool = False      # True = skip IC-weighting, just equal-weight
    
    # Risk model
    ema_halflife_risk: int = 60        # days for risk estimation
    ema_halflife_alpha: int = 120      # days for alpha IC tracking
    n_pca_factors: int = 5             # statistical factors beyond sectors
    
    # Neutralization
    dollar_neutral: bool = True
    sector_neutral: bool = True
    
    # Delay
    delay: int = 1


# ═══════════════════════════════════════════════════════════════
# STAGE 1: ALPHA IC TRACKER
# ═══════════════════════════════════════════════════════════════
class AlphaICTracker:
    """Track rolling Information Coefficient (rank correlation) for each alpha.
    
    IC = Spearman rank correlation between cross-sectional alpha ranks 
    and forward returns. This is the standard measure of alpha quality
    (Grinold & Kahn, Active Portfolio Management).
    
    The IC is used to weight alphas in combination: higher IC = more weight.
    This avoids the pitfalls of OLS scaling on rank-based signals.
    """
    
    def __init__(self, halflife: int = 120):
        self.halflife = halflife
        self._ema_decay = np.log(2) / halflife
        self._running_ic = None
        self._running_ic_sq = None   # For IC variance (stability)
        self._n_updates = 0
        
    def update(self, ranked_signal: np.ndarray, returns: np.ndarray):
        """Update IC estimate with today's cross-sectional observation.
        
        Args:
            ranked_signal: (N,) cross-sectional rank-normalized signal
            returns: (N,) realized forward returns
        """
        mask = np.isfinite(ranked_signal) & np.isfinite(returns) & (ranked_signal != 0)
        if mask.sum() < 30:
            return
        s = ranked_signal[mask]
        r = returns[mask]
        
        # Rank correlation (Spearman IC)
        from scipy.stats import spearmanr
        ic, _ = spearmanr(s, r)
        if not np.isfinite(ic):
            return
        
        alpha = 1 - np.exp(-self._ema_decay)
        if self._running_ic is None:
            self._running_ic = ic
            self._running_ic_sq = ic ** 2
        else:
            self._running_ic = alpha * ic + (1 - alpha) * self._running_ic
            self._running_ic_sq = alpha * ic**2 + (1 - alpha) * self._running_ic_sq
        self._n_updates += 1
    
    @property
    def ic(self) -> float:
        """Current smoothed IC estimate."""
        if self._running_ic is None:
            return 0.0
        return self._running_ic
    
    @property
    def ic_ir(self) -> float:
        """IC Information Ratio: IC / std(IC). Higher = more stable."""
        if self._running_ic is None or self._running_ic_sq is None:
            return 0.0
        var = self._running_ic_sq - self._running_ic ** 2
        if var <= 1e-10:
            return 0.0
        return self._running_ic / np.sqrt(var)
    
    @property
    def weight(self) -> float:
        """Combination weight: max(IC, 0). Anti-predictive alphas get zero."""
        return max(self._running_ic, 0.0) if self._running_ic is not None else 0.0
    
    @property
    def is_active(self) -> bool:
        return self._n_updates >= 20 and self.ic > 0.0
        return 1.0 / self._running_mse


# ═══════════════════════════════════════════════════════════════
# STAGE 2: RISK MODEL  
# ═══════════════════════════════════════════════════════════════
class FactorRiskModel:
    """
    Factor risk model: C = Diag(σ²) + B'FB
    
    Factors:
      - Sector dummies (11 GICS sectors)
      - Style: size, value, momentum, volatility, leverage
    """
    
    def __init__(self, halflife: int = 60):
        self.halflife = halflife
        self.factor_cov: Optional[np.ndarray] = None       # F (K×K)
        self.factor_loadings: Optional[np.ndarray] = None   # B (N×K)
        self.specific_var: Optional[np.ndarray] = None      # σ² (N,)
        self.factor_names: List[str] = []
        self._return_history: List[Tuple[np.ndarray, np.ndarray]] = []
        
    def build_loadings(self, tickers: List[str], 
                       classifications: Dict,
                       market_cap: pd.Series,
                       book_to_market: pd.Series,
                       momentum_12m: pd.Series,
                       volatility: pd.Series,
                       debt_to_equity: pd.Series) -> np.ndarray:
        """Build factor loading matrix B (N × K)."""
        N = len(tickers)
        
        # --- Sector dummies ---
        sectors = set()
        for t in tickers:
            if t in classifications and isinstance(classifications[t], dict):
                sectors.add(classifications[t].get("sector", "Unknown"))
        sectors = sorted(sectors)
        
        sector_cols = []
        for sector in sectors:
            col = np.zeros(N)
            for i, t in enumerate(tickers):
                if t in classifications and isinstance(classifications[t], dict):
                    if classifications[t].get("sector") == sector:
                        col[i] = 1.0
            sector_cols.append(col)
        
        # --- Style factors (z-scored cross-sectionally) ---
        def zscore_series(s: pd.Series) -> np.ndarray:
            vals = np.array([s.get(t, np.nan) for t in tickers], dtype=float)
            mask = np.isfinite(vals)
            if mask.sum() > 10:
                mu = np.nanmean(vals[mask])
                std = np.nanstd(vals[mask])
                if std > 1e-10:
                    vals[mask] = (vals[mask] - mu) / std
                else:
                    vals[mask] = 0.0
            vals[~mask] = 0.0
            return vals
        
        style_names = ["size", "value", "momentum", "volatility", "leverage"]
        style_cols = [
            zscore_series(market_cap.apply(np.log) if isinstance(market_cap, pd.Series) else pd.Series()),
            zscore_series(book_to_market),
            zscore_series(momentum_12m),
            zscore_series(volatility),
            zscore_series(debt_to_equity),
        ]
        
        # Combine
        self.factor_names = [f"sector_{s}" for s in sectors] + style_names
        all_cols = sector_cols + style_cols
        K = len(all_cols)
        
        B = np.column_stack(all_cols)  # N × K
        self.factor_loadings = B
        self._sector_names = sectors
        self._n_sectors = len(sectors)
        return B
    
    def update(self, returns: np.ndarray, weights: Optional[np.ndarray] = None):
        """Update factor covariance and specific variance with new daily returns.
        
        Args:
            returns: (N,) cross-sectional returns for one day
            weights: (N,) optional statistical weights
        """
        if self.factor_loadings is None:
            return
        
        B = self.factor_loadings  # N × K
        r = returns.copy()
        r[~np.isfinite(r)] = 0.0
        
        N, K = B.shape
        
        # Regress returns on factors: ρ = (B'B)^-1 B'r
        BtB = B.T @ B
        try:
            factor_returns = np.linalg.solve(BtB + 1e-8 * np.eye(K), B.T @ r)
        except np.linalg.LinAlgError:
            return
        
        # Residuals
        residuals = r - B @ factor_returns
        
        # EMA update
        alpha = 1 - np.exp(-np.log(2) / self.halflife)
        
        fr_outer = np.outer(factor_returns, factor_returns)
        res_sq = residuals ** 2
        
        if self.factor_cov is None:
            self.factor_cov = fr_outer
            self.specific_var = res_sq
        else:
            self.factor_cov = alpha * fr_outer + (1 - alpha) * self.factor_cov
            self.specific_var = alpha * res_sq + (1 - alpha) * self.specific_var
    
    def get_Q_matrix(self) -> np.ndarray:
        """Return Q such that Q'Q = B F B' (for factor risk term).
        Q = chol(F) @ B'  →  shape (K, N)
        """
        if self.factor_cov is None or self.factor_loadings is None:
            return None
        
        F = self.factor_cov
        # Ensure PSD
        eigvals, eigvecs = np.linalg.eigh(F)
        eigvals = np.maximum(eigvals, 1e-10)
        F_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        try:
            L = np.linalg.cholesky(F_psd)  # K × K lower triangular
        except np.linalg.LinAlgError:
            L = np.diag(np.sqrt(eigvals)) @ eigvecs.T
        
        Q = L @ self.factor_loadings.T  # K × N
        return Q


# ═══════════════════════════════════════════════════════════════
# STAGE 3: PORTFOLIO OPTIMIZER (CVXPY + OSQP)
# ═══════════════════════════════════════════════════════════════
class PortfolioOptimizerQP:
    """
    Weight-space formulation (standard Mosek / Boyd style):
    
    Maximize: α'w - ½κ w'Σw - tcost(w - w_prev)
    Subject to: sum(w) = 0 (dollar neutral), |w_i| <= w_max, sum(|w_i|) <= 1
    
    Variables are portfolio WEIGHTS (fractions of booksize), not dollar holdings.
    This gives better numerical conditioning and makes κ interpretable.
    
    α should be in units of expected return (e.g., IC * σ_cs ≈ 1e-3 to 1e-2).
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def optimize(self, 
                 alpha_vec: np.ndarray,        # (N,) return forecast (expected return units)
                 Q: np.ndarray,                # (K, N) factor risk sqrt
                 spec_var: np.ndarray,          # (N,) specific variance
                 w_prev: np.ndarray,            # (N,) previous weights (fractions of booksize)
                 adv: np.ndarray,               # (N,) dollar volume
                 sector_masks: List[np.ndarray], # list of boolean masks
                 ) -> np.ndarray:
        """Solve the QP and return optimal weights w*."""
        
        N = len(alpha_vec)
        cfg = self.config
        kappa = cfg.risk_aversion
        booksize = cfg.booksize
        
        # Position limits in weight space
        w_max_gmv = cfg.max_position_pct_gmv  # e.g., 0.02 = 2%
        w_max_adv = cfg.max_position_pct_adv * adv / booksize
        w_max = np.minimum(w_max_gmv, w_max_adv)
        w_max = np.maximum(w_max, 1e-6)
        
        # Transaction cost in weight space
        linear_cost_bps = cfg.slippage_bps + getattr(cfg, 'commission_bps', 0.0)
        linear_cost = linear_cost_bps * 1e-4  # cost per unit of weight traded
        
        # --- Build & solve CVXPY Problem ---
        w = cp.Variable(N)
        T = w - w_prev
        
        # 1. Factor risk: ½κ ||Qw||² (Q operates on weights)
        factor_risk = 0.5 * kappa * cp.sum_squares(Q @ w)
        
        # 2. Specific risk: ½κ w'Sw
        specific_risk = 0.5 * kappa * cp.sum(cp.multiply(spec_var, cp.square(w)))
        
        # 3. Expected return: α'w
        expected_return = alpha_vec @ w
        
        # 4. Linear transaction costs: c * |T|_1
        tcost = linear_cost * cp.norm(T, 1)
        
        # 5. SOFT sector neutrality penalty
        sector_penalty = 0.0
        sector_pen_coeff = kappa * 10
        if cfg.sector_neutral and sector_masks:
            for mask in sector_masks:
                if mask.sum() > 1:
                    sector_penalty += sector_pen_coeff * cp.square(cp.sum(w[mask]))
        
        # 6. Extra turnover aversion
        trade_penalty = cfg.trade_aversion * cp.sum_squares(T) if cfg.trade_aversion > 0 else 0.0
        
        objective = cp.Minimize(
            factor_risk + specific_risk - expected_return 
            + tcost + sector_penalty + trade_penalty
        )
        
        # Constraints
        constraints = [
            w >= -w_max,
            w <= w_max,
            cp.norm(w, 1) <= 1.0,  # GMV <= booksize (in weight space, sum |w| <= 1)
        ]
        if cfg.dollar_neutral:
            constraints.append(cp.sum(w) == 0)
        
        prob = cp.Problem(objective, constraints)
        
        # Try OSQP first (fastest), fall back to SCS
        for solver, kwargs in [
            (cp.OSQP, {"max_iter": 20000, "eps_abs": 1e-5, "eps_rel": 1e-5, "verbose": False, "warm_start": True, "time_limit": 30.0}),
            (cp.SCS, {"max_iters": 10000, "verbose": False}),
        ]:
            try:
                prob.solve(solver=solver, **kwargs)
                if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
                    return w.value
            except Exception:
                continue
        
        # Final fallback: simple analytical solution
        return self._fallback_optimize(alpha_vec, spec_var, w_prev, w_max, kappa)
    
    def _fallback_optimize(self, alpha_vec, spec_var, w_prev, w_max, kappa):
        """Simple analytical mean-variance solution as fallback."""
        safe_var = np.maximum(spec_var, 1e-8)
        w = alpha_vec / (kappa * safe_var + 1e-12)
        
        # Clip to position limits
        w = np.clip(w, -w_max, w_max)
        
        # Dollar neutralize
        w = w - np.mean(w)
        
        # Scale so GMV <= 1
        gmv = np.sum(np.abs(w))
        if gmv > 1.0:
            w = w / gmv
        
        return w


# ═══════════════════════════════════════════════════════════════
# STAGE 4: SEQUENTIAL BACKTESTER
# ═══════════════════════════════════════════════════════════════
@dataclass
class DailyResult:
    date: str
    pnl: float
    gross_pnl: float
    tcost: float
    turnover: float
    gmv: float
    n_long: int
    n_short: int
    factor_risk: float
    specific_risk: float


class IsichenkoPipeline:
    """Full day-by-day sequential backtester."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.risk_model = FactorRiskModel(halflife=config.ema_halflife_risk)
        self.optimizer = PortfolioOptimizerQP(config)
        self.alpha_scalers: Dict[str, AlphaScaler] = {}
        
    def run(self, 
            alpha_expressions: List[str],
            matrices: Dict[str, pd.DataFrame],
            classifications: Dict,
            universe_df: pd.DataFrame,
            expr_engine) -> Dict:
        """
        Run the full pipeline.
        
        Returns dict with daily results and statistics.
        """
        cfg = self.config
        
        # Get all trading dates
        close = matrices["close"]
        returns = matrices["returns"] if "returns" in matrices else close.pct_change()
        # CRITICAL: clip extreme returns to prevent NaN/Inf from bad data
        returns = returns.clip(-0.5, 0.5)
        all_dates = close.index.tolist()
        
        # Filter to IS/OOS period
        start_idx = 0
        for i, d in enumerate(all_dates):
            if str(d)[:10] >= cfg.is_start:
                start_idx = i
                break
        
        tickers = close.columns.tolist()
        N = len(tickers)
        
        # Universe mask
        if universe_df is not None:
            universe_mask_df = universe_df.reindex(index=close.index, columns=tickers).fillna(False)
        else:
            universe_mask_df = pd.DataFrame(True, index=close.index, columns=tickers)
        
        # Pre-compute alpha signals for all dates (heavy but avoids re-computing)
        print(f"  Pre-computing {len(alpha_expressions)} alpha signals...")
        alpha_signals = {}
        for i, expr in enumerate(alpha_expressions):
            try:
                sig = expr_engine.evaluate(expr)
                # Align to our columns
                sig = sig.reindex(index=close.index, columns=tickers)
                alpha_signals[expr] = sig
                print(f"    ✅ Alpha {i+1}: {expr[:55]}")
            except Exception as e:
                print(f"    ❌ Alpha {i+1} failed: {e}")
        
        print(f"  {len(alpha_signals)}/{len(alpha_expressions)} alphas loaded\n")
        
        if len(alpha_signals) == 0:
            print("  No valid alphas! Aborting.")
            return {}
        
        # Prepare style factor inputs
        log_market_cap = matrices.get("market_cap", pd.DataFrame()).apply(np.log)
        book_to_market = matrices.get("book_to_market", pd.DataFrame())
        volatility = matrices.get("historical_volatility_60", pd.DataFrame())
        debt_to_eq = matrices.get("debt_to_equity", pd.DataFrame())
        adv_df = matrices.get("adv20", matrices.get("dollars_traded", pd.DataFrame()))
        
        # Compute 12-month momentum
        momentum_12m = close.pct_change(252)
        
        # Initialize IC trackers
        for expr in alpha_signals:
            self.alpha_scalers[expr] = AlphaICTracker(halflife=cfg.ema_halflife_alpha)
        
        # Build risk model loadings using data from first available date
        ref_date_idx = min(start_idx, len(all_dates) - 1)
        ref_date = all_dates[ref_date_idx]
        
        def get_series_at(df, date):
            if df.empty:
                return pd.Series(dtype=float)
            if date in df.index:
                return df.loc[date]
            return pd.Series(dtype=float)
        
        self.risk_model.build_loadings(
            tickers=tickers,
            classifications=classifications,
            market_cap=get_series_at(log_market_cap, ref_date),
            book_to_market=get_series_at(book_to_market, ref_date),
            momentum_12m=get_series_at(momentum_12m, ref_date),
            volatility=get_series_at(volatility, ref_date),
            debt_to_equity=get_series_at(debt_to_eq, ref_date),
        )
        
        # Track when to rebuild factor loadings (every 20 days)
        loadings_rebuild_interval = 20
        last_loadings_rebuild = start_idx
        
        # Build sector masks for neutrality constraints
        sector_masks = []
        if cfg.sector_neutral:
            sector_set = set()
            for t in tickers:
                if t in classifications and isinstance(classifications[t], dict):
                    sector_set.add(classifications[t].get("sector", "Unknown"))
            for sector in sorted(sector_set):
                mask = np.array([
                    classifications.get(t, {}).get("sector") == sector 
                    if isinstance(classifications.get(t), dict) else False
                    for t in tickers
                ])
                if mask.sum() > 1:
                    sector_masks.append(mask)
        
        # ── Main Loop ──
        w_prev = np.zeros(N)  # Start flat (weight space)
        daily_results = []
        
        # Warmup: train risk model and alpha scalers
        warmup_end = start_idx
        print(f"  Warming up risk model ({cfg.warmup_days} days)...")
        warmup_start = max(0, start_idx - cfg.warmup_days)
        
        for t in range(warmup_start, start_idx):
            date = all_dates[t]
            r = returns.iloc[t].reindex(tickers).fillna(0.0).values
            self.risk_model.update(r)
            
            # Update IC trackers with RANK-normalized signals
            if t > 0:
                prev_date = all_dates[t - 1]
                fwd_ret = returns.iloc[t].reindex(tickers).fillna(0).values
                for expr, sig_df in alpha_signals.items():
                    if prev_date in sig_df.index:
                        raw = sig_df.loc[prev_date].reindex(tickers)
                        # Rank-normalize
                        ranked = raw.rank(pct=True).fillna(0.5).values - 0.5
                        self.alpha_scalers[expr].update(ranked, fwd_ret)
        
        print(f"  Running backtest from {all_dates[start_idx]} to {all_dates[-1]}...\n")
        
        n_days = len(all_dates) - start_idx
        report_every = max(1, n_days // 20)
        
        for t_idx in range(start_idx, len(all_dates)):
            date = all_dates[t_idx]
            date_str = str(date)[:10]
            
            # 1. Get today's returns (realized)
            r_today = returns.iloc[t_idx].reindex(tickers).fillna(0.0).values
            
            # 2. PnL from yesterday's positions (weights × booksize × returns)
            h_prev_dollars = w_prev * cfg.booksize
            gross_pnl = np.dot(h_prev_dollars, r_today)
            
            # 3. Update risk model with today's returns
            self.risk_model.update(r_today)
            
            # 3b. Rebuild factor loadings periodically (dynamic style factors)
            if t_idx - last_loadings_rebuild >= loadings_rebuild_interval:
                self.risk_model.build_loadings(
                    tickers=tickers,
                    classifications=classifications,
                    market_cap=get_series_at(log_market_cap, date),
                    book_to_market=get_series_at(book_to_market, date),
                    momentum_12m=get_series_at(momentum_12m, date),
                    volatility=get_series_at(volatility, date),
                    debt_to_equity=get_series_at(debt_to_eq, date),
                )
                last_loadings_rebuild = t_idx
            
            # 4. Update IC trackers with (yesterday's ranked signal, today's return)
            if t_idx > 0:
                prev_date = all_dates[t_idx - 1]
                fwd_ret = returns.iloc[t_idx].reindex(tickers).fillna(0).values
                for expr, sig_df in alpha_signals.items():
                    if prev_date in sig_df.index:
                        raw = sig_df.loc[prev_date].reindex(tickers)
                        ranked = raw.rank(pct=True).fillna(0.5).values - 0.5
                        self.alpha_scalers[expr].update(ranked, fwd_ret)
            
            # 5. RANK-NORMALIZE each alpha, then combine
            # This is the key fix: rank normalization makes all alphas comparable
            # regardless of their raw signal scale (from 0-1 to billions).
            combined_alpha = np.zeros(N)
            n_active = 0
            
            for expr, sig_df in alpha_signals.items():
                if date not in sig_df.index:
                    continue
                raw = sig_df.loc[date].reindex(tickers)
                # Cross-sectional rank normalize to [-0.5, +0.5]
                ranked = raw.rank(pct=True).fillna(0.5).values - 0.5
                ranked = np.nan_to_num(ranked, nan=0.0)
                
                if not np.any(ranked != 0):
                    continue
                
                if self.config.raw_signal_mode:
                    # Equal weight
                    w_alpha = 1.0
                else:
                    # IC-weighted: weight by trailing IC (higher IC = more weight)
                    w_alpha = self.alpha_scalers[expr].weight
                    if w_alpha <= 0:
                        continue
                
                combined_alpha += w_alpha * ranked
                n_active += 1
            
            # Average and scale to return forecast units
            if n_active > 0:
                combined_alpha /= n_active
                # Scale: multiply by cross-sectional return vol
                # This converts rank signal (O(0.1)) to expected return units (O(1e-3))
                # Grinold fundamental law: E[r] ≈ IC × vol × score
                cs_ret_vol = np.nanstd(r_today[r_today != 0]) if np.any(r_today != 0) else 0.01
                combined_alpha *= cs_ret_vol  # now in daily return units
            
            # Apply universe mask
            if date in universe_mask_df.index:
                umask = universe_mask_df.loc[date].reindex(tickers).fillna(False).values
                combined_alpha[~umask] = 0.0
            
            # Sanitize alpha forecast
            combined_alpha = np.nan_to_num(combined_alpha, nan=0.0, posinf=0.0, neginf=0.0)
            # NOTE: Do NOT normalize alpha to sum-to-1 or clip — it's a return forecast,
            # not a weight vector. The QP optimizer will determine weights subject to 
            # its own constraints (including max_position_pct_gmv = 0.01).
            
            # 6. Get risk model matrices
            Q = self.risk_model.get_Q_matrix()
            spec_var = self.risk_model.specific_var
            
            # 7. Get ADV
            if not adv_df.empty and date in adv_df.index:
                adv = adv_df.loc[date].reindex(tickers).fillna(1e5).values
                adv = np.maximum(adv, 1e3)
            else:
                adv = np.full(N, 1e6)
            
            # 8. Run optimizer (in weight space)
            if Q is not None and spec_var is not None and n_active > 0:
                w_new = self.optimizer.optimize(
                    alpha_vec=combined_alpha,
                    Q=Q,
                    spec_var=np.maximum(spec_var, 1e-10),
                    w_prev=w_prev,
                    adv=adv,
                    sector_masks=sector_masks,
                )
            else:
                w_new = w_prev
            
            # Convert weights to dollar holdings for cost computation
            h_new = w_new * cfg.booksize
            h_prev_dollars = w_prev * cfg.booksize
            
            # 9. Compute trading costs (in dollar space)
            trades = h_new - h_prev_dollars
            trade_notional = np.sum(np.abs(trades))
            slippage_cost = self.config.slippage_bps * 1e-4 * trade_notional
            commission_cost = getattr(self.config, 'commission_bps', 0.0) * 1e-4 * trade_notional
            # Market impact: Almgren-Chriss temporary impact = η * σ * √(|trade|/ADV) * |trade|
            # This gives impact proportional to participation rate^0.5, standard in industry
            sigma = np.sqrt(np.maximum(spec_var if spec_var is not None else np.zeros(N), 1e-10))
            safe_adv = np.maximum(adv, 1e3)
            abs_trades = np.abs(trades)
            participation = abs_trades / safe_adv  # fraction of ADV traded
            impact_cost = np.sum(self.config.impact_coeff * sigma * np.sqrt(participation) * abs_trades)
            total_tcost = slippage_cost + commission_cost + impact_cost
            
            # Borrow cost: charge on short positions held overnight
            short_notional = np.sum(np.abs(h_prev_dollars[h_prev_dollars < 0]))
            borrow_cost = cfg.borrow_cost_bps * 1e-4 * short_notional
            total_tcost += borrow_cost
            
            net_pnl = gross_pnl - total_tcost
            
            # 10. Record
            gmv = np.sum(np.abs(h_new))
            turnover = trade_notional / max(gmv, 1)
            
            daily_results.append(DailyResult(
                date=date_str,
                pnl=net_pnl,
                gross_pnl=gross_pnl,
                tcost=total_tcost,
                turnover=turnover,
                gmv=gmv,
                n_long=int(np.sum(w_new > 1e-6)),
                n_short=int(np.sum(w_new < -1e-6)),
                factor_risk=0.5 * self.config.risk_aversion * np.sum((Q @ w_new)**2) if Q is not None else 0,
                specific_risk=0.5 * self.config.risk_aversion * np.sum(spec_var * w_new**2) if spec_var is not None else 0,
            ))
            
            w_prev = w_new
            
            # Progress
            progress = t_idx - start_idx
            if progress % report_every == 0 or t_idx == len(all_dates) - 1:
                cum_pnl = sum(r.pnl for r in daily_results)
                print(f"    [{date_str}] day {progress+1}/{n_days} | "
                      f"cum PnL: ${cum_pnl:+,.0f} | GMV: ${gmv:,.0f} | TO: {turnover:.1%} | "
                      f"longs: {daily_results[-1].n_long} shorts: {daily_results[-1].n_short}")
        
        return self._compute_statistics(daily_results)
    
    def _compute_statistics(self, results: List[DailyResult]) -> Dict:
        """Compute comprehensive statistics from daily results."""
        if not results:
            return {}
        
        cfg = self.config
        oos_start = cfg.oos_start
        
        # Split IS / OOS
        is_results = [r for r in results if r.date < oos_start]
        oos_results = [r for r in results if r.date >= oos_start]
        
        def stats_for(res_list, label):
            if not res_list:
                return {}
            pnls = np.array([r.pnl for r in res_list])
            gross_pnls = np.array([r.gross_pnl for r in res_list])
            tcosts = np.array([r.tcost for r in res_list])
            turnovers = np.array([r.turnover for r in res_list])
            gmvs = np.array([r.gmv for r in res_list])
            
            cum_pnl = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cum_pnl)
            drawdown = cum_pnl - running_max
            
            n_days = len(pnls)
            sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0
            gross_sharpe = np.mean(gross_pnls) / np.std(gross_pnls) * np.sqrt(252) if np.std(gross_pnls) > 0 else 0
            
            avg_gmv = np.mean(gmvs)
            ann_return = np.mean(pnls) * 252 / avg_gmv if avg_gmv > 0 else 0
            ann_gross = np.mean(gross_pnls) * 252 / avg_gmv if avg_gmv > 0 else 0
            max_dd = np.min(drawdown) / avg_gmv if avg_gmv > 0 else 0
            
            # Calmar: ann_return / |max_dd|
            calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0
            
            # Win rate
            win_rate = np.mean(pnls > 0)
            
            # Profit factor
            gains = pnls[pnls > 0].sum()
            losses = abs(pnls[pnls < 0].sum())
            profit_factor = gains / losses if losses > 0 else float('inf')
            
            # Average trade stats
            avg_turnover = np.mean(turnovers)
            total_tcost = np.sum(tcosts)
            
            return {
                "label": label,
                "n_days": n_days,
                "start_date": res_list[0].date,
                "end_date": res_list[-1].date,
                "cum_pnl": float(cum_pnl[-1]),
                "sharpe": float(sharpe),
                "gross_sharpe": float(gross_sharpe),
                "ann_return": float(ann_return),
                "ann_gross_return": float(ann_gross),
                "max_drawdown": float(max_dd),
                "calmar": float(calmar),
                "win_rate": float(win_rate),
                "profit_factor": float(profit_factor),
                "avg_daily_pnl": float(np.mean(pnls)),
                "std_daily_pnl": float(np.std(pnls)),
                "avg_daily_gross": float(np.mean(gross_pnls)),
                "avg_daily_tcost": float(np.mean(tcosts)),
                "total_tcost": float(total_tcost),
                "avg_gmv": float(avg_gmv),
                "avg_turnover": float(avg_turnover),
                "avg_n_long": float(np.mean([r.n_long for r in res_list])),
                "avg_n_short": float(np.mean([r.n_short for r in res_list])),
                "daily_pnls": pnls.tolist(),
                "dates": [r.date for r in res_list],
            }
        
        all_stats = stats_for(results, "FULL")
        is_stats = stats_for(is_results, "IN-SAMPLE")
        oos_stats = stats_for(oos_results, "OUT-OF-SAMPLE")
        
        return {
            "full": all_stats,
            "is": is_stats,
            "oos": oos_stats,
            "daily_results": results,
        }
