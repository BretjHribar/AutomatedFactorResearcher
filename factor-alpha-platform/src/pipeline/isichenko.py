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
    risk_aversion: float = 1e-6        # κ in Isichenko Eq 6.6
    booksize: float = 20_000_000.0     # $20M
    max_position_pct_gmv: float = 0.02 # 2% of booksize per name
    max_position_pct_adv: float = 0.05 # 5% of ADV per name  
    max_trade_pct_adv: float = 0.10    # 10% of ADV max trade
    
    # Transaction costs (Isichenko Ch 5)
    slippage_bps: float = 1.0          # ~1 bps slippage per trade
    impact_coeff: float = 0.1          # Impact ∝ coeff * σ / √ADV
    borrow_cost_bps: float = 1.0       # ~1 bps/day short borrow (≈2.5% annualized)
    trade_aversion: float = 1e-6       # Turnover penalty (optimal from sweep)
    holding_bonus: float = 0.0         # Reward for holding existing positions
    
    # Alpha combination mode
    raw_signal_mode: bool = False      # True = skip OLS/MSE, just average raw signals
    
    # Risk model
    ema_halflife_risk: int = 60        # days for risk estimation
    ema_halflife_alpha: int = 120      # days for alpha scaling
    n_pca_factors: int = 5             # statistical factors beyond sectors
    
    # Neutralization
    dollar_neutral: bool = True
    sector_neutral: bool = True
    
    # Delay
    delay: int = 1


# ═══════════════════════════════════════════════════════════════
# STAGE 1: ALPHA SCALER
# ═══════════════════════════════════════════════════════════════
class AlphaScaler:
    """Scale raw alpha signals to return forecasts via rolling OLS.
    
    Also tracks MSE (mean squared error) of the OLS prediction for use
    in Isichenko Eq 2.38: optimal alpha combination weights = 1/MSE.
    """
    
    def __init__(self, halflife: int = 120):
        self.halflife = halflife
        self.ema_decay = np.log(2) / halflife
        self._running_cov = None   # Cov(signal, return)
        self._running_var = None   # Var(signal)
        self._running_mse = None   # MSE of forecast vs realized
        self._scale = 1e-5  # small non-zero initial so signal isn't killed before warmup
        
    def update(self, signal: pd.Series, returns: pd.Series):
        """Update the scaling factor with new cross-sectional observation."""
        # Only use non-NaN overlapping data
        common = signal.dropna().index.intersection(returns.dropna().index)
        if len(common) < 50:
            return
        s = signal.loc[common].values
        r = returns.loc[common].values
        
        # Demean
        s = s - np.nanmean(s)
        r = r - np.nanmean(r)
        
        # Cross-sectional Cov and Var
        cov_sr = np.nanmean(s * r)
        var_s = np.nanmean(s * s)
        
        alpha = 1 - np.exp(-self.ema_decay)
        if self._running_cov is None:
            self._running_cov = cov_sr
            self._running_var = var_s
        else:
            self._running_cov = alpha * cov_sr + (1 - alpha) * self._running_cov
            self._running_var = alpha * var_s + (1 - alpha) * self._running_var
        
        if self._running_var > 1e-20:
            self._scale = self._running_cov / self._running_var
        
        # Track MSE: E[(r - k*s)^2] for Isichenko Eq 2.38 weighting
        residual = r - self._scale * s
        mse = np.nanmean(residual ** 2)
        if self._running_mse is None:
            self._running_mse = mse
        else:
            self._running_mse = alpha * mse + (1 - alpha) * self._running_mse
    
    def scale(self, signal: pd.Series) -> pd.Series:
        """Apply the current scale factor to a signal to produce a forecast.
        
        Per Isichenko Eq 2.32: if k_a < 0, the alpha is anti-predictive
        and should be zeroed out (not flipped).
        """
        if self._scale <= 0:
            return signal * 0.0  # shut off anti-predictive alphas
        return signal * self._scale
    
    @property
    def is_active(self) -> bool:
        """Whether this scaler has a positive (predictive) scale factor."""
        return self._scale > 0
    
    @property
    def precision_weight(self) -> float:
        """Precision weight = 1/MSE for Isichenko Eq 2.38 combination.
        
        Returns 0 if alpha is anti-predictive or MSE is not yet estimated.
        """
        if self._scale <= 0 or self._running_mse is None or self._running_mse <= 1e-20:
            return 0.0
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
    Minimize: ½κ h'(BFB')h + ½κ h'Sh - α'h + (h-h₀)'Λ(h-h₀) + c'|h-h₀|
             + sector_penalty * Σ (sector_exposure)²
    Subject to: dollar neutral, position limits, GMV limit
    
    Based on Isichenko Eq. 6.6 / 6.63 and Udacity Barra reference.
    Uses soft sector neutrality (penalty) instead of hard constraint.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._prob = None
        self._h_param = None
    
    def optimize(self, 
                 alpha_vec: np.ndarray,        # (N,) combined forecast
                 Q: np.ndarray,                # (K, N) factor risk sqrt
                 spec_var: np.ndarray,          # (N,) specific variance
                 h_prev: np.ndarray,            # (N,) previous holdings
                 adv: np.ndarray,               # (N,) dollar volume
                 sector_masks: List[np.ndarray], # list of boolean masks
                 ) -> np.ndarray:
        """Solve the QP and return optimal holdings h*."""
        
        N = len(alpha_vec)
        cfg = self.config
        kappa = cfg.risk_aversion
        booksize = cfg.booksize
        
        # Position limits
        max_pos_gmv = cfg.max_position_pct_gmv * booksize
        max_pos_adv = cfg.max_position_pct_adv * adv
        max_pos = np.minimum(max_pos_gmv, max_pos_adv)
        max_pos = np.maximum(max_pos, 100.0)
        
        # Transaction cost parameters (Isichenko Eq 5.4)
        sigma = np.sqrt(np.maximum(spec_var, 1e-10))
        safe_adv = np.maximum(adv, 1e4)
        lambda_impact = cfg.impact_coeff * sigma / np.sqrt(safe_adv)
        slippage_cost = cfg.slippage_bps * 1e-4
        
        # --- Build & solve CVXPY Problem ---
        h = cp.Variable(N)
        T = h - h_prev
        
        # 1. Factor risk: ½κ ||Qh||²
        factor_risk = 0.5 * kappa * cp.sum_squares(Q @ h)
        
        # 2. Specific risk: ½κ h'Sh
        specific_risk = 0.5 * kappa * cp.sum(cp.multiply(spec_var, cp.square(h)))
        
        # 3. Expected return
        expected_return = alpha_vec @ h
        
        # 4. Impact cost (quadratic)
        impact_cost = cp.sum(cp.multiply(lambda_impact, cp.square(T)))
        
        # 5. Slippage (linear)
        slippage = slippage_cost * cp.norm(T, 1)
        
        # 6. SOFT sector neutrality penalty (instead of hard constraint)
        sector_penalty = 0.0
        sector_pen_coeff = kappa * 10  # strong penalty
        if cfg.sector_neutral and sector_masks:
            for mask in sector_masks:
                if mask.sum() > 1:
                    sector_penalty += sector_pen_coeff * cp.square(cp.sum(h[mask]))
        
        # 7. Extra turnover aversion (tunable knob)
        trade_penalty = cfg.trade_aversion * cp.sum_squares(T)
        
        objective = cp.Minimize(
            factor_risk + specific_risk - expected_return 
            + impact_cost + slippage + sector_penalty + trade_penalty
        )
        
        # Constraints
        constraints = [
            h >= -max_pos,
            h <= max_pos,
            cp.norm(h, 1) <= booksize,  # GMV constraint (Isichenko Eq 6.6)
        ]
        if cfg.dollar_neutral:
            constraints.append(cp.sum(h) == 0)
        
        prob = cp.Problem(objective, constraints)
        
        # Try OSQP first (fastest), fall back to SCS
        for solver, kwargs in [
            (cp.OSQP, {"max_iter": 10000, "eps_abs": 1e-3, "eps_rel": 1e-3, "verbose": False, "warm_start": True, "time_limit": 30.0}),
            (cp.SCS, {"max_iters": 5000, "verbose": False}),
        ]:
            try:
                prob.solve(solver=solver, **kwargs)
                if prob.status in ["optimal", "optimal_inaccurate"] and h.value is not None:
                    return h.value
            except Exception:
                continue
        
        # Final fallback: simple mean-variance without CVXPY
        return self._fallback_optimize(alpha_vec, spec_var, h_prev, max_pos, kappa, booksize)
    
    def _fallback_optimize(self, alpha_vec, spec_var, h_prev, max_pos, kappa, booksize):
        """Simple analytical mean-variance solution as fallback."""
        # h* = alpha / (kappa * sigma^2) — unconstrained Markowitz
        safe_var = np.maximum(spec_var, 1e-8)
        h = alpha_vec / (kappa * safe_var + 1e-12)
        
        # Clip to position limits
        h = np.clip(h, -max_pos, max_pos)
        
        # Dollar neutralize
        h = h - np.mean(h)
        
        # Scale to booksize
        gmv = np.sum(np.abs(h))
        if gmv > booksize:
            h = h * booksize / gmv
        
        return h


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
        
        # Initialize scalers
        for expr in alpha_signals:
            self.alpha_scalers[expr] = AlphaScaler(halflife=cfg.ema_halflife_alpha)
        
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
        h_prev = np.zeros(N)  # Start flat
        daily_results = []
        
        # Warmup: train risk model and alpha scalers
        warmup_end = start_idx
        print(f"  Warming up risk model ({cfg.warmup_days} days)...")
        warmup_start = max(0, start_idx - cfg.warmup_days)
        
        for t in range(warmup_start, start_idx):
            date = all_dates[t]
            r = returns.iloc[t].reindex(tickers).fillna(0.0).values
            self.risk_model.update(r)
            
            # Update alpha scalers
            if t > 0:
                prev_date = all_dates[t - 1]
                fwd_ret = returns.iloc[t].reindex(tickers)
                for expr, sig_df in alpha_signals.items():
                    if prev_date in sig_df.index:
                        sig = sig_df.loc[prev_date]
                        self.alpha_scalers[expr].update(sig, fwd_ret)
        
        print(f"  Running backtest from {all_dates[start_idx]} to {all_dates[-1]}...\n")
        
        n_days = len(all_dates) - start_idx
        report_every = max(1, n_days // 20)
        
        for t_idx in range(start_idx, len(all_dates)):
            date = all_dates[t_idx]
            date_str = str(date)[:10]
            
            # 1. Get today's returns (realized)
            r_today = returns.iloc[t_idx].reindex(tickers).fillna(0.0).values
            
            # 2. PnL from yesterday's positions (delay=1 means we entered yesterday)
            gross_pnl = np.dot(h_prev, r_today)
            
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
            
            # 4. Update alpha scalers with (yesterday's signal, today's return)
            if t_idx > 0:
                prev_date = all_dates[t_idx - 1]
                fwd_ret = returns.iloc[t_idx].reindex(tickers)
                for expr, sig_df in alpha_signals.items():
                    if prev_date in sig_df.index:
                        sig = sig_df.loc[prev_date]
                        self.alpha_scalers[expr].update(sig, fwd_ret)
            
            # 5. Construct combined alpha forecast for tomorrow
            combined_alpha = np.zeros(N)
            n_active = 0
            
            if self.config.raw_signal_mode:
                # RAW MODE: skip OLS scaling, cross-sectionally demean & scale
                for expr, sig_df in alpha_signals.items():
                    if date in sig_df.index:
                        raw_signal = sig_df.loc[date].reindex(tickers).fillna(0.0)
                        sv = raw_signal.values if isinstance(raw_signal, pd.Series) else raw_signal
                        sv = np.nan_to_num(sv, nan=0.0, posinf=0.0, neginf=0.0)
                        if np.any(sv != 0):
                            # Demean cross-sectionally (dollar neutral)
                            sv = sv - np.mean(sv[sv != 0]) if np.any(sv != 0) else sv
                            # Scale to bps-level returns (rank-based signals are O(1))
                            std = np.std(sv[sv != 0]) if np.any(sv != 0) else 1.0
                            if std > 0:
                                sv = sv / std * 0.001  # ~10 bps cross-sectional spread
                            combined_alpha += sv
                            n_active += 1
                # Average
                if n_active > 0:
                    combined_alpha /= n_active
            else:
                # OLS + MSE MODE: Isichenko Eq 2.32 + 2.38
                forecasts_and_weights = []
                for expr, sig_df in alpha_signals.items():
                    if date in sig_df.index:
                        raw_signal = sig_df.loc[date].reindex(tickers).fillna(0.0)
                        scaled = self.alpha_scalers[expr].scale(raw_signal)
                        sv = scaled.values if isinstance(scaled, pd.Series) else scaled
                        sv = np.nan_to_num(sv, nan=0.0, posinf=0.0, neginf=0.0)
                        w = self.alpha_scalers[expr].precision_weight
                        if w > 0 and np.any(sv != 0):
                            forecasts_and_weights.append((sv, w))
                            n_active += 1
                # Combine: scale by relative precision (w / mean_w)
                if forecasts_and_weights:
                    weights = np.array([w for _, w in forecasts_and_weights])
                    mean_w = np.mean(weights)
                    for sv, w in forecasts_and_weights:
                        combined_alpha += (w / mean_w) * sv
            
            # Apply universe mask
            if date in universe_mask_df.index:
                umask = universe_mask_df.loc[date].reindex(tickers).fillna(False).values
                combined_alpha[~umask] = 0.0
            
            # Sanitize & clip (Isichenko: forecasts should be small, bps-level)
            combined_alpha = np.nan_to_num(combined_alpha, nan=0.0, posinf=0.0, neginf=0.0)
            combined_alpha = np.clip(combined_alpha, -0.05, 0.05)
            
            # 6. Get risk model matrices
            Q = self.risk_model.get_Q_matrix()
            spec_var = self.risk_model.specific_var
            
            # 7. Get ADV
            if not adv_df.empty and date in adv_df.index:
                adv = adv_df.loc[date].reindex(tickers).fillna(1e5).values
                adv = np.maximum(adv, 1e3)
            else:
                adv = np.full(N, 1e6)
            
            # 8. Run optimizer
            if Q is not None and spec_var is not None and n_active > 0:
                h_new = self.optimizer.optimize(
                    alpha_vec=combined_alpha,
                    Q=Q,
                    spec_var=np.maximum(spec_var, 1e-10),
                    h_prev=h_prev,
                    adv=adv,
                    sector_masks=sector_masks,
                )
            else:
                h_new = h_prev
            
            # 9. Compute trading costs
            trades = h_new - h_prev
            slippage_cost = self.config.slippage_bps * 1e-4 * np.sum(np.abs(trades))
            sigma = np.sqrt(np.maximum(spec_var if spec_var is not None else np.zeros(N), 1e-10))
            safe_adv = np.maximum(adv, 1e3)
            impact_cost = np.sum(self.config.impact_coeff * sigma / np.sqrt(safe_adv) * trades**2)
            total_tcost = slippage_cost + impact_cost
            
            # Borrow cost: charge on short positions held overnight (Isichenko Ch 5.3)
            short_notional = np.sum(np.abs(h_prev[h_prev < 0]))
            borrow_cost = cfg.borrow_cost_bps * 1e-4 * short_notional
            total_tcost += borrow_cost
            
            net_pnl = gross_pnl - total_tcost
            
            # 10. Record
            gmv = np.sum(np.abs(h_new))
            turnover = np.sum(np.abs(trades)) / max(gmv, 1)
            
            daily_results.append(DailyResult(
                date=date_str,
                pnl=net_pnl,
                gross_pnl=gross_pnl,
                tcost=total_tcost,
                turnover=turnover,
                gmv=gmv,
                n_long=int(np.sum(h_new > 100)),
                n_short=int(np.sum(h_new < -100)),
                factor_risk=0.5 * self.config.risk_aversion * np.sum((Q @ h_new)**2) if Q is not None else 0,
                specific_risk=0.5 * self.config.risk_aversion * np.sum(spec_var * h_new**2) if spec_var is not None else 0,
            ))
            
            h_prev = h_new
            
            # Progress
            progress = t_idx - start_idx
            if progress % report_every == 0 or t_idx == len(all_dates) - 1:
                cum_pnl = sum(r.pnl for r in daily_results)
                avg_gmv = np.mean([r.gmv for r in daily_results[-20:]]) if daily_results else 0
                print(f"    [{date_str}] day {progress+1}/{n_days} | "
                      f"cum PnL: ${cum_pnl:+,.0f} | GMV: ${avg_gmv:,.0f} | "
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
