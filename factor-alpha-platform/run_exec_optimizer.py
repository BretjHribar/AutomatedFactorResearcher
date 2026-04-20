"""
run_exec_optimizer.py — Execution-Optimized Portfolio Construction

Takes the aggregate signal from each combiner (Billion Alphas, PCA Hedge,
Return-Weighted) and runs a CVXPY convex optimizer to produce optimal
portfolio weights that balance:
  1. Signal tracking (closeness to alpha signal)
  2. Risk minimization (PCA factor risk model from return covariance)
  3. Transaction cost minimization (L1 penalty on trades)

This replaces the naive `process_signal -> normalize` step with a proper
QP: max a'w - 0.5*k*w'Sw - lam*||w - w_prev||_1

Uses OSQP solver (already installed via cvxpy).
"""

import sys, os, time, sqlite3
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_alpha_5m as ea
import eval_portfolio_5m as ep

# ============================================================================
# CONFIGURATION
# ============================================================================

FEES_BPS     = 0.0   # Fees applied in simulation (separate from optimizer tcost)
UNIVERSE     = "BINANCE_TOP50"
INTERVAL     = "5m"
BOOKSIZE     = ea.BOOKSIZE
BARS_PER_DAY = ea.BARS_PER_DAY
NEUTRALIZE   = ea.NEUTRALIZE
DB_PATH      = ea.DB_PATH

ea.UNIVERSE = UNIVERSE
ucfg = ea.UNIVERSE_CONFIG.get(UNIVERSE, ea.UNIVERSE_CONFIG["BINANCE_TOP100"])
MAX_WEIGHT = ucfg["max_weight"]
ea.MAX_WEIGHT = MAX_WEIGHT

SPLITS = {
    "train": ("2025-02-01", "2026-02-01"),
    "val":   ("2026-02-01", "2026-03-01"),
    "test":  ("2026-03-01", "2026-03-27"),
}

# Optimizer params
RISK_AVERSION   = 1.0       # κ — risk penalty
TCOST_LAMBDA    = 0.0003    # λ — L1 turnover penalty (in weight-space)
REBALANCE_EVERY = 12        # Solve QP every N bars (12 = 1 hour)
N_PCA_FACTORS   = 3         # Number of PCA factors for risk model
RISK_LOOKBACK   = 576       # 2 days of 5m bars for covariance estimation
MAX_GMV         = 1.0       # Max gross exposure (as fraction of booksize)


# ============================================================================
# PCA RISK MODEL FOR CRYPTO
# ============================================================================

class CryptoPcaRiskModel:
    """Simple PCA-based risk model for crypto.
    
    No sectors/styles — uses rolling PCA on return covariance.
    C = B @ F @ B' + diag(spec_var)
    where B = top-k PCA loadings, F = factor covariance.
    """
    
    def __init__(self, n_factors=3, lookback=576, halflife=288):
        self.n_factors = n_factors
        self.lookback = lookback
        self.halflife = halflife
        self._return_buffer = []
        self.factor_loadings = None   # (N, K)
        self.factor_cov = None        # (K, K) 
        self.specific_var = None      # (N,)
        self._n_tickers = 0
    
    def update(self, returns_row: np.ndarray):
        """Add one bar of cross-sectional returns."""
        self._return_buffer.append(returns_row.copy())
        if len(self._return_buffer) > self.lookback * 2:
            self._return_buffer = self._return_buffer[-self.lookback:]
    
    def fit(self, n_tickers: int):
        """Fit PCA risk model from buffered returns."""
        self._n_tickers = n_tickers
        if len(self._return_buffer) < max(self.lookback // 2, 50):
            # Not enough data — use identity
            self.factor_loadings = np.zeros((n_tickers, self.n_factors))
            self.factor_cov = np.eye(self.n_factors) * 1e-6
            self.specific_var = np.ones(n_tickers) * 1e-4
            return
        
        # Use last `lookback` bars with exponential weighting
        buf = np.array(self._return_buffer[-self.lookback:])  # (T, N)
        buf = np.nan_to_num(buf, nan=0.0, posinf=0.0, neginf=0.0)
        T, N = buf.shape
        
        # EWM weights
        alpha = np.log(2) / self.halflife
        weights = np.exp(-alpha * np.arange(T)[::-1])
        weights /= weights.sum()
        
        # Weighted demeaning
        wmean = (weights[:, None] * buf).sum(axis=0)
        centered = buf - wmean[None, :]
        
        # Weighted covariance
        wcov = (centered * weights[:, None]).T @ centered  # (N, N)
        
        # PCA
        try:
            eigvals, eigvecs = np.linalg.eigh(wcov)
            # Take top-k
            idx = np.argsort(eigvals)[::-1][:self.n_factors]
            B = eigvecs[:, idx]  # (N, K)
            
            # Factor returns = B' @ r for each bar
            factor_rets = centered @ B  # (T, K)
            
            # Factor covariance
            F = (factor_rets * weights[:, None]).T @ factor_rets  # (K, K)
            
            # Specific variance = diag(residual cov)
            residuals = centered - factor_rets @ B.T
            spec_var = np.sum(weights[:, None] * residuals**2, axis=0)
            spec_var = np.maximum(spec_var, 1e-8)
            
            self.factor_loadings = B
            self.factor_cov = F
            self.specific_var = spec_var
        except np.linalg.LinAlgError:
            self.factor_loadings = np.zeros((N, self.n_factors))
            self.factor_cov = np.eye(self.n_factors) * 1e-6
            self.specific_var = np.ones(N) * 1e-4
    
    def get_Q_matrix(self):
        """Return Q such that Q'Q ≈ B @ F @ B' (factor risk component)."""
        if self.factor_cov is None or self.factor_loadings is None:
            return None
        F = self.factor_cov
        eigvals, eigvecs = np.linalg.eigh(F)
        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals))  # (K, K)
        Q = L.T @ self.factor_loadings.T          # (K, N)
        return Q


# ============================================================================
# CONVEX OPTIMIZER
# ============================================================================

def optimize_weights_qp(
    alpha_signal: np.ndarray,    # (N,) raw composite signal
    w_prev: np.ndarray,          # (N,) previous weights
    risk_model: CryptoPcaRiskModel,
    universe_mask: np.ndarray,   # (N,) bool — tradeable assets
    kappa: float = 1.0,          # risk aversion
    tcost_lambda: float = 0.0003,# turnover penalty
    max_wt: float = 0.05,       # per-stock max weight
    max_gmv: float = 1.0,       # max gross
) -> np.ndarray:
    """Solve the portfolio QP:
    
    max  α'w - ½κ w'Σw - λ||w - w_prev||₁
    s.t. sum(w) = 0 (dollar neutral)
         |w_i| <= max_wt
         sum(|w_i|) <= max_gmv
         w_i = 0 for non-universe assets
    """
    N = len(alpha_signal)
    
    # Prepare alpha — cross-sectional z-score, masked
    alpha = alpha_signal.copy()
    alpha[~universe_mask] = 0.0
    alpha = np.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize alpha to return-forecast scale
    astd = np.std(alpha[universe_mask]) if universe_mask.sum() > 5 else 1.0
    if astd > 0:
        alpha = alpha / astd * 0.01  # Scale to ~1% return forecast
    
    # Build CVXPY problem
    w = cp.Variable(N)
    T = w - w_prev
    
    # Objective components
    # 1. Alpha tracking: α'w
    expected_return = alpha @ w
    
    # 2. Factor risk: ½κ ||Qw||²
    Q = risk_model.get_Q_matrix()
    if Q is not None:
        factor_risk = 0.5 * kappa * cp.sum_squares(Q @ w)
    else:
        factor_risk = 0.0
    
    # 3. Specific risk: ½κ Σ spec_var_i * w_i²
    spec_var = risk_model.specific_var if risk_model.specific_var is not None else np.ones(N) * 1e-4
    specific_risk = 0.5 * kappa * cp.sum(cp.multiply(spec_var, cp.square(w)))
    
    # 4. Transaction cost: λ * ||T||₁
    tcost = tcost_lambda * cp.norm(T, 1)
    
    objective = cp.Minimize(
        factor_risk + specific_risk - expected_return + tcost
    )
    
    # Constraints
    w_max = np.full(N, max_wt)
    w_max[~universe_mask] = 0.0  # Force out-of-universe to zero
    
    constraints = [
        w >= -w_max,
        w <= w_max,
        cp.norm(w, 1) <= max_gmv,
        cp.sum(w) == 0,  # Dollar neutral
    ]
    
    prob = cp.Problem(objective, constraints)
    
    # Solve with OSQP (fast for QP), fallback to SCS
    for solver, kwargs in [
        (cp.OSQP, {"max_iter": 5000, "eps_abs": 1e-4, "eps_rel": 1e-4,
                    "verbose": False, "warm_start": True}),
        (cp.SCS, {"max_iters": 5000, "verbose": False}),
    ]:
        try:
            prob.solve(solver=solver, **kwargs)
            if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
                return w.value
        except Exception:
            continue
    
    # Fallback: simple signal-proportional weights
    return _fallback_weights(alpha, universe_mask, max_wt, max_gmv)


def _fallback_weights(alpha, universe_mask, max_wt, max_gmv):
    """Simple mean-variance fallback if QP fails."""
    w = alpha.copy()
    w[~universe_mask] = 0.0
    w = np.clip(w, -max_wt, max_wt)
    w = w - np.mean(w[universe_mask]) * universe_mask.astype(float)
    gmv = np.sum(np.abs(w))
    if gmv > max_gmv:
        w *= max_gmv / gmv
    return w


# ============================================================================
# EXECUTION-OPTIMIZED SIMULATION
# ============================================================================

def run_exec_optimized_sim(
    composite: pd.DataFrame,     # (T, N) raw composite signal per bar
    full_matrices: dict,
    universe_df: pd.DataFrame,
    valid_tickers: list,
    split_name: str,
    fees_bps: float = 0.0,
    rebalance_every: int = 12,
    kappa: float = 1.0,
    tcost_lambda: float = 0.0003,
    n_pca: int = 3,
):
    """Run the full exec-optimized simulation for one split.
    
    Returns the same SimResult as ea.simulate for comparison.
    """
    start, end = SPLITS[split_name]
    
    # Slice data
    close = full_matrices["close"].loc[start:end]
    returns = close.pct_change()
    uni = universe_df[valid_tickers].loc[start:end]
    
    # Slice composite
    comp = composite.loc[start:end]
    
    # Align columns
    tickers = valid_tickers
    N = len(tickers)
    T_bars = len(comp)
    
    # Initialize risk model — warm up from training data
    risk_model = CryptoPcaRiskModel(n_factors=n_pca, lookback=RISK_LOOKBACK, halflife=288)
    
    # Warm up risk model with pre-split data
    warmup_start = pd.Timestamp(start) - pd.Timedelta(days=5)
    warmup_returns = full_matrices["close"].loc[str(warmup_start):start].pct_change()
    for _, row in warmup_returns.iterrows():
        r = row.reindex(tickers).fillna(0.0).values
        risk_model.update(r)
    risk_model.fit(N)
    
    # Run bar-by-bar simulation
    w_prev = np.zeros(N)
    all_weights = []
    refit_counter = 0
    
    for t in range(T_bars):
        # Get this bar's return
        if t < len(returns):
            r_t = returns.iloc[t].reindex(tickers).fillna(0.0).values
        else:
            r_t = np.zeros(N)
        
        # Update risk model with returns
        risk_model.update(r_t)
        
        # Rebalance?
        if t % rebalance_every == 0:
            # Refit risk model periodically (every ~12h = every 144 rebalances)
            refit_counter += 1
            if refit_counter % (144 // max(rebalance_every, 1) + 1) == 0 or t == 0:
                risk_model.fit(N)
            
            # Get signal for this bar
            if t < len(comp):
                alpha_raw = comp.iloc[t].reindex(tickers).fillna(0.0).values
            else:
                alpha_raw = np.zeros(N)
            
            # Get universe mask
            if t < len(uni):
                umask = uni.iloc[t].reindex(tickers).fillna(False).values.astype(bool)
            else:
                umask = np.ones(N, dtype=bool)
            
            # Solve QP
            w_new = optimize_weights_qp(
                alpha_signal=alpha_raw,
                w_prev=w_prev,
                risk_model=risk_model,
                universe_mask=umask,
                kappa=kappa,
                tcost_lambda=tcost_lambda,
                max_wt=MAX_WEIGHT,
                max_gmv=MAX_GMV,
            )
            w_prev = w_new
        
        all_weights.append(w_prev.copy())
    
    # Convert weights to alpha DataFrame for ea.simulate
    weights_df = pd.DataFrame(all_weights, index=comp.index[:len(all_weights)], columns=tickers)
    
    # Simulate using eval_alpha_5m's exact pipeline
    sim = ea.simulate(
        alpha_df=weights_df,
        returns_df=returns,
        close_df=close,
        universe_df=uni,
        fees_bps=fees_bps,
    )
    return sim


# ============================================================================
# DATA LOADING (same as run_compare_combiners.py)
# ============================================================================

def load_alphas():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT a.id, a.expression, COALESCE(e.ic_mean, 0), COALESCE(e.sharpe_is, 0)
        FROM alphas a LEFT JOIN evaluations e ON e.alpha_id = a.id
        WHERE a.archived = 0 AND a.universe = ?
        ORDER BY COALESCE(e.ic_mean, 0) DESC
    """, (UNIVERSE,)).fetchall()
    conn.close()
    return rows


def load_full_data():
    mat_dir = Path(f"data/binance_cache/matrices/{INTERVAL}")
    uni_path = Path(f"data/binance_cache/universes/{UNIVERSE}_{INTERVAL}.parquet")
    universe_df = pd.read_parquet(uni_path)
    coverage = universe_df.sum(axis=0) / len(universe_df)
    valid_tickers = sorted(coverage[coverage > ea.COVERAGE_CUTOFF].index.tolist())
    
    print(f"  Loading 5m matrices ({len(valid_tickers)} tickers)...", flush=True)
    t0 = time.time()
    matrices = {}
    for fp in sorted(mat_dir.glob("*.parquet")):
        df = pd.read_parquet(fp)
        cols = [c for c in valid_tickers if c in df.columns]
        if cols:
            matrices[fp.stem] = df[cols]
    print(f"  Loaded {len(matrices)} fields in {time.time()-t0:.1f}s", flush=True)
    
    ea._DATA_CACHE['_full_matrices'] = matrices
    ea._DATA_CACHE['_tickers'] = valid_tickers
    ea._DATA_CACHE['_universe_df'] = universe_df
    
    return matrices, universe_df, valid_tickers


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0_total = time.time()
    
    alphas = load_alphas()
    print(f"\n{'='*80}")
    print(f"  EXECUTION OPTIMIZER COMPARISON")
    print(f"  Universe: {UNIVERSE} | Fees: {FEES_BPS} bps | Alphas: {len(alphas)}")
    print(f"  Risk: kappa={RISK_AVERSION}, PCA-{N_PCA_FACTORS} | TCost: lambda={TCOST_LAMBDA}")
    print(f"  Rebalance: every {REBALANCE_EVERY} bars ({REBALANCE_EVERY*5/60:.1f}h)")
    print(f"{'='*80}")
    for a in alphas[:5]:
        print(f"  #{a[0]:3d} IC={a[2]:+.05f} SR={a[3]:+.2f} | {a[1][:60]}")
    print(f"  ... ({len(alphas)} total)")
    
    if not alphas:
        print("  No alphas found!")
        return
    
    # Load data
    full_matrices, universe_df, valid_tickers = load_full_data()
    
    # Build full-range data
    full_start, full_end = "2025-02-01", "2026-03-27"
    full_mat = {name: df.loc[full_start:full_end] for name, df in full_matrices.items()}
    full_uni = universe_df[valid_tickers].loc[full_start:full_end]
    if "close" in full_mat:
        full_mat["returns"] = full_mat["close"].pct_change()
    full_returns = full_mat.get("returns")
    
    # ── Build composites for each combiner ──
    COMBINERS = {
        "PCA Hedge (3 factors)": {
            "func": "ts_autonorm",
            "kwargs": {
                "gamma": 0.01, "ic_weighted": True, "signal_smooth": 12,
                "signal_hedge": 3, "rolling_ic": 0, "rolling_return": 0,
                "concordance_boost": False, "rank_norm": False, "beta_hedge": False,
            },
        },
        "Billion Alphas": {
            "func": "billion_alphas",
            "kwargs": {
                "lookback_days": 15, "retrain_every": 288,
                "gamma": 0.01, "signal_smooth": 12,
            },
        },
        "Return-Weighted": {
            "func": "ts_autonorm",
            "kwargs": {
                "gamma": 0.01, "ic_weighted": True, "signal_smooth": 12,
                "signal_hedge": 0, "rolling_ic": 0, "rolling_return": 1440,
                "concordance_boost": False, "rank_norm": False, "beta_hedge": False,
            },
        },
    }
    
    composites = {}
    for cname, cfg in COMBINERS.items():
        print(f"\n  Building composite: {cname}...", flush=True)
        func_name = cfg["func"]
        kwargs = cfg["kwargs"]
        if func_name == "ts_autonorm":
            composite = ep.compute_composite_ts_autonorm(
                alphas, full_mat, full_uni, full_returns, **kwargs
            )
        elif func_name == "billion_alphas":
            composite = ep.compute_composite_billion_alphas(
                alphas, full_mat, full_uni, full_returns, **kwargs
            )
        else:
            raise ValueError(f"Unknown combiner: {func_name}")
        if composite is not None:
            composites[cname] = composite
    
    if not composites:
        print("  No composites built!")
        return
    
    # ── Run both RAW and EXEC-OPTIMIZED simulation for each combiner ──
    all_results = {}  # {method_label: {split: sim}}
    
    for cname, composite in composites.items():
        # --- RAW (baseline — current pipeline) ---
        raw_label = f"{cname} (raw)"
        all_results[raw_label] = {}
        for split_name in ["train", "val", "test"]:
            start, end = SPLITS[split_name]
            comp_slice = composite.loc[start:end]
            close = full_matrices["close"].loc[start:end]
            returns = close.pct_change()
            uni = universe_df[valid_tickers].loc[start:end]
            alpha_processed = ea.process_signal(comp_slice, universe_df=uni, max_wt=MAX_WEIGHT)
            sim = ea.simulate(alpha_df=alpha_processed, returns_df=returns,
                            close_df=close, universe_df=uni, fees_bps=FEES_BPS)
            all_results[raw_label][split_name] = sim
        
        print(f"  OK {raw_label}: train={all_results[raw_label]['train'].sharpe:+.2f} "
              f"val={all_results[raw_label]['val'].sharpe:+.2f} "
              f"test={all_results[raw_label]['test'].sharpe:+.2f}", flush=True)
        
        # --- EXEC-OPTIMIZED ---
        opt_label = f"{cname} (QP-opt)"
        all_results[opt_label] = {}
        for split_name in ["train", "val", "test"]:
            print(f"    QP-opt {cname} / {split_name}...", flush=True)
            t1 = time.time()
            sim = run_exec_optimized_sim(
                composite=composite,
                full_matrices=full_matrices,
                universe_df=universe_df,
                valid_tickers=valid_tickers,
                split_name=split_name,
                fees_bps=FEES_BPS,
                rebalance_every=REBALANCE_EVERY,
                kappa=RISK_AVERSION,
                tcost_lambda=TCOST_LAMBDA,
                n_pca=N_PCA_FACTORS,
            )
            all_results[opt_label][split_name] = sim
            elapsed = time.time() - t1
            print(f"      SR={sim.sharpe:+.3f} TO={sim.turnover:.4f} ({elapsed:.0f}s)", flush=True)
        
        print(f"  OK {opt_label}: train={all_results[opt_label]['train'].sharpe:+.2f} "
              f"val={all_results[opt_label]['val'].sharpe:+.2f} "
              f"test={all_results[opt_label]['test'].sharpe:+.2f}", flush=True)
    
    # ============================================================================
    # SUMMARY TABLE
    # ============================================================================
    print(f"\n\n{'='*100}")
    print(f"  EXECUTION OPTIMIZER RESULTS -- RAW vs QP-OPTIMIZED")
    print(f"  ({len(alphas)} alphas, {UNIVERSE}, {FEES_BPS:.0f} bps, rebal every {REBALANCE_EVERY} bars)")
    print(f"{'='*100}")
    
    for label in sorted(all_results.keys()):
        print(f"\n  -- {label} --")
        print(f"  {'Split':<8} {'Sharpe':>8} {'Ret%':>9} {'RetAnn%':>10} {'MaxDD%':>8} {'TO':>8} {'Fitness':>8}")
        print(f"  {'-'*70}")
        for split_name in ["train", "val", "test"]:
            s = all_results[label][split_name]
            ret_pct = s.total_pnl / BOOKSIZE * 100
            print(f"  {split_name:<8} {s.sharpe:+8.3f} {ret_pct:+8.2f}% "
                  f"{s.returns_ann*100:+9.2f}% {s.max_drawdown*100:7.2f}% "
                  f"{s.turnover:7.4f} {s.fitness:7.2f}")
    
    # ============================================================================
    # COMPARISON CHART
    # ============================================================================
    print(f"\n  Generating charts...", flush=True)
    
    # Colors: raw = dashed, opt = solid
    base_colors = {
        "PCA Hedge (3 factors)": "#2196F3",
        "Billion Alphas":       "#4CAF50",
        "Return-Weighted":      "#FF9800",
    }
    
    fig, axes = plt.subplots(2, 1, figsize=(20, 14),
                              gridspec_kw={'height_ratios': [3, 1.5]})
    
    # Panel 1: Stitched cumulative PnL
    ax1 = axes[0]
    
    # Draw split backgrounds
    ref_label = list(all_results.keys())[0]
    offset = 0
    for split_name in ["train", "val", "test"]:
        n = len(all_results[ref_label][split_name].daily_pnl)
        ax1.axvspan(offset, offset + n, alpha=0.06,
                   color=['#2196F3', '#FF9800', '#F44336'][["train","val","test"].index(split_name)])
        if split_name != "train":
            ax1.axvline(x=offset, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        offset += n
    
    for label in sorted(all_results.keys()):
        running_offset = 0.0
        bar_off = 0
        all_x, all_y = [], []
        
        for split_name in ["train", "val", "test"]:
            s = all_results[label][split_name]
            cum = s.cumulative_pnl.values / BOOKSIZE * 100 + running_offset
            n = len(cum)
            x = np.arange(bar_off, bar_off + n)
            all_x.extend(x)
            all_y.extend(cum)
            running_offset = cum[-1]
            bar_off += n
        
        # Color and style
        is_opt = "(QP-opt)" in label
        base_name = label.replace(" (raw)", "").replace(" (QP-opt)", "")
        color = base_colors.get(base_name, '#999999')
        linestyle = '-' if is_opt else '--'
        linewidth = 2.5 if is_opt else 1.5
        alpha_plot = 1.0 if is_opt else 0.5
        
        sr_test = all_results[label]["test"].sharpe
        ax1.plot(all_x, all_y, color=color, linewidth=linewidth,
                linestyle=linestyle, alpha=alpha_plot,
                label=f"{label}  (test SR={sr_test:+.2f})")
    
    ax1.set_title(f"Execution Optimizer: Raw Signal vs QP-Optimized\n"
                  f"{len(alphas)} alphas, {UNIVERSE}, {FEES_BPS:.0f} bps, "
                  f"rebal every {REBALANCE_EVERY} bars ({REBALANCE_EVERY*5/60:.1f}h)",
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel("Cumulative Return (%)", fontsize=12)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.25)
    ax1.axhline(0, color='black', linewidth=0.5)
    
    # Panel 2: Bar chart — test Sharpe comparison
    ax2 = axes[1]
    combiner_names = list(composites.keys())
    n_c = len(combiner_names)
    x_pos = np.arange(n_c)
    bar_width = 0.35
    
    raw_srs = [all_results[f"{c} (raw)"]["test"].sharpe for c in combiner_names]
    opt_srs = [all_results[f"{c} (QP-opt)"]["test"].sharpe for c in combiner_names]
    
    bars1 = ax2.bar(x_pos - bar_width/2, raw_srs, bar_width,
                    label='Raw (normalize)', color='#BBDEFB', edgecolor='#2196F3', linewidth=1.5)
    bars2 = ax2.bar(x_pos + bar_width/2, opt_srs, bar_width,
                    label='QP-Optimized', color='#4CAF50', edgecolor='#2E7D32', linewidth=1.5)
    
    for i, v in enumerate(raw_srs):
        ax2.text(x_pos[i] - bar_width/2, v + (0.1 if v >= 0 else -0.3),
                f"{v:+.2f}", ha='center', fontsize=10, fontweight='bold', color='#2196F3')
    for i, v in enumerate(opt_srs):
        ax2.text(x_pos[i] + bar_width/2, v + (0.1 if v >= 0 else -0.3),
                f"{v:+.2f}", ha='center', fontsize=10, fontweight='bold', color='#2E7D32')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(combiner_names, fontsize=11, fontweight='bold')
    ax2.set_ylabel("Test Sharpe Ratio", fontsize=11)
    ax2.set_title("Test Period: Raw vs QP-Optimized", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.grid(True, alpha=0.25, axis='y')
    
    plt.tight_layout()
    chart_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "exec_optimizer_comparison.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: {chart_path}")
    
    elapsed = time.time() - t0_total
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
