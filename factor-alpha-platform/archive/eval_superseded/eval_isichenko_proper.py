"""
Proper Isichenko Pipeline for Crypto — FOLLOWING THE BOOK.

Two-stage approach per Isichenko (2021):
  Stage 1 (Ch 3.3, Eq 3.18): Rolling ridge regression of forward returns on
           rank-normalized alpha signals. Output: combined return forecast μ̂(t)
           in return units (bps).
  Stage 2 (Ch 6, Eq 6.6): QP optimizer: max [f·P - c|T| - kPCP]
           With PCA covariance for risk, dollar-neutral, position limits.

Key insights from the book:
- Forecasts should predict RETURNS, not be treated as weights (Sec 2.12, 3.3)
- Ridge regression (Sec 2.4.11.6) is preferred for combining many correlated forecasts
- The combined forecast scale matters for the optimizer (Sec 3.3, bottom of p.174)
- Factor risk model C = BFB' + Dσ² (Ch 4) — for crypto, use PCA factors
- Utility Eq 6.6: F(P) = f·P - I·T - c·|T| - kPCP
"""

import sys, warnings, time
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import sqlite3
import cvxpy as cp
from pathlib import Path
from sklearn.linear_model import Ridge

sys.path.insert(0, '.')
from src.operators.fastexpression import FastExpressionEngine
from src.simulation.vectorized_sim_polars import simulate_vectorized_polars as sim_vec

DATA_DIR = Path("data/binance_cache")
INTERVAL = "4h"
TRAIN_END = "2024-04-27"
BOOKSIZE = 2_000_000.0
FEES_BPS = 5.0
MAX_WT = 0.05
ANN_FACTOR = np.sqrt(6 * 252)  # 4h bars → annual (1512 bars/year)

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
print("=" * 80)
print("ISICHENKO PIPELINE — CRYPTO (FOLLOWING THE BOOK)")
print("=" * 80)

t0 = time.time()

# Load matrices
matrices = {}
for f in sorted((DATA_DIR / "matrices" / INTERVAL).glob("*.parquet")):
    matrices[f.stem] = pd.read_parquet(f)
universe_df = pd.read_parquet(DATA_DIR / f"universes/BINANCE_TOP100_{INTERVAL}.parquet")

close = matrices["close"].copy()
returns = matrices["returns"].copy()
for col in close.columns:
    lv = close[col].last_valid_index()
    if lv is not None and lv < close.index[-1]:
        returns.loc[returns.index > lv, col] = 0.0
matrices["returns"] = returns

all_tickers = sorted(universe_df.columns[universe_df.any()].tolist())
for name in list(matrices.keys()):
    cols = [c for c in all_tickers if c in matrices[name].columns]
    if cols:
        matrices[name] = matrices[name][cols]
    else:
        del matrices[name]

tickers = matrices["close"].columns.tolist()
N = len(tickers)
dates = matrices["returns"].index
T_total = len(dates)

TERMINALS = [t for t in [
    "close", "open", "high", "low", "volume", "returns", "log_returns",
    "taker_buy_ratio", "taker_buy_volume", "vwap", "vwap_deviation",
    "high_low_range", "open_close_range", "adv20", "adv60",
    "volume_ratio_20d", "historical_volatility_20", "historical_volatility_60",
    "momentum_5d", "momentum_20d", "momentum_60d", "beta_to_btc",
    "overnight_gap", "upper_shadow", "lower_shadow",
    "close_position_in_range", "trades_count", "trades_per_volume",
    "parkinson_volatility_20", "quote_volume",
] if t in matrices]

print(f"  {N} tickers, {T_total} days, {len(TERMINALS)} data fields")

# ═══════════════════════════════════════════════════════════════
# LOAD ALPHAS (56 proven 4h expressions from alphas.db)
# ═══════════════════════════════════════════════════════════════
conn = sqlite3.connect("data/alphas.db")
cur = conn.cursor()
cur.execute("""SELECT expression FROM alphas WHERE asset_class='crypto' AND interval='4h'""")
expressions = [r[0] for r in cur.fetchall()]
conn.close()
print(f"  {len(expressions)} alpha expressions")

# Pre-compute alpha signals (rank-normalized)
print("  Pre-computing alpha signals...")
engine = FastExpressionEngine(data_fields={n: matrices[n] for n in TERMINALS if n in matrices})
signal_list = []
signal_names = []
for i, expr in enumerate(expressions):
    try:
        a = engine.evaluate(expr)
        if a is not None and not a.empty:
            ranked = a.reindex(index=dates, columns=tickers).rank(axis=1, pct=True) - 0.5
            ranked = ranked.fillna(0.0).values
            signal_list.append(ranked)
            signal_names.append(expr)
    except:
        pass

K = len(signal_list)
print(f"  {K} valid signals")

# Stack into (T, N, K) array
signal_arr = np.stack(signal_list, axis=-1)  # (T, N, K)
returns_arr = matrices["returns"].reindex(columns=tickers).fillna(0).values  # (T, N)
uni_mask = universe_df.reindex(index=dates, columns=tickers).fillna(False).values  # (T, N)

# ═══════════════════════════════════════════════════════════════
# STAGE 1: Rolling Ridge Regression (Eq 3.18)
# ═══════════════════════════════════════════════════════════════
RIDGE_WINDOW = 1512  # 1 year rolling window at 4h (6 bars/day × 252 days)

print(f"\nSTAGE 1: Rolling ridge regression (window={RIDGE_WINDOW}, K={K})...")

combined_forecast = np.zeros((T_total, N))
ridge_coefs = np.zeros((T_total, K))  # Store coefficients for analysis

train_end_idx = np.searchsorted(dates, pd.Timestamp(TRAIN_END))

for t in range(RIDGE_WINDOW + 1, T_total):
    # Training data: cross-sectional stacked observations from [t-WINDOW, t-1]
    # For day d in window: X = signal(d-1), y = return(d)  [delay=1]
    X_list, y_list = [], []
    for d in range(max(1, t - RIDGE_WINDOW), t):
        mask = uni_mask[d]
        if mask.sum() < 10:
            continue
        X_day = signal_arr[d - 1, mask, :]  # (n_uni, K) signal from yesterday
        y_day = returns_arr[d, mask]         # (n_uni,) return from today
        valid = np.isfinite(y_day) & np.all(np.isfinite(X_day), axis=1) & (np.abs(y_day) < 0.5)
        if valid.sum() > 5:
            X_list.append(X_day[valid])
            y_list.append(y_day[valid])

    if len(X_list) < 50:
        continue

    X_train = np.vstack(X_list)  # (n_samples, K)
    y_train = np.concatenate(y_list)  # (n_samples,)

    # Auto-calibrate ridge lambda: proportional to trace(X'X)/K
    # This is a standard heuristic (Sec 2.4.11.6)
    alpha_ridge = np.mean(X_train ** 2) * K  # regularization strength

    # Solve ridge: w = (X'X + λI)^{-1} X'y
    ridge = Ridge(alpha=alpha_ridge, fit_intercept=False)
    ridge.fit(X_train, y_train)
    ridge_coefs[t] = ridge.coef_

    # Predict: μ̂(t) = X(t) @ w  — already in return units!
    X_today = signal_arr[t, :, :]  # (N, K)
    mu_hat = X_today @ ridge.coef_  # (N,) return forecast

    # Apply universe mask
    umask = uni_mask[t].astype(bool)
    mu_hat[~umask] = 0.0
    mu_hat = np.nan_to_num(mu_hat, nan=0.0, posinf=0.0, neginf=0.0)

    combined_forecast[t] = mu_hat

    if t % 200 == 0:
        avg_mag = np.mean(np.abs(mu_hat[umask])) * 10000 if umask.sum() > 0 else 0
        r2 = ridge.score(X_train, y_train) if len(X_train) > 0 else 0
        n_pos = np.sum(ridge.coef_ > 0)
        n_neg = np.sum(ridge.coef_ < 0)
        print(f"  Day {t:>5}/{T_total}: |forecast|={avg_mag:.1f} bps, R²={r2:.4f}, coef +:{n_pos} -:{n_neg}")

print(f"  Ridge complete in {time.time()-t0:.0f}s")

# ═══════════════════════════════════════════════════════════════
# BASELINE: Simple sim with ridge forecast
# ═══════════════════════════════════════════════════════════════
forecast_df = pd.DataFrame(combined_forecast, index=dates, columns=tickers)

print(f"\n{'=' * 80}")
print("RESULTS COMPARISON")
print(f"{'=' * 80}")

# Also build EqWt signal for comparison
combined_eqwt = np.mean(signal_arr, axis=2)  # (T, N) — equal-weight average of rank signals
eqwt_df = pd.DataFrame(combined_eqwt, index=dates, columns=tickers)

print(f"\n--- Simple Vectorized Sim (rank-normalize + clip, 5 bps fees) ---")
print(f"  {'Method':<30} {'Delay':>5} {'Period':>5} {'Sharpe':>8} {'PnL':>14} {'Ann%':>8} {'TO':>6}")
print(f"  {'-' * 85}")

for method_name, sig_df in [("EqWt (rank avg)", eqwt_df), ("Ridge forecast", forecast_df)]:
    for delay in [0, 1]:
        for period_name, start, end in [('TRAIN', dates[0], TRAIN_END), ('OOS', TRAIN_END, dates[-1])]:
            p_sig = sig_df.loc[start:end]
            p_ret = matrices['returns'].reindex(columns=tickers).loc[start:end]
            p_close = matrices['close'].reindex(columns=tickers).loc[start:end]
            p_open = matrices['open'].reindex(columns=tickers).loc[start:end]
            p_uni = universe_df.loc[start:end]

            r = sim_vec(alpha_df=p_sig, returns_df=p_ret,
                       close_df=p_close, open_df=p_open, universe_df=p_uni,
                       booksize=BOOKSIZE, max_stock_weight=MAX_WT,
                       decay=0, delay=delay, neutralization='market', fees_bps=FEES_BPS)

            nbars = len(p_ret)
            ann = r.total_pnl / BOOKSIZE / max(1, nbars/1512) * 100  # 1512 4h bars/year
            print(f"  {method_name:<30} {delay:>5} {period_name:>5} {r.sharpe:>+8.2f} ${r.total_pnl:>13,.0f} {ann:>7.1f}% {r.turnover:>5.1%}")

# ═══════════════════════════════════════════════════════════════
# STAGE 2: QP Optimizer with Ridge Forecast + PCA Risk (Eq 6.6)
# ═══════════════════════════════════════════════════════════════
print(f"\n--- QP Optimizer with Ridge Forecast (Eq 6.6) ---")
print(f"  Using PCA risk model (5 factors, 60-day EMA)")

# PCA Risk Model
N_PCA = 5
EMA_HL = 60
ema_decay = np.log(2) / EMA_HL

# Initialize
ret_cov_ema = None  # (N, N) EMA covariance
oos_start = np.searchsorted(dates, pd.Timestamp(TRAIN_END))

def get_pca_risk(ret_cov, n_factors=N_PCA):
    """Extract PCA factor risk: Q (K×N) and spec_var (N,)."""
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(ret_cov)
        # Take top factors
        idx = np.argsort(eigenvalues)[::-1][:n_factors]
        L = eigenvectors[:, idx]  # (N, K)
        D = np.diag(np.sqrt(np.maximum(eigenvalues[idx], 1e-10)))  # (K, K)
        Q = (D @ L.T)  # (K, N)
        # Specific variance: total var minus factor var
        factor_var = np.sum((Q.T @ np.eye(n_factors)) ** 2, axis=1)  # rough
        total_var = np.diag(ret_cov)
        spec_var = np.maximum(total_var - np.sum(L ** 2 * eigenvalues[idx], axis=1), 1e-8)
        return Q, spec_var
    except:
        return np.zeros((n_factors, ret_cov.shape[0])), np.ones(ret_cov.shape[0]) * 1e-4

# Sweep kappa values for QP
for kappa in [1.0, 5.0, 10.0, 50.0, 100.0]:
    w_prev = np.zeros(N)
    pnls = []
    ret_history = []
    ret_cov_ema = None

    for t in range(oos_start, T_total):
        r_today = returns_arr[t]

        # PnL from previous positions
        h_prev = w_prev * BOOKSIZE
        gross_pnl = np.dot(h_prev, r_today)

        # Update covariance EMA
        r_clean = np.nan_to_num(r_today, nan=0.0)
        outer = np.outer(r_clean, r_clean)
        alpha_ema = 1 - np.exp(-ema_decay)
        if ret_cov_ema is None:
            ret_cov_ema = outer
        else:
            ret_cov_ema = alpha_ema * outer + (1 - alpha_ema) * ret_cov_ema

        # Get forecast (delay=1: use yesterday's forecast)
        mu_hat = combined_forecast[t - 1] if t > 0 else np.zeros(N)
        umask = uni_mask[t].astype(bool) if t < len(uni_mask) else np.ones(N, dtype=bool)

        uni_idx = np.where(umask)[0]
        n_uni = len(uni_idx)

        if n_uni >= 4 and t > oos_start + 60 and np.any(mu_hat[umask] != 0):
            a_sub = mu_hat[uni_idx]
            w_prev_sub = w_prev[uni_idx]

            # PCA risk on universe subset
            cov_sub = ret_cov_ema[np.ix_(uni_idx, uni_idx)]
            Q_sub, spec_sub = get_pca_risk(cov_sub)

            w = cp.Variable(n_uni)
            T_var = w - w_prev_sub

            # Eq 6.6: F(P) = f·P - c|T| - kPCP
            # Risk: factor + idiosyncratic
            factor_risk = 0.5 * kappa * cp.sum_squares(Q_sub @ w)
            idio_risk = 0.5 * kappa * cp.sum(cp.multiply(spec_sub, cp.square(w)))
            expected_return = a_sub @ w
            tcost = FEES_BPS * 1e-4 * cp.norm(T_var, 1)

            objective = cp.Minimize(factor_risk + idio_risk - expected_return + tcost)
            constraints = [
                w >= -MAX_WT, w <= MAX_WT,
                cp.norm(w, 1) <= 1.0,
                cp.sum(w) == 0  # dollar neutral
            ]
            prob = cp.Problem(objective, constraints)
            try:
                prob.solve(solver=cp.OSQP, max_iter=20000, eps_abs=1e-5, eps_rel=1e-5, verbose=False)
                if prob.status in ['optimal', 'optimal_inaccurate'] and w.value is not None:
                    w_new = np.zeros(N)
                    w_new[uni_idx] = w.value
                else:
                    w_new = w_prev
            except:
                w_new = w_prev
        else:
            w_new = np.zeros(N) if t <= oos_start + 60 else w_prev

        h_new = w_new * BOOKSIZE
        trades = h_new - h_prev
        tcost_real = FEES_BPS * 1e-4 * np.sum(np.abs(trades))
        pnls.append(gross_pnl - tcost_real)
        w_prev = w_new

    pnls = np.array(pnls)
    total = np.sum(pnls)
    sharpe = np.mean(pnls) / np.std(pnls) * ANN_FACTOR if np.std(pnls) > 0 else 0
    ann = total / BOOKSIZE / max(1, len(pnls)/1512) * 100  # 1512 4h bars/year
    n_active = np.sum(np.abs(w_prev) > 1e-6)
    print(f"  OOS Ridge+QP+PCA (κ={kappa:<5}): Sharpe={sharpe:+.2f}  PnL=${total:>11,.0f}  Ann={ann:>6.1f}%  Pos={n_active}")

# Also check GP status
print(f"\n--- GP Status ---")
try:
    conn2 = sqlite3.connect("data/alpha_gp_crypto_d0.db")
    cur2 = conn2.cursor()
    cur2.execute("SELECT COUNT(*), MAX(e.sharpe), MAX(e.fitness) FROM alphas a JOIN evaluations e ON a.alpha_id=e.alpha_id")
    r = cur2.fetchone()
    conn2.close()
    print(f"  D0 DB: {r[0]} alphas, max Sharpe={r[1]:+.2f}, max Fitness={r[2]:.2f}")
except:
    print("  D0 DB: unavailable")

print(f"\nTotal runtime: {time.time()-t0:.0f}s")
