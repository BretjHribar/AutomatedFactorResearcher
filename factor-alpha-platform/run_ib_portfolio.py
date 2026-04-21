"""
run_ib_portfolio.py -- IB Closing Auction Portfolio Construction & Evaluation

Evaluates combination methods across train/val/test with ACTUAL IBKR fees.
Produces equity curve charts with Sharpe + return labels.

Combiners:
  1. Equal Weight
  2. Adaptive (rolling expected return)
  3. Risk Parity (inverse volatility)
  4. Billions (exact port from CryptoStratMangerBillion.py, Isichenko 2021)
  + QP-optimized versions of each

IBKR Pro Fee Schedule (US Equities, per share):
  Tiered:
    <= 300,000 shares/month:         $0.0035/share  (min $0.35/order, max 1% trade value)
    300,001 - 3,000,000/month:       $0.0020/share
    3,000,001 - 20,000,000/month:    $0.0015/share
    20,000,001 - 100,000,000/month:  $0.0010/share
    > 100,000,000/month:             $0.0005/share
  Fixed:
    All volume:                      $0.0050/share  (min $1.00/order, max 1% trade value)

  Effective bps = (cost_per_share / avg_stock_price) * 10000
  Computed from actual average close price of the TOP2000TOP3000 universe.

Splits: train (2016-2023), val (2023-2024.5), test (2024.5-present)
"""

import sys, os, time, sqlite3
import warnings; warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import linear_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# CONFIGURATION
# ============================================================================

UNIVERSE      = "TOP2000TOP3000"
BOOKSIZE      = 20_000_000.0
BARS_PER_DAY  = 1
MAX_WEIGHT    = 0.01
NEUTRALIZE    = "market"
COVERAGE_CUTOFF = 0.3
DB_PATH       = "data/ib_alphas.db"

SPLITS = {
    "train": ("2016-01-01", "2023-01-01"),   # 7 years
    "val":   ("2023-01-01", "2024-07-01"),   # 1.5 years
    "test":  ("2024-07-01", None),           # to present
}

FULL_START = "2016-01-01"

# IBKR Pro per-share commission rates (USD)
IBKR_TIERED_RATES = {
    "Tiered >100M/mo  $0.0005/sh": 0.0005,
    "Tiered 20M-100M  $0.0010/sh": 0.0010,
    "Tiered 3M-20M    $0.0015/sh": 0.0015,
    "Tiered 300k-3M   $0.0020/sh": 0.0020,
    "Tiered <=300k    $0.0035/sh": 0.0035,
}
IBKR_FIXED_RATE = 0.0050   # Fixed: $0.005/share
IBKR_MIN_ORDER_TIERED = 0.35
IBKR_MIN_ORDER_FIXED  = 1.00
IBKR_MAX_ORDER_PCT    = 0.01   # 1% of trade value

# Billions lookback (daily equivalent of optimLookback=5 in crypto 15m bars)
# 5 * 96 15m-bars/day = 480min = ~2 days; for daily bars we use 60 (1 quarter)
BILLIONS_LOOKBACK = 60

REPORT_BUFFER = []
def log(msg="", flush=True):
    print(msg, flush=flush)
    REPORT_BUFFER.append(msg)


# ============================================================================
# DATA LOADING (reuses eval_alpha_ib.py's clean pipeline)
# ============================================================================

_FULL_DATA = None

def load_full_data():
    global _FULL_DATA
    if _FULL_DATA is not None:
        return _FULL_DATA

    import eval_alpha_ib
    eval_alpha_ib.UNIVERSE = UNIVERSE
    eval_alpha_ib.NEUTRALIZE = NEUTRALIZE

    matrices, universe, classifications = eval_alpha_ib.load_data("full")
    valid_tickers = universe.columns.tolist()

    print(f"  Loaded {len(matrices)} fields, {len(valid_tickers)} tickers", flush=True)
    print(f"  Date range: {matrices['close'].index[0]} to {matrices['close'].index[-1]}", flush=True)

    _FULL_DATA = (matrices, universe, classifications, valid_tickers)
    return _FULL_DATA


def load_alphas():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT a.id, a.expression, COALESCE(e.ic_mean, 0), COALESCE(e.sharpe_is, 0)
        FROM alphas a LEFT JOIN evaluations e ON e.alpha_id = a.id
        WHERE a.archived = 0 AND a.asset_class = 'equities_ib'
        ORDER BY COALESCE(e.sharpe_is, 0) DESC
    """).fetchall()
    conn.close()
    return rows


def evaluate_expression(expression, matrices):
    from src.operators.fastexpression import FastExpressionEngine
    engine = FastExpressionEngine(data_fields=matrices)
    return engine.evaluate(expression)


def process_signal(alpha_df, universe_df=None, max_wt=MAX_WEIGHT):
    signal = alpha_df.copy()
    if universe_df is not None:
        uni_mask = universe_df.reindex(index=signal.index, columns=signal.columns).fillna(False).astype(bool)
        signal = signal.where(uni_mask, np.nan)

    mean_val = signal.mean(axis=1)
    signal = signal.sub(mean_val, axis=0)
    abs_sum = signal.abs().sum(axis=1).replace(0, np.nan)
    signal = signal.div(abs_sum, axis=0)
    signal = signal.clip(lower=-max_wt, upper=max_wt)
    return signal.fillna(0.0)


def simulate(alpha_df, returns_df, close_df, universe_df, fees_bps=0.0,
             classifications=None):
    from src.simulation.vectorized_sim_polars import simulate_vectorized_polars
    return simulate_vectorized_polars(
        alpha_df=alpha_df, returns_df=returns_df, close_df=close_df,
        universe_df=universe_df, booksize=BOOKSIZE,
        max_stock_weight=MAX_WEIGHT, decay=0, delay=0,
        neutralization=NEUTRALIZE, fees_bps=fees_bps,
        bars_per_day=BARS_PER_DAY,
        classifications=classifications,
    )


# ============================================================================
# ALPHA SIGNAL LOADING
# ============================================================================

def load_alpha_signals(alphas, matrices):
    signals = {}
    for aid, expr, ic, sr in alphas:
        try:
            raw = evaluate_expression(expr, matrices)
            if raw is not None and not raw.empty:
                signals[aid] = raw
                print(f"    Alpha #{aid}: loaded (IS SR={sr:+.2f}, IC={ic:+.4f})", flush=True)
        except Exception as e:
            print(f"    Alpha #{aid}: FAILED ({e})", flush=True)
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


def combiner_adaptive(alpha_signals, matrices, universe_df, returns_df, lookback=504):
    """Adaptive: weight by rolling expected factor return (positive ER only)."""
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
    rolling_er = fr_df.rolling(window=lookback, min_periods=60).mean()
    weights = rolling_er.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)

    combined = pd.DataFrame(0.0, index=dates, columns=tickers)
    for aid, norm in normed_signals.items():
        w = weights_norm[aid].values if aid in weights_norm.columns else np.zeros(len(dates))
        combined = combined.add(norm.mul(pd.Series(w, index=dates), axis=0))

    return combined


def combiner_risk_parity(alpha_signals, matrices, universe_df, returns_df, lookback=504):
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
    rolling_vol = fr_df.rolling(window=lookback, min_periods=60).std()
    rolling_er  = fr_df.rolling(window=lookback, min_periods=60).mean()
    inv_vol = (1.0 / rolling_vol.replace(0, np.nan)).fillna(0)
    inv_vol = inv_vol.where(rolling_er > 0, 0)
    wsum = inv_vol.sum(axis=1).replace(0, np.nan)
    weights_norm = inv_vol.div(wsum, axis=0).fillna(0)

    combined = pd.DataFrame(0.0, index=dates, columns=tickers)
    for aid, norm in normed_signals.items():
        w = weights_norm[aid].values if aid in weights_norm.columns else np.zeros(len(dates))
        combined = combined.add(norm.mul(pd.Series(w, index=dates), axis=0))

    return combined


def combiner_billions(alpha_signals, matrices, universe_df, returns_df,
                      optim_lookback=BILLIONS_LOOKBACK):
    """
    BILLIONS regression combiner -- exact port of CryptoStratMangerBillion.py.

    Reference: Isichenko (2021) "Quantitative Portfolio Management"
    Algorithm:
      1. Compute factor returns for each alpha (daily scalar PnL of normed signal, lagged 1 bar)
      2. alphasExpectedReturns = rolling SMA(optim_lookback), shifted 1 bar, clipped >= 0
      3. Walk-forward (one bar at a time), for each bar t:
           a. Window = factor returns [t : t + optim_lookback]   (shape: lookback x N)
           b. bilAlphasDFdemeaned = window - window.mean(axis=0)  (demean along time)
           c. sampleStd = bilAlphasDFdemeaned.std(axis=0)         (std per alpha)
           d. normalizedDemeanedReturns = bilAlphasDFdemeaned / sampleStd
           e. Y_is = A_is = normalizedDemeanedReturns.iloc[:, :optim_lookback]
                     (keeps first min(N, lookback) columns -- exact from original)
           f. subAlphaExpRet = alphasExpectedReturns[t:t+lookback] / sampleStd  (fill NaN -> 0)
           g. reg.fit(A_is, subAlphaExpRet)  -- LinearRegression, no intercept, multi-output
           h. residuals = reg.predict(A_is) - subAlphaExpRet
           i. optimizedWeights = residuals / sampleStd
           j. optimizedWeights = optimizedWeights / optimizedWeights.sum(axis=1)  (row-norm)
           k. Store last row at alpha_weights_ts[t + optim_lookback + 1]  (1 bar causal gap)
      4. combined = sum_i( alpha_weights_ts[:,i] * normed_signals[i] )
    """
    close_df = matrices["close"]
    dates    = close_df.index
    tickers  = close_df.columns.tolist()
    n_bars   = len(dates)
    aid_list = list(alpha_signals.keys())
    n_alphas = len(aid_list)

    # ── Step 0: normalize each alpha's raw signal ──────────────────────────
    normed_signals = {
        aid: process_signal(raw, universe_df=universe_df, max_wt=MAX_WEIGHT)
        for aid, raw in alpha_signals.items()
    }

    # ── Step 1: compute factor returns (daily scalar PnL per alpha) ────────
    #   fr_i[t] = sum_j( normed_signal_i[t-1, j] * return_j[t] )
    #   Matching: g_alphas_arr in original = list of C = DFreturnsRowSum series
    ret_df = returns_df.reindex(index=dates, columns=tickers)
    fr_data = {}
    for aid, norm in normed_signals.items():
        lagged = norm.shift(1)
        ab = lagged.abs().sum(axis=1).replace(0, np.nan)
        n  = lagged.div(ab, axis=0)
        fr_data[aid] = (n * ret_df).sum(axis=1)

    # alphasDF.T in original = rows:time, cols:alphas
    fr_df = pd.DataFrame(fr_data, index=dates)   # shape (T, N_alphas)

    # ── Step 2: alphasExpectedReturns = sma(fr_df, lookback).shift(1), clip >= 0 ──
    #   Exact: alphasExpectedReturns = GPfunctions.sma(A_trans, optimLookback).shift(1)
    #          alphasExpectedReturns[alphasExpectedReturns < 0] = 0
    alphas_exp_ret = (
        fr_df.rolling(window=optim_lookback, min_periods=max(1, optim_lookback // 2))
             .mean()
             .shift(1)
    )
    alphas_exp_ret = alphas_exp_ret.clip(lower=0)   # zero out negative expected returns

    # ── Step 3: initialize alpha weights (equal weight at every bar) ───────
    alpha_weights_ts = pd.DataFrame(
        1.0 / n_alphas,
        index=dates, columns=aid_list
    )

    reg = linear_model.LinearRegression(fit_intercept=False)

    # ── Step 4: walk-forward loop ──────────────────────────────────────────
    #   Exact mirroring of:
    #     optimtestStart = testStart
    #     for blockStart in range(optimtestStart, len(alphasDF.T)-optimLookback-2):
    #         testStart = testStart + 1
    #         optimEnd  = testStart + optimLookback
    #   => first test_start used = optim_test_start + 1, first optimEnd = that + lookback
    optim_test_start = 0

    for test_start in range(optim_test_start + 1, n_bars - optim_lookback - 2):
        optim_end = test_start + optim_lookback

        if optim_end + 1 >= n_bars:
            break

        try:
            # ── a: window of factor returns ──────────────────────────────
            bil_alphas_df = fr_df.iloc[test_start:optim_end].copy()    # (lookback, N_alphas)

            # ── b: demean along time axis (axis=0) ──────────────────────
            bil_demeaned = bil_alphas_df - bil_alphas_df.mean(axis=0)

            # ── c: sample std per alpha ──────────────────────────────────
            sample_std = bil_demeaned.std(axis=0)                       # (N_alphas,)
            sample_std = sample_std.replace(0, np.nan)

            # ── d: normalize ─────────────────────────────────────────────
            normalized = bil_demeaned.divide(sample_std)                # (lookback, N_alphas)

            # ── e: Y_is = normalizedDemeanedReturns.iloc[:, :optimLookback] ─
            #   (exact from original -- keeps first min(N, lookback) cols)
            Y_is = normalized.iloc[:, :optim_lookback]
            A_is = Y_is

            # ── f: expected returns for window, normalized by same std ───
            sub_exp_ret = alphas_exp_ret.iloc[test_start:optim_end].copy()
            sub_exp_ret = sub_exp_ret.divide(sample_std)
            sub_exp_ret = sub_exp_ret.fillna(0.0)

            # ── g: linear regression (multi-output, no intercept) ────────
            X_train = A_is.fillna(0.0).values
            Y_train = sub_exp_ret.values
            reg.fit(X_train, Y_train)

            # ── h: residuals = predicted - actual ────────────────────────
            residuals_vals = reg.predict(X_train) - Y_train            # (lookback, N_alphas)
            residuals = pd.DataFrame(
                residuals_vals,
                index=sub_exp_ret.index, columns=sub_exp_ret.columns
            )

            # ── i: weights = residuals / std ─────────────────────────────
            opt_weights = residuals.divide(sample_std)

            # ── j: normalize rows to sum = 1 ─────────────────────────────
            row_sums = opt_weights.sum(axis=1).replace(0, np.nan)
            opt_weights = opt_weights.div(row_sums, axis=0)

            # ── k: take last row, store at optimEnd + 1 ──────────────────
            #   Exact: alphaweightsTS.iloc[optimEnd+1] = optimizedWeights.tail(1)
            final_weights = opt_weights.iloc[-1]
            alpha_weights_ts.iloc[optim_end + 1] = final_weights.values

        except Exception:
            pass

    # ── Step 5: combine alpha signals using time-varying scalar weights ────
    combined = pd.DataFrame(0.0, index=dates, columns=tickers)
    for aid in aid_list:
        w = alpha_weights_ts[aid]
        combined = combined.add(normed_signals[aid].mul(w, axis=0))

    return combined


# ============================================================================
# QP EXECUTION OPTIMIZER
# ============================================================================

def qp_optimize_signal(composite_df, matrices, universe_df, returns_df,
                       risk_aversion=0.10, tcost_lambda=0.00002,
                       lookback_bars=504, rebal_every=20):
    """
    Convex optimization on aggregate signal.
    Maximize: alpha'w - 0.5*kappa*w'Sigma*w - lambda*||w - w_prev||_1
    Subject to: sum(w) = 0, sum(|w|) <= 2

    Fee assumptions (IB Closing Auction via MOC orders):
      - IB Pro Tiered: $0.001-0.0035/share (volume-dependent)
      - MOC orders get exact closing price: negligible market-impact slippage
      - Therefore tcost_lambda is near-zero (0.00002)
      - Lower risk_aversion (0.10) allows optimizer to take more signal exposure
    """
    try:
        import cvxpy as cp
    except ImportError:
        print("  QP: cvxpy not installed, returning raw signal", flush=True)
        return composite_df

    close_df = matrices["close"]
    dates    = close_df.index
    tickers  = close_df.columns.tolist()
    n_tickers = len(tickers)
    n_bars    = len(dates)

    ret_df     = returns_df.reindex(index=dates, columns=tickers).fillna(0.0).values
    signal_vals = composite_df.reindex(index=dates, columns=tickers).fillna(0.0).values

    opt_weights = np.zeros((n_bars, n_tickers))
    prev_w = np.zeros(n_tickers)

    rebal_count = 0
    for t in range(lookback_bars, n_bars):
        if t % rebal_every != 0:
            opt_weights[t] = prev_w
            continue

        ret_window = ret_df[max(0, t - lookback_bars):t]
        if ret_window.shape[0] < 60:
            opt_weights[t] = prev_w
            continue

        alpha_t = signal_vals[t]
        alpha_t = np.nan_to_num(alpha_t, nan=0.0)

        cov = np.cov(ret_window.T)
        if cov.shape[0] != n_tickers:
            opt_weights[t] = prev_w
            continue

        # Ledoit-Wolf style shrinkage: blend sample + diagonal, small ridge
        cov = 0.5 * cov + 0.5 * np.diag(np.diag(cov)) + 1e-8 * np.eye(n_tickers)

        try:
            w   = cp.Variable(n_tickers)
            obj = alpha_t @ w - risk_aversion * cp.quad_form(w, cp.psd_wrap(cov))

            # Near-zero tcost penalty (IB MOC: exact closing price, negligible slippage)
            if np.any(prev_w != 0) and tcost_lambda > 0:
                obj -= tcost_lambda * cp.norm1(w - prev_w)

            constraints = [
                cp.sum(w) == 0,          # Market neutral
                cp.norm1(w) <= 2.0,      # Gross leverage <= 200%
                w >= -MAX_WEIGHT,        # Position limits
                w <= MAX_WEIGHT,
            ]

            prob = cp.Problem(cp.Maximize(obj), constraints)

            # Try CLARABEL first (faster, more accurate), fall back to SCS
            solved = False
            for solver in [cp.CLARABEL, cp.SCS]:
                try:
                    if solver == cp.CLARABEL:
                        prob.solve(solver=solver, verbose=False)
                    else:
                        prob.solve(solver=solver, verbose=False, max_iters=5000)
                    if prob.status in ["optimal", "optimal_inaccurate"]:
                        solved = True
                        break
                except Exception:
                    continue

            if solved and w.value is not None:
                opt_w = w.value
                opt_w = np.nan_to_num(opt_w, nan=0.0)
                prev_w = opt_w
                opt_weights[t] = opt_w
                rebal_count += 1
            else:
                opt_weights[t] = prev_w
        except Exception:
            opt_weights[t] = prev_w

    opt_weights[:lookback_bars] = 0.0

    print(f"  QP optimizer: {rebal_count} rebalances over {n_bars} bars", flush=True)
    return pd.DataFrame(opt_weights, index=dates, columns=tickers)


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    global UNIVERSE

    parser = argparse.ArgumentParser(description="IB Closing Auction Portfolio Evaluator")
    parser.add_argument("--universe", type=str, default=None,
                        help="Universe name to use (e.g. TOP1000TOP2000). Overrides config.")
    parser.add_argument("--skip-qp", action="store_true",
                        help="Skip QP optimization passes (faster iteration)")
    parser.add_argument("--skip-unclamped", action="store_true",
                        help="Skip unclamped combiner variants")
    args = parser.parse_args()

    if args.universe:
        UNIVERSE = args.universe
        print(f"  [CLI] Universe overridden to: {UNIVERSE}", flush=True)

    t0_total = time.time()

    # Load data
    alphas = load_alphas()
    full_matrices, full_universe, classifications, valid_tickers = load_full_data()
    full_returns = full_matrices.get("returns")

    # ── Compute average stock price for IBKR fee conversion ───────────────
    print("  Computing average universe stock price for IBKR fee schedule...", flush=True)
    close_in_uni = full_matrices["close"].where(full_universe.reindex(
        index=full_matrices["close"].index,
        columns=full_matrices["close"].columns).fillna(False).astype(bool))

    avg_price  = float(close_in_uni.stack().median())
    mean_price = float(close_in_uni.stack().mean())
    print(f"  Avg (median) close price in universe: ${avg_price:.2f}  (mean: ${mean_price:.2f})", flush=True)

    # Convert IBKR per-share rates to effective bps: bps = (cost/price) * 10000
    # The realistic tier depends on monthly share volume.
    # At $20M book, TO~1.1 daily, avg_price:
    #   daily_shares = BOOKSIZE * 1.1 / avg_price
    #   monthly_shares = daily_shares * 21
    est_daily_shares   = BOOKSIZE * 1.1 / avg_price
    est_monthly_shares = est_daily_shares * 21
    print(f"  Estimated monthly share volume: {est_monthly_shares/1e6:.1f}M shares "
          f"(daily: {est_daily_shares/1e3:.0f}k shares)", flush=True)

    # Build fee schedule: label -> bps
    fee_schedule = {"0 bps (fee-free)": 0.0}
    for label, rate in IBKR_TIERED_RATES.items():
        bps = (rate / avg_price) * 10000
        fee_schedule[f"IBKR {label} ({bps:.2f} bps eff.)"] = bps
    fixed_bps = (IBKR_FIXED_RATE / avg_price) * 10000
    fee_schedule[f"IBKR Fixed $0.005/sh ({fixed_bps:.2f} bps eff.)"] = fixed_bps

    fee_labels = list(fee_schedule.keys())
    fee_values = list(fee_schedule.values())

    log(f"\n{'='*120}")
    log(f"  IB CLOSING AUCTION PORTFOLIO CONSTRUCTION")
    log(f"  Universe: {UNIVERSE} | Alphas: {len(alphas)} | Booksize: ${BOOKSIZE/1e6:.0f}M")
    log(f"  Splits: train {SPLITS['train']}, val {SPLITS['val']}, test {SPLITS['test']}")
    log(f"  Neutralization: {NEUTRALIZE}")
    log(f"  Avg universe stock price: ${avg_price:.2f} (median), ${mean_price:.2f} (mean)")
    log(f"  Est. monthly share volume: {est_monthly_shares/1e6:.1f}M -> "
        f"IBKR Pro Tiered rate: ${_get_ibkr_tier_rate(est_monthly_shares):.4f}/share")
    log(f"  Billions lookback: {BILLIONS_LOOKBACK} bars | QP: risk_aversion=0.10, tcost_lambda=0.00002")
    log(f"{'='*120}")

    if not alphas:
        log("  ERROR: No alphas found in IB DB!")
        return

    log(f"\n  Loading alpha signals...")
    alpha_signals = load_alpha_signals(alphas, full_matrices)
    log(f"  Loaded {len(alpha_signals)}/{len(alphas)} alpha signals")

    if len(alpha_signals) < 2:
        log("  ERROR: Need at least 2 valid alpha signals!")
        return

    # ── Build all composites on full data ──────────────────────────────────
    COMBINERS = {
        "Equal Weight":  lambda: combiner_equal(alpha_signals, full_matrices, full_universe, full_returns),
        "Adaptive":      lambda: combiner_adaptive(alpha_signals, full_matrices, full_universe, full_returns, lookback=504),
        "Risk Parity":   lambda: combiner_risk_parity(alpha_signals, full_matrices, full_universe, full_returns, lookback=504),
        "Billions":      lambda: combiner_billions(alpha_signals, full_matrices, full_universe, full_returns, optim_lookback=BILLIONS_LOOKBACK),
    }

    composites = {}
    for cname, builder in COMBINERS.items():
        print(f"\n  Building: {cname}...", flush=True)
        t0 = time.time()
        comp = builder()
        if comp is not None:
            composites[cname] = comp
            print(f"    Done in {time.time()-t0:.1f}s", flush=True)

    # ── Build unclamped versions (no position clamping) ────────────────────
    if not args.skip_unclamped:
        UNCLAMPED_COMBINERS = {
            "Equal Weight [Unclamped]":  lambda: combiner_equal(alpha_signals, full_matrices, full_universe, full_returns),
            "Adaptive [Unclamped]":      lambda: combiner_adaptive(alpha_signals, full_matrices, full_universe, full_returns, lookback=504),
            "Risk Parity [Unclamped]":   lambda: combiner_risk_parity(alpha_signals, full_matrices, full_universe, full_returns, lookback=504),
            "Billions [Unclamped]":      lambda: combiner_billions(alpha_signals, full_matrices, full_universe, full_returns, optim_lookback=BILLIONS_LOOKBACK),
        }
        for cname, builder in UNCLAMPED_COMBINERS.items():
            print(f"\n  Building: {cname}...", flush=True)
            t0 = time.time()
            comp = builder()
            if comp is not None:
                composites[cname] = comp
                print(f"    Done in {time.time()-t0:.1f}s", flush=True)
    else:
        print("  [--skip-unclamped] Skipping unclamped combiner variants.", flush=True)

    # ── Build QP-optimized versions ────────────────────────────────────────
    qp_composites = {}
    if not args.skip_qp:
        for cname, comp in composites.items():
            qp_name = f"{cname} + QP"
            print(f"\n  Building QP: {qp_name}...", flush=True)
            t0 = time.time()
            qp_comp = qp_optimize_signal(comp, full_matrices, full_universe, full_returns)
            if qp_comp is not None:
                qp_composites[qp_name] = qp_comp
                print(f"    Done in {time.time()-t0:.1f}s", flush=True)
    else:
        print("  [--skip-qp] Skipping QP optimization passes.", flush=True)

    all_composites = {**composites, **qp_composites}

    # ── Evaluate all composites x splits x fees ───────────────────────────
    print(f"\n  Evaluating {len(all_composites)} methods x {len(SPLITS)} splits x {len(fee_values)} fees...")
    all_results = {}

    for cname, comp in all_composites.items():
        for split_name, (start, end) in SPLITS.items():
            comp_slice = comp.loc[start:end] if end else comp.loc[start:]
            split_matrices = {}
            for name, df in full_matrices.items():
                sliced = df.loc[start:end] if end else df.loc[start:]
                if len(sliced) > 0:
                    split_matrices[name] = sliced
            split_universe = full_universe.loc[start:end] if end else full_universe.loc[start:]
            close   = split_matrices.get("close")
            returns = split_matrices.get("returns")

            if "+ QP" not in cname:
                alpha_processed = process_signal(comp_slice, universe_df=split_universe, max_wt=MAX_WEIGHT)
            else:
                alpha_processed = comp_slice.clip(lower=-MAX_WEIGHT, upper=MAX_WEIGHT)

            for fee in fee_values:
                try:
                    sim = simulate(alpha_processed, returns, close, split_universe, fees_bps=fee,
                                  classifications=classifications)
                    all_results[(cname, split_name, fee)] = sim
                except Exception as e:
                    print(f"    ERROR: {cname}/{split_name}/{fee:.3f}bps: {e}")

    # ====================================================================
    # OUTPUT
    # ====================================================================

    log(f"\n\n{'='*160}")
    log(f"  FULL RESULTS TABLE -- ALL METHODS x SPLITS x FEES")
    log(f"  {len(alphas)} alphas, {UNIVERSE}, daily, market-neutral")
    log(f"  IBKR Fee note: effective bps = (per_share_rate / avg_price) * 10000")
    log(f"  Avg price: ${avg_price:.2f} | Est. monthly vol: {est_monthly_shares/1e6:.1f}M shares")
    log(f"  Val (2023-mid2024) vs Test (mid2024-now): genuine regime difference (rate-plateau era).")
    log(f"  Lookback data fully loaded (signals computed on 2016-present, then date-sliced).")
    log(f"{'='*160}")

    for fee_label, fee in zip(fee_labels, fee_values):
        log(f"\n  ---- {fee_label} ----")
        header = (f"  {'Method':<34s} | {'Train SR':>9} {'Train TO':>9} {'Trn $/day':>10} |"
                  f" {'Val SR':>8} {'Val TO':>8} {'Val $/day':>10} |"
                  f" {'Test SR':>8} {'Test TO':>8} {'Tst $/day':>10}")
        log(header)
        log(f"  {'-'*len(header)}")

        for cname in all_composites:
            train_r = all_results.get((cname, "train", fee))
            val_r   = all_results.get((cname, "val",   fee))
            test_r  = all_results.get((cname, "test",  fee))

            def fmt(r):
                if r is None:
                    return "  N/A", "  N/A", "  N/A"
                # Dollar turnover per day: turnover (fraction of booksize) * booksize
                dollar_to = r.turnover * BOOKSIZE
                return f"{r.sharpe:+8.3f}", f"{r.turnover:8.4f}", f"${dollar_to/1e3:8.0f}k"

            tr_sr, tr_to, tr_dt = fmt(train_r)
            v_sr,  v_to,  v_dt  = fmt(val_r)
            te_sr, te_to, te_dt = fmt(test_r)

            log(f"  {cname:<34s} | {tr_sr} {tr_to} {tr_dt} | {v_sr} {v_to} {v_dt} | {te_sr} {te_to} {te_dt}")

    # ── IBKR Fee Cost Breakdown ────────────────────────────────────────────
    log(f"\n\n{'='*120}")
    log(f"  IBKR COMMISSION COST BREAKDOWN (realistic on $20M book)")
    log(f"  Based on avg universe price ${avg_price:.2f}/share (median)")
    log(f"  Est. daily shares traded: {est_daily_shares/1e3:.0f}k | Monthly: {est_monthly_shares/1e6:.1f}M")
    log(f"{'='*120}")
    log(f"  {'Fee Scenario':<45s} | {'$/share':>8} | {'Eff. bps':>9} | {'$/day (TO=1.1)':>14} | {'$/year':>10}")
    log(f"  {'-'*100}")

    for label, rate in {**IBKR_TIERED_RATES, "Fixed": IBKR_FIXED_RATE}.items():
        bps = (rate / avg_price) * 10000
        daily_cost = est_daily_shares * rate
        annual_cost = daily_cost * 252
        log(f"  {'IBKR Pro ' + label:<45s} | ${rate:.4f}  | {bps:8.2f}  | ${daily_cost:13,.0f} | ${annual_cost/1e3:9,.0f}k")

    # ── Detailed per-combiner breakdown ───────────────────────────────────
    log(f"\n\n{'='*120}")
    log(f"  DETAILED PER-COMBINER BREAKDOWN (Sharpe | Ann.Ret% | MaxDD% | Turnover | $/day-TO | Fitness)")
    log(f"{'='*120}")

    for cname in all_composites:
        log(f"\n  -- {cname} --")
        log(f"  {'Fee Scenario':<45s} | {'Split':<6} | {'Sharpe':>8} {'AnnRet%':>8} {'MaxDD%':>7} {'TO':>7} {'$/day-TO':>9} {'Fitness':>7}")
        log(f"  {'-'*110}")
        for fee_label, fee in zip(fee_labels, fee_values):
            for split_name in ["train", "val", "test"]:
                r = all_results.get((cname, split_name, fee))
                if r is None:
                    log(f"  {fee_label:<45s} | {split_name:<6} | {'N/A':>8}")
                    continue
                dollar_to = r.turnover * BOOKSIZE
                log(f"  {fee_label:<45s} | {split_name:<6} | {r.sharpe:+8.3f} "
                    f"{r.returns_ann*100:+7.2f}% {r.max_drawdown*100:6.2f}% "
                    f"{r.turnover:6.4f} ${dollar_to/1e3:8.0f}k {r.fitness:6.2f}")

    # ── Individual alpha results ───────────────────────────────────────────
    log(f"\n\n{'='*120}")
    log(f"  INDIVIDUAL ALPHA RESULTS (for reference)")
    log(f"{'='*120}")

    # Show only at 0 bps and the realistic IBKR tier for brevity
    realistic_rate = _get_ibkr_tier_rate(est_monthly_shares)
    realistic_bps  = (realistic_rate / avg_price) * 10000
    individual_fees = [(0.0, "0 bps"), (realistic_bps, f"IBKR ~{realistic_bps:.2f}bps")]

    for aid, expr, ic, sr in alphas:
        if aid not in alpha_signals:
            continue
        raw = alpha_signals[aid]
        log(f"\n  Alpha #{aid}: {expr[:70]}")
        log(f"  {'Fee':<20s} | {'Split':<6} | {'Sharpe':>8} {'Ret%':>9} {'$/day-TO':>10}")
        log(f"  {'-'*60}")
        for fee, fee_lbl in individual_fees:
            for split_name, (start, end) in SPLITS.items():
                try:
                    split_matrices = {}
                    for name, df in full_matrices.items():
                        sliced = df.loc[start:end] if end else df.loc[start:]
                        if len(sliced) > 0:
                            split_matrices[name] = sliced
                    split_universe = full_universe.loc[start:end] if end else full_universe.loc[start:]
                    raw_slice = raw.loc[start:end] if end else raw.loc[start:]
                    alpha_p   = process_signal(raw_slice, universe_df=split_universe, max_wt=MAX_WEIGHT)
                    sim = simulate(alpha_p, split_matrices.get("returns"),
                                   split_matrices.get("close"), split_universe, fees_bps=fee,
                                   classifications=classifications)
                    ret_pct  = sim.total_pnl / BOOKSIZE * 100
                    dollar_to = sim.turnover * BOOKSIZE
                    log(f"  {fee_lbl:<20s} | {split_name:<6} | {sim.sharpe:+8.3f} {ret_pct:+8.2f}% ${dollar_to/1e3:8.0f}k")
                except Exception as e:
                    log(f"  {fee_lbl:<20s} | {split_name:<6} | ERROR: {str(e)[:40]}")

    elapsed = time.time() - t0_total
    log(f"\n\n  Total elapsed: {elapsed:.1f}s")
    log(f"{'='*120}")

    # ── Equity curve charts ───────────────────────────────────────────────
    print("\n  Generating equity curve charts...", flush=True)
    plot_equity_curves(all_composites, all_results, full_matrices,
                       fee_values, fee_labels, avg_price, est_monthly_shares)

    with open("ib_portfolio_results.txt", "w") as f:
        f.write("\n".join(REPORT_BUFFER))
    print(f"\n  Report saved to ib_portfolio_results.txt")


# ============================================================================
# EQUITY CURVE PLOTTING
# ============================================================================

def plot_equity_curves(all_composites, all_results, full_matrices, fee_values,
                       fee_labels, avg_price, est_monthly_shares):
    """
    Plot cumulative equity curves for all combiners.
    Two panels: fee-free and realistic IBKR tier fee.
    Vertical dashed lines mark val/test split boundaries.
    Legend shows combiner name + test Sharpe + test AnnRet.
    """
    realistic_rate = _get_ibkr_tier_rate(est_monthly_shares)
    realistic_bps  = (realistic_rate / avg_price) * 10000

    # Pick two fee scenarios to plot
    plot_scenarios = [
        (0.0,          "Fee-Free (0 bps)"),
        (realistic_bps, f"IBKR Pro Tiered ${realistic_rate:.4f}/sh  ({realistic_bps:.2f} bps eff. @ ${avg_price:.2f})"),
    ]

    n_panels = len(plot_scenarios)
    fig, axes = plt.subplots(n_panels, 1, figsize=(20, 7 * n_panels))
    if n_panels == 1:
        axes = [axes]

    # Colour palette: separate QP vs non-QP
    base_names = [c for c in all_composites if "+ QP" not in c]
    qp_names   = [c for c in all_composites if "+ QP" in c]
    base_cmap  = plt.cm.tab10(np.linspace(0,  0.6, len(base_names)))
    qp_cmap    = plt.cm.tab10(np.linspace(0.6, 1.0, len(qp_names)))
    color_map  = {n: c for n, c in zip(base_names, base_cmap)}
    color_map.update({n: c for n, c in zip(qp_names, qp_cmap)})

    val_date  = pd.Timestamp("2023-01-01")
    test_date = pd.Timestamp("2024-07-01")

    # Pre-build date arrays for each split (avoids repeated slicing)
    split_dates = {}
    for split_name, (start, end) in SPLITS.items():
        idx = full_matrices["close"].loc[start:end].index if end else full_matrices["close"].loc[start:].index
        split_dates[split_name] = idx

    for ax, (fee, fee_lbl) in zip(axes, plot_scenarios):
        # Find the closest stored fee value
        stored_fees = fee_values
        nearest_fee = min(stored_fees, key=lambda f: abs(f - fee))

        legend_handles = []

        for cname in all_composites:
            color = color_map.get(cname, 'gray')

            # Stitch daily PnL across train -> val -> test
            pnl_all   = []
            dates_all = []
            for split_name in ["train", "val", "test"]:
                r = all_results.get((cname, split_name, nearest_fee))
                if r is None or not hasattr(r, 'daily_pnl') or r.daily_pnl is None:
                    continue
                pnl_arr   = np.asarray(r.daily_pnl)
                sdates    = split_dates[split_name]
                n         = min(len(pnl_arr), len(sdates))
                # Skip the first bar of val and test (already in previous split)
                skip = 1 if pnl_all else 0
                pnl_all.append(pnl_arr[skip:n])
                dates_all.append(sdates[skip:n])

            if not pnl_all:
                continue

            pnl_cat   = np.concatenate(pnl_all)
            dates_cat = dates_all[0]
            for d in dates_all[1:]:
                dates_cat = dates_cat.append(d)

            cum_ret_pct = np.cumsum(pnl_cat) / BOOKSIZE * 100   # % of booksize

            # Labels from test window
            r_test = all_results.get((cname, "test", nearest_fee))
            if r_test:
                lbl = (f"{cname}  [Test SR={r_test.sharpe:+.2f}  "
                       f"AnnRet={r_test.returns_ann*100:+.1f}%  "
                       f"TO={r_test.turnover:.2f}]")
            else:
                lbl = cname

            ls = '--' if '+ QP' in cname else '-'
            lw = 1.4  if '+ QP' in cname else 2.0
            ax.plot(dates_cat[:len(cum_ret_pct)], cum_ret_pct,
                    label=lbl, color=color, linestyle=ls, linewidth=lw)

        # Split markers
        ymin, ymax = ax.get_ylim()
        ax.axvline(val_date,  color='dimgray', linestyle=':', linewidth=1.5, alpha=0.85)
        ax.axvline(test_date, color='black',   linestyle=':', linewidth=2.0, alpha=0.95)
        ax.text(val_date,  ymax * 0.97, '  Val start\n  2023-01', fontsize=8,  color='dimgray', va='top')
        ax.text(test_date, ymax * 0.97, '  Test start\n  2024-07', fontsize=8, color='black',   va='top')

        # Shade regions
        xmin_date = full_matrices["close"].index[0]
        xmax_date = full_matrices["close"].index[-1]
        ax.axvspan(xmin_date, val_date,  alpha=0.04, color='steelblue',  label='_train')
        ax.axvspan(val_date,  test_date, alpha=0.06, color='orange',     label='_val')
        ax.axvspan(test_date, xmax_date, alpha=0.06, color='green',      label='_test')

        ax.set_title(f"IB Closing Auction — All Combiners Equity Curves\n{fee_lbl}",
                     fontsize=13, fontweight='bold', pad=10)
        ax.set_ylabel("Cumulative Return (% of $20M Booksize)", fontsize=11)
        ax.legend(fontsize=8.5, loc='upper left', framealpha=0.85)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

    plt.tight_layout(pad=3.0)
    outfile = "ib_portfolio_equity_curves.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"\n  Equity curves saved -> {outfile}", flush=True)
    plt.close(fig)


def _get_ibkr_tier_rate(monthly_shares: float) -> float:
    """Return IBKR Pro Tiered per-share rate based on monthly share volume."""
    if monthly_shares <= 300_000:
        return 0.0035
    elif monthly_shares <= 3_000_000:
        return 0.0020
    elif monthly_shares <= 20_000_000:
        return 0.0015
    elif monthly_shares <= 100_000_000:
        return 0.0010
    else:
        return 0.0005


if __name__ == "__main__":
    main()
