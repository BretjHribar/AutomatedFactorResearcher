"""
prod/aipt_trader.py -- AIPT SDF Paper Trader for KuCoin Perps

Runs the AIPT (Didisheim-Ke-Kelly-Malamud 2025) RFF-Markowitz SDF portfolio
on KuCoin TOP100 perpetual futures. Designed to run every 4 hours on a
scheduler (Task Scheduler on Windows, or cron).

Pipeline:
  1. Refresh data (fetch latest 4h bars from KuCoin API + rebuild matrices)
  2. Load FULL research matrices (AIPT needs 4380+ bars for lambda training)
  3. Build characteristics Z_t at the latest bar
  4. Compute RFF signals S_t = sin/cos(gamma * Z_t * theta)
  5. Load persisted factor-return history from disk
  6. Re-estimate lambda every REBAL_EVERY bars using rolling ridge-Markowitz
  7. Compute target portfolio weights (sum|w| = 1)
  8. Record trades, returns, and positions

State persistence (survives restarts):
  - factor_return_history.npz   — array of past factor returns
  - aipt_state.json             — lambda, prev_weights, bars_since_rebal, etc.

Usage:
  python prod/aipt_trader.py              # single run (call from scheduler)
  python prod/aipt_trader.py --backfill   # backfill from OOS start to now

Schedule (Windows Task Scheduler):
  Run every 4h at 00:05, 04:05, 08:05, 12:05, 16:05, 20:05 UTC
  (5 min after bar close to ensure data is available)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# CONFIGURATION
# ============================================================================

UNIVERSE        = "KUCOIN_TOP100"
INTERVAL        = "4h"
BARS_PER_YEAR   = 6 * 365   # 2190
TRAIN_BARS      = 4380      # ~2 years of 4h bars
MIN_TRAIN_BARS  = 1000
REBAL_EVERY     = 12        # re-estimate SDF weights every 12 bars (~2 days)
COVERAGE_CUTOFF = 0.3       # min data coverage per ticker

# AIPT model
P_FACTORS       = 100
Z_RIDGE         = 1e-3
SEED            = 42
GAMMA_GRID      = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Fees
TAKER_BPS       = 3.0       # KuCoin VIP12 taker fee
TARGET_GMV      = 100_000   # $100K notional

# Characteristics to use (must match research matrices)
CHAR_NAMES = [
    "adv20", "adv60", "beta_to_btc", "close_position_in_range",
    "dollars_traded", "high_low_range",
    "historical_volatility_10", "historical_volatility_20",
    "historical_volatility_60", "historical_volatility_120",
    "log_returns", "momentum_5d", "momentum_20d", "momentum_60d",
    "open_close_range",
    "parkinson_volatility_10", "parkinson_volatility_20",
    "parkinson_volatility_60",
    "quote_volume", "turnover",
    "volume_momentum_1", "volume_momentum_5_20", "volume_ratio_20d",
    "vwap_deviation",
]

# Paths
# Live runs use PROD matrices (1500 rows, refreshed by data_refresh).
# Factor return history is persisted to disk, so we DON'T need 4380+ rows
# in the matrices — only the latest few bars for Z_t and R_t.
# Backfill uses the full research matrices (see run_backfill()).
MATRICES_DIR_PROD = PROJECT_ROOT / "data" / "kucoin_cache" / "matrices" / INTERVAL / "prod"
MATRICES_DIR_FULL = PROJECT_ROOT / "data" / "kucoin_cache" / "matrices" / INTERVAL
UNIVERSE_PATH     = PROJECT_ROOT / "data" / "kucoin_cache" / "universes" / f"{UNIVERSE}_{INTERVAL}.parquet"
STATE_DIR       = PROJECT_ROOT / "prod" / "state" / "aipt"
LOG_DIR         = PROJECT_ROOT / "prod" / "logs" / "kucoin" / "aipt"
TRADE_LOG_DIR   = LOG_DIR / "trades"
EQUITY_DIR      = LOG_DIR / "performance"

for d in [STATE_DIR, TRADE_LOG_DIR, EQUITY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging():
    log_file = LOG_DIR / f"aipt_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ]
    )
    return logging.getLogger("aipt_trader")

log = setup_logging()

# ============================================================================
# RFF / SDF CORE (same math as aipt_barbybar.py — verified no lookahead)
# ============================================================================

def generate_rff_params(D: int, P: int, seed: int = SEED):
    """Generate FIXED random RFF parameters (no data dependency)."""
    rng = np.random.default_rng(seed)
    n_pairs = P // 2
    theta = rng.standard_normal((n_pairs, D))
    gamma = rng.choice(GAMMA_GRID, size=n_pairs)
    return theta, gamma


def build_characteristics_vector(matrices: dict, tickers: list, bar_idx: int) -> np.ndarray:
    """Build characteristics Z_t at a single bar index. Returns (N, D) array."""
    available_chars = [c for c in CHAR_NAMES if c in matrices]
    D = len(available_chars)
    N = len(tickers)

    Z = np.full((N, D), np.nan)
    for j, char_name in enumerate(available_chars):
        df = matrices[char_name]
        row_vals = df.iloc[bar_idx].reindex(tickers).values.astype(np.float64)
        Z[:, j] = row_vals

    # Rank-standardize cross-sectionally to [-0.5, 0.5]
    for j in range(D):
        col = Z[:, j]
        valid = ~np.isnan(col)
        n_valid = valid.sum()
        if n_valid < 3:
            Z[:, j] = 0.0
            continue
        r = rankdata(col[valid], method='average')
        r = r / n_valid - 0.5
        Z[valid, j] = r
        Z[~valid, j] = 0.0

    return Z, available_chars


def compute_rff_signals(Z: np.ndarray, theta: np.ndarray, gamma: np.ndarray, P: int) -> np.ndarray:
    """Compute RFF signals S = [sin(gamma * Z * theta), cos(gamma * Z * theta)]."""
    proj = Z @ theta.T                    # (N, P//2)
    proj_scaled = proj * gamma[np.newaxis, :]
    S = np.empty((Z.shape[0], P))
    S[:, 0::2] = np.sin(proj_scaled)
    S[:, 1::2] = np.cos(proj_scaled)
    return S


def estimate_ridge_markowitz(F_train: np.ndarray, z: float) -> np.ndarray:
    """Ridge-Markowitz: lambda = (zI + E[FF'])^{-1} E[F]."""
    T, P = F_train.shape
    mu = F_train.mean(axis=0)
    FF = (F_train.T @ F_train) / T
    A = z * np.eye(P) + FF
    return np.linalg.solve(A, mu)


# ============================================================================
# STATE PERSISTENCE
# ============================================================================

STATE_FILE = STATE_DIR / "aipt_state.json"
FACTOR_RETURNS_FILE = STATE_DIR / "factor_returns.npz"
FACTOR_INDICES_FILE = STATE_DIR / "factor_indices.npy"

def save_state(lambda_hat, prev_weights, bars_since_rebal, last_bar_time,
               tickers):
    """Persist state to disk for next run."""
    state = {
        "bars_since_rebal": bars_since_rebal,
        "last_bar_time": str(last_bar_time),
        "tickers": tickers,
        "P": P_FACTORS,
        "z": Z_RIDGE,
        "seed": SEED,
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

    # Save lambda and prev_weights as npz
    np.savez(STATE_DIR / "weights.npz",
             lambda_hat=lambda_hat,
             prev_weights=prev_weights if prev_weights is not None else np.array([]))

    log.info(f"  State saved: {last_bar_time}")


def load_state():
    """Load persisted state from disk. Returns None if no state exists."""
    if not STATE_FILE.exists():
        return None

    with open(STATE_FILE) as f:
        state = json.load(f)

    weights = np.load(STATE_DIR / "weights.npz")
    lambda_hat = weights["lambda_hat"]
    prev_weights = weights["prev_weights"]
    if prev_weights.size == 0:
        prev_weights = None

    return {
        **state,
        "lambda_hat": lambda_hat,
        "prev_weights": prev_weights,
    }


def save_factor_returns(fr_history: list):
    """Save factor return history to disk."""
    if not fr_history:
        return
    timestamps = np.array([ts for ts, _ in fr_history], dtype='U30')
    vectors = np.vstack([v for _, v in fr_history])
    np.save(FACTOR_INDICES_FILE, timestamps)
    np.savez_compressed(FACTOR_RETURNS_FILE, factor_returns=vectors)
    log.info(f"  Factor returns saved: {len(fr_history)} bars")


def load_factor_returns() -> list:
    """Load factor return history from disk."""
    if not FACTOR_RETURNS_FILE.exists() or not FACTOR_INDICES_FILE.exists():
        return []
    timestamps = np.load(FACTOR_INDICES_FILE, allow_pickle=True)
    data = np.load(FACTOR_RETURNS_FILE)
    vectors = data["factor_returns"]
    return [(str(ts), vectors[i]) for i, ts in enumerate(timestamps)]


# ============================================================================
# TRADE LOGGING
# ============================================================================

def log_trade(timestamp, bar_time, target_weights, prev_weights, turnover,
              port_return, port_return_net, n_long, n_short):
    """Append a single trade record to the equity CSV and save a trade JSON."""

    # Equity CSV
    equity_file = EQUITY_DIR / "equity_aipt.csv"
    if not equity_file.exists():
        with open(equity_file, "w") as f:
            f.write("timestamp,bar_time,n_long,n_short,turnover,"
                    "port_return_gross,port_return_net,gmv\n")

    with open(equity_file, "a") as f:
        f.write(f"{timestamp},{bar_time},{n_long},{n_short},{turnover:.6f},"
                f"{port_return:.8f},{port_return_net:.8f},{TARGET_GMV}\n")

    # Trade JSON (detailed positions)
    trade_path = TRADE_LOG_DIR / f"aipt_{timestamp.replace(':', '-')}.json"
    # Only record non-zero weights for compactness
    target_dict = {}
    for sym, w in target_weights.items():
        if abs(w) > 1e-6:
            target_dict[sym] = round(float(w), 6)

    trade_log = {
        "timestamp": timestamp,
        "bar_time": str(bar_time),
        "model": {"P": P_FACTORS, "z": Z_RIDGE, "seed": SEED, "rebal": REBAL_EVERY},
        "portfolio": {
            "n_long": n_long,
            "n_short": n_short,
            "target_gmv": TARGET_GMV,
            "turnover": round(turnover, 6),
            "port_return_gross": round(port_return, 8),
            "port_return_net": round(port_return_net, 8),
        },
        "weights": target_dict,
    }
    with open(trade_path, "w") as f:
        json.dump(trade_log, f, indent=2)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline():
    """Run a single AIPT trading cycle."""
    t0 = time.time()
    now = dt.datetime.utcnow()
    timestamp = now.isoformat()

    log.info("=" * 70)
    log.info(f"  AIPT TRADER v1.0 -- {now.strftime('%Y-%m-%d %H:%M')} UTC")
    log.info(f"  P={P_FACTORS}, z={Z_RIDGE:.0e}, rebal={REBAL_EVERY}")
    log.info("=" * 70)

    # ── Phase 0: Data refresh ────────────────────────────────────────
    log.info("\nPhase 0: Refreshing data from KuCoin API...")
    try:
        from prod.data_refresh import refresh_kucoin
        latest_bar = refresh_kucoin()
        log.info(f"  Data refreshed. Latest bar: {latest_bar}")
    except Exception as e:
        log.warning(f"  Data refresh failed: {e}")
        log.warning(f"  Continuing with cached data...")

    # ── Phase 1: Load PROD matrices (1500 rows, refreshed by Phase 0) ──
    #    Factor return history is persisted to disk from backfill,
    #    so we only need the latest bars for Z_t and R_t computation.
    log.info("\nPhase 1: Loading prod matrices...")
    matrices = {}
    mat_dir = MATRICES_DIR_PROD if MATRICES_DIR_PROD.exists() else MATRICES_DIR_FULL
    for fp in sorted(mat_dir.glob("*.parquet")):
        matrices[fp.stem] = pd.read_parquet(fp)

    if not matrices or "close" not in matrices:
        log.error("No matrices found! Run download_kucoin.py first.")
        return

    # Load universe and filter tickers
    universe_df = pd.read_parquet(UNIVERSE_PATH)
    coverage = universe_df.sum(axis=0) / len(universe_df)
    tickers = sorted(coverage[coverage > COVERAGE_CUTOFF].index.tolist())
    tickers = [t for t in tickers if t in matrices["close"].columns]
    N = len(tickers)

    close = matrices["close"][tickers]
    dates = close.index
    T = len(dates)
    t_last = T - 1  # index of most recent bar
    last_bar_time = str(dates[t_last])

    log.info(f"  {len(matrices)} matrices, {N} tickers, {T} bars")
    log.info(f"  Latest bar: {dates[t_last]}")

    # ── Phase 2: Build characteristics at latest bar ─────────────────
    log.info("\nPhase 2: Building characteristics...")
    avail_chars = [c for c in CHAR_NAMES if c in matrices]
    D = len(avail_chars)
    theta, gamma = generate_rff_params(D, P_FACTORS, seed=SEED)

    Z_t, _ = build_characteristics_vector(matrices, tickers, t_last)
    S_t = compute_rff_signals(Z_t, theta, gamma, P_FACTORS)
    log.info(f"  D={D} characteristics, P={P_FACTORS} RFF factors")

    # ── Phase 3: Load state + compute factor return for this bar ─────
    log.info("\nPhase 3: Loading state...")
    state = load_state()
    fr_history = load_factor_returns()

    if state and state.get("last_bar_time") == last_bar_time:
        log.info(f"  Already processed {last_bar_time}, skipping (idempotent)")
        return

    # Compute factor return for this bar (using PREVIOUS bar's signal)
    # F_{t} = S_{t-1}' R_t / sqrt(N)
    if t_last > 0:
        # Previous bar's characteristics + signal
        Z_prev, _ = build_characteristics_vector(matrices, tickers, t_last - 1)
        S_prev = compute_rff_signals(Z_prev, theta, gamma, P_FACTORS)

        # Return at bar t_last
        close_vals = close.values
        R_t = (close_vals[t_last] - close_vals[t_last - 1]) / close_vals[t_last - 1]
        R_t = np.nan_to_num(R_t, nan=0.0)

        # Valid assets
        valid = ~np.isnan(Z_prev).any(axis=1) & ~np.isnan(close_vals[t_last]) & ~np.isnan(close_vals[t_last - 1])
        N_t = valid.sum()

        if N_t >= 5:
            F_t = (1.0 / np.sqrt(N_t)) * (S_prev[valid].T @ R_t[valid])
            fr_history.append((last_bar_time, F_t))
            log.info(f"  Factor return computed for {last_bar_time} (N_t={N_t})")
        else:
            log.warning(f"  Only {N_t} valid assets, skipping factor return")

    # Trim history to TRAIN_BARS + buffer
    max_keep = TRAIN_BARS + 500
    if len(fr_history) > max_keep:
        fr_history = fr_history[-max_keep:]

    save_factor_returns(fr_history)

    # ── Phase 4: Estimate lambda (if needed) ─────────────────────────
    log.info("\nPhase 4: SDF weight estimation...")
    if state:
        lambda_hat = state["lambda_hat"]
        prev_weights = state["prev_weights"]
        bars_since_rebal = state["bars_since_rebal"] + 1
    else:
        lambda_hat = None
        prev_weights = None
        bars_since_rebal = REBAL_EVERY  # force first estimation

    if bars_since_rebal >= REBAL_EVERY or lambda_hat is None:
        train_data = [
            fr for (ts, fr) in fr_history
            if ts < last_bar_time
        ]
        # Only use the most recent TRAIN_BARS
        train_data = train_data[-TRAIN_BARS:]
        if len(train_data) >= MIN_TRAIN_BARS:
            F_train = np.vstack(train_data)
            lambda_hat = estimate_ridge_markowitz(F_train, Z_RIDGE)
            bars_since_rebal = 0
            log.info(f"  Lambda re-estimated from {len(train_data)} training bars")
            log.info(f"  |lambda| mean={np.abs(lambda_hat).mean():.4f}, "
                     f"max={np.abs(lambda_hat).max():.4f}")
        else:
            log.warning(f"  Only {len(train_data)} training bars (need {MIN_TRAIN_BARS})")
            if lambda_hat is None:
                log.error("  No lambda available, aborting")
                return

    # ── Phase 5: Compute target portfolio ────────────────────────────
    log.info("\nPhase 5: Computing target portfolio...")

    # Valid assets at current bar
    valid_now = ~np.isnan(Z_t).any(axis=1)
    N_valid = valid_now.sum()

    # Raw weights: w = (1/sqrt(N)) * S_t @ lambda
    raw_w = np.zeros(N)
    raw_w_valid = (1.0 / np.sqrt(N_valid)) * (S_t[valid_now] @ lambda_hat)
    raw_w[valid_now] = raw_w_valid

    # Normalize: sum(|w|) = 1
    abs_sum = np.abs(raw_w).sum()
    if abs_sum < 1e-12:
        log.error("  Zero weights, aborting")
        return
    w_norm = raw_w / abs_sum

    n_long = int((w_norm > 1e-6).sum())
    n_short = int((w_norm < -1e-6).sum())

    # Turnover
    if prev_weights is not None and len(prev_weights) == N:
        turnover = np.abs(w_norm - prev_weights).sum() / 2.0
    else:
        turnover = 0.0

    # Return from PREVIOUS weights (this bar's realized return)
    port_return = 0.0
    if prev_weights is not None and len(prev_weights) == N and t_last > 0:
        R_t_all = (close_vals[t_last] - close_vals[t_last - 1]) / close_vals[t_last - 1]
        R_t_all = np.nan_to_num(R_t_all, nan=0.0)
        port_return = float(prev_weights @ R_t_all)

    # Fee drag
    fee_drag = turnover * TAKER_BPS / 10000 * 2  # 2 legs
    port_return_net = port_return - fee_drag

    log.info(f"  {n_long}L / {n_short}S | TO={turnover:.4f} | "
             f"Return={port_return:.6f} | Net={port_return_net:.6f}")

    # ── Phase 6: Log everything ──────────────────────────────────────
    log.info("\nPhase 6: Logging...")
    target_weights = dict(zip(tickers, w_norm))
    log_trade(
        timestamp=timestamp,
        bar_time=dates[t_last],
        target_weights=target_weights,
        prev_weights=dict(zip(tickers, prev_weights)) if prev_weights is not None else {},
        turnover=turnover,
        port_return=port_return,
        port_return_net=port_return_net,
        n_long=n_long,
        n_short=n_short,
    )

    # Save state for next run
    save_state(
        lambda_hat=lambda_hat,
        prev_weights=w_norm,
        bars_since_rebal=bars_since_rebal,
        last_bar_time=last_bar_time,
        tickers=tickers,
    )

    elapsed = time.time() - t0
    log.info(f"\n  COMPLETE in {elapsed:.1f}s")
    log.info(f"  Next rebal in {REBAL_EVERY - bars_since_rebal} bars")


# ============================================================================
# BACKFILL — replay all bars from OOS start to build factor return history
# ============================================================================

def run_backfill():
    """
    Backfill factor return history from OOS start to latest bar.
    This builds the state needed for the first live run.
    """
    OOS_START = "2024-09-01"

    log.info("=" * 70)
    log.info("  AIPT BACKFILL -- building factor return history")
    log.info("=" * 70)

    # Load FULL research matrices (backfill needs full history)
    matrices = {}
    for fp in sorted(MATRICES_DIR_FULL.glob("*.parquet")):
        if fp.parent.name == "prod":
            continue
        matrices[fp.stem] = pd.read_parquet(fp)

    universe_df = pd.read_parquet(UNIVERSE_PATH)
    coverage = universe_df.sum(axis=0) / len(universe_df)
    tickers = sorted(coverage[coverage > COVERAGE_CUTOFF].index.tolist())
    tickers = [t for t in tickers if t in matrices["close"].columns]
    N = len(tickers)

    close = matrices["close"][tickers]
    dates = close.index
    T = len(dates)

    avail_chars = [c for c in CHAR_NAMES if c in matrices]
    D = len(avail_chars)
    theta, gamma = generate_rff_params(D, P_FACTORS, seed=SEED)

    # Find start index (need TRAIN_BARS of warm-up before OOS)
    oos_idx = None
    for i, dt_val in enumerate(dates):
        if str(dt_val) >= OOS_START:
            oos_idx = i
            break
    start_bar = max(0, oos_idx - TRAIN_BARS - 10)

    log.info(f"  N={N}, D={D}, T={T}, OOS start={oos_idx}")
    log.info(f"  Building factor returns from bar {start_bar} to {T-2}...")

    close_vals = close.values
    fr_history = []

    for t in range(start_bar, T - 1):
        Z_t, _ = build_characteristics_vector(matrices, tickers, t)
        S_t = compute_rff_signals(Z_t, theta, gamma, P_FACTORS)

        R_t1 = (close_vals[t + 1] - close_vals[t]) / close_vals[t]
        R_t1 = np.nan_to_num(R_t1, nan=0.0)

        valid = ~np.isnan(Z_t).any(axis=1) & ~np.isnan(close_vals[t]) & ~np.isnan(close_vals[t + 1])
        N_t = valid.sum()
        if N_t < 5:
            continue

        F_t1 = (1.0 / np.sqrt(N_t)) * (S_t[valid].T @ R_t1[valid])
        fr_history.append((str(dates[t + 1]), F_t1))

        if len(fr_history) % 1000 == 0:
            log.info(f"    {len(fr_history)} factor returns computed...")

    save_factor_returns(fr_history)

    # Also do a final lambda estimation and save state at the last bar
    # Use the most recent TRAIN_BARS factor returns
    train_data = [fr for (_, fr) in fr_history[-TRAIN_BARS:]]
    if len(train_data) >= MIN_TRAIN_BARS:
        F_train = np.vstack(train_data)
        lambda_hat = estimate_ridge_markowitz(F_train, Z_RIDGE)

        # Compute weights at last bar
        Z_last, _ = build_characteristics_vector(matrices, tickers, T - 1)
        S_last = compute_rff_signals(Z_last, theta, gamma, P_FACTORS)
        valid_last = ~np.isnan(Z_last).any(axis=1)
        raw_w = np.zeros(N)
        raw_w[valid_last] = (1.0 / np.sqrt(valid_last.sum())) * (S_last[valid_last] @ lambda_hat)
        abs_sum = np.abs(raw_w).sum()
        if abs_sum > 1e-12:
            w_norm = raw_w / abs_sum
        else:
            w_norm = raw_w

        save_state(
            lambda_hat=lambda_hat,
            prev_weights=w_norm,
            bars_since_rebal=0,
            last_bar_time=str(dates[T - 1]),
            tickers=tickers,
        )
        log.info(f"  Backfill complete: {len(fr_history)} factor returns, "
                 f"lambda estimated from {len(train_data)} training bars")
    else:
        log.error(f"  Not enough training data: {len(train_data)} bars")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIPT SDF Paper Trader (KuCoin)")
    parser.add_argument("--backfill", action="store_true",
                        help="Backfill factor return history from OOS start")
    args = parser.parse_args()

    try:
        if args.backfill:
            run_backfill()
        else:
            run_pipeline()
    except Exception as e:
        log.error(f"FATAL: {e}", exc_info=True)
        sys.exit(1)
