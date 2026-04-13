"""
eval_portfolio_5m.py — Agent 2: Portfolio Construction for 5m Crypto Alphas

===============================================================================
OVERVIEW
===============================================================================

This module takes alpha signals discovered by Agent 1 (stored in data/alphas_5m.db)
and optimizes HOW to trade them. It handles signal combination, normalization,
threshold-based entry/exit, position sizing, and risk management.

Current best performance (walk-forward OOS, 7 bps fees):
  Sharpe: +4.91  |  Return: +36.87%  |  Period: 91 days (Dec 2025 - Feb 2026)

===============================================================================
ARCHITECTURE
===============================================================================

1. SIGNAL COMBINATION (compute_composite_ts_autonorm):
   - Loads 30 alpha expressions from SQLite DB
   - Evaluates each alpha on 5m matrix data (101 tickers x ~33k bars)
   - Per-alpha normalization via hybrid approach:
     * Alphas with zscore_cs() -> standard cross-sectional z-score
     * Raw alphas -> divide by EWMA std only (preserves directional alpha)
   - IC-weighted combination using max(IC, 0.0001) weights
   - Optional enhancements: signal smoothing, BTC beta-hedging

2. THRESHOLD STRATEGY (run_threshold_strategy):
   - Entry when composite z-score > entry_threshold (long) or < -entry (short)
   - Exit when |z-score| < exit_threshold
   - Emergency exit when signal fully reverses (overrides min hold)
   - Conviction sizing: weight proportional to |z-score|
   - Vol targeting and drawdown brake for risk management

3. WALK-FORWARD VALIDATION (walk_forward_validate):
   - Rolling 6-day train / 3-day test windows across trainval
   - 28 folds giving 91 days of pure OOS coverage
   - Fixed params preferred over per-fold grid search (more stable)

===============================================================================
DATA
===============================================================================

Splits (116-day window, Dec 2025 - Mar 2026):
  Train:  2025-12-01 to 2026-02-01  (62 days, 17,856 bars)
  Val:    2026-02-01 to 2026-03-01  (28 days, 8,064 bars)
  Test:   2026-03-01 to 2026-03-27  (26 days, 7,488 bars)

Universe: 101 Binance USDT-perpetual futures
Interval: 5-minute bars (288 bars/day)
Fees:     7 bps per side (configurable via --fees)
Alpha DB: data/alphas_5m.db (30 alphas from 298 agent trials)

===============================================================================
BEST CONFIGURATION
===============================================================================

python eval_portfolio_5m.py --walkforward \\
    --combine ts_autonorm --entry 1.2 --exit 0.3 --hold 36 \\
    --ts-gamma 0.01 --top-n 30 --max-pos 50 --fees 7 \\
    --signal-smooth 12 --beta-hedge

Key parameters:
  --combine ts_autonorm     Time-series auto-normalization (hybrid std-only)
  --signal-smooth 12        1-hour EMA on composite (reduces whipsaw, +0.24 SR)
  --beta-hedge              Remove BTC market beta from signal (+0.07 SR)
  --max-pos 50              50 concurrent positions (diversification, +0.11 SR)
  --entry 1.2 --exit 0.3    Entry at 1.2 sigma, exit at 0.3 sigma
  --hold 36                 3-hour minimum hold period
  --fees 7                  7 bps per side

===============================================================================
USAGE
===============================================================================

Walk-forward (primary evaluation):
    python eval_portfolio_5m.py --walkforward --combine ts_autonorm \\
        --entry 1.2 --exit 0.3 --hold 36 --top-n 30 --max-pos 50 \\
        --fees 7 --signal-smooth 12 --beta-hedge

Val quick-check:
    python eval_portfolio_5m.py --combine ts_autonorm --entry 1.2 --exit 0.3 \\
        --hold 36 --top-n 30 --max-pos 50 --fees 7 --signal-smooth 12 --beta-hedge

Compare methods:
    python eval_portfolio_5m.py --compare --entry 1.2 --exit 0.3 --hold 36

Grid search (AVOID - overfits):
    python eval_portfolio_5m.py --optimize --combine ts_autonorm
"""

import sys, os, time, argparse, sqlite3, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# CONFIGURATION
# ============================================================================

# ┌─────────────────────────────────────────────────────────────────────┐
# │  HARD RULE: DATA SPLIT DISCIPLINE                                  │
# │                                                                     │
# │  Train  (Sep 1 2025 – Feb 1 2026) → Alpha signal DISCOVERY        │
# │  Val    (Feb 1 – Mar 1)  → Portfolio optim / signal combination    │
# │  Test   (Mar 1 – Mar 27) → FINAL TEST ONLY — NEVER LOOK AT        │
# │                                                                     │
# │  NEVER discover alphas on Val or Test data. NEVER optimize          │
# │  portfolio params on Train data. NEVER evaluate Test until          │
# │  you are done with ALL research.                                    │
# └─────────────────────────────────────────────────────────────────────┘

# Data splits — portfolio construction window
# Alpha signals were discovered on 2025-02-01 to 2026-02-01 (eval_alpha_5m.py)
# Portfolio walk-forward uses Dec 2025 – Mar 2026 (in-sample for alphas, but OOS for combiner)
TRAIN_START = "2025-12-01"
TRAIN_END   = "2026-02-01"
VAL_START   = "2026-02-01"
VAL_END     = "2026-03-01"
TEST_START  = "2026-03-01"
TEST_END    = "2026-03-27"

# Matching eval_alpha_5m.py
UNIVERSE       = "BINANCE_TOP100"
INTERVAL       = "5m"
BOOKSIZE       = 2_000_000.0
MAX_WEIGHT     = 0.05
BARS_PER_DAY   = 288
COVERAGE_CUTOFF = 0.3

# Trading parameters
FEES_BPS       = 10.0    # 10 bps per side (configurable via --fees)
ENTRY_THRESHOLD = 0.60   # Enter when composite z-score > this (top ~27%)
EXIT_THRESHOLD  = 0.30   # Exit when composite drops below this
MIN_HOLD_BARS   = 36     # Minimum hold = 3 hours (avoid churn)
MAX_POSITIONS   = 20     # Max concurrent positions (0 = unlimited)
SIZING_MODE     = "conviction"  # "equal" or "conviction" (signal-proportional)
TOP_LIQUID      = 0      # Restrict TRADING to top N liquid tickers (0 = all).
                         # Signals are ALWAYS computed on the full signal universe.
                         # Only positions are filtered.
EXCHANGE        = "binance"  # "binance" or "kucoin"
PORTFOLIO_UNIVERSE = "TOP100"   # "TOP100", "TOP50", "TOP20" — set via --universe
_TRADING_TICKERS = None  # Set at runtime if TOP_LIQUID > 0; None = trade all

# Alpha DB
DB_PATH = "data/alphas_5m.db"

# ============================================================================
# DATA LOADING
# ============================================================================

_DATA_CACHE = {}

# ── Data Cleaning Pipeline ──

DQ_STALE_TICKER_THRESHOLD = 0.20   # Exclude ticker if >20% of bars are stale
DQ_NAN_TICKER_THRESHOLD = 0.30     # Exclude ticker if >30% of bars are NaN
DQ_MIN_QUOTE_VOL = 500.0           # Exclude ticker if median quote_vol < $500/bar
DQ_STALE_RUN_BARS = 6              # NaN out stale runs >= 6 bars (30 min) for Binance
DQ_STALE_RUN_BARS_KUCOIN = 24     # KuCoin has lower liquidity; allow up to 2hr stale [was 6]
DQ_ZERO_VOL_BARS = 6               # NaN out zero-volume runs >= 6 bars (30 min)
DQ_RETURN_HARD_CAP = 0.15          # Hard cap single-bar return at ±15%
DQ_RETURN_ADAPTIVE_MULT = 5.0      # Causal adaptive cap: 5x rolling median abs return
DQ_RETURN_ADAPTIVE_FLOOR = 0.02    # Adaptive cap floor: at least ±2%
DQ_QV_SPIKE_MULT = 50.0            # Flag quote_volume spikes > 50x rolling median
DQ_MIN_SPREAD_RATIO = 1e-6         # (high-low)/close must be > this to be valid
DQ_TICK_SIZE_RATIO = 0.40           # Exclude if > 40% of bars have zero price change
                                    # (tick-size discretization — e.g. CRV, where the
                                    #  min price increment is large relative to the price,
                                    #  producing quantized returns that poison alpha signals)


def _exclude_tickers(close_df, quote_volume_df=None):
    """
    Stage 1: Identify tickers to permanently exclude (vectorized).

    Catches:
    - Dead/delisted pairs with forward-filled stale prices (>20% stale bars)
    - Tickers with too much missing data (>30% NaN)
    - Tickers with too few data points (<100 bars)
    - Illiquid tickers below minimum quote volume threshold
    - Zero-price tickers (data corruption)
    - Tickers with zero variance over entire history (stuck prices)
    """
    exclude = set()
    reasons = {}

    # ── Vectorized checks (all tickers at once) ──

    # Too few data points (sum of non-NaN bars)
    valid_counts = close_df.notna().sum(axis=0)
    too_few = valid_counts[valid_counts < 100].index
    for col in too_few:
        exclude.add(col)
        reasons[col] = "too_few_bars"

    # NaN fraction
    nan_frac = close_df.isna().mean(axis=0)
    too_nan = nan_frac[(nan_frac > DQ_NAN_TICKER_THRESHOLD) &
                       (~nan_frac.index.isin(exclude))].index
    for col in too_nan:
        exclude.add(col)
        reasons[col] = f"nan_{nan_frac[col]*100:.0f}%"

    # Staleness: fraction of bars where price didn't change
    diffs = close_df.diff(axis=0)
    stale_frac = ((diffs == 0) & close_df.notna()).mean(axis=0)
    too_stale = stale_frac[
        (stale_frac > DQ_STALE_TICKER_THRESHOLD) &
        (~stale_frac.index.isin(exclude))
    ].index
    for col in too_stale:
        exclude.add(col)
        reasons[col] = f"stale_{stale_frac[col]*100:.0f}%"

    # Zero-price corruption (any bar with close <= 0)
    zero_price = (close_df <= 0).any(axis=0) & (~close_df.isna().all(axis=0))
    for col in zero_price[zero_price & ~zero_price.index.isin(exclude)].index:
        exclude.add(col)
        reasons[col] = "zero_price"

    # Zero variance: price literally never moves (entire series same value)
    price_std = close_df.std(axis=0)
    zero_var = price_std[price_std < 1e-10].index
    for col in zero_var:
        if col not in exclude:
            exclude.add(col)
            reasons[col] = "zero_variance"

    # Minimum liquidity: median quote_volume < DQ_MIN_QUOTE_VOL per bar
    if quote_volume_df is not None:
        common = [c for c in close_df.columns
                  if c in quote_volume_df.columns and c not in exclude]
        if common:
            med_vol = quote_volume_df[common].median(axis=0)
            too_illiquid = med_vol[med_vol < DQ_MIN_QUOTE_VOL].index
            for col in too_illiquid:
                exclude.add(col)
                reasons[col] = f"illiquid_medqv${med_vol[col]:.0f}"

    # Tick-size discretization: price can't move in small increments
    # If > DQ_TICK_SIZE_RATIO of bars have zero price change, the
    # tick size is too coarse relative to the price for meaningful
    # 5m alpha signals.  This is DIFFERENT from genuine staleness
    # (dead tickers) — discretized tickers are actively traded but
    # their returns are heavily quantized.
    remaining = [c for c in close_df.columns if c not in exclude]
    if remaining:
        diffs_rem = close_df[remaining].diff(axis=0)
        zero_frac = ((diffs_rem == 0) & close_df[remaining].notna()).mean(axis=0)
        discretized = zero_frac[zero_frac > DQ_TICK_SIZE_RATIO].index
        for col in discretized:
            if col not in exclude:
                exclude.add(col)
                reasons[col] = f"tick_discrete_{zero_frac[col]*100:.0f}%"

    return exclude, reasons


def _vectorized_run_lengths(bool_df):
    """
    Compute run lengths for a boolean DataFrame, fully vectorized.
    Returns a same-shape DataFrame where each True cell contains the
    length of the consecutive True run it belongs to.
    """
    # Assign a group ID to each run (increments at every False->True transition)
    not_true = ~bool_df
    run_id = not_true.cumsum(axis=0)         # (n_bars, n_tickers)
    # Count how many True cells share each run_id, per ticker
    run_lengths = bool_df.groupby(run_id, axis=0).transform('sum')
    return run_lengths


def _get_stale_run_bars():
    """Return the exchange-appropriate stale run threshold.
    
    KuCoin has lower liquidity for many contracts — periods of 30-60 min
    with no price change are common and legitimate (thinly-traded pairs).
    Using the Binance 6-bar (30min) threshold would wipe 98% of KuCoin bars.
    """
    if EXCHANGE == "kucoin":
        return DQ_STALE_RUN_BARS_KUCOIN
    return DQ_STALE_RUN_BARS


def _clean_bars(matrices):
    """
    Stage 2: NaN out individual bars where data is unreliable (vectorized).

    Catches:
    - Stale price runs (>=12 consecutive identical prices = 1 hour)
    - Zero-volume periods (>=6 consecutive bars = 30 min)
    - HLOC sanity violations (high < low, negative prices)
    - Post-gap price jumps (first bar after NaN >50% move, vectorized)
    """
    close_df = matrices.get("close")
    volume_df = matrices.get("volume")
    high_df   = matrices.get("high")
    low_df    = matrices.get("low")
    if close_df is None:
        return 0

    n_cleaned = 0
    bad_mask = pd.DataFrame(False, index=close_df.index, columns=close_df.columns)

    # ── A. Stale price runs (vectorized across all tickers) ──
    diffs = close_df.diff(axis=0)
    is_stale = (diffs == 0) & close_df.notna()             # (bars, tickers)
    if is_stale.values.any():
        not_stale = ~is_stale
        run_id = not_stale.cumsum(axis=0)
        # Use groupby on the stacked form, then unstack — works per-column
        run_lengths = is_stale.copy()
        for col in is_stale.columns:
            s = is_stale[col]
            if s.any():
                ri = not_stale[col].cumsum()
                run_lengths[col] = s.groupby(ri).transform('sum')
            else:
                run_lengths[col] = 0
        stale_threshold = _get_stale_run_bars()
        stale_mask = is_stale & (run_lengths >= stale_threshold)
        bad_mask |= stale_mask
        n_cleaned += int(stale_mask.values.sum())

    # ── B. Zero-volume runs (vectorized) ──
    # Only flag bars where volume is actually ZERO (not NaN).
    # NaN volume = ticker not yet listed / data gap — handled by universe mask.
    # We only want to catch dead periods where the ticker WAS trading but hit 0 volume.
    if volume_df is not None:
        common_cols = close_df.columns.intersection(volume_df.columns)
        vol_sub = volume_df[common_cols]
        close_sub = close_df[common_cols]
        # is_zero: volume == 0 AND close is valid (ticker existed but had no volume)
        is_zero = (vol_sub == 0) & close_sub.notna()
        if is_zero.values.any():
            zero_run_lengths = is_zero.copy()
            for col in is_zero.columns:
                s = is_zero[col]
                if s.any():
                    ri = (~s).cumsum()
                    zero_run_lengths[col] = s.groupby(ri).transform('sum')
                else:
                    zero_run_lengths[col] = 0
            zero_mask = is_zero & (zero_run_lengths >= DQ_ZERO_VOL_BARS)
            # Align to close_df columns
            zero_mask = zero_mask.reindex(columns=close_df.columns, fill_value=False)
            bad_mask |= zero_mask
            n_cleaned += int(zero_mask.values.sum())


    # ── C. HLOC sanity checks (vectorized) ──
    if high_df is not None and low_df is not None:
        common_cols = close_df.columns
        h = high_df.reindex(columns=common_cols)
        lo = low_df.reindex(columns=common_cols)
        # high < low — corrupted OHLC bar
        hl_bad = (h < lo) & h.notna() & lo.notna()
        # Negative or zero prices
        neg_price = (close_df <= 0) & close_df.notna()
        # Close outside [low, high] — impossible bar
        close_outside = (
            (close_df < lo) | (close_df > h)
        ) & close_df.notna() & h.notna() & lo.notna()
        # Degenerate bar with zero high-low range (not just stale, but H=L=O=C)
        flat_bar = (h == lo) & (h == close_df) & h.notna()
        sanity_mask = hl_bad | neg_price | close_outside | flat_bar
        bad_mask |= sanity_mask
        n_sanity = int(sanity_mask.values.sum())
        if n_sanity > 0:
            n_cleaned += n_sanity

    # ── D. Post-gap price jumps (vectorized, causal) ──
    # Identify the first valid bar after each NaN run
    was_nan = close_df.isna()
    resumed = (~was_nan) & was_nan.shift(1, fill_value=False)  # True at gap resumption
    if resumed.values.any():
        # Get last valid close BEFORE each bar (shift-1 ffill)
        prev_close = close_df.shift(1).ffill()
        # Compute jump magnitude only at resumption bars
        jump = ((close_df - prev_close) / prev_close.abs()).abs()
        jump_mask = resumed & (jump > 0.50) & prev_close.notna()
        bad_mask |= jump_mask
        n_cleaned += int(jump_mask.values.sum())

    # ── E. Suspiciously large quote_volume spikes (data corruption) ──
    # A 50x spike in quote_volume is almost certainly a data error, not real trading
    # Only NaN the return for that bar (don't NaN close — price may be valid)
    # We track these separately for return cleaning, not close cleaning.

    # Apply all bad bars to ALL OHLC fields (not just close) for full consistency
    for field_name, mat in [("close", close_df), ("high", high_df), ("low", low_df)]:
        if mat is not None:
            try:
                mat[bad_mask.reindex(columns=mat.columns, fill_value=False)] = np.nan
            except Exception:
                pass
    # close_df is already mutated in-place since it's the same object
    return n_cleaned


def _winsorize_returns(returns_df, hard_cap=DQ_RETURN_HARD_CAP,
                       adaptive_mult=DQ_RETURN_ADAPTIVE_MULT,
                       adaptive_floor=DQ_RETURN_ADAPTIVE_FLOOR,
                       quote_volume_df=None):
    """
    Stage 3: Cap extreme returns for PnL calculation (fully causal).

    Three layers:
    1. Hard cap: |return| > 15% in a single 5-min bar -> clip (data corruption)
    2. Causal adaptive cap: |return| > 5x rolling 1-day median abs return
       Uses shift(1) so the cap for bar t is computed from data through t-1 only.
    3. Quote-volume spike NaN: if quote_volume > 50x rolling median AND
       return > 2%, likely a bad print — NaN that return (don't trade on it).

    Returns a new DataFrame (does not mutate input).
    """
    clean = returns_df.copy()

    # Hard cap
    n_hard = int(((clean.abs() > hard_cap) & clean.notna()).sum().sum())
    clean = clean.clip(-hard_cap, hard_cap)

    # Causal adaptive cap: use shift(1) so no lookahead into current bar
    # Per-ticker rolling MAR: more robust than global cap
    mar = clean.abs().rolling(288, min_periods=50).median().shift(1)
    adaptive_cap = (mar * adaptive_mult).clip(lower=adaptive_floor)

    too_high = clean > adaptive_cap
    too_low  = clean < -adaptive_cap
    n_adaptive = int((too_high | too_low).sum().sum())
    clean = clean.clip(lower=-adaptive_cap, upper=adaptive_cap)

    # Quote-volume spike filter: NaN returns during volume explosions
    # These are typically exchange-side errors or wash trading artefacts
    n_qv_spike = 0
    if quote_volume_df is not None:
        common_cols = clean.columns.intersection(quote_volume_df.columns)
        if len(common_cols) > 0:
            qv = quote_volume_df[common_cols].reindex(index=clean.index)
            # Causal rolling median qv (shift 1)
            qv_med = qv.rolling(288, min_periods=50).median().shift(1)
            qv_spike = (qv > qv_med * DQ_QV_SPIKE_MULT) & (clean[common_cols].abs() > 0.02)
            n_qv_spike = int(qv_spike.sum().sum())
            clean.loc[:, common_cols] = clean[common_cols].where(~qv_spike, np.nan)

    return clean, n_hard, n_adaptive, n_qv_spike


def _update_universe(universe_df, close_df):
    """
    Stage 4: Disable universe membership for bars where close was NaN'd.
    """
    nan_mask = close_df.isna()
    common = universe_df.columns.intersection(nan_mask.columns)
    common_idx = universe_df.index.intersection(nan_mask.index)
    if len(common) > 0 and len(common_idx) > 0:
        universe_df.loc[common_idx, common] = (
            universe_df.loc[common_idx, common] & ~nan_mask.loc[common_idx, common]
        )
    return universe_df


def _run_data_cleaning(matrices, valid_tickers, universe_df):
    """
    Orchestrate the full data cleaning pipeline.
    Called once during initial data load, before any splits.

    Stages:
      1. Exclude tickers with structural problems (stale, illiquid, zero-var, NaN)
      2. NaN out individual bad bars (stale runs, zero-vol, HLOC violations, gap jumps)
      3. Winsorize returns: hard cap + causal adaptive cap + QV spike filter
      4. Update universe mask to disable bars where close was NaN'd
    """
    close = matrices.get("close")
    if close is None:
        return matrices, valid_tickers, universe_df

    n_tickers_raw = len(valid_tickers)
    print(f"\n  === DATA QUALITY PIPELINE ===", flush=True)

    # Stage 1: Exclude bad tickers (vectorized, includes liquidity filter)
    qv = matrices.get("quote_volume")
    excluded, reasons = _exclude_tickers(close, quote_volume_df=qv)
    if excluded:
        valid_tickers = [t for t in valid_tickers if t not in excluded]
        matrices = {k: v[[c for c in v.columns if c not in excluded]]
                    for k, v in matrices.items()}
        keep_cols = [c for c in universe_df.columns if c not in excluded]
        universe_df = universe_df[keep_cols]
        # Group reasons for compact display
        by_type = {}
        for t, r in reasons.items():
            kind = r.split("_")[0]
            by_type.setdefault(kind, []).append(f"{t}({r})")
        reason_parts = []
        for kind, items in sorted(by_type.items()):
            if len(items) <= 3:
                reason_parts.append(", ".join(items))
            else:
                reason_parts.append(", ".join(items[:3]) + f" +{len(items)-3} more")
        print(f"  Stage 1 -- Excluded {len(excluded)}/{n_tickers_raw} tickers: "
              f"{'; '.join(reason_parts)}", flush=True)
    else:
        print(f"  Stage 1 -- No tickers excluded", flush=True)

    # Stage 2: Clean bars (stale runs, zero-vol, HLOC violations, gap jumps)
    close = matrices.get("close")  # refresh after exclusion
    n_before_nan = int(close.isna().sum().sum())
    n_cleaned = _clean_bars(matrices)
    total_cells = close.shape[0] * close.shape[1]
    stale_thresh = _get_stale_run_bars()
    print(f"  Stage 2 -- NaN'd {n_cleaned:,} bars ({n_cleaned/total_cells*100:.3f}%)"
          f"  [stale<{stale_thresh}bars/{stale_thresh*5}min, zero-vol, HLOC]", flush=True)

    # Stage 3: Winsorize returns (causal adaptive + QV spike filter)
    raw_returns = close.pct_change()
    clean_returns, n_hard, n_adaptive, n_qv = _winsorize_returns(
        raw_returns, quote_volume_df=qv
    )
    matrices["returns_clean"] = clean_returns
    max_raw = raw_returns.abs().max().max()
    max_clean = clean_returns.abs().max().max()
    print(f"  Stage 3 -- Returns capped: {n_hard} hard ({DQ_RETURN_HARD_CAP*100:.0f}%), "
          f"{n_adaptive} adaptive causal ({DQ_RETURN_ADAPTIVE_MULT}x MAR), "
          f"{n_qv} QV spikes", flush=True)
    print(f"             Max |return|: {max_raw*100:.1f}% raw -> {max_clean*100:.1f}% clean",
          flush=True)

    # Stage 4: Update universe mask to disable NaN'd bars from trading
    _update_universe(universe_df, close)
    n_universe_disabled = int((~universe_df.values).sum()) - int(
        (close.isna() & ~universe_df.reindex(
            index=close.index, columns=close.columns, fill_value=False
        )).sum().sum()
    )  # rough count
    print(f"  Stage 4 -- Universe mask updated (disabled bad bars from trading)", flush=True)

    # Summary
    print(f"  -------------------------------", flush=True)
    print(f"  Result: {len(valid_tickers)} tickers, {close.shape[0]:,} bars", flush=True)
    print(f"  =============================\n", flush=True)

    return matrices, valid_tickers, universe_df


def _get_data_dir():
    """Return data directory based on current exchange."""
    if EXCHANGE == "kucoin":
        return "data/kucoin_cache"
    return "data/binance_cache"

def _get_universe_prefix():
    """Return universe prefix (BINANCE or KUCOIN)."""
    if EXCHANGE == "kucoin":
        return "KUCOIN"
    return "BINANCE"

def _get_universe_tickers():
    if '_tickers' in _DATA_CACHE:
        return _DATA_CACHE['_tickers'], _DATA_CACHE['_universe_df']
    data_dir = _get_data_dir()
    uni_prefix = _get_universe_prefix()
    uni_path = Path(f"{data_dir}/universes/{uni_prefix}_{PORTFOLIO_UNIVERSE}_{INTERVAL}.parquet")
    universe_df = pd.read_parquet(uni_path)
    coverage = universe_df.sum(axis=0) / len(universe_df)
    valid_tickers = sorted(coverage[coverage > COVERAGE_CUTOFF].index.tolist())

    # ── Signal/Trading Universe Split ──
    # TOP_LIQUID restricts TRADING only — signals are computed on ALL valid_tickers.
    # This gives cross-sectional breadth of the full universe (better z-scores)
    # while restricting execution to the most liquid names.
    global _TRADING_TICKERS
    if TOP_LIQUID > 0 and TOP_LIQUID < len(valid_tickers):
        mat_dir = Path(f"{data_dir}/matrices/{INTERVAL}")
        qv_path = mat_dir / "quote_volume.parquet"
        if qv_path.exists():
            qv = pd.read_parquet(qv_path)
            avg_qv = qv[[c for c in valid_tickers if c in qv.columns]].median()
            top_liquid_tickers = sorted(avg_qv.nlargest(TOP_LIQUID).index.tolist())
            _TRADING_TICKERS = set(top_liquid_tickers)
            removed = len(valid_tickers) - len(top_liquid_tickers)
            print(f"  Liquidity split: signal universe={len(valid_tickers)} tickers, "
                  f"trading universe={len(top_liquid_tickers)} tickers (removed {removed} from trading)",
                  flush=True)
        else:
            print(f"  Warning: quote_volume.parquet not found, skipping liquidity filter", flush=True)
            _TRADING_TICKERS = None
    else:
        _TRADING_TICKERS = None  # Trade the full signal universe

    _DATA_CACHE['_tickers'] = valid_tickers
    _DATA_CACHE['_universe_df'] = universe_df
    return valid_tickers, universe_df


def load_data(split="val"):
    if split in _DATA_CACHE:
        return _DATA_CACHE[split]

    mat_dir = Path(f"{_get_data_dir()}/matrices/{INTERVAL}")
    valid_tickers, universe_df = _get_universe_tickers()

    if '_full_matrices' not in _DATA_CACHE:
        print(f"  Loading 5m matrices ({len(valid_tickers)} tickers)...", flush=True)
        t0 = time.time()
        matrices = {}
        for fp in sorted(mat_dir.glob("*.parquet")):
            df = pd.read_parquet(fp)
            cols = [c for c in valid_tickers if c in df.columns]
            if cols:
                matrices[fp.stem] = df[cols]
        print(f"  Loaded {len(matrices)} fields in {time.time()-t0:.1f}s", flush=True)

        # ── Data Cleaning Pipeline ──
        matrices, valid_tickers, universe_df = _run_data_cleaning(
            matrices, valid_tickers, universe_df
        )
        # Update cache with cleaned tickers
        _DATA_CACHE['_tickers'] = valid_tickers
        _DATA_CACHE['_universe_df'] = universe_df

        _DATA_CACHE['_full_matrices'] = matrices
    else:
        matrices = _DATA_CACHE['_full_matrices']

    splits = {
        "train": (TRAIN_START, TRAIN_END),
        "val":   (VAL_START, VAL_END),
        "test":  (TEST_START, TEST_END),
        "trainval": (TRAIN_START, VAL_END),  # train+val combined
    }
    start, end = splits[split]
    split_matrices = {name: df.loc[start:end] for name, df in matrices.items()}
    split_universe = universe_df[valid_tickers].loc[start:end]

    # Fix universe coverage gap: if the split starts before the universe file,
    # extend the universe by inferring membership from close prices.
    if "close" in split_matrices:
        close_idx = split_matrices["close"].index
        if len(split_universe) < len(close_idx):
            # Reindex universe to match close prices, filling missing dates
            split_universe = split_universe.reindex(close_idx)
            # For dates not in original universe, infer: ticker is in-universe
            # if it has a valid (non-NaN) close price at that bar
            missing_mask = split_universe.isna().any(axis=1)
            if missing_mask.any():
                close_valid = split_matrices["close"].notna()
                split_universe.loc[missing_mask] = close_valid.loc[missing_mask]
            split_universe = split_universe.fillna(False).astype(bool)

    if "close" in split_matrices:
        split_matrices["returns"] = split_matrices["close"].pct_change()
        # Apply return winsorization to the PnL returns
        if "returns_clean" in matrices:
            split_matrices["returns_clean"] = matrices["returns_clean"].loc[start:end]

    result = (split_matrices, split_universe)
    _DATA_CACHE[split] = result
    return result


def evaluate_expression(expression, matrices):
    from src.operators.fastexpression import FastExpressionEngine
    engine = FastExpressionEngine(data_fields=matrices)
    return engine.evaluate(expression)


# ============================================================================
# ALPHA LOADING & COMBINATION
# ============================================================================

def load_alphas(universe=None):
    """Load all saved 5m alphas from DB, filtered by universe."""
    conn = sqlite3.connect(DB_PATH)
    if universe:
        db_universe = f"{_get_universe_prefix()}_{universe}"
        rows = conn.execute("""
            SELECT a.id, a.expression, COALESCE(e.ic_mean, 0), COALESCE(e.sharpe_is, 0)
            FROM alphas a LEFT JOIN evaluations e ON e.alpha_id = a.id
            WHERE a.archived = 0 AND a.universe = ?
            ORDER BY COALESCE(e.ic_mean, 0) DESC
        """, (db_universe,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT a.id, a.expression, COALESCE(e.ic_mean, 0), COALESCE(e.sharpe_is, 0)
            FROM alphas a LEFT JOIN evaluations e ON e.alpha_id = a.id
            WHERE a.archived = 0
            ORDER BY COALESCE(e.ic_mean, 0) DESC
        """).fetchall()
    conn.close()
    return rows


def compute_composite_signal(alphas, matrices, universe, method="ic_weighted"):
    """
    Combine alpha signals into a single composite per (bar, symbol).
    
    Methods:
      - equal: simple average of zscore_cs(alpha)
      - ic_weighted: weight by train-set IC (from DB)
      - rank: average of cross-sectional ranks
    
    Returns: DataFrame of composite signal values, range roughly [-3, 3]
    """
    signals = {}
    ic_weights = {}

    for alpha_id, expression, ic_mean, sharpe_is in alphas:
        try:
            raw = evaluate_expression(expression, matrices)
            if raw is None or raw.empty:
                continue
            # Apply universe mask
            uni_mask = universe.reindex(index=raw.index, columns=raw.columns).fillna(False)
            raw = raw.where(uni_mask, np.nan)
            signals[alpha_id] = raw
            ic_weights[alpha_id] = max(ic_mean, 0.0001)  # floor at small positive
        except Exception as e:
            print(f"  Alpha #{alpha_id} failed: {e}")

    if not signals:
        return None

    # Normalize each signal cross-sectionally (z-score)
    zscored = {}
    for aid, sig in signals.items():
        mu = sig.mean(axis=1)
        std = sig.std(axis=1).replace(0, np.nan)
        z = sig.sub(mu, axis=0).div(std, axis=0)
        z = z.clip(-3, 3)  # winsorize
        zscored[aid] = z

    if method == "equal":
        combined = sum(zscored.values()) / len(zscored)
    elif method == "ic_weighted":
        total_w = sum(ic_weights[aid] for aid in zscored)
        combined = sum(zscored[aid] * (ic_weights[aid] / total_w) for aid in zscored)
    elif method == "rank":
        ranked = {}
        for aid, sig in signals.items():
            ranked[aid] = sig.rank(axis=1, pct=True)  # 0 to 1
        combined = sum(ranked.values()) / len(ranked)
        # Shift to centered
        combined = combined - 0.5
    else:
        raise ValueError(f"Unknown method: {method}")

    # Final cross-sectional z-score of combined
    mu = combined.mean(axis=1)
    std = combined.std(axis=1).replace(0, np.nan)
    composite = combined.sub(mu, axis=0).div(std, axis=0)

    return composite


def compute_composite_ml(alphas, matrices, universe, returns,
                         train_bars=1728, retrain_every=288,
                         target_horizon=36, regularization=1.0,
                         use_enriched_features=False):
    """
    ML-based signal combination using Ridge regression.
    
    Rolling retrain:
      - Every `retrain_every` bars, fit Ridge on the last `train_bars` of data
      - Features: per-symbol alpha signal values (z-scored)
      - Target: forward `target_horizon`-bar risk-adjusted excess return
        (forward return / rolling volatility - market median)
      - The trained model predicts composite signal for the next chunk
    
    Creative target engineering:
      - Risk-adjusted: divide by local vol to equalize signal across vol regimes
      - Excess: subtract cross-sectional median to make it market-neutral
      - Multi-horizon: uses 3h forward return (36 bars) to match holding period
    """
    from sklearn.linear_model import Ridge
    
    # First evaluate all alpha signals
    signals = {}
    for alpha_id, expression, ic_mean, sharpe_is in alphas:
        try:
            raw = evaluate_expression(expression, matrices)
            if raw is None or raw.empty:
                continue
            uni_mask = universe.reindex(index=raw.index, columns=raw.columns).fillna(False)
            raw = raw.where(uni_mask, np.nan)
            # Z-score each alpha cross-sectionally
            mu = raw.mean(axis=1)
            std = raw.std(axis=1).replace(0, np.nan)
            z = raw.sub(mu, axis=0).div(std, axis=0).clip(-3, 3)
            signals[alpha_id] = z
        except Exception as e:
            print(f"  Alpha #{alpha_id} failed: {e}")
    
    if not signals:
        return None
    
    alpha_ids = sorted(signals.keys())
    dates = returns.index
    tickers = returns.columns
    n_bars = len(dates)
    n_tickers = len(tickers)
    K = len(alpha_ids)
    
    # Build 3D feature tensor: (bars, tickers, K)
    feat_3d = np.zeros((n_bars, n_tickers, K))
    for k, aid in enumerate(alpha_ids):
        sig = signals[aid].reindex(index=dates, columns=tickers).fillna(0).values
        feat_3d[:, :, k] = sig
    
    # Optionally use enriched features (momentum + interactions + vol regime)
    if use_enriched_features:
        features, n_feat = _build_enriched_features(signals, returns, alpha_ids, dates, tickers)
    else:
        features = feat_3d
        n_feat = K
    
    # Compute target: excess forward return in percent
    ret_np = returns.fillna(0).values
    # Forward N-bar cumulative return
    fwd_ret = np.zeros((n_bars, n_tickers))
    for t in range(n_bars - target_horizon):
        fwd_ret[t] = ret_np[t+1:t+1+target_horizon].sum(axis=0)
    
    # Convert to percent and make excess (subtract cross-sectional median)
    fwd_pct = fwd_ret * 100.0  # now in percent, e.g. 2.5 means +2.5%
    for t in range(n_bars):
        med = np.nanmedian(fwd_pct[t])
        fwd_pct[t] -= med
    
    # Winsorize at 1st/99th percentile per cross-section to clip outliers
    for t in range(n_bars):
        row = fwd_pct[t]
        finite = row[np.isfinite(row)]
        if len(finite) > 10:
            lo, hi = np.percentile(finite, [1, 99])
            fwd_pct[t] = np.clip(row, lo, hi)
    
    target = fwd_pct  # (n_bars, n_tickers) — in excess return percent
    
    # Rolling retrain with cross-validated regularization
    composite_out = np.zeros((n_bars, n_tickers))
    from sklearn.linear_model import RidgeCV
    model = RidgeCV(alphas=[10.0, 100.0, 1000.0, 10000.0])
    last_trained = -retrain_every  # force initial train
    
    for t in range(train_bars, n_bars, 1):
        if t - last_trained >= retrain_every or t == train_bars:
            # Train on [t-train_bars, t)
            t0 = max(0, t - train_bars)
            X_train = features[t0:t].reshape(-1, n_feat)  # flatten (bars*tickers, n_feat)
            y_train = target[t0:t].reshape(-1)
            
            # Filter valid samples (not NaN target, not all-zero features)
            valid = np.isfinite(y_train) & (np.abs(X_train).sum(axis=1) > 0)
            if valid.sum() < 100:
                composite_out[t] = 0
                continue
            
            model.fit(X_train[valid], y_train[valid])
            last_trained = t
        
        # Predict for bar t
        X_t = features[t]  # (n_tickers, n_feat)
        pred = model.predict(X_t)
        composite_out[t] = pred
    
    composite = pd.DataFrame(composite_out, index=dates, columns=tickers)
    
    # Final z-score
    mu = composite.mean(axis=1)
    std = composite.std(axis=1).replace(0, np.nan)
    composite = composite.sub(mu, axis=0).div(std, axis=0)
    
    enriched_str = f" (enriched: {n_feat} features)" if use_enriched_features else ""
    print(f"  ML Ridge: K={K} alphas, target_horizon={target_horizon} bars, "
          f"retrain_every={retrain_every} bars{enriched_str}", flush=True)
    if hasattr(model, 'coef_'):
        coef_str = ", ".join(f"#{alpha_ids[i]}:{model.coef_[i]:+.3f}" for i in range(min(K, len(alpha_ids))))
        alpha_str = f" (ridge_alpha={model.alpha_:.0f})" if hasattr(model, 'alpha_') else ""
        print(f"  Last model coefficients{alpha_str}: {coef_str}", flush=True)
    
    return composite


def _build_enriched_features(signals, returns, alpha_ids, dates, tickers):
    """
    Build enriched feature matrix beyond raw alpha z-scores.
    
    Features per (bar, symbol):
      - Raw alpha z-scores (K)
      - Alpha momentum: 72-bar change in z-score (K)
      - Cross-alpha interactions: pairwise products of top 3 (3)
      - Market vol regime: rolling 72-bar market vol z-score (1, broadcast)
      - Symbol vol regime: symbol's 72-bar vol z-score (1)
    """
    n_bars = len(dates)
    n_tickers = len(tickers)
    K = len(alpha_ids)
    
    feat_raw = np.zeros((n_bars, n_tickers, K))
    for k, aid in enumerate(alpha_ids):
        sig = signals[aid].reindex(index=dates, columns=tickers).fillna(0).values
        feat_raw[:, :, k] = sig
    
    feat_mom = np.zeros_like(feat_raw)
    feat_mom[72:] = feat_raw[72:] - feat_raw[:-72]
    
    n_interact = min(3, K)
    feat_interact = np.zeros((n_bars, n_tickers, n_interact))
    for i in range(n_interact):
        j = (i + 1) % K
        feat_interact[:, :, i] = feat_raw[:, :, i] * feat_raw[:, :, j]
    
    ret_np = returns.reindex(index=dates, columns=tickers).fillna(0).values
    mkt_ret = ret_np.mean(axis=1)
    mkt_vol = pd.Series(mkt_ret).rolling(72, min_periods=10).std().fillna(0).values
    mv_mean = np.mean(mkt_vol[72:]) if len(mkt_vol) > 72 else 0
    mv_std = max(np.std(mkt_vol[72:]), 1e-8) if len(mkt_vol) > 72 else 1
    feat_mkt_vol = np.tile(((mkt_vol - mv_mean) / mv_std).reshape(-1, 1), (1, n_tickers))
    
    sym_vol = returns.reindex(index=dates, columns=tickers).rolling(72, min_periods=10).std().fillna(0).values
    sym_vol_mu = np.nanmean(sym_vol[72:], axis=0, keepdims=True)
    sym_vol_std = np.maximum(np.nanstd(sym_vol[72:], axis=0, keepdims=True), 1e-8)
    feat_sym_vol = (sym_vol - sym_vol_mu) / sym_vol_std
    
    total_f = K + K + n_interact + 2
    features = np.zeros((n_bars, n_tickers, total_f))
    idx = 0
    features[:, :, idx:idx+K] = feat_raw; idx += K
    features[:, :, idx:idx+K] = feat_mom; idx += K
    features[:, :, idx:idx+n_interact] = feat_interact; idx += n_interact
    features[:, :, idx] = feat_mkt_vol; idx += 1
    features[:, :, idx] = feat_sym_vol
    
    return features, total_f


def compute_composite_gbt(alphas, matrices, universe, returns,
                          train_bars=1728, retrain_every=288,
                          target_horizon=36, n_estimators=100, max_depth=3):
    """
    GradientBoosting Regressor with Sharpe-optimal targets.
    
    Target = (fwd_return - cross_median) / max(fwd_vol, floor)
    Enriched features: raw alphas + momentum + interactions + vol regime.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    
    signals = {}
    for alpha_id, expression, ic_mean, sharpe_is in alphas:
        try:
            raw = evaluate_expression(expression, matrices)
            if raw is None or raw.empty:
                continue
            uni_mask = universe.reindex(index=raw.index, columns=raw.columns).fillna(False)
            raw = raw.where(uni_mask, np.nan)
            mu = raw.mean(axis=1); std = raw.std(axis=1).replace(0, np.nan)
            z = raw.sub(mu, axis=0).div(std, axis=0).clip(-3, 3)
            signals[alpha_id] = z
        except Exception as e:
            print(f"  Alpha #{alpha_id} failed: {e}")
    
    if not signals:
        return None
    
    alpha_ids = sorted(signals.keys())
    dates = returns.index; tickers = returns.columns
    n_bars = len(dates); n_tickers = len(tickers)
    
    features, n_feat = _build_enriched_features(signals, returns, alpha_ids, dates, tickers)
    
    ret_np = returns.fillna(0).values
    fwd_ret = np.zeros((n_bars, n_tickers))
    fwd_vol = np.zeros((n_bars, n_tickers))
    for t in range(n_bars - target_horizon):
        chunk = ret_np[t+1:t+1+target_horizon]
        fwd_ret[t] = chunk.sum(axis=0)
        fwd_vol[t] = chunk.std(axis=0)
    fwd_vol = np.maximum(fwd_vol, 1e-5)
    for t in range(n_bars):
        fwd_ret[t] -= np.nanmedian(fwd_ret[t])
    target = fwd_ret / fwd_vol
    
    composite_out = np.zeros((n_bars, n_tickers))
    model = GradientBoostingRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=0.1, subsample=0.8, random_state=42)
    last_trained = -retrain_every
    
    for t in range(train_bars, n_bars):
        if t - last_trained >= retrain_every or t == train_bars:
            t0 = max(0, t - train_bars)
            X = features[t0:t].reshape(-1, n_feat)
            y = target[t0:t].reshape(-1)
            valid = np.isfinite(y) & (np.abs(X).sum(axis=1) > 0)
            if valid.sum() < 200:
                continue
            if valid.sum() > 50000:
                idx = np.random.choice(np.where(valid)[0], 50000, replace=False)
                model.fit(X[idx], y[idx])
            else:
                model.fit(X[valid], y[valid])
            last_trained = t
        composite_out[t] = model.predict(features[t])
    
    composite = pd.DataFrame(composite_out, index=dates, columns=tickers)
    mu = composite.mean(axis=1); std = composite.std(axis=1).replace(0, np.nan)
    composite = composite.sub(mu, axis=0).div(std, axis=0)
    
    print(f"  GBT Regressor: {n_feat}feat, target=sharpe_opt_{target_horizon}bar, "
          f"trees={n_estimators}, depth={max_depth}", flush=True)
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        labels = ([f"raw_{a}" for a in alpha_ids] + [f"mom_{a}" for a in alpha_ids] +
                  [f"xint_{i}" for i in range(min(3, len(alpha_ids)))] + ["mkt_vol", "sym_vol"])
        top5 = np.argsort(imp)[-5:][::-1]
        print(f"  Top features: {', '.join(f'{labels[i]}:{imp[i]:.3f}' for i in top5)}", flush=True)
    return composite


def compute_composite_gbt_classifier(alphas, matrices, universe, returns,
                                      train_bars=1728, retrain_every=288,
                                      target_horizon=72, n_estimators=100, max_depth=3):
    """
    GBT Classifier: predicts top/bottom quintile of forward cross-sectional returns.
    Signal = P(top_quintile) - P(bottom_quintile).
    """
    from sklearn.ensemble import GradientBoostingClassifier
    
    signals = {}
    for alpha_id, expression, ic_mean, sharpe_is in alphas:
        try:
            raw = evaluate_expression(expression, matrices)
            if raw is None or raw.empty:
                continue
            uni_mask = universe.reindex(index=raw.index, columns=raw.columns).fillna(False)
            raw = raw.where(uni_mask, np.nan)
            mu = raw.mean(axis=1); std = raw.std(axis=1).replace(0, np.nan)
            z = raw.sub(mu, axis=0).div(std, axis=0).clip(-3, 3)
            signals[alpha_id] = z
        except Exception as e:
            print(f"  Alpha #{alpha_id} failed: {e}")
    
    if not signals:
        return None
    
    alpha_ids = sorted(signals.keys())
    dates = returns.index; tickers = returns.columns
    n_bars = len(dates); n_tickers = len(tickers)
    
    features, n_feat = _build_enriched_features(signals, returns, alpha_ids, dates, tickers)
    
    ret_np = returns.fillna(0).values
    fwd_ret = np.zeros((n_bars, n_tickers))
    for t in range(n_bars - target_horizon):
        fwd_ret[t] = ret_np[t+1:t+1+target_horizon].sum(axis=0)
    
    target = np.zeros((n_bars, n_tickers))
    for t in range(n_bars - target_horizon):
        row = fwd_ret[t]
        valid = np.isfinite(row) & (row != 0)
        if valid.sum() < 20:
            continue
        p20 = np.percentile(row[valid], 20)
        p80 = np.percentile(row[valid], 80)
        target[t, row <= p20] = -1
        target[t, row >= p80] = 1
    
    composite_out = np.zeros((n_bars, n_tickers))
    model = GradientBoostingClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=0.1, subsample=0.8, random_state=42)
    last_trained = -retrain_every
    
    for t in range(train_bars, n_bars):
        if t - last_trained >= retrain_every or t == train_bars:
            t0 = max(0, t - train_bars)
            X = features[t0:t].reshape(-1, n_feat)
            y = target[t0:t].reshape(-1)
            valid = np.isfinite(y) & (np.abs(X).sum(axis=1) > 0)
            if valid.sum() < 200 or len(np.unique(y[valid])) < 2:
                continue
            if valid.sum() > 50000:
                idx = np.random.choice(np.where(valid)[0], 50000, replace=False)
                model.fit(X[idx], y[idx])
            else:
                model.fit(X[valid], y[valid])
            last_trained = t
        try:
            probs = model.predict_proba(features[t])
            classes = list(model.classes_)
            p_top = probs[:, classes.index(1)] if 1 in classes else np.zeros(n_tickers)
            p_bot = probs[:, classes.index(-1)] if -1 in classes else np.zeros(n_tickers)
            composite_out[t] = p_top - p_bot
        except:
            composite_out[t] = 0
    
    composite = pd.DataFrame(composite_out, index=dates, columns=tickers)
    mu = composite.mean(axis=1); std = composite.std(axis=1).replace(0, np.nan)
    composite = composite.sub(mu, axis=0).div(std, axis=0)
    
    print(f"  GBT Classifier: {n_feat}feat, quintile_{target_horizon}bar, "
          f"trees={n_estimators}, depth={max_depth}", flush=True)
    return composite


def compute_composite_gbt_entry(alphas, matrices, universe, returns,
                                 train_bars=1728, retrain_every=288,
                                 lookahead=144, n_estimators=80, max_depth=4):
    """
    GBT Entry Predictor: directly predicts if entering NOW is profitable net of fees.
    
    Two classifiers: P(profitable_long) and P(profitable_short).
    Signal = P(long) - P(short). Bakes trading cost into the ML target.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    
    signals = {}
    for alpha_id, expression, ic_mean, sharpe_is in alphas:
        try:
            raw = evaluate_expression(expression, matrices)
            if raw is None or raw.empty:
                continue
            uni_mask = universe.reindex(index=raw.index, columns=raw.columns).fillna(False)
            raw = raw.where(uni_mask, np.nan)
            mu = raw.mean(axis=1); std = raw.std(axis=1).replace(0, np.nan)
            z = raw.sub(mu, axis=0).div(std, axis=0).clip(-3, 3)
            signals[alpha_id] = z
        except Exception as e:
            print(f"  Alpha #{alpha_id} failed: {e}")
    
    if not signals:
        return None
    
    alpha_ids = sorted(signals.keys())
    dates = returns.index; tickers = returns.columns
    n_bars = len(dates); n_tickers = len(tickers)
    
    features, n_feat = _build_enriched_features(signals, returns, alpha_ids, dates, tickers)
    
    ret_np = returns.fillna(0).values
    fee_cost = FEES_BPS / 10000.0 * 2  # round-trip
    
    target_long = np.zeros((n_bars, n_tickers))
    target_short = np.zeros((n_bars, n_tickers))
    for t in range(n_bars - lookahead):
        fwd = ret_np[t+1:t+1+lookahead].sum(axis=0)
        target_long[t] = (fwd > fee_cost).astype(float)
        target_short[t] = (fwd < -fee_cost).astype(float)
    
    composite_out = np.zeros((n_bars, n_tickers))
    model_l = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                          learning_rate=0.1, subsample=0.8, random_state=42)
    model_s = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                          learning_rate=0.1, subsample=0.8, random_state=43)
    last_trained = -retrain_every
    
    for t in range(train_bars, n_bars):
        if t - last_trained >= retrain_every or t == train_bars:
            t0 = max(0, t - train_bars)
            X = features[t0:t].reshape(-1, n_feat)
            yl = target_long[t0:t].reshape(-1)
            ys = target_short[t0:t].reshape(-1)
            valid = np.isfinite(yl) & (np.abs(X).sum(axis=1) > 0)
            if valid.sum() < 200:
                continue
            if valid.sum() > 40000:
                idx = np.random.choice(np.where(valid)[0], 40000, replace=False)
                Xs, yls, yss = X[idx], yl[idx], ys[idx]
            else:
                Xs, yls, yss = X[valid], yl[valid], ys[valid]
            if len(np.unique(yls)) >= 2:
                model_l.fit(Xs, yls)
            if len(np.unique(yss)) >= 2:
                model_s.fit(Xs, yss)
            last_trained = t
        try:
            pl = model_l.predict_proba(features[t])[:, 1]
        except:
            pl = np.zeros(n_tickers)
        try:
            ps = model_s.predict_proba(features[t])[:, 1]
        except:
            ps = np.zeros(n_tickers)
        composite_out[t] = pl - ps
    
    composite = pd.DataFrame(composite_out, index=dates, columns=tickers)
    mu = composite.mean(axis=1); std = composite.std(axis=1).replace(0, np.nan)
    composite = composite.sub(mu, axis=0).div(std, axis=0)
    
    print(f"  GBT Entry: {n_feat}feat, lookahead={lookahead}bar, fee_cost={fee_cost*100:.2f}%, "
          f"trees={n_estimators}, depth={max_depth}", flush=True)
    return composite


def compute_composite_concordance(alphas, matrices, universe, returns,
                                   vote_threshold=1.0, min_votes=3,
                                   ic_lookback=288, ic_gate=0.0):
    """
    Concordance voting: count how many alphas agree on direction.
    
    No regression needed — each alpha casts a binary vote based on its
    cross-sectional z-score exceeding +/- vote_threshold. The composite
    is the net vote count, optionally gated by rolling IC.
    
    For takeout strategies, this directly measures "breadth of conviction"
    rather than fitting noisy coefficients.
    """
    from scipy import stats as sp_stats
    
    K = len(alphas)
    alpha_ids = [a[0] for a in alphas]
    
    # Evaluate all alphas
    signals = {}
    for alpha_id, expression, ic_mean, sharpe_is in alphas:
        try:
            raw = evaluate_expression(expression, matrices)
            if raw is None or raw.empty:
                continue
            uni_mask = universe.reindex(index=raw.index, columns=raw.columns).fillna(False)
            raw = raw.where(uni_mask, np.nan)
            signals[alpha_id] = raw
        except Exception as e:
            print(f"  Alpha #{alpha_id} failed: {e}")
    
    if not signals:
        return None
    
    # Z-score each alpha cross-sectionally
    zscored = {}
    for aid, sig in signals.items():
        mu = sig.mean(axis=1)
        std = sig.std(axis=1).replace(0, np.nan)
        z = sig.sub(mu, axis=0).div(std, axis=0).clip(-5, 5)
        zscored[aid] = z
    
    # Get common index
    ref_aid = list(zscored.keys())[0]
    dates = zscored[ref_aid].index
    tickers = zscored[ref_aid].columns
    n_bars = len(dates)
    n_tickers = len(tickers)
    
    # Compute rolling IC for each alpha (per-bar IC then rolling mean)
    fwd_ret = returns.reindex(index=dates, columns=tickers).shift(-1)
    
    rolling_ic = {}
    for aid, z in zscored.items():
        ic_arr = np.full(n_bars, np.nan)
        for t in range(n_bars - 1):
            a = z.iloc[t]
            r = fwd_ret.iloc[t]
            valid = a.notna() & r.notna() & np.isfinite(a) & np.isfinite(r)
            a_v, r_v = a[valid].values, r[valid].values
            if len(a_v) >= 15 and np.std(a_v) > 1e-10:
                ic, _ = sp_stats.spearmanr(a_v, r_v)
                ic_arr[t] = ic
        # Rolling mean IC
        ic_series = pd.Series(ic_arr)
        rolling_ic[aid] = ic_series.rolling(ic_lookback, min_periods=72).mean().fillna(0).values
    
    # Build concordance signal
    composite_np = np.zeros((n_bars, n_tickers))
    
    for t in range(ic_lookback, n_bars):
        long_votes = np.zeros(n_tickers)
        short_votes = np.zeros(n_tickers)
        
        for aid, z in zscored.items():
            # Gate: only count vote if rolling IC is above threshold
            if rolling_ic[aid][t] < ic_gate:
                continue
            
            z_row = z.iloc[t].values
            z_row = np.nan_to_num(z_row, nan=0.0)
            
            long_votes += (z_row > vote_threshold).astype(float)
            short_votes += (z_row < -vote_threshold).astype(float)
        
        # Net votes, scaled by magnitude (average z-score of agreeing signals)
        net = long_votes - short_votes
        composite_np[t] = net
    
    composite = pd.DataFrame(composite_np, index=dates, columns=tickers)
    
    # Final z-score
    mu = composite.mean(axis=1)
    std = composite.std(axis=1).replace(0, np.nan)
    composite = composite.sub(mu, axis=0).div(std, axis=0)
    
    n_active = sum(1 for aid in rolling_ic if rolling_ic[aid][-1] > ic_gate)
    print(f"  Concordance: {len(signals)} alphas, vote_tau={vote_threshold}, "
          f"min_votes={min_votes}, IC gate={ic_gate}, active={n_active}", flush=True)
    
    return composite


def compute_composite_ic_rolling(alphas, matrices, universe, returns,
                                  ic_lookback=288, ic_smooth=72):
    """
    Rolling IC-weighted alpha combination.
    
    Each alpha is weighted by its trailing IC (Spearman with forward returns),
    smoothed with an SMA. Weights are naturally bounded [-1, 1] and self-deactivate
    for alphas that stop working.
    
    No regression coefficients to estimate — purely adaptive.
    """
    from scipy import stats as sp_stats
    
    # Evaluate all alphas
    signals = {}
    for alpha_id, expression, ic_mean, sharpe_is in alphas:
        try:
            raw = evaluate_expression(expression, matrices)
            if raw is None or raw.empty:
                continue
            uni_mask = universe.reindex(index=raw.index, columns=raw.columns).fillna(False)
            raw = raw.where(uni_mask, np.nan)
            signals[alpha_id] = raw
        except Exception as e:
            print(f"  Alpha #{alpha_id} failed: {e}")
    
    if not signals:
        return None
    
    # Z-score each alpha
    zscored = {}
    for aid, sig in signals.items():
        mu = sig.mean(axis=1)
        std = sig.std(axis=1).replace(0, np.nan)
        z = sig.sub(mu, axis=0).div(std, axis=0).clip(-5, 5)
        zscored[aid] = z
    
    ref_aid = list(zscored.keys())[0]
    dates = zscored[ref_aid].index
    tickers = zscored[ref_aid].columns
    n_bars = len(dates)
    n_tickers = len(tickers)
    
    fwd_ret = returns.reindex(index=dates, columns=tickers).shift(-1)
    
    # Compute per-bar cross-sectional IC for each alpha
    per_bar_ic = {}
    for aid, z in zscored.items():
        ic_arr = np.full(n_bars, np.nan)
        for t in range(n_bars - 1):
            a = z.iloc[t]
            r = fwd_ret.iloc[t]
            valid = a.notna() & r.notna() & np.isfinite(a) & np.isfinite(r)
            a_v, r_v = a[valid].values, r[valid].values
            if len(a_v) >= 15 and np.std(a_v) > 1e-10:
                ic, _ = sp_stats.spearmanr(a_v, r_v)
                ic_arr[t] = ic
        per_bar_ic[aid] = ic_arr
    
    # Rolling mean + SMA smoothing of IC
    rolling_weights = {}
    for aid, ic_arr in per_bar_ic.items():
        # Rolling mean IC
        ic_series = pd.Series(ic_arr)
        rolling_mean = ic_series.rolling(ic_lookback, min_periods=72).mean()
        # Smooth further
        smoothed = rolling_mean.rolling(ic_smooth, min_periods=1).mean()
        rolling_weights[aid] = smoothed.fillna(0).values
    
    # Build composite: IC-weighted sum of z-scored alphas
    composite_np = np.zeros((n_bars, n_tickers))
    
    for t in range(ic_lookback, n_bars):
        total_abs_w = sum(abs(rolling_weights[aid][t]) for aid in zscored)
        if total_abs_w < 1e-10:
            continue
        
        for aid, z in zscored.items():
            w = rolling_weights[aid][t]
            z_row = z.iloc[t].values
            z_row = np.nan_to_num(z_row, nan=0.0)
            composite_np[t] += w * z_row / total_abs_w
    
    composite = pd.DataFrame(composite_np, index=dates, columns=tickers)
    
    # Final z-score
    mu = composite.mean(axis=1)
    std = composite.std(axis=1).replace(0, np.nan)
    composite = composite.sub(mu, axis=0).div(std, axis=0)
    
    # Report final weights
    final_w = {aid: rolling_weights[aid][-1] for aid in zscored}
    top_w = sorted(final_w.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    w_str = ", ".join(f"#{aid}:{w:+.4f}" for aid, w in top_w)
    print(f"  IC Rolling: {len(signals)} alphas, lookback={ic_lookback}, "
          f"smooth={ic_smooth}", flush=True)
    print(f"  Top 5 IC weights: {w_str}", flush=True)
    
    return composite


def compute_composite_ts_autonorm(alphas, matrices, universe, returns,
                                    gamma=0.01, ic_weighted=True,
                                    signal_smooth=0, concordance_boost=False,
                                    rank_norm=False, beta_hedge=False,
                                    signal_hedge=0, rolling_ic=0,
                                    rolling_return=0):
    """
    Time-Series Auto-Normalization (Smart Hybrid).

    For each alpha, detects whether its expression already contains
    cross-sectional normalization (zscore_cs). Two normalization paths:

    A) RAW alphas (no zscore_cs): Apply per-asset EWMA detrending.
       z_ts = (raw - ewma_mean) / ewma_std
       Captures "this asset's signal is elevated vs its own recent history."

    B) PRE-NORMALIZED alphas (has zscore_cs): Apply per-asset CS z-score only.
       Uses standard cross-sectional z-scoring (same as ic_weighted method)
       to avoid double-normalizing.

    After per-alpha normalization, combine with IC weights and apply
    a final cross-sectional z-score so thresholds work consistently.

    gamma controls EWMA decay for raw alphas:
      0.01 ~ 100-bar window, 0.005 ~ 200-bar, 0.02 ~ 50-bar
    
    Creative enhancements:
      signal_smooth: EMA span to apply to composite (reduces whipsaw). 0=off.
      concordance_boost: When True, multiply signal by fraction of alphas
        that agree on direction (boosts high-conviction moments).
      rank_norm: When True, use cross-sectional rank percentile instead of
        z-score for the final normalization (more robust to outliers).
      beta_hedge: When True, regress out BTC returns from the signal to create
        a market-neutral composite.
    """
    signals = {}
    ic_weights = {}
    alpha_expressions = {}  # track expressions for smart normalization

    for alpha_id, expression, ic_mean, sharpe_is in alphas:
        try:
            raw = evaluate_expression(expression, matrices)
            if raw is None or raw.empty:
                continue
            uni_mask = universe.reindex(index=raw.index, columns=raw.columns).fillna(False)
            raw = raw.where(uni_mask, np.nan)
            signals[alpha_id] = raw
            ic_weights[alpha_id] = max(ic_mean, 0.0001)
            alpha_expressions[alpha_id] = expression
        except Exception as e:
            print(f"  Alpha #{alpha_id} failed: {e}")

    if not signals:
        return None

    # Smart normalization: detect pre-normalized vs raw
    ewm_span = int(1.0 / gamma)
    min_periods = max(ewm_span // 4, 5)
    normed = {}
    n_raw = 0
    n_prenorm = 0

    for aid, sig in signals.items():
        expr = alpha_expressions.get(aid, "")
        has_zscore_cs = "zscore_cs" in expr

        if has_zscore_cs:
            # Pre-normalized: use cross-sectional z-score (like ic_weighted)
            mu = sig.mean(axis=1)
            std = sig.std(axis=1).replace(0, np.nan)
            z = sig.sub(mu, axis=0).div(std, axis=0).clip(-3, 3)
            normed[aid] = z.fillna(0.0)
            n_prenorm += 1
        else:
            # Raw signal: normalize by per-asset EWMA std only (no detrending)
            # Preserves directional signal while standardizing scale
            ewm_std = sig.ewm(span=ewm_span, min_periods=min_periods).std().replace(0, np.nan)
            ts_z = sig.div(ewm_std).clip(-5, 5)
            normed[aid] = ts_z.fillna(0.0)
            n_raw += 1

    # ── Concordance boost: measure alpha agreement ──
    if concordance_boost and len(normed) > 3:
        # For each bar, count fraction of alphas agreeing on direction
        # Then multiply the composite by this "conviction" score
        sign_sum = sum(np.sign(normed[aid]) for aid in normed)
        n_alphas = len(normed)
        # agreement = |sum of signs| / n_alphas -> [0, 1]
        # when all agree: 1.0, when half: 0.5, when split: 0.0
        agreement = sign_sum.abs() / n_alphas
        # Rescale to [0.5, 1.5] so it modulates rather than kills signal
        concordance_multiplier = 0.5 + agreement
    else:
        concordance_multiplier = None

    # Combine via IC-weighted or equal average
    if ic_weighted:
        if rolling_return > 0:
            # ── Rolling causal RETURN weighting: weight by realized P&L ──
            #
            # For each alpha k at each bar t, compute the realized L/S return:
            #   ret_k(t) = mean(sign(signal_k[t-1, :]) * return[t, :])
            # This is the average cross-sectional return of going long when
            # signal > 0 and short when signal < 0.
            #
            # Then: ema_ret_k(t) = EMA(ret_k, span=rolling_return)
            # Weight_k(t) = max(ema_ret_k(t), 0.0001)
            #
            # vs Rolling IC: IC measures rank correlation (ordinal quality),
            # while rolling return measures actual P&L magnitude. Return
            # weighting directly optimizes what we care about — profit.
            #
            aid_list = list(normed.keys())
            n_alphas = len(aid_list)
            dates = normed[aid_list[0]].index
            tickers = normed[aid_list[0]].columns
            n_bars = len(dates)
            n_tickers = len(tickers)

            close = matrices.get('close')
            fwd_ret = close.pct_change().reindex(index=dates, columns=tickers).fillna(0.0)

            # Compute per-bar L/S return for each alpha (vectorized)
            ret_series = {}
            for aid in aid_list:
                sig_df = normed[aid]
                sig_lagged = sig_df.shift(1)  # causal: use signal from t-1
                # Normalize to unit weight per bar (simple sign weighting)
                wt = np.sign(sig_lagged)
                n_nonzero = (wt != 0).sum(axis=1).clip(lower=1)
                # Per-alpha L/S return: average of sign(signal) * return
                alpha_ret = (wt * fwd_ret).sum(axis=1) / n_nonzero
                ret_series[aid] = alpha_ret

            # EMA-smooth the per-bar returns
            ema_rets = {}
            for aid in aid_list:
                ema_rets[aid] = ret_series[aid].ewm(
                    span=rolling_return, min_periods=max(rolling_return // 4, 10)
                ).mean().fillna(0.0)

            # Build combined signal with time-varying return weights (vectorized)
            # CRITICAL: zero out negative-return alphas — they should get NO weight.
            # This naturally kills alphas that stop working in the current regime.
            ema_ret_matrix = np.column_stack([ema_rets[aid].values for aid in aid_list])
            ema_ret_matrix = np.maximum(ema_ret_matrix, 0.0)  # zero floor: negative return = zero weight
            weight_sums = ema_ret_matrix.sum(axis=1, keepdims=True)
            weight_sums = np.maximum(weight_sums, 1e-10)
            ema_ret_matrix = ema_ret_matrix / weight_sums  # normalize to sum=1 per bar

            signal_stack = np.stack([normed[aid].values for aid in aid_list], axis=2)
            combined_values = np.einsum('ijk,ik->ij', signal_stack, ema_ret_matrix)
            combined = pd.DataFrame(combined_values, index=dates, columns=tickers)

            mean_ema_rets = {aid: ema_rets[aid].mean() for aid in aid_list}
            avg_ret = np.mean(list(mean_ema_rets.values()))
            # Count how many alphas are active (positive return) at the last bar
            n_active_final = int((ema_ret_matrix[-1] > 0).sum())
            n_active_avg = float((ema_ret_matrix > 0).mean(axis=0).sum())
            print(f"  Rolling Return: EMA span={rolling_return} ({rolling_return*5/60:.1f}h), "
                  f"avg ret={avg_ret:.7f}, active={n_active_final}/{n_alphas} "
                  f"(avg {n_active_avg:.1f}), computed causally", flush=True)

        elif rolling_ic > 0:
            # ── Rolling causal IC: no lookahead in weights ──
            #
            # Instead of using static IC from the DB (which may be computed on
            # future data), compute the IC causally at each bar using trailing data.
            #
            # Method:
            #   For each bar t and alpha k:
            #     IC_k(t) = rank_corr(signal_k[t-1, :], returns[t, :])  across tickers
            #   This is causal: at time t, we know signal at t-1 and return at t.
            #   The IC measures "did yesterday's signal predict today's return?"
            #
            #   Then: ema_IC_k(t) = EMA(IC_k, span=rolling_ic)
            #   Weight_k(t) = max(ema_IC_k(t), 0.0001)
            #
            # This makes weights fully adaptive — alphas that work in the current
            # regime get higher weight, and alphas that stop working fade out.
            #
            from scipy.stats import spearmanr
            
            # Stack all normalized signals into a 3D array: (n_alphas, n_bars, n_tickers)
            aid_list = list(normed.keys())
            n_alphas = len(aid_list)
            dates = normed[aid_list[0]].index
            tickers = normed[aid_list[0]].columns
            n_bars = len(dates)
            n_tickers = len(tickers)
            
            # Get 1-bar forward returns (causal: return[t] is known at time t)
            close = matrices.get('close')
            fwd_ret = close.pct_change().reindex(index=dates, columns=tickers).fillna(0.0)
            
            # Compute per-bar cross-sectional IC for each alpha (VECTORIZED)
            # IC_k(t) = rank_corr(signal_k[t-1, :], return[t, :])
            # 
            # Instead of per-bar spearmanr (slow), compute cross-sectional rank
            # correlation efficiently:
            #   1. Rank each row (cross-sectionally) for both signal and returns
            #   2. Compute the Pearson correlation of the ranks at each bar
            #      using the formula: corr = (Σ(r_x - r̄)(r_y - r̄)) / (N × std_x × std_y)
            #
            ic_series = {}
            ret_ranked = fwd_ret.rank(axis=1, pct=True)  # (n_bars, n_tickers) ranks
            
            for aid in aid_list:
                sig_df = normed[aid]
                # Lag signal by 1 bar so IC_k(t) = corr(signal[t-1], return[t])
                sig_lagged = sig_df.shift(1)
                sig_ranked = sig_lagged.rank(axis=1, pct=True)  # cross-sectional ranks
                
                # Cross-sectional Pearson correlation of ranks at each bar
                # = Spearman rank correlation
                n_valid = sig_ranked.notna().sum(axis=1).clip(lower=1)
                sig_dm = sig_ranked.sub(sig_ranked.mean(axis=1), axis=0)
                ret_dm = ret_ranked.sub(ret_ranked.mean(axis=1), axis=0)
                
                cov_sr = (sig_dm * ret_dm).sum(axis=1)
                std_sig = (sig_dm ** 2).sum(axis=1).clip(lower=1e-10) ** 0.5
                std_ret = (ret_dm ** 2).sum(axis=1).clip(lower=1e-10) ** 0.5
                
                bar_ic = (cov_sr / (std_sig * std_ret)).fillna(0.0)
                ic_series[aid] = bar_ic
            
            # EMA-smooth the per-bar ICs
            ema_ics = {}
            for aid in aid_list:
                ema_ics[aid] = ic_series[aid].ewm(span=rolling_ic, min_periods=max(rolling_ic // 4, 10)).mean().fillna(0.0)
            
            # Build combined signal with time-varying IC weights (VECTORIZED)
            # Stack EMA ICs into (n_bars, n_alphas) weight matrix
            ema_ic_matrix = np.column_stack([ema_ics[aid].values for aid in aid_list])  # (n_bars, n_alphas)
            # HARD ZERO: alphas with negative rolling IC get ZERO weight.
            # They are regime-inappropriate and should contribute nothing.
            ema_ic_matrix = np.maximum(ema_ic_matrix, 0.0)  # hard zero floor
            weight_sums = ema_ic_matrix.sum(axis=1, keepdims=True)
            weight_sums = np.maximum(weight_sums, 1e-10)
            ema_ic_matrix = ema_ic_matrix / weight_sums  # normalize to sum=1 per bar
            
            # Stack all alpha signals into (n_bars, n_tickers, n_alphas)
            signal_stack = np.stack([normed[aid].values for aid in aid_list], axis=2)  # (n_bars, n_tickers, n_alphas)
            
            # Weighted sum: combined[t, j] = Σ_k signal[t, j, k] * weight[t, k]
            combined_values = np.einsum('ijk,ik->ij', signal_stack, ema_ic_matrix)
            
            combined = pd.DataFrame(combined_values, index=dates, columns=tickers)
            
            # Report average rolling IC stats
            mean_ema_ics = {aid: ema_ics[aid].mean() for aid in aid_list}
            avg_ic = np.mean(list(mean_ema_ics.values()))
            n_active_final = int((ema_ic_matrix[-1] > 0).sum())
            n_active_avg = float((ema_ic_matrix > 0).mean(axis=0).sum())
            print(f"  Rolling IC: EMA span={rolling_ic} ({rolling_ic*5/60:.1f}h), "
                  f"avg IC={avg_ic:.5f}, active={n_active_final}/{n_alphas} "
                  f"(avg {n_active_avg:.1f}), computed causally", flush=True)
        else:
            # Static IC from DB — simple weighted average
            total_w = sum(ic_weights[aid] for aid in normed)
            combined = sum(normed[aid] * (ic_weights[aid] / total_w) for aid in normed)
    else:
        combined = sum(normed.values()) / len(normed)

    # Apply concordance boost
    if concordance_multiplier is not None:
        combined = combined * concordance_multiplier
        print(f"  Concordance boost: avg agreement={concordance_multiplier.mean().mean():.3f}", flush=True)

    # ── Signal smoothing: EMA on combined to reduce whipsaw ──
    if signal_smooth > 0:
        combined = combined.ewm(span=signal_smooth, min_periods=1).mean()
        print(f"  Signal smoothing: EMA span={signal_smooth} ({signal_smooth*5/60:.1f}h)", flush=True)

    # ── Beta hedge: regress out BTC returns from signal ──
    if beta_hedge:
        close = matrices.get('close')
        if close is not None:
            # Find BTC column
            btc_col = None
            for col in close.columns:
                if 'BTC' in col.upper() and 'USD' in col.upper():
                    btc_col = col
                    break
            if btc_col is not None:
                btc_ret = close[btc_col].pct_change().fillna(0.0)
                # For each ticker, remove BTC beta from signal using rolling regression
                hedged = combined.copy()
                lookback = 576  # 2 days
                for col in combined.columns:
                    if col == btc_col:
                        continue
                    sig_col = combined[col]
                    # Rolling beta: cov(signal, btc_ret) / var(btc_ret)
                    rolling_cov = sig_col.rolling(lookback, min_periods=72).cov(btc_ret)
                    rolling_var = btc_ret.rolling(lookback, min_periods=72).var()
                    beta = (rolling_cov / rolling_var.replace(0, np.nan)).fillna(0).clip(-5, 5)
                    hedged[col] = sig_col - beta * btc_ret
                combined = hedged
                print(f"  Beta hedge: removed BTC beta (lookback={lookback}b)", flush=True)

    # ── PCA signal hedge: remove top N eigenvectors of return covariance ──
    #
    # Generalizes beta-hedge from 1 factor (BTC) to N factors (PCA eigenvectors).
    # For each bar (refit every 288 bars = 1 day):
    #   1. Estimate PCA on trailing 576 bars (2 days) of cross-sectional returns
    #   2. Project the combined signal vector onto the top N PC loadings
    #   3. Subtract the projection, leaving only the idiosyncratic signal
    #
    # This removes the component of the signal that is linearly explained by
    # systematic factors (market direction, alt-vs-major rotation, vol regime).
    # The residual signal is market-neutral by construction.
    #
    # Key difference from --pca flag: --pca operates on the FINAL composite
    # (after z-scoring), while signal_hedge operates on the RAW combined signal
    # BEFORE final normalization. This is important because the z-score step
    # can re-introduce factor exposure that was removed.
    #
    if signal_hedge > 0:
        from sklearn.decomposition import PCA
        close = matrices.get('close')
        if close is not None:
            ret_df = close.pct_change().fillna(0.0)
            ret_df = ret_df.reindex(index=combined.index, columns=combined.columns).fillna(0.0)
            ret_np = ret_df.values
            comb_np = combined.values.copy()
            n_bars, n_tickers = comb_np.shape
            
            pca_lookback = 576   # 2 days of trailing returns for covariance estimation
            pca_retrain = 288    # refit PCA every 1 day (288 5-min bars)
            components = None    # (n_components, n_tickers) PC loading matrix
            
            for t in range(n_bars):
                if t < pca_lookback:
                    continue  # not enough history, leave signal untouched
                
                # Periodically re-estimate PCA from trailing returns
                if components is None or t % pca_retrain == 0:
                    ret_window = ret_np[t - pca_lookback:t]  # (lookback, n_tickers)
                    col_std = ret_window.std(axis=0)
                    valid = col_std > 1e-8  # exclude zero-variance tickers
                    n_valid = valid.sum()
                    if n_valid > signal_hedge + 1:
                        pca = PCA(n_components=min(signal_hedge, n_valid - 1))
                        pca.fit(ret_window[:, valid])
                        # Expand components back to full ticker dimension
                        components = np.zeros((pca.n_components_, n_tickers))
                        components[:, valid] = pca.components_
                    else:
                        components = None
                
                if components is None:
                    continue
                
                # Project signal onto PC space and subtract
                # signal_residual = signal - sum_k (signal . pc_k) / (pc_k . pc_k) * pc_k
                signal = comb_np[t]
                for k in range(components.shape[0]):
                    pc = components[k]
                    pc_norm = np.dot(pc, pc)
                    if pc_norm > 1e-10:
                        loading = np.dot(signal, pc) / pc_norm
                        comb_np[t] = comb_np[t] - loading * pc
            
            combined = pd.DataFrame(comb_np, index=combined.index, columns=combined.columns)
            print(f"  PCA signal hedge: removed top {signal_hedge} eigenvectors "
                  f"(lookback={pca_lookback}b, retrain={pca_retrain}b)", flush=True)

    # ── Final normalization ──
    if rank_norm:
        # Cross-sectional rank percentile: more robust to outliers
        # Transforms to uniform [0,1] then to [-3, 3] for threshold compatibility
        composite = combined.rank(axis=1, pct=True)
        # Map [0,1] percentile to approximate z-score via inverse normal CDF
        from scipy import stats
        composite = composite.clip(0.01, 0.99)  # avoid inf at edges
        composite = composite.apply(lambda x: pd.Series(stats.norm.ppf(x), index=x.index), axis=1)
        composite = composite.fillna(0.0)
        print(f"  Rank normalization: percentile -> inverse-normal z", flush=True)
    else:
        # Standard cross-sectional z-score for threshold compatibility
        mu_cs = combined.mean(axis=1)
        std_cs = combined.std(axis=1).replace(0, np.nan)
        composite = combined.sub(mu_cs, axis=0).div(std_cs, axis=0).fillna(0.0)

    n_signals = len(normed)
    eff_window = 1.0 / gamma if gamma > 0 else float('inf')
    extras = []
    if concordance_boost: extras.append("concordance")
    if signal_smooth > 0: extras.append(f"smooth={signal_smooth}")
    if beta_hedge: extras.append("beta-hedge")
    if signal_hedge > 0: extras.append(f"pca-hedge={signal_hedge}")
    if rolling_ic > 0: extras.append(f"rolling-ic={rolling_ic}")
    if rolling_return > 0: extras.append(f"rolling-ret={rolling_return}")
    if rank_norm: extras.append("rank-norm")
    extras_str = f" [{', '.join(extras)}]" if extras else ""
    print(f"  TS AutoNorm (hybrid): {n_signals} alphas ({n_prenorm} CS-normed, "
          f"{n_raw} EWM-detrended), gamma={gamma:.4f} "
          f"(eff. window~{eff_window:.0f} bars / {eff_window*5/60:.0f}h), "
          f"IC-weighted={ic_weighted}{extras_str}", flush=True)

    return composite


ML_COMBINERS = {
    "ml_ridge": compute_composite_ml,
    "ml_gbt": compute_composite_gbt,
    "ml_gbt_class": compute_composite_gbt_classifier,
    "ml_gbt_entry": compute_composite_gbt_entry,
    "concordance": compute_composite_concordance,
    "ic_rolling": compute_composite_ic_rolling,
    "ts_autonorm": compute_composite_ts_autonorm,
    "hurdle": None,       # registered below after definition
    "asym_loss": None,
    "quantile": None,
}


def compute_composite_lgbm(alphas, matrices, universe, returns,
                            target_horizon=144, train_bars=4032,
                            retrain_every=576, gamma=0.01,
                            ic_weighted=True):
    """
    LightGBM-based signal combiner with smart target transforms.

    WARNING: This combiner OVERFITS BADLY on 5m data. Val Sharpe +5.12 but
    walk-forward -2.38. Do NOT use for production. Kept for reference only.

    Root cause: Only ~14 days of training per WF fold is insufficient for
    61 features (30 raw + 30 rank + 1 vol regime). The model finds spurious
    patterns that don't generalize. Would need 6+ months of data or much
    fewer features.

    Key innovations (that didn't help OOS):
    1. Target: sign(fwd_ret) * log1p(|fwd_ret| / fee_cost)
       - Measures directional alpha in fee-multiples
       - Emphasizes direction over magnitude
       - Naturally downweights tiny moves that don't cover fees

    2. Features:
       - Raw alpha signals (after ts_autonorm normalization)
       - Cross-sectional rank of each signal (nonlinear monotonic transform)
       - Market vol regime (rolling std of equal-weight portfolio)

    3. Winsorized returns to limit outlier influence on tree splits

    4. Rolling refit every retrain_every bars on trailing train_bars

    Parameters
    ----------
    target_horizon : int
        Forward return horizon for label construction (bars). Default: 144 (12h).
    train_bars : int
        Training window size. Default: 4032 (14 days).
    retrain_every : int
        Refit frequency. Default: 576 (2 days).
    gamma : float
        EWMA decay for signal normalization.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor as lgb
        lgb = None

    # First: compute ts_autonorm signals for all alphas
    normed_signals = {}
    alpha_expressions = {}
    ic_weights = {}

    for alpha_id, expression, ic_mean, sharpe_is in alphas:
        alpha_expressions[alpha_id] = expression
        ic_weights[alpha_id] = max(abs(ic_mean), 0.0001)

    # Build alpha signal matrices using ts_autonorm logic
    close = matrices.get('close')
    dates = close.index
    tickers = close.columns
    n_bars = len(dates)
    n_tickers = len(tickers)
    K = len(alphas)

    # Build per-alpha normalized signals using same hybrid logic
    ewm_span = max(int(1.0 / gamma), 2) if gamma > 0 else 100

    for alpha_id, expression, ic_mean, sharpe_is in alphas:
        try:
            sig = evaluate_expression(expression, matrices)
            if sig is None or sig.empty:
                continue
            uni_mask = universe.reindex(index=sig.index, columns=sig.columns).fillna(False)
            sig = sig.where(uni_mask, np.nan)
            sig = sig.reindex(index=dates, columns=tickers)
        except Exception as e:
            print(f"  LGBM: Alpha #{alpha_id} failed: {e}")
            continue

        has_zscore_cs = 'zscore_cs' in expression

        if has_zscore_cs:
            # Cross-sectional z-score (same as ic_weighted)
            mu = sig.mean(axis=1)
            std = sig.std(axis=1).replace(0, np.nan)
            normed = sig.sub(mu, axis=0).div(std, axis=0).clip(-3, 3)
        else:
            # Std-only normalization (preserve directional signal)
            ewm_std = sig.ewm(span=ewm_span, min_periods=max(ewm_span//2, 5)).std()
            ewm_std = ewm_std.clip(lower=1e-8)
            normed = sig.div(ewm_std).clip(-5, 5)

        normed_signals[alpha_id] = normed.fillna(0.0).values

    if not normed_signals:
        print("  LGBM: no signals computed", flush=True)
        return None

    alpha_ids = list(normed_signals.keys())
    K = len(alpha_ids)

    # Build raw feature matrix: (n_bars, n_tickers, K)
    feat_raw = np.zeros((n_bars, n_tickers, K))
    for k, aid in enumerate(alpha_ids):
        feat_raw[:, :, k] = normed_signals[aid]

    # Build rank features: cross-sectional rank at each bar
    feat_rank = np.zeros((n_bars, n_tickers, K))
    for t in range(n_bars):
        for k in range(K):
            row = feat_raw[t, :, k]
            valid = np.isfinite(row) & (row != 0)
            if valid.sum() > 5:
                ranks = np.zeros(n_tickers)
                ranks[valid] = (np.argsort(np.argsort(row[valid])) + 1.0) / valid.sum()
                feat_rank[t, :, k] = ranks

    # Market vol regime feature: rolling std of equal-weight portfolio return
    ret_np = returns.reindex(index=dates, columns=tickers).fillna(0).values
    port_ret = np.nanmean(ret_np, axis=1)
    vol_regime = np.zeros(n_bars)
    vol_lookback = 288  # 1 day
    for t in range(vol_lookback, n_bars):
        vol_regime[t] = np.std(port_ret[t-vol_lookback:t])
    # Normalize to [0, 1]
    vol_max = np.percentile(vol_regime[vol_regime > 0], 99) if (vol_regime > 0).any() else 1
    vol_regime = np.clip(vol_regime / max(vol_max, 1e-8), 0, 1)

    # Combined features: raw signals + rank signals + vol regime = 2*K + 1
    n_features = 2 * K + 1

    # Forward returns (target) - use multi-bar forward return
    fwd_ret = np.zeros((n_bars, n_tickers))
    for t in range(n_bars - target_horizon):
        fwd_ret[t] = ret_np[t+1:t+1+target_horizon].sum(axis=0)

    # Target transform: sign(ret) * log1p(|ret| / fee_cost)
    fee_cost = FEES_BPS / 10_000.0 * 2  # round-trip
    # Winsorize returns at ±5% before transform
    fwd_clipped = np.clip(fwd_ret, -0.05, 0.05)
    target = np.sign(fwd_clipped) * np.log1p(np.abs(fwd_clipped) / max(fee_cost, 1e-6))

    # Universe mask
    uni_np = universe.reindex(index=dates, columns=tickers).fillna(False).values

    composite_np = np.zeros((n_bars, n_tickers))

    # LightGBM params — shallow trees, strong regularization
    if lgb is not None:
        lgb_params = {
            'objective': 'regression',
            'metric': 'mae',
            'learning_rate': 0.05,
            'num_leaves': 15,
            'max_depth': 4,
            'min_child_samples': 200,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 1.0,
            'reg_lambda': 1.0,
            'verbose': -1,
            'n_jobs': 1,
        }
        n_rounds = 100
    else:
        lgb_params = None

    model = None
    last_trained = -retrain_every

    for t in range(train_bars, n_bars):
        if t - last_trained >= retrain_every or model is None:
            # Build training data
            t0 = max(0, t - train_bars)

            # Flatten (bars, tickers) -> rows
            X_raw = feat_raw[t0:t].reshape(-1, K)
            X_rank = feat_rank[t0:t].reshape(-1, K)
            vol_tile = np.tile(vol_regime[t0:t].reshape(-1, 1),
                              (1, n_tickers)).reshape(-1, 1)
            X = np.hstack([X_raw, X_rank, vol_tile])
            y = target[t0:t].reshape(-1)
            uni_flat = uni_np[t0:t].reshape(-1)

            # Filter: in universe, finite target, not too close to end
            valid = uni_flat & np.isfinite(y) & (np.abs(X).sum(axis=1) > 0)
            # Don't use last target_horizon bars (target not ready)
            n_train_rows = (t - t0) * n_tickers
            horizon_cutoff = target_horizon * n_tickers
            valid_idx = np.where(valid)[0]
            valid_idx = valid_idx[valid_idx < n_train_rows - horizon_cutoff]

            if len(valid_idx) < 500:
                last_trained = t
                continue

            X_train = X[valid_idx]
            y_train = y[valid_idx]

            try:
                if lgb is not None:
                    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
                    model = lgb.train(lgb_params, dtrain, num_boost_round=n_rounds,
                                     callbacks=[lgb.log_evaluation(period=-1)])
                else:
                    from sklearn.ensemble import GradientBoostingRegressor
                    model = GradientBoostingRegressor(
                        n_estimators=50, max_depth=4, learning_rate=0.05,
                        min_samples_leaf=200, subsample=0.7)
                    model.fit(X_train, y_train)
            except Exception as e:
                print(f"  LGBM train error at t={t}: {e}", flush=True)
                last_trained = t
                continue

            last_trained = t

        # Predict at time t
        if model is None:
            continue

        x_raw = feat_raw[t]  # (n_tickers, K)
        x_rank = feat_rank[t]
        x_vol = np.full((n_tickers, 1), vol_regime[t])
        x_t = np.hstack([x_raw, x_rank, x_vol])

        try:
            if lgb is not None:
                pred = model.predict(x_t)
            else:
                pred = model.predict(x_t)
            composite_np[t] = pred
        except Exception:
            pass

    # Final cross-sectional z-score
    composite = pd.DataFrame(composite_np, index=dates, columns=tickers)
    mu = composite.mean(axis=1)
    std = composite.std(axis=1).replace(0, np.nan)
    composite = composite.sub(mu, axis=0).div(std, axis=0).fillna(0.0)

    # Report
    n_active = int((np.abs(composite_np) > 0).any(axis=1).sum())
    print(f"  LGBM combiner: {K} alphas, {K*2+1} features (raw+rank+vol), "
          f"target_h={target_horizon}, train={train_bars}b, refit={retrain_every}b, "
          f"active_bars={n_active}/{n_bars}", flush=True)

    return composite


ML_COMBINERS["lgbm"] = compute_composite_lgbm



# ============================================================================
# KAKUSHADZE "HOW TO COMBINE A BILLION ALPHAS" COMBINER
# Reference: https://arxiv.org/abs/1603.05937
# ============================================================================

def compute_composite_billion_alphas(
    alphas, matrices, universe, returns,
    lookback_days=20,            # Rolling window in DAILY alpha returns (paper's M+1)
    retrain_every=288,           # Bars between weight recomputes (1 day = 288)
    rm_overall=True,             # Step 6-7: Cross-demean + drop col (remove 'overall' mode)
    gamma=0.01,                  # ts_autonorm normalization for per-alpha signals
    signal_smooth=12,            # EMA span for final composite smoothing
):
    """
    Kakushadze & Yu (2016) regression-based alpha combination.
    'How to Combine a Billion Alphas', Sec. 5.3, Appendix A

    CRITICAL DESIGN CONSTRAINT:
      The paper requires N (number of alphas) >> M (number of time observations).
      Alpha returns MUST be aggregated to DAILY so that M ~ 20, well below N ~ 27.
      Using bar-level data (M=1440) makes M >> N, which produces a degenerate
      underdetermined regression with zero residuals and zero weights.

    Faithful Python translation of the R code from Appendix A:

      calc.opt.weights <- function(e.r, ret, y=0, s=0, rm.overall=T) {
        s <- apply(ret, 1, sd)           # sample std per alpha
        x <- ret - rowMeans(ret)         # serial demean
        y <- x / s                       # normalize
        y <- y[, -ncol(x)]              # DROP last col → [N, M]
        if(rm.overall) {
          y <- t(t(y) - colMeans(y))     # cross-sectionally demean columns
          y <- y[, -ncol(y)]            # DROP last col again → [N, M-1]
        }
        e.r <- matrix(e.r / s, ...)     # normalize expected returns
        w <- t(y) %*% e.r               # Y^T @ e_tilde
        w <- solve(t(y) %*% y) %*% w    # beta = (Y^T Y)^-1 Y^T e_tilde
        w <- e.r - y %*% w              # residuals = e_tilde - Y @ beta
        w <- w / s                       # divide by sigma
        w <- w / sum(abs(w))             # L1 normalize
      }

    Parameters
    ----------
    lookback_days : int
        Rolling window in daily alpha returns. This is the paper's M+1.
        Must be < N (number of alphas) for the regression to be overdetermined.
        With N=27 alphas, use lookback_days=20 so the regression is [27, 17] or [27, 18].
    rm_overall : bool
        If True, remove the 'overall' mode via cross-sectional demeaning (paper step 6-7).
        This removes the analog of the market factor and reduces the number of
        negative weights. The R code variable is rm.overall.
    """
    BARS_PER_DAY_LOCAL = 288

    close = matrices["close"]
    dates = close.index
    tickers = close.columns
    n_bars = len(dates)

    # ── Step 0: Build per-alpha normalized signals ──
    print(f"  Billion Alphas: computing per-alpha signals ({len(alphas)} alphas)...", flush=True)
    alpha_signals = []   # list of DataFrames [n_bars, n_tickers]
    alpha_ids = []
    for alpha_id, expr, ic, sr in alphas:
        try:
            raw = evaluate_expression(expr, matrices)
            if raw is None or raw.shape != close.shape:
                continue
            # ts_autonorm normalization (same as production)
            if "zscore_cs" in expr:
                z = raw.sub(raw.mean(axis=1), axis=0)
                cs_std = raw.std(axis=1)
                z = z.div(cs_std.replace(0, np.nan), axis=0).clip(-3, 3)
            else:
                ewm_std = raw.ewm(alpha=gamma, min_periods=20).std()
                z = raw.div(ewm_std.replace(0, np.nan)).clip(-5, 5)
            z = z.ffill(limit=6)
            alpha_signals.append(z)
            alpha_ids.append(alpha_id)
        except Exception:
            continue

    if len(alpha_signals) < 3:
        print("  Billion Alphas: insufficient valid alphas, falling back to ts_autonorm", flush=True)
        return compute_composite_ts_autonorm(alphas, matrices, universe, returns,
                                             gamma=gamma, ic_weighted=True,
                                             signal_smooth=signal_smooth)

    N = len(alpha_signals)  # number of alphas

    # ── Compute per-alpha bar-level returns ──
    ret_df = returns.reindex(index=dates, columns=tickers)
    alpha_bar_ret = np.zeros((N, n_bars), dtype=np.float64)
    for i, z in enumerate(alpha_signals):
        z_lag = z.shift(1)  # causal
        uni_mask = universe.reindex_like(z_lag).fillna(False)
        z_masked = z_lag.where(uni_mask)
        r_masked = ret_df.where(uni_mask)
        cs_std = z_masked.std(axis=1).replace(0, np.nan)
        z_norm = z_masked.sub(z_masked.mean(axis=1), axis=0).div(cs_std, axis=0)
        alpha_bar_ret[i] = (z_norm * r_masked).sum(axis=1).fillna(0).values

    # ── Aggregate to DAILY alpha returns (sum of 288 bars per day) ──
    # This is critical: paper requires N >> M. With daily aggregation,
    # M ~ 20 days << N ~ 27 alphas → overdetermined regression.
    n_days = n_bars // BARS_PER_DAY_LOCAL
    alpha_daily_ret = np.zeros((N, n_days), dtype=np.float64)
    for d in range(n_days):
        s = d * BARS_PER_DAY_LOCAL
        e = s + BARS_PER_DAY_LOCAL
        alpha_daily_ret[:, d] = alpha_bar_ret[:, s:e].sum(axis=1)

    print(f"  Billion Alphas: {N} alphas, {n_days} daily obs, "
          f"lookback={lookback_days}d, retrain={retrain_every}bars",
          flush=True)

    if lookback_days >= N:
        print(f"  WARNING: lookback_days ({lookback_days}) >= N ({N}). "
              f"Regression will be underdetermined. Reducing to {N-3}.", flush=True)
        lookback_days = max(N - 3, 5)

    # ── Kakushadze regression procedure (Sec. 5.3, Appendix A R code) ──
    weights_ts = np.full((n_bars, N), fill_value=1.0 / N)
    last_refit = -retrain_every
    current_weights = np.full(N, 1.0 / N)
    min_lookback_bars = lookback_days * BARS_PER_DAY_LOCAL

    for t in range(min_lookback_bars, n_bars):
        if t - last_refit >= retrain_every:
            # Map bar index to day index
            day_idx = t // BARS_PER_DAY_LOCAL
            if day_idx < lookback_days:
                weights_ts[t] = current_weights
                continue

            # ret is [N, M+1] where M+1 = lookback_days (paper convention)
            M_plus_1 = lookback_days
            ret_window = alpha_daily_ret[:, day_idx - M_plus_1: day_idx]  # [N, M+1]

            if ret_window.shape[1] < 4:
                weights_ts[t] = current_weights
                continue

            try:
                # ── R code: s <- apply(ret, 1, sd) ──
                # R's sd uses ddof=1 (sample std). With M+1 observations, denominator = M.
                s = ret_window.std(axis=1, ddof=1)  # [N] — sample std per alpha
                s = np.where(s < 1e-15, 1e-15, s)

                # ── R code: x <- ret - rowMeans(ret) ──
                x = ret_window - ret_window.mean(axis=1, keepdims=True)  # [N, M+1] serial demean

                # ── R code: y <- x / s ──
                y = x / s[:, None]  # [N, M+1] normalized demeaned returns

                # ── R code: y <- y[, -ncol(x)] ──
                # Drop LAST column → [N, M]  (paper step 5: keep first M columns)
                y = y[:, :-1]       # [N, M] where M = M_plus_1 - 1

                # ── R code: if(rm.overall) { ... } ──
                if rm_overall:
                    # ── y <- t(t(y) - colMeans(y)) ──
                    # Cross-sectionally demean each column (subtract mean over alphas)
                    y = y - y.mean(axis=0, keepdims=True)  # [N, M]

                    # ── y <- y[, -ncol(y)] ──
                    # Drop LAST column AGAIN → [N, M-1]
                    y = y[:, :-1]   # [N, M-1]

                K = y.shape[1]  # number of risk factor columns

                if K < 2 or K >= N:
                    # Degenerate: too few factors or underdetermined
                    weights_ts[t] = current_weights
                    last_refit = t
                    continue

                # ── R code: e.r <- matrix(e.r / s, length(e.r), 1) ──
                # Expected returns: mean daily return over the lookback window
                E = ret_window.mean(axis=1)  # [N] — simple trailing mean
                E_tilde = (E / s).reshape(-1, 1)  # [N, 1] — normalized

                # ── R code: w <- t(y) %*% e.r ──
                # ── w <- solve(t(y) %*% y) %*% w ──
                # beta = (Y^T Y)^-1 @ Y^T @ e_tilde
                YtY = y.T @ y          # [K, K]
                YtE = y.T @ E_tilde    # [K, 1]
                beta = np.linalg.solve(YtY, YtE)  # [K, 1]

                # ── R code: w <- e.r - y %*% w ──
                # residuals = E_tilde - Y @ beta
                eps = E_tilde - y @ beta  # [N, 1]

                # ── R code: w <- w / s ──
                w = (eps.ravel()) / s    # [N]

                # ── R code: w <- w / sum(abs(w)) ──
                l1 = np.abs(w).sum()
                if l1 > 1e-15:
                    w = w / l1
                else:
                    w = np.full(N, 1.0 / N)

                current_weights = w

            except np.linalg.LinAlgError:
                pass  # singular matrix — keep previous weights
            except Exception as ex:
                print(f"  Billion Alphas: refit error at bar {t}: {ex}", flush=True)

            last_refit = t

        weights_ts[t] = current_weights

    # ── Apply scalar weights to per-alpha signals to get composite ──
    print("  Billion Alphas: assembling composite...", flush=True)
    composite_df = pd.DataFrame(0.0, index=dates, columns=tickers)
    w_ts = pd.DataFrame(weights_ts, index=dates)

    for i, z in enumerate(alpha_signals):
        wi = w_ts.iloc[:, i]
        composite_df = composite_df.add(z.mul(wi, axis=0))

    # Final cross-sectional z-score
    cs_mean = composite_df.mean(axis=1)
    cs_std = composite_df.std(axis=1).replace(0, np.nan)
    composite_df = composite_df.sub(cs_mean, axis=0).div(cs_std, axis=0)

    # Optional smoothing
    if signal_smooth > 0:
        composite_df = composite_df.ewm(span=signal_smooth, min_periods=1, axis=0).mean()

    # Print weight stats
    final_w = weights_ts[-1]
    sorted_idx = np.argsort(np.abs(final_w))[::-1]
    print(f"  Billion Alphas: final weights (top 5 by |w|):", flush=True)
    for rank_i in sorted_idx[:5]:
        aid = alpha_ids[rank_i]
        print(f"    alpha #{aid:2d}: w={final_w[rank_i]:+.4f}", flush=True)
    n_pos = (final_w > 0.001).sum()
    n_neg = (final_w < -0.001).sum()
    n_zero = N - n_pos - n_neg
    print(f"  Billion Alphas: pos={n_pos} neg={n_neg} ~zero={n_zero} "
          f"|w| sum={np.abs(final_w).sum():.4f}", flush=True)

    return composite_df


ML_COMBINERS["billion_alphas"] = compute_composite_billion_alphas


def compute_composite_hurdle(alphas, matrices, universe, returns,
                              fee_bps=7, target_horizon=144,
                              train_bars=1728, retrain_every=288,
                              precision_threshold=0.70):
    """
    Two-Stage Hurdle Model for takeout strategy.

    Stage 1 (Classifier): GBT predicts P(|fwd_ret_144| > 2*fee_cost).
                          Only pass entries with P >= precision_threshold.
    Stage 2 (Regressor):  Trained only on large-move events. Predicts
                          direction and magnitude in the tail.

    This separates "will there be a large move?" from "which direction?",
    preventing the mass of tiny returns from dominating the regression.
    """
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    import warnings
    warnings.filterwarnings('ignore')

    fee_cost = fee_bps / 10_000.0
    hurdle = 2 * fee_cost  # ~0.14% round-trip threshold

    # Evaluate all alphas
    signals = {}
    for alpha_id, expression, ic_mean, sharpe_is in alphas:
        try:
            raw = evaluate_expression(expression, matrices)
            if raw is None or raw.empty:
                continue
            uni_mask = universe.reindex(index=raw.index, columns=raw.columns).fillna(False)
            signals[alpha_id] = raw.where(uni_mask, np.nan)
        except Exception as e:
            print(f"  Alpha #{alpha_id} failed: {e}")

    if not signals:
        return None

    # Z-score each alpha
    zscored = {}
    for aid, sig in signals.items():
        mu = sig.mean(axis=1)
        std = sig.std(axis=1).replace(0, np.nan)
        zscored[aid] = sig.sub(mu, axis=0).div(std, axis=0).clip(-5, 5)

    ref_aid = list(zscored.keys())[0]
    dates = zscored[ref_aid].index
    tickers = zscored[ref_aid].columns
    n_bars, n_tickers = len(dates), len(tickers)
    K = len(zscored)
    alpha_ids = list(zscored.keys())

    # Build feature matrix: (n_bars, n_tickers, K) -> flattened for sklearn
    feat_3d = np.zeros((n_bars, n_tickers, K))
    for k, aid in enumerate(alpha_ids):
        feat_3d[:, :, k] = np.nan_to_num(zscored[aid].values, nan=0.0)

    # Build target: raw forward return, binary hurdle, signed return
    close = matrices.get('close')
    ret_np = close.pct_change().reindex(index=dates, columns=tickers).fillna(0).values

    fwd_ret = np.zeros((n_bars, n_tickers))
    for t in range(n_bars - target_horizon):
        fwd_ret[t] = ret_np[t+1:t+1+target_horizon].sum(axis=0)

    # Binary: |return| > hurdle (any large move, long or short)
    fwd_abs_large = (np.abs(fwd_ret) > hurdle).astype(float)
    # Signed: actual return (for regression, only used on large-move subset)
    fwd_signed = fwd_ret

    composite_np = np.zeros((n_bars, n_tickers))
    clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1)
    reg = GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1)
    last_trained = -retrain_every

    for t in range(train_bars, n_bars, 1):
        if t - last_trained >= retrain_every or t == train_bars:
            t0 = max(0, t - train_bars)
            X = feat_3d[t0:t].reshape(-1, K)
            y_bin = fwd_abs_large[t0:t].reshape(-1)
            y_ret = fwd_signed[t0:t].reshape(-1)

            valid = np.isfinite(y_bin) & (np.abs(X).sum(axis=1) > 0)
            X_v, y_bin_v, y_ret_v = X[valid], y_bin[valid], y_ret[valid]

            if y_bin_v.sum() < 50 or (y_bin_v == 0).sum() < 50:
                last_trained = t
                continue

            # Stage 1: classify large moves
            try:
                clf.fit(X_v, y_bin_v)
            except Exception:
                last_trained = t
                continue

            # Stage 2: regress on large-move subset only
            large_mask = y_bin_v == 1
            if large_mask.sum() >= 30:
                try:
                    reg.fit(X_v[large_mask], y_ret_v[large_mask])
                    reg_fitted = True
                except Exception:
                    reg_fitted = False
            else:
                reg_fitted = False

            last_trained = t

        # Predict at time t
        x_t = feat_3d[t].copy()  # (n_tickers, K)
        try:
            # Stage 1: P(large move)
            proba = clf.predict_proba(x_t)[:, 1]  # P(|ret| > hurdle)
            # Stage 2: direction prediction (only if high confidence)
            if reg_fitted:
                direction = reg.predict(x_t)
            else:
                direction = np.zeros(n_tickers)
            # Composite: P(large move) * direction sign (gated by threshold)
            gate = (proba >= precision_threshold).astype(float)
            composite_np[t] = gate * direction
        except Exception:
            pass

    composite = pd.DataFrame(composite_np, index=dates, columns=tickers)
    mu = composite.mean(axis=1)
    std = composite.std(axis=1).replace(0, np.nan)
    composite = composite.sub(mu, axis=0).div(std, axis=0)

    n_gated = int((np.abs(composite_np) > 0).sum())
    pct_large = float(fwd_abs_large.mean()) * 100
    print(f"  Hurdle: K={K}, hurdle={hurdle*100:.2f}%, P_threshold={precision_threshold}, "
          f"large_events={pct_large:.1f}%, gated_cells={n_gated}", flush=True)

    return composite


def compute_composite_quantile(alphas, matrices, universe, returns,
                                target_horizon=144, train_bars=1728,
                                retrain_every=288, tau=0.95):
    """
    Quantile Regression combiner (tau=0.95 by default).

    Uses the pinball loss to predict the conditional 95th percentile of
    forward returns. Focuses on the upper tail, robust to outliers,
    ignores the mass of small returns by design.

    For shorts: also run at tau=0.05 (lower tail), combine directionally.
    """
    from sklearn.linear_model import QuantileRegressor

    signals = {}
    for alpha_id, expression, ic_mean, sharpe_is in alphas:
        try:
            raw = evaluate_expression(expression, matrices)
            if raw is None or raw.empty:
                continue
            uni_mask = universe.reindex(index=raw.index, columns=raw.columns).fillna(False)
            signals[alpha_id] = raw.where(uni_mask, np.nan)
        except Exception as e:
            print(f"  Alpha #{alpha_id} failed: {e}")

    if not signals:
        return None

    zscored = {}
    for aid, sig in signals.items():
        mu = sig.mean(axis=1)
        std = sig.std(axis=1).replace(0, np.nan)
        zscored[aid] = sig.sub(mu, axis=0).div(std, axis=0).clip(-5, 5)

    ref_aid = list(zscored.keys())[0]
    dates = zscored[ref_aid].index
    tickers = zscored[ref_aid].columns
    n_bars, n_tickers = len(dates), len(tickers)
    K = len(zscored)
    alpha_ids = list(zscored.keys())

    feat_3d = np.zeros((n_bars, n_tickers, K))
    for k, aid in enumerate(alpha_ids):
        feat_3d[:, :, k] = np.nan_to_num(zscored[aid].values, nan=0.0)

    close = matrices.get('close')
    ret_np = close.pct_change().reindex(index=dates, columns=tickers).fillna(0).values
    fwd_ret = np.zeros((n_bars, n_tickers))
    for t in range(n_bars - target_horizon):
        fwd_ret[t] = ret_np[t+1:t+1+target_horizon].sum(axis=0)

    # Cross-sectional excess return in percent
    fwd_pct = fwd_ret * 100.0
    for t in range(n_bars):
        med = np.nanmedian(fwd_pct[t])
        fwd_pct[t] -= med

    # Use LightGBM quantile objective — much faster than sklearn LP solver
    try:
        import lightgbm as lgb
        HAS_LGB = True
    except ImportError:
        HAS_LGB = False

    composite_np = np.zeros((n_bars, n_tickers))
    model_high = None
    model_low = None
    last_trained = -retrain_every

    lgb_params_high = {
        'objective': 'quantile', 'alpha': tau,
        'num_leaves': 31, 'learning_rate': 0.05,
        'n_estimators': 100, 'verbose': -1,
        'min_child_samples': 50,
    }
    lgb_params_low = {
        'objective': 'quantile', 'alpha': 1 - tau,
        'num_leaves': 31, 'learning_rate': 0.05,
        'n_estimators': 100, 'verbose': -1,
        'min_child_samples': 50,
    }

    for t in range(train_bars, n_bars, 1):
        if t - last_trained >= retrain_every or t == train_bars:
            t0 = max(0, t - train_bars)
            X = feat_3d[t0:t].reshape(-1, K)
            y = fwd_pct[t0:t].reshape(-1)
            valid = np.isfinite(y) & (np.abs(X).sum(axis=1) > 0)
            if valid.sum() < 200:
                last_trained = t
                continue
            try:
                if HAS_LGB:
                    model_high = lgb.LGBMRegressor(**lgb_params_high)
                    model_high.fit(X[valid], y[valid])
                    model_low = lgb.LGBMRegressor(**lgb_params_low)
                    model_low.fit(X[valid], y[valid])
                else:
                    from sklearn.linear_model import QuantileRegressor
                    # subsample for speed
                    idx = np.random.choice(valid.sum(), min(5000, valid.sum()), replace=False)
                    model_high = QuantileRegressor(quantile=tau, alpha=0.1, solver='highs')
                    model_high.fit(X[valid][idx], y[valid][idx])
                    model_low = QuantileRegressor(quantile=1-tau, alpha=0.1, solver='highs')
                    model_low.fit(X[valid][idx], y[valid][idx])
            except Exception:
                last_trained = t
                continue
            last_trained = t

        if model_high is not None:
            x_t = feat_3d[t]
            try:
                pred_high = model_high.predict(x_t)
                pred_low = model_low.predict(x_t)
                composite_np[t] = pred_high + pred_low
            except Exception:
                pass

    composite = pd.DataFrame(composite_np, index=dates, columns=tickers)
    mu = composite.mean(axis=1)
    std = composite.std(axis=1).replace(0, np.nan)
    composite = composite.sub(mu, axis=0).div(std, axis=0)

    print(f"  Quantile (LGB): K={K}, tau={tau}, target_horizon={target_horizon}, "
          f"lgb={'yes' if HAS_LGB else 'sklearn fallback'}", flush=True)
    return composite


def compute_composite_asym_loss(alphas, matrices, universe, returns,
                                 target_horizon=144, train_bars=1728,
                                 retrain_every=288, fp_penalty=5.0):
    """
    Asymmetric Loss GBM combiner.

    Custom loss function that heavily penalizes false positives (predicted
    large move, actual small move) vs false negatives (missed large move).
    Implemented via LightGBM custom objective.

    fp_penalty: multiplier on FP loss vs FN loss.
    """
    try:
        import lightgbm as lgb
        HAS_LGB = True
    except ImportError:
        HAS_LGB = False
        print("  WARNING: lightgbm not installed, falling back to GBT with sample weights")

    signals = {}
    for alpha_id, expression, ic_mean, sharpe_is in alphas:
        try:
            raw = evaluate_expression(expression, matrices)
            if raw is None or raw.empty:
                continue
            uni_mask = universe.reindex(index=raw.index, columns=raw.columns).fillna(False)
            signals[alpha_id] = raw.where(uni_mask, np.nan)
        except Exception as e:
            print(f"  Alpha #{alpha_id} failed: {e}")

    if not signals:
        return None

    zscored = {}
    for aid, sig in signals.items():
        mu = sig.mean(axis=1)
        std = sig.std(axis=1).replace(0, np.nan)
        zscored[aid] = sig.sub(mu, axis=0).div(std, axis=0).clip(-5, 5)

    ref_aid = list(zscored.keys())[0]
    dates = zscored[ref_aid].index
    tickers = zscored[ref_aid].columns
    n_bars, n_tickers = len(dates), len(tickers)
    K = len(zscored)
    alpha_ids = list(zscored.keys())

    feat_3d = np.zeros((n_bars, n_tickers, K))
    for k, aid in enumerate(alpha_ids):
        feat_3d[:, :, k] = np.nan_to_num(zscored[aid].values, nan=0.0)

    close = matrices.get('close')
    ret_np = close.pct_change().reindex(index=dates, columns=tickers).fillna(0).values
    fwd_ret = np.zeros((n_bars, n_tickers))
    for t in range(n_bars - target_horizon):
        fwd_ret[t] = ret_np[t+1:t+1+target_horizon].sum(axis=0)

    fwd_pct = fwd_ret * 100.0
    for t in range(n_bars):
        med = np.nanmedian(fwd_pct[t])
        fwd_pct[t] -= med

    # Use LGBMRegressor with sample weights — simpler and more reliable than custom fobj.
    # Sample weight logic: samples where actual negative return (FP risk) get fp_penalty weight.
    # Samples where actual positive (desired signal) get weight 1.0.
    # Additionally weight large-magnitude returns more to focus on tails.
    composite_np = np.zeros((n_bars, n_tickers))
    model = None
    last_trained = -retrain_every

    lgb_params = {
        'objective': 'regression', 'num_leaves': 31,
        'learning_rate': 0.05, 'n_estimators': 100,
        'verbose': -1, 'min_child_samples': 50,
    }

    for t in range(train_bars, n_bars, 1):
        if t - last_trained >= retrain_every or t == train_bars:
            t0 = max(0, t - train_bars)
            X = feat_3d[t0:t].reshape(-1, K)
            y = fwd_pct[t0:t].reshape(-1)
            valid = np.isfinite(y) & (np.abs(X).sum(axis=1) > 0)
            if valid.sum() < 200:
                last_trained = t
                continue

            Xv, yv = X[valid], y[valid]
            # Asymmetric sample weights:
            # - negative return samples (potential false positives): upweight by fp_penalty
            # - positive return samples: weight by magnitude (focus on large moves)
            mag = np.abs(yv)
            mag_weight = 1.0 + mag / (np.percentile(mag, 75) + 1e-8)
            direction_weight = np.where(yv < 0, fp_penalty, 1.0)
            weights = mag_weight * direction_weight
            weights /= weights.mean()  # normalize

            try:
                if HAS_LGB:
                    model = lgb.LGBMRegressor(**lgb_params)
                    model.fit(Xv, yv, sample_weight=weights)
                else:
                    from sklearn.ensemble import GradientBoostingRegressor
                    model = GradientBoostingRegressor(n_estimators=50, max_depth=3)
                    model.fit(Xv, yv, sample_weight=weights)
            except Exception:
                model = None
            last_trained = t

        if model is not None:
            x_t = feat_3d[t]
            try:
                composite_np[t] = model.predict(x_t)
            except Exception:
                pass

    composite = pd.DataFrame(composite_np, index=dates, columns=tickers)
    mu = composite.mean(axis=1)
    std = composite.std(axis=1).replace(0, np.nan)
    composite = composite.sub(mu, axis=0).div(std, axis=0)

    print(f"  Asym Loss: K={K}, fp_penalty={fp_penalty}, target_horizon={target_horizon}, "
          f"lgb={'yes' if HAS_LGB else 'no (sklearn fallback)'}", flush=True)
    return composite


# Register new combiners
ML_COMBINERS["hurdle"] = compute_composite_hurdle
ML_COMBINERS["asym_loss"] = compute_composite_asym_loss
ML_COMBINERS["quantile"] = compute_composite_quantile

# Global ML config (set from CLI)
_ML_TARGET_HORIZON = 36
_ML_ENRICHED = False


# ============================================================================
# PCA MARKET NEUTRALIZATION
# ============================================================================

def pca_neutralize_signal(composite, returns_df, n_components=3, lookback=576):
    """
    Remove the top N principal components of the return matrix from the
    composite signal. This makes the portfolio approximately market-neutral
    by hedging out the dominant correlated factors (BTC, sector, momentum).
    
    For each bar:
      1. Compute PCA on trailing `lookback` bars of returns (cross-section)
      2. Project the composite signal onto the PC space
      3. Subtract the projection, leaving only idiosyncratic signal
    
    Args:
        composite: (n_bars, n_tickers) signal DataFrame
        returns_df: (n_bars, n_tickers) returns DataFrame
        n_components: number of PCs to remove (default 3)
        lookback: rolling window for PCA estimation (default 576 = 2 days)
    
    Returns:
        neutralized composite DataFrame
    """
    from sklearn.decomposition import PCA
    
    dates = composite.index
    tickers = composite.columns
    ret_np = returns_df.reindex(index=dates, columns=tickers).fillna(0).values
    comp_np = composite.values.copy()
    
    n_bars = len(dates)
    neutralized = np.zeros_like(comp_np)
    
    # Retrain PCA every 288 bars (1 day) for efficiency
    retrain_every = 288
    pca = None
    components = None
    
    for t in range(n_bars):
        if t < lookback:
            # Not enough data — pass through raw signal
            neutralized[t] = comp_np[t]
            continue
        
        # Retrain PCA periodically
        if pca is None or t % retrain_every == 0:
            ret_window = ret_np[t-lookback:t]  # (lookback, n_tickers)
            # Remove tickers with zero variance
            col_std = ret_window.std(axis=0)
            valid_cols = col_std > 1e-8
            if valid_cols.sum() < n_components + 1:
                neutralized[t] = comp_np[t]
                continue
            
            pca = PCA(n_components=min(n_components, valid_cols.sum() - 1))
            pca.fit(ret_window[:, valid_cols])
            # Build full-size components (n_components, n_tickers)
            components = np.zeros((pca.n_components_, len(tickers)))
            components[:, valid_cols] = pca.components_
        
        if components is None:
            neutralized[t] = comp_np[t]
            continue
        
        # Project signal onto PC space and subtract
        signal = comp_np[t]  # (n_tickers,)
        projection = np.zeros_like(signal)
        for k in range(components.shape[0]):
            pc = components[k]
            pc_norm = np.dot(pc, pc)
            if pc_norm > 1e-10:
                projection += np.dot(signal, pc) / pc_norm * pc
        
        neutralized[t] = signal - projection
    
    result = pd.DataFrame(neutralized, index=dates, columns=tickers)
    
    # Re-standardize cross-sectionally
    mu = result.mean(axis=1)
    std = result.std(axis=1).replace(0, np.nan)
    result = result.sub(mu, axis=0).div(std, axis=0)
    
    print(f"  PCA neutralization: removed top {n_components} PCs (lookback={lookback} bars)", flush=True)
    return result

# ============================================================================
# PCA RISK MODEL OPTIMIZER
# ============================================================================

class PcaRiskModel:
    """
    Barra-style PCA risk model for portfolio hedging.
    
    Estimates factor structure from trailing returns via PCA, then adjusts
    portfolio weights to minimize factor exposure while preserving alpha signal.
    
    NO LOOKAHEAD: PCA is estimated from [t-lookback, t) only.
    
    The optimization solves:
        min_w  ||w - w_alpha||^2 + hedge_strength * w^T B Sigma_f B^T w
    
    Where:
        w_alpha = alpha-driven target weights (from signal)
        B = factor loadings (PCA components transposed)
        Sigma_f = factor covariance matrix
        hedge_strength = controls risk penalty vs alpha fidelity
    
    This is equivalent to: keep weights close to alpha targets, but penalize
    factor risk heavily. The analytical solution is a ridge-like shrinkage
    toward the factor-neutral subspace.
    """
    
    def __init__(self, n_components=3, lookback=576, retrain_every=288,
                 hedge_strength=1.0):
        self.n_components = n_components
        self.lookback = lookback
        self.retrain_every = retrain_every
        self.hedge_strength = hedge_strength
        
        # State
        self.factor_loadings = None  # (n_tickers, n_components)
        self.factor_cov = None       # (n_components, n_components)
        self.risk_matrix = None      # B @ Sigma_f @ B^T (n_tickers, n_tickers)
        self.last_trained = -1
    
    def update(self, t, returns_np, n_tickers):
        """
        Re-estimate PCA factors from trailing returns.
        
        STRICTLY CAUSAL: uses returns_np[t-lookback:t] only.
        """
        if t < self.lookback:
            return False
        
        if self.last_trained >= 0 and (t - self.last_trained) < self.retrain_every:
            return self.factor_loadings is not None
        
        from sklearn.decomposition import PCA
        
        ret_window = returns_np[t - self.lookback:t]  # (lookback, n_tickers)
        
        # Remove zero-variance columns
        col_std = ret_window.std(axis=0)
        valid_cols = col_std > 1e-8
        n_valid = valid_cols.sum()
        
        if n_valid < self.n_components + 2:
            return False
        
        n_comp = min(self.n_components, n_valid - 1)
        pca = PCA(n_components=n_comp)
        
        # Standardize returns before PCA (z-score each ticker)
        # This makes eigenvalues O(1) instead of O(1e-8) for small crypto returns
        ret_std = ret_window[:, valid_cols].copy()
        col_means = ret_std.mean(axis=0)
        col_stds = ret_std.std(axis=0)
        col_stds[col_stds < 1e-10] = 1.0
        ret_std = (ret_std - col_means) / col_stds
        
        pca.fit(ret_std)
        
        # Factor loadings on standardized scale: B (n_tickers, n_components)
        B_valid = pca.components_.T  # (n_valid, n_components)
        B = np.zeros((n_tickers, n_comp))
        B[valid_cols] = B_valid
        
        # Use explained variance ratios (sum to 1.0) for well-scaled risk matrix
        # Each component's "risk" is proportional to variance fraction explained
        var_ratios = pca.explained_variance_ratio_
        Sigma_f = np.diag(var_ratios)
        
        # Risk matrix: B @ Sigma_f @ B^T — now O(1) scale
        self.factor_loadings = B
        self.factor_cov = Sigma_f
        self.risk_matrix = B @ Sigma_f @ B.T
        self.last_trained = t
        
        return True
    
    def hedge_weights(self, w_alpha, positions):
        """
        Adjust weights to reduce factor exposure while staying close to alpha targets.
        
        Analytical solution to:
            min_w  ||w - w_alpha||^2 + lambda * w^T R w
        
        where R = B @ Sigma_f @ B^T (factor risk matrix)
        
        Solution: w* = (I + lambda * R)^(-1) @ w_alpha
        
        This shrinks weights toward the factor-neutral subspace.
        Only adjusts weights for active positions (positions != 0).
        """
        if self.risk_matrix is None or self.hedge_strength <= 0:
            return w_alpha
        
        active = positions != 0
        n_active = active.sum()
        if n_active < 3:
            return w_alpha
        
        n = len(w_alpha)
        R = self.risk_matrix
        lam = self.hedge_strength
        
        # Solve (I + lam*R) @ w = w_alpha
        # Only for active positions — inactive stay at 0
        R_active = R[np.ix_(active, active)]
        w_a = w_alpha[active]
        
        try:
            A = np.eye(n_active) + lam * R_active
            w_hedged_active = np.linalg.solve(A, w_a)
        except np.linalg.LinAlgError:
            return w_alpha
        
        # Preserve sign from positions (don't flip long to short)
        pos_active = positions[active]
        for i in range(n_active):
            if pos_active[i] > 0 and w_hedged_active[i] < 0:
                w_hedged_active[i] = 0
            elif pos_active[i] < 0 and w_hedged_active[i] < 0:
                pass  # short weight is fine
            elif pos_active[i] < 0 and w_hedged_active[i] > 0:
                w_hedged_active[i] = 0
        
        # Re-normalize to sum to 1
        total = np.abs(w_hedged_active).sum()
        if total > 1e-10:
            w_hedged_active = w_hedged_active / total
        
        w_out = np.zeros(n)
        w_out[active] = np.abs(w_hedged_active)  # weights are always positive, sign from positions
        return w_out



def run_threshold_strategy(composite, returns_df, universe_df,
                           entry_threshold=ENTRY_THRESHOLD,
                           exit_threshold=EXIT_THRESHOLD,
                           min_hold_bars=MIN_HOLD_BARS,
                           fees_bps=FEES_BPS,
                           booksize=BOOKSIZE,
                           max_weight=MAX_WEIGHT,
                           max_positions=0,
                           sizing_mode="conviction",
                           vol_target=True,
                           pca_hedge=0.0):
    """
    Threshold-based entry/exit strategy with conviction sizing.
    
    This is the core backtest engine. For each symbol at each 5-minute bar:
    
    ENTRY LOGIC:
      - If flat & composite > entry_threshold  → enter LONG (buy)
      - If flat & composite < -entry_threshold → enter SHORT (sell)
    
    EXIT LOGIC:
      - If long  & composite < exit_threshold  → flatten (sell)
      - If short & composite > -exit_threshold → flatten (buy)
    
    EMERGENCY EXIT (overrides min hold):
      - If long  & composite < -exit_threshold → force exit (signal fully reversed)
      - If short & composite > exit_threshold  → force exit
      This prevents holding losing positions through the min-hold timer
      when the signal has completely reversed direction.
    
    POSITION SIZING (conviction mode):
      Weight each position proportional to |z-score| / sum(|z-scores|).
      Stronger signals get more capital. This naturally concentrates
      risk in the highest-conviction trades.
    
    POSITION LIMITS:
      When max_positions > 0 and more symbols qualify, keep only the
      top max_positions by |signal strength|. This reduces fee drag
      from marginal-conviction trades.
    
    RISK CONTROLS:
      - Vol targeting: Scales portfolio to target annualized volatility.
      - Drawdown brake: Reduces position sizes during drawdowns.
      - PCA hedge: Optionally neutralizes factor exposure via PCA risk model.
    
    Parameters
    ----------
    composite : pd.DataFrame
        Cross-sectional z-scored signal matrix (bars × tickers). Values > entry
        trigger longs, < -entry trigger shorts.
    returns_df : pd.DataFrame
        Per-bar returns matrix (bars × tickers). Used for PnL calculation.
    universe_df : pd.DataFrame
        Boolean mask (bars × tickers) indicating which tickers are tradeable.
    entry_threshold : float
        Z-score threshold to trigger entry. Higher = more selective.
        Best tested: 1.2
    exit_threshold : float
        Z-score threshold to trigger exit. Lower = tighter stops.
        Best tested: 0.3
    min_hold_bars : int
        Minimum bars to hold a position before exit (except emergency).
        Best tested: 36 (3 hours at 5-min bars).
    fees_bps : float
        Round-trip fee cost in basis points per side. Default: 7.
    max_positions : int
        Maximum concurrent positions. 0 = unlimited. Best tested: 50.
    sizing_mode : str
        'conviction' = weight by |signal|, 'equal' = 1/N.
    vol_target : bool
        Enable volatility targeting (default: True).
    pca_hedge : float
        PCA risk hedge strength. 0 = off, 1.0 = moderate. Default: 0.
    
    Returns
    -------
    dict with keys:
        'sharpe', 'return_net', 'return_gross', 'max_dd', 'ic', 'n_trades',
        'avg_hold_hours', 'avg_positions', 'bars', 'pct_invested'
    """
    dates = composite.index
    tickers = composite.columns
    n_bars = len(dates)
    n_tickers = len(tickers)
    
    fee_rate = fees_bps / 10_000.0
    
    # State tracking
    positions = np.zeros((n_bars, n_tickers))  # +1 long, -1 short, 0 flat
    hold_timer = np.zeros(n_tickers, dtype=int)  # bars since last entry
    
    # Universe mask (signal universe — all tickers)
    uni_np = universe_df.reindex(index=dates, columns=tickers).fillna(False).values
    comp_np = composite.values
    ret_np = returns_df.reindex(index=dates, columns=tickers).fillna(0).values

    # Trading universe mask: if TOP_LIQUID was set, restrict ENTRIES to liquid tickers.
    # Signals are still computed on the full signal universe (better cross-sectional z-scores).
    # Existing positions in illiquid tickers are allowed to exit normally.
    if _TRADING_TICKERS is not None:
        tradeable_cols = np.array([col in _TRADING_TICKERS for col in tickers], dtype=bool)
        # uni_np is True only where ticker is in signal universe AND in trading universe
        # We only restrict entries — exits from any ticker are still allowed
        trading_np = uni_np & tradeable_cols[np.newaxis, :]   # shape: (n_bars, n_tickers)
        n_trading = tradeable_cols.sum()

    else:
        trading_np = uni_np   # No restriction — trade any universe ticker


    # Replace NaN composites with 0
    comp_np = np.nan_to_num(comp_np, nan=0.0)

    pnl_bars = np.zeros(n_bars)
    fee_bars = np.zeros(n_bars)
    n_trades = 0
    n_longs = 0
    n_shorts = 0
    
    prev_pos = np.zeros(n_tickers)
    
    # PCA risk model
    risk_model = None
    if pca_hedge > 0:
        risk_model = PcaRiskModel(n_components=3, lookback=576,
                                  retrain_every=288, hedge_strength=pca_hedge)
    
    # Risk management state
    vol_scalar_history = []  # track for reporting
    peak_equity = 0.0
    cum_net = 0.0
    vol_lookback = 288  # 1 day
    
    for t in range(1, n_bars):
        # Decay hold timers
        hold_timer = np.maximum(hold_timer - 1, 0)
        
        curr_comp = comp_np[t]
        curr_uni  = uni_np[t]       # signal universe: used for PnL + exits
        curr_trade = trading_np[t]  # trading universe: used for entries only
        new_pos = prev_pos.copy()
        
        for j in range(n_tickers):
            if not curr_uni[j]:
                # Not in signal universe → flatten
                if prev_pos[j] != 0:
                    new_pos[j] = 0
                continue
                
            sig = curr_comp[j]
            
            if prev_pos[j] == 0:
                # Flat → check for entry (only in trading universe)
                if not curr_trade[j]:
                    continue  # Skip entry — not in trading universe
                if sig > entry_threshold:
                    new_pos[j] = 1
                    hold_timer[j] = min_hold_bars
                    n_trades += 1
                    n_longs += 1
                elif sig < -entry_threshold:
                    new_pos[j] = -1
                    hold_timer[j] = min_hold_bars
                    n_trades += 1
                    n_shorts += 1
            elif prev_pos[j] == 1:
                # Long → check for exit (normal or emergency)
                if hold_timer[j] == 0 and sig < exit_threshold:
                    new_pos[j] = 0
                    n_trades += 1
                elif sig < -exit_threshold:
                    # Emergency exit: signal fully reversed, override hold timer
                    new_pos[j] = 0
                    hold_timer[j] = 0
                    n_trades += 1
            elif prev_pos[j] == -1:
                # Short → check for exit (normal or emergency)
                if hold_timer[j] == 0 and sig > -exit_threshold:
                    new_pos[j] = 0
                    n_trades += 1
                elif sig > exit_threshold:
                    # Emergency exit: signal fully reversed, override hold timer
                    new_pos[j] = 0
                    hold_timer[j] = 0
                    n_trades += 1
        
        # Apply max positions cap: keep top N by signal strength
        if max_positions > 0:
            active_idx = np.where(new_pos != 0)[0]
            if len(active_idx) > max_positions:
                strengths = np.abs(curr_comp[active_idx])
                # Sort by signal strength, keep top max_positions
                keep_order = np.argsort(strengths)[-max_positions:]
                keep_set = set(active_idx[keep_order])
                for j in active_idx:
                    if j not in keep_set and hold_timer[j] == 0:
                        new_pos[j] = 0
        
        # Compute position weights
        active_mask = np.abs(new_pos) > 0
        n_active = active_mask.sum()
        weights = np.zeros(n_tickers)
        
        if n_active > 0:
            if sizing_mode == "conviction":
                # Signal-proportional sizing: weight by |signal| above threshold
                signal_strengths = np.abs(curr_comp) * active_mask.astype(float)
                total_strength = signal_strengths.sum()
                if total_strength > 0:
                    weights = signal_strengths / total_strength
                    weights = np.minimum(weights, max_weight)
                    w_sum = weights.sum()
                    if w_sum > 0:
                        weights = weights / w_sum  # re-normalize to sum=1
                else:
                    weights[active_mask] = 1.0 / n_active
            else:
                # Equal weight fallback
                w = min(1.0 / n_active, max_weight)
                weights[active_mask] = w
                w_sum = weights.sum()
                if w_sum > 0:
                    weights = weights / w_sum
        
        # --- PCA Risk Hedge ---
        if risk_model is not None:
            risk_model.update(t, ret_np, n_tickers)
            weights = risk_model.hedge_weights(weights, new_pos)
        
        # PnL from previous positions earning current returns
        # Use previous bar's weights for PnL (position was held from prev bar)
        prev_active = np.abs(prev_pos) > 0
        n_prev_active = prev_active.sum()
        if n_prev_active > 0:
            if sizing_mode == "conviction":
                prev_strengths = np.abs(comp_np[t-1]) * prev_active.astype(float)
                prev_total = prev_strengths.sum()
                if prev_total > 0:
                    prev_weights = prev_strengths / prev_total
                    prev_weights = np.minimum(prev_weights, max_weight)
                    pw_sum = prev_weights.sum()
                    if pw_sum > 0:
                        prev_weights = prev_weights / pw_sum
                else:
                    prev_weights = np.zeros(n_tickers)
                    prev_weights[prev_active] = 1.0 / n_prev_active
            else:
                prev_weights = np.zeros(n_tickers)
                w = min(1.0 / n_prev_active, max_weight)
                prev_weights[prev_active] = w
                pw_sum = prev_weights.sum()
                if pw_sum > 0:
                    prev_weights = prev_weights / pw_sum
        else:
            prev_weights = np.zeros(n_tickers)
        
        # --- Volatility Targeting ---
        vol_scale = 1.0
        if vol_target and t > vol_lookback:
            # Compute trailing portfolio vol from recent net PnL
            recent_net = pnl_bars[max(0, t-vol_lookback):t] - fee_bars[max(0, t-vol_lookback):t]
            nonzero = recent_net[recent_net != 0]
            if len(nonzero) > 20:
                port_vol = np.std(nonzero)
                # Target vol = median of all observed vols so far
                vol_scalar_history.append(port_vol)
                if len(vol_scalar_history) > 3:
                    target_vol = np.median(vol_scalar_history)
                    if port_vol > 0:
                        vol_scale = target_vol / port_vol
                        vol_scale = np.clip(vol_scale, 0.25, 1.5)  # floor/cap
        
        
        # Apply risk scaling to weights
        risk_scale = vol_scale
        prev_weights_scaled = prev_weights * risk_scale
        weights_scaled = weights * risk_scale
        
        bar_pnl = np.sum(prev_pos * prev_weights_scaled * ret_np[t])
        
        # Fees from position changes
        bar_fees = 0.0
        for j in range(n_tickers):
            if new_pos[j] != prev_pos[j]:
                # Fee on the notional change using current weights
                w_change = max(weights_scaled[j], prev_weights_scaled[j]) if (weights_scaled[j] > 0 or prev_weights_scaled[j] > 0) else (1.0 / max(n_active, 1)) * risk_scale
                bar_fees += w_change * fee_rate
        
        pnl_bars[t] = bar_pnl * booksize
        fee_bars[t] = bar_fees * booksize
        
        # Update cumulative equity for drawdown tracking
        cum_net += (bar_pnl - bar_fees) * booksize if booksize == 1 else (pnl_bars[t] - fee_bars[t])
        peak_equity = max(peak_equity, cum_net)
        
        positions[t] = new_pos
        prev_pos = new_pos.copy()
    
    # Compute metrics
    net_pnl = pnl_bars - fee_bars
    cum_pnl = np.cumsum(net_pnl)
    gross_pnl = np.cumsum(pnl_bars)
    
    # Annualized metrics (288 bars/day, 365 days/year)
    total_bars = n_bars
    ann_factor = np.sqrt(BARS_PER_DAY * 365)
    
    daily_pnl = []
    for d in range(0, total_bars, BARS_PER_DAY):
        daily_pnl.append(np.sum(net_pnl[d:d+BARS_PER_DAY]))
    daily_pnl = np.array(daily_pnl)
    
    if len(daily_pnl) > 1 and np.std(daily_pnl) > 0:
        sharpe = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(365)
    else:
        sharpe = 0.0
    
    total_return = cum_pnl[-1] / booksize if booksize > 0 else 0
    total_gross = gross_pnl[-1] / booksize if booksize > 0 else 0
    total_fees = np.sum(fee_bars) / booksize if booksize > 0 else 0
    
    # Max drawdown
    peak = np.maximum.accumulate(cum_pnl)
    dd = (cum_pnl - peak)
    max_dd = np.min(dd) / booksize if booksize > 0 else 0
    
    # Position stats
    pos_count = np.abs(positions).sum(axis=1)
    avg_positions = pos_count.mean()
    pct_time_invested = (pos_count > 0).mean()
    
    # Holding period
    holding_periods = []
    for j in range(n_tickers):
        in_pos = False
        entry_bar = 0
        for t in range(n_bars):
            if not in_pos and positions[t, j] != 0:
                in_pos = True
                entry_bar = t
            elif in_pos and positions[t, j] == 0:
                holding_periods.append(t - entry_bar)
                in_pos = False
        if in_pos:
            holding_periods.append(n_bars - entry_bar)
    
    avg_hold = np.mean(holding_periods) if holding_periods else 0
    avg_hold_hours = avg_hold * 5 / 60  # 5min bars → hours
    
    # IC of composite vs forward returns (on universe stocks)
    comp_lag = composite.shift(1)
    returns_fwd = returns_df.reindex(index=dates, columns=tickers)
    ics = []
    for t_idx in range(100, n_bars, 12):  # sample every hour
        dt = dates[t_idx]
        a = comp_lag.loc[dt]
        r = returns_fwd.loc[dt]
        valid = a.notna() & r.notna() & np.isfinite(a) & np.isfinite(r)
        av, rv = a[valid], r[valid]
        if len(av) < 10 or av.std() < 1e-15:
            continue
        ic, _ = stats.spearmanr(av, rv)
        ics.append(ic)
    ic_mean = np.mean(ics) if ics else 0
    ic_std = np.std(ics) if ics else 1
    
    pos_count = np.abs(positions).sum(axis=1)
    
    return {
        "sharpe": sharpe,
        "total_return": total_return,
        "total_gross": total_gross,
        "total_fees": total_fees,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
        "n_longs": n_longs,
        "n_shorts": n_shorts,
        "avg_positions": avg_positions,
        "pct_invested": pct_time_invested,
        "avg_hold_hours": avg_hold_hours,
        "n_bars": n_bars,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "daily_pnl": daily_pnl,
        "cum_pnl": cum_pnl,
        "entry_threshold": entry_threshold,
        "exit_threshold": exit_threshold,
        # Internal arrays for charting
        "_positions": positions,
        "_gross_pnl": gross_pnl,
        "_fee_cum": np.cumsum(fee_bars),
        "_pos_count": pos_count,
        "_booksize": booksize,
    }


def run_continuous_strategy(composite, returns_df, universe_df,
                            fees_bps=FEES_BPS, booksize=BOOKSIZE,
                            max_weight=MAX_WEIGHT, max_positions=0,
                            vol_lookback=72, trade_buffer=0.0,
                            position_smooth=0.0, rebalance_every=1):
    """
    Continuous signal-proportional portfolio.

    Instead of binary threshold entry/exit, positions are continuously
    proportional to the composite signal:
      target_weight[i] = signal[i] / sum(|signal|)          (dollar neutral)
      vol_adj_weight[i] = target_weight[i] / vol[i]         (vol normalized)

    Key features:
      - Per-position inverse-volatility weighting
      - Trade buffer: only change weight if |target - current| > buffer
        to reduce turnover and fee drag
      - Position smoothing via EMA to reduce whipsawing
      - No hard entry/exit thresholds — the signal IS the position

    This is the standard institutional approach for systematic alpha.
    """
    dates = composite.index
    tickers = composite.columns
    n_bars = len(dates)
    n_tickers = len(tickers)

    fee_rate = fees_bps / 10_000.0

    uni_np = universe_df.reindex(index=dates, columns=tickers).fillna(False).values
    comp_np = np.nan_to_num(composite.values, nan=0.0)
    ret_np = returns_df.reindex(index=dates, columns=tickers).fillna(0).values

    # Pre-compute per-asset rolling volatility for inverse-vol weighting
    ret_vol = np.full((n_bars, n_tickers), np.nan)
    for t in range(vol_lookback, n_bars):
        window = ret_np[t-vol_lookback:t]
        ret_vol[t] = np.nanstd(window, axis=0)
    # Fill early bars with expanding vol
    for t in range(1, vol_lookback):
        window = ret_np[:t+1]
        ret_vol[t] = np.nanstd(window, axis=0)
    ret_vol = np.maximum(ret_vol, 1e-8)  # floor

    pnl_bars = np.zeros(n_bars)
    fee_bars = np.zeros(n_bars)
    weights = np.zeros((n_bars, n_tickers))  # actual weights (signed)
    n_trades = 0
    n_longs = 0
    n_shorts = 0
    prev_weights = np.zeros(n_tickers)

    for t in range(1, n_bars):
        # PnL from previous positions earning current returns (always computed)
        pnl_bars[t] = np.sum(prev_weights * ret_np[t])

        # Skip rebalancing if not a rebalance bar
        if rebalance_every > 1 and t % rebalance_every != 0:
            weights[t] = prev_weights
            continue

        sig = comp_np[t].copy()
        uni = uni_np[t]

        # Zero out non-universe
        sig[~uni] = 0.0

        # Inverse-vol adjust: scale signal by 1/vol
        inv_vol = 1.0 / ret_vol[t]
        inv_vol[~np.isfinite(inv_vol)] = 0.0
        sig_vol_adj = sig * inv_vol

        # Dollar-neutral: demean
        valid_mask = uni & np.isfinite(sig_vol_adj) & (sig_vol_adj != 0)
        if valid_mask.sum() < 3:
            weights[t] = prev_weights
            continue

        cs_mean = sig_vol_adj[valid_mask].mean()
        sig_centered = sig_vol_adj - cs_mean

        # Normalize to sum of abs weights = 1 (book leverage)
        abs_sum = np.abs(sig_centered).sum()
        if abs_sum > 1e-10:
            target_weights = sig_centered / abs_sum
        else:
            target_weights = np.zeros(n_tickers)

        # Max positions: keep top N by abs weight
        if max_positions > 0:
            abs_w = np.abs(target_weights)
            sorted_idx = np.argsort(abs_w)
            # Zero out the weakest positions beyond max_positions
            n_nonzero = (abs_w > 1e-10).sum()
            if n_nonzero > max_positions:
                cutoff_idx = sorted_idx[:-max_positions]
                target_weights[cutoff_idx] = 0.0
                # Re-normalize
                abs_sum2 = np.abs(target_weights).sum()
                if abs_sum2 > 1e-10:
                    target_weights = target_weights / abs_sum2

        # Cap individual weights
        target_weights = np.clip(target_weights, -max_weight, max_weight)
        abs_sum3 = np.abs(target_weights).sum()
        if abs_sum3 > 1e-10:
            target_weights = target_weights / abs_sum3

        # Position smoothing (EMA blending with previous weights)
        if position_smooth > 0:
            alpha = 2.0 / (position_smooth + 1.0)
            target_weights = alpha * target_weights + (1 - alpha) * prev_weights

        # Trade buffer: only update position if change exceeds buffer
        if trade_buffer > 0:
            weight_change = np.abs(target_weights - prev_weights)
            # Only update positions where change exceeds buffer
            update_mask = weight_change > trade_buffer
            actual_weights = prev_weights.copy()
            actual_weights[update_mask] = target_weights[update_mask]
            # Re-normalize after selective update
            abs_sum4 = np.abs(actual_weights).sum()
            if abs_sum4 > 1e-10:
                actual_weights = actual_weights / abs_sum4
            target_weights = actual_weights

        # PnL already computed at top of loop

        # Fees from weight changes
        turnover = np.abs(target_weights - prev_weights).sum()
        fee_bars[t] = turnover * fee_rate * booksize

        # Count trades
        for j in range(n_tickers):
            old_dir = np.sign(prev_weights[j])
            new_dir = np.sign(target_weights[j])
            if old_dir != new_dir and new_dir != 0:
                n_trades += 1
                if new_dir > 0:
                    n_longs += 1
                else:
                    n_shorts += 1

        weights[t] = target_weights
        prev_weights = target_weights.copy()

    # Scale PnL to booksize
    pnl_bars = pnl_bars * booksize

    # Compute metrics
    net_pnl = pnl_bars - fee_bars
    cum_pnl = np.cumsum(net_pnl)
    gross_pnl = np.cumsum(pnl_bars)

    daily_pnl = []
    for d in range(0, n_bars, BARS_PER_DAY):
        daily_pnl.append(np.sum(net_pnl[d:d+BARS_PER_DAY]))
    daily_pnl = np.array(daily_pnl)

    if len(daily_pnl) > 1 and np.std(daily_pnl) > 0:
        sharpe = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(365)
    else:
        sharpe = 0.0

    total_return = cum_pnl[-1] / booksize if booksize > 0 else 0
    total_gross = gross_pnl[-1] / booksize if booksize > 0 else 0
    total_fees = np.sum(fee_bars) / booksize if booksize > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(cum_pnl)
    dd = cum_pnl - peak
    max_dd = np.min(dd) / booksize if booksize > 0 else 0

    # Position stats
    pos_count = (np.abs(weights) > 1e-8).sum(axis=1)
    avg_positions = pos_count.mean()
    pct_time_invested = (pos_count > 0).mean()

    # Average holding period (approximate from autocorrelation of weight signs)
    avg_hold_hours = 0.0
    holding_periods = []
    for j in range(n_tickers):
        in_pos = False
        entry_bar = 0
        for t in range(n_bars):
            w = weights[t, j]
            has_pos = abs(w) > 1e-8
            if not in_pos and has_pos:
                in_pos = True
                entry_bar = t
            elif in_pos and not has_pos:
                holding_periods.append(t - entry_bar)
                in_pos = False
        if in_pos:
            holding_periods.append(n_bars - entry_bar)
    avg_hold = np.mean(holding_periods) if holding_periods else 0
    avg_hold_hours = avg_hold * 5 / 60

    # IC of composite vs forward returns
    comp_lag = composite.shift(1)
    returns_fwd = returns_df.reindex(index=dates, columns=tickers)
    ics = []
    for t_idx in range(100, n_bars, 12):
        dt = dates[t_idx]
        a = comp_lag.loc[dt]
        r = returns_fwd.loc[dt]
        valid = a.notna() & r.notna() & np.isfinite(a) & np.isfinite(r)
        av, rv = a[valid], r[valid]
        if len(av) < 10 or av.std() < 1e-15:
            continue
        ic, _ = stats.spearmanr(av, rv)
        ics.append(ic)
    ic_mean = np.mean(ics) if ics else 0
    ic_std = np.std(ics) if ics else 1

    # Positions for compatibility (sign of weights)
    positions = np.sign(weights)
    pos_count = (np.abs(weights) > 1e-8).sum(axis=1)

    return {
        "sharpe": sharpe,
        "total_return": total_return,
        "total_gross": total_gross,
        "total_fees": total_fees,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
        "n_longs": n_longs,
        "n_shorts": n_shorts,
        "avg_positions": avg_positions,
        "pct_invested": pct_time_invested,
        "avg_hold_hours": avg_hold_hours,
        "n_bars": n_bars,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "daily_pnl": daily_pnl,
        "cum_pnl": cum_pnl,
        "entry_threshold": 0.0,
        "exit_threshold": 0.0,
        "_positions": positions,
        "_gross_pnl": gross_pnl,
        "_fee_cum": np.cumsum(fee_bars),
        "_pos_count": pos_count,
        "_booksize": booksize,
    }


def print_result(result, label=""):
    """Print backtest results in a compact format."""
    out = f"""
  {label}
  ----------------------------------------------
  Sharpe:       {result['sharpe']:+.3f}
  Return:       {result['total_return']*100:+.2f}% (gross {result['total_gross']*100:+.2f}%, fees {result['total_fees']*100:.2f}%)
  Max DD:       {result['max_drawdown']*100:.2f}%
  IC:           {result['ic_mean']:+.5f} (std {result['ic_std']:.5f})
  Trades:       {result['n_trades']} ({result['n_longs']} long, {result['n_shorts']} short)
  Avg Hold:     {result['avg_hold_hours']:.1f} hours
  Avg Pos:      {result['avg_positions']:.1f} symbols | {result['pct_invested']*100:.1f}% time invested
  Bars:         {result['n_bars']}
  Entry/Exit:   {result['entry_threshold']:.2f} / {result['exit_threshold']:.2f}"""
    print(out, flush=True)


def plot_results(result, composite, returns_df, label="", save_dir="."):
    """
    Generate two charts:
      1. Per-symbol cumulative returns (top 10 by total return)
      2. Aggregate PnL (gross vs net vs fees) + position count
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    dates = composite.index
    n_bars = len(dates)
    tickers = composite.columns.tolist()
    booksize = result.get('_booksize', BOOKSIZE)
    
    ret_np = returns_df.reindex(index=dates, columns=tickers).fillna(0).values
    x = np.arange(n_bars)
    
    # --- Chart 1: Per-symbol cumulative returns ---
    fig, ax = plt.subplots(figsize=(14, 7))
    cum_ret = np.cumsum(ret_np, axis=0) * 100  # percentage
    total_ret = cum_ret[-1]
    top_idx = np.argsort(total_ret)[-10:]
    bot_idx = np.argsort(total_ret)[:5]
    for j in np.concatenate([bot_idx, top_idx]):
        ax.plot(x, cum_ret[:, j], alpha=0.6, linewidth=0.8, label=tickers[j])
    ax.set_title(f'Symbol Cumulative Returns (top 10 + bottom 5) — {label}', fontsize=13)
    ax.set_xlabel('Bar')
    ax.set_ylabel('Cumulative Return (%)')
    ax.legend(fontsize=7, ncol=3, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)
    plt.tight_layout()
    p1 = os.path.join(save_dir, 'chart_symbol_returns.png')
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"  Chart saved: {p1}", flush=True)
    
    # --- Chart 2: Aggregate PnL + Position Count ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                    gridspec_kw={'height_ratios': [2, 1]})
    
    net_pct = result['cum_pnl'] / booksize * 100
    gross_pct = result['_gross_pnl'] / booksize * 100
    fee_pct = result['_fee_cum'] / booksize * 100
    
    ax1.plot(x, net_pct, color='#2ecc71', linewidth=2, label='Net PnL')
    ax1.plot(x, gross_pct, color='#3498db', linewidth=1.5, alpha=0.7, label='Gross PnL')
    ax1.fill_between(x, net_pct, gross_pct, color='#e74c3c', alpha=0.2, label='Fee Drag')
    ax1.set_title(f'Aggregate Portfolio PnL — {label} ({FEES_BPS:.0f} bps fees)', fontsize=13)
    ax1.set_ylabel('Cumulative Return (%)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='black', linewidth=0.5)
    
    pos_count = result['_pos_count']
    ax2.fill_between(x, pos_count, color='#9b59b6', alpha=0.4)
    ax2.plot(x, pos_count, color='#9b59b6', linewidth=0.5, alpha=0.8)
    ax2.set_title('Active Positions Over Time', fontsize=11)
    ax2.set_xlabel('Bar')
    ax2.set_ylabel('# Positions')
    ax2.set_ylim(0, max(pos_count.max() * 1.1, 1))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    p2 = os.path.join(save_dir, 'chart_aggregate_pnl.png')
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"  Chart saved: {p2}", flush=True)


# ============================================================================
# OPTIMIZATION
# ============================================================================

def optimize_thresholds(composite, returns_df, universe_df, 
                        split_label="val", 
                        objective="sharpe"):
    """Grid search over entry/exit thresholds to maximize Sharpe or return."""
    
    entry_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5]
    exit_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    hold_values = [12, 36, 72, 144]
    
    results = []
    total = len(entry_values) * len(exit_values) * len(hold_values)
    i = 0
    
    print(f"\n  Optimizing on {split_label} ({total} combinations)...", flush=True)
    
    for entry in entry_values:
        for exit_val in exit_values:
            if exit_val >= entry:
                continue  # exit must be below entry
            for hold in hold_values:
                i += 1
                r = run_threshold_strategy(
                    composite, returns_df, universe_df,
                    entry_threshold=entry,
                    exit_threshold=exit_val,
                    min_hold_bars=hold,
                )
                results.append({
                    "entry": entry, "exit": exit_val, "hold": hold,
                    "sharpe": r["sharpe"], "return": r["total_return"],
                    "ic": r["ic_mean"], "dd": r["max_drawdown"],
                    "trades": r["n_trades"], "avg_hold_h": r["avg_hold_hours"],
                    "pct_inv": r["pct_invested"],
                })
                if i % 20 == 0:
                    print(f"    {i}/{total}...", flush=True)
    
    # Sort by objective
    if objective == "sharpe":
        results.sort(key=lambda x: x["sharpe"], reverse=True)
    elif objective == "return":
        results.sort(key=lambda x: x["return"], reverse=True)
    elif objective == "ic":
        results.sort(key=lambda x: x["ic"], reverse=True)
    
    print(f"\n  Top 10 parameter sets (by {objective}):", flush=True)
    print(f"  {'Entry':>6s} {'Exit':>5s} {'Hold':>5s} {'SR':>7s} {'Ret%':>7s} {'IC':>8s} {'DD%':>7s} {'Trd':>5s} {'AvgH':>6s} {'%Inv':>5s}")
    print(f"  {'-'*70}")
    for r in results[:10]:
        print(f"  {r['entry']:6.2f} {r['exit']:5.2f} {r['hold']:5d} "
              f"{r['sharpe']:+7.2f} {r['return']*100:+7.2f} {r['ic']:+8.5f} "
              f"{r['dd']*100:7.2f} {r['trades']:5d} {r['avg_hold_h']:6.1f} "
              f"{r['pct_inv']*100:5.1f}")
    
    best = results[0]
    print(f"\n  BEST: entry={best['entry']:.2f} exit={best['exit']:.2f} hold={best['hold']} "
          f"-> SR={best['sharpe']:+.2f} Ret={best['return']*100:+.2f}% IC={best['ic']:+.5f}", flush=True)
    
    return best


# ============================================================================
# WALK-FORWARD VALIDATION
# ============================================================================

def walk_forward_validate(alphas, objective="sharpe", combine="ic_weighted",
                          train_days=6, test_days=3,
                          fixed_entry=None, fixed_exit=None, fixed_hold=None,
                          max_positions=0, sizing_mode="conviction",
                          vol_target=True,
                          pca_components=0, pca_hedge=0.0,
                          fees_bps=FEES_BPS):
    """
    Walk-Forward Cross-Validation over the full trainval period.
    
    This is the most reliable performance estimator. It simulates realistic
    out-of-sample performance by:
    
    1. Building the composite signal on the FULL trainval period (no lookahead
       in signal construction — EWMA normalization is causal).
    2. Splitting trainval into rolling windows of [train_days | test_days].
    3. For each fold:
       a. If fixed params provided: Use them directly (PREFERRED — more stable).
       b. Otherwise: Grid-search entry/exit/hold on the train window.
       c. Run the threshold strategy on the test window.
       d. Record OOS metrics (Sharpe, return, IC).
    4. Aggregate all OOS periods into a single equity curve.
    5. Report aggregate Sharpe, mean fold Sharpe, total return.
    
    IMPORTANT: The composite signal is built ONCE on the full trainval period.
    Individual fold train/test splits are over the SAME signal — they only
    differ in which bars are used for evaluation. This means there's no
    lookahead in the alpha combination (EWMA normalization is causal), but
    the alpha selection (which alphas from the DB) is global.
    
    Default window: 6-day train, 3-day test, rolling across Dec 1 – Feb 28.
    This produces 28 folds with 91 days of OOS coverage.
    
    Parameters
    ----------
    alphas : list of tuples
        (alpha_id, expression, ic_mean, sharpe_is) from the alpha DB.
    objective : str
        Grid-search objective: 'sharpe', 'return', or 'ic'.
    combine : str
        Signal combination method (e.g., 'ts_autonorm').
    train_days, test_days : int
        Walk-forward window sizes in days.
    fixed_entry, fixed_exit, fixed_hold : float/int or None
        If all provided, skip grid search and use these fixed params.
        STRONGLY PREFERRED for stability and to avoid optimizer noise.
    max_positions : int
        Max concurrent positions (0 = unlimited). Best: 50.
    fees_bps : float
        Fees in basis points per side. Default: 7.
    
    Returns
    -------
    dict with keys:
        'folds': list of per-fold results,
        'agg_sharpe': aggregate OOS Sharpe,
        'agg_return': aggregate OOS return,
        'mean_oos_sharpe': average across folds,
        'mean_oos_ic': average OOS IC
    """
    # Load full trainval data
    matrices, universe = load_data("trainval")
    close = matrices.get("close")
    returns = close.pct_change() if close is not None else matrices.get("returns")
    # Use winsorized returns for PnL attribution (falls back to raw if not available)
    returns_pnl = matrices.get("returns_clean", returns)
    
    # Build composite on full trainval
    if combine in ML_COMBINERS:
        if combine == "ml_ridge":
            composite = compute_composite_ml(alphas, matrices, universe, returns,
                                            target_horizon=_ML_TARGET_HORIZON,
                                            use_enriched_features=_ML_ENRICHED)
        elif combine == "ts_autonorm":
            composite = compute_composite_ts_autonorm(alphas, matrices, universe, returns,
                                                       gamma=_TS_GAMMA,
                                                       ic_weighted=not _EQUAL_WEIGHT,
                                                       signal_smooth=_SIGNAL_SMOOTH,
                                                       concordance_boost=_CONCORDANCE,
                                                       rank_norm=_RANK_NORM,
                                                       beta_hedge=_BETA_HEDGE,
                                                       signal_hedge=_SIGNAL_HEDGE,
                                                       rolling_ic=_ROLLING_IC,
                                                       rolling_return=_ROLLING_RETURN)
        elif combine == "lgbm":
            composite = compute_composite_lgbm(alphas, matrices, universe, returns,
                                               gamma=_TS_GAMMA)
        else:
            composite = ML_COMBINERS[combine](alphas, matrices, universe, returns)
    else:
        composite = compute_composite_signal(alphas, matrices, universe, method=combine)
    if composite is None:
        print("  Failed to build composite signal")
        return None
    
    # Apply PCA neutralization if requested
    if pca_components > 0:
        composite = pca_neutralize_signal(composite, returns, n_components=pca_components)
    
    dates = composite.index
    total_bars = len(dates)
    train_bars = train_days * BARS_PER_DAY
    test_bars = test_days * BARS_PER_DAY
    
    use_fixed = (fixed_entry is not None and fixed_exit is not None and fixed_hold is not None)
    
    print(f"\n  Walk-Forward Validation ({combine})", flush=True)
    print(f"  Total: {total_bars} bars ({total_bars/BARS_PER_DAY:.0f} days)")
    print(f"  Window: train={train_days}d ({train_bars}b), test={test_days}d ({test_bars}b)")
    if use_fixed:
        print(f"  FIXED PARAMS: entry={fixed_entry} exit={fixed_exit} hold={fixed_hold}")
    print(f"  Sizing: {sizing_mode} | Max positions: {max_positions if max_positions > 0 else 'unlimited'}")
    
    entry_values = [0.5, 0.7, 0.8, 1.0, 1.2, 1.5]
    exit_values = [0.0, 0.2, 0.3, 0.5]
    hold_values = [36, 72, 144]
    
    fold_results = []
    fold_id = 0
    
    start = 0
    while start + train_bars + test_bars <= total_bars:
        fold_id += 1
        train_end = start + train_bars
        test_end = train_end + test_bars
        
        # Slice data for this fold — use winsorized returns for PnL
        train_comp = composite.iloc[start:train_end]
        train_ret = returns_pnl.iloc[start:train_end]
        train_uni = universe.iloc[start:train_end]
        
        test_comp = composite.iloc[train_end:test_end]
        test_ret = returns_pnl.iloc[train_end:test_end]
        test_uni = universe.iloc[train_end:test_end]
        
        # Use fixed params or grid search
        if use_fixed:
            best_params = (fixed_entry, fixed_exit, fixed_hold)
            best_sr = 0  # not optimized
        else:
            # Grid search on train fold
            best_sr = -999
            best_params = (1.0, 0.3, 72)
            for entry in entry_values:
                for exit_val in exit_values:
                    if exit_val >= entry:
                        continue
                    for hold in hold_values:
                        r = run_threshold_strategy(
                            train_comp, train_ret, train_uni,
                            entry_threshold=entry, exit_threshold=exit_val,
                            min_hold_bars=hold,
                            max_positions=max_positions,
                            sizing_mode=sizing_mode,
                            vol_target=vol_target,
                            pca_hedge=pca_hedge,
                            fees_bps=fees_bps,
                        )
                        score = r["sharpe"] if objective == "sharpe" else r["total_return"]
                        if score > best_sr:
                            best_sr = score
                            best_params = (entry, exit_val, hold)
        
        # Test on OOS fold with best params
        oos = run_threshold_strategy(
            test_comp, test_ret, test_uni,
            entry_threshold=best_params[0],
            exit_threshold=best_params[1],
            min_hold_bars=best_params[2],
            max_positions=max_positions,
            sizing_mode=sizing_mode,
            vol_target=vol_target,
            pca_hedge=pca_hedge,
            fees_bps=fees_bps,
        )
        
        fold_results.append({
            "fold": fold_id,
            "train_start": str(dates[start].date()),
            "test_start": str(dates[train_end].date()),
            "test_end": str(dates[min(test_end, total_bars)-1].date()),
            "params": best_params,
            "train_sr": best_sr,
            "oos_sr": oos["sharpe"],
            "oos_ret": oos["total_return"],
            "oos_gross": oos["total_gross"],
            "oos_fees": oos["total_fees"],
            "oos_dd": oos["max_drawdown"],
            "oos_trades": oos["n_trades"],
            "oos_longs": oos["n_longs"],
            "oos_shorts": oos["n_shorts"],
            "oos_hold_h": oos["avg_hold_hours"],
            "oos_avg_pos": oos["avg_positions"],
            "oos_pct_inv": oos["pct_invested"],
            "oos_ic": oos["ic_mean"],
            "oos_daily_pnl": oos["daily_pnl"],
            # For charting: per-bar PnL and per-symbol returns
            "_oos_cum_pnl": oos["cum_pnl"],
            "_oos_gross_pnl": oos["_gross_pnl"],
            "_oos_fee_cum": oos["_fee_cum"],
            "_oos_pos_count": oos["_pos_count"],
            "_oos_positions": oos["_positions"],
            "_oos_booksize": oos.get("_booksize", BOOKSIZE),
            "_test_bar_start": train_end,
            "_test_bar_end": min(test_end, total_bars),
        })
        
        print(f"  Fold {fold_id}: train {dates[start].date()}->{dates[train_end-1].date()} | "
              f"test {dates[train_end].date()}->{dates[min(test_end,total_bars)-1].date()} | "
              f"params=({best_params[0]:.1f},{best_params[1]:.1f},{best_params[2]}) | "
              f"trainSR={best_sr:+.1f} oosSR={oos['sharpe']:+.1f} oosRet={oos['total_return']*100:+.2f}%",
              flush=True)
        
        start += test_bars
    
    if not fold_results:
        print("  Not enough data for walk-forward")
        return None
    
    # Aggregate OOS results
    all_oos_daily = np.concatenate([f["oos_daily_pnl"] for f in fold_results])
    if len(all_oos_daily) > 1 and np.std(all_oos_daily) > 0:
        agg_sr = np.mean(all_oos_daily) / np.std(all_oos_daily) * np.sqrt(365)
    else:
        agg_sr = 0
    agg_ret = sum(f["oos_ret"] for f in fold_results)
    agg_gross = sum(f["oos_gross"] for f in fold_results)
    agg_fees = sum(f["oos_fees"] for f in fold_results)
    avg_oos_sr = np.mean([f["oos_sr"] for f in fold_results])
    avg_oos_ic = np.mean([f["oos_ic"] for f in fold_results])
    sr_std = np.std([f["oos_sr"] for f in fold_results])
    
    # Full statistics
    total_trades = sum(f["oos_trades"] for f in fold_results)
    total_longs = sum(f["oos_longs"] for f in fold_results)
    total_shorts = sum(f["oos_shorts"] for f in fold_results)
    avg_hold_h = np.mean([f["oos_hold_h"] for f in fold_results if f["oos_hold_h"] > 0])
    avg_positions = np.mean([f["oos_avg_pos"] for f in fold_results])
    avg_pct_inv = np.mean([f["oos_pct_inv"] for f in fold_results])
    
    # Max drawdown from stitched equity curve
    booksize = fold_results[0].get('_oos_booksize', BOOKSIZE)
    stitched_net = []
    cum = 0.0
    for f in fold_results:
        fold_bar_pnl = np.diff(f['_oos_cum_pnl'], prepend=0)
        for p in fold_bar_pnl:
            cum += p
            stitched_net.append(cum)
    stitched_net = np.array(stitched_net)
    peak = np.maximum.accumulate(stitched_net)
    dd = (stitched_net - peak) / booksize
    max_dd = dd.min() if len(dd) > 0 else 0
    calmar = (agg_ret / abs(max_dd)) if abs(max_dd) > 1e-6 else float('inf')
    
    # Win rate (fold level)
    n_pos_folds = sum(1 for f in fold_results if f["oos_ret"] > 0)
    win_rate = n_pos_folds / len(fold_results) * 100
    
    # Best / worst folds
    best_fold = max(fold_results, key=lambda f: f["oos_ret"])
    worst_fold = min(fold_results, key=lambda f: f["oos_ret"])
    
    # Daily stats
    n_pos_days = sum(1 for d in all_oos_daily if d > 0)
    daily_win_rate = n_pos_days / len(all_oos_daily) * 100 if len(all_oos_daily) > 0 else 0
    avg_daily_ret = np.mean(all_oos_daily) / booksize * 100
    avg_win_day = np.mean([d for d in all_oos_daily if d > 0]) / booksize * 100 if n_pos_days > 0 else 0
    avg_lose_day = np.mean([d for d in all_oos_daily if d <= 0]) / booksize * 100 if (len(all_oos_daily) - n_pos_days) > 0 else 0
    profit_factor = abs(sum(d for d in all_oos_daily if d > 0) / sum(d for d in all_oos_daily if d < 0)) if sum(d for d in all_oos_daily if d < 0) != 0 else float('inf')
    
    # Trades per day
    total_oos_days = len(all_oos_daily)
    trades_per_day = total_trades / total_oos_days if total_oos_days > 0 else 0
    
    # Most commonly chosen parameters
    from collections import Counter
    param_counts = Counter(f["params"] for f in fold_results)
    most_common_params = param_counts.most_common(1)[0]
    
    out = f"""
  ===============================================
  WALK-FORWARD SUMMARY ({len(fold_results)} folds)
  ===============================================
  
  Performance
  -----------
  Aggregate OOS Sharpe:  {agg_sr:+.3f}
  Mean Fold OOS Sharpe:  {avg_oos_sr:+.3f} (std {sr_std:.3f})
  Total OOS Return:      {agg_ret*100:+.2f}% (gross {agg_gross*100:+.2f}%, fees {agg_fees*100:.2f}%)
  Max Drawdown:          {max_dd*100:.2f}%
  Calmar Ratio:          {calmar:.2f}
  Mean OOS IC:           {avg_oos_ic:+.5f}
  
  Trading Activity
  ----------------
  Total Trades:          {total_trades} ({total_longs} long, {total_shorts} short)
  Trades/Day:            {trades_per_day:.1f}
  Avg Hold:              {avg_hold_h:.1f} hours
  Avg Positions:         {avg_positions:.1f} symbols
  Time Invested:         {avg_pct_inv*100:.1f}%
  
  Win Rates
  ---------
  Fold Win Rate:         {n_pos_folds}/{len(fold_results)} ({win_rate:.0f}%)
  Daily Win Rate:        {n_pos_days}/{len(all_oos_daily)} ({daily_win_rate:.0f}%)
  Profit Factor:         {profit_factor:.2f}
  Avg Winning Day:       {avg_win_day:+.3f}%
  Avg Losing Day:        {avg_lose_day:+.3f}%
  
  Extremes
  --------
  Best Fold:             #{best_fold['fold']} ({best_fold['test_start']}) {best_fold['oos_ret']*100:+.2f}%
  Worst Fold:            #{worst_fold['fold']} ({worst_fold['test_start']}) {worst_fold['oos_ret']*100:+.2f}%
  
  Config
  ------
  Params:                entry={most_common_params[0][0]:.1f} exit={most_common_params[0][1]:.1f} hold={most_common_params[0][2]} ({most_common_params[1]}/{len(fold_results)} folds)
  Period:                {fold_results[0]['test_start']} -> {fold_results[-1]['test_end']} ({total_oos_days} days OOS)
  ==============================================="""
    print(out, flush=True)
    
    # Generate walk-forward charts
    plot_walkforward_results(fold_results, returns, composite, agg_sr, agg_ret)
    
    return {
        "folds": fold_results,
        "agg_sharpe": agg_sr,
        "avg_oos_sharpe": avg_oos_sr,
        "total_oos_return": agg_ret,
        "most_common_params": most_common_params[0],
    }



def plot_walkforward_results(fold_results, returns, composite, agg_sr, agg_ret,
                             save_dir="."):
    """
    Generate two walk-forward charts:
      1. Per-symbol equity curves (OOS only, stitched across all folds)
      2. Aggregate OOS PnL (net vs gross vs fees) + position count
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime

    tickers = composite.columns.tolist()
    all_dates = composite.index
    n_tickers = len(tickers)
    booksize = fold_results[0].get('_oos_booksize', BOOKSIZE)

    # ── Stitch all OOS fold data into continuous time series ──
    # Collect test bar indices and per-symbol returns for each fold
    oos_bar_indices = []
    for f in fold_results:
        b0 = f['_test_bar_start']
        b1 = f['_test_bar_end']
        oos_bar_indices.append((b0, b1))

    # Per-symbol cumulative return across OOS periods only
    ret_np = returns.reindex(index=all_dates, columns=tickers).fillna(0).values
    total_oos_bars = sum(b1 - b0 for b0, b1 in oos_bar_indices)

    # Build stitched per-symbol returns and aggregate PnL
    stitched_ret = np.zeros((total_oos_bars, n_tickers))
    stitched_net_pnl = np.zeros(total_oos_bars)
    stitched_gross_pnl = np.zeros(total_oos_bars)
    stitched_fee_cum = np.zeros(total_oos_bars)
    stitched_pos_count = np.zeros(total_oos_bars)
    stitched_dates = []

    offset = 0
    cumulative_net = 0.0
    cumulative_gross = 0.0
    cumulative_fees = 0.0

    for i, f in enumerate(fold_results):
        b0, b1 = oos_bar_indices[i]
        n_fold_bars = b1 - b0
        oos_cum = f['_oos_cum_pnl']
        oos_gross = f['_oos_gross_pnl']
        oos_fee = f['_oos_fee_cum']
        oos_pos = f['_oos_pos_count']

        # Per-symbol returns for this fold's OOS slice
        stitched_ret[offset:offset+n_fold_bars] = ret_np[b0:b1]

        # Compute per-bar net PnL (not cumulative) from fold's cumulative
        fold_net_bar = np.diff(oos_cum, prepend=0)
        fold_gross_bar = np.diff(oos_gross, prepend=0)
        fold_fee_bar = np.diff(oos_fee, prepend=0)

        for t in range(n_fold_bars):
            if t < len(fold_net_bar):
                cumulative_net += fold_net_bar[t]
                cumulative_gross += fold_gross_bar[t]
                cumulative_fees += fold_fee_bar[t]
            stitched_net_pnl[offset + t] = cumulative_net
            stitched_gross_pnl[offset + t] = cumulative_gross
            stitched_fee_cum[offset + t] = cumulative_fees
            if t < len(oos_pos):
                stitched_pos_count[offset + t] = oos_pos[t]

        # Dates for x-axis
        for bar_idx in range(b0, b1):
            if bar_idx < len(all_dates):
                stitched_dates.append(all_dates[bar_idx])
            else:
                stitched_dates.append(all_dates[-1])

        offset += n_fold_bars

    stitched_dates = pd.DatetimeIndex(stitched_dates)

    # ── Chart 1: Per-symbol equity curves ──
    fig, ax = plt.subplots(figsize=(16, 8))
    sym_cum = np.cumsum(stitched_ret, axis=0) * 100  # percentage
    final_ret = sym_cum[-1] if len(sym_cum) > 0 else np.zeros(n_tickers)

    # Plot all symbols in grey, then highlight top 10 + bottom 5
    for j in range(n_tickers):
        ax.plot(stitched_dates, sym_cum[:, j], color='#cccccc', alpha=0.15,
                linewidth=0.4)

    top_idx = np.argsort(final_ret)[-10:]
    bot_idx = np.argsort(final_ret)[:5]
    colors_top = plt.cm.Set2(np.linspace(0, 1, 10))
    colors_bot = plt.cm.Set1(np.linspace(0, 0.5, 5))

    for k, j in enumerate(top_idx):
        ax.plot(stitched_dates, sym_cum[:, j], color=colors_top[k],
                linewidth=1.4, alpha=0.9, label=f'{tickers[j]} ({final_ret[j]:+.0f}%)')
    for k, j in enumerate(bot_idx):
        ax.plot(stitched_dates, sym_cum[:, j], color=colors_bot[k],
                linewidth=1.2, alpha=0.8, linestyle='--',
                label=f'{tickers[j]} ({final_ret[j]:+.0f}%)')

    # Add fold boundaries
    for i, (b0, b1) in enumerate(oos_bar_indices):
        if b0 < len(all_dates):
            ax.axvline(all_dates[b0], color='#666666', linewidth=0.3, alpha=0.4)

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(f'Walk-Forward OOS: Per-Symbol Cumulative Returns\n'
                 f'{len(fold_results)} folds | {total_oos_bars} OOS bars | '
                 f'Top 10 + Bottom 5 highlighted',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Cumulative Return (%)', fontsize=11)
    ax.legend(fontsize=7, ncol=3, loc='upper left', framealpha=0.8)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    fig.autofmt_xdate()
    plt.tight_layout()
    p1 = os.path.join(save_dir, 'wf_symbol_equity_curves.png')
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"  Chart saved: {p1}", flush=True)

    # ── Chart 2: Aggregate OOS PnL + Position Count ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                    gridspec_kw={'height_ratios': [2.5, 1]})

    net_pct = stitched_net_pnl / booksize * 100
    gross_pct = stitched_gross_pnl / booksize * 100
    fee_pct = stitched_fee_cum / booksize * 100

    ax1.plot(stitched_dates, net_pct, color='#2ecc71', linewidth=2.2,
             label=f'Net PnL ({net_pct[-1]:+.1f}%)', zorder=3)
    ax1.plot(stitched_dates, gross_pct, color='#3498db', linewidth=1.5,
             alpha=0.7, label=f'Gross PnL ({gross_pct[-1]:+.1f}%)', zorder=2)
    ax1.fill_between(stitched_dates, net_pct, gross_pct,
                     color='#e74c3c', alpha=0.15, label=f'Fee Drag ({fee_pct[-1]:.1f}%)',
                     zorder=1)

    # Add fold boundaries
    for i, (b0, b1) in enumerate(oos_bar_indices):
        if b0 < len(all_dates):
            ax1.axvline(all_dates[b0], color='#999999', linewidth=0.3,
                       alpha=0.3, linestyle=':')

    # Drawdown shading
    peak_net = np.maximum.accumulate(net_pct)
    dd_pct = net_pct - peak_net
    ax1.fill_between(stitched_dates, net_pct, peak_net,
                     where=(dd_pct < 0), color='#e74c3c', alpha=0.08)

    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_title(f'Walk-Forward Aggregate OOS Equity Curve\n'
                  f'Sharpe: {agg_sr:+.2f} | Return: {agg_ret*100:+.1f}% | '
                  f'{len(fold_results)} folds | {FEES_BPS:.0f} bps fees',
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('Cumulative Return (%)', fontsize=11)
    ax1.legend(fontsize=9, loc='upper left', framealpha=0.8)
    ax1.grid(True, alpha=0.2)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

    # Position count subplot
    ax2.fill_between(stitched_dates, stitched_pos_count,
                     color='#9b59b6', alpha=0.3)
    ax2.plot(stitched_dates, stitched_pos_count,
             color='#9b59b6', linewidth=0.6, alpha=0.8)
    ax2.set_title('Active Positions Over Time', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('# Positions', fontsize=11)
    ax2.set_ylim(0, max(stitched_pos_count.max() * 1.15, 1))
    ax2.grid(True, alpha=0.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

    fig.autofmt_xdate()
    plt.tight_layout()
    p2 = os.path.join(save_dir, 'wf_aggregate_oos_pnl.png')
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"  Chart saved: {p2}", flush=True)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Agent 2: 5m Portfolio Construction")
    parser.add_argument("--split", default="val", choices=["train", "val", "test", "trainval"])
    ALL_METHODS = ["equal", "ic_weighted", "rank", "ml_ridge", "ml_gbt", "ml_gbt_class", "ml_gbt_entry", "concordance", "ic_rolling", "hurdle", "asym_loss", "quantile", "ts_autonorm", "lgbm", "billion_alphas"]
    parser.add_argument("--combine", default="ic_weighted", choices=ALL_METHODS)
    parser.add_argument("--entry", type=float, default=None)
    parser.add_argument("--exit", type=float, default=None)
    parser.add_argument("--hold", type=int, default=None)
    parser.add_argument("--fees", type=float, default=None, help="Fees in bps (default 10)")
    parser.add_argument("--optimize", action="store_true", help="Grid search thresholds")
    parser.add_argument("--walkforward", action="store_true", help="Walk-forward cross-validation")
    parser.add_argument("--wf-train-days", type=int, default=6, help="Walk-forward train window (days)")
    parser.add_argument("--wf-test-days", type=int, default=3, help="Walk-forward test window (days)")
    parser.add_argument("--objective", default="sharpe", choices=["sharpe", "return", "ic"])
    parser.add_argument("--compare", action="store_true", help="Compare combination methods")
    parser.add_argument("--top-n", type=int, default=None, help="Use only top N alphas by IC")
    parser.add_argument("--max-pos", type=int, default=0, help="Max concurrent positions (0=unlimited)")
    parser.add_argument("--sizing", default="conviction", choices=["equal", "conviction"],
                        help="Position sizing: equal or conviction (signal-proportional)")
    parser.add_argument("--top-liquid", type=int, default=0,
                        help="Only use top N tickers by median quote_volume (0=all)")
    parser.add_argument("--no-vol-target", action="store_true", help="Disable vol targeting")
    parser.add_argument("--target-horizon", type=int, default=36,
                        help="ML Ridge forward return horizon in bars (default 36=3h)")
    parser.add_argument("--enriched", action="store_true",
                        help="Use enriched features for ML Ridge (momentum + interactions + vol regime)")
    parser.add_argument("--pca", type=int, default=0,
                        help="PCA neutralization: remove top N return PCs from signal (0=off, 3=recommended)")
    parser.add_argument("--exchange", default="binance", choices=["binance", "kucoin"],
                        help="Exchange data source (default: binance)")
    parser.add_argument("--universe", default="TOP100", choices=["TOP100", "TOP50", "TOP20"],
                        help="Universe size (default: TOP100). Filters alphas from DB and selects universe file.")
    parser.add_argument("--alpha-ids", type=str, default=None,
                        help="Comma-separated alpha IDs to use (e.g. '1,6,9,11,17,18')")
    parser.add_argument("--pca-hedge", type=float, default=0.0,
                        help="PCA risk hedge strength (0=off, 1.0=moderate, 5.0=aggressive)")
    parser.add_argument("--ts-gamma", type=float, default=0.01,
                        help="EWMA decay for ts_autonorm (default 0.01 = 100-bar window)")
    parser.add_argument("--strategy", default="threshold", choices=["threshold", "continuous"],
                        help="Portfolio strategy: threshold (entry/exit) or continuous (signal-proportional)")
    parser.add_argument("--trade-buffer", type=float, default=0.0,
                        help="Trade buffer for continuous strategy: min weight change to execute (reduces turnover)")
    parser.add_argument("--rebalance-every", type=int, default=1,
                        help="Rebalance every N bars (1=every bar, 12=hourly, 288=daily). Reduces turnover.")
    parser.add_argument("--pos-smooth", type=float, default=0.0,
                        help="Position smoothing EMA span for continuous strategy (0=off, 12=moderate)")
    parser.add_argument("--equal-weight", action="store_true",
                        help="Use equal weighting instead of IC weighting in ts_autonorm (better for tail-trading)")
    parser.add_argument("--signal-smooth", type=int, default=0,
                        help="EMA span to smooth composite signal (reduces whipsaw). 0=off, 12=1h, 36=3h")
    parser.add_argument("--concordance", action="store_true",
                        help="Boost signal when multiple alphas agree on direction")
    parser.add_argument("--rank-norm", action="store_true",
                        help="Use rank percentile -> inverse-normal instead of z-score")
    parser.add_argument("--beta-hedge", action="store_true",
                        help="Regress out BTC beta from signal for market-neutrality")
    parser.add_argument("--signal-hedge", type=int, default=0,
                        help="PCA signal hedge: remove top N eigenvectors of return covariance from signal. "
                             "Generalizes beta-hedge to N factors. 0=off, 3=recommended.")
    parser.add_argument("--rolling-ic", type=int, default=0,
                        help="Rolling IC EMA span for causal IC weighting. 0=use static DB IC (default). "
                             "288=1-day EMA, 576=2-day EMA. Eliminates IC lookahead bias.")
    parser.add_argument("--rolling-return", type=int, default=0,
                        help="Rolling return EMA span for causal return weighting. 0=off (default). "
                             "1440=5-day EMA. Weights alphas by realized L/S returns instead of IC.")
    parser.add_argument("--ba-lookback", type=int, default=15,
                        help="billion_alphas: lookback in daily alpha returns (default 15). Must be < N alphas.")
    parser.add_argument("--ba-retrain", type=int, default=288,
                        help="billion_alphas: retrain every N bars (default 288 = 1 day).")
    args = parser.parse_args()
    
    # Load alphas filtered by universe
    alphas = load_alphas(universe=args.universe)
    if not alphas:
        print("No alphas found in DB. Run eval_alpha_5m.py first.")
        return
    
    # Prune to top N alphas by IC
    if args.top_n and args.top_n < len(alphas):
        alphas = alphas[:args.top_n]
        print(f"\n  Pruned to top {args.top_n} alphas by IC from {DB_PATH}", flush=True)
    elif args.alpha_ids:
        keep_ids = set(int(x.strip()) for x in args.alpha_ids.split(","))
        alphas = [a for a in alphas if a[0] in keep_ids]
        print(f"\n  Filtered to {len(alphas)} alphas by ID: {sorted(keep_ids)}", flush=True)
    else:
        print(f"\n  Loaded {len(alphas)} alphas from {DB_PATH}", flush=True)
    for a in alphas:
        print(f"    #{a[0]:2d} IC={a[2]:+.05f} SR={a[3]:+.2f} | {a[1][:50]}")
    
    # Set exchange and universe (must be before data loading)
    global EXCHANGE, PORTFOLIO_UNIVERSE
    EXCHANGE = args.exchange
    PORTFOLIO_UNIVERSE = args.universe
    
    # Set fees (KuCoin market maker = 3bps default)
    global FEES_BPS
    if args.fees is not None:
        FEES_BPS = args.fees
    elif EXCHANGE == "kucoin":
        FEES_BPS = 3.0  # KuCoin market maker tier
    global TOP_LIQUID
    TOP_LIQUID = args.top_liquid
    global _ML_TARGET_HORIZON, _ML_ENRICHED, _TS_GAMMA, _EQUAL_WEIGHT
    global _SIGNAL_SMOOTH, _CONCORDANCE, _RANK_NORM, _BETA_HEDGE
    _ML_TARGET_HORIZON = args.target_horizon
    _ML_ENRICHED = args.enriched
    _TS_GAMMA = args.ts_gamma
    _EQUAL_WEIGHT = args.equal_weight
    _SIGNAL_SMOOTH = args.signal_smooth
    _CONCORDANCE = args.concordance
    _RANK_NORM = args.rank_norm
    _BETA_HEDGE = args.beta_hedge
    global _SIGNAL_HEDGE
    _SIGNAL_HEDGE = args.signal_hedge
    global _ROLLING_IC, _ROLLING_RETURN
    _ROLLING_IC = args.rolling_ic
    _ROLLING_RETURN = args.rolling_return
    vol_t = not args.no_vol_target
    print(f"  Exchange: {EXCHANGE.upper()} | Fees: {FEES_BPS:.0f} bps | Sizing: {args.sizing} | Max pos: {args.max_pos if args.max_pos > 0 else 'unlimited'} | Liquid: top {args.top_liquid if args.top_liquid > 0 else 'all'}", flush=True)
    print(f"  Vol target: {vol_t} | ML horizon: {args.target_horizon} | PCA: {args.pca} | PCA hedge: {args.pca_hedge}", flush=True)
    
    # Walk-forward mode -- runs its own data loading
    if args.walkforward:
        walk_forward_validate(
            alphas, objective=args.objective, combine=args.combine,
            train_days=args.wf_train_days, test_days=args.wf_test_days,
            fixed_entry=args.entry, fixed_exit=args.exit, fixed_hold=args.hold,
            max_positions=args.max_pos, sizing_mode=args.sizing,
            vol_target=vol_t,
            pca_components=args.pca, pca_hedge=args.pca_hedge,
            fees_bps=args.fees,
        )
        return
    
    # Load data
    matrices, universe = load_data(args.split)
    close = matrices.get("close")
    returns = close.pct_change() if close is not None else matrices.get("returns")
    returns_pnl = matrices.get("returns_clean", returns)  # winsorized for PnL
    
    if args.compare:
        # Compare all combination methods with specified thresholds
        entry = args.entry if args.entry is not None else ENTRY_THRESHOLD
        exit_val = args.exit if args.exit is not None else EXIT_THRESHOLD
        hold = args.hold if args.hold is not None else MIN_HOLD_BARS
        print(f"\n  Comparing ALL combination methods on {args.split} (entry={entry} exit={exit_val} hold={hold}):", flush=True)
        for method in ALL_METHODS:
            print(f"\n  --- {method.upper()} ---", flush=True)
            if method in ML_COMBINERS:
                composite = ML_COMBINERS[method](alphas, matrices, universe, returns)
            else:
                composite = compute_composite_signal(alphas, matrices, universe, method=method)
            if composite is None:
                print(f"  {method} failed to build composite")
                continue
            r = run_threshold_strategy(composite, returns_pnl, universe,
                                       entry_threshold=entry, exit_threshold=exit_val,
                                       min_hold_bars=hold,
                                       fees_bps=FEES_BPS,
                                       max_positions=args.max_pos,
                                       sizing_mode=args.sizing,
                                       vol_target=vol_t)
            print_result(r, f"{method.upper()}")
        return
    
    # Build composite signal
    print(f"\n  Building composite signal ({args.combine})...", flush=True)
    if args.combine in ML_COMBINERS:
        if args.combine == "ml_ridge":
            composite = ML_COMBINERS[args.combine](alphas, matrices, universe, returns,
                                                    target_horizon=_ML_TARGET_HORIZON,
                                                    use_enriched_features=_ML_ENRICHED)
        elif args.combine == "ts_autonorm":
            composite = ML_COMBINERS[args.combine](alphas, matrices, universe, returns,
                                                    gamma=_TS_GAMMA,
                                                    ic_weighted=not _EQUAL_WEIGHT,
                                                    signal_smooth=_SIGNAL_SMOOTH,
                                                    concordance_boost=_CONCORDANCE,
                                                    rank_norm=_RANK_NORM,
                                                    beta_hedge=_BETA_HEDGE,
                                                    signal_hedge=_SIGNAL_HEDGE,
                                                    rolling_ic=_ROLLING_IC,
                                                    rolling_return=_ROLLING_RETURN)
        elif args.combine == "lgbm":
            composite = ML_COMBINERS[args.combine](alphas, matrices, universe, returns,
                                                    gamma=_TS_GAMMA)
        elif args.combine == "billion_alphas":
            composite = ML_COMBINERS[args.combine](alphas, matrices, universe, returns,
                                                    lookback_days=args.ba_lookback,
                                                    retrain_every=args.ba_retrain,
                                                    gamma=_TS_GAMMA,
                                                    signal_smooth=_SIGNAL_SMOOTH)
        else:
            composite = ML_COMBINERS[args.combine](alphas, matrices, universe, returns)
    else:
        composite = compute_composite_signal(alphas, matrices, universe, method=args.combine)
    if composite is None:
        print("  Failed to build composite signal")
        return
    
    # Apply PCA neutralization if requested
    if args.pca > 0:
        composite = pca_neutralize_signal(composite, returns, n_components=args.pca)
    
    # Apply signal smoothing (EMA) to reduce turnover — applies to ALL combiners
    if _SIGNAL_SMOOTH > 0 and args.combine not in ("ts_autonorm", "billion_alphas"):
        # ts_autonorm and billion_alphas apply smoothing internally
        print(f"  Signal smoothing: EMA span={_SIGNAL_SMOOTH} ({_SIGNAL_SMOOTH*5/60:.1f}h)", flush=True)
        composite = composite.ewm(span=_SIGNAL_SMOOTH, min_periods=1).mean()
    
    print(f"  Composite shape: {composite.shape}", flush=True)
    print(f"  Composite stats: mean={composite.values[np.isfinite(composite.values)].mean():.4f} "
          f"std={composite.values[np.isfinite(composite.values)].std():.4f}", flush=True)
    
    if args.optimize:
        best = optimize_thresholds(composite, returns_pnl, universe,
                                   split_label=args.split,
                                   objective=args.objective)
        # Run best params and print full result
        r = run_threshold_strategy(
            composite, returns_pnl, universe,
            entry_threshold=best["entry"],
            exit_threshold=best["exit"],
            min_hold_bars=best["hold"],
            fees_bps=FEES_BPS,
            max_positions=args.max_pos,
            sizing_mode=args.sizing,
            vol_target=vol_t,
            pca_hedge=args.pca_hedge,
        )
        print_result(r, f"BEST on {args.split}")
        plot_results(r, composite, returns_pnl, label=f"BEST {args.combine} on {args.split}")
    else:
        if args.strategy == "continuous":
            # Continuous signal-proportional portfolio
            r = run_continuous_strategy(
                composite, returns_pnl, universe,
                fees_bps=FEES_BPS,
                max_positions=args.max_pos,
                trade_buffer=args.trade_buffer,
                position_smooth=args.pos_smooth,
                rebalance_every=args.rebalance_every,
            )
            print_result(r, f"Continuous Strategy on {args.split}")
            plot_results(r, composite, returns_pnl, label=f"{args.combine} continuous on {args.split}")
        else:
            # Single run with specified or default thresholds
            entry = args.entry if args.entry is not None else ENTRY_THRESHOLD
            exit_val = args.exit if args.exit is not None else EXIT_THRESHOLD
            hold = args.hold if args.hold is not None else MIN_HOLD_BARS
            
            r = run_threshold_strategy(
                composite, returns_pnl, universe,
                entry_threshold=entry,
                exit_threshold=exit_val,
                min_hold_bars=hold,
                fees_bps=FEES_BPS,
                max_positions=args.max_pos,
                sizing_mode=args.sizing,
                vol_target=vol_t,
                pca_hedge=args.pca_hedge,
            )
            print_result(r, f"Threshold Strategy on {args.split}")
            plot_results(r, composite, returns_pnl, label=f"{args.combine} on {args.split}")


if __name__ == "__main__":
    main()
