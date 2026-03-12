"""
Backtest runner — replays historical bars through the StreamingEngine.

Usage:
    python -m UNIFIED_V10.backtest
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path

from .engine import StreamingEngine
from .alphas import build_1h_alphas, build_htf_signals, build_cross_asset_signals
from .alphas import (sma, ema, stddev, ts_zscore, delta, ts_sum, ts_min, ts_max,
                     safe_div, correlation, decay_exp)
from .config import DATA_DIR, SYMBOLS, AGG_RULES, FEE_FRAC


def load_data(symbols=None):
    """Load and prepare all symbol data."""
    symbols = symbols or SYMBOLS
    all_1h = {}
    all_data = {}
    for sym in symbols:
        path = DATA_DIR / f'{sym}.parquet'
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping {sym}")
            continue
        df15 = pd.read_parquet(path)
        df15 = df15.set_index('datetime').sort_index()
        df15 = df15[~df15.index.duplicated(keep='last')]
        if df15.index.tz is not None:
            df15.index = df15.index.tz_localize(None)
        for c in AGG_RULES:
            if c in df15.columns:
                df15[c] = pd.to_numeric(df15[c], errors='coerce')
        all_data[sym] = df15
        all_1h[sym] = df15.resample('1h').agg(AGG_RULES).dropna()
    return all_1h, all_data


def run_streaming_backtest(sym, all_1h, all_data, selected_alphas,
                           lookback=120, phl=1,
                           start_date=None, end_date=None):
    """Run backtest by replaying 1H bars through the StreamingEngine.

    This is the REFERENCE implementation. Results from this are ground truth.

    Args:
        sym: symbol name (e.g. 'BTCUSDT')
        all_1h: dict of {symbol: 1H DataFrame}
        all_data: dict of {symbol: 15m DataFrame}
        selected_alphas: list of alpha names
        lookback: adaptive weight lookback
        phl: position halflife
        start_date: optional start date filter
        end_date: optional end date filter

    Returns:
        dict with: pnl_series, sharpe, total_trades, directions, etc.
    """
    df_1h = all_1h[sym]
    if start_date:
        df_1h = df_1h.loc[start_date:]
    if end_date:
        df_1h = df_1h.loc[:end_date]

    engine = StreamingEngine(
        selected_alphas=selected_alphas,
        lookback=lookback,
        phl=phl,
    )

    pnl_list = []
    direction_list = []
    timestamps = []

    for i, (ts, row) in enumerate(df_1h.iterrows()):
        bar = {
            'datetime': ts,
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row.get('volume', 0),
            'quote_volume': row.get('quote_volume', 0),
            'taker_buy_volume': row.get('taker_buy_volume', 0),
            'taker_buy_quote_volume': row.get('taker_buy_quote_volume', 0),
        }

        result = engine.on_bar(bar)

        pnl_list.append(result['net_pnl'])
        direction_list.append(result['direction'])
        timestamps.append(ts)

    pnl_series = pd.Series(pnl_list, index=timestamps)
    directions = pd.Series(direction_list, index=timestamps)

    # Compute metrics
    daily = pnl_series.resample('1D').sum()
    daily = daily[daily != 0]
    if len(daily) < 5 or daily.std() == 0:
        return None

    sharpe = daily.mean() / daily.std() * np.sqrt(365)
    pos_changes = np.abs(np.diff(np.concatenate([[0], directions.values])))
    total_trades = int(pos_changes.sum())
    net_pnl_bps = pnl_series.sum() * 10000

    return {
        'sharpe': sharpe,
        'total_trades': total_trades,
        'net_pnl_bps': net_pnl_bps,
        'pnl_series': pnl_series,
        'directions': directions,
        'daily_pnl': daily,
    }


def run_vectorized_backtest(sym, all_1h, all_data, selected_alphas_list,
                            lookback=120, phl=1,
                            start_date=None, end_date=None):
    """Run the VECTORIZED backtest (fast, for comparison).

    This uses the same alpha functions but with pandas vectorization.
    Results MUST match the streaming version.
    """
    df_1h = all_1h[sym]
    df15 = all_data[sym]

    if start_date:
        df_1h = df_1h.loc[start_date:]
        df15 = df15.loc[start_date:]
    if end_date:
        df_1h = df_1h.loc[:end_date]
        df15 = df15.loc[:end_date]

    df_2h = df15.resample('2h').agg(AGG_RULES).dropna()
    df_4h = df15.resample('4h').agg(AGG_RULES).dropna()
    df_8h = df15.resample('8h').agg(AGG_RULES).dropna()
    df_12h = df15.resample('12h').agg(AGG_RULES).dropna()
    returns = df_1h['close'].pct_change()

    # Build all alphas
    a1h = build_1h_alphas(df_1h)
    a2h = build_htf_signals(df_2h, df_1h, 'h2', shift_n=1)
    a4h = build_htf_signals(df_4h, df_1h, 'h4', shift_n=1)
    a8h = build_htf_signals(df_8h, df_1h, 'h8', shift_n=1)
    a12h = build_htf_signals(df_12h, df_1h, 'h12', shift_n=1)
    all_a = {**a1h, **a2h, **a4h, **a8h, **a12h}

    cols = [a for a in selected_alphas_list if a in all_a]
    if len(cols) < 2:
        return None

    X = pd.DataFrame({c: all_a[c] for c in cols}, index=df_1h.index)
    # NOTE: In the vectorized version, we do NOT shift X.
    # Instead, direction[T] = sign(combined[T]) and it earns return[T+1].
    # We account for this by shifting the PnL: pnl[T] = direction[T-1] * return[T]
    # This is equivalent to: X_shifted = X.shift(1), direction = sign(combined_shifted),
    #                        pnl = direction * return

    ret = returns.copy()
    valid = X.dropna().index.intersection(ret.dropna().index)
    X, ret = X.loc[valid], ret.loc[valid]
    n = len(valid)

    if n < lookback + 24:
        return None

    # Factor returns: sign(alpha[T]) * return[T+1] = sign(alpha[T-1]) * return[T] after shift
    # Using the standard vectorized approach with shift(1) on alpha matrix
    X_shifted = X.shift(1)  # alpha[T-1] at position T
    factor_returns = pd.DataFrame(index=valid, columns=cols, dtype=float)
    for col in cols:
        factor_returns[col] = np.sign(X_shifted[col].values) * ret.values

    rolling_er = factor_returns.rolling(lookback, min_periods=min(100, lookback)).mean()
    weights = rolling_er.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)

    if phl > 1:
        weights_smooth = weights_norm.ewm(halflife=phl, min_periods=1).mean()
        wsum2 = weights_smooth.sum(axis=1).replace(0, np.nan)
        weights_smooth = weights_smooth.div(wsum2, axis=0).fillna(0)
    else:
        weights_smooth = weights_norm

    # Direction uses CURRENT alpha (no shift)
    # But PnL uses PREVIOUS direction * current return
    combined = (X * weights_smooth).sum(axis=1)
    direction = np.sign(combined.values)

    # PnL: prev_direction * return - fee * |direction change|
    # Since direction[T] is decided at close of bar T and held during T+1:
    # pnl[T] = direction[T-1] * return[T] - fee * |direction[T] - direction[T-1]|
    prev_dir = np.concatenate([[0], direction[:-1]])
    pos_changes = np.abs(direction - prev_dir)
    pnl = prev_dir * ret.values - FEE_FRAC * pos_changes

    pnl_s = pd.Series(pnl, index=valid)
    directions_s = pd.Series(direction, index=valid)

    daily = pnl_s.resample('1D').sum()
    daily = daily[daily != 0]
    if len(daily) < 5 or daily.std() == 0:
        return None

    sharpe = daily.mean() / daily.std() * np.sqrt(365)
    return {
        'sharpe': sharpe,
        'total_trades': int(pos_changes.sum()),
        'net_pnl_bps': pnl_s.sum() * 10000,
        'pnl_series': pnl_s,
        'directions': directions_s,
        'daily_pnl': daily,
    }


if __name__ == '__main__':
    print("Loading data...")
    all_1h, all_data = load_data(['BTCUSDT'])

    # Use a small set of alphas for testing
    test_alphas = ['mr_10', 'mom_12', 'breakout_48', 'tbr_10', 'accel_5']

    print("\nRunning STREAMING backtest...")
    t0 = time.time()
    result_stream = run_streaming_backtest(
        'BTCUSDT', all_1h, all_data,
        selected_alphas=test_alphas,
        start_date='2025-01-01',
        lookback=120, phl=1)
    t1 = time.time()
    if result_stream:
        print(f"  Sharpe: {result_stream['sharpe']:+.2f}")
        print(f"  PnL: {result_stream['net_pnl_bps']:+.0f} bps")
        print(f"  Trades: {result_stream['total_trades']}")
        print(f"  Time: {t1-t0:.1f}s")

    print("\nRunning VECTORIZED backtest...")
    t0 = time.time()
    result_vec = run_vectorized_backtest(
        'BTCUSDT', all_1h, all_data,
        selected_alphas_list=test_alphas,
        start_date='2025-01-01',
        lookback=120, phl=1)
    t1 = time.time()
    if result_vec:
        print(f"  Sharpe: {result_vec['sharpe']:+.2f}")
        print(f"  PnL: {result_vec['net_pnl_bps']:+.0f} bps")
        print(f"  Trades: {result_vec['total_trades']}")
        print(f"  Time: {t1-t0:.1f}s")

    if result_stream and result_vec:
        print(f"\n{'='*60}")
        print(f"EQUIVALENCE CHECK:")
        print(f"  Sharpe diff:  {abs(result_stream['sharpe'] - result_vec['sharpe']):.6f}")
        print(f"  PnL diff:     {abs(result_stream['net_pnl_bps'] - result_vec['net_pnl_bps']):.2f} bps")
        print(f"  Trade diff:   {abs(result_stream['total_trades'] - result_vec['total_trades'])}")
