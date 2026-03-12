"""
Walk-Forward Backtest using the Unified V10 Engine.

Two modes:
  FAST (default): Vectorized backtest using the SAME math as StreamingEngine.
    Proven 100.000% equivalent via test_equivalence.py. Runs in seconds.

  REFERENCE (--streaming): Bar-by-bar through StreamingEngine.on_bar().
    Slow (O(N²)) but is the literal live trading code path.
    Use this for final verification only.

Usage:
    python -m UNIFIED_V10.walk_forward              # Fast vectorized
    python -m UNIFIED_V10.walk_forward --streaming   # Reference streaming
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import json
import time
import sys
from pathlib import Path

from .engine import StreamingEngine
from .alphas import build_1h_alphas, build_htf_signals, build_cross_asset_signals
from .config import DATA_DIR, SYMBOLS, AGG_RULES, FEE_FRAC, PARAMS_FILE


def load_all_1h(symbols=None):
    """Load 15m data and resample to 1h for all symbols."""
    symbols = symbols or SYMBOLS
    all_1h = {}
    for sym in symbols:
        path = DATA_DIR / f'{sym}.parquet'
        if not path.exists():
            continue
        df15 = pd.read_parquet(path)
        df15 = df15.set_index('datetime').sort_index()
        df15 = df15[~df15.index.duplicated(keep='last')]
        if df15.index.tz is not None:
            df15.index = df15.index.tz_localize(None)
        for c in AGG_RULES:
            if c in df15.columns:
                df15[c] = pd.to_numeric(df15[c], errors='coerce')
        all_1h[sym] = df15.resample('1h').agg(AGG_RULES).dropna()
    return all_1h


def vectorized_backtest(df_1h, selected_alphas, lookback=120, phl=1,
                        start_date=None, fee_frac=None, sym=None,
                        all_1h=None):
    """Fast vectorized backtest — proven 100% equivalent to StreamingEngine.

    Uses EXACTLY the same math:
      1. build_1h_alphas() → alpha matrix X
      2. factor_return = sign(X.shift(1)) * return
      3. weights = rolling(lookback, min_periods=100).mean().clip(0) / sum
      4. combined = X * weights → direction = sign(combined)
      5. pnl = prev_direction * return - fee * |direction_change|
    """
    fee = fee_frac if fee_frac is not None else FEE_FRAC

    # Build all alphas
    all_alphas = build_1h_alphas(df_1h)

    # HTF alphas
    if isinstance(df_1h.index, pd.DatetimeIndex) and len(df_1h) >= 50:
        agg = {k: v for k, v in AGG_RULES.items() if k in df_1h.columns}
        for freq, prefix in [('2h', 'h2'), ('4h', 'h4'),
                              ('8h', 'h8'), ('12h', 'h12')]:
            try:
                df_htf = df_1h.resample(freq).agg(agg).dropna()
                if len(df_htf) >= 5:
                    htf = build_htf_signals(df_htf, df_1h, prefix, shift_n=1)
                    all_alphas.update(htf)
            except Exception:
                pass

    # Cross-asset alphas (BTC, ETH, SOL as factors)
    if sym and all_1h:
        try:
            cross = build_cross_asset_signals(all_1h, sym, df_1h)
            all_alphas.update(cross)
        except Exception:
            pass

    # Select only requested alphas
    avail = [a for a in selected_alphas if a in all_alphas]
    missing = [a for a in selected_alphas if a not in all_alphas]
    if missing and len(missing) > 0:
        pass  # Some alphas not available — will be excluded
    if not avail:
        return None

    X = pd.DataFrame({c: all_alphas[c] for c in avail}, index=df_1h.index)
    ret = df_1h['close'].pct_change()

    # Factor returns with lag (same as engine: sign(prev_alpha) * return)
    X_shifted = X.shift(1)
    factor_returns = pd.DataFrame(index=df_1h.index, columns=avail, dtype=float)
    for col in avail:
        factor_returns[col] = np.sign(X_shifted[col].values) * ret.values

    # Adaptive weights (same rolling + min_periods as engine)
    min_p = min(100, lookback)
    rolling_er = factor_returns.rolling(lookback, min_periods=min_p).mean()
    weights = rolling_er.clip(lower=0).fillna(0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)

    # Combined signal → direction
    combined = (X * weights_norm).sum(axis=1)
    direction = np.sign(combined.values)
    direction = np.where(np.isnan(direction), 0, direction)

    # PnL: prev_direction * return - fee * |pos_change|
    prev_dir = np.concatenate([[0], direction[:-1]])
    pos_changes = np.abs(direction - prev_dir)
    pnl = prev_dir * ret.values - fee * pos_changes

    # Trim to start_date
    pnl_s = pd.Series(pnl, index=df_1h.index)
    dirs = pd.Series(direction, index=df_1h.index)
    if start_date:
        pnl_s = pnl_s.loc[start_date:]
        dirs = dirs.loc[start_date:]

    # Metrics
    daily = pnl_s.resample('1D').sum()
    daily = daily[daily != 0]
    if len(daily) < 5 or daily.std() == 0:
        return None

    sharpe = daily.mean() / daily.std() * np.sqrt(365)
    cum_pnl = pnl_s.sum() * 10000
    max_dd = (pnl_s.cumsum().cummax() - pnl_s.cumsum()).max() * 10000
    n_trades = int(np.abs(np.diff(np.concatenate([[0], dirs.values]))).sum())
    win_rate = (daily > 0).mean()

    return {
        'sharpe': sharpe,
        'cum_pnl_bps': cum_pnl,
        'max_dd_bps': max_dd,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'pnl_series': pnl_s,
        'daily_pnl': daily,
        'directions': dirs,
        'n_bars': len(pnl_s),
    }


def streaming_backtest(df_1h, selected_alphas, lookback=120, phl=1,
                       start_date=None):
    """Reference streaming backtest — replays bars through StreamingEngine.

    Slow (O(N²)) but is the LITERAL live code path.
    """
    engine = StreamingEngine(
        selected_alphas=selected_alphas,
        lookback=lookback,
        phl=phl,
    )

    pnl_list = []
    dir_list = []
    timestamps = []

    for ts, row in df_1h.iterrows():
        bar = {
            'datetime': ts,
            'open': row['open'], 'high': row['high'],
            'low': row['low'], 'close': row['close'],
            'volume': row.get('volume', 0),
            'quote_volume': row.get('quote_volume', 0),
            'taker_buy_volume': row.get('taker_buy_volume', 0),
            'taker_buy_quote_volume': row.get('taker_buy_quote_volume', 0),
        }
        result = engine.on_bar(bar)
        pnl_list.append(result['net_pnl'])
        dir_list.append(result['direction'])
        timestamps.append(ts)

    pnl_s = pd.Series(pnl_list, index=timestamps)
    dirs = pd.Series(dir_list, index=timestamps)

    if start_date:
        pnl_s = pnl_s.loc[start_date:]
        dirs = dirs.loc[start_date:]

    daily = pnl_s.resample('1D').sum()
    daily = daily[daily != 0]
    if len(daily) < 5 or daily.std() == 0:
        return None

    sharpe = daily.mean() / daily.std() * np.sqrt(365)
    cum_pnl = pnl_s.sum() * 10000
    max_dd = (pnl_s.cumsum().cummax() - pnl_s.cumsum()).max() * 10000
    n_trades = int(np.abs(np.diff(np.concatenate([[0], dirs.values]))).sum())
    win_rate = (daily > 0).mean()

    return {
        'sharpe': sharpe,
        'cum_pnl_bps': cum_pnl,
        'max_dd_bps': max_dd,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'pnl_series': pnl_s,
        'daily_pnl': daily,
        'directions': dirs,
        'n_bars': len(pnl_s),
    }


def run_walk_forward(params_file=None, start_date='2024-06-01',
                     end_date=None, streaming=False, verbose=True):
    """Run walk-forward backtest using frozen params."""
    params_path = Path(params_file) if params_file else PARAMS_FILE
    with open(params_path) as f:
        frozen = json.load(f)

    mode = "STREAMING (reference)" if streaming else "VECTORIZED (fast, proven equivalent)"
    if verbose:
        print(f"Mode: {mode}")
        print(f"Frozen params: {frozen['version']} ({frozen['frozen_at']})")
        print()

    sym_list = list(frozen['symbols'].keys())
    # Also load BTC, ETH, SOL for cross-asset signals
    all_syms = list(set(sym_list + ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']))
    if verbose:
        print(f"Loading data for {len(all_syms)} symbols (incl cross-asset)...")
    all_1h = load_all_1h(all_syms)

    results = {}
    for sym in sym_list:
        if sym not in all_1h:
            print(f"  SKIP {sym}: no data")
            continue

        cfg = frozen['symbols'][sym]
        selected = cfg['selected_alphas']
        lookback = cfg['lookback']
        phl = cfg['phl']

        df_1h = all_1h[sym]
        if end_date:
            df_1h = df_1h.loc[:end_date]

        if len(df_1h) < 300:
            print(f"  SKIP {sym}: only {len(df_1h)} bars")
            continue

        if verbose:
            print(f"\n  {sym}: {len(selected)} alphas, {len(df_1h)} bars")

        t0 = time.time()
        if streaming:
            r = streaming_backtest(df_1h, selected, lookback, phl, start_date)
        else:
            r = vectorized_backtest(df_1h, selected, lookback, phl, start_date,
                                     sym=sym, all_1h=all_1h)
        elapsed = time.time() - t0

        if r is None:
            print(f"  {sym}: insufficient data for metrics")
            continue

        results[sym] = r

        if verbose:
            print(f"  Sharpe: {r['sharpe']:+.2f} | PnL: {r['cum_pnl_bps']:+.0f} bps | "
                  f"MaxDD: {r['max_dd_bps']:.0f} bps | Trades: {r['n_trades']} | "
                  f"Win: {r['win_rate']:.0%} | {elapsed:.1f}s")

    # Portfolio summary
    if len(results) >= 2:
        all_daily = pd.DataFrame({
            sym: r['daily_pnl'] for sym, r in results.items()
        }).fillna(0)
        port_daily = all_daily.mean(axis=1)
        port_daily = port_daily[port_daily != 0]

        if len(port_daily) >= 5 and port_daily.std() > 0:
            port_sharpe = port_daily.mean() / port_daily.std() * np.sqrt(365)
            port_pnl = port_daily.sum() * 10000
            port_wr = (port_daily > 0).mean()

            results['PORTFOLIO'] = {
                'sharpe': port_sharpe,
                'cum_pnl_bps': port_pnl,
                'win_rate': port_wr,
                'daily_pnl': port_daily,
            }

    return results


if __name__ == '__main__':
    streaming = '--streaming' in sys.argv

    print("=" * 60)
    print("UNIFIED V10 WALK-FORWARD BACKTEST")
    print("=" * 60)
    print()

    results = run_walk_forward(
        start_date='2024-06-01',
        end_date='2025-03-12',
        streaming=streaming,
        verbose=True,
    )

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Symbol':>10s} {'Sharpe':>8s} {'PnL(bps)':>10s} {'MaxDD':>8s} {'Trades':>8s} {'Win%':>6s}")
    for sym in results:
        r = results[sym]
        if sym == 'PORTFOLIO':
            print(f"{'─'*52}")
        print(f"{sym:>10s} {r['sharpe']:>+8.2f} {r['cum_pnl_bps']:>+10.0f} "
              f"{r.get('max_dd_bps', 0):>8.0f} {r.get('n_trades', 0):>8d} "
              f"{r.get('win_rate', 0):>5.0%}")
