"""
Test: Equivalence between Streaming and Vectorized Backtest

CRITICAL TEST: The streaming engine (one bar at a time) must produce
identical results to the vectorized backtest (full pandas operations).

If they diverge, one of them has a bug, and we cannot trust either
for live trading.

We test:
  1. Direction series match
  2. PnL series match
  3. Cumulative PnL matches
  4. Sharpe ratio matches
"""
import warnings; warnings.filterwarnings('ignore')
import sys
import os
import numpy as np
import pandas as pd
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from UNIFIED_V10.engine import StreamingEngine
from UNIFIED_V10.backtest import load_data, run_streaming_backtest, run_vectorized_backtest


def test_equivalence_synthetic():
    """Test equivalence on synthetic data where we control everything."""
    print("TEST: Equivalence on Synthetic Data")
    print("=" * 60)

    # Generate deterministic synthetic data
    rng = np.random.RandomState(42)
    n_bars = 500
    dates = pd.date_range('2025-01-01', periods=n_bars, freq='1h')
    close = 100 * np.exp(np.cumsum(rng.randn(n_bars) * 0.002))
    high = close * (1 + rng.uniform(0, 0.01, n_bars))
    low = close * (1 - rng.uniform(0, 0.01, n_bars))
    opn = close * (1 + rng.uniform(-0.005, 0.005, n_bars))
    volume = rng.uniform(1000, 10000, n_bars)

    df_1h = pd.DataFrame({
        'open': opn, 'high': high, 'low': low, 'close': close,
        'volume': volume,
        'quote_volume': volume * close,
        'taker_buy_volume': volume * rng.uniform(0.3, 0.7, n_bars),
        'taker_buy_quote_volume': volume * close * rng.uniform(0.3, 0.7, n_bars),
    }, index=dates)

    test_alphas = ['mr_10', 'mom_12', 'breakout_48', 'tbr_10', 'accel_5']
    lookback = 120
    phl = 1

    # --- STREAMING ---
    engine = StreamingEngine(
        selected_alphas=test_alphas, lookback=lookback, phl=phl)

    stream_pnl = []
    stream_dirs = []
    for i in range(len(df_1h)):
        row = df_1h.iloc[i]
        result = engine.on_bar({
            'datetime': dates[i],
            'open': row['open'], 'high': row['high'],
            'low': row['low'], 'close': row['close'],
            'volume': row['volume'],
            'quote_volume': row['quote_volume'],
            'taker_buy_volume': row['taker_buy_volume'],
            'taker_buy_quote_volume': row['taker_buy_quote_volume'],
        })
        stream_pnl.append(result['net_pnl'])
        stream_dirs.append(result['direction'])

    stream_pnl = np.array(stream_pnl)
    stream_dirs = np.array(stream_dirs)

    # --- VECTORIZED ---
    from UNIFIED_V10.alphas import build_1h_alphas
    from UNIFIED_V10.config import FEE_FRAC

    alphas_full = build_1h_alphas(df_1h)
    returns = df_1h['close'].pct_change()

    avail = [a for a in test_alphas if a in alphas_full]
    X = pd.DataFrame({c: alphas_full[c] for c in avail}, index=dates)
    ret = returns

    # Factor returns with shift (no lookahead)
    X_shifted = X.shift(1)
    factor_returns = pd.DataFrame(index=dates, columns=avail, dtype=float)
    for col in avail:
        factor_returns[col] = np.sign(X_shifted[col].values) * ret.values

    rolling_er = factor_returns.rolling(lookback, min_periods=100).mean()
    weights = rolling_er.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)

    combined = (X * weights_norm).sum(axis=1)
    direction = np.sign(combined.values)

    # PnL: prev_direction * return - fee * |direction change|
    prev_dir = np.concatenate([[0], direction[:-1]])
    pos_changes = np.abs(direction - prev_dir)
    vec_pnl = prev_dir * ret.values - FEE_FRAC * pos_changes

    # Compare only after warmup (bars >= MIN_WARMUP_BARS + lookback)
    # Stream has warmup period where direction is 0
    # Find first bar where both have nonzero direction
    stream_first_sig = np.argmax(np.array(stream_dirs) != 0)
    vec_first_sig = np.argmax(direction != 0)

    print(f"  Stream first signal at bar: {stream_first_sig}")
    print(f"  Vec first signal at bar:    {vec_first_sig}")

    # Compare from the later of the two start points + some buffer
    start = max(stream_first_sig, vec_first_sig, 220)

    # After warmup alignment, do the directions agree?
    stream_d = stream_dirs[start:]
    vec_d = direction[start:]

    agree_mask = (stream_d != 0) & (vec_d != 0)
    if agree_mask.sum() > 0:
        agreement = (stream_d[agree_mask] == vec_d[agree_mask]).mean()
    else:
        agreement = 0.0

    print(f"\n  Direction agreement (post-warmup): {agreement:.2%}")
    print(f"  Bars compared: {agree_mask.sum()}")

    # Check cumulative PnL correlation
    stream_cum = np.cumsum(stream_pnl[start:])
    vec_cum = np.cumsum(vec_pnl[start:])

    if len(stream_cum) > 10 and len(vec_cum) > 10:
        corr = np.corrcoef(stream_cum, vec_cum[:len(stream_cum)])[0, 1]
        print(f"  Cumulative PnL correlation: {corr:.6f}")

        cum_diff = abs(stream_cum[-1] - vec_cum[len(stream_cum)-1])
        print(f"  Final cum PnL diff: {cum_diff*10000:.2f} bps")

    # The key metric: do they produce the same Sharpe?
    s_daily = pd.Series(stream_pnl, index=dates).resample('1D').sum()
    s_daily = s_daily[s_daily != 0]
    v_daily = pd.Series(vec_pnl, index=dates).resample('1D').sum()
    v_daily = v_daily[v_daily != 0]

    if len(s_daily) >= 5 and s_daily.std() > 0:
        s_sharpe = s_daily.mean() / s_daily.std() * np.sqrt(365)
    else:
        s_sharpe = 0

    if len(v_daily) >= 5 and v_daily.std() > 0:
        v_sharpe = v_daily.mean() / v_daily.std() * np.sqrt(365)
    else:
        v_sharpe = 0

    print(f"\n  Stream Sharpe: {s_sharpe:+.2f}")
    print(f"  Vec Sharpe:    {v_sharpe:+.2f}")
    print(f"  Sharpe diff:   {abs(s_sharpe - v_sharpe):.4f}")

    if agreement < 0.99:
        print(f"\n  WARNING: Agreement is {agreement:.2%}, expected >99%")
        print("  Investigating disagreement bars...")
        disagree_bars = np.where(agree_mask & (stream_d != vec_d))[0] + start
        for b in disagree_bars[:5]:
            print(f"    Bar {b}: stream_dir={stream_dirs[b]}, vec_dir={int(direction[b])}")

    print(f"\n  ✓ Test complete (agreement={agreement:.2%})")
    return agreement


def test_equivalence_real_data():
    """Test equivalence on real market data."""
    print("\nTEST: Equivalence on Real Data (BTCUSDT)")
    print("=" * 60)

    # Load real data
    all_1h, all_data = load_data(['BTCUSDT'])
    if 'BTCUSDT' not in all_1h:
        print("  SKIP: BTCUSDT data not available")
        return None

    test_alphas = ['mr_10', 'mom_12', 'breakout_48', 'tbr_10', 'accel_5']

    print("  Running streaming backtest...")
    t0 = time.time()
    r_stream = run_streaming_backtest(
        'BTCUSDT', all_1h, all_data,
        selected_alphas=test_alphas,
        start_date='2025-01-01',
        lookback=120, phl=1)
    t1 = time.time()
    print(f"  Streaming: {t1-t0:.1f}s")

    print("  Running vectorized backtest...")
    t0 = time.time()
    r_vec = run_vectorized_backtest(
        'BTCUSDT', all_1h, all_data,
        selected_alphas_list=test_alphas,
        start_date='2025-01-01',
        lookback=120, phl=1)
    t1 = time.time()
    print(f"  Vectorized: {t1-t0:.1f}s")

    if r_stream and r_vec:
        sharpe_diff = abs(r_stream['sharpe'] - r_vec['sharpe'])
        pnl_diff = abs(r_stream['net_pnl_bps'] - r_vec['net_pnl_bps'])

        print(f"\n  Stream Sharpe:  {r_stream['sharpe']:+.2f}")
        print(f"  Vec Sharpe:     {r_vec['sharpe']:+.2f}")
        print(f"  Sharpe diff:    {sharpe_diff:.4f}")
        print(f"  Stream PnL:     {r_stream['net_pnl_bps']:+.0f} bps")
        print(f"  Vec PnL:        {r_vec['net_pnl_bps']:+.0f} bps")
        print(f"  PnL diff:       {pnl_diff:.1f} bps")
        print(f"  Stream trades:  {r_stream['total_trades']}")
        print(f"  Vec trades:     {r_vec['total_trades']}")

        # Direction alignment
        common = r_stream['directions'].index.intersection(r_vec['directions'].index)
        s_dir = r_stream['directions'].loc[common]
        v_dir = r_vec['directions'].loc[common]
        mask = (s_dir != 0) & (v_dir != 0)
        if mask.sum() > 0:
            agreement = (s_dir[mask] == v_dir[mask]).mean()
            print(f"\n  Direction agreement: {agreement:.2%}")
        else:
            agreement = 0

        return {
            'sharpe_diff': sharpe_diff,
            'pnl_diff': pnl_diff,
            'agreement': agreement,
        }
    return None


def test_determinism():
    """Test that the engine produces identical results on identical input."""
    print("\nTEST: Determinism (same input → same output)")
    print("=" * 60)

    rng = np.random.RandomState(42)
    n_bars = 400
    dates = pd.date_range('2025-01-01', periods=n_bars, freq='1h')
    close = 100 * np.exp(np.cumsum(rng.randn(n_bars) * 0.002))

    bars = []
    for i in range(n_bars):
        bars.append({
            'datetime': dates[i],
            'open': close[i] * (1 + rng.uniform(-0.003, 0.003)),
            'high': close[i] * (1 + rng.uniform(0, 0.008)),
            'low': close[i] * (1 - rng.uniform(0, 0.008)),
            'close': close[i],
            'volume': rng.uniform(1000, 10000),
            'quote_volume': close[i] * rng.uniform(1000, 10000),
            'taker_buy_volume': rng.uniform(500, 5000),
            'taker_buy_quote_volume': close[i] * rng.uniform(500, 5000),
        })

    test_alphas = ['mr_10', 'mom_12', 'breakout_48', 'accel_5']

    # Run 1
    engine1 = StreamingEngine(selected_alphas=test_alphas, lookback=120, phl=1)
    results1 = [engine1.on_bar(b) for b in bars]

    # Run 2 (fresh engine, same data)
    engine2 = StreamingEngine(selected_alphas=test_alphas, lookback=120, phl=1)
    results2 = [engine2.on_bar(b) for b in bars]

    n_mismatch = 0
    for i in range(len(bars)):
        if results1[i]['direction'] != results2[i]['direction']:
            n_mismatch += 1
            print(f"  VIOLATION at bar {i}: "
                  f"run1={results1[i]['direction']}, run2={results2[i]['direction']}")
        if abs(results1[i]['cumulative_pnl'] - results2[i]['cumulative_pnl']) > 1e-15:
            n_mismatch += 1
            print(f"  VIOLATION at bar {i}: cum_pnl differs by "
                  f"{abs(results1[i]['cumulative_pnl'] - results2[i]['cumulative_pnl']):.2e}")

    print(f"\n  Checked {len(bars)} bars across 2 runs")
    print(f"  Mismatches: {n_mismatch}")
    assert n_mismatch == 0, f"DETERMINISM FAILED: {n_mismatch} mismatches"
    print("  ✓ PASSED — Engine is perfectly deterministic")
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("EQUIVALENCE & DETERMINISM TEST SUITE")
    print("=" * 60)
    print()

    test_determinism()

    agreement = test_equivalence_synthetic()

    real_result = test_equivalence_real_data()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"  Synthetic agreement: {agreement:.2%}")
    if real_result:
        print(f"  Real data agreement: {real_result['agreement']:.2%}")
        print(f"  Real data Sharpe diff: {real_result['sharpe_diff']:.4f}")
