"""
Test: No Lookahead Bias

CRITICAL TEST: Verifies that alpha values computed at time T
depend ONLY on data from bars 0..T (inclusive).

Method:
  1. Compute alpha at time T using full data [0..N]
  2. Compute alpha at time T using truncated data [0..T]
  3. Assert they are IDENTICAL

This catches any accidental use of future data in alpha computation.
"""
import warnings; warnings.filterwarnings('ignore')
import sys
import os
import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from UNIFIED_V10.alphas import build_1h_alphas, build_htf_signals
from UNIFIED_V10.config import DATA_DIR, AGG_RULES


def generate_synthetic_data(n_bars=500, seed=42):
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range('2025-01-01', periods=n_bars, freq='1h')

    close = 100 * np.exp(np.cumsum(rng.randn(n_bars) * 0.002))
    high = close * (1 + rng.uniform(0, 0.01, n_bars))
    low = close * (1 - rng.uniform(0, 0.01, n_bars))
    opn = close * (1 + rng.uniform(-0.005, 0.005, n_bars))
    volume = rng.uniform(1000, 10000, n_bars)

    df = pd.DataFrame({
        'open': opn, 'high': high, 'low': low, 'close': close,
        'volume': volume,
        'quote_volume': volume * close,
        'taker_buy_volume': volume * rng.uniform(0.3, 0.7, n_bars),
        'taker_buy_quote_volume': volume * close * rng.uniform(0.3, 0.7, n_bars),
    }, index=dates)
    return df


def test_1h_alpha_no_lookahead():
    """Test that 1H alphas at time T don't use future data."""
    print("TEST: 1H Alpha No-Lookahead")
    print("="*60)

    df = generate_synthetic_data(500)

    # Compute alphas on FULL data
    alphas_full = build_1h_alphas(df)

    # Check several time points
    test_points = [200, 250, 300, 350, 400, 450]
    n_alphas_tested = 0
    n_violations = 0

    for T in test_points:
        # Compute alphas on TRUNCATED data [0..T]
        df_trunc = df.iloc[:T+1].copy()
        alphas_trunc = build_1h_alphas(df_trunc)

        for name in alphas_full:
            if name not in alphas_trunc:
                continue

            full_val = alphas_full[name].iloc[T]
            trunc_val = alphas_trunc[name].iloc[-1]  # Last value of truncated

            n_alphas_tested += 1

            if np.isnan(full_val) and np.isnan(trunc_val):
                continue  # Both NaN is OK

            if np.isnan(full_val) != np.isnan(trunc_val):
                print(f"  VIOLATION at T={T}, alpha={name}: "
                      f"full={full_val}, trunc={trunc_val}")
                n_violations += 1
                continue

            if abs(full_val - trunc_val) > 1e-10:
                print(f"  VIOLATION at T={T}, alpha={name}: "
                      f"full={full_val:.8f}, trunc={trunc_val:.8f}, "
                      f"diff={abs(full_val-trunc_val):.2e}")
                n_violations += 1

    print(f"\n  Tested {n_alphas_tested} alpha-timepoint combinations")
    print(f"  Violations: {n_violations}")
    assert n_violations == 0, f"LOOKAHEAD BIAS DETECTED: {n_violations} violations!"
    print("  ✓ PASSED — No lookahead bias in 1H alphas")
    return True


def test_htf_alpha_no_lookahead():
    """Test that HTF alphas at time T don't use future data."""
    print("\nTEST: HTF Alpha No-Lookahead")
    print("="*60)

    df_1h = generate_synthetic_data(500)
    agg = {k: v for k, v in AGG_RULES.items() if k in df_1h.columns}

    n_violations = 0
    n_tested = 0

    for freq, prefix in [('2h', 'h2'), ('4h', 'h4')]:
        df_htf = df_1h.resample(freq).agg(agg).dropna()

        # Full computation
        alphas_full = build_htf_signals(df_htf, df_1h, prefix, shift_n=1)

        for T in [200, 300, 400]:
            # Truncated
            df_1h_trunc = df_1h.iloc[:T+1]
            df_htf_trunc = df_1h_trunc.resample(freq).agg(agg).dropna()
            alphas_trunc = build_htf_signals(df_htf_trunc, df_1h_trunc,
                                              prefix, shift_n=1)

            for name in alphas_full:
                if name not in alphas_trunc:
                    continue
                n_tested += 1

                full_val = alphas_full[name].iloc[T]
                trunc_val = alphas_trunc[name].iloc[-1]

                if np.isnan(full_val) and np.isnan(trunc_val):
                    continue

                if np.isnan(full_val) != np.isnan(trunc_val):
                    n_violations += 1
                    print(f"  VIOLATION: {prefix} T={T} {name}: "
                          f"full={full_val}, trunc={trunc_val}")
                    continue

                if abs(full_val - trunc_val) > 1e-10:
                    n_violations += 1
                    print(f"  VIOLATION: {prefix} T={T} {name}: "
                          f"diff={abs(full_val-trunc_val):.2e}")

    print(f"\n  Tested {n_tested} HTF alpha-timepoint combinations")
    print(f"  Violations: {n_violations}")
    assert n_violations == 0, f"HTF LOOKAHEAD BIAS: {n_violations} violations!"
    print("  ✓ PASSED — No lookahead bias in HTF alphas")
    return True


def test_engine_no_lookahead():
    """Test that the StreamingEngine direction at T doesn't depend on future data.

    Method: Run engine on [0..T], record direction at T.
    Then run engine on [0..T+100], check direction at T is the same.
    """
    print("\nTEST: StreamingEngine No-Lookahead")
    print("="*60)

    from UNIFIED_V10.engine import StreamingEngine

    df = generate_synthetic_data(500)
    test_alphas = ['mr_10', 'mom_12', 'breakout_48', 'tbr_10', 'accel_5']

    n_violations = 0
    test_points = [250, 300, 350, 400]

    for T in test_points:
        # Run engine on [0..T]
        engine_short = StreamingEngine(
            selected_alphas=test_alphas, lookback=120, phl=1)
        for i in range(T + 1):
            row = df.iloc[i]
            result_short = engine_short.on_bar({
                'datetime': df.index[i],
                'open': row['open'], 'high': row['high'],
                'low': row['low'], 'close': row['close'],
                'volume': row['volume'],
                'quote_volume': row.get('quote_volume', 0),
                'taker_buy_volume': row.get('taker_buy_volume', 0),
                'taker_buy_quote_volume': row.get('taker_buy_quote_volume', 0),
            })
        dir_short = result_short['direction']
        sig_short = result_short['signal_value']
        pnl_short = result_short['cumulative_pnl']

        # Run engine on [0..T+100]
        engine_long = StreamingEngine(
            selected_alphas=test_alphas, lookback=120, phl=1)
        result_at_T = None
        for i in range(min(T + 101, len(df))):
            row = df.iloc[i]
            result = engine_long.on_bar({
                'datetime': df.index[i],
                'open': row['open'], 'high': row['high'],
                'low': row['low'], 'close': row['close'],
                'volume': row['volume'],
                'quote_volume': row.get('quote_volume', 0),
                'taker_buy_volume': row.get('taker_buy_volume', 0),
                'taker_buy_quote_volume': row.get('taker_buy_quote_volume', 0),
            })
            if i == T:
                result_at_T = result
        dir_long = result_at_T['direction']
        sig_long = result_at_T['signal_value']
        pnl_long = result_at_T['cumulative_pnl']

        if dir_short != dir_long:
            print(f"  VIOLATION at T={T}: short_dir={dir_short}, long_dir={dir_long}")
            n_violations += 1

        if abs(sig_short - sig_long) > 1e-10:
            print(f"  VIOLATION at T={T}: signal diff={abs(sig_short-sig_long):.2e}")
            n_violations += 1

        if abs(pnl_short - pnl_long) > 1e-10:
            print(f"  VIOLATION at T={T}: cum_pnl diff={abs(pnl_short-pnl_long):.2e}")
            n_violations += 1

    print(f"\n  Tested {len(test_points)} time points")
    print(f"  Violations: {n_violations}")
    assert n_violations == 0, f"ENGINE LOOKAHEAD: {n_violations} violations!"
    print("  ✓ PASSED — StreamingEngine has no lookahead bias")
    return True


def test_direction_only_uses_past():
    """Test that direction[T] only depends on bars 0..T.

    This is the most direct test: if we change bar T+1's data,
    direction[T] should NOT change.
    """
    print("\nTEST: Direction Uses Only Past Data")
    print("="*60)

    from UNIFIED_V10.engine import StreamingEngine

    df = generate_synthetic_data(400)
    test_alphas = ['mr_10', 'mom_12', 'breakout_48', 'accel_5']

    # Run engine normally
    engine1 = StreamingEngine(selected_alphas=test_alphas, lookback=120, phl=1)
    results1 = []
    for i in range(len(df)):
        row = df.iloc[i]
        r = engine1.on_bar({
            'datetime': df.index[i],
            'open': row['open'], 'high': row['high'],
            'low': row['low'], 'close': row['close'],
            'volume': row['volume'],
            'quote_volume': row.get('quote_volume', 0),
            'taker_buy_volume': row.get('taker_buy_volume', 0),
            'taker_buy_quote_volume': row.get('taker_buy_quote_volume', 0),
        })
        results1.append(r)

    # Now modify future data and re-run
    df_modified = df.copy()
    # Corrupt bars 300+ with random data
    rng = np.random.RandomState(999)
    for i in range(300, len(df_modified)):
        df_modified.iloc[i, df_modified.columns.get_loc('close')] *= (1 + rng.randn() * 0.05)
        df_modified.iloc[i, df_modified.columns.get_loc('high')] *= (1 + rng.randn() * 0.05)
        df_modified.iloc[i, df_modified.columns.get_loc('low')] *= (1 + rng.randn() * 0.05)

    engine2 = StreamingEngine(selected_alphas=test_alphas, lookback=120, phl=1)
    results2 = []
    for i in range(len(df_modified)):
        row = df_modified.iloc[i]
        r = engine2.on_bar({
            'datetime': df_modified.index[i],
            'open': row['open'], 'high': row['high'],
            'low': row['low'], 'close': row['close'],
            'volume': row['volume'],
            'quote_volume': row.get('quote_volume', 0),
            'taker_buy_volume': row.get('taker_buy_volume', 0),
            'taker_buy_quote_volume': row.get('taker_buy_quote_volume', 0),
        })
        results2.append(r)

    # Check: results before bar 300 should be IDENTICAL
    n_violations = 0
    for i in range(300):
        if results1[i]['direction'] != results2[i]['direction']:
            print(f"  VIOLATION at bar {i}: dir1={results1[i]['direction']}, "
                  f"dir2={results2[i]['direction']}")
            n_violations += 1
        if abs(results1[i]['signal_value'] - results2[i]['signal_value']) > 1e-10:
            print(f"  VIOLATION at bar {i}: signal diff="
                  f"{abs(results1[i]['signal_value'] - results2[i]['signal_value']):.2e}")
            n_violations += 1

    print(f"\n  Checked 300 bars before modification point")
    print(f"  Violations: {n_violations}")
    assert n_violations == 0, f"FUTURE DATA LEAKAGE: {n_violations} violations!"
    print("  ✓ PASSED — Direction is independent of future data")
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("LOOKAHEAD BIAS TEST SUITE")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    for test_fn in [test_1h_alpha_no_lookahead,
                    test_htf_alpha_no_lookahead,
                    test_engine_no_lookahead,
                    test_direction_only_uses_past]:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    if failed == 0:
        print("ALL LOOKAHEAD TESTS PASSED ✓")
    else:
        print("*** LOOKAHEAD BIAS DETECTED — DO NOT TRADE ***")
