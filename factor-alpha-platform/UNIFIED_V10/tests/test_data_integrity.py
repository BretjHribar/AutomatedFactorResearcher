"""
Test: Data Integrity

Validates that input data meets quality requirements before
any signal computation. Bad data → bad signals → losses.
"""
import warnings; warnings.filterwarnings('ignore')
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from UNIFIED_V10.config import DATA_DIR, AGG_RULES, SYMBOLS


def test_no_gaps(df, sym, freq='15min', max_gap_multiple=3):
    """Check for timestamp gaps in data."""
    expected_delta = pd.Timedelta(freq)
    diffs = df.index.to_series().diff().dropna()
    gaps = diffs[diffs > expected_delta * max_gap_multiple]
    if len(gaps) > 0:
        print(f"  WARNING [{sym}]: {len(gaps)} gaps found (>{max_gap_multiple}x {freq})")
        for ts, gap in gaps.head(5).items():
            print(f"    {ts}: gap of {gap}")
        return False
    return True


def test_no_duplicates(df, sym):
    """Check for duplicate timestamps."""
    dupes = df.index.duplicated()
    if dupes.any():
        n = dupes.sum()
        print(f"  FAIL [{sym}]: {n} duplicate timestamps")
        return False
    return True


def test_ohlc_consistency(df, sym):
    """Check that OHLC relationships hold."""
    violations = 0

    # High must be >= max(open, close)
    max_oc = np.maximum(df['open'], df['close'])
    bad = (df['high'] < max_oc - 1e-8)
    if bad.any():
        violations += bad.sum()
        print(f"  FAIL [{sym}]: {bad.sum()} bars where high < max(open, close)")

    # Low must be <= min(open, close)
    min_oc = np.minimum(df['open'], df['close'])
    bad = (df['low'] > min_oc + 1e-8)
    if bad.any():
        violations += bad.sum()
        print(f"  FAIL [{sym}]: {bad.sum()} bars where low > min(open, close)")

    # High >= Low
    bad = (df['high'] < df['low'] - 1e-8)
    if bad.any():
        violations += bad.sum()
        print(f"  FAIL [{sym}]: {bad.sum()} bars where high < low")

    # Positive prices
    for col in ['open', 'high', 'low', 'close']:
        bad = (df[col] <= 0)
        if bad.any():
            violations += bad.sum()
            print(f"  FAIL [{sym}]: {bad.sum()} bars where {col} <= 0")

    return violations == 0


def test_no_extreme_returns(df, sym, max_pct=50):
    """Check for unrealistic price moves (>50% in one bar)."""
    returns = df['close'].pct_change().abs()
    extreme = returns > (max_pct / 100)
    if extreme.any():
        n = extreme.sum()
        print(f"  WARNING [{sym}]: {n} bars with >{max_pct}% return")
        for ts in returns[extreme].head(5).index:
            print(f"    {ts}: {returns.loc[ts]*100:.1f}%")
        return False
    return True


def test_volume_positive(df, sym):
    """Check that volume is non-negative."""
    bad = (df['volume'] < 0)
    if bad.any():
        print(f"  FAIL [{sym}]: {bad.sum()} bars with negative volume")
        return False
    return True


def test_sufficient_history(df, sym, min_bars=5000):
    """Check we have enough data for reliable backtesting."""
    n = len(df)
    if n < min_bars:
        print(f"  WARNING [{sym}]: Only {n} bars (need {min_bars})")
        return False
    print(f"  OK [{sym}]: {n} bars, range {df.index[0]} to {df.index[-1]}")
    return True


def test_1h_resampling_integrity(df_15m, sym):
    """Check that 15m → 1h resampling produces valid data."""
    agg = {k: v for k, v in AGG_RULES.items() if k in df_15m.columns}
    df_1h = df_15m.resample('1h').agg(agg).dropna()

    # Each 1h bar should come from exactly 4 15m bars (mostly)
    counts = df_15m.resample('1h')['close'].count()
    incomplete = (counts > 0) & (counts < 4)
    if incomplete.sum() > len(df_1h) * 0.01:
        print(f"  WARNING [{sym}]: {incomplete.sum()} incomplete 1h bars "
              f"({incomplete.sum()/len(df_1h)*100:.1f}%)")

    # Verify OHLC consistency after resampling
    return test_ohlc_consistency(df_1h, f"{sym}_1h")


def run_all_integrity_tests():
    """Run all data integrity tests on available symbols."""
    print("=" * 60)
    print("DATA INTEGRITY TEST SUITE")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    for sym in SYMBOLS:
        path = DATA_DIR / f'{sym}.parquet'
        if not path.exists():
            print(f"  SKIP [{sym}]: file not found at {path}")
            continue

        print(f"\n--- {sym} ---")
        df = pd.read_parquet(path)
        df = df.set_index('datetime').sort_index()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        for c in AGG_RULES:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        tests = [
            ("Sufficient history", lambda: test_sufficient_history(df, sym)),
            ("No duplicates", lambda: test_no_duplicates(df, sym)),
            ("No gaps", lambda: test_no_gaps(df, sym, '15min')),
            ("OHLC consistency", lambda: test_ohlc_consistency(df, sym)),
            ("No extreme returns", lambda: test_no_extreme_returns(df, sym)),
            ("Volume positive", lambda: test_volume_positive(df, sym)),
            ("1H resampling", lambda: test_1h_resampling_integrity(df, sym)),
        ]

        for name, test_fn in tests:
            try:
                result = test_fn()
                if result:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  ERROR [{sym}] {name}: {e}")
                failed += 1

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed/warned")
    return failed == 0


if __name__ == '__main__':
    run_all_integrity_tests()
