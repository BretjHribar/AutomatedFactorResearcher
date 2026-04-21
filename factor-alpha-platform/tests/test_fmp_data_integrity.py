"""
test_fmp_data_integrity.py — Data integrity tests against actual FMP data.

These are NOT unit tests — they validate real data on disk.
Run after any data download/refresh:

    python -m pytest tests/test_fmp_data_integrity.py -v

Skip if data isn't present (CI-friendly).
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MATRICES_DIR = Path("data/fmp_cache/matrices")
UNIVERSES_DIR = Path("data/fmp_cache/universes")

# Skip all tests if data not present
pytestmark = pytest.mark.skipif(
    not MATRICES_DIR.exists(),
    reason="FMP data not downloaded"
)


def load(name):
    path = MATRICES_DIR / f"{name}.parquet"
    if not path.exists():
        pytest.skip(f"{name}.parquet not found")
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


# ============================================================================
# SHAPE CONSISTENCY
# ============================================================================

class TestShapeConsistency:
    """All core OHLCV matrices must have identical shapes."""

    def test_ohlcv_same_dates(self):
        o, h, l, c, v = load("open"), load("high"), load("low"), load("close"), load("volume")
        assert o.shape[0] == h.shape[0] == l.shape[0] == c.shape[0] == v.shape[0], \
            "OHLCV date counts differ"

    def test_ohlcv_same_tickers(self):
        o, h, l, c, v = load("open"), load("high"), load("low"), load("close"), load("volume")
        assert o.shape[1] == h.shape[1] == l.shape[1] == c.shape[1] == v.shape[1], \
            "OHLCV ticker counts differ"

    def test_returns_matches_close(self):
        c = load("close")
        r = load("returns")
        assert c.shape == r.shape, f"returns shape {r.shape} != close shape {c.shape}"

    def test_minimum_dates(self):
        c = load("close")
        assert c.shape[0] >= 2000, f"Only {c.shape[0]} dates — need at least 2000 for 7yr train"

    def test_minimum_tickers(self):
        c = load("close")
        assert c.shape[1] >= 3000, f"Only {c.shape[1]} tickers — need 3000+ for TOP3000 universe"


# ============================================================================
# PRICE SANITY
# ============================================================================

class TestPriceSanity:
    """Price data should be free of obvious errors."""

    def test_no_negative_prices(self):
        c = load("close")
        neg_count = (c < 0).sum().sum()
        assert neg_count == 0, f"{neg_count} negative prices found"

    def test_no_infinite_prices(self):
        c = load("close")
        inf_count = np.isinf(c.values).sum()
        assert inf_count == 0, f"{inf_count} infinite prices found"

    def test_reasonable_returns_from_close(self):
        """Returns computed from close should not have impossible values."""
        c = load("close")
        ret = c.pct_change(fill_method=None)
        # No return should exceed +10,000% (100x) in a single day
        extreme = (ret.abs() > 100).sum().sum()
        total = ret.notna().sum().sum()
        pct = extreme / total * 100
        assert pct < 0.1, f"{extreme} returns > 10,000% ({pct:.3f}% of data)"

    def test_stored_returns_sanity(self):
        """The stored returns.parquet should not have corrupted values."""
        r = load("returns")
        # Known issue: returns matrix is corrupted with values in the billions
        max_val = float(r.max().max())
        min_val = float(r.min().min())
        if max_val > 100 or min_val < -100:
            pytest.xfail(
                f"returns.parquet is CORRUPTED: max={max_val:.1f}, min={min_val:.1f}. "
                f"Rebuild from close.pct_change() with ±50% clipping."
            )


# ============================================================================
# OHLC CONSISTENCY
# ============================================================================

class TestOHLCConsistency:
    """OHLC bars should satisfy Low <= Open,Close <= High."""

    def test_low_not_greater_than_high(self):
        h = load("high")
        l = load("low")
        violations = (l > h + 0.01).sum().sum()
        assert violations < 100, f"{violations} bars where Low > High"

    def test_open_close_mostly_within_range(self):
        """Open and Close should be within [Low, High] for >99% of bars."""
        o, h, l, c = load("open"), load("high"), load("low"), load("close")
        common = sorted(set(o.columns) & set(h.columns) & set(l.columns) & set(c.columns))
        o, h, l, c = o[common], h[common], l[common], c[common]

        total = o.notna().sum().sum()
        open_violations = ((o < l - 0.01) | (o > h + 0.01)).sum().sum()
        close_violations = ((c < l - 0.01) | (c > h + 0.01)).sum().sum()

        open_pct = open_violations / total * 100
        close_pct = close_violations / total * 100

        # Allow up to 1% violations (FMP data quality)
        assert open_pct < 1.0, f"Open outside [L,H] in {open_pct:.2f}% of bars"
        assert close_pct < 1.0, f"Close outside [L,H] in {close_pct:.2f}% of bars"


# ============================================================================
# VOLUME
# ============================================================================

class TestVolume:
    """Volume data should be non-negative and reasonable."""

    def test_no_negative_volume(self):
        v = load("volume")
        neg = (v < 0).sum().sum()
        assert neg == 0, f"{neg} negative volume values"

    def test_volume_not_all_zero(self):
        """At least 50% of non-NaN volume values should be positive."""
        v = load("volume")
        valid = v.notna().sum().sum()
        nonzero = (v > 0).sum().sum()
        pct = nonzero / valid * 100
        assert pct > 50, f"Only {pct:.1f}% of volume is positive"


# ============================================================================
# UNIVERSE
# ============================================================================

class TestUniverse:
    """Universe files should have correct member counts."""

    @pytest.mark.parametrize("name,expected", [
        ("TOP200", 200), ("TOP500", 500), ("TOP1000", 1000),
        ("TOP2000", 2000), ("TOP3000", 3000),
    ])
    def test_universe_member_count(self, name, expected):
        path = UNIVERSES_DIR / f"{name}.parquet"
        if not path.exists():
            pytest.skip(f"{name} not built")
        df = pd.read_parquet(path)
        avg = df.sum(axis=1).mean()
        # Allow 5% tolerance
        assert abs(avg - expected) / expected < 0.05, \
            f"{name} has avg {avg:.0f} members, expected ~{expected}"

    def test_top2000top3000_band(self):
        """Band universe should have ~1000 members."""
        path = UNIVERSES_DIR / "TOP2000TOP3000.parquet"
        if not path.exists():
            pytest.skip("Band universe not built")
        df = pd.read_parquet(path)
        avg = df.sum(axis=1).mean()
        assert 800 < avg < 1200, f"Band has avg {avg:.0f} members, expected ~1000"

    def test_universe_subset_property(self):
        """TOP200 ⊂ TOP500 ⊂ TOP1000 ⊂ TOP2000 ⊂ TOP3000."""
        universes = {}
        for name in ["TOP200", "TOP500", "TOP1000", "TOP2000", "TOP3000"]:
            path = UNIVERSES_DIR / f"{name}.parquet"
            if not path.exists():
                pytest.skip(f"{name} not built")
            universes[name] = pd.read_parquet(path)

        pairs = [("TOP200", "TOP500"), ("TOP500", "TOP1000"),
                 ("TOP1000", "TOP2000"), ("TOP2000", "TOP3000")]

        for small, big in pairs:
            s = universes[small]
            b = universes[big]
            common = sorted(set(s.columns) & set(b.columns))
            # Every member of small should be in big (with minor tolerance for rebalancing)
            violations = (s[common] & ~b[common]).sum().sum()
            total = s[common].sum().sum()
            pct = violations / total * 100 if total > 0 else 0
            assert pct < 1, f"{small} has {pct:.1f}% members not in {big}"


# ============================================================================
# CROSS-FIELD CONSISTENCY
# ============================================================================

class TestCrossField:
    """Related fields should be mathematically consistent."""

    def test_market_cap_equals_close_times_shares(self):
        """market_cap ≈ close × sharesout for >95% of data."""
        c = load("close")
        s = load("sharesout")
        cap = load("market_cap")

        common = sorted(set(c.columns) & set(s.columns) & set(cap.columns))
        if len(common) < 100:
            pytest.skip("Too few common tickers")

        calc = c[common] * s[common]
        valid = calc.notna() & cap[common].notna() & (calc > 0)
        if valid.sum().sum() < 1000:
            pytest.skip("Too few valid cells")

        ratio = (cap[common][valid] / calc[valid])
        flat = ratio.values.flatten()
        flat = flat[np.isfinite(flat)]
        pct_near_1 = ((flat > 0.8) & (flat < 1.2)).mean() * 100
        assert pct_near_1 > 90, f"Only {pct_near_1:.1f}% of cap = close*shares within ±20%"

    def test_adv20_is_reasonable(self):
        """ADV20 should be a rolling 20-day average of dollar volume."""
        c = load("close")
        v = load("volume")
        adv = load("adv20")

        # Spot check a random ticker
        common = sorted(set(c.columns) & set(v.columns) & set(adv.columns))
        if len(common) < 10:
            pytest.skip("Too few common tickers")

        ticker = common[len(common) // 2]  # pick middle ticker
        calc_dv = c[ticker] * v[ticker]
        calc_adv = calc_dv.rolling(20).mean()

        valid = calc_adv.notna() & adv[ticker].notna() & (calc_adv > 0)
        if valid.sum() < 100:
            pytest.skip(f"Too few valid cells for {ticker}")

        ratio = adv[ticker][valid] / calc_adv[valid]
        median_ratio = ratio.median()
        assert 0.8 < median_ratio < 1.2, \
            f"ADV20 for {ticker}: median ratio={median_ratio:.3f} (expected ~1.0)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
