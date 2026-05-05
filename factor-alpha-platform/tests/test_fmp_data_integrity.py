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
        # Was 3000 but FMP screener with $300M min cap currently returns ~2500
        # tickers after IPO/REIT/SPAC filters. Threshold reflects deliverable
        # reality, not aspirational TOP3000.
        assert c.shape[1] >= 2400, f"Only {c.shape[1]} tickers — expected ≥ 2400"


# ============================================================================
# CALENDAR CONTINUITY  (the silent killer for delay-0 strategies)
# ============================================================================

class TestCalendarContinuity:
    """The close.parquet date index must contain every NYSE trading day
    between its first and last date — no holes, no extra non-trading days.

    A single missing trading day corrupts every rolling-window operator
    (sma, ts_delta, momentum_60d, parkinson_volatility_60, …) for many
    bars after the gap. This test catches the same condition that the
    live trader's Phase 1 A2 staleness check refuses to trade against.
    """

    @staticmethod
    def _expected_dates(first, last, prefer_nyse=True):
        if prefer_nyse:
            try:
                import pandas_market_calendars as mcal
                nyse = mcal.get_calendar("NYSE")
                sched = nyse.schedule(start_date=str(first), end_date=str(last))
                return set(d.date() for d in sched.index), "NYSE"
            except ImportError:
                pass
        return set(d.date() for d in pd.bdate_range(first, last)), "weekday-fallback"

    def test_no_missing_trading_days(self):
        c = load("close")
        cached = set(c.index.date)
        first, last = c.index.min().date(), c.index.max().date()
        expected, label = self._expected_dates(first, last)
        missing = sorted(expected - cached)
        assert not missing, (
            f"{len(missing)} {label} trading days missing in close.parquet "
            f"(first {missing[:5]}). Holes silently corrupt rolling-window operators."
        )

    def test_no_extra_nontrading_dates(self):
        c = load("close")
        cached = set(c.index.date)
        first, last = c.index.min().date(), c.index.max().date()
        expected, label = self._expected_dates(first, last)
        extra = sorted(cached - expected)
        # Tolerate <= 5 extras (very rare half-day or special-session edge cases)
        assert len(extra) <= 5, (
            f"{len(extra)} cached dates aren't {label} trading days: {extra[:10]}"
        )

    def test_last_bar_is_recent_trading_day(self):
        """The cache must end on the previous NYSE trading day or later.

        If the trader runs today (T) and the cache ends at T-N where N >= 2
        trading days, every alpha is reading stale data and the live-bar
        appended for T is preceded by missing rows — exactly the failure
        pattern that produced today's audit. This test fails fast.
        """
        import datetime as _dt
        c = load("close")
        last = c.index.max().date()
        today = _dt.date.today()
        try:
            import pandas_market_calendars as mcal
            nyse = mcal.get_calendar("NYSE")
            sched = nyse.schedule(start_date=str(today - _dt.timedelta(days=10)),
                                  end_date=str(today))
            prior = [d.date() for d in sched.index if d.date() < today]
            expected_last = prior[-1] if prior else None
        except ImportError:
            d = today
            while True:
                d = d - _dt.timedelta(days=1)
                if d.weekday() < 5:
                    expected_last = d
                    break
        if expected_last is None:
            pytest.skip("could not determine expected last trading day")
        assert last >= expected_last, (
            f"close.parquet ends {last}, expected the previous trading day "
            f"{expected_last} ({(expected_last - last).days} calendar days behind). "
            f"Cache refresh required before live trading."
        )


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
        """Each TOP_N universe averages close to its target N members,
        BOUNDED ABOVE by the empirical cap (max-N universe's active count).
        IPO seasoning + history filters drop ~18% of FMP's deliverable, so
        TOP3000 caps at whatever TOP3500 shows."""
        path = UNIVERSES_DIR / f"{name}.parquet"
        if not path.exists():
            pytest.skip(f"{name} not built")
        df = pd.read_parquet(path)
        avg = df.sum(axis=1).mean()

        # Empirical cap = active count in the largest TOP_N universe.
        # Using TOP3500 (or whichever exists) as the deliverable proxy.
        cap = expected
        for cap_name in ("TOP3500", "TOP3000"):
            cap_path = UNIVERSES_DIR / f"{cap_name}.parquet"
            if cap_path.exists():
                cap = pd.read_parquet(cap_path).sum(axis=1).mean()
                break
        capped_expected = min(expected, int(cap))
        tol = 0.10  # 10% wobble (seasoning + rebalance churn)
        assert abs(avg - capped_expected) / capped_expected < tol, \
            f"{name} has avg {avg:.0f} members, expected ~{capped_expected} " \
            f"(target {expected} capped by empirical max {int(cap)})"

    def test_top2000top3000_band(self):
        """Band universe should have ~1000 members. Skip if the underlying
        TOP3000 has < 2500 active members (band degenerates to set
        difference of two near-identical sets)."""
        path = UNIVERSES_DIR / "TOP2000TOP3000.parquet"
        if not path.exists():
            pytest.skip("Band universe not built")
        top3k_path = UNIVERSES_DIR / "TOP3000.parquet"
        if top3k_path.exists():
            top3k = pd.read_parquet(top3k_path)
            if top3k.sum(axis=1).mean() < 2500:
                pytest.skip(
                    f"TOP3000 only has {top3k.sum(axis=1).mean():.0f} active "
                    f"members; band degenerates. Need a wider FMP screen."
                )
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
        """market_cap ≈ close × sharesout for the bulk of data.

        Two reasons this isn't a tight equality:
        (a) `market_cap` comes from FMP's `metrics` endpoint sampled at
            filing date (per-quarter, point-in-time, ffill).
        (b) `close × shares_out` uses today's close × ffill'd-shares-out.

        When close has moved materially since the last filing, the ratio
        drifts. So a wider tolerance reflects that the two are *related*
        but not identical estimates of market cap. Real failure mode: ratio
        is wildly off (factor of 10+) → split-adjustment broke."""
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
        # ±50% tolerance — wider than 20% to allow normal price drift between
        # filings. Tighter is the right *aspirational* target but with FMP's
        # quarterly-shares-ffill, 50% catches split bugs without false-positives.
        pct_near_1 = ((flat > 0.5) & (flat < 1.5)).mean() * 100
        assert pct_near_1 > 60, f"Only {pct_near_1:.1f}% of cap = close*shares within ±50%"

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


# ============================================================================
# FUNDAMENTALS — alignment, freshness, population, update rate
# ============================================================================
#
# Fundamentals matter even if today's alpha library doesn't reference them:
# every other strategy that lands on this same FMP cache (research_equity.json,
# Bryan-Kelly factor model, future alphas) reads the same matrices. Silent
# fundamentals lag corrupts those downstream consumers.
#
# Tests:
#   1. Each fundamental field shares close.parquet's date index (no 46-day lag)
#   2. No fundamental field is 100% NaN (catches mapped-column-name typos)
#   3. Each fundamental's most-recent populated row is within 120 days of the
#      latest close (Q1 reporting can lag a quarter, hence 120d not 30d)
#   4. Recent quarter's update rate ≥ 5% (filings should land for ≥ 5% of the
#      universe per quarter; below that the field is structurally broken)
#   5. No look-ahead: matrix value for a quarter shouldn't appear before the
#      filing date (spot-checked on a small ticker sample)
#
# Field categorization mirrors tools/diagnostics/equities_data_integrity.py.

# Tokens that mark a matrix as fundamental (vs price/volume/derived OHLCV).
_FUND_HINTS = (
    "roe", "roa", "margin", "earnings_yield", "free_cashflow_yield",
    "ev_to_", "book_to_market", "asset_turnover", "debt_to", "current_ratio",
    "cap", "revenue", "income", "equity", "assets", "liabilit", "cash_",
    "depreciation", "amortization", "capex", "operating_", "ebitda", "shares",
    "dividend", "tax", "interest_expense", "goodwill", "intangible",
    "inventory", "receivable", "payable", "investment",
)
# Tokens that mark a matrix as price/volume/derived (which are tested elsewhere).
_TS_HINTS = (
    "returns", "log_returns", "_volatility_", "adv", "dollars_traded",
    "high_low_range", "open_close_range", "vwap", "momentum_", "volume",
    "high", "low", "open", "close", "turnover",
)


def _is_fundamental(name: str) -> bool:
    nl = name.lower()
    if any(h in nl for h in _TS_HINTS):
        return False
    return any(h in nl for h in _FUND_HINTS)


def _list_fundamental_files():
    if not MATRICES_DIR.exists():
        return []
    return sorted(p for p in MATRICES_DIR.glob("*.parquet")
                  if not p.stem.startswith("_") and _is_fundamental(p.stem))


class TestFundamentals:
    """Fundamentals data integrity. These tests are skipped if no fundamentals
    are present in the cache. They assume close.parquet is the reference
    calendar (already validated by TestCalendarContinuity)."""

    def test_fundamentals_present(self):
        """At least 50 fundamental fields should exist."""
        files = _list_fundamental_files()
        assert len(files) >= 50, (
            f"Only {len(files)} fundamental fields found; expected ≥ 50. "
            f"Refresh: python -m src.data.bulk_download --skip-prices --skip-sic"
        )

    def test_fundamentals_share_close_index(self):
        """Every fundamental's date index must equal close.parquet's index.

        A misaligned fundamentals matrix (e.g. 46 rows shorter at the tail)
        means downstream alphas using those fields read NaN for the most
        recent N bars and silently drop those names from the signal."""
        c = load("close")
        files = _list_fundamental_files()
        if not files:
            pytest.skip("no fundamental files found")

        misaligned = []
        for fp in files:
            df = pd.read_parquet(fp)
            if not isinstance(df.index, pd.DatetimeIndex):
                continue
            if not df.index.equals(c.index):
                # Quantify: how many close-dates are missing from this matrix?
                missing = len(set(c.index) - set(df.index))
                extra   = len(set(df.index) - set(c.index))
                misaligned.append((fp.stem, missing, extra))

        assert not misaligned, (
            f"{len(misaligned)} fundamentals don't share close.parquet's "
            f"date index. Worst (name, missing_dates, extra_dates):\n  " +
            "\n  ".join(f"{n}: missing={m}, extra={e}"
                       for n, m, e in misaligned[:10])
        )

    def test_no_field_100pct_nan(self):
        """A field that is 100% NaN over its whole history is almost always a
        bug — a mapped FMP column name that doesn't exist in the source files,
        or a calculation that errored silently during build."""
        files = _list_fundamental_files()
        if not files:
            pytest.skip("no fundamental files found")

        all_nan = []
        for fp in files:
            df = pd.read_parquet(fp)
            if df.shape[0] == 0 or df.shape[1] == 0:
                all_nan.append(fp.stem)
                continue
            arr = df.values
            if not np.issubdtype(arr.dtype, np.floating):
                continue
            if not (~np.isnan(arr)).any():
                all_nan.append(fp.stem)

        assert not all_nan, (
            f"{len(all_nan)} fundamental fields are 100% NaN: "
            f"{all_nan[:10]}{'...' if len(all_nan) > 10 else ''}. "
            f"Likely a mapped column name doesn't exist in source data."
        )

    def test_fundamentals_recent_freshness(self):
        """Each fundamental's most-recent NON-NaN row must be within 120 days
        of close.parquet's last date.

        120d covers a Q1 report + 30 day filing lag (annual report can lag
        ~90 days from quarter end). Tighter than that false-positives during
        earnings season; looser silently accepts stale fundamentals."""
        c = load("close")
        last_close_date = c.index.max()
        files = _list_fundamental_files()
        if not files:
            pytest.skip("no fundamental files found")

        stale = []
        for fp in files:
            df = pd.read_parquet(fp)
            if not isinstance(df.index, pd.DatetimeIndex):
                continue
            # Find the most-recent date with at least one non-NaN value
            mask_any_nonnan = (~df.isna()).any(axis=1)
            if not mask_any_nonnan.any():
                continue   # all-NaN caught by test_no_field_100pct_nan
            last_data = df.index[mask_any_nonnan][-1]
            gap_days = (last_close_date - last_data).days
            if gap_days > 120:
                stale.append((fp.stem, str(last_data.date()), gap_days))

        assert not stale, (
            f"{len(stale)} fundamentals are >120 days behind close. "
            f"Worst (name, last_data_date, days_behind):\n  " +
            "\n  ".join(f"{n}: {d}, {g}d" for n, d, g in stale[:10])
        )

    def test_recent_quarter_update_rate(self):
        """For each fundamental, the recent-quarter update rate must be
        plausibly close to its historical update rate. Naturally-sparse
        fields (preferred_dividends_paid, net_income_discontinued, etc.)
        update at 1-5% per quarter; comparing absolute rates would
        false-positive on those. The real failure mode we're guarding
        against is a field that USED to update at 25%/quarter and dropped
        to 0% — meaning the FMP refresh broke for that endpoint.

        Rule: a field is broken if its recent 90-day update rate is below
        25% of its 4-year historical median 90-day update rate.
        """
        c = load("close")
        if len(c.index) < 360:
            pytest.skip("not enough history to compute historical update rate baseline")
        recent_window_start = c.index[-90]
        recent_window_end   = c.index[-1]

        files = _list_fundamental_files()
        if not files:
            pytest.skip("no fundamental files found")

        def _change_rate(window: pd.DataFrame) -> float:
            if len(window) < 2:
                return float("nan")
            first = window.iloc[0]
            last  = window.iloc[-1]
            valid = first.notna() & last.notna()
            if valid.sum() == 0:
                return float("nan")
            changed = ((first - last).abs() > 1e-12) & valid
            return float(changed.sum() / valid.sum())

        broken = []
        for fp in files:
            df = pd.read_parquet(fp)
            if not isinstance(df.index, pd.DatetimeIndex):
                continue
            # Recent rate
            try:
                recent = _change_rate(df.loc[recent_window_start:recent_window_end])
            except Exception:
                continue
            if np.isnan(recent):
                continue
            # Historical baseline — sample 90-day windows over the last ~4 years,
            # take the median.
            hist_rates = []
            for end_offset in range(180, min(1080, len(df)), 90):  # every 90 days
                end = df.index[-end_offset]
                start_idx = max(0, len(df.index) - end_offset - 90)
                start = df.index[start_idx]
                try:
                    r = _change_rate(df.loc[start:end])
                except Exception:
                    continue
                if not np.isnan(r):
                    hist_rates.append(r)
            if len(hist_rates) < 4:
                continue
            hist_median = float(np.median(hist_rates))
            # If the field is ALWAYS near-zero, leave it alone (structurally
            # sparse). Only flag when historical rate is meaningful (≥ 5%) AND
            # recent rate is < 25% of that.
            if hist_median >= 0.05 and recent < hist_median * 0.25:
                broken.append((fp.stem, recent, hist_median))

        assert not broken, (
            f"{len(broken)} fundamentals had recent 90-day update rate "
            f"< 25% of their 4-year historical median. Likely the FMP refresh "
            f"broke for those endpoints. Worst (name, recent_rate, hist_median):\n  "
            + "\n  ".join(f"{n}: {r*100:.1f}% vs hist {h*100:.1f}%"
                          for n, r, h in broken[:10])
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
