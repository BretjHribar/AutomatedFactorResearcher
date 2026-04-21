"""
audit_fmp_data.py — Comprehensive Data Integrity Audit for FMP Historical Data.

Checks:
    1. Matrix shape consistency (all matrices have same date range)
    2. Price data sanity (no negative prices, extreme jumps)
    3. OHLC consistency (low <= open/close <= high)
    4. Volume sanity (no negatives, check for stale/zero volume days)
    5. Returns distribution (detect data errors via extreme returns)
    6. Universe coverage over time
    7. Survivorship bias detection
    8. Fundamental data staleness
    9. Missing data patterns
    10. Cross-field consistency (e.g., market_cap vs close * sharesout)

Run:
    python audit_fmp_data.py
"""

import sys
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MATRICES_DIR = Path("data/fmp_cache/matrices")
UNIVERSES_DIR = Path("data/fmp_cache/universes")
CACHE_DIR = Path("data/fmp_cache")


def load_matrix(name: str) -> pd.DataFrame | None:
    """Load a matrix parquet file."""
    path = MATRICES_DIR / f"{name}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def audit_matrix_shapes():
    """Check that core OHLCV matrices have consistent shapes."""
    print("\n" + "="*80)
    print("1. MATRIX SHAPE CONSISTENCY")
    print("="*80)

    core_fields = ["open", "high", "low", "close", "volume", "returns", "vwap"]
    shapes = {}
    date_ranges = {}
    issues = []

    for name in core_fields:
        df = load_matrix(name)
        if df is None:
            issues.append(f"  MISSING: {name}.parquet")
            continue
        shapes[name] = df.shape
        date_ranges[name] = (df.index[0], df.index[-1])
        print(f"  {name:12s}: {df.shape[0]:6d} dates x {df.shape[1]:5d} tickers "
              f"| {df.index[0].date()} to {df.index[-1].date()}")

    # Check consistency
    if len(set(s[0] for s in shapes.values())) > 1:
        issues.append(f"  INCONSISTENT date counts: {shapes}")
    if len(set(s[1] for s in shapes.values())) > 1:
        ticker_counts = {k: v[1] for k, v in shapes.items()}
        issues.append(f"  INCONSISTENT ticker counts: {ticker_counts}")

    # Check date alignment
    if "close" in date_ranges and "volume" in date_ranges:
        if date_ranges["close"] != date_ranges["volume"]:
            issues.append(f"  DATE MISMATCH: close={date_ranges['close']}, volume={date_ranges['volume']}")

    if issues:
        for i in issues:
            print(f"  ⚠️  {i}")
    else:
        print(f"  ✅ All core matrices consistent")

    return shapes, issues


def audit_price_sanity():
    """Check price data for obvious errors."""
    print("\n" + "="*80)
    print("2. PRICE DATA SANITY")
    print("="*80)

    close = load_matrix("close")
    if close is None:
        print("  ❌ close.parquet not found!")
        return {}

    issues = {}

    # Negative prices
    neg_count = (close < 0).sum().sum()
    if neg_count > 0:
        bad_tickers = close.columns[(close < 0).any()].tolist()
        issues["negative_prices"] = {"count": int(neg_count), "tickers": bad_tickers[:20]}
        print(f"  ❌ Negative prices: {neg_count} cells in {len(bad_tickers)} tickers")
        print(f"     Examples: {bad_tickers[:5]}")
    else:
        print(f"  ✅ No negative prices")

    # Zero prices (possible delistings or data errors)
    zero_count = (close == 0).sum().sum()
    zero_tickers = close.columns[(close == 0).any()].tolist()
    print(f"  {'⚠️' if zero_count > 0 else '✅'} Zero prices: {zero_count} cells in "
          f"{len(zero_tickers)} tickers")

    # Extreme prices (> $50,000 or < $0.01 — likely data errors)
    extreme_high = (close > 50000).sum().sum()
    extreme_low = ((close > 0) & (close < 0.01)).sum().sum()
    if extreme_high > 0:
        tickers = close.columns[(close > 50000).any()].tolist()
        issues["extreme_high_prices"] = {"count": int(extreme_high), "tickers": tickers[:10]}
        print(f"  ⚠️  Prices > $50,000: {extreme_high} cells ({tickers[:5]})")
    if extreme_low > 0:
        tickers = close.columns[((close > 0) & (close < 0.01)).any()].tolist()
        print(f"  ⚠️  Prices < $0.01 (penny): {extreme_low} cells ({len(tickers)} tickers)")

    # Price jumps (day-over-day > 100% or < -80%)
    returns = close.pct_change(fill_method=None)
    extreme_up = (returns > 1.0).sum().sum()
    extreme_down = (returns < -0.80).sum().sum()
    issues["extreme_returns_up"] = int(extreme_up)
    issues["extreme_returns_down"] = int(extreme_down)
    print(f"  ⚠️  Returns > +100% in a day: {extreme_up} instances")
    print(f"  ⚠️  Returns < -80% in a day: {extreme_down} instances")

    # Stale prices (identical close for 5+ consecutive days)
    stale = (close.diff() == 0)
    stale_streaks = stale.rolling(5).sum()
    stale_5d = (stale_streaks >= 5).sum().sum()
    print(f"  {'⚠️' if stale_5d > 100 else '✅'} Stale prices (5+ days identical): "
          f"{stale_5d} instances")

    return issues


def audit_ohlc_consistency():
    """Check OHLC relationship: Low <= Open,Close <= High."""
    print("\n" + "="*80)
    print("3. OHLC CONSISTENCY")
    print("="*80)

    o = load_matrix("open")
    h = load_matrix("high")
    l = load_matrix("low")
    c = load_matrix("close")

    if any(x is None for x in [o, h, l, c]):
        print("  ❌ Missing OHLC data!")
        return {}

    # Align
    common_cols = sorted(set(o.columns) & set(h.columns) & set(l.columns) & set(c.columns))
    common_dates = o.index.intersection(h.index).intersection(l.index).intersection(c.index)

    o = o.loc[common_dates, common_cols]
    h = h.loc[common_dates, common_cols]
    l = l.loc[common_dates, common_cols]
    c = c.loc[common_dates, common_cols]

    issues = {}

    # Low > High (impossible)
    low_gt_high = (l > h + 0.001).sum().sum()  # small tolerance for floating point
    if low_gt_high > 0:
        bad_mask = (l > h + 0.001).any()
        bad_tickers = [col for col, val in bad_mask.items() if val]
        issues["low_gt_high"] = int(low_gt_high)
        print(f"  ❌ Low > High: {low_gt_high} violations ({len(bad_tickers)} tickers)")
    else:
        print(f"  ✅ Low <= High: all good")

    # Open outside [Low, High]
    open_below_low = (o < l - 0.001).sum().sum()
    open_above_high = (o > h + 0.001).sum().sum()
    if open_below_low > 0 or open_above_high > 0:
        issues["open_violations"] = int(open_below_low + open_above_high)
        print(f"  ⚠️  Open < Low: {open_below_low} | Open > High: {open_above_high}")
    else:
        print(f"  ✅ Open within [Low, High]")

    # Close outside [Low, High]
    close_below_low = (c < l - 0.001).sum().sum()
    close_above_high = (c > h + 0.001).sum().sum()
    if close_below_low > 0 or close_above_high > 0:
        issues["close_violations"] = int(close_below_low + close_above_high)
        print(f"  ⚠️  Close < Low: {close_below_low} | Close > High: {close_above_high}")
    else:
        print(f"  ✅ Close within [Low, High]")

    # High-Low range as % of close — detect flat bars (possible data errors)
    range_pct = (h - l) / c
    flat_bars = (range_pct == 0).sum().sum()
    total_bars = range_pct.notna().sum().sum()
    print(f"  Flat bars (H=L): {flat_bars} / {total_bars} "
          f"({flat_bars/total_bars*100:.2f}%)")

    return issues


def audit_volume():
    """Check volume data sanity."""
    print("\n" + "="*80)
    print("4. VOLUME DATA SANITY")
    print("="*80)

    vol = load_matrix("volume")
    if vol is None:
        print("  ❌ volume.parquet not found!")
        return {}

    issues = {}

    # Negative volume
    neg = (vol < 0).sum().sum()
    print(f"  {'❌' if neg > 0 else '✅'} Negative volume: {neg}")

    # Zero volume days (could indicate halts or data gaps)
    zero = (vol == 0).sum().sum()
    total = vol.notna().sum().sum()
    print(f"  Zero volume days: {zero} / {total} ({zero/total*100:.2f}%)")

    # Very high volume spikes (> 50x 20-day average)
    vol_ma20 = vol.rolling(20).mean()
    spikes = (vol > vol_ma20 * 50).sum().sum()
    print(f"  Volume spikes (>50x 20d avg): {spikes}")

    # Coverage: % of cells that are NaN
    nan_pct = vol.isna().sum().sum() / (vol.shape[0] * vol.shape[1]) * 100
    print(f"  NaN coverage: {nan_pct:.1f}% of all cells")

    return issues


def audit_returns_distribution():
    """Analyze returns distribution for anomalies."""
    print("\n" + "="*80)
    print("5. RETURNS DISTRIBUTION ANALYSIS")
    print("="*80)

    ret = load_matrix("returns")
    if ret is None:
        close = load_matrix("close")
        if close is None:
            print("  ❌ No returns or close data!")
            return {}
        ret = close.pct_change(fill_method=None)

    # Basic stats
    all_rets = ret.values.flatten()
    all_rets = all_rets[np.isfinite(all_rets)]

    print(f"  Total return observations: {len(all_rets):,}")
    print(f"  Mean:     {np.mean(all_rets):+.6f}")
    print(f"  Std:      {np.std(all_rets):.6f}")
    print(f"  Skew:     {pd.Series(all_rets).skew():.4f}")
    print(f"  Kurtosis: {pd.Series(all_rets).kurtosis():.4f}")
    print(f"  Min:      {np.min(all_rets):+.4f}")
    print(f"  Max:      {np.max(all_rets):+.4f}")

    # Distribution of extreme returns
    thresholds = [0.20, 0.30, 0.50, 1.0]
    for t in thresholds:
        n_up = (all_rets > t).sum()
        n_down = (all_rets < -t).sum()
        print(f"  |return| > {t:.0%}: {n_up} up, {n_down} down "
              f"({(n_up + n_down)/len(all_rets)*100:.4f}%)")

    return {}


def audit_universe_coverage():
    """Analyze universe membership over time."""
    print("\n" + "="*80)
    print("6. UNIVERSE COVERAGE")
    print("="*80)

    for name in ["TOP200", "TOP500", "TOP1000", "TOP2000", "TOP3000"]:
        path = UNIVERSES_DIR / f"{name}.parquet"
        if not path.exists():
            print(f"  ❌ {name}: missing")
            continue

        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        members = df.sum(axis=1)
        print(f"  {name:8s}: {df.shape[0]:5d} dates x {df.shape[1]:5d} tickers | "
              f"avg={members.mean():.0f} min={members.min():.0f} max={members.max():.0f} | "
              f"{df.index[0].date()} to {df.index[-1].date()}")

    # Check for band universe prerequisites
    print(f"\n  Band universe prereqs (TOP1500, TOP2500, TOP3500):")
    for name in ["TOP1500", "TOP2500", "TOP3500"]:
        path = UNIVERSES_DIR / f"{name}.parquet"
        status = "EXISTS" if path.exists() else "MISSING — run bulk_download.py"
        print(f"    {name}: {status}")


def audit_survivorship_bias():
    """Check for survivorship bias in the data."""
    print("\n" + "="*80)
    print("7. SURVIVORSHIP BIAS DETECTION")  
    print("="*80)

    close = load_matrix("close")
    if close is None:
        print("  ❌ close data missing")
        return {}

    # Count tickers per year
    yearly_counts = {}
    for year in range(close.index[0].year, close.index[-1].year + 1):
        mask = close.index.year == year
        if mask.sum() == 0:
            continue
        year_data = close.loc[mask]
        active = year_data.notna().any(axis=0).sum()
        yearly_counts[year] = active

    print(f"  Active tickers per year:")
    for year, count in sorted(yearly_counts.items()):
        print(f"    {year}: {count:5d} tickers")

    # Check for delistings (tickers that stop having data)
    last_valid = close.apply(lambda x: x.last_valid_index())
    early_exits = last_valid[last_valid < close.index[-1] - pd.Timedelta(days=30)]
    print(f"\n  Tickers that stopped before last month: {len(early_exits)}")

    # Check delisted companies file
    delisted_path = CACHE_DIR / "delisted_companies.json"
    if delisted_path.exists():
        with open(delisted_path) as f:
            delisted = json.load(f)
        print(f"  Delisted companies in FMP: {len(delisted)}")
        if isinstance(delisted, list) and len(delisted) > 0:
            # Check if delisted are in our price data
            delisted_symbols = set(d.get("symbol", d) if isinstance(d, dict) else str(d) 
                                   for d in delisted[:1000])
            overlap = delisted_symbols & set(close.columns)
            print(f"  Delisted tickers in our data: {len(overlap)} "
                  f"(good = survivorship bias mitigation)")
    else:
        print(f"  ⚠️  No delisted_companies.json found")

    return {"yearly_counts": yearly_counts, "early_exits": len(early_exits)}


def audit_fundamental_staleness():
    """Check how stale fundamental data is (quarterly data forward-filled to daily)."""
    print("\n" + "="*80)
    print("8. FUNDAMENTAL DATA STALENESS")
    print("="*80)

    # Fundamental fields that should update quarterly
    fundamental_fields = ["revenue", "net_income", "ebitda", "eps", "bookvalue_ps",
                          "pe_ratio", "market_cap", "debt", "free_cashflow"]

    for name in fundamental_fields:
        df = load_matrix(name)
        if df is None:
            print(f"  {name:20s}: MISSING")
            continue

        # Count how many unique values each ticker has (vs total dates)
        # Fundamental data should have ~4 unique values per year (quarterly)
        n_dates = len(df)
        avg_unique = df.apply(lambda x: x.dropna().nunique()).mean()
        avg_coverage = df.notna().sum(axis=0).mean() / n_dates * 100
        
        # Check for NaN percentage
        nan_pct = df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100

        print(f"  {name:20s}: {df.shape[1]:5d} tickers | "
              f"avg_unique={avg_unique:.0f} | "
              f"coverage={avg_coverage:.0f}% | "
              f"NaN={nan_pct:.1f}%")


def audit_missing_data():
    """Analyze missing data patterns."""
    print("\n" + "="*80)
    print("9. MISSING DATA PATTERNS")
    print("="*80)

    close = load_matrix("close")
    vol = load_matrix("volume")
    vwap = load_matrix("vwap")

    if close is None:
        print("  ❌ No close data")
        return {}

    # Overall NaN rates for core fields
    for name, df in [("close", close), ("volume", vol), ("vwap", vwap)]:
        if df is None:
            print(f"  {name:10s}: MISSING FILE")
            continue
        total = df.shape[0] * df.shape[1]
        nan_count = df.isna().sum().sum()
        print(f"  {name:10s}: {nan_count/total*100:.1f}% NaN "
              f"({nan_count:,} / {total:,})")

    # Per-year NaN rates for close
    print(f"\n  Close NaN rate by year:")
    for year in range(close.index[0].year, close.index[-1].year + 1):
        mask = close.index.year == year
        if mask.sum() == 0:
            continue
        year_data = close.loc[mask]
        nan_rate = year_data.isna().sum().sum() / (year_data.shape[0] * year_data.shape[1]) * 100
        n_active = year_data.notna().any(axis=0).sum()
        print(f"    {year}: {nan_rate:.1f}% NaN | {n_active} active tickers")

    return {}


def audit_cross_field_consistency():
    """Check consistency between related fields."""
    print("\n" + "="*80)
    print("10. CROSS-FIELD CONSISTENCY")
    print("="*80)

    close = load_matrix("close")
    cap = load_matrix("market_cap")
    shares = load_matrix("sharesout")
    vwap_df = load_matrix("vwap")
    adv20 = load_matrix("adv20")
    vol = load_matrix("volume")
    dollars = load_matrix("dollars_traded")

    # 1. Market cap vs close * shares
    if close is not None and shares is not None and cap is not None:
        common = sorted(set(close.columns) & set(shares.columns) & set(cap.columns))
        if common:
            calc_cap = close[common] * shares[common]
            reported_cap = cap[common]
            
            # Sample 1000 random cells
            valid = calc_cap.notna() & reported_cap.notna() & (calc_cap > 0)
            if valid.sum().sum() > 0:
                ratio = (reported_cap[valid] / calc_cap[valid]).values.flatten()
                ratio = ratio[np.isfinite(ratio)]
                if len(ratio) > 100:
                    median_ratio = np.median(ratio)
                    pct_near_1 = ((ratio > 0.8) & (ratio < 1.2)).mean() * 100
                    print(f"  Market cap = close × shares: median ratio={median_ratio:.3f}, "
                          f"{pct_near_1:.1f}% within ±20%")
    
    # 2. VWAP should be between low and high
    if vwap_df is not None:
        h = load_matrix("high")
        l = load_matrix("low")
        if h is not None and l is not None:
            common = sorted(set(vwap_df.columns) & set(h.columns) & set(l.columns))
            if common:
                vwap_above_high = (vwap_df[common] > h[common] + 0.01).sum().sum()
                vwap_below_low = (vwap_df[common] < l[common] - 0.01).sum().sum()
                total = vwap_df[common].notna().sum().sum()
                print(f"  VWAP within [Low, High]: "
                      f"{vwap_above_high} above high, {vwap_below_low} below low "
                      f"(of {total:,} valid)")
    
    # 3. ADV20 vs volume * close
    if adv20 is not None and vol is not None and close is not None:
        common = sorted(set(adv20.columns) & set(vol.columns) & set(close.columns))
        if common:
            calc_dv = vol[common] * close[common]
            calc_adv20 = calc_dv.rolling(20).mean()
            valid = calc_adv20.notna() & adv20[common].notna() & (calc_adv20 > 0)
            if valid.sum().sum() > 0:
                ratio = (adv20[common][valid] / calc_adv20[valid]).values.flatten()
                ratio = ratio[np.isfinite(ratio)]
                if len(ratio) > 100:
                    median_ratio = np.median(ratio)
                    print(f"  ADV20 vs calc(vol*close, 20d): median ratio={median_ratio:.3f}")

    # 4. Dollars traded vs volume * close
    if dollars is not None and vol is not None and close is not None:
        common = sorted(set(dollars.columns) & set(vol.columns) & set(close.columns))
        if common:
            calc_dollars = vol[common] * close[common]
            valid = calc_dollars.notna() & dollars[common].notna() & (calc_dollars > 0)
            if valid.sum().sum() > 0:
                ratio = (dollars[common][valid] / calc_dollars[valid]).values.flatten()
                ratio = ratio[np.isfinite(ratio)]
                if len(ratio) > 100:
                    pct_match = ((ratio > 0.9) & (ratio < 1.1)).mean() * 100
                    print(f"  dollars_traded ≈ vol*close: {pct_match:.1f}% within ±10%")


def run_full_audit():
    """Run all audit checks."""
    print("╔" + "═"*78 + "╗")
    print("║" + "FMP HISTORICAL DATA INTEGRITY AUDIT".center(78) + "║")
    print("╚" + "═"*78 + "╝")

    # Check data exists
    if not MATRICES_DIR.exists():
        print(f"\n❌ Matrices directory not found: {MATRICES_DIR}")
        return

    n_files = len(list(MATRICES_DIR.glob("*.parquet")))
    print(f"\nMatrices directory: {MATRICES_DIR}")
    print(f"Total parquet files: {n_files}")

    shapes, shape_issues = audit_matrix_shapes()
    price_issues = audit_price_sanity()
    ohlc_issues = audit_ohlc_consistency()
    volume_issues = audit_volume()
    returns_info = audit_returns_distribution()
    audit_universe_coverage()
    surv_info = audit_survivorship_bias()
    audit_fundamental_staleness()
    audit_missing_data()
    audit_cross_field_consistency()

    # Summary
    print("\n" + "="*80)
    print("AUDIT SUMMARY")
    print("="*80)

    all_issues = []
    if shape_issues:
        all_issues.extend(shape_issues)
    if price_issues.get("negative_prices"):
        all_issues.append(f"Negative prices: {price_issues['negative_prices']['count']}")
    if price_issues.get("extreme_returns_up", 0) > 1000:
        all_issues.append(f"Many extreme +returns: {price_issues['extreme_returns_up']}")
    if ohlc_issues.get("low_gt_high", 0) > 0:
        all_issues.append(f"OHLC violations (Low > High): {ohlc_issues['low_gt_high']}")

    if all_issues:
        print("  ⚠️  Issues found:")
        for issue in all_issues:
            print(f"    - {issue}")
    else:
        print("  ✅ No critical issues found")

    print("\n  Run this audit periodically, especially after data refreshes.")
    print()


if __name__ == "__main__":
    run_full_audit()
