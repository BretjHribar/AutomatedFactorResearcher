"""
diag_data_freshness.py -- Audit the date coverage and recent null rates
for all FMP matrix and universe files to find why equity curves flatline.
"""
import pandas as pd
import numpy as np
from pathlib import Path

mat_dir = Path("data/fmp_cache/matrices")
uni_dir = Path("data/fmp_cache/universes")

print("=" * 90)
print("  MATRIX FILE DATE RANGES & RECENT NULL RATES")
print("=" * 90)
print(f"  {'Field':<30s}  {'Start':>12}  {'End':>12}  {'LastValid':>12}  {'Null_30d%':>10}")
print(f"  {'-'*80}")

for fp in sorted(mat_dir.glob("*.parquet")):
    try:
        df = pd.read_parquet(fp)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        non_null = df.dropna(how="all")
        last_valid = non_null.index[-1].date() if len(non_null) > 0 else "ALL_NAN"
        null_30d = df.iloc[-30:].isna().mean().mean() * 100
        end_date = df.index[-1].date()
        start_date = df.index[0].date()
        print(f"  {fp.stem:<30s}  {str(start_date):>12}  {str(end_date):>12}  {str(last_valid):>12}  {null_30d:9.1f}%")
    except Exception as e:
        print(f"  {fp.stem:<30s}  ERROR: {e}")

print()
print("=" * 90)
print("  UNIVERSE FILE DATE RANGES & COVERAGE")
print("=" * 90)
print(f"  {'Universe':<30s}  {'Start':>12}  {'End':>12}  {'Cvg_last%':>10}  {'Cvg_30d%':>10}")
print(f"  {'-'*80}")

for fp in sorted(uni_dir.glob("*.parquet")):
    try:
        df = pd.read_parquet(fp)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        cvg_last = float(df.iloc[-1].mean()) * 100
        cvg_30d  = float(df.iloc[-30:].mean().mean()) * 100
        print(f"  {fp.stem:<30s}  {str(df.index[0].date()):>12}  {str(df.index[-1].date()):>12}  {cvg_last:9.1f}%  {cvg_30d:9.1f}%")
    except Exception as e:
        print(f"  {fp.stem:<30s}  ERROR: {e}")

# ── Deep check: how many tickers have data in the last 30 trading days ──────
print()
print("=" * 90)
print("  CLOSE PRICE LAST 60 BARS: TICKER COVERAGE OVER TIME (TOP2000TOP3000 universe)")
print("=" * 90)
try:
    close = pd.read_parquet(mat_dir / "close.parquet")
    if not isinstance(close.index, pd.DatetimeIndex):
        close.index = pd.to_datetime(close.index)
    uni = pd.read_parquet(uni_dir / "TOP2000TOP3000.parquet")
    if not isinstance(uni.index, pd.DatetimeIndex):
        uni.index = pd.to_datetime(uni.index)

    # Align
    common_idx = close.index.intersection(uni.index)
    common_col = close.columns.intersection(uni.columns)
    close_u = close.loc[common_idx, common_col]
    uni_u   = uni.loc[common_idx, common_col]

    # Show month-by-month coverage for the last 18 months
    recent = close_u.iloc[-400:]
    uni_r  = uni_u.reindex(index=recent.index, columns=recent.columns).fillna(False)

    print(f"  {'Month':<12}  {'InUniverse':>12}  {'HaveClose':>12}  {'Coverage%':>10}")
    print(f"  {'-'*55}")

    by_month = recent.resample("ME")
    for month_end, grp in by_month:
        uni_grp = uni_r.reindex(index=grp.index)
        n_in_uni = int(uni_grp.any(axis=0).sum())
        n_have_close = int(grp.notna().any(axis=0).sum())
        # of tickers in universe, how many have at least 1 valid close this month
        in_uni_tickers = uni_grp.columns[uni_grp.any(axis=0)]
        have_data = grp[in_uni_tickers].notna().any(axis=0).sum()
        pct = 100.0 * have_data / max(n_in_uni, 1)
        print(f"  {str(month_end.date()):<12}  {n_in_uni:>12}  {n_have_close:>12}  {pct:9.1f}%")

except Exception as e:
    print(f"  ERROR in close/universe analysis: {e}")

print()
print("  Done.")
