"""
Equities Data Integrity Dashboard — analogous to data_integrity_dashboard.py
(crypto). PASS/FAIL on:

  1. Per-matrix shape consistency (all matrices should have same rows × cols)
  2. Date-index alignment (all matrices share the same calendar)
  3. Ticker-column alignment (all matrices share the same ticker set)
  4. NaN coverage per char (overall + recent window)
  5. **Freshness per char** — last date with > 30% non-NaN
  6. **Update-frequency anomaly** — fundamentals usually update on filings;
     flag months where < 5% of universe got an update
  7. **Look-ahead bias check** — for fundamentals where we have raw cache with
     acceptedDate, verify matrix value comes online no earlier than acceptedDate
  8. Price sanity (negative, zero, extreme returns)
  9. Cross-matrix consistency (close vs returns alignment)
 10. Universe membership consistency (TOP500/1000/2000 row sums)
 11. **Calendar continuity** — index has every NYSE trading day, no holes
 12. **End-staleness** — last bar is the previous NYSE trading day (or today)

Exit code 0 = ALL GREEN, 1 = FAILURES.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# ── Config / thresholds ──────────────────────────────────────────────────────
MATRICES_DIRS = {
    "matrices_pit_v2": PROJECT_ROOT / "data/fmp_cache/matrices_pit_v2",
    "matrices_pit":    PROJECT_ROOT / "data/fmp_cache/matrices_pit",
    "matrices_clean":  PROJECT_ROOT / "data/fmp_cache/matrices_clean",
    "matrices":        PROJECT_ROOT / "data/fmp_cache/matrices",
}
RAW_CACHE_DIRS = {
    "income":    PROJECT_ROOT / "data/fmp_cache/income",
    "balance":   PROJECT_ROOT / "data/fmp_cache/balance",
    "cashflow":  PROJECT_ROOT / "data/fmp_cache/cashflow",
    "metrics":   PROJECT_ROOT / "data/fmp_cache/metrics",
}
UNIVERSES_DIR = PROJECT_ROOT / "data/fmp_cache/universes"
RESULTS_DIR   = PROJECT_ROOT / "data/aipt_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FRESHNESS_DAYS_PRICES   = 7    # close should be < 7 days stale
FRESHNESS_DAYS_FUND     = 60   # fundamentals tolerate longer lag
MIN_NONNAN_FRAC         = 0.30 # bar with < 30% non-NaN ⇒ "no data here"
UPDATE_RATE_FLOOR_PRICE = 0.95 # close should change for ~all tickers per bar
UPDATE_RATE_FLOOR_FUND  = 0.20 # fundamentals: ≥20% of universe should update per month

# Files we treat as fundamentals (vs time-series chars)
FUNDAMENTAL_HINTS = ("roe", "roa", "margin", "earnings_yield", "free_cashflow_yield",
                     "ev_to_", "book_to_market", "asset_turnover", "debt_to",
                     "current_ratio", "cap")
# Files we treat as time-series chars (should change every bar)
TIMESERIES_HINTS = ("returns", "log_returns", "_volatility_", "adv", "dollars_traded",
                    "high_low_range", "open_close_range", "vwap", "momentum_",
                    "volume", "high", "low", "open", "close", "turnover")


# ─────────────────────────────────────────────────────────────────────────────
# Check helpers
# ─────────────────────────────────────────────────────────────────────────────
class CheckReport:
    def __init__(self):
        self.results = []

    def add(self, check, status, detail):
        self.results.append({"check": check, "status": status, "detail": detail})
        sym = "[PASS]" if status == "PASS" else "[FAIL]" if status == "FAIL" else "[WARN]"
        print(f"  {sym} {check}: {detail}", flush=True)

    def summary(self):
        passes = sum(1 for r in self.results if r["status"] == "PASS")
        warns  = sum(1 for r in self.results if r["status"] == "WARN")
        fails  = sum(1 for r in self.results if r["status"] == "FAIL")
        return passes, warns, fails


def is_fundamental(name):
    return any(h in name for h in FUNDAMENTAL_HINTS) and not is_timeseries(name)

def is_timeseries(name):
    return any(name.endswith(h) or h in name for h in TIMESERIES_HINTS)


# ─────────────────────────────────────────────────────────────────────────────
# Per-directory checks
# ─────────────────────────────────────────────────────────────────────────────

def audit_matrices_dir(name: str, mdir: Path, report: CheckReport):
    print(f"\n{'='*100}")
    print(f"AUDIT: {name}  ({mdir})")
    print(f"{'='*100}")
    if not mdir.exists():
        report.add(f"{name}/exists", "FAIL", f"directory missing: {mdir}")
        return

    parquet_files = sorted(mdir.glob("*.parquet"))
    if not parquet_files:
        report.add(f"{name}/files", "FAIL", "no parquet files")
        return
    report.add(f"{name}/files", "PASS", f"{len(parquet_files)} parquet files")

    # Load close to get reference index/cols
    close_path = mdir / "close.parquet"
    if not close_path.exists():
        report.add(f"{name}/close", "FAIL", "no close.parquet — can't audit")
        return
    close = pd.read_parquet(close_path)
    ref_index = close.index
    ref_cols = set(close.columns)
    report.add(f"{name}/close.shape", "PASS",
               f"{close.shape}, dates {close.index.min().date()} → {close.index.max().date()}")

    # ── Check 1: shape consistency
    bad_shape = []
    for fp in parquet_files:
        try:
            df = pd.read_parquet(fp)
        except Exception as e:
            bad_shape.append((fp.stem, f"read-error:{type(e).__name__}"))
            continue
        if df.shape[0] != close.shape[0] or df.shape[1] != close.shape[1]:
            bad_shape.append((fp.stem, df.shape))
    if bad_shape:
        report.add(f"{name}/shape_consistency", "FAIL",
                   f"{len(bad_shape)} files differ from close shape: " +
                   ", ".join(f"{n}{s}" for n, s in bad_shape[:5]))
    else:
        report.add(f"{name}/shape_consistency", "PASS", "all files match close.shape")

    # ── Check 2: date alignment (mtime-based last-touched + last data date)
    print(f"\n  Last data date by file (top 5 latest, top 5 earliest):")
    last_dates = {}
    last_mtimes = {}
    for fp in parquet_files:
        df = pd.read_parquet(fp)
        try:
            last_dt = pd.to_datetime(df.index.max())
        except Exception:
            last_dt = pd.Timestamp("1900-01-01")
        last_dates[fp.stem] = last_dt
        last_mtimes[fp.stem] = datetime.fromtimestamp(fp.stat().st_mtime)

    by_date_desc = sorted(last_dates.items(), key=lambda kv: kv[1], reverse=True)
    print(f"    {'file':<35} {'last_data_date':<22} {'mtime':<22}")
    for fname, dt in by_date_desc[:5]:
        print(f"    {fname:<35} {str(dt.date()):<22} {str(last_mtimes[fname]):<22}")
    print(f"    ...")
    for fname, dt in by_date_desc[-5:]:
        print(f"    {fname:<35} {str(dt.date()):<22} {str(last_mtimes[fname]):<22}")

    # Inconsistent freshness?
    date_set = {dt.date() for dt in last_dates.values()}
    if len(date_set) > 1:
        report.add(f"{name}/freshness_consistency", "FAIL",
                   f"{len(date_set)} different last-data dates across {len(parquet_files)} files: "
                   f"{sorted(date_set, reverse=True)[:3]} ... {sorted(date_set)[:3]}")
    else:
        report.add(f"{name}/freshness_consistency", "PASS",
                   f"all {len(parquet_files)} files end on {date_set.pop()}")

    # ── Check 3: ticker alignment
    bad_cols = []
    for fp in parquet_files:
        df = pd.read_parquet(fp)
        if set(df.columns) != ref_cols:
            diff_a = len(set(df.columns) - ref_cols)
            diff_b = len(ref_cols - set(df.columns))
            bad_cols.append((fp.stem, df.shape[1], diff_a, diff_b))
    if bad_cols:
        report.add(f"{name}/ticker_alignment", "FAIL",
                   f"{len(bad_cols)} files have different tickers from close. Worst:"
                   + " " + ", ".join(f"{n}(N={s},+{a},-{b})" for n,s,a,b in bad_cols[:3]))
    else:
        report.add(f"{name}/ticker_alignment", "PASS", f"all files share {len(ref_cols)} tickers")

    # ── Check 4a: 100%-NaN catastrophic failure (column-name mismatch in build)
    # This is a separate, stricter check than recent_coverage. A file that is
    # 100% NaN over its whole history is almost always a build bug — a mapped
    # FMP column name that doesn't exist in the cached files. Demoted from
    # WARN (its previous status) to FAIL so CI / scoreboard runs catch it.
    all_nan = []
    for fp in parquet_files:
        if fp.stem.startswith("_"):
            continue
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        arr = df.values
        if not np.issubdtype(arr.dtype, np.floating):
            continue
        if not (~np.isnan(arr)).any():
            all_nan.append(fp.stem)
    if all_nan:
        report.add(f"{name}/field_population", "FAIL",
                   f"{len(all_nan)} fields are 100% NaN over the whole index "
                   f"(likely a mapped column name doesn't exist in source files): "
                   + ", ".join(all_nan[:10])
                   + (f", ...+{len(all_nan)-10} more" if len(all_nan) > 10 else ""))
    else:
        report.add(f"{name}/field_population", "PASS",
                   f"all {len(parquet_files)} fields contain at least some non-NaN data")

    # ── Check 4b: per-char NaN coverage in recent window (last 60 trading days)
    recent_end = close.index.max()
    recent_start = close.index[max(0, len(close.index) - 60)]
    bad_recent = []
    for fp in parquet_files:
        if fp.stem.startswith("_"):
            continue
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        # Some auxiliary parquets have non-DatetimeIndex (e.g. classification
        # tables); skip them rather than crashing the whole audit.
        if not isinstance(df.index, pd.DatetimeIndex):
            continue
        try:
            recent = df.loc[recent_start:recent_end]
        except Exception:
            continue
        if len(recent) == 0:
            continue
        avg_avail = (~recent.isna()).mean(axis=1).mean()
        if avg_avail < MIN_NONNAN_FRAC:
            bad_recent.append((fp.stem, avg_avail))
    if bad_recent:
        report.add(f"{name}/recent_coverage", "WARN",
                   f"{len(bad_recent)} files have <{MIN_NONNAN_FRAC*100:.0f}% recent coverage: "
                   + ", ".join(f"{n}({a*100:.0f}%)" for n, a in bad_recent[:5]))
    else:
        report.add(f"{name}/recent_coverage", "PASS",
                   f"all files have ≥{MIN_NONNAN_FRAC*100:.0f}% non-NaN over last 60 bars")

    # ── Check 5: update-frequency anomaly
    # Skip files we can't categorize; check fundamentals only
    print(f"\n  UPDATE FREQUENCY CHECK (fundamentals — fraction of universe whose value changed in each month)")
    print(f"  {'file':<28} " + " ".join(f"{m:>10}" for m in
            ["2024-12", "2025-06", "2025-09", "2025-11", "2025-12", "2026-01", "2026-02"]))
    fund_files = [fp for fp in parquet_files if is_fundamental(fp.stem) and not is_timeseries(fp.stem)]
    flagged = []
    no_data = []  # fundamentals where every month is n/a (caught by field_population too, but recorded here for the per-month report)
    for fp in fund_files[:20]:  # cap output volume
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        if not isinstance(df.index, pd.DatetimeIndex):
            continue
        rates = []
        for month in ["2024-12", "2025-06", "2025-09", "2025-11", "2025-12", "2026-01", "2026-02"]:
            try:
                sub = df.loc[month]
            except Exception:
                rates.append(np.nan)
                continue
            if len(sub) < 2:
                rates.append(np.nan)
                continue
            first = sub.iloc[0]
            last = sub.iloc[-1]
            valid = first.notna() & last.notna()
            if valid.sum() == 0:
                rates.append(np.nan)
                continue
            changed = ((first - last).abs() > 1e-12) & valid
            rates.append(changed.sum() / valid.sum())
        # If EVERY month is n/a, the field has no usable data — record it.
        if all(np.isnan(r) for r in rates):
            no_data.append(fp.stem)
        # Flag if recent month is dramatically lower than the typical rate
        recent_three = [r for r in rates[-3:] if not np.isnan(r)]
        prior_three  = [r for r in rates[:-3] if not np.isnan(r)]
        if recent_three and prior_three:
            recent_min = min(recent_three)
            prior_med = np.median(prior_three)
            if recent_min < UPDATE_RATE_FLOOR_FUND and recent_min < 0.4 * prior_med:
                flagged.append((fp.stem, recent_min, prior_med))
        print(f"  {fp.stem:<28} " + " ".join(
            (f"{r*100:>9.1f}%" if not np.isnan(r) else "      n/a") for r in rates))
    if flagged:
        report.add(f"{name}/update_freq", "FAIL",
                   f"{len(flagged)} fundamentals had recent month update rate < "
                   f"{UPDATE_RATE_FLOOR_FUND*100:.0f}% AND < 40% of prior median: " +
                   ", ".join(f"{n}({r*100:.0f}% vs {p*100:.0f}%)" for n,r,p in flagged[:5]))
    else:
        report.add(f"{name}/update_freq", "PASS", "fundamental update rates within tolerance")
    if no_data:
        report.add(f"{name}/update_freq_no_data", "FAIL",
                   f"{len(no_data)} fundamentals had n/a in every sampled month "
                   f"(field is unpopulated, would be silently skipped by the rate "
                   f"comparison): " + ", ".join(no_data[:5]))


def audit_calendar_continuity(name: str, mdir: Path, report: CheckReport):
    """Check that the date index covers every NYSE trading day with no holes,
    and that the last bar is the previous NYSE trading day or today.

    A hole is the silent killer for delay-0 strategies — every rolling-window
    operator (sma, momentum_60d, parkinson_volatility_60, etc.) reads across
    the gap and produces corrupted values for many subsequent bars."""
    print(f"\n{'='*100}")
    print(f"CALENDAR CONTINUITY: {name}")
    print(f"{'='*100}")
    if not mdir.exists():
        return
    close_path = mdir / "close.parquet"
    if not close_path.exists():
        return
    close = pd.read_parquet(close_path)
    if not isinstance(close.index, pd.DatetimeIndex):
        return

    cached_dates = set(close.index.date)
    first, last = close.index.min().date(), close.index.max().date()

    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        sched = nyse.schedule(start_date=str(first), end_date=str(last))
        expected = set(d.date() for d in sched.index)
        cal_label = "NYSE"
    except ImportError:
        rng = pd.bdate_range(first, last)
        expected = set(d.date() for d in rng)
        cal_label = "weekday-only-fallback"

    missing = sorted(expected - cached_dates)
    extra   = sorted(cached_dates - expected)

    if missing:
        report.add(f"{name}/calendar_holes", "FAIL",
                   f"{len(missing)} {cal_label} trading days missing from "
                   f"close.parquet. First few: {missing[:10]}"
                   + (f", ...+{len(missing)-10} more" if len(missing) > 10 else ""))
    else:
        report.add(f"{name}/calendar_holes", "PASS",
                   f"no missing trading days ({cal_label}, "
                   f"{len(cached_dates)} cached over {first}→{last})")

    if extra:
        report.add(f"{name}/calendar_extra", "WARN",
                   f"{len(extra)} dates in cache that aren't {cal_label} "
                   f"trading days: {extra[:5]}")

    # End-staleness — the cache must include the previous trading day
    today = datetime.today().date()
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        sched_recent = nyse.schedule(start_date=str(today - timedelta(days=10)),
                                     end_date=str(today))
        prior_days = [d.date() for d in sched_recent.index if d.date() < today]
        expected_last = prior_days[-1] if prior_days else None
    except ImportError:
        # Fallback: nearest weekday strictly before today
        d = today
        while True:
            d = d - timedelta(days=1)
            if d.weekday() < 5:
                expected_last = d
                break

    if expected_last is None:
        report.add(f"{name}/end_staleness", "WARN", "could not determine expected last bar")
    elif last < expected_last:
        gap_days = (expected_last - last).days
        report.add(f"{name}/end_staleness", "FAIL",
                   f"cache ends {last}, expected last bar is {expected_last} "
                   f"({gap_days} calendar days behind). Refresh required before "
                   f"trading on stale data.")
    elif last > expected_last:
        # cache includes today — that's fine if a live bar was appended
        report.add(f"{name}/end_staleness", "PASS",
                   f"cache ends {last} (≥ expected {expected_last}; live bar present)")
    else:
        report.add(f"{name}/end_staleness", "PASS",
                   f"cache ends on previous trading day {last}")


def audit_lookahead_bias(report: CheckReport):
    """Check whether matrix forward-fills from period-end (lookahead!) vs filing-date (PIT)."""
    print(f"\n{'='*100}")
    print(f"AUDIT: LOOK-AHEAD BIAS  (matrix value available date vs raw filing date)")
    print(f"{'='*100}")

    # Pick 5 sample tickers, check earnings_yield value vs raw income statement
    matrices_dir = MATRICES_DIRS["matrices_clean"]
    income_dir = RAW_CACHE_DIRS["income"]
    if not matrices_dir.exists() or not income_dir.exists():
        report.add("lookahead/sources", "WARN", "missing dirs to audit")
        return

    try:
        ey = pd.read_parquet(matrices_dir / "earnings_yield.parquet")
    except FileNotFoundError:
        report.add("lookahead/sources", "WARN", "no earnings_yield in matrices_clean")
        return

    sample = [c for c in ["AAPL", "MSFT", "JPM", "XOM", "WMT"] if c in ey.columns]
    if not sample:
        report.add("lookahead/sample", "WARN", "no recognizable sample tickers in matrix")
        return

    print(f"\n  Sample of 5 large stocks: matrix-EY value first appears vs raw filing date")
    print(f"  {'symbol':<8} {'period_end':<14} {'filing_date':<14} {'matrix_first_change_after':<30} {'lookahead_days':>14}")
    issues = 0
    for sym in sample:
        income_path = income_dir / f"{sym}.parquet"
        if not income_path.exists():
            continue
        inc = pd.read_parquet(income_path)
        if "filingDate" not in inc.columns:
            continue
        # latest 4 quarters
        inc_sorted = inc.sort_values("filingDate" if "filingDate" in inc.columns else "date").tail(4)
        col = ey[sym].dropna()
        for _, row in inc_sorted.iterrows():
            # FMP `period` column has values like "Q2" / "FY" that can't be parsed
            # as dates. Use the row's index value (DatetimeIndex) when available,
            # then fall back to a `date` column. Skip rows where neither is parseable.
            period_end = None
            if hasattr(row, "name") and isinstance(row.name, pd.Timestamp):
                period_end = row.name.date()
            elif "date" in row.index:
                try:
                    period_end = pd.Timestamp(row["date"]).date()
                except Exception:
                    pass
            if period_end is None:
                continue
            try:
                filing = pd.to_datetime(row["filingDate"]).date()
            except Exception:
                continue
            # Find the first matrix date ≥ period_end where value differs from prior bar
            after = ey[sym].loc[str(period_end):].dropna()
            if len(after) < 2:
                continue
            # Look for first change after period_end
            change_dates = after[after.diff().fillna(0).abs() > 1e-12].index
            change_dates = [d for d in change_dates if d.date() > period_end]
            if not change_dates:
                continue
            first_change = change_dates[0].date()
            lookahead = (filing - first_change).days
            sym_short = sym
            print(f"  {sym_short:<8} {str(period_end):<14} {str(filing):<14} {str(first_change):<30} {lookahead:>14}")
            if lookahead > 5:
                issues += 1

    if issues > 0:
        report.add("lookahead/period_end_ffill", "FAIL",
                   f"{issues} sample (sym, quarter) showed matrix value appearing >5 days "
                   f"BEFORE the actual filing date — period-end forward-fill bug")
    else:
        report.add("lookahead/period_end_ffill", "PASS",
                   "no detected look-ahead in sampled fundamentals")


def main():
    t0 = time.time()
    print("=" * 100)
    print("EQUITIES DATA INTEGRITY DASHBOARD")
    print("=" * 100)

    report = CheckReport()
    for name, mdir in MATRICES_DIRS.items():
        audit_matrices_dir(name, mdir, report)
        audit_calendar_continuity(name, mdir, report)

    audit_lookahead_bias(report)

    # Summary
    p, w, f = report.summary()
    print(f"\n{'='*100}")
    print(f"SUMMARY: {p} PASS / {w} WARN / {f} FAIL  ({time.time()-t0:.1f}s)")
    print(f"{'='*100}")

    out = RESULTS_DIR / "equities_data_integrity_report.json"
    with open(out, "w") as fh:
        json.dump({"results": report.results, "pass": p, "warn": w, "fail": f}, fh, indent=2, default=str)
    print(f"\nReport JSON: {out}")
    sys.exit(1 if f > 0 else 0)


if __name__ == "__main__":
    main()
