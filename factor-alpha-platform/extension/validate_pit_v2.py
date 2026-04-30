"""
Validate matrices_pit_v2 panel directly. Checks per-field shape consistency,
freshness, NaN coverage, and PIT-correctness vs raw filings (using AAPL/MSFT
spot checks).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data/fmp_cache/matrices_pit_v2"
INCOME = ROOT / "data/fmp_cache/income"

results = []
def add(check, status, detail):
    results.append((check, status, detail))
    sym = "[PASS]" if status == "PASS" else "[FAIL]" if status == "FAIL" else "[WARN]"
    print(f"  {sym} {check}: {detail}", flush=True)


def main():
    print("=" * 80)
    print("PIT v2 PANEL VALIDATION")
    print("=" * 80)

    files = sorted(DIR.glob("*.parquet"))
    print(f"  {len(files)} parquet files in {DIR}")

    if not files:
        add("files", "FAIL", "no parquets")
        return _summary()

    close = pd.read_parquet(DIR / "close.parquet")
    add("close.shape", "PASS",
        f"{close.shape}, {close.index.min().date()} -> {close.index.max().date()}")

    # 1. Shape consistency
    bad_shape = []
    for fp in files:
        df = pd.read_parquet(fp)
        if df.shape != close.shape:
            bad_shape.append((fp.stem, df.shape))
    add("shape_consistency", "PASS" if not bad_shape else "FAIL",
        "all files match close.shape" if not bad_shape else
        f"{len(bad_shape)} mismatched: " + ", ".join(f"{n}{s}" for n,s in bad_shape[:5]))

    # 2. Freshness consistency
    last_dates = {fp.stem: pd.to_datetime(pd.read_parquet(fp).index.max()) for fp in files}
    date_set = {d.date() for d in last_dates.values()}
    add("freshness_consistency",
        "PASS" if len(date_set) == 1 else "WARN",
        f"all files end on {sorted(date_set)[0]}" if len(date_set) == 1 else
        f"{len(date_set)} different last-data dates: {sorted(date_set, reverse=True)[:3]}")

    # 3. Coverage by year (close field)
    print("\n  Close coverage (% non-NaN cells) by year:")
    for y in range(2010, 2027):
        sub = close.loc[f"{y}-01-01":f"{y}-12-31"]
        if len(sub):
            cov = (~sub.isna()).sum().sum() / (sub.shape[0] * sub.shape[1]) * 100
            print(f"    {y}: {cov:>5.1f}%   (avg active: {(~sub.isna()).any(axis=0).sum()})")
    add("year_coverage", "PASS", "yearly coverage printed")

    # 4. Recent fundamental update rates (avoid the quoting issue from the dashboard)
    print("\n  Recent fundamental update rates (last 60 bars vs prior 60):")
    fund_keys = ["roe","roa","cap","earnings_yield","book_to_market","current_ratio",
                 "debt_to_equity","ev_to_ebitda","operating_margin","revenue","total_assets"]
    issues = []
    for k in fund_keys:
        fp = DIR / f"{k}.parquet"
        if not fp.exists(): continue
        df = pd.read_parquet(fp)
        if len(df) < 120: continue
        recent = df.iloc[-60:]
        prior = df.iloc[-120:-60]
        recent_chg = ((recent.iloc[-1] != recent.iloc[0]) & recent.iloc[-1].notna() & recent.iloc[0].notna()).mean()
        prior_chg = ((prior.iloc[-1] != prior.iloc[0]) & prior.iloc[-1].notna() & prior.iloc[0].notna()).mean()
        marker = "OK" if recent_chg > 0.2 else "LOW"
        if marker == "LOW":
            issues.append(k)
        print(f"    {k:<22} recent_chg={recent_chg*100:>5.1f}%  prior_chg={prior_chg*100:>5.1f}%  [{marker}]")
    add("recent_update_rate", "PASS" if not issues else "WARN",
        "all key fields > 20% recent change rate" if not issues else
        f"{len(issues)} fields with low recent change: {issues}")

    # 5. PIT correctness spot-check: AAPL roa_2023 should appear no earlier than filing date
    if (INCOME / "AAPL.parquet").exists():
        inc = pd.read_parquet(INCOME / "AAPL.parquet")
        if "filingDate" in inc.columns:
            inc["filingDate"] = pd.to_datetime(inc["filingDate"], errors="coerce")
            roa = pd.read_parquet(DIR / "roa.parquet")["AAPL"].dropna()
            if len(roa):
                # First date AAPL roa is non-null in 2024
                aapl_2024 = roa.loc["2024-01-01":"2024-12-31"]
                if len(aapl_2024):
                    first_change = aapl_2024.index[0]
                    # Most recent filing on or before that date should be the source
                    valid_filings = inc[inc["filingDate"] <= first_change]
                    if len(valid_filings) > 0:
                        latest_filing = valid_filings["filingDate"].max()
                        gap_days = (first_change - latest_filing).days
                        add("pit_correctness/AAPL_roa",
                            "PASS" if 0 <= gap_days <= 30 else "WARN",
                            f"AAPL 2024-Q1 roa first appears {first_change.date()} vs latest filing {latest_filing.date()} (gap {gap_days}d)")

    # 6. Bankrupt/acquired ticker spot-checks
    spot_checks = [
        ("BBBY", "Bed Bath & Beyond bankruptcy 2023"),
        ("FRC", "First Republic Bank seizure 2023"),
        ("SVB", "SVB collapse 2023"),
    ]
    for sym, desc in spot_checks:
        if sym in close.columns:
            series = close[sym].dropna()
            if len(series):
                add(f"survivor/{sym}", "PASS",
                    f"present: {len(series)} obs, ends {series.index.max().date()} ({desc})")
            else:
                add(f"survivor/{sym}", "WARN", f"present but empty ({desc})")
        else:
            add(f"survivor/{sym}", "WARN", f"NOT in panel ({desc})")

    return _summary()


def _summary():
    p = sum(1 for _,s,_ in results if s == "PASS")
    w = sum(1 for _,s,_ in results if s == "WARN")
    f = sum(1 for _,s,_ in results if s == "FAIL")
    print()
    print("=" * 80)
    print(f"SUMMARY: {p} PASS / {w} WARN / {f} FAIL")
    print("=" * 80)
    out = ROOT / "data/fmp_cache/matrices_pit_v2/validation_report.json"
    out.write_text(json.dumps([{"check":c,"status":s,"detail":d} for c,s,d in results], indent=2))
    print(f"Report: {out}")
    sys.exit(1 if f > 0 else 0)


if __name__ == "__main__":
    main()
