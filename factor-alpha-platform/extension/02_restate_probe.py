"""
Step 6 (early) — Restate-detection probe.

For a sample of 20 tickers, RE-FETCH fundamentals NOW and DIFF against the
cached files (whose mtime tells us when we last fetched).

Quantifies: how often does FMP silently restate historical values? This is
Isichenko's "vendors sometimes 'improve' history" warning made measurable.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import requests
import pandas as pd
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
KEY = open(ROOT / ".env").read().split("=")[1].strip()
BASE = "https://financialmodelingprep.com"
LOG = ROOT / "extension/run_log.md"
OUT = ROOT / "extension/restate_probe_report.json"

SAMPLE_TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "WMT", "GE", "T", "PFE",
                  "BAC", "C", "GS", "GOOGL", "AMZN", "TSLA", "META",
                  "HD", "CSCO", "DIS", "VZ", "INTC"]
NUMERIC_FIELDS_TO_CHECK = [  # focus on common values that are restated
    "revenue", "grossProfit", "operatingIncome", "netIncome", "ebitda", "eps",
    "weightedAverageShsOut", "totalAssets", "totalEquity", "totalDebt",
    "operatingCashFlow", "capitalExpenditure", "freeCashFlow",
]


def log(msg):
    print(msg, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def fetch(endpoint, sym, period="quarter", limit=80):
    r = requests.get(f"{BASE}/stable/{endpoint}",
                     params={"symbol": sym, "period": period, "limit": str(limit), "apikey": KEY},
                     timeout=30)
    if r.status_code == 200:
        return r.json()
    return None


def diff_records(old_df, new_records, field_list):
    """For each (date, field) present in BOTH old and new, count differences > 1e-6."""
    if old_df is None or old_df.empty or not new_records:
        return {"old_n": 0, "new_n": len(new_records or []), "common": 0,
                "diffs_by_field": {}, "rel_diffs": []}
    new_df = pd.DataFrame(new_records)
    if "date" not in new_df.columns:
        return None
    new_df["date"] = pd.to_datetime(new_df["date"])
    new_df = new_df.set_index("date").sort_index()

    # Old df may have 'date' as index (cached files do) or as column
    if old_df.index.name != "date":
        if "date" in old_df.columns:
            old_df = old_df.copy()
            old_df["date"] = pd.to_datetime(old_df["date"])
            old_df = old_df.set_index("date").sort_index()
        else:
            return None

    common = old_df.index.intersection(new_df.index)
    diffs = {}
    rel_diffs = []
    for field in field_list:
        if field not in old_df.columns or field not in new_df.columns:
            continue
        n_diff = 0
        n_common = 0
        for d in common:
            try:
                v_old = float(old_df.loc[d, field])
                v_new = float(new_df.loc[d, field])
            except (ValueError, TypeError):
                continue
            if pd.isna(v_old) or pd.isna(v_new):
                continue
            n_common += 1
            if abs(v_old - v_new) > 1e-6 * max(abs(v_old), abs(v_new), 1):
                n_diff += 1
                if v_old != 0:
                    rel_diffs.append(abs(v_new - v_old) / max(abs(v_old), 1))
        diffs[field] = {"checked": n_common, "different": n_diff,
                         "pct_changed": (n_diff / n_common * 100) if n_common else 0.0}
    return {"old_n": len(old_df), "new_n": len(new_df),
            "common": len(common), "diffs_by_field": diffs,
            "rel_diffs": rel_diffs}


def main():
    log("# Step 6 — Restate-detection probe")
    log(f"  Comparing fresh fetch (today) vs cached values for {len(SAMPLE_TICKERS)} tickers")

    # Determine cache age
    cache_dir = ROOT / "data/fmp_cache/income"
    mtimes = []
    for sym in SAMPLE_TICKERS:
        fp = cache_dir / f"{sym}.parquet"
        if fp.exists():
            mtimes.append(fp.stat().st_mtime)
    if mtimes:
        avg_age_days = (time.time() - sum(mtimes)/len(mtimes)) / 86400
        log(f"  Average cached-file age: {avg_age_days:.1f} days")

    report = {"tickers": {}}
    total_checks = 0
    total_diffs = 0
    by_field_totals = {}
    for sym in SAMPLE_TICKERS:
        log(f"\n## {sym}")
        sym_report = {}
        for endpoint, fields in [
            ("income-statement", ["revenue","grossProfit","operatingIncome","netIncome","ebitda","eps","weightedAverageShsOut"]),
            ("balance-sheet-statement", ["totalAssets","totalEquity","totalDebt"]),
            ("cash-flow-statement", ["operatingCashFlow","capitalExpenditure","freeCashFlow"]),
        ]:
            new_records = fetch(endpoint, sym)
            cache_name = endpoint.replace("-statement", "").replace("-", "")
            cache_name = {"income": "income", "balancesheet": "balance",
                          "cashflow": "cashflow"}.get(cache_name, cache_name)
            old_path = ROOT / f"data/fmp_cache/{cache_name}/{sym}.parquet"
            try:
                old_df = pd.read_parquet(old_path) if old_path.exists() else None
            except Exception:
                old_df = None
            d = diff_records(old_df, new_records, fields)
            if d is None:
                continue
            sym_report[endpoint] = d
            for field, fd in d.get("diffs_by_field", {}).items():
                total_checks += fd["checked"]
                total_diffs += fd["different"]
                if field not in by_field_totals:
                    by_field_totals[field] = {"checked": 0, "different": 0}
                by_field_totals[field]["checked"] += fd["checked"]
                by_field_totals[field]["different"] += fd["different"]
            log(f"  {endpoint:<25} common={d['common']}  "
                f"diff_fields=[" +
                " ".join(f"{f}:{fd['different']}/{fd['checked']}" for f, fd in d["diffs_by_field"].items()) + "]")
        report["tickers"][sym] = sym_report
        time.sleep(0.05)

    log("\n## Aggregate restate rate by field:")
    for field, t in sorted(by_field_totals.items(), key=lambda x: -x[1]["different"]):
        pct = t["different"] / max(t["checked"], 1) * 100
        log(f"  {field:<30} {t['different']:>5d}/{t['checked']:<5d} = {pct:>6.2f}%")

    overall_pct = total_diffs / max(total_checks, 1) * 100
    log(f"\n## OVERALL: {total_diffs} / {total_checks} = {overall_pct:.2f}% of (ticker, quarter, field) values changed since last fetch")

    report["aggregate"] = {
        "total_checks": total_checks,
        "total_diffs": total_diffs,
        "overall_pct_changed": overall_pct,
        "by_field": by_field_totals,
    }
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    log(f"\nSaved report to {OUT}")


if __name__ == "__main__":
    main()
