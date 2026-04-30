"""
Step 1 — Enumerate FMP delisted-companies endpoint, filter to US common stocks.

Output: data/fmp_cache/delisted_universe.json (sym, name, exchange, ipoDate, delistedDate)
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import requests
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
KEY = open(ROOT / ".env").read().split("=")[1].strip()
BASE = "https://financialmodelingprep.com"
OUT = ROOT / "data/fmp_cache/delisted_universe.json"
LOG = ROOT / "extension/run_log.md"

US_EQUITY_EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "NYSEAMEX", "NYSEARCA"}
EXCLUDE_NAME_PATTERNS = ["ETF", "ETN", "Fund", "Index"]


def log(msg):
    print(msg, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def main():
    log("# Step 1 — Enumerate delisted universe")
    all_rows = []
    page = 0
    while True:
        r = requests.get(f"{BASE}/stable/delisted-companies",
                         params={"apikey": KEY, "page": str(page), "limit": "100"}, timeout=30)
        if r.status_code != 200:
            log(f"  page {page}: HTTP {r.status_code} — stopping")
            break
        d = r.json()
        if not d:
            break
        all_rows.extend(d)
        if (page + 1) % 10 == 0:
            log(f"  page {page+1}: cumulative {len(all_rows)} entries")
        page += 1
        if page > 500:
            log("  hit page cap (500)")
            break
        time.sleep(0.05)

    log(f"  Total delisted entries pulled: {len(all_rows)}")

    us_common = []
    excluded = {"non_us_exchange": 0, "etf_or_fund": 0, "delisted_pre_2010": 0,
                "missing_dates": 0, "kept": 0}
    for row in all_rows:
        sym = row.get("symbol", "")
        name = row.get("companyName", "") or ""
        exch = row.get("exchange", "")
        ipo = row.get("ipoDate", "")
        dlst = row.get("delistedDate", "")
        if not sym or not exch:
            excluded["missing_dates"] += 1; continue
        if exch not in US_EQUITY_EXCHANGES:
            excluded["non_us_exchange"] += 1; continue
        if any(p.lower() in name.lower() for p in EXCLUDE_NAME_PATTERNS):
            excluded["etf_or_fund"] += 1; continue
        if dlst and dlst < "2010-01-01":
            excluded["delisted_pre_2010"] += 1; continue
        excluded["kept"] += 1
        us_common.append(row)

    log("\n## Filter breakdown:")
    for k, v in excluded.items():
        log(f"  {k:<25} {v:>6,d}")
    log(f"\n## Kept: {len(us_common)} US common stocks delisted 2010+")

    by_year = {}
    for r in us_common:
        y = (r.get("delistedDate") or "")[:4]
        by_year[y] = by_year.get(y, 0) + 1
    log("\n## Delisting count by year:")
    for y in sorted(by_year.keys()):
        log(f"  {y}: {by_year[y]:>5,d}")

    by_exch = {}
    for r in us_common:
        e = r.get("exchange", "")
        by_exch[e] = by_exch.get(e, 0) + 1
    log("\n## Exchange breakdown of kept universe:")
    for e, c in sorted(by_exch.items(), key=lambda x: -x[1]):
        log(f"  {e:<12} {c:>5,d}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(us_common, fh, indent=2)
    log(f"\n## Saved {len(us_common)} entries to {OUT}")


if __name__ == "__main__":
    main()
