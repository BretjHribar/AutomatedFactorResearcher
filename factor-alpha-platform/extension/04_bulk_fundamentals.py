"""
Step 3 — Bulk fetch fundamentals for the augmented universe back to 2010.

For each ticker in the augmented universe:
  - income, balance, cashflow, metrics — quarterly, limit=80 (~20yr)
  - skip if cached file is fresh (mtime within MAX_CACHE_AGE_DAYS)
  - cache to data/fmp_cache/{income,balance,cashflow,metrics}/{TICKER}.parquet

Resumable.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import requests
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
KEY = open(ROOT / ".env").read().split("=")[1].strip()
BASE = "https://financialmodelingprep.com"
PRICES_DIR = ROOT / "data/fmp_cache/prices"
DELISTED_JSON = ROOT / "data/fmp_cache/delisted_universe.json"
LOG = ROOT / "extension/run_log.md"
MAX_CACHE_AGE_DAYS = 7   # refresh fundamentals if older than this

ENDPOINTS = [
    ("income-statement",        "income",   80),
    ("balance-sheet-statement", "balance",  80),
    ("cash-flow-statement",     "cashflow", 80),
    ("key-metrics",             "metrics",  80),
]

SLEEP = 0.05


def log(msg, also_print=True):
    if also_print:
        print(msg, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def needs_refresh(fp: Path):
    if not fp.exists():
        return True
    age_days = (time.time() - fp.stat().st_mtime) / 86400
    return age_days > MAX_CACHE_AGE_DAYS


def fetch_one(endpoint, sym, period="quarter", limit=80):
    while True:
        r = requests.get(f"{BASE}/stable/{endpoint}",
                         params={"symbol": sym, "period": period,
                                 "limit": str(limit), "apikey": KEY}, timeout=60)
        if r.status_code == 429:
            log(f"  429 — backing off 30s")
            time.sleep(30)
            continue
        if r.status_code != 200:
            return None, r.status_code
        d = r.json()
        if not isinstance(d, list) or not d:
            return None, "empty"
        df = pd.DataFrame(d)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.set_index("date").sort_index()
        return df, "ok"


def main():
    log("# Step 3 — Bulk fundamentals fetch")

    delisted = json.load(open(DELISTED_JSON)) if DELISTED_JSON.exists() else []
    delisted_syms = {d["symbol"] for d in delisted if d.get("symbol")}
    active_syms = {fp.stem for fp in PRICES_DIR.glob("*.parquet")}
    universe = sorted(active_syms | delisted_syms)
    log(f"  Universe size: {len(universe)} ({len(active_syms)} active + {len(delisted_syms)} delisted)")
    log(f"  Endpoints per ticker: {[e[0] for e in ENDPOINTS]}  → up to {len(universe)*len(ENDPOINTS)} API calls")

    for endpoint, dir_name, _ in ENDPOINTS:
        (ROOT / f"data/fmp_cache/{dir_name}").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_calls = 0
    n_fetched = 0
    n_failed = 0
    n_skipped = 0
    failures = []

    for i, sym in enumerate(universe):
        for endpoint, dir_name, limit in ENDPOINTS:
            fp = ROOT / f"data/fmp_cache/{dir_name}/{sym}.parquet"
            if not needs_refresh(fp):
                n_skipped += 1
                continue
            df, status = fetch_one(endpoint, sym, limit=limit)
            n_calls += 1
            if df is not None:
                try:
                    df.to_parquet(fp)
                    n_fetched += 1
                except Exception as e:
                    n_failed += 1
                    failures.append((sym, endpoint, f"write:{e}"))
            else:
                n_failed += 1
                failures.append((sym, endpoint, status))
            time.sleep(SLEEP)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = n_calls / max(elapsed, 1) * 60
            remaining = (len(universe) - i - 1) * len(ENDPOINTS)
            eta_min = remaining / max(rate / 60, 0.01) / 60
            log(f"  [{i+1}/{len(universe)}] calls={n_calls} fetched={n_fetched} "
                f"skipped={n_skipped} failed={n_failed}  rate={rate:.0f}/min  ETA={eta_min:.0f}min")

    log(f"\n## DONE in {(time.time()-t0)/60:.1f}min")
    log(f"  Total API calls: {n_calls}")
    log(f"  Fetched: {n_fetched}, Skipped: {n_skipped}, Failed: {n_failed}")
    with open(ROOT / "extension/bulk_fundamentals_failures.json", "w") as fh:
        json.dump([{"symbol": s, "endpoint": e, "reason": str(r)} for s, e, r in failures], fh, indent=2)


if __name__ == "__main__":
    main()
