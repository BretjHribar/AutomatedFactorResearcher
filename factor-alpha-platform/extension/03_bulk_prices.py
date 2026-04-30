"""
Step 2 — Bulk fetch prices for augmented universe back to 2010-01-01.

Universe = (currently_active in prices/) ∪ (delisted from delisted_universe.json)
For each ticker:
  - Skip if already cached AND covers 2010-01-01 to recent
  - Else fetch /stable/historical-price-eod/full from 2010-01-01
  - Cache to data/fmp_cache/prices/{TICKER}.parquet

Resumable: re-running just refreshes/adds. Logs progress to extension/run_log.md.
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
START_DATE = "2010-01-01"
SLEEP_BETWEEN = 0.05      # 1200/min theoretical
RETRY_429_BACKOFF = 30    # seconds


def log(msg, also_print=True):
    if also_print:
        print(msg, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def needs_refresh(fp: Path, target_start: str) -> bool:
    """True if cached file doesn't exist OR doesn't cover target_start."""
    if not fp.exists():
        return True
    try:
        df = pd.read_parquet(fp)
        if df.empty:
            return True
        earliest = df.index.min()
        if pd.Timestamp(target_start) < earliest:
            return True
    except Exception:
        return True
    return False


def fetch_one(sym, start_date=START_DATE):
    while True:
        r = requests.get(f"{BASE}/stable/historical-price-eod/full",
                         params={"symbol": sym, "from": start_date, "apikey": KEY},
                         timeout=60)
        if r.status_code == 429:
            log(f"  429 on {sym} — backing off {RETRY_429_BACKOFF}s")
            time.sleep(RETRY_429_BACKOFF)
            continue
        if r.status_code != 200:
            return None, r.status_code
        d = r.json()
        if not isinstance(d, list) or not d:
            return None, "empty"
        df = pd.DataFrame(d)
        if "date" not in df.columns:
            return None, "no_date"
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        keep = [c for c in ["open","high","low","close","volume","vwap"] if c in df.columns]
        return df[keep], "ok"


def main():
    log("# Step 2 — Bulk fetch prices for augmented universe (back to 2010-01-01)")

    # Build augmented universe
    delisted = []
    if DELISTED_JSON.exists():
        delisted = json.load(open(DELISTED_JSON))
    delisted_syms = {d["symbol"] for d in delisted if d.get("symbol")}
    log(f"  Delisted entries: {len(delisted_syms)}")

    active_syms = {fp.stem for fp in PRICES_DIR.glob("*.parquet")}
    log(f"  Currently cached (active): {len(active_syms)}")

    universe = sorted(active_syms | delisted_syms)
    log(f"  Augmented universe size: {len(universe)}")

    PRICES_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_skipped = n_fetched = n_failed = 0
    failed_syms = []

    for i, sym in enumerate(universe):
        fp = PRICES_DIR / f"{sym}.parquet"
        if not needs_refresh(fp, START_DATE):
            n_skipped += 1
            if (i + 1) % 500 == 0:
                log(f"  [{i+1}/{len(universe)}] skipped/fetched/failed = {n_skipped}/{n_fetched}/{n_failed}  "
                    f"(elapsed {(time.time()-t0)/60:.1f}min)")
            continue

        df, status = fetch_one(sym)
        if df is not None:
            try:
                df.to_parquet(fp)
                n_fetched += 1
            except Exception as e:
                n_failed += 1
                failed_syms.append((sym, f"write_err:{e}"))
        else:
            n_failed += 1
            failed_syms.append((sym, status))

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (n_fetched + n_failed) / max(elapsed, 1) * 60
            eta_min = (len(universe) - i - 1) / max(rate / 60, 0.01) / 60
            log(f"  [{i+1}/{len(universe)}] skipped={n_skipped} fetched={n_fetched} failed={n_failed}  "
                f"rate={rate:.0f}/min  ETA={eta_min:.0f}min", also_print=True)

        time.sleep(SLEEP_BETWEEN)

    log(f"\n## DONE in {(time.time()-t0)/60:.1f}min")
    log(f"  Skipped (already covered 2010+): {n_skipped}")
    log(f"  Fetched: {n_fetched}")
    log(f"  Failed: {n_failed}")
    if failed_syms:
        log(f"  First 30 failures: {failed_syms[:30]}")
    # Save failures list
    with open(ROOT / "extension/bulk_prices_failures.json", "w") as fh:
        json.dump([{"symbol": s, "reason": str(r)} for s, r in failed_syms], fh, indent=2)


if __name__ == "__main__":
    main()
