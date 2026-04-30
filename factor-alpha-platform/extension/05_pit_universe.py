"""
Step 4 — Build PIT-correct per-bar universe membership.

For each ticker in (currently_active ∪ delisted), determine on each trading bar:
  ticker is investable iff:
     first_available_date <= t < delisted_date_or_today

Output: data/fmp_cache/universes_pit/membership.parquet (T × N boolean DataFrame)
        + summary stats per year on universe size

This becomes the SURVIVOR-CORRECTED universe filter the new backtest uses.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
PRICES_DIR = ROOT / "data/fmp_cache/prices"
DELISTED_JSON = ROOT / "data/fmp_cache/delisted_universe.json"
OUT_DIR = ROOT / "data/fmp_cache/universes_pit"
LOG = ROOT / "extension/run_log.md"

START_DATE = "2010-01-01"


def log(msg):
    print(msg, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def main():
    log("# Step 4 — Build PIT universe membership")
    delisted = json.load(open(DELISTED_JSON)) if DELISTED_JSON.exists() else []
    delisted_dates = {d["symbol"]: d.get("delistedDate") for d in delisted if d.get("symbol")}

    # Use a reference calendar from a high-coverage ticker like AAPL
    ref_path = PRICES_DIR / "AAPL.parquet"
    if not ref_path.exists():
        log("  WARN: no AAPL prices to use as calendar reference")
        return
    ref = pd.read_parquet(ref_path)
    common_idx = ref.index[(ref.index >= pd.Timestamp(START_DATE))]
    log(f"  Calendar from AAPL: {len(common_idx)} bars, {common_idx.min()} -> {common_idx.max()}")

    # Process every per-ticker price file
    price_files = sorted(PRICES_DIR.glob("*.parquet"))
    log(f"  Tickers with price files: {len(price_files)}")

    # Build sparse membership: for each ticker, (first_available, last_available)
    intervals = {}
    skipped = 0
    for fp in price_files:
        sym = fp.stem
        try:
            df = pd.read_parquet(fp)
        except Exception:
            skipped += 1; continue
        df = df.dropna(subset=["close"]) if "close" in df.columns else df
        if df.empty:
            skipped += 1; continue
        first_avail = df.index.min()
        last_avail = df.index.max()
        # Apply delistedDate cap if known
        delist_str = delisted_dates.get(sym)
        if delist_str:
            try:
                delist_dt = pd.Timestamp(delist_str)
                last_avail = min(last_avail, delist_dt)
            except Exception:
                pass
        intervals[sym] = (first_avail, last_avail)

    log(f"  Got intervals for {len(intervals)} tickers (skipped {skipped})")

    # Build membership matrix (T × N) — boolean
    tickers = sorted(intervals.keys())
    membership = pd.DataFrame(False, index=common_idx, columns=tickers, dtype=bool)
    for sym, (first, last) in intervals.items():
        in_window = (common_idx >= first) & (common_idx <= last)
        membership.loc[in_window, sym] = True

    # Universe size per year stats
    log("\n## Universe size by year:")
    yearly = membership.sum(axis=1).resample("YE").mean()
    for y, n in yearly.items():
        log(f"  {y.year}: {n:>5.0f} avg active tickers")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "membership.parquet"
    # Cast to bool to save space
    membership.to_parquet(out_path)
    log(f"\n## Saved membership matrix to {out_path}")
    log(f"  shape: {membership.shape}")

    # Also save IPO/delist dates for reference
    intervals_df = pd.DataFrame(
        [(s, str(f), str(l)) for s, (f, l) in intervals.items()],
        columns=["symbol", "first_available", "last_available"]
    )
    intervals_df.to_parquet(OUT_DIR / "ticker_intervals.parquet")
    log(f"  Saved per-ticker intervals to {OUT_DIR / 'ticker_intervals.parquet'}")


if __name__ == "__main__":
    main()
