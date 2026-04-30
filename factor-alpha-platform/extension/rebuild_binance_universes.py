"""
Standalone rebuild of BINANCE_TOP{20,50,100} 1d universes from the freshly
re-ingested matrices, without waiting for ingest_binance.py to finish all
intervals.

Mirrors scripts/ingest_binance.build_universes() but for 1d only.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
EX_INFO = ROOT / "data/binance_cache/exchange_info.parquet"
UNIV_DIR = ROOT / "data/binance_cache/universes"

MIN_LISTING_AGE_DAYS = 365
# Per-interval rebal cadence (bars), matching scripts/ingest_binance.py
REBAL_BARS = {"1d": 20, "4h": 120, "12h": 40, "5m": 5760}


def main():
    interval = sys.argv[1] if len(sys.argv) > 1 else "1d"
    if interval == "1d":
        ADV20_PATH = ROOT / "data/binance_cache/matrices/adv20.parquet"
        suffix = ""
    else:
        ADV20_PATH = ROOT / f"data/binance_cache/matrices/{interval}/adv20.parquet"
        suffix = f"_{interval}"
    rebal_period = REBAL_BARS.get(interval, 20)

    print(f"Rebuilding {interval} universes from fresh matrices (rebal every {rebal_period} bars)...")
    if not ADV20_PATH.exists():
        print(f"  ERR: {ADV20_PATH} missing — run ingest first")
        sys.exit(1)
    adv20 = pd.read_parquet(ADV20_PATH)
    print(f"  adv20: {adv20.shape}, {adv20.index.min().date()} -> {adv20.index.max().date()}")

    ex = pd.read_parquet(EX_INFO)
    listing_dates = ex.set_index("symbol")["listing_date"]
    seasoning_delta = pd.Timedelta(days=MIN_LISTING_AGE_DAYS)
    pre = adv20.copy()
    for sym in adv20.columns:
        if sym in listing_dates.index:
            eligible = listing_dates[sym] + seasoning_delta
            adv20.loc[adv20.index < eligible, sym] = np.nan
        else:
            adv20[sym] = np.nan

    all_dates = adv20.index
    rebal_indices = list(range(0, len(all_dates), rebal_period))
    rebal_dates = all_dates[rebal_indices]
    print(f"  {len(rebal_dates)} rebalance dates (every {rebal_period} bars)")

    UNIV_DIR.mkdir(parents=True, exist_ok=True)
    for tier_name, tier_size in [("BINANCE_TOP100", 100), ("BINANCE_TOP50", 50), ("BINANCE_TOP20", 20)]:
        mask = pd.DataFrame(False, index=all_dates, columns=adv20.columns)
        for i, reb in enumerate(rebal_dates):
            row = adv20.loc[reb].dropna()
            row = row[row > 0]
            top_n = row.nlargest(min(tier_size, len(row))).index.tolist()
            if i + 1 < len(rebal_dates):
                end_date = rebal_dates[i + 1]
                period = all_dates[(all_dates >= reb) & (all_dates < end_date)]
            else:
                period = all_dates[all_dates >= reb]
            mask.loc[period, top_n] = True
        out_path = UNIV_DIR / f"{tier_name}{suffix}.parquet"
        mask.to_parquet(out_path)
        print(f"  {tier_name}{suffix}: avg {mask.sum(axis=1).mean():.0f} symbols/bar -> {out_path.name}")

    print("DONE")


if __name__ == "__main__":
    main()
