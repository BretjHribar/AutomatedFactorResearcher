"""
Step 5 — Re-run PIT matrix rebuild on extended date range (2010+) using the
augmented universe that includes delisted tickers.

This is a thin wrapper around rebuild_pit_matrices.py logic but:
  - START_DATE="2010-01-01"
  - Uses universes_pit/membership.parquet for PIT universe filtering
  - Outputs to data/fmp_cache/matrices_pit_v2/ (won't overwrite v1)
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Override constants in rebuild_pit_matrices, then call its main
import rebuild_pit_matrices as rpm

LOG = ROOT / "extension/run_log.md"


def log(msg):
    print(msg, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def main():
    log("# Step 5 — Re-run PIT matrix rebuild on extended (2010+) data")

    # Patch the rebuild script's constants
    rpm.OUT_DIR = ROOT / "data/fmp_cache/matrices_pit_v2"
    rpm.OUT_DIR.mkdir(parents=True, exist_ok=True)
    rpm.PIT_LAG_DAYS = 1   # next-day-tradeable

    # Universe pool: use augmented (active + delisted with cached prices)
    membership_path = ROOT / "data/fmp_cache/universes_pit/membership.parquet"
    if not membership_path.exists():
        log("  ERR: PIT membership matrix not built — run 05_pit_universe.py first")
        return
    membership = pd.read_parquet(membership_path)
    tickers = sorted(membership.columns.tolist())
    log(f"  Augmented universe: {len(tickers)} tickers from membership")

    # Replace UNIVERSE_PATH-driven filter — call load_prices directly with augmented list
    matrices, tickers_ok, dates = rpm.load_prices(tickers)
    # Restrict to dates >= 2010
    keep_idx = dates >= pd.Timestamp("2010-01-01")
    new_dates = dates[keep_idx]
    for k in list(matrices):
        matrices[k] = matrices[k].loc[new_dates]
    log(f"  Trimmed to 2010+: T={len(new_dates)} N={len(tickers_ok)}")

    prices_chars = rpm.derive_timeseries_chars(matrices, new_dates)
    fund = rpm.build_pit_fundamentals(tickers_ok, new_dates)
    ratios = rpm.derive_ratios(prices_chars, fund)

    # Save
    log(f"\n[save] Writing {len(prices_chars) + len(fund) + len(ratios)} parquets to {rpm.OUT_DIR}")
    t0 = time.time()
    all_out = {**prices_chars, **fund, **ratios}
    for name, df in all_out.items():
        df.to_parquet(rpm.OUT_DIR / f"{name}.parquet")
    log(f"  done in {time.time()-t0:.1f}s")

    # Manifest
    from datetime import datetime
    manifest = {
        "build_timestamp": datetime.now().isoformat(),
        "n_tickers": len(tickers_ok),
        "n_dates": len(new_dates),
        "date_range": [str(new_dates.min()), str(new_dates.max())],
        "pit_lag_days": rpm.PIT_LAG_DAYS,
        "fields": sorted(all_out.keys()),
        "augmented_universe": True,
        "delisted_included": True,
    }
    with open(rpm.OUT_DIR / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    log(f"  Manifest: {rpm.OUT_DIR / 'manifest.json'}")
    log(f"\n## DONE — extended PIT panel at {rpm.OUT_DIR}")


if __name__ == "__main__":
    main()
