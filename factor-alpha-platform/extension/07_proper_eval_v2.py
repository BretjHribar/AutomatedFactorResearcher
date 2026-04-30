"""
Step 7 — Re-run proper_evaluation on extended PIT-v2 panel (2010+).

Differences from proper_evaluation.py:
  - Reads matrices from matrices_pit_v2/ (survivorship-corrected, 2010+)
  - Uses PIT membership matrix universes_pit/membership.parquet for valid tickers
  - Extends OOS_WINDOWS back to 2013-04-01 (start needs ~3yr TRAIN buffer)
  - Result: 26 non-overlapping 6-month OOS windows × 5 seeds = 130 estimates
    vs. 12 × 5 = 60 in v1 → SE ~1.5x tighter
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import backtest_voc_equities_neutralized as base
import proper_evaluation as pe

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "extension/run_log.md"


def log(msg):
    print(msg, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def main():
    log("# Step 7 — Re-run proper_evaluation on extended (2010+) PIT-v2 panel")

    # ── Patch base + pe constants for v2 ──────────────────────────────────────
    pit_v2 = ROOT / "data/fmp_cache/matrices_pit_v2"
    if not pit_v2.exists():
        log("  ERR: matrices_pit_v2 not built yet — run 06_rebuild_pit_v2.py first")
        return
    base.PIT_DIR = pit_v2
    log(f"  base.PIT_DIR -> {pit_v2}")

    # Use PIT membership matrix as universe selector
    membership_path = ROOT / "data/fmp_cache/universes_pit/membership.parquet"
    if not membership_path.exists():
        log("  ERR: PIT membership matrix not built")
        return
    membership = pd.read_parquet(membership_path)
    # Coverage cutoff: ticker must be active for at least 25% of bars
    cov = membership.sum(axis=0) / len(membership)
    valid_universe = sorted(cov[cov > 0.25].index.tolist())
    log(f"  PIT universe (>25% coverage): {len(valid_universe)} tickers")

    # Replace the universe loader by writing valid_universe to a temp universe parquet
    # backtest_voc_equities_neutralized reads UNIVERSES_DIR/{UNIVERSE_NAME}.parquet
    # so we'll create a synthetic 'PITV2' universe parquet
    pit_universe_path = base.UNIVERSES_DIR / "PITV2.parquet"
    universe_df = pd.DataFrame(
        np.ones((len(membership.index), len(valid_universe)), dtype=bool),
        index=membership.index, columns=valid_universe
    )
    universe_df.to_parquet(pit_universe_path)
    base.UNIVERSE_NAME = "PITV2"
    log(f"  Synthetic PITV2 universe written: {universe_df.shape}")

    # Extended OOS windows: from 2013-04 (after 3yr TRAIN buffer) to 2026-04
    pe.OOS_RUN_START     = "2013-01-01"   # run AIPT from this bar (earlier than windows start)
    pe.OOS_WINDOWS_START = "2013-04-01"
    pe.OOS_WINDOWS_END   = "2026-04-20"
    log(f"  OOS windows: {pe.OOS_WINDOWS_START} → {pe.OOS_WINDOWS_END} (run from {pe.OOS_RUN_START})")

    # Output dir
    pe.OUT_DIR = base.RESULTS_DIR / "proper_eval_extended"
    pe.OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"  OUT_DIR: {pe.OUT_DIR}")

    # ── Run pe.main() with patched config ──────────────────────────────────────
    log("\n  Launching proper_evaluation.main() ...")
    pe.main()

    log("\n## DONE — extended evaluation complete")


if __name__ == "__main__":
    main()
