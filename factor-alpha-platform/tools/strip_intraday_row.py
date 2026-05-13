"""Strip a single bad date row from contaminated FMP matrices and universes.

Triggered by the 2026-05-13 incident: firing `research_signal_job` during NYSE
session hours pulled intraday quotes from FMP's daily endpoint and appended
them as today's "close" — which then tripped the moc_trader's data-integrity
guard at the 15:56 ET force-rerun.

This script:
  1. Snapshots every contaminated file to data/fmp_cache/_intraday_leak_backup/<ts>/.
  2. Removes the bad date row from each parquet's DatetimeIndex.
  3. Writes the cleaned parquet back in place.
  4. Skips files whose latest index row != BAD_DATE.

Run once. Production should re-fetch the legitimate close at 16:30+ ET via the
scheduled EOD refresh.
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BAD_DATE = pd.Timestamp("2026-05-13")
TARGETS = [
    ROOT / "data/fmp_cache/matrices",
    ROOT / "data/fmp_cache/universes",
]
BACKUP_BASE = ROOT / "data/fmp_cache/_intraday_leak_backup"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    backup_dir = BACKUP_BASE / ts
    backup_dir.mkdir(parents=True, exist_ok=True)
    print(f"backup -> {backup_dir.relative_to(ROOT)}", flush=True)

    n_cleaned = 0
    n_skipped = 0
    n_err = 0
    for src_dir in TARGETS:
        if not src_dir.exists():
            print(f"  skip (missing): {src_dir}")
            continue
        rel = src_dir.relative_to(ROOT)
        out_backup_dir = backup_dir / rel
        out_backup_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(src_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(f)
                if not isinstance(df.index, pd.DatetimeIndex) or len(df.index) == 0:
                    n_skipped += 1
                    continue
                last = df.index[-1]
                if last != BAD_DATE:
                    n_skipped += 1
                    continue
                # Backup
                shutil.copy2(f, out_backup_dir / f.name)
                # Strip the bad row
                cleaned = df.loc[df.index != BAD_DATE]
                if cleaned.shape == df.shape:
                    print(f"  WARN {f.name}: nothing dropped")
                    n_skipped += 1
                    continue
                cleaned.to_parquet(f)
                n_cleaned += 1
                if n_cleaned <= 5 or n_cleaned % 25 == 0:
                    print(f"  cleaned {n_cleaned:3d}: {rel}/{f.name}  {df.shape} -> {cleaned.shape}", flush=True)
            except Exception as exc:
                print(f"  ERR  {f.name}: {type(exc).__name__}: {exc}")
                n_err += 1

    print(f"\nDONE  cleaned={n_cleaned}  skipped={n_skipped}  errored={n_err}", flush=True)
    print(f"      backup at: {backup_dir}")
    print(f"\nNext steps:")
    print(f"  1. The 16:30 ET EOD refresh schedule will re-fetch 2026-05-13 from FMP")
    print(f"     once the actual close is published.")
    print(f"  2. Verify post-refresh with:")
    print(f"     python -c \"import pandas as pd; print(pd.read_parquet('data/fmp_cache/matrices/close.parquet').index[-3:])\"")


if __name__ == "__main__":
    main()
