"""
One-time DB merge: copy crypto research tables from data/alphas.db into the
unified data/alpha_results.db with `_crypto` suffix.

Equity tables (alphas, evaluations) stay untouched.
After this script runs, the unified DB holds:

  alphas, evaluations                        (equity, daily)
  alphas_crypto, evaluations_crypto,         (crypto, 4h)
  runs_crypto, selections_crypto,
  correlations_crypto, trial_log_crypto

Idempotent: drops `*_crypto` tables before re-inserting, so safe to re-run.
"""
from __future__ import annotations
import sys
import sqlite3
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
EQUITY_DB = ROOT / "data" / "alpha_results.db"
CRYPTO_DB = ROOT / "data" / "alphas.db"

CRYPTO_TABLES = [
    "alphas", "evaluations", "runs", "selections",
    "correlations", "trial_log",
]


def main():
    assert EQUITY_DB.exists(), f"missing {EQUITY_DB}"
    assert CRYPTO_DB.exists(), f"missing {CRYPTO_DB}"

    src = sqlite3.connect(CRYPTO_DB)
    dst = sqlite3.connect(EQUITY_DB)

    print(f"=== merging {CRYPTO_DB.name} → {EQUITY_DB.name} ===")
    src_counts = {}
    for t in CRYPTO_TABLES:
        try:
            src_counts[t] = src.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
        except sqlite3.OperationalError:
            src_counts[t] = None
    src.close()

    dst.execute("ATTACH DATABASE ? AS crypto", (str(CRYPTO_DB),))
    dst_counts = {}
    for t in CRYPTO_TABLES:
        if src_counts.get(t) is None:
            print(f"  [skip] crypto.{t} not present in source")
            continue
        new_name = f"{t}_crypto"
        # Drop existing copy
        dst.execute(f'DROP TABLE IF EXISTS "{new_name}"')
        # Recreate from source
        dst.execute(f'CREATE TABLE "{new_name}" AS SELECT * FROM crypto."{t}"')
        n = dst.execute(f'SELECT count(*) FROM "{new_name}"').fetchone()[0]
        dst_counts[t] = n
        match = "OK" if n == src_counts[t] else f"MISMATCH src={src_counts[t]} dst={n}"
        print(f"  {t:18s} → {new_name:25s} {n:>6d} rows  [{match}]")

    dst.execute("DETACH DATABASE crypto")
    dst.commit()
    dst.close()

    print()
    print("=== verification ===")
    con = sqlite3.connect(EQUITY_DB)
    for t in CRYPTO_TABLES:
        nm = f"{t}_crypto"
        try:
            n = con.execute(f'SELECT count(*) FROM "{nm}"').fetchone()[0]
            print(f"  {nm:25s} {n:>6d} rows")
        except sqlite3.OperationalError as e:
            print(f"  {nm:25s} ERROR: {e}")
    # Equity baseline (unchanged)
    for t in ("alphas", "evaluations"):
        n = con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
        print(f"  {t:25s} {n:>6d} rows  (equity, untouched)")
    con.close()


if __name__ == "__main__":
    main()
