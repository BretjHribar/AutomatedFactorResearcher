"""Verify that the bounded-history incremental signal compute matches the
full-history rerun byte-for-byte on the latest weights row.

Run before deploying any new alpha or any change to combiner / risk-model
parameters. Pin the per-market `max_lookback_bars` defaults in
`SignalServiceResource` to whatever this tool confirms.

Usage:
    python tools/verify_incremental_signal.py prod/config/research_equity.json --bars 400
    python tools/verify_incremental_signal.py prod/config/research_crypto.json --bars 1500

Exits 0 on byte-exact match (within atol), 1 on mismatch.
"""
from __future__ import annotations

import argparse
import json
import sys

from src.pipeline.signal_service import verify_incremental_signal_matches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str, help="Path to research config JSON")
    parser.add_argument("--bars", type=int, required=True,
                        help="Max lookback bars for the bounded run")
    parser.add_argument("--atol", type=float, default=1e-10,
                        help="Absolute tolerance on per-ticker weight delta (default 1e-10)")
    parser.add_argument("--rtol", type=float, default=0.0,
                        help="Relative tolerance on per-ticker weight delta")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    print(f"Verifying incremental signal compute for {args.config}", flush=True)
    print(f"  max_lookback_bars={args.bars}, atol={args.atol}, rtol={args.rtol}", flush=True)
    try:
        stats = verify_incremental_signal_matches(
            args.config,
            max_lookback_bars=args.bars,
            atol=args.atol,
            rtol=args.rtol,
            verbose=args.verbose,
        )
    except AssertionError as exc:
        print(f"FAIL: {exc}", flush=True)
        return 1
    print(json.dumps(stats, indent=2), flush=True)
    print("PASS: incremental run matches full rerun within tolerance.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
