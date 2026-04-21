"""
universe_band.py — Band Universe Builder for IB Closing Auction Strategy.

Builds "band" universes: TOP_N to TOP_M by ADV20, e.g. TOP2000TOP3000
is the set of symbols ranked 2001-3000 by 20-day average dollar volume,
rebalanced every 20 trading days.

This module:
1. Reads existing TOP_N universe parquet files (e.g., TOP3000 and TOP2000)
2. Computes the band as set difference: TOP3000 \ TOP2000
3. Saves the result as a new universe parquet file

Usage:
    python -m src.data.universe_band                     # Build all standard bands
    python -m src.data.universe_band --band 2000 3000    # Build specific band
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Default cache directories (matches bulk_download.py)
CACHE_DIR = Path("data/fmp_cache")
UNIVERSES_DIR = CACHE_DIR / "universes"

# Standard band universe definitions: (lower_rank, upper_rank)
BAND_UNIVERSES = {
    "TOP1500TOP2500": (1500, 2500),
    "TOP2000TOP3000": (2000, 3000),
    "TOP2500TOP3500": (2500, 3500),
}

# Required TOP_N universes for each band
# Band (lo, hi) requires TOP_{lo} and TOP_{hi} to exist
REQUIRED_TOPN = {
    "TOP1500TOP2500": ("TOP1500", "TOP2500"),
    "TOP2000TOP3000": ("TOP2000", "TOP3000"),
    "TOP2500TOP3500": ("TOP2500", "TOP3500"),
}

# Extended UNIVERSE_SIZES that includes the TOP_N needed for bands
# These will be added to bulk_download.py's UNIVERSE_SIZES
EXTENDED_UNIVERSE_SIZES = {
    "TOP1500": 1500,
    "TOP2500": 2500,
    "TOP3500": 3500,
}


def build_band_universe(
    lower_rank: int,
    upper_rank: int,
    universes_dir: Path = UNIVERSES_DIR,
    save: bool = True,
) -> pd.DataFrame | None:
    """
    Build a band universe from existing TOP_N universe parquet files.

    The band TOP{lower}TOP{upper} contains symbols ranked (lower+1) to upper
    by ADV20. It is computed as: TOP_{upper} \ TOP_{lower} (set difference).

    Args:
        lower_rank: Lower bound rank (exclusive). E.g. 2000.
        upper_rank: Upper bound rank (inclusive). E.g. 3000.
        universes_dir: Directory containing universe parquet files.
        save: Whether to save the result to disk.

    Returns:
        DataFrame (dates x tickers, bool) of band universe membership,
        or None if required universe files are missing.
    """
    band_name = f"TOP{lower_rank}TOP{upper_rank}"
    top_upper_path = universes_dir / f"TOP{upper_rank}.parquet"
    top_lower_path = universes_dir / f"TOP{lower_rank}.parquet"

    # Check that required universe files exist
    if not top_upper_path.exists():
        logger.error(f"Missing universe file: {top_upper_path}")
        logger.error(f"Run bulk_download.py first with TOP{upper_rank} in UNIVERSE_SIZES")
        return None
    if not top_lower_path.exists():
        logger.error(f"Missing universe file: {top_lower_path}")
        logger.error(f"Run bulk_download.py first with TOP{lower_rank} in UNIVERSE_SIZES")
        return None

    # Load the two universe masks
    top_upper = pd.read_parquet(top_upper_path)
    top_lower = pd.read_parquet(top_lower_path)

    if not isinstance(top_upper.index, pd.DatetimeIndex):
        top_upper.index = pd.to_datetime(top_upper.index)
    if not isinstance(top_lower.index, pd.DatetimeIndex):
        top_lower.index = pd.to_datetime(top_lower.index)

    # Align columns (tickers) — use union to handle different ticker sets
    all_tickers = sorted(set(top_upper.columns) | set(top_lower.columns))
    top_upper = top_upper.reindex(columns=all_tickers, fill_value=False)
    top_lower = top_lower.reindex(columns=all_tickers, fill_value=False)

    # Align dates — use intersection
    common_dates = top_upper.index.intersection(top_lower.index)
    top_upper = top_upper.loc[common_dates]
    top_lower = top_lower.loc[common_dates]

    # Band = in TOP_{upper} but NOT in TOP_{lower}
    band = top_upper & ~top_lower

    # Validate
    avg_members = band.sum(axis=1).mean()
    expected = upper_rank - lower_rank
    logger.info(
        f"  {band_name}: avg {avg_members:.0f} members/day "
        f"(expected ~{expected}), "
        f"{len(common_dates)} dates, "
        f"{band.any(axis=0).sum()} unique tickers ever in band"
    )

    if avg_members < expected * 0.5:
        logger.warning(
            f"  WARNING: {band_name} has fewer members than expected "
            f"({avg_members:.0f} vs {expected}). "
            f"Check that TOP{lower_rank} and TOP{upper_rank} are correctly built."
        )

    if save:
        out_path = universes_dir / f"{band_name}.parquet"
        band.to_parquet(out_path)
        logger.info(f"  Saved: {out_path}")

    return band


def build_all_band_universes(
    universes_dir: Path = UNIVERSES_DIR,
) -> dict[str, pd.DataFrame]:
    """Build all standard band universes."""
    results = {}
    for band_name, (lo, hi) in BAND_UNIVERSES.items():
        logger.info(f"Building {band_name} (ranks {lo+1}-{hi})...")
        df = build_band_universe(lo, hi, universes_dir=universes_dir)
        if df is not None:
            results[band_name] = df
    return results


def validate_band_universe(
    band_name: str,
    universes_dir: Path = UNIVERSES_DIR,
) -> dict:
    """
    Validate a band universe file exists and has reasonable statistics.

    Returns dict with validation results.
    """
    path = universes_dir / f"{band_name}.parquet"
    result = {
        "name": band_name,
        "exists": path.exists(),
        "valid": False,
        "avg_members": 0,
        "n_dates": 0,
        "n_unique_tickers": 0,
    }

    if not path.exists():
        return result

    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    members_per_day = df.sum(axis=1)
    result["avg_members"] = float(members_per_day.mean())
    result["n_dates"] = len(df)
    result["n_unique_tickers"] = int(df.any(axis=0).sum())
    result["min_members"] = int(members_per_day.min())
    result["max_members"] = int(members_per_day.max())
    result["date_range"] = (str(df.index[0].date()), str(df.index[-1].date()))

    # Check for reasonable member count
    if band_name in BAND_UNIVERSES:
        lo, hi = BAND_UNIVERSES[band_name]
        expected = hi - lo
        result["expected_members"] = expected
        result["valid"] = result["avg_members"] > expected * 0.3
    else:
        result["valid"] = result["avg_members"] > 0

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Build band universes for IB strategy")
    parser.add_argument("--band", nargs=2, type=int, default=None,
                        help="Build specific band: --band 2000 3000")
    parser.add_argument("--validate", action="store_true",
                        help="Validate existing band universes")
    parser.add_argument("--dir", type=str, default=str(UNIVERSES_DIR),
                        help="Universe directory")
    args = parser.parse_args()

    univ_dir = Path(args.dir)

    if args.validate:
        print("\nBand Universe Validation:")
        print("-" * 60)
        for name in BAND_UNIVERSES:
            v = validate_band_universe(name, univ_dir)
            status = "OK" if v["valid"] else "MISSING/INVALID"
            print(f"  {name}: {status}")
            if v["exists"]:
                print(f"    Avg members: {v['avg_members']:.0f}, "
                      f"Dates: {v['n_dates']}, "
                      f"Unique tickers: {v['n_unique_tickers']}")
                if "date_range" in v:
                    print(f"    Date range: {v['date_range'][0]} to {v['date_range'][1]}")
    elif args.band:
        lo, hi = args.band
        build_band_universe(lo, hi, universes_dir=univ_dir)
    else:
        build_all_band_universes(universes_dir=univ_dir)
