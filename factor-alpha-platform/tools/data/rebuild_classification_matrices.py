"""Rebuild equity classification JSON and encoded matrix parquets.

The live/research stack consumes both:
  - data/fmp_cache/classifications.json for descriptive ticker metadata
  - data/fmp_cache/matrices/{sector,industry,subindustry}.parquet for fast
    group-neutralization in vectorized research.

This tool keeps those artifacts in sync and guards against the historical
regression where FMP industry names were copied into subindustry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "fmp_cache"
MATRICES_DIR = CACHE_DIR / "matrices"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _normalize_entry(entry: dict | None, fallback: dict | None = None) -> dict:
    entry = dict(entry or {})
    fallback = dict(fallback or {})
    sector = str(entry.get("sector") or "Unknown")
    industry = str(entry.get("industry") or "000")
    subindustry = str(entry.get("subindustry") or "0000")
    return {
        **entry,
        "sector": sector,
        "industry": industry,
        "subindustry": subindustry,
        "sector_name": str(entry.get("sector_name") or sector),
        "industry_name": str(entry.get("industry_name") or industry),
        "subindustry_name": str(entry.get("subindustry_name") or subindustry),
        "fmp_sector": str(entry.get("fmp_sector") or fallback.get("fmp_sector") or fallback.get("sector") or ""),
        "fmp_industry": str(entry.get("fmp_industry") or fallback.get("fmp_industry") or fallback.get("industry") or ""),
    }


def _encoded_matrix(labels: dict[str, str], dates: pd.Index, tickers: list[str]):
    values = sorted(set(labels.values()))
    encoder = {label: i for i, label in enumerate(values)}
    row = pd.Series({ticker: encoder[labels[ticker]] for ticker in tickers}, dtype="int32")
    mat = pd.DataFrame([row.to_numpy()] * len(dates), index=dates, columns=tickers, dtype="int32")
    decoder = {str(i): label for label, i in encoder.items()}
    return mat, decoder


def rebuild(source: Path, *, write_json: bool) -> dict:
    close = pd.read_parquet(MATRICES_DIR / "close.parquet")
    if not isinstance(close.index, pd.DatetimeIndex):
        close.index = pd.to_datetime(close.index)
    tickers = list(close.columns)

    source_cls = _load_json(source)
    current_cls = _load_json(CACHE_DIR / "classifications.json")

    classifications: dict[str, dict] = {}
    for ticker in tickers:
        classifications[ticker] = _normalize_entry(source_cls.get(ticker), fallback=current_cls.get(ticker))

    labels_by_level = {
        level: {ticker: classifications[ticker][level] for ticker in tickers}
        for level in ("sector", "industry", "subindustry")
    }

    stats = {"tickers": len(tickers)}
    for level, labels in labels_by_level.items():
        mat, decoder = _encoded_matrix(labels, close.index, tickers)
        mat.to_parquet(MATRICES_DIR / f"{level}.parquet")
        _write_json(CACHE_DIR / f"{level}_decoder.json", decoder)
        stats[f"{level}_groups"] = len(decoder)

    # Backward-compatible Series artifacts used by a few older diagnostics.
    pd.Series(labels_by_level["sector"], name="sector").to_frame().to_parquet(
        MATRICES_DIR / "_sector_groups.parquet"
    )
    pd.Series(labels_by_level["industry"], name="industry").to_frame().to_parquet(
        MATRICES_DIR / "_industry_groups.parquet"
    )

    same = sum(
        1
        for entry in classifications.values()
        if entry["industry"] == entry["subindustry"]
    )
    stats["industry_subindustry_same_pct"] = same / len(classifications) * 100.0

    if write_json:
        _write_json(CACHE_DIR / "classifications.json", classifications)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=CACHE_DIR / "classifications_pre_edgar.json",
        help="Classification JSON to encode.",
    )
    parser.add_argument(
        "--write-json",
        action="store_true",
        help="Also replace data/fmp_cache/classifications.json with the normalized source.",
    )
    args = parser.parse_args()

    source = args.source
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    stats = rebuild(source, write_json=args.write_json)

    print("classification rebuild complete")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
