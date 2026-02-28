"""
Generate synthetic test fixture datasets and save as Parquet files.

Usage:
    python scripts/generate_fixtures.py

Generates:
    tests/fixtures/sample_data/tiny/      20 stocks × 252 days
    tests/fixtures/sample_data/small/     100 stocks × 504 days
    tests/fixtures/sample_data/medium/    500 stocks × 1260 days
"""

from __future__ import annotations

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthetic import (
    generate_tiny_fixture,
    generate_small_fixture,
    generate_medium_fixture,
    SyntheticDataset,
)


def save_dataset(dataset: SyntheticDataset, output_dir: str) -> None:
    """Save a SyntheticDataset to Parquet files."""
    os.makedirs(output_dir, exist_ok=True)

    dataset.prices.to_parquet(os.path.join(output_dir, "prices.parquet"), index=False)
    dataset.fundamentals.to_parquet(os.path.join(output_dir, "fundamentals.parquet"), index=False)
    dataset.estimates.to_parquet(os.path.join(output_dir, "estimates.parquet"), index=False)
    dataset.classifications.to_parquet(os.path.join(output_dir, "classifications.parquet"), index=False)
    dataset.universes.to_parquet(os.path.join(output_dir, "universes.parquet"), index=False)

    print(f"  Saved to {output_dir}")
    print(f"    Prices: {len(dataset.prices):,} rows")
    print(f"    Fundamentals: {len(dataset.fundamentals):,} rows")
    print(f"    Estimates: {len(dataset.estimates):,} rows")
    print(f"    Classifications: {len(dataset.classifications):,} rows")
    print(f"    Universes: {len(dataset.universes):,} rows")


def main() -> None:
    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests", "fixtures", "sample_data",
    )

    print("Generating tiny fixture (20 stocks × 252 days)...")
    save_dataset(generate_tiny_fixture(), os.path.join(base_dir, "tiny"))

    print("\nGenerating small fixture (100 stocks × 504 days)...")
    save_dataset(generate_small_fixture(), os.path.join(base_dir, "small"))

    print("\nGenerating medium fixture (500 stocks × 1260 days)...")
    save_dataset(generate_medium_fixture(), os.path.join(base_dir, "medium"))

    print("\nDone! Fixture data generated successfully.")


if __name__ == "__main__":
    main()
