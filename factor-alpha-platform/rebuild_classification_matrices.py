"""
rebuild_classification_matrices.py — Rebuild sector/industry/subindustry
matrix parquets from the merged classifications.json (SEC EDGAR SIC codes).

These 3 matrices are time-constant (each ticker always has the same classification)
stored as integer-encoded group IDs so they can be loaded as matrices.

After this: load_data() will return correct separate codes for all 3 levels.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

CACHE_DIR = Path("data/fmp_cache")
MATRICES_DIR = CACHE_DIR / "matrices"

# Load classifications
print("Loading classifications...")
with open(CACHE_DIR / "classifications.json") as f:
    cls_raw = json.load(f)

# Load close to get full date index and ticker list
close = pd.read_parquet(MATRICES_DIR / "close.parquet")
all_tickers = close.columns.tolist()
date_idx = close.index
print(f"  {len(all_tickers)} tickers, {len(date_idx)} dates")

# Build ticker -> code mappings
print("\nBuilding classification mappings...")

for level in ("sector", "industry", "subindustry"):
    # Get raw code for each ticker
    raw_map = {}
    for t in all_tickers:
        if t in cls_raw:
            raw_map[t] = cls_raw[t].get(level, "Unknown")
        else:
            raw_map[t] = "Unknown"

    # Label-encode: unique categories -> integer ids
    unique_vals = sorted(set(raw_map.values()))
    val_to_id = {v: i for i, v in enumerate(unique_vals)}
    id_to_val = {i: v for v, i in val_to_id.items()}

    print(f"\n  {level}: {len(unique_vals)} unique groups")
    if level in ("sector", "industry"):
        print(f"    Sample groups: {unique_vals[:8]}")

    # Build the encoded series (one value per ticker, constant over time)
    encoded = pd.Series({t: val_to_id[v] for t, v in raw_map.items()})

    # Broadcast to full date x ticker matrix (constant rows)
    mat = pd.DataFrame(
        np.tile(encoded.values, (len(date_idx), 1)),
        index=date_idx,
        columns=all_tickers,
        dtype=np.int32,
    )

    # Save
    out_path = MATRICES_DIR / f"{level}.parquet"
    mat.to_parquet(out_path)
    print(f"    Saved {out_path.name}: {mat.shape} (dtype={mat.dtypes.iloc[0]})")

    # Also save the label decoder as JSON for reference
    decoder_path = CACHE_DIR / f"{level}_decoder.json"
    with open(decoder_path, "w") as f:
        json.dump(id_to_val, f, indent=2)
    print(f"    Decoder saved: {decoder_path.name}")

# Verify the 3 files are now distinct
print("\nVerification:")
for level in ("sector", "industry", "subindustry"):
    df = pd.read_parquet(MATRICES_DIR / f"{level}.parquet")
    last = df.iloc[-1]
    unique_cnt = last.nunique()
    print(f"  {level}: {unique_cnt} unique groups (expecting 11 / 239 / 360)")

# Cross-check: are industry and subindustry different now?
ind = pd.read_parquet(MATRICES_DIR / "industry.parquet").iloc[-1]
sub = pd.read_parquet(MATRICES_DIR / "subindustry.parquet").iloc[-1]
same_pct = (ind == sub).mean() * 100
print(f"\n  industry == subindustry for {same_pct:.1f}% of tickers (should be much less than 100%)")
print("Done!")
