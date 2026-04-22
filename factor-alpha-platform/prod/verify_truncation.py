"""Verify that truncated matrix build produces identical last-row values vs full build."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

# Save the current full matrices' last rows
MATRICES_DIR = Path("data/binance_cache/matrices/4h")
fields = ["close", "returns", "adv20", "adv60", "vwap", "vwap_deviation",
          "taker_buy_ratio", "volume_ratio_20d", "historical_volatility_60",
          "historical_volatility_120", "parkinson_volatility_20",
          "high_low_range", "momentum_60d", "overnight_gap",
          "funding_rate", "funding_rate_zscore", "beta_to_btc",
          "volume_momentum_5_20", "upper_shadow", "lower_shadow"]

print("Loading full matrices (last row)...", flush=True)
full_last = {}
for f in fields:
    fpath = MATRICES_DIR / f"{f}.parquet"
    if fpath.exists():
        df = pd.read_parquet(fpath)
        full_last[f] = df.iloc[-1].copy()
        print(f"  {f}: {df.shape} -> last={df.index[-1]}")

# Now rebuild with truncation
print("\nRebuilding with tail=600...", flush=True)
t0 = time.time()
from prod.data_refresh import _build_binance_matrices
_build_binance_matrices()
elapsed = time.time() - t0
print(f"  Rebuild took {elapsed:.1f}s")

# Compare
print("\n=== COMPARISON: Full vs Truncated (last row) ===")
all_match = True
for f in fields:
    fpath = MATRICES_DIR / f"{f}.parquet"
    if not fpath.exists() or f not in full_last:
        continue
    trunc = pd.read_parquet(fpath)
    trunc_last = trunc.iloc[-1]
    full = full_last[f]
    
    # Align columns
    common = full.index.intersection(trunc_last.index)
    f_vals = full[common].astype(float)
    t_vals = trunc_last[common].astype(float)
    
    # Compare (ignoring NaN == NaN)
    both_nan = f_vals.isna() & t_vals.isna()
    both_valid = f_vals.notna() & t_vals.notna()
    
    if both_valid.sum() == 0:
        print(f"  {f:<35} NO VALID VALUES")
        continue
    
    max_diff = (f_vals[both_valid] - t_vals[both_valid]).abs().max()
    mean_diff = (f_vals[both_valid] - t_vals[both_valid]).abs().mean()
    nan_mismatch = (~both_nan & (f_vals.isna() != t_vals.isna())).sum()
    
    status = "MATCH" if max_diff < 1e-10 and nan_mismatch == 0 else "MISMATCH"
    if status == "MISMATCH":
        all_match = False
    print(f"  {f:<35} {status}  max_diff={max_diff:.2e}  nan_mismatch={nan_mismatch}  n_valid={both_valid.sum()}")

print(f"\n{'ALL FIELDS MATCH!' if all_match else 'SOME FIELDS DO NOT MATCH!'}")
print(f"Matrix shape: {trunc.shape}")
