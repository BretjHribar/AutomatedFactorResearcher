"""
build_custom_universes.py — Construct custom ADV-band and market-cap universes

Universes constructed:
  ADV bands (by rank of 20-day ADV descending):
    - TOP500TOP1000    : ranks 501-1000   (~$100M-250M ADV, large mid-cap)
    - TOP1000TOP2000   : ranks 1001-2000  (~$30M-100M ADV, mid-cap)
    - TOP1500TOP2500   : ranks 1501-2500  (~$10M-50M ADV, small/mid)
    - TOP2500TOP3500   : ranks 2501-3500  (~$1M-10M ADV, small-cap)
    - TOP1000TOP3000   : ranks 1001-3000  (wide band, combines mid + small)
    - TOP500TOP2000    : ranks 501-2000   (upper band)

  Market cap bands (using cap.parquet, filtered to TOP3500 for liquidity floor):
    - MCAP_2B_10B     : $2B-$10B market cap (large mid-cap)
    - MCAP_500M_2B    : $500M-$2B market cap (mid-cap sweet spot)
    - MCAP_200M_1B    : $200M-$1B market cap (lower mid-cap / small-cap)
    - MCAP_100M_500M  : $100M-$500M market cap (small-cap)
"""

import numpy as np
import pandas as pd
from pathlib import Path

MATRICES_DIR = Path("data/fmp_cache/matrices")
UNIVERSE_DIR = Path("data/fmp_cache/universes")

adv20_df = pd.read_parquet(MATRICES_DIR / "adv20.parquet")
cap_df   = pd.read_parquet(MATRICES_DIR / "cap.parquet")

# Existing universes for reference liquidity floors
top3500 = pd.read_parquet(UNIVERSE_DIR / "TOP3500.parquet")

def make_adv_band(adv_df, rank_lo, rank_hi, name):
    """1=in universe, 0=out, based on ADV rank [rank_lo, rank_hi] inclusive."""
    # Rank descending by ADV (rank 1 = highest ADV)
    ranks = adv_df.rank(axis=1, ascending=False, method='first')
    mask = ((ranks >= rank_lo) & (ranks <= rank_hi)).astype(int)
    out = UNIVERSE_DIR / f"{name}.parquet"
    mask.to_parquet(out)
    n_med = mask.sum(axis=1).median()
    print(f"  {name:<25}: {n_med:.0f} tickers (median) -> {out.name}")
    return mask

def make_mcap_band(cap_df, adv_df, lo_m, hi_m, name, adv_rank_floor=3500):
    """Market cap band with ADV rank floor to ensure tradability."""
    cap_m = cap_df / 1e6  # in $M
    adv_ranks = adv_df.rank(axis=1, ascending=False, method='first')
    
    in_cap = (cap_m >= lo_m) & (cap_m < hi_m)
    in_liq = adv_ranks <= adv_rank_floor   # avoid micro-caps with near-zero ADV
    mask = (in_cap & in_liq).astype(int)
    
    out = UNIVERSE_DIR / f"{name}.parquet"
    mask.to_parquet(out)
    n_med = mask.sum(axis=1).median()
    print(f"  {name:<25}: {n_med:.0f} tickers (median) -> {out.name}")
    return mask

print("Building custom ADV-band universes...")
make_adv_band(adv20_df, 501,  1000, "TOP500TOP1000")
make_adv_band(adv20_df, 1001, 2000, "TOP1000TOP2000")
make_adv_band(adv20_df, 1501, 2500, "TOP1500TOP2500")
make_adv_band(adv20_df, 2501, 3500, "TOP2500TOP3500")
make_adv_band(adv20_df, 1001, 3000, "TOP1000TOP3000")
make_adv_band(adv20_df, 501,  2000, "TOP500TOP2000")

print("\nBuilding custom market-cap universes (ADV floor: top 3500)...")
make_mcap_band(cap_df, adv20_df, 2000,  10000, "MCAP_2B_10B",    adv_rank_floor=3500)
make_mcap_band(cap_df, adv20_df, 500,   2000,  "MCAP_500M_2B",   adv_rank_floor=3500)
make_mcap_band(cap_df, adv20_df, 200,   1000,  "MCAP_200M_1B",   adv_rank_floor=3500)
make_mcap_band(cap_df, adv20_df, 100,   500,   "MCAP_100M_500M", adv_rank_floor=3500)

print("\nDone. New universes:")
for f in sorted(UNIVERSE_DIR.glob("*.parquet")):
    n = pd.read_parquet(f).clip(0,1).sum(axis=1).median()
    print(f"  {f.stem:<25}: {n:.0f} tickers")
