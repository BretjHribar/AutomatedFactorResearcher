"""
merge_sic_classifications.py — Merge SEC EDGAR authoritative SIC codes into classifications.json
and run data integrity checks on price/fundamental matrices.

This enriches every ticker with:
  - SIC code (4-digit, from SEC EDGAR)
  - SIC sector (division level)
  - SIC industry (3-digit)
  - SIC subindustry (4-digit)
  - FMP sector/industry (if available)
"""
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path("data/fmp_cache")

# ═══════════════════════════════════════════════════════════════
# Load all data sources
# ═══════════════════════════════════════════════════════════════
print("Loading data sources...")

# SEC EDGAR SIC codes (authoritative)
edgar_path = CACHE_DIR / "edgar_sic_full.json"
with open(edgar_path) as f:
    edgar_sic = json.load(f)
print(f"  SEC EDGAR SIC: {len(edgar_sic)} tickers")

# Existing classifications (from FMP)
cls_path = CACHE_DIR / "classifications.json"
with open(cls_path) as f:
    existing_cls = json.load(f)
print(f"  Existing classifications: {len(existing_cls)}")

# FMP profiles (for sector/industry names)
profiles_dir = CACHE_DIR / "profiles"

# Matrix tickers
close_df = pd.read_parquet(CACHE_DIR / "matrices" / "close.parquet")
all_tickers = close_df.columns.tolist()
print(f"  Matrix tickers: {len(all_tickers)}")

# ═══════════════════════════════════════════════════════════════
# SIC Division Mapping
# ═══════════════════════════════════════════════════════════════
SIC_DIVISIONS = {
    (1, 9): "Agriculture, Forestry & Fishing",
    (10, 14): "Mining",
    (15, 17): "Construction",
    (20, 39): "Manufacturing",
    (40, 49): "Transportation & Public Utilities",
    (50, 51): "Wholesale Trade",
    (52, 59): "Retail Trade",
    (60, 67): "Finance, Insurance & Real Estate",
    (70, 89): "Services",
    (91, 99): "Public Administration",
}

def sic_to_sector_name(sic_code):
    try:
        sic2 = int(str(sic_code)[:2])
    except (ValueError, TypeError):
        return "Unknown"
    for (lo, hi), name in SIC_DIVISIONS.items():
        if lo <= sic2 <= hi:
            return name
    return "Unknown"

# ═══════════════════════════════════════════════════════════════
# Merge: SEC EDGAR SIC + FMP into comprehensive classifications
# ═══════════════════════════════════════════════════════════════
print("\nMerging classifications...")

merged = {}
stats = Counter()

for ticker in all_tickers:
    entry = {}
    
    # Start with existing FMP classification
    if ticker in existing_cls:
        entry = dict(existing_cls[ticker])
        stats["had_fmp"] += 1
    
    # Overlay SEC EDGAR SIC (authoritative)
    if ticker in edgar_sic:
        edgar = edgar_sic[ticker]
        sic = edgar.get("sic", "")
        if sic:
            entry["sic_code"] = str(sic)
            entry["sic_description"] = edgar.get("sicDescription", "")
            entry["sic_sector"] = sic_to_sector_name(sic)
            entry["sic_industry"] = str(sic)[:3]  # 3-digit
            entry["sic_subindustry"] = str(sic)     # 4-digit
            entry["sec_name"] = edgar.get("name", "")
            entry["sec_category"] = edgar.get("category", "")
            entry["sec_entity_type"] = edgar.get("entityType", "")
            
            # Use SIC-based hierarchy for WQ-compatible neutralization
            entry["sector"] = sic_to_sector_name(sic)
            entry["sector_name"] = sic_to_sector_name(sic)
            entry["industry"] = str(sic)[:3]
            entry["industry_name"] = edgar.get("sicDescription", "")
            entry["subindustry"] = str(sic)
            entry["subindustry_name"] = edgar.get("sicDescription", "")
            
            stats["has_edgar_sic"] += 1
        else:
            stats["edgar_no_sic"] += 1
    else:
        stats["not_in_edgar"] += 1
    
    # Load FMP profile for richer sector/industry names 
    profile_path = profiles_dir / f"{ticker}.json"
    if profile_path.exists():
        try:
            with open(profile_path) as f:
                prof = json.load(f)
            if isinstance(prof, list):
                prof = prof[0] if prof else {}
            fmp_sector = prof.get("sector", "")
            fmp_industry = prof.get("industry", "")
            if fmp_sector:
                entry["fmp_sector"] = fmp_sector
            if fmp_industry:
                entry["fmp_industry"] = fmp_industry
        except Exception:
            pass
    
    if entry:
        merged[ticker] = entry
    else:
        # Completely unknown ticker
        merged[ticker] = {
            "sector": "Unknown",
            "industry": "000",
            "subindustry": "0000",
            "sector_name": "Unknown",
            "industry_name": "Unknown",
            "subindustry_name": "Unknown",
        }
        stats["unknown"] += 1

# Save merged classifications
import shutil
shutil.copy(cls_path, CACHE_DIR / "classifications_pre_edgar.json")
with open(cls_path, "w") as f:
    json.dump(merged, f, indent=2)

print(f"\nMerge stats:")
for k, v in stats.most_common():
    print(f"  {k}: {v}")
print(f"Total merged: {len(merged)}")

# ═══════════════════════════════════════════════════════════════
# Sector/Industry/Subindustry Coverage Report
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("  CLASSIFICATION HIERARCHY REPORT")
print(f"{'='*60}")

sectors = Counter(v.get("sector", "Unknown") for v in merged.values())
industries = Counter(v.get("industry", "000") for v in merged.values())
subindustries = Counter(v.get("subindustry", "0000") for v in merged.values())

print(f"\n  Unique sectors: {len(sectors)}")
print(f"  Unique industries (3-digit SIC): {len(industries)}")
print(f"  Unique subindustries (4-digit SIC): {len(subindustries)}")

print(f"\n  Sector distribution:")
for s, c in sectors.most_common():
    print(f"    {s:<45} {c:>5} ({c/len(merged)*100:5.1f}%)")

# ═══════════════════════════════════════════════════════════════
# DATA INTEGRITY CHECKS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("  DATA INTEGRITY CHECKS")
print(f"{'='*60}")

# Load all matrices
matrices = {}
mat_dir = CACHE_DIR / "matrices"
for f in mat_dir.iterdir():
    if f.suffix == ".parquet":
        matrices[f.stem] = pd.read_parquet(f)
print(f"\n  Loaded {len(matrices)} matrices")

# Check 1: Missing data coverage
print(f"\n  [1] Missing Data Coverage:")
for name, df in sorted(matrices.items()):
    total = df.size
    missing = df.isna().sum().sum()
    pct = missing / total * 100
    print(f"    {name:<25} {df.shape[0]:>5}d x {df.shape[1]:>5}t  | missing: {pct:5.1f}%")

# Check 2: Extreme returns (potential split issues)
if "returns" in matrices:
    ret = matrices["returns"]
    extreme_threshold = 1.0  # 100% daily return
    extremes = (ret.abs() > extreme_threshold).sum().sum()
    total_obs = ret.notna().sum().sum()
    print(f"\n  [2] Extreme Returns (|ret| > 100%):")
    print(f"    Total extreme observations: {extremes} ({extremes/total_obs*100:.4f}%)")
    
    # Show worst offenders
    extreme_counts = (ret.abs() > extreme_threshold).sum()
    worst = extreme_counts[extreme_counts > 0].sort_values(ascending=False).head(10)
    if len(worst) > 0:
        print(f"    Worst tickers:")
        for ticker, count in worst.items():
            max_ret = ret[ticker].abs().max()
            print(f"      {ticker:<8} {count:>3} extreme days  (max |ret|: {max_ret:.1%})")

# Check 3: Stale prices (same close for consecutive days)
if "close" in matrices:
    close = matrices["close"]
    changes = close.diff().abs()
    stale_pct = (changes == 0).sum() / changes.notna().sum() * 100
    worst_stale = stale_pct.sort_values(ascending=False).head(10)
    print(f"\n  [3] Stale Prices (unchanged close):")
    print(f"    Average stale rate: {stale_pct.mean():.1f}%")
    print(f"    Worst tickers:")
    for ticker, pct in worst_stale.items():
        print(f"      {ticker:<8} {pct:5.1f}% stale days")

# Check 4: Volume anomalies
if "volume" in matrices:
    vol = matrices["volume"]
    zero_vol = (vol == 0).sum()
    worst_zero = zero_vol.sort_values(ascending=False).head(10)
    print(f"\n  [4] Zero Volume Days:")
    print(f"    Average zero-vol rate: {(zero_vol / vol.notna().sum()).mean()*100:.1f}%")
    for ticker, count in worst_zero.items():
        total_days = vol[ticker].notna().sum()
        print(f"      {ticker:<8} {count:>5} zero-vol days ({count/total_days*100:.1f}%)")

# Check 5: Price reasonableness (penny stocks, extremely high prices)
if "close" in matrices:
    close = matrices["close"]
    last_prices = close.iloc[-1].dropna()
    penny = (last_prices < 1.0).sum()
    extreme_high = (last_prices > 10000).sum()
    print(f"\n  [5] Price Reasonableness (current):")
    print(f"    Penny stocks (< $1): {penny}")
    print(f"    Ultra-high (> $10k): {extreme_high}")
    print(f"    Median price: ${last_prices.median():.2f}")
    print(f"    Mean price: ${last_prices.mean():.2f}")

# Check 6: Split detection (large overnight price jumps with inverse volume change)
if "close" in matrices and "volume" in matrices:
    ret = matrices.get("returns", close.pct_change())
    vol_change = matrices["volume"].pct_change()
    
    # Potential splits: price drops >40% AND volume increases >100%
    potential_splits = ((ret < -0.40) & (vol_change > 1.0)).sum().sum()
    # Potential reverse splits: price jumps >100% AND volume drops >50%
    potential_rsplits = ((ret > 1.0) & (vol_change < -0.50)).sum().sum() 
    print(f"\n  [6] Split/Reverse Split Indicators:")
    print(f"    Potential unhandled splits: {potential_splits}")
    print(f"    Potential reverse splits: {potential_rsplits}")

# Check 7: ADV data coverage for universe construction
if "adv20" in matrices:
    adv = matrices["adv20"]
    last_adv = adv.iloc[-20:].mean()
    valid = last_adv.dropna()
    print(f"\n  [7] ADV20 Coverage (recent 20 days):")
    print(f"    Tickers with valid ADV: {len(valid)}/{len(all_tickers)}")
    print(f"    Median ADV: ${valid.median():,.0f}")
    
    # Check TOP3500 feasibility
    ranked = valid.sort_values(ascending=False)
    for n in [2000, 2500, 3000, 3500]:
        if len(ranked) >= n:
            adv_at_rank = ranked.iloc[n-1]
            print(f"    ADV at rank {n}: ${adv_at_rank:,.0f}")
        else:
            print(f"    ADV at rank {n}: INSUFFICIENT DATA (only {len(ranked)} tickers)")

print(f"\n{'='*60}")
print("  DONE — Classifications merged, integrity checks complete")
print(f"{'='*60}")
