"""
fetch_complete_sic.py — Get complete SIC codes for all US-traded stocks from SEC EDGAR + FMP

Two sources:
1. SEC EDGAR company_tickers.json — Has CIK→ticker→SIC for ~10k+ registrants (authoritative)
2. FMP profile endpoint — Has sector/industry for all FMP-covered stocks

This builds a comprehensive classifications.json covering the full TOP3500+ universe.
"""
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

API_KEY = os.environ.get("FMP_API_KEY", "C6T2KGmSbbsDL3sM7gjx680hmUTiEXfy")
CACHE_DIR = Path("data/fmp_cache")

# ═══════════════════════════════════════════════════════════════
# Step 1: Get all tickers in our matrices (need classifications for all)
# ═══════════════════════════════════════════════════════════════
close_df = pd.read_parquet(CACHE_DIR / "matrices" / "close.parquet")
all_tickers = close_df.columns.tolist()
print(f"Total tickers in matrices: {len(all_tickers)}")

# Load existing classifications
cls_path = CACHE_DIR / "classifications.json"
if cls_path.exists():
    with open(cls_path) as f:
        existing_cls = json.load(f)
    print(f"Existing classifications: {len(existing_cls)}")
else:
    existing_cls = {}

# Find missing
missing = [t for t in all_tickers if t not in existing_cls]
print(f"Missing classifications: {len(missing)}")

# ═══════════════════════════════════════════════════════════════
# Step 2: Fetch SEC EDGAR company tickers with SIC codes
# ═══════════════════════════════════════════════════════════════
print("\n--- SEC EDGAR: Fetching company tickers ---")
edgar_cache = CACHE_DIR / "edgar_company_tickers.json"

if edgar_cache.exists():
    with open(edgar_cache) as f:
        edgar_data = json.load(f)
    print(f"  Loaded from cache: {len(edgar_data)} entries")
else:
    headers = {
        "User-Agent": "PersonalResearch bretjhribar@gmail.com",
        "Accept": "application/json"
    }
    r = requests.get("https://efts.sec.gov/LATEST/search-index?q=*&dateRange=custom&startdt=2024-01-01&enddt=2025-01-01&forms=10-K", 
                      headers=headers, timeout=30)
    print(f"  EDGAR search: {r.status_code}")
    
    # Try the company_tickers endpoint  
    r = requests.get("https://www.sec.gov/files/company_tickers.json", 
                      headers=headers, timeout=30)
    if r.status_code == 200:
        edgar_data = r.json()
        with open(edgar_cache, "w") as f:
            json.dump(edgar_data, f)
        print(f"  Downloaded {len(edgar_data)} company entries from SEC EDGAR")
    else:
        print(f"  SEC EDGAR failed: {r.status_code}")
        edgar_data = {}

# Build ticker → SIC lookup from EDGAR
edgar_sic_cache = CACHE_DIR / "edgar_sic_lookup.json"
if edgar_sic_cache.exists():
    with open(edgar_sic_cache) as f:
        edgar_sic = json.load(f)
    print(f"  Loaded EDGAR SIC lookup: {len(edgar_sic)} tickers")
else:
    edgar_sic = {}
    if edgar_data:
        for key, entry in edgar_data.items():
            ticker = entry.get("ticker", "")
            if ticker:
                edgar_sic[ticker] = {
                    "cik": entry.get("cik_str", ""),
                    "title": entry.get("title", ""),
                }
        with open(edgar_sic_cache, "w") as f:
            json.dump(edgar_sic, f, indent=2)
        print(f"  Built EDGAR SIC lookup: {len(edgar_sic)} tickers")

# ═══════════════════════════════════════════════════════════════
# Step 3: Fetch SIC codes from SEC EDGAR submissions endpoint
# ═══════════════════════════════════════════════════════════════
print("\n--- Fetching SIC codes from SEC EDGAR submissions ---")
headers = {
    "User-Agent": "PersonalResearch bretjhribar@gmail.com",
    "Accept": "application/json"
}

# First try the bulk company_tickers_exchange endpoint which has SIC
sic_exchange_cache = CACHE_DIR / "edgar_company_tickers_exchange.json"
if sic_exchange_cache.exists():
    with open(sic_exchange_cache) as f:
        exchange_data = json.load(f)
    print(f"  Loaded exchange data: {len(exchange_data)} entries")
else:
    r = requests.get("https://www.sec.gov/files/company_tickers_exchange.json",
                      headers=headers, timeout=30)
    if r.status_code == 200:
        exchange_data = r.json()
        with open(sic_exchange_cache, "w") as f:
            json.dump(exchange_data, f)
        print(f"  Downloaded exchange data: {len(exchange_data)} entries")
    else:
        print(f"  Exchange data failed: {r.status_code}, trying alternative...")
        exchange_data = None

# Build ticker → SIC from exchange data (has SIC field)
sec_sic_map = {}
if exchange_data and "data" in exchange_data:
    fields = exchange_data.get("fields", [])
    sic_idx = fields.index("sic") if "sic" in fields else None
    ticker_idx = fields.index("ticker") if "ticker" in fields else None
    name_idx = fields.index("name") if "name" in fields else None
    exchange_idx = fields.index("exchange") if "exchange" in fields else None
    
    if sic_idx is not None and ticker_idx is not None:
        for row in exchange_data["data"]:
            ticker = row[ticker_idx]
            sic = row[sic_idx] if sic_idx < len(row) else None
            name = row[name_idx] if name_idx and name_idx < len(row) else ""
            exchange = row[exchange_idx] if exchange_idx and exchange_idx < len(row) else ""
            if ticker and sic:
                sec_sic_map[ticker] = {
                    "sic": str(sic),
                    "name": name,
                    "exchange": exchange,
                }
        print(f"  Built SIC map from exchange data: {len(sec_sic_map)} tickers")
    else:
        print(f"  Fields: {fields}")
elif exchange_data:
    print(f"  Exchange data format: {list(exchange_data.keys())[:5]}")

# ═══════════════════════════════════════════════════════════════
# Step 4: Fetch missing profiles from FMP (for sector/industry names)
# ═══════════════════════════════════════════════════════════════
print(f"\n--- FMP: Fetching profiles for missing tickers ---")

profiles_dir = CACHE_DIR / "profiles"
profiles_dir.mkdir(exist_ok=True)

# Which tickers still need FMP profile data?
still_missing = [t for t in all_tickers if t not in existing_cls]
print(f"  Tickers needing FMP profile: {len(still_missing)}")

# Batch fetch profiles (FMP supports comma-separated up to 50)  
batch_size = 50
fetched = 0
for i in range(0, len(still_missing), batch_size):
    batch = still_missing[i:i+batch_size]
    symbols = ",".join(batch)
    
    # Check cache first
    cached = all(os.path.exists(profiles_dir / f"{s}.json") for s in batch)
    if cached:
        continue
    
    url = f"https://financialmodelingprep.com/api/v3/profile/{symbols}?apikey={API_KEY}"
    try:
        r = requests.get(url, timeout=30)
        time.sleep(0.15)
        if r.status_code == 200:
            data = r.json()
            if data:
                for item in data:
                    sym = item.get("symbol", "")
                    if sym:
                        with open(profiles_dir / f"{sym}.json", "w") as f:
                            json.dump(item, f)
                        fetched += 1
        if (i // batch_size) % 10 == 0:
            print(f"  Batch {i//batch_size}: {fetched} profiles fetched so far...")
    except Exception as e:
        print(f"  Error fetching batch at {i}: {e}")
        time.sleep(2)

print(f"  Total FMP profiles fetched: {fetched}")

# ═══════════════════════════════════════════════════════════════
# Step 5: Build comprehensive classifications
# ═══════════════════════════════════════════════════════════════
print("\n--- Building comprehensive classifications ---")

# SIC division mapping (2-digit SIC → sector name)
SIC_DIVISIONS = {
    range(1, 10): "Agriculture, Forestry & Fishing",
    range(10, 15): "Mining",
    range(15, 18): "Construction",
    range(20, 40): "Manufacturing",
    range(40, 50): "Transportation & Public Utilities",
    range(50, 52): "Wholesale Trade",
    range(52, 60): "Retail Trade",
    range(60, 68): "Finance, Insurance & Real Estate",
    range(70, 90): "Services",
    range(91, 100): "Public Administration",
}

def sic_to_sector(sic_code):
    """Map 4-digit SIC to sector name."""
    try:
        sic2 = int(str(sic_code)[:2])
    except (ValueError, TypeError):
        return "Unknown"
    for r, name in SIC_DIVISIONS.items():
        if sic2 in r:
            return name
    return "Unknown"

def sic_to_industry(sic_code):
    """Map 4-digit SIC to 3-digit industry code."""
    try:
        return str(sic_code)[:3]
    except (ValueError, TypeError):
        return "000"

all_cls = dict(existing_cls)  # start with existing
new_count = 0

for ticker in all_tickers:
    if ticker in all_cls:
        continue
    
    # Try SEC EDGAR first (authoritative SIC)
    sic_info = sec_sic_map.get(ticker, {})
    sic_code = sic_info.get("sic", "")
    
    # Try FMP profile
    profile_path = profiles_dir / f"{ticker}.json"
    fmp_sector = ""
    fmp_industry = ""
    if profile_path.exists():
        try:
            with open(profile_path) as f:
                prof = json.load(f)
            if isinstance(prof, list):
                prof = prof[0] if prof else {}
            fmp_sector = prof.get("sector", "")
            fmp_industry = prof.get("industry", "")
            if not sic_code:
                sic_code = str(prof.get("sicCode", "") or "")
        except Exception:
            pass
    
    if sic_code or fmp_sector:
        sector = sic_to_sector(sic_code) if sic_code else fmp_sector
        industry = sic_to_industry(sic_code) if sic_code else ""
        subindustry = str(sic_code) if sic_code else ""
        
        all_cls[ticker] = {
            "sector": sector,
            "industry": industry,
            "subindustry": subindustry,
            "sector_name": sector,
            "industry_name": fmp_industry or sic_info.get("name", ""),
            "subindustry_name": fmp_industry or sic_info.get("name", ""),
            "fmp_sector": fmp_sector,
            "fmp_industry": fmp_industry,
            "sic_code": sic_code,
        }
        new_count += 1

print(f"  Added {new_count} new classifications")
print(f"  Total classifications: {len(all_cls)}")

# Coverage check
covered = sum(1 for t in all_tickers if t in all_cls)
print(f"  Coverage: {covered}/{len(all_tickers)} ({covered/len(all_tickers)*100:.1f}%)")

# Save
backup_path = CACHE_DIR / "classifications_backup.json"
if cls_path.exists():
    import shutil
    shutil.copy(cls_path, backup_path)
    print(f"  Backed up existing → {backup_path.name}")

with open(cls_path, "w") as f:
    json.dump(all_cls, f, indent=2)
print(f"  Saved classifications.json ({len(all_cls)} tickers)")

# Save separate SIC-only file from EDGAR
if sec_sic_map:
    sic_out = CACHE_DIR / "sic_codes_edgar.json"
    with open(sic_out, "w") as f:
        json.dump(sec_sic_map, f, indent=2)
    print(f"  Saved SEC EDGAR SIC codes → {sic_out.name} ({len(sec_sic_map)} tickers)")

# ═══════════════════════════════════════════════════════════════
# Step 6: Data Integrity Report
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("  DATA INTEGRITY REPORT")
print(f"{'='*60}")

# Sector distribution
sectors = {}
for v in all_cls.values():
    s = v.get("fmp_sector") or v.get("sector", "Unknown")
    sectors[s] = sectors.get(s, 0) + 1
print("\n  Sector distribution:")
for s in sorted(sectors, key=sectors.get, reverse=True):
    print(f"    {s:<40} {sectors[s]:>5}")

# Missing tickers
still_uncovered = [t for t in all_tickers if t not in all_cls]
if still_uncovered:
    print(f"\n  Still missing ({len(still_uncovered)}): {still_uncovered[:20]}")
else:
    print(f"\n  ✓ All {len(all_tickers)} tickers have classifications!")
