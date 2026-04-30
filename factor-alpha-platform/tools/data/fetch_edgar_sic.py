"""
fetch_edgar_sic.py — Fetch SIC codes from SEC EDGAR submissions API

Uses: https://data.sec.gov/submissions/CIK{cik}.json
Each submission has 'sic' and 'sicDescription' fields.

Rate limit: 10 req/sec with proper User-Agent.
"""
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path("data/fmp_cache")

headers = {
    "User-Agent": "PersonalResearch bretjhribar@gmail.com",
    "Accept": "application/json",
}

# Load EDGAR company tickers (ticker -> CIK mapping)
edgar_path = CACHE_DIR / "edgar_company_tickers.json"
with open(edgar_path) as f:
    edgar_data = json.load(f)

# Build CIK lookup
ticker_to_cik = {}
for key, entry in edgar_data.items():
    ticker = entry.get("ticker", "")
    cik = entry.get("cik_str", "")
    if ticker and cik:
        ticker_to_cik[ticker] = str(cik)

print(f"EDGAR tickers with CIK: {len(ticker_to_cik)}")

# Get our matrix tickers
close_df = pd.read_parquet(CACHE_DIR / "matrices" / "close.parquet")
our_tickers = close_df.columns.tolist()
print(f"Our matrix tickers: {len(our_tickers)}")

# See which we can look up
can_lookup = [t for t in our_tickers if t in ticker_to_cik]
print(f"Tickers with CIK mapping: {len(can_lookup)}")

# Cache for SIC results
sic_cache_path = CACHE_DIR / "edgar_sic_full.json"
if sic_cache_path.exists():
    with open(sic_cache_path) as f:
        sic_results = json.load(f)
    print(f"Loaded existing SIC cache: {len(sic_results)} entries")
else:
    sic_results = {}

# Fetch SIC codes for each ticker via submissions API
to_fetch = [t for t in can_lookup if t not in sic_results]
print(f"Need to fetch: {len(to_fetch)} tickers")

batch_save = 100
for i, ticker in enumerate(to_fetch):
    cik = ticker_to_cik[ticker]
    cik_padded = cik.zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            data = r.json()
            sic_results[ticker] = {
                "sic": data.get("sic", ""),
                "sicDescription": data.get("sicDescription", ""),
                "name": data.get("name", ""),
                "category": data.get("category", ""),
                "entityType": data.get("entityType", ""),
                "exchanges": data.get("exchanges", []),
                "tickers": data.get("tickers", []),
            }
        elif r.status_code == 429:
            print(f"  Rate limited at {i}, waiting 30s...")
            time.sleep(30)
            continue
        else:
            sic_results[ticker] = {"sic": "", "error": f"HTTP {r.status_code}"}
        
        # Rate limit: ~8 req/sec
        time.sleep(0.12)
        
    except Exception as e:
        sic_results[ticker] = {"sic": "", "error": str(e)}
        time.sleep(1)
    
    if (i + 1) % batch_save == 0:
        with open(sic_cache_path, "w") as f:
            json.dump(sic_results, f)
        filled = sum(1 for v in sic_results.values() if v.get("sic"))
        print(f"  Progress: {i+1}/{len(to_fetch)} ({filled} with SIC)")

# Final save
with open(sic_cache_path, "w") as f:
    json.dump(sic_results, f, indent=2)

# Stats
total = len(sic_results)
with_sic = sum(1 for v in sic_results.values() if v.get("sic"))
print(f"\nFinal: {total} tickers queried, {with_sic} have SIC codes")

# SIC sector distribution
from collections import Counter
sic_divisions = {
    (1, 9): "Agriculture",
    (10, 14): "Mining",
    (15, 17): "Construction",
    (20, 39): "Manufacturing",
    (40, 49): "Transport & Utilities",
    (50, 51): "Wholesale Trade",
    (52, 59): "Retail Trade",
    (60, 67): "Finance/Insurance/RE",
    (70, 89): "Services",
    (91, 99): "Public Admin",
}

sector_counts = Counter()
for v in sic_results.values():
    sic = v.get("sic", "")
    if sic:
        try:
            sic2 = int(str(sic)[:2])
            for (lo, hi), name in sic_divisions.items():
                if lo <= sic2 <= hi:
                    sector_counts[name] += 1
                    break
            else:
                sector_counts["Other"] += 1
        except ValueError:
            sector_counts["Invalid"] += 1

print("\nSIC Sector Distribution:")
for sector, count in sector_counts.most_common():
    print(f"  {sector:<30} {count:>5}")

# Coverage check
our_covered = sum(1 for t in our_tickers if t in sic_results and sic_results[t].get("sic"))
print(f"\nOur ticker coverage: {our_covered}/{len(our_tickers)} ({our_covered/len(our_tickers)*100:.1f}%)")
