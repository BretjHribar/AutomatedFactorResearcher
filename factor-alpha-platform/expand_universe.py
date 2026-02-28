"""
Expand the FMP data universe to include delisted stocks.

The screener only returns currently active stocks (isActivelyTrading=true),
which creates survivorship bias. Over 10 years (2016-2026), hundreds of 
stocks get delisted via M&A, bankruptcies, going private, etc.

WorldQuant TOP3000 includes ~3000+ unique stocks at any point because they
have the full historical universe. We need to match this.

Steps:
1. Download delisted companies list from FMP
2. Download their historical prices and fundamentals  
3. Expand the matrices with the new tickers
4. Rebuild the universes
"""
import datetime as dt
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

API_KEY = os.environ.get("FMP_API_KEY", "C6T2KGmSbbsDL3sM7gjx680hmUTiEXfy")
BASE_URL = "https://financialmodelingprep.com"
CACHE_DIR = Path("data/fmp_cache")
START_DATE = "2016-01-01"
REQUEST_DELAY = 0.15

def fmp_get(endpoint, params=None, retries=3):
    url = f"{BASE_URL}/{endpoint}"
    all_params = {"apikey": API_KEY}
    if params:
        all_params.update(params)
    for attempt in range(retries):
        try:
            r = requests.get(url, params=all_params, timeout=30)
            time.sleep(REQUEST_DELAY)
            if r.status_code == 429:
                wait = 30 * (attempt + 1)
                logger.warning(f"Rate limit hit, waiting {wait}s...")
                time.sleep(wait)
                continue
            if r.status_code != 200:
                return None
            data = r.json()
            if isinstance(data, dict) and "Error Message" in data:
                return None
            return data
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                logger.error(f"Request failed: {e}")
    return None

# ══════════════════════════════════════════════════════════════════════
# Step 1: Download delisted companies
# ══════════════════════════════════════════════════════════════════════
print("Step 1: Downloading delisted companies...")
delisted_cache = CACHE_DIR / "delisted_companies.json"

if delisted_cache.exists():
    with open(delisted_cache) as f:
        all_delisted = json.load(f)
    print(f"  Loaded from cache: {len(all_delisted)} delisted companies")
else:
    all_delisted = []
    page = 0
    while True:
        data = fmp_get("stable/delisted-companies", {"page": str(page), "limit": "1000"})
        if not data or len(data) == 0:
            break
        all_delisted.extend(data)
        print(f"  Page {page}: {len(data)} companies (total: {len(all_delisted)})")
        page += 1

    with open(delisted_cache, "w") as f:
        json.dump(all_delisted, f)
    print(f"  Saved {len(all_delisted)} delisted companies")

# Filter to US stocks delisted after 2016
delisted_df = pd.DataFrame(all_delisted)
print(f"  Total delisted: {len(delisted_df)}")
if 'delistedDate' in delisted_df.columns:
    delisted_df['delistedDate'] = pd.to_datetime(delisted_df['delistedDate'], errors='coerce')
    recent = delisted_df[delisted_df['delistedDate'] >= '2016-01-01']
    print(f"  Delisted since 2016: {len(recent)}")
if 'exchange' in delisted_df.columns:
    us_exchanges = ['NYSE', 'NASDAQ', 'AMEX', 'New York Stock Exchange', 
                    'NASDAQ Global Select', 'NASDAQ Capital Market', 'NASDAQ Global Market']
    # Show all exchanges
    print(f"  Exchanges: {delisted_df['exchange'].value_counts().head(10).to_dict()}")

# Get symbols we don't already have
existing = pd.read_parquet(CACHE_DIR / "matrices" / "close.parquet").columns.tolist()
delisted_symbols = delisted_df['symbol'].dropna().unique().tolist()
# Filter: common stock symbols only (no preferred, warrants, units)
delisted_symbols = [s for s in delisted_symbols 
                    if s.isalpha() or (s.replace('.','').replace('-','').isalnum() and len(s) <= 5)]
new_symbols = [s for s in delisted_symbols if s not in existing]
print(f"  Already have: {len(existing)} tickers")
print(f"  New delisted symbols: {len(new_symbols)}")

# ══════════════════════════════════════════════════════════════════════
# Step 2: Download prices for new symbols
# ══════════════════════════════════════════════════════════════════════
print(f"\nStep 2: Downloading prices for {len(new_symbols)} new symbols...")

prices_dir = CACHE_DIR / "prices"
prices_dir.mkdir(exist_ok=True)

def download_one_price(symbol):
    cache = prices_dir / f"{symbol}.json"
    if cache.exists():
        return symbol, True
    data = fmp_get("stable/historical-price-eod/full", {
        "symbol": symbol,
        "from": START_DATE,
    })
    if data and isinstance(data, list) and len(data) > 0:
        with open(cache, "w") as f:
            json.dump(data, f)
        return symbol, True
    return symbol, False

downloaded = 0
failed = 0
with ThreadPoolExecutor(max_workers=5) as pool:
    futures = {pool.submit(download_one_price, s): s for s in new_symbols}
    for i, future in enumerate(as_completed(futures), 1):
        sym, ok = future.result()
        if ok:
            downloaded += 1
        else:
            failed += 1
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(new_symbols)} ({downloaded} OK, {failed} failed)")

print(f"  Done: {downloaded} downloaded, {failed} failed")

# ══════════════════════════════════════════════════════════════════════
# Step 3: Expand matrices with new tickers
# ══════════════════════════════════════════════════════════════════════
print(f"\nStep 3: Expanding matrices with new price data...")

# Load existing close to get date index
close_orig = pd.read_parquet(CACHE_DIR / "matrices" / "close.parquet")
date_idx = close_orig.index

# Build new matrices from downloaded prices
new_data = {"close": {}, "open": {}, "high": {}, "low": {}, "volume": {}, "vwap": {}}
for pfile in prices_dir.iterdir():
    if not pfile.suffix == '.json':
        continue
    sym = pfile.stem
    if sym in existing:
        continue  # already in matrices
    try:
        with open(pfile) as f:
            records = json.load(f)
        if not records:
            continue
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        # Only keep data in our date range
        df = df[df.index.isin(date_idx)]
        if len(df) < 20:  # need at least ~1 month
            continue
        for field in ['close', 'open', 'high', 'low', 'volume']:
            if field in df.columns:
                new_data[field][sym] = df[field]
        # Compute VWAP approximation = (high+low+close)/3
        if all(f in df.columns for f in ['high', 'low', 'close']):
            new_data['vwap'][sym] = (df['high'] + df['low'] + df['close']) / 3.0
    except Exception:
        continue

added = len(new_data['close'])
print(f"  New tickers with valid price data: {added}")

if added > 0:
    # Expand each price matrix
    for field in ['close', 'open', 'high', 'low', 'volume', 'vwap']:
        mat_path = CACHE_DIR / "matrices" / f"{field}.parquet"
        if mat_path.exists() and new_data[field]:
            orig = pd.read_parquet(mat_path)
            new_df = pd.DataFrame(new_data[field], index=date_idx)
            combined = pd.concat([orig, new_df], axis=1)
            # Remove duplicate columns
            combined = combined.loc[:, ~combined.columns.duplicated()]
            combined.to_parquet(mat_path)
            print(f"  {field}: {orig.shape[1]} -> {combined.shape[1]} tickers")

    # Compute derived price fields for new tickers (returns, adv20, etc.)
    close_new = pd.read_parquet(CACHE_DIR / "matrices" / "close.parquet")
    vol_new = pd.read_parquet(CACHE_DIR / "matrices" / "volume.parquet")
    
    # Returns
    returns_path = CACHE_DIR / "matrices" / "returns.parquet"
    returns_new = close_new.pct_change()
    returns_new.to_parquet(returns_path)
    print(f"  returns: {returns_new.shape[1]} tickers")

    # Log returns
    log_returns = np.log(close_new / close_new.shift(1))
    log_returns.to_parquet(CACHE_DIR / "matrices" / "log_returns.parquet")

    # ADV20 = rolling 20-day average of (close * volume)
    dollars = close_new * vol_new
    adv20 = dollars.rolling(20, min_periods=10).mean()
    adv20.to_parquet(CACHE_DIR / "matrices" / "adv20.parquet")
    print(f"  adv20: {adv20.shape[1]} tickers")

    # ADV60
    adv60 = dollars.rolling(60, min_periods=30).mean()
    adv60.to_parquet(CACHE_DIR / "matrices" / "adv60.parquet")

    # Dollars traded
    dollars.to_parquet(CACHE_DIR / "matrices" / "dollars_traded.parquet")

# ══════════════════════════════════════════════════════════════════════
# Step 4: Rebuild universes
# ══════════════════════════════════════════════════════════════════════
print(f"\nStep 4: Rebuilding universes...")

close_final = pd.read_parquet(CACHE_DIR / "matrices" / "close.parquet")
vol_final = pd.read_parquet(CACHE_DIR / "matrices" / "volume.parquet")
dollars_final = close_final * vol_final
adv20_final = dollars_final.rolling(20, min_periods=10).mean()

N_DAYS = len(close_final)
REBALANCE_FREQ = 20  # trading days

UNIVERSE_SIZES = {"TOP200": 200, "TOP500": 500, "TOP1000": 1000, "TOP2000": 2000, "TOP3000": 3000}

for name, size in UNIVERSE_SIZES.items():
    membership = pd.DataFrame(False, index=close_final.index, columns=close_final.columns)
    current_members = set()
    
    for i in range(N_DAYS):
        if i % REBALANCE_FREQ == 0 or i == 0:
            # Rank by ADV20 at this date
            row = adv20_final.iloc[i].dropna().sort_values(ascending=False)
            current_members = set(row.head(size).index.tolist())
        membership.iloc[i, membership.columns.isin(current_members)] = True
    
    uni_path = CACHE_DIR / "universes" / f"{name}.parquet"
    membership.to_parquet(uni_path)
    
    daily_count = membership.sum(axis=1)
    total_unique = membership.any(axis=0).sum()
    print(f"  {name}: avg={daily_count.mean():.0f}/day, total unique={total_unique}, "
          f"max/day={daily_count.max():.0f}")

# Final stats
print(f"\n{'='*60}")
print(f"  Final: {close_final.shape[1]} total tickers x {close_final.shape[0]} days")
print(f"  Date range: {close_final.index[0].date()} to {close_final.index[-1].date()}")

# Update metadata
meta_path = CACHE_DIR / "metadata.json"
if meta_path.exists():
    with open(meta_path) as f:
        meta = json.load(f)
    meta["n_tickers"] = close_final.shape[1]
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
