"""
Bulk FMP Data Download Script.

Downloads the full US equity universe from FMP into Parquet cache:
1. Company screener -> universe of US stocks by market cap
2. Profiles with sector/industry/IPO date
3. Filter out ETFs, REITs, Funds, and stocks < 1yr since IPO
4. Historical daily OHLCV (7 years)
5. Quarterly fundamentals (income, balance sheet, cash flow, key metrics)
6. Build WQ-style universes (TOP200..TOP3000) ranked by ADV20, rebalanced every 20 trading days

Run with: python -m src.data.bulk_download

Uses concurrent downloads with rate limiting to maximize throughput.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

API_KEY = os.environ.get("FMP_API_KEY", "C6T2KGmSbbsDL3sM7gjx680hmUTiEXfy")
BASE_URL = "https://financialmodelingprep.com"
CACHE_DIR = Path("data/fmp_cache")
START_DATE = "2016-01-01"  # 10+ years of history

# Rate limiting: Premium allows much more
MAX_WORKERS = 5
REQUEST_DELAY = 0.15  # seconds between requests per thread

# Excluded industries (REITs, SPACs, shell companies)
EXCLUDED_INDUSTRIES = {
    "REIT - Diversified", "REIT - Healthcare Facilities", "REIT - Hotel & Motel",
    "REIT - Industrial", "REIT - Mortgage", "REIT - Office", "REIT - Residential",
    "REIT - Retail", "REIT - Specialty",
    "Shell Companies", "Blank Checks",
}

# Excluded sectors
EXCLUDED_SECTORS = set()  # We filter at industry level for more precision

# Minimum trading history in days before a stock enters the universe
MIN_TRADING_DAYS = 252  # ~1 year


def fmp_get(endpoint: str, params: dict | None = None, retries: int = 3) -> Any:
    """Make a GET request to FMP with retries."""
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
                logger.error(f"Request failed after {retries} retries: {e}")
    return None


# ===========================================================================
# Step 1: Get US Stock Universe
# ===========================================================================

def download_universe(min_market_cap: int = 100_000_000) -> pd.DataFrame:
    """Download full US stock universe from company screener."""
    cache_path = CACHE_DIR / "universe.parquet"
    if cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < 24:
            df = pd.read_parquet(cache_path)
            logger.info(f"Universe loaded from cache: {len(df)} stocks")
            return df

    logger.info("Downloading US stock universe from screener...")
    all_stocks = []

    for exchange in ["NYSE", "NASDAQ", "AMEX"]:
        data = fmp_get("stable/company-screener", {
            "country": "US",
            "exchange": exchange,
            "isActivelyTrading": "true",
            "isEtf": "false",
            "isFund": "false",
            "marketCapMoreThan": str(min_market_cap),
            "limit": "10000",
        })
        if data:
            all_stocks.extend(data)
            logger.info(f"  {exchange}: {len(data)} stocks")

    df = pd.DataFrame(all_stocks)
    df = df.drop_duplicates(subset="symbol")

    # Filter out REITs and excluded industries at screener level
    before = len(df)
    df = df[~df["industry"].isin(EXCLUDED_INDUSTRIES)]
    # Also catch any REIT-like industries we may have missed
    df = df[~df["industry"].str.contains("REIT", case=False, na=False)]
    logger.info(f"  Filtered out {before - len(df)} REITs/excluded industries")

    df.to_parquet(cache_path, index=False)
    logger.info(f"Universe: {len(df)} unique US stocks > ${min_market_cap/1e6:.0f}M (excl REITs)")
    return df


# ===========================================================================
# Step 2: Download Profiles (sector/industry/IPO date)
# ===========================================================================

def download_profile(symbol: str) -> dict | None:
    """Download and cache a single company profile."""
    cache_path = CACHE_DIR / "profiles" / f"{symbol}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400
        if age_days < 30:
            try:
                with open(cache_path) as f:
                    return json.load(f)
            except Exception:
                pass

    data = fmp_get("stable/profile", {"symbol": symbol})
    if data and isinstance(data, list) and len(data) > 0:
        profile = data[0]
        with open(cache_path, "w") as f:
            json.dump(profile, f)
        return profile
    return None


def download_all_profiles(symbols: list[str]) -> dict[str, dict]:
    """Download profiles for all symbols concurrently."""
    logger.info(f"Downloading profiles for {len(symbols)} symbols...")
    profiles = {}
    to_download = []

    for sym in symbols:
        cache_path = CACHE_DIR / "profiles" / f"{sym}.json"
        if cache_path.exists():
            age_days = (time.time() - cache_path.stat().st_mtime) / 86400
            if age_days < 30:
                try:
                    with open(cache_path) as f:
                        profiles[sym] = json.load(f)
                    continue
                except Exception:
                    pass
        to_download.append(sym)

    logger.info(f"  {len(profiles)} cached, {len(to_download)} to download")

    if to_download:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(download_profile, sym): sym for sym in to_download}
            done = 0
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    p = future.result()
                    if p:
                        profiles[sym] = p
                except Exception:
                    pass
                done += 1
                if done % 100 == 0:
                    logger.info(f"  Profiles: {done}/{len(to_download)}")

    logger.info(f"  Total profiles: {len(profiles)}")
    return profiles


def filter_by_profile(symbols: list[str], profiles: dict[str, dict],
                       reference_date: str = None) -> list[str]:
    """
    Filter symbols based on profile data:
    - Remove ETFs, Funds
    - Remove REITs (industry contains 'REIT')
    - Remove stocks with IPO date < 1 year before reference_date
    - Remove blank check / shell companies
    """
    if reference_date is None:
        reference_date = dt.date.today().isoformat()
    ref = pd.Timestamp(reference_date)
    min_ipo = ref - pd.Timedelta(days=365)

    kept = []
    removed = {"etf": 0, "fund": 0, "reit": 0, "ipo_too_new": 0, "no_profile": 0, "shell": 0}

    for sym in symbols:
        p = profiles.get(sym)
        if not p:
            removed["no_profile"] += 1
            continue

        # ETF / Fund check
        if p.get("isEtf", False):
            removed["etf"] += 1
            continue
        if p.get("isFund", False):
            removed["fund"] += 1
            continue

        # REIT check
        industry = p.get("industry", "") or ""
        if "REIT" in industry.upper():
            removed["reit"] += 1
            continue

        # Shell company check
        if industry in EXCLUDED_INDUSTRIES:
            removed["shell"] += 1
            continue

        # IPO date check (must have been trading for >= 1 year)
        ipo_str = p.get("ipoDate", "")
        if ipo_str:
            try:
                ipo_date = pd.Timestamp(ipo_str)
                if ipo_date > min_ipo:
                    removed["ipo_too_new"] += 1
                    continue
            except Exception:
                pass

        kept.append(sym)

    logger.info(f"Profile filter: {len(kept)} kept, {sum(removed.values())} removed")
    for reason, count in removed.items():
        if count > 0:
            logger.info(f"  Removed {count} ({reason})")

    return kept


# ===========================================================================
# Step 3: Download Historical Prices
# ===========================================================================

def download_prices(symbol: str, start_date: str = START_DATE) -> pd.DataFrame | None:
    """Download and cache historical prices for one symbol."""
    cache_dir = CACHE_DIR / "prices"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{symbol}.parquet"

    if cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < 12:
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                pass

    data = fmp_get("stable/historical-price-eod/full", {
        "symbol": symbol,
        "from": start_date,
    })
    if not data or not isinstance(data, list) or len(data) == 0:
        return None

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    cols = [c for c in ["open", "high", "low", "close", "volume", "vwap",
                         "adjClose", "changePercent"] if c in df.columns]
    df = df[cols]

    df.to_parquet(cache_path)
    return df


def download_all_prices(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Download prices for all symbols concurrently."""
    logger.info(f"Downloading prices for {len(symbols)} symbols...")
    prices = {}
    to_download = []

    for sym in symbols:
        cache_path = CACHE_DIR / "prices" / f"{sym}.parquet"
        if cache_path.exists():
            age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
            if age_hours < 12:
                try:
                    prices[sym] = pd.read_parquet(cache_path)
                    continue
                except Exception:
                    pass
        to_download.append(sym)

    logger.info(f"  {len(prices)} cached, {len(to_download)} to download")

    if to_download:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(download_prices, sym): sym for sym in to_download}
            done = 0
            failed = 0
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        prices[sym] = df
                    else:
                        failed += 1
                except Exception:
                    failed += 1
                done += 1
                if done % 100 == 0:
                    logger.info(f"  Prices: {done}/{len(to_download)} ({failed} failed)")

    logger.info(f"  Total prices: {len(prices)} symbols")
    return prices


# ===========================================================================
# Step 4: Download Fundamentals
# ===========================================================================

FUNDAMENTAL_ENDPOINTS = {
    "income": "stable/income-statement",
    "balance": "stable/balance-sheet-statement",
    "cashflow": "stable/cash-flow-statement",
    "metrics": "stable/key-metrics",
}


def download_fundamental(symbol: str, endpoint_key: str) -> pd.DataFrame | None:
    """Download and cache one fundamental dataset for one symbol."""
    cache_dir = CACHE_DIR / endpoint_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{symbol}.parquet"

    if cache_path.exists():
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400
        if age_days < 7:
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                pass

    endpoint = FUNDAMENTAL_ENDPOINTS[endpoint_key]
    data = fmp_get(endpoint, {"symbol": symbol, "period": "quarter", "limit": "40"})
    if not data or not isinstance(data, list):
        return None

    df = pd.DataFrame(data)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

    df.to_parquet(cache_path)
    return df


def download_all_fundamentals(symbols: list[str]):
    """Download all fundamental data for all symbols."""
    for ftype in FUNDAMENTAL_ENDPOINTS:
        logger.info(f"Downloading {ftype} statements for {len(symbols)} symbols...")
        to_download = []

        for sym in symbols:
            cache_path = CACHE_DIR / ftype / f"{sym}.parquet"
            if cache_path.exists():
                age_days = (time.time() - cache_path.stat().st_mtime) / 86400
                if age_days < 7:
                    continue
            to_download.append(sym)

        logger.info(f"  {len(symbols) - len(to_download)} cached, {len(to_download)} to download")

        if to_download:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futures = {pool.submit(download_fundamental, sym, ftype): sym for sym in to_download}
                done = 0
                for future in as_completed(futures):
                    done += 1
                    if done % 200 == 0:
                        logger.info(f"  {ftype}: {done}/{len(to_download)}")
            logger.info(f"  {ftype}: done")


# ===========================================================================
# Step 5: Download SIC codes from SEC EDGAR (Legacy Fallback)
#
# NOTE: Production classifications use GICS codes built by
# build_gics_classifications.py, which maps FMP sector/industry names to
# official 8-digit GICS sub-industry codes. The SIC-based code below is
# retained as a fallback for tickers where GICS mapping is unavailable.
# ===========================================================================

# SIC Division Table: 2-digit SIC -> Sector name
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


def sic_to_sector(sic_code: str) -> str:
    """Map 4-digit SIC code to sector name using official division table."""
    try:
        sic2 = int(sic_code[:2])
    except (ValueError, TypeError):
        return "Unknown"
    for rng, name in SIC_DIVISIONS.items():
        if sic2 in rng:
            return name
    return "Unknown"


def download_sic_codes(symbols: list[str]) -> dict[str, dict]:
    """
    Download SIC codes from SEC EDGAR for all symbols.

    Uses SEC company_tickers.json for ticker->CIK mapping,
    then fetches SIC from submissions endpoint.

    Returns dict of symbol -> {sic, sicDescription, sector, industry, subindustry}
    """
    cache_path = CACHE_DIR / "sic_codes.json"
    if cache_path.exists():
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400
        if age_days < 30:
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                # Check if we have enough coverage
                missing = [s for s in symbols if s not in cached]
                if len(missing) < len(symbols) * 0.1:  # <10% missing is OK
                    logger.info(f"SIC codes loaded from cache: {len(cached)} "
                                f"({len(missing)} missing)")
                    if missing:
                        logger.info(f"  Downloading {len(missing)} missing SIC codes...")
                        new = _fetch_sic_from_edgar(missing)
                        cached.update(new)
                        with open(cache_path, "w") as f:
                            json.dump(cached, f, indent=2)
                    return cached
            except Exception:
                pass

    logger.info(f"Downloading SIC codes from SEC EDGAR for {len(symbols)} symbols...")
    result = _fetch_sic_from_edgar(symbols)

    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"  SIC codes downloaded: {len(result)}")
    return result


def _fetch_sic_from_edgar(symbols: list[str]) -> dict[str, dict]:
    """Fetch SIC codes from SEC EDGAR submissions API."""
    headers = {"User-Agent": "AlphaResearch research@example.com"}

    # Get ticker -> CIK mapping
    try:
        r = requests.get("https://www.sec.gov/files/company_tickers.json",
                         headers=headers, timeout=30)
        r.raise_for_status()
        tickers_data = r.json()
    except Exception as e:
        logger.error(f"Failed to get SEC ticker mapping: {e}")
        return {}

    ticker_to_cik = {}
    for entry in tickers_data.values():
        ticker_to_cik[entry["ticker"]] = entry["cik_str"]

    result = {}
    done = 0
    missing_cik = 0

    for sym in symbols:
        cik = ticker_to_cik.get(sym)
        if not cik:
            missing_cik += 1
            continue

        try:
            r = requests.get(
                f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json",
                headers=headers, timeout=10,
            )
            time.sleep(0.11)  # SEC rate limit: 10 req/sec

            if r.status_code == 200:
                d = r.json()
                sic = d.get("sic", "")
                sic_desc = d.get("sicDescription", "")

                if sic:
                    result[sym] = {
                        "sic": sic,
                        "sicDescription": sic_desc,
                        "sector": sic_to_sector(sic),
                        "industry": sic[:3],  # 3-digit SIC = industry group
                        "industryDescription": sic_desc,
                        "subindustry": sic,   # 4-digit SIC = subindustry
                        "subindustryDescription": sic_desc,
                    }
        except Exception:
            pass

        done += 1
        if done % 200 == 0:
            logger.info(f"  SIC: {done}/{len(symbols)} ({missing_cik} no CIK)")

    logger.info(f"  SIC complete: {len(result)} found, {missing_cik} no CIK")
    return result


def build_classification_map(
    profiles: dict[str, dict],
    sic_codes: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """
    Build classification hierarchy for neutralization (LEGACY SIC fallback).

    NOTE: For production use, run build_gics_classifications.py instead,
    which produces GICS-based classifications:
      - sector: 2-digit GICS sector code (e.g., "45" = Info Tech)
      - industry_group: 4-digit GICS group (e.g., "4520")
      - industry: 6-digit GICS industry (e.g., "452030")
      - subindustry: 8-digit GICS sub-industry (e.g., "45203010")

    This SIC-based fallback uses SEC EDGAR SIC codes when available,
    falling back to FMP sector/industry as a secondary source.
    """
    classifications = {}
    sector_counts = {}
    industry_counts = {}
    subindustry_counts = {}

    for sym, profile in profiles.items():
        sic = sic_codes.get(sym, {}) if sic_codes else {}

        if sic and sic.get("sic"):
            # Use SIC-based classification (preferred)
            sector = sic["sector"]
            industry = sic["industry"]
            subindustry = sic["subindustry"]
        else:
            # Fallback to FMP profile data
            sector = profile.get("sector", "Unknown") or "Unknown"
            industry = profile.get("industry", "Unknown") or "Unknown"
            subindustry = industry

        classifications[sym] = {
            "sector": sector,
            "industry": industry,
            "subindustry": subindustry,
            # Keep descriptive names for UI display
            "sector_name": sic.get("sector", sector) if sic else sector,
            "industry_name": sic.get("industryDescription", industry) if sic else industry,
            "subindustry_name": sic.get("subindustryDescription", subindustry) if sic else subindustry,
            # Also keep FMP sector/industry as supplementary
            "fmp_sector": profile.get("sector", ""),
            "fmp_industry": profile.get("industry", ""),
        }

        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        subindustry_counts[subindustry] = subindustry_counts.get(subindustry, 0) + 1

    # Save classification map
    map_path = CACHE_DIR / "classifications.json"
    with open(map_path, "w") as f:
        json.dump(classifications, f, indent=2)

    logger.info(f"Classifications (SIC fallback): {len(classifications)} symbols")
    logger.info(f"  {len(sector_counts)} sectors, "
                f"{len(industry_counts)} industries, "
                f"{len(subindustry_counts)} subindustries")

    # Print sector breakdown
    for sec in sorted(sector_counts, key=sector_counts.get, reverse=True):
        logger.info(f"  {sec}: {sector_counts[sec]}")

    return classifications


# ===========================================================================
# Step 6: Build WQ-style Universes (ADV20-ranked, rebalanced every 20 days)
# ===========================================================================

UNIVERSE_SIZES = {
    "TOP200": 200,
    "TOP500": 500,
    "TOP1000": 1000,
    "TOP1500": 1500,
    "TOP2000": 2000,
    "TOP2500": 2500,
    "TOP3000": 3000,
    "TOP3500": 3500,
}


def build_universes(
    prices: dict[str, pd.DataFrame],
    profiles: dict[str, dict],
    rebalance_period: int = 20,
) -> dict[str, pd.DataFrame]:
    """
    Build WQ-compatible universe membership.

    Rules (matching WorldQuant BRAIN):
    - Ranked by ADV (Average Daily dollar Volume) over the trailing 20 days
    - Rebalanced every `rebalance_period` trading days (default 20)
    - No IPOs or new stocks until they've been trading for >= 1 year (252 days)
    - No ETFs, REITs, Funds (already filtered earlier)
    - Universe membership is a boolean mask (dates x tickers)

    Returns dict of universe_name -> DataFrame (dates x tickers, bool).
    """
    logger.info("Building WQ-style universes...")

    # Get all tickers and dates
    tickers = sorted(prices.keys())
    all_dates = sorted(set().union(*(df.index for df in prices.values())))

    # Build close x volume matrix for ADV computation
    close_mat = pd.DataFrame(index=all_dates, columns=tickers, dtype=float)
    volume_mat = pd.DataFrame(index=all_dates, columns=tickers, dtype=float)
    for sym, df in prices.items():
        if "close" in df.columns:
            close_mat[sym] = df["close"].reindex(all_dates)
        if "volume" in df.columns:
            volume_mat[sym] = df["volume"].reindex(all_dates)

    # Dollar volume = close * volume
    dv = close_mat * volume_mat

    # ADV20 = 20-day rolling mean of dollar volume
    adv20 = dv.rolling(20, min_periods=10).mean()

    # Track first valid date per stock (for IPO/seasoning filter)
    first_valid = {}
    for sym in tickers:
        valid = close_mat[sym].dropna()
        if not valid.empty:
            first_valid[sym] = valid.index[0]

    # Get IPO dates from profiles (more accurate than first price date)
    ipo_dates = {}
    for sym in tickers:
        p = profiles.get(sym, {})
        ipo_str = p.get("ipoDate", "")
        if ipo_str:
            try:
                ipo_dates[sym] = pd.Timestamp(ipo_str)
            except Exception:
                pass
        # Fallback: first date with valid price
        if sym not in ipo_dates and sym in first_valid:
            ipo_dates[sym] = first_valid[sym]

    # Identify rebalance dates (every rebalance_period trading days)
    rebal_dates = all_dates[::rebalance_period]
    logger.info(f"  {len(rebal_dates)} rebalance dates (every {rebalance_period} trading days)")

    # Build universe membership for each rebalance date
    universes = {}
    for univ_name in UNIVERSE_SIZES:
        universes[univ_name] = pd.DataFrame(
            False, index=all_dates, columns=tickers
        )

    for i, reb_date in enumerate(rebal_dates):
        # ADV20 snapshot at rebalance date
        adv_row = adv20.loc[reb_date].dropna()

        # Filter: must have been trading for >= MIN_TRADING_DAYS
        eligible = []
        for sym in adv_row.index:
            if sym not in ipo_dates:
                continue
            days_since_ipo = (reb_date - ipo_dates[sym]).days
            if days_since_ipo >= MIN_TRADING_DAYS:
                eligible.append(sym)

        # Rank eligible stocks by ADV20 descending
        adv_eligible = adv_row[eligible].sort_values(ascending=False)

        # Determine date range this rebalance covers
        if i + 1 < len(rebal_dates):
            end_date = rebal_dates[i + 1]
        else:
            end_date = all_dates[-1]

        date_mask = (pd.Index(all_dates) >= reb_date) & (pd.Index(all_dates) < end_date)
        period_dates = pd.Index(all_dates)[date_mask]

        # Assign to each universe tier
        for univ_name, size in UNIVERSE_SIZES.items():
            top_n = adv_eligible.head(min(size, len(adv_eligible))).index.tolist()
            universes[univ_name].loc[period_dates, top_n] = True

    # Handle last rebalance covering through end
    if rebal_dates:
        last_reb = rebal_dates[-1]
        remaining = [d for d in all_dates if d >= last_reb]
        if remaining:
            adv_row = adv20.loc[last_reb].dropna()
            eligible = [s for s in adv_row.index
                        if s in ipo_dates and (last_reb - ipo_dates[s]).days >= MIN_TRADING_DAYS]
            adv_eligible = adv_row[eligible].sort_values(ascending=False)
            for univ_name, size in UNIVERSE_SIZES.items():
                top_n = adv_eligible.head(min(size, len(adv_eligible))).index.tolist()
                universes[univ_name].loc[remaining, top_n] = True

    # Save universes
    univ_dir = CACHE_DIR / "universes"
    univ_dir.mkdir(exist_ok=True)
    for name, df in universes.items():
        df.to_parquet(univ_dir / f"{name}.parquet")
        member_count = df.sum(axis=1).mean()
        logger.info(f"  {name}: avg {member_count:.0f} members/day")

    return universes


# ===========================================================================
# Step 7: Build Aligned Matrices (save to Parquet)
# ===========================================================================

def build_matrices(
    prices: dict[str, pd.DataFrame],
    classifications: dict[str, dict],
    fundamentals_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Build aligned (dates x tickers) matrices from downloaded data."""
    logger.info("Building aligned matrices...")

    # Deduplicate price indices (FMP sometimes returns duplicate dates)
    for sym in list(prices.keys()):
        df = prices[sym]
        if df.index.duplicated().any():
            prices[sym] = df[~df.index.duplicated(keep='last')]

    tickers = sorted(prices.keys())
    all_dates = sorted(set().union(*(df.index for df in prices.values())))

    matrices = {}

    # Price fields
    for field in ["open", "high", "low", "close", "volume", "vwap"]:
        mat = pd.DataFrame(index=all_dates, columns=tickers, dtype=float)
        for sym, df in prices.items():
            if field in df.columns:
                mat[sym] = df[field].reindex(all_dates)
        matrices[field] = mat

    # Derived fields
    close = matrices["close"]
    vol = matrices["volume"]
    matrices["returns"] = close.pct_change()
    matrices["dollars_traded"] = close * vol
    matrices["adv20"] = matrices["dollars_traded"].rolling(20).mean()
    matrices["adv60"] = matrices["dollars_traded"].rolling(60).mean()

    # Log returns (vectorized, avoid lambda)
    ratio = close / close.shift(1)
    matrices["log_returns"] = np.log(ratio.where(ratio > 0))

    # Cap (market cap proxy = close * avg shares, or just close for now)
    matrices["cap"] = close  # Will be replaced with real market cap from metrics

    # Classification-based group matrices
    sector_map = {}
    industry_map = {}
    for sym in tickers:
        c = classifications.get(sym, {})
        sector_map[sym] = c.get("sector", "Unknown")
        industry_map[sym] = c.get("industry", "Unknown")

    # Create group Series for neutralization
    matrices["_sector_groups"] = pd.Series(sector_map, name="sector")
    matrices["_industry_groups"] = pd.Series(industry_map, name="industry")

    # Also create the 'sector' and 'industry' and 'subindustry' fields
    # as categorical integer-encoded matrices (for group_rank operator)
    sector_codes = {s: i for i, s in enumerate(sorted(set(sector_map.values())))}
    industry_codes = {s: i for i, s in enumerate(sorted(set(industry_map.values())))}

    sector_mat = pd.DataFrame(index=all_dates, columns=tickers, dtype=float)
    industry_mat = pd.DataFrame(index=all_dates, columns=tickers, dtype=float)
    for sym in tickers:
        sector_mat[sym] = sector_codes.get(sector_map.get(sym, "Unknown"), -1)
        industry_mat[sym] = industry_codes.get(industry_map.get(sym, "Unknown"), -1)

    matrices["sector"] = sector_mat
    matrices["industry"] = industry_mat
    matrices["subindustry"] = industry_mat.copy()  # Same as industry for FMP

    # Fundamental data (forward-filled to daily)
    if fundamentals_dir is None:
        fundamentals_dir = CACHE_DIR

    # Income statement fields
    income_map = {
        "revenue": "revenue",
        "operating_income": "operatingIncome",
        "gross_profit": "grossProfit",
        "net_income": "netIncome",
        "rd_expense": "researchAndDevelopmentExpenses",
        "ebitda": "ebitda",
        "eps": "eps",
        "eps_diluted": "epsdiluted",
        "sales": "revenue",
        "cost_of_revenue": "costOfRevenue",
        "sga_expense": "sellingGeneralAndAdministrativeExpenses",
        "interest_expense": "interestExpense",
        "income_tax": "incomeTaxExpense",
    }

    balance_map = {
        "total_assets": "totalAssets",
        "total_equity": "totalStockholdersEquity",
        "total_debt": "totalDebt",
        "assets_curr": "totalCurrentAssets",
        "liabilities_curr": "totalCurrentLiabilities",
        "cash": "cashAndCashEquivalents",
        "inventory": "inventory",
        "total_liabilities": "totalLiabilities",
        "shares_out": "weightedAverageShsOut",
        "goodwill": "goodwill",
        "intangibles": "intangibleAssets",
        "net_debt": "netDebt",
        "retained_earnings": "retainedEarnings",
        "receivables": "netReceivables",
        "payables": "accountPayables",
        "ppe_net": "propertyPlantEquipmentNet",
    }

    cashflow_map = {
        "cashflow_op": "operatingCashFlow",
        "capex": "capitalExpenditure",
        "free_cashflow": "freeCashFlow",
        "dividends_paid": "dividendsPaid",
        "depreciation": "depreciationAndAmortization",
        "stock_repurchase": "commonStockRepurchased",
        "debt_repayment": "debtRepayment",
    }

    metric_map = {
        "enterprise_value": "enterpriseValue",
        "invested_capital": "investedCapital",
        "market_cap_metric": "marketCap",
        "pe_ratio": "peRatio",
        "pb_ratio": "pbRatio",
        "ev_to_ebitda": "enterpriseValueOverEBITDA",
        "debt_to_equity": "debtToEquity",
        "current_ratio": "currentRatio",
        "roe": "roe",
        "roa": "roic",
        "dividend_yield": "dividendYield",
        "revenue_per_share": "revenuePerShare",
        "book_value_per_share": "bookValuePerShare",
        "tangible_book_per_share": "tangibleBookValuePerShare",
        "fcf_per_share": "freeCashFlowPerShare",
    }

    fund_configs = [
        ("income", income_map),
        ("balance", balance_map),
        ("cashflow", cashflow_map),
        ("metrics", metric_map),
    ]

    for ftype, field_map in fund_configs:
        fund_dir = fundamentals_dir / ftype
        if not fund_dir.exists():
            continue

        for our_name in field_map:
            if our_name not in matrices:
                matrices[our_name] = pd.DataFrame(
                    index=all_dates, columns=tickers, dtype=float
                )

        for sym in tickers:
            fpath = fund_dir / f"{sym}.parquet"
            if not fpath.exists():
                continue
            try:
                fdf = pd.read_parquet(fpath)
                # Deduplicate fundamental indices
                if fdf.index.duplicated().any():
                    fdf = fdf[~fdf.index.duplicated(keep='last')]
            except Exception:
                continue

            # --- Point-in-Time (PIT) fix ---
            # Use acceptedDate (SEC filing acceptance) as the date data
            # becomes available, NOT the fiscal period end date.
            # This prevents look-ahead bias in backtesting.
            pit_index = fdf.index  # default: period end date
            if "acceptedDate" in fdf.columns:
                try:
                    accepted = pd.to_datetime(fdf["acceptedDate"])
                    # Use just the date part (strip time)
                    pit_index = accepted.dt.normalize()
                except Exception:
                    pass
            elif "filingDate" in fdf.columns:
                try:
                    filed = pd.to_datetime(fdf["filingDate"])
                    # Add 1 business day buffer after filing
                    pit_index = filed + pd.Timedelta(days=1)
                except Exception:
                    pass
            else:
                # No filing date available — assume 90-day lag
                # (SEC 10-Q deadline is 40-45 days, 10-K is 60-90 days)
                pit_index = fdf.index + pd.Timedelta(days=90)

            # Re-index with PIT dates
            fdf_pit = fdf.copy()
            fdf_pit.index = pit_index
            fdf_pit = fdf_pit.sort_index()
            if fdf_pit.index.duplicated().any():
                fdf_pit = fdf_pit[~fdf_pit.index.duplicated(keep='last')]

            for our_name, fmp_name in field_map.items():
                if fmp_name in fdf_pit.columns:
                    series = fdf_pit[fmp_name]
                    daily = series.reindex(all_dates, method="ffill")
                    matrices[our_name][sym] = daily

    # --- Field aliasing to match field_catalog.py names ---
    # Map catalog names → existing matrix names so the expression engine
    # and LLM researcher can use either form
    FIELD_ALIASES = {
        # Financial Statement Data → our matrix names
        "assets": "total_assets",
        "liabilities": "total_liabilities",
        "sales": "revenue",
        "debt": "total_debt",
        "equity": "total_equity",
        "ebit": "operating_income",       # close approximation
        "income": "net_income",
        "debt_lt": "total_debt",          # long-term approximation
        "bookvalue_ps": "book_value_per_share",
        "ppent": "ppe_net",
        "cogs": "cost_of_revenue",
        "receivable": "receivables",
        "cashflow": "free_cashflow",
        "depre_amort": "depreciation",
        "cashflow_dividends": "dividends_paid",
        "cashflow_invst": "capex",        # investing CF approximation
        "return_equity": "roe",
        "return_assets": "roa",
        "gross_profit_field": "gross_profit",
        "operating_expense": "sga_expense",   # approximation
        "shares_out": "shares_out",           # from metrics
        "sharesout": "shares_out",
    }

    for alias, source in FIELD_ALIASES.items():
        if alias not in matrices and source in matrices:
            matrices[alias] = matrices[source]

    # --- Derived fields ---
    # Market cap from close × shares outstanding
    if "market_cap_metric" in matrices:
        matrices["market_cap"] = matrices["market_cap_metric"]
        matrices["cap"] = matrices["market_cap_metric"]  # Override simple proxy
    elif "close" in matrices and "shares_out" in matrices:
        matrices["market_cap"] = matrices["close"] * matrices["shares_out"]
        matrices["cap"] = matrices["market_cap"]

    # Working capital = current assets - current liabilities
    if "assets_curr" in matrices and "liabilities_curr" in matrices:
        matrices["working_capital"] = matrices["assets_curr"] - matrices["liabilities_curr"]

    # Net debt = total debt - cash
    if "total_debt" in matrices and "cash" in matrices:
        matrices["net_debt_calc"] = matrices["total_debt"] - matrices["cash"]

    # Sales per share
    if "revenue" in matrices and "shares_out" in matrices:
        so = matrices["shares_out"].replace(0, np.nan)
        matrices["sales_ps"] = matrices["revenue"] / so

    # FCF per share
    if "free_cashflow" in matrices and "shares_out" in matrices:
        so = matrices["shares_out"].replace(0, np.nan)
        matrices["fcf_per_share"] = matrices["free_cashflow"] / so

    # Inventory turnover = COGS / Inventory
    if "cost_of_revenue" in matrices and "inventory" in matrices:
        inv = matrices["inventory"].replace(0, np.nan)
        matrices["inventory_turnover"] = matrices["cost_of_revenue"] / inv

    # Gross profit = Revenue - COGS  (if not already from FMP)
    if "gross_profit" not in matrices and "revenue" in matrices and "cost_of_revenue" in matrices:
        matrices["gross_profit"] = matrices["revenue"] - matrices["cost_of_revenue"]

    # Volatility fields (computed from returns)
    if "returns" in matrices:
        for window in [10, 20, 30, 60, 90, 120, 150, 180]:
            matrices[f"historical_volatility_{window}"] = matrices["returns"].rolling(window).std() * np.sqrt(252)
            # Parkinson volatility: uses high/low range
            if "high" in matrices and "low" in matrices:
                h = matrices["high"]
                l = matrices["low"]
                ratio = np.log(h / l.replace(0, np.nan)) ** 2
                matrices[f"parkinson_volatility_{window}"] = np.sqrt(ratio.rolling(window).mean() / (4 * np.log(2)))

    logger.info(f"Matrices built: {len(matrices)} fields x {len(tickers)} tickers x {len(all_dates)} days")

    # Save metadata
    meta = {
        "tickers": tickers,
        "n_tickers": len(tickers),
        "n_days": len(all_dates),
        "start_date": str(all_dates[0]),
        "end_date": str(all_dates[-1]),
        "fields": sorted([k for k in matrices if not k.startswith("_")]),
        "n_sectors": len(set(sector_map.values())),
        "n_industries": len(set(industry_map.values())),
        "sectors": dict(sorted(
            {v: sum(1 for x in sector_map.values() if x == v)
             for v in set(sector_map.values())}.items(),
            key=lambda x: -x[1]
        )),
    }
    with open(CACHE_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save matrices
    matrices_dir = CACHE_DIR / "matrices"
    matrices_dir.mkdir(exist_ok=True)
    for name, df in matrices.items():
        if isinstance(df, pd.DataFrame):
            df.to_parquet(matrices_dir / f"{name}.parquet")
        elif isinstance(df, pd.Series):
            df.to_frame().to_parquet(matrices_dir / f"{name}.parquet")
    logger.info(f"Matrices saved to {matrices_dir}")

    return matrices


# ===========================================================================
# Main
# ===========================================================================

def main():
    global MAX_WORKERS, START_DATE

    import argparse
    parser = argparse.ArgumentParser(description="Bulk download FMP data")
    parser.add_argument("--min-cap", type=int, default=300_000_000,
                        help="Minimum market cap in USD (default 300M)")
    parser.add_argument("--max-symbols", type=int, default=3000,
                        help="Max symbols to download (default 3000)")
    parser.add_argument("--skip-fundamentals", action="store_true")
    parser.add_argument("--skip-prices", action="store_true",
                        help="Skip price download (use cached)")
    parser.add_argument("--skip-sic", action="store_true",
                        help="Skip SIC code download from SEC EDGAR")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--start-date", default=None)
    args = parser.parse_args()

    MAX_WORKERS = args.workers
    if args.start_date:
        START_DATE = args.start_date

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. Universe (screener)
    universe = download_universe(min_market_cap=args.min_cap)
    symbols = universe["symbol"].tolist()[:args.max_symbols]
    logger.info(f"Target universe: {len(symbols)} symbols")

    # 2. Profiles (FMP)
    profiles = download_all_profiles(symbols)

    # 3. Filter by profile (ETF, REIT, IPO)
    symbols = filter_by_profile(symbols, profiles)

    # 4. SIC codes from SEC EDGAR
    sic_codes = None
    if not args.skip_sic:
        sic_codes = download_sic_codes(symbols)

    # 5. Classifications (SIC-based hierarchy for neutralization)
    classifications = build_classification_map(
        {s: profiles[s] for s in symbols if s in profiles},
        sic_codes=sic_codes,
    )

    # 6. Prices
    if not args.skip_prices:
        prices = download_all_prices(symbols)
    else:
        logger.info("Loading prices from cache...")
        prices = {}
        for sym in symbols:
            p = CACHE_DIR / "prices" / f"{sym}.parquet"
            if p.exists():
                try:
                    prices[sym] = pd.read_parquet(p)
                except Exception:
                    pass
        logger.info(f"  {len(prices)} prices loaded from cache")

    # Remove symbols with no price data
    symbols = [s for s in symbols if s in prices]
    logger.info(f"Symbols with price data: {len(symbols)}")

    # 7. Fundamentals
    if not args.skip_fundamentals:
        download_all_fundamentals(symbols)

    # 8. Build universes (ADV20-ranked, rebalanced every 20 days)
    build_universes(prices, profiles)

    # 9. Build matrices
    build_matrices(prices, classifications, CACHE_DIR)

    elapsed = time.time() - t0
    logger.info(f"\n{'='*60}")
    logger.info(f"Done in {elapsed/60:.1f} minutes")
    logger.info(f"  Universe: {len(symbols)} symbols")
    logger.info(f"  Prices loaded: {len(prices)}")
    logger.info(f"  Classifications: {len(classifications)}")
    logger.info(f"  SIC codes: {len(sic_codes) if sic_codes else 'skipped'}")
    logger.info(f"  Cache: {CACHE_DIR}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
