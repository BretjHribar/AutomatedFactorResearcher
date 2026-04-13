"""
Binance Futures Data Ingestion Pipeline.

Downloads USDT-margined perpetual futures kline data (daily, 12h, 4h),
funding rates, and builds matrix-format parquets for the GP alpha engine.

Handles:
- Lookahead-free universe construction (point-in-time TOP100/50/20)
- Delisted symbols (keeps history up to delist date)
- Derived signals (returns, volatility, volume metrics)
- Rate limiting (2400 req/min weight-based)

Usage:
    python scripts/ingest_binance.py [--start 2020-01-01] [--update]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ── Config ──
BINANCE_BASE = "https://fapi.binance.com"
DATA_DIR = Path("data/binance_cache")
KLINES_DIR = DATA_DIR / "klines"
MATRICES_DIR = DATA_DIR / "matrices"
FUNDING_DIR = DATA_DIR / "funding_rates"
UNIVERSE_DIR = DATA_DIR / "universes"

INTERVALS = ["1d", "12h", "4h", "5m"]
INTERVAL_MS = {"1d": 86400000, "12h": 43200000, "4h": 14400000, "5m": 300000}
MAX_CANDLES = 1500        # Binance max per request
REQUEST_WEIGHT = 5        # Weight per klines request
MAX_WEIGHT_PER_MIN = 2400
PAUSE_SECONDS = 0.1       # Small pause between requests


def log(msg: str):
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════
# 1. Exchange Info — get all USDT perpetuals with listing dates
# ═══════════════════════════════════════════════════════════════

def get_exchange_info() -> pd.DataFrame:
    """Fetch all USDT-margined perpetual futures from Binance."""
    log("Fetching exchange info...")
    resp = requests.get(f"{BINANCE_BASE}/fapi/v1/exchangeInfo")
    resp.raise_for_status()
    data = resp.json()

    symbols = []
    for s in data["symbols"]:
        if s["contractType"] == "PERPETUAL" and s["quoteAsset"] == "USDT":
            symbols.append({
                "symbol": s["symbol"],
                "base_asset": s["baseAsset"],
                "status": s["status"],
                "listing_date": pd.Timestamp(s["onboardDate"], unit="ms"),
                "price_precision": s["pricePrecision"],
                "quantity_precision": s["quantityPrecision"],
            })

    df = pd.DataFrame(symbols)
    log(f"  Found {len(df)} USDT perpetuals "
        f"({(df['status'] == 'TRADING').sum()} active, "
        f"{(df['status'] != 'TRADING').sum()} inactive/delisted)")
    return df


# ═══════════════════════════════════════════════════════════════
# 2. Kline Download — paginated with rate limiting
# ═══════════════════════════════════════════════════════════════

def download_klines(
    symbol: str,
    interval: str,
    start_ts: int,
    end_ts: Optional[int] = None,
) -> pd.DataFrame:
    """Download klines for a single symbol/interval, paginating forward."""
    all_rows = []
    current_start = start_ts
    final_end = end_ts or int(datetime.utcnow().timestamp() * 1000)
    step_ms = INTERVAL_MS.get(interval, 86400000)

    while current_start < final_end:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "limit": MAX_CANDLES,
        }

        try:
            resp = requests.get(f"{BINANCE_BASE}/fapi/v1/klines", params=params)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            log(f"    ⚠ Error fetching {symbol} {interval}: {e}")
            break

        rows = resp.json()
        if not rows:
            break

        all_rows.extend(rows)

        # Move window forward: start after the last candle in this batch
        latest = rows[-1][0]  # open_time of last candle
        if latest <= current_start:
            break  # No progress, avoid infinite loop
        current_start = latest + step_ms

        time.sleep(PAUSE_SECONDS)

    if not all_rows:
        return pd.DataFrame()

    # Parse kline data
    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades_count",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore",
    ])

    # Convert to proper types
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume",
                 "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trades_count"] = pd.to_numeric(df["trades_count"], errors="coerce").astype(int)

    # Keep only needed columns
    df = df[["datetime", "open", "high", "low", "close", "volume",
             "quote_volume", "trades_count", "taker_buy_volume",
             "taker_buy_quote_volume"]].copy()
    df = df.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)

    return df


def download_all_klines(
    symbols_df: pd.DataFrame,
    start_date: str,
    update_mode: bool = False,
    intervals: Optional[List[str]] = None,
):
    """Download klines for all symbols across all intervals.
    
    Always resumes from last available bar if file exists.
    Full re-download only happens if file doesn't exist.
    """
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    intervals_to_use = intervals or INTERVALS

    for interval in intervals_to_use:
        interval_dir = KLINES_DIR / interval
        interval_dir.mkdir(parents=True, exist_ok=True)

        symbols = symbols_df["symbol"].tolist()
        log(f"\nDownloading {interval} klines for {len(symbols)} symbols...")

        skipped = 0
        for i, sym in enumerate(symbols):
            fpath = interval_dir / f"{sym}.parquet"

            # Always resume from last bar if file exists
            if fpath.exists():
                try:
                    existing = pd.read_parquet(fpath)
                    last_ts = int(existing["datetime"].max().timestamp() * 1000)
                    sym_start = last_ts + INTERVAL_MS[interval]
                    
                    # Skip if already up to date (within 1 day of now)
                    now_ts = int(datetime.utcnow().timestamp() * 1000)
                    if now_ts - last_ts < 2 * INTERVAL_MS[interval]:
                        skipped += 1
                        if (i + 1) % 100 == 0:
                            log(f"    {i+1}/{len(symbols)} ({skipped} skipped)")
                        continue
                except Exception:
                    sym_start = start_ts
                    existing = pd.DataFrame()
            else:
                sym_start = start_ts
                existing = pd.DataFrame()

            df = download_klines(sym, interval, sym_start)

            if fpath.exists() and not df.empty and not existing.empty:
                df = pd.concat([existing, df]).drop_duplicates("datetime").sort_values("datetime")
            
            if not df.empty:
                df.to_parquet(fpath, index=False)

            if (i + 1) % 10 == 0:
                log(f"    {i+1}/{len(symbols)} done ({sym}: {len(df)} bars, {skipped} skipped)")


# ═══════════════════════════════════════════════════════════════
# 3. Funding Rates
# ═══════════════════════════════════════════════════════════════

def download_funding_rates(
    symbols_df: pd.DataFrame,
    start_date: str,
):
    """Download historical funding rates for all symbols."""
    FUNDING_DIR.mkdir(parents=True, exist_ok=True)
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    log(f"\nDownloading funding rates...")

    all_funding = []
    for i, sym in enumerate(symbols_df["symbol"]):
        rows = []
        current_start = start_ts
        while True:
            params = {
                "symbol": sym,
                "startTime": current_start,
                "limit": 1000,
            }
            try:
                resp = requests.get(f"{BINANCE_BASE}/fapi/v1/fundingRate", params=params)
                resp.raise_for_status()
                data = resp.json()
            except:
                break

            if not data:
                break
            rows.extend(data)
            current_start = data[-1]["fundingTime"] + 1
            time.sleep(PAUSE_SECONDS)

        for r in rows:
            all_funding.append({
                "datetime": pd.Timestamp(r["fundingTime"], unit="ms"),
                "symbol": sym,
                "funding_rate": float(r["fundingRate"]),
            })

        if (i + 1) % 50 == 0:
            log(f"    {i+1}/{len(symbols_df)} symbols done")

    if all_funding:
        df = pd.DataFrame(all_funding)
        # Pivot to matrix format: datetime × symbol
        pivot = df.pivot_table(index="datetime", columns="symbol",
                               values="funding_rate", aggfunc="first")
        pivot.to_parquet(FUNDING_DIR / "funding_rates.parquet")
        log(f"  Saved funding rates: {pivot.shape}")


# ═══════════════════════════════════════════════════════════════
# 3b. Open Interest
# ═══════════════════════════════════════════════════════════════

OI_DIR = DATA_DIR / "open_interest"

def download_open_interest(
    symbols_df: pd.DataFrame,
    start_date: str,
    interval: str = "4h",
):
    """Download historical open interest for all symbols."""
    OI_DIR.mkdir(parents=True, exist_ok=True)
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    log(f"\nDownloading open interest ({interval})...")

    all_oi = []
    symbols = symbols_df[symbols_df["status"] == "TRADING"]["symbol"].tolist()
    for i, sym in enumerate(symbols):
        rows = []
        current_start = start_ts
        while True:
            params = {
                "symbol": sym,
                "period": interval,
                "startTime": current_start,
                "limit": 500,
            }
            try:
                resp = requests.get(
                    f"{BINANCE_BASE}/futures/data/openInterestHist", params=params
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                # This endpoint is more restricted, skip on error
                break

            if not data:
                break
            rows.extend(data)
            current_start = data[-1]["timestamp"] + 1
            time.sleep(PAUSE_SECONDS * 2)  # Extra caution — lower rate limit

        for r in rows:
            all_oi.append({
                "datetime": pd.Timestamp(r["timestamp"], unit="ms"),
                "symbol": sym,
                "open_interest": float(r["sumOpenInterest"]),
                "open_interest_value": float(r["sumOpenInterestValue"]),
            })

        if (i + 1) % 50 == 0:
            log(f"    {i+1}/{len(symbols)} symbols done")

    if all_oi:
        df = pd.DataFrame(all_oi)
        for col in ["open_interest", "open_interest_value"]:
            pivot = df.pivot_table(
                index="datetime", columns="symbol", values=col, aggfunc="first"
            )
            pivot.to_parquet(OI_DIR / f"{col}.parquet")
            log(f"  Saved {col}: {pivot.shape}")


# ═══════════════════════════════════════════════════════════════
# 4. Build Matrices (dates × tickers format)
# ═══════════════════════════════════════════════════════════════

def build_matrices(interval: str = "1d"):
    """
    Pivot per-symbol parquets into dates × tickers matrix format.
    Same structure as equity matrices for compatibility with GP engine.
    """
    interval_dir = KLINES_DIR / interval
    if not interval_dir.exists():
        log(f"  No data for interval {interval}")
        return

    suffix = "" if interval == "1d" else f"_{interval}"
    out_dir = MATRICES_DIR if interval == "1d" else MATRICES_DIR / interval
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"\nBuilding matrices for {interval}...")

    # Load all symbol data
    all_data = {}
    for fpath in sorted(interval_dir.glob("*.parquet")):
        sym = fpath.stem
        df = pd.read_parquet(fpath)
        if df.empty:
            continue
        df = df.set_index("datetime")
        all_data[sym] = df

    if not all_data:
        log("  No data found!")
        return

    log(f"  {len(all_data)} symbols loaded")

    # Build each field as a dates × tickers DataFrame
    fields = ["open", "high", "low", "close", "volume", "quote_volume",
              "trades_count", "taker_buy_volume", "taker_buy_quote_volume"]

    for field in fields:
        series_dict = {}
        for sym, df in all_data.items():
            if field in df.columns:
                series_dict[sym] = df[field]
        if series_dict:
            mat = pd.DataFrame(series_dict)
            mat.index.name = "date"
            mat.to_parquet(out_dir / f"{field}.parquet")
            log(f"    {field}: {mat.shape}")

    # ── Derived fields ──
    close = pd.DataFrame({s: d["close"] for s, d in all_data.items()})
    open_ = pd.DataFrame({s: d["open"] for s, d in all_data.items()})
    high = pd.DataFrame({s: d["high"] for s, d in all_data.items()})
    low = pd.DataFrame({s: d["low"] for s, d in all_data.items()})
    volume = pd.DataFrame({s: d["volume"] for s, d in all_data.items()})
    qv = pd.DataFrame({s: d["quote_volume"] for s, d in all_data.items()})
    tbv = pd.DataFrame({s: d.get("taker_buy_volume", pd.Series(dtype=float)) for s, d in all_data.items()})

    # Returns
    returns = close.pct_change()
    returns.to_parquet(out_dir / "returns.parquet")
    log(f"    returns: {returns.shape}")

    log_returns = np.log(close / close.shift(1))
    log_returns.to_parquet(out_dir / "log_returns.parquet")

    # VWAP proxy
    safe_vol = volume.replace(0, np.nan)
    vwap = qv / safe_vol
    vwap.to_parquet(out_dir / "vwap.parquet")

    # VWAP deviation
    vwap_dev = (close - vwap) / vwap
    vwap_dev.to_parquet(out_dir / "vwap_deviation.parquet")

    # Taker buy ratio (orderflow imbalance)
    tbr = tbv / safe_vol
    tbr.to_parquet(out_dir / "taker_buy_ratio.parquet")

    # ADV (rolling average daily quote volume)
    for window in [20, 60]:
        adv = qv.rolling(window, min_periods=max(window // 2, 1)).mean()
        adv.to_parquet(out_dir / f"adv{window}.parquet")
        log(f"    adv{window}: {adv.shape}")

    # Volume ratio
    adv20 = qv.rolling(20, min_periods=10).mean()
    vol_ratio = qv / adv20
    vol_ratio.to_parquet(out_dir / "volume_ratio_20d.parquet")

    # Volatility
    for window in [10, 20, 60, 120]:
        hvol = returns.rolling(window, min_periods=max(window // 2, 1)).std() * np.sqrt(252)
        hvol.to_parquet(out_dir / f"historical_volatility_{window}.parquet")
        if window <= 60:
            log(f"    historical_volatility_{window}: {hvol.shape}")

    # Parkinson volatility
    hl_ratio = np.log(high / low)
    for window in [10, 20, 60]:
        pvol = hl_ratio.rolling(window, min_periods=max(window // 2, 1)).apply(
            lambda x: np.sqrt(np.mean(x**2) / (4 * np.log(2))) * np.sqrt(252),
            raw=True,
        )
        pvol.to_parquet(out_dir / f"parkinson_volatility_{window}.parquet")

    # High-low range
    hlr = (high - low) / close
    hlr.to_parquet(out_dir / "high_low_range.parquet")

    # Open-close range
    ocr = (close - open_).abs() / close
    ocr.to_parquet(out_dir / "open_close_range.parquet")

    # Momentum
    for window in [5, 20, 60]:
        mom = close / close.shift(window) - 1
        mom.to_parquet(out_dir / f"momentum_{window}d.parquet")

    # Overnight gap (open / prev_close - 1)
    gap = open_ / close.shift(1) - 1
    gap.to_parquet(out_dir / "overnight_gap.parquet")

    # Trades per volume (fragmentation)
    tc = pd.DataFrame({s: d["trades_count"] for s, d in all_data.items()})
    tpv = tc / safe_vol
    tpv.to_parquet(out_dir / "trades_per_volume.parquet")

    # Upper/lower shadow ratios
    max_oc = pd.DataFrame(np.maximum(open_.values, close.values),
                          index=close.index, columns=close.columns)
    min_oc = pd.DataFrame(np.minimum(open_.values, close.values),
                          index=close.index, columns=close.columns)
    hl_range = high - low
    safe_hl = hl_range.replace(0, np.nan)
    upper_shadow = (high - max_oc) / safe_hl
    lower_shadow = (min_oc - low) / safe_hl
    upper_shadow.to_parquet(out_dir / "upper_shadow.parquet")
    lower_shadow.to_parquet(out_dir / "lower_shadow.parquet")

    # Close position in range
    cpr = (close - low) / safe_hl
    cpr.to_parquet(out_dir / "close_position_in_range.parquet")

    # Dollar volume = quote volume
    qv.to_parquet(out_dir / "dollars_traded.parquet")

    # ── Funding Rate integration ──
    funding_path = FUNDING_DIR / "funding_rates.parquet"
    if funding_path.exists():
        log(f"    Integrating funding rates...")
        fr_raw = pd.read_parquet(funding_path)
        # Resample funding rate to match the interval index
        fr_resampled = fr_raw.reindex(close.index, method="ffill")
        # Only keep columns that exist in close
        common_cols = [c for c in fr_resampled.columns if c in close.columns]
        if common_cols:
            fr_aligned = fr_resampled[common_cols]
            fr_aligned.to_parquet(out_dir / "funding_rate.parquet")
            log(f"    funding_rate: {fr_aligned.shape}")

            # Cumulative funding (rolling sum over ~1 day = 3 funding periods)
            fr_cumsum_3 = fr_aligned.rolling(3, min_periods=1).sum()
            fr_cumsum_3.to_parquet(out_dir / "funding_rate_cumsum_3.parquet")

            # Rolling average funding (8h funding × 21 periods ≈ 7 days)
            fr_avg_21 = fr_aligned.rolling(21, min_periods=5).mean()
            fr_avg_21.to_parquet(out_dir / "funding_rate_avg_7d.parquet")

            # Funding rate z-score (surprise vs recent average)
            fr_mean = fr_aligned.rolling(42, min_periods=10).mean()
            fr_std = fr_aligned.rolling(42, min_periods=10).std()
            fr_zscore = (fr_aligned - fr_mean) / fr_std.replace(0, np.nan)
            fr_zscore.to_parquet(out_dir / "funding_rate_zscore.parquet")
            log(f"    funding_rate_zscore: {fr_zscore.shape}")

    # ── Open Interest integration ──
    oi_path = OI_DIR / "open_interest_value.parquet" if OI_DIR.exists() else None
    if oi_path and oi_path.exists():
        log(f"    Integrating open interest...")
        oi_raw = pd.read_parquet(oi_path)
        oi_resampled = oi_raw.reindex(close.index, method="ffill")
        common_cols = [c for c in oi_resampled.columns if c in close.columns]
        if common_cols:
            oi_aligned = oi_resampled[common_cols]
            oi_aligned.to_parquet(out_dir / "open_interest_value.parquet")
            log(f"    open_interest_value: {oi_aligned.shape}")

            # OI change (percent)
            oi_change = oi_aligned.pct_change()
            oi_change.to_parquet(out_dir / "oi_change.parquet")

            # OI / Volume ratio (crowding signal)
            oi_vol_ratio = oi_aligned / qv.replace(0, np.nan)
            oi_vol_ratio.to_parquet(out_dir / "oi_volume_ratio.parquet")
            log(f"    oi_change + oi_volume_ratio built")

    # ── Additional derived signals ──
    # Dollar volume acceleration (rate of change of volume)
    dv_mom = qv / qv.shift(1).replace(0, np.nan)
    dv_mom.to_parquet(out_dir / "volume_momentum_1.parquet")

    dv_mom5 = qv.rolling(5, min_periods=2).mean() / qv.rolling(20, min_periods=5).mean()
    dv_mom5.to_parquet(out_dir / "volume_momentum_5_20.parquet")

    # Taker buy quote volume (raw)
    tbqv = pd.DataFrame({s: d.get("taker_buy_quote_volume", pd.Series(dtype=float))
                          for s, d in all_data.items()})
    if not tbqv.empty:
        tbqv.to_parquet(out_dir / "taker_buy_quote_volume.parquet")

    log(f"  ✅ All matrices built for {interval}")


# ═══════════════════════════════════════════════════════════════
# 5. Beta to BTC
# ═══════════════════════════════════════════════════════════════

def compute_btc_beta(interval: str = "1d", window: int = 60):
    """Compute rolling beta of each coin to BTCUSDT."""
    out_dir = MATRICES_DIR if interval == "1d" else MATRICES_DIR / interval
    returns = pd.read_parquet(out_dir / "returns.parquet")

    if "BTCUSDT" not in returns.columns:
        log("  ⚠ BTCUSDT not found, skipping beta computation")
        return

    btc_ret = returns["BTCUSDT"]
    btc_var = btc_ret.rolling(window, min_periods=window // 2).var()

    betas = {}
    for col in returns.columns:
        if col == "BTCUSDT":
            betas[col] = pd.Series(1.0, index=returns.index)
            continue
        cov = returns[col].rolling(window, min_periods=window // 2).cov(btc_ret)
        betas[col] = cov / btc_var

    beta_df = pd.DataFrame(betas)
    beta_df.to_parquet(out_dir / "beta_to_btc.parquet")
    log(f"  beta_to_btc: {beta_df.shape}")


# ═══════════════════════════════════════════════════════════════
# 6. Universe Construction (Point-in-Time)
# ═══════════════════════════════════════════════════════════════

# Minimum listing age in calendar days before a symbol enters the universe.
# Matches equities MIN_TRADING_DAYS = 252 (~1 year). Using 365 calendar days
# for crypto since markets trade 24/7 (365 cal days ≈ 252 equity trading days).
MIN_LISTING_AGE_DAYS = 365

# Rebalance period: how often to re-rank the universe (in bars).
# Equities use 20 trading days. For crypto we scale by interval:
#   1d  → every 20 bars = 20 days
#   4h  → every 120 bars = 20 days
#   12h → every 40 bars  = 20 days
REBAL_BARS = {"1d": 20, "12h": 40, "4h": 120, "5m": 5760}


def build_universes(symbols_df: pd.DataFrame, interval: str = "1d"):
    """Build point-in-time TOP100/TOP50/TOP20 universe masks.

    Rules (matching WorldQuant BRAIN / equities pipeline):
    - Ranked by ADV (20-bar rolling average quote volume)
    - Rebalanced every ~20 calendar days (scaled to bar frequency)
    - No new listings until they have >= 1 year of history
    - ADV must be > 0
    """
    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = MATRICES_DIR if interval == "1d" else MATRICES_DIR / interval

    log(f"\nBuilding point-in-time universes (interval={interval})...")
    log(f"  Min listing age: {MIN_LISTING_AGE_DAYS} days")

    # Load ADV20 and check listing dates
    adv_path = out_dir / "adv20.parquet"
    if not adv_path.exists():
        log("  ⚠ adv20 not found, skipping universe construction")
        return

    adv20 = pd.read_parquet(adv_path)
    listing_dates = symbols_df.set_index("symbol")["listing_date"]

    # For each symbol, mask out bars before listing_date + seasoning period
    seasoning_delta = pd.Timedelta(days=MIN_LISTING_AGE_DAYS)
    for sym in adv20.columns:
        if sym in listing_dates.index:
            eligible_date = listing_dates[sym] + seasoning_delta
            adv20.loc[adv20.index < eligible_date, sym] = np.nan
        else:
            # No listing info → NaN everywhere (exclude entirely)
            adv20[sym] = np.nan

    # Determine rebalance schedule
    rebal_period = REBAL_BARS.get(interval, 20)
    all_dates = adv20.index
    rebal_indices = list(range(0, len(all_dates), rebal_period))
    rebal_dates = all_dates[rebal_indices]
    log(f"  {len(rebal_dates)} rebalance dates (every {rebal_period} bars ≈ 20 days)")

    # Build universe membership per rebalance window
    for tier_name, tier_size in [("TOP100", 100), ("TOP50", 50), ("TOP20", 20)]:
        mask = pd.DataFrame(False, index=all_dates, columns=adv20.columns)

        for i, reb_date in enumerate(rebal_dates):
            # ADV20 snapshot at rebalance date
            adv_row = adv20.loc[reb_date].dropna()
            # Require ADV > 0
            adv_row = adv_row[adv_row > 0]
            # Rank and take top N
            top_n = adv_row.nlargest(min(tier_size, len(adv_row))).index.tolist()

            # Window: from this rebal date to next (exclusive)
            if i + 1 < len(rebal_dates):
                end_date = rebal_dates[i + 1]
                period = all_dates[(all_dates >= reb_date) & (all_dates < end_date)]
            else:
                period = all_dates[all_dates >= reb_date]

            mask.loc[period, top_n] = True

        suffix = "" if interval == "1d" else f"_{interval}"
        mask.to_parquet(UNIVERSE_DIR / f"BINANCE_{tier_name}{suffix}.parquet")
        avg_count = mask.sum(axis=1).mean()
        log(f"  {tier_name}: avg {avg_count:.0f} symbols per bar")


# ═══════════════════════════════════════════════════════════════
# 7. Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Ingest Binance futures data")
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--update", action="store_true", help="Update existing data only")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, just rebuild matrices")
    parser.add_argument("--daily-only", action="store_true", help="Only download daily klines")
    parser.add_argument("--intervals", nargs="+", default=None,
                        help="Specific intervals to download, e.g. --intervals 4h")
    parser.add_argument("--skip-funding", action="store_true", help="Skip funding rate download")
    parser.add_argument("--skip-oi", action="store_true", help="Skip open interest download")
    args = parser.parse_args()

    # Determine which intervals to download
    if args.intervals:
        download_intervals = args.intervals
    elif args.daily_only:
        download_intervals = ["1d"]
    else:
        download_intervals = INTERVALS

    print("=" * 60, flush=True)
    print("BINANCE FUTURES DATA INGESTION", flush=True)
    print(f"Start date: {args.start}", flush=True)
    print(f"Mode: {'update' if args.update else 'full download'}", flush=True)
    print(f"Intervals: {download_intervals}", flush=True)
    print("=" * 60, flush=True)

    # Create directories
    for d in [DATA_DIR, KLINES_DIR, MATRICES_DIR, FUNDING_DIR, UNIVERSE_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: Get exchange info
    symbols_df = get_exchange_info()
    symbols_df.to_parquet(DATA_DIR / "exchange_info.parquet", index=False)

    # Save metadata
    metadata = {
        "last_updated": datetime.utcnow().isoformat(),
        "start_date": args.start,
        "n_symbols": len(symbols_df),
        "n_active": int((symbols_df["status"] == "TRADING").sum()),
    }
    with open(DATA_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if not args.skip_download:
        # Step 2: Download klines using proper resume logic
        # download_all_klines always resumes from last bar if file exists
        download_all_klines(symbols_df, args.start, update_mode=True, intervals=download_intervals)

        # Step 3: Download funding rates
        if not args.skip_funding:
            download_funding_rates(symbols_df, args.start)

        # Step 3b: Download open interest
        if not args.skip_oi:
            for oi_interval in ["4h"]:
                download_open_interest(symbols_df, args.start, interval=oi_interval)

    # Step 4: Build matrices for ALL downloaded intervals
    build_matrices("1d")
    for iv in ["12h", "4h", "5m"]:
        if (KLINES_DIR / iv).exists():
            build_matrices(iv)

    # Step 5: BTC beta for all intervals
    compute_btc_beta("1d")
    for iv in ["4h", "5m"]:
        iv_dir = MATRICES_DIR / iv
        if iv_dir.exists() and (iv_dir / "returns.parquet").exists():
            compute_btc_beta(iv)
            log(f"  BTC beta computed for {iv}")

    # Step 6: Universes for all intervals
    build_universes(symbols_df, "1d")
    for iv in ["4h", "5m"]:
        iv_dir = MATRICES_DIR / iv
        if iv_dir.exists() and (iv_dir / "adv20.parquet").exists():
            build_universes(symbols_df, iv)
            log(f"  Universes built for {iv}")

    print("\n" + "=" * 60, flush=True)
    print("✅ INGESTION COMPLETE", flush=True)
    print(f"  Data stored in: {DATA_DIR}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
