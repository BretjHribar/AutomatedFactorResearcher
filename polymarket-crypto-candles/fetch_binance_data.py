"""
fetch_binance_data.py — Download historical klines from Binance public API.
Stores data as parquet files: data/{symbol}_{interval}.parquet
"""
import time, sys, os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SYMBOLS, INTERVALS, DATA_DIR, DATA_START, DATA_END

BINANCE_API = "https://api.binance.com/api/v3/klines"
BATCH_SIZE = 1000
RATE_LIMIT_SLEEP = 0.35  # seconds between API calls


def ts_to_ms(date_str: str) -> int:
    """Convert date string to milliseconds since epoch."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def interval_to_ms(interval: str) -> int:
    """Convert interval string to milliseconds."""
    mapping = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
    return mapping[interval]


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch all klines for a symbol/interval in batches."""
    all_rows = []
    current_start = start_ms
    interval_ms = interval_to_ms(interval)
    total_candles = (end_ms - start_ms) // interval_ms
    fetched = 0

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": BATCH_SIZE,
        }
        try:
            resp = requests.get(BINANCE_API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Error fetching {symbol} {interval}: {e}, retrying in 5s...")
            time.sleep(5)
            continue

        if not data:
            break

        all_rows.extend(data)
        fetched += len(data)
        pct = min(100, fetched / max(total_candles, 1) * 100)
        print(f"  {symbol} {interval}: {fetched:,} candles ({pct:.0f}%)", end="\r")

        # Move start to after the last candle
        last_open_time = data[-1][0]
        current_start = last_open_time + interval_ms

        if len(data) < BATCH_SIZE:
            break

        time.sleep(RATE_LIMIT_SLEEP)

    print(f"  {symbol} {interval}: {len(all_rows):,} candles (100%)     ")

    if not all_rows:
        return pd.DataFrame()

    # Parse kline data
    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])

    # Convert types
    for col in ["open", "high", "low", "close", "volume", "quote_volume",
                "taker_buy_base", "taker_buy_quote"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["trades"] = df["trades"].astype(int)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df.drop(columns=["ignore"], inplace=True)
    df.set_index("open_time", inplace=True)
    df.sort_index(inplace=True)

    # Remove duplicates
    df = df[~df.index.duplicated(keep="first")]

    return df


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    start_ms = ts_to_ms(DATA_START)
    end_ms = ts_to_ms(DATA_END)

    for symbol in SYMBOLS:
        for interval in INTERVALS:
            out_path = DATA_DIR / f"{symbol}_{interval}.parquet"
            if out_path.exists():
                existing = pd.read_parquet(out_path)
                print(f"  {symbol} {interval}: Already have {len(existing):,} candles, skipping")
                continue

            print(f"\nFetching {symbol} {interval}...")
            df = fetch_klines(symbol, interval, start_ms, end_ms)
            if df.empty:
                print(f"  WARNING: No data for {symbol} {interval}")
                continue

            df.to_parquet(out_path)
            print(f"  Saved {len(df):,} candles to {out_path.name}")

    # Summary
    print("\n" + "=" * 60)
    print("DATA DOWNLOAD COMPLETE")
    print("=" * 60)
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            fp = DATA_DIR / f"{symbol}_{interval}.parquet"
            if fp.exists():
                df = pd.read_parquet(fp)
                print(f"  {symbol} {interval}: {len(df):,} candles, "
                      f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    main()
