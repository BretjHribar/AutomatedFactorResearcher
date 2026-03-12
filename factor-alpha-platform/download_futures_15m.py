"""Download 15m klines from Binance Futures (fapi) for our 5 symbols."""
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

BINANCE_FAPI = "https://fapi.binance.com/fapi/v1/klines"
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'DOGEUSDT']
INTERVAL = "15m"
INTERVAL_MS = 900_000  # 15 min in ms
MAX_CANDLES = 1500
OUT_DIR = Path("data/binance_futures_15m")
START_DATE = "2023-01-01"

def download_klines(symbol, start_ms):
    all_rows = []
    current = start_ms
    end_ms = int(datetime.utcnow().timestamp() * 1000)
    
    while current < end_ms:
        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "startTime": current,
            "limit": MAX_CANDLES,
        }
        try:
            resp = requests.get(BINANCE_FAPI, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Error: {e}, retrying in 5s...")
            time.sleep(5)
            continue
        
        if not data:
            break
        
        all_rows.extend(data)
        last_time = data[-1][0]
        if last_time <= current:
            break
        current = last_time + INTERVAL_MS
        
        if len(all_rows) % 10000 < MAX_CANDLES:
            print(f"  {symbol}: {len(all_rows):,} candles...", flush=True)
        
        time.sleep(0.12)  # rate limit
    
    if not all_rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades_count",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore",
    ])
    
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume",
                 "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trades_count"] = pd.to_numeric(df["trades_count"], errors="coerce").astype(int)
    
    df = df[["datetime", "open", "high", "low", "close", "volume",
             "quote_volume", "trades_count", "taker_buy_volume",
             "taker_buy_quote_volume"]].copy()
    df = df.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)
    return df

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start_ms = int(pd.Timestamp(START_DATE).timestamp() * 1000)
    
    print("=" * 60)
    print(f"BINANCE FUTURES 15m DOWNLOAD (fapi.binance.com)")
    print(f"Symbols: {SYMBOLS}")
    print(f"Start: {START_DATE}")
    print("=" * 60)
    
    for sym in SYMBOLS:
        fpath = OUT_DIR / f"{sym}.parquet"
        print(f"\n{sym}:", flush=True)
        
        t0 = time.time()
        df = download_klines(sym, start_ms)
        elapsed = time.time() - t0
        
        if not df.empty:
            df.to_parquet(fpath, index=False)
            print(f"  Saved {len(df):,} candles to {fpath}")
            print(f"  Range: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
            print(f"  Time: {elapsed:.0f}s")
        else:
            print(f"  WARNING: No data!")
    
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
