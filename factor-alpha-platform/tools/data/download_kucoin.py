"""
Download KuCoin Futures 4h OHLCV data and build matrices.
KuCoin data starts from ~2023 for most symbols.
"""
import requests
import pandas as pd
import numpy as np
import time as time_mod
import json
from pathlib import Path
from datetime import datetime

OUT_DIR = Path("data/kucoin_cache")
MATRICES_DIR = OUT_DIR / "matrices" / "4h"
KLINES_DIR = OUT_DIR / "klines" / "4h"
for d in [OUT_DIR, MATRICES_DIR, KLINES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://api-futures.kucoin.com"
GRAN = 240  # 4h in minutes
BAR_MS = GRAN * 60 * 1000

# Step 1: Contracts
print("Fetching contracts...", flush=True)
r = requests.get(f"{BASE_URL}/api/v1/contracts/active")
contracts = r.json()["data"]
usdt_perps = [c for c in contracts if c["quoteCurrency"] == "USDT" 
              and not c["isInverse"] and c["status"] == "Open"]
print(f"  {len(usdt_perps)} USDT perpetuals", flush=True)

tick_sizes = {c["symbol"]: float(c["tickSize"]) for c in usdt_perps}
with open(OUT_DIR / "tick_sizes.json", "w") as f:
    json.dump(tick_sizes, f)

# Step 2: Download klines
START_MS = int(datetime(2023, 7, 1).timestamp() * 1000)
END_MS = int(datetime.now().timestamp() * 1000)

def download_klines(symbol):
    cache_file = KLINES_DIR / f"{symbol}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)
    
    all_candles = []
    current = START_MS
    empty_chunks = 0
    
    while current < END_MS:
        to_ms = min(current + 200 * BAR_MS, END_MS)
        try:
            r = requests.get(f"{BASE_URL}/api/v1/kline/query", params={
                "symbol": symbol, "granularity": GRAN, "from": current, "to": to_ms
            })
            data = r.json()
            
            if data.get("code") != "200000" or not data.get("data"):
                # No data for this range — advance to next chunk
                current = to_ms
                empty_chunks += 1
                if empty_chunks > 5 and not all_candles:
                    # Too many empty chunks at start, try jumping ahead more
                    pass
                continue
            
            candles = data["data"]
            all_candles.extend(candles)
            last_ts = candles[-1][0]
            if last_ts <= current:
                break
            current = last_ts + BAR_MS
            empty_chunks = 0
            time_mod.sleep(0.12)
            
        except Exception as e:
            current = to_ms
            time_mod.sleep(0.5)
    
    if not all_candles:
        return None
    
    # KuCoin Futures API returns [time_ms, open, HIGH, LOW, CLOSE, volume, turnover]
    # (verified empirically; docs claiming [t, o, c, h, l, v, tv] are wrong for futures v1).
    df = pd.DataFrame(all_candles, columns=["time", "open", "high", "low", "close", "volume", "turnover"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.set_index("time").sort_index()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[~df.index.duplicated(keep="last")]
    
    if len(df) > 50:
        df.to_parquet(cache_file)
    return df

symbols = [c["symbol"] for c in usdt_perps]
print(f"\nDownloading {len(symbols)} symbols...", flush=True)

all_data = {}
for i, sym in enumerate(symbols):
    if (i+1) % 20 == 0 or i < 5:
        print(f"  [{i+1}/{len(symbols)}] {sym}...", end=" ", flush=True)
    
    df = download_klines(sym)
    if df is not None and len(df) > 100:
        all_data[sym] = df
        if (i+1) % 20 == 0 or i < 5:
            print(f"{len(df)} bars ({df.index[0].date()} -> {df.index[-1].date()})", flush=True)
    else:
        if (i+1) % 20 == 0 or i < 5:
            bars = len(df) if df is not None else 0
            print(f"SKIP ({bars} bars)", flush=True)

print(f"\n  Total: {len(all_data)} symbols with data", flush=True)

# Step 3: Build matrices
print("\nBuilding matrices...", flush=True)

all_indices = set()
for df in all_data.values():
    all_indices.update(df.index)
common_idx = sorted(all_indices)
print(f"  Common timeline: {len(common_idx)} bars, {common_idx[0]} -> {common_idx[-1]}", flush=True)

# Base OHLCV
base_mats = {}
for field in ["open", "close", "high", "low", "volume", "turnover"]:
    mat = pd.DataFrame(index=common_idx)
    for sym, df in all_data.items():
        if field in df.columns:
            mat[sym] = df[field]
    base_mats[field] = mat
    mat.to_parquet(MATRICES_DIR / f"{field}.parquet")

close = base_mats["close"]
high = base_mats["high"]
low = base_mats["low"]
opn = base_mats["open"]
vol = base_mats["volume"]
turnover = base_mats["turnover"]

# Derived
print("  Computing derived fields...", flush=True)
ret = close.pct_change()

derived = {
    "returns": ret,
    "log_returns": np.log1p(ret.fillna(0)),
    "vwap": (high + low + close) / 3,
    "adv20": turnover.rolling(120, min_periods=60).mean(),
    "adv60": turnover.rolling(360, min_periods=180).mean(),
    "high_low_range": (high - low) / close,
    "open_close_range": (close - opn).abs() / close,
    "close_position_in_range": (close - low) / (high - low + 1e-10),
    "volume_momentum_5_20": vol.rolling(30).mean() / vol.rolling(120).mean(),
    "historical_volatility_10": ret.rolling(60).std() * np.sqrt(6*365),
    "historical_volatility_20": ret.rolling(120).std() * np.sqrt(6*365),
    "historical_volatility_60": ret.rolling(360).std() * np.sqrt(6*365),
    "momentum_5d": close / close.shift(30) - 1,
    "momentum_20d": close / close.shift(120) - 1,
    "momentum_60d": close / close.shift(360) - 1,
    "vwap_deviation": (close - (high + low + close) / 3) / ((high + low + close) / 3),
    "dollars_traded": turnover,
    "quote_volume": turnover,
    "volume_ratio_20d": vol / vol.rolling(120).mean(),
    "volume_momentum_1": vol / vol.shift(1) - 1,
}

# Parkinson vol
hl = np.log(high / low)
derived["parkinson_volatility_10"] = hl.pow(2).rolling(60).mean().pow(0.5) / (2 * np.log(2))**0.5
derived["parkinson_volatility_20"] = hl.pow(2).rolling(120).mean().pow(0.5) / (2 * np.log(2))**0.5
derived["parkinson_volatility_60"] = hl.pow(2).rolling(360).mean().pow(0.5) / (2 * np.log(2))**0.5

for name, mat in derived.items():
    mat.to_parquet(MATRICES_DIR / f"{name}.parquet")

print(f"\nDone! {len(all_data)} symbols, {len(common_idx)} bars, {6 + len(derived)} matrices", flush=True)
print(f"  Range: {common_idx[0]} -> {common_idx[-1]}", flush=True)
