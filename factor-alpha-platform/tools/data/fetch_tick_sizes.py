"""Fetch Binance Futures tick sizes and save."""
import requests, json
import pandas as pd
import numpy as np

r = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo")
data = r.json()
ticks = {}
for s in data["symbols"]:
    if s["contractType"] == "PERPETUAL" and s["quoteAsset"] == "USDT":
        for f in s["filters"]:
            if f["filterType"] == "PRICE_FILTER":
                ticks[s["symbol"]] = float(f["tickSize"])
                break
print(f"Got {len(ticks)} perpetual USDT symbols")

with open("data/binance_tick_sizes.json", "w") as f:
    json.dump(ticks, f)

# Examples
for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT", "1000PEPEUSDT"]:
    print(f"  {sym}: tick=${ticks.get(sym, 'N/A')}")

# Tick size in bps relative to last close
c = pd.read_parquet("data/binance_cache/matrices/4h/close.parquet")
last = c.iloc[-1].dropna()
tick_bps = []
for sym in last.index:
    tick = ticks.get(sym, None)
    if tick and last[sym] > 0:
        tick_bps.append(tick / last[sym] * 10000)

print(f"\nTick size in bps (relative to price):")
print(f"  Mean:   {np.mean(tick_bps):.1f} bps")
print(f"  Median: {np.median(tick_bps):.1f} bps")
print(f"  P25:    {np.percentile(tick_bps, 25):.1f} bps")
print(f"  P75:    {np.percentile(tick_bps, 75):.1f} bps")
print(f"  P95:    {np.percentile(tick_bps, 95):.1f} bps")
