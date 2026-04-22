"""Quick debug: download BTC klines from KuCoin Futures."""
import requests
from datetime import datetime

BASE = "https://api-futures.kucoin.com"
symbol = "XBTUSDTM"

# Try a recent range
start_ms = int(datetime(2025, 1, 1).timestamp() * 1000)
end_ms = int(datetime(2025, 1, 10).timestamp() * 1000)

r = requests.get(f"{BASE}/api/v1/kline/query", params={
    "symbol": symbol, "granularity": 240, "from": start_ms, "to": end_ms
})
d = r.json()
print(f"Code: {d['code']}")
print(f"Candles: {len(d.get('data', []))}")
if d.get('data'):
    for c in d['data'][:3]:
        ts = datetime.fromtimestamp(c[0] / 1000)
        print(f"  {ts}: O={c[1]} C={c[2]} H={c[3]} L={c[4]} V={c[5]}")

# Now try iterating from 2020
print("\n--- Full download test ---")
start_ms = int(datetime(2020, 1, 1).timestamp() * 1000)
end_ms = int(datetime.now().timestamp() * 1000)
BAR_MS = 240 * 60 * 1000

all_candles = []
current = start_ms
chunks = 0

while current < end_ms and chunks < 100:
    to_ms = min(current + 200 * BAR_MS, end_ms)
    r = requests.get(f"{BASE}/api/v1/kline/query", params={
        "symbol": symbol, "granularity": 240, "from": current, "to": to_ms
    })
    d = r.json()
    
    if d.get("code") != "200000" or not d.get("data"):
        print(f"  Chunk {chunks}: no data at {datetime.fromtimestamp(current/1000)}")
        current = to_ms
        chunks += 1
        continue
    
    candles = d["data"]
    all_candles.extend(candles)
    last_ts = candles[-1][0]
    print(f"  Chunk {chunks}: got {len(candles)} candles, {datetime.fromtimestamp(candles[0][0]/1000)} -> {datetime.fromtimestamp(last_ts/1000)}")
    
    if last_ts <= current:
        break
    current = last_ts + BAR_MS
    chunks += 1
    
    import time
    time.sleep(0.15)

print(f"\nTotal: {len(all_candles)} candles for {symbol}")
