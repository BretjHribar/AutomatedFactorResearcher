"""Diagnose industry vs subindustry classification and universe size."""
import json
import pandas as pd

# Load classifications
cls = json.load(open("data/fmp_cache/classifications.json"))

# Load universe
uni = pd.read_parquet("data/fmp_cache/universes/TOP2000TOP3000.parquet")

# Active tickers (recent 60 days)
recent = uni.iloc[-60:]
ever_active = recent.any(axis=0)
active_tickers = ever_active[ever_active].index.tolist()
print(f"Active tickers in TOP2000TOP3000 (recent 60d): {len(active_tickers)}")

daily_count = recent.sum(axis=1)
print(f"Daily avg count: {daily_count.mean():.0f}")
print(f"Daily min count: {daily_count.min():.0f}")
print(f"Daily max count: {daily_count.max():.0f}")

# Check industry vs subindustry for active tickers
industries = set()
subindustries = set()
both_same = 0
both_diff = 0
missing_cls = 0

for t in active_tickers:
    if t not in cls:
        missing_cls += 1
        continue
    ind = cls[t].get("industry", "")
    sub = cls[t].get("subindustry", "")
    industries.add(ind)
    subindustries.add(sub)
    if ind == sub:
        both_same += 1
    else:
        both_diff += 1

print(f"\nClassification analysis for {len(active_tickers)} active tickers:")
print(f"  Unique industry codes:    {len(industries)}")
print(f"  Unique subindustry codes: {len(subindustries)}")
print(f"  industry == subindustry:  {both_same} ({both_same/len(active_tickers)*100:.0f}%)")
print(f"  industry != subindustry:  {both_diff}")
print(f"  Missing classifications:  {missing_cls}")

# Show where they differ
print("\nSample where industry != subindustry:")
count = 0
for t in active_tickers:
    if t not in cls:
        continue
    ind = cls[t].get("industry", "?")
    sub = cls[t].get("subindustry", "?")
    sic = cls[t].get("sic_code", "?")
    if ind != sub:
        print(f"  {t:<8} ind={ind:<5} sub={sub:<6} sic4={sic}")
        count += 1
        if count >= 15:
            break
if count == 0:
    print("  NONE — all industry == subindustry!")

# Show first 10 tickers
print("\nSample (first 10 active tickers):")
for t in active_tickers[:10]:
    if t in cls:
        ind = cls[t].get("industry", "?")
        sub = cls[t].get("subindustry", "?")
        sic = cls[t].get("sic_code", "?")
        sic_desc = cls[t].get("sic_description", "?")[:30]
        same = "==" if ind == sub else "!="
        print(f"  {t:<8} ind={ind:<5} {same} sub={sub:<6}  sic={sic}  {sic_desc}")

# What did merge_sic set?
print("\nRaw classification entry for first active ticker:")
t0 = active_tickers[0]
for k, v in cls[t0].items():
    print(f"  {k}: {v}")
