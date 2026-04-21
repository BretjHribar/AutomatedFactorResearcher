import pandas as pd
import numpy as np

adv20 = pd.read_parquet("data/fmp_cache/matrices/adv20.parquet")
uni   = pd.read_parquet("data/fmp_cache/universes/TOP2000TOP3000.parquet")
close = pd.read_parquet("data/fmp_cache/matrices/close.parquet")
cap   = pd.read_parquet("data/fmp_cache/matrices/market_cap.parquet")

for df in [adv20, uni, close, cap]:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

last = uni.index[-1]
in_uni_tickers = uni.loc[last][uni.loc[last]].index.tolist()

adv_today   = adv20.loc[last].reindex(in_uni_tickers).dropna().sort_values(ascending=False)
close_today = close.loc[last].reindex(in_uni_tickers).dropna()
cap_today   = cap.loc[last].reindex(in_uni_tickers).dropna()

print(f"Date: {last.date()}")
print(f"Members in universe: {len(in_uni_tickers)} tickers")
print()
print("=== ADV20 distribution for TOP2000TOP3000 members ===")
for label, q in [("Min", 0.0), ("10th pct", 0.10), ("25th pct", 0.25),
                 ("Median", 0.50), ("75th pct", 0.75), ("90th pct", 0.90), ("Max", 1.0)]:
    v = adv_today.quantile(q) / 1e6
    print(f"  {label:<12}: {v:.1f}M/day")

print()
print("=== Market Cap distribution ===")
cap_m = cap_today / 1e6
for label, q in [("Min", 0.0), ("25th pct", 0.25), ("Median", 0.50),
                 ("75th pct", 0.75), ("Max", 1.0)]:
    v = cap_m.quantile(q)
    print(f"  {label:<12}: {v:,.0f}M")

print()
print("=== Avg stock price ===")
print(f"  Median: {close_today.median():.2f}")
print(f"  Mean:   {close_today.mean():.2f}")

print()
print("=== ADV rank cutoffs (all universe tickers, most recent date) ===")
all_adv = adv20.loc[last].dropna().sort_values(ascending=False)
print(f"  Total ranked tickers: {len(all_adv)}")
for rank in [500, 1000, 1500, 2000, 2500, 3000]:
    if rank <= len(all_adv):
        thresh = all_adv.iloc[rank - 1] / 1e6
        print(f"  Rank {rank:>4d} cutoff: {thresh:.1f}M/day ADV")

print()
print("  TOP2000TOP3000 = stocks between the Rank-2000 and Rank-3000 ADV cutoffs")
print("  These are SMALL-CAP stocks in the lower Russell 3000 band,")
print("  NOT in the top 2000 most liquid names.")
