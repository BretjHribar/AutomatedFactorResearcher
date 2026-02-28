import pandas as pd, os

# Check TOP3000 universe
uni3000 = pd.read_parquet('data/fmp_cache/universes/TOP3000.parquet')
print(f'TOP3000 universe: {uni3000.shape[1]} tickers x {uni3000.shape[0]} days')
daily_count = uni3000.sum(axis=1)
print(f'  Avg per day: {daily_count.mean():.0f}')
print(f'  Max per day: {daily_count.max():.0f}')
cov = uni3000.sum(axis=0) / len(uni3000)
for t in [0.0, 0.1, 0.3]:
    print(f'  >{t:.0%} coverage: {(cov > t).sum()} tickers')

# Check TOP1000
uni1000 = pd.read_parquet('data/fmp_cache/universes/TOP1000.parquet')
daily_1k = uni1000.sum(axis=1)
print(f'\nTOP1000: avg={daily_1k.mean():.0f}/day, max={daily_1k.max():.0f}/day')

# Screener
if os.path.exists('data/fmp_cache/universe.parquet'):
    scr = pd.read_parquet('data/fmp_cache/universe.parquet')
    print(f'\nScreener: {len(scr)} stocks')
    for col in ['exchange']:
        print(f'  {scr[col].value_counts().to_dict()}')
    mcap = scr['marketCap']
    print(f'  Market cap: ${mcap.min()/1e6:.0f}M - ${mcap.max()/1e9:.0f}B')
    print(f'  >$100M: {(mcap > 1e8).sum()}')
    print(f'  >$500M: {(mcap > 5e8).sum()}')
    print(f'  >$1B: {(mcap > 1e9).sum()}')

# The ROOT CAUSE: how many tickers did we download prices for?
close = pd.read_parquet('data/fmp_cache/matrices/close.parquet')
print(f'\nClose matrix: {close.shape[1]} tickers total')
has_data = close.notna().any(axis=0).sum()
print(f'  With any close data: {has_data}')

tickers_per_day = close.notna().sum(axis=1)
print(f'  Avg tickers/day with close data: {tickers_per_day.mean():.0f}')
print(f'  Recent (last 252 days): {tickers_per_day.iloc[-252:].mean():.0f}')
print(f'  2016: {tickers_per_day.iloc[:252].mean():.0f}')
print(f'  2020: {tickers_per_day.iloc[1000:1252].mean():.0f}')
