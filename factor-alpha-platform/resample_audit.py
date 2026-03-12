"""Spot-Futures Basis Analysis: Can spot data provide unique signals?"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd

print('='*70)
print('SPOT-FUTURES BASIS SIGNAL ANALYSIS')
print('='*70)

# Load futures 15m (our existing data)
df15_fut = pd.read_parquet('data/binance_cache/klines/15m/BTCUSDT.parquet')
df15_fut = df15_fut.set_index('datetime').sort_index()
df15_fut = df15_fut[~df15_fut.index.duplicated(keep='last')]
for c in ['open','high','low','close','volume','quote_volume','taker_buy_volume']:
    df15_fut[c] = pd.to_numeric(df15_fut[c], errors='coerce')

# Resample futures to 1H
fut_1h = df15_fut.resample('1h').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
    'volume': 'sum', 'quote_volume': 'sum', 'taker_buy_volume': 'sum',
}).dropna()

# Load spot 1H
spot_1h = pd.read_parquet('../polymarket-crypto-candles/data/BTCUSDT_1h.parquet')
spot_1h.index = spot_1h.index.tz_localize(None)
spot_1h = spot_1h.rename(columns={'taker_buy_base': 'taker_buy_volume'})

# Align
common = fut_1h.index.intersection(spot_1h.index)
print(f'Common timestamps: {len(common)}')
print(f'Range: {common[0]} to {common[-1]}')

fut = fut_1h.loc[common]
spot = spot_1h.loc[common]

# Futures return (what we actually trade)
fut_ret = fut['close'].pct_change()

# === POTENTIAL SPOT-DERIVED SIGNALS ===

# 1. Basis (futures premium)
basis = (fut['close'] - spot['close']) / spot['close']
print(f'\nBasis stats:')
print(f'  Mean: {basis.mean()*10000:.1f} bps')
print(f'  Std:  {basis.std()*10000:.1f} bps')
print(f'  Min:  {basis.min()*10000:.1f} bps')
print(f'  Max:  {basis.max()*10000:.1f} bps')

# 2. Basis change (delta)
basis_delta = basis.diff()

# 3. Basis z-score (mean reversion of basis)
for w in [12, 24, 48, 96]:
    bz = (basis - basis.rolling(w).mean()) / basis.rolling(w).std().replace(0, np.nan)
    # Does basis z-score predict futures return?
    sig = -bz.shift(1)  # negative: when basis is high, expect reversion (futures down)
    valid = sig.dropna().index.intersection(fut_ret.dropna().index)
    d = np.sign(sig.loc[valid])
    pnl = (d * fut_ret.loc[valid])
    daily = pnl.resample('1D').sum()
    daily = daily[daily != 0]
    if len(daily) > 10 and daily.std() > 0:
        sr = daily.mean() / daily.std() * np.sqrt(365)
        print(f'  basis_zscore_{w}: Sharpe={sr:+.2f}')

# 4. Spot-futures volume divergence
fut_vol = fut['volume']
spot_vol = spot['volume']
vol_ratio = fut_vol / spot_vol.replace(0, np.nan)
for w in [12, 24, 48]:
    vrz = (vol_ratio - vol_ratio.rolling(w).mean()) / vol_ratio.rolling(w).std().replace(0, np.nan)
    sig = vrz.shift(1)  # high futures/spot volume = bullish?
    valid = sig.dropna().index.intersection(fut_ret.dropna().index)
    d = np.sign(sig.loc[valid])
    pnl = (d * fut_ret.loc[valid])
    daily = pnl.resample('1D').sum()
    daily = daily[daily != 0]
    if len(daily) > 10 and daily.std() > 0:
        sr = daily.mean() / daily.std() * np.sqrt(365)
        print(f'  fut/spot_vol_zscore_{w}: Sharpe={sr:+.2f}')

# 5. Spot taker buy imbalance vs futures taker buy imbalance
fut_tbr = fut['taker_buy_volume'] / fut['volume'].replace(0, np.nan)
spot_tbr = spot['taker_buy_volume'] / spot['volume'].replace(0, np.nan)
tbr_diff = fut_tbr - spot_tbr
for w in [12, 24, 48]:
    tbr_z = (tbr_diff - tbr_diff.rolling(w).mean()) / tbr_diff.rolling(w).std().replace(0, np.nan)
    sig = tbr_z.shift(1)
    valid = sig.dropna().index.intersection(fut_ret.dropna().index)
    d = np.sign(sig.loc[valid])
    pnl = (d * fut_ret.loc[valid])
    daily = pnl.resample('1D').sum()
    daily = daily[daily != 0]
    if len(daily) > 10 and daily.std() > 0:
        sr = daily.mean() / daily.std() * np.sqrt(365)
        print(f'  tbr_diff_zscore_{w}: Sharpe={sr:+.2f}')

# 6. Spot return leads futures (does spot lead?)
spot_ret = spot['close'].pct_change()
for lag in [1, 2, 3]:
    sig = spot_ret.shift(lag)  # spot return [lag] bars ago predicts futures now?
    valid = sig.dropna().index.intersection(fut_ret.dropna().index)
    d = np.sign(sig.loc[valid])
    pnl = (d * fut_ret.loc[valid])
    daily = pnl.resample('1D').sum()
    daily = daily[daily != 0]
    if len(daily) > 10 and daily.std() > 0:
        sr = daily.mean() / daily.std() * np.sqrt(365)
        print(f'  spot_ret_lag{lag}: Sharpe={sr:+.2f}')

# 7. Compute correlation between spot signals and existing alpha signals
print(f'\n--- UNIQUENESS: Correlation with futures-only signals ---')
# Build a few representative futures signals
fut_close = fut['close']
fut_lr = np.log(fut_close / fut_close.shift(1))
mr10 = -(fut_close - fut_close.rolling(10).mean()) / fut_close.rolling(10).std()
mom12 = fut_lr.rolling(12).sum()

# Basis zscore as candidate new signal
bz24 = -(basis - basis.rolling(24).mean()) / basis.rolling(24).std().replace(0, np.nan)
bz48 = -(basis - basis.rolling(48).mean()) / basis.rolling(48).std().replace(0, np.nan)
vr24 = (vol_ratio - vol_ratio.rolling(24).mean()) / vol_ratio.rolling(24).std().replace(0, np.nan)

print(f'  corr(basis_z24, mr10):  {bz24.corr(mr10):.3f}')
print(f'  corr(basis_z48, mr10):  {bz48.corr(mr10):.3f}')
print(f'  corr(basis_z24, mom12): {bz24.corr(mom12):.3f}')
print(f'  corr(vr24, mr10):       {vr24.corr(mr10):.3f}')
print(f'  corr(vr24, mom12):      {vr24.corr(mom12):.3f}')
print(f'  corr(basis_z24, vr24):  {bz24.corr(vr24):.3f}')

print('\nDone')
