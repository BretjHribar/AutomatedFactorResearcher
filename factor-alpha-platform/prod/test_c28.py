"""Run C28 solo: batch vs incremental, Jan 1 - Apr 22 2026."""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '.')
import json, numpy as np, pandas as pd, time, sqlite3
from pathlib import Path
from src.operators.fastexpression import FastExpressionEngine
from src.portfolio.combiners import process_signal

# Load
mdir = Path('data/binance_cache/matrices/4h')
full = {f.stem: pd.read_parquet(f) for f in sorted(mdir.glob('*.parquet')) if f.parent.name != 'prod'}
close = full['close']; returns = full['returns']

qv = full.get('quote_volume', full.get('turnover', full['volume']))
adv20 = qv.rolling(20, min_periods=10).mean()
rank = adv20.rank(axis=1, ascending=False)
universe = rank <= 100

# Get C28 expression
conn = sqlite3.connect('data/alphas.db')
expr = conn.execute('SELECT expression FROM alphas WHERE id=28').fetchone()[0]
conn.close()
print(f'C28: {expr[:80]}...', flush=True)

# Evaluate C28 on full data
engine = FastExpressionEngine(data_fields=full)
t0 = time.time()
raw_c28 = engine.evaluate(expr)
raw_c28 = raw_c28.where(universe, np.nan)
print(f'C28 evaluated in {time.time()-t0:.1f}s', flush=True)

# Normalize
normed = process_signal(raw_c28, universe_df=universe, max_wt=0.10)

# Batch PnL (just the normalized signal × forward returns)
fwd_ret = close.pct_change().shift(-1)
batch_pnl = (normed * fwd_ret).sum(axis=1)

oos = '2026-01-01'
batch_oos = batch_pnl.loc[oos:]

# Incremental PnL (bar by bar with TAIL=1500, same normalized signal sliced)
TAIL = 1500
oos_idx = close.index[close.index >= oos]
oos_positions = [close.index.get_loc(t) for t in oos_idx if close.index.get_loc(t) < len(close)-1]

incr_pnl = []
incr_times = []
prev_positions = pd.Series(dtype=float)
GMV = 100_000
TAKER_BPS = 1.7 / 10_000

for bar_pos in oos_positions:
    bar_time = close.index[bar_pos]
    
    # Signal at this bar (same whether sliced or full — cross-sectional)
    target = normed.iloc[bar_pos]
    target = target[target.abs() > 1e-10]
    
    # PnL
    next_ret = fwd_ret.iloc[bar_pos]
    pnl = (target * next_ret.reindex(target.index, fill_value=0)).sum()
    incr_pnl.append(pnl)
    incr_times.append(bar_time)

incr_s = pd.Series(incr_pnl, index=incr_times)
batch_s = batch_oos.reindex(incr_times)

# Compare
diff = (incr_s - batch_s).abs()
corr = incr_s.corr(batch_s)

print(f'\n=== C28 SOLO: Batch vs Incremental (GROSS) ===')
print(f'Max diff: {diff.max():.2e}, Correlation: {corr:.10f}')

# Stats
daily_pnl = (batch_s * GMV).resample('D').sum()
daily_pnl = daily_pnl[daily_pnl != 0]
sr = daily_pnl.mean() / daily_pnl.std() * np.sqrt(365) if daily_pnl.std() > 0 else 0
total = batch_s.sum() * GMV

# Cost estimate
turnover = normed.diff().abs().sum(axis=1).loc[oos:].mean()
cost_per_day = turnover * GMV * TAKER_BPS * 6
daily_net = daily_pnl.mean() - cost_per_day
sr_net = daily_net / daily_pnl.std() * np.sqrt(365)

print(f'\nC28 solo OOS (Jan 1 - Apr 22, 2026):')
print(f'  Total PnL (gross):  ${total:+,.0f}')
print(f'  Daily PnL (gross):  ${daily_pnl.mean():+,.0f}')
print(f'  Daily costs:        ${cost_per_day:,.0f}')
print(f'  Daily PnL (net):    ${daily_net:+,.0f}')
print(f'  Sharpe (gross):     {sr:.2f}')
print(f'  Sharpe (net):       {sr_net:.2f}')
print(f'  Avg turnover/bar:   {turnover:.3f}')
print(f'  Bars: {len(incr_s)}, Days: {len(daily_pnl)}')

# Full IS Sharpe for comparison
full_daily = (batch_pnl * GMV).resample('D').sum()
full_daily = full_daily[full_daily != 0]
sr_full = full_daily.mean() / full_daily.std() * np.sqrt(365)
print(f'\n  Full IS Sharpe:     {sr_full:.2f} (all 2020-2026)')

# The eval reported SR_net=17.17 for C28. Let's see what that annualization is
bar_pnl_full = batch_pnl * GMV
bar_sr = bar_pnl_full.mean() / bar_pnl_full.std() * np.sqrt(6*365)
print(f'  Full IS bar-Sharpe: {bar_sr:.2f} (this is what eval reports as SR)')
