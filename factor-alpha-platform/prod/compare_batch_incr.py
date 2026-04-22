"""Compare batch combiner vs incremental combiner to find Sharpe discrepancy."""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '.')
import json, numpy as np, pandas as pd, time
from pathlib import Path
from src.operators.fastexpression import FastExpressionEngine
from src.portfolio.combiners import combiner_risk_parity, process_signal

# Load full matrices
mdir = Path('data/binance_cache/matrices/4h')
full = {f.stem: pd.read_parquet(f) for f in sorted(mdir.glob('*.parquet')) if f.parent.name != 'prod'}
close = full['close']; returns = full['returns']

with open('prod/config/binance.json') as f:
    cfg = json.load(f)

qv = full.get('quote_volume', full.get('turnover', full['volume']))
adv20 = qv.rolling(20, min_periods=10).mean()
rank = adv20.rank(axis=1, ascending=False)
universe = rank <= 100

engine = FastExpressionEngine(data_fields=full)
raw = {}
for a in cfg['alphas']:
    try:
        sig = engine.evaluate(a['expression'])
        sig = sig.where(universe, np.nan)
        raw[a['id']] = sig
    except:
        pass
print(f'{len(raw)} alphas evaluated', flush=True)

# === BATCH: Run combiner on FULL data ===
t0 = time.time()
combined_batch = combiner_risk_parity(raw, full, universe, returns, lookback=504, max_wt=0.10)
print(f'Batch combiner: {time.time()-t0:.1f}s', flush=True)

# === Compute batch PnL for Jan 1+ ===
oos = '2026-01-01'
fwd_ret = close.pct_change().shift(-1)
batch_pnl = (combined_batch * fwd_ret).sum(axis=1)
batch_oos = batch_pnl.loc[oos:]

# === INCREMENTAL: same logic as test_incremental.py ===
TAIL = 1500
normed = {aid: process_signal(r, universe_df=universe, max_wt=0.10) for aid, r in raw.items()}
aid_list = list(raw.keys())

oos_idx = close.index[close.index >= oos]
oos_positions = [close.index.get_loc(t) for t in oos_idx if close.index.get_loc(t) < len(close)-1]

incr_pnl = []
incr_times = []
t0 = time.time()
for bar_pos in oos_positions:
    bar_time = close.index[bar_pos]
    start = max(0, bar_pos - TAIL + 1)
    
    sliced_n = {aid: normed[aid].iloc[start:bar_pos+1] for aid in aid_list}
    sliced_r = returns.iloc[start:bar_pos+1]
    
    fr = {}
    for aid in aid_list:
        lag = sliced_n[aid].shift(1)
        ab = lag.abs().sum(axis=1).replace(0, np.nan)
        n = lag.div(ab, axis=0)
        fr[aid] = (n * sliced_r).sum(axis=1)
    fr_df = pd.DataFrame(fr)
    
    rvol = fr_df.rolling(504, min_periods=60).std()
    rer = fr_df.rolling(504, min_periods=60).mean()
    iv = (1.0 / rvol.replace(0, np.nan)).fillna(0)
    iv = iv.where(rer > 0, 0)
    ws = iv.sum(axis=1).replace(0, np.nan)
    wn = iv.div(ws, axis=0).fillna(0)
    
    comb = pd.Series(0.0, index=close.columns)
    for aid in aid_list:
        w = wn[aid].iloc[-1] if aid in wn.columns else 0.0
        comb = comb + sliced_n[aid].iloc[-1] * w
    
    pnl = (comb * fwd_ret.iloc[bar_pos]).sum()
    incr_pnl.append(pnl)
    incr_times.append(bar_time)

print(f'Incremental combiner: {time.time()-t0:.1f}s', flush=True)

incr_s = pd.Series(incr_pnl, index=incr_times)
batch_s = batch_oos.reindex(incr_times)

# Compare
print(f'\n=== COMPARISON: Batch vs Incremental (GROSS, no costs) ===')
print(f'Bars compared: {len(incr_s)}')
diff = (incr_s - batch_s).abs()
print(f'Max PnL diff:     {diff.max():.6e}')
print(f'Mean PnL diff:    {diff.mean():.6e}')
corr = incr_s.corr(batch_s)
print(f'Correlation:      {corr:.10f}')

# Show first 5 bars side by side
print(f'\nFirst 10 bars:')
print(f'{"Time":<22} {"Batch":>12} {"Incr":>12} {"Diff":>12}')
for t in incr_times[:10]:
    b = batch_s.get(t, np.nan)
    ic = incr_s.get(t, np.nan)
    print(f'{str(t):<22} {b:>12.6f} {ic:>12.6f} {abs(b-ic):>12.2e}')

GMV = 100000

# Sharpe comparison
daily_batch = (batch_s * GMV).resample('D').sum()
daily_batch = daily_batch[daily_batch != 0]
sr_batch = daily_batch.mean() / daily_batch.std() * np.sqrt(365)

daily_incr = (incr_s * GMV).resample('D').sum()
daily_incr = daily_incr[daily_incr != 0]
sr_incr = daily_incr.mean() / daily_incr.std() * np.sqrt(365)

print(f'\nBatch Sharpe (daily, GROSS):  {sr_batch:.2f}')
print(f'Incr  Sharpe (daily, GROSS):  {sr_incr:.2f}')
pnl_batch = batch_s.sum() * GMV
pnl_incr = incr_s.sum() * GMV
print(f'Batch total PnL (gross):  ${pnl_batch:+,.0f}')
print(f'Incr  total PnL (gross):  ${pnl_incr:+,.0f}')

# Check eval's test split 
n = len(close)
test_start_idx = int(n * 0.8)
test_start = close.index[test_start_idx]
print(f'\nEval test split starts at bar {test_start_idx} = {test_start}')
print(f'Eval test covers {n - test_start_idx} bars')
print(f'Incr test covers {len(incr_s)} bars (from {oos})')

# Compute batch Sharpe for the eval test split period
test_pnl = (batch_pnl.iloc[test_start_idx:] * GMV).resample('D').sum()
test_pnl = test_pnl[test_pnl != 0]
sr_test = test_pnl.mean() / test_pnl.std() * np.sqrt(365) if test_pnl.std() > 0 else 0
print(f'Batch Sharpe on eval test period: {sr_test:.2f}')

print('\n\nIf batch==incr, the incremental pipeline is correct.')
print('If batch SR >> incr SR, there is a bug in the incremental combiner.')
print('If batch SR for Jan-Apr is ~4, the IS SR of 15.7 was on a different period.')
