"""
CAUSAL alpha evaluation — correct version.
Uses signal[t-1] * return[t] to avoid concurrent lookahead bias.
Compares to the eval script's concurrent numbers.
"""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, json, time, sqlite3
from pathlib import Path
from src.operators.fastexpression import FastExpressionEngine

# Load
mdir = Path('data/binance_cache/matrices/4h')
full = {f.stem: pd.read_parquet(f) for f in sorted(mdir.glob('*.parquet')) if f.parent.name != 'prod'}
close = full['close']; returns = full['returns']

with open('data/binance_tick_sizes.json') as f:
    tick_sizes_raw = json.load(f)

# Tick cost matrix
tick_bps_matrix = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
for sym in close.columns:
    tick = tick_sizes_raw.get(sym, None)
    if tick and tick > 0:
        tick_bps_matrix[sym] = tick / close[sym] * 10000
    else:
        tick_bps_matrix[sym] = 2.8

# Universe
qv = full.get('quote_volume', full['volume'])
adv20 = qv.rolling(20, min_periods=10).mean()
rank = adv20.rank(axis=1, ascending=False)
top100 = rank <= 100

# Load all alphas
conn = sqlite3.connect('data/alphas.db')
crypto_alphas = conn.execute('SELECT id, expression FROM alphas WHERE archived=0').fetchall()
conn.close()

conn2 = sqlite3.connect('data/ib_alphas.db')
ib12 = conn2.execute('SELECT expression FROM alphas WHERE id=12').fetchone()
conn2.close()

all_alphas = [(f'C{aid}', expr) for aid, expr in crypto_alphas]
all_alphas.append(('IB12', ib12[0]))

engine = FastExpressionEngine(data_fields=full)
bpy = 6 * 365
TAKER_BPS = 1.7

print(f'Loaded {len(all_alphas)} alphas, {close.shape[0]} bars x {close.shape[1]} tickers\n')
print(f'{"Alpha":<8} {"Concurrent":>12} {"Causal":>12} {"Causal_Net":>12} {"Return%":>10} '
      f'{"TO":>8} {"DD%":>8} {"Inflate":>8} {"Time":>6}')
print('-' * 100)

results = []

for alpha_name, expr in sorted(all_alphas, key=lambda x: x[0]):
    t0 = time.time()
    try:
        raw = engine.evaluate(expr)
        if raw is None or raw.empty:
            continue

        sig = raw.where(top100, np.nan)
        sig = sig.sub(sig.mean(axis=1), axis=0)
        sig_abs = sig.abs().sum(axis=1).replace(0, np.nan)
        sig_norm = sig.div(sig_abs, axis=0)

        # CONCURRENT (what eval does — WRONG)
        pnl_conc = (sig_norm * returns).sum(axis=1).dropna()
        sr_conc = pnl_conc.mean() / pnl_conc.std() * np.sqrt(bpy) if pnl_conc.std() > 0 else 0

        # CAUSAL (correct: signal[t-1] predicts return[t])
        pnl_causal_g = (sig_norm.shift(1) * returns).sum(axis=1).dropna()

        # Turnover and costs on the LAGGED signal
        lagged_norm = sig_norm.shift(1)
        turnover = lagged_norm.diff().abs()
        cost_per_bar = turnover * (TAKER_BPS + tick_bps_matrix.reindex(
            index=turnover.index, columns=turnover.columns, method='ffill').fillna(2.8)) / 10000
        total_cost = cost_per_bar.sum(axis=1).reindex(pnl_causal_g.index, fill_value=0)

        pnl_causal_n = pnl_causal_g - total_cost
        pnl_causal_n = pnl_causal_n.dropna()

        sr_causal_g = pnl_causal_g.mean() / pnl_causal_g.std() * np.sqrt(bpy) if pnl_causal_g.std() > 0 else 0
        sr_causal_n = pnl_causal_n.mean() / pnl_causal_n.std() * np.sqrt(bpy) if pnl_causal_n.std() > 0 else 0
        ar_n = pnl_causal_n.mean() * bpy * 100
        mean_to = turnover.sum(axis=1).mean()
        cum = (1 + pnl_causal_n).cumprod()
        dd = (cum / cum.cummax() - 1).min() * 100

        ratio = sr_conc / sr_causal_g if abs(sr_causal_g) > 0.1 else float('inf')
        elapsed = time.time() - t0

        print(f'{alpha_name:<8} {sr_conc:>12.2f} {sr_causal_g:>12.2f} {sr_causal_n:>12.2f} '
              f'{ar_n:>10.1f} {mean_to:>8.3f} {dd:>8.1f} {ratio:>8.1f}x {elapsed:>6.1f}s')

        results.append({
            'alpha': alpha_name, 'sr_concurrent': sr_conc,
            'sr_causal_gross': sr_causal_g, 'sr_causal_net': sr_causal_n,
            'ann_return_pct': ar_n, 'turnover': mean_to, 'max_dd_pct': dd,
            'inflation': ratio, 'expr': expr
        })

    except Exception as e:
        elapsed = time.time() - t0
        print(f'{alpha_name:<8} ERROR: {str(e)[:60]} ({elapsed:.1f}s)')

# Summary
print(f'\n{"="*100}')
print('SUMMARY — TRUE CAUSAL SHARPE RATIOS (annualized, bar-level)')
print(f'{"="*100}\n')

df = pd.DataFrame(results).sort_values('sr_causal_net', ascending=False)

viable = df[df['sr_causal_net'] > 1.0]
print(f'Alphas with causal SR_net > 1.0: {len(viable)}/{len(df)}')
print(f'Alphas with causal SR_net > 0.0: {(df["sr_causal_net"] > 0).sum()}/{len(df)}')
print(f'Alphas with NEGATIVE causal SR:  {(df["sr_causal_net"] < 0).sum()}/{len(df)}\n')

# Flag alphas that should be archived
for _, row in df.iterrows():
    status = 'KEEP' if row['sr_causal_net'] > 1.0 else 'WEAK' if row['sr_causal_net'] > 0 else 'ARCHIVE'
    print(f"  {row['alpha']:<8} SR_net={row['sr_causal_net']:>6.2f}  Ret={row['ann_return_pct']:>7.1f}%  "
          f"TO={row['turnover']:.3f}  DD={row['max_dd_pct']:>6.1f}%  → {status}")

# Save results
out = Path('data/crypto_results/causal_alpha_eval.csv')
df.to_csv(out, index=False)
print(f'\nResults saved to {out}')
