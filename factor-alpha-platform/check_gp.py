import sqlite3, time, sys, os
sys.path.insert(0, '.')

conn = sqlite3.connect('data/alpha_gp_crypto_4h.db')
cur = conn.cursor()
cur.execute('SELECT run_id, started_at, status FROM runs')
for r in cur.fetchall():
    print(f'Run {r[0]}: started={r[1]} status={r[2]}')
cur.execute('SELECT COUNT(*) FROM alphas')
total = cur.fetchone()[0]
print(f'\nTotal alphas: {total}')
if total > 0:
    cur.execute('''SELECT a.alpha_id, a.expression, e.sharpe, e.fitness, e.turnover
                   FROM alphas a JOIN evaluations e ON a.alpha_id = e.alpha_id
                   ORDER BY e.sharpe DESC''')
    for aid, expr, sharpe, fitness, to in cur.fetchall():
        print(f'  #{aid} Sharpe={sharpe:+.3f} Fitness={fitness:.3f} TO={to:.1%} | {expr}')
conn.close()

# Quick sanity check: what Sharpe does a simple alpha get on 4h?
import pandas as pd, numpy as np
from src.simulation.vectorized_sim_polars import simulate_vectorized_polars as sim_vec

mdir = 'data/binance_cache/matrices/4h'
close = pd.read_parquet(f'{mdir}/close.parquet')
returns = pd.read_parquet(f'{mdir}/returns.parquet')
universe = pd.read_parquet('data/binance_cache/universes/BINANCE_TOP50_4h.parquet')
mom5 = pd.read_parquet(f'{mdir}/momentum_5d.parquet')

# Train period only
train_end = pd.Timestamp("2024-04-27")
alpha = -mom5.loc[:train_end]
ret = returns.loc[:train_end]
cl = close.loc[:train_end]
uni = universe.loc[:train_end]

r = sim_vec(alpha_df=alpha, returns_df=ret, close_df=cl, universe_df=uni,
            booksize=2000000, max_stock_weight=0.05, decay=0, delay=0,
            neutralization='market', fees_bps=0.0)
print(f'\nSanity check (train only, 0 fees):')
print(f'  -momentum_5d: Sharpe={r.sharpe:+.3f} TO={r.turnover:.1%}')

tbr = pd.read_parquet(f'{mdir}/taker_buy_ratio.parquet').loc[:train_end]
r2 = sim_vec(alpha_df=tbr, returns_df=ret, close_df=cl, universe_df=uni,
             booksize=2000000, max_stock_weight=0.05, decay=0, delay=0,
             neutralization='market', fees_bps=0.0)
print(f'  taker_buy_ratio: Sharpe={r2.sharpe:+.3f} TO={r2.turnover:.1%}')

fr = pd.read_parquet(f'{mdir}/funding_rate.parquet').loc[:train_end]
r3 = sim_vec(alpha_df=-fr, returns_df=ret, close_df=cl, universe_df=uni,
             booksize=2000000, max_stock_weight=0.05, decay=0, delay=0,
             neutralization='market', fees_bps=0.0)
print(f'  -funding_rate: Sharpe={r3.sharpe:+.3f} TO={r3.turnover:.1%}')
