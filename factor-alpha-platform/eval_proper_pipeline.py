"""Compare: per-alpha normalize+sum vs rank avg."""
import sys, warnings, time
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sqlite3
from pathlib import Path
sys.path.insert(0, '.')
from src.operators.fastexpression import FastExpressionEngine
from src.simulation.vectorized_sim_polars import simulate_vectorized_polars as sim_vec

DATA_DIR = Path('data/binance_cache')
TRAIN_END = '2024-04-27'
MAX_WT = 0.05
t0 = time.time()

# Load data
matrices = {}
for f in sorted((DATA_DIR / 'matrices').glob('*.parquet')):
    matrices[f.stem] = pd.read_parquet(f)
universe_df = pd.read_parquet(DATA_DIR / 'universes/BINANCE_TOP50.parquet')
close = matrices['close'].copy()
returns = matrices['returns'].copy()
for col in close.columns:
    lv = close[col].last_valid_index()
    if lv is not None and lv < close.index[-1]:
        returns.loc[returns.index > lv, col] = 0.0
matrices['returns'] = returns
all_tickers = sorted(universe_df.columns[universe_df.any()].tolist())
for name in list(matrices.keys()):
    cols = [c for c in all_tickers if c in matrices[name].columns]
    if cols:
        matrices[name] = matrices[name][cols]
    else:
        del matrices[name]
tickers = matrices['close'].columns.tolist()
dates = matrices['returns'].index
uni_np = universe_df.reindex(index=dates, columns=tickers).fillna(False).values.astype(bool)

TERMINALS = [t for t in [
    'close','open','high','low','volume','returns','log_returns',
    'taker_buy_ratio','taker_buy_volume','vwap','vwap_deviation',
    'high_low_range','open_close_range','adv20','adv60',
    'volume_ratio_20d','historical_volatility_20','historical_volatility_60',
    'momentum_5d','momentum_20d','momentum_60d','beta_to_btc',
    'overnight_gap','upper_shadow','lower_shadow',
    'close_position_in_range','trades_count','trades_per_volume',
    'parkinson_volatility_20','quote_volume',
] if t in matrices]

print(f'Data loaded in {time.time()-t0:.0f}s', flush=True)

# Load alphas
conn = sqlite3.connect('data/alpha_gp_crypto.db')
cur = conn.cursor()
cur.execute("SELECT a.expression FROM alphas a JOIN evaluations e ON a.alpha_id=e.alpha_id WHERE e.sharpe >= 1.0 AND e.fitness >= 1.0")
expressions = [r[0] for r in cur.fetchall()]
conn.close()
print(f'{len(expressions)} alpha expressions loaded', flush=True)

# Evaluate and combine
engine = FastExpressionEngine(data_fields={n: matrices[n] for n in TERMINALS if n in matrices})
combined_proper = np.zeros((len(dates), len(tickers)))
combined_rank = np.zeros((len(dates), len(tickers)))
n_good = 0

for i, expr in enumerate(expressions):
    try:
        a = engine.evaluate(expr)
        if a is not None and not a.empty:
            arr = a.reindex(index=dates, columns=tickers).values.astype(np.float64).copy()
            arr[np.isinf(arr)] = 0.0
            arr[~uni_np[:len(arr)]] = np.nan

            # === PROPER: per-alpha neutralize -> abs-sum normalize -> clip ===
            rm = np.nanmean(arr, axis=1, keepdims=True)
            arr -= rm
            ab = np.nansum(np.abs(arr), axis=1, keepdims=True)
            ab[ab == 0] = np.nan
            arr = arr / ab
            arr = np.clip(arr, -MAX_WT, MAX_WT)
            arr = np.nan_to_num(arr, nan=0.0)
            combined_proper += arr

            # === RANK: rank-normalize then sum (old way) ===
            ranked = a.reindex(index=dates, columns=tickers).rank(axis=1, pct=True).values - 0.5
            combined_rank += np.nan_to_num(ranked, nan=0.0)

            n_good += 1
            if (i + 1) % 50 == 0:
                print(f'  {i+1}/{len(expressions)} processed...', flush=True)
    except:
        pass

combined_rank /= max(n_good, 1)
print(f'{n_good} alphas processed in {time.time()-t0:.0f}s', flush=True)
print()

# Run sims
hdr = f'{"Method":>20} {"d":>1} {"Per":>5} {"Sharpe":>8} {"PnL":>14} {"Ann":>8} {"TO":>6} {"MaxDD":>7}'
print(hdr)
print("-" * len(hdr))

for label, arr in [('Proper(norm+sum)', combined_proper), ('Rank avg', combined_rank)]:
    sig_df = pd.DataFrame(arr, index=dates, columns=tickers)
    for delay in [0, 1]:
        for pname, s, e in [('TRAIN', dates[0], TRAIN_END), ('OOS', TRAIN_END, dates[-1])]:
            r = sim_vec(
                alpha_df=sig_df.loc[s:e],
                returns_df=matrices['returns'].reindex(columns=tickers).loc[s:e],
                close_df=matrices['close'].reindex(columns=tickers).loc[s:e],
                open_df=matrices['open'].reindex(columns=tickers).loc[s:e],
                universe_df=universe_df.loc[s:e],
                booksize=2e6, max_stock_weight=0.05,
                decay=0, delay=delay, neutralization='market', fees_bps=5.0,
            )
            ndays = len(matrices['returns'].loc[s:e])
            ann = r.total_pnl / 1e6 / max(1, ndays / 365) * 100
            print(f'{label:>20} {delay:>1} {pname:>5} {r.sharpe:>+8.2f} ${r.total_pnl:>13,.0f} {ann:>7.1f}% {r.turnover:>5.1%} {r.max_drawdown:>6.1%}')
    print()

print(f'Done in {time.time()-t0:.0f}s')
