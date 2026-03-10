"""Universe expansion test: TOP50 vs TOP100."""
import sys, warnings, time
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sqlite3
from pathlib import Path
sys.path.insert(0, '.')
from src.operators.fastexpression import FastExpressionEngine
from src.simulation.vectorized_sim_polars import simulate_vectorized_polars as sim_vec

DATA_DIR = Path('data/binance_cache')
TRAIN_END = '2024-04-27'
t0 = time.time()

# Load full data
matrices = {}
for f in sorted((DATA_DIR / 'matrices').glob('*.parquet')):
    matrices[f.stem] = pd.read_parquet(f)

# Load both universes
uni50 = pd.read_parquet(DATA_DIR / 'universes/BINANCE_TOP50.parquet')
uni100 = pd.read_parquet(DATA_DIR / 'universes/BINANCE_TOP100.parquet')

close = matrices['close'].copy()
returns = matrices['returns'].copy()
for col in close.columns:
    lv = close[col].last_valid_index()
    if lv is not None and lv < close.index[-1]:
        returns.loc[returns.index > lv, col] = 0.0
matrices['returns'] = returns

print(f'Data loaded in {time.time()-t0:.0f}s')
print(f'TOP50 universe: {uni50.shape[1]} cols, TOP100: {uni100.shape[1]} cols')

# For each universe, get the tickers and evaluate
conn = sqlite3.connect('data/alpha_gp_crypto.db')
cur = conn.cursor()
cur.execute("SELECT a.expression FROM alphas a JOIN evaluations e ON a.alpha_id=e.alpha_id WHERE e.sharpe >= 1.0 AND e.fitness >= 1.0")
expressions = [r[0] for r in cur.fetchall()]
conn.close()

dates = matrices['returns'].index
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

print(f'\nTesting against both TOP50 and TOP100 universes...')
print(f'{"="*90}')

for uni_name, uni_df in [('TOP50', uni50), ('TOP100', uni100)]:
    # Get tickers for this universe
    all_tickers = sorted(uni_df.columns[uni_df.any()].tolist())
    # Filter data to these tickers
    local_matrices = {}
    for name in matrices:
        cols = [c for c in all_tickers if c in matrices[name].columns]
        if cols:
            local_matrices[name] = matrices[name][cols]
    
    tickers = local_matrices['close'].columns.tolist()
    N = len(tickers)
    
    # Evaluate alphas on this universe
    engine = FastExpressionEngine(data_fields={n: local_matrices[n] for n in TERMINALS if n in local_matrices})
    combined = None
    n_good = 0
    for expr in expressions:
        try:
            a = engine.evaluate(expr)
            if a is not None and not a.empty:
                ranked = a.reindex(index=dates, columns=tickers).rank(axis=1, pct=True) - 0.5
                if combined is None:
                    combined = ranked.copy()
                else:
                    combined = combined.add(ranked, fill_value=0)
                n_good += 1
        except:
            pass
    combined = combined / n_good
    
    print(f'\n{uni_name}: {N} tickers, {n_good} valid alphas')
    hdr = f'  {"Decay":>5} {"Dly":>3} {"Per":>5} {"Sharpe":>8} {"PnL":>14} {"Ann%":>8} {"TO":>6} {"MaxDD":>7}'
    print(hdr)
    print(f'  {"-"*70}')
    
    for decay_val in [0, 5, 10]:
        for delay in [1]:
            for pname, s, e in [('TRAIN', dates[0], TRAIN_END), ('OOS', TRAIN_END, dates[-1])]:
                r = sim_vec(
                    alpha_df=combined.loc[s:e],
                    returns_df=local_matrices['returns'].reindex(columns=tickers).loc[s:e],
                    close_df=local_matrices['close'].reindex(columns=tickers).loc[s:e],
                    open_df=local_matrices['open'].reindex(columns=tickers).loc[s:e],
                    universe_df=uni_df.loc[s:e],
                    booksize=2e6, max_stock_weight=0.05,
                    decay=decay_val, delay=delay, neutralization='market', fees_bps=5.0,
                )
                ndays = len(local_matrices['returns'].loc[s:e])
                ann = r.total_pnl / 1e6 / max(1, ndays / 365) * 100
                print(f'  {decay_val:>5} {delay:>3} {pname:>5} {r.sharpe:>+8.2f} ${r.total_pnl:>13,.0f} {ann:>7.1f}% {r.turnover:>5.1%} {r.max_drawdown:>6.1%}')

print(f'\nDone in {time.time()-t0:.0f}s')
