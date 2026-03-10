"""
Rolling 20-day MA factor shutoff test.
For each alpha: compute its daily cross-sectional return, take 20-day MA.
If MA <= 0, zero out the alpha until it recovers. Compare with/without.
Uses top 30 alphas by in-sample fitness.
"""
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
MA_WINDOW = 20
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
N = len(tickers)
T = len(dates)
returns_arr = matrices['returns'].reindex(columns=tickers).fillna(0).values
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

# Get top 30 alphas by in-sample fitness
conn = sqlite3.connect('data/alpha_gp_crypto.db')
cur = conn.cursor()
cur.execute("""SELECT a.expression, e.sharpe, e.fitness 
               FROM alphas a JOIN evaluations e ON a.alpha_id=e.alpha_id 
               WHERE e.sharpe >= 1.0 AND e.fitness >= 1.0
               ORDER BY e.fitness DESC LIMIT 30""")
rows = cur.fetchall()
conn.close()
expressions = [r[0] for r in rows]
print(f'Top 30 alphas by fitness (range: {rows[-1][2]:.2f} to {rows[0][2]:.2f})', flush=True)

# Evaluate all alphas and compute their rank-normalized signals
engine = FastExpressionEngine(data_fields={n: matrices[n] for n in TERMINALS if n in matrices})
alpha_signals = []  # list of (T, N) arrays — rank-normalized
alpha_names = []

for i, expr in enumerate(expressions):
    try:
        a = engine.evaluate(expr)
        if a is not None and not a.empty:
            ranked = a.reindex(index=dates, columns=tickers).rank(axis=1, pct=True) - 0.5
            ranked = ranked.fillna(0).values
            alpha_signals.append(ranked)
            alpha_names.append(expr[:60])
    except:
        pass

K = len(alpha_signals)
print(f'{K} valid alphas evaluated', flush=True)

# Compute daily factor return for each alpha:
# factor_return[t, k] = cross-sectional correlation between signal[t-1, :] and return[t, :]
# This is the "PnL" of the alpha on day t (using delay=1)
print(f'Computing daily factor returns...', flush=True)
factor_returns = np.zeros((T, K))

for k in range(K):
    sig = alpha_signals[k]  # (T, N)
    for t in range(1, T):
        s = sig[t-1]  # yesterday's signal
        r = returns_arr[t]  # today's return
        mask = uni_np[t] & (s != 0) & np.isfinite(r)
        if mask.sum() < 5:
            continue
        # Factor return = mean return of long positions minus mean return of short positions
        # Or equivalently: dot(normalized_signal, returns)
        s_m = s[mask]
        r_m = r[mask]
        # Abs-sum normalize the signal for this day
        abs_s = np.sum(np.abs(s_m))
        if abs_s > 0:
            factor_returns[t, k] = np.dot(s_m / abs_s, r_m)

# Rolling 20-day MA of factor returns
print(f'Computing rolling {MA_WINDOW}-day MA...', flush=True)
factor_ma = np.zeros((T, K))
for t in range(MA_WINDOW, T):
    factor_ma[t] = np.mean(factor_returns[t-MA_WINDOW:t], axis=0)

# Alpha is "on" when its MA > 0
alpha_on = factor_ma > 0  # (T, K) boolean

# Print some stats
train_end_idx = np.searchsorted(dates, pd.Timestamp(TRAIN_END))
for k in range(min(5, K)):
    on_pct_train = alpha_on[MA_WINDOW:train_end_idx, k].mean() * 100
    on_pct_oos = alpha_on[train_end_idx:, k].mean() * 100
    avg_ret_train = factor_returns[MA_WINDOW:train_end_idx, k].mean() * 10000
    avg_ret_oos = factor_returns[train_end_idx:, k].mean() * 10000
    print(f'  Alpha {k}: ON {on_pct_train:.0f}% train / {on_pct_oos:.0f}% OOS, '
          f'avg ret {avg_ret_train:.1f} / {avg_ret_oos:.1f} bps')

# Build combined signals: with and without shutoff
print(f'\nBuilding combined signals...', flush=True)

# Method 1: All alphas always on (baseline)
combined_always = np.zeros((T, N))
for k in range(K):
    combined_always += alpha_signals[k]
combined_always /= K

# Method 2: Shutoff when MA <= 0
combined_shutoff = np.zeros((T, N))
for t in range(T):
    n_on = 0
    for k in range(K):
        if alpha_on[t, k]:
            combined_shutoff[t] += alpha_signals[k][t]
            n_on += 1
    if n_on > 0:
        combined_shutoff[t] /= n_on

# Run sims
print(f'\nRESULTS (top {K} alphas, {MA_WINDOW}-day MA shutoff)', flush=True)
print(f'{"="*85}', flush=True)
hdr = f'{"Method":>25} {"d":>1} {"Per":>5} {"Sharpe":>8} {"PnL":>14} {"Ann":>8} {"TO":>6} {"MaxDD":>7}'
print(hdr, flush=True)
print("-" * len(hdr), flush=True)

for label, arr in [
    (f'Top{K} always-on', combined_always),
    (f'Top{K} MA{MA_WINDOW} shutoff', combined_shutoff),
]:
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
            print(f'{label:>25} {delay:>1} {pname:>5} {r.sharpe:>+8.2f} ${r.total_pnl:>13,.0f} {ann:>7.1f}% {r.turnover:>5.1%} {r.max_drawdown:>6.1%}', flush=True)
    print(flush=True)

# Also test all 209 alphas with shutoff for comparison
print(f'\n--- Now testing ALL 209 alphas with shutoff ---', flush=True)
conn2 = sqlite3.connect('data/alpha_gp_crypto.db')
cur2 = conn2.cursor()
cur2.execute("SELECT a.expression FROM alphas a JOIN evaluations e ON a.alpha_id=e.alpha_id WHERE e.sharpe >= 1.0 AND e.fitness >= 1.0")
all_exprs = [r[0] for r in cur2.fetchall()]
conn2.close()

all_signals = []
for expr in all_exprs:
    try:
        a = engine.evaluate(expr)
        if a is not None and not a.empty:
            ranked = a.reindex(index=dates, columns=tickers).rank(axis=1, pct=True) - 0.5
            all_signals.append(ranked.fillna(0).values)
    except:
        pass

K_all = len(all_signals)
print(f'{K_all} valid alphas', flush=True)

# Factor returns for all
fr_all = np.zeros((T, K_all))
for k in range(K_all):
    sig = all_signals[k]
    for t in range(1, T):
        s = sig[t-1]
        r = returns_arr[t]
        mask = uni_np[t] & (s != 0) & np.isfinite(r)
        if mask.sum() < 5:
            continue
        s_m = s[mask]
        r_m = r[mask]
        abs_s = np.sum(np.abs(s_m))
        if abs_s > 0:
            fr_all[t, k] = np.dot(s_m / abs_s, r_m)

ma_all = np.zeros((T, K_all))
for t in range(MA_WINDOW, T):
    ma_all[t] = np.mean(fr_all[t-MA_WINDOW:t], axis=0)
on_all = ma_all > 0

# All always on
comb_all_always = np.mean(np.stack(all_signals, axis=-1), axis=-1)

# All with shutoff
comb_all_shutoff = np.zeros((T, N))
for t in range(T):
    n_on = 0
    for k in range(K_all):
        if on_all[t, k]:
            comb_all_shutoff[t] += all_signals[k][t]
            n_on += 1
    if n_on > 0:
        comb_all_shutoff[t] /= n_on

print(f'\n{"Method":>25} {"d":>1} {"Per":>5} {"Sharpe":>8} {"PnL":>14} {"Ann":>8} {"TO":>6} {"MaxDD":>7}', flush=True)
print("-" * 85, flush=True)

for label, arr in [
    (f'All{K_all} always-on', comb_all_always),
    (f'All{K_all} MA{MA_WINDOW} shutoff', comb_all_shutoff),
]:
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
            print(f'{label:>25} {delay:>1} {pname:>5} {r.sharpe:>+8.2f} ${r.total_pnl:>13,.0f} {ann:>7.1f}% {r.turnover:>5.1%} {r.max_drawdown:>6.1%}', flush=True)
    print(flush=True)

print(f'\nDone in {time.time()-t0:.0f}s', flush=True)
