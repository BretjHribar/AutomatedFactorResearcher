"""Decay sweep + Rolling IC weighting test."""
import sys, warnings, time
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sqlite3
from pathlib import Path
from scipy.stats import spearmanr
sys.path.insert(0, '.')
from src.operators.fastexpression import FastExpressionEngine
from src.simulation.vectorized_sim_polars import simulate_vectorized_polars as sim_vec

DATA_DIR = Path('data/binance_cache')
TRAIN_END = '2024-04-27'
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
T_total = len(dates)
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

print(f'Data loaded in {time.time()-t0:.0f}s')

# Load 209 alphas
conn = sqlite3.connect('data/alpha_gp_crypto.db')
cur = conn.cursor()
cur.execute("SELECT a.expression FROM alphas a JOIN evaluations e ON a.alpha_id=e.alpha_id WHERE e.sharpe >= 1.0 AND e.fitness >= 1.0")
expressions = [r[0] for r in cur.fetchall()]
conn.close()

# Evaluate all alphas
engine = FastExpressionEngine(data_fields={n: matrices[n] for n in TERMINALS if n in matrices})
alpha_signals = []  # list of (T, N) rank-normalized arrays
for i, expr in enumerate(expressions):
    try:
        a = engine.evaluate(expr)
        if a is not None and not a.empty:
            ranked = a.reindex(index=dates, columns=tickers).rank(axis=1, pct=True) - 0.5
            alpha_signals.append(ranked.fillna(0).values)
    except:
        pass

K = len(alpha_signals)
print(f'{K} alphas evaluated in {time.time()-t0:.0f}s')

# ═══════════════════════════════════════════════════════════════
# 1. EQUAL-WEIGHT AVERAGE (baseline) + DECAY SWEEP
# ═══════════════════════════════════════════════════════════════
combined_eqwt = np.mean(np.stack(alpha_signals, axis=-1), axis=-1)  # (T, N)
eqwt_df = pd.DataFrame(combined_eqwt, index=dates, columns=tickers)

print(f'\n{"="*90}')
print(f'TEST 1: DECAY SWEEP (209 alphas, rank avg, delay=1, 5 bps)')
print(f'{"="*90}')
hdr = f'  {"Decay":>5} {"Per":>5} {"Sharpe":>8} {"PnL":>14} {"Ann%":>8} {"TO":>6} {"MaxDD":>7}'
print(hdr)
print(f'  {"-"*65}')

for decay_val in [0, 3, 5, 10, 15, 20]:
    for pname, s, e in [('TRAIN', dates[0], TRAIN_END), ('OOS', TRAIN_END, dates[-1])]:
        r = sim_vec(
            alpha_df=eqwt_df.loc[s:e],
            returns_df=matrices['returns'].reindex(columns=tickers).loc[s:e],
            close_df=matrices['close'].reindex(columns=tickers).loc[s:e],
            open_df=matrices['open'].reindex(columns=tickers).loc[s:e],
            universe_df=universe_df.loc[s:e],
            booksize=2e6, max_stock_weight=0.05,
            decay=decay_val, delay=1, neutralization='market', fees_bps=5.0,
        )
        ndays = len(matrices['returns'].loc[s:e])
        ann = r.total_pnl / 1e6 / max(1, ndays / 365) * 100
        print(f'  {decay_val:>5} {pname:>5} {r.sharpe:>+8.2f} ${r.total_pnl:>13,.0f} {ann:>7.1f}% {r.turnover:>5.1%} {r.max_drawdown:>6.1%}')

# ═══════════════════════════════════════════════════════════════
# 2. ROLLING IC WEIGHTING
# ═══════════════════════════════════════════════════════════════
print(f'\n{"="*90}')
print(f'TEST 2: ROLLING IC WEIGHTING (209 alphas, delay=1, 5 bps)')
print(f'{"="*90}')

# For each alpha, compute rolling 60-day Spearman IC
# IC[t,k] = spearman(signal[t-1,:], return[t,:]) averaged over last 60 days
IC_WINDOW = 60

# Compute daily cross-sectional IC for each alpha
daily_ic = np.zeros((T_total, K))
for k in range(K):
    sig = alpha_signals[k]
    for t in range(1, T_total):
        s = sig[t-1]
        r = returns_arr[t]
        mask = uni_np[t] & (s != 0) & np.isfinite(r) & (r != 0)
        if mask.sum() >= 10:
            ic, _ = spearmanr(s[mask], r[mask])
            if np.isfinite(ic):
                daily_ic[t, k] = ic

print(f'Daily ICs computed in {time.time()-t0:.0f}s')

# Rolling IC (60-day EMA)
ema_decay = np.log(2) / IC_WINDOW
rolling_ic = np.zeros((T_total, K))
for t in range(1, T_total):
    alpha = 1 - np.exp(-ema_decay)
    if t == 1:
        rolling_ic[t] = daily_ic[t]
    else:
        rolling_ic[t] = alpha * daily_ic[t] + (1 - alpha) * rolling_ic[t-1]

# IC weights: max(rolling_ic, 0) — kill negative IC alphas
ic_weights = np.maximum(rolling_ic, 0)  # (T, K)

# Build IC-weighted combined signal
# At time t: combined[t] = sum_k( w_k[t] * signal_k[t] ) / sum_k(w_k[t])
signal_stack = np.stack(alpha_signals, axis=-1)  # (T, N, K)
combined_ic = np.zeros((T_total, N))
for t in range(IC_WINDOW, T_total):
    w = ic_weights[t]  # (K,) — weights based on IC through time t (which uses returns through t, not t+1)
    w_sum = w.sum()
    if w_sum > 0:
        # Weighted average of signals
        combined_ic[t] = (signal_stack[t] * w[np.newaxis, :]).sum(axis=1) / w_sum
    else:
        combined_ic[t] = combined_eqwt[t]  # fallback to equal weight

n_active_oos = np.mean([np.sum(ic_weights[t] > 0) for t in range(np.searchsorted(dates, pd.Timestamp(TRAIN_END)), T_total)])
print(f'Avg active alphas OOS: {n_active_oos:.0f}/{K}')

ic_df = pd.DataFrame(combined_ic, index=dates, columns=tickers)

hdr = f'  {"Method":>20} {"Per":>5} {"Sharpe":>8} {"PnL":>14} {"Ann%":>8} {"TO":>6} {"MaxDD":>7}'
print(hdr)
print(f'  {"-"*75}')

for label, sig_df in [('EqWt (baseline)', eqwt_df), ('IC-weighted', ic_df)]:
    for pname, s, e in [('TRAIN', dates[0], TRAIN_END), ('OOS', TRAIN_END, dates[-1])]:
        r = sim_vec(
            alpha_df=sig_df.loc[s:e],
            returns_df=matrices['returns'].reindex(columns=tickers).loc[s:e],
            close_df=matrices['close'].reindex(columns=tickers).loc[s:e],
            open_df=matrices['open'].reindex(columns=tickers).loc[s:e],
            universe_df=universe_df.loc[s:e],
            booksize=2e6, max_stock_weight=0.05,
            decay=0, delay=1, neutralization='market', fees_bps=5.0,
        )
        ndays = len(matrices['returns'].loc[s:e])
        ann = r.total_pnl / 1e6 / max(1, ndays / 365) * 100
        print(f'  {label:>20} {pname:>5} {r.sharpe:>+8.2f} ${r.total_pnl:>13,.0f} {ann:>7.1f}% {r.turnover:>5.1%} {r.max_drawdown:>6.1%}')

# Also test IC + best decay
print(f'\n  IC-weighted + decay sweep:')
for decay_val in [0, 3, 5, 10]:
    for pname, s, e in [('OOS', TRAIN_END, dates[-1])]:
        r = sim_vec(
            alpha_df=ic_df.loc[s:e],
            returns_df=matrices['returns'].reindex(columns=tickers).loc[s:e],
            close_df=matrices['close'].reindex(columns=tickers).loc[s:e],
            open_df=matrices['open'].reindex(columns=tickers).loc[s:e],
            universe_df=universe_df.loc[s:e],
            booksize=2e6, max_stock_weight=0.05,
            decay=decay_val, delay=1, neutralization='market', fees_bps=5.0,
        )
        ndays = len(matrices['returns'].loc[s:e])
        ann = r.total_pnl / 1e6 / max(1, ndays / 365) * 100
        print(f'  IC+decay={decay_val:<3} {pname:>5} {r.sharpe:>+8.2f} ${r.total_pnl:>13,.0f} {ann:>7.1f}% {r.turnover:>5.1%} {r.max_drawdown:>6.1%}')

print(f'\nTotal runtime: {time.time()-t0:.0f}s')
