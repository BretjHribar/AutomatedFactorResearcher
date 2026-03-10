"""
Sharpe scaling law: Sharpe vs number of alphas.
Test with and without correlation cutoff.
Fit log-linear model: Sharpe = a * log(N) + b
"""
import sys, warnings, time, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sqlite3
from pathlib import Path
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
uni100 = pd.read_parquet(DATA_DIR / 'universes/BINANCE_TOP100.parquet')
close = matrices['close'].copy()
returns = matrices['returns'].copy()
for col in close.columns:
    lv = close[col].last_valid_index()
    if lv is not None and lv < close.index[-1]:
        returns.loc[returns.index > lv, col] = 0.0
matrices['returns'] = returns

all_tickers = sorted(uni100.columns[uni100.any()].tolist())
for name in list(matrices.keys()):
    cols = [c for c in all_tickers if c in matrices[name].columns]
    if cols:
        matrices[name] = matrices[name][cols]
    else:
        del matrices[name]
tickers = matrices['close'].columns.tolist()
dates = matrices['returns'].index
N = len(tickers)

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

print(f'Data loaded: {N} tickers, {len(dates)} days')

# Load alphas sorted by fitness (best first)
conn = sqlite3.connect('data/alpha_gp_crypto.db')
cur = conn.cursor()
cur.execute("""SELECT a.expression, e.sharpe, e.fitness 
               FROM alphas a JOIN evaluations e ON a.alpha_id=e.alpha_id 
               WHERE e.sharpe >= 1.0 AND e.fitness >= 1.0
               ORDER BY e.fitness DESC""")
rows = cur.fetchall()
conn.close()
expressions = [r[0] for r in rows]
print(f'{len(expressions)} alphas loaded (sorted by fitness)')

# Evaluate all alphas and get their rank-normalized signals
engine = FastExpressionEngine(data_fields={n: matrices[n] for n in TERMINALS if n in matrices})
alpha_signals = []  # (T, N) arrays
alpha_exprs = []

for i, expr in enumerate(expressions):
    try:
        a = engine.evaluate(expr)
        if a is not None and not a.empty:
            ranked = a.reindex(index=dates, columns=tickers).rank(axis=1, pct=True) - 0.5
            alpha_signals.append(ranked.fillna(0).values)
            alpha_exprs.append(expr)
    except:
        pass

K = len(alpha_signals)
print(f'{K} valid alphas evaluated in {time.time()-t0:.0f}s')

# Compute pairwise correlations between alpha signals
# Use the TRAIN period only for correlation calc
train_end_idx = np.searchsorted(dates, pd.Timestamp(TRAIN_END))
print(f'Computing pairwise correlations (train period)...')

# Flatten each signal to a 1D vector (train period only, times x stocks)
flat_signals = []
for k in range(K):
    s = alpha_signals[k][:train_end_idx].flatten()
    flat_signals.append(s)
flat_mat = np.stack(flat_signals, axis=0)  # (K, T*N)

# Correlation matrix
corr_mat = np.corrcoef(flat_mat)
print(f'Correlation matrix computed: {corr_mat.shape}')

# Mean and max pairwise correlation
upper = corr_mat[np.triu_indices(K, k=1)]
print(f'Pairwise corr stats: mean={np.mean(upper):.3f}, median={np.median(upper):.3f}, '
      f'p90={np.percentile(upper, 90):.3f}, max={np.max(upper):.3f}')

# Helper: simulate a subset of alphas
def sim_subset(indices, period='OOS'):
    combined = np.mean(np.stack([alpha_signals[i] for i in indices], axis=-1), axis=-1)
    sig_df = pd.DataFrame(combined, index=dates, columns=tickers)
    if period == 'OOS':
        s, e = TRAIN_END, dates[-1]
    else:
        s, e = dates[0], TRAIN_END
    r = sim_vec(
        alpha_df=sig_df.loc[s:e],
        returns_df=matrices['returns'].reindex(columns=tickers).loc[s:e],
        close_df=matrices['close'].reindex(columns=tickers).loc[s:e],
        open_df=matrices['open'].reindex(columns=tickers).loc[s:e],
        universe_df=uni100.loc[s:e],
        booksize=2e6, max_stock_weight=0.05,
        decay=0, delay=1, neutralization='market', fees_bps=5.0,
    )
    return r.sharpe, r.total_pnl, r.turnover, r.max_drawdown

# ═══════════════════════════════════════════════════════════════
# TEST 1: Progressive alpha addition (no correlation filter)
# ═══════════════════════════════════════════════════════════════
print(f'\n{"="*90}')
print(f'SCALING LAW: Sharpe vs N_alphas (TOP100, delay=1, 5 bps)')
print(f'{"="*90}')

counts = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 209]
counts = [c for c in counts if c <= K]

print(f'\n--- No correlation filter (add in fitness order) ---')
print(f'  {"N":>5} {"OOS Sharpe":>12} {"Train Sharpe":>14} {"OOS PnL":>14} {"TO":>6} {"MaxDD":>7}')
print(f'  {"-"*65}')

no_filter_results = []
for n_alpha in counts:
    idx = list(range(n_alpha))
    s_oos, pnl_oos, to_oos, dd_oos = sim_subset(idx, 'OOS')
    s_train, _, _, _ = sim_subset(idx, 'TRAIN')
    no_filter_results.append((n_alpha, s_oos, s_train, pnl_oos))
    print(f'  {n_alpha:>5} {s_oos:>+12.2f} {s_train:>+14.2f} ${pnl_oos:>13,.0f} {to_oos:>5.1%} {dd_oos:>6.1%}')

# ═══════════════════════════════════════════════════════════════
# TEST 2: With correlation cutoff
# ═══════════════════════════════════════════════════════════════
for corr_cutoff in [0.7, 0.5, 0.3]:
    print(f'\n--- Correlation cutoff = {corr_cutoff} ---')
    # Greedy selection: add alpha if max corr with existing < cutoff
    selected = [0]  # Start with best alpha
    for k in range(1, K):
        max_corr_with_existing = max(abs(corr_mat[k, s]) for s in selected)
        if max_corr_with_existing < corr_cutoff:
            selected.append(k)
    
    print(f'  {len(selected)} alphas survive cutoff')
    
    # Progressive addition from the selected set
    sub_counts = [c for c in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200] if c <= len(selected)]
    
    print(f'  {"N":>5} {"OOS Sharpe":>12} {"Train Sharpe":>14} {"OOS PnL":>14}')
    print(f'  {"-"*55}')
    
    corr_results = []
    for n_alpha in sub_counts:
        idx = selected[:n_alpha]
        s_oos, pnl_oos, _, _ = sim_subset(idx, 'OOS')
        s_train, _, _, _ = sim_subset(idx, 'TRAIN')
        corr_results.append((n_alpha, s_oos, s_train, pnl_oos))
        print(f'  {n_alpha:>5} {s_oos:>+12.2f} {s_train:>+14.2f} ${pnl_oos:>13,.0f}')

# ═══════════════════════════════════════════════════════════════
# FIT LOG-LINEAR MODEL
# ═══════════════════════════════════════════════════════════════
print(f'\n{"="*90}')
print(f'LOG-LINEAR FIT: Sharpe = a * ln(N) + b')
print(f'{"="*90}')

for label, results in [('No filter', no_filter_results)]:
    ns = np.array([r[0] for r in results])
    sharpes_oos = np.array([r[1] for r in results])
    sharpes_train = np.array([r[2] for r in results])
    
    # Only fit where N >= 3
    mask = ns >= 3
    log_ns = np.log(ns[mask])
    
    # OOS fit
    coeffs_oos = np.polyfit(log_ns, sharpes_oos[mask], 1)
    a_oos, b_oos = coeffs_oos
    r2_oos = 1 - np.sum((sharpes_oos[mask] - (a_oos * log_ns + b_oos))**2) / np.sum((sharpes_oos[mask] - np.mean(sharpes_oos[mask]))**2)
    
    # Train fit
    coeffs_train = np.polyfit(log_ns, sharpes_train[mask], 1)
    a_train, b_train = coeffs_train
    
    print(f'\n{label}:')
    print(f'  OOS:   Sharpe = {a_oos:.3f} * ln(N) + {b_oos:.3f}  (R2 = {r2_oos:.3f})')
    print(f'  Train: Sharpe = {a_train:.3f} * ln(N) + {b_train:.3f}')
    
    # Predict how many alphas needed for target Sharpes
    for target in [1.5, 2.0, 2.5, 3.0]:
        n_needed = np.exp((target - b_oos) / a_oos) if a_oos > 0 else float('inf')
        print(f'  -> OOS Sharpe {target:.1f} needs ~{n_needed:.0f} alphas')

# Also fit sqrt(N) model: Sharpe = a * sqrt(N) + b (theoretical for uncorrelated alphas)
print(f'\nSQRT FIT: Sharpe = a * sqrt(N) + b')
ns = np.array([r[0] for r in no_filter_results])
sharpes_oos = np.array([r[1] for r in no_filter_results])
mask = ns >= 3
sqrt_ns = np.sqrt(ns[mask])
coeffs_sqrt = np.polyfit(sqrt_ns, sharpes_oos[mask], 1)
a_sq, b_sq = coeffs_sqrt
r2_sqrt = 1 - np.sum((sharpes_oos[mask] - (a_sq * sqrt_ns + b_sq))**2) / np.sum((sharpes_oos[mask] - np.mean(sharpes_oos[mask]))**2)
print(f'  OOS: Sharpe = {a_sq:.3f} * sqrt(N) + {b_sq:.3f}  (R2 = {r2_sqrt:.3f})')
for target in [1.5, 2.0, 2.5, 3.0]:
    n_needed = ((target - b_sq) / a_sq) ** 2 if a_sq > 0 else float('inf')
    print(f'  -> OOS Sharpe {target:.1f} needs ~{n_needed:.0f} alphas')

print(f'\nDone in {time.time()-t0:.0f}s')
