"""
QP Pipeline for Crypto — NO RISK MODEL, just dollar-neutral + position limits.

The QP just maximizes alpha - tcost, subject to:
  - sum(w) = 0 (dollar-neutral)
  - |w_i| <= 5%
  - sum(|w_i|) <= 1 (GMV <= booksize)

No PCA, no covariance matrix, no factor risk.
Delay=0, 5 bps fees (same as simple sim for fair comparison).
"""
import sys, os, sqlite3, time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import cvxpy as cp
from pathlib import Path
sys.path.insert(0, '.')

from src.operators.fastexpression import FastExpressionEngine

TRAIN_END = "2024-04-27"
DATA_DIR = Path("data/binance_cache")
DB_PATH = "data/alpha_gp_crypto.db"
RESULTS_FILE = "crypto_qp_simple.txt"

BOOKSIZE = 2_000_000.0
MAX_WEIGHT = 0.05  # 5% per coin
FEES_BPS = 5.0     # Same as simple sim
DELAY = 0

out = open(RESULTS_FILE, 'w', encoding='ascii', errors='replace')
def log(msg):
    print(msg, flush=True)
    out.write(msg + '\n')
    out.flush()

log("=" * 90)
log("CRYPTO QP -- NO RISK MODEL, DOLLAR-NEUTRAL ONLY")
log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 90)
log(f"\nConfig:")
log(f"  Booksize:     ${BOOKSIZE:,.0f}")
log(f"  Max weight:   {MAX_WEIGHT:.0%} per coin")
log(f"  Fees:         {FEES_BPS} bps one-way")
log(f"  Delay:        {DELAY}")
log(f"  Risk model:   NONE")
log(f"  Constraints:  sum(w)=0 (dollar-neutral), |w_i|<=5%, sum(|w|)<=1")

# ── Load data ──
t0 = time.time()
log("\n[1/4] Loading data...")

matrices = {}
for f in sorted((DATA_DIR / "matrices").glob("*.parquet")):
    matrices[f.stem] = pd.read_parquet(f)

universe_df = pd.read_parquet(DATA_DIR / "universes/BINANCE_TOP50.parquet")

# Handle delistings
close = matrices['close'].copy()
returns = matrices['returns'].copy()
n_del = 0
for col in close.columns:
    lv = close[col].last_valid_index()
    if lv is not None and lv < close.index[-1]:
        returns.loc[returns.index > lv, col] = 0.0
        n_del += 1
matrices['returns'] = returns
matrices['close'] = close

all_tickers = sorted(universe_df.columns[universe_df.any()].tolist())
for name in list(matrices.keys()):
    cols = [c for c in all_tickers if c in matrices[name].columns]
    if cols:
        matrices[name] = matrices[name][cols]
    else:
        del matrices[name]

tickers = matrices['close'].columns.tolist()
N = len(tickers)
log(f"  {N} tickers, {n_del} delisted")

TERMINALS = [t for t in [
    "close", "open", "high", "low", "volume", "returns", "log_returns",
    "taker_buy_ratio", "taker_buy_volume", "vwap", "vwap_deviation",
    "high_low_range", "open_close_range", "adv20", "adv60",
    "volume_ratio_20d", "historical_volatility_20", "historical_volatility_60",
    "momentum_5d", "momentum_20d", "momentum_60d", "beta_to_btc",
    "overnight_gap", "upper_shadow", "lower_shadow",
    "close_position_in_range", "trades_count", "trades_per_volume",
    "parkinson_volatility_20", "quote_volume",
] if t in matrices]

log(f"  Load time: {time.time()-t0:.0f}s")

# ── Get qualifying alphas ──
log("\n[2/4] Loading alphas...")
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("""SELECT a.expression, e.sharpe, e.fitness 
               FROM alphas a JOIN evaluations e ON a.alpha_id=e.alpha_id
               WHERE e.sharpe >= 1.0 AND e.fitness >= 1.0
               ORDER BY e.fitness DESC""")
qualifying = cur.fetchall()
conn.close()
expressions = [r[0] for r in qualifying]
log(f"  {len(qualifying)} qualifying alphas")

# ── Pre-compute combined alpha signal ──
log("\n[3/4] Building combined alpha signal...")
full_engine = FastExpressionEngine(
    data_fields={n: matrices[n] for n in TERMINALS if n in matrices}
)

combined_signal = None
n_good = 0
for expr in expressions:
    try:
        a = full_engine.evaluate(expr)
        if a is not None and not a.empty:
            a = a.reindex(index=matrices['close'].index, columns=tickers)
            ranked = a.rank(axis=1, pct=True) - 0.5
            ranked = ranked.fillna(0)
            if combined_signal is None:
                combined_signal = ranked.copy()
            else:
                combined_signal = combined_signal + ranked
            n_good += 1
    except:
        pass

if n_good > 0:
    combined_signal = combined_signal / n_good
log(f"  {n_good}/{len(expressions)} alphas combined")

# ── Day-by-day QP backtest ──
log("\n[4/4] Running day-by-day QP backtest...")

returns_df = matrices['returns']
all_dates = returns_df.index.tolist()

# Find OOS start
oos_start_idx = 0
for i, d in enumerate(all_dates):
    if str(d)[:10] >= TRAIN_END:
        oos_start_idx = i
        break

# Universe mask
uni_mask = universe_df.reindex(index=returns_df.index, columns=tickers).fillna(False)

# Pre-compute arrays
alpha_arr = combined_signal.reindex(index=returns_df.index, columns=tickers).fillna(0).values
returns_arr = returns_df.reindex(columns=tickers).fillna(0).values
uni_arr = uni_mask.values

# QP solve function (no risk model)
def solve_qp(alpha_vec, w_prev, uni_mask, w_max=MAX_WEIGHT):
    """Solve simple QP: max alpha'w - tcost, s.t. dollar-neutral + position limits."""
    n = len(alpha_vec)
    
    # Only optimize over universe stocks
    uni_idx = np.where(uni_mask)[0]
    n_uni = len(uni_idx)
    if n_uni < 4:
        return np.zeros(n)
    
    alpha_sub = alpha_vec[uni_idx]
    w_prev_sub = w_prev[uni_idx]
    
    w = cp.Variable(n_uni)
    T = w - w_prev_sub
    
    # Objective: maximize alpha'w - linear tcost - small L2 regularization
    linear_cost = FEES_BPS * 1e-4
    reg = 0.001  # tiny L2 to make QP well-posed
    
    objective = cp.Minimize(
        -alpha_sub @ w           # negative because we minimize
        + linear_cost * cp.norm(T, 1)  # transaction cost
        + reg * cp.sum_squares(w)      # small regularization
    )
    
    constraints = [
        w >= -w_max,
        w <= w_max,
        cp.norm(w, 1) <= 1.0,   # GMV <= booksize
        cp.sum(w) == 0,          # dollar-neutral
    ]
    
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.OSQP, max_iter=20000, eps_abs=1e-5, 
                   eps_rel=1e-5, verbose=False, warm_start=True)
        if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
            result = np.zeros(n)
            result[uni_idx] = w.value
            return result
    except:
        pass
    
    try:
        prob.solve(solver=cp.SCS, max_iters=10000, verbose=False)
        if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
            result = np.zeros(n)
            result[uni_idx] = w.value
            return result
    except:
        pass
    
    return w_prev  # fallback: hold

# Run
w_prev = np.zeros(N)
daily_results = []
n_days = len(all_dates)
report_every = max(1, n_days // 40)

log(f"  Total days: {n_days}")
log(f"  Train: to day {oos_start_idx}")
log(f"  OOS:   from day {oos_start_idx}")

for t in range(n_days):
    date_str = str(all_dates[t])[:10]
    r_today = returns_arr[t]
    
    # 1. PnL from previous positions
    h_prev = w_prev * BOOKSIZE
    gross_pnl = np.dot(h_prev, r_today)
    
    # 2. Get alpha signal
    alpha_signal = alpha_arr[t] if DELAY == 0 else (alpha_arr[t-1] if t > 0 else np.zeros(N))
    
    # 3. Apply universe mask
    umask = uni_arr[t].astype(bool) if t < len(uni_arr) else np.ones(N, dtype=bool)
    
    # Zero prev weights for exited universe
    w_prev_clean = w_prev.copy()
    w_prev_clean[~umask] = 0.0
    
    # 4. QP optimize
    if t >= 10 and np.any(alpha_signal[umask] != 0):
        w_new = solve_qp(alpha_signal, w_prev_clean, umask)
    else:
        w_new = np.zeros(N)
    
    # 5. Costs
    h_new = w_new * BOOKSIZE
    trades = h_new - h_prev
    trade_notional = np.sum(np.abs(trades))
    tcost = FEES_BPS * 1e-4 * trade_notional
    
    net_pnl = gross_pnl - tcost
    gmv = np.sum(np.abs(h_new))
    turnover = trade_notional / max(gmv, 1) if gmv > 0 else 0
    n_long = int(np.sum(w_new > 1e-6))
    n_short = int(np.sum(w_new < -1e-6))
    
    daily_results.append({
        'date': date_str, 'pnl': net_pnl, 'gross_pnl': gross_pnl,
        'tcost': tcost, 'gmv': gmv, 'turnover': turnover,
        'n_long': n_long, 'n_short': n_short,
        'is_oos': t >= oos_start_idx,
    })
    
    w_prev = w_new
    
    if t % report_every == 0 or t == n_days - 1:
        cum_pnl = sum(r['pnl'] for r in daily_results)
        log(f"    [{date_str}] day {t+1}/{n_days} | "
            f"cumPnL: ${cum_pnl:+,.0f} | GMV: ${gmv:,.0f} | "
            f"TO: {turnover:.1%} | L/S: {n_long}/{n_short}")

# ── Stats ──
def compute_stats(results, label):
    pnls = [r['pnl'] for r in results]
    gross = [r['gross_pnl'] for r in results]
    tcosts = [r['tcost'] for r in results]
    gmvs = [r['gmv'] for r in results]
    
    n = len(pnls)
    total_pnl = sum(pnls)
    total_gross = sum(gross)
    total_tcost = sum(tcosts)
    avg_gmv = np.mean(gmvs)
    avg_to = np.mean([r['turnover'] for r in results])
    
    daily_rets = [p / max(g, 1) for p, g in zip(pnls, gmvs) if g > 1000]
    sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(365) if len(daily_rets) > 10 else 0
    
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    max_dd = np.min(cum - peak) / max(BOOKSIZE * 0.5, 1) if len(cum) > 0 else 0
    
    years = n / 365
    ann_ret = total_pnl / (BOOKSIZE * 0.5) / max(years, 0.1) * 100
    
    avg_l = np.mean([r['n_long'] for r in results])
    avg_s = np.mean([r['n_short'] for r in results])
    
    log(f"\n  {label}:")
    log(f"    Sharpe:     {sharpe:+.2f}")
    log(f"    Total PnL:  ${total_pnl:>12,.0f}")
    log(f"    Gross PnL:  ${total_gross:>12,.0f}")
    log(f"    Total Costs:${total_tcost:>12,.0f}")
    log(f"    Ann Return: {ann_ret:>8.1f}%")
    log(f"    Avg GMV:    ${avg_gmv:>12,.0f}")
    log(f"    Avg TO:     {avg_to:.1%}")
    log(f"    Max DD:     {max_dd:.1%}")
    log(f"    Avg L/S:    {avg_l:.0f}/{avg_s:.0f}")
    return sharpe, total_pnl, max_dd

train_r = [r for r in daily_results if not r['is_oos'] and r['gmv'] > 1000]
oos_r = [r for r in daily_results if r['is_oos'] and r['gmv'] > 1000]
all_r = [r for r in daily_results if r['gmv'] > 1000]

s_t, p_t, d_t = compute_stats(train_r, "TRAINING")
s_o, p_o, d_o = compute_stats(oos_r, "OOS (GP never saw this)")
s_f, p_f, d_f = compute_stats(all_r, "FULL")

log(f"\n{'='*90}")
log(f"SUMMARY")
log(f"{'='*90}")
log(f"  {'':>10} | {'Sharpe':>8} | {'PnL':>14} | {'MaxDD':>8}")
log(f"  {'-'*50}")
log(f"  {'TRAIN':>10} | {s_t:+8.2f} | ${p_t:>13,.0f} | {d_t:>7.1%}")
log(f"  {'OOS':>10} | {s_o:+8.2f} | ${p_o:>13,.0f} | {d_o:>7.1%}")
log(f"  {'FULL':>10} | {s_f:+8.2f} | ${p_f:>13,.0f} | {d_f:>7.1%}")

log(f"\nCompleted: {time.strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Runtime: {time.time()-t0:.0f}s")
out.close()
print(f"\nResults saved to {RESULTS_FILE}")
