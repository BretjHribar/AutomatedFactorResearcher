"""
EWMA sweep on FULL alpha training window (365 days) + top-N position filter.
Also tests 3 bps specifically.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import eval_alpha_5m as ea
import eval_portfolio_5m as ep

BOOKSIZE = ep.BOOKSIZE
MAX_WT = ea.MAX_WEIGHT
BARS_PER_DAY = ep.BARS_PER_DAY

def simulate_ewma(alpha_np, ret_np, ewma_span=1, fees_bps=0, top_n=0,
                  booksize=BOOKSIZE, max_wt=MAX_WT):
    """
    EWMA smoothed signal + optional top-N concentration.
    top_n=0 means trade all positions.
    """
    n_bars, n_tickers = alpha_np.shape
    
    # Apply EWMA to signal
    sig_arr = alpha_np.copy()
    if ewma_span > 1:
        ema_alpha = 2.0 / (ewma_span + 1.0)
        for t in range(1, n_bars):
            sig_arr[t] = ema_alpha * sig_arr[t] + (1 - ema_alpha) * sig_arr[t-1]
    
    fee_rate = fees_bps / 10_000.0
    positions = np.zeros(n_tickers)
    pnl_arr = np.zeros(n_bars)
    turnover_total = 0.0
    
    for t in range(1, n_bars):
        pnl_arr[t] = np.sum(positions * ret_np[t]) * booksize
        
        sig = sig_arr[t].copy()
        
        # Top-N filter: zero out all but top N by absolute signal
        if top_n > 0 and np.count_nonzero(sig) > top_n:
            abs_sig = np.abs(sig)
            threshold = np.sort(abs_sig)[-top_n]
            sig[abs_sig < threshold] = 0.0
        
        abs_sum = np.abs(sig).sum()
        if abs_sum > 1e-10:
            new_pos = np.clip(sig / abs_sum, -max_wt, max_wt)
            abs_sum2 = np.abs(new_pos).sum()
            if abs_sum2 > 1e-10: new_pos /= abs_sum2
        else:
            new_pos = positions.copy()
        
        turnover = np.abs(new_pos - positions).sum()
        turnover_total += turnover
        pnl_arr[t] -= turnover * fee_rate * booksize
        positions = new_pos
    
    cum_pnl = np.cumsum(pnl_arr)
    daily_pnl = [np.sum(pnl_arr[d:d+BARS_PER_DAY]) for d in range(0, n_bars, BARS_PER_DAY)]
    daily_pnl = np.array(daily_pnl)
    
    sharpe = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(365) if np.std(daily_pnl) > 0 else 0
    total_ret = cum_pnl[-1] / booksize
    avg_to = turnover_total / (n_bars - 1)
    gross_pnl = np.sum(pnl_arr) + turnover_total * fee_rate * booksize
    
    return {
        'sharpe': sharpe, 'total_return': total_ret, 'cum_pnl': cum_pnl,
        'avg_turnover': avg_to, 'daily_pnl': daily_pnl,
        'gross_return': gross_pnl / booksize,
    }


# ===== FULL ALPHA TRAIN (365 days, Feb 25 - Feb 26) =====
print("Loading FULL alpha training data (365 days)...")
matrices, universe = ea.load_data("train")
close = matrices["close"]
returns = close.pct_change()

alphas = ep.load_alphas(universe="TOP100")

# Build composite
raw_sum = None; n = 0
for aid, expr, ic, sr in alphas:
    try:
        raw = ea.evaluate_expression(expr, matrices)
        if raw is None: continue
        r = raw.fillna(0)
        if raw_sum is None: raw_sum = r.copy()
        else: raw_sum = raw_sum.add(r, fill_value=0)
        n += 1
    except: pass
raw_avg = raw_sum / n
composite = ea.process_signal(raw_avg, universe_df=universe, max_wt=MAX_WT)

# Align
common = composite.columns.intersection(returns.columns).tolist()
idx = composite.index.intersection(returns.index)
alpha_np = np.nan_to_num(composite.loc[idx, common].values.astype(np.float64), nan=0)
ret_np = np.nan_to_num(returns.loc[idx, common].values.astype(np.float64), nan=0)
print(f"Full year: {len(idx)} bars, {len(common)} tickers\n")

# ===== SWEEP: EWMA × fees × top-N =====
spans = [1, 6, 12, 24, 36, 48, 72, 96, 144, 288, 576]
fees_list = [0, 1, 2, 3, 5, 7]
top_ns = [0, 40, 20, 10]  # 0=all, then concentrated

for top_n in top_ns:
    label = f"All {len(common)}" if top_n == 0 else f"Top {top_n}"
    print(f"\n{'='*90}")
    print(f"TOP-N = {label}")
    print(f"{'='*90}")
    print(f"{'Span':>6} {'Period':>8} {'TO':>7} {'SR(0)':>7} {'SR(1)':>7} {'SR(2)':>7} {'SR(3)':>7} {'SR(5)':>7} {'SR(7)':>7} {'Ret(0)':>8} {'Ret(3)':>8} {'Ret(7)':>8}")
    
    for sp in spans:
        hl = f"{sp*5/60:.0f}h" if sp < 288 else f"{sp*5/60/24:.0f}d"
        results = {}
        for fee in fees_list:
            r = simulate_ewma(alpha_np, ret_np, ewma_span=sp, fees_bps=fee, top_n=top_n)
            results[fee] = r
        
        r0, r3, r7 = results[0], results[3], results[7]
        print(f"{sp:>6} {hl:>8} {r0['avg_turnover']:7.4f} "
              f"{results[0]['sharpe']:+7.2f} {results[1]['sharpe']:+7.2f} {results[2]['sharpe']:+7.2f} "
              f"{results[3]['sharpe']:+7.2f} {results[5]['sharpe']:+7.2f} {results[7]['sharpe']:+7.2f} "
              f"{r0['total_return']:+8.1%} {r3['total_return']:+8.1%} {r7['total_return']:+8.1%}")

# ===== Best config deep dive =====
print(f"\n{'='*90}")
print(f"BEST CONFIGS: EMA × Top-N at 3 bps")
print(f"{'='*90}")

best_configs = []
for sp in [48, 72, 96, 144, 288]:
    for top_n in [0, 40, 20, 10]:
        r = simulate_ewma(alpha_np, ret_np, ewma_span=sp, fees_bps=3, top_n=top_n)
        label = f"All" if top_n == 0 else f"Top{top_n}"
        best_configs.append((sp, top_n, r))
        if r['sharpe'] > 0:
            print(f"  ✓ EMA={sp} ({sp*5/60:.0f}h) {label:>5}: SR={r['sharpe']:+.2f}  Ret={r['total_return']:+.1%}  TO={r['avg_turnover']:.4f}")

# Plot: equity curves for best 3bps configs
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. EWMA sweep at 3 bps, all positions
ax = axes[0, 0]
for sp in [1, 12, 48, 96, 144, 288]:
    r = simulate_ewma(alpha_np, ret_np, ewma_span=sp, fees_bps=3, top_n=0)
    hl = f"{sp*5/60:.0f}h" if sp < 288 else "1d"
    ax.plot(r['cum_pnl'], label=f"EMA {hl} (SR={r['sharpe']:+.1f})", linewidth=1.5)
ax.set_title('Full Year, 3 bps, All Positions')
ax.set_xlabel('Bar')
ax.set_ylabel('Cum PnL ($)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2. Top-N comparison at best EMA (96)
ax = axes[0, 1]
for top_n in [0, 40, 20, 10]:
    r = simulate_ewma(alpha_np, ret_np, ewma_span=96, fees_bps=3, top_n=top_n)
    label = f"All" if top_n == 0 else f"Top {top_n}"
    ax.plot(r['cum_pnl'], label=f"{label} (SR={r['sharpe']:+.1f}, TO={r['avg_turnover']:.3f})", linewidth=1.5)
ax.set_title('EMA=96 (8h), 3 bps: Position Concentration')
ax.set_xlabel('Bar')
ax.set_ylabel('Cum PnL ($)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 3. SR surface: EWMA × fees for all positions
ax = axes[1, 0]
for fee, color in [(0, 'blue'), (1, 'cyan'), (2, 'green'), (3, 'orange'), (5, 'red'), (7, 'darkred')]:
    srs = [simulate_ewma(alpha_np, ret_np, ewma_span=sp, fees_bps=fee, top_n=0)['sharpe'] for sp in spans]
    ax.semilogx(spans, srs, 'o-', color=color, linewidth=2, label=f'{fee}bps')
ax.axhline(y=0, color='gray', linestyle='--')
ax.set_xlabel('EWMA Span')
ax.set_ylabel('Sharpe')
ax.set_title('Full Year: SR vs EMA Span at Different Fee Levels')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. 3bps equity curves across top-N at best EMA
ax = axes[1, 1]
for sp in [72, 96, 144]:
    for top_n in [0, 20]:
        r = simulate_ewma(alpha_np, ret_np, ewma_span=sp, fees_bps=3, top_n=top_n)
        hl = f"{sp*5/60:.0f}h"
        label = f"All" if top_n == 0 else f"Top{top_n}"
        ax.plot(r['cum_pnl'], label=f"EMA {hl} {label} (SR={r['sharpe']:+.1f})", linewidth=1.5)
ax.set_title('3 bps: EMA × Top-N Comparison')
ax.set_xlabel('Bar')
ax.set_ylabel('Cum PnL ($)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diag_ewma_full_year.png', dpi=150)
print(f"\nChart saved: diag_ewma_full_year.png")
plt.close()
print("\nDone.")
