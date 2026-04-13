"""
Debug: Why does daily rebalancing outperform 6h rebalancing at 0 fees?
This tests whether it's a timing artifact from t % N == 0 alignment.
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

def simulate_rebal(alpha_np, ret_np, rebal_every=1, offset=0, fees_bps=0, 
                   booksize=BOOKSIZE, max_wt=MAX_WT):
    """Sim with rebalance every N bars, starting at offset."""
    n_bars, n_tickers = alpha_np.shape
    fee_rate = fees_bps / 10_000.0
    positions = np.zeros(n_tickers)
    pnl_arr = np.zeros(n_bars)
    
    for t in range(1, n_bars):
        # PnL from holding
        pnl_arr[t] = np.sum(positions * ret_np[t]) * booksize
        
        # Rebalance?
        if (t - offset) % rebal_every == 0:
            sig = alpha_np[t].copy()
            abs_sum = np.abs(sig).sum()
            if abs_sum > 1e-10:
                new_pos = np.clip(sig / abs_sum, -max_wt, max_wt)
                abs_sum2 = np.abs(new_pos).sum()
                if abs_sum2 > 1e-10: new_pos /= abs_sum2
            else:
                new_pos = np.zeros(n_tickers)
            
            turnover = np.abs(new_pos - positions).sum()
            pnl_arr[t] -= turnover * fee_rate * booksize
            positions = new_pos
    
    cum_pnl = np.cumsum(pnl_arr)
    daily_pnl = [np.sum(pnl_arr[d:d+BARS_PER_DAY]) for d in range(0, n_bars, BARS_PER_DAY)]
    daily_pnl = np.array(daily_pnl)
    
    if len(daily_pnl) > 1 and np.std(daily_pnl) > 0:
        sharpe = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(365)
    else:
        sharpe = 0
    total_ret = cum_pnl[-1] / booksize
    
    return sharpe, total_ret, cum_pnl, daily_pnl


# Load val data
alphas = ep.load_alphas(universe="TOP100")
matrices, universe = ep.load_data("val")
returns = matrices["returns"]

# Build EW composite using process_signal (NOT compute_composite_signal)
print("Building EW composite with process_signal...")
raw_sum = None; n = 0
for aid, expr, ic, sr in alphas:
    try:
        raw = ep.evaluate_expression(expr, matrices)
        if raw is None: continue
        r = raw.fillna(0)
        if raw_sum is None: raw_sum = r.copy()
        else: raw_sum = raw_sum.add(r, fill_value=0)
        n += 1
    except: pass
raw_avg = raw_sum / n
composite = ea.process_signal(raw_avg, universe_df=universe, max_wt=MAX_WT)
print(f"Composite: {composite.shape}, {n} alphas")

# Align
common = composite.columns.intersection(returns.columns).tolist()
idx = composite.index.intersection(returns.index)
alpha_np = composite.loc[idx, common].values.astype(np.float64)
ret_np = returns.loc[idx, common].values.astype(np.float64)
alpha_np = np.nan_to_num(alpha_np, nan=0)
ret_np = np.nan_to_num(ret_np, nan=0)
n_bars = len(idx)
print(f"N bars: {n_bars}, N tickers: {len(common)}")

# Test: rebalance frequency sweep with MULTIPLE OFFSETS
rebal_bars = [1, 3, 6, 12, 24, 36, 72, 144, 288]
n_offsets = 10

print(f"\n{'='*80}")
print(f"REBALANCE FREQUENCY SWEEP WITH OFFSET AVERAGING (val, 0 fees)")
print(f"{'='*80}")
print(f"{'Rebal':>8} {'Period':>8} {'SR(off=0)':>10} {'SR(avg)':>10} {'Ret(off=0)':>12} {'Ret(avg)':>12} {'SR_std':>8}")

results = {}
for rb in rebal_bars:
    srs = []
    rets = []
    for off in range(min(rb, n_offsets)):
        sr, ret, _, _ = simulate_rebal(alpha_np, ret_np, rebal_every=rb, offset=off, fees_bps=0)
        srs.append(sr)
        rets.append(ret)
    
    sr0, ret0, cum0, dpnl0 = simulate_rebal(alpha_np, ret_np, rebal_every=rb, offset=0, fees_bps=0)
    results[rb] = (sr0, ret0, cum0, np.mean(srs), np.std(srs), np.mean(rets))
    
    print(f"{rb:>8} {f'{rb*5}m':>8} {sr0:+10.2f} {np.mean(srs):+10.2f} {ret0:+12.1%} {np.mean(rets):+12.1%} {np.std(srs):8.2f}")

# Also test: what if we look at cumulative returns by bar, not just final
print(f"\n{'='*80}")
print(f"BAR-LEVEL STATS")
print(f"{'='*80}")
for rb in [1, 72, 288]:
    sr, ret, cum, dpnl = simulate_rebal(alpha_np, ret_np, rebal_every=rb, offset=0, fees_bps=0)
    print(f"\nRebal every {rb} bars ({rb*5}m):")
    print(f"  Final cum PnL: ${cum[-1]:,.0f}")
    print(f"  Mean daily PnL: ${np.mean(dpnl):,.0f}")
    print(f"  Std daily PnL: ${np.std(dpnl):,.0f}")
    print(f"  Daily SR: {np.mean(dpnl)/np.std(dpnl):.4f}")
    print(f"  N days: {len(dpnl)}")
    print(f"  Daily PnLs: {['$'+f'{p:,.0f}' for p in dpnl]}")

# PLOT
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Equity curves at different rebalance freqs
ax = axes[0, 0]
for rb in [1, 12, 36, 72, 144, 288]:
    sr, ret, cum, dpnl = results[rb][:4] if rb in results else simulate_rebal(alpha_np, ret_np, rebal_every=rb)[:3]
    cum = results[rb][2]
    ax.plot(cum, label=f"{rb*5}m (SR={results[rb][0]:+.1f})", linewidth=1.5)
ax.set_title('Val (0 fees): Equity Curves by Rebalance Freq')
ax.set_xlabel('Bar')
ax.set_ylabel('Cumulative PnL ($)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2. SR with offset averaging
ax = axes[0, 1]
rbs = list(results.keys())
sr_off0 = [results[rb][0] for rb in rbs]
sr_avg = [results[rb][3] for rb in rbs]
sr_std = [results[rb][4] for rb in rbs]
ax.errorbar(range(len(rbs)), sr_avg, yerr=sr_std, fmt='ro-', capsize=5, label='Avg over offsets ± std')
ax.plot(range(len(rbs)), sr_off0, 'bs--', label='Offset=0 only')
ax.set_xticks(range(len(rbs)))
ax.set_xticklabels([f"{rb*5}m" if rb < 288 else "1d" for rb in rbs], rotation=45)
ax.set_ylabel('Sharpe Ratio')
ax.set_title('SR: Offset=0 vs Averaged Over Offsets')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linestyle='--')

# 3. Daily bar PnL for daily rebalance
ax = axes[1, 0]
_, _, _, dpnl_1 = simulate_rebal(alpha_np, ret_np, rebal_every=1, fees_bps=0)
_, _, _, dpnl_288 = simulate_rebal(alpha_np, ret_np, rebal_every=288, fees_bps=0)
x = np.arange(len(dpnl_1))
w = 0.35
ax.bar(x - w/2, dpnl_1, w, label=f'5m rebal (SR={results[1][0]:+.1f})', alpha=0.7)
ax.bar(x + w/2, dpnl_288, w, label=f'Daily rebal (SR={results[288][0]:+.1f})', alpha=0.7)
ax.set_xlabel('Day')
ax.set_ylabel('Daily PnL ($)')
ax.set_title('Daily PnL: 5m vs Daily Rebalancing')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Return at each rebalance freq (should be monotonically decreasing at 0 fees)
ax = axes[1, 1]
ret_off0 = [results[rb][1] for rb in rbs]
ret_avg = [results[rb][5] for rb in rbs]
ax.plot(range(len(rbs)), [r*100 for r in ret_off0], 'bs--', label='Return (off=0)', markersize=8)
ax.plot(range(len(rbs)), [r*100 for r in ret_avg], 'ro-', label='Return (avg offsets)', markersize=8)
ax.set_xticks(range(len(rbs)))
ax.set_xticklabels([f"{rb*5}m" if rb < 288 else "1d" for rb in rbs], rotation=45)
ax.set_ylabel('Total Return (%)')
ax.set_title('Total Return vs Rebalance Freq\n(Should decrease at 0 fees)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diag_rebal_debug.png', dpi=150)
print(f"\nChart saved: diag_rebal_debug.png")
plt.close()
