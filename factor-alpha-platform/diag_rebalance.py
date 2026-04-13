"""
Diagnostic Part 3b: Rebalance frequency sweep using PORTFOLIO data loader.
This correctly handles train/val/test splits with data through Mar 2026.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import eval_alpha_5m as ea
import eval_portfolio_5m as ep

OUT_DIR = "."
BOOKSIZE = ep.BOOKSIZE
MAX_WT = ea.MAX_WEIGHT
BARS_PER_DAY = ep.BARS_PER_DAY

def simulate_rebalance_every_n(alpha_df, returns_df, rebal_every=1, fees_bps=0, 
                                booksize=BOOKSIZE, max_wt=MAX_WT):
    """Simple sim: rebalance every N bars, hold between rebalances."""
    common = alpha_df.columns.intersection(returns_df.columns).tolist()
    idx = alpha_df.index.intersection(returns_df.index)
    
    alpha_np = alpha_df.loc[idx, common].values.astype(np.float64).copy()
    ret_np = returns_df.loc[idx, common].values.astype(np.float64)
    n_bars, n_tickers = alpha_np.shape
    
    alpha_np[~np.isfinite(alpha_np)] = 0
    ret_np[~np.isfinite(ret_np)] = 0
    
    fee_rate = fees_bps / 10_000.0
    positions = np.zeros(n_tickers)
    pnl_arr = np.zeros(n_bars)
    
    for t in range(1, n_bars):
        pnl_arr[t] = np.sum(positions * ret_np[t]) * booksize
        
        if t % rebal_every == 0:
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
    
    sharpe = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(365) if len(daily_pnl) > 1 and np.std(daily_pnl) > 0 else 0
    total_ret = cum_pnl[-1] / booksize
    
    return {'sharpe': sharpe, 'total_return': total_ret, 'cum_pnl': cum_pnl, 'n_bars': n_bars, 'daily_pnl': daily_pnl}


# Load data using PORTFOLIO data loader (loads through Mar 2026)
print("Loading data via portfolio loader (includes Mar 2026)...", flush=True)
for split in ["train", "trainval", "val"]:
    matrices_s, universe_s = ep.load_data(split)
    close = matrices_s["close"]
    returns = matrices_s["returns"]
    print(f"  {split}: {close.index[0]} to {close.index[-1]}, {len(close)} bars, {len(close.columns)} tickers")

# Build composites per split
def build_ew_composite(matrices, universe, alphas):
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
    return ea.process_signal(raw_sum / n, universe_df=universe, max_wt=MAX_WT), n

alphas = ep.load_alphas(universe="TOP100")

# Run on each split
results_all = {}
for split_name in ["train", "trainval", "val"]:
    matrices_s, universe_s = ep.load_data(split_name)
    returns_s = matrices_s["returns"]
    comp_s, n_a = build_ew_composite(matrices_s, universe_s, alphas)
    print(f"\n{'='*70}")
    print(f"SPLIT: {split_name.upper()} — {len(returns_s)} bars, {n_a} alphas")
    print(f"{'='*70}")
    
    results_split = []
    rebal_bars = [1, 3, 6, 12, 24, 36, 72, 144, 288]
    
    print(f"{'Rebal':>8} {'Period':>8} {'SR(0bp)':>8} {'SR(3bp)':>8} {'SR(7bp)':>8} {'Ret(0)':>10} {'Ret(7)':>10}")
    for rb in rebal_bars:
        period = f"{rb*5}m" if rb < 288 else "1day"
        r0 = simulate_rebalance_every_n(comp_s, returns_s, rebal_every=rb, fees_bps=0)
        r3 = simulate_rebalance_every_n(comp_s, returns_s, rebal_every=rb, fees_bps=3)
        r7 = simulate_rebalance_every_n(comp_s, returns_s, rebal_every=rb, fees_bps=7)
        results_split.append((rb, r0, r3, r7))
        print(f"{rb:>8} {period:>8} {r0['sharpe']:+8.2f} {r3['sharpe']:+8.2f} {r7['sharpe']:+8.2f} {r0['total_return']:+10.1%} {r7['total_return']:+10.1%}")
    
    results_all[split_name] = results_split

# ===== PLOTS =====
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
rebal_bars = [1, 3, 6, 12, 24, 36, 72, 144, 288]
labels = [f"{rb*5}m" if rb < 288 else "1d" for rb in rebal_bars]

colors = {'train': 'blue', 'trainval': 'green', 'val': 'red'}

# 1. SR(0bps) vs rebalance freq, all splits
ax = axes[0, 0]
for split_name in ["train", "trainval", "val"]:
    srs = [r[1]['sharpe'] for r in results_all[split_name]]
    ax.semilogx(rebal_bars, srs, 'o-', color=colors[split_name], linewidth=2, markersize=8, label=f'{split_name} (0 bps)')
ax.axhline(y=0, color='gray', linestyle='--')
ax.set_xlabel('Rebalance every N bars')
ax.set_ylabel('Sharpe Ratio')
ax.set_title('0 Fees: Sharpe vs Rebalance Frequency')
ax.set_xticks(rebal_bars)
ax.set_xticklabels(labels, rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)

# 2. SR(7bps) vs rebalance freq, all splits
ax = axes[0, 1]
for split_name in ["train", "trainval", "val"]:
    srs = [r[3]['sharpe'] for r in results_all[split_name]]
    ax.semilogx(rebal_bars, srs, 'o-', color=colors[split_name], linewidth=2, markersize=8, label=f'{split_name} (7 bps)')
ax.axhline(y=0, color='gray', linestyle='--')
ax.set_xlabel('Rebalance every N bars')
ax.set_ylabel('Sharpe Ratio')
ax.set_title('7 bps Fees: Sharpe vs Rebalance Frequency')
ax.set_xticks(rebal_bars)
ax.set_xticklabels(labels, rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Equity curves: train 0 bps, best rebalance freqs
ax = axes[1, 0]
for rb, r0, r3, r7 in results_all["train"]:
    if rb in [1, 12, 36, 72, 288]:
        lbl = f"{rb*5}m" if rb < 288 else "1d"
        ax.plot(r0['cum_pnl'], label=f"Rebal {lbl} (SR={r0['sharpe']:+.1f})", linewidth=1.5)
ax.set_title('Train: Equity Curves (0 fees)')
ax.set_xlabel('Bar')
ax.set_ylabel('Cumulative PnL ($)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 4. Equity curves: val 0 bps + 7 bps
ax = axes[1, 1]
for rb, r0, r3, r7 in results_all["val"]:
    if rb in [1, 12, 72]:
        lbl = f"{rb*5}m" if rb < 288 else "1d"
        ax.plot(r0['cum_pnl'], label=f"Rebal {lbl} 0bp (SR={r0['sharpe']:+.1f})", linewidth=2)
        ax.plot(r7['cum_pnl'], '--', label=f"Rebal {lbl} 7bp (SR={r7['sharpe']:+.1f})", linewidth=1)
ax.set_title('Val (TRUE OOS): Equity Curves')
ax.set_xlabel('Bar')
ax.set_ylabel('Cumulative PnL ($)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/diag_rebalance_sweep.png', dpi=150)
print(f"\n  Chart saved: {OUT_DIR}/diag_rebalance_sweep.png")
plt.close()
print("\nDone.")
