"""
EWMA signal smoothing sweep: the standard turnover reduction approach.
Apply EMA to the composite signal, then rebalance every bar.
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

def simulate_ewma(alpha_df, ret_df, ewma_span=1, fees_bps=0, 
                  booksize=BOOKSIZE, max_wt=MAX_WT):
    """
    Sim: apply EWMA to signal, rebalance every bar.
    ewma_span=1 means no smoothing.
    """
    common = alpha_df.columns.intersection(ret_df.columns).tolist()
    idx = alpha_df.index.intersection(ret_df.index)
    
    alpha_np = alpha_df.loc[idx, common].values.astype(np.float64).copy()
    ret_np = ret_df.loc[idx, common].values.astype(np.float64)
    n_bars, n_tickers = alpha_np.shape
    
    alpha_np[~np.isfinite(alpha_np)] = 0
    ret_np[~np.isfinite(ret_np)] = 0
    
    # Apply EWMA to signal
    if ewma_span > 1:
        ema_alpha = 2.0 / (ewma_span + 1.0)
        smoothed = np.zeros_like(alpha_np)
        smoothed[0] = alpha_np[0]
        for t in range(1, n_bars):
            smoothed[t] = ema_alpha * alpha_np[t] + (1 - ema_alpha) * smoothed[t-1]
        alpha_np = smoothed
    
    fee_rate = fees_bps / 10_000.0
    positions = np.zeros(n_tickers)
    pnl_arr = np.zeros(n_bars)
    turnover_arr = np.zeros(n_bars)
    
    for t in range(1, n_bars):
        # PnL from holding
        pnl_arr[t] = np.sum(positions * ret_np[t]) * booksize
        
        # New target
        sig = alpha_np[t].copy()
        abs_sum = np.abs(sig).sum()
        if abs_sum > 1e-10:
            new_pos = np.clip(sig / abs_sum, -max_wt, max_wt)
            abs_sum2 = np.abs(new_pos).sum()
            if abs_sum2 > 1e-10: new_pos /= abs_sum2
        else:
            new_pos = positions.copy()
        
        turnover = np.abs(new_pos - positions).sum()
        turnover_arr[t] = turnover
        pnl_arr[t] -= turnover * fee_rate * booksize
        positions = new_pos
    
    cum_pnl = np.cumsum(pnl_arr)
    daily_pnl = [np.sum(pnl_arr[d:d+BARS_PER_DAY]) for d in range(0, n_bars, BARS_PER_DAY)]
    daily_pnl = np.array(daily_pnl)
    
    sharpe = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(365) if np.std(daily_pnl) > 0 else 0
    total_ret = cum_pnl[-1] / booksize
    avg_to = np.mean(turnover_arr[1:])
    gross_pnl = np.sum(pnl_arr) + np.sum(turnover_arr * fee_rate * booksize)
    fee_pnl = np.sum(turnover_arr * fee_rate * booksize)
    
    return {
        'sharpe': sharpe, 'total_return': total_ret, 'cum_pnl': cum_pnl,
        'avg_turnover': avg_to, 'daily_pnl': daily_pnl,
        'gross_return': gross_pnl / booksize, 'fee_cost': fee_pnl / booksize,
    }


# Load data
alphas = ep.load_alphas(universe="TOP100")

# Run on BOTH train and val
for split_name in ["train", "val"]:
    matrices, universe = ep.load_data(split_name)
    returns = matrices["returns"]
    
    # Build EW composite
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
    
    print(f"\n{'='*80}")
    print(f"EWMA SIGNAL SMOOTHING SWEEP — {split_name.upper()} ({len(composite)} bars)")
    print(f"{'='*80}")
    
    # Spans from no smoothing to 2-day
    spans = [1, 3, 6, 12, 24, 36, 48, 72, 96, 144, 288, 576]
    
    print(f"{'Span':>6} {'Half-life':>10} {'TO':>8} {'SR(0bp)':>8} {'SR(3bp)':>8} {'SR(5bp)':>8} {'SR(7bp)':>8} {'Ret(0)':>10} {'Ret(7)':>10} {'Gross':>10} {'Fees':>10}")
    
    results = []
    for sp in spans:
        hl = f"{sp*5/60:.1f}h" if sp < 288 else f"{sp*5/60/24:.1f}d"
        r0 = simulate_ewma(composite, returns, ewma_span=sp, fees_bps=0)
        r3 = simulate_ewma(composite, returns, ewma_span=sp, fees_bps=3)
        r5 = simulate_ewma(composite, returns, ewma_span=sp, fees_bps=5)
        r7 = simulate_ewma(composite, returns, ewma_span=sp, fees_bps=7)
        results.append((sp, r0, r3, r5, r7))
        print(f"{sp:>6} {hl:>10} {r0['avg_turnover']:8.4f} {r0['sharpe']:+8.2f} {r3['sharpe']:+8.2f} {r5['sharpe']:+8.2f} {r7['sharpe']:+8.2f} {r0['total_return']:+10.1%} {r7['total_return']:+10.1%} {r7['gross_return']:+10.1%} {r7['fee_cost']:+10.1%}")
    
    # Find breakeven span for 7bps
    for sp, r0, r3, r5, r7 in results:
        if r7['sharpe'] > 0:
            print(f"\n  >>> BREAKEVEN at span={sp} ({sp*5/60:.1f}h): SR={r7['sharpe']:+.2f}, Return={r7['total_return']:+.1%}")
            break
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. SR vs EWMA span at different fee levels
    ax = axes[0, 0]
    for fee, color, label in [(0, 'blue', '0 bps'), (3, 'green', '3 bps'), (5, 'orange', '5 bps'), (7, 'red', '7 bps')]:
        idx_fee = {0: 1, 3: 2, 5: 3, 7: 4}[fee]
        srs = [r[idx_fee]['sharpe'] for r in results]
        ax.semilogx([r[0] for r in results], srs, 'o-', color=color, linewidth=2, markersize=8, label=label)
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xlabel('EWMA Span (bars)')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title(f'{split_name.upper()}: Sharpe vs EWMA Signal Smoothing')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Turnover vs span
    ax = axes[0, 1]
    tos = [r[1]['avg_turnover'] for r in results]
    ax.semilogx([r[0] for r in results], tos, 'rs-', linewidth=2, markersize=8)
    ax.set_xlabel('EWMA Span (bars)')
    ax.set_ylabel('Avg Turnover per Bar')
    ax.set_title(f'{split_name.upper()}: Turnover Reduction')
    ax.grid(True, alpha=0.3)
    
    # 3. Equity curves at key spans (7 bps)
    ax = axes[1, 0]
    for sp, r0, r3, r5, r7 in results:
        if sp in [1, 12, 36, 72, 144, 288]:
            hl = f"{sp*5/60:.0f}h" if sp < 288 else "1d"
            ax.plot(r7['cum_pnl'], label=f"EMA {hl} (SR={r7['sharpe']:+.1f})", linewidth=1.5)
    ax.set_title(f'{split_name.upper()}: Equity Curves at 7 bps')
    ax.set_xlabel('Bar')
    ax.set_ylabel('Cumulative PnL ($)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. Gross vs Fee cost
    ax = axes[1, 1]
    gross = [r[4]['gross_return']*100 for r in results]
    fees = [r[4]['fee_cost']*100 for r in results]
    net = [r[4]['total_return']*100 for r in results]
    x = range(len(results))
    ax.bar(x, gross, color='green', alpha=0.7, label='Gross')
    ax.bar(x, [-f for f in fees], color='red', alpha=0.7, label='Fees')
    ax.plot(x, net, 'ko-', linewidth=2, markersize=6, label='Net')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{sp*5/60:.0f}h" if sp < 288 else "1d" for sp, *_ in results], rotation=45)
    ax.set_ylabel('Return (%)')
    ax.set_title(f'{split_name.upper()}: Gross vs Fees at 7 bps')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'diag_ewma_sweep_{split_name}.png', dpi=150)
    print(f"  Chart saved: diag_ewma_sweep_{split_name}.png")
    plt.close()

print("\nDone.")
