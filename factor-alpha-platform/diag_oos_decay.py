"""
Diagnostic Part 2: True OOS validation + decay sweep for fee survival.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import eval_alpha_5m as ea
from eval_portfolio_5m import load_alphas
from src.simulation.vectorized_sim_polars import simulate_vectorized_polars

OUT_DIR = "."
alphas = load_alphas(universe="TOP100")
print(f"Loaded {len(alphas)} alphas\n")

# Load full train data (Feb 25 - Feb 26)
matrices, universe = ea.load_data("train")
close = matrices["close"]
returns = close.pct_change()

# ===== Build EW composite =====
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
composite = ea.process_signal(raw_avg, universe_df=universe, max_wt=ea.MAX_WEIGHT)

# ===== TRUE OOS: Val split (Feb-Mar 2026) =====
print("=" * 70)
print("TRUE OOS: Val split (Feb 2026 - Mar 2026)")
print("  Alpha train end: Feb 2026. Val is FULLY out-of-sample.")
print("=" * 70)

val_start, val_end = "2026-02-01", "2026-03-01"
val_matrices = {name: df.loc[val_start:val_end] for name, df in matrices.items()}
val_universe = universe.loc[val_start:val_end]
val_close = val_matrices["close"]
val_returns = val_close.pct_change()

# Individual alphas on val
val_sharpes = []
for aid, expr, ic, sr_db in alphas:
    try:
        raw = ea.evaluate_expression(expr, val_matrices)
        if raw is None: continue
        proc = ea.process_signal(raw, universe_df=val_universe, max_wt=ea.MAX_WEIGHT)
        result = simulate_vectorized_polars(proc, val_returns, val_close, val_universe,
                                             booksize=ea.BOOKSIZE, max_stock_weight=ea.MAX_WEIGHT,
                                             decay=0, delay=0, neutralization="market",
                                             fees_bps=0, bars_per_day=ea.BARS_PER_DAY)
        val_sharpes.append((aid, result.sharpe, result.returns_ann, result.turnover))
        print(f"  #{aid:2d}: SR={result.sharpe:+7.2f}  Ret={result.returns_ann:+.1%}  TO={result.turnover:.3f}")
    except Exception as e:
        print(f"  #{aid:2d}: FAILED - {e}")

print(f"\n  Val Mean SR: {np.mean([s[1] for s in val_sharpes]):+.2f}")

# EW composite on val
raw_sum_val = None; n = 0
for aid, expr, ic, sr_db in alphas:
    try:
        raw = ea.evaluate_expression(expr, val_matrices)
        if raw is None: continue
        r = raw.fillna(0)
        if raw_sum_val is None: raw_sum_val = r.copy()
        else: raw_sum_val = raw_sum_val.add(r, fill_value=0)
        n += 1
    except: pass
raw_avg_val = raw_sum_val / n
comp_val = ea.process_signal(raw_avg_val, universe_df=val_universe, max_wt=ea.MAX_WEIGHT)
r_val = simulate_vectorized_polars(comp_val, val_returns, val_close, val_universe,
                                    booksize=ea.BOOKSIZE, max_stock_weight=ea.MAX_WEIGHT,
                                    decay=0, delay=0, neutralization="market",
                                    fees_bps=0, bars_per_day=ea.BARS_PER_DAY)
print(f"  EW Composite (val, 0 fees): SR={r_val.sharpe:+.2f}  Ret={r_val.returns_ann:+.1%}  TO={r_val.turnover:.3f}")

# ===== DECAY SWEEP: Find fee-survivable config =====
print("\n" + "=" * 70)
print("DECAY + FEES SWEEP (on train, full year)")
print("=" * 70)

results_sweep = []
for decay in [0, 3, 6, 12, 24, 48]:
    for fees in [0, 3, 5, 7, 10]:
        r = simulate_vectorized_polars(composite, returns, close, universe,
                                        booksize=ea.BOOKSIZE, max_stock_weight=ea.MAX_WEIGHT,
                                        decay=decay, delay=0, neutralization="market",
                                        fees_bps=fees, bars_per_day=ea.BARS_PER_DAY)
        results_sweep.append((decay, fees, r.sharpe, r.returns_ann, r.turnover, r.total_pnl))
        if fees in (0, 7):
            print(f"  decay={decay:2d}  fees={fees:2d}bps  SR={r.sharpe:+7.2f}  Ret={r.returns_ann:+.1%}  TO={r.turnover:.3f}")

# ===== DECAY SWEEP on VAL =====
print("\n" + "=" * 70)
print("DECAY + FEES SWEEP (on val, TRUE OOS)")
print("=" * 70)

results_sweep_val = []
for decay in [0, 3, 6, 12, 24, 48]:
    for fees in [0, 7]:
        r = simulate_vectorized_polars(comp_val, val_returns, val_close, val_universe,
                                        booksize=ea.BOOKSIZE, max_stock_weight=ea.MAX_WEIGHT,
                                        decay=decay, delay=0, neutralization="market",
                                        fees_bps=fees, bars_per_day=ea.BARS_PER_DAY)
        results_sweep_val.append((decay, fees, r.sharpe, r.returns_ann, r.turnover, r.total_pnl))
        print(f"  decay={decay:2d}  fees={fees:2d}bps  SR={r.sharpe:+7.2f}  Ret={r.returns_ann:+.1%}  TO={r.turnover:.3f}")

# ===== PLOTS =====
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Val split equity curves
ax = axes[0, 0]
# Re-run best individual + composite to get curves
best_aid = max(val_sharpes, key=lambda x: x[1])[0]
best_expr = next(expr for aid, expr, _, _ in alphas if aid == best_aid)
raw = ea.evaluate_expression(best_expr, val_matrices)
proc = ea.process_signal(raw, universe_df=val_universe, max_wt=ea.MAX_WEIGHT)
r_best = simulate_vectorized_polars(proc, val_returns, val_close, val_universe,
                                     booksize=ea.BOOKSIZE, max_stock_weight=ea.MAX_WEIGHT,
                                     decay=0, delay=0, neutralization="market",
                                     fees_bps=0, bars_per_day=ea.BARS_PER_DAY)
ax.plot(r_best.cumulative_pnl.values, 'b-', alpha=0.5, label=f'Best #{best_aid} (SR={r_best.sharpe:.1f})')
ax.plot(r_val.cumulative_pnl.values, 'k-', linewidth=2, label=f'EW Composite (SR={r_val.sharpe:.1f})')
ax.set_title(f'TRUE OOS: Val (Feb-Mar 2026, 0 fees)\nComposite SR={r_val.sharpe:.1f}')
ax.set_xlabel('Bar')
ax.set_ylabel('Cumulative PnL ($)')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Decay sweep: SR vs decay at different fee levels
ax = axes[0, 1]
for fees in [0, 3, 5, 7, 10]:
    pts = [(d, sr) for d, f, sr, _, _, _ in results_sweep if f == fees]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], 'o-', label=f'{fees}bps', linewidth=2)
ax.set_xlabel('Decay (bars)')
ax.set_ylabel('Sharpe Ratio')
ax.set_title('Train: Sharpe vs Decay at Different Fee Levels')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linestyle='--')

# 3. Decay sweep on VAL
ax = axes[1, 0]
for fees in [0, 7]:
    pts = [(d, sr) for d, f, sr, _, _, _ in results_sweep_val if f == fees]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], 'o-', label=f'{fees}bps', linewidth=2, markersize=8)
ax.set_xlabel('Decay (bars)')
ax.set_ylabel('Sharpe Ratio')
ax.set_title('TRUE OOS (Val): Sharpe vs Decay')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linestyle='--')

# 4. Turnover vs Decay
ax = axes[1, 1]
pts_to = [(d, to) for d, f, _, _, to, _ in results_sweep if f == 0]
ax.plot([p[0] for p in pts_to], [p[1] for p in pts_to], 'rs-', linewidth=2, markersize=8, label='Turnover')
ax2 = ax.twinx()
pts_ret = [(d, ret) for d, f, _, ret, _, _ in results_sweep if f == 7]
ax2.plot([p[0] for p in pts_ret], [p[1] for p in pts_ret], 'b^-', linewidth=2, markersize=8, label='Ret (7bps)')
ax.set_xlabel('Decay (bars)')
ax.set_ylabel('Turnover (red)', color='red')
ax2.set_ylabel('Return Ann (7bps, blue)', color='blue')
ax.set_title('Turnover Reduction vs Return at 7bps')
ax.legend(loc='upper right')
ax2.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/diag_oos_decay.png', dpi=150)
print(f"\n  Chart saved: {OUT_DIR}/diag_oos_decay.png")
plt.close()
print("\nDone.")
