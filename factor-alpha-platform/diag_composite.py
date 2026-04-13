"""
Full diagnostic: Individual alphas + composites + equity curves + correlation matrix.
All run through eval_alpha_5m vectorized sim (ground truth).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import eval_alpha_5m as ea
from eval_portfolio_5m import load_alphas
from scipy.stats import spearmanr

OUT_DIR = "."

# Load alphas
alphas = load_alphas(universe="TOP100")
print(f"Loaded {len(alphas)} alphas\n")

# ===== Run on TRAIN split (Feb 2025 - Feb 2026, IS for alphas) =====
print("=" * 70)
print("SPLIT: TRAIN (Feb 2025 - Feb 2026) — IN-SAMPLE for alphas")
print("=" * 70)
matrices_train, universe_train = ea.load_data("train")
close_train = matrices_train["close"]
returns_train = close_train.pct_change()

alpha_raws = []
alpha_pnls = []
sharpes_train = []

for aid, expr, ic, sr_db in alphas:
    try:
        raw = ea.evaluate_expression(expr, matrices_train)
        if raw is None: continue
        alpha_raws.append((aid, raw))
        processed = ea.process_signal(raw, universe_df=universe_train, max_wt=ea.MAX_WEIGHT)
        result = ea.simulate(processed, returns_train, close_train, universe_train, fees_bps=0)
        sharpes_train.append((aid, result.sharpe, result.returns_ann, result.turnover))
        alpha_pnls.append((aid, result.cumulative_pnl))
        print(f"  #{aid:2d}: SR={result.sharpe:+7.2f}  Ret={result.returns_ann:+.1%}  TO={result.turnover:.3f}")
    except Exception as e:
        print(f"  #{aid:2d}: FAILED - {e}")

print(f"\n  Mean SR:   {np.mean([s[1] for s in sharpes_train]):+.2f}")
print(f"  Median SR: {np.median([s[1] for s in sharpes_train]):+.2f}")

# EW Composite on TRAIN
raw_sum = None; n = 0
for aid, raw in alpha_raws:
    r = raw.fillna(0)
    if raw_sum is None: raw_sum = r.copy()
    else: raw_sum = raw_sum.add(r, fill_value=0)
    n += 1
raw_avg = raw_sum / n
composite = ea.process_signal(raw_avg, universe_df=universe_train, max_wt=ea.MAX_WEIGHT)
result_ew_train = ea.simulate(composite, returns_train, close_train, universe_train, fees_bps=0)
print(f"\n  EW Composite: SR={result_ew_train.sharpe:+7.2f}  Ret={result_ew_train.returns_ann:+.1%}  TO={result_ew_train.turnover:.3f}")

# ===== Run on TRAINVAL (Dec 2025 - Mar 2026) =====
# Note: Dec-Jan is IS, Feb-Mar is OOS for alphas
print("\n" + "=" * 70)
print("SPLIT: TRAINVAL (Dec 2025 - Mar 2026) — partially OOS")
print("=" * 70)

tv_start, tv_end = "2025-12-01", "2026-03-01"
tv_matrices = {name: df.loc[tv_start:tv_end] for name, df in matrices_train.items()}
tv_universe = universe_train.loc[tv_start:tv_end]
tv_close = tv_matrices["close"]
tv_returns = tv_close.pct_change()

alpha_raws_tv = []
sharpes_tv = []
alpha_pnls_tv = []

for aid, expr, ic, sr_db in alphas:
    try:
        raw = ea.evaluate_expression(expr, tv_matrices)
        if raw is None: continue
        alpha_raws_tv.append((aid, raw))
        processed = ea.process_signal(raw, universe_df=tv_universe, max_wt=ea.MAX_WEIGHT)
        result = ea.simulate(processed, tv_returns, tv_close, tv_universe, fees_bps=0)
        sharpes_tv.append((aid, result.sharpe, result.returns_ann, result.turnover))
        alpha_pnls_tv.append((aid, result.cumulative_pnl))
        print(f"  #{aid:2d}: SR={result.sharpe:+7.2f}  Ret={result.returns_ann:+.1%}  TO={result.turnover:.3f}")
    except Exception as e:
        print(f"  #{aid:2d}: FAILED - {e}")

print(f"\n  Mean SR:   {np.mean([s[1] for s in sharpes_tv]):+.2f}")

# EW Composite on TRAINVAL  
raw_sum_tv = None; n = 0
for aid, raw in alpha_raws_tv:
    r = raw.fillna(0)
    if raw_sum_tv is None: raw_sum_tv = r.copy()
    else: raw_sum_tv = raw_sum_tv.add(r, fill_value=0)
    n += 1
raw_avg_tv = raw_sum_tv / n
comp_tv = ea.process_signal(raw_avg_tv, universe_df=tv_universe, max_wt=ea.MAX_WEIGHT)
result_ew_tv = ea.simulate(comp_tv, tv_returns, tv_close, tv_universe, fees_bps=0)
print(f"  EW Composite: SR={result_ew_tv.sharpe:+7.2f}  Ret={result_ew_tv.returns_ann:+.1%}  TO={result_ew_tv.turnover:.3f}")

# ===== CORRELATION MATRIX (full NxN) =====
print("\n" + "=" * 70)
print("FULL PAIRWISE CORRELATION MATRIX (on train)")
print("=" * 70)

# Use flattened processed signals for correlation
aid_list = [a[0] for a in alpha_raws]
n_alphas = len(alpha_raws)
corr_matrix = np.zeros((n_alphas, n_alphas))
for i in range(n_alphas):
    for j in range(i, n_alphas):
        ai = alpha_raws[i][1].values.flatten()
        aj = alpha_raws[j][1].values.flatten()
        valid = np.isfinite(ai) & np.isfinite(aj)
        if valid.sum() > 100:
            c, _ = spearmanr(ai[valid][:50000], aj[valid][:50000])
        else:
            c = 0
        corr_matrix[i, j] = c
        corr_matrix[j, i] = c

# Count truly independent signals (corr < 0.5)
n_independent = 0
used = set()
for i in range(n_alphas):
    if i in used: continue
    n_independent += 1
    for j in range(i+1, n_alphas):
        if abs(corr_matrix[i, j]) >= 0.5:
            used.add(j)

print(f"  Estimated independent signals: {n_independent}")
print(f"  Mean pairwise |corr|: {np.mean(np.abs(corr_matrix[np.triu_indices(n_alphas, k=1)])):.3f}")
print(f"  Max pairwise |corr|:  {np.max(np.abs(corr_matrix[np.triu_indices(n_alphas, k=1)])):.3f}")

# ===== PLOTS =====
# 1. Equity curves — individual alphas + composite on TRAIN
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax = axes[0, 0]
for aid, pnl in alpha_pnls:
    ax.plot(pnl.values, alpha=0.3, linewidth=0.5)
ax.plot(result_ew_train.cumulative_pnl.values, 'k-', linewidth=2, label=f'EW Composite (SR={result_ew_train.sharpe:.1f})')
ax.set_title(f'TRAIN: Individual Alphas + EW Composite (0 fees)\nMean SR={np.mean([s[1] for s in sharpes_train]):.1f}, Composite SR={result_ew_train.sharpe:.1f}')
ax.set_xlabel('Bar')
ax.set_ylabel('Cumulative PnL ($)')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Equity curves on TRAINVAL
ax = axes[0, 1]
for aid, pnl in alpha_pnls_tv:
    ax.plot(pnl.values, alpha=0.3, linewidth=0.5)
ax.plot(result_ew_tv.cumulative_pnl.values, 'k-', linewidth=2, label=f'EW Composite (SR={result_ew_tv.sharpe:.1f})')
ax.set_title(f'TRAINVAL (Dec-Mar): Alphas + Composite (0 fees)\nMean SR={np.mean([s[1] for s in sharpes_tv]):.1f}, Composite SR={result_ew_tv.sharpe:.1f}')
ax.set_xlabel('Bar')
ax.set_ylabel('Cumulative PnL ($)')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Correlation heatmap
ax = axes[1, 0]
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(n_alphas))
ax.set_yticks(range(n_alphas))
ax.set_xticklabels([f'#{a}' for a in aid_list], rotation=90, fontsize=7)
ax.set_yticklabels([f'#{a}' for a in id_list], fontsize=7) if 'id_list' in dir() else ax.set_yticklabels([f'#{a}' for a in aid_list], fontsize=7)
ax.set_title(f'Pairwise Spearman Correlation\nMean |corr|={np.mean(np.abs(corr_matrix[np.triu_indices(n_alphas, k=1)])):.2f}')
plt.colorbar(im, ax=ax)

# 4. Sharpe comparison: train vs trainval
ax = axes[1, 1]
train_srs = [s[1] for s in sharpes_train]
tv_srs = [s[1] for s in sharpes_tv]
x = np.arange(len(train_srs))
width = 0.35
ax.bar(x - width/2, train_srs, width, label='Train (Feb25-Feb26)', alpha=0.8)
ax.bar(x + width/2, tv_srs, width, label='TrainVal (Dec25-Mar26)', alpha=0.8)
ax.axhline(y=result_ew_train.sharpe, color='blue', linestyle='--', alpha=0.5, label=f'EW Train SR={result_ew_train.sharpe:.1f}')
ax.axhline(y=result_ew_tv.sharpe, color='orange', linestyle='--', alpha=0.5, label=f'EW TrainVal SR={result_ew_tv.sharpe:.1f}')
ax.set_xlabel('Alpha')
ax.set_ylabel('Sharpe Ratio')
ax.set_title('Individual vs Composite Sharpe')
ax.set_xticks(x)
ax.set_xticklabels([f'#{a[0]}' for a in sharpes_train], rotation=90, fontsize=7)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/diag_full_audit.png', dpi=150)
print(f"\n  Chart saved: {OUT_DIR}/diag_full_audit.png")

plt.close()
print("\nDone.")
