"""
Deep signal analysis to inform sparse trading approaches.
What does the SIGNAL ITSELF look like over time?
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

# Load full year data
matrices, universe = ea.load_data("train")
close = matrices["close"]
returns = close.pct_change()

alphas = ep.load_alphas(universe="TOP100")
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
print(f"Composite: {composite.shape}")

common = composite.columns.intersection(returns.columns).tolist()
idx = composite.index.intersection(returns.index)
sig = composite.loc[idx, common].values.astype(np.float64)
ret = returns.loc[idx, common].values.astype(np.float64)
sig = np.nan_to_num(sig, nan=0)
ret = np.nan_to_num(ret, nan=0)
n_bars, n_tickers = sig.shape

# 1. Signal autocorrelation at different lags
print("\n=== SIGNAL AUTOCORRELATION ===")
acorrs = []
for lag in [1, 3, 6, 12, 24, 36, 72, 144, 288]:
    # Per-ticker autocorrelation, then average
    per_tick_corrs = []
    for j in range(n_tickers):
        s = sig[:, j]
        if np.std(s) > 0:
            c = np.corrcoef(s[:-lag], s[lag:])[0, 1]
            if np.isfinite(c):
                per_tick_corrs.append(c)
    avg_c = np.mean(per_tick_corrs) if per_tick_corrs else 0
    acorrs.append((lag, avg_c))
    print(f"  Lag={lag:>4} ({lag*5:>5}min): autocorr={avg_c:.4f}")

# 2. Signal persistence: how often does top/bottom rank persist?
print("\n=== RANK PERSISTENCE ===")
for k in [5, 10, 20]:
    persist_counts = []
    for lag in [1, 6, 12, 36, 72, 288]:
        # What fraction of top-K at time t is still top-K at t+lag?
        overlaps = []
        for t in range(0, n_bars - lag, max(lag, 12)):
            top_now = set(np.argsort(sig[t])[-k:])
            top_later = set(np.argsort(sig[t+lag])[-k:])
            overlap = len(top_now & top_later) / k
            overlaps.append(overlap)
        persist_counts.append((lag, np.mean(overlaps)))
    persist_str = "  ".join([f"lag={lag}:{ov:.2f}" for lag, ov in persist_counts])
    print(f"  Top {k}: {persist_str}")

# 3. Signal vs future return at different horizons  
print("\n=== SIGNAL PREDICTIVE POWER BY HORIZON ===")
for horizon in [1, 3, 6, 12, 24, 72, 288]:
    # IC at each bar, then average
    ics = []
    for t in range(0, n_bars - horizon, max(horizon, 12)):
        fwd_ret = ret[t+1:t+1+horizon].sum(axis=0)  # cumulative forward return
        s = sig[t]
        mask = (np.abs(s) > 1e-10) & np.isfinite(fwd_ret)
        if mask.sum() > 10:
            ic = np.corrcoef(s[mask], fwd_ret[mask])[0, 1]
            if np.isfinite(ic):
                ics.append(ic)
    mean_ic = np.mean(ics)
    print(f"  Horizon={horizon:>4} ({horizon*5:>5}min): IC={mean_ic:+.5f} (n={len(ics)})")

# 4. Distribution of per-ticker signal changes (what fraction of turnover is "noise"?)
print("\n=== SIGNAL CHANGE DISTRIBUTION ===")
sig_changes = np.abs(np.diff(sig, axis=0))
pcts = [50, 75, 90, 95, 99]
for p in pcts:
    v = np.percentile(sig_changes[sig_changes > 0], p)
    print(f"  P{p}: signal change = {v:.5f}")

median_change = np.median(sig_changes[sig_changes > 0])
print(f"  Median per-bar signal change: {median_change:.5f}")
print(f"  Mean per-bar abs signal: {np.mean(np.abs(sig[sig != 0])):.5f}")
print(f"  Change/Signal ratio: {median_change / np.mean(np.abs(sig[sig != 0])):.3f}")

# 5. Cross-sectional signal dispersion over time
cs_dispersion = np.std(sig, axis=1)
print(f"\n=== CROSS-SECTIONAL DISPERSION ===")
print(f"  Mean: {np.mean(cs_dispersion):.5f}")
print(f"  Std:  {np.std(cs_dispersion):.5f}")
print(f"  CV:   {np.std(cs_dispersion)/np.mean(cs_dispersion):.3f}")

# 6. Extreme signal analysis: what if we only trade when signal > X sigma?
print(f"\n=== EXTREME SIGNAL ANALYSIS (per-asset ts z-score) ===")
# Compute rolling z-score of each asset's signal
lookback = 288  # 1 day
ts_zscore = np.zeros_like(sig)
for t in range(lookback, n_bars):
    window = sig[t-lookback:t]
    mu = np.mean(window, axis=0)
    sd = np.std(window, axis=0)
    sd[sd < 1e-8] = 1e-8
    ts_zscore[t] = (sig[t] - mu) / sd

for threshold in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    mask = np.abs(ts_zscore) > threshold
    frac_active = mask.mean()
    # IC conditional on being above threshold
    ics = []
    for t in range(lookback, n_bars - 1):
        m = mask[t]
        if m.sum() > 3:
            ic = np.corrcoef(sig[t, m], ret[t+1, m])[0, 1]
            if np.isfinite(ic):
                ics.append(ic)
    mean_ic = np.mean(ics) if ics else 0
    print(f"  |z|>{threshold:.1f}: {frac_active:6.1%} active, IC={mean_ic:+.5f}")

# Plot
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Signal autocorrelation
ax = axes[0, 0]
lags, acvals = zip(*acorrs)
ax.plot(lags, acvals, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('Lag (bars)')
ax.set_ylabel('Autocorrelation')
ax.set_title('Signal Autocorrelation')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linestyle='--')

# 2. IC by horizon
ax = axes[0, 1]
horizons = [1, 3, 6, 12, 24, 72, 288]
ics_by_h = []
for h in horizons:
    ic_list = []
    for t in range(0, n_bars - h, max(h, 12)):
        fwd = ret[t+1:t+1+h].sum(axis=0)
        s = sig[t]
        m = (np.abs(s) > 1e-10) & np.isfinite(fwd)
        if m.sum() > 10:
            ic = np.corrcoef(s[m], fwd[m])[0, 1]
            if np.isfinite(ic): ic_list.append(ic)
    ics_by_h.append(np.mean(ic_list))
ax.plot(horizons, ics_by_h, 'rs-', linewidth=2, markersize=8)
ax.set_xlabel('Forward Horizon (bars)')
ax.set_ylabel('Mean IC')
ax.set_title('Predictive Power by Horizon')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)

# 3. CS dispersion over time
ax = axes[0, 2]
# Subsample for plotting
step = 288
ax.plot(range(0, len(cs_dispersion), step), cs_dispersion[::step], 'g-', linewidth=1)
ax.set_xlabel('Day')
ax.set_ylabel('CS Dispersion (std of signal)')
ax.set_title('Signal Dispersion Over Time')
ax.grid(True, alpha=0.3)

# 4. Signal change distribution
ax = axes[1, 0]
changes = sig_changes.ravel()
changes = changes[changes > 0]
ax.hist(changes, bins=100, alpha=0.7, color='steelblue', density=True)
ax.axvline(np.median(changes), color='red', linestyle='--', label=f'Median={np.median(changes):.4f}')
ax.set_xlabel('Per-bar signal change (abs)')
ax.set_ylabel('Density')
ax.set_title('Distribution of Signal Changes')
ax.set_xlim(0, np.percentile(changes, 99))
ax.legend()

# 5. ts z-score distribution  
ax = axes[1, 1]
zs = ts_zscore[lookback:].ravel()
zs = zs[np.isfinite(zs) & (zs != 0)]
ax.hist(zs, bins=100, alpha=0.7, color='purple', density=True)
ax.set_xlabel('Time-series Z-score')
ax.set_ylabel('Density')
ax.set_title('Signal TS Z-score Distribution')
ax.set_xlim(-5, 5)

# 6. IC conditional on |z| threshold
ax = axes[1, 2]
thresholds = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
cond_ics = []
frac_actives = []
for threshold in thresholds:
    m = np.abs(ts_zscore) > threshold
    frac_actives.append(m.mean())
    ics = []
    for t in range(lookback, n_bars - 1):
        mm = m[t]
        if mm.sum() > 3:
            ic = np.corrcoef(sig[t, mm], ret[t+1, mm])[0, 1]
            if np.isfinite(ic): ics.append(ic)
    cond_ics.append(np.mean(ics) if ics else 0)
ax2 = ax.twinx()
ax.bar(thresholds, [f*100 for f in frac_actives], width=0.3, alpha=0.3, color='blue', label='% active')
ax2.plot(thresholds, cond_ics, 'ro-', linewidth=2, markersize=8, label='Cond IC')
ax.set_xlabel('|TS Z-score| Threshold')
ax.set_ylabel('% Active Positions', color='blue')
ax2.set_ylabel('Conditional IC', color='red')
ax.set_title('Conditional IC vs Activity')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('diag_signal_analysis.png', dpi=150)
print(f"\nChart saved: diag_signal_analysis.png")
plt.close()
print("Done.")
