"""
Comparison chart: BillionsQP vs Billions(ol=120) vs CorrSelect vs Equal-Weight
Plots cumulative net PnL (5bps fees) on a single chart.
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from eval_portfolio import (
    load_raw_alpha_signals, load_alpha_signals,
    strategy_billions_qp, strategy_billions,
    strategy_corr_select, strategy_proper_equal,
    simulate, VAL_START, VAL_END,
)

# ── Load data ─────────────────────────────────────────────────────────────────
raw_signals, returns_pct, close, universe = load_raw_alpha_signals()
signals, _, _, _ = load_alpha_signals()
print(f"Loaded {len(raw_signals)} alphas")

# ── Run strategies ─────────────────────────────────────────────────────────────
runs = []

print("Running BillionsQP...")
r, lbl = strategy_billions_qp(raw_signals, returns_pct, close, universe,
                               optim_lookback=120, risk_aversion=2.0, tc_penalty=0.5)
runs.append(("BillionsQP (SR=+4.62, TO=0.20)", r.daily_pnl, "#00C49A", 2.5))

print("Running Billions(ol=120)...")
r, lbl = strategy_billions(raw_signals, returns_pct, close, universe, optim_lookback=120)
runs.append(("Billions ol=120 (SR=+3.98, TO=0.24)", r.daily_pnl, "#5B8FFF", 1.8))

print("Running CorrSelect...")
r, lbl = strategy_corr_select(raw_signals, returns_pct, close, universe,
                               max_wt=0.03, max_corr=0.3, top_n=2)
runs.append(("CorrSelect (SR=+4.40, TO=0.25)", r.daily_pnl, "#FF9F40", 1.8))

print("Running ProperEqual...")
r, lbl = strategy_proper_equal(raw_signals, returns_pct, close, universe, max_wt=0.02)
runs.append(("Equal-Weight (SR=+3.47, TO=0.22)", r.daily_pnl, "#AAAAAA", 1.2))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor('#0F1117')
ax.set_facecolor('#161B22')

colors_grid = '#2A2F3A'
ax.grid(True, color=colors_grid, linewidth=0.6, zorder=0)
ax.spines[['top','right','left','bottom']].set_color(colors_grid)
ax.tick_params(colors='#B0B8C8', labelsize=10)
ax.xaxis.label.set_color('#B0B8C8')
ax.yaxis.label.set_color('#B0B8C8')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator())

BOOKSIZE = 20_000_000.0
for label, daily_pnl, color, lw in runs:
    cum = (daily_pnl * BOOKSIZE).cumsum()
    ax.plot(cum.index, cum.values, label=label, color=color, linewidth=lw, zorder=3)

ax.axhline(0, color='#444', linewidth=0.8, linestyle='--', zorder=2)
ax.set_title(
    f'Portfolio Comparison — Validation {VAL_START} → {VAL_END}  (5bps fees, $20M book)',
    color='white', fontsize=13, pad=14, fontweight='bold'
)
ax.set_ylabel('Cumulative Net PnL', color='#B0B8C8', fontsize=11)
ax.legend(facecolor='#1E242E', edgecolor='#3A424F', labelcolor='white',
          fontsize=10, loc='upper left', framealpha=0.9)

plt.tight_layout()
out = 'comparison_billions_qp.png'
plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
plt.close()
print(f"\nSaved to {out}")
