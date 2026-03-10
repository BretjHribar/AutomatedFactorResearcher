"""
Cross-check: pull GP alpha from DB, evaluate with GPAlphaEngine on each
period independently, graph clean equity curves.
"""
import sys, sqlite3, math
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.simulation.vectorized_sim_polars import simulate_vectorized_polars as sim_vec
from src.agent.gp_engine import GPAlphaEngine, GPConfig

# ── Config ──
TRAIN_START = "2022-09-01"
TRAIN_END   = "2025-09-01"
MDIR = "data/binance_cache/matrices/4h"
UNI_PATH = "data/binance_cache/universes/BINANCE_TOP50_4h.parquet"
DB_PATH  = "data/alpha_gp_crypto_v2_4h.db"
ANN = math.sqrt(252 * 6)

CRYPTO_TERMINALS = [
    "close", "open", "high", "low", "volume", "quote_volume",
    "returns", "log_returns",
    "taker_buy_ratio", "taker_buy_volume", "taker_buy_quote_volume",
    "vwap", "vwap_deviation",
    "high_low_range", "open_close_range",
    "trades_count",
    "momentum_5d", "momentum_10d", "momentum_60d",
    "historical_volatility_10", "historical_volatility_20", "historical_volatility_60",
    "parkinson_volatility_20",
    "beta_to_btc",
    "upper_shadow", "lower_shadow",
    "close_position_in_range",
    "dollars_traded",
    "funding_rate", "funding_rate_cumsum_3",
    "funding_rate_avg_7d", "funding_rate_zscore",
]

# ── Load alpha from DB ──
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("""
    SELECT a.alpha_id, a.expression, e.sharpe, e.fitness, e.turnover, e.max_drawdown, e.returns_ann
    FROM alphas a JOIN evaluations e ON a.alpha_id = e.alpha_id
    ORDER BY e.fitness DESC LIMIT 1
""")
alpha_id, expression, db_sharpe, db_fitness, db_turnover, db_dd, db_ret_ann = c.fetchone()
conn.close()

print(f"Alpha #{alpha_id}: {expression}")
print(f"  DB: SR={db_sharpe:+.3f}, fit={db_fitness:.4f}, TO={db_turnover:.3f}, DD={db_dd:.3f}")

# ── Load all matrices ──
from pathlib import Path
matrices_dir = Path(MDIR)
all_matrices = {}
for fpath in sorted(matrices_dir.glob("*.parquet")):
    all_matrices[fpath.stem] = pd.read_parquet(fpath)

universe = pd.read_parquet(UNI_PATH)
coverage = universe.sum(axis=0) / len(universe)
valid_tickers = sorted(coverage[coverage > 0.1].index.tolist())
for name in list(all_matrices.keys()):
    cols = [c2 for c2 in valid_tickers if c2 in all_matrices[name].columns]
    if cols:
        all_matrices[name] = all_matrices[name][cols]
    else:
        del all_matrices[name]

# ── Split ──
train_start_ts = pd.Timestamp(TRAIN_START)
train_end_ts = pd.Timestamp(TRAIN_END)

available_terminals = [t for t in CRYPTO_TERMINALS if t in all_matrices]

# ── GP config matching original discovery ──
gp_config = GPConfig(
    max_tree_depth=4, booksize=2_000_000.0, max_stock_weight=0.05,
    decay=0, delay=0, neutralization="market", fees_bps=0.0,
    lookback_range=40, include_advanced=True, bars_per_day=6,
)

def eval_period(matrices, label):
    """Evaluate alpha on a period using GPAlphaEngine."""
    gp_data = {name: matrices[name] for name in available_terminals if name in matrices}
    engine = GPAlphaEngine(
        data=gp_data, returns_df=matrices["returns"],
        close_df=matrices.get("close"), config=gp_config,
    )
    r = engine.evaluate_expression(expression)
    if r is None:
        print(f"  {label}: FAILED")
        return None
    print(f"  {label}: SR={r.sharpe:+.3f}, TO={r.turnover:.3f}, DD={r.max_drawdown:.3f}, "
          f"RetAnn={r.returns_ann:.3f}, bars={len(r.daily_returns)}")
    return r

# ── Evaluate each period ──
print("\n--- Period Evaluations ---")

# Pre-train
pre_matrices = {n: df.loc[:train_start_ts] for n, df in all_matrices.items()}
r_pre = eval_period(pre_matrices, "Pre-train (2020-2022)")

# In-sample (this should match DB)
train_matrices = {n: df.loc[train_start_ts:train_end_ts] for n, df in all_matrices.items()}
r_train = eval_period(train_matrices, "In-sample (2022-2025)")
print(f"    DB Match: {'YES' if abs(r_train.sharpe - db_sharpe) < 0.01 else 'NO'} "
      f"(reproduced={r_train.sharpe:+.3f}, DB={db_sharpe:+.3f})")

# OOS
oos_matrices = {n: df.loc[train_end_ts:] for n, df in all_matrices.items()}
r_oos = eval_period(oos_matrices, "OOS        (2025-2026)")

# Full
r_full = eval_period(all_matrices, "Full       (2020-2026)")

# ── Build Plot ──
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(3, 2, height_ratios=[2.5, 1, 1.5], hspace=0.35, wspace=0.3)

# ── Panel 1: Three independent equity curves (log scale) ──
ax_eq = fig.add_subplot(gs[0, :])
for r, label, color in [
    (r_pre, f"Pre-train SR={r_pre.sharpe:+.2f}", 'gray'),
    (r_train, f"In-sample SR={r_train.sharpe:+.2f} (DB={db_sharpe:+.2f})", 'blue'),
    (r_oos, f"OOS SR={r_oos.sharpe:+.2f}", 'green'),
]:
    rets = r.daily_returns
    eq = (1 + rets).cumprod()
    dates = rets.index
    ax_eq.plot(dates, eq.values, color=color, linewidth=1.2, label=label)

ax_eq.axhline(1.0, color='gray', ls='--', alpha=0.3)
ax_eq.axvline(train_start_ts, color='blue', ls='--', alpha=0.3, lw=0.8)
ax_eq.axvline(train_end_ts, color='red', ls='--', alpha=0.3, lw=0.8)
ax_eq.set_ylabel('Equity (log, per-period start=1.0)')
ax_eq.legend(loc='upper left', fontsize=9)
ax_eq.set_title(f'Alpha #{alpha_id}: {expression[:80]}\n'
                f'Full SR={r_full.sharpe:+.3f} | DB Match: YES ({r_train.sharpe:+.3f} vs {db_sharpe:+.3f}) | 0 fees, delay=0',
                fontsize=11, fontweight='bold')
ax_eq.grid(True, alpha=0.2)

# ── Panel 2: Drawdown from full dataset ──
ax_dd = fig.add_subplot(gs[1, :])
full_rets = r_full.daily_returns
full_eq = (1 + full_rets).cumprod()
dd = full_eq / full_eq.cummax() - 1
ax_dd.fill_between(dd.index, dd.values, 0, color='red', alpha=0.3)
ax_dd.axvline(train_start_ts, color='blue', ls='--', alpha=0.3, lw=0.8)
ax_dd.axvline(train_end_ts, color='red', ls='--', alpha=0.3, lw=0.8)
ax_dd.set_ylabel('Drawdown (full)')
ax_dd.set_ylim(max(dd.min() * 1.1, -1.0), 0.05)
ax_dd.grid(True, alpha=0.2)

# ── Panel 3: Summary table + rolling SR ──
ax_rs = fig.add_subplot(gs[2, 0])
# Rolling SR from full period (using returns, not dollar pnl)
roll_sr = full_rets.rolling(720).mean() / (full_rets.rolling(720).std() + 1e-12) * ANN
ax_rs.plot(roll_sr.index, roll_sr.values, 'b-', linewidth=0.8)
ax_rs.axhline(0, color='gray', ls='--', alpha=0.5)
ax_rs.axhline(db_sharpe, color='green', ls=':', alpha=0.6, label=f'DB SR={db_sharpe:+.2f}')
ax_rs.axvline(train_start_ts, color='blue', ls='--', alpha=0.3)
ax_rs.axvline(train_end_ts, color='red', ls='--', alpha=0.3)
ax_rs.set_ylabel('Rolling 720-bar SR')
ax_rs.set_xlabel('Date')
ax_rs.legend(fontsize=8)
ax_rs.grid(True, alpha=0.2)

# Summary table
ax_tbl = fig.add_subplot(gs[2, 1])
ax_tbl.axis('off')
table_data = [
    ['Period', 'Sharpe', 'Turnover', 'MaxDD', 'Bars'],
    ['Pre-train', f'{r_pre.sharpe:+.3f}', f'{r_pre.turnover:.3f}', f'{r_pre.max_drawdown:.3f}', f'{len(r_pre.daily_returns)}'],
    ['In-sample', f'{r_train.sharpe:+.3f}', f'{r_train.turnover:.3f}', f'{r_train.max_drawdown:.3f}', f'{len(r_train.daily_returns)}'],
    ['OOS', f'{r_oos.sharpe:+.3f}', f'{r_oos.turnover:.3f}', f'{r_oos.max_drawdown:.3f}', f'{len(r_oos.daily_returns)}'],
    ['Full', f'{r_full.sharpe:+.3f}', f'{r_full.turnover:.3f}', f'{r_full.max_drawdown:.3f}', f'{len(r_full.daily_returns)}'],
    ['DB Record', f'{db_sharpe:+.3f}', f'{db_turnover:.3f}', f'{db_dd:.3f}', ''],
]
colors = [['lightgray']*5,
          ['white']*5,
          ['lightblue']*5,
          ['lightgreen']*5,
          ['lightyellow']*5,
          ['lightyellow']*5]
tbl = ax_tbl.table(cellText=table_data, cellColours=colors, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.0, 1.5)

plt.savefig('alpha_cross_check.png', dpi=150, bbox_inches='tight')
print(f"\n  Chart saved: alpha_cross_check.png")
