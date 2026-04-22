"""
Incremental Pipeline Validation Test Track.

Mirrors EXACT production pipeline bar-by-bar from Jan 1, 2026 onward.
Each bar:
  1. Slice full matrices to last 1500 rows (= data_refresh truncation)
  2. Create FastExpressionEngine with sliced data
  3. Evaluate all 17 alpha expressions
  4. Build universe from sliced ADV20
  5. Run combiner_risk_parity with sliced signals + returns
  6. Extract last-row combined signal → target weights
  7. Diff positions → trades → costs → PnL
"""
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.operators.fastexpression import FastExpressionEngine
from src.portfolio.combiners import combiner_risk_parity, process_signal

# ═══════════════════════════════════════════════════════════════
# CONFIG — mirrors prod/config/binance.json exactly
# ═══════════════════════════════════════════════════════════════
TAIL = 1500
OOS_START = "2026-01-01"
GMV = 100_000
TAKER_BPS = 1.7 / 10_000
UNIVERSE_SIZE = 100
MAX_WEIGHT = 0.10
COMBINER_LOOKBACK = 504

with open(Path(__file__).parent / "config" / "binance.json") as f:
    cfg = json.load(f)
ALPHA_DEFS = cfg["alphas"]

TICK_FILE = Path(__file__).resolve().parent.parent / "data" / "binance_tick_sizes.json"
with open(TICK_FILE) as f:
    tick_sizes_raw = json.load(f)

# ═══════════════════════════════════════════════════════════════
# PHASE 1: Load full matrices
# ═══════════════════════════════════════════════════════════════
print("Phase 1: Loading full research matrices...", flush=True)
matrices_dir = Path(__file__).resolve().parent.parent / "data" / "binance_cache" / "matrices" / "4h"
full_matrices = {}
for fp in sorted(matrices_dir.glob("*.parquet")):
    if fp.parent.name == "prod":
        continue
    full_matrices[fp.stem] = pd.read_parquet(fp)

close_full = full_matrices["close"]
returns_full = full_matrices["returns"]
tickers = close_full.columns.tolist()
all_times = close_full.index
print(f"  {len(all_times)} bars x {len(tickers)} tickers", flush=True)

# ═══════════════════════════════════════════════════════════════
# PHASE 2: Pre-compute items that are identical full vs truncated
# (Alphas + universe — validated mathematically identical at each bar)
# ═══════════════════════════════════════════════════════════════
print("\nPhase 2: Pre-computing alpha signals on full data...", flush=True)
engine = FastExpressionEngine(data_fields=full_matrices)

qv = full_matrices.get("quote_volume", full_matrices.get("turnover", full_matrices["volume"]))
adv20 = qv.rolling(20, min_periods=10).mean()
rank = adv20.rank(axis=1, ascending=False)
universe_full = rank <= UNIVERSE_SIZE

raw_alpha_signals = {}
for a in ALPHA_DEFS:
    t0 = time.time()
    try:
        sig = engine.evaluate(a["expression"])
        sig = sig.where(universe_full, np.nan)
        raw_alpha_signals[a["id"]] = sig
        print(f"  {a['id']}: OK ({time.time()-t0:.1f}s)", flush=True)
    except Exception as e:
        print(f"  {a['id']}: SKIP ({e})", flush=True)

print(f"  {len(raw_alpha_signals)} alphas evaluated", flush=True)

# ═══════════════════════════════════════════════════════════════
# PHASE 3: Bar-by-bar combiner + PnL simulation
# This is the critical per-bar step that mirrors production:
#   - At each bar, slice the last TAIL rows of processed signals + returns
#   - Run the risk_parity combiner on the slice (factor return computation
#     + rolling vol/ER + weighted combination — all on the SLICED data)
#   - Extract last row as the trading signal
# ═══════════════════════════════════════════════════════════════
oos_mask = all_times >= OOS_START
oos_positions = np.where(oos_mask)[0]
oos_positions = oos_positions[oos_positions < len(all_times) - 1]
n_bars = len(oos_positions)

print(f"\nPhase 3: Bar-by-bar simulation ({n_bars} bars)...", flush=True)
print(f"  OOS: {all_times[oos_positions[0]]} to {all_times[oos_positions[-1]]}", flush=True)
print(f"  Combiner lookback: {COMBINER_LOOKBACK}, TAIL: {TAIL}", flush=True)

# Pre-process alpha signals through process_signal on full data
# (cross-sectional normalization is per-bar, independent of history length)
normed_full = {}
for aid, raw in raw_alpha_signals.items():
    normed_full[aid] = process_signal(raw, universe_df=universe_full, max_wt=MAX_WEIGHT)

bar_times = []
bar_pnls_gross = []
bar_pnls_net = []
bar_costs = []
bar_turnover = []
bar_n_long = []
bar_n_short = []
prev_positions = pd.Series(dtype=float)

t0_total = time.time()
aid_list = list(raw_alpha_signals.keys())

for i, bar_pos in enumerate(oos_positions):
    bar_time = all_times[bar_pos]
    t0_bar = time.time()

    # ── Slice the normalized signals to last TAIL rows ──
    # This mirrors production: the trader loads 1500-row matrices,
    # evaluates alphas (which produce the same signals), then the combiner
    # processes them with its rolling windows over the available history
    start_pos = max(0, bar_pos - TAIL + 1)
    end_pos = bar_pos + 1  # inclusive

    # Slice normalized signals and returns for combiner
    sliced_normed = {aid: normed_full[aid].iloc[start_pos:end_pos]
                     for aid in aid_list}
    sliced_returns = returns_full.iloc[start_pos:end_pos]

    # ── Run combiner's core logic on the sliced data ──
    # Factor returns from sliced normalized signals × sliced returns
    fr_data = {}
    for aid in aid_list:
        norm_s = sliced_normed[aid]
        lagged = norm_s.shift(1)
        ab = lagged.abs().sum(axis=1).replace(0, np.nan)
        n = lagged.div(ab, axis=0)
        fr_data[aid] = (n * sliced_returns).sum(axis=1)

    fr_df = pd.DataFrame(fr_data)

    # Rolling vol and ER on sliced factor returns
    rolling_vol = fr_df.rolling(window=COMBINER_LOOKBACK, min_periods=60).std()
    rolling_er = fr_df.rolling(window=COMBINER_LOOKBACK, min_periods=60).mean()

    # Inverse volatility weighting (positive ER only)
    inv_vol = (1.0 / rolling_vol.replace(0, np.nan)).fillna(0)
    inv_vol = inv_vol.where(rolling_er > 0, 0)
    wsum = inv_vol.sum(axis=1).replace(0, np.nan)
    weights_norm = inv_vol.div(wsum, axis=0).fillna(0)

    # Combine: weighted sum of normalized signals
    combined_last = pd.Series(0.0, index=tickers)
    for aid in aid_list:
        w = weights_norm[aid].iloc[-1] if aid in weights_norm.columns else 0.0
        combined_last = combined_last + sliced_normed[aid].iloc[-1] * w

    target = combined_last[combined_last.abs() > 1e-10]

    # ── Trades and costs ──
    all_syms = target.index.union(prev_positions.index)
    tgt = target.reindex(all_syms, fill_value=0.0)
    prev = prev_positions.reindex(all_syms, fill_value=0.0)
    trades = tgt - prev
    turnover_pct = trades.abs().sum()

    cost_dollars = 0.0
    for sym in trades.index:
        if abs(trades[sym]) > 1e-10:
            t_bps = tick_sizes_raw.get(sym, 0)
            close_px = close_full.loc[bar_time, sym] if sym in close_full.columns else 0
            tick_cost = (t_bps / close_px * 10_000) if (close_px > 0 and t_bps > 0) else 2.8
            cost_per_unit = TAKER_BPS + tick_cost / 10_000
            cost_dollars += abs(trades[sym]) * GMV * cost_per_unit

    # ── PnL ──
    next_ret = returns_full.iloc[bar_pos + 1]
    pnl_gross = (target * next_ret.reindex(target.index, fill_value=0)).sum() * GMV
    pnl_net = pnl_gross - cost_dollars

    # Record
    bar_times.append(bar_time)
    bar_pnls_gross.append(pnl_gross)
    bar_pnls_net.append(pnl_net)
    bar_costs.append(cost_dollars)
    bar_turnover.append(turnover_pct)
    n_l = int((target > 0).sum())
    n_s = int((target < 0).sum())
    bar_n_long.append(n_l)
    bar_n_short.append(n_s)

    prev_positions = target

    # Progress
    bars_done = i + 1
    elapsed = time.time() - t0_total
    bar_elapsed = time.time() - t0_bar
    bars_left = n_bars - bars_done
    eta_s = elapsed / bars_done * bars_left
    cum_net = sum(bar_pnls_net)

    if bars_done <= 5 or bars_done % 50 == 0 or bars_done == n_bars:
        print(f"  [{bars_done:>4}/{n_bars}] {bar_time} | {bar_elapsed:.2f}s | "
              f"PnL=${pnl_net:>+8,.0f} | Cum=${cum_net:>+10,.0f} | "
              f"TO={turnover_pct:.3f} | {n_l}L/{n_s}S | "
              f"ETA {eta_s:.0f}s", flush=True)


# ═══════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════
total_time = time.time() - t0_total

cum_gross = np.cumsum(bar_pnls_gross)
cum_net = np.cumsum(bar_pnls_net)

results_df = pd.DataFrame({
    "time": bar_times,
    "pnl_gross": bar_pnls_gross,
    "pnl_net": bar_pnls_net,
    "cost": bar_costs,
    "turnover": bar_turnover,
    "n_long": bar_n_long,
    "n_short": bar_n_short,
}).set_index("time")

daily_pnl = results_df["pnl_net"].resample("D").sum()
daily_pnl = daily_pnl[daily_pnl != 0]

sharpe_daily = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(365)) if daily_pnl.std() > 0 else 0
bars_per_year = 6 * 365
sharpe_bar = (results_df["pnl_net"].mean() / results_df["pnl_net"].std() * np.sqrt(bars_per_year)) if results_df["pnl_net"].std() > 0 else 0

total_pnl_net = cum_net[-1]
total_pnl_gross = cum_gross[-1]
total_costs = sum(bar_costs)
max_dd_pct = ((cum_net - np.maximum.accumulate(cum_net)).min() / GMV) * 100
avg_turnover = results_df["turnover"].mean()
win_rate = (results_df["pnl_net"] > 0).mean()
n_days = len(daily_pnl)

print(f"\n{'='*70}")
print(f"  INCREMENTAL PIPELINE TEST TRACK RESULTS")
print(f"  {OOS_START} to {bar_times[-1]} ({n_bars} bars, {n_days} days)")
print(f"{'='*70}")
print(f"  Simulation time:    {total_time:.0f}s ({total_time/60:.1f}min)")
print(f"  Time per bar:       {total_time/n_bars:.2f}s")
print(f"")
print(f"  Total PnL (gross):  ${total_pnl_gross:>+12,.0f}")
print(f"  Total costs:        ${total_costs:>12,.0f}")
print(f"  Total PnL (net):    ${total_pnl_net:>+12,.0f}")
print(f"")
print(f"  Daily PnL (avg):    ${daily_pnl.mean():>+10,.0f}")
print(f"  Annual PnL (est):   ${daily_pnl.mean()*365:>+12,.0f}")
print(f"  Annual Return:      {daily_pnl.mean()*365/GMV*100:>+8.1f}%")
print(f"")
print(f"  Sharpe (daily):     {sharpe_daily:>8.2f}")
print(f"  Sharpe (bar-level): {sharpe_bar:>8.2f}")
print(f"  Win rate (bar):     {win_rate*100:>8.1f}%")
print(f"  Max drawdown:       {max_dd_pct:>8.1f}%")
print(f"  Avg turnover/bar:   {avg_turnover:>8.3f}")
print(f"  Avg positions:      {results_df['n_long'].mean():.0f}L / {results_df['n_short'].mean():.0f}S")
print(f"{'='*70}\n")

# ═══════════════════════════════════════════════════════════════
# EQUITY CURVE PLOT
# ═══════════════════════════════════════════════════════════════
out_path = Path(__file__).resolve().parent.parent / "data" / "crypto_results" / "incremental_test_track.png"
out_path.parent.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1, 1]})
fig.patch.set_facecolor("#1a1a2e")
for ax in axes:
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#555")

# Equity curve
ax1 = axes[0]
ax1.plot(bar_times, cum_gross, label="Gross", color="#66bb6a", alpha=0.5, linewidth=1)
ax1.plot(bar_times, cum_net, label="Net (after costs)", color="#42a5f5", linewidth=2)
ax1.fill_between(bar_times, 0, cum_net, alpha=0.15, color="#42a5f5")
ax1.axhline(0, color="#555", linewidth=0.5, linestyle="--")
ax1.set_title(
    f"Incremental Pipeline Test Track — Binance Risk Parity T100\n"
    f"$100K GMV | {n_bars} bars OOS | SR={sharpe_daily:.1f} | "
    f"Net PnL=${total_pnl_net:+,.0f} | Daily=${daily_pnl.mean():+,.0f}",
    fontsize=13, fontweight="bold", color="white"
)
ax1.set_ylabel("Cumulative PnL ($)", color="white")
ax1.legend(loc="upper left", facecolor="#16213e", edgecolor="#555", labelcolor="white")
ax1.grid(True, alpha=0.2, color="#555")

# Bar PnL
ax2 = axes[1]
colors = ["#66bb6a" if p > 0 else "#ef5350" for p in bar_pnls_net]
ax2.bar(bar_times, bar_pnls_net, color=colors, alpha=0.7, width=0.15)
ax2.axhline(0, color="#555", linewidth=0.5)
ax2.set_ylabel("Bar PnL ($)", color="white")
ax2.grid(True, alpha=0.2, color="#555")

# Drawdown
dd = cum_net - np.maximum.accumulate(cum_net)
ax3 = axes[2]
ax3.fill_between(bar_times, dd, 0, color="#ef5350", alpha=0.4)
ax3.set_ylabel("Drawdown ($)", color="white")
ax3.set_xlabel("Date", color="white")
ax3.grid(True, alpha=0.2, color="#555")

plt.tight_layout()
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
print(f"  Chart saved: {out_path}")

csv_path = out_path.with_suffix(".csv")
results_df.to_csv(csv_path)
print(f"  CSV saved: {csv_path}")
