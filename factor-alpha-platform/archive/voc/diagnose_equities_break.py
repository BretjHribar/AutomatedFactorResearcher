"""Diagnose why TOP2000 equities AIPT broke around 2025-12-01.

Checks (in order of likelihood):
1. Data quality — is char availability dropping? (NaN density per char per bar)
2. Universe size — N_valid per bar over time
3. IC time series — when does IC drop to zero?
4. Position activity — are weights collapsing to zero?
5. Returns dispersion — is cross-sectional return std collapsing?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest_voc_equities import (
    BARS_PER_YEAR, OOS_START, COVERAGE_CUTOFF,
    MATRICES_DIR, UNIVERSES_DIR, RESULTS_DIR, CHAR_NAMES,
)
from backtest_voc_equities_sweep import load_data_universe

print("Loading TOP2000...")
matrices, tickers, dates, close_vals, chars = load_data_universe("TOP2000")
N = len(tickers)
print(f"  N={N}, T={len(dates)}, chars={len(chars)}")
print(f"  date range: {dates.min()} → {dates.max()}\n")

# ── 1. Per-char NaN density over time ────────────────────────────────────────
print("="*90)
print("1. CHAR AVAILABILITY OVER TIME (fraction of universe with non-NaN per bar)")
print("="*90)
char_avail = pd.DataFrame(index=dates, columns=chars, dtype=float)
for cn in chars:
    char_avail[cn] = (~matrices[cn].isna()).mean(axis=1)

# Print availability around the break
break_date = pd.Timestamp("2025-12-01")
window_start = pd.Timestamp("2025-09-01")
window_end = pd.Timestamp("2026-02-01")
window = char_avail.loc[window_start:window_end]
print(f"\nChar availability {window_start.date()} → {window_end.date()}:")
print(f"  {'char':<30} {'2025-09':>8} {'2025-10':>8} {'2025-11':>8} {'2025-12':>8} {'2026-01':>8}")
for cn in chars:
    vals = []
    for date_str in ["2025-09-15", "2025-10-15", "2025-11-15", "2025-12-15", "2026-01-15"]:
        ts = pd.Timestamp(date_str)
        ix = char_avail.index.get_indexer([ts], method="nearest")[0]
        vals.append(char_avail[cn].iloc[ix])
    print(f"  {cn:<30}  {vals[0]:>8.2%} {vals[1]:>8.2%} {vals[2]:>8.2%} {vals[3]:>8.2%} {vals[4]:>8.2%}")

# Mean availability across chars over time
avg_avail = char_avail.mean(axis=1)
print(f"\nMean char availability:")
for date_str in ["2024-06-01", "2024-12-01", "2025-06-01", "2025-09-01", "2025-11-01",
                 "2025-12-01", "2026-01-01", "2026-02-01"]:
    ts = pd.Timestamp(date_str)
    ix = avg_avail.index.get_indexer([ts], method="nearest")[0]
    actual_date = avg_avail.index[ix]
    print(f"  {actual_date.date()}: {avg_avail.iloc[ix]:.2%}")

# ── 2. Universe size — how many tickers have any data per bar ────────────────
print(f"\n{'='*90}")
print("2. ACTIVE UNIVERSE SIZE OVER TIME (tickers with non-NaN close)")
print("="*90)
active = (~matrices["close"].isna()).sum(axis=1)
print(f"\nActive ticker count over time:")
for date_str in ["2024-01-01", "2024-12-01", "2025-06-01", "2025-09-01", "2025-11-01",
                 "2025-12-01", "2026-01-01", "2026-02-01"]:
    ts = pd.Timestamp(date_str)
    ix = active.index.get_indexer([ts], method="nearest")[0]
    print(f"  {active.index[ix].date()}: {active.iloc[ix]} active tickers (of {N})")

# ── 3. Per-bar IC + cumulative return from saved baseline ───────────────────
print(f"\n{'='*90}")
print("3. PER-BAR IC OVER TIME (from saved baseline)")
print("="*90)
csv = RESULTS_DIR / "voc_equities_baseline.csv"
df = pd.read_csv(csv, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Rolling IC (60-day window)
df["rolling_ic_60d"] = df["ic_p"].rolling(60).mean()
df["rolling_ir_60d"] = df["ic_p"].rolling(60).mean() / df["ic_p"].rolling(60).std() * np.sqrt(252)
df["rolling_sr_60d"] = df["net_1bps"].rolling(60).mean() / df["net_1bps"].rolling(60).std() * np.sqrt(252)
df["rolling_to_60d"] = df["turnover"].rolling(60).mean()

print(f"\n60-bar rolling stats:")
print(f"  {'date':<12} {'IC_p':>8} {'rolling_IC':>12} {'rolling_SR_n':>14} {'rolling_TO%':>12}")
for date_str in ["2024-06-01", "2024-12-01", "2025-06-01", "2025-09-01", "2025-10-01",
                 "2025-11-01", "2025-12-01", "2026-01-01", "2026-02-01"]:
    ts = pd.Timestamp(date_str)
    sub = df[df["date"] >= ts].head(1)
    if sub.empty: continue
    r = sub.iloc[0]
    print(f"  {r['date'].date()!s:<12} {r['ic_p']:>+8.4f} {r['rolling_ic_60d']:>+12.5f} "
          f"{r['rolling_sr_60d']:>+14.2f} {r['rolling_to_60d']*100:>11.1f}%")

# ── 4. Cross-sectional return dispersion ────────────────────────────────────
print(f"\n{'='*90}")
print("4. CROSS-SECTIONAL RETURN DISPERSION (does the market still differentiate?)")
print("="*90)
returns = matrices["close"].pct_change(fill_method=None)
returns = returns.replace([np.inf, -np.inf], np.nan)
ret_xs_std = returns.std(axis=1)
print(f"\nCross-sectional return std dev over time:")
for date_str in ["2024-06-01", "2024-12-01", "2025-06-01", "2025-09-01", "2025-11-01",
                 "2025-12-01", "2026-01-01", "2026-02-01"]:
    ts = pd.Timestamp(date_str)
    ix = ret_xs_std.index.get_indexer([ts], method="nearest")[0]
    actual = ret_xs_std.index[ix]
    print(f"  {actual.date()}: xs_std={ret_xs_std.iloc[ix]*100:.2f}%")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

axes[0].plot(df["date"], df["net_1bps"].cumsum() * 100, color="tab:red", linewidth=1.5)
axes[0].axvline(pd.Timestamp("2025-12-01"), color="black", linestyle="--", alpha=0.6, label="2025-12-01")
axes[0].axhline(0, color="black", linewidth=0.5, alpha=0.4)
axes[0].set_title("Cumulative net return (1bp)", fontweight="bold")
axes[0].set_ylabel("Cum return %")
axes[0].grid(alpha=0.3); axes[0].legend()

axes[1].plot(df["date"], df["rolling_ic_60d"], color="tab:blue", linewidth=1.4, label="60-day rolling IC (Pearson)")
axes[1].axvline(pd.Timestamp("2025-12-01"), color="black", linestyle="--", alpha=0.6)
axes[1].axhline(0, color="black", linewidth=0.5, alpha=0.4)
axes[1].set_title("60-bar rolling IC", fontweight="bold")
axes[1].set_ylabel("IC (Pearson)")
axes[1].grid(alpha=0.3); axes[1].legend()

axes[2].plot(avg_avail.index, avg_avail.values, color="tab:green", linewidth=1.4)
axes[2].axvline(pd.Timestamp("2025-12-01"), color="black", linestyle="--", alpha=0.6)
axes[2].set_title("Mean character data availability (fraction non-NaN)", fontweight="bold")
axes[2].set_ylabel("Avg availability")
axes[2].grid(alpha=0.3)

axes[3].plot(active.index, active.values, color="tab:purple", linewidth=1.4, label="Active tickers")
axes[3].axvline(pd.Timestamp("2025-12-01"), color="black", linestyle="--", alpha=0.6)
axes[3].set_title("Active universe size (tickers with non-NaN close)", fontweight="bold")
axes[3].set_ylabel("# tickers")
axes[3].set_xlabel("Date")
axes[3].grid(alpha=0.3); axes[3].legend()

# Limit x-axis to OOS-relevant period
axes[3].set_xlim(pd.Timestamp("2023-06-01"), pd.Timestamp("2026-03-01"))

fig.tight_layout()
out = RESULTS_DIR / "diagnose_equities_break.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nPlot: {out}")
