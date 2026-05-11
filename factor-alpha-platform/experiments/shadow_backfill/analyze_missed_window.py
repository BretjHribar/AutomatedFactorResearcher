"""Quantify the gap between actual (frozen at 5/5) and counterfactual
(rebalanced daily) for the 5/5 -> 5/7 missed-trades window.

Reads weights.parquet from backfill_ib_moc_signals.py and the FMP close matrix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/shadow_backfill"
W = pd.read_parquet(OUT / "weights.parquet")
CLOSE = pd.read_parquet(ROOT / "data/fmp_cache/matrices/close.parquet")

BOOK = 500000.0
START = pd.Timestamp("2026-05-05")
# Fwd window: signals as of 5/5, 5/6, 5/7 — drives returns on 5/6, 5/7, 5/8
# But 5/8 close not yet in data, so we only have realised 5/6 and 5/7.

w_cols_in_close = [c for c in W.columns if c in CLOSE.columns]
W = W[w_cols_in_close]
CLOSE = CLOSE[w_cols_in_close]

# Bar-to-bar returns (close-to-close)
RET = CLOSE.pct_change()

# Trading days available
trading_days = [d for d in W.index if d >= START]
print(f"Backfill window starts {trading_days[0].date()}, ends {trading_days[-1].date()}", flush=True)
print(f"Last close in matrix: {CLOSE.index[-1].date()}", flush=True)
print(f"Last weight row: {W.index[-1].date()}", flush=True)

# Counterfactual A: HELD = freeze 5/5 weights, mark-to-market through last close.
held_w = W.loc[START].fillna(0.0)
# Counterfactual B: REBAL = use w[t] for the return from t to t+1.
print("\nDay-by-day comparison:")
print(f"{'date':<12} {'held_$pnl':>12} {'rebal_$pnl':>12} {'gap_$':>10} {'rebal_TO_$':>11}")
print("-" * 60)

held_cum = 0.0
rebal_cum = 0.0
gap_cum = 0.0

# Iterate transitions: for trading_day t and next trading day t+1,
#   held_pnl(t->t+1) = sum_i held_w[i] * (close[t+1]/close[t] - 1)
#   rebal_pnl(t->t+1) = sum_i W.loc[t, i] * (close[t+1]/close[t] - 1)
#   rebal_TO(t) = sum_i |W.loc[t] - W.loc[t-1]| * BOOK
all_dates = list(W.index)
idx_start = all_dates.index(START)
prev_w = W.loc[all_dates[idx_start - 1]].fillna(0.0)
for j in range(idx_start, len(all_dates) - 1):
    d_t = all_dates[j]
    d_next = all_dates[j + 1]
    if d_next not in CLOSE.index or d_t not in CLOSE.index:
        continue
    rets = (CLOSE.loc[d_next] / CLOSE.loc[d_t] - 1.0).fillna(0.0)
    held_pnl = float((held_w * rets).sum()) * BOOK
    rebal_w = W.loc[d_t].fillna(0.0)
    rebal_pnl = float((rebal_w * rets).sum()) * BOOK
    # turnover from previous rebal weights to today's
    rebal_to = float((rebal_w - prev_w).abs().sum()) * BOOK
    held_cum += held_pnl
    rebal_cum += rebal_pnl
    gap = rebal_pnl - held_pnl
    gap_cum += gap
    print(f"{d_t.date()} -> {d_next.date()}  "
          f"{held_pnl:>+10.0f}  {rebal_pnl:>+10.0f}  {gap:>+8.0f}  {rebal_to:>9.0f}",
          flush=True)
    prev_w = rebal_w

print("-" * 60)
print(f"CUMULATIVE:  held_$={held_cum:+.0f}   rebal_$={rebal_cum:+.0f}   "
      f"gap_$={gap_cum:+.0f} (rebal - held)", flush=True)

# T-cost estimate: 1bp impact + $0.0045/share commission ~ approx 2bp on a $500K book
# rebal_TO_total * 2bp / 100 = approx t-cost
all_to = 0.0
prev_w = W.loc[all_dates[idx_start - 1]].fillna(0.0)
for j in range(idx_start, len(all_dates)):
    rebal_w = W.loc[all_dates[j]].fillna(0.0)
    all_to += float((rebal_w - prev_w).abs().sum()) * BOOK
    prev_w = rebal_w
est_tc = all_to * 0.0002  # 2bp per side estimate
print(f"\nEstimated incremental t-cost of daily rebal vs held (turnover x 2bp): "
      f"${est_tc:,.0f}", flush=True)
print(f"Net counterfactual edge of rebal: ${gap_cum - est_tc:+,.0f}", flush=True)

# Also compute: forward-looking signal for trade tomorrow (5/8 close -> 5/11 close)
# That uses W.loc[5/8] which doesn't exist yet (5/7 is the last weight row).
# So tomorrow's MOC submission will use a freshly recomputed signal at 15:30 ET tomorrow.
print(f"\nLatest signal we have: weights as of {W.index[-1].date()} "
      f"(today's MOC will use a fresh signal computed at 15:30 ET).", flush=True)
