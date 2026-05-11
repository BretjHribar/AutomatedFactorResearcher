"""Backfill the IB MOC signal that *would* have fired on each recent trading day.

Runs the canonical research_equity pipeline once on the full history, then dumps
per-day target-weight rows + per-day gross/cost/net P&L. This is the
deterministic record of "what would the system have traded?" — it does not
need IB Gateway, FMP intraday, or `prod/moc_trader.py`'s import chain.

Outputs (all under experiments/shadow_backfill/):
  weights.parquet      — T x N target weights (book = $500K)
  pnl.parquet          — T x 3: gross_pnl, cost, net_pnl (in $)
  daily_summary.csv    — per-day n_long/n_short/gross/net + cumulative net
  recent_diffs.csv     — order_diffs for the last 6 trading days
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.pipeline.runner import run  # noqa: E402

CFG = ROOT / "prod/config/research_equity.json"
OUT = ROOT / "experiments/shadow_backfill"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print(f"[backfill] running pipeline {CFG} ...", flush=True)
    res = run(CFG, root=ROOT)
    print(f"[backfill] alpha_signals_n={res.alpha_signals_n}  universe={res.universe_size}  "
          f"n_bars={res.n_bars}  elapsed={res.elapsed_sec:.1f}s", flush=True)

    weights: pd.DataFrame = res.weights
    book = float(res.config.get("book", 1.0))
    gross_pnl: pd.Series = res.gross_pnl
    cost: pd.Series = res.cost
    net_pnl: pd.Series = res.net_pnl

    # Persist raw matrices
    weights.to_parquet(OUT / "weights.parquet")
    pnl_df = pd.DataFrame({"gross_pnl": gross_pnl, "cost": cost, "net_pnl": net_pnl})
    pnl_df.to_parquet(OUT / "pnl.parquet")

    # Per-day summary (book-scaled $)
    summary_rows = []
    cum_gross = 0.0
    cum_cost = 0.0
    cum_net = 0.0
    for d in weights.index:
        row = weights.loc[d].dropna()
        n_long = int((row > 1e-6).sum())
        n_short = int((row < -1e-6).sum())
        gross_l1 = float(row.abs().sum())
        net_l1 = float(row.sum())
        g = float(gross_pnl.get(d, np.nan)) if d in gross_pnl.index else np.nan
        c = float(cost.get(d, np.nan)) if d in cost.index else np.nan
        n = float(net_pnl.get(d, np.nan)) if d in net_pnl.index else np.nan
        if not np.isnan(g):
            cum_gross += g
        if not np.isnan(c):
            cum_cost += c
        if not np.isnan(n):
            cum_net += n
        summary_rows.append({
            "date": d.date().isoformat() if hasattr(d, "date") else str(d),
            "n_long": n_long,
            "n_short": n_short,
            "gross_l1": gross_l1,
            "net_l1": net_l1,
            "gross_$": book * gross_l1,
            "gross_pnl_$": g,
            "cost_$": c,
            "net_pnl_$": n,
            "cum_net_pnl_$": cum_net,
            "cum_gross_pnl_$": cum_gross,
            "cum_cost_$": cum_cost,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "daily_summary.csv", index=False)
    print(f"[backfill] wrote {OUT / 'daily_summary.csv'}  rows={len(summary)}", flush=True)

    # Tail (last 12 trading days) — what user wants for "what would we have traded?"
    tail = summary.tail(12)
    print("\nLast 12 trading days (signal-side, book = $500K):", flush=True)
    print(tail[["date", "n_long", "n_short", "gross_$", "net_l1",
                "gross_pnl_$", "cost_$", "net_pnl_$", "cum_net_pnl_$"]].to_string(index=False), flush=True)

    # Order diffs for the 5 most-recent dates: target_t - target_{t-1}
    recent_dates = list(weights.index[-7:])
    if len(recent_dates) >= 2:
        diffs_rows = []
        for i in range(1, len(recent_dates)):
            d_prev, d_now = recent_dates[i - 1], recent_dates[i]
            w_prev = weights.loc[d_prev]
            w_now = weights.loc[d_now]
            delta = (w_now - w_prev).fillna(0.0)
            # Order $ flow at $500K book
            order_l1 = float(delta.abs().sum())
            diffs_rows.append({
                "as_of": d_now.date().isoformat() if hasattr(d_now, "date") else str(d_now),
                "rebal_from": d_prev.date().isoformat() if hasattr(d_prev, "date") else str(d_prev),
                "n_changes": int((delta.abs() > 1e-6).sum()),
                "turnover_l1": order_l1,
                "turnover_$": book * order_l1,
            })
        diffs_df = pd.DataFrame(diffs_rows)
        diffs_df.to_csv(OUT / "recent_diffs.csv", index=False)
        print("\nRebalance turnover for the last 6 transitions:", flush=True)
        print(diffs_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
