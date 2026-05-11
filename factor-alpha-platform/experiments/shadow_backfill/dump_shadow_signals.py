"""Dump per-day shadow_signals records for prior dates from the already-computed
weights.parquet. Mirrors the layout the live `equity_signal_shadow_record` asset
writes so historical backfill and live records share one schema.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/shadow_backfill"
OUT = ROOT / "prod/logs/shadow_signals"
OUT.mkdir(parents=True, exist_ok=True)

W = pd.read_parquet(EXP / "weights.parquet")
CLOSE = pd.read_parquet(ROOT / "data/fmp_cache/matrices/close.parquet")
BOOK = 500_000.0

# Backfill the last ~60 trading days so the dashboard has a meaningful curve.
# (The live `equity_signal_shadow_record` asset will start writing today's row
# on each new schedule fire — this is just the historical seed.)
N_DAYS = 60
TARGET_DATES = list(W.index[-N_DAYS:])

for d in TARGET_DATES:
    if d not in W.index:
        print(f"[skip] {d.date()} not in weights index", flush=True)
        continue
    weights_row = W.loc[d].dropna()
    if d in CLOSE.index:
        last_close = CLOSE.loc[d]
    else:
        # fallback: use the prior close
        idx = CLOSE.index[CLOSE.index <= d]
        last_close = CLOSE.loc[idx[-1]] if len(idx) else pd.Series(dtype="float64")

    rows = []
    for sym, w in weights_row.items():
        w_val = float(w)
        if abs(w_val) < 1e-12:
            continue
        dollar = BOOK * w_val
        price = float(last_close.get(sym, np.nan))
        shares_est = int(round(dollar / price)) if price and price > 0 and not np.isnan(price) else 0
        rows.append({
            "ticker": sym,
            "weight": w_val,
            "dollar": dollar,
            "last_close": price if not np.isnan(price) else None,
            "shares_est": shares_est,
            "side": "long" if w_val > 0 else "short",
        })
    df = pd.DataFrame(rows).sort_values("dollar", key=lambda s: s.abs(), ascending=False)

    label = d.date().isoformat()
    parquet_path = OUT / f"equity_{label}.parquet"
    json_path = OUT / f"equity_{label}.json"

    tmp_p = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    df.to_parquet(tmp_p, index=False)
    os.replace(tmp_p, parquet_path)

    summary = {
        "signal_date": str(d),
        "strategy": "equity_1d",
        "config_hash": None,
        "book": BOOK,
        "n_positions": int(len(df)),
        "n_long": int((df["weight"] > 0).sum()) if not df.empty else 0,
        "n_short": int((df["weight"] < 0).sum()) if not df.empty else 0,
        "gross_weight_l1": float(df["weight"].abs().sum()) if not df.empty else 0.0,
        "net_weight_l1": float(df["weight"].sum()) if not df.empty else 0.0,
        "gross_dollar": float(df["dollar"].abs().sum()) if not df.empty else 0.0,
        "net_dollar": float(df["dollar"].sum()) if not df.empty else 0.0,
        "gross_shares_est": int(df["shares_est"].abs().sum()) if not df.empty else 0,
        "alpha_signals_n": 45,
        "universe_size": 220,
        "max_lookback_bars": None,
        "parquet_path": str(parquet_path),
        "source": "experiments.shadow_backfill.dump_shadow_signals",
    }
    tmp_j = json_path.with_suffix(json_path.suffix + ".tmp")
    tmp_j.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    os.replace(tmp_j, json_path)
    print(f"[ok] {label}  n={summary['n_positions']}  L/S={summary['n_long']}/{summary['n_short']}  "
          f"gross_$={summary['gross_dollar']:,.0f}  shares={summary['gross_shares_est']:,}", flush=True)
