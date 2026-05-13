"""Backfill the preferreds-excluded variant of the shadow signal.

Runs the canonical pipeline ONCE with `data.exclude_tickers` set to the
PREFERRED list from the classification cache, then dumps per-day target
weights to `prod/logs/shadow_signals/equity_<DATE>_ex_pref.{parquet,json}`
for the last ~60 trading days. Mirrors `dump_shadow_signals.py` for the
full-universe variant.
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
sys.path.insert(0, str(ROOT))

from src.pipeline.runner import run, merge_overrides  # noqa: E402

CFG_PATH = ROOT / "prod/config/research_equity.json"
CLS_PATH = ROOT / "data/fmp_cache/classifications/preferred.parquet"
OUT = ROOT / "prod/logs/shadow_signals"
OUT.mkdir(parents=True, exist_ok=True)
BOOK = 500_000.0
N_DAYS = 60


def main() -> None:
    cls = pd.read_parquet(CLS_PATH)
    excluded = sorted(cls.loc[cls["is_preferred"] == True, "ticker"].astype(str).tolist())  # noqa: E712
    print(f"[ex_pref] excluding {len(excluded)} preferreds: {excluded}", flush=True)

    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    cfg = merge_overrides(cfg, {"data": {"exclude_tickers": excluded}})
    print("[ex_pref] running pipeline with preferreds excluded ...", flush=True)
    res = run(cfg, root=ROOT)
    print(f"[ex_pref] alpha_signals_n={res.alpha_signals_n}  universe={res.universe_size}  "
          f"n_bars={res.n_bars}  elapsed={res.elapsed_sec:.1f}s", flush=True)

    weights = res.weights
    close = pd.read_parquet(ROOT / "data/fmp_cache/matrices/close.parquet")

    target_dates = list(weights.index[-N_DAYS:])
    for d in target_dates:
        row = weights.loc[d].dropna()
        if d in close.index:
            last_close = close.loc[d]
        else:
            idx = close.index[close.index <= d]
            last_close = close.loc[idx[-1]] if len(idx) else pd.Series(dtype="float64")
        rows = []
        for sym, w in row.items():
            wv = float(w)
            if abs(wv) < 1e-12:
                continue
            dollar = BOOK * wv
            price = float(last_close.get(sym, np.nan))
            shares_est = int(round(dollar / price)) if (price and price > 0 and not np.isnan(price)) else 0
            rows.append({"ticker": sym, "weight": wv, "dollar": dollar,
                         "last_close": price if not np.isnan(price) else None,
                         "shares_est": shares_est,
                         "side": "long" if wv > 0 else "short"})
        df = pd.DataFrame(rows).sort_values("dollar", key=lambda s: s.abs(), ascending=False)
        label = d.date().isoformat()
        pq = OUT / f"equity_{label}_ex_pref.parquet"
        js = OUT / f"equity_{label}_ex_pref.json"
        tmp_pq = pq.with_suffix(pq.suffix + ".tmp")
        df.to_parquet(tmp_pq, index=False)
        os.replace(tmp_pq, pq)

        summary = {
            "signal_date": str(d),
            "strategy": "equity_1d",
            "variant": "ex_pref",
            "excluded_tickers": excluded,
            "n_excluded": len(excluded),
            "book": BOOK,
            "n_positions": int(len(df)),
            "n_long": int((df["weight"] > 0).sum()) if not df.empty else 0,
            "n_short": int((df["weight"] < 0).sum()) if not df.empty else 0,
            "gross_weight_l1": float(df["weight"].abs().sum()) if not df.empty else 0.0,
            "net_weight_l1": float(df["weight"].sum()) if not df.empty else 0.0,
            "gross_dollar": float(df["dollar"].abs().sum()) if not df.empty else 0.0,
            "net_dollar": float(df["dollar"].sum()) if not df.empty else 0.0,
            "gross_shares_est": int(df["shares_est"].abs().sum()) if not df.empty else 0,
            "parquet_path": str(pq),
            "source": "experiments.shadow_backfill.dump_shadow_signals_ex_pref",
        }
        tmp_js = js.with_suffix(js.suffix + ".tmp")
        tmp_js.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        os.replace(tmp_js, js)
    print(f"[ok] wrote {len(target_dates)} days of ex_pref shadows to {OUT}", flush=True)


if __name__ == "__main__":
    main()
