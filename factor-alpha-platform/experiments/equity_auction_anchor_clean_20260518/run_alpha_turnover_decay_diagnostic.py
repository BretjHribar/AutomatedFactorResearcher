"""Per-alpha turnover diagnostic for the original AUCT_ANCHOR alphas.

This is an execution diagnostic, not alpha discovery. It keeps the DB alpha
expressions fixed and applies the existing vectorized Decay_lin operator as a
post-signal smoothing layer before the normal subindustry/L1/clip preprocessing.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
OUT_DIR = EXP_DIR / "outputs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.equity_auction_anchor_clean_20260518.run_qp_tuning_original_alphas import (  # noqa: E402
    BOOK,
    VAL_END,
    VAL_START,
    _ann_sr,
    _max_dd,
)
from src.operators.fastexpression import FastExpressionEngine  # noqa: E402
from src.operators.vectorized import Decay_lin  # noqa: E402
from src.pipeline.fees import make_cost_fn  # noqa: E402
from src.pipeline.runner import _load_alphas, _load_universe_and_matrices  # noqa: E402
from src.portfolio.preprocessing import apply_preprocess  # noqa: E402


BARS_PER_YEAR = 252
TRAIN_END = "2023-01-01"
DECAYS = [0, 5, 10]


def _metrics(weights: pd.DataFrame, ret: pd.DataFrame, close: pd.DataFrame, fee_fn) -> dict:
    gross = (weights * ret.shift(-1)).sum(axis=1).fillna(0.0)
    cost = fee_fn(weights, close, BOOK)
    net = gross - cost
    turn = weights.diff().abs().sum(axis=1).fillna(0.0)
    out = {}
    for label, sl in {
        "train": slice(None, TRAIN_END),
        "val": slice(TRAIN_END, VAL_END),
    }.items():
        g = gross.loc[sl]
        n = net.loc[sl]
        c = cost.loc[sl]
        t = turn.loc[sl]
        out[f"{label}_SR_gross"] = _ann_sr(g)
        out[f"{label}_SR_net"] = _ann_sr(n)
        out[f"{label}_ret_ann_net"] = float(n.mean() * BARS_PER_YEAR)
        out[f"{label}_cost_ann"] = float(c.mean() * BARS_PER_YEAR)
        out[f"{label}_turnover"] = float(t.mean())
        out[f"{label}_max_dd_net"] = _max_dd(n)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = json.loads((OUT_DIR / "validation_config_base.json").read_text())
    fee_fn = make_cost_fn(base["fees"]["model"], base["fees"]["params"], bars_per_year=BARS_PER_YEAR)
    uni, dates, tickers, mats, close, ret, classifications, groups = _load_universe_and_matrices(base, root=ROOT)
    rows, _train_sharpes = _load_alphas(base, root=ROOT)
    engine = FastExpressionEngine(data_fields=mats, groups=groups)

    out_rows = []
    for aid, expr in rows:
        print(f"=== alpha {aid} ===", flush=True)
        raw = engine.evaluate(expr).reindex(index=dates, columns=tickers)
        for decay in DECAYS:
            if decay > 0:
                sig = Decay_lin(raw, decay)
                variant = f"post_decay_{decay}"
            else:
                sig = raw
                variant = "original"
            weights = apply_preprocess(
                sig,
                universe_mask=True,
                universe=uni,
                demean_method="subindustry",
                classifications=classifications,
                normalize="l1",
                clip_max_w=0.02,
            )
            stats = _metrics(weights, ret, close, fee_fn)
            row = {
                "alpha_id": aid,
                "variant": variant,
                "post_signal_decay": decay,
                **stats,
            }
            out_rows.append(row)
            print(
                f"a{aid} {variant}: train net={row['train_SR_net']:+.3f} "
                f"TO={row['train_turnover']:.3f}; val net={row['val_SR_net']:+.3f} "
                f"TO={row['val_turnover']:.3f} cost={row['val_cost_ann']:.3f}",
                flush=True,
            )
            pd.DataFrame(out_rows).to_csv(OUT_DIR / "alpha_turnover_decay_diagnostic.csv", index=False)

    summary = pd.DataFrame(out_rows)
    summary.to_csv(OUT_DIR / "alpha_turnover_decay_diagnostic.csv", index=False)
    print("\n=== best per alpha by val net with turnover <= original ===", flush=True)
    original_turn = summary[summary["variant"] == "original"].set_index("alpha_id")["val_turnover"]
    filt = summary[summary.apply(lambda r: r["val_turnover"] <= original_turn.loc[r["alpha_id"]], axis=1)]
    best = filt.sort_values(["alpha_id", "val_SR_net"], ascending=[True, False]).groupby("alpha_id").head(1)
    cols = ["alpha_id", "variant", "train_SR_net", "val_SR_net", "train_turnover", "val_turnover", "val_cost_ann"]
    print(best[cols].to_markdown(index=False, floatfmt=".3f"), flush=True)


if __name__ == "__main__":
    main()
