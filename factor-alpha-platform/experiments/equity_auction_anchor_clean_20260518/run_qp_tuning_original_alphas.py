"""Tune project-native QP on the original six AUCT_ANCHOR alphas.

This script does not modify alpha expressions. It precomputes the original
subindustry-neutralized alpha signals with train-history warmup, builds the
library combiner target, then runs src.portfolio.qp.run_walkforward over a
focused QP grid.

The important knob is alpha_scale. The project-native QP objective consumes
alpha scores, while the pipeline had been passing already-normalized portfolio
weights with magnitudes around 1e-3. Scaling restores the relative size of the
alpha term versus the L1 turnover and risk penalties.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
OUT_DIR = EXP_DIR / "outputs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.operators.fastexpression import FastExpressionEngine  # noqa: E402
from src.pipeline.fees import make_cost_fn  # noqa: E402
from src.pipeline.runner import (  # noqa: E402
    _load_alphas,
    _load_universe_and_matrices,
    _post_combiner,
)
from src.portfolio.combiners import (  # noqa: E402
    combiner_adaptive,
    combiner_equal,
    combiner_ic_weighted,
)
from src.portfolio.preprocessing import apply_preprocess  # noqa: E402
from src.portfolio.qp import run_walkforward  # noqa: E402
from src.portfolio.risk_model import build_diagonal  # noqa: E402


VAL_START = "2023-01-01"
VAL_END = "2024-07-01"
QP_START = "2022-01-01"
BARS_PER_YEAR = 252
BOOK = 500_000.0


COMBINERS = {
    "equal": (combiner_equal, {"max_wt": 0.02}),
    "adaptive": (combiner_adaptive, {}),
    "ic_wt": (combiner_ic_weighted, {}),
}

# Focused grid. The previous pipeline cell was effectively alpha_scale=1,
# lambda=5, kappa=30, which destroyed gross SR. These cells test restoring
# signal strength and then adding only enough L1 turnover penalty to matter.
GRID = [
    {"alpha_scale": 10.0, "lambda_risk": 0.0, "kappa_tc": 0.0},
    {"alpha_scale": 10.0, "lambda_risk": 0.1, "kappa_tc": 1.0},
    {"alpha_scale": 10.0, "lambda_risk": 0.5, "kappa_tc": 5.0},
    {"alpha_scale": 30.0, "lambda_risk": 0.0, "kappa_tc": 0.0},
    {"alpha_scale": 30.0, "lambda_risk": 0.1, "kappa_tc": 1.0},
    {"alpha_scale": 30.0, "lambda_risk": 0.5, "kappa_tc": 5.0},
    {"alpha_scale": 100.0, "lambda_risk": 0.0, "kappa_tc": 0.0},
    {"alpha_scale": 100.0, "lambda_risk": 0.1, "kappa_tc": 1.0},
    {"alpha_scale": 100.0, "lambda_risk": 0.5, "kappa_tc": 5.0},
    {"alpha_scale": 300.0, "lambda_risk": 0.1, "kappa_tc": 1.0},
    {"alpha_scale": 300.0, "lambda_risk": 0.5, "kappa_tc": 5.0},
    {"alpha_scale": 300.0, "lambda_risk": 1.0, "kappa_tc": 10.0},
]


def _ann_sr(s: pd.Series) -> float:
    clean = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) == 0 or clean.std() <= 0:
        return float("nan")
    return float(clean.mean() / clean.std() * math.sqrt(BARS_PER_YEAR))


def _max_dd(s: pd.Series) -> float:
    clean = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(clean) == 0:
        return float("nan")
    eq = (1.0 + clean).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def _metrics(weights: pd.DataFrame, ret: pd.DataFrame, close: pd.DataFrame, fee_fn) -> tuple[pd.Series, pd.Series, pd.Series, dict]:
    gross = (weights * ret.shift(-1)).sum(axis=1).fillna(0.0)
    cost = fee_fn(weights, close, BOOK)
    net = gross - cost
    val_net = net.loc[VAL_START:VAL_END]
    val_gross = gross.loc[VAL_START:VAL_END]
    val_cost = cost.loc[VAL_START:VAL_END]
    stats = {
        "val_SR_gross": _ann_sr(val_gross),
        "val_SR_net": _ann_sr(val_net),
        "val_ret_ann_net": float(val_net.mean() * BARS_PER_YEAR),
        "val_cost_ann": float(val_cost.mean() * BARS_PER_YEAR),
        "val_max_dd_net": _max_dd(val_net),
        "turnover_full": float(weights.diff().abs().sum(axis=1).mean()),
        "turnover_val": float(weights.diff().abs().sum(axis=1).loc[VAL_START:VAL_END].mean()),
        "gross_l1_val": float(weights.abs().sum(axis=1).loc[VAL_START:VAL_END].mean()),
        "max_abs_w_val": float(weights.abs().max(axis=1).loc[VAL_START:VAL_END].mean()),
    }
    return gross, cost, net, stats


def _corr(a: pd.Series, b: pd.Series) -> float:
    aa = a.loc[VAL_START:VAL_END].replace([np.inf, -np.inf], np.nan)
    bb = b.loc[VAL_START:VAL_END].replace([np.inf, -np.inf], np.nan)
    idx = aa.dropna().index.intersection(bb.dropna().index)
    if len(idx) < 20:
        return float("nan")
    return float(aa.loc[idx].corr(bb.loc[idx]))


def _risk_model_fn(_i, _idx, vol_today, _ret_mat, _factor_window):
    return build_diagonal(vol_today)


def _build_targets(base: dict):
    uni, dates, tickers, mats, close, ret, classifications, groups = _load_universe_and_matrices(base, root=ROOT)
    rows, train_sharpes = _load_alphas(base, root=ROOT)
    engine = FastExpressionEngine(data_fields=mats, groups=groups)
    alpha_signals = {}
    for aid, expr in rows:
        raw = engine.evaluate(expr).reindex(index=dates, columns=tickers)
        alpha_signals[aid] = apply_preprocess(
            raw,
            universe_mask=True,
            universe=uni,
            demean_method="subindustry",
            classifications=classifications,
            normalize="l1",
            clip_max_w=0.02,
        )

    targets = {}
    for name, (fn, params) in COMBINERS.items():
        local = {**params, "signals_are_preprocessed": True}
        if name == "topn_train":
            local["train_sharpes"] = train_sharpes
        combined = fn(alpha_signals, mats, uni, ret, **local)
        targets[name] = _post_combiner(combined, base, dates, tickers)
    return targets, uni, mats, close, ret, dates, tickers


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = json.loads((OUT_DIR / "validation_config_base.json").read_text())
    fee_fn = make_cost_fn(base["fees"]["model"], base["fees"]["params"], bars_per_year=BARS_PER_YEAR)

    print("=== building original alpha targets ===", flush=True)
    targets, uni, mats, close, ret, dates, tickers = _build_targets(base)

    # QP warmup slice: train-history warmup before validation, but avoids the
    # full 2016-2021 solve cost for a grid search.
    qslice = slice(QP_START, VAL_END)
    uni_q = uni.loc[qslice]
    close_q = close.loc[qslice]
    ret_q = ret.loc[qslice]

    rows = []
    series = {}
    for combiner, target in targets.items():
        target_q = target.loc[qslice]
        gross0, cost0, net0, stats0 = _metrics(target_q, ret_q, close_q, fee_fn)
        base_row = {
            "combiner": combiner,
            "cell": f"{combiner}_no_qp",
            "alpha_scale": 0.0,
            "lambda_risk": 0.0,
            "kappa_tc": 0.0,
            "qp_solves": 0,
            "qp_failures": 0,
            "elapsed_sec": 0.0,
            **stats0,
            "corr_vs_noqp_net": 1.0,
        }
        rows.append(base_row)
        series[f"{combiner}_noqp_net"] = net0

        for spec in GRID:
            label = (
                f"{combiner}_qp_s{spec['alpha_scale']:g}"
                f"_l{spec['lambda_risk']:g}_k{spec['kappa_tc']:g}"
            ).replace(".", "p")
            print(f"=== {label} ===", flush=True)
            t0 = time.time()
            weights = run_walkforward(
                target_q * float(spec["alpha_scale"]),
                close_q,
                ret_q,
                uni_q,
                _risk_model_fn,
                lambda_risk=float(spec["lambda_risk"]),
                kappa_tc=float(spec["kappa_tc"]),
                max_w=0.02,
                commission_per_share=0.0045,
                impact_bps=0.0,
                vol_window=60,
                factor_window=126,
                dollar_neutral=True,
                max_gross_leverage=1.0,
                label=label,
                verbose=True,
            )
            elapsed = time.time() - t0
            gross, cost, net, stats = _metrics(weights, ret_q, close_q, fee_fn)
            row = {
                "combiner": combiner,
                "cell": label,
                **spec,
                "elapsed_sec": elapsed,
                **stats,
                "corr_vs_noqp_net": _corr(net, net0),
            }
            rows.append(row)
            series[f"{label}_net"] = net
            print(
                f"{label}: VAL gross={row['val_SR_gross']:+.3f} "
                f"net={row['val_SR_net']:+.3f} TO={row['turnover_val']:.3f} "
                f"cost={row['val_cost_ann']:.3f} corr={row['corr_vs_noqp_net']:+.3f}",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(OUT_DIR / "qp_tuning_original_alphas.csv", index=False)

    out = pd.DataFrame(rows).sort_values(["val_SR_net", "val_SR_gross"], ascending=False)
    out.to_csv(OUT_DIR / "qp_tuning_original_alphas.csv", index=False)
    pd.DataFrame(series).sort_index().to_parquet(OUT_DIR / "qp_tuning_original_returns.parquet")
    print("\n=== best cells ===", flush=True)
    cols = [
        "cell",
        "val_SR_gross",
        "val_SR_net",
        "turnover_val",
        "val_cost_ann",
        "corr_vs_noqp_net",
        "alpha_scale",
        "lambda_risk",
        "kappa_tc",
    ]
    print(out[cols].head(20).to_markdown(index=False, floatfmt=".3f"), flush=True)


if __name__ == "__main__":
    main()
