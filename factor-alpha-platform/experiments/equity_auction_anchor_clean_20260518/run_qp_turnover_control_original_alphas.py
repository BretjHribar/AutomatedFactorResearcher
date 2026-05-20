"""Focused QP + turnover-control tuning for the original AUCT_ANCHOR alphas.

This follows the existing library workflow:
  - target construction from run_qp_tuning_original_alphas.py
  - QP through src.portfolio.qp.run_walkforward
  - style+pca risk model through src.pipeline.runner._build_risk_model_fn
  - post-QP L1 turnover cap through experiments.aipt_stepwise_constraints
  - IB per-share MOC fee model from src.pipeline.fees

The goal is execution tuning, not alpha discovery. Alpha expressions are not
changed here.
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

from experiments.aipt_stepwise_constraints import _apply_turnover_control  # noqa: E402
from experiments.equity_auction_anchor_clean_20260518.run_qp_tuning_original_alphas import (  # noqa: E402
    BOOK,
    QP_START,
    VAL_END,
    VAL_START,
    _ann_sr,
    _build_targets,
    _corr,
    _max_dd,
)
from src.pipeline.fees import make_cost_fn  # noqa: E402
from src.pipeline.runner import _build_risk_model_fn  # noqa: E402
from src.portfolio.qp import run_walkforward  # noqa: E402


BARS_PER_YEAR = 252
RISK_MODEL_NAME = "style+pca"
RISK_MODEL_PARAMS = {"factor_window": 126, "n_pca_factors": 5, "vol_window": 60}


QP_SPECS = [
    # Best net from the first pass, then higher cost/risk pressure.
    {"alpha_scale": 10.0, "lambda_risk": 0.5, "kappa_tc": 5.0},
    {"alpha_scale": 10.0, "lambda_risk": 1.0, "kappa_tc": 20.0},
    {"alpha_scale": 10.0, "lambda_risk": 2.0, "kappa_tc": 50.0},
    {"alpha_scale": 30.0, "lambda_risk": 2.0, "kappa_tc": 100.0},
]

TURNOVER_CONTROLS = [
    {"turnover_cap": 0.0, "blend": 1.0},
    {"turnover_cap": 0.50, "blend": 1.0},
    {"turnover_cap": 0.35, "blend": 1.0},
    {"turnover_cap": 0.50, "blend": 0.75},
]


def _metrics(weights: pd.DataFrame, ret: pd.DataFrame, close: pd.DataFrame, fee_fn) -> tuple[pd.Series, pd.Series, pd.Series, dict]:
    gross = (weights * ret.shift(-1)).sum(axis=1).fillna(0.0)
    cost = fee_fn(weights, close, BOOK)
    net = gross - cost
    val_net = net.loc[VAL_START:VAL_END]
    val_gross = gross.loc[VAL_START:VAL_END]
    val_cost = cost.loc[VAL_START:VAL_END]
    turn = weights.diff().abs().sum(axis=1).fillna(0.0)
    stats = {
        "val_SR_gross": _ann_sr(val_gross),
        "val_SR_net": _ann_sr(val_net),
        "val_ret_ann_net": float(val_net.mean() * BARS_PER_YEAR),
        "val_vol_ann_net": float(val_net.std() * math.sqrt(BARS_PER_YEAR)),
        "val_cost_ann": float(val_cost.mean() * BARS_PER_YEAR),
        "val_max_dd_net": _max_dd(val_net),
        "turnover_full": float(turn.mean()),
        "turnover_val": float(turn.loc[VAL_START:VAL_END].mean()),
        "gross_l1_val": float(weights.abs().sum(axis=1).loc[VAL_START:VAL_END].mean()),
        "max_abs_w_val": float(weights.abs().max(axis=1).loc[VAL_START:VAL_END].mean()),
    }
    return gross, cost, net, stats


def _label_float(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def _apply_turnover_frame(weights: pd.DataFrame, turnover_cap: float, blend: float) -> pd.DataFrame:
    if turnover_cap <= 0 and abs(blend - 1.0) < 1e-12:
        return weights.copy()
    out = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
    prev = np.zeros(weights.shape[1], dtype=np.float64)
    for i in range(len(weights.index)):
        target = weights.iloc[i].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float64)
        new_w = _apply_turnover_control(target, prev, turnover_cap, blend)
        out.iloc[i] = new_w
        prev = new_w
    return out


def _append_row(rows: list[dict], series: dict[str, pd.Series], row: dict, net: pd.Series) -> None:
    rows.append(row)
    series[f"{row['cell']}_net"] = net
    pd.DataFrame(rows).to_csv(OUT_DIR / "qp_turnover_control_original_alphas.csv", index=False)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = json.loads((OUT_DIR / "validation_config_base.json").read_text())
    fee_fn = make_cost_fn(base["fees"]["model"], base["fees"]["params"], bars_per_year=BARS_PER_YEAR)

    print("=== building original alpha targets ===", flush=True)
    targets, uni, mats, close, ret, dates, tickers = _build_targets(base)
    print(f"=== building {RISK_MODEL_NAME} risk model fn ===", flush=True)
    risk_fn = _build_risk_model_fn(RISK_MODEL_NAME, RISK_MODEL_PARAMS, mats, dates, tickers)

    qslice = slice(QP_START, VAL_END)
    uni_q = uni.loc[qslice]
    close_q = close.loc[qslice]
    ret_q = ret.loc[qslice]

    rows: list[dict] = []
    series: dict[str, pd.Series] = {}

    for combiner, target in targets.items():
        target_q = target.loc[qslice]
        gross0, cost0, net0, stats0 = _metrics(target_q, ret_q, close_q, fee_fn)
        base_row = {
            "combiner": combiner,
            "cell": f"{combiner}_no_qp",
            "risk_model": "none",
            "alpha_scale": 0.0,
            "lambda_risk": 0.0,
            "kappa_tc": 0.0,
            "turnover_cap": 0.0,
            "blend": 1.0,
            "elapsed_sec": 0.0,
            **stats0,
            "corr_vs_noqp_net": 1.0,
        }
        _append_row(rows, series, base_row, net0)
        print(
            f"{base_row['cell']}: VAL gross={base_row['val_SR_gross']:+.3f} "
            f"net={base_row['val_SR_net']:+.3f} TO={base_row['turnover_val']:.3f} "
            f"cost={base_row['val_cost_ann']:.3f}",
            flush=True,
        )

        for tc in TURNOVER_CONTROLS[1:]:
            controlled = _apply_turnover_frame(target_q, tc["turnover_cap"], tc["blend"])
            _g, _c, net_tc, stats_tc = _metrics(controlled, ret_q, close_q, fee_fn)
            label = (
                f"{combiner}_no_qp_turncap{_label_float(tc['turnover_cap'])}"
                f"_blend{_label_float(tc['blend'])}"
            )
            row = {
                "combiner": combiner,
                "cell": label,
                "risk_model": "none",
                "alpha_scale": 0.0,
                "lambda_risk": 0.0,
                "kappa_tc": 0.0,
                **tc,
                "elapsed_sec": 0.0,
                **stats_tc,
                "corr_vs_noqp_net": _corr(net_tc, net0),
            }
            _append_row(rows, series, row, net_tc)
            print(
                f"{label}: VAL gross={row['val_SR_gross']:+.3f} "
                f"net={row['val_SR_net']:+.3f} TO={row['turnover_val']:.3f} "
                f"cost={row['val_cost_ann']:.3f} corr={row['corr_vs_noqp_net']:+.3f}",
                flush=True,
            )

        for spec in QP_SPECS:
            qp_label = (
                f"{combiner}_qp_{RISK_MODEL_NAME.replace('+', 'p')}"
                f"_s{_label_float(spec['alpha_scale'])}"
                f"_l{_label_float(spec['lambda_risk'])}"
                f"_k{_label_float(spec['kappa_tc'])}"
            )
            print(f"=== {qp_label} ===", flush=True)
            t0 = time.time()
            weights_qp = run_walkforward(
                target_q * float(spec["alpha_scale"]),
                close_q,
                ret_q,
                uni_q,
                risk_fn,
                lambda_risk=float(spec["lambda_risk"]),
                kappa_tc=float(spec["kappa_tc"]),
                max_w=0.02,
                commission_per_share=0.0045,
                impact_bps=0.0,
                vol_window=int(RISK_MODEL_PARAMS["vol_window"]),
                factor_window=int(RISK_MODEL_PARAMS["factor_window"]),
                dollar_neutral=True,
                max_gross_leverage=1.0,
                label=qp_label,
                verbose=True,
            )
            elapsed = time.time() - t0
            for tc in TURNOVER_CONTROLS:
                weights = _apply_turnover_frame(weights_qp, tc["turnover_cap"], tc["blend"])
                _g, _c, net, stats = _metrics(weights, ret_q, close_q, fee_fn)
                if tc["turnover_cap"] > 0 or abs(tc["blend"] - 1.0) > 1e-12:
                    label = (
                        f"{qp_label}_turncap{_label_float(tc['turnover_cap'])}"
                        f"_blend{_label_float(tc['blend'])}"
                    )
                else:
                    label = qp_label
                row = {
                    "combiner": combiner,
                    "cell": label,
                    "risk_model": RISK_MODEL_NAME,
                    **spec,
                    **tc,
                    "elapsed_sec": elapsed,
                    **stats,
                    "corr_vs_noqp_net": _corr(net, net0),
                }
                _append_row(rows, series, row, net)
                print(
                    f"{label}: VAL gross={row['val_SR_gross']:+.3f} "
                    f"net={row['val_SR_net']:+.3f} TO={row['turnover_val']:.3f} "
                    f"cost={row['val_cost_ann']:.3f} corr={row['corr_vs_noqp_net']:+.3f}",
                    flush=True,
                )

    out = pd.DataFrame(rows).sort_values(["val_SR_net", "val_SR_gross"], ascending=False)
    out.to_csv(OUT_DIR / "qp_turnover_control_original_alphas.csv", index=False)
    pd.DataFrame(series).sort_index().to_parquet(OUT_DIR / "qp_turnover_control_original_returns.parquet")
    print("\n=== best cells ===", flush=True)
    cols = [
        "combiner",
        "cell",
        "risk_model",
        "val_SR_gross",
        "val_SR_net",
        "turnover_val",
        "val_cost_ann",
        "corr_vs_noqp_net",
        "alpha_scale",
        "lambda_risk",
        "kappa_tc",
        "turnover_cap",
        "blend",
    ]
    print(out[cols].head(30).to_markdown(index=False, floatfmt=".3f"), flush=True)


if __name__ == "__main__":
    main()
