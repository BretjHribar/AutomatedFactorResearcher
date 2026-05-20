"""Test-set run for the validation-selected AUCT QP strategy.

Selection was done on train/validation. This script freezes those settings,
loads the full PIT universe/matrices, and reports only the post-validation
test window against the two existing paper-production references.
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

from experiments.equity_auction_anchor_clean_20260518.run_qp_tuning_original_alphas import (  # noqa: E402
    BOOK,
    QP_START,
    _build_targets,
)
from src.pipeline.fees import make_cost_fn  # noqa: E402
from src.pipeline.runner import _build_risk_model_fn, merge_overrides, run  # noqa: E402
from src.portfolio.qp import run_walkforward  # noqa: E402


BARS_PER_YEAR = 252
TRAIN_END = "2023-01-01"
VAL_END = "2024-07-01"
TEST_START = "2024-07-02"
PROD_TEST_START = "2025-04-01"
FULL_AUCT_UNIVERSE = EXP_DIR / "universes" / "AUCT_ANCHOR_MCAP90M_550M_DAILY.parquet"
RISK_MODEL_NAME = "style+pca"
RISK_MODEL_PARAMS = {"factor_window": 126, "n_pca_factors": 5, "vol_window": 60}
SELECTED_QP = {"alpha_scale": 10.0, "lambda_risk": 2.0, "kappa_tc": 50.0}
PROD_DASHBOARD_ALPHA_FILTER = (
    "archived=0 AND asset_class='equities' AND "
    "(notes LIKE '%SMALLCAP_D0_v2%' OR notes LIKE '%SMALLCAP_D0_v3%')"
)


def _rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


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


def _corr(a: pd.Series, b: pd.Series, start: str, end: str) -> float:
    aa = a.loc[start:end].replace([np.inf, -np.inf], np.nan)
    bb = b.loc[start:end].replace([np.inf, -np.inf], np.nan)
    idx = aa.dropna().index.intersection(bb.dropna().index)
    if len(idx) < 20:
        return float("nan")
    return float(aa.loc[idx].corr(bb.loc[idx]))


def _metrics(
    *,
    label: str,
    gross: pd.Series,
    cost: pd.Series,
    net: pd.Series,
    turnover: pd.Series | None,
    start: str,
    end: str,
) -> dict:
    g = gross.loc[start:end]
    c = cost.loc[start:end]
    n = net.loc[start:end]
    t = turnover.loc[start:end] if turnover is not None else None
    return {
        "series": label,
        "test_start": start,
        "test_end": end,
        "test_n_bars": int(n.replace([np.inf, -np.inf], np.nan).dropna().shape[0]),
        "test_SR_gross": _ann_sr(g),
        "test_SR_net": _ann_sr(n),
        "test_ret_ann_net": float(n.mean() * BARS_PER_YEAR),
        "test_vol_ann_net": float(n.std() * math.sqrt(BARS_PER_YEAR)),
        "test_max_dd_net": _max_dd(n),
        "test_cost_ann": float(c.mean() * BARS_PER_YEAR),
        "test_turnover": float(t.mean()) if t is not None else float("nan"),
    }


def _base_full_auct_config() -> dict:
    base = json.loads((OUT_DIR / "validation_config_base.json").read_text())
    base["data"]["universe_path"] = _rel(FULL_AUCT_UNIVERSE)
    base["data"]["universe_filter"] = {"method": "coverage", "threshold": 0.0}
    base["splits"] = {"train_end": TRAIN_END, "val_end": VAL_END}
    base["book"] = BOOK
    return base


def _prod_reference() -> object:
    prod_cfg = json.loads((ROOT / "prod" / "config" / "research_equity.json").read_text())
    cfg = merge_overrides(
        prod_cfg,
        {
            "alpha_source": {
                # Matches the paper IB MOC dashboard snapshot alpha set
                # (45 signals as of 2026-05-14). The currently edited prod
                # config adds an evaluation subquery and drops this to 36.
                "filter_sql": PROD_DASHBOARD_ALPHA_FILTER,
                "train_sharpe_table": None,
                "train_sharpe_column": None,
            },
            "data": {"max_lookback_bars": 400},
            "book": BOOK,
            "fees": {
                "model": "per_share_ib",
                "params": {
                    "commission_per_share": 0.0045,
                    "per_order_min": 0.35,
                    "sec_fee_per_dollar": 27.80e-6,
                    "sell_fraction": 0.50,
                    "impact_bps": 0.0,
                    "borrow_bps_annual": 50,
                },
            },
            "qp": {
                **prod_cfg.get("qp", {}),
                "enabled": True,
                "commission_per_share": 0.0045,
                "impact_bps": 0.0,
            },
        },
    )
    return run(cfg, verbose=False)


def _aipt_reference() -> tuple[pd.Series, pd.Series, pd.Series]:
    path = (
        ROOT
        / "experiments"
        / "results"
        / "aipt_live_refresh_20260514"
        / "equity_smallcap_d0__prox_l1_gross1_cap_fee__P256__z0p001__seed1__maxw0p02__tau5__turncap0__blend1__qpscale1__qprisk0.returns.parquet"
    )
    r = pd.read_parquet(path)
    if not isinstance(r.index, pd.DatetimeIndex):
        r.index = pd.to_datetime(r.index)
    return r["gross"], r["cost"], r["net"]


def _run_selected_auct() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = _base_full_auct_config()
    fee_fn = make_cost_fn(base["fees"]["model"], base["fees"]["params"], bars_per_year=BARS_PER_YEAR)
    print("=== building full AUCT target for frozen VAL-selected setting ===", flush=True)
    targets, uni, mats, close, ret, dates, tickers = _build_targets(base)
    print(f"=== building {RISK_MODEL_NAME} risk model fn ===", flush=True)
    risk_fn = _build_risk_model_fn(RISK_MODEL_NAME, RISK_MODEL_PARAMS, mats, dates, tickers)

    qslice = slice(QP_START, None)
    uni_q = uni.loc[qslice]
    close_q = close.loc[qslice]
    ret_q = ret.loc[qslice]

    weights_out: dict[str, pd.DataFrame] = {}
    label = "ic_wt_qp_styleppca_s10_l2_k50"
    print(f"=== {label} full test run ===", flush=True)
    t0 = time.time()
    weights = run_walkforward(
        targets["ic_wt"].loc[qslice] * SELECTED_QP["alpha_scale"],
        close_q,
        ret_q,
        uni_q,
        risk_fn,
        lambda_risk=SELECTED_QP["lambda_risk"],
        kappa_tc=SELECTED_QP["kappa_tc"],
        max_w=0.02,
        commission_per_share=0.0045,
        impact_bps=0.0,
        vol_window=RISK_MODEL_PARAMS["vol_window"],
        factor_window=RISK_MODEL_PARAMS["factor_window"],
        dollar_neutral=True,
        max_gross_leverage=1.0,
        label=label,
        verbose=True,
    )
    print(f"{label} elapsed_sec={time.time() - t0:.1f}", flush=True)
    weights_out[label] = weights

    returns = {}
    for label, weights in weights_out.items():
        gross = (weights * ret_q.shift(-1)).sum(axis=1).fillna(0.0)
        cost = fee_fn(weights, close_q, BOOK)
        net = gross - cost
        turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
        returns[f"{label}_gross"] = gross
        returns[f"{label}_cost"] = cost
        returns[f"{label}_net"] = net
        returns[f"{label}_turnover"] = turnover
    return weights_out, pd.DataFrame(returns).sort_index(), close_q, ret_q


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    weights, auct_returns, _close, _ret = _run_selected_auct()
    print("=== running existing IB MOC dashboard reference on production test panel ===", flush=True)
    prod = _prod_reference()
    print("=== loading AIPT paper reference ===", flush=True)
    aipt_gross, aipt_cost, aipt_net = _aipt_reference()

    all_net = {
        "paper_ib_moc_equity": prod.net_pnl,
        "paper_aipt_smallcap_d0": aipt_net,
        "auct_ic_wt_qp_styleppca_s10_l2_k50": auct_returns["ic_wt_qp_styleppca_s10_l2_k50_net"],
    }
    common_end = min(s.dropna().index.max() for s in all_net.values())
    test_end = str(pd.Timestamp(common_end).date())
    comparison_start = PROD_TEST_START

    rows = []
    rows.append(
        _metrics(
            label="paper_ib_moc_equity",
            gross=prod.gross_pnl,
            cost=prod.cost,
            net=prod.net_pnl,
            turnover=prod.weights.diff().abs().sum(axis=1).fillna(0.0),
            start=comparison_start,
            end=test_end,
        )
    )
    rows.append(
        _metrics(
            label="paper_aipt_smallcap_d0",
            gross=aipt_gross,
            cost=aipt_cost,
            net=aipt_net,
            turnover=None,
            start=comparison_start,
            end=test_end,
        )
    )
    for label in (
        "ic_wt_qp_styleppca_s10_l2_k50",
    ):
        rows.append(
            _metrics(
                label=f"auct_{label}",
                gross=auct_returns[f"{label}_gross"],
                cost=auct_returns[f"{label}_cost"],
                net=auct_returns[f"{label}_net"],
                turnover=auct_returns[f"{label}_turnover"],
                start=comparison_start,
                end=test_end,
            )
        )

    series = pd.DataFrame(all_net).sort_index().loc[comparison_start:test_end]
    labels = list(series.columns)
    corr = pd.DataFrame(index=labels, columns=labels, dtype=float)
    overlap = pd.DataFrame(index=labels, columns=labels, dtype=int)
    for a in labels:
        for b in labels:
            aa = series[a].replace([np.inf, -np.inf], np.nan)
            bb = series[b].replace([np.inf, -np.inf], np.nan)
            idx = aa.dropna().index.intersection(bb.dropna().index)
            overlap.loc[a, b] = len(idx)
            corr.loc[a, b] = float(aa.loc[idx].corr(bb.loc[idx])) if len(idx) >= 20 else float("nan")

    summary = pd.DataFrame(rows)
    for ref in ("paper_ib_moc_equity", "paper_aipt_smallcap_d0"):
        summary[f"corr_vs_{ref}"] = [
            corr.loc[row["series"].replace("auct_", "auct_"), ref]
            if row["series"] in corr.index else float("nan")
            for row in summary.to_dict("records")
        ]

    summary_path = OUT_DIR / "test_prod_window_corrected_summary.csv"
    corr_path = OUT_DIR / "test_prod_window_corrected_correlation_matrix.csv"
    overlap_path = OUT_DIR / "test_prod_window_corrected_correlation_overlap.csv"
    returns_path = OUT_DIR / "test_prod_window_corrected_returns.parquet"
    weights_path = OUT_DIR / "test_selected_strategy_weights.parquet"
    summary.to_csv(summary_path, index=False)
    corr.to_csv(corr_path)
    overlap.to_csv(overlap_path)
    pd.concat([auct_returns, pd.DataFrame({"paper_ib_moc_equity_net": prod.net_pnl, "paper_aipt_smallcap_d0_net": aipt_net})], axis=1).sort_index().to_parquet(returns_path)

    # Store selected AUCT weights only; production/AIPT weights are not all available here.
    pd.concat({k: v for k, v in weights.items()}, names=["series", "date"]).to_parquet(weights_path)

    print("\n=== TEST SUMMARY ===", flush=True)
    print(summary.sort_values("test_SR_net", ascending=False).to_markdown(index=False, floatfmt=".3f"), flush=True)
    print("\n=== TEST NET RETURN CORRELATION ===", flush=True)
    print(corr.to_markdown(floatfmt=".3f"), flush=True)
    print(f"\nSaved {summary_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
