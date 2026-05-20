"""Diagnose why the AUCT strategy underperformed on the production TEST window.

This is a diagnostic wrapper around the existing project libraries. It does
not change alpha expressions or portfolio construction.
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
FULL_AUCT_UNIVERSE = EXP_DIR / "universes" / "AUCT_ANCHOR_MCAP90M_550M_DAILY.parquet"

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


BARS_PER_YEAR = 252
BOOK = 500_000.0
SPLITS = {
    "TRAIN": ("2016-01-04", "2023-01-01"),
    "VAL": ("2023-01-01", "2024-07-01"),
    "AUCT_TEST": ("2024-07-02", "2026-05-14"),
    "PROD_TEST": ("2025-04-01", "2026-05-14"),
}
HALF_YEAR_WINDOWS = {
    "2023H1": ("2023-01-01", "2023-06-30"),
    "2023H2": ("2023-07-01", "2023-12-31"),
    "2024H1": ("2024-01-01", "2024-07-01"),
    "2024H2": ("2024-07-02", "2024-12-31"),
    "2025H1": ("2025-01-01", "2025-06-30"),
    "2025H2": ("2025-07-01", "2025-12-31"),
    "2026YTD": ("2026-01-01", "2026-05-14"),
}
COMBINERS = {
    "equal": (combiner_equal, {"max_wt": 0.02}),
    "adaptive": (combiner_adaptive, {}),
    "ic_wt": (combiner_ic_weighted, {}),
}


def _rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _ann_sr(s: pd.Series) -> float:
    clean = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) <= 1 or clean.std() <= 0:
        return float("nan")
    return float(clean.mean() / clean.std() * math.sqrt(BARS_PER_YEAR))


def _max_dd(s: pd.Series) -> float:
    clean = s.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return float("nan")
    eq = (1.0 + clean).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def _metrics(
    label: str,
    split: str,
    start: str,
    end: str,
    gross: pd.Series,
    cost: pd.Series,
    net: pd.Series,
    turnover: pd.Series,
) -> dict:
    g = gross.loc[start:end]
    c = cost.loc[start:end]
    n = net.loc[start:end]
    t = turnover.loc[start:end]
    clean_n = n.replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "series": label,
        "split": split,
        "start": start,
        "end": end,
        "n_bars": int(len(clean_n)),
        "SR_gross": _ann_sr(g),
        "SR_net": _ann_sr(n),
        "ret_ann_gross": float(g.mean() * BARS_PER_YEAR),
        "ret_ann_net": float(n.mean() * BARS_PER_YEAR),
        "cost_ann": float(c.mean() * BARS_PER_YEAR),
        "turnover": float(t.mean()),
        "vol_ann_net": float(n.std() * math.sqrt(BARS_PER_YEAR)),
        "max_dd_net": _max_dd(n),
    }


def _series_metrics(label: str, gross: pd.Series, cost: pd.Series, net: pd.Series, turnover: pd.Series) -> list[dict]:
    rows = []
    for split, (start, end) in SPLITS.items():
        rows.append(_metrics(label, split, start, end, gross, cost, net, turnover))
    return rows


def _window_metrics(label: str, gross: pd.Series, cost: pd.Series, net: pd.Series, turnover: pd.Series) -> list[dict]:
    rows = []
    for split, (start, end) in HALF_YEAR_WINDOWS.items():
        rows.append(_metrics(label, split, start, end, gross, cost, net, turnover))
    return rows


def _portfolio_returns(weights: pd.DataFrame, ret: pd.DataFrame, close: pd.DataFrame, fee_fn):
    gross = (weights * ret.shift(-1)).sum(axis=1).fillna(0.0)
    cost = fee_fn(weights, close, BOOK)
    net = gross - cost
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    return gross, cost, net, turnover


def _base_config() -> dict:
    base = json.loads((OUT_DIR / "validation_config_base.json").read_text())
    base["data"]["universe_path"] = _rel(FULL_AUCT_UNIVERSE)
    base["data"]["universe_filter"] = {"method": "coverage", "threshold": 0.0}
    base["book"] = BOOK
    return base


def _load_alpha_signals(base: dict):
    uni, dates, tickers, mats, close, ret, classifications, groups = _load_universe_and_matrices(base, root=ROOT)
    rows, train_sharpes = _load_alphas(base, root=ROOT)
    engine = FastExpressionEngine(data_fields=mats, groups=groups)
    alpha_signals: dict[int, pd.DataFrame] = {}
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
    return rows, train_sharpes, alpha_signals, uni, mats, close, ret, dates, tickers


def _combined_targets(base: dict, alpha_signals: dict[int, pd.DataFrame], mats, uni, ret, dates, tickers, train_sharpes):
    targets = {}
    for name, (fn, params) in COMBINERS.items():
        local = {**params, "signals_are_preprocessed": True}
        if name == "topn_train":
            local["train_sharpes"] = train_sharpes
        combined = fn(alpha_signals, mats, uni, ret, **local)
        targets[name] = _post_combiner(combined, base, dates, tickers)
    return targets


def _data_quality(uni: pd.DataFrame, mats: dict[str, pd.DataFrame]) -> pd.DataFrame:
    close = mats["close"]
    high = mats["high"]
    low = mats["low"]
    vwap = mats.get("vwap")
    rows = []
    for split, (start, end) in SPLITS.items():
        active = uni.loc[start:end].astype(bool)
        cells = int(active.to_numpy().sum())
        active_counts = active.sum(axis=1)
        c = close.loc[start:end].where(active)
        h = high.loc[start:end].where(active)
        l = low.loc[start:end].where(active)
        valid_hloc = c.notna() & h.notna() & l.notna()
        hloc_bad = valid_hloc & ((c > h * 1.0001) | (c < l * 0.9999) | (h < l))
        row = {
            "split": split,
            "start": start,
            "end": end,
            "active_cells": cells,
            "median_active_names": float(active_counts.median()),
            "min_active_names": int(active_counts.min()),
            "max_active_names": int(active_counts.max()),
            "close_coverage": float(c.notna().sum().sum() / cells) if cells else float("nan"),
            "hloc_violation_rate": float(hloc_bad.sum().sum() / max(int(valid_hloc.sum().sum()), 1)),
        }
        if vwap is not None:
            vw = vwap.loc[start:end].where(active)
            valid_vwap = vw.notna() & h.notna() & l.notna()
            vwap_bad = valid_vwap & ((vw > h * 1.05) | (vw < l * 0.95))
            vwap_ratio = (vw / c).replace([np.inf, -np.inf], np.nan).stack()
            row.update({
                "vwap_coverage": float(vw.notna().sum().sum() / cells) if cells else float("nan"),
                "vwap_far_outside_hl_rate": float(vwap_bad.sum().sum() / max(int(valid_vwap.sum().sum()), 1)),
                "vwap_close_ratio_p01": float(vwap_ratio.quantile(0.01)) if len(vwap_ratio) else float("nan"),
                "vwap_close_ratio_p99": float(vwap_ratio.quantile(0.99)) if len(vwap_ratio) else float("nan"),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = _base_config()
    fee_fn = make_cost_fn(base["fees"]["model"], base["fees"]["params"], bars_per_year=BARS_PER_YEAR)

    print("loading AUCT alphas and full universe", flush=True)
    rows, train_sharpes, alpha_signals, uni, mats, close, ret, dates, tickers = _load_alpha_signals(base)
    print(f"alphas={len(rows)} tickers={len(tickers)} bars={len(dates)}", flush=True)

    per_alpha_rows = []
    half_rows = []
    for aid, _expr in rows:
        label = f"alpha_{aid}"
        gross, cost, net, turnover = _portfolio_returns(alpha_signals[aid], ret, close, fee_fn)
        per_alpha_rows.extend(_series_metrics(label, gross, cost, net, turnover))
        half_rows.extend(_window_metrics(label, gross, cost, net, turnover))

    targets = _combined_targets(base, alpha_signals, mats, uni, ret, dates, tickers, train_sharpes)
    layer_rows = []
    for name, weights in targets.items():
        gross, cost, net, turnover = _portfolio_returns(weights, ret, close, fee_fn)
        layer_rows.extend(_series_metrics(f"{name}_no_qp", gross, cost, net, turnover))
        half_rows.extend(_window_metrics(f"{name}_no_qp", gross, cost, net, turnover))

    qpret_path = OUT_DIR / "test_selected_strategy_returns.parquet"
    if qpret_path.exists():
        qpret = pd.read_parquet(qpret_path)
        label = "ic_wt_qp_styleppca_s10_l2_k50"
        gross = qpret[f"{label}_gross"]
        cost = qpret[f"{label}_cost"]
        net = qpret[f"{label}_net"]
        turnover = qpret[f"{label}_turnover"]
        layer_rows.extend(_series_metrics(label, gross, cost, net, turnover))
        half_rows.extend(_window_metrics(label, gross, cost, net, turnover))

    per_alpha = pd.DataFrame(per_alpha_rows)
    layers = pd.DataFrame(layer_rows)
    half = pd.DataFrame(half_rows)
    dq = _data_quality(uni, mats)

    per_alpha.to_csv(OUT_DIR / "test_decay_per_alpha_metrics.csv", index=False)
    layers.to_csv(OUT_DIR / "test_decay_layer_metrics.csv", index=False)
    half.to_csv(OUT_DIR / "test_decay_halfyear_metrics.csv", index=False)
    dq.to_csv(OUT_DIR / "test_decay_data_quality.csv", index=False)

    print("\n=== layer metrics ===", flush=True)
    print(layers.sort_values(["split", "SR_net"], ascending=[True, False]).to_markdown(index=False, floatfmt=".3f"), flush=True)
    print("\n=== per-alpha PROD_TEST ===", flush=True)
    print(
        per_alpha[per_alpha["split"].eq("PROD_TEST")]
        .sort_values("SR_net", ascending=False)
        .to_markdown(index=False, floatfmt=".3f"),
        flush=True,
    )
    print("\n=== half-year selected layers ===", flush=True)
    show = half[half["series"].isin(["ic_wt_no_qp", "ic_wt_qp_styleppca_s10_l2_k50"])]
    print(show.to_markdown(index=False, floatfmt=".3f"), flush=True)
    print("\n=== data quality ===", flush=True)
    print(dq.to_markdown(index=False, floatfmt=".6f"), flush=True)


if __name__ == "__main__":
    main()
