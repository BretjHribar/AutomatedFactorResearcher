"""Validation-only combiner sweep for AUCT_ANCHOR saved alphas.

Uses the project pipeline runner end to end:
- expression evaluation through FastExpressionEngine
- subindustry preprocessing through src.portfolio.preprocessing
- library combiners through src.portfolio.combiners
- QP execution through src.portfolio.qp
- IB MOC fee model through src.pipeline.fees

The run truncates universes at VAL_END so no post-validation/test rows are
loaded by the validation sweep.
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
VAL_UNIVERSE = EXP_DIR / "universes" / "AUCT_ANCHOR_MCAP90M_550M_DAILY_TRAIN_VAL_201601_202407.parquet"
PROD_VAL_UNIVERSE = OUT_DIR / "MCAP_100M_500M_TRAIN_VAL_201601_202407.parquet"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.runner import merge_overrides, run  # noqa: E402


LABEL = "AUCT_ANCHOR_MCAP90_550_D0_CLEAN_S5_F5_20260518"
TRAIN_START = "2016-01-01"
TRAIN_END = "2023-01-01"
VAL_START = "2023-01-01"
VAL_END = "2024-07-01"
BARS_PER_YEAR = 252


COMBINERS = [
    ("equal", {"name": "equal", "params": {"max_wt": 0.02}}),
    ("adaptive", {"name": "adaptive", "params": {}}),
    ("risk_par", {"name": "risk_par", "params": {}}),
    ("billions", {"name": "billions", "params": {}}),
    ("ic_wt", {"name": "ic_wt", "params": {}}),
    ("sharpe_wt", {"name": "sharpe_wt", "params": {}}),
    ("topn_sharpe", {"name": "topn_sharpe", "params": {}}),
    ("topn_train", {"name": "topn_train", "params": {}}),
]


def _rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _finite(value: float) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except Exception:
        return None
    return f if math.isfinite(f) else None


def _make_train_val_universe(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    uni = pd.read_parquet(source)
    if not isinstance(uni.index, pd.DatetimeIndex):
        uni.index = pd.to_datetime(uni.index)
    clipped = uni.loc[TRAIN_START:VAL_END].astype(bool)
    clipped.to_parquet(target)


def _base_cfg() -> dict:
    return {
        "market": "equity",
        "interval": "1d",
        "annualization": {"bars_per_year": BARS_PER_YEAR},
        "data": {
            "matrices_dir": "data/fmp_cache/matrices",
            "universe_path": _rel(VAL_UNIVERSE),
            "universe_filter": {"method": "coverage", "threshold": 0.0},
            "returns_source": "compute_from_close",
        },
        "alpha_source": {
            "db_path": "data/ib_alphas.db",
            "table": "alphas",
            "filter_sql": (
                "archived=0 AND asset_class='equities_ib' "
                f"AND category='{LABEL}'"
            ),
            "train_sharpe_table": "evaluations",
            "train_sharpe_column": "sharpe_is",
        },
        "preprocessing": {
            "universe_mask": True,
            "demean_method": "subindustry",
            "subindustry_field": "subindustry",
            "normalize": "l1",
            "clip_max_w": 0.02,
        },
        "combiner": {"name": "equal", "params": {"max_wt": 0.02}},
        "post_combiner": {
            "renormalize_l1": True,
            "clip_max_w": 0.02,
        },
        "risk_model": {
            "name": "diagonal",
            "params": {"factor_window": 126, "n_pca_factors": 5, "vol_window": 60},
        },
        "qp": {
            "enabled": True,
            "lambda_risk": 5.0,
            "kappa_tc": 30.0,
            "max_w": 0.02,
            "dollar_neutral": True,
            "max_gross_leverage": 1.0,
            "commission_per_share": 0.0045,
            "impact_bps": 0.0,
            "adv_cap": None,
        },
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
        "splits": {"train_end": TRAIN_END, "val_end": VAL_END},
        "book": 500000,
    }


def _ann_metrics(series: pd.Series) -> dict[str, float | int | None]:
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return {"n_bars": 0, "SR": None, "ret_ann": None, "max_dd": None}
    std = float(s.std())
    sr = float(s.mean() / std * np.sqrt(BARS_PER_YEAR)) if std > 0 else None
    eq = (1.0 + s).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min()) if len(eq) else None
    return {
        "n_bars": int(len(s)),
        "SR": _finite(sr),
        "ret_ann": _finite(float(s.mean()) * BARS_PER_YEAR),
        "max_dd": _finite(dd),
    }


def _corr(a: pd.Series, b: pd.Series, start: str, end: str) -> float | None:
    aa = a.loc[start:end].replace([np.inf, -np.inf], np.nan)
    bb = b.loc[start:end].replace([np.inf, -np.inf], np.nan)
    idx = aa.dropna().index.intersection(bb.dropna().index)
    if len(idx) < 20:
        return None
    val = float(aa.loc[idx].corr(bb.loc[idx]))
    return _finite(val)


def _avg_weight_corr(a: pd.DataFrame, b: pd.DataFrame, start: str, end: str) -> float | None:
    aa = a.loc[start:end]
    bb = b.loc[start:end]
    dates = aa.index.intersection(bb.index)
    cols = aa.columns.intersection(bb.columns)
    vals = []
    for dt in dates:
        x = aa.loc[dt, cols].fillna(0.0)
        y = bb.loc[dt, cols].fillna(0.0)
        if float(x.std()) <= 0 or float(y.std()) <= 0:
            continue
        c = float(x.corr(y))
        if math.isfinite(c):
            vals.append(c)
    return _finite(float(np.mean(vals))) if vals else None


def _prod_reference() -> tuple[object, dict]:
    prod_cfg_path = ROOT / "prod" / "config" / "research_equity.json"
    prod_base = json.loads(prod_cfg_path.read_text())
    _make_train_val_universe(ROOT / "data" / "fmp_cache" / "universes" / "MCAP_100M_500M.parquet", PROD_VAL_UNIVERSE)
    cfg = merge_overrides(
        prod_base,
        {
            "data": {
                "universe_path": _rel(PROD_VAL_UNIVERSE),
                "universe_filter": {"method": "coverage", "threshold": 0.0},
            },
            "splits": {"train_end": TRAIN_END, "val_end": VAL_END},
            "book": 500000,
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
                **prod_base.get("qp", {}),
                "enabled": True,
                "commission_per_share": 0.0045,
                "impact_bps": 0.0,
            },
        },
    )
    res = run(cfg, verbose=False)
    return res, cfg


def _aipt_reference() -> tuple[pd.Series | None, dict]:
    summary_path = ROOT / "experiments" / "results" / "aipt_live_refresh_20260514" / "aipt_stepwise_summary.csv"
    returns_path = (
        ROOT
        / "experiments"
        / "results"
        / "aipt_live_refresh_20260514"
        / "equity_smallcap_d0__prox_l1_gross1_cap_fee__P256__z0p001__seed1__maxw0p02__tau5__turncap0__blend1__qpscale1__qprisk0.returns.parquet"
    )
    ref = {
        "source_summary": _rel(summary_path),
        "source_returns": _rel(returns_path),
    }
    if summary_path.exists():
        s = pd.read_csv(summary_path)
        val = s[s["split"].astype(str).eq("VAL")]
        if not val.empty:
            row = val.iloc[0]
            ref.update(
                {
                    "reported_val_n_bars": int(row["n_bars"]),
                    "reported_val_SR_net": _finite(row["SR_net"]),
                    "reported_val_ret_ann_net": _finite(row["ret_ann_net"]),
                    "reported_val_max_dd_net": _finite(row["max_dd_net"]),
                    "reported_val_turnover": _finite(row["turnover_per_bar"]),
                    "reported_val_cost_ann": _finite(row["cost_ann"]),
                }
            )
    if returns_path.exists():
        r = pd.read_parquet(returns_path)
        if not isinstance(r.index, pd.DatetimeIndex):
            r.index = pd.to_datetime(r.index)
        ref["calendar_val_net_metrics"] = _ann_metrics(r["net"].loc[VAL_START:VAL_END])
        return r["net"], ref
    return None, ref


def _row_for(label: str, res, prod_res, aipt_net: pd.Series | None, elapsed: float) -> dict:
    m = res.metrics
    val = m["VAL"]
    return {
        "cell": label,
        "combiner": res.config["combiner"]["name"],
        "risk_model": res.config["risk_model"]["name"],
        "qp_enabled": bool(res.config.get("qp", {}).get("enabled")),
        "alpha_signals_n": int(res.alpha_signals_n),
        "universe_size": int(res.universe_size),
        "val_n_bars": int(val["n_bars"]),
        "val_SR_gross": _finite(val["SR_gross"]),
        "val_SR_net": _finite(val["SR_net"]),
        "val_ret_ann_net": _finite(val["ret_ann_net"]),
        "val_max_dd_net": _finite(val["max_dd_net"]),
        "turnover_per_bar": _finite(m["_turnover_per_bar"]),
        "cost_ann": _finite(float(res.cost.loc[VAL_START:VAL_END].mean()) * BARS_PER_YEAR),
        "corr_net_vs_ib_moc_val": _corr(res.net_pnl, prod_res.net_pnl, VAL_START, VAL_END),
        "avg_weight_corr_vs_ib_moc_val": _avg_weight_corr(res.weights, prod_res.weights, VAL_START, VAL_END),
        "corr_net_vs_aipt_val": _corr(res.net_pnl, aipt_net, VAL_START, VAL_END) if aipt_net is not None else None,
        "elapsed_sec": _finite(elapsed),
    }


def _reference_rows(prod_res, aipt_net: pd.Series | None, aipt_ref: dict) -> list[dict]:
    prod_val = prod_res.metrics["VAL"]
    rows = [
        {
            "reference": "ib_moc_equity",
            "source": "prod/config/research_equity.json recomputed on train+val only",
            "val_n_bars": int(prod_val["n_bars"]),
            "val_SR_gross": _finite(prod_val["SR_gross"]),
            "val_SR_net": _finite(prod_val["SR_net"]),
            "val_ret_ann_net": _finite(prod_val["ret_ann_net"]),
            "val_max_dd_net": _finite(prod_val["max_dd_net"]),
            "turnover_per_bar": _finite(prod_res.metrics["_turnover_per_bar"]),
            "cost_ann": _finite(float(prod_res.cost.loc[VAL_START:VAL_END].mean()) * BARS_PER_YEAR),
            "calendar_val_SR_net": _finite(_ann_metrics(prod_res.net_pnl.loc[VAL_START:VAL_END])["SR"]),
        },
        {
            "reference": "aipt_smallcap_d0_prox_l1_tau5",
            "source": aipt_ref.get("source_summary"),
            "val_n_bars": aipt_ref.get("reported_val_n_bars"),
            "val_SR_gross": None,
            "val_SR_net": aipt_ref.get("reported_val_SR_net"),
            "val_ret_ann_net": aipt_ref.get("reported_val_ret_ann_net"),
            "val_max_dd_net": aipt_ref.get("reported_val_max_dd_net"),
            "turnover_per_bar": aipt_ref.get("reported_val_turnover"),
            "cost_ann": aipt_ref.get("reported_val_cost_ann"),
            "calendar_val_SR_net": (
                aipt_ref.get("calendar_val_net_metrics", {}).get("SR")
                if isinstance(aipt_ref.get("calendar_val_net_metrics"), dict)
                else None
            ),
        },
    ]
    if aipt_net is not None:
        rows[0]["corr_net_vs_aipt_calendar_val"] = _corr(prod_res.net_pnl, aipt_net, VAL_START, VAL_END)
    return rows


def _write_log(rows: list[dict], ref_rows: list[dict]) -> None:
    log_path = EXP_DIR / "research_log.md"
    best = max(rows, key=lambda r: r["val_SR_net"] if r["val_SR_net"] is not None else -1e9)
    lines = [
        "",
        "## Validation Combiner Sweep",
        "",
        f"Run timestamp UTC: {pd.Timestamp.utcnow().isoformat()}",
        f"Validation window: {VAL_START} through {VAL_END}; no rows after validation end loaded for this sweep.",
        "Implementation: shared `src.pipeline.runner` with library combiners, subindustry preprocessing, QP enabled, IB per-share MOC fees, `impact_bps=0.0`, book `$500,000`.",
        f"Saved outputs: `{_rel(OUT_DIR / 'validation_combiner_results.csv')}`, `{_rel(OUT_DIR / 'validation_reference_comparison.csv')}`, `{_rel(OUT_DIR / 'validation_returns.parquet')}`.",
        "",
        f"Best validation combiner: `{best['cell']}` with net SR={best['val_SR_net']:+.3f}, turnover={best['turnover_per_bar']:.3f}, cost_ann={best['cost_ann']:.3f}.",
        "",
        "Reference rows:",
    ]
    for row in ref_rows:
        sr = row.get("val_SR_net")
        lines.append(f"- {row['reference']}: VAL net SR={sr:+.3f}" if sr is not None else f"- {row['reference']}: VAL net SR=n/a")
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    source_uni = EXP_DIR / "universes" / "AUCT_ANCHOR_MCAP90M_550M_DAILY.parquet"
    _make_train_val_universe(source_uni, VAL_UNIVERSE)
    base = _base_cfg()
    (OUT_DIR / "validation_config_base.json").write_text(json.dumps(base, indent=2), encoding="utf-8")

    print("=== loading references ===", flush=True)
    prod_res, prod_cfg = _prod_reference()
    aipt_net, aipt_ref = _aipt_reference()
    (OUT_DIR / "validation_prod_reference_config.json").write_text(json.dumps(prod_cfg, indent=2), encoding="utf-8")
    (OUT_DIR / "validation_aipt_reference.json").write_text(json.dumps(aipt_ref, indent=2, default=str), encoding="utf-8")

    rows = []
    returns = {
        "ib_moc_equity_net": prod_res.net_pnl,
        "ib_moc_equity_gross": prod_res.gross_pnl,
    }
    if aipt_net is not None:
        returns["aipt_smallcap_net"] = aipt_net

    for label, combiner in COMBINERS:
        cfg = merge_overrides(base, {"combiner": combiner})
        print(f"=== running {label} ===", flush=True)
        t0 = time.time()
        res = run(cfg, verbose=True)
        elapsed = time.time() - t0
        rows.append(_row_for(label, res, prod_res, aipt_net, elapsed))
        returns[f"{label}_net"] = res.net_pnl
        returns[f"{label}_gross"] = res.gross_pnl

        val = res.metrics["VAL"]
        print(
            f"{label}: VAL SR_g={val['SR_gross']:+.3f} "
            f"SR_n={val['SR_net']:+.3f} "
            f"ret_n={val['ret_ann_net'] * 100:+.1f}% "
            f"TO={res.metrics['_turnover_per_bar']:.3f} "
            f"cost_ann={rows[-1]['cost_ann']:.4f}",
            flush=True,
        )

    results_df = pd.DataFrame(rows).sort_values("val_SR_net", ascending=False)
    results_df.to_csv(OUT_DIR / "validation_combiner_results.csv", index=False)

    ref_rows = _reference_rows(prod_res, aipt_net, aipt_ref)
    pd.DataFrame(ref_rows).to_csv(OUT_DIR / "validation_reference_comparison.csv", index=False)

    ret_df = pd.DataFrame(returns).sort_index()
    ret_df.to_parquet(OUT_DIR / "validation_returns.parquet")

    summary = {
        "label": LABEL,
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "val_start": VAL_START,
        "val_end": VAL_END,
        "book": 500000,
        "fees": base["fees"],
        "qp": base["qp"],
        "risk_model": base["risk_model"],
        "preprocessing": base["preprocessing"],
        "n_combiner_cells": len(COMBINERS),
        "best_by_val_SR_net": results_df.iloc[0].to_dict() if not results_df.empty else None,
        "references": ref_rows,
    }
    (OUT_DIR / "validation_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _write_log(rows, ref_rows)

    print("\n=== validation results sorted by net SR ===", flush=True)
    cols = [
        "cell",
        "val_SR_gross",
        "val_SR_net",
        "val_ret_ann_net",
        "turnover_per_bar",
        "cost_ann",
        "corr_net_vs_ib_moc_val",
        "corr_net_vs_aipt_val",
        "avg_weight_corr_vs_ib_moc_val",
    ]
    print(results_df[cols].to_string(index=False), flush=True)
    print("\n=== references ===", flush=True)
    print(pd.DataFrame(ref_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
