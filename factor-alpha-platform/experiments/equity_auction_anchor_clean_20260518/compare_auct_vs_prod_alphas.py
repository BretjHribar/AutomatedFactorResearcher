"""Compare AUCT research alphas against production IB MOC alphas.

This diagnostic uses existing project libraries for expression evaluation,
subindustry preprocessing, portfolio returns, and IB fee modeling. It evaluates:

  - production 45-alpha IB MOC set on its native MCAP_100M_500M universe
  - production 45-alpha set on the AUCT universe
  - AUCT six saved research alphas on the AUCT universe
  - AUCT six saved research alphas on production MCAP_100M_500M

The goal is to separate alpha-quality problems from universe, fee, and
pipeline issues.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
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
from src.pipeline.runner import _load_alphas, _load_universe_and_matrices, _post_combiner  # noqa: E402
from src.portfolio.combiners import combiner_equal, combiner_ic_weighted  # noqa: E402
from src.portfolio.preprocessing import apply_preprocess  # noqa: E402


BARS_PER_YEAR = 252
BOOK = 500_000.0
PROD_FILTER = (
    "archived=0 AND asset_class='equities' AND "
    "(notes LIKE '%SMALLCAP_D0_v2%' OR notes LIKE '%SMALLCAP_D0_v3%')"
)
SPLITS = {
    "AUCT_TRAIN": ("2016-01-04", "2023-01-01"),
    "AUCT_VAL": ("2023-01-01", "2024-07-01"),
    "AUCT_TEST": ("2024-07-02", "2026-05-14"),
    "PROD_TRAIN": ("2020-01-01", "2024-01-01"),
    "PROD_VAL": ("2024-01-01", "2025-04-01"),
    "PROD_TEST": ("2025-04-01", "2026-05-14"),
}
FIELD_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
FUNCTION_WORDS = {
    "abs", "add", "and", "correlation", "decay_exp", "decay_linear", "delay",
    "df_max", "divide", "greater", "if_else", "less", "log", "log_diff",
    "max", "min", "multiply", "negative", "or", "rank", "signed_power",
    "sma", "stddev", "subtract", "trade_when", "true_divide", "ts_delta",
    "ts_max", "ts_min", "ts_rank", "ts_zscore",
}


def _rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _ann_sr(s: pd.Series) -> float:
    clean = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) <= 1 or clean.std() <= 0:
        return float("nan")
    return float(clean.mean() / clean.std() * math.sqrt(BARS_PER_YEAR))


def _max_dd(s: pd.Series) -> float:
    clean = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if clean.empty:
        return float("nan")
    eq = (1.0 + clean).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def _metrics(
    *,
    group: str,
    item: str,
    split: str,
    gross: pd.Series,
    cost: pd.Series,
    net: pd.Series,
    turnover: pd.Series,
) -> dict:
    start, end = SPLITS[split]
    g = gross.loc[start:end]
    c = cost.loc[start:end]
    n = net.loc[start:end]
    t = turnover.loc[start:end]
    return {
        "group": group,
        "item": item,
        "split": split,
        "start": start,
        "end": end,
        "n_bars": int(n.replace([np.inf, -np.inf], np.nan).dropna().shape[0]),
        "SR_gross": _ann_sr(g),
        "SR_net": _ann_sr(n),
        "ret_ann_gross": float(g.mean() * BARS_PER_YEAR),
        "ret_ann_net": float(n.mean() * BARS_PER_YEAR),
        "vol_ann_net": float(n.std() * math.sqrt(BARS_PER_YEAR)),
        "cost_ann": float(c.mean() * BARS_PER_YEAR),
        "turnover": float(t.mean()),
        "cost_per_turnover": float(c.mean() * BARS_PER_YEAR / t.mean()) if t.mean() > 0 else float("nan"),
        "max_dd_net": _max_dd(n),
    }


def _portfolio_returns(weights: pd.DataFrame, ret: pd.DataFrame, close: pd.DataFrame, fee_fn):
    gross = (weights * ret.shift(-1)).sum(axis=1).fillna(0.0)
    cost = fee_fn(weights, close, BOOK)
    net = gross - cost
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    return gross, cost, net, turnover


def _prod_base_config() -> dict:
    cfg = json.loads((ROOT / "prod/config/research_equity.json").read_text())
    cfg["alpha_source"]["filter_sql"] = PROD_FILTER
    cfg["alpha_source"]["train_sharpe_table"] = None
    cfg["alpha_source"]["train_sharpe_column"] = None
    cfg["data"]["universe_filter"] = {"method": "coverage", "threshold": 0.0}
    cfg["book"] = BOOK
    return cfg


def _auct_base_config() -> dict:
    cfg = json.loads((OUT_DIR / "validation_config_base.json").read_text())
    cfg["data"]["universe_path"] = _rel(FULL_AUCT_UNIVERSE)
    cfg["data"]["universe_filter"] = {"method": "coverage", "threshold": 0.0}
    cfg["book"] = BOOK
    return cfg


def _load_auct_rows() -> list[tuple[str, str]]:
    saved = pd.read_csv(OUT_DIR / "saved_alphas.csv")
    return [(f"auct_saved_{int(row.alpha_id)}", row.expression) for row in saved.itertuples(index=False)]


def _load_prod_rows() -> list[tuple[int, str]]:
    con = sqlite3.connect(ROOT / "data/alpha_results.db")
    try:
        rows = pd.read_sql(
            f"SELECT id, expression FROM alphas WHERE {PROD_FILTER} ORDER BY id",
            con,
        )
    finally:
        con.close()
    return [(int(row.id), str(row.expression)) for row in rows.itertuples(index=False)]


def _eval_rows(
    rows: list[tuple[int | str, str]],
    *,
    base: dict,
) -> tuple[dict[int | str, pd.DataFrame], pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict]:
    uni, dates, tickers, mats, close, ret, classifications, groups = _load_universe_and_matrices(base, root=ROOT)
    engine = FastExpressionEngine(data_fields=mats, groups=groups)
    signals = {}
    for aid, expr in rows:
        raw = engine.evaluate(expr).reindex(index=dates, columns=tickers)
        signals[aid] = apply_preprocess(
            raw,
            universe_mask=True,
            universe=uni,
            demean_method="subindustry",
            classifications=classifications,
            normalize="l1",
            clip_max_w=0.02,
        )
    return signals, uni, mats, close, ret, dates, tickers, classifications


def _combiner_targets(
    signals: dict[int | str, pd.DataFrame],
    *,
    base: dict,
    mats: dict,
    uni: pd.DataFrame,
    ret: pd.DataFrame,
    dates: pd.Index,
    tickers: list[str],
) -> dict[str, pd.DataFrame]:
    return {
        "equal_no_qp": _post_combiner(
            combiner_equal(signals, mats, uni, ret, max_wt=0.02, signals_are_preprocessed=True),
            base,
            dates,
            tickers,
        ),
        "ic_wt_no_qp": _post_combiner(
            combiner_ic_weighted(signals, mats, uni, ret, signals_are_preprocessed=True),
            base,
            dates,
            tickers,
        ),
    }


def _field_summary(rows: list[tuple[int | str, str]], group: str) -> list[dict]:
    out = []
    for aid, expr in rows:
        fields = sorted(
            {
                tok
                for tok in FIELD_RE.findall(expr)
                if tok not in FUNCTION_WORDS and not tok.isupper() and not tok.replace("_", "").isdigit()
            }
        )
        out.append({
            "group": group,
            "item": str(aid),
            "n_fields": len(fields),
            "fields": ",".join(fields),
            "uses_fundamental": int(any(f in fields for f in {
                "assets", "cashflow_op", "net_stock_issuance", "roic",
            })),
            "uses_price_volume_only": int(all(f in {
                "adv20", "adv60", "close", "dollars_traded", "high", "low", "volume", "vwap",
            } for f in fields)),
        })
    return out


def _universe_diag(label: str, uni: pd.DataFrame, mats: dict, classifications: dict) -> list[dict]:
    close = mats["close"].reindex(index=uni.index, columns=uni.columns)
    volume = mats.get("volume", pd.DataFrame(index=uni.index, columns=uni.columns)).reindex(index=uni.index, columns=uni.columns)
    dollars = mats.get("dollars_traded")
    if dollars is None:
        dollars = close * volume
    else:
        dollars = dollars.reindex(index=uni.index, columns=uni.columns)
    sub = classifications.get("subindustry")
    rows = []
    for split, (start, end) in SPLITS.items():
        active = uni.loc[start:end].astype(bool)
        if active.empty:
            continue
        c = close.loc[start:end].where(active)
        d = dollars.loc[start:end].where(active)
        active_counts = active.sum(axis=1)
        sub_counts = []
        singleton_counts = []
        if sub is not None:
            sub_series = pd.Series(sub)
            for dt in active.index:
                names = active.columns[active.loc[dt].values]
                counts = sub_series.reindex(names).dropna().value_counts()
                sub_counts.append(len(counts))
                singleton_counts.append(int((counts == 1).sum()))
        rows.append({
            "universe": label,
            "split": split,
            "median_active_names": float(active_counts.median()),
            "mean_active_names": float(active_counts.mean()),
            "median_price": float(c.stack().median()),
            "p25_price": float(c.stack().quantile(0.25)),
            "p75_price": float(c.stack().quantile(0.75)),
            "median_dollars_traded": float(d.stack().median()),
            "p25_dollars_traded": float(d.stack().quantile(0.25)),
            "p75_dollars_traded": float(d.stack().quantile(0.75)),
            "mean_subindustries": float(np.mean(sub_counts)) if sub_counts else float("nan"),
            "mean_singleton_subindustries": float(np.mean(singleton_counts)) if singleton_counts else float("nan"),
        })
    return rows


def _overlap_diag(prod_uni: pd.DataFrame, auct_uni: pd.DataFrame) -> pd.DataFrame:
    dates = prod_uni.index.intersection(auct_uni.index)
    cols = prod_uni.columns.union(auct_uni.columns)
    p = prod_uni.reindex(index=dates, columns=cols).fillna(False).astype(bool)
    a = auct_uni.reindex(index=dates, columns=cols).fillna(False).astype(bool)
    rows = []
    for split, (start, end) in SPLITS.items():
        pp = p.loc[start:end]
        aa = a.loc[start:end]
        inter = pp & aa
        union = pp | aa
        rows.append({
            "split": split,
            "prod_active_mean": float(pp.sum(axis=1).mean()),
            "auct_active_mean": float(aa.sum(axis=1).mean()),
            "overlap_mean": float(inter.sum(axis=1).mean()),
            "jaccard_mean": float((inter.sum(axis=1) / union.sum(axis=1).replace(0, np.nan)).mean()),
            "prod_share_in_auct": float((inter.sum(axis=1) / pp.sum(axis=1).replace(0, np.nan)).mean()),
            "auct_share_in_prod": float((inter.sum(axis=1) / aa.sum(axis=1).replace(0, np.nan)).mean()),
        })
    return pd.DataFrame(rows)


def _corr_summary(group: str, returns: dict[str, pd.Series]) -> list[dict]:
    rows = []
    frame = pd.DataFrame(returns)
    for split, (start, end) in SPLITS.items():
        sub = frame.loc[start:end].dropna(how="all")
        corr = sub.corr()
        vals = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
        rows.append({
            "group": group,
            "split": split,
            "n_pairs": int(vals.shape[0]),
            "mean_pair_corr": float(vals.mean()) if len(vals) else float("nan"),
            "median_pair_corr": float(vals.median()) if len(vals) else float("nan"),
            "p75_pair_corr": float(vals.quantile(0.75)) if len(vals) else float("nan"),
            "max_pair_corr": float(vals.max()) if len(vals) else float("nan"),
        })
    return rows


def _run_group(
    group: str,
    rows: list[tuple[int | str, str]],
    base: dict,
) -> tuple[list[dict], list[dict], list[dict], pd.DataFrame, dict]:
    fee_fn = make_cost_fn(base["fees"]["model"], base["fees"]["params"], bars_per_year=BARS_PER_YEAR)
    signals, uni, mats, close, ret, dates, tickers, classifications = _eval_rows(rows, base=base)
    metric_rows = []
    returns_for_corr = {}
    for aid, weights in signals.items():
        gross, cost, net, turnover = _portfolio_returns(weights, ret, close, fee_fn)
        returns_for_corr[str(aid)] = net
        for split in SPLITS:
            metric_rows.append(_metrics(group=group, item=str(aid), split=split, gross=gross, cost=cost, net=net, turnover=turnover))
    combo_rows = []
    for combo_name, weights in _combiner_targets(signals, base=base, mats=mats, uni=uni, ret=ret, dates=dates, tickers=tickers).items():
        gross, cost, net, turnover = _portfolio_returns(weights, ret, close, fee_fn)
        for split in SPLITS:
            combo_rows.append(_metrics(group=group, item=combo_name, split=split, gross=gross, cost=cost, net=net, turnover=turnover))
    corr_rows = _corr_summary(group, returns_for_corr)
    uni_rows = _universe_diag(group, uni, mats, classifications)
    return metric_rows, combo_rows, corr_rows, pd.DataFrame(uni_rows), {"uni": uni}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prod_base = _prod_base_config()
    auct_base = _auct_base_config()
    prod_rows = _load_prod_rows()
    auct_rows = _load_auct_rows()
    print(f"prod_alphas={len(prod_rows)} auct_alphas={len(auct_rows)}", flush=True)

    all_metrics = []
    all_combos = []
    all_corr = []
    all_uni = []
    universes = {}
    specs = [
        ("prod_alphas_on_prod_universe", prod_rows, prod_base),
        ("prod_alphas_on_auct_universe", prod_rows, auct_base),
        ("auct_alphas_on_auct_universe", auct_rows, auct_base),
        ("auct_alphas_on_prod_universe", auct_rows, prod_base),
    ]
    for group, rows, base in specs:
        print(f"=== {group} ===", flush=True)
        metric_rows, combo_rows, corr_rows, uni_diag, extra = _run_group(group, rows, base)
        all_metrics.extend(metric_rows)
        all_combos.extend(combo_rows)
        all_corr.extend(corr_rows)
        all_uni.append(uni_diag)
        universes[group] = extra["uni"]

    field_rows = _field_summary(prod_rows, "prod_alphas") + _field_summary(auct_rows, "auct_alphas")
    per_alpha = pd.DataFrame(all_metrics)
    combos = pd.DataFrame(all_combos)
    corr = pd.DataFrame(all_corr)
    universe_diag = pd.concat(all_uni, ignore_index=True)
    field_summary = pd.DataFrame(field_rows)

    group_summary = per_alpha.groupby(["group", "split"]).agg(
        n=("item", "nunique"),
        median_SR_gross=("SR_gross", "median"),
        median_SR_net=("SR_net", "median"),
        mean_SR_net=("SR_net", "mean"),
        positive_gross=("SR_gross", lambda s: int((s > 0).sum())),
        positive_net=("SR_net", lambda s: int((s > 0).sum())),
        median_ret_ann_net=("ret_ann_net", "median"),
        median_cost_ann=("cost_ann", "median"),
        median_turnover=("turnover", "median"),
        median_cost_per_turnover=("cost_per_turnover", "median"),
        median_max_dd_net=("max_dd_net", "median"),
    ).reset_index()

    overlap = _overlap_diag(universes["prod_alphas_on_prod_universe"], universes["prod_alphas_on_auct_universe"])

    per_alpha.to_csv(OUT_DIR / "auct_vs_prod_per_alpha_cross_universe.csv", index=False)
    combos.to_csv(OUT_DIR / "auct_vs_prod_combiner_cross_universe.csv", index=False)
    group_summary.to_csv(OUT_DIR / "auct_vs_prod_group_summary.csv", index=False)
    corr.to_csv(OUT_DIR / "auct_vs_prod_alpha_return_corr_summary.csv", index=False)
    universe_diag.to_csv(OUT_DIR / "auct_vs_prod_universe_diagnostics.csv", index=False)
    overlap.to_csv(OUT_DIR / "auct_vs_prod_universe_overlap.csv", index=False)
    field_summary.to_csv(OUT_DIR / "auct_vs_prod_field_summary.csv", index=False)

    print("\n=== group summary PROD_TEST ===", flush=True)
    print(group_summary[group_summary["split"].eq("PROD_TEST")].to_markdown(index=False, floatfmt=".4f"), flush=True)
    print("\n=== combiner summary PROD_TEST ===", flush=True)
    print(combos[combos["split"].eq("PROD_TEST")].to_markdown(index=False, floatfmt=".4f"), flush=True)
    print("\n=== universe overlap PROD_TEST ===", flush=True)
    print(overlap[overlap["split"].eq("PROD_TEST")].to_markdown(index=False, floatfmt=".4f"), flush=True)


if __name__ == "__main__":
    main()
