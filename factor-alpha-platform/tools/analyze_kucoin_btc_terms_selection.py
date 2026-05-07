"""Analyze and combine KuCoin BTC-terms discovery candidates.

The scanner ranks individual expressions. This script applies selection rules
that do not use TEST, prunes correlated factor PnL, combines candidates, and
reports TRAIN/VAL/TEST metrics. It also includes clearly labeled diagnostic
lookahead baskets so regime breaks are easy to spot without confusing them for
deployable research.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.operators.fastexpression import FastExpressionEngine
from tools.run_kucoin_btc_terms_discovery import (
    load_research_data,
    signal_to_portfolio,
    summarize_pnl,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _score_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.drop_duplicates("expression").reset_index(drop=True)
    if {"TRAIN_SR_net", "VAL_SR_net"}.issubset(out.columns):
        out["TRAIN_VAL_MIN"] = out[["TRAIN_SR_net", "VAL_SR_net"]].min(axis=1)
        out["TRAIN_VAL_MEAN"] = out[["TRAIN_SR_net", "VAL_SR_net"]].mean(axis=1)
        out["TRAIN_VAL_HARM"] = 2.0 / (
            1.0 / out["TRAIN_SR_net"].clip(lower=1e-9)
            + 1.0 / out["VAL_SR_net"].clip(lower=1e-9)
        )
    out["ABS_TRAIN"] = out["TRAIN_SR_net"].abs()
    return out


def build_candidate_sets(
    df: pd.DataFrame,
    top_n: int,
    *,
    allow_oos_selection: bool = False,
    allow_test_diagnostics: bool = False,
) -> dict[str, pd.DataFrame]:
    """Build candidate sets.

    Default selection is TRAIN-only. VAL/TEST-based sets require explicit
    diagnostic flags and must not be used for model promotion.
    """
    sets: dict[str, pd.DataFrame] = {}
    d = _score_frame(df)

    sets[f"top{top_n}_train"] = d[d["TRAIN_SR_net"] > 0].sort_values(
        ["TRAIN_SR_net"], ascending=False
    ).head(top_n)

    sets[f"top{top_n}_abs_train_signfit"] = d[d["ABS_TRAIN"] > 0].sort_values(
        ["ABS_TRAIN"], ascending=False
    ).head(top_n)

    if allow_oos_selection and {"TRAIN_SR_net", "VAL_SR_net"}.issubset(d.columns):
        sets[f"top{top_n}_train_val_min_DIAG_OOS_SELECTION"] = d[
            (d["TRAIN_SR_net"] > 0) & (d["VAL_SR_net"] > 0)
        ].sort_values(["TRAIN_VAL_MIN", "TRAIN_SR_net"], ascending=False).head(top_n)

        sets[f"top{top_n}_train_val_mean_DIAG_OOS_SELECTION"] = d[
            (d["TRAIN_SR_net"] > 0) & (d["VAL_SR_net"] > 0)
        ].sort_values(["TRAIN_VAL_MEAN", "TRAIN_SR_net"], ascending=False).head(top_n)

        sets[f"top{top_n}_train_gt05_val_confirm_DIAG_OOS_SELECTION"] = d[
            (d["TRAIN_SR_net"] > 0.5) & (d["VAL_SR_net"] > 0.5)
        ].sort_values(["VAL_SR_net", "TRAIN_SR_net"], ascending=False).head(top_n)

    if allow_test_diagnostics and {"VAL_SR_net", "TEST_SR_net", "VAL+TEST_SR_net"}.issubset(d.columns):
        sets[f"top{top_n}_val_DIAG_LOOKAHEAD"] = d.sort_values(
            ["VAL_SR_net", "TRAIN_SR_net"], ascending=False
        ).head(top_n)
        sets[f"top{top_n}_val_test_DIAG_LOOKAHEAD"] = d.sort_values(
            ["VAL+TEST_SR_net", "TRAIN_SR_net"], ascending=False
        ).head(top_n)
        sets[f"top{top_n}_test_DIAG_LOOKAHEAD"] = d.sort_values(
            ["TEST_SR_net", "VAL_SR_net"], ascending=False
        ).head(top_n)
    return {k: v for k, v in sets.items() if not v.empty}


def evaluate_candidate_weights(
    engine: FastExpressionEngine,
    expressions: list[str],
    universe: pd.DataFrame,
    *,
    max_weight: float,
) -> dict[str, pd.DataFrame]:
    weights: dict[str, pd.DataFrame] = {}
    for i, expr in enumerate(expressions, 1):
        try:
            raw = engine.evaluate(expr)
            weights[expr] = signal_to_portfolio(raw, universe, max_weight)
        except Exception as exc:
            log(f"skip expression {i}/{len(expressions)}: {type(exc).__name__}: {str(exc)[:80]}")
    return weights


def _portfolio_pnl(w: pd.DataFrame, returns: pd.DataFrame, cost_bps: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    common_idx = w.index.intersection(returns.index)
    common_cols = w.columns.intersection(returns.columns)
    w = w.loc[common_idx, common_cols].fillna(0.0)
    r = returns.loc[common_idx, common_cols].fillna(0.0)
    gross = (w * r.shift(-1)).sum(axis=1).fillna(0.0)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - turnover * cost_bps / 10000.0
    return gross, net, turnover


def _train_pnl(w: pd.DataFrame, returns: pd.DataFrame, train_start: str, train_end: str) -> pd.Series:
    gross, _, _ = _portfolio_pnl(w, returns, cost_bps=0.0)
    return gross.loc[pd.Timestamp(train_start):pd.Timestamp(train_end)].fillna(0.0)


def correlation_prune(
    rows: pd.DataFrame,
    weights: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    *,
    train_start: str,
    train_end: str,
    corr_cutoff: float,
    signfit: bool = False,
) -> list[tuple[str, float]]:
    selected: list[tuple[str, float]] = []
    selected_pnls: list[pd.Series] = []
    for _, row in rows.iterrows():
        expr = row["expression"]
        if expr not in weights:
            continue
        sign = -1.0 if signfit and row.get("TRAIN_SR_net", 0.0) < 0 else 1.0
        pnl = _train_pnl(weights[expr] * sign, returns, train_start, train_end)
        if pnl.std(ddof=1) <= 0:
            continue
        keep = True
        for existing in selected_pnls:
            common = pnl.index.intersection(existing.index)
            if len(common) < 50:
                continue
            corr = pnl.loc[common].corr(existing.loc[common])
            if pd.notna(corr) and abs(corr) >= corr_cutoff:
                keep = False
                break
        if keep:
            selected.append((expr, sign))
            selected_pnls.append(pnl)
    return selected


def combine_weights(selected: list[tuple[str, float]], weights: dict[str, pd.DataFrame], *, renormalize: bool) -> pd.DataFrame:
    if not selected:
        return pd.DataFrame()
    combined = None
    for expr, sign in selected:
        w = weights[expr] * sign
        combined = w.copy() if combined is None else combined.add(w, fill_value=0.0)
    assert combined is not None
    combined = combined / len(selected)
    if renormalize:
        gross = combined.abs().sum(axis=1).replace(0, np.nan)
        combined = combined.div(gross, axis=0)
    return combined.fillna(0.0)


def plot_equity(curves: dict[str, pd.Series], out_path: Path, title: str) -> None:
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(13, 7))
    for name, pnl in curves.items():
        eq = (1.0 + pnl.fillna(0.0)).cumprod()
        ax.plot(eq.index, eq.values, lw=1.2, label=name)
    ax.axvline(pd.Timestamp("2025-09-01"), color="black", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(pd.Timestamp("2026-01-01"), color="black", ls="--", lw=0.8, alpha=0.5)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("equity, start=1.0")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze BTC-terms candidate selection and combinations.")
    p.add_argument("--scan-csv", type=Path, required=True)
    p.add_argument("--matrices-dir", type=Path, default=ROOT / "data/kucoin_cache/matrices/4h_btc")
    p.add_argument("--universe", type=Path, default=ROOT / "data/kucoin_cache/universes/KUCOIN_TOP100_BTC_4h.parquet")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data/aipt_results/kucoin_btc_terms_selection")
    p.add_argument("--coverage", type=float, default=0.3)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--corr-cutoff", type=float, default=0.70)
    p.add_argument("--max-weight", type=float, default=0.10)
    p.add_argument("--cost-bps", type=float, default=3.5)
    p.add_argument("--train-start", default="2023-09-01")
    p.add_argument("--train-end", default="2025-09-01")
    p.add_argument("--val-end", default="2026-01-01")
    p.add_argument("--test-end", default=None, help="Optional inclusive cap for TEST and VAL+TEST splits.")
    p.add_argument(
        "--allow-oos-selection",
        action="store_true",
        help="Allow VAL-based selection sets. Diagnostic only; do not use for promotion.",
    )
    p.add_argument(
        "--allow-test-diagnostics",
        action="store_true",
        help="Allow TEST/VAL+TEST lookahead sets. Diagnostic only.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log(f"loading scan {args.scan_csv}")
    scan = pd.read_csv(args.scan_csv)
    protocol_status = (
        "DIAGNOSTIC_OOS_SELECTION"
        if args.allow_oos_selection or args.allow_test_diagnostics
        else "TRAIN_ONLY_SELECTION"
    )
    log(f"selection protocol: {protocol_status}")
    sets = build_candidate_sets(
        scan,
        args.top_n,
        allow_oos_selection=args.allow_oos_selection,
        allow_test_diagnostics=args.allow_test_diagnostics,
    )
    expressions = sorted({expr for frame in sets.values() for expr in frame["expression"].tolist()})
    log(f"{len(scan)} scan rows -> {len(sets)} selection sets, {len(expressions)} unique expressions to re-evaluate")

    matrices, universe, tickers = load_research_data(args.matrices_dir, args.universe, args.coverage)
    returns = matrices["returns"]
    engine = FastExpressionEngine(data_fields=matrices)
    weights = evaluate_candidate_weights(engine, expressions, universe, max_weight=args.max_weight)
    log(f"evaluated weights for {len(weights)} expressions")

    report_rows: list[dict[str, Any]] = []
    curves: dict[str, pd.Series] = {}

    for set_name, frame in sets.items():
        signfit = set_name.endswith("abs_train_signfit")
        for renormalize in (False, True):
            selected = correlation_prune(
                frame,
                weights,
                returns,
                train_start=args.train_start,
                train_end=args.train_end,
                corr_cutoff=args.corr_cutoff,
                signfit=signfit,
            )
            combined = combine_weights(selected, weights, renormalize=renormalize)
            if combined.empty:
                continue
            gross, net, turnover = _portfolio_pnl(combined, returns, args.cost_bps)
            metrics = summarize_pnl(
                gross,
                net,
                turnover,
                train_start=args.train_start,
                train_end=args.train_end,
                val_end=args.val_end,
                test_end=args.test_end,
            )
            name = f"{set_name}_{'renorm' if renormalize else 'avg'}"
            row: dict[str, Any] = {
                "name": name,
                "selection": set_name,
                "renormalize": renormalize,
                "n_input": int(len(frame)),
                "n_selected_after_corr": int(len(selected)),
                "corr_cutoff": args.corr_cutoff,
                "diagnostic_lookahead": "DIAG_LOOKAHEAD" in set_name or "DIAG_OOS_SELECTION" in set_name,
                "protocol_status": protocol_status,
                "expressions": [expr if sign > 0 else f"negative({expr})" for expr, sign in selected],
            }
            for split, vals in metrics.items():
                for key, value in vals.items():
                    row[f"{split}_{key}"] = value
            report_rows.append(row)
            if len(curves) < 12:
                curves[name] = net
            log(
                f"{name}: n={len(selected)} "
                f"TRAIN={metrics.get('TRAIN', {}).get('SR_net', float('nan')):+.2f} "
                f"VAL={metrics.get('VAL', {}).get('SR_net', float('nan')):+.2f} "
                f"TEST={metrics.get('TEST', {}).get('SR_net', float('nan')):+.2f} "
                f"V+T={metrics.get('VAL+TEST', {}).get('SR_net', float('nan')):+.2f}"
            )

    report = pd.DataFrame(report_rows)
    if report.empty:
        log("no combinations produced")
        return 1
    sort_cols = ["diagnostic_lookahead"]
    sort_ascending = [True]
    for col in ["TRAIN_SR_net", "VAL_SR_net", "TEST_SR_net", "VAL+TEST_SR_net"]:
        if col in report.columns:
            sort_cols.append(col)
            sort_ascending.append(False)
    report_sorted = report.sort_values(sort_cols, ascending=sort_ascending)
    csv_path = args.out_dir / "selection_report.csv"
    json_path = args.out_dir / "selection_report.json"
    png_path = args.out_dir / "selection_equity.png"
    report_sorted.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(report_sorted.replace({np.nan: None}).to_dict(orient="records"), indent=2), encoding="utf-8")
    plot_equity(curves, png_path, f"KuCoin BTC-Terms Candidate Baskets ({args.scan_csv.name})")

    log(f"wrote {csv_path}")
    log(f"wrote {json_path}")
    log(f"wrote {png_path}")
    display_cols = [
        "name",
        "protocol_status",
        "diagnostic_lookahead",
        "n_selected_after_corr",
        "TRAIN_SR_net",
        "VAL_SR_net",
        "TEST_SR_net",
        "VAL+TEST_SR_net",
        "FULL_SR_net",
    ]
    display_cols = [c for c in display_cols if c in report_sorted.columns]
    print(
        report_sorted[display_cols].to_string(index=False),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
