"""Scan KuCoin 4h alpha candidates on BTC-denominated matrices.

This is a read-only discovery pass: it evaluates the same KuCoin 4h candidate
library used by the archived discovery workflow, but writes ranked CSV/JSON
artifacts instead of inserting into the production alpha DB. That keeps BTC
terms research isolated until we decide a candidate deserves promotion.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.operators.fastexpression import FastExpressionEngine

BARS_PER_YEAR = 6 * 365
ALL_SPLITS = ("TRAIN", "VAL", "TEST", "VAL+TEST", "FULL")
TRAIN_ONLY_SPLITS = ("TRAIN",)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_candidate_library(seed: int) -> list[str]:
    module = importlib.import_module("archive.discovery.discover_alphas_kucoin_4h")
    return module.generate_candidates(seed=seed)


def _add_unique(out: list[str], seen: set[str], expr: str) -> None:
    if expr not in seen:
        seen.add(expr)
        out.append(expr)


def load_btc_specific_candidates(seed: int) -> list[str]:
    """Candidate families aimed at BTC-relative crypto cross-sections."""
    rng = np.random.default_rng(seed)
    out: list[str] = []
    seen: set[str] = set()

    price_windows = [1, 2, 3, 6, 12, 18, 24, 36, 48, 72, 120, 180, 240]
    smooth_windows = [3, 6, 12, 24, 48, 60, 120]
    vol_windows = [10, 20, 60, 120]

    mom_atoms: list[str] = []
    for w in price_windows:
        mom_atoms.extend(
            [
                f"ts_delta(close, {w})",
                f"zscore_cs(ts_delta(close, {w}))",
                f"rank(ts_delta(close, {w}))",
                f"ts_rank(close, {w})",
                f"ts_zscore(close, {w})",
                f"sma(ts_delta(close, {w}), 6)",
            ]
        )
    mom_atoms.extend(
        [
            "returns",
            "log_returns",
            "momentum_5d",
            "momentum_20d",
            "momentum_60d",
            "zscore_cs(momentum_5d)",
            "zscore_cs(momentum_20d)",
            "zscore_cs(momentum_60d)",
            "sma(returns, 6)",
            "sma(returns, 12)",
            "sma(returns, 24)",
            "sma(returns, 60)",
            "sma(log_returns, 60)",
            "sma(log_returns, 120)",
        ]
    )

    range_atoms = [
        "open_close_range",
        "high_low_range",
        "close_position_in_range",
        "upper_shadow",
        "lower_shadow",
        "vwap_deviation",
        "true_divide(open_close_range, df_max(high_low_range, 0.0001))",
        "true_divide(upper_shadow, df_max(high_low_range, 0.0001))",
        "true_divide(lower_shadow, df_max(high_low_range, 0.0001))",
        "subtract(close_position_in_range, 0.5)",
    ]

    volume_atoms = [
        "volume_momentum_1",
        "volume_momentum_5_20",
        "volume_ratio_20d",
        "zscore_cs(s_log_1p(adv20))",
        "zscore_cs(s_log_1p(adv60))",
        "zscore_cs(s_log_1p(quote_volume))",
        "ts_delta(s_log_1p(quote_volume), 6)",
        "ts_delta(s_log_1p(quote_volume), 30)",
        "ts_delta(s_log_1p(adv20), 30)",
        "ts_delta(s_log_1p(adv60), 30)",
    ]

    risk_atoms = [
        "historical_volatility_10",
        "historical_volatility_20",
        "historical_volatility_60",
        "historical_volatility_120",
        "parkinson_volatility_10",
        "parkinson_volatility_20",
        "parkinson_volatility_60",
        "beta_to_btc",
        "subtract(beta_to_btc, sma(beta_to_btc, 60))",
        "ts_delta(beta_to_btc, 6)",
        "ts_delta(beta_to_btc, 12)",
    ]

    # Simple standalone transforms.
    for atom in mom_atoms + range_atoms + volume_atoms + risk_atoms:
        _add_unique(out, seen, atom)
        _add_unique(out, seen, f"negative({atom})")
        _add_unique(out, seen, f"zscore_cs({atom})")
        for sw in [6, 24, 60]:
            _add_unique(out, seen, f"sma({atom}, {sw})")

    # Momentum / reversion gated by range, volatility, and volume.
    for mom in mom_atoms[:70]:
        for gate in range_atoms + volume_atoms[:6]:
            _add_unique(out, seen, f"multiply({mom}, {gate})")
            _add_unique(out, seen, f"negative(multiply({mom}, {gate}))")
        for vol in risk_atoms[:7]:
            _add_unique(out, seen, f"true_divide({mom}, df_max({vol}, 0.0001))")
            _add_unique(out, seen, f"negative(true_divide({mom}, df_max({vol}, 0.0001)))")

    # Slower liquidity and microstructure composites.
    cs_atoms = [
        "zscore_cs(sma(ts_delta(s_log_1p(adv20), 30), 120))",
        "zscore_cs(sma(ts_delta(s_log_1p(adv60), 30), 120))",
        "zscore_cs(sma(ts_delta(s_log_1p(quote_volume), 30), 120))",
        "zscore_cs(sma(volume_momentum_5_20, 60))",
        "zscore_cs(sma(ts_corr(log_returns, volume_ratio_20d, 60), 60))",
        "zscore_cs(sma(ts_corr(log_returns, delay(log_returns, 1), 60), 60))",
        "zscore_cs(sma(ts_corr(close_position_in_range, open_close_range, 60), 60))",
        "zscore_cs(sma(ts_zscore(log_returns, 120), 120))",
        "zscore_cs(sma(ts_skewness(log_returns, 60), 60))",
        "zscore_cs(sma(ts_kurtosis(log_returns, 60), 60))",
        "zscore_cs(sma(vwap_deviation, 120))",
        "zscore_cs(sma(close_position_in_range, 120))",
    ]
    for _ in range(700):
        picks = rng.choice(cs_atoms, size=int(rng.choice([2, 3, 4])), replace=False).tolist()
        expr = picks[0]
        for atom in picks[1:]:
            expr = f"add({expr},{atom})"
        _add_unique(out, seen, f"df_min(df_max({expr}, -1.5), 1.5)")
        _add_unique(out, seen, f"negative(df_min(df_max({expr}, -1.5), 1.5))")

    # Compact smoothed variants of the empirically interesting range-momentum family.
    for pw in price_windows:
        base = f"multiply(ts_delta(close, {pw}), true_divide(open_close_range, df_max(high_low_range, 0.0001)))"
        for sw in smooth_windows:
            _add_unique(out, seen, f"sma({base}, {sw})")
            _add_unique(out, seen, f"negative(sma({base}, {sw}))")
        for alpha in [0.02, 0.05, 0.1, 0.2]:
            _add_unique(out, seen, f"Decay_exp({base}, {alpha})")
            _add_unique(out, seen, f"negative(Decay_exp({base}, {alpha}))")

    for vw in vol_windows:
        for pw in [6, 12, 24, 48, 120]:
            base = f"multiply(Sign(ts_delta(close, {pw})), true_divide(volume_momentum_5_20, df_max(historical_volatility_{vw}, 0.0001)))"
            _add_unique(out, seen, base)
            _add_unique(out, seen, f"negative({base})")
            for sw in [6, 12, 24]:
                _add_unique(out, seen, f"sma({base}, {sw})")
                _add_unique(out, seen, f"negative(sma({base}, {sw}))")

    return out


def build_candidates(mode: str, seed: int, include_negative: bool) -> list[str]:
    parts: list[str] = []
    if mode in {"kucoin_library", "both"}:
        parts.extend(load_candidate_library(seed))
    if mode in {"btc_specific", "both"}:
        parts.extend(load_btc_specific_candidates(seed))

    out: list[str] = []
    seen: set[str] = set()
    for expr in parts:
        _add_unique(out, seen, expr)
        if include_negative and not expr.startswith("negative("):
            _add_unique(out, seen, f"negative({expr})")
    return out


def load_research_data(matrices_dir: Path, universe_path: Path, coverage: float) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[str]]:
    uni = pd.read_parquet(universe_path)
    if not isinstance(uni.index, pd.DatetimeIndex):
        uni.index = pd.to_datetime(uni.index, errors="coerce")
        uni = uni[uni.index.notna()]
    uni = uni.astype(bool)
    cov = uni.sum(axis=0) / len(uni)
    valid = sorted(cov[cov > coverage].index.tolist())
    dates = uni.index

    matrices: dict[str, pd.DataFrame] = {}
    for fp in sorted(matrices_dir.glob("*.parquet")):
        if fp.stem.startswith("_") or fp.parent != matrices_dir:
            continue
        df = pd.read_parquet(fp)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[df.index.notna()]
        cols = [c for c in valid if c in df.columns]
        if cols:
            matrices[fp.stem] = df.loc[df.index.isin(dates), cols].reindex(index=dates, columns=cols)

    tickers = [c for c in valid if c in matrices["close"].columns]
    for name, mat in list(matrices.items()):
        matrices[name] = mat.reindex(index=dates, columns=tickers)
    return matrices, uni.reindex(index=dates, columns=tickers).fillna(False).astype(bool), tickers


def signal_to_portfolio(sig: pd.DataFrame, universe: pd.DataFrame, max_weight: float) -> pd.DataFrame:
    s = sig.replace([np.inf, -np.inf], np.nan).reindex(index=universe.index, columns=universe.columns)
    s = s.where(universe, np.nan)
    s = s.sub(s.mean(axis=1), axis=0)
    gross = s.abs().sum(axis=1).replace(0, np.nan)
    w = s.div(gross, axis=0)
    if max_weight > 0:
        w = w.clip(lower=-max_weight, upper=max_weight)
    return w.fillna(0.0)


def _slice_range(
    label: str,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    test_end: pd.Timestamp | None = None,
):
    if label == "TRAIN":
        return slice(train_start, train_end)
    if label == "VAL":
        return slice(train_end, val_end)
    if label == "TEST":
        return slice(val_end, test_end)
    if label == "VAL+TEST":
        return slice(train_end, test_end)
    if label == "FULL":
        return slice(None, None)
    raise ValueError(label)


def summarize_pnl(
    gross: pd.Series,
    net: pd.Series,
    turnover: pd.Series,
    *,
    train_start: str,
    train_end: str,
    val_end: str,
    test_end: str | None = None,
    split_labels: tuple[str, ...] = ALL_SPLITS,
) -> dict[str, dict[str, float]]:
    ts = pd.Timestamp(train_start)
    te = pd.Timestamp(train_end)
    ve = pd.Timestamp(val_end)
    tte = pd.Timestamp(test_end) if test_end else None
    out: dict[str, dict[str, float]] = {}
    for label in split_labels:
        sl = _slice_range(label, ts, te, ve, tte)
        g = gross.loc[sl].dropna()
        n = net.loc[sl].dropna()
        to = turnover.loc[sl].dropna()
        if len(g) < 50 or g.std(ddof=1) <= 0 or n.std(ddof=1) <= 0:
            out[label] = {
                "n_bars": int(len(g)),
                "SR_gross": float("nan"),
                "SR_net": float("nan"),
                "ret_ann_net": float("nan"),
                "max_dd_net": float("nan"),
                "turnover": float(to.mean()) if len(to) else float("nan"),
            }
            continue
        eq_n = (1.0 + n).cumprod()
        out[label] = {
            "n_bars": int(len(g)),
            "SR_gross": float(g.mean() / g.std(ddof=1) * math.sqrt(BARS_PER_YEAR)),
            "SR_net": float(n.mean() / n.std(ddof=1) * math.sqrt(BARS_PER_YEAR)),
            "ret_ann_net": float(n.mean() * BARS_PER_YEAR),
            "max_dd_net": float((eq_n / eq_n.cummax() - 1.0).min()),
            "turnover": float(to.mean()) if len(to) else float("nan"),
        }
    return out


def evaluate_expression(
    engine: FastExpressionEngine,
    expression: str,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    max_weight: float,
    cost_bps: float,
    train_start: str,
    train_end: str,
    val_end: str,
    test_end: str | None = None,
    split_labels: tuple[str, ...] = ALL_SPLITS,
) -> dict[str, Any] | None:
    raw = engine.evaluate(expression)
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return None
    w = signal_to_portfolio(raw, universe, max_weight)
    common_idx = w.index.intersection(returns.index)
    common_cols = w.columns.intersection(returns.columns)
    w = w.loc[common_idx, common_cols]
    r = returns.loc[common_idx, common_cols]
    gross = (w * r.shift(-1)).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - turnover * cost_bps / 10000.0
    metrics = summarize_pnl(
        gross,
        net,
        turnover,
        train_start=train_start,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
        split_labels=split_labels,
    )
    return {"weights": w, "metrics": metrics}


def flatten_result(expression: str, metrics: dict[str, dict[str, float]], rank_index: int) -> dict[str, Any]:
    row: dict[str, Any] = {"rank_index": rank_index, "expression": expression}
    for split, vals in metrics.items():
        for key, value in vals.items():
            row[f"{split}_{key}"] = value
    return row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a KuCoin BTC-terms alpha discovery scan.")
    p.add_argument("--matrices-dir", type=Path, default=ROOT / "data/kucoin_cache/matrices/4h_btc")
    p.add_argument("--universe", type=Path, default=ROOT / "data/kucoin_cache/universes/KUCOIN_TOP100_BTC_4h.parquet")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data/aipt_results/kucoin_btc_terms_discovery")
    p.add_argument("--coverage", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--candidate-mode", choices=["kucoin_library", "btc_specific", "both"], default="kucoin_library")
    p.add_argument("--include-negative", action="store_true", help="Also scan negative(expr) for candidate sign discovery.")
    p.add_argument("--max-candidates", type=int, default=0, help="0 means all generated candidates.")
    p.add_argument("--candidate-start", type=int, default=1, help="1-based inclusive candidate start after generation.")
    p.add_argument("--candidate-end", type=int, default=0, help="1-based inclusive candidate end; 0 means the last candidate.")
    p.add_argument("--rank-split", choices=ALL_SPLITS, default="TRAIN", help="Split used for output ranking.")
    p.add_argument(
        "--allow-oos-diagnostics",
        action="store_true",
        help="Compute/report VAL/TEST metrics. Diagnostic only; do not use for candidate selection.",
    )
    p.add_argument("--time-limit-min", type=float, default=45.0)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--cost-bps", type=float, default=3.5)
    p.add_argument("--max-weight", type=float, default=0.10)
    p.add_argument("--train-start", default="2023-09-01")
    p.add_argument("--train-end", default="2025-09-01")
    p.add_argument("--val-end", default="2026-01-01")
    p.add_argument("--test-end", default=None, help="Optional inclusive cap for TEST and VAL+TEST splits.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.rank_split != "TRAIN" and not args.allow_oos_diagnostics:
        log("--rank-split other than TRAIN requires --allow-oos-diagnostics")
        return 2
    split_labels = ALL_SPLITS if args.allow_oos_diagnostics else TRAIN_ONLY_SPLITS
    rank_col = f"{args.rank_split}_SR_net"
    protocol_status = (
        "DIAGNOSTIC_OOS_METRICS_NOT_FOR_SELECTION"
        if args.allow_oos_diagnostics
        else "TRAIN_ONLY_DISCOVERY"
    )
    log(f"research protocol: {protocol_status}; rank_split={args.rank_split}")

    log(f"loading BTC-term matrices from {args.matrices_dir}")
    matrices, universe, tickers = load_research_data(args.matrices_dir, args.universe, args.coverage)
    returns = matrices["returns"]
    log(
        f"loaded {len(matrices)} fields, {len(tickers)} tickers, {len(universe)} bars "
        f"({universe.index.min()} -> {universe.index.max()})"
    )

    candidates = build_candidates(args.candidate_mode, args.seed, args.include_negative)
    candidates_generated = len(candidates)
    start = max(int(args.candidate_start), 1)
    end = int(args.candidate_end) if int(args.candidate_end) > 0 else candidates_generated
    end = min(end, candidates_generated)
    if start > end:
        log(f"candidate window is empty: start={start}, end={end}, generated={candidates_generated}")
        return 1
    candidate_offset = start - 1
    candidates = candidates[candidate_offset:end]
    if args.max_candidates and args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]
    log(
        f"candidate expressions: {len(candidates)} "
        f"(window {start}-{candidate_offset + len(candidates)} of {candidates_generated})"
    )

    engine = FastExpressionEngine(data_fields=matrices)
    rows: list[dict[str, Any]] = []
    best_train = -999.0
    best_rank = -999.0
    t0 = time.time()
    deadline = t0 + args.time_limit_min * 60.0

    for local_i, expr in enumerate(candidates, 1):
        i = candidate_offset + local_i
        if time.time() >= deadline:
            log(f"time limit reached at local {local_i - 1}/{len(candidates)} candidates")
            break
        try:
            result = evaluate_expression(
                engine,
                expr,
                returns,
                universe,
                max_weight=args.max_weight,
                cost_bps=args.cost_bps,
                train_start=args.train_start,
                train_end=args.train_end,
                val_end=args.val_end,
                test_end=args.test_end,
                split_labels=split_labels,
            )
        except Exception:
            result = None
        if result is None:
            continue

        row = flatten_result(expr, result["metrics"], i)
        rows.append(row)
        tr = row.get("TRAIN_SR_net", float("nan"))
        rank_value = row.get(rank_col, float("nan"))
        if pd.notna(tr):
            best_train = max(best_train, float(tr))
        if pd.notna(rank_value):
            best_rank = max(best_rank, float(rank_value))

        if local_i % args.progress_every == 0:
            rate = local_i / max(time.time() - t0, 1e-9)
            log(
                f"{local_i}/{len(candidates)} evaluated (global {i}); results={len(rows)}; "
                f"best TRAIN net SR={best_train:+.2f}; best {args.rank_split} net SR={best_rank:+.2f}; "
                f"rate={rate:.2f}/s"
            )

    df = pd.DataFrame(rows)
    if df.empty:
        log("no successful candidate evaluations")
        return 1

    if rank_col not in df.columns:
        log(f"rank column missing from scan output: {rank_col}")
        return 1
    sort_cols = [rank_col]
    if rank_col != "TRAIN_SR_net" and "TRAIN_SR_net" in df.columns:
        sort_cols.append("TRAIN_SR_net")
    df_sorted = df.sort_values(sort_cols, ascending=False)
    csv_path = args.out_dir / "candidate_scan.csv"
    top_path = args.out_dir / "top_candidates.json"
    summary_path = args.out_dir / "summary.json"
    df_sorted.to_csv(csv_path, index=False)

    top = df_sorted.head(25).replace({np.nan: None}).to_dict(orient="records")
    top_path.write_text(json.dumps(top, indent=2), encoding="utf-8")

    summary = {
        "matrices_dir": str(args.matrices_dir),
        "universe": str(args.universe),
        "n_fields": len(matrices),
        "n_tickers": len(tickers),
        "n_bars": len(universe),
        "candidates_generated": candidates_generated,
        "candidate_start": start,
        "candidate_end": candidate_offset + len(candidates),
        "candidates_requested": len(candidates),
        "candidates_evaluated_successfully": int(len(df)),
        "elapsed_sec": time.time() - t0,
        "cost_bps": args.cost_bps,
        "max_weight": args.max_weight,
        "protocol_status": protocol_status,
        "rank_split": args.rank_split,
        "allow_oos_diagnostics": bool(args.allow_oos_diagnostics),
        "train_start": args.train_start,
        "train_end": args.train_end,
        "val_end": args.val_end,
        "test_end": args.test_end,
        "best_by_rank_net": top[:10],
        "best_by_train_net": (
            df.sort_values(["TRAIN_SR_net"], ascending=False)
            .head(10)
            .replace({np.nan: None})
            .to_dict(orient="records")
        ),
    }
    if args.allow_oos_diagnostics:
        summary["oos_diagnostics_warning"] = (
            "VAL/TEST metrics were computed for diagnostics only. Do not use this run "
            "for candidate selection or model promotion."
        )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log(f"wrote {csv_path}")
    log(f"wrote {top_path}")
    log(f"wrote {summary_path}")
    log(f"top 10 by {args.rank_split} net SR:")
    cols = []
    for col in [rank_col, "TRAIN_SR_net", "VAL_SR_net", "TEST_SR_net", "VAL+TEST_SR_net", "TRAIN_turnover", "expression"]:
        if col in df_sorted.columns and col not in cols:
            cols.append(col)
    print(df_sorted[cols].head(10).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
