"""Strict simple-alpha benchmark for AIPT comparisons.

This evaluates a small set of price/volume alpha motifs that showed up in the
legacy alpha database, but on the experiment PIT universes and the same delay
convention used by the AIPT runs:

  delay=0: signal at close[t] earns close[t+1] / close[t] - 1
  delay=1: signal at close[t] earns open[t+2] / open[t+1] - 1

No fitting is done, so TRAIN/VAL/TEST Sharpes here are a benchmark for signal
families, not selected model performance.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.aipt_replication import SCENARIOS, load_market_data, make_forward_returns
from src.operators.fastexpression import FastExpressionEngine


FIELDS = [
    "close",
    "open",
    "high",
    "low",
    "vwap",
    "volume",
    "dollars_traded",
    "adv20",
    "adv60",
    "returns",
]


ALPHAS: dict[str, str] = {
    "vwap_reversal": "rank(negative(true_divide(close, vwap)))",
    "delta3_reversal": "rank(negative(ts_delta(close, 3)))",
    "delta5_reversal": "rank(negative(ts_delta(close, 5)))",
    "delta21_reversal": "rank(negative(ts_delta(close, 21)))",
    "vwap_x_delta5": (
        "rank(multiply(rank(negative(true_divide(close, vwap))), "
        "rank(negative(ts_delta(close, 5)))))"
    ),
    "volume_surge_x_vwap": (
        "rank(multiply(rank(true_divide(volume, sma(volume, 20))), "
        "rank(negative(true_divide(close, vwap)))))"
    ),
    "volume_surge_plus_vwap_plus_delta5": (
        "rank(add(add(rank(true_divide(volume, sma(volume, 20))), "
        "rank(negative(true_divide(close, vwap)))), "
        "rank(negative(ts_delta(close, 5)))))"
    ),
    "zscore20_reversal": (
        "rank(negative(true_divide(subtract(close, sma(close, 20)), "
        "df_max(stddev(close, 20), 0.001))))"
    ),
}


@dataclass(frozen=True)
class SimpleAlphaSpec:
    scenario: str
    alpha_name: str
    expression: str
    demean: str
    clip_max_w: float
    dynamic_universe: bool


def _parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _metrics(x: pd.Series, bars_per_year: int) -> dict[str, float]:
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 3:
        return {
            "n_bars": int(len(x)),
            "SR": float("nan"),
            "mean_ann": float("nan"),
            "vol_ann": float("nan"),
            "cum_return_approx": float("nan"),
        }
    sd = float(x.std(ddof=1))
    mu = float(x.mean())
    return {
        "n_bars": int(len(x)),
        "SR": (mu / sd) * math.sqrt(bars_per_year) if sd > 0 else float("nan"),
        "mean_ann": mu * bars_per_year,
        "vol_ann": sd * math.sqrt(bars_per_year),
        "cum_return_approx": float(x.sum()),
    }


def _split_metrics(pnl: pd.Series, scenario_name: str) -> dict[str, dict[str, float]]:
    s = SCENARIOS[scenario_name]
    splits = {
        "TRAIN": slice(None, s.split_train_end),
        "VAL": slice(s.split_train_end, s.split_val_end),
        "TEST": slice(s.split_val_end, None),
        "VAL+TEST": slice(s.split_train_end, None),
        "FULL": slice(None, None),
    }
    return {name: _metrics(pnl.loc[sl], s.bars_per_year) for name, sl in splits.items()}


def _make_weights(
    raw_signal: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    demean: str,
    clip_max_w: float,
) -> pd.DataFrame:
    sig = raw_signal.replace([np.inf, -np.inf], np.nan).reindex(index=universe.index, columns=universe.columns)
    sig = sig.where(universe)
    if demean == "market":
        sig = sig.sub(sig.mean(axis=1), axis=0)
    elif demean == "center_rank":
        # For expressions ending in rank(...), this preserves the rank-only signal
        # while removing the long-only market book implied by [0, 1] ranks.
        sig = sig.sub(0.5)
    elif demean == "none":
        pass
    else:
        raise ValueError(f"unknown demean mode {demean!r}")

    gross = sig.abs().sum(axis=1).replace(0.0, np.nan)
    w = sig.div(gross, axis=0)
    if clip_max_w > 0:
        w = w.clip(lower=-clip_max_w, upper=clip_max_w)
    return w.fillna(0.0)


def run_one(spec: SimpleAlphaSpec, out_dir: Path) -> dict:
    scenario = SCENARIOS[spec.scenario]
    print(
        f"[simple] {spec.scenario} {spec.alpha_name} demean={spec.demean} "
        f"clip={spec.clip_max_w:g} dynamic_uni={spec.dynamic_universe}",
        flush=True,
    )
    t0 = time.time()
    uni, mats, _close, _available = load_market_data(
        scenario,
        FIELDS,
        root=ROOT,
        dynamic_universe=spec.dynamic_universe,
    )
    engine = FastExpressionEngine(data_fields={k: v.reindex(index=uni.index, columns=uni.columns) for k, v in mats.items()})
    raw = engine.evaluate(spec.expression)
    weights = _make_weights(raw, uni, demean=spec.demean, clip_max_w=spec.clip_max_w)
    fwd_ret = make_forward_returns(mats, scenario.delay).reindex(index=uni.index, columns=uni.columns)
    pnl = (weights * fwd_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    metrics = _split_metrics(pnl, spec.scenario)
    elapsed = time.time() - t0
    result = {
        "spec": asdict(spec),
        "scenario": asdict(scenario),
        "n_dates": len(uni.index),
        "n_names": len(uni.columns),
        "metrics": metrics,
        "turnover": {
            split: float(turnover.loc[sl].mean())
            for split, sl in {
                "TRAIN": slice(None, scenario.split_train_end),
                "VAL": slice(scenario.split_train_end, scenario.split_val_end),
                "TEST": slice(scenario.split_val_end, None),
                "VAL+TEST": slice(scenario.split_train_end, None),
                "FULL": slice(None, None),
            }.items()
        },
        "elapsed_sec": elapsed,
        "lookahead_audit": {
            "signal": "expression evaluated from same-row PIT matrices only",
            "universe": "dynamic PIT daily mask" if spec.dynamic_universe else "initial-fit frozen column set",
            "delay": scenario.delay,
            "fwd_return": "close[t+1]/close[t]-1 for d0; open[t+2]/open[t+1]-1 for d1",
            "training_selection": "none; fixed alpha list declared in script",
            "fees": False,
        },
    }
    tag = (
        f"{spec.scenario}__{spec.alpha_name}__{spec.demean}"
        f"__clip{spec.clip_max_w:g}{'__dynuni' if spec.dynamic_universe else ''}"
    ).replace(".", "p")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    pd.DataFrame({"pnl": pnl, "turnover": turnover}).to_parquet(out_dir / f"{tag}.returns.parquet")
    print(
        f"      SR train={metrics['TRAIN']['SR']:+.2f} val={metrics['VAL']['SR']:+.2f} "
        f"test={metrics['TEST']['SR']:+.2f} v+t={metrics['VAL+TEST']['SR']:+.2f} "
        f"to={result['turnover']['FULL']*100:.1f}% {elapsed:.1f}s",
        flush=True,
    )
    return result


def write_summary(results: list[dict], out_dir: Path) -> Path:
    rows = []
    for r in results:
        spec = r["spec"]
        base = {
            "scenario": spec["scenario"],
            "alpha_name": spec["alpha_name"],
            "expression": spec["expression"],
            "demean": spec["demean"],
            "clip_max_w": spec["clip_max_w"],
            "dynamic_universe": spec["dynamic_universe"],
            "n_names": r["n_names"],
            "elapsed_sec": r["elapsed_sec"],
        }
        for split, metrics in r["metrics"].items():
            row = dict(base)
            row["split"] = split
            row["turnover_per_bar"] = r["turnover"][split]
            row.update(metrics)
            rows.append(row)
    out = out_dir / "aipt_simple_alpha_benchmark_summary.csv"
    if rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (out_dir / "aipt_simple_alpha_benchmark_summary.json").write_text(
        json.dumps(results, indent=2, default=float),
        encoding="utf-8",
    )
    return out


def write_manifest(args: argparse.Namespace, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "script": "experiments/aipt_simple_alpha_benchmark.py",
        "started_at": datetime.now().isoformat(),
        "cwd": str(ROOT),
        "argv": sys.argv,
        "args": vars(args),
        "mode": "strict_simple_alpha_no_fee_benchmark",
        "notes": "Fixed price/volume motifs from legacy DB, evaluated on PIT universes with no fitting and no fees.",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", nargs="*", default=["equity_top3000_d0", "equity_top3000_d1"])
    p.add_argument("--alphas", default="all")
    p.add_argument("--demeans", default="none,market,center_rank")
    p.add_argument("--clip-grid", default="0,0.005")
    p.add_argument("--dynamic-universe", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out-dir", default="experiments/results/aipt_simple_alpha_benchmark_top3000")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    out_dir = ROOT / args.out_dir
    write_manifest(args, out_dir)

    alpha_names = list(ALPHAS) if args.alphas == "all" else _parse_csv(args.alphas)
    demeans = _parse_csv(args.demeans)
    clips = [float(x) for x in _parse_csv(args.clip_grid)]
    specs = [
        SimpleAlphaSpec(
            scenario=scenario,
            alpha_name=name,
            expression=ALPHAS[name],
            demean=demean,
            clip_max_w=clip,
            dynamic_universe=args.dynamic_universe,
        )
        for scenario in args.scenarios
        for name in alpha_names
        for demean in demeans
        for clip in clips
    ]
    if args.limit:
        specs = specs[: args.limit]
    print(f"[setup] simple-alpha cells={len(specs)} -> {out_dir.relative_to(ROOT)}", flush=True)
    results: list[dict] = []
    failures: list[dict] = []
    for spec in specs:
        try:
            results.append(run_one(spec, out_dir))
            write_summary(results, out_dir)
        except Exception as exc:
            failures.append({"spec": asdict(spec), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[fail] {spec}: {type(exc).__name__}: {exc}", flush=True)
            (out_dir / "aipt_simple_alpha_benchmark_failures.json").write_text(
                json.dumps(failures, indent=2),
                encoding="utf-8",
            )
    summary = write_summary(results, out_dir)
    print(f"[done] results={len(results)} failures={len(failures)} summary={summary.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
