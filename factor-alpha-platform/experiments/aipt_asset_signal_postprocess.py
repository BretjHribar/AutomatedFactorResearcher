"""Postprocess AIPT asset-signal runs.

Creates parameter-surface aggregates and fixed seed ensembles from
`aipt_asset_signal_unconstrained.py` outputs. Seed ensembling is ex ante:
for a given (scenario, source_set, P, z, weight_mode, train_window, ...), it
averages all available return streams across seeds without looking at their
future performance.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _metrics(x: pd.Series, bars_per_year: int) -> dict[str, float]:
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 3:
        return {"n_bars": int(len(x)), "SR": float("nan"), "mean_ann": float("nan"), "vol_ann": float("nan")}
    sd = float(x.std(ddof=1))
    mu = float(x.mean())
    return {
        "n_bars": int(len(x)),
        "SR": (mu / sd) * math.sqrt(bars_per_year) if sd > 0 else float("nan"),
        "mean_ann": mu * bars_per_year,
        "vol_ann": sd * math.sqrt(bars_per_year),
    }


def _splits(scenario: dict) -> dict[str, slice]:
    return {
        "TRAIN": slice(None, scenario["split_train_end"]),
        "VAL": slice(scenario["split_train_end"], scenario["split_val_end"]),
        "TEST": slice(scenario["split_val_end"], None),
        "VAL+TEST": slice(scenario["split_train_end"], None),
        "FULL": slice(None, None),
    }


def _read_results(run_dirs: list[Path]) -> list[dict]:
    rows = []
    for run_dir in run_dirs:
        for path in sorted(run_dir.glob("*.json")):
            if path.name.endswith("summary.json") or path.name in {"run_manifest.json"}:
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            ret_path = path.with_suffix(".returns.parquet")
            if not ret_path.exists():
                continue
            data["_json_path"] = str(path.relative_to(ROOT)).replace("\\", "/")
            data["_returns_path"] = str(ret_path.relative_to(ROOT)).replace("\\", "/")
            rows.append(data)
    return rows


def _key_without_seed(result: dict) -> tuple:
    spec = result["spec"]
    return (
        spec["scenario"],
        spec["source_set"],
        spec["n_features"],
        spec["ridge_z"],
        spec["activation"],
        spec["demean_features"],
        spec["projected_sources"],
        spec["project_top_k"],
        spec["dynamic_universe"],
        spec["train_window"],
        spec["rebalance_every"],
        spec["start_override"],
        spec["weight_mode"],
    )


def write_surface_aggregate(results: list[dict], out_dir: Path) -> Path:
    rows = []
    for result in results:
        spec = result["spec"]
        for split, metrics in result["metrics"].items():
            rows.append(
                {
                    "scenario": spec["scenario"],
                    "source_set": spec["source_set"],
                    "P": spec["n_features"],
                    "z": spec["ridge_z"],
                    "activation": spec["activation"],
                    "seed": spec["seed"],
                    "weight_mode": spec["weight_mode"],
                    "dynamic_universe": spec["dynamic_universe"],
                    "train_window": spec["train_window"],
                    "rebalance_every": spec["rebalance_every"],
                    "start_override": spec["start_override"],
                    "split": split,
                    "SR": metrics["SR"],
                    "sdf_SR": result["sdf_metrics"][split]["SR"],
                    "turnover_per_bar": result["turnover"][split],
                    "net_exposure_mean": result["net_exposure"][split],
                    "json_path": result["_json_path"],
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        out = out_dir / "aipt_asset_signal_surface_aggregate.csv"
        out.write_text("", encoding="utf-8")
        return out
    group_cols = [
        "scenario",
        "source_set",
        "P",
        "z",
        "activation",
        "weight_mode",
        "dynamic_universe",
        "train_window",
        "rebalance_every",
        "start_override",
        "split",
    ]
    agg = (
        frame.groupby(group_cols, dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            SR_mean=("SR", "mean"),
            SR_std=("SR", "std"),
            SR_min=("SR", "min"),
            SR_max=("SR", "max"),
            sdf_SR_mean=("sdf_SR", "mean"),
            turnover_mean=("turnover_per_bar", "mean"),
            net_exposure_mean=("net_exposure_mean", "mean"),
        )
        .reset_index()
    )
    out = out_dir / "aipt_asset_signal_surface_aggregate.csv"
    agg.to_csv(out, index=False)
    frame.to_csv(out_dir / "aipt_asset_signal_surface_cells.csv", index=False)
    return out


def write_seed_ensembles(results: list[dict], out_dir: Path) -> Path:
    rows = []
    for key, group in sorted({k: [r for r in results if _key_without_seed(r) == k] for k in {_key_without_seed(r) for r in results}}.items()):
        if len(group) < 2:
            continue
        first = group[0]
        spec = first["spec"]
        scenario = first["scenario"]
        series = []
        sdf_series = []
        turns = []
        nets = []
        seeds = []
        for result in sorted(group, key=lambda r: r["spec"]["seed"]):
            df = pd.read_parquet(ROOT / result["_returns_path"])
            series.append(df["asset_return"].rename(str(result["spec"]["seed"])))
            sdf_series.append(df["sdf_return"].rename(str(result["spec"]["seed"])))
            turns.append(df["turnover"].rename(str(result["spec"]["seed"])))
            nets.append(df["net_exposure"].rename(str(result["spec"]["seed"])))
            seeds.append(result["spec"]["seed"])
        ret = pd.concat(series, axis=1).mean(axis=1)
        sdf = pd.concat(sdf_series, axis=1).mean(axis=1)
        turnover = pd.concat(turns, axis=1).mean(axis=1)
        net_exposure = pd.concat(nets, axis=1).mean(axis=1)

        tag = (
            f"{spec['scenario']}__{spec['source_set']}__P{spec['n_features']}__z{spec['ridge_z']:g}"
            f"__{spec['activation']}__tw{spec['train_window']}__rb{spec['rebalance_every']}"
            f"__{spec['weight_mode']}__seedens{len(seeds)}"
        ).replace(".", "p")
        pd.DataFrame(
            {
                "asset_return": ret,
                "sdf_return": sdf,
                "turnover_component_mean": turnover,
                "net_exposure_component_mean": net_exposure,
            }
        ).to_parquet(out_dir / f"{tag}.returns.parquet")
        for split, sl in _splits(scenario).items():
            m = _metrics(ret.loc[sl], scenario["bars_per_year"])
            rows.append(
                {
                    "scenario": spec["scenario"],
                    "source_set": spec["source_set"],
                    "P": spec["n_features"],
                    "z": spec["ridge_z"],
                    "activation": spec["activation"],
                    "weight_mode": spec["weight_mode"],
                    "dynamic_universe": spec["dynamic_universe"],
                    "train_window": spec["train_window"],
                    "rebalance_every": spec["rebalance_every"],
                    "start_override": spec["start_override"],
                    "n_seeds": len(seeds),
                    "seeds": ",".join(map(str, seeds)),
                    "split": split,
                    "SR": m["SR"],
                    "mean_ann": m["mean_ann"],
                    "vol_ann": m["vol_ann"],
                    "sdf_SR": _metrics(sdf.loc[sl], scenario["bars_per_year"])["SR"],
                    "turnover_component_mean": float(turnover.loc[sl].dropna().mean()),
                    "net_exposure_component_mean": float(net_exposure.loc[sl].dropna().mean()),
                    "returns_path": str((out_dir / f"{tag}.returns.parquet").relative_to(ROOT)).replace("\\", "/"),
                }
            )
    out = out_dir / "aipt_asset_signal_seed_ensemble_summary.csv"
    if rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        out.write_text("", encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dirs", nargs="+", required=True)
    p.add_argument("--out-dir", default="experiments/results/aipt_asset_signal_postprocess")
    args = p.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "script": "experiments/aipt_asset_signal_postprocess.py",
        "started_at": datetime.now().isoformat(),
        "argv": sys.argv,
        "run_dirs": args.run_dirs,
        "mode": "asset_signal_seed_surface_postprocess",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    results = _read_results([ROOT / d for d in args.run_dirs])
    surface = write_surface_aggregate(results, out_dir)
    ensemble = write_seed_ensembles(results, out_dir)
    print(f"[done] inputs={len(results)} surface={surface.relative_to(ROOT)} ensemble={ensemble.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
