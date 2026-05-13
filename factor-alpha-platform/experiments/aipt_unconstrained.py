"""
Pure AIPT replication: unconstrained rolling ridge SDF.

This intentionally does NOT create executable asset weights. It matches the
paper's first object:

    F[t+1] = S(Z[t])' R[t+1] / sqrt(N[t])
    lambda[t] = (E_t[F F'] + z I)^(-1) E_t[F]
    R_sdf[t+1] = lambda[t]' F[t+1]

No dollar neutrality, no L1 normalization, no max position caps, no fees, no
execution-cost kernel. Constraints should be layered on only after this object
is behaving like the paper.
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

from experiments.aipt_replication import (
    SCENARIOS,
    ExperimentSpec,
    _fit_lambda,
    _parse_csv_floats,
    _parse_csv_ints,
    compute_hjd,
    load_market_data,
    make_characteristic_tensor,
    make_forward_returns,
    make_random_params,
    project_datasources,
    random_features_for_date,
)


@dataclass(frozen=True)
class UnconstrainedSpec:
    scenario: str
    source_set: str
    n_features: int
    ridge_z: float
    activation: str
    seed: int
    demean_features: bool
    projected_sources: bool
    project_top_k: int
    dynamic_universe: bool


def _metrics(x: pd.Series, bars_per_year: int) -> dict[str, float]:
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 3:
        return {"n_bars": int(len(x)), "SR": float("nan"), "mean_ann": float("nan"), "vol_ann": float("nan")}
    sd = float(x.std())
    mu = float(x.mean())
    return {
        "n_bars": int(len(x)),
        "SR": (mu / sd) * math.sqrt(bars_per_year) if sd > 0 else float("nan"),
        "mean_ann": mu * bars_per_year,
        "vol_ann": sd * math.sqrt(bars_per_year),
    }


def build_factor_returns_unconstrained(
    x: np.ndarray,
    active: np.ndarray,
    fwd_ret: pd.DataFrame,
    random_w: np.ndarray,
    gamma: np.ndarray,
    activation: str,
    *,
    min_names: int,
    demean_features: bool,
) -> np.ndarray:
    ret_np = fwd_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float64)
    n_dates = x.shape[0]
    n_features = random_w.shape[1] * 2 if activation == "sincos" else random_w.shape[1]
    factors = np.zeros((n_dates, n_features), dtype=np.float64)
    for t in range(n_dates):
        n_active = int(active[t].sum())
        if n_active < min_names:
            continue
        s = random_features_for_date(
            x[t],
            active[t],
            random_w,
            gamma,
            activation,
            demean_features=demean_features,
        )
        factors[t] = (s.T @ ret_np[t]) / math.sqrt(n_active)
    return factors


def rolling_sdf_returns(
    factors: np.ndarray,
    dates: pd.DatetimeIndex,
    *,
    train_window: int,
    rebalance_every: int,
    delay: int,
    ridge_z: float,
) -> tuple[pd.Series, pd.DataFrame]:
    sdf = pd.Series(0.0, index=dates, dtype=float)
    n_features = factors.shape[1]
    lambda_ts = pd.DataFrame(np.nan, index=dates, columns=[f"f{i}" for i in range(n_features)])
    lam = np.zeros(n_features, dtype=np.float64)
    last_fit = -10**9
    for t in range(len(dates)):
        train_end = t - delay
        train_start = train_end - train_window
        if train_start < 0 or train_end <= train_start:
            continue
        if (t - last_fit) >= rebalance_every:
            lam = _fit_lambda(factors[train_start:train_end], ridge_z)
            last_fit = t
        sdf.iloc[t] = float(np.nan_to_num(factors[t]) @ lam)
        if t == last_fit:
            lambda_ts.iloc[t] = lam
    return sdf, lambda_ts


def split_sdf_metrics(sdf: pd.Series, scenario_name: str) -> dict[str, dict[str, float]]:
    scenario = SCENARIOS[scenario_name]
    splits = {
        "TRAIN": slice(None, scenario.split_train_end),
        "VAL": slice(scenario.split_train_end, scenario.split_val_end),
        "TEST": slice(scenario.split_val_end, None),
        "VAL+TEST": slice(scenario.split_train_end, None),
        "FULL": slice(None, None),
    }
    return {name: _metrics(sdf.loc[sl], scenario.bars_per_year) for name, sl in splits.items()}


def run_one(spec: UnconstrainedSpec, out_dir: Path) -> dict:
    scenario = SCENARIOS[spec.scenario]
    fields = scenario.source_sets[spec.source_set]
    print(
        f"[sdf] {spec.scenario} source={spec.source_set} P={spec.n_features} "
        f"z={spec.ridge_z:g} act={spec.activation} seed={spec.seed} "
        f"demean={spec.demean_features} dynamic_uni={spec.dynamic_universe}",
        flush=True,
    )
    t0 = time.time()
    uni, mats, _close, available_fields = load_market_data(
        scenario,
        fields,
        root=ROOT,
        dynamic_universe=spec.dynamic_universe,
    )
    fwd_ret = make_forward_returns(mats, scenario.delay)
    selected_fields = available_fields
    field_scales = {f: 1.0 for f in selected_fields}
    projection_scores: dict[str, float] = {}
    if spec.projected_sources:
        selected_fields, field_scales, projection_scores = project_datasources(
            scenario,
            uni,
            mats,
            available_fields,
            fwd_ret,
            top_k=spec.project_top_k,
        )
    x, active, used_fields = make_characteristic_tensor(uni, mats, selected_fields, field_scales=field_scales)
    random_w, gamma = make_random_params(len(used_fields), spec.n_features, spec.activation, spec.seed)
    factors = build_factor_returns_unconstrained(
        x,
        active,
        fwd_ret,
        random_w,
        gamma,
        spec.activation,
        min_names=scenario.min_names,
        demean_features=spec.demean_features,
    )
    sdf, lambda_ts = rolling_sdf_returns(
        factors,
        uni.index,
        train_window=scenario.train_window,
        rebalance_every=scenario.rebalance_every,
        delay=scenario.delay,
        ridge_z=spec.ridge_z,
    )
    metrics = split_sdf_metrics(sdf, spec.scenario)
    hjd = compute_hjd(sdf, factors, uni.index, scenario.split_val_end)
    elapsed = time.time() - t0
    result = {
        "spec": asdict(spec),
        "scenario": asdict(scenario),
        "n_dates": len(uni.index),
        "n_names": len(uni.columns),
        "n_fields": len(used_fields),
        "used_fields": used_fields,
        "field_scales": field_scales,
        "projection_scores": projection_scores,
        "metrics": metrics,
        "hjd_test": hjd,
        "elapsed_sec": elapsed,
        "lookahead_audit": {
            "features": "S(Z[t]) only",
            "factor_return": "S(Z[t])' R[t+1] / sqrt(N[t])",
            "fit_uses_factor_rows": f"< t-{scenario.delay}",
            "dollar_neutral": False,
            "asset_weight_constraints": False,
            "fees": False,
            "dynamic_universe": spec.dynamic_universe,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"{spec.scenario}__{spec.source_set}{'__proj' if spec.projected_sources else ''}"
        f"{'__dynuni' if spec.dynamic_universe else ''}"
        f"__P{spec.n_features}__z{spec.ridge_z:g}"
        f"__{spec.activation}__seed{spec.seed}__demean{int(spec.demean_features)}"
    ).replace(".", "p")
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    pd.DataFrame({"sdf_return": sdf}).to_parquet(out_dir / f"{tag}.sdf.parquet")
    lambda_ts.dropna(how="all").tail(200).to_parquet(out_dir / f"{tag}.lambda_tail.parquet")
    print(
        f"      SDF SR VAL={metrics['VAL']['SR']:+.2f} TEST={metrics['TEST']['SR']:+.2f} "
        f"V+T={metrics['VAL+TEST']['SR']:+.2f} FULL={metrics['FULL']['SR']:+.2f} "
        f"HJD={hjd:.4g} {elapsed:.1f}s",
        flush=True,
    )
    return result


def make_specs(args: argparse.Namespace) -> list[UnconstrainedSpec]:
    scenarios = args.scenarios or []
    if args.scenario:
        scenarios.append(args.scenario)
    if not scenarios:
        scenarios = ["equity_smallcap_d0", "equity_smallcap_d1", "kucoin_top100"]
    p_grid = _parse_csv_ints(args.p_grid)
    z_grid = _parse_csv_floats(args.z_grid)
    seeds = _parse_csv_ints(args.seeds)
    activations = [x.strip() for x in args.activations.split(",") if x.strip()]
    specs: list[UnconstrainedSpec] = []
    for scenario_name in scenarios:
        scenario = SCENARIOS[scenario_name]
        source_sets = [x.strip() for x in args.source_sets.split(",") if x.strip()]
        if not source_sets or source_sets == ["default"]:
            source_sets = [scenario.default_source_set]
        for source_set in source_sets:
            if source_set not in scenario.source_sets:
                continue
            for p in p_grid:
                for z in z_grid:
                    for activation in activations:
                        if activation == "sincos" and p % 2:
                            continue
                        for seed in seeds:
                            specs.append(
                                UnconstrainedSpec(
                                    scenario=scenario_name,
                                    source_set=source_set,
                                    n_features=p,
                                    ridge_z=z,
                                    activation=activation,
                                    seed=seed,
                                    demean_features=args.demean_features,
                                    projected_sources=args.projected_sources,
                                    project_top_k=args.project_top_k,
                                    dynamic_universe=args.dynamic_universe,
                                )
                            )
    if args.limit:
        specs = specs[: args.limit]
    return specs


def write_summary(results: list[dict], out_dir: Path) -> Path:
    rows = []
    for r in results:
        spec = r["spec"]
        base = {
            "scenario": spec["scenario"],
            "source_set": spec["source_set"],
            "P": spec["n_features"],
            "z": spec["ridge_z"],
            "activation": spec["activation"],
            "seed": spec["seed"],
            "demean_features": spec["demean_features"],
            "projected_sources": spec["projected_sources"],
            "project_top_k": spec["project_top_k"],
            "dynamic_universe": spec.get("dynamic_universe", False),
            "n_names": r["n_names"],
            "n_fields": r["n_fields"],
            "hjd_test": r["hjd_test"],
            "elapsed_sec": r["elapsed_sec"],
        }
        for split, metrics in r["metrics"].items():
            row = dict(base)
            row["split"] = split
            row.update(metrics)
            rows.append(row)
    out = out_dir / "aipt_unconstrained_summary.csv"
    if rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (out_dir / "aipt_unconstrained_summary.json").write_text(
        json.dumps(results, indent=2, default=float),
        encoding="utf-8",
    )
    return out


def write_run_manifest(args: argparse.Namespace, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "script": "experiments/aipt_unconstrained.py",
        "started_at": datetime.now().isoformat(),
        "cwd": str(ROOT),
        "argv": sys.argv,
        "args": vars(args),
        "mode": "unconstrained_paper_sdf",
        "notes": "No dollar neutrality, no asset caps, no gross normalization, no fees.",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=sorted(SCENARIOS), default=None)
    p.add_argument("--scenarios", nargs="*", default=None)
    p.add_argument("--source-sets", default="default")
    p.add_argument("--p-grid", default="64,256,1024")
    p.add_argument("--z-grid", default="0.001,0.01,0.1")
    p.add_argument("--activations", default="sincos")
    p.add_argument("--seeds", default="1")
    p.add_argument("--demean-features", action="store_true")
    p.add_argument("--projected-sources", action="store_true")
    p.add_argument("--project-top-k", type=int, default=12)
    p.add_argument("--dynamic-universe", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out-dir", default="experiments/results/aipt_unconstrained")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    out_dir = ROOT / args.out_dir
    write_run_manifest(args, out_dir)
    specs = make_specs(args)
    print(f"[setup] unconstrained cells={len(specs)} -> {out_dir.relative_to(ROOT)}", flush=True)
    results: list[dict] = []
    failures: list[dict] = []
    for spec in specs:
        try:
            results.append(run_one(spec, out_dir))
            write_summary(results, out_dir)
        except Exception as exc:
            failures.append({"spec": asdict(spec), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[fail] {spec}: {type(exc).__name__}: {exc}", flush=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "aipt_unconstrained_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    summary = write_summary(results, out_dir)
    print(f"[done] results={len(results)} failures={len(failures)} summary={summary.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
