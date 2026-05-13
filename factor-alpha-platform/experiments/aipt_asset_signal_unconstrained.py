"""Unconstrained AIPT asset-signal benchmark.

The paper's first object is an SDF return. For trading-factor comparison, this
script converts the same rolling lambda into per-name scores

    alpha_i[t] = S(Z_i[t]) lambda[t] / sqrt(N[t])

and evaluates simple no-fee portfolios built from those scores. There is no QP,
no execution-cost kernel, no realized fee subtraction, and no required dollar
neutrality. Weight modes are deliberately primitive so we can see whether the
learned AIPT function is useful before adding constraints:

  raw_gross         weights = alpha / sum(abs(alpha))
  demean_gross      weights = (alpha - mean(alpha)) / sum(abs(.))
  rank_center_gross weights = (rank(alpha) - 0.5) / sum(abs(.))
  long_rank_gross   weights = rank(alpha) / sum(abs(.))
  subindustry_gross group-demean alpha as a WorldQuant Brain risk proxy

Timing follows the strict AIPT convention:
  delay=0 fit at t uses factor rows < t, earns close[t+1]/close[t]-1
  delay=1 fit at t uses factor rows < t-1, earns open[t+2]/open[t+1]-1
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.aipt_replication import (
    SCENARIOS,
    Scenario,
    _fit_lambda,
    _parse_csv_floats,
    _parse_csv_ints,
    load_market_data,
    make_characteristic_tensor,
    make_forward_returns,
    make_random_params,
    project_datasources,
    random_features_for_date,
)
from experiments.aipt_unconstrained import build_factor_returns_unconstrained


GROUP_LEVELS = ("sector", "industry", "subindustry")
WEIGHT_MODES = (
    "raw_gross",
    "demean_gross",
    "rank_center_gross",
    "long_rank_gross",
    "sector_gross",
    "industry_gross",
    "subindustry_gross",
    "sector_rank_gross",
    "industry_rank_gross",
    "subindustry_rank_gross",
)


@dataclass(frozen=True)
class AssetSignalSpec:
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
    train_window: int
    rebalance_every: int
    start_override: str | None
    weight_mode: str


def _scenario_for(spec: AssetSignalSpec) -> Scenario:
    base = SCENARIOS[spec.scenario]
    train_window = spec.train_window if spec.train_window > 0 else base.train_window
    rebalance_every = spec.rebalance_every if spec.rebalance_every > 0 else base.rebalance_every
    start = spec.start_override if spec.start_override else base.start
    suffix = f"__tw{train_window}__rb{rebalance_every}"
    if start:
        suffix += "__start" + start.replace("-", "")
    # load_market_data caches by scenario.name, so make override scenarios unique.
    return replace(
        base,
        name=base.name + suffix,
        train_window=train_window,
        rebalance_every=rebalance_every,
        start=start,
    )


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


def _split_metrics(pnl: pd.Series, scenario: Scenario) -> dict[str, dict[str, float]]:
    splits = {
        "TRAIN": slice(None, scenario.split_train_end),
        "VAL": slice(scenario.split_train_end, scenario.split_val_end),
        "TEST": slice(scenario.split_val_end, None),
        "VAL+TEST": slice(scenario.split_train_end, None),
        "FULL": slice(None, None),
    }
    return {label: _metrics(pnl.loc[sl], scenario.bars_per_year) for label, sl in splits.items()}


def _load_group_labels(tickers: list[str], level: str | None) -> np.ndarray | None:
    if level is None:
        return None
    path = ROOT / "data/fmp_cache/classifications.json"
    if not path.exists():
        raise FileNotFoundError(f"classifications file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return np.array([str(raw.get(t, {}).get(level, "Unknown")) for t in tickers], dtype=object)


def _mode_group_level(mode: str) -> str | None:
    for level in GROUP_LEVELS:
        if mode == f"{level}_gross" or mode == f"{level}_rank_gross":
            return level
    return None


def _group_demean(vals: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = vals.astype(np.float64, copy=True)
    for group in pd.unique(groups):
        mask = groups == group
        if mask.sum() > 0:
            out[mask] -= out[mask].mean()
    return out


def _weights_from_scores(
    scores: np.ndarray,
    active: np.ndarray,
    mode: str,
    group_labels: np.ndarray | None,
) -> np.ndarray:
    out = np.zeros_like(scores, dtype=np.float64)
    mask = active & np.isfinite(scores)
    if mask.sum() < 2:
        return out
    vals = scores[mask].astype(np.float64, copy=True)
    groups = group_labels[mask] if group_labels is not None else None
    if mode == "raw_gross":
        pass
    elif mode == "demean_gross":
        vals -= vals.mean()
    elif mode == "rank_center_gross":
        vals = rankdata(vals, method="average") / len(vals) - 0.5
    elif mode == "long_rank_gross":
        vals = rankdata(vals, method="average") / len(vals)
    elif mode.endswith("_rank_gross") and _mode_group_level(mode) is not None:
        if groups is None:
            raise ValueError(f"{mode} requires group labels")
        vals = rankdata(vals, method="average") / len(vals)
        vals = _group_demean(vals, groups)
    elif mode.endswith("_gross") and _mode_group_level(mode) is not None:
        if groups is None:
            raise ValueError(f"{mode} requires group labels")
        vals = _group_demean(vals, groups)
    else:
        raise ValueError(f"unknown weight_mode {mode!r}")
    gross = np.abs(vals).sum()
    if gross <= 1e-12:
        return out
    out[mask] = vals / gross
    return out


def run_asset_signal(
    *,
    scenario: Scenario,
    x: np.ndarray,
    active: np.ndarray,
    fwd_ret: pd.DataFrame,
    factors: np.ndarray,
    random_w: np.ndarray,
    gamma: np.ndarray,
    activation: str,
    demean_features: bool,
    ridge_z: float,
    weight_mode: str,
    group_labels: np.ndarray | None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    dates = fwd_ret.index
    ret_np = fwd_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float64)
    pnl = pd.Series(np.nan, index=dates, dtype=float)
    sdf = pd.Series(np.nan, index=dates, dtype=float)
    turnover = pd.Series(np.nan, index=dates, dtype=float)
    net_exposure = pd.Series(np.nan, index=dates, dtype=float)
    lam = np.zeros(factors.shape[1], dtype=np.float64)
    w_prev: np.ndarray | None = None
    last_fit = -10**9

    for t in range(len(dates)):
        train_end = t - scenario.delay
        train_start = train_end - scenario.train_window
        if train_start < 0 or train_end <= train_start:
            continue
        if int(active[t].sum()) < scenario.min_names:
            continue
        if (t - last_fit) >= scenario.rebalance_every:
            lam = _fit_lambda(factors[train_start:train_end], ridge_z)
            last_fit = t

        s_t = random_features_for_date(
            x[t],
            active[t],
            random_w,
            gamma,
            activation,
            demean_features=demean_features,
        )
        n_active = max(int(active[t].sum()), 1)
        scores = (s_t @ lam) / math.sqrt(n_active)
        weights = _weights_from_scores(scores, active[t], weight_mode, group_labels)
        pnl.iloc[t] = float(weights @ ret_np[t])
        sdf.iloc[t] = float(np.nan_to_num(factors[t]) @ lam)
        turnover.iloc[t] = float(np.abs(weights - w_prev).sum()) if w_prev is not None else 0.0
        net_exposure.iloc[t] = float(weights.sum())
        w_prev = weights

    return pnl, sdf, turnover, net_exposure


def run_one(spec: AssetSignalSpec, out_dir: Path) -> dict:
    scenario = _scenario_for(spec)
    base_name = spec.scenario
    fields = SCENARIOS[base_name].source_sets[spec.source_set]
    print(
        f"[asset] {base_name} source={spec.source_set} P={spec.n_features} z={spec.ridge_z:g} "
        f"seed={spec.seed} mode={spec.weight_mode} tw={scenario.train_window} "
        f"rb={scenario.rebalance_every} start={scenario.start} dyn={spec.dynamic_universe}",
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
    group_level = _mode_group_level(spec.weight_mode)
    group_labels = _load_group_labels(uni.columns.tolist(), group_level) if scenario.market == "equity" else None
    pnl, sdf, turnover, net_exposure = run_asset_signal(
        scenario=scenario,
        x=x,
        active=active,
        fwd_ret=fwd_ret,
        factors=factors,
        random_w=random_w,
        gamma=gamma,
        activation=spec.activation,
        demean_features=spec.demean_features,
        ridge_z=spec.ridge_z,
        weight_mode=spec.weight_mode,
        group_labels=group_labels,
    )
    metrics = _split_metrics(pnl, scenario)
    sdf_metrics = _split_metrics(sdf, scenario)
    elapsed = time.time() - t0
    split_slices = {
        "TRAIN": slice(None, scenario.split_train_end),
        "VAL": slice(scenario.split_train_end, scenario.split_val_end),
        "TEST": slice(scenario.split_val_end, None),
        "VAL+TEST": slice(scenario.split_train_end, None),
        "FULL": slice(None, None),
    }
    result = {
        "spec": asdict(spec),
        "scenario": asdict(scenario),
        "base_scenario": asdict(SCENARIOS[base_name]),
        "n_dates": len(uni.index),
        "n_names": len(uni.columns),
        "n_fields": len(used_fields),
        "used_fields": used_fields,
        "field_scales": field_scales,
        "projection_scores": projection_scores,
        "metrics": metrics,
        "sdf_metrics": sdf_metrics,
        "turnover": {k: float(turnover.loc[v].replace([np.inf, -np.inf], np.nan).dropna().mean()) for k, v in split_slices.items()},
        "net_exposure": {
            k: float(net_exposure.loc[v].replace([np.inf, -np.inf], np.nan).dropna().mean())
            for k, v in split_slices.items()
        },
        "elapsed_sec": elapsed,
        "lookahead_audit": {
            "signal_index": "scores use S(Z[t]) and lambda fit only on known factor rows",
            "delay": scenario.delay,
            "fit_uses_factor_rows": f"< t-{scenario.delay}",
            "fwd_return": "close[t+1]/close[t]-1 for delay=0; open[t+2]/open[t+1]-1 for delay=1",
            "fees": False,
            "qp": False,
            "execution_cost_kernel": False,
            "weight_mode": spec.weight_mode,
            "group_neutralization": (
                f"{group_level} groups from data/fmp_cache/classifications.json; static classification proxy"
                if group_level
                else "none"
            ),
            "dynamic_universe": spec.dynamic_universe,
            "start_override": spec.start_override,
        },
    }
    tag = (
        f"{base_name}__{spec.source_set}{'__proj' if spec.projected_sources else ''}"
        f"{'__dynuni' if spec.dynamic_universe else ''}__P{spec.n_features}"
        f"__z{spec.ridge_z:g}__{spec.activation}__seed{spec.seed}"
        f"__tw{scenario.train_window}__rb{scenario.rebalance_every}"
        f"__{spec.weight_mode}__demeanfeat{int(spec.demean_features)}"
    ).replace(".", "p")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    pd.DataFrame(
        {
            "asset_return": pnl,
            "sdf_return": sdf,
            "turnover": turnover,
            "net_exposure": net_exposure,
        }
    ).to_parquet(out_dir / f"{tag}.returns.parquet")
    print(
        f"      asset SR train={metrics['TRAIN']['SR']:+.2f} val={metrics['VAL']['SR']:+.2f} "
        f"test={metrics['TEST']['SR']:+.2f} v+t={metrics['VAL+TEST']['SR']:+.2f}; "
        f"sdf v+t={sdf_metrics['VAL+TEST']['SR']:+.2f}; "
        f"to={result['turnover']['VAL+TEST']*100:.1f}% net={result['net_exposure']['VAL+TEST']:+.2f} "
        f"{elapsed:.1f}s",
        flush=True,
    )
    return result


def make_specs(args: argparse.Namespace) -> list[AssetSignalSpec]:
    scenarios = args.scenarios or []
    if args.scenario:
        scenarios.append(args.scenario)
    if not scenarios:
        scenarios = ["equity_top3000_d0", "equity_top3000_d1"]
    p_grid = _parse_csv_ints(args.p_grid)
    z_grid = _parse_csv_floats(args.z_grid)
    seeds = _parse_csv_ints(args.seeds)
    activations = [x.strip() for x in args.activations.split(",") if x.strip()]
    weight_modes = [x.strip() for x in args.weight_modes.split(",") if x.strip()]
    bad_modes = [x for x in weight_modes if x not in WEIGHT_MODES]
    if bad_modes:
        raise ValueError(f"unknown weight modes: {bad_modes}")
    specs: list[AssetSignalSpec] = []
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
                            for mode in weight_modes:
                                specs.append(
                                    AssetSignalSpec(
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
                                        train_window=args.train_window,
                                        rebalance_every=args.rebalance_every,
                                        start_override=args.start_override,
                                        weight_mode=mode,
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
            "dynamic_universe": spec["dynamic_universe"],
            "train_window": spec["train_window"],
            "rebalance_every": spec["rebalance_every"],
            "start_override": spec["start_override"],
            "weight_mode": spec["weight_mode"],
            "n_names": r["n_names"],
            "n_fields": r["n_fields"],
            "elapsed_sec": r["elapsed_sec"],
        }
        for split, metrics in r["metrics"].items():
            row = dict(base)
            row["split"] = split
            row["turnover_per_bar"] = r["turnover"][split]
            row["net_exposure_mean"] = r["net_exposure"][split]
            row["sdf_SR"] = r["sdf_metrics"][split]["SR"]
            row.update(metrics)
            rows.append(row)
    out = out_dir / "aipt_asset_signal_unconstrained_summary.csv"
    if rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (out_dir / "aipt_asset_signal_unconstrained_summary.json").write_text(
        json.dumps(results, indent=2, default=float),
        encoding="utf-8",
    )
    return out


def write_manifest(args: argparse.Namespace, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "script": "experiments/aipt_asset_signal_unconstrained.py",
        "started_at": datetime.now().isoformat(),
        "cwd": str(ROOT),
        "argv": sys.argv,
        "args": vars(args),
        "mode": "unconstrained_aipt_asset_signal_no_fee",
        "notes": "No QP, no cost kernel, no realized fees. Converts AIPT lambda to primitive gross-normalized asset portfolios.",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=sorted(SCENARIOS), default=None)
    p.add_argument("--scenarios", nargs="*", default=None)
    p.add_argument("--source-sets", default="default")
    p.add_argument("--p-grid", default="64,256")
    p.add_argument("--z-grid", default="0.001,0.01")
    p.add_argument("--activations", default="sincos")
    p.add_argument("--seeds", default="1")
    p.add_argument("--demean-features", action="store_true")
    p.add_argument("--projected-sources", action="store_true")
    p.add_argument("--project-top-k", type=int, default=12)
    p.add_argument("--dynamic-universe", action="store_true")
    p.add_argument("--train-window", type=int, default=0)
    p.add_argument("--rebalance-every", type=int, default=0)
    p.add_argument("--start-override", default=None)
    p.add_argument("--weight-modes", default="raw_gross,demean_gross,subindustry_gross,subindustry_rank_gross")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out-dir", default="experiments/results/aipt_asset_signal_unconstrained")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    out_dir = ROOT / args.out_dir
    write_manifest(args, out_dir)
    specs = make_specs(args)
    print(f"[setup] asset-signal cells={len(specs)} -> {out_dir.relative_to(ROOT)}", flush=True)
    results: list[dict] = []
    failures: list[dict] = []
    for spec in specs:
        try:
            results.append(run_one(spec, out_dir))
            write_summary(results, out_dir)
        except Exception as exc:
            failures.append({"spec": asdict(spec), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[fail] {spec}: {type(exc).__name__}: {exc}", flush=True)
            (out_dir / "aipt_asset_signal_unconstrained_failures.json").write_text(
                json.dumps(failures, indent=2),
                encoding="utf-8",
            )
    summary = write_summary(results, out_dir)
    print(f"[done] results={len(results)} failures={len(failures)} summary={summary.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
