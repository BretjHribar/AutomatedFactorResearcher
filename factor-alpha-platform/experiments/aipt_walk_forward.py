"""Walk-forward selector over already-run AIPT candidate return files.

For each scenario and fold:

  1. Score every candidate only on a trailing train window.
  2. Drop a calendar washout gap.
  3. Trade the selected candidate for the next calendar month.
  4. Roll forward by one live window and repeat.

The train mask is stricter than a simple date slice: a row is eligible for
training only when the forward-return label endpoint is still before the
washout starts. The live mask similarly excludes rows whose return endpoint
would spill into the next fold.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.aipt_replication import SCENARIOS


@dataclass
class Candidate:
    candidate_id: str
    run_dir: str
    tag: str
    kind: str
    scenario: str
    return_col: str
    returns: pd.Series
    spec: dict[str, Any]


def _json_scalar(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (str, bool, int)):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        return x.item()
    except Exception:
        return str(x)


def _metrics(x: pd.Series, bars_per_year: int) -> dict[str, float]:
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 3:
        return {
            "n_bars": int(len(x)),
            "SR": float("nan"),
            "mean_ann": float("nan"),
            "vol_ann": float("nan"),
            "max_dd": float("nan"),
        }
    sd = float(x.std())
    mu = float(x.mean())
    eq = (1.0 + x).cumprod()
    return {
        "n_bars": int(len(x)),
        "SR": (mu / sd) * math.sqrt(bars_per_year) if sd > 0 else float("nan"),
        "mean_ann": mu * bars_per_year,
        "vol_ann": sd * math.sqrt(bars_per_year),
        "max_dd": float((eq / eq.cummax() - 1.0).min()),
    }


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _read_result_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or "spec" not in data:
        return None
    if not isinstance(data.get("spec"), dict):
        return None
    return data


def _candidate_return_path(json_path: Path) -> tuple[Path | None, str, str]:
    sdf = json_path.with_suffix(".sdf.parquet")
    if sdf.exists():
        return sdf, "sdf_return", "unconstrained_sdf"
    ret = json_path.with_suffix(".returns.parquet")
    if ret.exists():
        return ret, "auto", "stepwise"
    return None, "", ""


def _choose_return_column(df: pd.DataFrame, requested: str, default_col: str) -> str:
    if requested != "auto":
        if requested not in df.columns:
            raise ValueError(f"requested return column {requested!r} not in {list(df.columns)}")
        return requested
    if default_col != "auto" and default_col in df.columns:
        return default_col
    for col in ["net", "sdf_return", "sdf", "gross"]:
        if col in df.columns:
            return col
    raise ValueError(f"could not infer return column from {list(df.columns)}")


def load_candidates(
    input_dirs: list[Path],
    *,
    return_col: str,
    scenarios: set[str] | None,
    include_layers: set[str] | None,
    exclude_layers: set[str] | None,
) -> list[Candidate]:
    out: list[Candidate] = []
    for run_dir in input_dirs:
        for json_path in sorted(run_dir.glob("*.json")):
            data = _read_result_json(json_path)
            if data is None:
                continue
            spec = dict(data["spec"])
            scenario = str(spec.get("scenario", ""))
            if scenario not in SCENARIOS:
                continue
            if scenarios and scenario not in scenarios:
                continue
            layer = spec.get("layer")
            if include_layers and str(layer) not in include_layers:
                continue
            if exclude_layers and str(layer) in exclude_layers:
                continue
            ret_path, default_col, kind = _candidate_return_path(json_path)
            if ret_path is None:
                continue
            try:
                df = pd.read_parquet(ret_path)
                col = _choose_return_column(df, return_col, default_col)
            except Exception as exc:
                print(f"[skip] {ret_path}: {type(exc).__name__}: {exc}", flush=True)
                continue
            ser = df[col]
            if not isinstance(ser.index, pd.DatetimeIndex):
                ser.index = pd.to_datetime(ser.index, errors="coerce")
                ser = ser[ser.index.notna()]
            ser = ser.sort_index().astype(float)
            tag = json_path.stem
            candidate_id = f"{run_dir.name}/{tag}/{col}"
            out.append(
                Candidate(
                    candidate_id=candidate_id,
                    run_dir=_rel(run_dir),
                    tag=tag,
                    kind=kind,
                    scenario=scenario,
                    return_col=col,
                    returns=ser,
                    spec=spec,
                )
            )
    return out


def _label_end_dates(index: pd.DatetimeIndex, delay: int) -> pd.Series:
    n = len(index)
    endpoints: list[pd.Timestamp | pd.NaT] = []
    horizon = delay + 1
    for i in range(n):
        j = i + horizon
        endpoints.append(index[j] if j < n else pd.NaT)
    return pd.Series(endpoints, index=index)


def _month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts).to_period("M").to_timestamp()


def _first_live_start(index: pd.DatetimeIndex, train_months: int, washout_days: int) -> pd.Timestamp:
    raw = index.min() + pd.DateOffset(months=train_months) + pd.Timedelta(days=washout_days)
    first = _month_start(raw)
    if first < raw:
        first = first + pd.DateOffset(months=1)
    return first


def _fold_starts(
    index: pd.DatetimeIndex,
    *,
    train_months: int,
    live_months: int,
    washout_days: int,
    start: str | None,
    end: str | None,
) -> list[pd.Timestamp]:
    first = _first_live_start(index, train_months, washout_days)
    if start:
        first = max(first, _month_start(pd.Timestamp(start)))
    last_allowed = index.max()
    if end:
        last_allowed = min(last_allowed, pd.Timestamp(end))
    starts: list[pd.Timestamp] = []
    cur = first
    while cur + pd.DateOffset(months=live_months) <= last_allowed:
        starts.append(pd.Timestamp(cur))
        cur = cur + pd.DateOffset(months=live_months)
    return starts


def _spec_flat(spec: dict[str, Any]) -> dict[str, Any]:
    out = {
        "source_set": spec.get("source_set"),
        "P": spec.get("n_features"),
        "z": spec.get("ridge_z"),
        "activation": spec.get("activation"),
        "seed": spec.get("seed"),
        "ensemble_n": spec.get("ensemble_n"),
        "layer": spec.get("layer"),
        "cost_tau": spec.get("cost_tau"),
        "turnover_cap": spec.get("turnover_cap"),
        "blend": spec.get("blend"),
        "qp_alpha_scale": spec.get("qp_alpha_scale"),
        "qp_risk_lambda": spec.get("qp_risk_lambda"),
        "max_weight": spec.get("max_weight"),
        "demean_features": spec.get("demean_features"),
        "projected_sources": spec.get("projected_sources"),
        "project_top_k": spec.get("project_top_k"),
        "dynamic_universe": spec.get("dynamic_universe"),
        "train_window": spec.get("train_window"),
        "rebalance_every": spec.get("rebalance_every"),
        "start_override": spec.get("start_override"),
        "weight_mode": spec.get("weight_mode"),
    }
    return {k: _json_scalar(v) for k, v in out.items() if v is not None}


def ensemble_by_seed(candidates: list[Candidate]) -> list[Candidate]:
    groups: dict[str, list[Candidate]] = {}
    group_specs: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        flat = _spec_flat(cand.spec)
        flat.pop("seed", None)
        key_obj = {
            "scenario": cand.scenario,
            "kind": cand.kind,
            "return_col": cand.return_col,
            "spec": flat,
        }
        key = json.dumps(key_obj, sort_keys=True, default=str)
        groups.setdefault(key, []).append(cand)
        group_specs[key] = flat

    out: list[Candidate] = []
    for key, members in sorted(groups.items()):
        if len(members) == 1:
            out.append(members[0])
            continue
        common_index = members[0].returns.index
        for member in members[1:]:
            common_index = common_index.intersection(member.returns.index)
        common_index = pd.DatetimeIndex(common_index).sort_values()
        if common_index.empty:
            continue
        stacked = pd.concat([m.returns.reindex(common_index) for m in members], axis=1)
        ser = stacked.mean(axis=1)
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        first = members[0]
        spec = dict(first.spec)
        spec["seed"] = "ensemble"
        spec["ensemble_n"] = len(members)
        spec["ensemble_member_ids"] = [m.candidate_id for m in members]
        candidate_id = f"seed_ensemble/{first.scenario}/{digest}/{first.return_col}"
        out.append(
            Candidate(
                candidate_id=candidate_id,
                run_dir=";".join(sorted({m.run_dir for m in members})),
                tag=f"seed_ensemble_{digest}",
                kind=f"{first.kind}_seed_ensemble",
                scenario=first.scenario,
                return_col=first.return_col,
                returns=ser,
                spec=spec,
            )
        )
    return out


def run_walk_forward(
    candidates: list[Candidate],
    *,
    train_months: int,
    live_months: int,
    washout_days: int,
    min_train_bars: int,
    min_live_bars: int,
    start: str | None,
    end: str | None,
    selection_metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    selected_return_rows: list[dict[str, Any]] = []

    by_scenario: dict[str, list[Candidate]] = {}
    for cand in candidates:
        by_scenario.setdefault(cand.scenario, []).append(cand)

    for scenario_name, group in sorted(by_scenario.items()):
        scenario = SCENARIOS[scenario_name]
        common_index = group[0].returns.index
        for cand in group[1:]:
            common_index = common_index.intersection(cand.returns.index)
        common_index = pd.DatetimeIndex(common_index).sort_values()
        if common_index.empty:
            continue
        end_dates = _label_end_dates(common_index, scenario.delay)
        starts = _fold_starts(
            common_index,
            train_months=train_months,
            live_months=live_months,
            washout_days=washout_days,
            start=start,
            end=end,
        )
        print(f"[walk] {scenario_name}: candidates={len(group)} folds={len(starts)}", flush=True)

        aligned = {cand.candidate_id: cand.returns.reindex(common_index) for cand in group}
        cand_by_id = {cand.candidate_id: cand for cand in group}

        for fold_idx, live_start in enumerate(starts, start=1):
            live_end = live_start + pd.DateOffset(months=live_months)
            train_end = live_start - pd.Timedelta(days=washout_days)
            train_start = train_end - pd.DateOffset(months=train_months)

            train_mask = (
                (common_index >= train_start)
                & (common_index < train_end)
                & end_dates.notna().values
                & (end_dates.values < np.datetime64(train_end))
            )
            live_mask = (
                (common_index >= live_start)
                & (common_index < live_end)
                & end_dates.notna().values
                & (end_dates.values < np.datetime64(live_end))
            )
            train_dates = common_index[train_mask]
            live_dates = common_index[live_mask]
            if len(train_dates) < min_train_bars or len(live_dates) < min_live_bars:
                continue

            fold_candidates: list[dict[str, Any]] = []
            for cand in group:
                ser = aligned[cand.candidate_id]
                train_m = _metrics(ser.loc[train_dates], scenario.bars_per_year)
                live_m = _metrics(ser.loc[live_dates], scenario.bars_per_year)
                row = {
                    "scenario": scenario_name,
                    "fold": fold_idx,
                    "train_start": train_start.isoformat(),
                    "train_end_exclusive": train_end.isoformat(),
                    "washout_days": washout_days,
                    "live_start": live_start.isoformat(),
                    "live_end_exclusive": live_end.isoformat(),
                    "candidate_id": cand.candidate_id,
                    "run_dir": cand.run_dir,
                    "tag": cand.tag,
                    "kind": cand.kind,
                    "return_col": cand.return_col,
                    "train_n_bars": train_m["n_bars"],
                    "train_SR": train_m["SR"],
                    "train_mean_ann": train_m["mean_ann"],
                    "train_vol_ann": train_m["vol_ann"],
                    "train_max_dd": train_m["max_dd"],
                    "live_n_bars": live_m["n_bars"],
                    "live_SR": live_m["SR"],
                    "live_mean_ann": live_m["mean_ann"],
                    "live_vol_ann": live_m["vol_ann"],
                    "live_max_dd": live_m["max_dd"],
                    "selected": False,
                    **_spec_flat(cand.spec),
                }
                fold_candidates.append(row)

            usable = pd.DataFrame(fold_candidates)
            usable[selection_metric] = pd.to_numeric(usable[f"train_{selection_metric}"], errors="coerce")
            usable = usable.dropna(subset=[selection_metric])
            if usable.empty:
                continue
            selected = usable.sort_values(selection_metric, ascending=False).iloc[0]
            selected_id = str(selected["candidate_id"])
            selected_ser = aligned[selected_id].loc[live_dates]
            for row in fold_candidates:
                if row["candidate_id"] == selected_id:
                    row["selected"] = True
            candidate_rows.extend(fold_candidates)

            max_train_signal = train_dates.max()
            max_train_label = end_dates.loc[train_dates].max()
            min_live_signal = live_dates.min()
            min_live_label = end_dates.loc[live_dates].min()
            audit_no_signal_overlap = bool(max_train_signal < min_live_signal)
            audit_no_label_overlap = bool(max_train_label < min_live_signal)
            audit_gap_days = float((min_live_signal - max_train_label).total_seconds() / 86400.0)

            fold_row = {
                "scenario": scenario_name,
                "fold": fold_idx,
                "train_start": train_start.isoformat(),
                "train_end_exclusive": train_end.isoformat(),
                "live_start": live_start.isoformat(),
                "live_end_exclusive": live_end.isoformat(),
                "washout_days": washout_days,
                "train_months": train_months,
                "live_months": live_months,
                "selection_metric": f"train_{selection_metric}",
                "selected_candidate_id": selected_id,
                "selected_train_SR": float(selected["train_SR"]),
                "selected_live_SR": float(selected["live_SR"]),
                "selected_train_mean_ann": float(selected["train_mean_ann"]),
                "selected_live_mean_ann": float(selected["live_mean_ann"]),
                "selected_train_n_bars": int(selected["train_n_bars"]),
                "selected_live_n_bars": int(selected["live_n_bars"]),
                "max_train_signal_date": max_train_signal.isoformat(),
                "max_train_label_end_date": pd.Timestamp(max_train_label).isoformat(),
                "min_live_signal_date": min_live_signal.isoformat(),
                "min_live_label_end_date": pd.Timestamp(min_live_label).isoformat(),
                "audit_no_signal_overlap": audit_no_signal_overlap,
                "audit_no_label_overlap": audit_no_label_overlap,
                "audit_label_to_live_gap_days": audit_gap_days,
                **_spec_flat(cand_by_id[selected_id].spec),
            }
            fold_rows.append(fold_row)

            for dt, value in selected_ser.items():
                selected_return_rows.append(
                    {
                        "scenario": scenario_name,
                        "date": pd.Timestamp(dt).isoformat(),
                        "fold": fold_idx,
                        "selected_candidate_id": selected_id,
                        "return": float(value),
                    }
                )

    cand_df = pd.DataFrame(candidate_rows)
    folds_df = pd.DataFrame(fold_rows)
    returns_df = pd.DataFrame(selected_return_rows)
    summary_rows: list[dict[str, Any]] = []
    if not returns_df.empty:
        returns_df["date"] = pd.to_datetime(returns_df["date"])
        for scenario_name, g in returns_df.groupby("scenario"):
            scenario = SCENARIOS[scenario_name]
            ser = pd.Series(g["return"].values, index=pd.DatetimeIndex(g["date"])).sort_index()
            live_m = _metrics(ser, scenario.bars_per_year)
            f = folds_df[folds_df["scenario"] == scenario_name]
            train_sr = float(pd.to_numeric(f["selected_train_SR"], errors="coerce").mean())
            train_mean = float(pd.to_numeric(f["selected_train_mean_ann"], errors="coerce").mean())
            base = {
                "scenario": scenario_name,
                "train_months": train_months,
                "live_months": live_months,
                "washout_days": washout_days,
                "selection_metric": f"train_{selection_metric}",
                "folds": int(len(f)),
                "candidates": int(len([c for c in candidates if c.scenario == scenario_name])),
                "audit_all_no_signal_overlap": bool(f["audit_no_signal_overlap"].all()) if not f.empty else False,
                "audit_all_no_label_overlap": bool(f["audit_no_label_overlap"].all()) if not f.empty else False,
            }
            summary_rows.append({**base, "split": "TRAIN", "SR": train_sr, "mean_ann": train_mean, "n_bars": int(f["selected_train_n_bars"].sum())})
            summary_rows.append({**base, "split": "VAL+TEST", **live_m})
            summary_rows.append({**base, "split": "FULL", **live_m})
    summary_df = pd.DataFrame(summary_rows)
    return cand_df, folds_df, returns_df, summary_df


def write_plots(out_dir: Path, cand_df: pd.DataFrame, folds_df: pd.DataFrame) -> None:
    if cand_df.empty or folds_df.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot-skip] matplotlib unavailable: {exc}", flush=True)
        return
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for scenario, f in folds_df.groupby("scenario"):
        f = f.copy()
        f["live_start_dt"] = pd.to_datetime(f["live_start"])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(f["live_start_dt"], f["selected_train_SR"], marker="o", label="selected train SR")
        ax.plot(f["live_start_dt"], f["selected_live_SR"], marker="o", label="live month SR")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{scenario} walk-forward selected train vs live SR")
        ax.set_xlabel("live month")
        ax.set_ylabel("Sharpe")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{scenario}__fold_sr.png", dpi=150)
        plt.close(fig)

        cols = [c for c in ["P", "z", "cost_tau", "turnover_cap", "qp_alpha_scale", "qp_risk_lambda"] if c in f.columns]
        for col in cols:
            vals = pd.to_numeric(f[col], errors="coerce")
            if vals.notna().sum() == 0:
                continue
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(f["live_start_dt"], vals, marker="o")
            ax.set_title(f"{scenario} selected {col}")
            ax.set_xlabel("live month")
            ax.set_ylabel(col)
            if col == "z" or col == "cost_tau":
                ax.set_yscale("log")
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(plot_dir / f"{scenario}__selected_{col}.png", dpi=150)
            plt.close(fig)

    if {"scenario", "P", "z", "train_SR"}.issubset(cand_df.columns):
        for scenario, g in cand_df.groupby("scenario"):
            work = g.copy()
            work["P"] = pd.to_numeric(work["P"], errors="coerce")
            work["z"] = pd.to_numeric(work["z"], errors="coerce")
            work["train_SR"] = pd.to_numeric(work["train_SR"], errors="coerce")
            work = work.dropna(subset=["P", "z", "train_SR"])
            if work.empty:
                continue
            pivot = work.pivot_table(index="P", columns="z", values="train_SR", aggfunc="mean")
            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(pivot.values, aspect="auto", origin="lower")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{x:g}" for x in pivot.columns], rotation=45, ha="right")
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{int(x)}" for x in pivot.index])
            ax.set_title(f"{scenario} mean walk-forward train SR surface")
            ax.set_xlabel("ridge z")
            ax.set_ylabel("P")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(plot_dir / f"{scenario}__mean_train_surface_P_z.png", dpi=150)
            plt.close(fig)

    index_lines = ["# AIPT Walk-Forward Plots", ""]
    for path in sorted(plot_dir.glob("*.png")):
        index_lines.append(f"- `{_rel(path)}`")
    (plot_dir / "index.md").write_text("\n".join(index_lines), encoding="utf-8")


def write_outputs(
    out_dir: Path,
    args: argparse.Namespace,
    candidates: list[Candidate],
    cand_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "script": "experiments/aipt_walk_forward.py",
        "started_at": datetime.now().isoformat(),
        "cwd": str(ROOT),
        "argv": sys.argv,
        "args": vars(args),
        "candidate_count": len(candidates),
        "lookahead_audit": {
            "selection": "candidate chosen only by trailing train-window score",
            "washout": f"{args.washout_days} calendar days between train label endpoints and live signals",
            "train_label_rule": "training rows require forward-return endpoint before train_end_exclusive",
            "live_label_rule": "live rows require forward-return endpoint before live_end_exclusive",
            "overlap": "fold output records signal and label overlap booleans",
        },
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    cand_df.to_csv(out_dir / "aipt_walk_forward_candidates.csv", index=False)
    folds_df.to_csv(out_dir / "aipt_walk_forward_folds.csv", index=False)
    summary_df.to_csv(out_dir / "aipt_walk_forward_summary.csv", index=False)
    if not returns_df.empty:
        returns_df.to_parquet(out_dir / "aipt_walk_forward_selected_returns.parquet", index=False)
    write_plots(out_dir, cand_df, folds_df)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dirs", nargs="+", required=True)
    p.add_argument("--scenarios", nargs="*", default=None)
    p.add_argument("--return-col", default="auto", help="auto, net, gross, sdf, or sdf_return")
    p.add_argument("--include-layers", default="", help="comma-separated stepwise layers to include")
    p.add_argument("--exclude-layers", default="", help="comma-separated stepwise layers to exclude")
    p.add_argument("--train-months", type=int, default=12)
    p.add_argument("--live-months", type=int, default=1)
    p.add_argument("--washout-days", type=int, default=7)
    p.add_argument("--min-train-bars", type=int, default=60)
    p.add_argument("--min-live-bars", type=int, default=10)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--selection-metric", choices=["SR"], default="SR")
    p.add_argument("--ensemble-seeds", action="store_true")
    p.add_argument("--out-dir", default="experiments/results/aipt_walk_forward")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    input_dirs = [(ROOT / x).resolve() for x in args.input_dirs]
    scenario_filter = set(args.scenarios) if args.scenarios else None
    include_layers = {x.strip() for x in args.include_layers.split(",") if x.strip()} or None
    exclude_layers = {x.strip() for x in args.exclude_layers.split(",") if x.strip()} or None
    candidates = load_candidates(
        input_dirs,
        return_col=args.return_col,
        scenarios=scenario_filter,
        include_layers=include_layers,
        exclude_layers=exclude_layers,
    )
    if args.ensemble_seeds:
        before = len(candidates)
        candidates = ensemble_by_seed(candidates)
        print(f"[ensemble] seed groups {before} -> {len(candidates)} candidates", flush=True)
    print(f"[setup] candidates={len(candidates)} from {len(input_dirs)} dirs", flush=True)
    cand_df, folds_df, returns_df, summary_df = run_walk_forward(
        candidates,
        train_months=args.train_months,
        live_months=args.live_months,
        washout_days=args.washout_days,
        min_train_bars=args.min_train_bars,
        min_live_bars=args.min_live_bars,
        start=args.start,
        end=args.end,
        selection_metric=args.selection_metric,
    )
    out_dir = ROOT / args.out_dir
    write_outputs(out_dir, args, candidates, cand_df, folds_df, returns_df, summary_df)
    print(
        f"[done] folds={len(folds_df)} candidate_scores={len(cand_df)} "
        f"summary={_rel(out_dir / 'aipt_walk_forward_summary.csv')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
