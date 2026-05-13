"""Generate a central registry for all AIPT experiment outputs.

This scans `experiments/results/aipt*` and writes:

  - experiments/AIPT_EXPERIMENT_REGISTRY.json
  - experiments/AIPT_EXPERIMENT_REGISTRY.md

It intentionally includes diagnostic and stopped runs. Some early runs predate
command manifests, so their commands are recorded as reconstructed when exact
launch strings were not captured at run time.
"""
from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"
RESULTS = EXPERIMENTS / "results"


KNOWN_NOTES: dict[str, dict[str, str]] = {
    "aipt_smoke": {
        "class": "diagnostic constrained/cost smoke",
        "status_note": "early implementation smoke; not a paper-match result",
        "command": "not captured; reconstructed from output specs",
    },
    "aipt_smoke_eq": {
        "class": "diagnostic constrained/cost smoke",
        "status_note": "early equity smoke; not a paper-match result",
        "command": "not captured; reconstructed from output specs",
    },
    "aipt_smoke_p256": {
        "class": "diagnostic constrained/cost smoke",
        "status_note": "early P=256 smoke; constrained/cost layer",
        "command": "not captured; reconstructed from output specs",
    },
    "aipt_smoke_p1024": {
        "class": "diagnostic constrained/cost smoke",
        "status_note": "early P=1024 smoke; constrained/cost layer",
        "command": "not captured; reconstructed from output specs",
    },
    "aipt_extended": {
        "class": "diagnostic constrained/cost sweep",
        "status_note": "superseded by unconstrained-first workflow",
        "command": "not captured; reconstructed from output specs",
    },
    "aipt_extended2": {
        "class": "diagnostic constrained/cost sweep",
        "status_note": "completed but superseded; used before strict PIT-universe baseline",
        "command": "not captured; reconstructed from output specs",
    },
    "aipt_unconstrained_smoke": {
        "class": "unconstrained SDF smoke",
        "status_note": "legacy equity universe/matrix path; diagnostic only",
        "command": "python experiments\\aipt_unconstrained.py --scenario equity_smallcap_d0 --p-grid 64 --z-grid 0.001 --seeds 1 --limit 1 --out-dir experiments/results/aipt_unconstrained_smoke",
    },
    "aipt_unconstrained_main": {
        "class": "unconstrained SDF sweep",
        "status_note": "stopped when PIT issue found; legacy matrix path, diagnostic only",
        "command": "python experiments\\aipt_unconstrained.py --scenarios equity_smallcap_d0 equity_smallcap_d1 equity_top1000_d0 equity_top1000_d1 kucoin_top100 --source-sets default --p-grid 64,256,1024 --z-grid 0.00001,0.0001,0.001,0.01,0.1,1 --seeds 1,2,3 --out-dir experiments/results/aipt_unconstrained_main",
    },
    "aipt_unconstrained_pit_smoke": {
        "class": "unconstrained SDF smoke",
        "status_note": "PIT matrices with legacy universe; superseded by strict PIT universe smoke",
        "command": "python experiments\\aipt_unconstrained.py --scenario equity_smallcap_d0 --p-grid 64 --z-grid 0.001 --seeds 1 --limit 1 --out-dir experiments/results/aipt_unconstrained_pit_smoke",
    },
    "aipt_unconstrained_pit_main": {
        "class": "unconstrained SDF sweep",
        "status_note": "stopped when legacy equity universe survivorship channel was identified",
        "command": "python experiments\\aipt_unconstrained.py --scenarios equity_smallcap_d0 equity_smallcap_d1 equity_top1000_d0 equity_top1000_d1 kucoin_top100 --source-sets default --p-grid 64,256,1024 --z-grid 0.00001,0.0001,0.001,0.01,0.1,1 --seeds 1,2,3 --out-dir experiments/results/aipt_unconstrained_pit_main",
    },
    "aipt_unconstrained_strict_smoke": {
        "class": "strict unconstrained SDF smoke",
        "status_note": "PIT matrices plus experiment-local PIT universe; accepted baseline smoke",
        "command": "python experiments\\aipt_unconstrained.py --scenario equity_smallcap_d0 --p-grid 64 --z-grid 0.001 --seeds 1 --limit 1 --out-dir experiments/results/aipt_unconstrained_strict_smoke",
    },
    "aipt_unconstrained_strict_main": {
        "class": "strict unconstrained SDF sweep",
        "status_note": "current primary paper-matching baseline; running until all 270 cells finish",
        "command": "python experiments\\aipt_unconstrained.py --scenarios equity_smallcap_d0 equity_smallcap_d1 equity_top1000_d0 equity_top1000_d1 kucoin_top100 --source-sets default --p-grid 64,256,1024 --z-grid 0.00001,0.0001,0.001,0.01,0.1,1 --seeds 1,2,3 --out-dir experiments/results/aipt_unconstrained_strict_main",
    },
    "aipt_unconstrained_projection_smoke": {
        "class": "strict datasource smoke",
        "status_note": "price-only comparison, no costs/constraints",
        "command": "python experiments\\aipt_unconstrained.py --scenario equity_smallcap_d0 --source-sets price --p-grid 64 --z-grid 0.001 --seeds 1 --limit 1 --out-dir experiments/results/aipt_unconstrained_projection_smoke",
    },
    "aipt_unconstrained_projection_smoke_proj": {
        "class": "strict datasource projection smoke",
        "status_note": "train-only projected-source comparison, no costs/constraints",
        "command": "python experiments\\aipt_unconstrained.py --scenario equity_smallcap_d0 --source-sets all --projected-sources --project-top-k 8 --p-grid 64 --z-grid 0.001 --seeds 1 --limit 1 --out-dir experiments/results/aipt_unconstrained_projection_smoke_proj",
    },
    "aipt_unconstrained_dynamic_smallcap_d0_smoke": {
        "class": "strict dynamic-universe unconstrained smoke",
        "status_note": "Smallcap d0 using daily PIT top max_names by adv60, not frozen first-fit cohort; no costs/constraints",
    },
    "aipt_unconstrained_dynamic_smallcap_d0d1_p256": {
        "class": "strict dynamic-universe unconstrained sweep",
        "status_note": "Smallcap d0/d1 using daily PIT top max_names by adv60; P=256 ridge/seed comparison, no costs/constraints",
    },
    "aipt_unconstrained_top3000_dynamic_smoke": {
        "class": "strict dynamic-universe unconstrained smoke",
        "status_note": "TOP3000 ADV60 rolling PIT universe smoke; no costs/constraints",
    },
    "aipt_unconstrained_top3000_dynamic_d0_p64p256": {
        "class": "strict dynamic-universe unconstrained sweep",
        "status_note": "TOP3000 d0 rolling PIT ADV60 universe; P=64/256 ridge/seed comparison, no costs/constraints",
    },
    "aipt_unconstrained_top3000_dynamic_d1_p64p256": {
        "class": "strict dynamic-universe unconstrained sweep",
        "status_note": "TOP3000 d1 rolling PIT ADV60 universe; P=64/256 ridge/seed comparison, no costs/constraints",
    },
    "aipt_top3000_fixed_seed_ensemble": {
        "class": "strict TOP3000 factor post-analysis",
        "status_note": "Fixed-spec seed-ensemble TOP3000 no-cost SDF factor summaries; no QP, no execution costs",
    },
    "aipt_top3000_factor_postprocess": {
        "class": "strict TOP3000 factor post-analysis",
        "status_note": "TOP3000 seed-ensemble no-cost SDF factor with raw and trailing-known-vol-targeted variants; no QP, no execution costs",
    },
    "aipt_stepwise_smoke": {
        "class": "stepwise constraints/cost smoke",
        "status_note": "post-baseline decomposition smoke; no dollar neutrality",
        "command": "python experiments\\aipt_stepwise_constraints.py --scenario equity_smallcap_d0 --p-grid 64 --z-grid 0.001 --seeds 1 --layers raw_sdf,gross1_cap_fee --cost-taus 1 --limit 2 --out-dir experiments/results/aipt_stepwise_smoke",
    },
    "aipt_stepwise_strict_smallcap_d0_p1024": {
        "class": "strict stepwise execution-cost sweep",
        "status_note": "smallcap d0 best unconstrained spec; no dollar neutrality; includes full local cost kernel",
    },
    "aipt_stepwise_strict_smallcap_d1_p1024": {
        "class": "strict stepwise execution-cost sweep",
        "status_note": "smallcap d1 best unconstrained spec; no dollar neutrality; includes full local cost kernel",
    },
    "aipt_stepwise_strict_top1000_d0_p1024": {
        "class": "strict stepwise execution-cost sweep",
        "status_note": "top1000 d0 best VAL+TEST unconstrained spec; no dollar neutrality; includes full local cost kernel",
    },
    "aipt_stepwise_strict_top1000_d1_p1024": {
        "class": "strict stepwise execution-cost sweep",
        "status_note": "top1000 d1 best VAL+TEST unconstrained spec; no dollar neutrality; includes full local cost kernel",
    },
    "aipt_stepwise_strict_kucoin_pilot": {
        "class": "strict stepwise execution-cost pilot",
        "status_note": "KuCoin P=64/256 pilot over P,z,seed,layer,tau; no dollar neutrality",
    },
    "aipt_stepwise_strict_kucoin_p1024": {
        "class": "strict stepwise execution-cost sweep",
        "status_note": "KuCoin P=1024 high-complexity cost sweep around best unconstrained ridge values",
    },
    "aipt_stepwise_strict_kucoin_turnover_p1024": {
        "class": "strict stepwise turnover-control execution-cost sweep",
        "status_note": "KuCoin P=1024 cost sweep with blend and per-bar L1 turnover caps; no dollar neutrality",
    },
    "aipt_stepwise_strict_kucoin_qp_p1024": {
        "class": "strict project-native QP execution-cost sweep",
        "status_note": "KuCoin P=1024 using src.portfolio.qp.solve_qp; no dollar neutrality",
    },
    "aipt_stepwise_strict_smallcap_d0_qp_p1024_pilot": {
        "class": "strict project-native QP execution-cost pilot",
        "status_note": "Smallcap d0 P=1024 QP pilot using src.portfolio.qp.solve_qp; no dollar neutrality",
    },
    "aipt_stepwise_strict_kucoin_qp_turnover_p1024": {
        "class": "strict project-native QP plus turnover-control execution-cost sweep",
        "status_note": "KuCoin P=1024 using src.portfolio.qp.solve_qp plus post-QP L1 turnover caps; no dollar neutrality",
    },
    "aipt_stepwise_strict_kucoin_kernel_qp_turnover_p1024_pilot": {
        "class": "strict cost-kernel plus project-native QP execution pilot",
        "status_note": "KuCoin P=1024 cost-aware lambda fit followed by src.portfolio.qp.solve_qp and L1 turnover caps; no dollar neutrality",
    },
    "aipt_stepwise_strict_smallcap_d0_turnover_p1024": {
        "class": "strict turnover-control execution-cost sweep",
        "status_note": "Smallcap d0 P=1024 fee/kernel layers with post-target L1 turnover caps; no dollar neutrality",
    },
    "aipt_stepwise_strict_smallcap_d1_turnover_p1024": {
        "class": "strict turnover-control execution-cost sweep",
        "status_note": "Smallcap d1 P=1024 fee/kernel layers with post-target L1 turnover caps; no dollar neutrality",
    },
    "aipt_walkforward_unconstrained_equity_12m_washout7": {
        "class": "strict walk-forward selector",
        "status_note": "Equity unconstrained candidates; 12-month trailing train, 7-calendar-day washout, 1-month live, no train/live label overlap",
    },
    "aipt_walkforward_unconstrained_kucoin_3m_washout7": {
        "class": "strict walk-forward selector",
        "status_note": "KuCoin unconstrained candidates; 3-month trailing train, 7-calendar-day washout, 1-month live, no train/live label overlap",
    },
    "aipt_walkforward_unconstrained_kucoin_seedens_3m_washout7": {
        "class": "strict walk-forward seed-ensemble selector",
        "status_note": "KuCoin unconstrained seed-ensemble candidates; 3-month trailing train, 7-calendar-day washout, 1-month live",
    },
    "aipt_walkforward_unconstrained_dynamic_smallcap_p256_12m_washout7": {
        "class": "strict dynamic-universe walk-forward selector",
        "status_note": "Smallcap d0/d1 dynamic PIT daily ADV universe P=256 candidates; 12-month trailing train, 7-calendar-day washout, 1-month live",
    },
    "aipt_walkforward_unconstrained_dynamic_smallcap_p256_seedens_12m_washout7": {
        "class": "strict dynamic-universe walk-forward seed-ensemble selector",
        "status_note": "Smallcap d0/d1 dynamic PIT daily ADV universe P=256 seed-ensemble candidates; 12-month trailing train, 7-calendar-day washout, 1-month live",
    },
    "aipt_walkforward_unconstrained_top3000_dynamic_p64p256_12m_washout7": {
        "class": "strict TOP3000 no-cost walk-forward selector",
        "status_note": "TOP3000 rolling PIT ADV universe unconstrained no-cost candidates; 12-month trailing train, 7-calendar-day washout, 1-month live",
    },
    "aipt_walkforward_unconstrained_top3000_dynamic_p64p256_seedens_12m_washout7": {
        "class": "strict TOP3000 no-cost walk-forward seed-ensemble selector",
        "status_note": "TOP3000 rolling PIT ADV universe unconstrained no-cost seed-ensemble candidates; 12-month trailing train, 7-calendar-day washout, 1-month live",
    },
    "aipt_walkforward_unconstrained_top3000_dynamic_p64p256_2024live_12m_washout7": {
        "class": "strict TOP3000 recent no-cost walk-forward selector",
        "status_note": "TOP3000 unconstrained no-cost candidates; live months start in 2024, 12-month trailing train, 7-calendar-day washout",
    },
    "aipt_walkforward_unconstrained_top3000_dynamic_p64p256_seedens_2024live_12m_washout7": {
        "class": "strict TOP3000 recent no-cost walk-forward seed-ensemble selector",
        "status_note": "TOP3000 unconstrained no-cost seed-ensemble candidates; live months start in 2024, 12-month trailing train, 7-calendar-day washout",
    },
    "aipt_walkforward_cost_equity_base_12m_washout7": {
        "class": "strict costed walk-forward selector",
        "status_note": "Equity after-fee gross1_cap_fee/kernel candidates; 12-month trailing train, 7-calendar-day washout, 1-month live",
    },
    "aipt_walkforward_cost_equity_turnover_12m_washout7": {
        "class": "strict costed walk-forward selector",
        "status_note": "Equity after-fee gross1_cap_fee/kernel candidates including smallcap turnover caps; 12-month trailing train, 7-calendar-day washout, 1-month live",
    },
    "aipt_walkforward_cost_equity_all_exec_12m_washout7": {
        "class": "strict costed walk-forward selector",
        "status_note": "Equity after-fee candidates including smallcap turnover caps and smallcap d0 project QP pilot; 12-month trailing train, 7-calendar-day washout, 1-month live",
    },
    "aipt_walkforward_cost_kucoin_completed_3m_washout7": {
        "class": "strict costed walk-forward selector",
        "status_note": "KuCoin completed after-fee candidates; 3-month trailing train, 7-calendar-day washout, 1-month live",
    },
    "aipt_walkforward_cost_kucoin_seedens_3m_washout7": {
        "class": "strict costed walk-forward seed-ensemble selector",
        "status_note": "KuCoin completed after-fee seed-ensemble candidates; 3-month trailing train, 7-calendar-day washout, 1-month live",
    },
    "aipt_walkforward_cost_kucoin_qp_seedens_3m_washout7": {
        "class": "strict costed walk-forward seed-ensemble selector",
        "status_note": "KuCoin after-fee seed-ensemble candidates including project QP+turnover; 3-month trailing train, 7-calendar-day washout, 1-month live",
    },
    "aipt_walkforward_cost_kucoin_all_exec_seedens_3m_washout7": {
        "class": "strict costed walk-forward seed-ensemble selector",
        "status_note": "KuCoin after-fee seed-ensemble candidates including kernel, QP, and kernel+QP execution families; 3-month trailing train, 7-calendar-day washout, 1-month live",
    },
}


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _pid_running(pid: int) -> bool:
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
            check=False,
        )
        return str(pid) in result.stdout
    except Exception:
        return False


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _summary_path(run_dir: Path) -> Path | None:
    for name in [
        "aipt_unconstrained_summary.csv",
        "aipt_stepwise_summary.csv",
        "aipt_walk_forward_summary.csv",
        "aipt_asset_signal_unconstrained_summary.csv",
        "aipt_asset_signal_seed_ensemble_summary.csv",
        "aipt_asset_signal_surface_aggregate.csv",
        "aipt_simple_alpha_benchmark_summary.csv",
        "aipt_top3000_fixed_seed_ensemble_summary.csv",
        "aipt_top3000_factor_postprocess_summary.csv",
        "aipt_summary.csv",
    ]:
        path = run_dir / name
        if path.exists():
            return path
    return None


def _score_columns(df: pd.DataFrame) -> tuple[str, str]:
    if "SR" in df.columns:
        return "SR", "SR"
    if "SR_net" in df.columns:
        return "SR_net", "SR_net"
    if "sdf_raw_SR" in df.columns:
        return "sdf_raw_SR", "sdf_raw_SR"
    if "SR_sdf" in df.columns:
        return "SR_sdf", "SR_sdf"
    return "", ""


def _cell_keys(df: pd.DataFrame) -> list[str]:
    keys = []
    for col in [
        "scenario",
        "source_set",
        "projected",
        "projected_sources",
        "dynamic_universe",
        "P",
        "z",
        "activation",
        "seed",
        "n_seeds",
        "weight_mode",
        "train_window",
        "rebalance_every",
        "start_override",
        "alpha_name",
        "demean",
        "clip_max_w",
        "demean_features",
        "cost_tau",
        "turnover_cap",
        "blend",
        "qp_alpha_scale",
        "qp_risk_lambda",
        "layer",
        "max_weight",
        "variant",
        "ensemble_n",
        "train_months",
        "live_months",
        "washout_days",
        "selection_metric",
    ]:
        if col in df.columns:
            keys.append(col)
    return keys


SPLIT_ORDER = ["TRAIN", "VAL", "TEST", "VAL+TEST", "FULL"]


def _split_score_record(row: pd.Series, keys: list[str], group: str, selected_split: str, score_label: str) -> dict[str, Any]:
    split_scores = {split: _json_scalar(row[split]) for split in SPLIT_ORDER if split in row.index}
    train = split_scores.get("TRAIN")
    valtest = split_scores.get("VAL+TEST")
    test = split_scores.get("TEST")
    overfit_vs_valtest = None if train is None or valtest is None else float(train) - float(valtest)
    overfit_vs_test = None if train is None or test is None else float(train) - float(test)
    selected_score = split_scores.get(selected_split)
    return {
        "group": group,
        "selected_on": selected_split,
        "score_column": score_label,
        "score": selected_score,
        "split_scores": split_scores,
        "overfit_gap_train_minus_valtest": overfit_vs_valtest,
        "overfit_gap_train_minus_test": overfit_vs_test,
        "spec": {k: _json_scalar(row[k]) for k in keys if k in row.index},
    }


def _summarize_csv(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    out: dict[str, Any] = {
        "summary_csv": _rel(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
    }
    if df.empty:
        return out
    keys = _cell_keys(df)
    if "split" in df.columns and keys:
        out["completed_cells"] = int(df[keys].drop_duplicates().shape[0])
        out["splits"] = sorted(str(x) for x in df["split"].dropna().unique())
    score_col, score_label = _score_columns(df)
    if score_col and "split" in df.columns:
        tops = []
        train_tops = []
        selection_pairs = []
        preferred_split = "VAL+TEST" if "VAL+TEST" in set(df["split"].astype(str)) else "FULL"
        work = df.copy()
        work["split"] = work["split"].astype(str)
        work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
        pivot = (
            work.pivot_table(index=keys, columns="split", values=score_col, aggfunc="mean")
            .reset_index()
            .rename_axis(None, axis=1)
        )
        group_cols = [c for c in ["scenario", "layer"] if c in pivot.columns]

        def add_group_records(group_key: Any, g: pd.DataFrame) -> None:
            group = group_key if isinstance(group_key, str) else " / ".join(str(x) for x in group_key)
            if preferred_split in g.columns:
                row = g.sort_values(preferred_split, ascending=False).head(1)
                if not row.empty:
                    tops.append(_split_score_record(row.iloc[0], keys, group, preferred_split, score_label))
            if "TRAIN" in g.columns:
                row = g.sort_values("TRAIN", ascending=False).head(1)
                if not row.empty:
                    train_record = _split_score_record(row.iloc[0], keys, group, "TRAIN", score_label)
                    train_tops.append(train_record)
                    if preferred_split in g.columns:
                        val_row = g.sort_values(preferred_split, ascending=False).head(1)
                        if not val_row.empty:
                            selection_pairs.append(
                                {
                                    "group": group,
                                    "score_column": score_label,
                                    "selected_by_train": train_record,
                                    "selected_by_valtest": _split_score_record(
                                        val_row.iloc[0],
                                        keys,
                                        group,
                                        preferred_split,
                                        score_label,
                                    ),
                                }
                            )

        if group_cols:
            for group_key, g in pivot.groupby(group_cols, dropna=False):
                add_group_records(group_key, g)
        else:
            add_group_records("all", pivot)
        out["top_results"] = tops
        out["top_train_results"] = train_tops
        out["selection_overfit"] = selection_pairs
    return out


def _json_scalar(x: Any) -> Any:
    if pd.isna(x):
        return None
    if isinstance(x, (str, bool, int)):
        return x
    if isinstance(x, float):
        if math.isfinite(x):
            return x
        return None
    try:
        return x.item()
    except Exception:
        return str(x)


def _run_status(run_dir: Path) -> dict[str, Any]:
    pid_path = run_dir / "run.pid"
    if not pid_path.exists():
        return {"state": "completed_or_smoke", "pid": None, "running": False}
    try:
        pid_text = pid_path.read_text(encoding="utf-8-sig").strip().lstrip("\ufeff")
        pid = int(pid_text)
    except Exception:
        return {"state": "pid_unreadable", "pid": None, "running": False}
    running = _pid_running(pid)
    return {"state": "running" if running else "stopped_or_completed", "pid": pid, "running": running}


def _command_for(run_dir: Path, note: dict[str, str]) -> str:
    if note.get("command"):
        return note["command"]
    launch = run_dir / "launch_command.txt"
    if launch.exists():
        text = launch.read_text(encoding="utf-8-sig").strip()
        if text:
            return text
    manifest = _read_json(run_dir / "run_manifest.json")
    if isinstance(manifest, dict) and manifest.get("argv"):
        return "python " + " ".join(str(x) for x in manifest["argv"])
    return "not captured"


def collect() -> dict[str, Any]:
    runs = []
    for run_dir in sorted(p for p in RESULTS.glob("aipt*") if p.is_dir()):
        note = KNOWN_NOTES.get(run_dir.name, {})
        status = _run_status(run_dir)
        summary = _summary_path(run_dir)
        entry: dict[str, Any] = {
            "name": run_dir.name,
            "path": _rel(run_dir),
            "class": note.get("class", "unclassified AIPT output"),
            "status": status,
            "status_note": note.get("status_note", ""),
            "command": _command_for(run_dir, note),
            "last_write_time": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(),
            "files": {
                "summary": _rel(summary) if summary else None,
                "stdout": _rel(run_dir / "run.log") if (run_dir / "run.log").exists() else None,
                "stderr": _rel(run_dir / "run.err") if (run_dir / "run.err").exists() else None,
                "pid": _rel(run_dir / "run.pid") if (run_dir / "run.pid").exists() else None,
                "failures": _rel(next(iter(run_dir.glob("*failures.json")), Path())) if list(run_dir.glob("*failures.json")) else None,
            },
        }
        manifest = run_dir / "run_manifest.json"
        if manifest.exists():
            entry["run_manifest"] = _read_json(manifest)
        if summary:
            entry["summary"] = _summarize_csv(summary)
        runs.append(entry)

    return {
        "generated_at": datetime.now().isoformat(),
        "root": str(ROOT),
        "paper_reference": _rel(ROOT / "references/ssrn-4388526.pdf"),
        "extracted_paper_text": _rel(EXPERIMENTS / "ssrn_4388526_extracted.txt"),
        "audit_report": _rel(RESULTS / "aipt_no_lookahead_audit_strict.json"),
        "pit_universe_manifest": _rel(EXPERIMENTS / "data/aipt_universes/manifest.json"),
        "runs": runs,
    }


def write_markdown(registry: dict[str, Any], path: Path) -> None:
    def fmt_score(x: Any) -> str:
        return "nan" if x is None else f"{float(x):.3f}"

    def split_txt(record: dict[str, Any]) -> str:
        scores = record.get("split_scores", {})
        parts = []
        for split in SPLIT_ORDER:
            if split in scores:
                parts.append(f"{split}={fmt_score(scores[split])}")
        gap = record.get("overfit_gap_train_minus_valtest")
        if gap is not None:
            parts.append(f"TRAIN-VAL+TEST={float(gap):+.3f}")
        return ", ".join(parts)

    lines = [
        "# AIPT Experiment Registry",
        "",
        f"Generated: `{registry['generated_at']}`",
        f"Paper: `{registry['paper_reference']}`",
        f"No-lookahead audit: `{registry['audit_report']}`",
        f"PIT universe manifest: `{registry['pit_universe_manifest']}`",
        "",
        "This registry intentionally records diagnostic, stopped, superseded, and strict runs.",
        "",
    ]
    for run in registry["runs"]:
        lines.extend(
            [
                f"## {run['name']}",
                "",
                f"- Path: `{run['path']}`",
                f"- Class: {run['class']}",
                f"- Status: {run['status']['state']}" + (f" (pid {run['status']['pid']})" if run["status"]["pid"] else ""),
                f"- Note: {run['status_note'] or 'none'}",
                f"- Command: `{run['command']}`",
            ]
        )
        files = run["files"]
        for label in ["summary", "stdout", "stderr", "pid", "failures"]:
            if files.get(label):
                lines.append(f"- {label}: `{files[label]}`")
        summary = run.get("summary", {})
        if summary:
            lines.append(f"- Rows: {summary.get('rows', 0)}; completed cells: {summary.get('completed_cells', 'n/a')}")
            tops = summary.get("top_results", [])
            if tops:
                lines.append("- Top recorded results selected by VAL+TEST/FULL:")
                for top in tops[:8]:
                    spec = ", ".join(f"{k}={v}" for k, v in top["spec"].items())
                    lines.append(
                        f"  - {top['group']} {top['score_column']}: {split_txt(top)}; {spec}"
                    )
            train_tops = summary.get("top_train_results", [])
            if train_tops:
                lines.append("- Top recorded results selected by TRAIN:")
                for top in train_tops[:8]:
                    spec = ", ".join(f"{k}={v}" for k, v in top["spec"].items())
                    lines.append(
                        f"  - {top['group']} {top['score_column']}: {split_txt(top)}; {spec}"
                    )
            overfit = summary.get("selection_overfit", [])
            if overfit:
                lines.append("- Selection overfit check:")
                for pair in overfit[:8]:
                    train = pair["selected_by_train"]
                    valtest = pair["selected_by_valtest"]
                    lines.append(
                        f"  - {pair['group']}: TRAIN-selected [{split_txt(train)}] | "
                        f"VAL+TEST-selected [{split_txt(valtest)}]"
                    )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_overfit_markdown(registry: dict[str, Any], path: Path) -> None:
    def fmt_score(x: Any) -> str:
        return "nan" if x is None else f"{float(x):.3f}"

    def split_txt(record: dict[str, Any]) -> str:
        scores = record.get("split_scores", {})
        parts = []
        for split in SPLIT_ORDER:
            if split in scores:
                parts.append(f"{split}={fmt_score(scores[split])}")
        gap_vt = record.get("overfit_gap_train_minus_valtest")
        gap_test = record.get("overfit_gap_train_minus_test")
        if gap_vt is not None:
            parts.append(f"TRAIN-VAL+TEST={float(gap_vt):+.3f}")
        if gap_test is not None:
            parts.append(f"TRAIN-TEST={float(gap_test):+.3f}")
        return ", ".join(parts)

    lines = [
        "# AIPT Selection Overfit Report",
        "",
        f"Generated: `{registry['generated_at']}`",
        "",
        "For each result folder and scenario/layer, this compares the spec selected by TRAIN Sharpe to the spec selected by VAL+TEST/FULL Sharpe.",
        "",
    ]
    for run in registry["runs"]:
        summary = run.get("summary", {})
        pairs = summary.get("selection_overfit", [])
        if not pairs:
            continue
        lines.extend([f"## {run['name']}", "", f"- Class: {run['class']}", f"- Status: {run['status']['state']}", ""])
        for pair in pairs:
            train = pair["selected_by_train"]
            valtest = pair["selected_by_valtest"]
            train_spec = ", ".join(f"{k}={v}" for k, v in train["spec"].items())
            val_spec = ", ".join(f"{k}={v}" for k, v in valtest["spec"].items())
            lines.extend(
                [
                    f"### {pair['group']}",
                    "",
                    f"- TRAIN-selected: {split_txt(train)}",
                    f"- TRAIN-selected spec: `{train_spec}`",
                    f"- VAL+TEST-selected: {split_txt(valtest)}",
                    f"- VAL+TEST-selected spec: `{val_spec}`",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    registry = collect()
    json_path = EXPERIMENTS / "AIPT_EXPERIMENT_REGISTRY.json"
    md_path = EXPERIMENTS / "AIPT_EXPERIMENT_REGISTRY.md"
    overfit_path = EXPERIMENTS / "AIPT_SELECTION_OVERFIT.md"
    json_path.write_text(json.dumps(registry, indent=2, default=str), encoding="utf-8")
    write_markdown(registry, md_path)
    write_overfit_markdown(registry, overfit_path)
    print(f"wrote {_rel(json_path)}")
    print(f"wrote {_rel(md_path)}")
    print(f"wrote {_rel(overfit_path)}")


if __name__ == "__main__":
    main()
