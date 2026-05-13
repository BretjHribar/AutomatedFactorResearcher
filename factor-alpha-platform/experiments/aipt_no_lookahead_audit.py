"""No-lookahead audit checks for the AIPT experiment scripts.

This is intentionally small and mechanical. It verifies the timing convention
used by the replication:

  delay=0: S(Z[t]) earns close[t+1] / close[t] - 1
  delay=1: S(Z[t]) earns open[t+2] / open[t+1] - 1

and checks that rolling fits at t only use factor rows strictly known by t.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.aipt_replication import EQUITY_MATRICES_DIR, ROOT, SCENARIOS, make_forward_returns


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not np.isfinite(actual) or abs(actual - expected) > 1e-12:
        raise AssertionError(f"{label}: got {actual}, expected {expected}")


def audit_forward_returns() -> dict[str, float]:
    idx = pd.date_range("2026-01-01", periods=5, freq="D")
    close = pd.DataFrame({"AAA": [100.0, 110.0, 121.0, 133.1, 146.41]}, index=idx)
    open_ = pd.DataFrame({"AAA": [50.0, 55.0, 66.0, 72.6, 87.12]}, index=idx)
    d0 = make_forward_returns({"close": close}, delay=0)
    d1 = make_forward_returns({"close": close, "open": open_}, delay=1)
    _assert_close(float(d0.iloc[0, 0]), 0.10, "delay0 row0 close[1]/close[0]-1")
    _assert_close(float(d0.iloc[1, 0]), 0.10, "delay0 row1 close[2]/close[1]-1")
    _assert_close(float(d1.iloc[0, 0]), 0.20, "delay1 row0 open[2]/open[1]-1")
    _assert_close(float(d1.iloc[1, 0]), 0.10, "delay1 row1 open[3]/open[2]-1")
    return {
        "delay0_row0": float(d0.iloc[0, 0]),
        "delay1_row0": float(d1.iloc[0, 0]),
    }


def audit_training_windows() -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for delay in [0, 1]:
        for t in [252, 253, 400, 1000]:
            train_window = 252
            train_end = t - delay
            train_start = train_end - train_window
            if train_start < 0 or train_end <= train_start:
                continue
            max_used = train_end - 1
            max_known = t - delay - 1
            if max_used > max_known:
                raise AssertionError(f"delay={delay} t={t}: used {max_used}, known {max_known}")
            rows.append(
                {
                    "delay": delay,
                    "t": t,
                    "train_start": train_start,
                    "train_end_exclusive": train_end,
                    "max_factor_row_used": max_used,
                    "max_known_factor_row": max_known,
                }
            )
    return rows


def audit_equity_pit_manifest() -> dict[str, object]:
    matrices = ROOT / EQUITY_MATRICES_DIR
    if "matrices_pit" not in EQUITY_MATRICES_DIR.replace("\\", "/"):
        raise AssertionError(f"equity scenarios are not using PIT matrices: {EQUITY_MATRICES_DIR}")
    manifest_path = matrices / "manifest.json"
    if not manifest_path.exists():
        raise AssertionError(f"PIT manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lag = int(manifest.get("pit_lag_days", -1))
    if lag < 1:
        raise AssertionError(f"PIT lag must be >= 1 trading day, got {lag}")
    return {
        "matrices_dir": EQUITY_MATRICES_DIR,
        "pit_lag_days": lag,
        "date_range": manifest.get("date_range"),
        "delisted_included": bool(manifest.get("delisted_included", False)),
    }


def audit_pit_universe_manifest() -> dict[str, object]:
    manifest_path = ROOT / "experiments/data/aipt_universes/manifest.json"
    if not manifest_path.exists():
        raise AssertionError(f"experiment PIT universe manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs", {})
    required = ["MCAP_100M_500M_PITV2.parquet", "TOP1000_ADV60_PITV2.parquet", "TOP3000_ADV60_PITV2.parquet"]
    missing = [name for name in required if name not in outputs]
    if missing:
        raise AssertionError(f"missing PIT universe outputs: {missing}")
    for scenario_name in ["equity_smallcap_d0", "equity_smallcap_d1"]:
        if "MCAP_100M_500M_PITV2" not in SCENARIOS[scenario_name].universe_path:
            raise AssertionError(f"{scenario_name} is not using the experiment PIT smallcap universe")
    for scenario_name in ["equity_top1000_d0", "equity_top1000_d1"]:
        if "TOP1000_ADV60_PITV2" not in SCENARIOS[scenario_name].universe_path:
            raise AssertionError(f"{scenario_name} is not using the experiment PIT top1000 universe")
    for scenario_name in ["equity_top3000_d0", "equity_top3000_d1"]:
        if "TOP3000_ADV60_PITV2" not in SCENARIOS[scenario_name].universe_path:
            raise AssertionError(f"{scenario_name} is not using the experiment PIT top3000 universe")
    return {
        "source_membership": manifest.get("source_membership"),
        "construction": manifest.get("construction"),
        "outputs": outputs,
    }


def audit_no_backfill_in_experiment_code() -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    checked = []
    for path in sorted((ROOT / "experiments").glob("aipt_*.py")):
        if path.name == "aipt_no_lookahead_audit.py":
            continue
        checked.append(str(path.relative_to(ROOT)).replace("\\", "/"))
        text = path.read_text(encoding="utf-8")
        bad = []
        for needle in [".bfill(", "method=\"bfill\"", "method='bfill'"]:
            if needle in text:
                bad.append(needle)
        if bad:
            hits[str(path.relative_to(ROOT))] = bad
    if hits:
        raise AssertionError(f"backfill-like operations found: {hits}")
    return {"checked": checked}


def audit_walkforward_outputs() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for run_dir in sorted((ROOT / "experiments/results").glob("aipt_walkforward*")):
        folds_path = run_dir / "aipt_walk_forward_folds.csv"
        summary_path = run_dir / "aipt_walk_forward_summary.csv"
        if not folds_path.exists() or not summary_path.exists():
            continue
        folds = pd.read_csv(folds_path)
        summary = pd.read_csv(summary_path)
        if folds.empty:
            continue
        no_signal = folds["audit_no_signal_overlap"].astype(bool)
        no_label = folds["audit_no_label_overlap"].astype(bool)
        if not bool(no_signal.all()) or not bool(no_label.all()):
            bad = folds.loc[~(no_signal & no_label), ["scenario", "fold", "train_end_exclusive", "live_start"]]
            raise AssertionError(f"{run_dir.name}: walk-forward overlap found: {bad.to_dict('records')[:5]}")
        min_gap = float(pd.to_numeric(folds["audit_label_to_live_gap_days"], errors="coerce").min())
        if min_gap <= 0:
            raise AssertionError(f"{run_dir.name}: non-positive label/live gap {min_gap}")
        if "audit_all_no_label_overlap" in summary.columns:
            ok = summary["audit_all_no_label_overlap"].astype(bool)
            if not bool(ok.all()):
                raise AssertionError(f"{run_dir.name}: summary reports label overlap")
        out.append(
            {
                "run": str(run_dir.relative_to(ROOT)).replace("\\", "/"),
                "folds": int(len(folds)),
                "min_label_to_live_gap_days": min_gap,
                "all_no_signal_overlap": bool(no_signal.all()),
                "all_no_label_overlap": bool(no_label.all()),
            }
        )
    return out


def audit_scenarios() -> list[dict[str, object]]:
    out = []
    for name in [
        "equity_smallcap_d0",
        "equity_smallcap_d1",
        "equity_top1000_d0",
        "equity_top1000_d1",
        "equity_top3000_d0",
        "equity_top3000_d1",
        "kucoin_top100",
    ]:
        s = SCENARIOS[name]
        out.append(
            {
                "scenario": name,
                "matrices_dir": s.matrices_dir,
                "universe_path": s.universe_path,
                "delay": s.delay,
                "train_window": s.train_window,
                "rebalance_every": s.rebalance_every,
            }
        )
    return out


def main() -> None:
    report = {
        "forward_returns": audit_forward_returns(),
        "training_windows": audit_training_windows(),
        "equity_pit_manifest": audit_equity_pit_manifest(),
        "pit_universe_manifest": audit_pit_universe_manifest(),
        "no_backfill": audit_no_backfill_in_experiment_code(),
        "walkforward_outputs": audit_walkforward_outputs(),
        "scenarios": audit_scenarios(),
    }
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
