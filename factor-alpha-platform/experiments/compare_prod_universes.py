"""Compare the live production config (45 SMALLCAP_D0_v2 alphas) across three
universes:

  A. MCAP_100M_500M (current production, frozen-cohort)
  B. SMALLCAP_100M_500M_REBAL20D (new, 20-day rebalanced)
  C. MIDCAP_500M_5B_REBAL20D (new, 20-day rebalanced)

Uses src.pipeline.runner so the alpha eval, preprocessing (subindustry demean),
combiner, post-norm, optional QP, and fee subtraction are byte-identical to
the production prod/config/research_equity.json path.

Prints a side-by-side metrics table for TRAIN / VAL / TEST / VAL+TEST / FULL.
The decision lever is the VAL+TEST and TEST splits — production should change
only if a candidate dominates those splits without a Sharpe drop on TRAIN.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from copy import deepcopy

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.runner import run


BASE_CONFIG_PATH = ROOT / "prod/config/research_equity.json"

UNIVERSE_CONFIGS = [
    {
        "label": "PROD  (MCAP_100M_500M, frozen)",
        "universe_path": "data/fmp_cache/universes/MCAP_100M_500M.parquet",
    },
    {
        "label": "SMALLCAP_100M_500M_REBAL20D (new)",
        "universe_path": "experiments/data/aipt_universes/SMALLCAP_100M_500M_REBAL20D.parquet",
    },
    {
        "label": "MIDCAP_500M_5B_REBAL20D (new)",
        "universe_path": "experiments/data/aipt_universes/MIDCAP_500M_5B_REBAL20D.parquet",
    },
]


def _run_one(label: str, universe_path: str, verbose: bool = False) -> dict:
    base = json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    cfg = deepcopy(base)
    cfg["data"]["universe_path"] = universe_path
    print(f"\n{'=' * 80}\n[run] {label}\n  universe: {universe_path}\n{'=' * 80}", flush=True)
    result = run(cfg, root=ROOT, verbose=verbose)
    return {
        "label": label,
        "universe_path": universe_path,
        "alpha_signals_n": result.alpha_signals_n,
        "universe_size": result.universe_size,
        "n_bars": result.n_bars,
        "metrics": result.metrics,
    }


def _print_compare(results: list[dict]) -> None:
    splits = ["TRAIN", "VAL", "TEST", "VAL+TEST", "FULL"]
    print("\n" + "=" * 100)
    print("COMPARISON: 45 SMALLCAP_D0_v2 alphas across universes")
    print("=" * 100)
    # Header
    cols = "{:<40s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}".format(
        "Universe", *splits
    )
    print()
    for metric_key, metric_label in [
        ("SR_gross", "SR_gross"),
        ("SR_net", "SR_net"),
        ("ret_ann_net", "ret_ann_net"),
        ("max_dd_net", "max_dd_net"),
    ]:
        print(f"\n--- {metric_label} ---")
        print(cols)
        for r in results:
            row = "{:<40s}".format(r["label"][:40])
            for s in splits:
                v = r["metrics"].get(s, {}).get(metric_key, float("nan"))
                if isinstance(v, float):
                    row += " {:>+8.3f}".format(v)
                else:
                    row += " {:>8s}".format(str(v))
            print(row)
    # Auxiliary
    print(f"\n--- turnover_per_bar / n_alphas / universe_size ---")
    for r in results:
        to = r["metrics"].get("_turnover_per_bar")
        if to is None:
            # Per-split turnover lives at top level of metrics; fall back to FULL turnover
            to_str = "n/a"
        else:
            to_str = f"{to:.3%}"
        print(f"  {r['label']:<40s} turnover={to_str}  n_alphas={r['alpha_signals_n']:3d}  uni_size={r['universe_size']:5d}  n_bars={r['n_bars']:5d}")


def main() -> None:
    results = []
    for entry in UNIVERSE_CONFIGS:
        try:
            results.append(_run_one(entry["label"], entry["universe_path"], verbose=False))
        except Exception as exc:
            print(f"\n[error] {entry['label']}: {type(exc).__name__}: {exc}", flush=True)
            results.append({
                "label": entry["label"],
                "universe_path": entry["universe_path"],
                "alpha_signals_n": 0,
                "universe_size": 0,
                "n_bars": 0,
                "metrics": {s: {} for s in ["TRAIN", "VAL", "TEST", "VAL+TEST", "FULL"]},
            })
    out = ROOT / "experiments/results/aipt_audit_prod_universe_compare"
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare_summary.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    _print_compare(results)
    print(f"\nsaved: {(out / 'compare_summary.json').relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
