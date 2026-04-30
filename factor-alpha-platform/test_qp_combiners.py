"""
Combiner × risk-model sweep on EQUITY (research_equity.json).

Thin driver over src.pipeline.runner — each cell is a config override.
Reproduces the prior 14-row table (7 combiners × 2 risk models).

Sanity goal: (equal × diag) FULL net SR must reproduce ~+4.98 (prior baseline).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pipeline.runner import run, merge_overrides

CONFIG = ROOT / "prod" / "config" / "research_equity.json"

COMBINERS = [
    ("equal",       {"max_wt": 0.02}),
    ("billions",    {"max_wt": 0.02}),
    ("risk_par",    {"max_wt": 0.02}),
    ("ic_wt",       {"max_wt": 0.02}),
    ("sharpe_wt",   {"max_wt": 0.02}),
    ("topn_sharpe", {"max_wt": 0.02, "top_n": 10}),
    ("adaptive",    {"max_wt": 0.02}),
]
RISK_MODELS = ["diagonal", "style+pca"]


def main():
    base = json.loads(CONFIG.read_text())
    print(f"=== loading from {CONFIG.relative_to(ROOT)} ===", flush=True)
    results = {}
    for cname, cparams in COMBINERS:
        for rname in RISK_MODELS:
            label = f"{cname}/{rname}"
            print(f"\n>>> {label}", flush=True)
            cfg = merge_overrides(base, {
                "combiner":   {"name": cname, "params": cparams},
                "risk_model": {"name": rname,
                                "params": base["risk_model"]["params"]},
            })
            res = run(cfg, verbose=False)
            results[(cname, rname)] = res
            m = res.metrics
            cost_yr = float(res.cost.mean()) * base["annualization"]["bars_per_year"] * 100
            print(f"  TO={m['_turnover_per_bar']*100:.1f}%/d  cost={cost_yr:.2f}%/yr  "
                  f"TRAIN={m['TRAIN']['SR_net']:+.2f} VAL={m['VAL']['SR_net']:+.2f} "
                  f"TEST={m['TEST']['SR_net']:+.2f} FULL={m['FULL']['SR_net']:+.2f}",
                  flush=True)

    # Final summary table
    print("\n" + "=" * 90)
    print("SUMMARY: net SR by (combiner × risk_model)")
    print("=" * 90)
    print(f"{'combiner':10s} | {'risk':10s} | "
          f"{'TRAIN':>7s} | {'VAL':>7s} | {'TEST':>7s} | {'FULL':>7s} | "
          f"{'TO%/d':>6s} | {'cost(bps)':>9s}")
    print("-" * 90)
    for cname, _ in COMBINERS:
        for rname in RISK_MODELS:
            res = results[(cname, rname)]
            m = res.metrics
            cost_bps_per_day = float(res.cost.mean()) * 1e4
            print(f"{cname:10s} | {rname:10s} | "
                  f"{m['TRAIN']['SR_net']:+7.2f} | {m['VAL']['SR_net']:+7.2f} | "
                  f"{m['TEST']['SR_net']:+7.2f} | {m['FULL']['SR_net']:+7.2f} | "
                  f"{m['_turnover_per_bar']*100:6.1f} | {cost_bps_per_day:9.2f}")

    base_full = results[("equal", "diagonal")].metrics["FULL"]["SR_net"]
    print(f"\nSANITY: equal/diag FULL net SR = {base_full:+.3f}  (prior baseline ~4.98)")


if __name__ == "__main__":
    main()
