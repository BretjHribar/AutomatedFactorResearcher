"""
Risk-model A/B/C on EQUITY: same alpha composite, three risk models in QP.

  DIAG_60   : ½λ Σ σ²_60_i w²_i              (current production: vol_window=60)
  DIAG_126  : ½λ Σ σ²_126_i w²_i             (window-control for FACTOR_K5)
  FACTOR_K5 : ½λ (||B_pca' w||² + s² w²)     (PCA from rolling 126d cov, K=5)

Thin driver over src.pipeline.runner — only `risk_model` (and vol_window for
the DIAG_126 variant) varies between cells.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pipeline.runner import run, merge_overrides

CONFIG = ROOT / "prod" / "config" / "research_equity.json"

VARIANTS = [
    ("DIAG_60",   {"name": "diagonal",
                    "params": {"vol_window": 60}}),
    ("DIAG_126",  {"name": "diagonal",
                    "params": {"vol_window": 126}}),
    ("FACTOR_K5", {"name": "pca",
                    "params": {"vol_window": 60, "factor_window": 126,
                               "n_pca_factors": 5}}),
]


def main():
    base = json.loads(CONFIG.read_text())
    print(f"=== risk-model A/B on equity (equal-weight composite, ADV cap off) ===", flush=True)

    summary = {}
    for label, rm in VARIANTS:
        cfg = merge_overrides(base, {"risk_model": rm})
        res = run(cfg, verbose=False)
        m = res.metrics
        summary[label] = m
        cost_yr = float(res.cost.mean()) * base["annualization"]["bars_per_year"] * 100
        print(f"\n=== {label} ===  TO={m['_turnover_per_bar']*100:.1f}%/d  cost={cost_yr:.2f}%/yr",
              flush=True)
        for split in ("TRAIN", "VAL", "TEST", "FULL"):
            s = m[split]
            print(f"  {split:6s}  SR_g={s['SR_gross']:+5.2f}  SR_n={s['SR_net']:+5.2f}  "
                  f"ret_g={s['ret_ann_gross']*100:+5.1f}%  ret_n={s['ret_ann_net']*100:+5.1f}%",
                  flush=True)

    print("\n" + "=" * 72)
    print("SUMMARY (net SR)")
    print("=" * 72)
    print(f"{'split':6s} | " + " | ".join(f"{lab:>10s}" for lab, _ in VARIANTS))
    print("-" * 72)
    for split in ("TRAIN", "VAL", "TEST", "FULL"):
        cells = [f"{summary[lab][split]['SR_net']:+10.3f}" for lab, _ in VARIANTS]
        print(f"{split:6s} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
