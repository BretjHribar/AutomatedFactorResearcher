"""
Crypto sweep on KuCoin 4h: top-N combiner × {no QP, QP diag, QP pca}.

Thin driver over src.pipeline.runner — every cell is a config override on
prod/config/research_crypto.json.

The baseline cell (top_n=50, qp.enabled=false) is the canonical reproduction
target: TRAIN gross SR ≈ +5.30 (we hit +5.348 today).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pipeline.runner import run, merge_overrides

CONFIG = ROOT / "prod" / "config" / "research_crypto.json"

# Cells to run: (label, override-dict)
CELLS = [
    ("top50_noQP",      {"combiner": {"name": "topn_train", "params": {"top_n": 50}},
                          "qp": {"enabled": False}}),
    ("top30_noQP",      {"combiner": {"name": "topn_train", "params": {"top_n": 30}},
                          "qp": {"enabled": False}}),
    ("top50_QP_diag",   {"combiner": {"name": "topn_train", "params": {"top_n": 50}},
                          "risk_model": {"name": "diagonal"},
                          "qp": {"enabled": True, "max_w": 0.10, "lambda_risk": 5.0,
                                  "kappa_tc": 30.0, "impact_bps": 3.5,
                                  "commission_per_share": 0.0,
                                  "dollar_neutral": True,
                                  "max_gross_leverage": 1.0}}),
    ("top50_QP_pca",    {"combiner": {"name": "topn_train", "params": {"top_n": 50}},
                          "risk_model": {"name": "pca",
                                          "params": {"vol_window": 120,
                                                     "factor_window": 360,
                                                     "n_pca_factors": 5}},
                          "qp": {"enabled": True, "max_w": 0.10, "lambda_risk": 5.0,
                                  "kappa_tc": 30.0, "impact_bps": 3.5,
                                  "commission_per_share": 0.0,
                                  "dollar_neutral": True,
                                  "max_gross_leverage": 1.0}}),
]


def main():
    base = json.loads(CONFIG.read_text())
    print(f"=== crypto sweep ({CONFIG.name}) ===", flush=True)
    print(f"{'cell':16s} | "
          f"{'TRAIN_g':>7s} {'VAL_g':>6s} {'TEST_g':>7s} | "
          f"{'TRAIN_n':>7s} {'VAL_n':>6s} {'TEST_n':>7s} | "
          f"{'V+T_n':>6s} {'FULL_n':>7s} | {'TO/bar':>7s} {'cost/yr':>8s}",
          flush=True)
    print("-" * 130, flush=True)

    results = {}
    for label, ov in CELLS:
        cfg = merge_overrides(base, ov)
        res = run(cfg, verbose=False)
        m = res.metrics
        results[label] = m
        cost_yr = float(res.cost.mean()) * base["annualization"]["bars_per_year"] * 100
        print(f"{label:16s} | "
              f"{m['TRAIN']['SR_gross']:>+6.2f} "
              f"{m['VAL']['SR_gross']:>+5.2f} "
              f"{m['TEST']['SR_gross']:>+6.2f} | "
              f"{m['TRAIN']['SR_net']:>+6.2f} "
              f"{m['VAL']['SR_net']:>+5.2f} "
              f"{m['TEST']['SR_net']:>+6.2f} | "
              f"{m['VAL+TEST']['SR_net']:>+5.2f} "
              f"{m['FULL']['SR_net']:>+6.2f} | "
              f"{m['_turnover_per_bar']*100:>5.2f}% {cost_yr:>6.2f}%",
              flush=True)

    base_train = results["top50_noQP"]["TRAIN"]["SR_gross"]
    print(f"\nSANITY: top50_noQP TRAIN gross SR = {base_train:+.2f}  (target ~+5.30)")


if __name__ == "__main__":
    main()
