"""
run_top30_portfolio_qp.py — Portfolio combination + QP sweep against the 12
KUCOIN_TOP30 alphas in data/alphas.db (#229–#240).

Builds an in-memory config that overrides research_crypto.json to point at
the new universe + new alpha source, then sweeps combiners × QP variants and
prints TRAIN/VAL/TEST metrics for each cell.

Usage: python experiments/run_top30_portfolio_qp.py
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from src.pipeline.runner import run, merge_overrides

BASE_CFG = ROOT / "prod" / "config" / "research_crypto.json"


# Override: TOP30 universe + alphas.db source. Keep splits/fees/annualization.
OVERRIDE_BASE = {
    "data": {
        "matrices_dir": "data/kucoin_cache/matrices/4h",
        "universe_path": "data/kucoin_cache/universes/KUCOIN_TOP30_4h.parquet",
        "universe_filter": {"method": "coverage", "threshold": 0.0},
        "returns_source": "matrix:returns",
    },
    "alpha_source": {
        "db_path": "data/alphas.db",
        "table": "alphas",
        "filter_sql": "archived=0 AND universe='KUCOIN_TOP30'",
        "train_sharpe_table": "evaluations",
        "train_sharpe_column": "sharpe_train",
    },
}


# Cell sweep: (label, override).  All 12 alphas → top_n=12 means "use them all".
TOP_N = 12
QP_KW = {
    "max_w": 0.10, "lambda_risk": 5.0, "kappa_tc": 30.0, "impact_bps": 3.5,
    "commission_per_share": 0.0, "dollar_neutral": True, "max_gross_leverage": 1.0,
}

CELLS = [
    # ---- raw combiners (no QP) ----
    ("equal_noQP",       {"combiner": {"name": "equal", "params": {}},
                          "qp": {"enabled": False}}),
    ("billions_noQP",    {"combiner": {"name": "billions",
                                       "params": {"optim_lookback": 10}},
                          "qp": {"enabled": False}}),
    ("topn_train_noQP",  {"combiner": {"name": "topn_train", "params": {"top_n": TOP_N}},
                          "qp": {"enabled": False}}),
    ("ic_wt_noQP",       {"combiner": {"name": "ic_wt", "params": {"lookback": 60}},
                          "qp": {"enabled": False}}),
    ("sharpe_wt_noQP",   {"combiner": {"name": "sharpe_wt", "params": {"lookback": 240}},
                          "qp": {"enabled": False}}),
    ("risk_par_noQP",    {"combiner": {"name": "risk_par", "params": {"lookback": 60}},
                          "qp": {"enabled": False}}),

    # ---- QP-optimized variants on top of equal-weight + billions + topn ----
    ("equal_QPdiag",     {"combiner": {"name": "equal", "params": {}},
                          "risk_model": {"name": "diagonal"},
                          "qp": {"enabled": True, **QP_KW}}),
    ("billions_QPdiag",  {"combiner": {"name": "billions",
                                       "params": {"optim_lookback": 10}},
                          "risk_model": {"name": "diagonal"},
                          "qp": {"enabled": True, **QP_KW}}),
    ("topn_train_QPdiag",{"combiner": {"name": "topn_train", "params": {"top_n": TOP_N}},
                          "risk_model": {"name": "diagonal"},
                          "qp": {"enabled": True, **QP_KW}}),

    ("equal_QPpca",      {"combiner": {"name": "equal", "params": {}},
                          "risk_model": {"name": "pca",
                                         "params": {"vol_window": 120,
                                                    "factor_window": 360,
                                                    "n_pca_factors": 5}},
                          "qp": {"enabled": True, **QP_KW}}),
    ("billions_QPpca",   {"combiner": {"name": "billions",
                                       "params": {"optim_lookback": 10}},
                          "risk_model": {"name": "pca",
                                         "params": {"vol_window": 120,
                                                    "factor_window": 360,
                                                    "n_pca_factors": 5}},
                          "qp": {"enabled": True, **QP_KW}}),
    ("topn_train_QPpca", {"combiner": {"name": "topn_train", "params": {"top_n": TOP_N}},
                          "risk_model": {"name": "pca",
                                         "params": {"vol_window": 120,
                                                    "factor_window": 360,
                                                    "n_pca_factors": 5}},
                          "qp": {"enabled": True, **QP_KW}}),

    # ---- Bianchi-Babiak IPCA risk model — sweep K=2,3,5, window 240/360/540 ----
    ("equal_QPipca_K3_w360",     {"combiner": {"name": "equal", "params": {}},
                                  "risk_model": {"name": "ipca",
                                                 "params": {"vol_window": 120,
                                                            "factor_window": 360,
                                                            "n_factors": 3}},
                                  "qp": {"enabled": True, **QP_KW}}),
    ("equal_QPipca_K2_w360",     {"combiner": {"name": "equal", "params": {}},
                                  "risk_model": {"name": "ipca",
                                                 "params": {"vol_window": 120,
                                                            "factor_window": 360,
                                                            "n_factors": 2}},
                                  "qp": {"enabled": True, **QP_KW}}),
    ("equal_QPipca_K5_w360",     {"combiner": {"name": "equal", "params": {}},
                                  "risk_model": {"name": "ipca",
                                                 "params": {"vol_window": 120,
                                                            "factor_window": 360,
                                                            "n_factors": 5}},
                                  "qp": {"enabled": True, **QP_KW}}),
    ("equal_QPipca_K3_w240",     {"combiner": {"name": "equal", "params": {}},
                                  "risk_model": {"name": "ipca",
                                                 "params": {"vol_window": 120,
                                                            "factor_window": 240,
                                                            "n_factors": 3}},
                                  "qp": {"enabled": True, **QP_KW}}),
    ("equal_QPipca_K3_w540",     {"combiner": {"name": "equal", "params": {}},
                                  "risk_model": {"name": "ipca",
                                                 "params": {"vol_window": 120,
                                                            "factor_window": 540,
                                                            "n_factors": 3}},
                                  "qp": {"enabled": True, **QP_KW}}),
    ("topn_train_QPipca_K3_w360",{"combiner": {"name": "topn_train", "params": {"top_n": TOP_N}},
                                  "risk_model": {"name": "ipca",
                                                 "params": {"vol_window": 120,
                                                            "factor_window": 360,
                                                            "n_factors": 3}},
                                  "qp": {"enabled": True, **QP_KW}}),

    # ---- Billions lookback sweep — short to catch factor momentum ----
    # Asness-Frazzini-Moskowitz-Ooi (2021) "Factor Momentum Everywhere":
    # factor returns are positively auto-correlated; need short lookback to
    # track regime. At 4h crypto cadence this means single-digit bars to ~30.
    # Original Kakushadze paper used 10 bars at daily equity = ~2 weeks.
    ("billions_ol3",        {"combiner": {"name": "billions", "params": {"optim_lookback": 3}},
                              "qp": {"enabled": False}}),
    ("billions_ol5",        {"combiner": {"name": "billions", "params": {"optim_lookback": 5}},
                              "qp": {"enabled": False}}),
    ("billions_ol10",       {"combiner": {"name": "billions", "params": {"optim_lookback": 10}},
                              "qp": {"enabled": False}}),
    ("billions_ol15",       {"combiner": {"name": "billions", "params": {"optim_lookback": 15}},
                              "qp": {"enabled": False}}),
    ("billions_ol20",       {"combiner": {"name": "billions", "params": {"optim_lookback": 20}},
                              "qp": {"enabled": False}}),
    ("billions_ol30",       {"combiner": {"name": "billions", "params": {"optim_lookback": 30}},
                              "qp": {"enabled": False}}),
    ("billions_ol60",       {"combiner": {"name": "billions", "params": {"optim_lookback": 10}},
                              "qp": {"enabled": False}}),
    # Best-of-billions × QP risk models (use the winner of the lookback sweep)
    ("billions_ol10_QPdiag",{"combiner": {"name": "billions", "params": {"optim_lookback": 10}},
                              "risk_model": {"name": "diagonal"},
                              "qp": {"enabled": True, **QP_KW}}),
    ("billions_ol10_QPipca",{"combiner": {"name": "billions", "params": {"optim_lookback": 10}},
                              "risk_model": {"name": "ipca",
                                             "params": {"vol_window": 120,
                                                        "factor_window": 360,
                                                        "n_factors": 3}},
                              "qp": {"enabled": True, **QP_KW}}),
    ("billions_ol30_QPdiag",{"combiner": {"name": "billions", "params": {"optim_lookback": 30}},
                              "risk_model": {"name": "diagonal"},
                              "qp": {"enabled": True, **QP_KW}}),
    ("billions_ol30_QPipca",{"combiner": {"name": "billions", "params": {"optim_lookback": 30}},
                              "risk_model": {"name": "ipca",
                                             "params": {"vol_window": 120,
                                                        "factor_window": 360,
                                                        "n_factors": 3}},
                              "qp": {"enabled": True, **QP_KW}}),
]


def main():
    base = json.loads(BASE_CFG.read_text())
    base = merge_overrides(base, OVERRIDE_BASE)

    print("=" * 130, flush=True)
    print(f"KUCOIN_TOP30 portfolio sweep — 12 alphas (#229–#240)", flush=True)
    print(f"  universe: {base['data']['universe_path']}", flush=True)
    print(f"  alphas:   {base['alpha_source']['db_path']} :: "
          f"{base['alpha_source']['table']} :: {base['alpha_source']['filter_sql']}", flush=True)
    print(f"  splits:   train_end={base['splits']['train_end']}  "
          f"val_end={base['splits']['val_end']}", flush=True)
    print(f"  fees:     {base['fees']}", flush=True)
    print("=" * 130, flush=True)
    header = (f"{'cell':22s} | "
              f"{'TRAIN_g':>8s} {'VAL_g':>7s} {'TEST_g':>7s} | "
              f"{'TRAIN_n':>8s} {'VAL_n':>7s} {'TEST_n':>7s} | "
              f"{'V+T_n':>6s} {'FULL_n':>7s} | "
              f"{'TO/bar':>7s} {'cost%/yr':>9s} {'sec':>5s}")
    print(header, flush=True)
    print("-" * 130, flush=True)

    results = {}
    for label, ov in CELLS:
        cfg = merge_overrides(base, ov)
        if "combiner" in ov:
            cfg["combiner"] = ov["combiner"]
        if "risk_model" in ov:
            cfg["risk_model"] = ov["risk_model"]
        if "qp" in ov:
            cfg["qp"] = {**base.get("qp", {}), **ov["qp"]}
        t0 = time.time()
        try:
            res = run(cfg, verbose=False)
        except Exception as e:
            print(f"{label:22s} | ERROR: {type(e).__name__}: {e}", flush=True)
            continue
        m = res.metrics
        cost_yr = float(res.cost.mean()) * base["annualization"]["bars_per_year"] * 100
        elapsed = time.time() - t0
        results[label] = {"metrics": m, "to": float(m["_turnover_per_bar"]),
                          "cost_yr": cost_yr, "elapsed": elapsed}
        print(f"{label:22s} | "
              f"{m['TRAIN']['SR_gross']:>+7.2f} "
              f"{m['VAL']['SR_gross']:>+6.2f} "
              f"{m['TEST']['SR_gross']:>+6.2f} | "
              f"{m['TRAIN']['SR_net']:>+7.2f} "
              f"{m['VAL']['SR_net']:>+6.2f} "
              f"{m['TEST']['SR_net']:>+6.2f} | "
              f"{m['VAL+TEST']['SR_net']:>+5.2f} "
              f"{m['FULL']['SR_net']:>+6.2f} | "
              f"{m['_turnover_per_bar']*100:>5.2f}% {cost_yr:>7.2f}% {elapsed:>4.0f}s",
              flush=True)

    # Save results JSON for post-hoc inspection
    out = {
        "config": {"universe": "KUCOIN_TOP30", "n_alphas": TOP_N,
                   "base_cfg": str(BASE_CFG.relative_to(ROOT)),
                   "splits": base["splits"], "fees": base["fees"]},
        "cells": {label: {"metrics": r["metrics"], "to_per_bar": r["to"],
                          "cost_pct_per_yr": r["cost_yr"], "elapsed_sec": r["elapsed"]}
                  for label, r in results.items()},
    }
    out_path = ROOT / "experiments" / "results" / "top30_portfolio_qp.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[saved] {out_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
