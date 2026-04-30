"""
Sweep #alphas × fee_bps for ProperEqual / Billions / BillionsQP, plot SR/Ret/TO/DD.

Loads raw signals once; for each (N, fees) runs all 3 strategies on first N alphas.
Saves metrics CSV + grid PNG.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import eval_portfolio as ep

N_GRID    = [5, 10, 15, 18, 25, 35, 50, 75, 100, 130, 160, 190]
FEE_GRID  = [0.0, 3.0, 5.0]

OUT_DIR = ROOT / "data/aipt_results/billions_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_strategies(raw_signals_subset, returns_pct, close, universe, fees_bps):
    out = {}
    for name, fn, kwargs in [
        ("ProperEqual",   ep.strategy_proper_equal,    dict()),
        ("Billions",      ep.strategy_billions,        dict(optim_lookback=60)),
        # tc_bps=None → QP uses outer fees_bps (correct net-utility framing)
        ("ProperEqualQP", ep.strategy_proper_equal_qp, dict(qp_lookback=120, rebal_every=1,
                                                              track_aversion=1.0,
                                                              risk_aversion=0.0,
                                                              tc_bps=None)),
        ("BillionsQP",    ep.strategy_billions_qp,     dict(optim_lookback=120, qp_lookback=120,
                                                              rebal_every=1,
                                                              track_aversion=1.0,
                                                              risk_aversion=0.0,
                                                              tc_bps=None)),
    ]:
        t0 = time.time()
        try:
            sim, label = fn(raw_signals_subset, returns_pct, close, universe,
                              max_wt=ep.MAX_WEIGHT, fees_bps=fees_bps, **kwargs)
            out[name] = {
                "sharpe":   float(sim.sharpe),
                "fitness":  float(sim.fitness)        if hasattr(sim, "fitness")        else float("nan"),
                "ann_ret":  float(sim.returns_ann)    if hasattr(sim, "returns_ann")    else float("nan"),
                "turnover": float(sim.turnover),
                "max_dd":   float(sim.max_drawdown)   if hasattr(sim, "max_drawdown")   else float("nan"),
                "elapsed_s": time.time() - t0,
                "label":    label,
            }
            print(f"      {name:12s} SR={out[name]['sharpe']:+.3f}  ann={out[name]['ann_ret']*100:+5.1f}%  "
                  f"TO={out[name]['turnover']:.3f}  DD={out[name]['max_dd']*100:+5.1f}%  "
                  f"({out[name]['elapsed_s']:.1f}s)", flush=True)
        except Exception as e:
            print(f"      {name:12s} FAILED: {type(e).__name__}: {e}", flush=True)
            out[name] = dict(sharpe=np.nan, fitness=np.nan, ann_ret=np.nan,
                             turnover=np.nan, max_dd=np.nan,
                             elapsed_s=time.time() - t0, label=str(e))
    return out


def main():
    print("=" * 80)
    print("SWEEP: alpha count × fees vs Sharpe — ProperEqual / Billions / BillionsQP")
    print(f"  N grid:   {N_GRID}")
    print(f"  Fee grid: {FEE_GRID} bps")
    print("=" * 80)

    print("\n[1/2] Loading all 190 raw signals once...", flush=True)
    t0 = time.time()
    raw_signals_all, returns_pct, close, universe = ep.load_raw_alpha_signals()
    if raw_signals_all is None:
        print("  no signals loaded"); return
    aid_sorted = sorted(raw_signals_all.keys())
    print(f"  loaded {len(aid_sorted)} signals  ({time.time()-t0:.1f}s)", flush=True)
    print(f"  alpha id range: {min(aid_sorted)} → {max(aid_sorted)}", flush=True)

    rows = []
    for n in N_GRID:
        if n > len(aid_sorted):
            continue
        subset_ids = aid_sorted[:n]
        raw_subset = {aid: raw_signals_all[aid] for aid in subset_ids}
        for fees in FEE_GRID:
            print(f"\n  ── N={n}  fees={fees:.0f}bps ──", flush=True)
            results = run_strategies(raw_subset, returns_pct, close, universe, fees)
            for strat, m in results.items():
                rows.append({"n_alphas": n, "fees_bps": fees, "strategy": strat, **m})
            pd.DataFrame(rows).to_csv(OUT_DIR / "sweep_metrics.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "sweep_metrics.csv", index=False)
    print(f"\n## SWEEP DONE — metrics CSV: {OUT_DIR / 'sweep_metrics.csv'}")

    # ── Plot: 4 metric rows × 3 fee cols ────────────────────────────────────
    metrics = [
        ("sharpe",   "Sharpe Ratio",  False),
        ("ann_ret",  "Annual Return", True),    # display as %
        ("turnover", "Avg Turnover",  False),
        ("max_dd",   "Max Drawdown",  True),    # display as %
    ]
    colors = {"ProperEqual": "#2ca02c", "Billions": "#1f77b4",
              "ProperEqualQP": "#9467bd", "BillionsQP": "#d62728"}

    fig, axes = plt.subplots(len(metrics), len(FEE_GRID), figsize=(15, 4*len(metrics)),
                              sharex=True, sharey="row")
    for r, (col, label, as_pct) in enumerate(metrics):
        for c, fees in enumerate(FEE_GRID):
            ax = axes[r, c]
            for strat, sub in df[df.fees_bps == fees].groupby("strategy"):
                y = sub[col] * 100 if as_pct else sub[col]
                ax.plot(sub["n_alphas"], y, "o-", label=strat,
                        color=colors.get(strat, "gray"), lw=1.6, ms=5)
            if r == 0:
                ax.set_title(f"fees = {fees:.0f} bps")
            if c == 0:
                ax.set_ylabel(label + (" (%)" if as_pct else ""))
            if r == len(metrics) - 1:
                ax.set_xlabel("# alphas (first N by id)")
            ax.grid(alpha=0.3)
            if col == "sharpe":
                ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
            if r == 0 and c == len(FEE_GRID) - 1:
                ax.legend(fontsize=9, loc="best")
    fig.suptitle(f"KuCoin 4h: alpha-count sweep — VAL {ep.VAL_START} → {ep.VAL_END}",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sweep_alpha_count.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"   plot: {OUT_DIR / 'sweep_alpha_count.png'}")


if __name__ == "__main__":
    main()
