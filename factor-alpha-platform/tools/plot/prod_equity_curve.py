"""
Run the prod config (equal × diagonal on MCAP_100M_500M) and plot the
equity curve. Uses the unified pipeline runner.

Output:
  data/prod_equity_curve.png — log-scale equity curve with split markers
  Console — per-split SR + ann return summary
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.runner import run, merge_overrides

CONFIG_PATH = ROOT / "prod" / "config" / "research_equity.json"
OUT_PNG = ROOT / "data" / "prod_equity_curve.png"


def main():
    base = json.loads(CONFIG_PATH.read_text())
    splits = base["splits"]
    train_end = pd.Timestamp(splits["train_end"])
    val_end   = pd.Timestamp(splits["val_end"])

    # Prod config = equal x diagonal (per user direction). Also include
    # equal x style+pca as a sanity comparison.
    cells = [
        ("equal x diag",       {"combiner": {"name": "equal", "params": {"max_wt": 0.02}},
                                  "risk_model": {"name": "diagonal"}}),
        ("equal x style+pca",  {"combiner": {"name": "equal", "params": {"max_wt": 0.02}},
                                  "risk_model": {"name": "style+pca"}}),
    ]

    results = {}
    print(f"=== prod equity-curve runs (post fundamental-archive) ===", flush=True)
    print(f"alpha set: {base['alpha_source']['filter_sql']}", flush=True)
    print()

    for label, ov in cells:
        print(f"--- running: {label} ---", flush=True)
        cfg = merge_overrides(base, ov)
        res = run(cfg, verbose=False)
        results[label] = res
        m = res.metrics
        cost_yr = float(res.cost.mean()) * base["annualization"]["bars_per_year"] * 100
        print(f"  alphas={res.alpha_signals_n}  "
              f"TO/d={m['_turnover_per_bar']*100:.1f}%  "
              f"cost%/yr={cost_yr:.2f}", flush=True)
        for split in ("TRAIN", "VAL", "TEST", "VAL+TEST", "FULL"):
            s = m[split]
            print(f"  {split:9s}  SR_g={s['SR_gross']:+5.2f}  SR_n={s['SR_net']:+5.2f}  "
                  f"ret_n={s['ret_ann_net']*100:+6.1f}%/yr  n={s['n_bars']}", flush=True)
        print()

    # Plot equity curves (gross + net) for both
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    for label, color in zip(results, ["C0", "C3"]):
        res = results[label]
        net = res.net_pnl
        gross = res.gross_pnl
        eq_net = (1 + net).cumprod()
        eq_gross = (1 + gross).cumprod()
        m = res.metrics
        sr_n_full = m["FULL"]["SR_net"]
        sr_n_vt   = m["VAL+TEST"]["SR_net"]
        sr_n_test = m["TEST"]["SR_net"]
        ret_full  = m["FULL"]["ret_ann_net"] * 100
        axes[0].plot(eq_net.index, eq_net.values, color=color, lw=1.4,
                      label=f"{label}  net SR_FULL={sr_n_full:+.2f}  "
                            f"VAL+TEST={sr_n_vt:+.2f}  TEST={sr_n_test:+.2f}  "
                            f"ret={ret_full:+.1f}%/yr")
        axes[1].plot(eq_gross.index, eq_gross.values, color=color, lw=1.4, linestyle="--",
                      label=f"{label}  GROSS SR_FULL={m['FULL']['SR_gross']:+.2f}")

    for ax in axes:
        ax.axvline(train_end, color="grey", ls=":", lw=0.8, alpha=0.7)
        ax.axvline(val_end,   color="grey", ls=":", lw=0.8, alpha=0.7)
        ax.text(train_end, ax.get_ylim()[0], " TRAIN | VAL", fontsize=9, color="grey", va="bottom")
        ax.text(val_end,   ax.get_ylim()[0], " VAL | TEST",  fontsize=9, color="grey", va="bottom")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=10)
        ax.set_ylabel("Equity (log, start=1.0)")

    axes[0].set_title(
        f"Prod equity curve  —  MCAP_100M_500M  —  {results['equal x diag'].alpha_signals_n} alphas "
        f"(price/vol only, 36 fundamental-using alphas archived 2026-05-01)\n"
        f"Top: net of realistic IB MOC fees (book ${base['book']:,.0f})  |  "
        f"Bottom: gross PnL")
    axes[1].set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"=== saved: {OUT_PNG.relative_to(ROOT)} ===", flush=True)


if __name__ == "__main__":
    main()
