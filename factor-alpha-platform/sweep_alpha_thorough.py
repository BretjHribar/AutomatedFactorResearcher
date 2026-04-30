"""
Thorough alpha-count sweep:
  Mode A — sequential first-N (chronological discovery order)
  Mode B — random K subsets of N alphas (multiple seeds per N)

Strategies: ProperEqual, ProperEqualQP, Billions, BillionsQP.
Fees: production-realistic 5bps (single level for runtime).

Periodic reports printed after each N value completes.
"""
from __future__ import annotations
import sys, time, json
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

# Finer N grid for sequential sweep
N_GRID = [5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50, 60, 75, 90, 100, 125, 150, 175, 190]
# Random sampling: smaller grid (each random run × N_SEEDS × 4 strategies = ~5min per N)
N_GRID_RANDOM = [5, 10, 15, 20, 30, 50, 75, 100, 150]
N_RANDOM_SEEDS = 8                # 8 random subsets per N
FEES_BPS = 5.0                    # production fee
SEEDS_FOR_RANDOM = list(range(1000, 1000 + N_RANDOM_SEEDS))

OUT_DIR = ROOT / "data/aipt_results/billions_sweep_thorough"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_strategies(raw_subset, returns_pct, close, universe, fees_bps=FEES_BPS):
    out = {}
    for name, fn, kwargs in [
        ("ProperEqual",   ep.strategy_proper_equal,    dict()),
        ("ProperEqualQP", ep.strategy_proper_equal_qp, dict(qp_lookback=120, rebal_every=1,
                                                              track_aversion=1.0,
                                                              risk_aversion=0.0,
                                                              tc_bps=None)),
        ("Billions",      ep.strategy_billions,        dict(optim_lookback=60)),
        ("BillionsQP",    ep.strategy_billions_qp,     dict(optim_lookback=120, qp_lookback=120,
                                                              rebal_every=1,
                                                              track_aversion=1.0,
                                                              risk_aversion=0.0,
                                                              tc_bps=None)),
    ]:
        t0 = time.time()
        try:
            sim, _ = fn(raw_subset, returns_pct, close, universe,
                          max_wt=ep.MAX_WEIGHT, fees_bps=fees_bps, **kwargs)
            out[name] = {
                "sharpe":   float(sim.sharpe),
                "ann_ret":  float(sim.returns_ann),
                "turnover": float(sim.turnover),
                "max_dd":   float(sim.max_drawdown),
                "elapsed_s": time.time() - t0,
            }
        except Exception as e:
            out[name] = dict(sharpe=np.nan, ann_ret=np.nan, turnover=np.nan, max_dd=np.nan,
                             elapsed_s=time.time() - t0, error=str(e))
    return out


def main():
    print("=" * 80)
    print("THOROUGH ALPHA-COUNT SWEEP")
    print(f"  Sequential N grid: {N_GRID}  ({len(N_GRID)} values)")
    print(f"  Random N grid: {N_GRID_RANDOM}  ×  {N_RANDOM_SEEDS} seeds  ({len(N_GRID_RANDOM)*N_RANDOM_SEEDS} configs)")
    print(f"  Strategies: PE, PE+QP, Billions, BillionsQP")
    print(f"  Fees: {FEES_BPS} bps")
    print("=" * 80)

    # Load all alphas once
    print("\n[1/2] Loading 190 raw signals...")
    t0 = time.time()
    raw_all, returns_pct, close, universe = ep.load_raw_alpha_signals()
    aids = sorted(raw_all.keys())
    print(f"  Loaded {len(aids)} signals  ({time.time()-t0:.1f}s)\n")

    # ── MODE A — Sequential first-N (resumable) ────────────────────────────
    print("=" * 80)
    print("MODE A — SEQUENTIAL (first-N by alpha id)")
    print("=" * 80)
    seq_path = OUT_DIR / "sequential_metrics.csv"
    if seq_path.exists():
        existing = pd.read_csv(seq_path)
        seq_rows = existing.to_dict("records")
        done_n = set(existing["n_alphas"].unique().tolist())
        print(f"  RESUME: loaded {len(seq_rows)} prior rows, skipping N={sorted(done_n)}")
    else:
        seq_rows, done_n = [], set()
    for n in N_GRID:
        if n > len(aids) or n in done_n: continue
        subset = {aid: raw_all[aid] for aid in aids[:n]}
        results = run_strategies(subset, returns_pct, close, universe)
        for strat, m in results.items():
            seq_rows.append({"n_alphas": n, "mode": "sequential", "seed": -1,
                              "strategy": strat, **m})
        line = f"  N={n:>3}  " + "  ".join(
            f"{s[:3]}={results[s]['sharpe']:+.2f}/TO={results[s]['turnover']:.3f}"
            for s in ["ProperEqual","ProperEqualQP","Billions","BillionsQP"])
        print(line, flush=True)
        pd.DataFrame(seq_rows).to_csv(seq_path, index=False)

    # ── MODE B — Random subsets ────────────────────────────────────────────
    print()
    print("=" * 80)
    print(f"MODE B — RANDOM ({N_RANDOM_SEEDS} subsets per N)")
    print("=" * 80)
    rng_path = OUT_DIR / "random_metrics.csv"
    if rng_path.exists():
        existing = pd.read_csv(rng_path)
        rng_rows = existing.to_dict("records")
        done_pairs = set((int(r["n_alphas"]), int(r["seed"])) for _, r in existing.iterrows())
        print(f"  RESUME: loaded {len(rng_rows)} prior rows, skipping {len(done_pairs)} (N,seed) pairs")
    else:
        rng_rows, done_pairs = [], set()
    for n in N_GRID_RANDOM:
        if n > len(aids): continue
        per_n_results = {s: [] for s in ["ProperEqual","ProperEqualQP","Billions","BillionsQP"]}
        for seed in SEEDS_FOR_RANDOM:
            if (n, seed) in done_pairs:
                # Pull from existing for the report
                for strat in per_n_results:
                    sub = [r for r in rng_rows if r["n_alphas"]==n and r["seed"]==seed and r["strategy"]==strat]
                    if sub:
                        per_n_results[strat].append(float(sub[0]["sharpe"]))
                continue
            rng = np.random.default_rng(seed)
            chosen = sorted(rng.choice(aids, size=n, replace=False).tolist())
            subset = {aid: raw_all[aid] for aid in chosen}
            results = run_strategies(subset, returns_pct, close, universe)
            for strat, m in results.items():
                rng_rows.append({"n_alphas": n, "mode": "random", "seed": seed,
                                  "strategy": strat, **m})
                per_n_results[strat].append(m["sharpe"])
            pd.DataFrame(rng_rows).to_csv(rng_path, index=False)
        # Periodic per-N report (mean ± std across seeds)
        line = f"  N={n:>3}  " + "  ".join(
            f"{s[:3]}={np.nanmean(per_n_results[s]):+.2f}±{np.nanstd(per_n_results[s]):.2f}"
            for s in ["ProperEqual","ProperEqualQP","Billions","BillionsQP"])
        print(line, flush=True)

    print()
    print("=" * 80)
    print("PLOTTING")
    print("=" * 80)

    seq_df = pd.DataFrame(seq_rows)
    rng_df = pd.DataFrame(rng_rows)

    colors = {"ProperEqual": "#2ca02c", "Billions": "#1f77b4",
              "ProperEqualQP": "#9467bd", "BillionsQP": "#d62728"}

    # 4 panels: Sharpe / TO sequential vs random
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    # Sharpe — sequential (top-left)
    ax = axes[0, 0]
    for strat, sub in seq_df.groupby("strategy"):
        ax.plot(sub["n_alphas"], sub["sharpe"], "o-", label=strat,
                color=colors.get(strat,"gray"), lw=1.6, ms=4)
    ax.set_title("Sharpe — sequential (first-N)")
    ax.set_ylabel("Sharpe")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")

    # Sharpe — random (top-right): error bars
    ax = axes[0, 1]
    for strat in ["ProperEqual","ProperEqualQP","Billions","BillionsQP"]:
        sub = rng_df[rng_df.strategy == strat]
        agg = sub.groupby("n_alphas")["sharpe"].agg(["mean","std","count"]).reset_index()
        ax.errorbar(agg["n_alphas"], agg["mean"], yerr=agg["std"],
                     marker="o", capsize=3, label=strat,
                     color=colors.get(strat,"gray"), lw=1.5, ms=4)
    ax.set_title(f"Sharpe — random ({N_RANDOM_SEEDS} subsets per N, mean±std)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")

    # Turnover — sequential (bottom-left)
    ax = axes[1, 0]
    for strat, sub in seq_df.groupby("strategy"):
        ax.plot(sub["n_alphas"], sub["turnover"], "o-", label=strat,
                color=colors.get(strat,"gray"), lw=1.6, ms=4)
    ax.set_title("Turnover — sequential")
    ax.set_xlabel("# alphas")
    ax.set_ylabel("Turnover")
    ax.grid(alpha=0.3)

    # Turnover — random (bottom-right)
    ax = axes[1, 1]
    for strat in ["ProperEqual","ProperEqualQP","Billions","BillionsQP"]:
        sub = rng_df[rng_df.strategy == strat]
        agg = sub.groupby("n_alphas")["turnover"].agg(["mean","std"]).reset_index()
        ax.errorbar(agg["n_alphas"], agg["mean"], yerr=agg["std"],
                     marker="o", capsize=3, label=strat,
                     color=colors.get(strat,"gray"), lw=1.5, ms=4)
    ax.set_title("Turnover — random (mean±std)")
    ax.set_xlabel("# alphas")
    ax.grid(alpha=0.3)

    fig.suptitle(f"KuCoin 4h: thorough alpha-count sweep — VAL {ep.VAL_START} → {ep.VAL_END}, {FEES_BPS}bps fees",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sweep_thorough.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"   plot: {OUT_DIR / 'sweep_thorough.png'}")
    print(f"   sequential CSV: {OUT_DIR / 'sequential_metrics.csv'}")
    print(f"   random     CSV: {OUT_DIR / 'random_metrics.csv'}")
    print(f"\n## DONE")


if __name__ == "__main__":
    main()
