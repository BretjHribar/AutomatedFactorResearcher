"""
sweep_neutralization.py — Compare alpha performance across neutralization levels.

Tests the top seed alphas with different neutralization:
    - sector
    - industry
    - subindustry
    - market (cross-sectional demean only)

Usage:
    python sweep_neutralization.py
"""

import sys, os, time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from eval_alpha_ib but we'll override NEUTRALIZE
import eval_alpha_ib

# Top alphas to test (from seed run results)
TEST_ALPHAS = [
    ("close_near_low",      "rank((low - close) / (high - low + 0.001))"),
    ("upper_wick_ratio",    "rank((high - close) / (high - low + 0.001))"),
    ("vwap_deviation",      "rank(-(close - vwap) / close)"),
    ("reversal_1d",         "rank(-ts_delta(close, 1))"),
    ("reversal_3d",         "rank(-ts_delta(close, 3))"),
    ("low_vol_premium",     "rank(-ts_std_dev(close, 10))"),
    ("daily_range_pct",     "rank((high - low) / close)"),
    ("bollinger_z_5d",      "rank((close - ts_mean(close, 5)) / (ts_std_dev(close, 5) + 0.001))"),
]

NEUTRALIZATIONS = ["sector", "industry", "subindustry", "market"]


def run_sweep():
    print(f"\n{'='*100}")
    print(f"NEUTRALIZATION SWEEP — TOP2000TOP3000, delay=0, fee-free")
    print(f"{'='*100}")

    results = []

    for neut in NEUTRALIZATIONS:
        print(f"\n--- Neutralization: {neut.upper()} ---")
        # Override the global
        eval_alpha_ib.NEUTRALIZE = neut
        # Clear data cache so it reloads with new neutralization
        eval_alpha_ib._DATA_CACHE.clear()

        for name, expr in TEST_ALPHAS:
            t0 = time.time()
            r = eval_alpha_ib.eval_single(expr, split="train")
            elapsed = time.time() - t0

            if r["success"]:
                results.append({
                    "alpha": name,
                    "neutralization": neut,
                    "sharpe": r["sharpe"],
                    "fitness": r["fitness"],
                    "turnover": r["turnover"],
                    "returns_ann": r["returns_ann"],
                    "max_drawdown": r["max_drawdown"],
                    "elapsed": elapsed,
                })
                print(f"  {name:25s} | SR={r['sharpe']:+.3f} Fit={r['fitness']:.3f} "
                      f"TO={r['turnover']:.4f} DD={r['max_drawdown']:.3f} | {elapsed:.1f}s")
            else:
                print(f"  {name:25s} | FAILED: {r['error'][:60]}")
                results.append({
                    "alpha": name,
                    "neutralization": neut,
                    "sharpe": np.nan,
                    "fitness": np.nan,
                    "turnover": np.nan,
                    "returns_ann": np.nan,
                    "max_drawdown": np.nan,
                    "elapsed": elapsed,
                })

    # Reset to default
    eval_alpha_ib.NEUTRALIZE = "sector"
    eval_alpha_ib._DATA_CACHE.clear()

    # Print comparison table
    df = pd.DataFrame(results)
    print(f"\n\n{'='*100}")
    print("NEUTRALIZATION COMPARISON TABLE")
    print(f"{'='*100}")

    pivot = df.pivot_table(index="alpha", columns="neutralization", values="sharpe")
    pivot = pivot[NEUTRALIZATIONS]  # order columns
    print("\nSharpe Ratios:")
    print(pivot.to_string(float_format="{:+.3f}".format))

    pivot_fit = df.pivot_table(index="alpha", columns="neutralization", values="fitness")
    pivot_fit = pivot_fit[NEUTRALIZATIONS]
    print("\nFitness:")
    print(pivot_fit.to_string(float_format="{:.3f}".format))

    pivot_to = df.pivot_table(index="alpha", columns="neutralization", values="turnover")
    pivot_to = pivot_to[NEUTRALIZATIONS]
    print("\nTurnover:")
    print(pivot_to.to_string(float_format="{:.4f}".format))

    # Best neutralization per alpha
    print(f"\n{'='*100}")
    print("BEST NEUTRALIZATION PER ALPHA (by Sharpe)")
    print(f"{'='*100}")
    for alpha_name in pivot.index:
        row = pivot.loc[alpha_name]
        best_neut = row.idxmax()
        best_sr = row.max()
        print(f"  {alpha_name:25s} -> {best_neut:15s} (SR={best_sr:+.3f})")

    # Overall best
    print(f"\n  Average Sharpe by neutralization:")
    for neut in NEUTRALIZATIONS:
        avg = pivot[neut].mean()
        print(f"    {neut:15s}: {avg:+.3f}")

    return df


if __name__ == "__main__":
    run_sweep()
