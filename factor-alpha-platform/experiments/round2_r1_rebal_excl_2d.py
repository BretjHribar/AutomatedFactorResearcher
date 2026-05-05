"""
Round 2 — R1: rebalance × exclusion 2D sweep.

Theory (from H3+H4): regime-commitment + meme-overfit interact. 60d rebal won
because the universe stabilized; meme-exclusion won because hype phases didn't
transfer. Combined, the optimal might NOT be at the marginal optima of each
axis.

Hypothesis: 30d-90d rebal × no-meme exclusion has a non-trivial 2D optimum.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"experiments"))
from universe_experiments import (
    build_universe_topn_rebal, eval_signal, signal_to_portfolio,
    split_metrics, load_all,
    BENCHMARK_EXPR, BARS_PER_DAY, OUT, log,
)

MEMES = ["DOGE", "SHIB", "PEPE", "BONK", "WIF", "FLOKI", "MEME",
         "MOG", "MOODENG", "TRUMP", "FART"]


def main():
    matrices = load_all()
    sig = eval_signal(BENCHMARK_EXPR, matrices)
    log("R1: rebal × exclusion 2D sweep starting")
    rows = []
    for rebal_days in [10, 20, 30, 60, 90, 120]:
        for excl_set in [("none", []), ("memes", MEMES)]:
            t0 = time.time()
            uni = build_universe_topn_rebal(matrices["adv20"], matrices["close"],
                                            top_n=30, rebal_bars=rebal_days*BARS_PER_DAY,
                                            min_history_days=365,
                                            exclusions=excl_set[1])
            w = signal_to_portfolio(sig, uni)
            m = split_metrics(w, matrices["returns"], fee_bps=3.0)
            m.update({"rebal_days": rebal_days, "excl": excl_set[0],
                      "avg_active": float(uni.sum(axis=1).mean())})
            rows.append(m)
            print(f"  rebal={rebal_days:>3d}d  excl={excl_set[0]:>5s}  "
                  f"TR={m['TRAIN_SR_n']:+.2f} VAL={m['VAL_SR_n']:+.2f} "
                  f"TEST={m['TEST_SR_n']:+.2f} VT={m['VT_SR_n']:+.2f} "
                  f"DD={m['VT_dd_n']*100:+.0f}% TO={m['to_per_bar']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT/"round2_r1_rebal_excl_2d.csv", index=False, float_format="%.4f")
    log("R1 DONE")


if __name__ == "__main__":
    main()
