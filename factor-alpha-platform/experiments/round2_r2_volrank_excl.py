"""
Round 2 — R2: vol-rank exclusion (test the meme-proxy hypothesis).

Theory (from H4): excluding memes by NAME helped OOS. The proposed mechanism
was that memes are high-vol idiosyncratic regime-specific signals that overfit
TRAIN.

Hypothesis: if the mechanism is "high vol = bad", then excluding the top-K
highest-vol coins (by parkinson_volatility_60 rank) should match or beat the
explicit name-pattern meme exclusion. If vol-based exclusion does NOT match
name-based, then memes are special beyond just being high-vol.

Build TOP30 universe with min_hist=365d, rebal=20d, then for each candidate
ticker test if its avg vol over TRAIN exceeds threshold. Exclude those.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"experiments"))
from universe_experiments import (
    build_universe_topn_rebal, eval_signal, signal_to_portfolio,
    split_metrics, load_all,
    BENCHMARK_EXPR, BARS_PER_DAY, TRAIN_END, OUT, log,
)


def main():
    matrices = load_all()
    sig = eval_signal(BENCHMARK_EXPR, matrices)
    log("R2: vol-rank exclusion sweep")

    # For each ticker, compute mean parkinson_vol over TRAIN
    pv = matrices["parkinson_volatility_60"]
    train_vol = pv.loc[:TRAIN_END].mean()  # per-ticker mean vol

    rows = []
    for vol_pct in [None, 95, 90, 85, 75, 50]:
        excl = []
        if vol_pct is not None:
            threshold = train_vol.quantile(vol_pct/100.0)
            excl = train_vol[train_vol > threshold].index.tolist()
        # Use as exclusion list (matches names exactly)
        uni = build_universe_topn_rebal(matrices["adv20"], matrices["close"],
                                         top_n=30, rebal_bars=20*BARS_PER_DAY,
                                         min_history_days=365,
                                         exclusions=excl if vol_pct else None)
        if (uni.sum(axis=1) > 0).sum() < 100:
            print(f"  vol_pct={vol_pct}: skipped (too few eligible)"); continue
        w = signal_to_portfolio(sig, uni)
        m = split_metrics(w, matrices["returns"], fee_bps=3.0)
        m["excl_above_pct"] = vol_pct if vol_pct else "none"
        m["n_excluded"] = len(excl)
        m["avg_active"] = float(uni.sum(axis=1).mean())
        m["unique_tickers"] = int(uni.any(axis=0).sum())
        rows.append(m)
        print(f"  excl_top {str(vol_pct or 'none'):>4s}-pct-vol  n_excl={len(excl):>3d}  "
              f"unique={m['unique_tickers']:>3d}  TR={m['TRAIN_SR_n']:+.2f} "
              f"VAL={m['VAL_SR_n']:+.2f} TEST={m['TEST_SR_n']:+.2f} "
              f"VT={m['VT_SR_n']:+.2f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT/"round2_r2_volrank.csv", index=False, float_format="%.4f")
    log("R2 DONE")


if __name__ == "__main__":
    main()
