"""
Round 2 — R3: ADV-weighted positions vs equal-weighted (Russell-style).

Theory: equal-weight inside the universe means BTC gets the same weight as the
30th coin even though BTC has 10× the ADV. Russell uses *float-adjusted market
cap* weighting. The crypto analog is ADV-weighting.

Hypothesis: ADV-weighting trades some signal sharpness for capacity. SR may
drop slightly but DD should drop too (bigger names are less volatile). The
real question: is it a wash on net, or does it actively help/hurt?

Implementation: scale demeaned signal by ADV-rank weight before gross-norm.
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
    build_universe_topn_rebal, eval_signal,
    split_metrics, load_all,
    BENCHMARK_EXPR, BARS_PER_DAY, OUT, log,
)


def signal_to_portfolio_weighted(sig, universe, adv,
                                  weight_method: str = "equal",
                                  max_wt: float = 0.10):
    """Demean within universe, optionally tilt by ADV, gross-normalize, clip."""
    common = sig.columns.intersection(universe.columns)
    s = sig[common].replace([np.inf, -np.inf], np.nan)
    uni_mask = universe.reindex(index=s.index, columns=common).fillna(False).astype(bool)
    s = s.where(uni_mask, np.nan)
    demean = s.sub(s.mean(axis=1), axis=0)

    if weight_method == "adv":
        # Per-bar ADV rank-based weight: rank ascending, larger ADV gets bigger weight
        adv_aligned = adv.reindex(index=s.index, columns=common)
        adv_in_uni = adv_aligned.where(uni_mask, np.nan)
        # log-ADV scale (preserves ratio sense)
        log_adv = np.log(adv_in_uni.clip(lower=1.0))
        # Normalize to mean 1.0 per bar (preserves overall scale)
        scale = log_adv.div(log_adv.mean(axis=1), axis=0).fillna(1.0)
        demean = demean.mul(scale, axis=0)
    elif weight_method == "inv_vol":
        # Inverse-vol rank weight: smaller vol gets bigger weight (risk parity)
        # using parkinson_vol_60 already in matrices won't work here, skip
        raise NotImplementedError
    # equal: no scaling

    demean = demean.where(uni_mask, np.nan)
    gross = demean.abs().sum(axis=1).replace(0, np.nan)
    w = demean.div(gross, axis=0)
    w = w.clip(lower=-max_wt, upper=max_wt)
    return w.fillna(0)


def main():
    matrices = load_all()
    sig = eval_signal(BENCHMARK_EXPR, matrices)
    log("R3: ADV-weighted vs equal-weighted")
    rows = []
    for top_n in [20, 30, 50]:
        for weight_method in ["equal", "adv"]:
            uni = build_universe_topn_rebal(matrices["adv20"], matrices["close"],
                                             top_n=top_n, rebal_bars=20*BARS_PER_DAY,
                                             min_history_days=365)
            w = signal_to_portfolio_weighted(sig, uni, matrices["adv20"],
                                             weight_method=weight_method)
            m = split_metrics(w, matrices["returns"], fee_bps=3.0)
            m["top_n"] = top_n
            m["weight_method"] = weight_method
            m["avg_active"] = float(uni.sum(axis=1).mean())
            rows.append(m)
            print(f"  TOP{top_n}/{weight_method:>5s}  TR={m['TRAIN_SR_n']:+.2f} "
                  f"VAL={m['VAL_SR_n']:+.2f} TEST={m['TEST_SR_n']:+.2f} "
                  f"VT={m['VT_SR_n']:+.2f} DD={m['VT_dd_n']*100:+.0f}% "
                  f"TO={m['to_per_bar']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT/"round2_r3_adv_weighted.csv", index=False, float_format="%.4f")
    log("R3 DONE")


if __name__ == "__main__":
    main()
