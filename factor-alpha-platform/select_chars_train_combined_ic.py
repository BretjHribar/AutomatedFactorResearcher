"""
Forward selection by COMBINED IC marginal gain (TRAIN only).

At each step, for each remaining candidate char c:
  - Build combined signal = sign(c)·rank(c_t-1) summed across S ∪ {c} for each bar
  - Score = mean cross-sectional IC of combined signal vs next-bar returns over TRAIN
Pick the candidate with highest combined IC. Repeat until no candidate improves
the combined IC by more than EPS, or MAX_CHARS reached.

This is the "real" forward selection that measures whether each char ADDS
predictive value, not just whether it's rank-uncorrelated.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import backtest_voc_equities_neutralized as base
from backtest_voc_equities_d44_fund import D44, RAW_FIELDS, add_extra_fundamentals
from select_chars_train import ALL_CANDIDATES, load_all, rank_normalize_row

DELAY = 1
MAX_CHARS = 30
EPS = 0.00002    # only add a char if it improves combined IC by this much
OUT_JSON = base.RESULTS_DIR / "selected_chars_combined_ic.json"


def precompute_ranks(matrices, candidates, train_start, train_end):
    """For each candidate char, pre-compute the rank-normalized vectors for every TRAIN bar.
    Returns dict char -> (T_train, N) array of rank-normalized values."""
    print(f"  pre-computing rank-normalized panels for {len(candidates)} chars...", flush=True)
    t0 = time.time()
    out = {}
    bars = list(range(train_start, train_end))
    for c in candidates:
        arr = matrices[c].values
        N = arr.shape[1]
        rk = np.zeros((len(bars), N))
        for i, t in enumerate(bars):
            z_idx = t - DELAY
            if z_idx < 0:
                continue
            rk[i] = rank_normalize_row(arr[z_idx])
        out[c] = rk
    print(f"  done in {time.time()-t0:.1f}s", flush=True)
    return out


def precompute_returns(matrices, train_start, train_end):
    """Per-bar realized return + valid mask for each TRAIN bar."""
    close = matrices["close"].values
    bars = list(range(train_start, train_end))
    rets = []
    for t in bars:
        if t + 1 >= close.shape[0]:
            rets.append((None, None))
            continue
        r = (close[t + 1] - close[t]) / close[t]
        valid = np.isfinite(r) & np.isfinite(close[t]) & np.isfinite(close[t + 1])
        rets.append((r, valid))
    return rets


def combined_ic_of(selected, signs, ranks_dict, returns_per_bar):
    """Mean cross-sectional Spearman IC of equal-weight sum of (signed × rank-normalized) chars
    over TRAIN bars."""
    if not selected:
        return 0.0
    # Combined signal per bar
    n_bars = len(returns_per_bar)
    sums = np.zeros_like(ranks_dict[selected[0]])
    for c in selected:
        sums = sums + signs[c] * ranks_dict[c]

    ics = []
    for i in range(n_bars):
        r, valid = returns_per_bar[i]
        if r is None or valid.sum() < 30:
            continue
        s = sums[i]
        v = valid & np.isfinite(s)
        if v.sum() < 30:
            continue
        s_v, r_v = s[v], r[v]
        if s_v.std() > 0 and r_v.std() > 0:
            # Spearman: corr of ranks
            ic = np.corrcoef(rankdata(s_v), rankdata(r_v))[0, 1]
            ics.append(ic)
    return float(np.mean(ics)) if ics else 0.0


def main():
    overall_t0 = time.time()
    matrices, tickers, dates, available = load_all(base.PIT_DIR, ALL_CANDIDATES, RAW_FIELDS)
    oos_start_idx = next(i for i, d in enumerate(dates) if str(d) >= base.OOS_START)
    start_bar = max(1, oos_start_idx - base.TRAIN_BARS - 10)
    print(f"  TRAIN: bars {start_bar}..{oos_start_idx}  "
          f"({dates[start_bar].date()} → {dates[oos_start_idx].date()})", flush=True)

    # Pre-compute ranks once (reused at every selection step)
    ranks = precompute_ranks(matrices, available, start_bar, oos_start_idx)
    returns_per_bar = precompute_returns(matrices, start_bar, oos_start_idx)

    # First pass: univariate IC to determine sign per char (we still want signed-positive entries)
    print(f"\n[Sign estimation] univariate IC for {len(available)} chars to fix sign...", flush=True)
    signs = {}
    univ_ic = {}
    for c in available:
        ic = combined_ic_of([c], {c: 1}, ranks, returns_per_bar)
        signs[c] = 1 if ic >= 0 else -1
        univ_ic[c] = abs(ic)
    print(f"  signs assigned (negative-IC chars get sign=-1 so combined sum is positive)", flush=True)

    # ── Forward selection by combined-IC marginal gain ───────────────────
    print(f"\n[Forward selection by combined IC, MAX={MAX_CHARS}, EPS={EPS}]", flush=True)
    selected = []
    history = []
    current_ic = 0.0
    candidates = list(available)

    for step in range(MAX_CHARS):
        ts = time.time()
        best_c, best_ic = None, current_ic
        for c in candidates:
            trial_set = selected + [c]
            ic = combined_ic_of(trial_set, signs, ranks, returns_per_bar)
            if ic > best_ic + EPS:
                best_c, best_ic = c, ic
        if best_c is None:
            print(f"  no further improvement (current_ic={current_ic:.5f}). Stopping at D={len(selected)}.",
                  flush=True)
            break
        gain = best_ic - current_ic
        selected.append(best_c)
        candidates.remove(best_c)
        history.append({"step": step + 1, "char": best_c, "combined_ic": best_ic,
                         "marginal_gain": gain, "secs": time.time() - ts})
        current_ic = best_ic
        print(f"  [{len(selected):>2d}] +{best_c:<28}  combined_IC={best_ic:+.5f}  "
              f"gain=+{gain:.5f}  ({time.time()-ts:.1f}s)", flush=True)

    # Save
    payload = {
        "selected": selected,
        "n_selected": len(selected),
        "signs": {c: int(signs[c]) for c in selected},
        "univariate_abs_ic": univ_ic,
        "history": history,
        "final_combined_ic": current_ic,
        "n_candidates": len(available),
        "train_window": [str(dates[start_bar]), str(dates[oos_start_idx])],
        "elapsed_min": (time.time() - overall_t0) / 60,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved to {OUT_JSON}")
    print(f"Selected D={len(selected)}, final TRAIN combined IC={current_ic:+.5f}")
    print(f"Total: {(time.time()-overall_t0)/60:.1f}min")


if __name__ == "__main__":
    main()
