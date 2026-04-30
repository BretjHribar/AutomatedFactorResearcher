"""
Forward selection by COMBINED IC, with BLOCK CROSS-VALIDATION on TRAIN.

TRAIN split into K=5 contiguous blocks (~302 bars / ~14 months each, covering
2017-2024 including COVID crash). At each forward step, the candidate's score
is the MIN combined IC across all K blocks — forcing the selection to favor
chars that work in EVERY sub-regime, not just the average.

This addresses the regime-overfit problem from the simple combined-IC
selection (which picked all-quality/momentum chars that crushed in 2024-2026).
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
from backtest_voc_equities_d44_fund import RAW_FIELDS, add_extra_fundamentals
from select_chars_train import ALL_CANDIDATES, load_all, rank_normalize_row
from select_chars_train_combined_ic import precompute_ranks, precompute_returns

K_BLOCKS = 5
MAX_CHARS = 30
EPS = 0.00001  # need positive marginal min-IC gain to add
DELAY = 1
OUT_JSON = base.RESULTS_DIR / "selected_chars_block_cv.json"


def block_ic(selected_sums, returns_per_bar, block_indices):
    """Mean IC over a single block of bar indices (within TRAIN)."""
    ics = []
    for i in block_indices:
        r, valid = returns_per_bar[i]
        if r is None or valid.sum() < 30:
            continue
        s = selected_sums[i]
        v = valid & np.isfinite(s)
        if v.sum() < 30:
            continue
        s_v, r_v = s[v], r[v]
        if s_v.std() > 0 and r_v.std() > 0:
            ics.append(np.corrcoef(rankdata(s_v), rankdata(r_v))[0, 1])
    return float(np.mean(ics)) if ics else 0.0


def all_block_ics(selected, signs, ranks_dict, returns_per_bar, blocks):
    """Compute combined IC in each of K blocks."""
    if not selected:
        return [0.0] * len(blocks)
    sums = np.zeros_like(ranks_dict[selected[0]])
    for c in selected:
        sums = sums + signs[c] * ranks_dict[c]
    return [block_ic(sums, returns_per_bar, idxs) for idxs in blocks]


def main():
    overall_t0 = time.time()
    matrices, tickers, dates, available = load_all(base.PIT_DIR, ALL_CANDIDATES, RAW_FIELDS)
    oos_start_idx = next(i for i, d in enumerate(dates) if str(d) >= base.OOS_START)
    start_bar = max(1, oos_start_idx - base.TRAIN_BARS - 10)
    print(f"  TRAIN: bars {start_bar}..{oos_start_idx}  "
          f"({dates[start_bar].date()} → {dates[oos_start_idx].date()})", flush=True)

    n_train = oos_start_idx - start_bar
    block_size = n_train // K_BLOCKS
    blocks = []
    for k in range(K_BLOCKS):
        s = k * block_size
        e = (k + 1) * block_size if k < K_BLOCKS - 1 else n_train
        blocks.append(list(range(s, e)))
        d_s, d_e = dates[start_bar + s].date(), dates[start_bar + e - 1].date()
        print(f"    block {k+1}: bars {s}..{e}  ({d_s} → {d_e})  n={e-s}", flush=True)

    ranks = precompute_ranks(matrices, available, start_bar, oos_start_idx)
    returns_per_bar = precompute_returns(matrices, start_bar, oos_start_idx)

    # Sign per char from full-train univariate IC (so combined sums positive)
    print(f"\n[Sign estimation]", flush=True)
    signs = {}
    univ_ic = {}
    for c in available:
        ics = all_block_ics([c], {c: 1}, ranks, returns_per_bar, blocks)
        avg = float(np.mean(ics))
        signs[c] = 1 if avg >= 0 else -1
        univ_ic[c] = avg

    # ── Forward selection by MIN-IC across blocks ────────────────────────
    print(f"\n[Forward selection by MIN-IC across {K_BLOCKS} blocks, MAX={MAX_CHARS}, EPS={EPS}]",
          flush=True)
    selected = []
    history = []
    current_min_ic = -np.inf
    candidates = list(available)

    for step in range(MAX_CHARS):
        ts = time.time()
        best_c, best_min_ic, best_block_ics = None, current_min_ic, None
        for c in candidates:
            trial_set = selected + [c]
            block_ics = all_block_ics(trial_set, signs, ranks, returns_per_bar, blocks)
            min_ic = min(block_ics)
            if min_ic > best_min_ic + EPS:
                best_c, best_min_ic, best_block_ics = c, min_ic, block_ics
        if best_c is None:
            print(f"  no further improvement (current_min_IC={current_min_ic:.5f}). "
                  f"Stopping at D={len(selected)}.", flush=True)
            break
        gain = best_min_ic - current_min_ic
        selected.append(best_c)
        candidates.remove(best_c)
        history.append({"step": step + 1, "char": best_c,
                         "min_ic": best_min_ic, "block_ics": best_block_ics,
                         "marginal_gain": gain, "secs": time.time() - ts})
        current_min_ic = best_min_ic
        block_str = "  ".join(f"{ic:+.4f}" for ic in best_block_ics)
        print(f"  [{len(selected):>2d}] +{best_c:<28}  min_IC={best_min_ic:+.5f}  "
              f"per-block: [{block_str}]  ({time.time()-ts:.1f}s)", flush=True)

    payload = {
        "selected": selected,
        "n_selected": len(selected),
        "signs": {c: int(signs[c]) for c in selected},
        "univariate_avg_ic": univ_ic,
        "history": history,
        "k_blocks": K_BLOCKS,
        "final_min_ic": current_min_ic,
        "n_candidates": len(available),
        "train_window": [str(dates[start_bar]), str(dates[oos_start_idx])],
        "elapsed_min": (time.time() - overall_t0) / 60,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved to {OUT_JSON}")
    print(f"Selected D={len(selected)}, final TRAIN min-IC across {K_BLOCKS} blocks = {current_min_ic:+.5f}")


if __name__ == "__main__":
    main()
