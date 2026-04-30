"""
Extended candidate pool (~100 chars) for block-CV forward selection.
Same min-IC criterion across 5 TRAIN blocks. May find D=4-6 if there are
more regime-robust chars hiding in the wider pool.
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
from select_chars_train import rank_normalize_row
from select_chars_train_combined_ic import precompute_ranks, precompute_returns
from select_chars_block_cv import all_block_ics, K_BLOCKS, MAX_CHARS, EPS, DELAY

OUT_JSON = base.RESULTS_DIR / "selected_chars_block_cv_extended.json"

# Build a wider candidate pool: every PIT field + extra ratios computed inline.
PIT_DIR = base.PIT_DIR


def add_extra_ratios(matrices):
    """Compute extra ratios beyond what add_extra_fundamentals does."""
    def safe_div(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            out = a / b
        return out.replace([np.inf, -np.inf], np.nan)

    # Additional ratios not yet computed
    if all(f in matrices for f in ["ebitda","revenue"]):
        matrices["ebitda_margin"] = safe_div(matrices["ebitda"], matrices["revenue"])
    if all(f in matrices for f in ["net_debt","total_equity"]):
        matrices["net_debt_to_equity"] = safe_div(matrices["net_debt"], matrices["total_equity"])
    if all(f in matrices for f in ["inventory","revenue"]):
        matrices["inventory_to_revenue"] = safe_div(matrices["inventory"], matrices["revenue"])
    if all(f in matrices for f in ["operating_cashflow","net_income"]):
        matrices["earnings_quality"] = safe_div(matrices["operating_cashflow"], matrices["net_income"])
    if "operating_cashflow" in matrices:
        matrices["ocf_growth_252d"] = matrices["operating_cashflow"].pct_change(252, fill_method=None)
    if "free_cashflow" in matrices:
        matrices["fcf_growth_252d"] = matrices["free_cashflow"].pct_change(252, fill_method=None)
    if "ebit" in matrices:
        matrices["ebit_growth_252d"] = matrices["ebit"].pct_change(252, fill_method=None)
    # Price ratios
    if "close" in matrices and "eps" in matrices:
        matrices["pe_ratio"] = safe_div(matrices["close"], matrices["eps"])
    if "close" in matrices and "bookvalue_ps" in matrices:
        matrices["price_to_book"] = safe_div(matrices["close"], matrices["bookvalue_ps"])
    if all(f in matrices for f in ["close","revenue","shares_out"]):
        rps = safe_div(matrices["revenue"], matrices["shares_out"])
        matrices["price_to_sales"] = safe_div(matrices["close"], rps)
    # Momentum ratios (price relative to MAs)
    if "close" in matrices:
        for w in [20, 50, 200]:
            sma = matrices["close"].rolling(w, min_periods=w//2).mean()
            matrices[f"price_to_sma{w}"] = safe_div(matrices["close"], sma) - 1.0
    return matrices


def load_extended_pool(matrices_dir):
    print(f"Loading PIT matrices...", flush=True)
    uni = pd.read_parquet(base.UNIVERSES_DIR / f"{base.UNIVERSE_NAME}.parquet")
    cov = uni.sum(axis=0) / len(uni)
    valid_tickers = sorted(cov[cov > base.COVERAGE_CUTOFF].index.tolist())

    # Load EVERY parquet in matrices_pit + raw fields
    matrices = {}
    for fp in sorted(matrices_dir.glob("*.parquet")):
        name = fp.stem
        df = pd.read_parquet(fp)
        cols = [c for c in valid_tickers if c in df.columns]
        if cols:
            matrices[name] = df[cols]
    common_idx = matrices["close"].index
    for k in list(matrices):
        matrices[k] = matrices[k].reindex(index=common_idx)
    tickers = sorted(set(matrices["close"].columns))
    for k in list(matrices):
        matrices[k] = matrices[k].reindex(columns=tickers)

    # Add the 20 fundamental ratios from d44_fund + additional ratios
    add_extra_fundamentals(matrices)
    add_extra_ratios(matrices)

    # Filter to numeric chars suitable as features (not 'close' itself, not classifications)
    EXCLUDE = {"close", "_industry_groups", "_sector_groups"}
    candidates = sorted([k for k in matrices if k not in EXCLUDE
                          and matrices[k].notna().sum().sum() > 0
                          and matrices[k].iloc[-540:].notna().mean().mean() > 0.30])  # OOS coverage > 30%
    print(f"  Total chars in extended pool: {len(candidates)}", flush=True)
    return matrices, tickers, matrices["close"].index, candidates


def main():
    overall_t0 = time.time()
    matrices, tickers, dates, candidates = load_extended_pool(PIT_DIR)
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
    print(f"  K={K_BLOCKS} blocks of ~{block_size} bars each", flush=True)

    print(f"  Pre-computing ranks for {len(candidates)} chars...", flush=True)
    ranks = precompute_ranks(matrices, candidates, start_bar, oos_start_idx)
    returns_per_bar = precompute_returns(matrices, start_bar, oos_start_idx)

    print(f"  Sign estimation ...", flush=True)
    signs = {}
    univ_ic = {}
    for c in candidates:
        ics = all_block_ics([c], {c: 1}, ranks, returns_per_bar, blocks)
        avg = float(np.mean(ics))
        signs[c] = 1 if avg >= 0 else -1
        univ_ic[c] = avg

    print(f"\n[Forward selection min-IC across {K_BLOCKS} blocks, MAX={MAX_CHARS}, EPS={EPS}]",
          flush=True)
    selected = []
    history = []
    current_min_ic = -np.inf
    cands = list(candidates)

    for step in range(MAX_CHARS):
        ts = time.time()
        best_c, best_min_ic, best_block_ics = None, current_min_ic, None
        for c in cands:
            trial = selected + [c]
            block_ics = all_block_ics(trial, signs, ranks, returns_per_bar, blocks)
            min_ic = min(block_ics)
            if min_ic > best_min_ic + EPS:
                best_c, best_min_ic, best_block_ics = c, min_ic, block_ics
        if best_c is None:
            print(f"  no further improvement (current_min_IC={current_min_ic:.5f}). "
                  f"Stopping at D={len(selected)}.", flush=True)
            break
        gain = best_min_ic - current_min_ic
        selected.append(best_c)
        cands.remove(best_c)
        history.append({"step": step+1, "char": best_c, "min_ic": best_min_ic,
                         "block_ics": best_block_ics, "marginal_gain": gain,
                         "secs": time.time()-ts})
        current_min_ic = best_min_ic
        block_str = "  ".join(f"{ic:+.4f}" for ic in best_block_ics)
        print(f"  [{len(selected):>2d}] +{best_c:<28}  min_IC={best_min_ic:+.5f}  "
              f"per-block: [{block_str}]  ({time.time()-ts:.1f}s)", flush=True)

    payload = {
        "selected": selected, "n_selected": len(selected),
        "signs": {c: int(signs[c]) for c in selected},
        "univariate_avg_ic": univ_ic, "history": history,
        "k_blocks": K_BLOCKS, "final_min_ic": current_min_ic,
        "n_candidates": len(candidates),
        "train_window": [str(dates[start_bar]), str(dates[oos_start_idx])],
        "elapsed_min": (time.time() - overall_t0) / 60,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved to {OUT_JSON}")
    print(f"Selected D={len(selected)}, final TRAIN min-IC={current_min_ic:+.5f}")


if __name__ == "__main__":
    main()
