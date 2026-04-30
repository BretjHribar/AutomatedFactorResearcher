"""
Methodical char selection on TRAIN set only (no OOS leakage).

Phase 1: univariate IS IC for ~55 candidate chars (TRAIN: bars before 2024-01-01)
Phase 2: forward selection by IS IC + redundancy filter
         (greedy: add next-best by univariate |IC|, skip if correlation with
          already-selected combined signal exceeds CORR_THRESHOLD)
Phase 3: final selected D-set → AIPT P=2000 backtest on OOS

NOTHING from OOS touches the selection. Selected list saved to JSON.
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

CORR_THRESHOLD = 0.70   # skip char if its rank-corr with selected combined signal > this
MAX_CHARS = 30          # cap selection at this size
DELAY = 1
OUT_JSON = base.RESULTS_DIR / "selected_chars_train.json"

# Candidate pool = everything we might consider
ALL_CANDIDATES = list(set(D44 + [
    # extra momentum / vol horizons
    "momentum_5d", "momentum_20d", "momentum_60d", "momentum_120d", "momentum_252d",
    "historical_volatility_10", "historical_volatility_30", "historical_volatility_90",
    "parkinson_volatility_10", "parkinson_volatility_30", "parkinson_volatility_90",
    # microstructure / volume
    "high_low_range", "open_close_range", "close_position_in_range", "vwap_deviation",
    "volume_momentum_1", "volume_momentum_5_20", "volume_ratio_20d", "turnover",
    # beta / per-share value
    "beta_to_btc", "bookvalue_ps", "tangible_book_per_share", "fcf_per_share",
    "debt_to_assets",
]))


def load_all(matrices_dir, candidates, raw_fields):
    print(f"Loading PIT matrices...", flush=True)
    uni = pd.read_parquet(base.UNIVERSES_DIR / f"{base.UNIVERSE_NAME}.parquet")
    cov = uni.sum(axis=0) / len(uni)
    valid_tickers = sorted(cov[cov > base.COVERAGE_CUTOFF].index.tolist())

    matrices = {}
    fields = set(candidates) | set(raw_fields) | {"close"}
    for name in fields:
        fp = matrices_dir / f"{name}.parquet"
        if not fp.exists():
            continue
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

    add_extra_fundamentals(matrices)

    available = [c for c in candidates if c in matrices]
    print(f"  available chars in pool: {len(available)}/{len(candidates)}", flush=True)
    return matrices, tickers, matrices["close"].index, available


def rank_normalize_row(arr):
    out = np.zeros_like(arr, dtype=float)
    ok = np.isfinite(arr)
    if ok.sum() < 3:
        return out
    out[ok] = rankdata(arr[ok], method="average") / ok.sum() - 0.5
    return out


def univariate_ic(matrices, char, returns, train_start, train_end):
    """Mean cross-sectional Spearman IC of char (delay=1) vs next-bar returns over TRAIN."""
    z = matrices[char].values
    r = returns.values
    ics = []
    for t in range(train_start, train_end - 1):
        z_idx = t - DELAY
        if z_idx < 0:
            continue
        zr = rank_normalize_row(z[z_idx])
        r_t = r[t + 1]
        valid = np.isfinite(zr) & np.isfinite(r_t)
        if valid.sum() < 30:
            continue
        z_v, r_v = zr[valid], r_t[valid]
        if z_v.std() > 0 and r_v.std() > 0:
            ic = np.corrcoef(rankdata(z_v), rankdata(r_v))[0, 1]
            ics.append(ic)
    return float(np.mean(ics)) if ics else 0.0, len(ics)


def combined_signal_at_t(matrices, chars, t, sign_dict):
    """Sum of (sign × rank-normalized) chars at bar t, then re-rank-normalize."""
    if not chars:
        return None
    sums = None
    for c in chars:
        z_t = matrices[c].iloc[t].values
        r = rank_normalize_row(z_t)
        if sign_dict.get(c, 1) < 0:
            r = -r
        sums = r if sums is None else (sums + r)
    return rank_normalize_row(sums)


def correlation_with_selected(matrices, candidate, selected, sign_dict, dates,
                                train_start, train_end):
    """Mean cross-sectional rank-corr of candidate vs combined signal of selected (over TRAIN)."""
    if not selected:
        return 0.0
    corrs = []
    cand_z = matrices[candidate].values
    for t in range(train_start, train_end - 1, 10):  # sample every 10 bars for speed
        z_idx = t - DELAY
        if z_idx < 0:
            continue
        cand = rank_normalize_row(cand_z[z_idx])
        comb = combined_signal_at_t(matrices, selected, z_idx, sign_dict)
        if comb is None:
            continue
        valid = np.isfinite(cand) & np.isfinite(comb)
        if valid.sum() < 30:
            continue
        a, b = cand[valid], comb[valid]
        if a.std() > 0 and b.std() > 0:
            corrs.append(abs(np.corrcoef(rankdata(a), rankdata(b))[0, 1]))
    return float(np.mean(corrs)) if corrs else 0.0


def main():
    overall_t0 = time.time()
    matrices, tickers, dates, available = load_all(base.PIT_DIR, ALL_CANDIDATES, RAW_FIELDS)
    returns = (matrices["close"].pct_change(fill_method=None))

    # Train period: from start_bar to oos_start_idx
    oos_start_idx = next(i for i, d in enumerate(dates) if str(d) >= base.OOS_START)
    start_bar = max(1, oos_start_idx - base.TRAIN_BARS - 10)
    print(f"  TRAIN period: bars {start_bar}..{oos_start_idx}  "
          f"({dates[start_bar].date()} → {dates[oos_start_idx].date()})", flush=True)

    # ── Phase 1: univariate IC for each candidate ─────────────────────────
    print(f"\n[Phase 1] Univariate IS IC for {len(available)} candidates ...", flush=True)
    rows = []
    for i, c in enumerate(available):
        ts = time.time()
        ic, n = univariate_ic(matrices, c, returns, start_bar, oos_start_idx)
        rows.append({"char": c, "univariate_ic": ic, "abs_ic": abs(ic),
                     "n_bars": n, "secs": time.time()-ts})
        if (i+1) % 10 == 0:
            print(f"    [{i+1}/{len(available)}] {c:<28}  IC={ic:+.5f}  ({time.time()-ts:.1f}s)", flush=True)
    df = pd.DataFrame(rows).sort_values("abs_ic", ascending=False)
    print(f"\n  Top 15 by |IC|:")
    for _, r in df.head(15).iterrows():
        sign = "+" if r["univariate_ic"] >= 0 else "-"
        print(f"    {sign}{r['abs_ic']:.5f}  {r['char']}", flush=True)

    # Sign per char so we always combine signed-positive
    sign_dict = {r["char"]: (1 if r["univariate_ic"] >= 0 else -1) for _, r in df.iterrows()}

    # ── Phase 2: forward selection by univariate IC + redundancy filter ────
    print(f"\n[Phase 2] Forward selection (CORR_THRESHOLD={CORR_THRESHOLD}, MAX={MAX_CHARS})",
          flush=True)
    selected = []
    skipped = []
    for _, r in df.iterrows():
        if len(selected) >= MAX_CHARS:
            break
        c = r["char"]
        corr = correlation_with_selected(matrices, c, selected, sign_dict, dates,
                                          start_bar, oos_start_idx)
        if corr > CORR_THRESHOLD:
            skipped.append((c, corr, r["univariate_ic"]))
            continue
        selected.append(c)
        print(f"    [{len(selected):>2d}] +{c:<28}  univIC={r['univariate_ic']:+.5f}  "
              f"corr_w_existing={corr:.3f}", flush=True)

    print(f"\n  Skipped {len(skipped)} (corr > {CORR_THRESHOLD}):")
    for c, cr, ic in skipped[:10]:
        print(f"    -{c:<28}  univIC={ic:+.5f}  corr={cr:.3f}")

    # Save selection
    payload = {
        "selected": selected,
        "n_selected": len(selected),
        "signs": {c: int(sign_dict[c]) for c in selected},
        "univariate_ic": {r["char"]: r["univariate_ic"] for _, r in df.iterrows()},
        "skipped_redundant": [{"char": c, "corr_with_existing": cr, "univariate_ic": ic}
                              for c, cr, ic in skipped],
        "train_window": [str(dates[start_bar]), str(dates[oos_start_idx])],
        "n_candidates": len(available),
        "elapsed_min": (time.time() - overall_t0) / 60,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved selection to {OUT_JSON}")
    print(f"Selected D = {len(selected)}")


if __name__ == "__main__":
    main()
