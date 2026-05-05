"""
Build top-N-by-ADV KuCoin 4h universes with 20-day (120 × 4h bars) rebalance.

At each rebalance bar:
  1. Compute the trailing 20-day mean of `quote_volume` (or `adv20`) per ticker.
  2. Rank descending; mark top-N as universe=True.
  3. Hold that membership for the next 120 bars.

Saves:
  data/kucoin_cache/universes/KUCOIN_TOP50_REBAL20D_4h.parquet
  data/kucoin_cache/universes/KUCOIN_TOP100_REBAL20D_4h.parquet

These are drop-in replacements for KUCOIN_TOP50_4h / KUCOIN_TOP100_4h.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent.parent

ADV_FIELD = "adv20"   # already 20-day rolling mean of quote_volume in dollars
BARS_PER_DAY = 6      # 4h bars
REBAL_BARS = 20 * BARS_PER_DAY   # = 120 bars between rebalances
MIN_HISTORY_BARS = 365 * BARS_PER_DAY   # 1 year of bars required before eligibility


def build(top_n: int):
    adv   = pd.read_parquet(ROOT / "data/kucoin_cache/matrices/4h" / f"{ADV_FIELD}.parquet")
    close = pd.read_parquet(ROOT / "data/kucoin_cache/matrices/4h" / "close.parquet")
    print(f"\nbuilding TOP{top_n}_REBAL20D — adv shape {adv.shape}, "
          f"span {adv.index.min()} -> {adv.index.max()}")

    # Per-ticker first-active-bar = first index where close has data (proxy for listing date)
    # If a ticker's first bar == dataset start, it was likely listed long before — give it
    # the benefit of the doubt and treat as already-eligible (set first_active = -inf bar).
    data_start = close.index[0]
    first_active = {}
    for col in close.columns:
        nonna = close[col].dropna().index
        if not len(nonna): continue
        if nonna[0] == data_start:
            # Pre-existing ticker — treat as if listed >= 1 yr before data start
            first_active[col] = data_start - pd.Timedelta(days=400)
        else:
            first_active[col] = nonna[0]
    fa = pd.Series(first_active)
    n_pre = int((fa < data_start).sum())
    print(f"  per-ticker first-active dates: "
          f"min={fa.min()}, max={fa.max()}, n={len(fa)} ({n_pre} treated as pre-existing)")

    out = pd.DataFrame(False, index=adv.index, columns=adv.columns)
    rebal_bars = list(range(0, len(adv), REBAL_BARS))
    print(f"  rebalances at bars: n={len(rebal_bars)}, "
          f"first={adv.index[rebal_bars[0]]}, last={adv.index[rebal_bars[-1]]}")
    print(f"  min-history requirement: {MIN_HISTORY_BARS} bars "
          f"({MIN_HISTORY_BARS / BARS_PER_DAY:.0f} days)")

    last_members = None
    eligibility_log = []
    for i, b in enumerate(rebal_bars):
        next_b = rebal_bars[i+1] if i+1 < len(rebal_bars) else len(adv)
        rebal_ts = adv.index[b]

        # Eligibility: a ticker is eligible only if it has been listed >= 365 days
        # (proxy: rebal_ts - first_active >= 365 days)
        min_listing_days = MIN_HISTORY_BARS / BARS_PER_DAY
        eligible = [col for col in adv.columns
                    if col in fa and (rebal_ts - fa[col]).days >= min_listing_days]
        adv_row = adv.iloc[b].reindex(eligible).dropna()
        eligibility_log.append((rebal_ts, len(eligible), len(adv_row)))

        if len(adv_row) < top_n:
            if last_members is None:
                continue
            members = last_members
        else:
            members = adv_row.nlargest(top_n).index.tolist()
            last_members = members
        out.iloc[b:next_b, out.columns.get_indexer(members)] = True

    # Print eligibility growth at first/middle/last rebalance
    el_df = pd.DataFrame(eligibility_log, columns=["date", "eligible", "with_adv"])
    print(f"\n  eligibility over time (eligible / with_adv):")
    for idx in [0, len(el_df)//4, len(el_df)//2, 3*len(el_df)//4, len(el_df)-1]:
        r = el_df.iloc[idx]
        print(f"    {r['date']}: {r['eligible']:>4d} eligible, {r['with_adv']:>4d} with ADV data")

    n_active = out.sum(axis=1)
    print(f"  active per bar:  min/median/max = "
          f"{int(n_active.min())} / {int(n_active.median())} / {int(n_active.max())}")
    n_unique = out.any(axis=0).sum()
    print(f"  unique tickers ever in universe: {int(n_unique)}")

    diffs = out.astype(int).diff().abs().sum(axis=1)
    rebal_change_bars = (diffs > 0).sum()
    avg_change = float(diffs[diffs > 0].mean()) if (diffs > 0).any() else 0
    print(f"  bars with membership change: {int(rebal_change_bars)} (expected ~{len(rebal_bars)})")
    print(f"  avg tickers swapped per rebalance: {avg_change:.1f}")

    out_path = ROOT / "data/kucoin_cache/universes" / f"KUCOIN_TOP{top_n}_REBAL20D_4h.parquet"
    out.to_parquet(out_path)
    print(f"  saved: {out_path.relative_to(ROOT)}")

    # Coverage by split for sanity
    import json
    cfg = json.load(open(ROOT / "prod/config/research_crypto.json"))
    te = pd.Timestamp(cfg["splits"]["train_end"])
    ve = pd.Timestamp(cfg["splits"]["val_end"])
    print(f"  active/bar by split:")
    for lab, sl in [("TRAIN", slice(None, te)), ("VAL", slice(te, ve)), ("TEST", slice(ve, None))]:
        a = out.loc[sl].sum(axis=1)
        print(f"    {lab:5s} mean={a.mean():.1f}  min={int(a.min())}  max={int(a.max())}")
    return out


if __name__ == "__main__":
    build(30)
