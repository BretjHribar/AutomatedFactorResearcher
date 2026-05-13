"""
Build top-N-by-ADV equity universes with 20-day rebalance (Isichenko convention).

Equity analogue of tools/build_kucoin_universe_20d.py:
  1. At each rebalance bar (every 20 trading days), compute adv60[t] per ticker.
  2. Apply eligibility filter: 252-day min history + PIT membership True + close[t] > 0.
  3. Rank descending; mark top-N as universe=True.
  4. Hold that membership for the next 20 bars.

Saves drop-in replacements for the static frozen universes used by AIPT:
  experiments/data/aipt_universes/TOP1000_REBAL20D.parquet
  experiments/data/aipt_universes/TOP3000_REBAL20D.parquet
  experiments/data/aipt_universes/MIDCAP_500M_5B_REBAL20D.parquet

The midcap variant is market_cap in [500MM, 5B] (so distinct from the existing
100MM-500MM smallcap), still subject to 20-day rebalance + min-history filter.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent.parent

PIT_MAT = ROOT / "data/fmp_cache/matrices_pit_v2"
PIT_MEMBERSHIP = ROOT / "data/fmp_cache/universes_pit/membership.parquet"
OUT_DIR = ROOT / "experiments/data/aipt_universes"

ADV_FIELD = "adv60"
REBAL_BARS = 20            # rebalance every 20 trading days
MIN_HISTORY_BARS = 252     # require 1 year of close history before eligibility

MIDCAP_LOW = 500_000_000.0
MIDCAP_HIGH = 5_000_000_000.0
SMALLCAP_LOW = 100_000_000.0
SMALLCAP_HIGH = 500_000_000.0


def _read(name: str) -> pd.DataFrame:
    df = pd.read_parquet(PIT_MAT / f"{name}.parquet")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _first_active_dates(close: pd.DataFrame) -> pd.Series:
    """First date each ticker has finite close. Pre-existing tickers (active at
    data_start) are treated as if listed long before the dataset start."""
    data_start = close.index[0]
    fa = {}
    for col in close.columns:
        nonna = close[col].dropna().index
        if not len(nonna):
            continue
        if nonna[0] == data_start:
            fa[col] = data_start - pd.Timedelta(days=400)
        else:
            fa[col] = nonna[0]
    return pd.Series(fa)


def build_top_n(top_n: int, adv: pd.DataFrame, close: pd.DataFrame,
                membership: pd.DataFrame, first_active: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(False, index=adv.index, columns=adv.columns)
    rebal_bars = list(range(0, len(adv), REBAL_BARS))
    min_listing_days = MIN_HISTORY_BARS
    last_members = None
    log = []
    for i, b in enumerate(rebal_bars):
        next_b = rebal_bars[i + 1] if i + 1 < len(rebal_bars) else len(adv)
        rebal_ts = adv.index[b]
        # Eligibility: PIT-listed >= 1yr, PIT-member at t, close>0 at t.
        member_row = membership.iloc[b]
        close_row = close.iloc[b]
        eligible = [
            c for c in adv.columns
            if c in first_active
            and (rebal_ts - first_active[c]).days >= min_listing_days
            and bool(member_row.get(c, False))
            and np.isfinite(close_row.get(c, np.nan))
            and close_row.get(c, 0.0) > 0
        ]
        adv_row = adv.iloc[b].reindex(eligible).dropna()
        log.append((rebal_ts, len(eligible), len(adv_row)))
        if len(adv_row) < top_n:
            members = last_members if last_members is not None else adv_row.index.tolist()
        else:
            members = adv_row.nlargest(top_n).index.tolist()
        last_members = members
        out.iloc[b:next_b, out.columns.get_indexer(members)] = True
    el_df = pd.DataFrame(log, columns=["date", "eligible", "with_adv"])
    for idx in [0, len(el_df) // 4, len(el_df) // 2, 3 * len(el_df) // 4, len(el_df) - 1]:
        r = el_df.iloc[idx]
        print(f"    {r['date'].strftime('%Y-%m-%d')}: {r['eligible']:>5d} eligible, "
              f"{r['with_adv']:>5d} with ADV", flush=True)
    return out


def build_midcap(adv: pd.DataFrame, close: pd.DataFrame,
                 membership: pd.DataFrame, market_cap: pd.DataFrame,
                 first_active: pd.Series) -> pd.DataFrame:
    return _build_cap_band(adv, close, membership, market_cap, first_active,
                           MIDCAP_LOW, MIDCAP_HIGH)


def build_smallcap(adv: pd.DataFrame, close: pd.DataFrame,
                   membership: pd.DataFrame, market_cap: pd.DataFrame,
                   first_active: pd.Series) -> pd.DataFrame:
    return _build_cap_band(adv, close, membership, market_cap, first_active,
                           SMALLCAP_LOW, SMALLCAP_HIGH)


def _build_cap_band(adv: pd.DataFrame, close: pd.DataFrame,
                    membership: pd.DataFrame, market_cap: pd.DataFrame,
                    first_active: pd.Series,
                    low: float, high: float) -> pd.DataFrame:
    """Market cap band [low, high) at the rebalance bar, hold for 20 bars."""
    out = pd.DataFrame(False, index=adv.index, columns=adv.columns)
    rebal_bars = list(range(0, len(adv), REBAL_BARS))
    min_listing_days = MIN_HISTORY_BARS
    last_members = None
    for i, b in enumerate(rebal_bars):
        next_b = rebal_bars[i + 1] if i + 1 < len(rebal_bars) else len(adv)
        rebal_ts = adv.index[b]
        member_row = membership.iloc[b]
        close_row = close.iloc[b]
        cap_row = market_cap.iloc[b]
        eligible = []
        for c in adv.columns:
            if c not in first_active:
                continue
            if (rebal_ts - first_active[c]).days < min_listing_days:
                continue
            if not bool(member_row.get(c, False)):
                continue
            px = close_row.get(c, np.nan)
            if not np.isfinite(px) or px <= 0:
                continue
            mc = cap_row.get(c, np.nan)
            if not np.isfinite(mc) or mc < low or mc >= high:
                continue
            eligible.append(c)
        members = eligible if eligible else (last_members or [])
        last_members = members
        out.iloc[b:next_b, out.columns.get_indexer(members)] = True
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    adv = _read(ADV_FIELD)
    close = _read("close")
    market_cap = _read("market_cap")
    membership = pd.read_parquet(PIT_MEMBERSHIP).astype(bool).sort_index()
    # Align all matrices on the common index/columns.
    idx = adv.index.intersection(close.index).intersection(membership.index).intersection(market_cap.index)
    cols = sorted(set(adv.columns) & set(close.columns) & set(membership.columns) & set(market_cap.columns))
    adv = adv.reindex(index=idx, columns=cols)
    close = close.reindex(index=idx, columns=cols)
    membership = membership.reindex(index=idx, columns=cols).fillna(False).astype(bool)
    market_cap = market_cap.reindex(index=idx, columns=cols)

    print(f"data span: {idx.min()} -> {idx.max()}, n_dates={len(idx)}, n_tickers={len(cols)}", flush=True)
    fa = _first_active_dates(close)
    n_pre = int((fa < idx[0]).sum())
    print(f"per-ticker first-active dates: n={len(fa)} ({n_pre} treated as pre-existing)", flush=True)

    summaries = {}
    for top_n, name in [(1000, "TOP1000_REBAL20D"), (3000, "TOP3000_REBAL20D")]:
        print(f"\nbuilding {name}...", flush=True)
        uni = build_top_n(top_n, adv, close, membership, fa)
        n_active = uni.sum(axis=1)
        n_unique = uni.any(axis=0).sum()
        diffs = uni.astype(int).diff().abs().sum(axis=1)
        rebal_change_bars = int((diffs > 0).sum())
        avg_change = float(diffs[diffs > 0].mean()) if (diffs > 0).any() else 0.0
        path = OUT_DIR / f"{name}.parquet"
        uni.to_parquet(path)
        summaries[f"{name}.parquet"] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "shape": list(uni.shape),
            "first_date": str(uni.index.min()),
            "last_date": str(uni.index.max()),
            "active_per_bar_min": int(n_active.min()),
            "active_per_bar_median": int(n_active.median()),
            "active_per_bar_max": int(n_active.max()),
            "unique_members": int(n_unique),
            "rebalance_change_bars": rebal_change_bars,
            "avg_swaps_per_rebal": avg_change,
        }
        print(f"  active/bar min/med/max = {int(n_active.min())}/{int(n_active.median())}/{int(n_active.max())}",
              flush=True)
        print(f"  unique tickers ever in universe: {int(n_unique)}", flush=True)
        print(f"  saved: {path.relative_to(ROOT)}", flush=True)

    print(f"\nbuilding SMALLCAP_100M_500M_REBAL20D...", flush=True)
    uni_small = build_smallcap(adv, close, membership, market_cap, fa)
    n_active_s = uni_small.sum(axis=1)
    n_unique_s = uni_small.any(axis=0).sum()
    diffs_s = uni_small.astype(int).diff().abs().sum(axis=1)
    path_s = OUT_DIR / "SMALLCAP_100M_500M_REBAL20D.parquet"
    uni_small.to_parquet(path_s)
    summaries["SMALLCAP_100M_500M_REBAL20D.parquet"] = {
        "path": str(path_s.relative_to(ROOT)).replace("\\", "/"),
        "shape": list(uni_small.shape),
        "first_date": str(uni_small.index.min()),
        "last_date": str(uni_small.index.max()),
        "active_per_bar_min": int(n_active_s.min()),
        "active_per_bar_median": int(n_active_s.median()),
        "active_per_bar_max": int(n_active_s.max()),
        "unique_members": int(n_unique_s),
        "rebalance_change_bars": int((diffs_s > 0).sum()),
    }
    print(f"  active/bar min/med/max = {int(n_active_s.min())}/{int(n_active_s.median())}/{int(n_active_s.max())}",
          flush=True)
    print(f"  unique tickers ever in universe: {int(n_unique_s)}", flush=True)
    print(f"  saved: {path_s.relative_to(ROOT)}", flush=True)

    print(f"\nbuilding MIDCAP_500M_5B_REBAL20D...", flush=True)
    uni_mid = build_midcap(adv, close, membership, market_cap, fa)
    n_active = uni_mid.sum(axis=1)
    n_unique = uni_mid.any(axis=0).sum()
    diffs = uni_mid.astype(int).diff().abs().sum(axis=1)
    path = OUT_DIR / "MIDCAP_500M_5B_REBAL20D.parquet"
    uni_mid.to_parquet(path)
    summaries["MIDCAP_500M_5B_REBAL20D.parquet"] = {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "shape": list(uni_mid.shape),
        "first_date": str(uni_mid.index.min()),
        "last_date": str(uni_mid.index.max()),
        "active_per_bar_min": int(n_active.min()),
        "active_per_bar_median": int(n_active.median()),
        "active_per_bar_max": int(n_active.max()),
        "unique_members": int(n_unique),
        "rebalance_change_bars": int((diffs > 0).sum()),
    }
    print(f"  active/bar min/med/max = {int(n_active.min())}/{int(n_active.median())}/{int(n_active.max())}",
          flush=True)
    print(f"  unique tickers ever in universe: {int(n_unique)}", flush=True)
    print(f"  saved: {path.relative_to(ROOT)}", flush=True)

    manifest = {
        "source_membership": str(PIT_MEMBERSHIP.relative_to(ROOT)).replace("\\", "/"),
        "source_matrices": str(PIT_MAT.relative_to(ROOT)).replace("\\", "/"),
        "rebalance_bars": REBAL_BARS,
        "min_history_bars": MIN_HISTORY_BARS,
        "adv_field": ADV_FIELD,
        "midcap_range": [MIDCAP_LOW, MIDCAP_HIGH],
        "outputs": summaries,
    }
    (OUT_DIR / "manifest_rebal20d.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nmanifest: {(OUT_DIR / 'manifest_rebal20d.json').relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
