"""Build experiment-local PIT equity universes for AIPT.

The platform's legacy equity universe files can have a narrower column set than
the PIT v2 matrices. These universes are built from:

  - data/fmp_cache/universes_pit/membership.parquet
  - data/fmp_cache/matrices_pit_v2/market_cap.parquet
  - data/fmp_cache/matrices_pit_v2/adv60.parquet
  - data/fmp_cache/matrices_pit_v2/close.parquet

All filters are same-date only; no future membership, return, or validation
metric is used.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "experiments/data/aipt_universes"
PIT_MAT = ROOT / "data/fmp_cache/matrices_pit_v2"
PIT_UNI = ROOT / "data/fmp_cache/universes_pit/membership.parquet"


def _read(name: str) -> pd.DataFrame:
    path = PIT_MAT / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    membership = pd.read_parquet(PIT_UNI).astype(bool).sort_index()
    market_cap = _read("market_cap")
    adv60 = _read("adv60")
    close = _read("close")

    common_index = membership.index.intersection(market_cap.index).intersection(adv60.index).intersection(close.index)
    common_cols = sorted(set(membership.columns) & set(market_cap.columns) & set(adv60.columns) & set(close.columns))
    membership = membership.reindex(index=common_index, columns=common_cols).fillna(False).astype(bool)
    market_cap = market_cap.reindex(index=common_index, columns=common_cols)
    adv60 = adv60.reindex(index=common_index, columns=common_cols)
    close = close.reindex(index=common_index, columns=common_cols)

    tradable = membership & close.gt(0) & close.notna()
    smallcap = tradable & market_cap.ge(100_000_000.0) & market_cap.lt(500_000_000.0)
    adv_rank = adv60.where(tradable).rank(axis=1, ascending=False, method="first")
    top1000 = adv_rank.le(1000).fillna(False)
    top3000 = adv_rank.le(3000).fillna(False)

    outputs = {
        "MCAP_100M_500M_PITV2.parquet": smallcap.astype(bool),
        "TOP1000_ADV60_PITV2.parquet": top1000.astype(bool),
        "TOP3000_ADV60_PITV2.parquet": top3000.astype(bool),
    }
    summaries = {}
    for name, df in outputs.items():
        path = OUT_DIR / name
        df.to_parquet(path)
        summaries[name] = {
            "path": str(path.relative_to(ROOT)),
            "shape": list(df.shape),
            "first_date": str(df.index.min()),
            "last_date": str(df.index.max()),
            "last_members": int(df.tail(1).sum(axis=1).iloc[0]),
            "max_members": int(df.sum(axis=1).max()),
            "unique_members": int((df.sum(axis=0) > 0).sum()),
        }

    manifest = {
        "source_membership": str(PIT_UNI.relative_to(ROOT)),
        "source_matrices": str(PIT_MAT.relative_to(ROOT)),
        "construction": {
            "tradable": "PIT membership at t and close[t] > 0",
            "smallcap": "100MM <= market_cap[t] < 500MM",
            "top1000": "top 1000 by adv60[t] among tradable names",
            "top3000": "top 3000 by adv60[t] among tradable names",
        },
        "outputs": summaries,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
