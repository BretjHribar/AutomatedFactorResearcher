"""Classify every ticker in a universe parquet by IB-reported stockType.

IB's `reqContractDetails` returns `stockType` ∈ {COMMON, PREFERRED, ADR, ETF,
RIGHT, ...}. We persist a `ticker -> stockType` mapping at
`data/fmp_cache/classifications/preferred.parquet` so the moc_trader and the
shadow-signal-ex-preferreds asset can filter out preferreds before order
generation (they can't be MOC'd — see the SAT/HTFC/XOMAP/SAJ cancellations on
5/8 and 5/11).

The classification is stable; rerun monthly or whenever the universe
membership rotates significantly. Incremental: only queries tickers not
already in the cache.

Usage:
    python tools/classify_universe_preferred.py
    python tools/classify_universe_preferred.py --universe MCAP_100M_500M
    python tools/classify_universe_preferred.py --refresh   # query all again
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--universe", default="MCAP_100M_500M")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=4002)
    p.add_argument("--client-id", type=int, default=34)
    p.add_argument("--refresh", action="store_true",
                   help="Re-query all tickers (default: only ones missing from the cache)")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after this many queries (for smoke tests)")
    args = p.parse_args()

    # Resolve current active universe tickers.
    uni_path = ROOT / f"data/fmp_cache/universes/{args.universe}.parquet"
    if not uni_path.exists():
        print(f"[err] universe parquet not found: {uni_path}", flush=True)
        return 1
    uni = pd.read_parquet(uni_path)
    # Most recent active membership row
    last = uni.iloc[-1].fillna(False).astype(bool)
    tickers = sorted(last.index[last].tolist())
    print(f"[uni] {args.universe}: {len(tickers)} active tickers as of {uni.index[-1].date()}",
          flush=True)

    out_dir = ROOT / "data/fmp_cache/classifications"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "preferred.parquet"
    if cache_path.exists() and not args.refresh:
        existing = pd.read_parquet(cache_path)
    else:
        existing = pd.DataFrame(columns=["ticker", "stockType", "longName", "primaryExchange",
                                          "is_preferred", "queried_at"])
    if "ticker" in existing.columns:
        cached_tickers = set(existing["ticker"].astype(str))
    else:
        cached_tickers = set()
    todo = [t for t in tickers if t not in cached_tickers]
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print(f"[cache] {len(cached_tickers)} tickers already classified; nothing to do "
              f"(use --refresh to re-query)", flush=True)
        _print_summary(existing)
        return 0
    print(f"[run] querying {len(todo)} new tickers (cache had {len(cached_tickers)})",
          flush=True)

    from ib_insync import IB, Stock

    ib = IB()
    rows: list[dict] = []
    if not args.refresh:
        rows.extend(existing.to_dict("records"))
    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=15)
        for i, sym in enumerate(todo, 1):
            try:
                details = ib.reqContractDetails(Stock(sym, "SMART", "USD"))
            except Exception as exc:
                print(f"  [err] {sym}: {type(exc).__name__}: {exc}", flush=True)
                continue
            if not details:
                rows.append({"ticker": sym, "stockType": None, "longName": None,
                             "primaryExchange": None, "is_preferred": None,
                             "queried_at": pd.Timestamp.utcnow().isoformat()})
                continue
            d = details[0]
            stock_type = d.stockType
            is_pref = (str(stock_type).upper() == "PREFERRED")
            rows.append({
                "ticker": sym,
                "stockType": stock_type,
                "longName": d.longName,
                "primaryExchange": d.contract.primaryExchange,
                "is_preferred": is_pref,
                "queried_at": pd.Timestamp.utcnow().isoformat(),
            })
            if i % 20 == 0 or i == len(todo):
                print(f"  [{i:>3}/{len(todo)}] last={sym} type={stock_type}", flush=True)
            ib.sleep(0.05)  # pacing
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass

    df = pd.DataFrame(rows).drop_duplicates("ticker", keep="last").sort_values("ticker")
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(cache_path)
    print(f"\n[ok] wrote {cache_path}  total={len(df)}", flush=True)
    _print_summary(df)
    return 0


def _print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        return
    pref = df[df["is_preferred"] == True]  # noqa: E712
    print(f"\nstockType breakdown:")
    print(df["stockType"].fillna("UNKNOWN").value_counts().to_string())
    print(f"\npreferreds in cache: {len(pref)}")
    if len(pref):
        print(pref[["ticker", "longName", "primaryExchange"]].to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
