"""
ib_universe_borrow_probe.py — query IB for shortable status on every ticker
in the active midcap universe and dump a CSV.

Outputs `experiments/results/borrow_probe_<UNIVERSE>_<DATE>.csv` with columns:
  symbol, shortable (bool), shares_avail, shortable_indicator,
  fee_rate, status, kind, ts.

The shortable_indicator from IB:
  > 2.5  EASY      (deep borrow pool)
  1.5-2.5 LIMITED  (some availability, may incur fees)
  < 1.5  HARD      (locate likely required)
  None / 0 shares  NO_DATA / HARD_TO_BORROW

Run from the factor-alpha-platform root:
    python experiments/ib_universe_borrow_probe.py
    python experiments/ib_universe_borrow_probe.py --universe TOP2000TOP3000
    python experiments/ib_universe_borrow_probe.py --window-days 60
    python experiments/ib_universe_borrow_probe.py --client-id 12
"""
from __future__ import annotations
import argparse, csv, datetime as dt, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import pandas as pd

IB_HOST   = "127.0.0.1"
IB_PORT   = 4002
CLIENT_ID = 11

UNIVERSES_DIR = ROOT / "data/fmp_cache/universes"


def select_tickers(universe: str, window_days: int) -> list[str]:
    path = UNIVERSES_DIR / f"{universe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"universe parquet not found: {path}")
    uni = pd.read_parquet(path)
    if window_days <= 0:
        active = uni.iloc[-1]
        tickers = sorted(active[active].index.tolist())
        scope = f"active on {uni.index[-1].date()}"
    else:
        tail = uni.tail(window_days)
        active = tail.any(axis=0)
        tickers = sorted(active[active].index.tolist())
        scope = f"in last {window_days} bars (since {tail.index[0].date()})"
    print(f"[{universe}] {len(tickers)} tickers {scope}", flush=True)
    return tickers


def load_latest_close(tickers: list[str]) -> dict[str, float]:
    """Latest close from FMP cache. Stale by 1 day but adequate for $-shortable."""
    p = ROOT / "data/fmp_cache/matrices/close.parquet"
    df = pd.read_parquet(p)
    last = df.tail(5).ffill().iloc[-1]   # forward-fill last 5 trading days then take latest
    last_date = df.index[-1].date()
    print(f"  [price] FMP close as of {last_date} (forward-filled over 5 prior bars)",
          flush=True)
    out = {}
    for t in tickers:
        if t in last.index:
            v = last[t]
            if pd.notna(v) and v > 0:
                out[t] = float(v)
    print(f"  [price] {len(out)}/{len(tickers)} tickers have a close", flush=True)
    return out


def probe_borrow(ib, tickers: list[str], batch_size: int = 50, settle_sec: float = 6.0):
    from ib_insync import Stock
    out = []
    n = len(tickers)
    print(f"  Probing {n} tickers (batch={batch_size})...", flush=True)

    # Switch to delayed data so shortableShares populates without realtime entitlement
    try:
        ib.reqMarketDataType(3)
        print(f"  [data] market_data_type=3 (delayed)", flush=True)
    except Exception as e:
        print(f"  [data] reqMarketDataType(3) failed: {e}", flush=True)

    t_total_start = time.time()
    for batch_idx, i in enumerate(range(0, n, batch_size)):
        t_b = time.time()
        batch = tickers[i:i + batch_size]
        contracts = []
        for sym in batch:
            try:
                c = Stock(sym, "SMART", "USD")
                ib.qualifyContracts(c)
                contracts.append((sym, c))
            except Exception as e:
                out.append({
                    "symbol": sym, "shortable": False, "shortableShares": 0,
                    "shortable_indicator": None, "fee_rate": None,
                    "status": f"UNQUALIFIED: {type(e).__name__}",
                })

        reqs = []
        for sym, c in contracts:
            try:
                td = ib.reqMktData(c, genericTickList="236")
                reqs.append((sym, c, td))
            except Exception as e:
                out.append({
                    "symbol": sym, "shortable": False, "shortableShares": 0,
                    "shortable_indicator": None, "fee_rate": None,
                    "status": f"REQ_FAILED: {type(e).__name__}",
                })

        # Let quotes arrive
        ib.sleep(settle_sec)

        for sym, c, td in reqs:
            shares = getattr(td, "shortableShares", None)
            ind    = getattr(td, "shortable", None)
            fee    = getattr(td, "shortFeeRate", None)
            if shares is not None and shares > 0:
                if ind is not None:
                    if ind > 2.5:
                        status = "EASY"
                    elif ind > 1.5:
                        status = "LIMITED"
                    else:
                        status = "HARD"
                else:
                    status = "AVAILABLE"
                row = {
                    "symbol": sym, "shortable": True,
                    "shortableShares": int(shares),
                    "shortable_indicator": float(ind) if ind is not None else None,
                    "fee_rate": float(fee) if fee is not None else None,
                    "status": status,
                }
            else:
                row = {
                    "symbol": sym, "shortable": False,
                    "shortableShares": 0,
                    "shortable_indicator": float(ind) if ind is not None else None,
                    "fee_rate": None,
                    "status": "NO_DATA" if shares is None else "HARD_TO_BORROW",
                }
            out.append(row)
            ib.cancelMktData(c)

        t_elapsed = time.time() - t_b
        n_done = i + len(batch)
        n_short = sum(1 for r in out if r["shortable"])
        print(f"  batch {batch_idx+1}: {n_done}/{n} done  "
              f"({n_short} shortable so far)  [{t_elapsed:.1f}s]", flush=True)

    print(f"  [done] total {time.time()-t_total_start:.1f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="TOP2000TOP3000")
    ap.add_argument("--window-days", type=int, default=0,
                    help="Pull tickers in universe over last N bars; 0 = today only.")
    ap.add_argument("--client-id", type=int, default=CLIENT_ID)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--settle-sec", type=float, default=6.0)
    args = ap.parse_args()

    tickers = select_tickers(args.universe, args.window_days)
    if not tickers:
        print("no tickers selected, exiting", flush=True)
        return

    from ib_insync import IB
    ib = IB()
    print(f"Connecting to IB Gateway at {IB_HOST}:{IB_PORT} (clientId={args.client_id})...",
          flush=True)
    ib.connect(IB_HOST, IB_PORT, clientId=args.client_id, timeout=15)
    accts = ib.managedAccounts()
    print(f"  Connected. Accounts={accts}  Paper={'DU' in str(accts)}", flush=True)

    try:
        rows = probe_borrow(ib, tickers,
                            batch_size=args.batch_size, settle_sec=args.settle_sec)
    finally:
        ib.disconnect()
        print(f"  Disconnected.", flush=True)

    # Attach last close + dollar capacity
    closes = load_latest_close(tickers)
    for r in rows:
        px = closes.get(r["symbol"])
        r["last_close"] = round(px, 4) if px else None
        if px and r.get("shortableShares"):
            r["shortable_dollars"] = round(px * r["shortableShares"], 2)
        else:
            r["shortable_dollars"] = 0.0 if not r.get("shortable") else None

    # Stamp + save
    today = dt.date.today().isoformat()
    out_dir = ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"borrow_probe_{args.universe}_{today}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["symbol", "shortable", "shortableShares",
                        "last_close", "shortable_dollars",
                        "shortable_indicator", "fee_rate", "status"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[saved] {out_path.relative_to(ROOT)}  ({len(rows)} rows)", flush=True)

    # Quick summary
    n = len(rows)
    n_short  = sum(1 for r in rows if r["shortable"])
    n_easy   = sum(1 for r in rows if r["status"] == "EASY")
    n_lim    = sum(1 for r in rows if r["status"] == "LIMITED")
    n_hard   = sum(1 for r in rows if r["status"] == "HARD")
    n_avail  = sum(1 for r in rows if r["status"] == "AVAILABLE")
    n_htb    = sum(1 for r in rows if r["status"] == "HARD_TO_BORROW")
    n_nodat  = sum(1 for r in rows if r["status"] == "NO_DATA")
    n_unq    = sum(1 for r in rows if "UNQUALIFIED" in r["status"]
                   or "REQ_FAILED" in r["status"])
    print(f"\n  total:       {n}", flush=True)
    print(f"  shortable:   {n_short}  ({n_short/n*100:.1f}%)", flush=True)
    print(f"    EASY:      {n_easy}", flush=True)
    print(f"    LIMITED:   {n_lim}", flush=True)
    print(f"    HARD:      {n_hard}", flush=True)
    print(f"    AVAILABLE (no indicator): {n_avail}", flush=True)
    print(f"  HTB:         {n_htb}", flush=True)
    print(f"  NO_DATA:     {n_nodat}", flush=True)
    print(f"  unqualified: {n_unq}", flush=True)


if __name__ == "__main__":
    main()
