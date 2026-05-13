"""Diagnose why specific MOC orders didn't fill.

For a given date, reads `prod/logs/trades/trade_<DATE>.json` (intent) and
`prod/logs/reconciliation/equity_<DATE>.json` (fills), then connects to IB and
asks for completed orders + open orders. For each unfilled permId, reports
the IB-side `status`, any `whyHeld`, and the contract details. Also captures
errors emitted by ib_insync.

Usage:
    python tools/diag_unfilled_orders.py --date 2026-05-11
    python tools/diag_unfilled_orders.py --date 2026-05-08 --date 2026-05-11
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")


def diag_date(date: str, host: str, port: int, client_id: int) -> int:
    trade_path = ROOT / "prod/logs/trades" / f"trade_{date}.json"
    recon_path = ROOT / "prod/logs/reconciliation" / f"equity_{date}.json"
    if not trade_path.exists():
        print(f"[{date}] no intent log at {trade_path}", flush=True)
        return 1
    intent = json.loads(trade_path.read_text(encoding="utf-8"))
    fills_by_perm: set[int] = set()
    recon_msg = "(no recon file)"
    if recon_path.exists():
        recon = json.loads(recon_path.read_text(encoding="utf-8"))
        recon_msg = (f"recon status={recon.get('status')}  "
                     f"filled={recon.get('n_orders_with_fills')}/{recon.get('n_orders_intent')}  "
                     f"qty_rate={recon.get('fill_rate_qty')}")
        for f in recon.get("fills") or []:
            pid = f.get("perm_id")
            if pid:
                fills_by_perm.add(int(pid))

    # Build the unfilled set from intent
    intent_orders = intent.get("order_records") or []
    unfilled: list[dict] = []
    for r in intent_orders:
        pid = r.get("perm_id")
        if not pid:
            unfilled.append({"symbol": r.get("symbol"), "action": r.get("action"),
                             "quantity": r.get("quantity"), "perm_id": None,
                             "order_id": r.get("order_id"),
                             "submission_status": r.get("status"),
                             "reason": "intent record has no perm_id (probably PendingSubmit @ snapshot)"})
            continue
        if int(pid) not in fills_by_perm:
            unfilled.append({"symbol": r.get("symbol"), "action": r.get("action"),
                             "quantity": r.get("quantity"), "perm_id": int(pid),
                             "order_id": r.get("order_id"),
                             "submission_status": r.get("status")})

    print(f"\n=== {date} ===")
    print(f"  intent orders: {len(intent_orders)}")
    print(f"  {recon_msg}")
    print(f"  unfilled per intent vs recon: {len(unfilled)}")
    if not unfilled:
        print("  (nothing to diagnose)")
        return 0

    print("\n  intent-side rows that had no matching fill:")
    for u in unfilled:
        print(f"    {u['symbol']:>6} {u['action']:>4} {u['quantity']:>7}  perm_id={u['perm_id']}  "
              f"order_id={u['order_id']}  submission_status={u['submission_status']}")

    # Connect to IB and pull completedOrders + openOrders, then match
    from ib_insync import IB

    ib = IB()
    errors: list[str] = []
    ib.errorEvent += lambda reqId, code, msg, contract: errors.append(
        f"reqId={reqId} code={code} contract={getattr(contract,'symbol',None)} msg={msg}"
    )

    try:
        ib.connect(host, port, clientId=client_id, timeout=15)
        ib.reqAllOpenOrders()
        ib.sleep(1.5)
        open_trades = list(ib.openTrades() or [])
        completed = list(ib.reqCompletedOrders(apiOnly=False) or [])
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass

    # Index by permId
    by_perm: dict[int, dict] = {}
    for tr in (open_trades + completed):
        try:
            pid = int(getattr(tr.order, "permId", 0) or 0)
        except Exception:
            pid = 0
        if pid:
            os_ = tr.orderStatus
            ctr = tr.contract
            by_perm[pid] = {
                "status": getattr(os_, "status", None),
                "filled": float(getattr(os_, "filled", 0) or 0),
                "remaining": float(getattr(os_, "remaining", 0) or 0),
                "avgFillPrice": float(getattr(os_, "avgFillPrice", 0) or 0),
                "whyHeld": getattr(os_, "whyHeld", None),
                "lastError": (tr.log[-1].errorCode if getattr(tr, "log", None) and tr.log else None),
                "contract_symbol": getattr(ctr, "symbol", None),
                "exchange": getattr(ctr, "exchange", None),
                "secType": getattr(ctr, "secType", None),
            }

    print(f"\n  IB completedOrders+openTrades indexed by permId: {len(by_perm)} entries")
    print(f"  IB-side records for our unfilled perm_ids:")
    for u in unfilled:
        pid = u.get("perm_id")
        if not pid:
            print(f"    {u['symbol']:>6}: no perm_id captured at submission -> can't look up")
            continue
        rec = by_perm.get(int(pid))
        if rec is None:
            print(f"    {u['symbol']:>6}: NOT FOUND in IB completedOrders. Likely never reached IB.")
            continue
        print(f"    {u['symbol']:>6}: status={rec['status']:<12}  filled={rec['filled']}/{u['quantity']}  "
              f"avgPx={rec['avgFillPrice']}  whyHeld={rec['whyHeld']}  lastErr={rec['lastError']}")

    if errors:
        print("\n  ib_insync errorEvent log (during this session):")
        for e in errors:
            print(f"    {e}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--date", action="append", required=True, help="YYYY-MM-DD; repeat for multiple")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=4002)
    p.add_argument("--client-id", type=int, default=32)
    args = p.parse_args()
    rc = 0
    for d in args.date:
        rc = max(rc, diag_date(d, args.host, args.port, args.client_id))
    return rc


if __name__ == "__main__":
    sys.exit(main())
