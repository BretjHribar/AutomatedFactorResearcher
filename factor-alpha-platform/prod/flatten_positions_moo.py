"""
flatten_positions_moo.py — submit Market-On-Open orders to flatten every
position in the IB paper account.

Behavior:
  1. Connect to IB Gateway paper (port 4002, clientId 13 to avoid collisions).
  2. Pull current positions via ib.positions().
  3. For each non-zero position, submit an MOO order in the opposite direction:
       SELL  for longs   (action='SELL', orderType='MKT', tif='OPG')
       BUY   for shorts  (action='BUY',  orderType='MKT', tif='OPG')
  4. tif='OPG' = "On the Open" — order participates in the opening auction
     at the next regular session open (9:30 ET).
  5. Log every order to prod/logs/trades/flatten_<DATE>.json.

NYSE is closed for the day if you run this after 16:00 ET — orders queue
overnight and match in tomorrow morning's opening auction. Paper account
auction-fill caveat (per fill_vs_close findings): IB paper does NOT
participate in the actual exchange auction; fills happen at the last
printed trade just before/after the open.

Usage:
    python prod/flatten_positions_moo.py            # dry-run (default)
    python prod/flatten_positions_moo.py --live     # actually submit
"""
from __future__ import annotations
import argparse, datetime as dt, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

CFG = json.loads((ROOT / "prod" / "config" / "strategy.json").read_text())
IB_HOST   = os.environ.get("IB_HOST", CFG["ibkr"]["host"])
IB_PORT   = int(os.environ.get("IB_PORT_PAPER", os.environ.get("IB_PORT", CFG["ibkr"]["port_paper"])))
CLIENT_ID = int(os.environ.get("IB_CLIENT_ID_FLATTEN", "13"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="Actually submit orders (default: dry-run, no orders sent)")
    args = ap.parse_args()
    mode = "LIVE" if args.live else "DRY-RUN"

    from ib_insync import IB, Stock, Order

    ib = IB()
    print(f"Connecting to IB Gateway at {IB_HOST}:{IB_PORT} (clientId={CLIENT_ID})...",
          flush=True)
    ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID, timeout=10)
    accts = ib.managedAccounts()
    is_paper = "DU" in str(accts)
    print(f"  Connected. Accounts={accts}  Paper={is_paper}  Mode={mode}", flush=True)
    if not is_paper and args.live:
        print("REFUSING to submit live flatten orders against a NON-paper account.",
              flush=True)
        ib.disconnect()
        return

    # Pull positions
    print("\nPulling current positions...", flush=True)
    positions = []
    for p in ib.positions():
        qty = int(p.position)
        if qty == 0:
            continue
        positions.append({
            "symbol": p.contract.symbol,
            "shares": qty,
            "side": "LONG" if qty > 0 else "SHORT",
            "avg_cost": float(p.avgCost) if p.avgCost else None,
            "exchange": p.contract.primaryExchange or p.contract.exchange,
        })
    positions.sort(key=lambda r: r["symbol"])
    print(f"  {len(positions)} non-zero positions held", flush=True)

    if not positions:
        print("\nNo positions to flatten.", flush=True)
        ib.disconnect()
        return

    # Build flatten orders
    n_long  = sum(1 for p in positions if p["shares"] > 0)
    n_short = sum(1 for p in positions if p["shares"] < 0)
    gross_shares = sum(abs(p["shares"]) for p in positions)
    print(f"  Longs to sell : {n_long:>3d}", flush=True)
    print(f"  Shorts to buy : {n_short:>3d}", flush=True)
    print(f"  Gross shares  : {gross_shares:>6d}", flush=True)

    # Submit (or report)
    print(f"\n{'Submitting' if args.live else 'WOULD SUBMIT'} MOO orders ({mode}):",
          flush=True)
    print(f"  {'sym':6s} {'side':5s} {'shares':>7s}  {'order':10s}", flush=True)
    print("  " + "-" * 36, flush=True)
    order_records = []
    for p in positions:
        action = "SELL" if p["shares"] > 0 else "BUY"
        qty = abs(p["shares"])
        record = {
            "symbol": p["symbol"],
            "starting_position": p["shares"],
            "action": action,
            "quantity": qty,
            "order_type": "MKT",
            "tif": "OPG",
        }
        print(f"  {p['symbol']:6s} {p['side']:5s} {qty:>7d}  {action} MOO",
              flush=True)
        if args.live:
            try:
                contract = Stock(p["symbol"], "SMART", "USD")
                ib.qualifyContracts(contract)
                # MOO = Market order with tif='OPG' (On the OPenGate auction)
                order = Order(action=action, totalQuantity=qty,
                              orderType="MKT", tif="OPG")
                trade = ib.placeOrder(contract, order)
                record["order_id"] = trade.order.orderId
                record["status"]   = "SUBMITTED"
            except Exception as e:
                record["status"] = f"FAILED: {type(e).__name__}: {e}"
                print(f"    ! {p['symbol']} FAILED: {e}", flush=True)
        else:
            record["status"] = "DRY_RUN"
        order_records.append(record)

    ib.disconnect()
    print("\nDisconnected.", flush=True)

    # Log
    out_dir = ROOT / "prod" / "logs" / "trades"
    out_dir.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    out_path = out_dir / f"flatten_{today}.json"
    out_path.write_text(json.dumps({
        "ts": dt.datetime.now().isoformat(),
        "mode": mode,
        "account": list(accts),
        "n_positions": len(positions),
        "starting_positions": positions,
        "orders": order_records,
    }, indent=2, default=str))
    print(f"[saved] {out_path.relative_to(ROOT)}", flush=True)
    print(f"\nMOO orders queue overnight; they match in tomorrow's "
          f"opening auction (~9:30 ET).", flush=True)


if __name__ == "__main__":
    main()
