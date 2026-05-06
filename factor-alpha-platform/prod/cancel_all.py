"""Force cancel ALL orders on IB regardless of clientId.

Reads port from prod/config/strategy.json (port_paper). Pass --port to override.
"""
from ib_insync import IB
import argparse, json, logging
import os
from pathlib import Path

CFG = json.loads((Path(__file__).parent / "config" / "strategy.json").read_text())
DEFAULT_IB_HOST = os.environ.get("IB_HOST", CFG["ibkr"]["host"])
DEFAULT_IB_PORT = int(os.environ.get("IB_PORT_PAPER", os.environ.get("IB_PORT", CFG["ibkr"]["port_paper"])))

p = argparse.ArgumentParser()
p.add_argument("--host", default=DEFAULT_IB_HOST)
p.add_argument("--port", type=int, default=DEFAULT_IB_PORT)
p.add_argument("--client-id", type=int, default=CFG["ibkr"]["client_id"])
args = p.parse_args()

logging.basicConfig(level=logging.INFO, format="%(message)s")

ib = IB()
ib.connect(args.host, args.port, clientId=args.client_id)
print(f"Connected to {args.host}:{args.port}, clientId={args.client_id}")
ib.sleep(2)

trades = ib.openTrades()
print(f"Open trades: {len(trades)}")

# Global cancel - kills everything
print("Sending reqGlobalCancel()...")
ib.reqGlobalCancel()
ib.sleep(5)

# Check again
trades2 = ib.openTrades()
print(f"After cancel: {len(trades2)} remaining")

# Also try individual cancels if any remain
if trades2:
    for t in trades2:
        print(f"  Force-cancelling {t.contract.symbol} {t.order.orderId}...")
        ib.cancelOrder(t.order)
    ib.sleep(3)
    trades3 = ib.openTrades()
    print(f"After individual cancel: {len(trades3)} remaining")

ib.disconnect()
print("Done.")
