"""Force cancel ALL orders on IB regardless of clientId."""
from ib_insync import IB
import logging, time

logging.basicConfig(level=logging.INFO, format="%(message)s")

ib = IB()
ib.connect("127.0.0.1", 7497, clientId=10)  # Same clientId that placed them
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
