"""Quick borrow check on easily-borrowed mega-cap stocks to verify paper trading data."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ib_insync import IB, Stock

EASY_BORROWS = [
    "AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "NVDA", "SPY", "QQQ", "IWM",
    "JPM", "BAC", "XOM", "JNJ", "PG", "KO", "WMT", "DIS", "NFLX", "AMD",
]

ib = IB()
ib.connect("127.0.0.1", 7497, clientId=99)
print(f"Connected. Accounts: {ib.managedAccounts()}")
print(f"\n{'Symbol':<8} {'Shortable':>10} {'Shares':>12} {'Fee Rate':>10} {'Status'}")
print("-" * 55)

for sym in EASY_BORROWS:
    try:
        c = Stock(sym, "SMART", "USD")
        ib.qualifyContracts(c)
        td = ib.reqMktData(c, genericTickList="236")
        ib.sleep(2)

        shares = getattr(td, "shortableShares", None)
        shortable = getattr(td, "shortable", None)
        fee = getattr(td, "shortFeeRate", None)

        sh_str = f"{int(shares):,}" if shares and shares > 0 else "0"
        fee_str = f"{fee:.2f}%" if fee else "N/A"
        s_val = f"{shortable:.1f}" if shortable else "N/A"

        if shares and shares > 0:
            status = "EASY" if (shortable and shortable > 2.5) else "LIMITED"
        else:
            status = "NONE"

        print(f"{sym:<8} {s_val:>10} {sh_str:>12} {fee_str:>10} {status}")
        ib.cancelMktData(c)
    except Exception as e:
        print(f"{sym:<8} ERROR: {e}")

ib.disconnect()
print("\nDone.")
