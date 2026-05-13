"""Pull IB contract details for a list of tickers — useful for diagnosing
why specific MOC orders got cancelled. Particularly checks:
  - secType / stockType (common vs preferred vs other)
  - primaryExchange + market_rules (the closing-auction venue/rules)
  - tradingClass (NMS vs SCM — NASDAQ Capital Market tier matters)
  - longName (manual sanity check)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from ib_insync import IB, Stock  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("tickers", nargs="+")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=4002)
    p.add_argument("--client-id", type=int, default=33)
    args = p.parse_args()

    ib = IB()
    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=15)
        for sym in args.tickers:
            c = Stock(sym, "SMART", "USD")
            details = ib.reqContractDetails(c)
            if not details:
                print(f"{sym}: NO DETAILS RETURNED")
                continue
            for d in details:
                contract = d.contract
                print(f"\n=== {sym} ===")
                print(f"  longName       : {d.longName}")
                print(f"  primaryExchange: {contract.primaryExchange}")
                print(f"  exchange       : {contract.exchange}")
                print(f"  secType        : {contract.secType}")
                print(f"  stockType      : {d.stockType}")
                print(f"  tradingClass   : {contract.tradingClass}")
                print(f"  industry/cat   : {d.industry} / {d.category} / {d.subcategory}")
                print(f"  validExchanges : {d.validExchanges}")
                print(f"  orderTypes head: {(d.orderTypes or '')[:160]}")
                # Does the order types string include 'MOC'?
                ot = (d.orderTypes or "")
                supports_moc = "MOC" in ot
                supports_loc = "LOC" in ot
                print(f"  supports MOC?  : {supports_moc}     supports LOC?: {supports_loc}")
                print(f"  marketRuleIds  : {d.marketRuleIds}")
                print(f"  timeZoneId     : {d.timeZoneId}")
                print(f"  liquidHours    : {d.liquidHours[:80] if d.liquidHours else ''}")
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
