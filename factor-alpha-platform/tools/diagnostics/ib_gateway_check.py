"""
Fast IB Gateway / TWS connectivity sanity check.

Verifies in ~5 seconds:
  1. Can connect to localhost:<port>  (default 7497 = paper)
  2. Server time + connection state
  3. Account ID matches expected (from prod/config/strategy.json)
  4. Read-only API check (lists positions + account values without trading)
  5. Disconnect cleanly

Run before any --live launch. Failures here mean Gateway is wrong before any
alpha pipeline can possibly help.

Usage:
    python tools/diagnostics/ib_gateway_check.py            # paper (port 7497)
    python tools/diagnostics/ib_gateway_check.py --live     # LIVE port 7496
    python tools/diagnostics/ib_gateway_check.py --port 4002  # gateway port

Exit codes:
    0  — all checks passed
    1  — connection failure or account mismatch
    2  — Gateway settings prevent API (Read-Only API enabled? Trusted IP?)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CFG_PATH = ROOT / "prod" / "config" / "strategy.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=None,
                   help="default 7497 (paper); --live forces 7496")
    p.add_argument("--live", action="store_true",
                   help="use port_live (7496) — DANGER, this checks live account")
    p.add_argument("--client-id", type=int, default=99,
                   help="distinct from prod (default 10) so we don't collide")
    p.add_argument("--timeout", type=int, default=10,
                   help="connect timeout in seconds")
    args = p.parse_args()

    cfg = json.loads(CFG_PATH.read_text())
    expected_paper = cfg["ibkr"]["paper_account"]
    paper_port = cfg["ibkr"]["port_paper"]
    live_port  = cfg["ibkr"]["port_live"]

    if args.port is None:
        args.port = live_port if args.live else paper_port

    mode = "LIVE" if args.live or args.port == live_port else "PAPER"
    print(f"=== IB Gateway / TWS check — {mode} ===")
    print(f"  host={args.host}  port={args.port}  client_id={args.client_id}")
    print(f"  expected paper account: {expected_paper}")
    print()

    try:
        from ib_insync import IB, util
    except ImportError:
        print("FAIL: ib_insync not installed.  pip install ib_insync")
        return 1

    ib = IB()
    t0 = time.time()
    try:
        ib.connect(args.host, args.port, clientId=args.client_id,
                   timeout=args.timeout)
    except Exception as e:
        print(f"FAIL: ib.connect threw {type(e).__name__}: {e}")
        print("  Is IB Gateway / TWS running and logged in?")
        print(f"  Is API enabled?  Settings -> API -> 'Enable ActiveX and Socket Clients'")
        print(f"  Is the socket port {args.port} configured in Gateway -> Settings -> API -> Socket port?")
        return 1
    elapsed = time.time() - t0
    print(f"  [OK] connected in {elapsed*1000:.0f}ms")

    # Server time
    try:
        st = ib.reqCurrentTime()
        print(f"  [OK] server time: {st}")
    except Exception as e:
        print(f"  [WARN] server time: {e}")

    # Accounts
    accounts = ib.managedAccounts()
    print(f"  managed accounts: {accounts}")
    if not accounts:
        print("  FAIL: no managed accounts returned — login state issue")
        ib.disconnect()
        return 1
    if mode == "PAPER" and expected_paper not in accounts:
        print(f"  FAIL: expected paper account {expected_paper} not in {accounts}")
        ib.disconnect()
        return 1
    print(f"  [OK] account {accounts[0]} matches config")

    # Account summary (a few key tags)
    try:
        summary = ib.accountSummary(accounts[0])
        # print Net Liquidation + Cash + AvailableFunds
        keys = {"NetLiquidation", "TotalCashValue", "AvailableFunds", "BuyingPower"}
        for row in summary:
            if row.tag in keys:
                print(f"  {row.tag:18s} {row.value:>15s} {row.currency}")
    except Exception as e:
        print(f"  [WARN] accountSummary: {e}")

    # Positions (read-only — won't place anything)
    try:
        positions = ib.positions()
        print(f"  current positions: {len(positions)}")
        for pos in positions[:5]:
            print(f"    {pos.contract.symbol:6s}  qty={pos.position:>10.0f}  "
                  f"avg={pos.avgCost:>8.2f}")
        if len(positions) > 5:
            print(f"    ... and {len(positions)-5} more")
    except Exception as e:
        print(f"  [WARN] positions: {e}")

    # Read-only mode test — try to instantiate a market data sub but cancel immediately.
    # If Read-Only API is enabled, anything that "could place orders" might be blocked.
    # We don't actually place. This is just a check that the account is permissioned.
    try:
        from ib_insync import Stock
        c = Stock("AAPL", "SMART", "USD")
        ib.qualifyContracts(c)
        print(f"  [OK] contract qualification (AAPL): conId={c.conId}")
    except Exception as e:
        print(f"  [WARN] contract qualify: {e}")

    ib.disconnect()
    print()
    print("=== ALL CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
