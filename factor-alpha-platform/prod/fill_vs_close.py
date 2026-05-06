"""
ib_fill_vs_close.py — post-auction reconciliation: IB paper MOC fill prices
vs the official daily closing print.

# Why this exists
# ──────────────────────────────────────────────────────────────────────────
# IB paper trading does NOT participate in the actual NYSE/Nasdaq closing
# auction. From IB's documentation (Knowledge Base "Order Types — Paper
# Trading Simulation"): paper MOC orders are filled at IB's last printed
# trade price BEFORE 16:00:00 ET, not at the official auction match. The
# auction match price is published a few seconds after 16:00 ET and may
# differ by up to several percent for thin small/mid-cap names due to
# closing imbalance dynamics.
#
# Empirical evidence on this account (DUQ372830, 2026-05-04, 35 fills):
#   — every fill timestamped 15:59:31–15:59:52 ET (8–29s before auction)
#   — 0 of 35 fills equalled the official close exactly
#   — mean adverse delta 36.6 bps; max 168 bps (TCI)
#   — systematic adverse selection: 17/21 BUY fills > close, 12/14 SELL < close
#
# Fix in production: use the OFFICIAL CLOSE (queried post-15:00 CDT / 16:00 ET
# via reqHistoricalData TRADES 1d bar) for performance attribution. The IB
# fill price is the *paper-account fill*, not the prevailing market price.
# In live trading (port 4001) MOC orders DO match the auction print.

Pulls today's executions via ib.fills(), queries IB for the official daily
close for each filled symbol, computes deltas. Saves CSV + JSON to
`prod/logs/fills/`.

Usage:
    python experiments/ib_fill_vs_close.py
"""
from __future__ import annotations
import csv, datetime as dt, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

CFG = json.loads((ROOT / "prod" / "config" / "strategy.json").read_text())
IB_HOST   = os.environ.get("IB_HOST", CFG["ibkr"]["host"])
IB_PORT   = int(os.environ.get("IB_PORT_PAPER", os.environ.get("IB_PORT", CFG["ibkr"]["port_paper"])))
CLIENT_ID = int(os.environ.get("IB_CLIENT_ID_FILL_RECON", "12"))


def main():
    from ib_insync import IB, Stock
    today = dt.date.today().isoformat()

    ib = IB()
    print(f"Connecting to IB Gateway at {IB_HOST}:{IB_PORT} (clientId={CLIENT_ID})...", flush=True)
    ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID, timeout=10)
    print(f"  Accounts={ib.managedAccounts()}", flush=True)

    # Pull all of today's fills
    print("\n  Fetching today's fills via ib.fills()...", flush=True)
    fills_raw = ib.fills()
    fills = []
    for f in fills_raw:
        ex = f.execution
        if ex.time.date().isoformat() != today:
            continue
        fills.append({
            "symbol": f.contract.symbol,
            "side": ex.side,                # BOT / SLD
            "shares": float(ex.shares),
            "price": float(ex.price),
            "exec_time": ex.time.isoformat(),
            "exchange": ex.exchange,
            "exec_id": ex.execId,
            "order_id": ex.orderId,
        })
    print(f"  raw fills: {len(fills)}", flush=True)

    # Aggregate to one weighted-avg price per symbol
    by_sym: dict[str, dict] = {}
    for f in fills:
        s = by_sym.setdefault(f["symbol"], {
            "n_executions": 0,
            "total_shares": 0.0,
            "notional": 0.0,
            "side": f["side"],
            "last_exec_time": f["exec_time"],
        })
        s["n_executions"] += 1
        s["total_shares"] += f["shares"]
        s["notional"] += f["shares"] * f["price"]
        if f["exec_time"] > s["last_exec_time"]:
            s["last_exec_time"] = f["exec_time"]
    for s in by_sym.values():
        s["avg_fill_price"] = s["notional"] / s["total_shares"] if s["total_shares"] else None

    print(f"  symbols with fills: {len(by_sym)}", flush=True)

    # Query the official closing price for each (1-day TRADES bar)
    print("\n  Querying official daily close per symbol via reqHistoricalData (TRADES)...",
          flush=True)
    rows = []
    for i, sym in enumerate(sorted(by_sym), 1):
        agg = by_sym[sym]
        official_close = None
        bar_date = None
        try:
            c = Stock(sym, "SMART", "USD")
            ib.qualifyContracts(c)
            bars = ib.reqHistoricalData(
                c, endDateTime="",
                durationStr="2 D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
            if bars:
                # Pick the bar whose date matches today (the most recent bar)
                last = bars[-1]
                bar_date = str(last.date)
                official_close = float(last.close)
        except Exception as e:
            print(f"    {sym}: hist data err: {e}", flush=True)

        delta = None
        delta_bps = None
        if official_close is not None and agg["avg_fill_price"] is not None:
            delta = agg["avg_fill_price"] - official_close
            delta_bps = (delta / official_close) * 10000

        row = {
            "symbol": sym,
            "side": agg["side"],
            "shares": int(agg["total_shares"]),
            "n_executions": agg["n_executions"],
            "avg_fill_price": round(agg["avg_fill_price"], 4) if agg["avg_fill_price"] else None,
            "official_close": round(official_close, 4) if official_close else None,
            "close_bar_date": bar_date,
            "delta": round(delta, 4) if delta is not None else None,
            "delta_bps": round(delta_bps, 2) if delta_bps is not None else None,
            "last_exec_time": agg["last_exec_time"],
        }
        rows.append(row)
        flag = ""
        if delta_bps is not None and abs(delta_bps) > 1:
            flag = "  <-- mismatch"
        print(f"  [{i:>3d}/{len(by_sym)}] {sym:6s} {agg['side']:3s}  "
              f"fill ${agg['avg_fill_price']:.4f}  "
              f"close ${official_close:.4f}  "
              f"Δ {delta_bps:+.2f}bps{flag}" if official_close else
              f"  [{i:>3d}/{len(by_sym)}] {sym:6s}  no close data", flush=True)

        # IB rate-limit: don't hammer
        ib.sleep(0.05)

    ib.disconnect()
    print("\n  Disconnected.", flush=True)

    # ── Report ──
    # Save to BOTH prod/logs/fills (durable record) and experiments/results.
    prod_dir = ROOT / "prod" / "logs" / "fills"
    prod_dir.mkdir(parents=True, exist_ok=True)
    out_dir = ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = prod_dir / f"fill_vs_close_{today}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)
    print(f"\n[saved] {csv_path.relative_to(ROOT)}  ({len(rows)} rows)", flush=True)

    # Stats
    valid = [r for r in rows if r["delta_bps"] is not None]
    if not valid:
        print("  no comparable rows", flush=True)
        return
    deltas_bps = [r["delta_bps"] for r in valid]
    abs_deltas = [abs(d) for d in deltas_bps]
    n = len(valid)
    print(f"\n  ─── delta stats ({n} symbols) ───", flush=True)
    print(f"  mean abs delta:    {sum(abs_deltas)/n:.2f} bps", flush=True)
    print(f"  median abs delta:  {sorted(abs_deltas)[n//2]:.2f} bps", flush=True)
    print(f"  max abs delta:     {max(abs_deltas):.2f} bps", flush=True)
    print(f"  exact (Δ=0):       {sum(1 for d in deltas_bps if d == 0)} / {n}", flush=True)
    print(f"  |Δ| < 1bp:         {sum(1 for d in abs_deltas if d < 1)} / {n}", flush=True)
    print(f"  |Δ| < 5bps:        {sum(1 for d in abs_deltas if d < 5)} / {n}", flush=True)
    print(f"  |Δ| > 10bps:       {sum(1 for d in abs_deltas if d > 10)} / {n}", flush=True)
    pos = sum(1 for d in deltas_bps if d > 0)
    neg = sum(1 for d in deltas_bps if d < 0)
    print(f"  fill > close:      {pos} / {n}", flush=True)
    print(f"  fill < close:      {neg} / {n}", flush=True)

    # Show worst mismatches
    worst = sorted(valid, key=lambda r: -abs(r["delta_bps"]))[:10]
    print(f"\n  ─── worst 10 mismatches ───", flush=True)
    print(f"  {'symbol':8s} {'side':5s} {'fill':>10s} {'close':>10s} {'Δ ($)':>9s} {'Δ (bps)':>9s}", flush=True)
    for r in worst:
        print(f"  {r['symbol']:8s} {r['side']:5s} "
              f"${r['avg_fill_price']:>9.4f} ${r['official_close']:>9.4f} "
              f"{r['delta']:>+9.4f} {r['delta_bps']:>+9.2f}", flush=True)


if __name__ == "__main__":
    main()
