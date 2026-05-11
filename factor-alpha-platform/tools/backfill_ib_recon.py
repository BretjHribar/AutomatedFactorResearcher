"""One-time backfill of IB execution reconciliation for historical live trades.

Per-day: read prod/logs/trades/trade_<DATE>.json, query IB for executions on
that date, write prod/logs/reconciliation/equity_<DATE>.json. Skips dates that
already have a reconciliation file.

IB retains paper-account executions for ~7 trading days. Use from the host
(direct 127.0.0.1:4002) or inside the container (host.docker.internal:4002)
while the gateway is up.

Usage:
    python tools/backfill_ib_recon.py --dates 2026-05-05 2026-05-08
    python tools/backfill_ib_recon.py --dates 2026-05-05 --host host.docker.internal
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.execution.ib_recon import reconcile_from_ib, write_recon  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dates", nargs="+", required=True, help="YYYY-MM-DD list")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=4002)
    p.add_argument("--client-id", type=int, default=31, help="distinct from prod recon (30)")
    p.add_argument("--force", action="store_true", help="overwrite existing recon files")
    args = p.parse_args()

    trades_dir = ROOT / "prod/logs/trades"
    recon_dir = ROOT / "prod/logs/reconciliation"

    rc = 0
    for date_str in args.dates:
        trade_path = trades_dir / f"trade_{date_str}.json"
        if not trade_path.exists():
            print(f"[skip] {date_str}: no intent log at {trade_path}", flush=True)
            rc = max(rc, 1)
            continue
        recon_path = recon_dir / f"equity_{date_str}.json"
        if recon_path.exists() and not args.force:
            print(f"[skip] {date_str}: {recon_path.name} already exists (use --force to redo)", flush=True)
            continue
        try:
            trade_log = json.loads(trade_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[err]  {date_str}: cannot read intent: {exc}", flush=True)
            rc = max(rc, 1)
            continue
        if str(trade_log.get("mode") or "").lower() != "live":
            print(f"[skip] {date_str}: mode={trade_log.get('mode')!r} (not live)", flush=True)
            continue
        print(f"[run]  {date_str}: querying IB at {args.host}:{args.port} clientId={args.client_id}...", flush=True)
        try:
            result, stale = reconcile_from_ib(
                host=args.host, port=args.port, client_id=args.client_id,
                trade_log=trade_log, date=date_str,
            )
        except Exception as exc:
            print(f"[err]  {date_str}: {type(exc).__name__}: {exc}", flush=True)
            rc = max(rc, 2)
            continue
        out = write_recon(result, recon_dir=recon_dir, stale_unavailable=stale)
        if stale:
            print(f"[stale] {date_str}: IB has no execution history (outside retention). "
                  f"Day kept on the provisional curve. -> {out.name}", flush=True)
        else:
            print(f"[ok]    {date_str}: {result.n_orders_with_fills}/{result.n_orders_intent} orders filled, "
                  f"qty fill-rate {result.fill_rate_qty:.1%}, gross_$={result.gross_filled_dollar:,.0f}, "
                  f"commission=${result.total_commission:,.2f} -> {out.name}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
