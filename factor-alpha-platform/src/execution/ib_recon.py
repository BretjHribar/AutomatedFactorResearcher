"""Post-close IB MOC reconciliation: pull realized fills + commissions and
write a deterministic per-day record.

The intent record (`prod/logs/trades/trade_<date>.json`) only captures order
*intent* at submission time — every order shows `filled=0` because the JSON is
written before the 16:00 ET MOC auction. This module closes the loop: connect
to IB after the auction, pull `ib.fills()` (or `ib.reqExecutions()` for older
sessions), join on `permId`, aggregate per-symbol, and persist to
`prod/logs/reconciliation/equity_<date>.json`.

The dashboard's "live curve" reads recon files when present (solid line, real
fills) and falls back to the intent-based proxy (dashed line, provisional)
when recon is missing or stale.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FillRow:
    symbol: str
    action: str  # BUY or SELL (IB side)
    perm_id: int | None
    order_id: int | None
    filled_qty: int
    avg_price: float
    gross_dollar: float
    commission: float | None
    realized_pnl: float | None
    fill_count: int


@dataclass(frozen=True)
class ReconResult:
    date: str
    timestamp: str
    n_orders_intent: int
    n_orders_with_fills: int
    n_unfilled: int
    fill_rate_qty: float  # Σ filled_qty / Σ |intent_qty|
    fill_rate_orders: float  # filled_orders / total_orders
    gross_filled_shares: int
    gross_filled_dollar: float
    total_commission: float
    fills: list[FillRow] = field(default_factory=list)
    unfilled: list[dict[str, Any]] = field(default_factory=list)
    extras: list[dict[str, Any]] = field(default_factory=list)  # fills with no matching intent
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fills"] = [asdict(f) if not isinstance(f, dict) else f for f in self.fills]
        return d


def _load_intent(trade_log: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Map perm_id -> order_record for all intent orders. Records without
    perm_id are kept under a synthetic negative key so they still appear in
    the unfilled list."""
    by_perm: dict[int, dict[str, Any]] = {}
    fallback_key = -1
    for rec in trade_log.get("order_records") or []:
        pid = rec.get("perm_id")
        if pid is None or pid == 0:
            by_perm[fallback_key] = rec
            fallback_key -= 1
        else:
            by_perm[int(pid)] = rec
    return by_perm


def reconcile_from_ib(
    *,
    host: str,
    port: int,
    client_id: int,
    trade_log: dict[str, Any],
    date: str,
    fetch_window_start_et: dt.datetime | None = None,
    timeout: float = 12.0,
) -> ReconResult:
    """Connect to IB, fetch executions for `date`, join to `trade_log`'s
    intent records by permId, return a ReconResult.

    Designed for IBKR paper accounts (`DU*`). For a live account the same
    code path works but the caller must guard.
    """
    from ib_insync import IB, ExecutionFilter

    ib = IB()
    started = dt.datetime.now(dt.timezone.utc)
    try:
        ib.connect(host, port, clientId=client_id, timeout=timeout)

        # ExecutionFilter.time is "earliest fill timestamp to retrieve" in
        # YYYYMMDD-HH:MM:SS format (UTC). For a same-day recon we anchor at
        # 14:00 ET (pre-MOC) on `date` so we capture every today fill.
        if fetch_window_start_et is None:
            try:
                anchor = dt.datetime.fromisoformat(f"{date}T14:00:00")
            except Exception:
                anchor = dt.datetime.fromisoformat(f"{date}T00:00:00")
            anchor_et = anchor.replace(tzinfo=dt.timezone(dt.timedelta(hours=-4)))
        else:
            anchor_et = fetch_window_start_et
        anchor_utc = anchor_et.astimezone(dt.timezone.utc)
        time_str = anchor_utc.strftime("%Y%m%d-%H:%M:%S")

        ib.reqExecutions(ExecutionFilter(time=time_str))
        # ib.fills() returns Fill objects after reqExecutions has populated.
        ib.sleep(2)
        raw_fills = list(ib.fills() or [])
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass

    intent_by_perm = _load_intent(trade_log)
    matched_perm_ids: set[int] = set()
    aggregates: dict[int, dict[str, Any]] = {}

    for fill in raw_fills:
        execu = getattr(fill, "execution", None)
        comm = getattr(fill, "commissionReport", None)
        if execu is None:
            continue
        symbol = getattr(getattr(fill, "contract", None), "symbol", None) or ""
        # Filter to today's session only (avoid stale fills if we widen window).
        fill_time = getattr(execu, "time", None)
        if fill_time is not None:
            try:
                if str(fill_time)[:10] != date:
                    continue
            except Exception:
                pass

        pid = int(getattr(execu, "permId", 0) or 0)
        oid = int(getattr(execu, "orderId", 0) or 0)
        side = (getattr(execu, "side", "") or "").upper()
        # IB's side is BOT/SLD; normalize to BUY/SELL to match intent records.
        if side == "BOT":
            side = "BUY"
        elif side == "SLD":
            side = "SELL"
        shares = float(getattr(execu, "shares", 0) or 0)
        price = float(getattr(execu, "price", 0) or 0)
        commission = float(comm.commission) if comm and getattr(comm, "commission", None) is not None else None
        rpnl = float(comm.realizedPNL) if comm and getattr(comm, "realizedPNL", None) is not None else None

        bucket_key = pid if pid > 0 else (-(oid or 0))
        b = aggregates.setdefault(bucket_key, {
            "symbol": symbol, "action": side, "perm_id": pid or None, "order_id": oid or None,
            "filled_qty": 0.0, "gross_notional": 0.0, "commission": 0.0,
            "realized_pnl": 0.0, "fill_count": 0, "has_commission": False, "has_rpnl": False,
        })
        b["filled_qty"] += shares
        b["gross_notional"] += shares * price
        if commission is not None:
            b["commission"] += commission
            b["has_commission"] = True
        if rpnl is not None:
            b["realized_pnl"] += rpnl
            b["has_rpnl"] = True
        b["fill_count"] += 1
        if pid > 0:
            matched_perm_ids.add(pid)

    fills: list[FillRow] = []
    extras: list[dict[str, Any]] = []
    for key, b in aggregates.items():
        avg_price = (b["gross_notional"] / b["filled_qty"]) if b["filled_qty"] else 0.0
        # Sign the qty to match intent (negative for SELL).
        signed_qty = int(round(b["filled_qty"] * (1 if b["action"] == "BUY" else -1)))
        row = FillRow(
            symbol=b["symbol"],
            action=b["action"],
            perm_id=b["perm_id"],
            order_id=b["order_id"],
            filled_qty=signed_qty,
            avg_price=round(avg_price, 4),
            gross_dollar=round(b["gross_notional"] * (1 if b["action"] == "BUY" else -1), 2),
            commission=round(b["commission"], 4) if b["has_commission"] else None,
            realized_pnl=round(b["realized_pnl"], 4) if b["has_rpnl"] else None,
            fill_count=b["fill_count"],
        )
        if b["perm_id"] and b["perm_id"] in intent_by_perm:
            fills.append(row)
        else:
            extras.append(asdict(row))

    unfilled: list[dict[str, Any]] = []
    intent_qty_total = 0
    for pid, rec in intent_by_perm.items():
        if pid in matched_perm_ids:
            continue
        try:
            qty = int(rec.get("quantity") or 0)
        except (TypeError, ValueError):
            qty = 0
        if rec.get("action", "").upper() == "SELL":
            qty = -qty
        intent_qty_total += abs(qty)
        unfilled.append({
            "symbol": rec.get("symbol"),
            "action": rec.get("action"),
            "ordered_qty": qty,
            "perm_id": rec.get("perm_id"),
            "order_id": rec.get("order_id"),
            "submission_status": rec.get("status"),
        })

    # Total intent qty (filled + unfilled, denominator for fill_rate_qty)
    full_intent_qty_total = 0
    for rec in trade_log.get("order_records") or []:
        try:
            q = int(rec.get("quantity") or 0)
        except (TypeError, ValueError):
            q = 0
        full_intent_qty_total += abs(q)

    filled_qty_total = sum(abs(f.filled_qty) for f in fills)
    n_intent = len(trade_log.get("order_records") or [])
    n_filled = len(fills)
    fill_rate_qty = (filled_qty_total / full_intent_qty_total) if full_intent_qty_total else 0.0
    fill_rate_orders = (n_filled / n_intent) if n_intent else 0.0
    gross_filled_dollar = sum(abs(f.gross_dollar) for f in fills)
    total_commission = sum((f.commission or 0.0) for f in fills)

    # Detect IB's retention drop-off: paper accounts don't retain executions
    # across daily restarts (~7-session window, often less). When the intent
    # shows >=10 orders that were Submitted/PreSubmitted/PendingSubmit but IB
    # returns 0 matching fills, the most likely explanation is "IB forgot",
    # not "every order failed". Flag the day as stale_unavailable so the
    # dashboard keeps it on the provisional curve instead of stamping a
    # misleading $0/0% recon.
    intent_submitted = sum(
        1 for r in trade_log.get("order_records") or []
        if str(r.get("status", "")).upper() in {
            "SUBMITTED", "PRESUBMITTED", "PENDINGSUBMIT", "FILLED", "PARTIALLYFILLED",
        }
    )
    if n_filled == 0 and intent_submitted >= 10:
        message = (
            f"IB returned 0 fills for {n_intent} submitted orders — likely "
            f"outside the paper-account execution retention window. Day kept "
            f"as provisional on the dashboard."
        )
        return ReconResult(
            date=date, timestamp=started.isoformat(),
            n_orders_intent=n_intent, n_orders_with_fills=0,
            n_unfilled=len(unfilled),
            fill_rate_qty=0.0, fill_rate_orders=0.0,
            gross_filled_shares=0, gross_filled_dollar=0.0, total_commission=0.0,
            fills=[], unfilled=unfilled, extras=extras,
            message=message,
        ), True  # second value = stale_unavailable flag

    return ReconResult(
        date=date,
        timestamp=started.isoformat(),
        n_orders_intent=n_intent,
        n_orders_with_fills=n_filled,
        n_unfilled=len(unfilled),
        fill_rate_qty=round(fill_rate_qty, 4),
        fill_rate_orders=round(fill_rate_orders, 4),
        gross_filled_shares=int(filled_qty_total),
        gross_filled_dollar=round(gross_filled_dollar, 2),
        total_commission=round(total_commission, 4),
        fills=fills,
        unfilled=unfilled,
        extras=extras,
        message=(f"reconciled {n_filled}/{n_intent} orders, "
                 f"qty fill-rate {fill_rate_qty:.1%}"),
    ), False


def write_recon(result: ReconResult, *, recon_dir: Path,
                stale_unavailable: bool = False) -> Path:
    """Atomically write a recon JSON. `stale_unavailable=True` stamps the
    file with status="stale_unavailable" so the dashboard treats the day as
    provisional rather than as a real "0% fills" recon."""
    recon_dir.mkdir(parents=True, exist_ok=True)
    path = recon_dir / f"equity_{result.date}.json"
    payload = result.to_dict()
    payload["status"] = "stale_unavailable" if stale_unavailable else "reconciled"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path
