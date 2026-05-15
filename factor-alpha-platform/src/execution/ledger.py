"""Durable multi-strategy execution ledger.

This ledger is the source of truth for the new execution path:

- strategy targets and child deltas
- internal crosses
- residual broker net orders
- strategy-level fill allocations
- virtual strategy positions
- strategy PnL curve snapshots

It deliberately lives next to, not inside, the legacy one-file trade logs while
the rollout is shadow/paper-first.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from src.execution.netting import (
    ChildOrder,
    FillAllocation,
    InternalCross,
    NetOrder,
    NettingResult,
    StrategyTarget,
    new_id,
)


DEFAULT_LEDGER_DB = Path("data/execution_ledger.db")


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS execution_batches (
    batch_id        TEXT PRIMARY KEY,
    created_at_utc TEXT NOT NULL,
    mode            TEXT NOT NULL,
    status          TEXT NOT NULL,
    metadata_json   TEXT
);

CREATE TABLE IF NOT EXISTS strategy_targets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id        TEXT NOT NULL,
    strategy_id     TEXT NOT NULL,
    asset_type      TEXT NOT NULL,
    venue           TEXT NOT NULL,
    account         TEXT NOT NULL,
    bucket          TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    target_qty      REAL NOT NULL,
    target_notional REAL NOT NULL,
    price           REAL NOT NULL,
    weight          REAL,
    order_type      TEXT NOT NULL,
    signal_time     TEXT,
    metadata_json   TEXT
);

CREATE INDEX IF NOT EXISTS idx_strategy_targets_batch
    ON strategy_targets(batch_id, strategy_id, symbol);

CREATE TABLE IF NOT EXISTS child_orders (
    child_order_id  TEXT PRIMARY KEY,
    batch_id        TEXT NOT NULL,
    strategy_id     TEXT NOT NULL,
    asset_type      TEXT NOT NULL,
    venue           TEXT NOT NULL,
    account         TEXT NOT NULL,
    bucket          TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    delta_qty       REAL NOT NULL,
    reference_price REAL NOT NULL,
    order_type      TEXT NOT NULL,
    metadata_json   TEXT
);

CREATE INDEX IF NOT EXISTS idx_child_orders_batch
    ON child_orders(batch_id, bucket, symbol);

CREATE TABLE IF NOT EXISTS internal_crosses (
    cross_id         TEXT PRIMARY KEY,
    batch_id         TEXT NOT NULL,
    bucket           TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    buy_strategy_id  TEXT NOT NULL,
    sell_strategy_id TEXT NOT NULL,
    quantity         REAL NOT NULL,
    price            REAL NOT NULL,
    notional         REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS net_orders (
    net_order_id    TEXT PRIMARY KEY,
    batch_id        TEXT NOT NULL,
    asset_type      TEXT NOT NULL,
    venue           TEXT NOT NULL,
    account         TEXT NOT NULL,
    bucket          TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    quantity        REAL NOT NULL,
    reference_price REAL NOT NULL,
    order_type      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'created',
    broker_order_id TEXT,
    metadata_json   TEXT
);

CREATE INDEX IF NOT EXISTS idx_net_orders_batch
    ON net_orders(batch_id, bucket, symbol);

CREATE TABLE IF NOT EXISTS fill_allocations (
    allocation_id TEXT PRIMARY KEY,
    batch_id      TEXT NOT NULL,
    strategy_id   TEXT NOT NULL,
    bucket        TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    quantity      REAL NOT NULL,
    price         REAL NOT NULL,
    notional      REAL NOT NULL,
    source        TEXT NOT NULL,
    source_id     TEXT NOT NULL,
    fee           REAL NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_fill_allocations_batch
    ON fill_allocations(batch_id, strategy_id, symbol);

CREATE TABLE IF NOT EXISTS strategy_positions (
    strategy_id    TEXT NOT NULL,
    asset_type     TEXT NOT NULL,
    venue          TEXT NOT NULL,
    account        TEXT NOT NULL,
    bucket         TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    quantity       REAL NOT NULL,
    avg_price      REAL,
    last_price     REAL,
    updated_at_utc TEXT NOT NULL,
    PRIMARY KEY(strategy_id, bucket, symbol)
);

CREATE TABLE IF NOT EXISTS strategy_pnl (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id     TEXT NOT NULL,
    asset_type      TEXT NOT NULL,
    venue           TEXT NOT NULL,
    account         TEXT NOT NULL,
    bucket          TEXT,
    timestamp_utc   TEXT NOT NULL,
    pnl             REAL NOT NULL,
    fees            REAL NOT NULL DEFAULT 0,
    cumulative_pnl  REAL NOT NULL,
    gross_exposure  REAL,
    net_exposure    REAL,
    n_positions     INTEGER,
    source          TEXT NOT NULL,
    metadata_json   TEXT
);

CREATE INDEX IF NOT EXISTS idx_strategy_pnl_strategy_time
    ON strategy_pnl(strategy_id, timestamp_utc);
"""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def connect(db_path: str | Path = DEFAULT_LEDGER_DB) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


class ExecutionLedger:
    def __init__(self, db_path: str | Path = DEFAULT_LEDGER_DB):
        self.db_path = Path(db_path)
        with connect(self.db_path):
            pass

    def start_batch(self, *, mode: str, metadata: dict[str, Any] | None = None,
                    batch_id: str | None = None) -> str:
        batch_id = batch_id or new_id("batch")
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO execution_batches (
                    batch_id, created_at_utc, mode, status, metadata_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (batch_id, utc_now(), mode, "started", _json(metadata)),
            )
            conn.commit()
        return batch_id

    def finish_batch(self, batch_id: str, *, status: str,
                     metadata: dict[str, Any] | None = None) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                "UPDATE execution_batches SET status=?, metadata_json=? WHERE batch_id=?",
                (status, _json(metadata), batch_id),
            )
            conn.commit()

    def record_targets(self, batch_id: str, targets: Iterable[StrategyTarget]) -> None:
        rows = [
            (
                batch_id, t.strategy_id, t.asset_type, t.venue, t.account,
                t.execution_bucket(), t.symbol, float(t.target_qty),
                float(t.target_notional), float(t.price),
                None if t.weight is None else float(t.weight), t.order_type,
                t.signal_time, _json(t.metadata),
            )
            for t in targets
        ]
        if not rows:
            return
        with connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO strategy_targets (
                    batch_id, strategy_id, asset_type, venue, account, bucket,
                    symbol, target_qty, target_notional, price, weight,
                    order_type, signal_time, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    def record_netting_result(self, result: NettingResult) -> None:
        with connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO child_orders (
                    child_order_id, batch_id, strategy_id, asset_type, venue,
                    account, bucket, symbol, delta_qty, reference_price,
                    order_type, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        o.child_order_id, o.batch_id, o.strategy_id, o.asset_type,
                        o.venue, o.account, o.bucket, o.symbol, float(o.delta_qty),
                        float(o.reference_price), o.order_type, _json(o.metadata),
                    )
                    for o in result.child_orders
                ],
            )
            conn.executemany(
                """
                INSERT INTO internal_crosses (
                    cross_id, batch_id, bucket, symbol, buy_strategy_id,
                    sell_strategy_id, quantity, price, notional
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c.cross_id, c.batch_id, c.bucket, c.symbol,
                        c.buy_strategy_id, c.sell_strategy_id, float(c.quantity),
                        float(c.price), float(c.notional),
                    )
                    for c in result.internal_crosses
                ],
            )
            conn.executemany(
                """
                INSERT INTO net_orders (
                    net_order_id, batch_id, asset_type, venue, account, bucket,
                    symbol, quantity, reference_price, order_type, status,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        o.net_order_id, o.batch_id, o.asset_type, o.venue,
                        o.account, o.bucket, o.symbol, float(o.quantity),
                        float(o.reference_price), o.order_type, "created",
                        _json({
                            "child_order_ids": list(o.child_order_ids),
                            "residual_child_qty": o.residual_child_qty,
                        }),
                    )
                    for o in result.net_orders
                ],
            )
            conn.commit()
        self.record_allocations(result.internal_allocations)

    def record_allocations(self, allocations: Iterable[FillAllocation]) -> None:
        allocations = list(allocations)
        if not allocations:
            return
        with connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO fill_allocations (
                    allocation_id, batch_id, strategy_id, bucket, symbol,
                    quantity, price, notional, source, source_id, fee
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        a.allocation_id, a.batch_id, a.strategy_id, a.bucket,
                        a.symbol, float(a.quantity), float(a.price),
                        float(a.notional), a.source, a.source_id, float(a.fee),
                    )
                    for a in allocations
                ],
            )
            conn.commit()

    def apply_allocations_to_positions(
        self,
        allocations: Iterable[FillAllocation],
        *,
        route_by_strategy: dict[str, dict[str, str]],
        timestamp_utc: str | None = None,
    ) -> None:
        """Apply strategy-level fills to the virtual books and charge fees."""
        timestamp = timestamp_utc or utc_now()
        allocations = list(allocations)
        if not allocations:
            return
        with connect(self.db_path) as conn:
            for alloc in allocations:
                route = route_by_strategy.get(alloc.strategy_id, {})
                row = conn.execute(
                    """
                    SELECT * FROM strategy_positions
                    WHERE strategy_id=? AND bucket=? AND symbol=?
                    """,
                    (alloc.strategy_id, alloc.bucket, alloc.symbol),
                ).fetchone()
                old_qty = float(row["quantity"]) if row else 0.0
                old_avg = float(row["avg_price"]) if row and row["avg_price"] is not None else float(alloc.price)
                new_qty = old_qty + float(alloc.quantity)
                if abs(new_qty) <= 1e-9:
                    conn.execute(
                        """
                        DELETE FROM strategy_positions
                        WHERE strategy_id=? AND bucket=? AND symbol=?
                        """,
                        (alloc.strategy_id, alloc.bucket, alloc.symbol),
                    )
                else:
                    avg_price = _updated_avg_price(old_qty, old_avg, float(alloc.quantity), float(alloc.price))
                    conn.execute(
                        """
                        INSERT INTO strategy_positions (
                            strategy_id, asset_type, venue, account, bucket, symbol,
                            quantity, avg_price, last_price, updated_at_utc
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(strategy_id, bucket, symbol) DO UPDATE SET
                            quantity=excluded.quantity,
                            avg_price=excluded.avg_price,
                            last_price=excluded.last_price,
                            updated_at_utc=excluded.updated_at_utc
                        """,
                        (
                            alloc.strategy_id,
                            route.get("asset_type", _bucket_part(alloc.bucket, 0)),
                            route.get("venue", _bucket_part(alloc.bucket, 1)),
                            route.get("account", _bucket_part(alloc.bucket, 2)),
                            alloc.bucket,
                            alloc.symbol,
                            new_qty,
                            avg_price,
                            float(alloc.price),
                            timestamp,
                        ),
                    )
                if alloc.fee:
                    self._record_pnl_row(
                        conn,
                        strategy_id=alloc.strategy_id,
                        asset_type=route.get("asset_type", _bucket_part(alloc.bucket, 0)),
                        venue=route.get("venue", _bucket_part(alloc.bucket, 1)),
                        account=route.get("account", _bucket_part(alloc.bucket, 2)),
                        bucket=alloc.bucket,
                        timestamp_utc=timestamp,
                        pnl=-float(alloc.fee),
                        fees=float(alloc.fee),
                        source="allocated_fee",
                        metadata={"allocation_id": alloc.allocation_id},
                    )
            conn.commit()

    def mark_to_market(
        self,
        *,
        prices: dict[str, float],
        timestamp_utc: str | None = None,
        source: str = "mark_to_market",
    ) -> list[dict[str, Any]]:
        """Mark open virtual strategy positions and record PnL rows."""
        timestamp = timestamp_utc or utc_now()
        rows_written: list[dict[str, Any]] = []
        with connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM strategy_positions").fetchall()
            by_strategy: dict[str, list[sqlite3.Row]] = {}
            for row in rows:
                by_strategy.setdefault(row["strategy_id"], []).append(row)

            for strategy_id, pos_rows in by_strategy.items():
                pnl = 0.0
                gross = 0.0
                net = 0.0
                n_positions = 0
                for row in pos_rows:
                    symbol = row["symbol"]
                    if symbol not in prices:
                        continue
                    price = float(prices[symbol])
                    qty = float(row["quantity"])
                    last_price = row["last_price"]
                    if last_price is not None:
                        pnl += qty * (price - float(last_price))
                    gross += abs(qty * price)
                    net += qty * price
                    n_positions += 1
                    conn.execute(
                        """
                        UPDATE strategy_positions
                        SET last_price=?, updated_at_utc=?
                        WHERE strategy_id=? AND bucket=? AND symbol=?
                        """,
                        (price, timestamp, strategy_id, row["bucket"], symbol),
                    )
                if n_positions:
                    first = pos_rows[0]
                    cumulative = self._record_pnl_row(
                        conn,
                        strategy_id=strategy_id,
                        asset_type=first["asset_type"],
                        venue=first["venue"],
                        account=first["account"],
                        bucket=None,
                        timestamp_utc=timestamp,
                        pnl=pnl,
                        fees=0.0,
                        source=source,
                        gross_exposure=gross,
                        net_exposure=net,
                        n_positions=n_positions,
                    )
                    rows_written.append({
                        "strategy_id": strategy_id,
                        "pnl": pnl,
                        "cumulative_pnl": cumulative,
                        "gross_exposure": gross,
                        "net_exposure": net,
                        "n_positions": n_positions,
                    })
            conn.commit()
        return rows_written

    def current_positions(self, strategy_ids: Iterable[str] | None = None) -> dict[tuple[str, str], float]:
        params: list[Any] = []
        where = ""
        if strategy_ids is not None:
            ids = list(strategy_ids)
            if not ids:
                return {}
            where = f"WHERE strategy_id IN ({','.join('?' for _ in ids)})"
            params.extend(ids)
        with connect(self.db_path) as conn:
            rows = conn.execute(f"SELECT strategy_id, symbol, quantity FROM strategy_positions {where}", params).fetchall()
        return {(r["strategy_id"], r["symbol"]): float(r["quantity"]) for r in rows}

    def strategy_summaries(self) -> list[dict[str, Any]]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT p.strategy_id, p.asset_type, p.venue, p.account,
                       SUM(ABS(p.quantity * COALESCE(p.last_price, p.avg_price, 0))) AS gross_exposure,
                       SUM(p.quantity * COALESCE(p.last_price, p.avg_price, 0)) AS net_exposure,
                       COUNT(*) AS n_positions,
                       COALESCE((SELECT cumulative_pnl FROM strategy_pnl q
                                 WHERE q.strategy_id=p.strategy_id
                                 ORDER BY q.timestamp_utc DESC, q.id DESC LIMIT 1), 0) AS cumulative_pnl
                FROM strategy_positions p
                GROUP BY p.strategy_id, p.asset_type, p.venue, p.account
                ORDER BY p.asset_type, p.strategy_id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def strategy_curves(self, *, max_points: int = 360) -> dict[str, list[dict[str, Any]]]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT strategy_id, timestamp_utc, cumulative_pnl
                FROM strategy_pnl
                ORDER BY strategy_id, timestamp_utc, id
                """
            ).fetchall()
        curves: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            curves.setdefault(row["strategy_id"], []).append({
                "timestamp": row["timestamp_utc"],
                "pnl": float(row["cumulative_pnl"]),
            })
        if max_points > 0:
            for sid, points in list(curves.items()):
                if len(points) > max_points:
                    if max_points == 1:
                        curves[sid] = [points[-1]]
                        continue
                    step = (len(points) - 1) / float(max_points - 1)
                    indices = [round(i * step) for i in range(max_points)]
                    curves[sid] = [points[int(i)] for i in indices]
        return curves

    def latest_netting_stats(self) -> dict[str, Any]:
        with connect(self.db_path) as conn:
            batch = conn.execute(
                "SELECT * FROM execution_batches ORDER BY created_at_utc DESC LIMIT 1"
            ).fetchone()
            if not batch:
                return {}
            batch_id = batch["batch_id"]
            child = conn.execute(
                "SELECT COUNT(*) AS n, SUM(ABS(delta_qty * reference_price)) AS gross FROM child_orders WHERE batch_id=?",
                (batch_id,),
            ).fetchone()
            net = conn.execute(
                "SELECT COUNT(*) AS n, SUM(ABS(quantity * reference_price)) AS gross FROM net_orders WHERE batch_id=?",
                (batch_id,),
            ).fetchone()
            cross = conn.execute(
                "SELECT COUNT(*) AS n, SUM(notional) AS gross FROM internal_crosses WHERE batch_id=?",
                (batch_id,),
            ).fetchone()
        child_gross = float(child["gross"] or 0.0)
        net_gross = float(net["gross"] or 0.0)
        return {
            "batch_id": batch_id,
            "created_at_utc": batch["created_at_utc"],
            "mode": batch["mode"],
            "status": batch["status"],
            "n_child_orders": int(child["n"] or 0),
            "child_notional": child_gross,
            "n_net_orders": int(net["n"] or 0),
            "net_notional": net_gross,
            "n_internal_crosses": int(cross["n"] or 0),
            "crossed_notional": float(cross["gross"] or 0.0),
            "compression_ratio": 0.0 if child_gross <= 0 else 1.0 - (net_gross / child_gross),
        }

    def _record_pnl_row(
        self,
        conn: sqlite3.Connection,
        *,
        strategy_id: str,
        asset_type: str,
        venue: str,
        account: str,
        bucket: str | None,
        timestamp_utc: str,
        pnl: float,
        fees: float,
        source: str,
        gross_exposure: float | None = None,
        net_exposure: float | None = None,
        n_positions: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        prev = conn.execute(
            """
            SELECT cumulative_pnl FROM strategy_pnl
            WHERE strategy_id=?
            ORDER BY timestamp_utc DESC, id DESC LIMIT 1
            """,
            (strategy_id,),
        ).fetchone()
        cumulative = float(prev["cumulative_pnl"]) if prev else 0.0
        cumulative += float(pnl)
        conn.execute(
            """
            INSERT INTO strategy_pnl (
                strategy_id, asset_type, venue, account, bucket, timestamp_utc,
                pnl, fees, cumulative_pnl, gross_exposure, net_exposure,
                n_positions, source, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                strategy_id, asset_type, venue, account, bucket, timestamp_utc,
                float(pnl), float(fees), cumulative, gross_exposure,
                net_exposure, n_positions, source, _json(metadata),
            ),
        )
        return cumulative


def _updated_avg_price(old_qty: float, old_avg: float, fill_qty: float, fill_price: float) -> float:
    if old_qty == 0 or (old_qty > 0) != (fill_qty > 0):
        if abs(fill_qty) >= abs(old_qty):
            return fill_price
        return old_avg
    new_qty = old_qty + fill_qty
    if abs(new_qty) <= 1e-9:
        return fill_price
    return ((abs(old_qty) * old_avg) + (abs(fill_qty) * fill_price)) / abs(new_qty)


def _bucket_part(bucket: str, idx: int) -> str:
    parts = bucket.split(":")
    return parts[idx] if idx < len(parts) else "unknown"


def _json(value: dict[str, Any] | None) -> str:
    return json.dumps(value or {}, sort_keys=True, default=str)
