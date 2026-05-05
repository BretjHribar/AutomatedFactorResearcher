"""SQLite store for strategy health, integrity checks, and alerts."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_STATE_DB = Path("data/prod_state.db")


@dataclass(frozen=True)
class StrategyState:
    strategy_id: str
    market: str
    mode: str
    status: str
    last_data_bar: str | None = None
    last_signal_bar: str | None = None
    last_trade_time: str | None = None
    config_hash: str | None = None
    git_sha: str | None = None
    gross_exposure: float | None = None
    net_exposure: float | None = None
    n_positions: int | None = None
    metrics_json: dict[str, Any] | None = None


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS strategy_state (
    strategy_id      TEXT PRIMARY KEY,
    market           TEXT NOT NULL,
    mode             TEXT NOT NULL,
    status           TEXT NOT NULL,
    last_data_bar    TEXT,
    last_signal_bar  TEXT,
    last_trade_time  TEXT,
    config_hash      TEXT,
    git_sha          TEXT,
    gross_exposure   REAL,
    net_exposure     REAL,
    n_positions      INTEGER,
    metrics_json     TEXT,
    updated_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS data_integrity_checks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL,
    market        TEXT NOT NULL,
    check_name    TEXT NOT NULL,
    status        TEXT NOT NULL,
    severity      TEXT NOT NULL,
    message       TEXT NOT NULL,
    value         TEXT,
    threshold     TEXT,
    metadata_json TEXT,
    checked_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_data_integrity_latest
    ON data_integrity_checks(market, check_name, checked_at);

CREATE TABLE IF NOT EXISTS alerts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id   TEXT,
    market        TEXT,
    severity      TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'open',
    message       TEXT NOT NULL,
    metadata_json TEXT,
    created_at    TEXT NOT NULL,
    resolved_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_alerts_open
    ON alerts(status, severity, created_at);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(db_path: str | Path = DEFAULT_STATE_DB) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def upsert_strategy_state(state: StrategyState, db_path: str | Path = DEFAULT_STATE_DB) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO strategy_state (
                strategy_id, market, mode, status, last_data_bar, last_signal_bar,
                last_trade_time, config_hash, git_sha, gross_exposure, net_exposure,
                n_positions, metrics_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(strategy_id) DO UPDATE SET
                market=excluded.market,
                mode=excluded.mode,
                status=excluded.status,
                last_data_bar=excluded.last_data_bar,
                last_signal_bar=excluded.last_signal_bar,
                last_trade_time=excluded.last_trade_time,
                config_hash=excluded.config_hash,
                git_sha=excluded.git_sha,
                gross_exposure=excluded.gross_exposure,
                net_exposure=excluded.net_exposure,
                n_positions=excluded.n_positions,
                metrics_json=excluded.metrics_json,
                updated_at=excluded.updated_at
            """,
            (
                state.strategy_id,
                state.market,
                state.mode,
                state.status,
                state.last_data_bar,
                state.last_signal_bar,
                state.last_trade_time,
                state.config_hash,
                state.git_sha,
                state.gross_exposure,
                state.net_exposure,
                state.n_positions,
                json.dumps(state.metrics_json or {}, sort_keys=True),
                utc_now(),
            ),
        )
        conn.commit()


def record_check_results(results: Iterable[Any], *, run_id: str, market: str,
                         db_path: str | Path = DEFAULT_STATE_DB) -> None:
    rows = []
    checked_at = utc_now()
    for result in results:
        data = asdict(result) if hasattr(result, "__dataclass_fields__") else dict(result)
        rows.append(
            (
                run_id,
                market,
                data["name"],
                data["status"],
                data["severity"],
                data["message"],
                None if data.get("value") is None else str(data.get("value")),
                None if data.get("threshold") is None else str(data.get("threshold")),
                json.dumps(data.get("metadata") or {}, sort_keys=True),
                checked_at,
            )
        )
    with connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO data_integrity_checks (
                run_id, market, check_name, status, severity, message,
                value, threshold, metadata_json, checked_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()


def latest_checks(market: str | None = None, db_path: str | Path = DEFAULT_STATE_DB) -> list[dict[str, Any]]:
    where = ""
    params: list[Any] = []
    if market:
        where = "WHERE d.market=?"
        params.append(market)
    sql = f"""
        SELECT d.*
        FROM data_integrity_checks d
        JOIN (
            SELECT market, check_name, MAX(checked_at) AS checked_at
            FROM data_integrity_checks
            GROUP BY market, check_name
        ) latest
          ON latest.market=d.market
         AND latest.check_name=d.check_name
         AND latest.checked_at=d.checked_at
        {where}
        ORDER BY d.market, d.check_name
    """
    with connect(db_path) as conn:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def strategy_states(db_path: str | Path = DEFAULT_STATE_DB) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        return [
            dict(row)
            for row in conn.execute("SELECT * FROM strategy_state ORDER BY strategy_id").fetchall()
        ]


def open_alerts(db_path: str | Path = DEFAULT_STATE_DB) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM alerts WHERE status='open' ORDER BY severity DESC, created_at DESC"
            ).fetchall()
        ]


def raise_alert(*, message: str, severity: str = "warning",
                strategy_id: str | None = None, market: str | None = None,
                metadata: dict[str, Any] | None = None,
                db_path: str | Path = DEFAULT_STATE_DB) -> None:
    """Create an open alert if the same open alert does not already exist."""
    with connect(db_path) as conn:
        existing = conn.execute(
            """
            SELECT id FROM alerts
            WHERE status='open'
              AND COALESCE(strategy_id, '')=COALESCE(?, '')
              AND COALESCE(market, '')=COALESCE(?, '')
              AND message=?
            """,
            (strategy_id, market, message),
        ).fetchone()
        if existing:
            return
        conn.execute(
            """
            INSERT INTO alerts (
                strategy_id, market, severity, status, message, metadata_json, created_at
            ) VALUES (?, ?, ?, 'open', ?, ?, ?)
            """,
            (
                strategy_id,
                market,
                severity,
                message,
                json.dumps(metadata or {}, sort_keys=True),
                utc_now(),
            ),
        )
        conn.commit()


def resolve_alerts(*, message_prefix: str, strategy_id: str | None = None,
                   market: str | None = None,
                   db_path: str | Path = DEFAULT_STATE_DB) -> int:
    """Resolve open alerts whose message starts with the given prefix."""
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE alerts
               SET status='resolved',
                   resolved_at=?
             WHERE status='open'
               AND COALESCE(strategy_id, '')=COALESCE(?, '')
               AND COALESCE(market, '')=COALESCE(?, '')
               AND message LIKE ?
            """,
            (utc_now(), strategy_id, market, f"{message_prefix}%"),
        )
        conn.commit()
        return int(cur.rowcount or 0)


def sync_alerts_from_latest_checks(db_path: str | Path = DEFAULT_STATE_DB) -> int:
    """Promote latest warning/failing checks and resolve checks that recovered."""
    count = 0
    for check in latest_checks(db_path=db_path):
        if check["status"] not in ("fail", "warn"):
            resolve_alerts(
                message_prefix=f"{check['check_name']}:",
                market=check["market"],
                db_path=db_path,
            )
            continue
        raise_alert(
            message=f"{check['check_name']}: {check['message']}",
            severity=check["severity"],
            market=check["market"],
            metadata={"run_id": check["run_id"], "value": check["value"], "threshold": check["threshold"]},
            db_path=db_path,
        )
        count += 1
    return count
