"""
Unified Alpha Database — permanent store for alphas across asset classes and timescales.

Schema design:
  - alphas:       Expression definitions with category, asset class, interval, source
  - evaluations:  Per-alpha backtest metrics with proper train/val/test timeline tracking
  - runs:         Backtest configuration snapshots
  - selections:   Walk-forward alpha selection history (which alphas chosen at each bar)
  - correlations: Pairwise alpha correlation cache

Key features vs old per-file approach:
  - Single database for all asset classes (equity, crypto) and intervals (1d, 4h, 1h)
  - Expressions are UNIQUE — same alpha won't be stored twice
  - Full timeline tracking (train_start/end, val_start/end, test_start/end)
  - Walk-forward selection journal for reproducibility
  - Correlation cache to avoid recomputation
  - Source tagging (gp, manual, cryptorl, llm) for provenance

Usage:
    from src.data.alpha_db import AlphaDB
    db = AlphaDB()  # defaults to data/alphas.db
    alpha_id = db.upsert_alpha("rank(sma(returns, 90))", name="mom_90bar",
                                category="momentum", asset_class="crypto", interval="4h")
    db.add_evaluation(alpha_id, run_id, sharpe_train=1.5, sharpe_val=0.8, ...)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/alphas.db"

SCHEMA_SQL = """
-- ═══════════════════════════════════════════════════════════════
-- Core alpha definitions
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS alphas (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    expression      TEXT NOT NULL,
    name            TEXT,
    category        TEXT,           -- 'momentum', 'reversal', 'volume', 'correlation', etc.
    asset_class     TEXT NOT NULL,  -- 'equity', 'crypto'
    interval        TEXT NOT NULL,  -- '1d', '4h', '1h'
    source          TEXT,           -- 'gp', 'manual', 'cryptorl', 'llm'
    created_at      TEXT DEFAULT (datetime('now')),
    archived        INTEGER DEFAULT 0,
    notes           TEXT,
    UNIQUE(expression, asset_class, interval)
);

-- ═══════════════════════════════════════════════════════════════
-- Backtest runs (configuration snapshots)
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT,
    asset_class     TEXT NOT NULL,
    interval        TEXT NOT NULL,
    universe        TEXT,           -- 'BINANCE_TOP50', 'SP500', 'TOP1000', etc.
    fee_bps         REAL DEFAULT 5.0,
    delay           INTEGER DEFAULT 1,
    neutralization  TEXT DEFAULT 'market',
    train_bars      INTEGER,
    val_bars        INTEGER,
    n_alphas_tested INTEGER,
    n_alphas_passed INTEGER,
    sharpe_net      REAL,
    sharpe_gross    REAL,
    total_return    REAL,
    max_drawdown    REAL,
    started_at      TEXT DEFAULT (datetime('now')),
    completed_at    TEXT,
    config_json     TEXT,           -- Full config dump for exact reproducibility
    notes           TEXT
);

-- ═══════════════════════════════════════════════════════════════
-- Per-alpha evaluation results within a run
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS evaluations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    alpha_id        INTEGER NOT NULL REFERENCES alphas(id),
    run_id          INTEGER REFERENCES runs(id),
    -- Performance metrics
    sharpe_is       REAL,           -- In-sample Sharpe
    sharpe_oos      REAL,           -- Out-of-sample Sharpe
    sharpe_train    REAL,           -- Training period only
    sharpe_val      REAL,           -- Validation period only  
    sharpe_test     REAL,           -- True test period
    return_total    REAL,
    return_ann      REAL,
    max_drawdown    REAL,
    turnover        REAL,
    fitness         REAL,
    ic_mean         REAL,           -- Mean information coefficient
    ic_ir           REAL,           -- IC information ratio
    psr             REAL,           -- Probabilistic Sharpe Ratio
    -- Timeline tracking
    train_start     TEXT,
    train_end       TEXT,
    val_start       TEXT,
    val_end         TEXT,
    test_start      TEXT,
    test_end        TEXT,
    n_bars          INTEGER,
    -- Metadata
    evaluated_at    TEXT DEFAULT (datetime('now'))
);

-- ═══════════════════════════════════════════════════════════════
-- Walk-forward selection journal
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS selections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES runs(id),
    bar_index       INTEGER NOT NULL,
    bar_datetime    TEXT NOT NULL,
    alpha_id        INTEGER NOT NULL REFERENCES alphas(id),
    qp_weight       REAL,           -- Weight from QP combiner
    train_sharpe    REAL,
    val_sharpe      REAL,
    selected_at     TEXT DEFAULT (datetime('now'))
);

-- ═══════════════════════════════════════════════════════════════
-- Pairwise correlation cache
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS correlations (
    alpha_id_a      INTEGER NOT NULL REFERENCES alphas(id),
    alpha_id_b      INTEGER NOT NULL REFERENCES alphas(id),
    asset_class     TEXT NOT NULL,
    interval        TEXT NOT NULL,
    correlation     REAL NOT NULL,
    computed_at     TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (alpha_id_a, alpha_id_b, asset_class, interval)
);

-- ═══════════════════════════════════════════════════════════════
-- Indexes for common queries
-- ═══════════════════════════════════════════════════════════════
CREATE INDEX IF NOT EXISTS idx_alphas_asset_interval 
    ON alphas(asset_class, interval);
CREATE INDEX IF NOT EXISTS idx_alphas_category 
    ON alphas(category, asset_class, interval);
CREATE INDEX IF NOT EXISTS idx_evaluations_alpha 
    ON evaluations(alpha_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_run 
    ON evaluations(run_id);
CREATE INDEX IF NOT EXISTS idx_selections_run 
    ON selections(run_id, bar_index);
CREATE INDEX IF NOT EXISTS idx_alphas_source 
    ON alphas(source);
"""


class AlphaDB:
    """Unified alpha database with multi-asset, multi-timescale support."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ─── Alpha CRUD ──────────────────────────────────────────────

    def upsert_alpha(self, expression: str, *,
                     name: str | None = None,
                     category: str | None = None,
                     asset_class: str = "crypto",
                     interval: str = "4h",
                     source: str = "manual",
                     notes: str | None = None) -> int:
        """Insert or update an alpha. Returns the alpha ID."""
        cur = self.conn.execute(
            "SELECT id FROM alphas WHERE expression=? AND asset_class=? AND interval=?",
            (expression, asset_class, interval),
        )
        row = cur.fetchone()
        if row:
            alpha_id = row["id"]
            # Update metadata if provided
            updates = []
            params = []
            if name is not None:
                updates.append("name=?")
                params.append(name)
            if category is not None:
                updates.append("category=?")
                params.append(category)
            if source is not None:
                updates.append("source=?")
                params.append(source)
            if notes is not None:
                updates.append("notes=?")
                params.append(notes)
            if updates:
                params.append(alpha_id)
                self.conn.execute(
                    f"UPDATE alphas SET {', '.join(updates)} WHERE id=?", params
                )
                self.conn.commit()
            return alpha_id
        else:
            cur = self.conn.execute(
                """INSERT INTO alphas (expression, name, category, asset_class, interval, source, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (expression, name, category, asset_class, interval, source, notes),
            )
            self.conn.commit()
            return cur.lastrowid

    def bulk_upsert_alphas(self, alphas: list[dict]) -> list[int]:
        """Insert/update many alphas at once. Each dict needs 'expression' at minimum."""
        ids = []
        for a in alphas:
            alpha_id = self.upsert_alpha(
                a["expression"],
                name=a.get("name"),
                category=a.get("category"),
                asset_class=a.get("asset_class", "crypto"),
                interval=a.get("interval", "4h"),
                source=a.get("source", "manual"),
                notes=a.get("notes"),
            )
            ids.append(alpha_id)
        return ids

    def get_alpha(self, alpha_id: int) -> dict | None:
        """Get alpha by ID."""
        row = self.conn.execute("SELECT * FROM alphas WHERE id=?", (alpha_id,)).fetchone()
        return dict(row) if row else None

    def get_alpha_by_expr(self, expression: str, asset_class: str, interval: str) -> dict | None:
        """Get alpha by expression + asset class + interval."""
        row = self.conn.execute(
            "SELECT * FROM alphas WHERE expression=? AND asset_class=? AND interval=?",
            (expression, asset_class, interval),
        ).fetchone()
        return dict(row) if row else None

    def list_alphas(self, *,
                    asset_class: str | None = None,
                    interval: str | None = None,
                    category: str | None = None,
                    source: str | None = None,
                    archived: bool = False) -> list[dict]:
        """List alphas with optional filters."""
        where = ["archived=?"]
        params: list[Any] = [int(archived)]
        if asset_class:
            where.append("asset_class=?")
            params.append(asset_class)
        if interval:
            where.append("interval=?")
            params.append(interval)
        if category:
            where.append("category=?")
            params.append(category)
        if source:
            where.append("source=?")
            params.append(source)
        sql = f"SELECT * FROM alphas WHERE {' AND '.join(where)} ORDER BY id"
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def archive_alpha(self, alpha_id: int):
        """Soft-delete an alpha."""
        self.conn.execute("UPDATE alphas SET archived=1 WHERE id=?", (alpha_id,))
        self.conn.commit()

    # ─── Runs ────────────────────────────────────────────────────

    def create_run(self, name: str, *,
                   asset_class: str = "crypto",
                   interval: str = "4h",
                   universe: str = "BINANCE_TOP50",
                   fee_bps: float = 5.0,
                   config: dict | None = None,
                   notes: str | None = None) -> int:
        """Create a new backtest run. Returns run ID."""
        cur = self.conn.execute(
            """INSERT INTO runs (name, asset_class, interval, universe, fee_bps, config_json, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (name, asset_class, interval, universe, fee_bps,
             json.dumps(config) if config else None, notes),
        )
        self.conn.commit()
        return cur.lastrowid

    def complete_run(self, run_id: int, *,
                     sharpe_net: float | None = None,
                     sharpe_gross: float | None = None,
                     total_return: float | None = None,
                     max_drawdown: float | None = None,
                     n_alphas_tested: int | None = None,
                     n_alphas_passed: int | None = None):
        """Mark a run as completed with final metrics."""
        self.conn.execute(
            """UPDATE runs SET completed_at=datetime('now'),
               sharpe_net=?, sharpe_gross=?, total_return=?, max_drawdown=?,
               n_alphas_tested=?, n_alphas_passed=?
               WHERE id=?""",
            (sharpe_net, sharpe_gross, total_return, max_drawdown,
             n_alphas_tested, n_alphas_passed, run_id),
        )
        self.conn.commit()

    def list_runs(self, asset_class: str | None = None,
                  interval: str | None = None) -> list[dict]:
        """List backtest runs."""
        where = []
        params: list[Any] = []
        if asset_class:
            where.append("asset_class=?")
            params.append(asset_class)
        if interval:
            where.append("interval=?")
            params.append(interval)
        sql = "SELECT * FROM runs"
        if where:
            sql += f" WHERE {' AND '.join(where)}"
        sql += " ORDER BY id DESC"
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    # ─── Evaluations ─────────────────────────────────────────────

    def add_evaluation(self, alpha_id: int, run_id: int | None = None, **kwargs) -> int:
        """Add an evaluation result for an alpha."""
        cols = ["alpha_id", "run_id"]
        vals: list[Any] = [alpha_id, run_id]
        valid_cols = {
            "sharpe_is", "sharpe_oos", "sharpe_train", "sharpe_val", "sharpe_test",
            "return_total", "return_ann", "max_drawdown", "turnover", "fitness",
            "ic_mean", "ic_ir", "psr",
            "train_start", "train_end", "val_start", "val_end",
            "test_start", "test_end", "n_bars",
        }
        for k, v in kwargs.items():
            if k in valid_cols:
                cols.append(k)
                vals.append(v)
        placeholders = ", ".join("?" * len(cols))
        col_names = ", ".join(cols)
        cur = self.conn.execute(
            f"INSERT INTO evaluations ({col_names}) VALUES ({placeholders})", vals
        )
        self.conn.commit()
        return cur.lastrowid

    def get_best_alphas(self, asset_class: str, interval: str, *,
                        metric: str = "sharpe_oos",
                        min_value: float = 0.0,
                        limit: int = 50) -> list[dict]:
        """Get top alphas by a performance metric."""
        sql = f"""
            SELECT a.*, e.{metric}, e.sharpe_train, e.sharpe_val, e.sharpe_test, e.turnover
            FROM alphas a
            JOIN evaluations e ON a.id = e.alpha_id
            WHERE a.asset_class=? AND a.interval=? AND a.archived=0
              AND e.{metric} > ?
            ORDER BY e.{metric} DESC
            LIMIT ?
        """
        return [dict(r) for r in self.conn.execute(
            sql, (asset_class, interval, min_value, limit)
        ).fetchall()]

    # ─── Walk-Forward Selections ─────────────────────────────────

    def add_selections(self, run_id: int, bar_index: int, bar_datetime: str,
                       selections: list[dict]):
        """Record which alphas were selected at a walk-forward rebalance point."""
        for s in selections:
            self.conn.execute(
                """INSERT INTO selections 
                   (run_id, bar_index, bar_datetime, alpha_id, qp_weight, train_sharpe, val_sharpe)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (run_id, bar_index, bar_datetime,
                 s["alpha_id"], s.get("qp_weight"), s.get("train_sharpe"), s.get("val_sharpe")),
            )
        self.conn.commit()

    def get_selection_history(self, run_id: int) -> list[dict]:
        """Get full selection history for a run."""
        sql = """
            SELECT s.*, a.name, a.expression, a.category
            FROM selections s
            JOIN alphas a ON s.alpha_id = a.id
            WHERE s.run_id = ?
            ORDER BY s.bar_index, s.qp_weight DESC
        """
        return [dict(r) for r in self.conn.execute(sql, (run_id,)).fetchall()]

    # ─── Correlations ────────────────────────────────────────────

    def set_correlation(self, alpha_id_a: int, alpha_id_b: int,
                        correlation: float, asset_class: str, interval: str):
        """Store pairwise correlation (upsert)."""
        a, b = min(alpha_id_a, alpha_id_b), max(alpha_id_a, alpha_id_b)
        self.conn.execute(
            """INSERT OR REPLACE INTO correlations 
               (alpha_id_a, alpha_id_b, asset_class, interval, correlation)
               VALUES (?, ?, ?, ?, ?)""",
            (a, b, asset_class, interval, correlation),
        )
        self.conn.commit()

    def get_correlation(self, alpha_id_a: int, alpha_id_b: int,
                        asset_class: str, interval: str) -> float | None:
        """Get cached correlation between two alphas."""
        a, b = min(alpha_id_a, alpha_id_b), max(alpha_id_a, alpha_id_b)
        row = self.conn.execute(
            """SELECT correlation FROM correlations
               WHERE alpha_id_a=? AND alpha_id_b=? AND asset_class=? AND interval=?""",
            (a, b, asset_class, interval),
        ).fetchone()
        return row["correlation"] if row else None

    # ─── Summary Stats ───────────────────────────────────────────

    def summary(self) -> dict:
        """Get database summary stats."""
        counts = {}
        for table in ["alphas", "evaluations", "runs", "selections", "correlations"]:
            row = self.conn.execute(f"SELECT COUNT(*) as n FROM {table}").fetchone()
            counts[table] = row["n"]

        # Breakdown by asset class / interval
        breakdown = self.conn.execute("""
            SELECT asset_class, interval, COUNT(*) as n, 
                   SUM(CASE WHEN archived=0 THEN 1 ELSE 0 END) as active
            FROM alphas 
            GROUP BY asset_class, interval
        """).fetchall()

        return {
            "total_counts": counts,
            "alpha_breakdown": [dict(r) for r in breakdown],
        }
