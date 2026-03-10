"""
Alpha Database — SQLite-backed storage for alpha research results.

Stores every alpha that has been generated, tested, and evaluated so the
LLM agent can learn from prior attempts and avoid duplicates.

Tables:
    runs         — research sessions
    alphas       — generated alpha expressions
    evaluations  — simulation results for each alpha
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class AlphaDatabase:
    """SQLite-based alpha result storage for LLM learning."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "data" / "alpha_research.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=60)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=60000")
        self._create_tables()
        logger.info(f"AlphaDatabase initialized: {self.db_path}")

    def _create_tables(self) -> None:
        c = self._conn.cursor()
        c.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at     TEXT NOT NULL DEFAULT (datetime('now')),
            strategy       TEXT,
            llm_model      TEXT,
            config_json    TEXT,
            status         TEXT DEFAULT 'running',
            notes          TEXT
        );

        CREATE TABLE IF NOT EXISTS alphas (
            alpha_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id         INTEGER REFERENCES runs(run_id),
            trial_index    INTEGER,
            created_at     TEXT NOT NULL DEFAULT (datetime('now')),
            expression     TEXT NOT NULL,
            params_json    TEXT,
            reasoning      TEXT,
            source         TEXT DEFAULT 'llm',
            status         TEXT DEFAULT 'created',
            wq_alpha_id    TEXT,
            pnl_blob       BLOB,
            UNIQUE(expression, params_json)
        );

        CREATE TABLE IF NOT EXISTS evaluations (
            eval_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            alpha_id       INTEGER REFERENCES alphas(alpha_id),
            evaluated_at   TEXT NOT NULL DEFAULT (datetime('now')),
            source         TEXT DEFAULT 'local',
            sharpe         REAL,
            fitness        REAL,
            turnover       REAL,
            max_drawdown   REAL,
            returns_ann    REAL,
            profit_dollar  REAL,
            margin_bps     REAL,
            passed_checks  INTEGER,
            error          TEXT,
            metrics_json   TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_alphas_expression ON alphas(expression);
        CREATE INDEX IF NOT EXISTS idx_alphas_run_id ON alphas(run_id);
        CREATE INDEX IF NOT EXISTS idx_evaluations_alpha_id ON evaluations(alpha_id);
        CREATE INDEX IF NOT EXISTS idx_evaluations_sharpe ON evaluations(sharpe);
        CREATE INDEX IF NOT EXISTS idx_evaluations_fitness ON evaluations(fitness);
        """)
        self._conn.commit()

        # Add pnl_blob column if it doesn't exist (migration)
        try:
            c.execute("ALTER TABLE alphas ADD COLUMN pnl_blob BLOB")
            self._conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def create_run(self, strategy: str = "", llm_model: str = "",
                   config: dict | None = None, notes: str = "") -> int:
        """Create a new research run. Returns run_id."""
        c = self._conn.cursor()
        c.execute(
            "INSERT INTO runs (strategy, llm_model, config_json, notes) VALUES (?, ?, ?, ?)",
            (strategy, llm_model, json.dumps(config or {}), notes)
        )
        self._conn.commit()
        return c.lastrowid

    def finish_run(self, run_id: int, status: str = "completed") -> None:
        self._conn.execute("UPDATE runs SET status = ? WHERE run_id = ?", (status, run_id))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Alphas
    # ------------------------------------------------------------------

    def insert_alpha(self, expression: str, params: dict | None = None,
                     reasoning: str = "", source: str = "llm",
                     run_id: int | None = None, trial_index: int | None = None,
                     wq_alpha_id: str | None = None) -> int:
        """Insert a new alpha. Returns alpha_id. If duplicate, returns existing id."""
        params_json = json.dumps(params or {}, sort_keys=True)
        c = self._conn.cursor()
        # Check for existing
        c.execute("SELECT alpha_id FROM alphas WHERE expression = ? AND params_json = ?",
                  (expression, params_json))
        row = c.fetchone()
        if row:
            return row["alpha_id"]

        c.execute("""
            INSERT INTO alphas (run_id, trial_index, expression, params_json,
                              reasoning, source, wq_alpha_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (run_id, trial_index, expression, params_json, reasoning, source, wq_alpha_id))
        self._conn.commit()
        return c.lastrowid

    def store_pnl(self, alpha_id: int, pnl_vec: np.ndarray) -> None:
        """Store PnL vector as blob for cross-process correlation checking."""
        blob = pnl_vec.astype(np.float64).tobytes()
        c = self._conn.cursor()
        c.execute("UPDATE alphas SET pnl_blob = ? WHERE alpha_id = ?", (blob, alpha_id))
        self._conn.commit()

    def get_all_pnl_vectors(self) -> list[np.ndarray]:
        """Retrieve all stored PnL vectors from the DB (for correlation check)."""
        c = self._conn.cursor()
        c.execute("SELECT pnl_blob FROM alphas WHERE pnl_blob IS NOT NULL")
        results = []
        for row in c.fetchall():
            blob = row[0]
            if blob:
                results.append(np.frombuffer(blob, dtype=np.float64))
        return results

    def check_diversity_against_db(self, pnl_vec: np.ndarray, corr_cutoff: float = 0.7) -> bool:
        """Check if new PnL vector is sufficiently uncorrelated with ALL DB alphas."""
        existing_vectors = self.get_all_pnl_vectors()
        if not existing_vectors:
            return True
        for existing_pnl in existing_vectors:
            try:
                min_len = min(len(pnl_vec), len(existing_pnl))
                if min_len < 20:
                    continue
                corr = np.corrcoef(pnl_vec[:min_len], existing_pnl[:min_len])[0, 1]
                if not np.isnan(corr) and abs(corr) > corr_cutoff:
                    return False
            except Exception:
                continue
        return True

    def update_alpha_status(self, alpha_id: int, status: str,
                            wq_alpha_id: str | None = None) -> None:
        if wq_alpha_id:
            self._conn.execute(
                "UPDATE alphas SET status = ?, wq_alpha_id = ? WHERE alpha_id = ?",
                (status, wq_alpha_id, alpha_id))
        else:
            self._conn.execute(
                "UPDATE alphas SET status = ? WHERE alpha_id = ?", (status, alpha_id))
        self._conn.commit()

    def get_alpha(self, alpha_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM alphas WHERE alpha_id = ?", (alpha_id,)
        ).fetchone()
        return dict(row) if row else None

    def alpha_exists(self, expression: str, params: dict | None = None) -> bool:
        """Check if this exact expression + params combo already exists."""
        params_json = json.dumps(params or {}, sort_keys=True)
        row = self._conn.execute(
            "SELECT 1 FROM alphas WHERE expression = ? AND params_json = ?",
            (expression, params_json)
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Evaluations
    # ------------------------------------------------------------------

    def insert_evaluation(self, alpha_id: int, source: str = "local",
                          sharpe: float | None = None,
                          fitness: float | None = None,
                          turnover: float | None = None,
                          max_drawdown: float | None = None,
                          returns_ann: float | None = None,
                          profit_dollar: float | None = None,
                          margin_bps: float | None = None,
                          passed_checks: int | None = None,
                          error: str | None = None,
                          metrics: dict | None = None) -> int:
        c = self._conn.cursor()
        c.execute("""
            INSERT INTO evaluations (alpha_id, source, sharpe, fitness, turnover,
                                    max_drawdown, returns_ann, profit_dollar,
                                    margin_bps, passed_checks, error, metrics_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (alpha_id, source, sharpe, fitness, turnover, max_drawdown,
              returns_ann, profit_dollar, margin_bps, passed_checks, error,
              json.dumps(metrics or {})))
        self._conn.commit()
        return c.lastrowid

    # ------------------------------------------------------------------
    # Queries for LLM learning
    # ------------------------------------------------------------------

    def get_recent_alphas(self, limit: int = 30, run_id: int | None = None) -> List[Dict]:
        """Get recent alphas with their best evaluation.

        Returns list of dicts with keys: alpha_id, expression, sharpe, fitness,
        status, reasoning, params_json.
        """
        q = """
            SELECT a.alpha_id, a.expression, a.reasoning, a.params_json,
                   a.status, a.source, a.run_id, a.trial_index,
                   e.sharpe, e.fitness, e.turnover, e.error
            FROM alphas a
            LEFT JOIN evaluations e ON a.alpha_id = e.alpha_id
                AND e.eval_id = (SELECT MAX(eval_id) FROM evaluations WHERE alpha_id = a.alpha_id)
        """
        params = []
        if run_id is not None:
            q += " WHERE a.run_id = ?"
            params.append(run_id)
        q += " ORDER BY a.created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(q, params).fetchall()
        return [dict(r) for r in rows]

    def get_top_alphas(self, metric: str = "fitness", limit: int = 20,
                       min_sharpe: float | None = None) -> List[Dict]:
        """Get top alphas by metric (fitness or sharpe)."""
        q = """
            SELECT a.alpha_id, a.expression, a.params_json, a.reasoning,
                   a.source, a.created_at,
                   e.sharpe, e.fitness, e.turnover, e.max_drawdown,
                   e.returns_ann, e.margin_bps, e.passed_checks
            FROM alphas a
            JOIN evaluations e ON a.alpha_id = e.alpha_id
            WHERE e.error IS NULL AND e.{col} IS NOT NULL
        """.format(col=metric)
        params = []
        if min_sharpe is not None:
            q += " AND e.sharpe >= ?"
            params.append(min_sharpe)
        q += f" ORDER BY e.{metric} DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(q, params).fetchall()
        return [dict(r) for r in rows]

    def get_history_for_prompt(self, limit: int = 20,
                               run_id: int | None = None) -> List[Dict[str, Any]]:
        """Format recent history exactly like enhanced_gemini_agent expects.

        Returns list of dicts with: alpha_id, fitness, sharpe, code.
        """
        alphas = self.get_recent_alphas(limit=limit, run_id=run_id)
        history = []
        for a in reversed(alphas):  # chronological order
            history.append({
                "alpha_id": a.get("wq_alpha_id") or f"local_{a['alpha_id']}",
                "fitness": a.get("fitness") or 0.0,
                "sharpe": a.get("sharpe") or 0.0,
                "code": a["expression"],
            })
        return history

    def get_failed_patterns(self, limit: int = 50) -> List[str]:
        """Get expressions that failed, for the LLM to avoid."""
        rows = self._conn.execute("""
            SELECT DISTINCT a.expression
            FROM alphas a
            JOIN evaluations e ON a.alpha_id = e.alpha_id
            WHERE e.error IS NOT NULL
            ORDER BY a.created_at DESC LIMIT ?
        """, (limit,)).fetchall()
        return [r["expression"] for r in rows]

    def count_alphas(self, run_id: int | None = None) -> int:
        if run_id is not None:
            return self._conn.execute(
                "SELECT COUNT(*) FROM alphas WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
        return self._conn.execute("SELECT COUNT(*) FROM alphas").fetchone()[0]

    def get_stats(self) -> Dict[str, Any]:
        """Aggregate stats for the database."""
        total = self.count_alphas()
        c = self._conn
        successful = c.execute("""
            SELECT COUNT(DISTINCT a.alpha_id) FROM alphas a
            JOIN evaluations e ON a.alpha_id = e.alpha_id
            WHERE e.sharpe > 0 AND e.error IS NULL
        """).fetchone()[0]
        top_sharpe = c.execute("""
            SELECT MAX(e.sharpe) FROM evaluations e WHERE e.error IS NULL
        """).fetchone()[0]
        top_fitness = c.execute("""
            SELECT MAX(e.fitness) FROM evaluations e WHERE e.error IS NULL
        """).fetchone()[0]
        return {
            "total_alphas": total,
            "successful_alphas": successful,
            "top_sharpe": top_sharpe,
            "top_fitness": top_fitness,
        }

    def close(self) -> None:
        self._conn.close()

    def __del__(self):
        try:
            self._conn.close()
        except Exception:
            pass
