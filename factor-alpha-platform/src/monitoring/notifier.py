"""Critical-alert pager: posts open critical alerts to a Discord webhook.

Designed for the Phase 0 reliability plan from docs/RELIABILITY_PLAN.md.
The existing alert pipeline writes to SQLite but nothing pages -- this module
adds the missing wakeup channel.

Wiring:
  1. Set DISCORD_ALERT_WEBHOOK_URL in deploy/dagster/.env (a Discord channel
     webhook URL: Server -> Channel -> Edit -> Integrations -> Webhooks -> Copy).
  2. The `critical_alert_pager_sensor` in src.orchestration.dagster_defs fires
     every 60s, scans for open critical alerts that have never been notified
     OR whose last notification is older than `re_notify_after_min` (default
     60). For each, it POSTs to the webhook and stamps `alerts.last_notified_at`.

No webhook URL = no-op (the function returns 0 notified; safe to ship).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _format_for_discord(alert: dict[str, Any]) -> dict[str, Any]:
    """Build a Discord webhook payload for an alert row."""
    sev = (alert.get("severity") or "warning").upper()
    market = alert.get("market") or "unknown"
    strategy = alert.get("strategy_id") or "—"
    msg = (alert.get("message") or "").strip()
    created = alert.get("created_at") or ""

    color = {
        "critical": 0xFF0000,
        "warning": 0xFFAA00,
        "info": 0x3399FF,
    }.get(alert.get("severity"), 0x999999)

    embed = {
        "title": f"[{sev}] {market}: {msg.split(':')[0][:80] if ':' in msg else msg[:80]}",
        "description": msg[:3500],
        "color": color,
        "timestamp": created,
        "fields": [
            {"name": "Market", "value": market, "inline": True},
            {"name": "Strategy", "value": strategy, "inline": True},
            {"name": "Alert ID", "value": str(alert.get("id", "?")), "inline": True},
            {"name": "Created (UTC)", "value": created, "inline": False},
        ],
        "footer": {"text": "factor-alpha-platform critical-alert-pager"},
    }
    return {"embeds": [embed]}


def _post_discord(webhook_url: str, payload: dict[str, Any], timeout_sec: int = 8) -> bool:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "factor-alpha-pager/1.0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as e:
        # 204 No Content is success but urllib treats it as 204; some webhook
        # variants return non-2xx with informational reasons.
        return 200 <= e.code < 300
    except Exception:
        return False


def notify_open_critical_alerts(
    *,
    db_path: str | Path,
    webhook_url: str | None = None,
    severities: tuple[str, ...] = ("critical",),
    re_notify_after_min: int = 60,
) -> dict[str, Any]:
    """Page each unnotified or stale open alert. Returns a small report dict.

    Designed to be called from a Dagster sensor every ~60s. Idempotent within
    `re_notify_after_min`: alerts already notified within that window are
    skipped. Safe to call even with no webhook configured (returns counts).
    """
    webhook_url = webhook_url or os.environ.get("DISCORD_ALERT_WEBHOOK_URL", "")
    cutoff = (
        dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=re_notify_after_min)
    ).isoformat(timespec="seconds")
    placeholders = ",".join("?" for _ in severities)
    sql = f"""
        SELECT id, strategy_id, market, severity, message, metadata_json,
               created_at, resolved_at, last_notified_at
        FROM alerts
        WHERE status = 'open'
          AND severity IN ({placeholders})
          AND (last_notified_at IS NULL OR last_notified_at < ?)
        ORDER BY created_at ASC
    """
    db_path = str(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(sql, (*severities, cutoff)).fetchall()]

    report = {
        "candidates": len(rows),
        "notified": 0,
        "failed": 0,
        "skipped_no_webhook": 0,
        "alert_ids_notified": [],
    }
    if not rows:
        return report
    if not webhook_url:
        report["skipped_no_webhook"] = len(rows)
        return report

    now = _utc_iso()
    with sqlite3.connect(db_path) as conn:
        for row in rows:
            payload = _format_for_discord(row)
            ok = _post_discord(webhook_url, payload)
            if ok:
                conn.execute(
                    "UPDATE alerts SET last_notified_at = ? WHERE id = ?",
                    (now, row["id"]),
                )
                report["notified"] += 1
                report["alert_ids_notified"].append(row["id"])
            else:
                report["failed"] += 1
        conn.commit()
    return report
