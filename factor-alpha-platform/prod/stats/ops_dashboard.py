"""Generate a consolidated production operations dashboard.

This is a static HTML/JSON dashboard for the current single-box deployment
stage. It reads:
  - strategy_state, data_integrity_checks, alerts from data/prod_state.db
  - exchange performance CSVs from prod/logs/**/performance

Usage:
    python -m src.data.integrity --market all --write-db
    python prod/stats/ops_dashboard.py
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.state_store import (
    StrategyState,
    latest_checks,
    open_alerts,
    strategy_states,
    sync_alerts_from_latest_checks,
    upsert_strategy_state,
)
from src.execution.ib_gateway_health import check_ib_gateway


LOGS_ROOT = PROJECT_ROOT / "prod" / "logs"
OUT_DIR = PROJECT_ROOT / "prod" / "stats" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCHANGES = {
    "ib": LOGS_ROOT / "performance",
    "binance": LOGS_ROOT / "binance" / "performance",
    "kucoin": LOGS_ROOT / "kucoin" / "performance",
}

TRADE_LOG_DIRS = {
    "ib": LOGS_ROOT / "trades",
    "binance": LOGS_ROOT / "binance" / "trades",
    "kucoin": LOGS_ROOT / "kucoin" / "trades",
}


def _latest_csv(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    files = sorted(directory.glob("equity_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _config_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _exchange_enabled(exchange: str) -> tuple[bool, str | None]:
    if exchange == "ib":
        return True, None
    config_path = PROJECT_ROOT / "prod" / "config" / f"{exchange}.json"
    config = _read_json(config_path)
    execution = config.get("execution") or {}
    if execution.get("enabled") is False:
        return False, execution.get("disabled_reason") or "disabled in exchange config"
    return True, None


def _latest_json(directory: Path, pattern: str = "trade_*.json",
                 *, exclude_suffix: str | None = None) -> Path | None:
    if not directory.exists():
        return None
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if exclude_suffix:
        files = [p for p in files if not p.name.endswith(exclude_suffix)]
    return files[0] if files else None


def _last_parquet_index(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if df.empty:
        return None
    return str(df.index[-1])


def _market_status(checks: list[dict[str, Any]], market: str) -> str:
    market_checks = [c for c in checks if c.get("market") == market]
    if not market_checks:
        return "unknown"
    statuses = {str(c.get("status", "")).lower() for c in market_checks}
    if "fail" in statuses:
        return "blocked"
    if "warn" in statuses:
        return "warning"
    return "ready"


def _position_stats(position_map: dict[str, Any] | None) -> tuple[float | None, float | None, int | None]:
    if not position_map:
        return None, None, None
    values: list[float] = []
    for raw in position_map.values():
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    if not values:
        return None, None, None
    gross = float(sum(abs(v) for v in values))
    net = float(sum(values))
    n_positions = int(sum(1 for v in values if abs(v) > 1e-9))
    return gross, net, n_positions


def _portfolio_summary_stats(data: dict[str, Any]) -> tuple[float | None, float | None, int | None]:
    summary = data.get("portfolio_summary") or data.get("portfolio") or {}
    n_long = summary.get("n_long")
    n_short = summary.get("n_short")
    n_positions = None
    if n_long is not None or n_short is not None:
        try:
            n_positions = int(n_long or 0) + int(n_short or 0)
        except (TypeError, ValueError):
            n_positions = None

    gross = summary.get("gross_exposure") or summary.get("target_gmv")
    if gross is None:
        gross = (data.get("config") or {}).get("target_gmv")
    net = summary.get("net_exposure")

    try:
        gross_float = None if gross is None else float(gross)
    except (TypeError, ValueError):
        gross_float = None
    try:
        net_float = None if net is None else float(net)
    except (TypeError, ValueError):
        net_float = None
    return gross_float, net_float, n_positions


def _equity_curve(df: pd.DataFrame, *, max_points: int = 360) -> list[dict[str, Any]]:
    pnl = df.get("pnl_bar", pd.Series(dtype=float))
    cum = df.get("cumulative_pnl", pnl.cumsum())
    if len(cum) == 0:
        return []

    curve = pd.DataFrame({"pnl": pd.to_numeric(cum, errors="coerce")})
    time_col = "bar_time" if "bar_time" in df and not df["bar_time"].isna().all() else "timestamp"
    if time_col in df:
        curve["timestamp"] = pd.to_datetime(df[time_col], errors="coerce").astype(str)
    else:
        curve["timestamp"] = [str(i) for i in range(len(curve))]
    curve = curve.dropna(subset=["pnl"])
    if curve.empty:
        return []
    if len(curve) > max_points:
        idx = np.unique(np.linspace(0, len(curve) - 1, max_points).round().astype(int))
        curve = curve.iloc[idx]
    return [
        {"timestamp": str(row["timestamp"]), "pnl": float(row["pnl"])}
        for row in curve.to_dict("records")
    ]


def _performance_with_bar_times(exchange: str, df: pd.DataFrame) -> pd.DataFrame:
    """Attach trade-log bar_time to legacy performance CSV rows when possible."""
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    if "bar_time" in out.columns:
        out["bar_time"] = pd.to_datetime(out["bar_time"], errors="coerce")
        return out

    trade_dir = TRADE_LOG_DIRS.get(exchange)
    if trade_dir is None or not trade_dir.exists():
        out["bar_time"] = pd.NaT
        return out

    rows = []
    for path in trade_dir.glob("trade_*.json"):
        data = _read_json(path)
        if not data.get("timestamp") or not data.get("bar_time"):
            continue
        rows.append(
            {
                "timestamp": pd.to_datetime(data["timestamp"], errors="coerce"),
                "bar_time": pd.to_datetime(data["bar_time"], errors="coerce"),
            }
        )
    if not rows:
        out["bar_time"] = pd.NaT
        return out

    trade_df = pd.DataFrame(rows).dropna(subset=["timestamp"])
    trade_df = trade_df.drop_duplicates("timestamp", keep="last")
    out = out.merge(trade_df, on="timestamp", how="left")
    return out


def _dedupe_performance(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return one row per exchange bar plus duplicate/missing-bar diagnostics."""
    raw_rows = int(len(df))
    out = df.sort_values("timestamp").copy()
    diagnostics: dict[str, Any] = {
        "raw_rows": raw_rows,
        "unique_bars": raw_rows,
        "duplicate_bar_rows": 0,
        "duplicate_bar_times": 0,
        "missing_4h_bars": None,
        "max_bar_gap_hours": None,
        "dedupe_rule": "none",
    }

    if "bar_time" not in out.columns or out["bar_time"].isna().all():
        return out, diagnostics

    valid_bar = out.dropna(subset=["bar_time"]).copy()
    duplicate_mask = valid_bar.duplicated("bar_time", keep="first")
    diagnostics["duplicate_bar_rows"] = int(duplicate_mask.sum())
    diagnostics["duplicate_bar_times"] = int(
        valid_bar.loc[valid_bar.duplicated("bar_time", keep=False), "bar_time"].nunique()
    )
    diagnostics["dedupe_rule"] = "first row per bar_time"

    clean = valid_bar.drop_duplicates("bar_time", keep="first").sort_values("bar_time").copy()
    diagnostics["unique_bars"] = int(len(clean))

    if len(clean) >= 2:
        diffs = clean["bar_time"].diff().dropna()
        diagnostics["max_bar_gap_hours"] = float(diffs.max().total_seconds() / 3600)
        start = clean["bar_time"].iloc[0]
        end = clean["bar_time"].iloc[-1]
        expected = int(((end - start).total_seconds() // (4 * 3600)) + 1)
        diagnostics["missing_4h_bars"] = max(0, expected - len(clean))

    if "pnl_bar" in clean:
        clean["cumulative_pnl"] = pd.to_numeric(clean["pnl_bar"], errors="coerce").fillna(0.0).cumsum()
    return clean, diagnostics


def _performance_summary(exchange: str, path: Path | None) -> dict[str, Any]:
    enabled, disabled_reason = _exchange_enabled(exchange)
    if not enabled:
        return {
            "exchange": exchange,
            "status": "disabled",
            "message": disabled_reason,
            "path": str(path) if path else None,
            "curve": [],
        }
    if path is None:
        return {"exchange": exchange, "status": "missing", "message": "no equity CSV", "curve": []}
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
    except Exception as exc:
        return {"exchange": exchange, "status": "error", "message": str(exc), "path": str(path), "curve": []}
    if df.empty:
        return {"exchange": exchange, "status": "empty", "path": str(path), "curve": []}

    df = _performance_with_bar_times(exchange, df)
    clean_df, diagnostics = _dedupe_performance(df)
    pnl = clean_df.get("pnl_bar", pd.Series(dtype=float)).astype(float)
    cum = clean_df.get("cumulative_pnl", pnl.cumsum()).astype(float)
    raw_pnl = df.get("pnl_bar", pd.Series(dtype=float)).astype(float)
    bars_per_year = 252 if exchange == "ib" else 6 * 365
    sharpe = float(pnl.mean() / pnl.std() * np.sqrt(bars_per_year)) if pnl.std() > 0 else 0.0
    gmv = clean_df.get("gmv", pd.Series(dtype=float))
    n_positions = clean_df.get("n_positions", pd.Series(dtype=float))
    status = "warn" if diagnostics["duplicate_bar_rows"] or diagnostics["missing_4h_bars"] else "ok"
    return {
        "exchange": exchange,
        "status": status,
        "path": str(path),
        "last_timestamp": str(df["timestamp"].iloc[-1]) if "timestamp" in df and len(df) else None,
        "last_bar_time": str(clean_df["bar_time"].iloc[-1]) if "bar_time" in clean_df and len(clean_df) else None,
        "n_bars": int(len(clean_df)),
        "raw_rows": diagnostics["raw_rows"],
        "unique_bars": diagnostics["unique_bars"],
        "duplicate_bar_rows": diagnostics["duplicate_bar_rows"],
        "duplicate_bar_times": diagnostics["duplicate_bar_times"],
        "missing_4h_bars": diagnostics["missing_4h_bars"],
        "max_bar_gap_hours": diagnostics["max_bar_gap_hours"],
        "raw_total_pnl": float(raw_pnl.sum()) if len(raw_pnl) else 0.0,
        "total_pnl": float(cum.iloc[-1]) if len(cum) else 0.0,
        "sharpe": sharpe,
        "max_drawdown": float((cum - cum.cummax()).min()) if len(cum) else 0.0,
        "avg_gmv": float(gmv.mean()) if len(gmv) else None,
        "avg_positions": float(n_positions.mean()) if len(n_positions) else None,
        "curve": _equity_curve(clean_df),
        "dedupe_rule": diagnostics["dedupe_rule"],
    }


def _sync_ib_strategy_state(state_db: Path, checks: list[dict[str, Any]]) -> None:
    config_path = PROJECT_ROOT / "prod" / "config" / "strategy.json"
    config = _read_json(config_path)
    matrices_path = PROJECT_ROOT / "data" / "fmp_cache" / "matrices" / "close.parquet"
    trade_path = _latest_json(TRADE_LOG_DIRS["ib"], exclude_suffix="_dry_run.json")
    trade = _read_json(trade_path)
    dashboard_client_id = int(os.environ.get("IB_CLIENT_ID_DASHBOARD_HEALTHCHECK", "29"))
    gateway_status = check_ib_gateway(config, client_id=dashboard_client_id).to_dict()

    gross, net, n_positions = _portfolio_summary_stats(trade)
    status = _market_status(checks, "equity")
    if not gateway_status.get("connected"):
        status = "unreachable"
    elif trade:
        trade_date = str(trade.get("date") or "")[:10]
        today = datetime.now().date().isoformat()
        if trade_date == today:
            status = "orders_submitted" if int(trade.get("n_orders") or 0) else "no_orders"

    ibkr = config.get("ibkr", {})
    state = StrategyState(
        strategy_id="ib_moc_equity",
        market="equity",
        mode="paper" if ibkr.get("port_paper") else "unknown",
        status=status,
        last_data_bar=_last_parquet_index(matrices_path),
        last_signal_bar=trade.get("signal_date"),
        last_trade_time=trade.get("timestamp"),
        config_hash=_config_hash(config_path),
        gross_exposure=gross,
        net_exposure=net,
        n_positions=n_positions,
        metrics_json={
            "source": "ops_dashboard_file_sync",
            "strategy": (config.get("strategy") or {}).get("name"),
            "universe": (config.get("strategy") or {}).get("universe"),
            "min_alpha_sharpe": (config.get("strategy") or {}).get("min_alpha_sharpe"),
            "ib_port_paper": ibkr.get("port_paper"),
            "ib_port_live": ibkr.get("port_live"),
            "latest_trade_log": str(trade_path) if trade_path else None,
            "target_gmv": (config.get("account") or {}).get("target_gmv"),
            "ib_gateway_connected": bool(gateway_status.get("connected")),
            "ib_gateway_host": gateway_status.get("host"),
            "ib_gateway_port": gateway_status.get("port"),
            "ib_gateway_mode": gateway_status.get("mode"),
            "ib_gateway_message": gateway_status.get("message"),
            "ib_gateway_checked_at_utc": gateway_status.get("checked_at_utc"),
            "ib_gateway_error_type": gateway_status.get("error_type"),
            "ib_gateway_accounts_count": len(gateway_status.get("accounts") or []),
        },
    )
    upsert_strategy_state(state, state_db)


def _sync_kucoin_strategy_state(state_db: Path, checks: list[dict[str, Any]]) -> None:
    config_path = PROJECT_ROOT / "prod" / "config" / "kucoin.json"
    config = _read_json(config_path)
    matrices_path = PROJECT_ROOT / "data" / "kucoin_cache" / "matrices" / "4h" / "prod" / "close.parquet"
    trade_path = _latest_json(TRADE_LOG_DIRS["kucoin"])
    trade = _read_json(trade_path)

    positions = trade.get("positions") or {}
    gross, net, n_positions = _position_stats(positions.get("after"))
    if gross is None:
        gross, net, n_positions = _portfolio_summary_stats(trade)

    status = _market_status(checks, "crypto")
    if trade.get("timestamp"):
        status = "traded" if status == "ready" else status

    state = StrategyState(
        strategy_id="crypto_4h",
        market="crypto",
        mode=str(trade.get("mode") or ("paper" if (config.get("execution") or {}).get("paper_mode") else "live")).lower(),
        status=status,
        last_data_bar=_last_parquet_index(matrices_path),
        last_signal_bar=trade.get("bar_time") or trade.get("signal_date"),
        last_trade_time=trade.get("timestamp"),
        config_hash=_config_hash(config_path),
        gross_exposure=gross,
        net_exposure=net,
        n_positions=n_positions,
        metrics_json={
            "source": "ops_dashboard_file_sync",
            "exchange": "kucoin",
            "strategy": (config.get("strategy") or {}).get("name"),
            "latest_trade_log": str(trade_path) if trade_path else None,
            "target_gmv": (config.get("account") or {}).get("target_gmv"),
        },
    )
    upsert_strategy_state(state, state_db)


def _sync_binance_strategy_state(state_db: Path) -> None:
    config_path = PROJECT_ROOT / "prod" / "config" / "binance.json"
    config = _read_json(config_path)
    matrices_path = PROJECT_ROOT / "data" / "binance_cache" / "matrices" / "4h" / "prod" / "close.parquet"
    trade_path = _latest_json(TRADE_LOG_DIRS["binance"])
    trade = _read_json(trade_path)

    positions = trade.get("positions") or {}
    gross, net, n_positions = _position_stats(positions.get("after"))
    if gross is None:
        gross, net, n_positions = _portfolio_summary_stats(trade)

    last_data_bar = _last_parquet_index(matrices_path)
    enabled, disabled_reason = _exchange_enabled("binance")
    status = "unknown"
    if not enabled:
        status = "disabled"
    elif last_data_bar:
        try:
            age_hours = (
                pd.Timestamp.now("UTC").tz_localize(None) - pd.Timestamp(last_data_bar)
            ).total_seconds() / 3600
            status = "stale" if age_hours > 12 else "ready"
        except Exception:
            status = "unknown"
    if trade.get("bar_time") and trade.get("bar_time") == last_data_bar:
        status = "traded" if status == "ready" else status

    state = StrategyState(
        strategy_id="binance_4h",
        market="crypto",
        mode=str(trade.get("mode") or ("paper" if (config.get("execution") or {}).get("paper_mode") else "live")).lower(),
        status=status,
        last_data_bar=last_data_bar,
        last_signal_bar=trade.get("bar_time") or trade.get("signal_date"),
        last_trade_time=trade.get("timestamp"),
        config_hash=_config_hash(config_path),
        gross_exposure=gross,
        net_exposure=net,
        n_positions=n_positions,
        metrics_json={
            "source": "ops_dashboard_file_sync",
            "exchange": "binance",
            "strategy": (config.get("strategy") or {}).get("name"),
            "latest_trade_log": str(trade_path) if trade_path else None,
            "target_gmv": (config.get("account") or {}).get("target_gmv"),
            "enabled": enabled,
            "disabled_reason": disabled_reason,
        },
    )
    upsert_strategy_state(state, state_db)


def _sync_strategy_states_from_files(state_db: Path, checks: list[dict[str, Any]]) -> None:
    _sync_ib_strategy_state(state_db, checks)
    _sync_binance_strategy_state(state_db)
    _sync_kucoin_strategy_state(state_db, checks)


def build_payload(state_db: Path) -> dict[str, Any]:
    sync_alerts_from_latest_checks(state_db)
    checks = latest_checks(db_path=state_db)
    _sync_strategy_states_from_files(state_db, checks)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy_states": strategy_states(state_db),
        "integrity_checks": checks,
        "alerts": open_alerts(state_db),
        "performance": [
            _performance_summary(exchange, _latest_csv(directory))
            for exchange, directory in EXCHANGES.items()
        ],
    }


def _badge(status: str) -> str:
    cls = {
        "pass": "ok",
        "ok": "ok",
        "ready": "ok",
        "signal_ready": "ok",
        "traded": "ok",
        "orders_submitted": "ok",
        "no_orders": "ok",
        "warn": "warn",
        "warning": "warn",
        "stale": "warn",
        "unknown": "warn",
        "disabled": "neutral",
        "unreachable": "fail",
        "fail": "fail",
        "blocked": "fail",
        "critical": "fail",
        "missing": "warn",
        "error": "fail",
        "empty": "warn",
    }.get(str(status).lower(), "neutral")
    return f'<span class="badge {cls}">{html.escape(str(status).upper())}</span>'


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:,.3f}"
    return html.escape(str(value))


def _money(value: Any) -> str:
    try:
        return f"${float(value):,.0f}"
    except (TypeError, ValueError):
        return ""


def _svg_curve(curve: list[dict[str, Any]]) -> str:
    if not curve:
        return '<div class="empty-chart">No performance equity CSV found.</div>'

    width = 520
    height = 180
    pad_x = 32
    pad_y = 22
    values = [float(p["pnl"]) for p in curve]
    y_min = min(values)
    y_max = max(values)
    if y_min == y_max:
        spread = max(abs(y_min) * 0.05, 1.0)
        y_min -= spread
        y_max += spread

    x_span = width - 2 * pad_x
    y_span = height - 2 * pad_y
    points = []
    for i, value in enumerate(values):
        x = pad_x + (i / max(len(values) - 1, 1)) * x_span
        y = height - pad_y - ((value - y_min) / (y_max - y_min)) * y_span
        points.append(f"{x:.1f},{y:.1f}")

    zero_line = ""
    if y_min < 0 < y_max:
        y_zero = height - pad_y - ((0 - y_min) / (y_max - y_min)) * y_span
        zero_line = f'<line class="zero-line" x1="{pad_x}" y1="{y_zero:.1f}" x2="{width - pad_x}" y2="{y_zero:.1f}" />'

    start = html.escape(str(curve[0].get("timestamp", ""))[:16])
    end = html.escape(str(curve[-1].get("timestamp", ""))[:16])
    last = _money(values[-1])
    top = _money(y_max)
    bottom = _money(y_min)

    return f"""
      <svg class="curve" viewBox="0 0 {width} {height}" role="img" aria-label="Equity curve">
        <line class="axis" x1="{pad_x}" y1="{height - pad_y}" x2="{width - pad_x}" y2="{height - pad_y}" />
        <line class="axis" x1="{pad_x}" y1="{pad_y}" x2="{pad_x}" y2="{height - pad_y}" />
        {zero_line}
        <polyline class="curve-line" fill="none" points="{' '.join(points)}" />
        <text x="{pad_x}" y="14">{top}</text>
        <text x="{pad_x}" y="{height - pad_y - 5}">{bottom}</text>
        <text x="{width - pad_x}" y="14" text-anchor="end">{last}</text>
        <text x="{pad_x}" y="{height - 4}" class="date-label">{start}</text>
        <text x="{width - pad_x}" y="{height - 4}" text-anchor="end" class="date-label">{end}</text>
      </svg>
    """


def _curve_panel(performance: dict[str, Any]) -> str:
    path = performance.get("path")
    path_label = html.escape(Path(path).name if path else performance.get("message", ""))
    meta = " | ".join(
        part for part in [
            f"Unique bars {_fmt(performance.get('unique_bars'))}" if performance.get("unique_bars") is not None else "",
            f"Raw rows {_fmt(performance.get('raw_rows'))}" if performance.get("raw_rows") is not None else "",
            f"Dup rows {_fmt(performance.get('duplicate_bar_rows'))}" if performance.get("duplicate_bar_rows") else "",
            f"PnL {_money(performance.get('total_pnl'))}" if performance.get("total_pnl") is not None else "",
            f"SR {_fmt(performance.get('sharpe'))}" if performance.get("sharpe") is not None else "",
        ] if part
    )
    return f"""
      <article class="chart-panel">
        <div class="chart-head">
          <div>
            <div class="chart-title">{html.escape(performance['exchange'].upper())}</div>
            <div class="chart-path">{path_label}</div>
          </div>
          {_badge(performance['status'])}
        </div>
        {_svg_curve(performance.get('curve', []))}
        <div class="chart-meta">{html.escape(meta)}</div>
      </article>
    """


def render_html(payload: dict[str, Any]) -> str:
    checks = payload["integrity_checks"]
    alerts = payload["alerts"]
    states = payload["strategy_states"]
    performance = payload["performance"]
    curve_panels = "\n".join(_curve_panel(p) for p in performance)

    alert_rows = "\n".join(
        f"<tr><td>{_badge(a['severity'])}</td><td>{html.escape(str(a.get('market') or ''))}</td>"
        f"<td>{html.escape(a['message'])}</td><td>{html.escape(a['created_at'])}</td></tr>"
        for a in alerts
    ) or "<tr><td colspan='4'>No open alerts.</td></tr>"

    state_rows = "\n".join(
        f"<tr><td>{html.escape(s['strategy_id'])}</td><td>{html.escape(s['market'])}</td>"
        f"<td>{_fmt(s.get('mode'))}</td><td>{_badge(s['status'])}</td><td>{_fmt(s['last_data_bar'])}</td>"
        f"<td>{_fmt(s['last_signal_bar'])}</td><td>{_fmt(s['gross_exposure'])}</td>"
        f"<td>{_fmt(s['net_exposure'])}</td><td>{_fmt(s['n_positions'])}</td></tr>"
        for s in states
    ) or "<tr><td colspan='9'>No strategy state has been recorded yet.</td></tr>"

    check_rows = "\n".join(
        f"<tr><td>{html.escape(c['market'])}</td><td>{html.escape(c['check_name'])}</td>"
        f"<td>{_badge(c['status'])}</td><td>{html.escape(c['message'])}</td>"
        f"<td>{_fmt(c['value'])}</td><td>{_fmt(c['threshold'])}</td>"
        f"<td>{html.escape(c['checked_at'])}</td></tr>"
        for c in checks
    ) or "<tr><td colspan='7'>No integrity checks recorded.</td></tr>"

    perf_rows = "\n".join(
        f"<tr><td>{html.escape(p['exchange'])}</td><td>{_badge(p['status'])}</td>"
        f"<td>{_fmt(p.get('last_timestamp'))}</td><td>{_fmt(p.get('last_bar_time'))}</td>"
        f"<td>{_fmt(p.get('raw_rows'))}</td><td>{_fmt(p.get('unique_bars'))}</td>"
        f"<td>{_fmt(p.get('duplicate_bar_rows'))}</td><td>{_fmt(p.get('missing_4h_bars'))}</td>"
        f"<td>{_fmt(p.get('total_pnl'))}</td><td>{_fmt(p.get('raw_total_pnl'))}</td>"
        f"<td>{_fmt(p.get('sharpe'))}</td><td>{_fmt(p.get('max_drawdown'))}</td>"
        f"<td>{_fmt(p.get('avg_gmv'))}</td><td>{_fmt(p.get('avg_positions'))}</td></tr>"
        for p in performance
    )

    fail_count = sum(1 for c in checks if c["status"] == "fail")
    warn_count = sum(1 for c in checks if c["status"] == "warn")
    open_alert_count = len(alerts)
    status = "fail" if fail_count else "warn" if warn_count or open_alert_count else "ok"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Factor Alpha Ops Dashboard</title>
  <style>
    :root {{ color-scheme: dark; font-family: Inter, Segoe UI, Arial, sans-serif; }}
    body {{ margin: 0; background: #101214; color: #e7ecef; }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 24px; }}
    header {{ display: flex; justify-content: space-between; gap: 16px; align-items: baseline; }}
    h1 {{ margin: 0; font-size: 28px; font-weight: 700; }}
    h2 {{ margin: 28px 0 10px; font-size: 17px; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; margin-top: 18px; }}
    .tile {{ border: 1px solid #2d3439; border-radius: 6px; padding: 14px; background: #171b1f; }}
    .label {{ color: #9aa7af; font-size: 12px; }}
    .value {{ margin-top: 6px; font-size: 24px; font-weight: 700; }}
    .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 12px; }}
    .chart-panel {{ border: 1px solid #2d3439; border-radius: 6px; padding: 12px; background: #15191d; min-width: 0; }}
    .chart-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; margin-bottom: 8px; }}
    .chart-title {{ font-weight: 700; letter-spacing: 0; }}
    .chart-path, .chart-meta {{ color: #9aa7af; font-size: 12px; overflow-wrap: anywhere; }}
    .curve {{ width: 100%; height: auto; display: block; background: #11161a; border: 1px solid #263036; border-radius: 4px; }}
    .curve text {{ fill: #aeb8bf; font-size: 10px; }}
    .curve .date-label {{ fill: #78858e; }}
    .axis {{ stroke: #303940; stroke-width: 1; }}
    .zero-line {{ stroke: #52606a; stroke-dasharray: 3 4; stroke-width: 1; }}
    .curve-line {{ stroke: #6bbcff; stroke-width: 2.2; stroke-linejoin: round; stroke-linecap: round; }}
    .empty-chart {{ display: grid; place-items: center; min-height: 180px; color: #9aa7af; background: #11161a; border: 1px solid #263036; border-radius: 4px; font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; background: #15191d; border: 1px solid #2b3338; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid #263036; text-align: left; font-size: 13px; vertical-align: top; }}
    th {{ color: #aeb8bf; font-weight: 600; background: #1b2025; }}
    .badge {{ display: inline-block; min-width: 54px; padding: 3px 7px; border-radius: 4px; font-size: 11px; text-align: center; font-weight: 700; }}
    .ok {{ background: #164a34; color: #8df0bd; }}
    .warn {{ background: #574316; color: #ffd36e; }}
    .fail {{ background: #5a1d25; color: #ff99a8; }}
    .neutral {{ background: #303940; color: #cbd5dc; }}
    .generated {{ color: #9aa7af; font-size: 12px; }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>Factor Alpha Ops Dashboard</h1>
    <div>{_badge(status)}</div>
  </header>
  <div class="generated">Generated {html.escape(payload['generated_at'])}</div>
  <section class="summary">
    <div class="tile"><div class="label">Open Alerts</div><div class="value">{open_alert_count}</div></div>
    <div class="tile"><div class="label">Failed Checks</div><div class="value">{fail_count}</div></div>
    <div class="tile"><div class="label">Warning Checks</div><div class="value">{warn_count}</div></div>
    <div class="tile"><div class="label">Strategies</div><div class="value">{len(states)}</div></div>
  </section>
  <h2>Equity Curves</h2>
  <section class="charts">{curve_panels}</section>
  <h2>Strategy State</h2>
  <table><thead><tr><th>Strategy</th><th>Market</th><th>Mode</th><th>Status</th><th>Last Data</th><th>Last Signal</th><th>Gross</th><th>Net</th><th>Positions</th></tr></thead><tbody>{state_rows}</tbody></table>
  <h2>Open Alerts</h2>
  <table><thead><tr><th>Severity</th><th>Market</th><th>Message</th><th>Created</th></tr></thead><tbody>{alert_rows}</tbody></table>
  <h2>Data Integrity</h2>
  <table><thead><tr><th>Market</th><th>Check</th><th>Status</th><th>Message</th><th>Value</th><th>Threshold</th><th>Checked</th></tr></thead><tbody>{check_rows}</tbody></table>
  <h2>Performance Logs</h2>
  <table><thead><tr><th>Exchange</th><th>Status</th><th>Last Run</th><th>Last Bar</th><th>Raw Rows</th><th>Unique Bars</th><th>Dup Rows</th><th>Missing 4h</th><th>Dedup PnL</th><th>Raw PnL</th><th>Sharpe</th><th>Max DD</th><th>Avg GMV</th><th>Avg Positions</th></tr></thead><tbody>{perf_rows}</tbody></table>
</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate consolidated ops dashboard.")
    parser.add_argument("--state-db", type=Path, default=PROJECT_ROOT / "data" / "prod_state.db")
    parser.add_argument("--output", type=Path, default=OUT_DIR / "ops_dashboard.html")
    parser.add_argument("--json-output", type=Path, default=OUT_DIR / "ops_dashboard.json")
    args = parser.parse_args()

    payload = build_payload(args.state_db)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(payload), encoding="utf-8")
    args.json_output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {args.json_output}")


if __name__ == "__main__":
    main()
