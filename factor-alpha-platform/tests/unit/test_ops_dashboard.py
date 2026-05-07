from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd

from src.monitoring.state_store import strategy_states
from prod.stats import ops_dashboard as dash
from prod.stats.ops_dashboard import (
    _disabled_exchange_checks,
    _visible_integrity_checks,
    build_payload,
)


def test_dashboard_hides_retired_aggregate_crypto_checks_when_exchange_checks_exist():
    checks = [
        {
            "market": "crypto",
            "check_name": "crypto_latest_bar_freshness",
            "status": "pass",
        },
        {
            "market": "crypto",
            "check_name": "kucoin_crypto_latest_bar_freshness",
            "status": "pass",
        },
        {
            "market": "equity",
            "check_name": "ohlc_consistency",
            "status": "pass",
        },
    ]

    visible = _visible_integrity_checks(checks)

    assert [row["check_name"] for row in visible] == [
        "kucoin_crypto_latest_bar_freshness",
        "ohlc_consistency",
    ]


def test_build_payload_is_read_only_by_default(tmp_path):
    """The dashboard must not write to the state DB unless refresh_integrity=True.

    Dagster owns the integrity write path; a dashboard render that also writes
    can silently auto-resolve an alert Dagster just raised.
    """
    state_db = tmp_path / "prod_state.db"
    sentinels = {
        "_refresh_integrity_state": False,
        "sync_alerts_from_latest_checks": False,
    }

    def fake_refresh(_db):
        sentinels["_refresh_integrity_state"] = True

    def fake_sync(_db):
        sentinels["sync_alerts_from_latest_checks"] = True
        return 0

    with patch("prod.stats.ops_dashboard._refresh_integrity_state", side_effect=fake_refresh), \
         patch("prod.stats.ops_dashboard.sync_alerts_from_latest_checks", side_effect=fake_sync), \
         patch("prod.stats.ops_dashboard.latest_checks", return_value=[]), \
         patch("prod.stats.ops_dashboard._sync_strategy_states_from_files"), \
         patch("prod.stats.ops_dashboard.strategy_states", return_value=[]), \
         patch("prod.stats.ops_dashboard.open_alerts", return_value=[]), \
         patch("prod.stats.ops_dashboard._performance_summary", return_value={}):
        build_payload(state_db)

    assert sentinels == {
        "_refresh_integrity_state": False,
        "sync_alerts_from_latest_checks": False,
    }


def test_build_payload_writes_only_when_refresh_integrity_true(tmp_path):
    state_db = tmp_path / "prod_state.db"
    calls = {"refresh": 0, "sync": 0}

    with patch("prod.stats.ops_dashboard._refresh_integrity_state",
               side_effect=lambda _db: calls.__setitem__("refresh", calls["refresh"] + 1)), \
         patch("prod.stats.ops_dashboard.sync_alerts_from_latest_checks",
               side_effect=lambda _db: calls.__setitem__("sync", calls["sync"] + 1) or 0), \
         patch("prod.stats.ops_dashboard.latest_checks", return_value=[]), \
         patch("prod.stats.ops_dashboard._sync_strategy_states_from_files"), \
         patch("prod.stats.ops_dashboard.strategy_states", return_value=[]), \
         patch("prod.stats.ops_dashboard.open_alerts", return_value=[]), \
         patch("prod.stats.ops_dashboard._performance_summary", return_value={}):
        build_payload(state_db, refresh_integrity=True)

    assert calls == {"refresh": 1, "sync": 1}


def test_disabled_exchange_checks_cover_all_crypto_guardrails():
    names = {row["name"] for row in _disabled_exchange_checks("binance", "blocked")}

    assert {
        "binance_crypto_latest_bar_freshness",
        "binance_crypto_bar_index_continuity",
        "binance_crypto_latest_coverage",
        "binance_ohlc_consistency",
        "binance_crypto_universe_exists",
        "binance_crypto_universe_current",
        "binance_crypto_universe_membership",
        "binance_alpha_database_active_alphas",
    } <= names


def test_ib_dashboard_state_prefers_latest_execution_failure(tmp_path, monkeypatch):
    project_root = tmp_path
    state_db = tmp_path / "data/prod_state.db"
    config_path = project_root / "prod/config/strategy.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({
        "strategy": {
            "name": "IB Closing Auction L/S Equity",
            "universe": "MCAP_100M_500M",
            "min_alpha_sharpe": 5.0,
        },
        "account": {"target_gmv": 500000},
        "ibkr": {"port_paper": 4002, "port_live": 4001},
        "paths": {"db": "data/ib_alphas.db"},
    }), encoding="utf-8")

    matrix_path = project_root / "data/fmp_cache/matrices/close.parquet"
    matrix_path.parent.mkdir(parents=True)
    pd.DataFrame({"AAA": [1.0]}, index=pd.to_datetime(["2026-05-07"])).to_parquet(matrix_path)

    trade_dir = project_root / "prod/logs/trades"
    trade_dir.mkdir(parents=True)
    (trade_dir / "trade_2026-05-05.json").write_text(json.dumps({
        "date": "2026-05-05",
        "signal_date": "2026-05-05",
        "timestamp": "2026-05-05T14:54:15",
        "mode": "live",
        "n_orders": 41,
        "portfolio_summary": {"n_long": 21, "n_short": 38, "gross_shares": 9492, "net_shares": 2042},
    }), encoding="utf-8")

    execution_dir = project_root / "prod/logs/execution"
    execution_dir.mkdir(parents=True)
    (execution_dir / "ib_paper_moc_failed.json").write_text(json.dumps({
        "venue": "ib_equity",
        "status": "failed",
        "message": "subprocess failed with returncode 1",
        "started_at_utc": "2026-05-07T19:38:39+00:00",
        "ended_at_utc": "2026-05-07T19:38:41+00:00",
        "elapsed_sec": 1.496,
        "summary_path": "prod/logs/execution/ib_paper_moc_failed.json",
    }), encoding="utf-8")

    monkeypatch.setattr(dash, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(dash, "LOGS_ROOT", project_root / "prod/logs")
    monkeypatch.setitem(dash.TRADE_LOG_DIRS, "ib", trade_dir)
    monkeypatch.setattr(
        dash,
        "check_ib_gateway",
        lambda *_args, **_kwargs: type("Status", (), {
            "to_dict": lambda self: {
                "connected": True,
                "host": "host.docker.internal",
                "port": 4002,
                "mode": "ib_insync",
                "message": "ok",
                "checked_at_utc": "2026-05-07T21:00:00+00:00",
                "error_type": None,
                "accounts": ["DUQ372830"],
            }
        })(),
    )

    dash._sync_ib_strategy_state(state_db, checks=[])

    ib_state = next(row for row in strategy_states(state_db) if row["strategy_id"] == "ib_moc_equity")
    assert ib_state["status"] == "execution_failed"
    metrics = json.loads(ib_state["metrics_json"])
    assert metrics["latest_execution_status"] == "failed"
