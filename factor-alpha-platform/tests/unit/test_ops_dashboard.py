from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.monitoring.state_store import strategy_states
from prod.stats import ops_dashboard as dash
from prod.stats.ops_dashboard import (
    _disabled_exchange_checks,
    _ib_live_curves_split,
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


def test_build_payload_survives_malformed_state_db(tmp_path):
    state_db = tmp_path / "prod_state.db"

    with patch("prod.stats.ops_dashboard.latest_checks",
               side_effect=sqlite3.DatabaseError("database disk image is malformed")), \
         patch("prod.stats.ops_dashboard._sync_strategy_states_from_files",
               side_effect=sqlite3.DatabaseError("database disk image is malformed")), \
         patch("prod.stats.ops_dashboard.strategy_states",
               side_effect=sqlite3.DatabaseError("database disk image is malformed")), \
         patch("prod.stats.ops_dashboard.open_alerts",
               side_effect=sqlite3.DatabaseError("database disk image is malformed")), \
         patch("prod.stats.ops_dashboard._performance_summary", return_value={}):
        payload = build_payload(state_db)

    assert payload["strategy_states"] == []
    assert payload["alerts"] == []
    warning_names = {row["check_name"] for row in payload["integrity_checks"]}
    assert "state_db_latest_checks" in warning_names
    assert "state_db_strategy_states" in warning_names


def _write_aipt_dashboard_fixture(tmp_path, *, signal_time: str, data_dates: list[str]):
    price_rel = "data/fmp_cache/matrices_pit_v2/close.parquet"
    weights_rel = "prod/config/strategies/artifacts/aipt_weights.json"
    price_path = tmp_path / price_rel
    weights_path = tmp_path / weights_rel
    price_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"AAA": [10.0 + i for i, _ in enumerate(data_dates)]},
        index=pd.to_datetime(data_dates),
    ).to_parquet(price_path)
    weights_path.write_text(
        json.dumps({"signal_time": signal_time, "weights": {"AAA": 1.0}}),
        encoding="utf-8",
    )
    return SimpleNamespace(
        enabled=True,
        signal={
            "adapter": "aipt_weights_tail_artifact",
            "weights_path": weights_rel,
            "price_matrix": price_rel,
        },
    )


def test_aipt_dashboard_truth_labels_current_artifact_as_shadow_current(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "PROJECT_ROOT", tmp_path)
    cfg = _write_aipt_dashboard_fixture(
        tmp_path,
        signal_time="2026-05-14 00:00:00",
        data_dates=["2026-05-13", "2026-05-14"],
    )

    truth = dash._aipt_artifact_freshness(cfg)

    assert truth["truth_status"] == "ok"
    assert truth["scope"] == "shadow_current_artifact"
    assert truth["position_source"] == "current artifact weights; shadow netting only"
    assert "stale" not in truth["position_source"].lower()
    assert "included in shadow netting" in truth["description_suffix"]


def test_aipt_dashboard_truth_labels_stale_artifact_as_stale_shadow(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "PROJECT_ROOT", tmp_path)
    cfg = _write_aipt_dashboard_fixture(
        tmp_path,
        signal_time="2026-05-13 00:00:00",
        data_dates=["2026-05-13", "2026-05-14"],
    )

    truth = dash._aipt_artifact_freshness(cfg)

    assert truth["truth_status"] == "warn"
    assert truth["scope"] == "stale_shadow"
    assert truth["position_source"] == "stale artifact weights, not routed"
    assert "Static AIPT artifact is stale" in truth["description_suffix"]


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


def test_ib_live_curves_split_reconciled_diverges_from_provisional(tmp_path, monkeypatch):
    """When the recon JSON shows fewer filled shares than the intent, the
    reconciled curve should track real fills × Δclose minus commission, while
    the provisional curve stays on the intent assumption. Days marked
    `stale_unavailable` collapse onto the provisional path on the recon
    curve too (we don't have the truth)."""
    # Build a tiny synthetic close matrix: D1, D2, D3 for two tickers.
    close = pd.DataFrame(
        {"AAA": [100.0, 102.0, 105.0], "BBB": [50.0, 49.0, 48.0]},
        index=pd.to_datetime(["2026-05-01", "2026-05-04", "2026-05-05"]),
    )

    trade_dir = tmp_path / "prod/logs/trades"
    recon_dir = tmp_path / "prod/logs/reconciliation"
    trade_dir.mkdir(parents=True)
    recon_dir.mkdir(parents=True)

    # Live trade on D1 (5/1): intent = 100 AAA long, 200 BBB short.
    (trade_dir / "trade_2026-05-01.json").write_text(json.dumps({
        "date": "2026-05-01", "mode": "live",
        "target_portfolio": {"AAA": 100, "BBB": -200},
        "order_records": [{"perm_id": 1, "status": "Filled", "symbol": "AAA",
                           "action": "BUY", "quantity": 100},
                          {"perm_id": 2, "status": "Filled", "symbol": "BBB",
                           "action": "SELL", "quantity": 200}],
    }))
    # Recon on D1: only AAA filled (50 shares of 100), BBB didn't fill.
    # Plus $5 commission.
    (recon_dir / "equity_2026-05-01.json").write_text(json.dumps({
        "date": "2026-05-01", "status": "reconciled",
        "fills": [{"symbol": "AAA", "filled_qty": 50}],
        "total_commission": 5.0,
    }))

    monkeypatch.setattr(dash, "TRADE_LOG_DIRS", {"ib": trade_dir})
    monkeypatch.setattr(dash, "RECON_DIR", recon_dir)

    recon_curve, prov_curve, _proxy_curve = _ib_live_curves_split(close)

    # Both curves anchor at 0 on D1 and have a point at each subsequent bar.
    assert len(recon_curve) == len(prov_curve) == 3
    assert recon_curve[0]["pnl"] == 0.0
    assert prov_curve[0]["pnl"] == 0.0

    # D1 -> D2 (5/1 -> 5/4):
    # delta AAA = +2, BBB = -1
    # Provisional: 100*2 + (-200)*-1 = 200 + 200 = 400
    # Reconciled: 50*2 = 100, minus $5 commission charged on D1 = 95
    assert prov_curve[1]["pnl"] == 400.0
    assert recon_curve[1]["pnl"] == 95.0

    # D2 -> D3 (5/4 -> 5/5):
    # delta AAA = +3, BBB = -1
    # Provisional: 100*3 + (-200)*-1 = 300 + 200 = 500   (cum 900)
    # Reconciled: 50*3 = 150 (cum 245)
    assert prov_curve[2]["pnl"] == 900.0
    assert recon_curve[2]["pnl"] == 245.0


def test_ib_live_curves_split_stale_uses_intent_for_both(tmp_path, monkeypatch):
    """A `stale_unavailable` recon (IB lost execution history) must NOT zero
    the reconciled curve — both curves should fall back to intent-shares."""
    close = pd.DataFrame(
        {"AAA": [100.0, 110.0]},
        index=pd.to_datetime(["2026-04-01", "2026-04-02"]),
    )
    trade_dir = tmp_path / "prod/logs/trades"
    recon_dir = tmp_path / "prod/logs/reconciliation"
    trade_dir.mkdir(parents=True)
    recon_dir.mkdir(parents=True)
    (trade_dir / "trade_2026-04-01.json").write_text(json.dumps({
        "date": "2026-04-01", "mode": "live",
        "target_portfolio": {"AAA": 10},
        "order_records": [{"perm_id": 7, "status": "Filled", "symbol": "AAA",
                           "action": "BUY", "quantity": 10}],
    }))
    (recon_dir / "equity_2026-04-01.json").write_text(json.dumps({
        "date": "2026-04-01", "status": "stale_unavailable",
        "fills": [],
    }))

    monkeypatch.setattr(dash, "TRADE_LOG_DIRS", {"ib": trade_dir})
    monkeypatch.setattr(dash, "RECON_DIR", recon_dir)

    recon, prov, _proxy = _ib_live_curves_split(close)
    # Δclose = 10. 10 shares × 10 = $100 on both.
    assert prov[1]["pnl"] == 100.0
    assert recon[1]["pnl"] == 100.0
