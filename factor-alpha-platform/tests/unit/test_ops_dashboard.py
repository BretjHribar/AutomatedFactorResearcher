from __future__ import annotations

from unittest.mock import patch

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
