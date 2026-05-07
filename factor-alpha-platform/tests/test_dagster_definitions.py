from __future__ import annotations

import json

import dagster as dg

from src.orchestration.dagster_defs import (
    PlatformPathsResource,
    defs,
    _ib_runtime_dependency_check_payload,
    _ib_runtime_dependency_payload,
    production_strategy_config,
)


def test_dagster_definitions_are_loadable():
    dg.Definitions.validate_loadable(defs)

    asset_keys = {key.to_user_string() for key in defs.resolve_all_asset_keys()}
    assert "production_strategy_config" in asset_keys
    assert "equity_eod_data_refresh_result" in asset_keys
    assert "ib_gateway_connectivity_status" in asset_keys
    assert "ib_runtime_dependency_status" in asset_keys
    assert "equity_integrity_results" in asset_keys
    assert "kucoin_integrity_results" in asset_keys
    assert "binance_integrity_results" in asset_keys
    assert "live_equity_quote_snapshot" in asset_keys
    assert "research_equity_signal_snapshot" in asset_keys
    assert "ib_paper_moc_execution_result" in asset_keys
    assert "kucoin_paper_execution_record" in asset_keys
    assert "binance_paper_execution_record" in asset_keys
    assert "ops_dashboard_state" in asset_keys


def test_dagster_schedules_cover_intraday_preflight_and_postclose():
    schedules = {schedule.name: schedule for schedule in defs.schedules}
    names = set(schedules)

    assert {
        "equity_preflight_1515_et",
        "equity_eod_refresh_hourly_after_close_et",
        "equity_eod_refresh_hourly_overnight_catchup_et",
        "live_quote_collector_5m_regular_session",
        "equity_research_signal_1530_et",
        "post_eod_integrity_hourly_after_close_et",
        "post_eod_integrity_hourly_overnight_catchup_et",
        "crypto_research_signal_4h",
        "ib_paper_moc_execution_1538_et",
        "crypto_paper_execution_4h_utc",
    }.issubset(names)
    assert schedules["crypto_research_signal_4h"].cron_schedule == "2 */4 * * *"
    assert schedules["crypto_paper_execution_4h_utc"].cron_schedule == "3 */4 * * *"
    assert schedules["equity_eod_refresh_hourly_after_close_et"].cron_schedule == "30 16-23 * * 1-5"
    assert schedules["equity_eod_refresh_hourly_overnight_catchup_et"].cron_schedule == "30 0-8 * * 2-6"
    assert schedules["post_eod_integrity_hourly_after_close_et"].cron_schedule == "50 16-23 * * 1-5"
    assert schedules["post_eod_integrity_hourly_overnight_catchup_et"].cron_schedule == "50 0-8 * * 2-6"
    assert defs.resolve_job_def("equity_eod_data_refresh_job").tags["factor_alpha/eod_refresh"] == "equity_eod"


def test_production_strategy_config_asset_materializes_from_config(tmp_path):
    config_path = tmp_path / "prod" / "config" / "strategy.json"
    config_path.parent.mkdir(parents=True)
    config = {
        "strategy": {
            "name": "IB Closing Auction L/S Equity",
            "version": "1.0.0",
            "universe": "MCAP_100M_500M",
            "min_alpha_sharpe": 5.0,
        },
        "paths": {"db": "data/alpha_results.db", "universes": "data/fmp_cache/universes"},
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = dg.materialize(
        [production_strategy_config],
        resources={
            "paths": PlatformPathsResource(
                root=str(tmp_path),
                strategy_config_rel="prod/config/strategy.json",
            )
        },
    )

    assert result.success
    output = result.output_for_node("production_strategy_config")
    assert output["strategy"]["universe"] == "MCAP_100M_500M"
    assert len(output["_config_hash"]) == 64


def test_ib_runtime_dependency_payload_reports_missing_modules(tmp_path):
    payload = _ib_runtime_dependency_payload(tmp_path)
    check = _ib_runtime_dependency_check_payload(payload)

    assert payload["status"] == "fail"
    assert check["name"] == "ib_runtime_dependencies"
    assert check["status"] == "fail"
    assert "eval_alpha_ib.py" in " ".join(payload["missing"])


def test_upsert_ib_execution_state_preserves_signal_side_gross(tmp_path):
    """The execution writer must NOT clobber the signal asset's dynamic
    gross/net exposure. Both writers use merge=True; only fields the
    execution writer actually owns (status, last_trade_time, n_positions,
    execution metadata) should change."""
    from src.orchestration.dagster_defs import _upsert_ib_execution_state
    from src.monitoring.state_store import StrategyState, strategy_states, upsert_strategy_state

    state_db = tmp_path / "data/prod_state.db"
    state_db.parent.mkdir(parents=True)

    # Pretend the signal asset wrote first.
    upsert_strategy_state(
        StrategyState(
            strategy_id="ib_moc_equity",
            market="equity",
            mode="paper",
            status="signal_ready",
            last_signal_bar="2026-05-07",
            gross_exposure=512345.67,
            net_exposure=-1234.5,
            n_positions=200,
            metrics_json={"source": "dagster_signal_snapshot", "gross_weight_l1": 1.0247},
        ),
        db_path=state_db,
        merge=True,
    )

    # Then the execution writer runs (orders blocked path).
    _upsert_ib_execution_state(
        db_path=state_db,
        production_strategy_config={
            "account": {"target_gmv": 500000},
            "_config_hash": "abc123",
        },
        status="execution_blocked",
        message="orders disabled",
    )

    rows = strategy_states(state_db)
    state = next(r for r in rows if r["strategy_id"] == "ib_moc_equity")
    # status/config_hash come from the execution writer
    assert state["status"] == "execution_blocked"
    assert state["config_hash"] == "abc123"
    # but gross/net stay at the signal-side computed values
    assert state["gross_exposure"] == 512345.67
    assert state["net_exposure"] == -1234.5
    assert state["last_signal_bar"] == "2026-05-07"
    # n_positions wasn't in the trade summary, so signal-side count is preserved
    assert state["n_positions"] == 200


def test_ib_alert_resolve_only_fires_on_completed_status():
    """Regression: blocked/skipped must NOT auto-resolve a still-real alert.

    The recorder returns `status="blocked"` whenever ALLOW_IB_PAPER_ORDERS=0
    (the default). If the asset called resolve_alerts on the blocked branch,
    every scheduled tick with orders disabled would silently clear an alert
    raised by a prior real failure. Resolution must require an actual
    successful execution.
    """
    from pathlib import Path

    src_path = Path(__file__).resolve().parents[1] / "src/orchestration/dagster_defs.py"
    src = src_path.read_text(encoding="utf-8")
    # Find the asset definition and its body.
    asset_marker = "def ib_paper_moc_execution_result("
    asset_start = src.index(asset_marker)
    # The next top-level def or @ marks the end of this asset's body.
    rest = src[asset_start:]
    body_end = min(
        idx for idx in (
            rest.find("\n\n\n@dg.asset"),
            rest.find("\n\n\n@dg.asset_check"),
            rest.find("\n\n\nequity_integrity_job"),
            rest.find("\n\n\nschedules ="),
        ) if idx > 0
    )
    body = rest[:body_end]
    assert 'if payload["status"] == "completed":' in body, (
        "resolve_alerts must be gated on status==completed; current source does not contain that guard"
    )
    gate_idx = body.index('if payload["status"] == "completed":')
    resolve_idx = body.index("resolve_alerts(")
    assert resolve_idx > gate_idx, (
        "resolve_alerts must be inside the completed-only branch, not before it"
    )
