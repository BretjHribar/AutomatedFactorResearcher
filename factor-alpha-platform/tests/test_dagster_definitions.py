from __future__ import annotations

import json

import dagster as dg

from src.orchestration.dagster_defs import (
    PlatformPathsResource,
    defs,
    production_strategy_config,
)


def test_dagster_definitions_are_loadable():
    dg.Definitions.validate_loadable(defs)

    asset_keys = {key.to_user_string() for key in defs.resolve_all_asset_keys()}
    assert "production_strategy_config" in asset_keys
    assert "equity_eod_data_refresh_result" in asset_keys
    assert "ib_gateway_connectivity_status" in asset_keys
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
        "post_close_integrity_1610_et",
        "crypto_research_signal_4h",
        "ib_paper_moc_execution_1538_et",
        "crypto_paper_execution_4h_utc",
    }.issubset(names)
    assert schedules["crypto_research_signal_4h"].cron_schedule == "2 */4 * * *"
    assert schedules["crypto_paper_execution_4h_utc"].cron_schedule == "3 */4 * * *"
    assert schedules["equity_eod_refresh_hourly_after_close_et"].cron_schedule == "30 16-23 * * 1-5"
    assert schedules["equity_eod_refresh_hourly_overnight_catchup_et"].cron_schedule == "30 0-8 * * 2-6"
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
