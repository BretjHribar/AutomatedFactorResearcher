from __future__ import annotations

import json

from src.runtime.decision import (
    DecisionDataset,
    read_decision_dataset,
    stable_config_hash,
    summarize_vwap_sources,
    write_decision_dataset,
)


def test_stable_config_hash_ignores_dict_order():
    left = {"strategy": {"name": "x", "min_alpha_sharpe": 5}, "paths": {"db": "a.db"}}
    right = {"paths": {"db": "a.db"}, "strategy": {"min_alpha_sharpe": 5, "name": "x"}}

    assert stable_config_hash(left) == stable_config_hash(right)


def test_vwap_coverage_counts_good_sources_fallbacks_and_missing():
    coverage = summarize_vwap_sources(
        ["AAA", "BBB", "CCC", "DDD"],
        {
            "AAA": "fmp_intraday",
            "BBB": "quote_tape",
            "CCC": "typical_price",
        },
    )

    assert coverage.active_count == 4
    assert coverage.covered_count == 2
    assert coverage.fallback_count == 1
    assert coverage.missing_count == 1
    assert coverage.coverage == 0.5
    assert coverage.by_source == {"fmp_intraday": 1, "quote_tape": 1, "typical_price": 1}
    assert coverage.fallback_symbols == ["CCC"]
    assert coverage.missing_symbols == ["DDD"]


def test_decision_dataset_validation_passes_for_full_intraday_vwap_coverage():
    dataset = DecisionDataset(
        strategy_id="midcap_moc",
        market="equity",
        signal_date="2026-05-05",
        decision_time_utc="2026-05-05T20:30:00+00:00",
        data_last_bar="2026-05-05",
        active_tickers=["AAA", "BBB"],
        alpha_ids=[1, 2, 3],
        vwap_sources={"AAA": "fmp_intraday", "BBB": "ib_stream"},
    )

    checks = dataset.validate(min_active_tickers=2, min_alpha_count=3, min_vwap_coverage=1.0)

    assert all(c.status == "pass" for c in checks)


def test_decision_dataset_validation_fails_missing_vwap_provenance():
    dataset = DecisionDataset(
        strategy_id="midcap_moc",
        market="equity",
        signal_date="2026-05-05",
        decision_time_utc="2026-05-05T20:30:00+00:00",
        data_last_bar="2026-05-05",
        active_tickers=["AAA", "BBB"],
        alpha_ids=[1],
        vwap_sources={"AAA": "fmp_intraday"},
    )

    checks = dataset.validate(min_active_tickers=2, min_alpha_count=1, min_vwap_coverage=1.0)

    assert [c.name for c in checks if c.status == "fail"] == ["live_vwap_coverage"]
    assert checks[-1].metadata["missing_symbols"] == ["BBB"]


def test_decision_dataset_round_trip_json(tmp_path):
    dataset = DecisionDataset(
        strategy_id="midcap_moc",
        market="equity",
        signal_date="2026-05-05",
        decision_time_utc="2026-05-05T20:30:00+00:00",
        data_last_bar="2026-05-05",
        active_tickers=["AAA"],
        alpha_ids=[101],
        config_hash="abc",
        git_sha="def",
        vwap_sources={"AAA": "fmp_intraday"},
        artifact_paths={"live_quotes": "prod/logs/live_quotes/fmp_quotes_2026-05-05.csv"},
    )
    path = tmp_path / "decision.json"

    write_decision_dataset(dataset, path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    loaded = read_decision_dataset(path)

    assert raw["strategy_id"] == "midcap_moc"
    assert loaded == dataset

