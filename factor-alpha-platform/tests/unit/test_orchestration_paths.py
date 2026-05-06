from __future__ import annotations

import json

import pandas as pd

from src.orchestration.paths import PlatformPaths, active_universe_tickers, load_json


def test_platform_paths_resolve_relative_paths(tmp_path):
    paths = PlatformPaths(root=tmp_path, state_db_rel="state/prod.db")

    assert paths.resolve("data/example.parquet") == tmp_path / "data" / "example.parquet"
    assert paths.state_db == tmp_path / "state" / "prod.db"


def test_load_json_reads_utf8_payload(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"strategy": {"name": "demo"}}), encoding="utf-8")

    assert load_json(path) == {"strategy": {"name": "demo"}}


def test_active_universe_tickers_reads_latest_membership(tmp_path):
    universes = tmp_path / "data" / "fmp_cache" / "universes"
    universes.mkdir(parents=True)
    pd.DataFrame(
        {
            "AAA": [True, False],
            "BBB": [False, True],
            "CCC": [True, True],
        },
        index=pd.to_datetime(["2026-05-04", "2026-05-05"]),
    ).to_parquet(universes / "MCAP_100M_500M.parquet")
    cfg = {
        "strategy": {"universe": "MCAP_100M_500M"},
        "paths": {"universes": "data/fmp_cache/universes"},
    }

    tickers = active_universe_tickers(cfg, PlatformPaths(root=tmp_path))

    assert tickers == ["BBB", "CCC"]

