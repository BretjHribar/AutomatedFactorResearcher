from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from src.data.equity_refresh import (
    build_market_cap_band_universes,
    latest_closed_nyse_session,
    merge_price_history,
    rebuild_classification_matrices,
    rebuild_equity_cache_from_prices,
    refresh_equity_eod_cache,
)


def test_latest_closed_nyse_session_waits_for_post_close_buffer():
    before_buffer = pd.Timestamp("2026-05-06 16:29:00", tz="America/New_York")
    after_buffer = pd.Timestamp("2026-05-06 16:30:00", tz="America/New_York")

    assert latest_closed_nyse_session(before_buffer) == dt.date(2026, 5, 5)
    assert latest_closed_nyse_session(after_buffer) == dt.date(2026, 5, 6)


def test_latest_closed_nyse_session_weekend_uses_prior_trading_day():
    saturday = pd.Timestamp("2026-05-09 12:00:00", tz="America/New_York")

    assert latest_closed_nyse_session(saturday) == dt.date(2026, 5, 8)


def test_merge_price_history_detects_revisions_and_new_bars():
    existing = pd.DataFrame(
        {
            "open": [10.0, 11.0],
            "high": [11.0, 12.0],
            "low": [9.0, 10.0],
            "close": [10.5, 11.5],
            "volume": [100.0, 110.0],
            "vwap": [10.3, 11.3],
        },
        index=pd.to_datetime(["2026-05-04", "2026-05-05"]),
    )
    fetched = pd.DataFrame(
        {
            "open": [11.0, 12.0],
            "high": [12.2, 13.0],
            "low": [10.0, 11.0],
            "close": [11.7, 12.5],
            "volume": [115.0, 120.0],
            "vwap": [11.4, 12.3],
        },
        index=pd.to_datetime(["2026-05-05", "2026-05-06"]),
    )

    merged, changed = merge_price_history(existing, fetched)

    assert changed
    assert merged.loc[pd.Timestamp("2026-05-05"), "close"] == 11.7
    assert merged.loc[pd.Timestamp("2026-05-06"), "close"] == 12.5


def test_merge_price_history_ignores_dtype_only_differences():
    existing = pd.DataFrame(
        {
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [100.0],
            "vwap": [10.25],
            "adjClose": [10.5],
            "changePercent": [0.0],
        },
        index=pd.to_datetime(["2026-05-05"]),
    )
    fetched = pd.DataFrame(
        {
            "open": ["10.0"],
            "high": ["11.0"],
            "low": ["9.0"],
            "close": ["10.50000000001"],
            "volume": ["100"],
            "vwap": ["10.25"],
            "adjClose": ["10.5"],
            "changePercent": ["0"],
        },
        index=pd.to_datetime(["2026-05-05"]),
    )

    _, changed = merge_price_history(existing, fetched)

    assert not changed


def test_build_market_cap_band_universes(tmp_path):
    matrices = tmp_path / "matrices"
    matrices.mkdir()
    dates = pd.to_datetime(["2026-05-05", "2026-05-06"])
    close = pd.DataFrame({"AAA": [10.0, 10.5], "BBB": [20.0, 20.5], "CCC": [30.0, 30.5]}, index=dates)
    market_cap = pd.DataFrame(
        {
            "AAA": [150_000_000.0, 180_000_000.0],
            "BBB": [750_000_000.0, 900_000_000.0],
            "CCC": [3_000_000_000.0, 4_000_000_000.0],
        },
        index=dates,
    )
    close.to_parquet(matrices / "close.parquet")
    market_cap.to_parquet(matrices / "market_cap.parquet")

    summaries = build_market_cap_band_universes(tmp_path)
    small = pd.read_parquet(tmp_path / "universes" / "MCAP_100M_500M.parquet")

    assert summaries["MCAP_100M_500M"]["last_members"] == 1
    assert bool(small.loc[pd.Timestamp("2026-05-06"), "AAA"])
    assert not bool(small.loc[pd.Timestamp("2026-05-06"), "BBB"])


def test_rebuild_classification_matrices_preserves_subindustry_hierarchy(tmp_path):
    matrices = tmp_path / "matrices"
    matrices.mkdir()
    dates = pd.to_datetime(["2026-05-05", "2026-05-06"])
    pd.DataFrame({"AAA": [10.0, 10.5], "BBB": [20.0, 20.5]}, index=dates).to_parquet(matrices / "close.parquet")
    (tmp_path / "classifications.json").write_text(
        json.dumps(
            {
                "AAA": {"sector": "Manufacturing", "industry": "382", "subindustry": "3826"},
                "BBB": {"sector": "Manufacturing", "industry": "382", "subindustry": "3827"},
            }
        ),
        encoding="utf-8",
    )

    stats = rebuild_classification_matrices(tmp_path)
    industry = pd.read_parquet(matrices / "industry.parquet")
    subindustry = pd.read_parquet(matrices / "subindustry.parquet")

    assert stats["subindustry_groups"] == 2
    assert stats["industry_subindustry_same_pct"] == 0.0
    assert not industry.equals(subindustry)


def test_rebuild_equity_cache_from_prices_incrementally_updates_matrices(tmp_path):
    cache = tmp_path
    prices_dir = cache / "prices"
    matrices = cache / "matrices"
    prices_dir.mkdir()
    matrices.mkdir()
    dates = pd.to_datetime(["2026-05-05", "2026-05-06"])
    for symbol, close_values in {"AAA": [10.0, 11.0], "BBB": [20.0, 21.0]}.items():
        pd.DataFrame(
            {
                "open": close_values,
                "high": [v + 1 for v in close_values],
                "low": [v - 1 for v in close_values],
                "close": close_values,
                "volume": [100.0, 120.0],
                "vwap": close_values,
            },
            index=dates,
        ).to_parquet(prices_dir / f"{symbol}.parquet")
    pd.DataFrame({"AAA": [10_000_000.0], "BBB": [20_000_000.0]}, index=dates[:1]).to_parquet(
        matrices / "shares_out.parquet"
    )
    pd.DataFrame({"AAA": [1.0], "BBB": [2.0]}, index=dates[:1]).to_parquet(matrices / "revenue.parquet")
    (cache / "classifications.json").write_text(
        json.dumps(
            {
                "AAA": {"sector": "Technology", "industry": "737", "subindustry": "7372"},
                "BBB": {"sector": "Finance", "industry": "602", "subindustry": "6021"},
            }
        ),
        encoding="utf-8",
    )

    summary = rebuild_equity_cache_from_prices(cache, ["AAA", "BBB"])

    close = pd.read_parquet(matrices / "close.parquet")
    revenue = pd.read_parquet(matrices / "revenue.parquet")
    market_cap = pd.read_parquet(matrices / "market_cap.parquet")
    universe = pd.read_parquet(cache / "universes" / "MCAP_100M_500M.parquet")
    assert summary["matrix_end"] == "2026-05-06"
    assert close.loc[pd.Timestamp("2026-05-06"), "AAA"] == 11.0
    assert revenue.loc[pd.Timestamp("2026-05-06"), "AAA"] == 1.0
    assert market_cap.loc[pd.Timestamp("2026-05-06"), "AAA"] == 110_000_000.0
    assert bool(universe.loc[pd.Timestamp("2026-05-06"), "AAA"])


def test_rebuild_equity_cache_from_prices_uses_metadata_universe(tmp_path):
    cache = tmp_path
    prices_dir = cache / "prices"
    matrices = cache / "matrices"
    prices_dir.mkdir()
    matrices.mkdir()
    dates = pd.to_datetime(["2026-05-05", "2026-05-06"])
    for symbol in ["AAA", "BBB", "EXTRA"]:
        pd.DataFrame(
            {
                "open": [10.0, 11.0],
                "high": [11.0, 12.0],
                "low": [9.0, 10.0],
                "close": [10.0, 11.0],
                "volume": [100.0, 120.0],
                "vwap": [10.0, 11.0],
            },
            index=dates,
        ).to_parquet(prices_dir / f"{symbol}.parquet")
    pd.DataFrame({"AAA": [150_000_000.0], "BBB": [250_000_000.0]}, index=dates[:1]).to_parquet(
        matrices / "market_cap_metric.parquet"
    )
    (cache / "metadata.json").write_text(
        json.dumps({"tickers": ["AAA", "BBB"], "start_date": "2026-05-05 00:00:00"}),
        encoding="utf-8",
    )
    (cache / "classifications.json").write_text(
        json.dumps(
            {
                "AAA": {"sector": "Technology", "industry": "737", "subindustry": "7372"},
                "BBB": {"sector": "Finance", "industry": "602", "subindustry": "6021"},
                "EXTRA": {"sector": "Energy", "industry": "131", "subindustry": "1311"},
            }
        ),
        encoding="utf-8",
    )

    summary = rebuild_equity_cache_from_prices(cache, ["AAA", "BBB", "EXTRA"])

    close = pd.read_parquet(matrices / "close.parquet")
    market_cap = pd.read_parquet(matrices / "market_cap.parquet")
    assert summary["matrix_cols"] == 2
    assert list(close.columns) == ["AAA", "BBB"]
    assert "EXTRA" not in close.columns
    assert market_cap.loc[pd.Timestamp("2026-05-06"), "AAA"] == 150_000_000.0


def test_refresh_equity_eod_cache_up_to_date_without_fetch(tmp_path):
    root = tmp_path
    matrices = root / "data" / "fmp_cache" / "matrices"
    universes = root / "data" / "fmp_cache" / "universes"
    matrices.mkdir(parents=True)
    universes.mkdir(parents=True)
    dates = pd.to_datetime(["2026-05-05"])
    for field in ["open", "high", "low", "close", "volume"]:
        pd.DataFrame({"AAA": [10.0], "BBB": [20.0]}, index=dates).to_parquet(matrices / f"{field}.parquet")
    pd.DataFrame({"AAA": [True], "BBB": [True]}, index=dates).to_parquet(universes / "MCAP_100M_500M.parquet")

    result = refresh_equity_eod_cache(
        root,
        expected_date=dt.date(2026, 5, 5),
        recheck_recent=False,
        min_active_coverage=1.0,
    )

    assert result["status"] == "up_to_date"
    assert result["symbols_checked"] == 0
    assert result["active_coverage"]["coverage"] == 1.0
