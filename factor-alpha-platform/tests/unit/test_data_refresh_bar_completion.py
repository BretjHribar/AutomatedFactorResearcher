from __future__ import annotations

import pandas as pd

from prod import data_refresh
from prod.data_refresh import _drop_incomplete_4h_bars, _latest_completed_4h_start


def test_latest_completed_4h_start_before_boundary_excludes_current_open_bar():
    assert _latest_completed_4h_start("2026-05-06T03:55:00Z") == pd.Timestamp(
        "2026-05-05 20:00:00"
    )


def test_latest_completed_4h_start_after_boundary_uses_prior_open_bar():
    assert _latest_completed_4h_start("2026-05-06T04:03:00Z") == pd.Timestamp(
        "2026-05-06 00:00:00"
    )


def test_drop_incomplete_4h_bars_filters_exchange_open_timestamped_candle():
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-05-05 20:00:00", "2026-05-06 00:00:00", "2026-05-06 04:00:00"]
            ),
            "close": [100.0, 101.0, 102.0],
        }
    )

    before_close = _drop_incomplete_4h_bars(df, "time", now_ts="2026-05-06T03:55:00Z")
    after_close = _drop_incomplete_4h_bars(df, "time", now_ts="2026-05-06T04:03:00Z")

    assert before_close["time"].tolist() == [pd.Timestamp("2026-05-05 20:00:00")]
    assert after_close["time"].tolist() == [
        pd.Timestamp("2026-05-05 20:00:00"),
        pd.Timestamp("2026-05-06 00:00:00"),
    ]


def test_kucoin_update_overwrites_existing_recent_completed_bar(monkeypatch, tmp_path):
    kline_dir = tmp_path / "kucoin"
    kline_dir.mkdir()
    symbol_path = kline_dir / "XBTUSDTM.parquet"
    existing = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-05-05 20:00:00", "2026-05-06 00:00:00"]),
            "open": [100.0, 101.0],
            "high": [100.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [10.0, 11.0],
            "turnover": [1000.0, 1111.0],
        }
    ).set_index("time")
    existing.to_parquet(symbol_path)

    fetched = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-05-06 00:00:00", "2026-05-06 04:00:00"]),
            "open": [101.0, 200.0],
            "high": [112.0, 205.0],
            "low": [100.0, 199.0],
            "close": [111.0, 204.0],
            "volume": [15.0, 50.0],
            "turnover": [1665.0, 10000.0],
        }
    )

    monkeypatch.setattr(data_refresh, "KUCOIN_KLINES_DIR", kline_dir)
    monkeypatch.setattr(data_refresh, "_latest_completed_4h_start", lambda now_ts=None: pd.Timestamp("2026-05-06 00:00:00"))
    monkeypatch.setattr(data_refresh, "_kucoin_fetch_klines", lambda symbol, limit: fetched)
    monkeypatch.setattr(data_refresh.time, "sleep", lambda seconds: None)

    assert data_refresh._kucoin_update_klines() == 1

    saved = pd.read_parquet(symbol_path).reset_index()
    assert saved["time"].tolist() == [
        pd.Timestamp("2026-05-05 20:00:00"),
        pd.Timestamp("2026-05-06 00:00:00"),
    ]
    assert saved.loc[saved["time"] == pd.Timestamp("2026-05-06 00:00:00"), "close"].iloc[0] == 111.0
