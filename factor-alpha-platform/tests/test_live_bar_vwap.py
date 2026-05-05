from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prod"))

import live_bar


def test_intraday_rows_to_vwap_uses_today_volume_weighted_typical_price():
    rows = [
        {"date": "2026-05-05 15:55:00", "high": 13.0, "low": 10.0, "close": 13.0, "volume": 100},
        {"date": "2026-05-05 15:50:00", "high": 22.0, "low": 19.0, "close": 19.0, "volume": 300},
        {"date": "2026-05-04 15:55:00", "high": 100.0, "low": 100.0, "close": 100.0, "volume": 10000},
    ]

    vwap = live_bar._intraday_rows_to_vwap(rows, today=dt.date(2026, 5, 5))

    expected = (((13 + 10 + 13) / 3) * 100 + ((22 + 19 + 19) / 3) * 300) / 400
    assert vwap == pytest.approx(expected)


def test_quote_tape_vwap_includes_opening_cumulative_volume(monkeypatch):
    monkeypatch.setattr(live_bar, "QUOTE_TAPE_MIN_SNAPSHOTS", 3)
    monkeypatch.setattr(live_bar, "QUOTE_TAPE_MIN_SPAN_MINUTES", 30)
    tape = pd.DataFrame(
        {
            "captured_at_utc": [
                "2026-05-05T13:35:00+00:00",
                "2026-05-05T14:35:00+00:00",
                "2026-05-05T15:35:00+00:00",
            ],
            "price": [10.0, 20.0, 30.0],
            "volume": [100, 160, 220],
        }
    )

    vwap = live_bar._quote_tape_vwap_for_symbol(tape)

    expected = (10 * 100 + 20 * 60 + 30 * 60) / 220
    assert vwap == pytest.approx(expected)


def test_quote_tape_vwap_excludes_first_late_snapshot_volume(monkeypatch):
    monkeypatch.setattr(live_bar, "QUOTE_TAPE_MIN_SNAPSHOTS", 3)
    monkeypatch.setattr(live_bar, "QUOTE_TAPE_MIN_SPAN_MINUTES", 30)
    tape = pd.DataFrame(
        {
            "captured_at_utc": [
                "2026-05-05T14:30:00+00:00",
                "2026-05-05T15:30:00+00:00",
                "2026-05-05T16:30:00+00:00",
            ],
            "price": [10.0, 20.0, 30.0],
            "volume": [100, 160, 220],
        }
    )

    vwap = live_bar._quote_tape_vwap_for_symbol(tape)

    expected = (20 * 60 + 30 * 60) / 120
    assert vwap == pytest.approx(expected)


def test_construct_live_bar_uses_supplied_intraday_vwap_before_fallback():
    dates = pd.DatetimeIndex(["2026-05-04"])
    cols = ["AAA", "BBB", "CCC"]
    matrices = {
        name: pd.DataFrame([[10.0, 20.0, 30.0]], index=dates, columns=cols)
        for name in ["close", "open", "high", "low", "volume", "vwap"]
    }
    quotes = {
        "AAA": {"price": 11.0, "open": 10.5, "dayHigh": 11.5, "dayLow": 10.0},
        "BBB": {"price": 21.0, "open": 20.5, "dayHigh": 21.5, "dayLow": 20.0},
        "CCC": {"price": 31.0, "open": 30.5, "dayHigh": 32.0, "dayLow": 29.0},
    }

    extended = live_bar.construct_live_bar(
        matrices,
        quotes,
        flagged_tickers=[],
        ib_vwap={"AAA": 10.75, "BBB": 20.75},
        vwap_sources={"AAA": "fmp_intraday", "BBB": "quote_tape"},
    )

    live_vwap = extended["vwap"].iloc[-1]
    assert live_vwap["AAA"] == pytest.approx(10.75)
    assert live_vwap["BBB"] == pytest.approx(20.75)
    assert live_vwap["CCC"] == pytest.approx((32.0 + 29.0 + 31.0) / 3.0)
