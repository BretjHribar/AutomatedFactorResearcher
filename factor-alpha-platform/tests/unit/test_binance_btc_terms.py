from __future__ import annotations

import pandas as pd

from tools.build_binance_btc_terms import build_btc_terms_matrices


def _df(rows):
    idx = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 04:00", "2026-01-01 08:00"])
    return pd.DataFrame(rows, index=idx)


def test_binance_btc_terms_converts_prices_and_quote_notional():
    all_data = {
        "BTCUSDT": _df(
            {
                "open": [100.0, 200.0, 100.0],
                "high": [110.0, 210.0, 110.0],
                "low": [90.0, 190.0, 90.0],
                "close": [100.0, 200.0, 100.0],
                "volume": [10.0, 10.0, 10.0],
                "quote_volume": [1000.0, 2000.0, 1000.0],
                "trades_count": [5.0, 6.0, 7.0],
                "taker_buy_volume": [4.0, 5.0, 6.0],
                "taker_buy_quote_volume": [400.0, 1000.0, 600.0],
            }
        ),
        "ALTUSDT": _df(
            {
                "open": [10.0, 40.0, 30.0],
                "high": [12.0, 44.0, 33.0],
                "low": [9.0, 38.0, 27.0],
                "close": [10.0, 50.0, 25.0],
                "volume": [100.0, 200.0, 300.0],
                "quote_volume": [1000.0, 10000.0, 7500.0],
                "trades_count": [10.0, 20.0, 30.0],
                "taker_buy_volume": [40.0, 80.0, 120.0],
                "taker_buy_quote_volume": [400.0, 4000.0, 3000.0],
            }
        ),
    }

    mats, meta = build_btc_terms_matrices(all_data, btc_symbol="BTCUSDT")

    assert "BTCUSDT" not in mats["close"].columns
    assert meta["n_symbols_written"] == 1

    close = mats["close"]["ALTUSDT"]
    pd.testing.assert_series_equal(close, pd.Series([0.10, 0.25, 0.25], index=close.index, name="ALTUSDT"))

    returns = mats["returns"]["ALTUSDT"]
    assert pd.isna(returns.iloc[0])
    assert returns.iloc[1] == 1.5
    assert returns.iloc[2] == 0.0

    btc_typical_0 = (110.0 + 90.0 + 100.0) / 3.0
    assert mats["quote_volume"]["ALTUSDT"].iloc[0] == 1000.0 / btc_typical_0
    assert mats["dollars_traded"]["ALTUSDT"].iloc[0] == 1000.0 / btc_typical_0
    assert mats["volume"]["ALTUSDT"].tolist() == [100.0, 200.0, 300.0]
    assert mats["taker_buy_quote_volume"]["ALTUSDT"].iloc[0] == 400.0 / btc_typical_0


def test_binance_range_envelope_keeps_ohlc_consistent():
    all_data = {
        "BTCUSDT": _df(
            {
                "open": [100.0, 100.0, 100.0],
                "high": [130.0, 130.0, 130.0],
                "low": [70.0, 70.0, 70.0],
                "close": [120.0, 120.0, 120.0],
                "volume": [1.0, 1.0, 1.0],
                "quote_volume": [100.0, 100.0, 100.0],
            }
        ),
        "ALTUSDT": _df(
            {
                "open": [10.0, 10.0, 10.0],
                "high": [11.0, 11.0, 11.0],
                "low": [9.0, 9.0, 9.0],
                "close": [10.0, 10.0, 10.0],
                "volume": [1.0, 1.0, 1.0],
                "quote_volume": [10.0, 10.0, 10.0],
            }
        ),
    }

    mats, meta = build_btc_terms_matrices(all_data, btc_symbol="BTCUSDT", ohlc_mode="range_envelope")

    assert meta["high_low_inversions_after_sanitize"] == 0
    assert (mats["high"] >= mats["low"]).all().all()
    assert (mats["high"] >= mats["open"]).all().all()
    assert (mats["low"] <= mats["close"]).all().all()
