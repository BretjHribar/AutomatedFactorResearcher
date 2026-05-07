from __future__ import annotations

import pandas as pd

from tools.build_kucoin_btc_terms import build_btc_terms_matrices


def _df(rows):
    idx = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 04:00", "2026-01-01 08:00"])
    return pd.DataFrame(rows, index=idx)


def test_build_btc_terms_recomputes_point_prices_returns_and_notional_volume():
    all_data = {
        "XBTUSDTM": _df(
            {
                "open": [100.0, 200.0, 100.0],
                "high": [110.0, 210.0, 110.0],
                "low": [90.0, 190.0, 90.0],
                "close": [100.0, 200.0, 100.0],
                "volume": [10.0, 10.0, 10.0],
                "turnover": [1000.0, 2000.0, 1000.0],
            }
        ),
        "ALTUSDTM": _df(
            {
                "open": [10.0, 40.0, 30.0],
                "high": [12.0, 44.0, 33.0],
                "low": [9.0, 38.0, 27.0],
                "close": [10.0, 50.0, 25.0],
                "volume": [100.0, 200.0, 300.0],
                "turnover": [1000.0, 10000.0, 7500.0],
            }
        ),
    }

    mats, meta = build_btc_terms_matrices(all_data, btc_symbol="XBTUSDTM")

    assert "XBTUSDTM" not in mats["close"].columns
    assert meta["n_symbols_written"] == 1

    close = mats["close"]["ALTUSDTM"]
    pd.testing.assert_series_equal(close, pd.Series([0.10, 0.25, 0.25], index=close.index, name="ALTUSDTM"))

    returns = mats["returns"]["ALTUSDTM"]
    assert pd.isna(returns.iloc[0])
    assert returns.iloc[1] == 1.5
    assert returns.iloc[2] == 0.0

    # Native contract/base volume is not divided by BTC; USDT notional is.
    assert mats["volume"]["ALTUSDTM"].tolist() == [100.0, 200.0, 300.0]
    btc_typical_0 = (110.0 + 90.0 + 100.0) / 3.0
    assert mats["turnover"]["ALTUSDTM"].iloc[0] == 1000.0 / btc_typical_0


def test_range_envelope_keeps_synthetic_ohlc_non_inverted():
    all_data = {
        "XBTUSDTM": _df(
            {
                "open": [100.0, 100.0, 100.0],
                "high": [130.0, 130.0, 130.0],
                "low": [70.0, 70.0, 70.0],
                "close": [120.0, 120.0, 120.0],
                "volume": [1.0, 1.0, 1.0],
                "turnover": [100.0, 100.0, 100.0],
            }
        ),
        "ALTUSDTM": _df(
            {
                "open": [10.0, 10.0, 10.0],
                "high": [11.0, 11.0, 11.0],
                "low": [9.0, 9.0, 9.0],
                "close": [10.0, 10.0, 10.0],
                "volume": [1.0, 1.0, 1.0],
                "turnover": [10.0, 10.0, 10.0],
            }
        ),
    }

    mats, meta = build_btc_terms_matrices(all_data, btc_symbol="XBTUSDTM", ohlc_mode="range_envelope")

    assert meta["high_low_inversions_after_sanitize"] == 0
    assert (mats["high"] >= mats["low"]).all().all()
    assert (mats["high"] >= mats["open"]).all().all()
    assert (mats["low"] <= mats["close"]).all().all()
