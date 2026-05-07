from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from prod.data_refresh import _write_top_universe
from src.data.integrity import run_crypto_integrity


def _write_alpha_db(root: Path) -> None:
    db_path = root / "data" / "alpha_results.db"
    db_path.parent.mkdir(parents=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE alphas_crypto (archived INTEGER)")
        conn.execute("INSERT INTO alphas_crypto VALUES (0)")
        conn.commit()


def _write_matrices(root: Path, index: pd.DatetimeIndex) -> Path:
    matrices = root / "matrices"
    matrices.mkdir()
    close = pd.DataFrame(
        {
            "BTCUSDTM": [100 + i for i in range(len(index))],
            "ETHUSDTM": [50 + i for i in range(len(index))],
            "SOLUSDTM": [20 + i for i in range(len(index))],
        },
        index=index,
    )
    open_ = close - 0.5
    high = close + 1.0
    low = close - 1.0
    volume = close * 10
    for name, frame in {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }.items():
        frame.to_parquet(matrices / f"{name}.parquet")
    return matrices


def _write_universe(root: Path, index: pd.DatetimeIndex) -> Path:
    path = root / "universes" / "KUCOIN_TOP2_4h.parquet"
    path.parent.mkdir()
    pd.DataFrame(
        {
            "BTCUSDTM": [True] * len(index),
            "ETHUSDTM": [True] * len(index),
            "SOLUSDTM": [False] * len(index),
        },
        index=index,
    ).to_parquet(path)
    return path


def _statuses(results: list) -> dict[str, str]:
    return {result.name: result.status for result in results}


def test_crypto_integrity_passes_for_current_contiguous_4h_cache(tmp_path):
    index = pd.date_range("2026-05-07 00:00:00", periods=3, freq="4h")
    _write_matrices(tmp_path, index)
    universe = _write_universe(tmp_path, index)
    _write_alpha_db(tmp_path)

    results = run_crypto_integrity(
        tmp_path,
        matrices_rel="matrices",
        universe_rel=str(universe.relative_to(tmp_path)),
        expected_universe_size=2,
        now_ts="2026-05-07T12:20:00Z",
    )

    statuses = _statuses(results)
    assert statuses["crypto_latest_bar_freshness"] == "pass"
    assert statuses["crypto_bar_index_continuity"] == "pass"
    assert statuses["crypto_latest_coverage"] == "pass"
    assert statuses["crypto_universe_current"] == "pass"
    assert statuses["crypto_universe_membership"] == "pass"


def test_crypto_integrity_fails_on_recent_4h_gap(tmp_path):
    index = pd.DatetimeIndex(["2026-05-07 00:00:00", "2026-05-07 08:00:00"])
    _write_matrices(tmp_path, index)
    universe = _write_universe(tmp_path, index)
    _write_alpha_db(tmp_path)

    results = run_crypto_integrity(
        tmp_path,
        matrices_rel="matrices",
        universe_rel=str(universe.relative_to(tmp_path)),
        expected_universe_size=2,
        now_ts="2026-05-07T12:20:00Z",
    )

    assert _statuses(results)["crypto_bar_index_continuity"] == "fail"


def test_crypto_integrity_fails_when_universe_lags_close_matrix(tmp_path):
    index = pd.date_range("2026-05-07 00:00:00", periods=3, freq="4h")
    _write_matrices(tmp_path, index)
    universe = _write_universe(tmp_path, index[:-1])
    _write_alpha_db(tmp_path)

    results = run_crypto_integrity(
        tmp_path,
        matrices_rel="matrices",
        universe_rel=str(universe.relative_to(tmp_path)),
        expected_universe_size=2,
        now_ts="2026-05-07T12:20:00Z",
    )

    assert _statuses(results)["crypto_universe_current"] == "fail"


def test_top_universe_writer_matches_live_adv_rank_rule(tmp_path):
    adv = pd.DataFrame(
        {
            "BTCUSDTM": [100.0, 100.0],
            "ETHUSDTM": [90.0, 60.0],
            "SOLUSDTM": [80.0, 110.0],
        },
        index=pd.date_range("2026-05-07 00:00:00", periods=2, freq="4h"),
    )
    output_path = tmp_path / "universes" / "KUCOIN_LIVE_TOP2_4h.parquet"

    _write_top_universe(adv, output_path, size=2)

    saved = pd.read_parquet(output_path)
    assert saved.iloc[-1].to_dict() == {
        "BTCUSDTM": True,
        "ETHUSDTM": False,
        "SOLUSDTM": True,
    }


def test_top_universe_writer_refuses_to_clobber_curated_path(tmp_path):
    """The writer must refuse non-LIVE filenames to protect the research universe."""
    import pytest

    adv = pd.DataFrame(
        {"BTCUSDTM": [1.0]}, index=pd.to_datetime(["2026-05-07"]),
    )
    curated = tmp_path / "universes" / "KUCOIN_TOP2_4h.parquet"
    with pytest.raises(ValueError, match="LIVE"):
        _write_top_universe(adv, curated, size=2)
