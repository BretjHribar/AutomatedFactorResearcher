"""Byte-exact verification of bounded-history signal compute.

Builds a synthetic universe + matrices + alpha DB on disk under tmp_path, then
runs the canonical pipeline twice — once with full history, once with a bounded
slice — and asserts the LAST row of the produced weights matrices is identical
within tight fp64 tolerance.

This is the contract the Dagster signal-snapshot assets rely on for incremental
recompute. If a future alpha or combiner change introduces a longer-than-bound
lookback, this test fails loudly and the operator must bump the max_lookback_bars.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pipeline.signal_service import (
    latest_signal_snapshot,
    verify_incremental_signal_matches,
)


# ---------------------------------------------------------------------------
# Synthetic data setup
# ---------------------------------------------------------------------------


def _make_synthetic_root(tmp_path: Path, *, n_bars: int, n_tickers: int = 12, seed: int = 7):
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="D")

    # Random-walk close, log-returns, volume, etc.
    log_ret = rng.normal(0.0, 0.01, size=(n_bars, n_tickers))
    close = pd.DataFrame(np.exp(np.cumsum(log_ret, axis=0)) * 100.0,
                         index=dates, columns=tickers)
    volume = pd.DataFrame(rng.lognormal(15.0, 0.5, size=(n_bars, n_tickers)),
                          index=dates, columns=tickers)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = close * (1 + rng.uniform(0, 0.01, size=(n_bars, n_tickers)))
    low = close * (1 - rng.uniform(0, 0.01, size=(n_bars, n_tickers)))
    returns = close.pct_change(fill_method=None).fillna(0.0)

    matrices_dir = tmp_path / "data" / "fmp_cache" / "matrices"
    matrices_dir.mkdir(parents=True)
    close.to_parquet(matrices_dir / "close.parquet")
    open_.to_parquet(matrices_dir / "open.parquet")
    high.to_parquet(matrices_dir / "high.parquet")
    low.to_parquet(matrices_dir / "low.parquet")
    volume.to_parquet(matrices_dir / "volume.parquet")
    returns.to_parquet(matrices_dir / "returns.parquet")

    # All tickers active throughout — coverage 100%, threshold 0.5.
    universes_dir = tmp_path / "data" / "fmp_cache" / "universes"
    universes_dir.mkdir(parents=True)
    universe = pd.DataFrame(True, index=dates, columns=tickers)
    universe.to_parquet(universes_dir / "TEST_UNIVERSE.parquet")

    # Alpha DB with two ts_* alphas. ts_* operators have STRICT bounded lookback —
    # if N >= window, the last-bar value is byte-exact equal between bounded and
    # full runs. Decay_exp is tested separately because EWM has an exponentially
    # decaying tail that crosses the slice boundary.
    db_path = tmp_path / "data" / "alpha_results.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE alphas (id INTEGER PRIMARY KEY, expression TEXT, archived INTEGER)")
        conn.executemany("INSERT INTO alphas VALUES (?, ?, 0)", [
            (1, "ts_zscore(close, 60)"),
            (2, "ts_zscore(sma(close, 30), 60)"),
        ])
        conn.commit()

    config = {
        "market": "equity",
        "interval": "1d",
        "annualization": {"bars_per_year": 252},
        "data": {
            "matrices_dir": "data/fmp_cache/matrices",
            "universe_path": "data/fmp_cache/universes/TEST_UNIVERSE.parquet",
            "universe_filter": {"method": "coverage", "threshold": 0.5},
            "returns_source": "matrix:returns",
        },
        "alpha_source": {
            "db_path": "data/alpha_results.db",
            "table": "alphas",
            "filter_sql": "archived=0",
            "train_sharpe_table": None,
            "train_sharpe_column": None,
        },
        "preprocessing": {
            "universe_mask": False,
            "demean_method": "cross_section",
            "normalize": "l1",
            "clip_max_w": None,
        },
        "combiner": {"name": "equal", "params": {"max_wt": 0.5}},
        "post_combiner": {"renormalize_l1": True, "clip_max_w": None},
        "risk_model": {"name": "diagonal", "params": {"vol_window": 30, "factor_window": 60, "n_pca_factors": 3}},
        "qp": {"enabled": False},
        "fees": {"model": "bps_taker", "params": {"taker_bps": 0.0, "slippage_bps": 0.0}},
        "splits": {"train_end": "2024-04-01", "val_end": "2024-08-01"},
        "book": 100000,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_incremental_matches_full_when_lookback_covers_chained_ts_ops(tmp_path):
    """N=200 covers the chained ts_zscore(sma(close,30),60) (need ≥ 90) plus
    enough Decay_exp tail for fp64 cleanliness at α=0.05."""
    config_path = _make_synthetic_root(tmp_path, n_bars=400)

    stats = verify_incremental_signal_matches(
        config_path,
        max_lookback_bars=200,
        root=tmp_path,
        atol=1e-10,
    )
    assert stats["passed"] is True
    assert stats["max_abs_diff"] <= 1e-10


def test_incremental_diverges_when_lookback_starves_chained_lookback(tmp_path):
    """If max_lookback_bars < the chained ts_* requirement, the LAST row of the
    incremental run is NaN/zero (alphas can't be evaluated) and the assertion fails."""
    config_path = _make_synthetic_root(tmp_path, n_bars=400)

    with pytest.raises(AssertionError):
        verify_incremental_signal_matches(
            config_path,
            max_lookback_bars=50,  # < 90 needed by alpha 2
            root=tmp_path,
            atol=1e-10,
        )


def test_latest_signal_snapshot_passes_through_max_lookback(tmp_path):
    """The signal service's LatestSignalSnapshot reports the effective lookback used."""
    config_path = _make_synthetic_root(tmp_path, n_bars=300)

    snap = latest_signal_snapshot(
        config_path, root=tmp_path, max_lookback_bars=200,
    )
    assert snap.max_lookback_bars == 200
    # Bounded-mode notes flag picked up by the runner.
    assert any("max_lookback_bars=200" in n for n in snap.config_notes)


def test_full_compute_unaffected_when_max_lookback_is_none(tmp_path):
    """Default behaviour — no slicing — must produce identical results to
    explicitly passing a max_lookback_bars >= history length."""
    config_path = _make_synthetic_root(tmp_path, n_bars=300)

    full = latest_signal_snapshot(config_path, root=tmp_path)
    bounded = latest_signal_snapshot(
        config_path, root=tmp_path, max_lookback_bars=300,
    )
    assert full.weights == bounded.weights
    assert full.signal_date == bounded.signal_date


def _make_decay_exp_root(tmp_path: Path, *, n_bars: int, alpha_exp: float, seed: int = 11):
    rng = np.random.default_rng(seed)
    n_tickers = 8
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="D")
    log_ret = rng.normal(0.0, 0.01, size=(n_bars, n_tickers))
    close = pd.DataFrame(np.exp(np.cumsum(log_ret, axis=0)) * 100.0,
                         index=dates, columns=tickers)
    volume = pd.DataFrame(rng.lognormal(15.0, 0.5, size=(n_bars, n_tickers)),
                          index=dates, columns=tickers)
    returns = close.pct_change(fill_method=None).fillna(0.0)

    matrices_dir = tmp_path / "data" / "fmp_cache" / "matrices"
    matrices_dir.mkdir(parents=True)
    close.to_parquet(matrices_dir / "close.parquet")
    volume.to_parquet(matrices_dir / "volume.parquet")
    returns.to_parquet(matrices_dir / "returns.parquet")
    universes_dir = tmp_path / "data" / "fmp_cache" / "universes"
    universes_dir.mkdir(parents=True)
    pd.DataFrame(True, index=dates, columns=tickers).to_parquet(
        universes_dir / "TEST_UNIVERSE.parquet"
    )

    db_path = tmp_path / "data" / "alpha_results.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE alphas (id INTEGER PRIMARY KEY, expression TEXT, archived INTEGER)")
        conn.executemany("INSERT INTO alphas VALUES (?, ?, 0)", [
            (1, f"Decay_exp(returns, {alpha_exp})"),
        ])
        conn.commit()

    config = {
        "market": "equity",
        "interval": "1d",
        "annualization": {"bars_per_year": 252},
        "data": {
            "matrices_dir": "data/fmp_cache/matrices",
            "universe_path": "data/fmp_cache/universes/TEST_UNIVERSE.parquet",
            "universe_filter": {"method": "coverage", "threshold": 0.5},
            "returns_source": "matrix:returns",
        },
        "alpha_source": {
            "db_path": "data/alpha_results.db",
            "table": "alphas", "filter_sql": "archived=0",
            "train_sharpe_table": None, "train_sharpe_column": None,
        },
        "preprocessing": {"universe_mask": False, "demean_method": "cross_section", "normalize": "l1", "clip_max_w": None},
        "combiner": {"name": "equal", "params": {"max_wt": 0.5}},
        "post_combiner": {"renormalize_l1": True, "clip_max_w": None},
        "risk_model": {"name": "diagonal", "params": {"vol_window": 30, "factor_window": 60, "n_pca_factors": 3}},
        "qp": {"enabled": False},
        "fees": {"model": "bps_taker", "params": {"taker_bps": 0.0, "slippage_bps": 0.0}},
        "splits": {"train_end": "2024-04-01", "val_end": "2024-08-01"},
        "book": 100000,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_decay_exp_alpha_005_diverges_when_lookback_under_threshold(tmp_path):
    """Decay_exp(α=0.05): truncation error at last bar ≈ (1-α)^N. N=200 → 3.5e-5
    weight tail; after L1 normalization that's ~1e-6 per ticker — far above
    fp64 atol=1e-10. Must fail loud."""
    config_path = _make_decay_exp_root(tmp_path, n_bars=800, alpha_exp=0.05)
    with pytest.raises(AssertionError):
        verify_incremental_signal_matches(
            config_path, max_lookback_bars=200, root=tmp_path, atol=1e-10,
        )


def test_decay_exp_alpha_005_byte_exact_at_n_600(tmp_path):
    """N=600 → (1-0.05)^600 ≈ 1.5e-14, well under fp64 atol=1e-10."""
    config_path = _make_decay_exp_root(tmp_path, n_bars=800, alpha_exp=0.05)
    stats = verify_incremental_signal_matches(
        config_path, max_lookback_bars=600, root=tmp_path, atol=1e-10,
    )
    assert stats["passed"]
    assert stats["max_abs_diff"] <= 1e-10


def test_decay_exp_alpha_002_needs_deeper_lookback(tmp_path):
    """α=0.02 → need K≥1500 for fp64-clean (1-0.02)^1500 ≈ 8e-14."""
    config_path = _make_decay_exp_root(tmp_path, n_bars=2000, alpha_exp=0.02)
    # Tight bound: 200 bars far too short.
    with pytest.raises(AssertionError):
        verify_incremental_signal_matches(
            config_path, max_lookback_bars=200, root=tmp_path, atol=1e-10,
        )
    # Sufficient bound.
    stats = verify_incremental_signal_matches(
        config_path, max_lookback_bars=1500, root=tmp_path, atol=1e-10,
    )
    assert stats["passed"]
