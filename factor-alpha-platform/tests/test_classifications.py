"""
tests/test_classifications.py — Regression tests for classification data integrity.

Run: python -m pytest tests/test_classifications.py -v
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CACHE_DIR = Path("data/fmp_cache")
MATRICES_DIR = CACHE_DIR / "matrices"


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cls():
    """Load classifications.json once for all tests."""
    with open(CACHE_DIR / "classifications.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def sector_mat():
    return pd.read_parquet(MATRICES_DIR / "sector.parquet")


@pytest.fixture(scope="module")
def industry_mat():
    return pd.read_parquet(MATRICES_DIR / "industry.parquet")


@pytest.fixture(scope="module")
def subindustry_mat():
    return pd.read_parquet(MATRICES_DIR / "subindustry.parquet")


@pytest.fixture(scope="module")
def close():
    return pd.read_parquet(MATRICES_DIR / "close.parquet")


# ─── Classification JSON Tests ───────────────────────────────────────────────


def test_classifications_not_empty(cls):
    """Must have at least 2000 ticker entries."""
    assert len(cls) >= 2000, f"Only {len(cls)} entries — expected >= 2000"


def test_required_keys_present(cls):
    """Every entry must have sector, industry, subindustry keys."""
    required = {"sector", "industry", "subindustry"}
    missing_any = [t for t, v in cls.items() if not required.issubset(v.keys())]
    assert not missing_any, f"Missing required keys for: {missing_any[:10]}"


def test_sector_not_all_same(cls):
    """Sectors must have at least 5 distinct values (not collapsed to one)."""
    unique_sectors = set(v["sector"] for v in cls.values())
    assert len(unique_sectors) >= 5, f"Only {len(unique_sectors)} unique sectors — data may be collapsed"


def test_industry_more_granular_than_sector(cls):
    """Industries must have strictly more unique values than sectors."""
    unique_sectors = set(v["sector"] for v in cls.values())
    unique_industries = set(v["industry"] for v in cls.values())
    assert len(unique_industries) > len(unique_sectors), (
        f"industries ({len(unique_industries)}) must be > sectors ({len(unique_sectors)})"
    )


def test_subindustry_more_granular_than_industry(cls):
    """Subindustries must have strictly more unique values than industries."""
    unique_industries = set(v["industry"] for v in cls.values())
    unique_subs = set(v["subindustry"] for v in cls.values())
    assert len(unique_subs) > len(unique_industries), (
        f"subindustries ({len(unique_subs)}) must be > industries ({len(unique_industries)})"
    )


def test_industry_subindustry_not_identical(cls):
    """Industry and subindustry must differ for the majority of tickers.

    Previously broken: both were stored as 3-digit codes.
    """
    same_count = sum(1 for v in cls.values() if v["industry"] == v["subindustry"])
    pct_same = same_count / len(cls) * 100
    assert pct_same < 20, (
        f"{pct_same:.1f}% of tickers have industry == subindustry — "
        "this likely means classification build is broken (should be < 20%)"
    )


def test_no_unknown_sector_majority(cls):
    """'Unknown' sector should not dominate (max 20% of tickers)."""
    unknown_count = sum(1 for v in cls.values() if v.get("sector", "").lower() in ("unknown", ""))
    pct = unknown_count / len(cls) * 100
    assert pct < 20, f"{pct:.1f}% of tickers have Unknown sector — coverage too low"


# ─── Matrix Parquet Tests ─────────────────────────────────────────────────────


def test_sector_matrix_exists():
    assert (MATRICES_DIR / "sector.parquet").exists(), "sector.parquet is missing"


def test_industry_matrix_exists():
    assert (MATRICES_DIR / "industry.parquet").exists(), "industry.parquet is missing"


def test_subindustry_matrix_exists():
    assert (MATRICES_DIR / "subindustry.parquet").exists(), "subindustry.parquet is missing"


def test_sector_unique_count(sector_mat):
    """Sector matrix must have 8-15 unique values (SIC divisions)."""
    n = sector_mat.iloc[-1].nunique()
    assert 8 <= n <= 15, f"sector.parquet has {n} unique values (expected 8-15)"


def test_industry_unique_count(industry_mat):
    """Industry matrix must have at least 50 unique values."""
    n = industry_mat.iloc[-1].nunique()
    assert n >= 50, f"industry.parquet has only {n} unique values (expected >= 50)"


def test_subindustry_unique_count_exceeds_industry(industry_mat, subindustry_mat):
    """Subindustry parquet must have more unique groups than industry parquet."""
    n_ind = industry_mat.iloc[-1].nunique()
    n_sub = subindustry_mat.iloc[-1].nunique()
    assert n_sub > n_ind, (
        f"subindustry.parquet ({n_sub}) must have > industry.parquet ({n_ind}) unique groups — "
        "this was the regression: both had 239 groups"
    )


def test_matrices_not_identical(industry_mat, subindustry_mat):
    """industry and subindustry matrix rows must differ for most tickers.

    Core regression test: previously industry.parquet == subindustry.parquet entirely.
    """
    last_ind = industry_mat.iloc[-1]
    last_sub = subindustry_mat.iloc[-1]
    pct_same = (last_ind == last_sub).mean() * 100
    assert pct_same < 20, (
        f"{pct_same:.1f}% of tickers have the same industry and subindustry group ID — "
        "matrices may be duplicates of each other"
    )


def test_matrices_are_integer_encoded(sector_mat, industry_mat, subindustry_mat):
    """All classification matrices must be integer (label-encoded)."""
    for name, mat in [("sector", sector_mat), ("industry", industry_mat), ("subindustry", subindustry_mat)]:
        assert np.issubdtype(mat.dtypes.iloc[0], np.integer), (
            f"{name}.parquet dtype is {mat.dtypes.iloc[0]} — expected integer (label-encoded)"
        )


def test_matrices_no_negative_values(sector_mat, industry_mat, subindustry_mat):
    """Label-encoded IDs must be non-negative."""
    for name, mat in [("sector", sector_mat), ("industry", industry_mat), ("subindustry", subindustry_mat)]:
        assert (mat.values >= 0).all(), f"{name}.parquet has negative values"


def test_matrices_ticker_alignment(sector_mat, industry_mat, subindustry_mat, close):
    """Classification matrices must cover all tickers in close.parquet."""
    close_tickers = set(close.columns)
    for name, mat in [("sector", sector_mat), ("industry", industry_mat), ("subindustry", subindustry_mat)]:
        mat_tickers = set(mat.columns)
        missing = close_tickers - mat_tickers
        assert not missing, (
            f"{name}.parquet missing {len(missing)} tickers from close.parquet: {list(missing)[:10]}"
        )


def test_matrices_constant_over_time(sector_mat):
    """Classification matrices should be time-constant (sector doesn't change daily)."""
    # Check that first and last row are identical
    first = sector_mat.iloc[0]
    last = sector_mat.iloc[-1]
    pct_changed = (first != last).mean() * 100
    assert pct_changed < 1, (
        f"{pct_changed:.1f}% of tickers changed sector between first and last date — "
        "classifications should be static"
    )


# ─── Integration: load_data returns correct structure ────────────────────────


def test_load_data_classifications_structure():
    """eval_alpha_ib.load_data() must return classifications as a dict with all 3 levels."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import eval_alpha_ib
    _, _, cls_loaded = eval_alpha_ib.load_data("full")
    assert isinstance(cls_loaded, dict), "classifications should be a dict"
    for level in ("sector", "industry", "subindustry"):
        assert level in cls_loaded, f"Missing '{level}' key in loaded classifications"
        assert isinstance(cls_loaded[level], pd.Series), f"'{level}' should be a pd.Series"
        n = cls_loaded[level].nunique()
        assert n > 1, f"'{level}' has only {n} unique values after load"


def test_load_data_industry_vs_subindustry_differ():
    """Loaded industry and subindustry series must be distinct."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import eval_alpha_ib
    _, _, cls_loaded = eval_alpha_ib.load_data("full")
    if "industry" in cls_loaded and "subindustry" in cls_loaded:
        ind = cls_loaded["industry"]
        sub = cls_loaded["subindustry"]
        aligned = ind.reindex(sub.index)
        pct_same = (aligned == sub).mean() * 100
        assert pct_same < 20, (
            f"{pct_same:.1f}% of loaded industry == subindustry — "
            "loaded classification data is incorrect"
        )
