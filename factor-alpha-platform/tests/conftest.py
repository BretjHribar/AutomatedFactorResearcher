"""
Shared test fixtures for the factor alpha platform.

Provides pre-configured synthetic datasets and DataContexts for testing.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import (
    SyntheticDataGenerator,
    SyntheticDataset,
    generate_tiny_fixture,
)
from src.data.context_research import InMemoryDataContext


@pytest.fixture(scope="session")
def tiny_dataset() -> SyntheticDataset:
    """20 stocks × 252 days. Fastest possible fixture for unit tests."""
    return generate_tiny_fixture(seed=42)


@pytest.fixture(scope="session")
def tiny_ctx(tiny_dataset: SyntheticDataset) -> InMemoryDataContext:
    """DataContext from tiny fixture."""
    return InMemoryDataContext(tiny_dataset)


@pytest.fixture
def simple_series() -> pd.Series:
    """Simple numeric Series for operator testing."""
    return pd.Series(
        [20.2, 15.6, 10.0, 5.7, 50.2, 18.4],
        index=["A", "B", "C", "D", "E", "F"],
    )


@pytest.fixture
def simple_matrix() -> pd.DataFrame:
    """Simple (5 dates × 4 tickers) matrix for time-series operator tests."""
    dates = pd.date_range("2023-01-02", periods=5, freq="B")
    data = {
        "X": [10.0, 12.0, 11.0, 13.0, 14.0],
        "Y": [20.0, 19.0, 21.0, 22.0, 20.5],
        "Z": [5.0, 5.5, 4.5, 6.0, 5.8],
        "W": [100.0, 102.0, 101.0, 103.0, 105.0],
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def groups_series() -> pd.Series:
    """Group assignments for neutralization tests."""
    return pd.Series(
        ["tech", "tech", "fin", "fin", "health", "health"],
        index=["A", "B", "C", "D", "E", "F"],
    )
