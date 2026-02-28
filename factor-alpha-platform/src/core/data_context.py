"""
DataContext protocol — the abstract data access layer.

Factor code accesses data exclusively through this interface.
Research and production implement differently, factor code stays the same.
"""

from __future__ import annotations

import datetime as dt
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DataContext(Protocol):
    """
    Abstract data access layer.

    Research uses DuckDB + Parquet files.
    Production uses PostgreSQL / TimescaleDB.
    Testing uses in-memory synthetic data.
    """

    def get_price(self, ticker: str, field: str, date: dt.date) -> float:
        """
        Get a price/volume field for a ticker on a date.

        Fields: open, high, low, close, vwap, volume, returns, adv20, sharesout
        """
        ...

    def get_fundamental(self, ticker: str, field: str, date: dt.date, pit: bool = True) -> float:
        """
        Get a fundamental data field for a ticker as of a date.

        Fields: sales, income, eps, assets, equity, debt, ebitda, etc.
        pit=True uses point-in-time (filing date) values to avoid look-ahead bias.
        """
        ...

    def get_estimate(self, ticker: str, field: str, date: dt.date, period: str = "FY1") -> float:
        """
        Get analyst consensus estimate for a ticker.

        Fields: eps, revenue, ebitda
        Periods: FY1, FY2, Q0, Q1, Q2
        """
        ...

    def get_universe(self, date: dt.date, universe: str = "TOP3000") -> list[str]:
        """
        Return list of tickers in universe on given date.

        Universes: TOP200, TOP500, TOP1000, TOP2000, TOP3000
        """
        ...

    def get_industry(self, ticker: str, date: dt.date, level: str = "industry") -> str:
        """
        Return industry/sector/subindustry classification.

        Levels: sector, industry, subindustry
        """
        ...

    def get_matrix(
        self, field: str, date: dt.date, lookback: int, universe: str = "TOP3000"
    ) -> pd.DataFrame:
        """
        Return (lookback x n_instruments) DataFrame of field values.

        Rows = dates (ascending, oldest first), Columns = tickers.
        This is the primary bulk-access method for vectorized factor computation.
        """
        ...

    def get_trading_days(self, start_date: dt.date, end_date: dt.date) -> list[dt.date]:
        """Return sorted list of trading days in [start_date, end_date]."""
        ...
