"""
In-Memory DataContext — loads from SyntheticDataset or Parquet files.

This is the research/testing implementation of the DataContext protocol.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any

import numpy as np
import pandas as pd

from src.data.synthetic import SyntheticDataset


class InMemoryDataContext:
    """
    DataContext backed by in-memory DataFrames.

    Loads data from a SyntheticDataset or from Parquet files.
    Provides all DataContext protocol methods.
    """

    def __init__(self, dataset: SyntheticDataset | None = None) -> None:
        # Price data: pivoted to ticker columns for fast matrix access
        self._prices_raw: pd.DataFrame = pd.DataFrame()
        self._price_matrices: dict[str, pd.DataFrame] = {}  # field -> (dates x tickers)
        self._fundamentals: pd.DataFrame = pd.DataFrame()
        self._estimates: pd.DataFrame = pd.DataFrame()
        self._classifications: dict[str, dict[str, str]] = {}  # ticker -> {sector, industry, subindustry}
        self._universes: dict[str, dict[str, list[str]]] = {}  # universe -> {date_str -> [tickers]}
        self._trading_days: list[dt.date] = []

        if dataset is not None:
            self._load_from_dataset(dataset)

    def _load_from_dataset(self, ds: SyntheticDataset) -> None:
        """Load all data from a SyntheticDataset into efficient lookup structures."""
        self._prices_raw = ds.prices.copy()
        self._trading_days = sorted(ds.trading_days)

        # Build pivoted price matrices for each field
        price_fields = ["open", "high", "low", "close", "volume", "vwap", "returns", "adv20", "sharesout"]
        for field in price_fields:
            if field in self._prices_raw.columns:
                pivot = self._prices_raw.pivot_table(
                    index="date", columns="ticker", values=field, aggfunc="first"
                )
                pivot = pivot.sort_index()
                self._price_matrices[field] = pivot

        # Fundamentals
        self._fundamentals = ds.fundamentals.copy()

        # Estimates
        self._estimates = ds.estimates.copy()

        # Classifications lookup
        for _, row in ds.classifications.iterrows():
            self._classifications[row["ticker"]] = {
                "sector": row["sector"],
                "industry": row["industry"],
                "subindustry": row["subindustry"],
            }

        # Universe lookup: universe -> {date -> [tickers]}
        for univ_name in ds.universes["universe"].unique():
            univ_data = ds.universes[ds.universes["universe"] == univ_name]
            date_groups: dict[str, list[str]] = {}
            for _, row in univ_data.iterrows():
                d = str(row["date"])
                if d not in date_groups:
                    date_groups[d] = []
                date_groups[d].append(row["ticker"])
            self._universes[univ_name] = date_groups

    def load_from_matrices(
        self,
        matrices: dict[str, pd.DataFrame],
        classifications: dict[str, dict[str, str]] | None = None,
    ) -> None:
        """
        Load directly from pre-built wide-format matrices.

        Args:
            matrices: Dict of field_name → DataFrame (dates × tickers).
                      Must include at least 'close'.
            classifications: Optional dict of ticker → {sector, industry, subindustry}.
        """
        self._price_matrices = {}
        for field, df in matrices.items():
            self._price_matrices[field] = df.sort_index()

        # Infer trading days from close matrix
        if "close" in matrices:
            idx = matrices["close"].index
            if hasattr(idx[0], "date"):
                self._trading_days = sorted([d.date() for d in idx])
            else:
                self._trading_days = sorted(idx.tolist())

        if classifications:
            self._classifications = classifications

    def load_from_cache(self, cache_dir: str = "data/fmp_cache") -> None:
        """
        Load pre-built matrices from the bulk download cache.

        Reads all .parquet files from cache_dir/matrices/ and
        the classifications.json for sector/industry groups.
        """
        import os

        matrices_dir = os.path.join(cache_dir, "matrices")
        if not os.path.exists(matrices_dir):
            raise FileNotFoundError(f"No matrices directory: {matrices_dir}")

        matrices = {}
        for fname in os.listdir(matrices_dir):
            if fname.endswith(".parquet"):
                field = fname.replace(".parquet", "")
                if field.startswith("_"):
                    continue  # Skip helper matrices
                df = pd.read_parquet(os.path.join(matrices_dir, fname))
                matrices[field] = df

        # Classifications
        classifications = None
        cls_path = os.path.join(cache_dir, "classifications.json")
        if os.path.exists(cls_path):
            import json
            with open(cls_path) as f:
                classifications = json.load(f)

        self.load_from_matrices(matrices, classifications)

        # Also load group series for neutralization
        for group_name in ["_sector_groups", "_industry_groups"]:
            gpath = os.path.join(matrices_dir, f"{group_name}.parquet")
            if os.path.exists(gpath):
                gs = pd.read_parquet(gpath)
                if isinstance(gs, pd.DataFrame) and len(gs.columns) == 1:
                    self._price_matrices[group_name] = gs.iloc[:, 0]
                else:
                    self._price_matrices[group_name] = gs



    def load_from_parquet(self, data_dir: str) -> None:
        """Load data from Parquet files in the given directory."""
        import os

        prices_path = os.path.join(data_dir, "prices.parquet")
        if os.path.exists(prices_path):
            self._prices_raw = pd.read_parquet(prices_path)
            # Ensure date column is date type
            if "date" in self._prices_raw.columns:
                self._prices_raw["date"] = pd.to_datetime(self._prices_raw["date"]).dt.date
            self._trading_days = sorted(self._prices_raw["date"].unique().tolist())

            # Build matrices
            for field in ["open", "high", "low", "close", "volume", "vwap", "returns", "adv20", "sharesout"]:
                if field in self._prices_raw.columns:
                    pivot = self._prices_raw.pivot_table(
                        index="date", columns="ticker", values=field, aggfunc="first"
                    )
                    self._price_matrices[field] = pivot.sort_index()

        fund_path = os.path.join(data_dir, "fundamentals.parquet")
        if os.path.exists(fund_path):
            self._fundamentals = pd.read_parquet(fund_path)

        est_path = os.path.join(data_dir, "estimates.parquet")
        if os.path.exists(est_path):
            self._estimates = pd.read_parquet(est_path)

        class_path = os.path.join(data_dir, "classifications.parquet")
        if os.path.exists(class_path):
            cdf = pd.read_parquet(class_path)
            for _, row in cdf.iterrows():
                self._classifications[row["ticker"]] = {
                    "sector": row["sector"],
                    "industry": row["industry"],
                    "subindustry": row["subindustry"],
                }

        univ_path = os.path.join(data_dir, "universes.parquet")
        if os.path.exists(univ_path):
            udf = pd.read_parquet(univ_path)
            for univ_name in udf["universe"].unique():
                univ_data = udf[udf["universe"] == univ_name]
                date_groups: dict[str, list[str]] = {}
                for _, row in univ_data.iterrows():
                    d = str(row["date"])
                    if d not in date_groups:
                        date_groups[d] = []
                    date_groups[d].append(row["ticker"])
                self._universes[univ_name] = date_groups

    # -------------------------------------------------------------------
    # DataContext Protocol Implementation
    # -------------------------------------------------------------------

    def get_price(self, ticker: str, field: str, date: dt.date) -> float:
        """Get a single price/volume value."""
        mat = self._price_matrices.get(field)
        if mat is None:
            return float("nan")
        if date not in mat.index or ticker not in mat.columns:
            return float("nan")
        val = mat.at[date, ticker]
        if pd.isna(val):
            return float("nan")
        return float(val)

    def get_fundamental(self, ticker: str, field: str, date: dt.date, pit: bool = True) -> float:
        """Get fundamental value with point-in-time semantics."""
        if self._fundamentals.empty:
            return float("nan")

        mask = self._fundamentals["ticker"] == ticker
        mask &= self._fundamentals["field"] == field

        # Compare date-to-date (avoid Timestamp vs date mismatch)
        date_col = "filing_date" if pit else "report_date"
        col_vals = self._fundamentals[date_col]
        mask &= col_vals.apply(lambda d: d <= date if isinstance(d, dt.date) else pd.Timestamp(d).date() <= date)

        subset = self._fundamentals[mask]
        if subset.empty:
            return float("nan")

        latest = subset.sort_values(date_col).iloc[-1]
        return float(latest["value"])

    def get_estimate(self, ticker: str, field: str, date: dt.date, period: str = "FY1") -> float:
        """Get analyst consensus estimate."""
        if self._estimates.empty:
            return float("nan")

        mask = (
            (self._estimates["ticker"] == ticker)
            & (self._estimates["field"] == field)
            & (self._estimates["period"] == period)
            & (self._estimates["date"].apply(lambda d: d <= date if isinstance(d, dt.date) else pd.Timestamp(d).date() <= date))
        )
        subset = self._estimates[mask]
        if subset.empty:
            return float("nan")

        latest = subset.sort_values("date").iloc[-1]
        return float(latest["consensus"])

    def get_universe(self, date: dt.date, universe: str = "TOP3000") -> list[str]:
        """Return list of tickers in universe on given date."""
        univ = self._universes.get(universe)
        if univ is None:
            # Fallback: return all tickers with data on this date
            if "close" in self._price_matrices:
                mat = self._price_matrices["close"]
                if date in mat.index:
                    return [t for t in mat.columns if not pd.isna(mat.at[date, t])]
            return []

        # Find the most recent rebalance date <= date
        date_str = str(date)
        all_dates = sorted(univ.keys())

        best_date = None
        for d in all_dates:
            if d <= date_str:
                best_date = d

        if best_date is None:
            return []
        return univ[best_date]

    def get_industry(self, ticker: str, date: dt.date, level: str = "industry") -> str:
        """Return classification at given level."""
        info = self._classifications.get(ticker)
        if info is None:
            return "Unknown"
        return info.get(level, "Unknown")

    def get_matrix(
        self, field: str, date: dt.date, lookback: int, universe: str = "TOP3000"
    ) -> pd.DataFrame:
        """
        Return (lookback x n_instruments) DataFrame.

        Rows = dates (ascending), Columns = tickers in universe.
        """
        mat = self._price_matrices.get(field)
        if mat is None:
            return pd.DataFrame()

        # Find the date index
        if date not in mat.index:
            # Find closest date <= date
            valid_dates = mat.index[mat.index <= date]
            if len(valid_dates) == 0:
                return pd.DataFrame()
            date = valid_dates[-1]

        date_idx = mat.index.get_loc(date)
        start_idx = max(0, date_idx - lookback + 1)

        # Get universe tickers
        universe_tickers = self.get_universe(date, universe)
        if not universe_tickers:
            universe_tickers = list(mat.columns)

        # Filter to tickers present in matrix
        valid_tickers = [t for t in universe_tickers if t in mat.columns]

        return mat.iloc[start_idx:date_idx + 1][valid_tickers]

    def get_trading_days(self, start_date: dt.date, end_date: dt.date) -> list[dt.date]:
        """Return sorted list of trading days in [start_date, end_date]."""
        return [d for d in self._trading_days if start_date <= d <= end_date]
