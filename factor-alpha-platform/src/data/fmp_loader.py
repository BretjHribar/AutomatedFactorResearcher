"""
FMP (Financial Modeling Prep) Data Loader.

Fetches real US equity data from the FMP API and converts it into
the InMemoryDataContext format used by the alpha research platform.

Free tier limits:
- 250 API requests/day
- 5 years of historical data
- Single symbol per request (no batch)
- End-of-day prices only

Caches all data to disk (Parquet) to avoid re-fetching.

Usage:
    loader = FMPDataLoader(api_key="YOUR_KEY")
    ctx = loader.build_context(
        symbols=["AAPL", "MSFT", "GOOGL"],
        start_date="2021-01-01",
        end_date="2025-12-31",
    )
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from src.data.context_research import InMemoryDataContext

logger = logging.getLogger(__name__)

BASE_URL = "https://financialmodelingprep.com"

# Default US stock universe (mix of large/mid cap for testing)
DEFAULT_US_SYMBOLS = [
    # Tech mega-cap
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL", "CRM",
    # Tech large
    "ADBE", "CSCO", "ACN", "INTC", "AMD", "QCOM", "TXN", "AMAT", "MU", "LRCX",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "MCD", "NKE", "SBUX", "TGT", "HD",
    # Industrials
    "CAT", "DE", "UNP", "HON", "GE", "MMM", "RTX", "LMT", "BA", "UPS",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX", "OXY", "HES",
    # Materials
    "LIN", "APD", "SHW", "ECL", "DD",
    # Telecom / Utilities
    "T", "VZ", "NEE", "DUK", "SO",
    # REITs
    "AMT", "PLD", "CCI", "EQIX", "SPG",
    # Other
    "DIS", "NFLX", "PYPL", "V", "MA",
]


class FMPDataLoader:
    """Loads and caches equity data from Financial Modeling Prep API."""

    def __init__(
        self,
        api_key: str,
        cache_dir: str = "data/fmp_cache",
        requests_per_minute: int = 10,
    ):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._request_interval = 60.0 / requests_per_minute
        self._last_request_time = 0.0
        self._request_count = 0

    # ------------------------------------------------------------------
    # API methods
    # ------------------------------------------------------------------

    def _get(self, endpoint: str, params: dict | None = None) -> Any:
        """Make a rate-limited GET request to FMP API."""
        url = f"{BASE_URL}{endpoint}"
        all_params = {"apikey": self.api_key}
        if params:
            all_params.update(params)

        # Rate limiting (only delays actual network calls)
        elapsed = time.time() - self._last_request_time
        if elapsed < self._request_interval:
            time.sleep(self._request_interval - elapsed)

        try:
            r = requests.get(url, params=all_params, timeout=30)
            self._last_request_time = time.time()
            self._request_count += 1

            if r.status_code == 402:
                logger.warning(f"FMP endpoint restricted (402): {endpoint}")
                return None
            if r.status_code == 429:
                logger.warning("FMP rate limit hit, waiting 60s...")
                time.sleep(60)
                return self._get(endpoint, params)
            if r.status_code != 200:
                logger.warning(f"FMP error {r.status_code}: {endpoint} → {r.text[:100]}")
                return None

            data = r.json()
            if isinstance(data, dict) and "Error Message" in data:
                logger.warning(f"FMP error: {data['Error Message'][:100]}")
                return None
            return data

        except Exception as e:
            logger.error(f"FMP request failed: {e}")
            return None

    def _cache_path(self, category: str, symbol: str, suffix: str = "parquet") -> Path:
        """Get cache file path for a symbol's data."""
        cat_dir = self.cache_dir / category
        cat_dir.mkdir(exist_ok=True)
        return cat_dir / f"{symbol}.{suffix}"

    def _is_cached(self, category: str, symbol: str, max_age_days: int = 1, suffix: str = "parquet") -> bool:
        """Check if cached data exists and is fresh enough."""
        path = self._cache_path(category, symbol, suffix)
        if not path.exists():
            return False
        age = time.time() - path.stat().st_mtime
        return age < max_age_days * 86400

    # ------------------------------------------------------------------
    # Price data
    # ------------------------------------------------------------------

    def fetch_historical_prices(
        self,
        symbol: str,
        start_date: str = "2021-01-01",
        end_date: str | None = None,
        force: bool = False,
    ) -> pd.DataFrame | None:
        """
        Fetch daily OHLCV data for a symbol. Caches to Parquet.

        Returns DataFrame with columns: open, high, low, close, volume, vwap
        Index: DatetimeIndex
        """
        cache_key = f"prices_{start_date}"
        if not force and self._is_cached(cache_key, symbol, max_age_days=1):
            try:
                return pd.read_parquet(self._cache_path(cache_key, symbol))
            except Exception:
                pass

        params = {"symbol": symbol, "from": start_date}
        if end_date:
            params["to"] = end_date

        data = self._get("/stable/historical-price-eod/full", params)
        if not data or not isinstance(data, list):
            logger.warning(f"No price data for {symbol}")
            return None

        df = pd.DataFrame(data)
        if df.empty:
            return None

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df[["open", "high", "low", "close", "volume"]].copy()

        # Add VWAP if present
        if "vwap" in pd.DataFrame(data).columns:
            df["vwap"] = pd.DataFrame(data).set_index(pd.to_datetime(pd.DataFrame(data)["date"])).sort_index()["vwap"]

        # Cache
        try:
            df.to_parquet(self._cache_path(cache_key, symbol))
        except Exception as e:
            logger.debug(f"Cache write failed: {e}")

        return df

    # ------------------------------------------------------------------
    # Fundamental data
    # ------------------------------------------------------------------

    def fetch_profile(self, symbol: str, force: bool = False) -> dict | None:
        """Fetch company profile (sector, industry, market cap, etc.)."""
        if not force and self._is_cached("profiles", symbol, max_age_days=7, suffix="json"):
            try:
                with open(self._cache_path("profiles", symbol, "json")) as f:
                    return json.load(f)
            except Exception:
                pass

        data = self._get("/stable/profile", {"symbol": symbol})
        if data and isinstance(data, list) and len(data) > 0:
            profile = data[0]
            try:
                with open(self._cache_path("profiles", symbol, "json"), "w") as f:
                    json.dump(profile, f)
            except Exception:
                pass
            return profile
        return None

    def fetch_key_metrics(
        self, symbol: str, period: str = "annual", limit: int = 5, force: bool = False
    ) -> pd.DataFrame | None:
        """Fetch key financial metrics (enterprise value, ratios, etc.)."""
        cache_key = f"metrics_{period}"
        if not force and self._is_cached(cache_key, symbol, max_age_days=7):
            try:
                return pd.read_parquet(self._cache_path(cache_key, symbol))
            except Exception:
                pass

        data = self._get("/stable/key-metrics", {
            "symbol": symbol, "period": period, "limit": str(limit)
        })
        if not data or not isinstance(data, list):
            return None

        df = pd.DataFrame(data)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

        try:
            df.to_parquet(self._cache_path(cache_key, symbol))
        except Exception:
            pass
        return df

    def fetch_income_statement(
        self, symbol: str, period: str = "quarter", limit: int = 20, force: bool = False
    ) -> pd.DataFrame | None:
        """Fetch income statement data."""
        cache_key = f"income_{period}"
        if not force and self._is_cached(cache_key, symbol, max_age_days=7):
            try:
                return pd.read_parquet(self._cache_path(cache_key, symbol))
            except Exception:
                pass

        data = self._get("/stable/income-statement", {
            "symbol": symbol, "period": period, "limit": str(limit)
        })
        if not data or not isinstance(data, list):
            return None

        df = pd.DataFrame(data)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

        try:
            df.to_parquet(self._cache_path(cache_key, symbol))
        except Exception:
            pass
        return df

    def fetch_balance_sheet(
        self, symbol: str, period: str = "quarter", limit: int = 20, force: bool = False
    ) -> pd.DataFrame | None:
        """Fetch balance sheet data."""
        cache_key = f"balance_{period}"
        if not force and self._is_cached(cache_key, symbol, max_age_days=7):
            try:
                return pd.read_parquet(self._cache_path(cache_key, symbol))
            except Exception:
                pass

        data = self._get("/stable/balance-sheet-statement", {
            "symbol": symbol, "period": period, "limit": str(limit)
        })
        if not data or not isinstance(data, list):
            return None

        df = pd.DataFrame(data)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

        try:
            df.to_parquet(self._cache_path(cache_key, symbol))
        except Exception:
            pass
        return df

    def fetch_cash_flow(
        self, symbol: str, period: str = "quarter", limit: int = 20, force: bool = False
    ) -> pd.DataFrame | None:
        """Fetch cash flow statement."""
        cache_key = f"cashflow_{period}"
        if not force and self._is_cached(cache_key, symbol, max_age_days=7):
            try:
                return pd.read_parquet(self._cache_path(cache_key, symbol))
            except Exception:
                pass

        data = self._get("/stable/cash-flow-statement", {
            "symbol": symbol, "period": period, "limit": str(limit)
        })
        if not data or not isinstance(data, list):
            return None

        df = pd.DataFrame(data)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

        try:
            df.to_parquet(self._cache_path(cache_key, symbol))
        except Exception:
            pass
        return df

    # ------------------------------------------------------------------
    # Build InMemoryDataContext
    # ------------------------------------------------------------------

    def build_context(
        self,
        symbols: list[str] | None = None,
        start_date: str = "2021-01-01",
        end_date: str | None = None,
        include_fundamentals: bool = False,
        force: bool = False,
    ) -> InMemoryDataContext:
        """
        Build an InMemoryDataContext from FMP data.

        This is the main entry point: fetches prices for all symbols,
        optionally fetches fundamentals, and constructs the context
        with aligned DataFrames.

        Args:
            symbols: List of tickers. Defaults to DEFAULT_US_SYMBOLS.
            start_date: Start date for historical data.
            end_date: End date (None = today).
            include_fundamentals: Whether to fetch income/balance/cashflow.
            force: Force re-fetch even if cached.
        """
        symbols = symbols or DEFAULT_US_SYMBOLS
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")

        print(f"\n[FMP] Data Loader -- {len(symbols)} symbols")
        print(f"   Period: {start_date} to {end_date}")
        print(f"   Cache: {self.cache_dir}")
        print(f"   Requests used so far: {self._request_count}")

        # --- Fetch prices ---
        price_dfs = {}
        failed = []
        for i, sym in enumerate(symbols):
            if (i + 1) % 10 == 0:
                print(f"   [{i + 1}/{len(symbols)}] Fetching prices...")

            df = self.fetch_historical_prices(sym, start_date, end_date, force=force)
            if df is not None and not df.empty:
                price_dfs[sym] = df
            else:
                failed.append(sym)

        print(f"   OK: {len(price_dfs)} symbols loaded, {len(failed)} failed")
        if failed:
            print(f"   Failed: {', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}")

        if not price_dfs:
            raise ValueError("No price data loaded!")

        # --- Build aligned matrices ---
        all_dates = sorted(set().union(*(df.index for df in price_dfs.values())))
        all_tickers = sorted(price_dfs.keys())

        matrices = {}
        for field in ["open", "high", "low", "close", "volume"]:
            mat = pd.DataFrame(index=all_dates, columns=all_tickers, dtype=float)
            for sym, df in price_dfs.items():
                if field in df.columns:
                    mat[sym] = df[field].reindex(all_dates)
            matrices[field] = mat

        # Derived fields
        close = matrices["close"]
        volume = matrices["volume"]
        matrices["returns"] = close.pct_change()
        matrices["dollars_traded"] = close * volume
        matrices["adv20"] = matrices["dollars_traded"].rolling(20).mean()

        # --- Fetch profiles (sector/industry) ---
        classifications = {}
        print(f"   Fetching profiles...")
        for sym in all_tickers:
            profile = self.fetch_profile(sym, force=force)
            if profile:
                classifications[sym] = {
                    "sector": profile.get("sector", "Unknown"),
                    "industry": profile.get("industry", "Unknown"),
                    "subindustry": profile.get("industry", "Unknown"),
                    "market_cap": profile.get("marketCap", 0),
                    "country": profile.get("country", "US"),
                }

        # --- Fundamentals (optional, uses many API calls) ---
        if include_fundamentals:
            print(f"   Fetching fundamentals (this uses many API calls)...")
            self._add_fundamentals(matrices, all_tickers, all_dates, force)

        # --- Build context ---
        ctx = InMemoryDataContext()
        ctx.load_from_matrices(matrices, classifications)

        print(f"\n   Context ready:")
        print(f"      {len(all_tickers)} tickers x {len(all_dates)} trading days")
        print(f"      Fields: {', '.join(matrices.keys())}")
        print(f"      Date range: {all_dates[0]} to {all_dates[-1]}")
        print(f"      API requests used: {self._request_count}")

        return ctx

    def _add_fundamentals(
        self, matrices: dict, tickers: list, dates: list, force: bool
    ) -> None:
        """Add fundamental data fields to the matrices dict."""
        # Key fundamental fields to extract
        income_fields = {
            "revenue": "revenue",
            "operating_income": "operatingIncome",
            "gross_profit": "grossProfit",
            "net_income": "netIncome",
            "rd_expense": "researchAndDevelopmentExpenses",
            "ebitda": "ebitda",
        }
        balance_fields = {
            "total_assets": "totalAssets",
            "total_equity": "totalStockholdersEquity",
            "total_debt": "totalDebt",
            "assets_curr": "totalCurrentAssets",
            "cash": "cashAndCashEquivalents",
        }
        cashflow_fields = {
            "cashflow_op": "operatingCashFlow",
            "capex": "capitalExpenditure",
            "free_cashflow": "freeCashFlow",
        }
        metric_fields = {
            "enterprise_value": "enterpriseValue",
            "market_cap_metric": "marketCap",
            "invested_capital": "investedCapital",
        }

        # Initialize empty matrices
        all_fund_fields = {}
        all_fund_fields.update(income_fields)
        all_fund_fields.update(balance_fields)
        all_fund_fields.update(cashflow_fields)
        all_fund_fields.update(metric_fields)

        for field_name in all_fund_fields:
            matrices[field_name] = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

        for idx, sym in enumerate(tickers):
            if (idx + 1) % 20 == 0:
                print(f"      [{idx + 1}/{len(tickers)}] fundamentals...")

            # Income
            inc_df = self.fetch_income_statement(sym, force=force)
            if inc_df is not None:
                for our_name, fmp_name in income_fields.items():
                    if fmp_name in inc_df.columns:
                        self._forward_fill_fundamental(
                            matrices[our_name], sym, inc_df[fmp_name], dates
                        )

            # Balance
            bal_df = self.fetch_balance_sheet(sym, force=force)
            if bal_df is not None:
                for our_name, fmp_name in balance_fields.items():
                    if fmp_name in bal_df.columns:
                        self._forward_fill_fundamental(
                            matrices[our_name], sym, bal_df[fmp_name], dates
                        )

            # Cash flow
            cf_df = self.fetch_cash_flow(sym, force=force)
            if cf_df is not None:
                for our_name, fmp_name in cashflow_fields.items():
                    if fmp_name in cf_df.columns:
                        self._forward_fill_fundamental(
                            matrices[our_name], sym, cf_df[fmp_name], dates
                        )

            # Metrics
            met_df = self.fetch_key_metrics(sym, period="quarter", force=force)
            if met_df is not None:
                for our_name, fmp_name in metric_fields.items():
                    if fmp_name in met_df.columns:
                        self._forward_fill_fundamental(
                            matrices[our_name], sym, met_df[fmp_name], dates
                        )

    @staticmethod
    def _forward_fill_fundamental(
        matrix: pd.DataFrame,
        symbol: str,
        quarterly_series: pd.Series,
        dates: list,
    ) -> None:
        """
        Forward-fill quarterly fundamental data to daily frequency.

        Fundamental data is reported quarterly; between reports we carry
        the last known value forward (point-in-time, no look-ahead).
        """
        if quarterly_series.empty:
            return

        # Reindex to daily dates and forward fill
        daily = quarterly_series.reindex(dates, method="ffill")
        matrix[symbol] = daily

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_request_count(self) -> int:
        """Get total API requests made this session."""
        return self._request_count

    def remaining_requests(self, daily_limit: int = 250) -> int:
        """Estimate remaining requests (approximate, resets daily)."""
        return max(0, daily_limit - self._request_count)


def load_fmp_context(
    api_key: str,
    symbols: list[str] | None = None,
    start_date: str = "2021-01-01",
    cache_dir: str = "data/fmp_cache",
    include_fundamentals: bool = False,
) -> InMemoryDataContext:
    """
    Convenience function: load FMP data into an InMemoryDataContext.

    Usage:
        ctx = load_fmp_context("YOUR_API_KEY", symbols=["AAPL", "MSFT"])
    """
    loader = FMPDataLoader(api_key=api_key, cache_dir=cache_dir)
    return loader.build_context(
        symbols=symbols,
        start_date=start_date,
        include_fundamentals=include_fundamentals,
    )
