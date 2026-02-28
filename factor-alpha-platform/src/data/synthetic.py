"""
Synthetic Data Generator — the cornerstone of zero-dependency development.

Generates realistic market data with configurable statistical properties.
Embeds known signals (mean-reversion, momentum) so the backtester can be
validated against analytically computable expected Sharpes.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# SyntheticDataset — the output of the generator
# ---------------------------------------------------------------------------

@dataclass
class SyntheticDataset:
    """Complete synthetic dataset for development and testing."""

    prices: pd.DataFrame        # date, ticker, open, high, low, close, volume, vwap, returns, adv20, sharesout
    fundamentals: pd.DataFrame  # ticker, report_date, filing_date, period, field, value
    estimates: pd.DataFrame     # ticker, date, period, field, consensus, high, low, n_analysts
    classifications: pd.DataFrame  # ticker, sector, industry, subindustry
    universes: pd.DataFrame     # date, universe, tickers (as list or exploded)
    trading_days: list[dt.date]

    # Metadata
    n_stocks: int = 0
    n_days: int = 0
    start_date: dt.date = dt.date(2019, 1, 2)
    seed: int = 42


# ---------------------------------------------------------------------------
# GICS-style sector/industry/subindustry definitions
# ---------------------------------------------------------------------------

SECTORS = [
    "Energy", "Materials", "Industrials", "Consumer Discretionary",
    "Consumer Staples", "Health Care", "Financials", "Information Technology",
    "Communication Services", "Utilities", "Real Estate",
]

INDUSTRIES = {
    "Energy": ["Oil & Gas Exploration", "Oil & Gas Equipment", "Oil & Gas Refining"],
    "Materials": ["Chemicals", "Metals & Mining", "Construction Materials", "Packaging"],
    "Industrials": ["Aerospace & Defense", "Machinery", "Construction", "Transportation"],
    "Consumer Discretionary": ["Automobiles", "Retail", "Hotels & Restaurants", "Media"],
    "Consumer Staples": ["Food Products", "Beverages", "Household Products", "Tobacco"],
    "Health Care": ["Pharmaceuticals", "Biotechnology", "Medical Devices", "Health Services"],
    "Financials": ["Banks", "Insurance", "Capital Markets", "Consumer Finance"],
    "Information Technology": ["Software", "Hardware", "Semiconductors", "IT Services"],
    "Communication Services": ["Telecom", "Entertainment", "Interactive Media"],
    "Utilities": ["Electric Utilities", "Gas Utilities", "Water Utilities"],
    "Real Estate": ["REITs", "Real Estate Services", "Real Estate Development"],
}


def _generate_subindustries(industry: str, n: int = 3) -> list[str]:
    """Generate sub-industry names from an industry."""
    suffixes = ["Large Cap", "Mid Cap", "Small Cap", "Diversified", "Specialty"]
    return [f"{industry} - {suffixes[i % len(suffixes)]}" for i in range(n)]


# ---------------------------------------------------------------------------
# The Generator
# ---------------------------------------------------------------------------

class SyntheticDataGenerator:
    """
    Generate realistic synthetic market data for development and testing.

    All tests run against this — no external dependency ever needed.

    Key design:
    - Prices follow geometric Brownian motion with sector-correlated noise
    - Known mean-reversion and momentum signals are embedded at configurable strength
    - Fundamentals have realistic quarterly cadence with 30-60 day filing lag
    - Some stocks 'delist' mid-series (for survivorship bias testing)
    - Some data points are intentionally NaN (for missing data handling)
    """

    def generate(
        self,
        n_stocks: int = 500,
        n_days: int = 252 * 5,
        start_date: str = "2019-01-02",
        seed: int = 42,
        # Statistical properties
        annual_return_mean: float = 0.08,
        annual_vol_mean: float = 0.25,
        mean_reversion_strength: float = 0.02,
        momentum_strength: float = 0.01,
        # Structure
        n_sectors: int = 11,
        sector_correlation: float = 0.3,
        market_correlation: float = 0.4,
        # Data quality
        nan_fraction: float = 0.005,
        delist_fraction: float = 0.02,
    ) -> SyntheticDataset:
        """Generate a complete synthetic dataset."""
        rng = np.random.default_rng(seed)
        sd = dt.date.fromisoformat(start_date)
        n_sectors = min(n_sectors, len(SECTORS))

        # --- Generate trading days (skip weekends) ---
        trading_days = self._generate_trading_days(sd, n_days)
        actual_n_days = len(trading_days)

        # --- Assign tickers, sectors, industries, subindustries ---
        tickers = [f"SYN{i:04d}" for i in range(n_stocks)]
        classifications = self._assign_classifications(tickers, n_sectors, rng)

        # --- Generate correlated returns ---
        returns_matrix = self._generate_returns(
            n_stocks=n_stocks,
            n_days=actual_n_days,
            rng=rng,
            annual_return_mean=annual_return_mean,
            annual_vol_mean=annual_vol_mean,
            market_correlation=market_correlation,
            sector_correlation=sector_correlation,
            classifications=classifications,
            mean_reversion_strength=mean_reversion_strength,
            momentum_strength=momentum_strength,
        )

        # --- Generate OHLCV from returns ---
        prices_df = self._generate_ohlcv(
            returns_matrix, tickers, trading_days, rng, annual_vol_mean
        )

        # --- Handle delistings ---
        prices_df = self._apply_delistings(
            prices_df, tickers, trading_days, delist_fraction, rng
        )

        # --- Inject NaN ---
        prices_df = self._inject_nans(prices_df, nan_fraction, rng)

        # --- Compute derived fields (adv20, returns column) ---
        prices_df = self._compute_derived_fields(prices_df)

        # --- Generate fundamentals ---
        fundamentals_df = self._generate_fundamentals(tickers, trading_days, rng)

        # --- Generate estimates ---
        estimates_df = self._generate_estimates(tickers, trading_days, rng)

        # --- Generate universe membership ---
        universes_df = self._generate_universes(prices_df, trading_days)

        # --- Classifications DataFrame ---
        class_df = pd.DataFrame(classifications)

        return SyntheticDataset(
            prices=prices_df,
            fundamentals=fundamentals_df,
            estimates=estimates_df,
            classifications=class_df,
            universes=universes_df,
            trading_days=trading_days,
            n_stocks=n_stocks,
            n_days=actual_n_days,
            start_date=sd,
            seed=seed,
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _generate_trading_days(self, start: dt.date, n_days: int) -> list[dt.date]:
        """Generate n trading days (skip weekends), starting from start."""
        days: list[dt.date] = []
        current = start
        while len(days) < n_days:
            if current.weekday() < 5:  # Mon-Fri
                days.append(current)
            current += dt.timedelta(days=1)
        return days

    def _assign_classifications(
        self, tickers: list[str], n_sectors: int, rng: np.random.Generator
    ) -> list[dict[str, str]]:
        """Assign each ticker to a sector, industry, subindustry."""
        sectors_used = SECTORS[:n_sectors]
        result = []
        for i, ticker in enumerate(tickers):
            sector = sectors_used[i % n_sectors]
            industries_in_sector = INDUSTRIES[sector]
            industry = industries_in_sector[i % len(industries_in_sector)]
            subindustries = _generate_subindustries(industry)
            subindustry = subindustries[i % len(subindustries)]
            result.append({
                "ticker": ticker,
                "sector": sector,
                "industry": industry,
                "subindustry": subindustry,
            })
        return result

    def _generate_returns(
        self,
        n_stocks: int,
        n_days: int,
        rng: np.random.Generator,
        annual_return_mean: float,
        annual_vol_mean: float,
        market_correlation: float,
        sector_correlation: float,
        classifications: list[dict[str, str]],
        mean_reversion_strength: float,
        momentum_strength: float,
    ) -> np.ndarray:
        """
        Generate correlated daily returns with embedded alpha signals.

        Returns: (n_days, n_stocks) array of daily returns.
        """
        daily_mu = annual_return_mean / 252
        daily_vol = annual_vol_mean / np.sqrt(252)

        # Per-stock volatility (varied around mean)
        stock_vols = daily_vol * (0.5 + rng.random(n_stocks))

        # Market factor
        market_factor = rng.normal(daily_mu, daily_vol, n_days)

        # Sector factors
        sectors = [c["sector"] for c in classifications]
        unique_sectors = sorted(set(sectors))
        sector_map = {s: i for i, s in enumerate(unique_sectors)}
        sector_ids = np.array([sector_map[s] for s in sectors])
        n_sector_factors = len(unique_sectors)
        sector_factors = rng.normal(0, daily_vol * 0.5, (n_days, n_sector_factors))

        # Idiosyncratic noise
        idio_noise = rng.normal(0, 1, (n_days, n_stocks)) * stock_vols[np.newaxis, :]

        # Combine: returns = market + sector + idiosyncratic
        returns = np.zeros((n_days, n_stocks))
        for t in range(n_days):
            for i in range(n_stocks):
                returns[t, i] = (
                    market_correlation * market_factor[t]
                    + sector_correlation * sector_factors[t, sector_ids[i]]
                    + (1 - market_correlation - sector_correlation) * idio_noise[t, i]
                )

        # --- Embed mean-reversion signal ---
        # After large moves, inject a pull-back the next day
        if mean_reversion_strength > 0:
            for t in range(1, n_days):
                returns[t, :] -= mean_reversion_strength * returns[t - 1, :]

        # --- Embed momentum signal ---
        # Trailing 20-day return predicts next-day return
        if momentum_strength > 0:
            for t in range(21, n_days):
                trailing = returns[t - 20:t, :].sum(axis=0)
                returns[t, :] += momentum_strength * np.sign(trailing) * daily_vol * 0.1

        return returns

    def _generate_ohlcv(
        self,
        returns_matrix: np.ndarray,
        tickers: list[str],
        trading_days: list[dt.date],
        rng: np.random.Generator,
        annual_vol: float,
    ) -> pd.DataFrame:
        """Convert returns matrix to OHLCV DataFrame."""
        n_days, n_stocks = returns_matrix.shape

        # Starting prices (varied)
        starting_prices = 20 + rng.random(n_stocks) * 180  # $20 to $200

        # Shares outstanding (varied)
        shares_out = (rng.random(n_stocks) * 900 + 100) * 1_000_000  # 100M to 1B

        records = []
        close_prev = starting_prices.copy()

        for t in range(n_days):
            close_t = close_prev * (1 + returns_matrix[t])

            # Intraday range: close ± some noise
            daily_range = np.abs(close_t * rng.normal(0, 0.005, n_stocks))
            high_t = close_t + daily_range * rng.random(n_stocks)
            low_t = close_t - daily_range * rng.random(n_stocks)
            low_t = np.maximum(low_t, 0.01)  # prices can't go negative
            open_t = close_prev + (close_t - close_prev) * rng.random(n_stocks)
            open_t = np.clip(open_t, low_t, high_t)

            # VWAP: approximate as weighted avg of open, high, low, close
            vwap_t = (open_t + high_t + low_t + close_t) / 4.0

            # Volume: base volume with noise
            base_volume = shares_out * 0.01  # ~1% of shares trade daily
            volume_t = base_volume * (0.5 + rng.random(n_stocks))

            for i, ticker in enumerate(tickers):
                records.append({
                    "date": trading_days[t],
                    "ticker": ticker,
                    "open": round(float(open_t[i]), 2),
                    "high": round(float(high_t[i]), 2),
                    "low": round(float(low_t[i]), 2),
                    "close": round(float(close_t[i]), 2),
                    "volume": int(volume_t[i]),
                    "vwap": round(float(vwap_t[i]), 2),
                    "sharesout": int(shares_out[i]),
                })

            close_prev = close_t.copy()

        return pd.DataFrame(records)

    def _apply_delistings(
        self,
        df: pd.DataFrame,
        tickers: list[str],
        trading_days: list[dt.date],
        delist_fraction: float,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Remove data for 'delisted' stocks after their delist date."""
        n_delist = max(1, int(len(tickers) * delist_fraction))
        delist_tickers = rng.choice(tickers, size=n_delist, replace=False)
        n_days = len(trading_days)

        for ticker in delist_tickers:
            # Delist somewhere in the second half
            delist_idx = rng.integers(n_days // 2, n_days)
            delist_date = trading_days[delist_idx]

            # Apply delisting return (large negative) on delist day
            mask_delist = (df["ticker"] == ticker) & (df["date"] == delist_date)
            if mask_delist.any():
                df.loc[mask_delist, "close"] = df.loc[mask_delist, "close"] * 0.3
                df.loc[mask_delist, "low"] = df.loc[mask_delist, "close"]

            # Remove all data after delist date
            mask_remove = (df["ticker"] == ticker) & (df["date"] > delist_date)
            df = df[~mask_remove]

        return df.reset_index(drop=True)

    def _inject_nans(
        self, df: pd.DataFrame, nan_fraction: float, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Randomly set some price values to NaN."""
        numeric_cols = ["open", "high", "low", "close", "volume", "vwap"]
        n_total = len(df) * len(numeric_cols)
        n_nans = int(n_total * nan_fraction)

        for _ in range(n_nans):
            row_idx = rng.integers(0, len(df))
            col = rng.choice(numeric_cols)
            df.iloc[row_idx, df.columns.get_loc(col)] = np.nan

        return df

    def _compute_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute returns, adv20 (trailing 20-day avg dollar volume)."""
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        # Daily returns: close-to-close
        df["returns"] = df.groupby("ticker")["close"].pct_change()

        # Dollar volume
        df["dollar_volume"] = df["close"] * df["volume"]

        # ADV20: trailing 20-day average dollar volume
        df["adv20"] = (
            df.groupby("ticker")["dollar_volume"]
            .transform(lambda x: x.rolling(20, min_periods=1).mean())
        )

        df = df.drop(columns=["dollar_volume"])
        return df

    def _generate_fundamentals(
        self, tickers: list[str], trading_days: list[dt.date], rng: np.random.Generator
    ) -> pd.DataFrame:
        """Generate quarterly fundamental data with realistic filing lags."""
        fields = [
            "sales", "income", "eps", "ebitda", "assets", "equity", "debt",
            "cashflow", "bookvalue_ps", "return_assets", "return_equity",
        ]

        records = []
        for ticker in tickers:
            # Base values per stock (vary by company "size")
            base_sales = rng.uniform(500, 50000)  # $M
            growth_rate = rng.uniform(-0.05, 0.15)

            # Generate quarterly reports
            start_year = trading_days[0].year
            end_year = trading_days[-1].year

            for year in range(start_year, end_year + 1):
                for q in range(1, 5):
                    report_month = q * 3
                    report_date = dt.date(year, report_month, 28)

                    # Filing date: 30-60 days after quarter end
                    filing_lag = rng.integers(30, 61)
                    filing_date = report_date + dt.timedelta(days=int(filing_lag))

                    # Only include if within our date range
                    if filing_date > trading_days[-1]:
                        continue
                    if report_date < trading_days[0]:
                        continue

                    quarter_label = f"{year}Q{q}"
                    quarterly_sales = base_sales * (1 + growth_rate) ** ((year - start_year) + q / 4)

                    values = {
                        "sales": quarterly_sales,
                        "income": quarterly_sales * rng.uniform(0.02, 0.25),
                        "eps": quarterly_sales * rng.uniform(0.001, 0.01),
                        "ebitda": quarterly_sales * rng.uniform(0.1, 0.35),
                        "assets": quarterly_sales * rng.uniform(2, 8),
                        "equity": quarterly_sales * rng.uniform(0.5, 3),
                        "debt": quarterly_sales * rng.uniform(0.2, 2),
                        "cashflow": quarterly_sales * rng.uniform(-0.1, 0.2),
                        "bookvalue_ps": rng.uniform(5, 100),
                        "return_assets": rng.uniform(0.01, 0.15),
                        "return_equity": rng.uniform(0.02, 0.30),
                    }

                    for f, v in values.items():
                        records.append({
                            "ticker": ticker,
                            "report_date": report_date,
                            "filing_date": filing_date,
                            "period": quarter_label,
                            "field": f,
                            "value": round(float(v), 4),
                        })

        return pd.DataFrame(records)

    def _generate_estimates(
        self, tickers: list[str], trading_days: list[dt.date], rng: np.random.Generator
    ) -> pd.DataFrame:
        """Generate analyst consensus estimates with revision patterns."""
        records = []
        # Monthly estimate snapshots
        for ticker in tickers:
            base_eps = rng.uniform(1, 10)
            for day in trading_days[::21]:  # ~monthly
                for period in ["FY1", "FY2"]:
                    noise = rng.normal(0, 0.1) * base_eps
                    consensus = base_eps + noise
                    records.append({
                        "ticker": ticker,
                        "date": day,
                        "period": period,
                        "field": "eps",
                        "consensus": round(float(consensus), 4),
                        "high": round(float(consensus * 1.15), 4),
                        "low": round(float(consensus * 0.85), 4),
                        "n_analysts": rng.integers(3, 30),
                    })
                    # Revenue estimate
                    base_rev = base_eps * rng.uniform(50, 200)
                    rev_noise = rng.normal(0, 0.05) * base_rev
                    records.append({
                        "ticker": ticker,
                        "date": day,
                        "period": period,
                        "field": "revenue",
                        "consensus": round(float(base_rev + rev_noise), 2),
                        "high": round(float((base_rev + rev_noise) * 1.1), 2),
                        "low": round(float((base_rev + rev_noise) * 0.9), 2),
                        "n_analysts": rng.integers(3, 30),
                    })

        return pd.DataFrame(records)

    def _generate_universes(
        self, prices_df: pd.DataFrame, trading_days: list[dt.date]
    ) -> pd.DataFrame:
        """Generate universe membership based on dollar volume ranking."""
        universe_sizes = {"TOP200": 200, "TOP500": 500, "TOP1000": 1000, "TOP2000": 2000, "TOP3000": 3000}

        # Rebalance monthly (first trading day of each month)
        rebalance_dates = []
        current_month = None
        for d in trading_days:
            ym = (d.year, d.month)
            if ym != current_month:
                rebalance_dates.append(d)
                current_month = ym

        records = []
        for reb_date in rebalance_dates:
            # Get average dollar volume over trailing 63 days (or available)
            lookback_start = reb_date - dt.timedelta(days=100)
            mask = (prices_df["date"] >= lookback_start) & (prices_df["date"] <= reb_date)
            subset = prices_df[mask]

            if subset.empty:
                continue

            avg_dv = subset.groupby("ticker")["adv20"].last().dropna()
            ranked = avg_dv.sort_values(ascending=False)

            for univ_name, size in universe_sizes.items():
                top_n = ranked.head(min(size, len(ranked))).index.tolist()
                for ticker in top_n:
                    records.append({
                        "date": reb_date,
                        "universe": univ_name,
                        "ticker": ticker,
                    })

        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Convenience: pre-configured fixture generators
# ---------------------------------------------------------------------------

def generate_tiny_fixture(seed: int = 42) -> SyntheticDataset:
    """20 stocks × 252 days. For unit tests (< 1 sec)."""
    return SyntheticDataGenerator().generate(
        n_stocks=20, n_days=252, seed=seed,
        start_date="2023-01-03",
        mean_reversion_strength=0.03,
        momentum_strength=0.015,
    )


def generate_small_fixture(seed: int = 42) -> SyntheticDataset:
    """100 stocks × 504 days (2 years). For integration tests (< 10 sec)."""
    return SyntheticDataGenerator().generate(
        n_stocks=100, n_days=504, seed=seed,
        start_date="2022-01-03",
    )


def generate_medium_fixture(seed: int = 42) -> SyntheticDataset:
    """500 stocks × 1260 days (5 years). For E2E/GUI demo (< 30 sec)."""
    return SyntheticDataGenerator().generate(
        n_stocks=500, n_days=1260, seed=seed,
        start_date="2019-01-02",
    )
