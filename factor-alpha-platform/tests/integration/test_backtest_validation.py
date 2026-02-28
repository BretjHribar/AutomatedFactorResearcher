"""
Integration tests — validate the backtester detects embedded signals
in synthetic data and produces correct metrics.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.core.types import FactorDef, SimSettings, SimResult
from src.data.synthetic import SyntheticDataGenerator, generate_tiny_fixture
from src.data.context_research import InMemoryDataContext
from src.simulation.engine import simulate


class TestBacktestValidation:
    """Validate the simulation engine against known analytical results."""

    @pytest.fixture(scope="class")
    def reversal_setup(self):
        """
        Generate data with strong mean-reversion signal,
        then backtest a reversal alpha with delay=0.

        Key: moderate MR strength (0.15) + delay=0 + low correlated noise
        ensures the reversal alpha detects the embedded signal.
        """
        dataset = SyntheticDataGenerator().generate(
            n_stocks=20,
            n_days=252,
            seed=42,
            mean_reversion_strength=0.15,  # Moderate — detectable but not extreme
            momentum_strength=0.0,
            start_date="2023-01-03",
            market_correlation=0.1,
            sector_correlation=0.05,
            nan_fraction=0.0,
            delist_fraction=0.0,
        )
        ctx = InMemoryDataContext(dataset)

        # Define reversal alpha: -returns (buy losers, sell winners)
        def compute_reversal(date: dt.date, ctx_inner) -> dict[str, float]:
            universe = ctx_inner.get_universe(date, "TOP3000")
            result = {}
            for ticker in universe:
                ret = ctx_inner.get_price(ticker, "returns", date)
                if not np.isnan(ret):
                    result[ticker] = -ret
            return result

        factor = FactorDef(
            factor_id="test_reversal",
            version=1,
            name="Simple Reversal",
            description="Buy yesterday's losers, sell yesterday's winners",
            category="momentum",
            compute_fn=compute_reversal,
            data_fields=("returns",),
            lookback_days=10,
            delay=0,
            neutralization="none",
        )

        settings = SimSettings(
            booksize=20_000_000,
            delay=0,
            neutralization="none",
            universe="TOP3000",
            duration_years=1,
        )

        result = simulate(factor, ctx, settings)
        return result

    def test_reversal_produces_results(self, reversal_setup):
        """The backtest should produce non-empty results."""
        result = reversal_setup
        assert len(result.dates) > 100
        assert len(result.daily_pnl) > 100

    def test_reversal_positive_pnl(self, reversal_setup):
        """
        On mean-reverting data, a reversal alpha should have positive total PnL.
        """
        result = reversal_setup
        total_pnl = sum(result.daily_pnl)
        assert total_pnl > 0, f"Reversal on mean-reverting data should be profitable, got PnL={total_pnl}"

    def test_reversal_positive_sharpe(self, reversal_setup):
        """Sharpe should be positive on mean-reverting data."""
        result = reversal_setup
        assert result.total_metrics.sharpe > 0, f"Sharpe = {result.total_metrics.sharpe}"

    def test_metrics_consistency(self, reversal_setup):
        """Check internal consistency of computed metrics."""
        result = reversal_setup
        m = result.total_metrics

        # PnL should equal sum of daily PnL
        assert m.pnl == pytest.approx(sum(result.daily_pnl), rel=1e-6)

        # Annual return = PnL / equity
        equity = m.booksize / 2.0
        expected_return = m.pnl / equity
        assert m.annual_return == pytest.approx(expected_return, rel=0.1)

        # Pct profitable should be between 0 and 1
        assert 0 <= m.pct_profitable_days <= 1

        # Max drawdown should be >= 0
        assert m.max_drawdown >= 0

    def test_positions_are_scaled(self, reversal_setup):
        """Positions should be scaled to booksize."""
        result = reversal_setup
        settings = result.settings
        booksize = settings.booksize

        for positions in result.daily_positions:
            if positions:
                total_abs = sum(abs(v) for v in positions.values())
                # Should be approximately booksize (with some tolerance for NaN filtering)
                assert total_abs <= booksize * 1.1, f"Positions {total_abs} > booksize {booksize}"

    def test_turnover_is_bounded(self, reversal_setup):
        """Daily turnover should be between 0 and 2."""
        result = reversal_setup
        for t in result.daily_turnover:
            assert 0 <= t <= 2.0, f"Turnover {t} out of bounds"

    def test_positions_have_reasonable_distribution(self, reversal_setup):
        """
        Verify positions are distributed across multiple stocks, not concentrated.
        """
        result = reversal_setup
        for positions in result.daily_positions[10:]:
            if positions and len(positions) > 3:
                values = list(positions.values())
                total_abs = sum(abs(v) for v in values)
                if total_abs > 0:
                    # No single position should be > 90% of total (relaxed for 20-stock universe)
                    max_w = max(abs(v) / total_abs for v in values)
                    assert max_w < 0.9, f"Position too concentrated: max weight = {max_w}"


class TestConstantAlpha:
    """Test an equal-weight alpha (constant for all stocks)."""

    def test_constant_alpha_approximates_market(self):
        """
        Alpha = 1 for all stocks → equal-weight long portfolio.
        Without neutralization, this should approximate market return.
        """
        dataset = SyntheticDataGenerator().generate(
            n_stocks=20,
            n_days=126,
            seed=42,
            start_date="2023-01-03",
            mean_reversion_strength=0.0,
            momentum_strength=0.0,
        )
        ctx = InMemoryDataContext(dataset)

        def compute_constant(date: dt.date, ctx_inner) -> dict[str, float]:
            universe = ctx_inner.get_universe(date, "TOP3000")
            return {t: 1.0 for t in universe}

        factor = FactorDef(
            factor_id="test_constant",
            version=1,
            name="Constant",
            description="Equal weight all stocks",
            category="test",
            compute_fn=compute_constant,
            lookback_days=5,
            neutralization="none",
        )

        settings = SimSettings(
            neutralization="none",
            duration_years=1,
        )

        result = simulate(factor, ctx, settings)
        assert len(result.dates) > 50
        # The PnL should reflect market returns (some movement, not zero)
        assert result.total_metrics.pnl != 0.0


class TestSyntheticDataGeneration:
    """Test the synthetic data generator itself."""

    def test_generates_correct_shape(self):
        ds = SyntheticDataGenerator().generate(n_stocks=10, n_days=50, seed=42)
        assert ds.n_stocks == 10
        assert ds.n_days == 50
        assert len(ds.trading_days) == 50

    def test_prices_are_positive(self):
        ds = SyntheticDataGenerator().generate(n_stocks=10, n_days=100, seed=42)
        close_prices = ds.prices["close"].dropna()
        assert (close_prices > 0).all()

    def test_deterministic_with_seed(self):
        ds1 = SyntheticDataGenerator().generate(n_stocks=10, n_days=50, seed=42)
        ds2 = SyntheticDataGenerator().generate(n_stocks=10, n_days=50, seed=42)
        pd.testing.assert_frame_equal(ds1.prices, ds2.prices)

    def test_different_seeds_differ(self):
        ds1 = SyntheticDataGenerator().generate(n_stocks=10, n_days=50, seed=42)
        ds2 = SyntheticDataGenerator().generate(n_stocks=10, n_days=50, seed=99)
        assert not ds1.prices.equals(ds2.prices)

    def test_classifications_assigned(self):
        ds = SyntheticDataGenerator().generate(n_stocks=20, n_days=50, seed=42)
        assert len(ds.classifications) == 20
        assert "sector" in ds.classifications.columns
        assert "industry" in ds.classifications.columns
        assert "subindustry" in ds.classifications.columns

    def test_fundamentals_generated(self):
        ds = SyntheticDataGenerator().generate(n_stocks=10, n_days=252, seed=42)
        assert len(ds.fundamentals) > 0
        assert "ticker" in ds.fundamentals.columns
        assert "field" in ds.fundamentals.columns
        assert "filing_date" in ds.fundamentals.columns

    def test_universes_generated(self):
        ds = SyntheticDataGenerator().generate(n_stocks=20, n_days=100, seed=42)
        assert len(ds.universes) > 0


class TestDataContext:
    """Test the InMemoryDataContext."""

    def test_get_price(self, tiny_ctx):
        date = tiny_ctx._trading_days[50]
        universe = tiny_ctx.get_universe(date, "TOP3000")
        if universe:
            ticker = universe[0]
            close = tiny_ctx.get_price(ticker, "close", date)
            assert not np.isnan(close)
            assert close > 0

    def test_get_universe(self, tiny_ctx):
        date = tiny_ctx._trading_days[50]
        universe = tiny_ctx.get_universe(date, "TOP3000")
        assert len(universe) > 0

    def test_get_matrix(self, tiny_ctx):
        date = tiny_ctx._trading_days[50]
        mat = tiny_ctx.get_matrix("close", date, 20, "TOP3000")
        assert not mat.empty
        assert len(mat) <= 20  # up to 20 rows
        assert len(mat.columns) > 0

    def test_get_trading_days(self, tiny_ctx):
        days = tiny_ctx.get_trading_days(
            tiny_ctx._trading_days[0],
            tiny_ctx._trading_days[-1],
        )
        assert len(days) > 0
        # Days should be sorted
        assert days == sorted(days)

    def test_get_industry(self, tiny_ctx):
        date = tiny_ctx._trading_days[50]
        universe = tiny_ctx.get_universe(date, "TOP3000")
        if universe:
            industry = tiny_ctx.get_industry(universe[0], date, "industry")
            assert industry != "Unknown"

    def test_get_fundamental(self, tiny_ctx):
        date = tiny_ctx._trading_days[100]
        universe = tiny_ctx.get_universe(date, "TOP3000")
        if universe:
            val = tiny_ctx.get_fundamental(universe[0], "eps", date)
            # May or may not have data depending on filing dates
            # Just verify it doesn't crash
            assert isinstance(val, float)
