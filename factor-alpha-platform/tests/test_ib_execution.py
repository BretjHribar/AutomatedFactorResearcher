"""
test_ib_execution.py — Tests for IB execution modules.

Tests the portfolio construction logic, trade computation,
and position sizing — the core business logic that doesn't
require an actual IB connection.

Run:
    python -m pytest tests/test_ib_execution.py -v
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestPortfolioConstruction:
    """Test portfolio construction from alpha weights to share positions."""

    def test_basic_long_short_split(self):
        """Alpha weights should produce both long and short positions."""
        from src.execution.ib_trader import compute_target_portfolio

        weights = pd.Series({
            "AAPL": 0.10, "GOOG": 0.08, "MSFT": 0.07,  # longs
            "XOM": -0.09, "CVX": -0.08, "BP": -0.06,    # shorts
        })
        prices = pd.Series({
            "AAPL": 150.0, "GOOG": 140.0, "MSFT": 400.0,
            "XOM": 100.0, "CVX": 150.0, "BP": 35.0,
        })

        positions = compute_target_portfolio(weights, prices, per_side=220_000)
        longs = [p for p in positions if p.is_long]
        shorts = [p for p in positions if not p.is_long]

        assert len(longs) > 0, "Should have long positions"
        assert len(shorts) > 0, "Should have short positions"

    def test_shares_are_whole_numbers(self):
        """All positions should be rounded to whole shares (not fractional)."""
        from src.execution.ib_trader import compute_target_portfolio

        weights = pd.Series({"A": 0.5, "B": 0.5, "C": -0.5, "D": -0.5})
        prices = pd.Series({"A": 33.33, "B": 77.77, "C": 15.50, "D": 123.45})

        positions = compute_target_portfolio(weights, prices, per_side=110_000)
        for pos in positions:
            assert pos.shares == int(pos.shares), \
                f"{pos.symbol}: shares={pos.shares} is not a whole number"

    def test_min_position_filter(self):
        """Positions below minimum dollar threshold should be skipped."""
        from src.execution.ib_trader import compute_target_portfolio

        # With per_side=1000 and min_position=500:
        # After normalization, BIG gets ~990, TINY gets ~10 -> below $500
        weights = pd.Series({"BIG": 0.99, "TINY": 0.001, "SHORT": -1.0})
        prices = pd.Series({"BIG": 50.0, "TINY": 50.0, "SHORT": 50.0})

        positions = compute_target_portfolio(
            weights, prices, per_side=1_000, min_position=500
        )
        symbols = {p.symbol for p in positions}
        # TINY gets ~0.001/0.991 * $1000 = ~$1.01, below $500 min
        assert "TINY" not in symbols, "TINY should be filtered (below $500 min)"
        assert "BIG" in symbols

    def test_max_weight_clipping(self):
        """Weights should be clipped to max_weight."""
        from src.execution.ib_trader import compute_target_portfolio

        weights = pd.Series({"A": 0.8, "B": 0.2, "C": -0.8, "D": -0.2})
        prices = pd.Series({"A": 25.0, "B": 25.0, "C": 25.0, "D": 25.0})

        positions = compute_target_portfolio(
            weights, prices, per_side=220_000, max_weight=0.05
        )
        # All positions should exist since clipping doesn't eliminate them,
        # it just reduces their size
        assert len(positions) == 4

    def test_zero_price_skipped(self):
        """Symbols with zero or missing price should be skipped."""
        from src.execution.ib_trader import compute_target_portfolio

        weights = pd.Series({"GOOD": 0.5, "ZERO": 0.5, "MISS": -1.0})
        prices = pd.Series({"GOOD": 25.0, "ZERO": 0.0})

        positions = compute_target_portfolio(weights, prices, per_side=110_000)
        symbols = {p.symbol for p in positions}
        assert "GOOD" in symbols
        assert "ZERO" not in symbols
        assert "MISS" not in symbols


class TestTradeComputation:
    """Test trade computation from current to target positions."""

    def test_new_positions(self):
        """Starting from empty, all targets should become trades."""
        from src.execution.ib_trader import compute_trades, TargetPosition

        targets = [
            TargetPosition("AAPL", 100, "BUY", 15000, 0.1),
            TargetPosition("XOM", -50, "SELL", 5000, 0.05),
        ]
        current = {}

        trades = compute_trades(targets, current)
        assert len(trades) == 2

        trade_map = {t.symbol: t.shares for t in trades}
        assert trade_map["AAPL"] == 100
        assert trade_map["XOM"] == -50

    def test_no_change_needed(self):
        """If current matches target, no trades needed."""
        from src.execution.ib_trader import compute_trades, TargetPosition

        targets = [
            TargetPosition("AAPL", 100, "BUY", 15000, 0.1),
        ]
        current = {"AAPL": 100}

        trades = compute_trades(targets, current)
        assert len(trades) == 0

    def test_partial_rebalance(self):
        """Trades should be the delta between target and current."""
        from src.execution.ib_trader import compute_trades, TargetPosition

        targets = [
            TargetPosition("AAPL", 150, "BUY", 22500, 0.15),
        ]
        current = {"AAPL": 100}

        trades = compute_trades(targets, current)
        assert len(trades) == 1
        assert trades[0].shares == 50  # Need to buy 50 more
        assert trades[0].side == "BUY"

    def test_close_position(self):
        """Removing a symbol from target should generate a closing trade."""
        from src.execution.ib_trader import compute_trades, TargetPosition

        targets = []  # No target positions
        current = {"AAPL": 100, "XOM": -50}

        trades = compute_trades(targets, current)
        assert len(trades) == 2

        trade_map = {t.symbol: t.shares for t in trades}
        assert trade_map["AAPL"] == -100  # Sell to close long
        assert trade_map["XOM"] == 50     # Buy to close short

    def test_flip_position(self):
        """Going from long to short should generate correct delta."""
        from src.execution.ib_trader import compute_trades, TargetPosition

        targets = [
            TargetPosition("AAPL", -80, "SELL", 12000, 0.08),
        ]
        current = {"AAPL": 100}

        trades = compute_trades(targets, current)
        assert len(trades) == 1
        assert trades[0].shares == -180  # Sell 100 to close + sell 80 to short


class TestSchedulerConfig:
    """Test scheduler configuration."""

    def test_is_market_day(self):
        """Market days should be Mon-Fri only."""
        from src.execution.scheduler import is_market_day
        from datetime import datetime

        monday = datetime(2024, 1, 8)     # Monday
        saturday = datetime(2024, 1, 6)   # Saturday
        sunday = datetime(2024, 1, 7)     # Sunday

        assert is_market_day(monday) is True
        assert is_market_day(saturday) is False
        assert is_market_day(sunday) is False

    def test_schedule_time(self):
        """Schedule should be at 3:30 PM ET."""
        from src.execution.scheduler import SCHEDULE_TIME
        from datetime import time

        assert SCHEDULE_TIME == time(15, 30)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
