"""
Unit tests for all BRAIN-compatible performance metrics.

Tests every metric computation against hand-computed expected values.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.simulation.metrics import (
    compute_sharpe,
    compute_max_drawdown,
    compute_pct_profitable,
    compute_daily_turnover,
    compute_fitness,
    compute_profit_per_dollar,
    compute_margin_bps,
    compute_total_dollars_traded,
)
from src.core.types import get_sharpe_rating


class TestSharpe:
    def test_sharpe_basic(self):
        daily_returns = np.array([0.01, -0.005, 0.008, 0.002, -0.003])
        expected_ir = np.mean(daily_returns) / np.std(daily_returns, ddof=1)
        expected_sharpe = expected_ir * math.sqrt(252)

        result = compute_sharpe(daily_returns)
        assert result == pytest.approx(expected_sharpe, rel=1e-10)

    def test_sharpe_positive(self):
        # Consistently positive returns → positive Sharpe
        returns = np.array([0.01] * 100)
        result = compute_sharpe(returns)
        assert result > 0

    def test_sharpe_negative(self):
        # Consistently negative returns → negative Sharpe
        returns = np.array([-0.01] * 100)
        result = compute_sharpe(returns)
        assert result < 0

    def test_sharpe_zero_std(self):
        # All identical returns → std = 0 → Sharpe = 0
        returns = np.array([0.01, 0.01, 0.01])
        result = compute_sharpe(returns)
        assert result == 0.0

    def test_sharpe_empty(self):
        result = compute_sharpe(np.array([]))
        assert result == 0.0

    def test_sharpe_single(self):
        result = compute_sharpe(np.array([0.01]))
        assert result == 0.0


class TestMaxDrawdown:
    def test_no_drawdown(self):
        # Monotonically increasing cumPnL
        cum_pnl = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_max_drawdown(cum_pnl, 10.0)
        assert result == pytest.approx(0.0)

    def test_full_drawdown(self):
        # Goes up then back to 0
        cum_pnl = np.array([0.0, 5.0, 10.0, 5.0, 0.0])
        result = compute_max_drawdown(cum_pnl, 10.0)
        assert result == pytest.approx(1.0)  # 10/10

    def test_specific_drawdown(self):
        cum_pnl = np.array([0.0, 2.0, 4.0, 3.0, 1.0, 3.0, 5.0])
        equity = 10.0
        result = compute_max_drawdown(cum_pnl, equity)
        # Max peak=4.0, trough after=1.0, drawdown=3.0
        # Max drawdown = 3.0 / 10.0 = 0.3
        assert result == pytest.approx(0.3)


class TestPctProfitable:
    def test_all_profitable(self):
        pnl = np.array([1.0, 2.0, 3.0])
        assert compute_pct_profitable(pnl) == pytest.approx(1.0)

    def test_none_profitable(self):
        pnl = np.array([-1.0, -2.0, -3.0])
        assert compute_pct_profitable(pnl) == pytest.approx(0.0)

    def test_half_profitable(self):
        pnl = np.array([1.0, -1.0, 2.0, -2.0])
        assert compute_pct_profitable(pnl) == pytest.approx(0.5)

    def test_zero_not_profitable(self):
        pnl = np.array([0.0, 0.0, 1.0])
        assert compute_pct_profitable(pnl) == pytest.approx(1 / 3)


class TestDailyTurnover:
    def test_no_change(self):
        positions = [{"A": 100.0}, {"A": 100.0}]
        result = compute_daily_turnover(positions, 200.0)
        assert len(result) == 1
        assert result[0] == pytest.approx(0.0)

    def test_full_turnover(self):
        positions = [{"A": 100.0}, {"B": 100.0}]
        result = compute_daily_turnover(positions, 200.0)
        # Close A (+100 change) + Open B (+100 change) = 200 / 200 = 1.0
        assert result[0] == pytest.approx(1.0)


class TestFitness:
    def test_fitness_basic(self):
        # fitness = sharpe * sqrt(|returnsPerc| / max(turnover, 0.125))
        sharpe = 2.0
        returns_pct = 10.0
        turnover = 0.5
        expected = 2.0 * math.sqrt(10.0 / 0.5)
        result = compute_fitness(sharpe, returns_pct, turnover)
        assert result == pytest.approx(expected)

    def test_fitness_low_turnover_floor(self):
        # When turnover < 0.125, use 0.125
        sharpe = 2.0
        returns_pct = 10.0
        turnover = 0.01  # very low
        expected = 2.0 * math.sqrt(10.0 / 0.125)
        result = compute_fitness(sharpe, returns_pct, turnover)
        assert result == pytest.approx(expected)


class TestProfitPerDollar:
    def test_basic(self):
        result = compute_profit_per_dollar(1000.0, 100000.0)
        assert result == pytest.approx(1.0)  # 1 cent per dollar

    def test_zero_traded(self):
        result = compute_profit_per_dollar(1000.0, 0.0)
        assert result == 0.0


class TestMarginBps:
    def test_basic(self):
        result = compute_margin_bps(100.0, 100000.0)
        assert result == pytest.approx(10.0)  # 10 bps

    def test_zero(self):
        result = compute_margin_bps(0.0, 100000.0)
        assert result == pytest.approx(0.0)


class TestSharpeRating:
    def test_delay1_ratings(self):
        assert get_sharpe_rating(5.0, delay=1) == "Spectacular"
        assert get_sharpe_rating(4.0, delay=1) == "Excellent"
        assert get_sharpe_rating(3.2, delay=1) == "Good"
        assert get_sharpe_rating(2.8, delay=1) == "Average"
        assert get_sharpe_rating(1.5, delay=1) == "Inferior"
        assert get_sharpe_rating(0.5, delay=1) == "Poor"

    def test_delay0_ratings(self):
        assert get_sharpe_rating(7.0, delay=0) == "Spectacular"
        assert get_sharpe_rating(5.5, delay=0) == "Excellent"
        assert get_sharpe_rating(5.0, delay=0) == "Good"
        assert get_sharpe_rating(4.0, delay=0) == "Average"
        assert get_sharpe_rating(2.0, delay=0) == "Inferior"
        assert get_sharpe_rating(0.5, delay=0) == "Poor"
