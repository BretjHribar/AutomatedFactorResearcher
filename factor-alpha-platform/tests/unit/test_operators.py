"""
Unit tests for all WebSim-compatible operators.

Tests cross-sectional, time-series, and element-wise operators
against known inputs and expected outputs.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.operators.cross_sectional import (
    rank,
    scale,
    ind_neutralize,
    zscore,
    winsorize,
)
from src.operators.time_series import (
    delay,
    delta,
    ts_sum,
    ts_mean,
    ts_std,
    ts_min,
    ts_max,
    ts_rank,
    ts_skewness,
    ts_kurtosis,
    decay_linear,
    decay_exp,
    product,
    count_nans,
    correlation,
    covariance,
)
from src.operators.element_wise import (
    op_abs,
    sign,
    log,
    signed_power,
    pasteurize,
    op_min,
    op_max,
    tail,
)


# ==========================================================================
# Cross-Sectional Operator Tests
# ==========================================================================


class TestRank:
    def test_basic_rank(self, simple_series):
        result = rank(simple_series)
        # Values: [20.2, 15.6, 10.0, 5.7, 50.2, 18.4]
        # Sorted: [5.7, 10.0, 15.6, 18.4, 20.2, 50.2]
        # Ranks:  [1,    2,    3,    4,    5,    6]  → pct: [1/6, 2/6, 3/6, 4/6, 5/6, 6/6]
        assert result["E"] == pytest.approx(1.0)  # 50.2 is highest
        assert result["D"] == pytest.approx(1 / 6)  # 5.7 is lowest
        # All values should be in [0, 1]
        assert result.min() > 0
        assert result.max() <= 1.0

    def test_rank_preserves_nan(self):
        x = pd.Series([1.0, np.nan, 3.0, 2.0])
        result = rank(x)
        assert np.isnan(result.iloc[1])
        assert result.iloc[0] == pytest.approx(1 / 3)

    def test_rank_uniform_distribution(self):
        """Rank output should be roughly uniformly distributed."""
        rng = np.random.default_rng(42)
        x = pd.Series(rng.normal(0, 1, 1000))
        result = rank(x)
        assert result.min() > 0
        assert result.max() <= 1.0
        # Check approximate uniformity
        assert abs(result.mean() - 0.5) < 0.05

    def test_rank_ties(self):
        """Ties should get average rank."""
        x = pd.Series([1.0, 1.0, 3.0])
        result = rank(x)
        # Two tied values at rank 1 and 2 → average = 1.5 → pct = 1.5/3 = 0.5
        assert result.iloc[0] == result.iloc[1]


class TestScale:
    def test_basic_scale(self):
        x = pd.Series([2.0, -3.0, 5.0])
        result = scale(x)
        assert abs(result.abs().sum() - 1.0) < 1e-10

    def test_scale_zero(self):
        x = pd.Series([0.0, 0.0])
        result = scale(x)
        assert (result == 0.0).all()

    def test_scale_preserves_direction(self):
        x = pd.Series([2.0, -3.0, 5.0])
        result = scale(x)
        assert result.iloc[0] > 0  # positive stays positive
        assert result.iloc[1] < 0  # negative stays negative


class TestIndNeutralize:
    def test_basic_neutralization(self, simple_series, groups_series):
        result = ind_neutralize(simple_series, groups_series)
        # Within each group, values should sum to approximately 0
        for group in groups_series.unique():
            group_mask = groups_series == group
            group_sum = result[group_mask].sum()
            assert abs(group_sum) < 1e-10, f"Group {group} sum = {group_sum}"

    def test_industry_neutralization_specific(self):
        alpha = pd.Series([1.0, 2.0, 3.0, 4.0], index=["A", "B", "C", "D"])
        industries = pd.Series(["tech", "tech", "fin", "fin"], index=["A", "B", "C", "D"])
        result = ind_neutralize(alpha, industries)
        # tech: mean(1,2)=1.5 → A=-0.5, B=0.5
        # fin: mean(3,4)=3.5 → C=-0.5, D=0.5
        assert result["A"] == pytest.approx(-0.5)
        assert result["B"] == pytest.approx(0.5)
        assert result["C"] == pytest.approx(-0.5)
        assert result["D"] == pytest.approx(0.5)


class TestZscore:
    def test_basic_zscore(self):
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = zscore(x)
        assert abs(result.mean()) < 1e-10
        assert abs(result.std() - 1.0) < 0.1  # approximately unit variance

    def test_zscore_constant(self):
        x = pd.Series([5.0, 5.0, 5.0])
        result = zscore(x)
        assert (result == 0.0).all()


class TestWinsorize:
    def test_basic_winsorize(self):
        rng = np.random.default_rng(42)
        x = pd.Series(rng.normal(0, 1, 1000))
        result = winsorize(x, limits=(0.05, 0.95))
        assert result.min() >= x.quantile(0.05) - 1e-10
        assert result.max() <= x.quantile(0.95) + 1e-10


# ==========================================================================
# Time-Series Operator Tests
# ==========================================================================


class TestDelay:
    def test_delay_1(self, simple_matrix):
        result = delay(simple_matrix, 1)
        # delay(x, 1) should be the second-to-last row
        expected = simple_matrix.iloc[-2]
        pd.testing.assert_series_equal(result, expected)

    def test_delay_2(self, simple_matrix):
        result = delay(simple_matrix, 2)
        expected = simple_matrix.iloc[-3]
        pd.testing.assert_series_equal(result, expected)


class TestDelta:
    def test_delta_1(self, simple_matrix):
        result = delta(simple_matrix, 1)
        expected = simple_matrix.iloc[-1] - simple_matrix.iloc[-2]
        pd.testing.assert_series_equal(result, expected)

    def test_delta_specific(self, simple_matrix):
        result = delta(simple_matrix, 2)
        # X: 14.0 - 11.0 = 3.0
        assert result["X"] == pytest.approx(3.0)


class TestTsSum:
    def test_ts_sum_all(self, simple_matrix):
        result = ts_sum(simple_matrix, 5)
        # X: 10+12+11+13+14 = 60
        assert result["X"] == pytest.approx(60.0)

    def test_ts_sum_partial(self, simple_matrix):
        result = ts_sum(simple_matrix, 3)
        # X: last 3 = 11+13+14 = 38
        assert result["X"] == pytest.approx(38.0)


class TestTsMean:
    def test_ts_mean(self, simple_matrix):
        result = ts_mean(simple_matrix, 5)
        assert result["X"] == pytest.approx(12.0)  # mean(10,12,11,13,14)

    def test_ts_mean_3(self, simple_matrix):
        result = ts_mean(simple_matrix, 3)
        # X: mean(11, 13, 14) = 12.667
        assert result["X"] == pytest.approx(38.0 / 3)


class TestTsStd:
    def test_ts_std(self, simple_matrix):
        result = ts_std(simple_matrix, 5)
        x_vals = np.array([10.0, 12.0, 11.0, 13.0, 14.0])
        expected_std = np.std(x_vals, ddof=1)
        assert result["X"] == pytest.approx(expected_std)


class TestTsMinMax:
    def test_ts_min(self, simple_matrix):
        result = ts_min(simple_matrix, 5)
        assert result["X"] == pytest.approx(10.0)

    def test_ts_max(self, simple_matrix):
        result = ts_max(simple_matrix, 5)
        assert result["X"] == pytest.approx(14.0)


class TestTsRank:
    def test_ts_rank_max(self, simple_matrix):
        result = ts_rank(simple_matrix, 5)
        # X: current=14.0, which is the max → rank should be 1.0
        assert result["X"] == pytest.approx(1.0)

    def test_ts_rank_min(self):
        dates = pd.date_range("2023-01-02", periods=5, freq="B")
        data = {"A": [5.0, 4.0, 3.0, 2.0, 1.0]}  # Decreasing → current is min
        mat = pd.DataFrame(data, index=dates)
        result = ts_rank(mat, 5)
        assert result["A"] == pytest.approx(0.0)


class TestDecayLinear:
    def test_decay_linear_3(self):
        """decay_linear(x, 3) = (x[t]*3 + x[t-1]*2 + x[t-2]*1) / 6"""
        dates = pd.date_range("2023-01-02", periods=3, freq="B")
        data = {"A": [10.0, 20.0, 30.0]}
        mat = pd.DataFrame(data, index=dates)
        result = decay_linear(mat, 3)
        # Weights: [1, 2, 3] for oldest to newest
        # (10*1 + 20*2 + 30*3) / 6 = (10 + 40 + 90) / 6 = 140/6 = 23.333...
        expected = (10 * 1 + 20 * 2 + 30 * 3) / (1 + 2 + 3)
        assert result["A"] == pytest.approx(expected)


class TestDecayExp:
    def test_decay_exp(self):
        dates = pd.date_range("2023-01-02", periods=3, freq="B")
        data = {"A": [10.0, 20.0, 30.0]}
        mat = pd.DataFrame(data, index=dates)
        f = 0.5
        result = decay_exp(mat, f, 3)
        # Weights: f^2, f^1, f^0 = 0.25, 0.5, 1.0
        # (10*0.25 + 20*0.5 + 30*1.0) / (0.25+0.5+1.0) = (2.5+10+30)/1.75
        expected = (10 * 0.25 + 20 * 0.5 + 30 * 1.0) / (0.25 + 0.5 + 1.0)
        assert result["A"] == pytest.approx(expected)


class TestCountNans:
    def test_count_nans_none(self, simple_matrix):
        result = count_nans(simple_matrix, 5)
        assert (result == 0).all()

    def test_count_nans_some(self):
        dates = pd.date_range("2023-01-02", periods=3, freq="B")
        data = {"A": [1.0, np.nan, 3.0], "B": [np.nan, np.nan, 1.0]}
        mat = pd.DataFrame(data, index=dates)
        result = count_nans(mat, 3)
        assert result["A"] == 1
        assert result["B"] == 2


class TestProduct:
    def test_product(self):
        dates = pd.date_range("2023-01-02", periods=3, freq="B")
        data = {"A": [2.0, 3.0, 4.0]}
        mat = pd.DataFrame(data, index=dates)
        result = product(mat, 3)
        assert result["A"] == pytest.approx(24.0)


class TestCorrelation:
    def test_perfect_correlation(self):
        dates = pd.date_range("2023-01-02", periods=10, freq="B")
        x_data = {"A": list(range(10))}
        y_data = {"A": [v * 2 for v in range(10)]}
        x = pd.DataFrame(x_data, index=dates, dtype=float)
        y = pd.DataFrame(y_data, index=dates, dtype=float)
        result = correlation(x, y, 10)
        assert result["A"] == pytest.approx(1.0, abs=1e-10)


# ==========================================================================
# Element-Wise Operator Tests
# ==========================================================================


class TestElementWise:
    def test_abs(self):
        x = pd.Series([-1.0, 2.0, -3.0])
        result = op_abs(x)
        expected = pd.Series([1.0, 2.0, 3.0])
        pd.testing.assert_series_equal(result, expected)

    def test_sign(self):
        x = pd.Series([-5.0, 0.0, 3.0])
        result = sign(x)
        expected = pd.Series([-1.0, 0.0, 1.0])
        pd.testing.assert_series_equal(result, expected)

    def test_log(self):
        x = pd.Series([1.0, math.e, math.e**2])
        result = log(x)
        expected = pd.Series([0.0, 1.0, 2.0])
        pd.testing.assert_series_equal(result, expected, atol=1e-10)

    def test_log_negative(self):
        x = pd.Series([-1.0, 0.0, 1.0])
        result = log(x)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(0.0)

    def test_signed_power(self):
        x = pd.Series([-2.0, 0.0, 3.0])
        result = signed_power(x, 2)
        # sign(x) * |x|^2
        expected = pd.Series([-4.0, 0.0, 9.0])
        pd.testing.assert_series_equal(result, expected)

    def test_pasteurize_removes_inf(self):
        x = pd.Series([1.0, np.inf, -np.inf, 3.0], index=["A", "B", "C", "D"])
        result = pasteurize(x)
        assert np.isnan(result["B"])
        assert np.isnan(result["C"])
        assert result["A"] == 1.0
        assert result["D"] == 3.0

    def test_pasteurize_filters_universe(self):
        x = pd.Series([1.0, 2.0, 3.0, 4.0], index=["A", "B", "C", "D"])
        result = pasteurize(x, universe=["A", "C"])
        assert result["A"] == 1.0
        assert result["C"] == 3.0
        assert np.isnan(result["B"])
        assert np.isnan(result["D"])

    def test_min_max(self):
        x = pd.Series([1.0, 5.0, 3.0])
        y = pd.Series([2.0, 4.0, 3.0])
        min_result = op_min(x, y)
        max_result = op_max(x, y)
        assert min_result.tolist() == [1.0, 4.0, 3.0]
        assert max_result.tolist() == [2.0, 5.0, 3.0]

    def test_tail(self):
        x = pd.Series([1.0, 5.0, 3.0, 10.0])
        result = tail(x, 2.0, 6.0, 0.0)
        # Values between 2 and 6 → set to 0
        assert result.iloc[0] == 1.0  # outside range
        assert result.iloc[1] == 0.0  # 5 is in [2, 6]
        assert result.iloc[2] == 0.0  # 3 is in [2, 6]
        assert result.iloc[3] == 10.0  # outside range
