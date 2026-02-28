"""
Tests for vectorized operators and the FastExpression engine.

Validates:
- All vectorized operators match GPfunctions.py behavior
- FastExpression parser handles BRAIN expression syntax
- group_neutralize, group_rank, ts_regression work correctly
- Complex real-world expressions from the user's examples parse and evaluate
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from src.operators.vectorized import (
    ts_sum, sma, ts_mean, ts_rank, ts_min, ts_max,
    delta, ts_delta, stddev, ts_std_dev,
    correlation, ts_corr, covariance, ts_cov,
    Product, delay, ts_delay,
    ArgMax, ArgMin, ts_argmax, ts_argmin,
    ts_skewness, ts_kurtosis, ts_zscore, ts_av_diff,
    ts_regression, ts_moment,
    Decay_lin, decay_linear, Decay_exp, decay_exp,
    rank, scale, zscore_cs, group_rank, group_neutralize, market_neutralize,
    add, subtract, multiply, divide, true_divide, protectedDiv,
    negative, Abs, abs_op, Sign, sign,
    SignedPower, signed_power, Inverse, inverse,
    log, log10, sqrt, square, log_diff, s_log_1p,
    Tail, tail, df_max, df_min, if_else, power,
    npfadd, npfsub, npfmul, npfdiv,
    pasteurize, winsorize, truncate, ts_count_nans,
)
from src.operators.fastexpression import (
    FastExpressionEngine, FastExpressionLexer, FastExpressionParser,
    create_engine_from_context,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """5 days × 3 tickers DataFrame for testing."""
    dates = pd.date_range("2023-01-01", periods=5, freq="B")
    data = {
        "A": [10.0, 12.0, 11.0, 14.0, 13.0],
        "B": [20.0, 18.0, 22.0, 21.0, 25.0],
        "C": [5.0,  6.0,  4.0,  7.0,  8.0],
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_df2():
    """Second DataFrame for two-argument operators."""
    dates = pd.date_range("2023-01-01", periods=5, freq="B")
    data = {
        "A": [1.0, 2.0, 3.0, 4.0, 5.0],
        "B": [5.0, 4.0, 3.0, 2.0, 1.0],
        "C": [2.0, 2.0, 2.0, 2.0, 2.0],
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def groups():
    """Industry group assignments."""
    return pd.Series({"A": "Tech", "B": "Tech", "C": "Finance"})


@pytest.fixture
def engine(sample_df, sample_df2, groups):
    """Pre-configured FastExpressionEngine for testing."""
    return FastExpressionEngine(
        data_fields={
            "close": sample_df,
            "volume": sample_df2,
            "returns": sample_df.pct_change(),
            "open": sample_df + 0.5,
            "high": sample_df + 1.0,
            "low": sample_df - 1.0,
        },
        groups={"industry": groups, "subindustry": groups, "sector": groups},
    )


@pytest.fixture
def big_df():
    """Larger DataFrame for ts_regression and window operators."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    tickers = ["A", "B", "C", "D", "E"]
    data = np.random.randn(300, 5).cumsum(axis=0) + 100
    return pd.DataFrame(data, index=dates, columns=tickers)


# ===========================================================================
# TESTS: Vectorized Operators
# ===========================================================================

class TestTimeSeriesOperators:
    """Test time-series operators (operate down each column)."""

    def test_ts_sum(self, sample_df):
        result = ts_sum(sample_df, 3)
        # ts_sum of A for last 3 days: 11+14+13 = 38
        assert result["A"].iloc[4] == pytest.approx(38.0)

    def test_sma(self, sample_df):
        result = sma(sample_df, 3)
        assert result["A"].iloc[4] == pytest.approx((11 + 14 + 13) / 3)

    def test_ts_mean_is_sma(self, sample_df):
        pd.testing.assert_frame_equal(ts_mean(sample_df, 3), sma(sample_df, 3))

    def test_ts_rank(self, sample_df):
        result = ts_rank(sample_df, 5)
        # All values are ranks within window as pct
        assert result.min().min() >= 0
        assert result.max().max() <= 1

    def test_ts_min(self, sample_df):
        result = ts_min(sample_df, 3)
        assert result["A"].iloc[4] == 11.0  # min of [11, 14, 13]

    def test_ts_max(self, sample_df):
        result = ts_max(sample_df, 3)
        assert result["A"].iloc[4] == 14.0  # max of [11, 14, 13]

    def test_delta(self, sample_df):
        result = delta(sample_df, 1)
        assert result["A"].iloc[1] == pytest.approx(2.0)  # 12 - 10

    def test_ts_delta_is_delta(self, sample_df):
        pd.testing.assert_frame_equal(ts_delta(sample_df, 2), delta(sample_df, 2))

    def test_stddev(self, sample_df):
        result = stddev(sample_df, 5)
        expected = sample_df["A"].rolling(5, min_periods=2).std()
        pd.testing.assert_series_equal(result["A"], expected, check_names=False)

    def test_ts_std_dev_is_stddev(self, sample_df):
        pd.testing.assert_frame_equal(ts_std_dev(sample_df, 3), stddev(sample_df, 3))

    def test_correlation(self, sample_df, sample_df2):
        result = correlation(sample_df, sample_df2, 3)
        assert result.shape == sample_df.shape

    def test_correlation_same_returns_zero(self, sample_df):
        result = correlation(sample_df, sample_df, 3)
        assert (result == 0).all().all()

    def test_covariance(self, sample_df, sample_df2):
        result = covariance(sample_df, sample_df2, 3)
        assert result.shape == sample_df.shape

    def test_Product(self, sample_df):
        result = Product(sample_df, 2)
        # Product of last 2: A = 14*13 = 182
        assert result["A"].iloc[4] == pytest.approx(14 * 13)

    def test_delay(self, sample_df):
        result = delay(sample_df, 1)
        assert result["A"].iloc[1] == 10.0  # shifted by 1

    def test_ts_delay_is_delay(self, sample_df):
        pd.testing.assert_frame_equal(ts_delay(sample_df, 2), delay(sample_df, 2))

    def test_ArgMax(self, sample_df):
        result = ArgMax(sample_df, 5)
        # In window [10, 12, 11, 14, 13], argmax=3 (0-indexed) → +1 = 4
        assert result["A"].iloc[4] == 4

    def test_ArgMin(self, sample_df):
        result = ArgMin(sample_df, 5)
        # In window [10, 12, 11, 14, 13], argmin=0 → +1 = 1
        assert result["A"].iloc[4] == 1

    def test_ts_skewness(self, sample_df):
        result = ts_skewness(sample_df, 5)
        assert result.shape == sample_df.shape

    def test_ts_kurtosis(self, sample_df):
        result = ts_kurtosis(sample_df, 5)
        assert result.shape == sample_df.shape

    def test_ts_zscore(self, sample_df):
        result = ts_zscore(sample_df, 3)
        assert result.shape == sample_df.shape
        # z-score should have some non-NaN values
        assert result.dropna(how="all").shape[0] > 0

    def test_ts_av_diff(self, sample_df):
        result = ts_av_diff(sample_df, 3)
        # x - sma(x, 3)
        assert result.shape == sample_df.shape

    def test_ts_regression(self, big_df):
        """Test ts_regression with rettype=2 (slope)."""
        y = big_df
        x = big_df + np.random.randn(*big_df.shape) * 0.1  # noisy version
        result = ts_regression(y, x, window=60, lag=0, rettype=2)
        # Slope should be close to 1 (since y ≈ x)
        last_row = result.iloc[-1].dropna()
        assert len(last_row) > 0
        assert last_row.mean() > 0.5  # slope > 0.5

    def test_ts_regression_with_lag(self, big_df):
        """Test ts_regression with lag parameter."""
        result = ts_regression(big_df, big_df, window=60, lag=5, rettype=0)
        assert result.shape == big_df.shape


class TestDecayOperators:
    """Test linear and exponential decay."""

    def test_Decay_lin(self, sample_df):
        result = Decay_lin(sample_df, 3)
        assert result.shape == sample_df.shape
        # Linear decay should be weighted average with [1, 2, 3] weights
        a_vals = [11.0, 14.0, 13.0]
        weights = [1, 2, 3]
        expected = sum(v * w for v, w in zip(a_vals, weights)) / sum(weights)
        assert result["A"].iloc[4] == pytest.approx(expected)

    def test_decay_linear_is_Decay_lin(self, sample_df):
        pd.testing.assert_frame_equal(decay_linear(sample_df, 3), Decay_lin(sample_df, 3))

    def test_Decay_exp(self, sample_df):
        result = Decay_exp(sample_df, 0.5)
        assert result.shape == sample_df.shape

    def test_Decay_exp_clamps_alpha(self, sample_df):
        # alpha > 1 should get clamped to 0.99
        result = Decay_exp(sample_df, 1.5)
        assert result.shape == sample_df.shape


class TestCrossSectionalOperators:
    """Test cross-sectional operators (operate across instruments)."""

    def test_rank(self, sample_df):
        result = rank(sample_df)
        # Each row should have ranks [1/3, 2/3, 1.0] in some order
        assert result.shape == sample_df.shape
        for i in range(len(result)):
            row_vals = sorted(result.iloc[i].values)
            assert row_vals == pytest.approx([1 / 3, 2 / 3, 1.0], rel=0.01)

    def test_scale(self, sample_df):
        result = scale(sample_df, 1.0)
        # sum(abs(row)) should be ~1 for each row
        for i in range(len(result)):
            assert result.iloc[i].abs().sum() == pytest.approx(1.0, rel=0.01)

    def test_zscore_cs(self, sample_df):
        result = zscore_cs(sample_df)
        # mean across instruments should be ~0
        for i in range(len(result)):
            assert result.iloc[i].mean() == pytest.approx(0.0, abs=1e-10)

    def test_group_rank(self, sample_df, groups):
        result = group_rank(sample_df, groups)
        assert result.shape == sample_df.shape

    def test_group_neutralize(self, sample_df, groups):
        result = group_neutralize(sample_df, groups)
        # Within Tech (A, B), mean should be ~0
        for i in range(len(result)):
            tech_mean = result.iloc[i][["A", "B"]].mean()
            assert tech_mean == pytest.approx(0.0, abs=1e-10)

    def test_market_neutralize(self, sample_df):
        result = market_neutralize(sample_df)
        # Mean across all instruments should be ~0
        for i in range(len(result)):
            assert result.iloc[i].mean() == pytest.approx(0.0, abs=1e-10)


class TestElementWiseOperators:
    """Test element-wise operators."""

    def test_add(self, sample_df, sample_df2):
        result = add(sample_df, sample_df2)
        assert result["A"].iloc[0] == 11.0  # 10 + 1

    def test_subtract(self, sample_df, sample_df2):
        result = subtract(sample_df, sample_df2)
        assert result["A"].iloc[0] == 9.0

    def test_multiply(self, sample_df, sample_df2):
        result = multiply(sample_df, sample_df2)
        assert result["A"].iloc[0] == 10.0

    def test_divide(self, sample_df, sample_df2):
        result = divide(sample_df, sample_df2)
        assert result["A"].iloc[0] == 10.0

    def test_protectedDiv_zero(self):
        df1 = pd.DataFrame({"A": [1.0, 2.0]})
        df2 = pd.DataFrame({"A": [0.0, 2.0]})
        result = protectedDiv(df1, df2)
        assert result["A"].iloc[0] == 0.0
        assert result["A"].iloc[1] == 1.0

    def test_negative(self, sample_df):
        result = negative(sample_df)
        assert result["A"].iloc[0] == -10.0

    def test_Abs(self, sample_df):
        neg_df = negative(sample_df)
        result = Abs(neg_df)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_Sign(self, sample_df):
        result = Sign(sample_df)
        assert (result == 1.0).all().all()

    def test_SignedPower(self, sample_df):
        result = SignedPower(sample_df, 2)
        assert result["A"].iloc[0] == pytest.approx(100.0)

    def test_Inverse(self, sample_df):
        result = Inverse(sample_df)
        assert result["A"].iloc[0] == pytest.approx(0.1)

    def test_log(self, sample_df):
        result = log(sample_df)
        assert result["A"].iloc[0] == pytest.approx(np.log(10.0))

    def test_sqrt(self, sample_df):
        result = sqrt(sample_df)
        assert result["A"].iloc[0] == pytest.approx(np.sqrt(10.0))

    def test_square(self, sample_df):
        result = square(sample_df)
        assert result["A"].iloc[0] == pytest.approx(100.0)

    def test_log_diff(self, sample_df):
        result = log_diff(sample_df)
        assert result["A"].iloc[1] == pytest.approx(2.0)  # 12 - 10

    def test_s_log_1p(self, sample_df):
        result = s_log_1p(sample_df)
        assert result["A"].iloc[0] == pytest.approx(np.log1p(10.0))

    def test_Tail(self, sample_df):
        mid_df = sample_df - 12.0  # some values near zero
        result = Tail(mid_df, 2.0)
        # Values in [-2, 2] should be set to 0
        for col in result.columns:
            for val in result[col]:
                assert abs(val) >= 2.0 or val == 0.0

    def test_df_max(self, sample_df, sample_df2):
        result = df_max(sample_df, sample_df2)
        assert result["A"].iloc[0] == 10.0  # max(10, 1)
        assert result["B"].iloc[0] == 20.0  # max(20, 5)

    def test_df_min(self, sample_df, sample_df2):
        result = df_min(sample_df, sample_df2)
        assert result["A"].iloc[0] == 1.0

    def test_npfadd(self, sample_df):
        result = npfadd(sample_df, 5.0)
        assert result["A"].iloc[0] == 15.0

    def test_npfmul(self, sample_df):
        result = npfmul(sample_df, 2.0)
        assert result["A"].iloc[0] == 20.0

    def test_pasteurize(self):
        df = pd.DataFrame({"A": [1.0, np.inf, -np.inf, np.nan, 5.0]})
        result = pasteurize(df)
        assert np.isnan(result["A"].iloc[1])
        assert np.isnan(result["A"].iloc[2])
        assert result["A"].iloc[0] == 1.0

    def test_truncate(self, sample_df):
        result = truncate(sample_df, 15.0)
        assert result.max().max() <= 15.0


# ===========================================================================
# TESTS: FastExpression Lexer
# ===========================================================================

class TestFastExpressionLexer:
    """Test the lexer."""

    def test_simple_function(self):
        tokens = FastExpressionLexer("rank(close)").tokenize()
        assert tokens[0].value == "rank"
        assert tokens[1].type == "LPAREN"
        assert tokens[2].value == "close"
        assert tokens[3].type == "RPAREN"

    def test_nested_function(self):
        tokens = FastExpressionLexer("rank(ts_delta(close, 5))").tokenize()
        types = [t.type for t in tokens[:-1]]  # exclude EOF
        assert types == ["IDENT", "LPAREN", "IDENT", "LPAREN", "IDENT", "COMMA", "NUMBER", "RPAREN", "RPAREN"]

    def test_multiplication(self):
        tokens = FastExpressionLexer("rank(a) * rank(b)").tokenize()
        assert any(t.value == "*" for t in tokens)

    def test_keyword_arg(self):
        tokens = FastExpressionLexer("ts_regression(y, x, 252, lag=126, rettype=2)").tokenize()
        # Should have EQUALS tokens
        equals_tokens = [t for t in tokens if t.type == "EQUALS"]
        assert len(equals_tokens) == 2

    def test_unary_minus(self):
        tokens = FastExpressionLexer("-rank(close)").tokenize()
        assert tokens[0].value == "NEG"

    def test_division_slash(self):
        tokens = FastExpressionLexer("close / volume").tokenize()
        assert any(t.value == "/" for t in tokens)

    def test_complex_expression(self):
        expr = "rank(ts_delta(divide(close, volume), 120)) * rank(ts_rank(returns, 60))"
        tokens = FastExpressionLexer(expr).tokenize()
        idents = [t.value for t in tokens if t.type == "IDENT"]
        assert "rank" in idents
        assert "ts_delta" in idents
        assert "divide" in idents
        assert "close" in idents
        assert "volume" in idents
        assert "ts_rank" in idents
        assert "returns" in idents


# ===========================================================================
# TESTS: FastExpression Engine - End-to-End
# ===========================================================================

class TestFastExpressionEngine:
    """Test the full parse → evaluate pipeline."""

    def test_simple_field(self, engine, sample_df):
        result = engine.evaluate("close")
        pd.testing.assert_frame_equal(result, sample_df)

    def test_rank(self, engine, sample_df):
        result = engine.evaluate("rank(close)")
        expected = rank(sample_df)
        pd.testing.assert_frame_equal(result, expected)

    def test_negative_rank(self, engine, sample_df):
        result = engine.evaluate("-rank(close)")
        expected = -rank(sample_df)
        pd.testing.assert_frame_equal(result, expected)

    def test_delta(self, engine, sample_df):
        result = engine.evaluate("delta(close, 1)")
        expected = delta(sample_df, 1)
        pd.testing.assert_frame_equal(result, expected)

    def test_ts_delta(self, engine, sample_df):
        result = engine.evaluate("ts_delta(close, 2)")
        expected = ts_delta(sample_df, 2)
        pd.testing.assert_frame_equal(result, expected)

    def test_rank_of_delta(self, engine, sample_df):
        result = engine.evaluate("rank(delta(close, 1))")
        expected = rank(delta(sample_df, 1))
        pd.testing.assert_frame_equal(result, expected)

    def test_divide_function(self, engine, sample_df, sample_df2):
        result = engine.evaluate("divide(close, volume)")
        expected = divide(sample_df, sample_df2)
        pd.testing.assert_frame_equal(result, expected)

    def test_division_operator(self, engine, sample_df, sample_df2):
        result = engine.evaluate("close / volume")
        expected = sample_df / sample_df2
        pd.testing.assert_frame_equal(result, expected)

    def test_multiplication(self, engine, sample_df):
        result = engine.evaluate("rank(close) * rank(volume)")
        expected = rank(sample_df) * rank(engine.data_fields["volume"])
        pd.testing.assert_frame_equal(result, expected)

    def test_nested_three_deep(self, engine, sample_df, sample_df2):
        result = engine.evaluate("rank(ts_delta(divide(close, volume), 2))")
        expected = rank(ts_delta(divide(sample_df, sample_df2), 2))
        pd.testing.assert_frame_equal(result, expected)

    def test_two_multiplied_ranks(self, engine, sample_df, sample_df2):
        result = engine.evaluate("rank(delta(close, 1)) * rank(ts_rank(volume, 3))")
        expected = rank(delta(sample_df, 1)) * rank(ts_rank(sample_df2, 3))
        pd.testing.assert_frame_equal(result, expected)

    def test_sma(self, engine, sample_df):
        result = engine.evaluate("sma(close, 3)")
        expected = sma(sample_df, 3)
        pd.testing.assert_frame_equal(result, expected)

    def test_Decay_lin(self, engine, sample_df):
        result = engine.evaluate("Decay_lin(close, 3)")
        expected = Decay_lin(sample_df, 3)
        pd.testing.assert_frame_equal(result, expected)

    def test_inverse(self, engine, sample_df):
        result = engine.evaluate("inverse(close)")
        expected = inverse(sample_df)
        pd.testing.assert_frame_equal(result, expected)

    def test_abs(self, engine, sample_df):
        result = engine.evaluate("Abs(-close)")
        # -close then abs → same as close
        expected = Abs(-sample_df)
        pd.testing.assert_frame_equal(result, expected)

    def test_ts_zscore(self, engine, sample_df):
        result = engine.evaluate("ts_zscore(close, 3)")
        expected = ts_zscore(sample_df, 3)
        pd.testing.assert_frame_equal(result, expected)

    def test_stddev(self, engine, sample_df):
        result = engine.evaluate("stddev(close, 3)")
        expected = stddev(sample_df, 3)
        pd.testing.assert_frame_equal(result, expected)

    def test_ts_std_dev(self, engine, sample_df):
        result = engine.evaluate("ts_std_dev(close, 3)")
        expected = ts_std_dev(sample_df, 3)
        pd.testing.assert_frame_equal(result, expected)

    def test_group_neutralize(self, engine, sample_df, groups):
        result = engine.evaluate("group_neutralize(close, industry)")
        expected = group_neutralize(sample_df, groups)
        pd.testing.assert_frame_equal(result, expected)

    def test_group_rank(self, engine, sample_df, groups):
        result = engine.evaluate("group_rank(close, subindustry)")
        expected = group_rank(sample_df, groups)
        pd.testing.assert_frame_equal(result, expected)

    def test_scalar_constant(self, engine, sample_df):
        result = engine.evaluate("close + 5")
        expected = sample_df + 5
        pd.testing.assert_frame_equal(result, expected)

    def test_comparison(self, engine, sample_df):
        result = engine.evaluate("close > 12")
        assert result["A"].iloc[0] == 0.0  # 10 > 12 is False
        assert result["A"].iloc[3] == 1.0  # 14 > 12 is True

    def test_ternary(self, engine, sample_df):
        result = engine.evaluate("close > 12 ? close : -close")
        assert result["A"].iloc[0] == -10.0  # condition false → -close
        assert result["A"].iloc[3] == 14.0   # condition true → close

    def test_log_diff(self, engine, sample_df):
        result = engine.evaluate("log_diff(close)")
        expected = log_diff(sample_df)
        pd.testing.assert_frame_equal(result, expected)

    def test_s_log_1p(self, engine, sample_df):
        result = engine.evaluate("s_log_1p(close)")
        expected = s_log_1p(sample_df)
        pd.testing.assert_frame_equal(result, expected)


class TestFastExpressionRealWorld:
    """Test real-world BRAIN expressions from the user's examples."""

    @pytest.fixture
    def real_engine(self, big_df):
        """Engine with many data fields for real expressions."""
        np.random.seed(42)
        n_rows, n_cols = big_df.shape

        # Generate realistic financial data
        close = big_df
        returns = close.pct_change()
        volume = pd.DataFrame(
            np.abs(np.random.randn(n_rows, n_cols)) * 1e6,
            index=big_df.index, columns=big_df.columns
        )
        revenue = pd.DataFrame(
            np.abs(np.random.randn(n_rows, n_cols)) * 500 + 200,
            index=big_df.index, columns=big_df.columns
        )
        assets = revenue * 3
        operating_income = revenue * 0.15
        enterprise_value = revenue * 10
        invested_capital = revenue * 2
        cashflow_op = revenue * 0.1
        assets_curr = revenue * 0.3
        rd_expense = revenue * 0.05
        sales = revenue * 1.0
        debt = revenue * 0.5

        groups = pd.Series({"A": "Tech", "B": "Tech", "C": "Finance", "D": "Finance", "E": "Energy"})

        return FastExpressionEngine(
            data_fields={
                "close": close,
                "volume": volume,
                "returns": returns,
                "open": close + 0.5,
                "high": close + 1,
                "low": close - 1,
                "revenue": revenue,
                "assets": assets,
                "operating_income": operating_income,
                "enterprise_value": enterprise_value,
                "invested_capital": invested_capital,
                "cashflow_op": cashflow_op,
                "assets_curr": assets_curr,
                "rd_expense": rd_expense,
                "sales": sales,
                "debt": debt,
            },
            groups={"industry": groups, "subindustry": groups, "sector": groups},
        )

    def test_rank_times_rank(self, real_engine):
        """rank(delta(close, 5)) * rank(ts_rank(returns, 60))"""
        result = real_engine.evaluate("rank(delta(close, 5)) * rank(ts_rank(returns, 60))")
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 300
        assert result.shape[1] == 5
        # Last row should have values
        last = result.iloc[-1].dropna()
        assert len(last) > 0

    def test_divide_in_rank(self, real_engine):
        """rank(ts_delta(divide(operating_income, revenue), 120))"""
        result = real_engine.evaluate("rank(ts_delta(divide(operating_income, revenue), 120))")
        assert isinstance(result, pd.DataFrame)
        last = result.iloc[-1].dropna()
        assert len(last) > 0

    def test_ts_zscore_with_division(self, real_engine):
        """rank(ts_zscore(close / enterprise_value, 120))"""
        result = real_engine.evaluate("rank(ts_zscore(close / enterprise_value, 120))")
        assert isinstance(result, pd.DataFrame)
        last = result.iloc[-1].dropna()
        assert len(last) > 0

    def test_group_rank_in_expression(self, real_engine):
        """rank(group_rank(inverse(ts_std_dev(divide(rd_expense, sales), 60)), subindustry))"""
        result = real_engine.evaluate(
            "rank(group_rank(inverse(ts_std_dev(divide(rd_expense, sales), 60)), subindustry))"
        )
        assert isinstance(result, pd.DataFrame)

    def test_group_neutralize_in_expression(self, real_engine):
        """group_neutralize(rank(delta(close, 5)), sector)"""
        result = real_engine.evaluate("group_neutralize(rank(delta(close, 5)), sector)")
        assert isinstance(result, pd.DataFrame)

    def test_three_way_product(self, real_engine):
        """Three rank terms multiplied together."""
        expr = (
            "rank(ts_delta(divide(operating_income, revenue), 120)) * "
            "rank(ts_rank(divide(cashflow_op, assets_curr), 60)) * "
            "rank(ts_zscore(close / enterprise_value, 120))"
        )
        result = real_engine.evaluate(expr)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 300

    def test_ts_regression_expression(self, real_engine):
        """Test ts_regression with keyword args."""
        expr = "rank(ts_regression(revenue, assets, 200, lag=100, rettype=2))"
        result = real_engine.evaluate(expr)
        assert isinstance(result, pd.DataFrame)
        # Should have valid values after window period
        last = result.iloc[-1].dropna()
        assert len(last) > 0

    def test_negative_expression(self, real_engine):
        """Test unary minus: -rank(delta(close, 5))"""
        result = real_engine.evaluate("-rank(delta(close, 5))")
        positive = real_engine.evaluate("rank(delta(close, 5))")
        pd.testing.assert_frame_equal(result, -positive)


class TestFastExpressionFromContext:
    """Test creating engines from InMemoryDataContext."""

    def test_create_engine(self):
        """Test that create_engine_from_context works."""
        from src.data.synthetic import SyntheticDataGenerator
        from src.data.context_research import InMemoryDataContext

        ds = SyntheticDataGenerator().generate(
            n_stocks=10, n_days=50, seed=42,
            start_date="2023-01-03",
        )
        ctx = InMemoryDataContext(ds)
        engine = create_engine_from_context(ctx)

        # Should have data fields
        assert "close" in engine.data_fields
        assert "volume" in engine.data_fields
        assert "returns" in engine.data_fields

        # Should have groups
        assert "industry" in engine.groups or "sector" in engine.groups

        # Should be able to evaluate expressions
        result = engine.evaluate("rank(delta(close, 5))")
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] > 0

    def test_full_pipeline(self):
        """End-to-end: generate data → context → engine → evaluate expression."""
        from src.data.synthetic import SyntheticDataGenerator
        from src.data.context_research import InMemoryDataContext

        ds = SyntheticDataGenerator().generate(
            n_stocks=20, n_days=100, seed=42,
            start_date="2023-01-03",
        )
        ctx = InMemoryDataContext(ds)
        engine = create_engine_from_context(ctx)

        # Multi-operator expression
        result = engine.evaluate("rank(delta(close, 5)) * rank(sma(close, 10))")
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] > 50
        assert result.shape[1] > 10

        # Values should be bounded (rank * rank ∈ [0, 1])
        last = result.iloc[-1].dropna()
        assert all(0 <= v <= 1 for v in last), f"Values out of rank range: {last.values}"
