"""
Unit tests for neutralization methods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.simulation.neutralization import neutralize, _market_neutralize, risk_neutralize


class TestMarketNeutralization:
    def test_sums_to_zero(self):
        alpha = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _market_neutralize(alpha)
        assert abs(result.sum()) < 1e-10

    def test_preserves_relative_order(self):
        alpha = pd.Series([1.0, 5.0, 3.0])
        result = _market_neutralize(alpha)
        assert result.iloc[1] > result.iloc[2] > result.iloc[0]


class TestGroupNeutralization:
    def test_industry_neutralization_sums_to_zero(self, tiny_ctx):
        """After neutralization, each group sums to approximately zero."""
        import datetime as dt

        date = tiny_ctx._trading_days[100] if len(tiny_ctx._trading_days) > 100 else tiny_ctx._trading_days[-1]
        universe = tiny_ctx.get_universe(date, "TOP3000")[:10]

        alpha = pd.Series(
            np.random.default_rng(42).normal(0, 1, len(universe)),
            index=universe,
        )

        result = neutralize(alpha, "industry", tiny_ctx, date)

        # Check that each industry group sums to ~0
        groups = pd.Series({t: tiny_ctx.get_industry(t, date, "industry") for t in result.index})
        for group_name, group_vals in result.groupby(groups):
            assert abs(group_vals.sum()) < 1e-10, f"Group {group_name} sum = {group_vals.sum()}"

    def test_none_is_identity(self):
        alpha = pd.Series([1.0, 2.0, 3.0])
        result = neutralize(alpha, "none", None, None)
        pd.testing.assert_series_equal(result, alpha)


class TestRiskNeutralization:
    def test_removes_factor_exposure(self):
        rng = np.random.default_rng(42)
        n = 100

        # Alpha = 2 * factor + noise
        factor = pd.Series(rng.normal(0, 1, n), index=[f"S{i}" for i in range(n)])
        noise = pd.Series(rng.normal(0, 0.5, n), index=factor.index)
        alpha = 2.0 * factor + noise

        risk_factors = pd.DataFrame({"factor1": factor})
        result = risk_neutralize(alpha, risk_factors)

        # After neutralization, correlation with factor should be ~0
        corr = result.corr(factor)
        assert abs(corr) < 0.15, f"Residual correlation = {corr}"
