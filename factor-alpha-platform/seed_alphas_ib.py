"""Seed alpha library for IB closing-auction equity research.

These expressions are deliberately simple, liquid-data-only starting points for
delay-0 small-cap discovery. They are not production selections by themselves;
`eval_alpha_ib.py --run-seeds` evaluates and gates them before anything is
saved to `data/ib_alphas.db`.
"""
from __future__ import annotations


SEED_ALPHAS = [
    {
        "name": "candle_close_location_1d",
        "category": "candle",
        "expr": "rank(true_divide(subtract(close, low), add(subtract(high, low), 0.001)))",
        "reasoning": "Closing near the high captures same-day demand into the auction while normalizing by daily range.",
    },
    {
        "name": "candle_intraday_reversal",
        "category": "candle",
        "expr": "rank(negative(true_divide(subtract(close, open), add(subtract(high, low), 0.001))))",
        "reasoning": "Large intraday extensions often mean revert at the next close in noisy small-cap baskets.",
    },
    {
        "name": "candle_upper_shadow_pressure",
        "category": "candle",
        "expr": "rank(true_divide(subtract(high, df_max(open, close)), add(close, 0.001)))",
        "reasoning": "Persistent upper shadows can indicate failed upside auctions and short-term supply pressure.",
    },
    {
        "name": "reversal_1d",
        "category": "reversal",
        "expr": "rank(negative(returns))",
        "reasoning": "One-day reversal is a durable microstructure baseline for diversified daily equity signals.",
    },
    {
        "name": "reversal_3d_delta",
        "category": "reversal",
        "expr": "rank(negative(ts_delta(close, 3)))",
        "reasoning": "Short multi-day price displacement can mean revert after temporary liquidity shocks.",
    },
    {
        "name": "reversal_5d_rank",
        "category": "reversal",
        "expr": "rank(negative(ts_rank(close, 5)))",
        "reasoning": "Recent high closes within a one-week window are penalized to capture local exhaustion.",
    },
    {
        "name": "momentum_20d",
        "category": "momentum",
        "expr": "rank(momentum_20d)",
        "reasoning": "Twenty-day continuation is a low-complexity trend anchor with broad cross-sectional coverage.",
    },
    {
        "name": "momentum_60d_smooth",
        "category": "momentum",
        "expr": "rank(sma(momentum_60d, 5))",
        "reasoning": "Smoothing medium-horizon momentum reduces single-bar noise before cross-sectional ranking.",
    },
    {
        "name": "momentum_vol_adjusted",
        "category": "momentum",
        "expr": "rank(true_divide(momentum_20d, add(historical_volatility_20, 0.001)))",
        "reasoning": "Risk-adjusted momentum prefers names with trend strength that is not only volatility expansion.",
    },
    {
        "name": "volume_regime_20d",
        "category": "volume",
        "expr": "rank(ts_rank(volume, 20))",
        "reasoning": "Relative volume regime changes can identify attention and liquidity shifts before close.",
    },
    {
        "name": "volume_dollar_flow",
        "category": "volume",
        "expr": "rank(ts_rank(dollars_traded, 20))",
        "reasoning": "Dollar volume is more comparable than share volume across differently priced securities.",
    },
    {
        "name": "volume_price_confirmation",
        "category": "volume",
        "expr": "rank(multiply(ts_rank(volume, 20), returns))",
        "reasoning": "Price moves confirmed by relative volume are separated from thin, noisy prints.",
    },
    {
        "name": "volatility_low_20d",
        "category": "volatility",
        "expr": "rank(negative(historical_volatility_20))",
        "reasoning": "Lower realized volatility names are useful anchors for capacity and optimizer stability.",
    },
    {
        "name": "volatility_range_compression",
        "category": "volatility",
        "expr": "rank(negative(sma(high_low_range, 10)))",
        "reasoning": "Range compression can precede cleaner next-bar execution and lower close-to-close noise.",
    },
    {
        "name": "composite_reversal_volume",
        "category": "composite",
        "expr": "rank(add(rank(negative(returns)), rank(ts_rank(volume, 20))))",
        "reasoning": "Combines short-term reversal with attention from relative volume to improve signal breadth.",
    },
    {
        "name": "composite_momentum_quality",
        "category": "composite",
        "expr": "rank(add(rank(momentum_20d), rank(negative(historical_volatility_20))))",
        "reasoning": "Pairs medium-term continuation with lower volatility to avoid chasing unstable names.",
    },
]


def get_seed_expressions() -> list[str]:
    """Return seed expressions in library order."""
    return [alpha["expr"] for alpha in SEED_ALPHAS]
