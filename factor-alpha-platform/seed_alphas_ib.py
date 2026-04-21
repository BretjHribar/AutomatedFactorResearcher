"""
seed_alphas_ib.py — Seed Alpha Library for IB Closing Auction Strategy.

Price-action delay-0 alphas optimized for the closing auction.
These are evaluated fee-free to discover raw signal quality,
then combined at the portfolio level with IB commission model.

All alphas target TOP2000-TOP3000 universe with sector neutralization.

Categories:
    - candle:    Intraday candle body/wick features
    - reversal:  Short-term mean reversion signals
    - momentum:  Short-term momentum/continuation
    - volume:    Volume-price interaction signals
    - volatility: Volatility-based features
    - composite: Multi-factor combinations
"""

from __future__ import annotations

# ============================================================================
# SEED ALPHA EXPRESSIONS
# ============================================================================

SEED_ALPHAS: list[dict[str, str]] = [
    # === CANDLE BODY / WICK FEATURES ===
    {
        "expr": "(close - open) / (high - low + 0.001)",
        "name": "candle_body_ratio",
        "category": "candle",
        "reasoning": "Body-to-range ratio. Large body = strong directional conviction. "
                     "In small-caps, extreme body ratios tend to reverse at delay-0.",
    },
    {
        "expr": "rank((close - open) / (high - low + 0.001))",
        "name": "candle_body_rank",
        "category": "candle",
        "reasoning": "Cross-sectional rank of body ratio. Captures relative candle "
                     "strength across the universe.",
    },
    {
        "expr": "rank((low - close) / (high - low + 0.001))",
        "name": "close_near_low",
        "category": "candle",
        "reasoning": "Close near intraday low suggests selling exhaustion. "
                     "Delay-0 reversal signal. [SAVED #1 — SR +4.73]",
    },
    {
        "expr": "rank((high - close) / (high - low + 0.001))",
        "name": "upper_wick_ratio",
        "category": "candle",
        "reasoning": "Large upper wick = rejection at highs. Bearish signal. "
                     "Rank-transformed for cross-sectional use.",
    },
    {
        "expr": "rank((open - low) / (high - low + 0.001))",
        "name": "lower_wick_ratio",
        "category": "candle",
        "reasoning": "Large lower wick = buying support. Bullish candle pattern.",
    },
    {
        "expr": "rank((high - low) / close)",
        "name": "daily_range_pct",
        "category": "candle",
        "reasoning": "Daily range as percentage of close. High range = high uncertainty, "
                     "often followed by mean reversion.",
    },

    # === VWAP DEVIATION ===
    {
        "expr": "rank(-(close - vwap) / close)",
        "name": "vwap_deviation",
        "category": "reversal",
        "reasoning": "Close far above VWAP = overbought intraday. Negative sign = "
                     "buy cheaply at close when price is below VWAP.",
    },

    # === SHORT-TERM REVERSAL ===
    {
        "expr": "rank(-ts_delta(close, 1))",
        "name": "reversal_1d",
        "category": "reversal",
        "reasoning": "1-day reversal: today's losers are tomorrow's winners. "
                     "Classic short-term mean reversion. [SAVED #2 — SR +1.82]",
    },
    {
        "expr": "rank(-ts_delta(close, 3))",
        "name": "reversal_3d",
        "category": "reversal",
        "reasoning": "3-day reversal: slightly longer horizon mean reversion. "
                     "Complementary to 1-day. [SAVED #3 — SR +1.96]",
    },
    {
        "expr": "rank(ts_delta(close, 5))",
        "name": "momentum_5d",
        "category": "momentum",
        "reasoning": "5-day momentum: winners keep winning at the weekly horizon. "
                     "Delay-0 captures the continuation before close.",
    },

    # === VOLUME-PRICE INTERACTION ===
    {
        "expr": "rank(ts_corr(close, volume, 10))",
        "name": "price_volume_corr_10d",
        "category": "volume",
        "reasoning": "10-day price-volume correlation. Positive = volume confirms "
                     "price trend. Negative = divergence warning.",
    },
    {
        "expr": "rank(-ts_delta(volume, 5)) * rank(ts_delta(close, 5))",
        "name": "volume_price_divergence",
        "category": "volume",
        "reasoning": "Volume declining + price rising = hidden strength. "
                     "Composite signal combining volume and price momentum.",
    },

    # === VOLATILITY FEATURES ===
    {
        "expr": "rank((close - ts_mean(close, 5)) / (ts_std_dev(close, 5) + 0.001))",
        "name": "bollinger_z_5d",
        "category": "volatility",
        "reasoning": "5-day Bollinger Z-score. Extreme Z = likely reversion. "
                     "Positive Z = overbought, negative = oversold.",
    },
    {
        "expr": "rank(-ts_std_dev(close, 10))",
        "name": "low_vol_premium",
        "category": "volatility",
        "reasoning": "Low volatility premium: less volatile stocks outperform "
                     "risk-adjusted. Well-documented anomaly. [SAVED #4 — SR +1.66]",
    },

    # === MULTI-DAY CANDLE FEATURES ===
    {
        "expr": "rank(ts_mean((close - open) / (high - low + 0.001), 5))",
        "name": "avg_body_ratio_5d",
        "category": "candle",
        "reasoning": "5-day average candle body ratio. Persistent directional candles "
                     "indicate trend strength.",
    },
    {
        "expr": "rank(ts_rank(close, 20))",
        "name": "price_rank_20d",
        "category": "momentum",
        "reasoning": "20-day percentile rank of current close within its own history. "
                     "High rank = near 20-day high = momentum.",
    },
    {
        "expr": "rank(-ts_rank(volume, 20))",
        "name": "volume_rank_reversal",
        "category": "volume",
        "reasoning": "Negative volume rank: low relative volume = accumulation phase. "
                     "Contrarian volume signal.",
    },

    # === COMPOSITE SIGNALS ===
    {
        "expr": "rank((close - open) / (high - low + 0.001)) * rank(-ts_delta(close, 3))",
        "name": "body_x_reversal",
        "category": "composite",
        "reasoning": "Interaction of candle body strength with 3-day reversal. "
                     "Strong bullish candle after 3 days of decline = buy.",
    },
    {
        "expr": "rank(ts_corr(close, volume, 20)) * rank(-ts_delta(close, 5))",
        "name": "corr_x_reversal",
        "category": "composite",
        "reasoning": "Price-volume correlation confirms when 5-day decline is "
                     "a buying opportunity vs genuine breakdown.",
    },

    # =========================================================================
    # WAVE 2: 10 HIGH-CONVICTION ORTHOGONAL CANDIDATES (Target: IS SR > 4.0)
    # =========================================================================

    # 1. MULTI-DAY RANGE EXHAUSTION
    # Theory: Close near the 3-day low is a stronger exhaustion signal than
    # single-day close-near-low because it requires sustained multi-day selling.
    # Structurally different from #1 (3-day range context, not today-only H-L).
    {
        "expr": "rank(-(close - ts_min(low, 3)) / (ts_max(high, 3) - ts_min(low, 3) + 0.001))",
        "name": "w2_3d_range_exhaustion",
        "category": "candle",
        "reasoning": "Close near the 3-day low: requires sustained multi-day selling to "
                     "score high. Stronger exhaustion than single-day close_near_low. "
                     "Uses 3-day H-L range as denominator so it's not dominated by "
                     "today's intraday volatility.",
    },

    # 1B. DECAYED 3-DAY RANGE EXHAUSTION (solves turnover)
    # The raw 3d_range_exhaustion hit SR +3.39 but TO=0.93 (gate <= 0.50).
    # Applying Decay_exp smoothing reduces turnover by ~50% by dampening daily flips.
    {
        "expr": "rank(Decay_exp(-(close - ts_min(low, 3)) / (ts_max(high, 3) - ts_min(low, 3) + 0.001), 0.7))",
        "name": "w2_3d_range_exhaustion_decay",
        "category": "candle",
        "reasoning": "Decayed version of 3d_range_exhaustion (SR +3.39, failed TO gate at 0.93). "
                     "Decay_exp(0.7) = halflife ~2 days smoothing that reduces daily signal "
                     "flips, cutting turnover while preserving the exhaustion pattern.",
    },

    # Theory: Stocks that open BELOW prior close have absorbed overnight sellers.
    # MOC buyers at today's close step in because the gap creates a mispricing.
    # Uses open + prior close - completely orthogonal to all saved signals.
    {
        "expr": "rank(delay(close, 1) - open)",
        "name": "w2_gap_down_reversal",
        "category": "reversal",
        "reasoning": "Gap-down reversal: prior_close - open. High rank = stock gapped "
                     "down at open. Small-cap gap-downs are driven by retail panic and "
                     "tend to mean-revert by the end of the following day. Captures "
                     "OVERNIGHT information not in any close-to-close signal.",
    },

    # 3. INTRADAY SELL EXHAUSTION
    # Theory: Stock fell from open to close (open > close = down-day body).
    # Intraday sellers are exhausted; MOC buyers absorb the residual sell flow.
    # Mathematically distinct from close_near_low: uses open vs close (body direction),
    # not close vs high-low range (candle position).
    {
        "expr": "rank(open - close)",
        "name": "w2_intraday_sell_exhaustion",
        "category": "candle",
        "reasoning": "Open minus close. Positive when stock fell all day — pure intraday "
                     "sell pressure. MOC buyer absorbs final sellers at auction. Different "
                     "from close_near_low: body direction (up/down) vs candle wick position.",
    },

    # 4. RANGE-AMPLIFIED EXHAUSTION
    # Theory: close_near_low is a stronger signal on HIGH-RANGE days because
    # high-range = lots of trading activity = more informed selling.
    # Filter out weak-candle days by requiring unusual range expansion.
    {
        "expr": "rank((high - low) / ts_mean(high - low, 20) * (low - close) / (high - low + 0.001))",
        "name": "w2_range_amplified_exhaustion",
        "category": "candle",
        "reasoning": "Range-expansion × close-near-low. Multiplies the exhaustion signal "
                     "by how unusual today's range was relative to 20-day average. "
                     "High-volume, high-range down-close days = strongest reversals. "
                     "Filters out mundane close-near-low signals on quiet days.",
    },

    # 5. PRIOR-DAY UPPER WICK (YESTERDAY'S EXHAUSTION)
    # Theory: When a stock closed near its HIGH yesterday, there is overhead supply
    # today. Sellers who bought at yesterday's high exit near break-even.
    # Uses only prior-day candle data — zero overlap with today's candle signals.
    {
        "expr": "rank(-(delay(close, 1) - delay(low, 1)) / (delay(high, 1) - delay(low, 1) + 0.001))",
        "name": "w2_prior_day_high_exhaustion",
        "category": "candle",
        "reasoning": "Yesterday's close near yesterday's HIGH. Overhead supply today "
                     "from lateday buyers exiting. Predicts a weak open and close TODAY. "
                     "Entirely lagged (ts_lag) — orthogonal to any signal using today's OHLC.",
    },

    # 6. PANIC SELL (1D REVERSAL × HIGH RELATIVE VOLUME)
    # Theory: A 1-day price drop on UNUSUALLY HIGH volume = panic/forced selling.
    # Normal drops can be rational; volume-amplified drops are more likely irrational.
    # Correlation with reversal_1d ~0.5 because same price direction but very different
    # cross-sectional ordering (stock A: -5% on 3x vol ranks much higher than B: -5% on 0.5x vol).
    {
        "expr": "rank(-ts_delta(close, 1) * ts_rank(volume, 20))",
        "name": "w2_panic_sell_reversal",
        "category": "reversal",
        "reasoning": "1-day reversal weighted by relative volume. ts_rank(volume,20) "
                     "gives 0-1 percentile of today's volume in 20-day context. "
                     "Big drops on abnormally high volume = institutional forced selling = "
                     "strongest reversal opportunities in the auction.",
    },

    # 7. NEGATIVE 5-DAY BOLLINGER Z (NORMALIZED REVERSAL)
    # Theory: bollinger_z_5d (positive Z) had SR = -1.78 with market neutral,
    # so its inverse (negative Z = below 5-day mean in volatility-adjusted units)
    # should have SR ~+1.78 at minimum. Better than raw reversal_3d because it
    # normalizes by local volatility — a -5% move in a 1%-vol stock ranks higher
    # than -5% in a 5%-vol stock. Genuinely different cross-sectional ordering.
    {
        "expr": "rank((ts_mean(close, 5) - close) / (ts_std_dev(close, 5) + 0.001))",
        "name": "w2_neg_bollinger_z_5d",
        "category": "reversal",
        "reasoning": "5-day Bollinger Z, inverted. Buys stocks more than 1 std below "
                     "their 5-day mean. Volatility-normalized reversal: a given raw "
                     "return scores higher for low-vol stocks, distinguishing routine "
                     "pullbacks (high-vol stocks) from genuine statistical extremes.",
    },

    # 8. RETURN-VOLATILITY LOW-VOL PREMIUM
    # Theory: low_vol_premium uses PRICE volatility (std of close levels),
    # which is contaminated by the price level itself. Return volatility
    # (std of daily pct changes) is scale-free and measures true risk exposure.
    # Same anomaly, cleaner measurement. Low-vol stocks outperform after controlling
    # for market (well-documented: Frazzini & Pedersen 2014).
    {
        "expr": "rank(-ts_std_dev(returns, 20))",
        "name": "w2_return_vol_premium",
        "category": "volatility",
        "reasoning": "Low realized-return volatility premium. Uses log returns std "
                     "instead of price std — scale-free, not confounded by share price "
                     "level. 20-day window captures stable volatility regime. "
                     "Orthogonal to low_vol_premium (#4) via different denominator.",
    },

    # 9. EARNINGS YIELD (VALUE SIGNAL)
    # Theory: Value factor in small-caps is well-documented (Fama & French 1992).
    # Earnings yield (E/P) is quarterly-rebalanced, so turnover ~0.1 — fee-immune.
    # Zero correlation expected with any price-action signal above.
    # Will fail SR > 4 gate if pure value doesn't reach that bar — that's fine.
    {
        "expr": "rank(earnings_yield)",
        "name": "w2_earnings_yield",
        "category": "fundamental",
        "reasoning": "Earnings yield (earnings/price) value factor. Quarterly data, "
                     "low turnover ~0.1. Known anomaly in small caps. Tests whether "
                     "a fundamental anchor improves portfolio when combined with "
                     "price-action signals. Orthogonal by construction.",
    },

    # 10. BOOK-TO-MARKET VALUE
    # Theory: Classic value factor (Fama & French HML). In small-cap band,
    # historically strong (smallcap value premium). Different from earnings_yield:
    # captures asset cheapness not earnings flow, better for capital-intensive firms.
    # Quarterly data, very low turnover, zero correlation with technicals.
    {
        "expr": "rank(book_to_market)",
        "name": "w2_book_to_market",
        "category": "fundamental",
        "reasoning": "Book-to-market ratio: classic value signal. Low P/B stocks "
                     "outperform in small-cap universe. Quarterly rebalancing means "
                     "turnover ~0.05-0.10. If SR > 4, it provides fundamentally "
                     "orthogonal diversification to intraday candle signals.",
    },

    # =========================================================================
    # WAVE 2B: CROSS-SECTIONAL ZSCORE VARIANTS
    # zscore() differs from rank() in that it preserves signal magnitude —
    # a stock at 5 std below cross-sectional mean gets a MUCH higher weight
    # than one at 2 std, whereas rank() would compress them both near 1.0.
    # For reversal signals driven by extreme events (auctions, pan-ic selling),
    # this magnitude preservation can amplify the signal significantly.
    # =========================================================================

    # ZS-1: ZSCORE CLOSE-NEAR-LOW (the proven signal, alternative normalization)
    # Theory: same intraday exhaustion signal as close_near_low (#1, SR +4.73),
    # but now the stock at the very bottom of its range (e.g., close = low)
    # scores 3-5x higher than a merely weak close. This concentrates position
    # in the most extreme exhaustion cases rather than treating all below-mean
    # stocks equally. May increase SR by better sizing to the signal strength.
    {
        "expr": "zscore((low - close) / (high - low + 0.001))",
        "name": "w2z_close_near_low_zs",
        "category": "candle",
        "reasoning": "Cross-sectional zscore of the proven close_near_low signal. "
                     "Where rank() gives all stocks a value in [0,1], zscore() "
                     "lets a stock with close=low score 5+ std above mean, "
                     "concentrating weight on the most extreme exhaustion cases. "
                     "Expect different PnL profile: fewer but stronger positions.",
    },

    # ZS-2: ZSCORE INTRADAY SELL EXHAUSTION (open - close)
    # Theory: open-close body gives a continuous signal. zscore() means a stock
    # that fell $5 from open gets proportionally more weight than one that fell $0.50.
    # This is more natural than rank() for dollar-amount signals (not %-based).
    {
        "expr": "zscore(open - close)",
        "name": "w2z_intraday_sell_zs",
        "category": "candle",
        "reasoning": "Cross-sectional zscore of intraday sell pressure (open-close). "
                     "Dollar-amount signal where zscore is more natural than rank: "
                     "a $5 drop is proportionally emphasized over a $0.50 drop. "
                     "In small-cap universe, extreme intraday drops often reflect "
                     "uninformed retail selling that reverses at the MOC auction.",
    },

    # ZS-3: ZSCORE GAP-DOWN
    # Theory: gap-down magnitude matters. A 5% gap down is a much stronger
    # reversal signal than a 0.5% gap. rank() loses this. zscore() preserves it.
    {
        "expr": "zscore(delay(close, 1) - open)",
        "name": "w2z_gap_down_zs",
        "category": "reversal",
        "reasoning": "Cross-sectional zscore of overnight gap-down size. "
                     "rank(ts_lag(close,1)-open) treats a 0.1% and 5% gap equally "
                     "if they're at the same cross-sectional percentile. "
                     "zscore() weights the 5% gap ~50x heavier, concentrating "
                     "positions in true panic gaps which have the strongest reversals.",
    },
]


def get_seed_expressions() -> list[str]:
    """Return just the expression strings for batch evaluation."""
    return [a["expr"] for a in SEED_ALPHAS]


def get_seed_alphas() -> list[dict[str, str]]:
    """Return full seed alpha definitions with metadata."""
    return SEED_ALPHAS.copy()


if __name__ == "__main__":
    print(f"Seed Alpha Library: {len(SEED_ALPHAS)} alphas\n")
    for i, alpha in enumerate(SEED_ALPHAS, 1):
        print(f"  {i:2d}. [{alpha['category']:12s}] {alpha['name']}")
        print(f"      Expr: {alpha['expr']}")
        print()
