"""
ib_cost_model.py — IBKR Pro Tiered Commission Model for Closing Auction Strategy.

This module provides the ONLY execution cost model for the IB closing auction
trading system. It is applied ONLY at the portfolio combination / live simulation
stage — NOT during individual alpha research (which is fee-free per WQ standard).

Key principle: The NYSE closing auction is a single-price auction. All participants
receive the official closing price regardless of order size. There is ZERO spread.
The only costs are IB's commission and regulatory fees.

IBKR Pro Tiered (US Equities):
    <= 300,000 shares/month:    $0.0035/share ($0.35 min/order)
    300,001 - 3,000,000:        $0.0020/share
    3,000,001 - 20,000,000:     $0.0015/share
    > 20,000,000:               $0.0010/share
    Max per order: 1% of trade value

Regulatory fees (pass-through, applied to both tiers):
    SEC fee:   ~$0.0000278 per $ of sell value
    FINRA TAF: $0.000145 per share sold (capped at $7.27/trade)
    Exchange:  ~$0.0003 per share for MOC on NYSE
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class IBCostResult:
    """Result of cost calculation for a single trade."""
    commission: float       # IB commission
    regulatory_fees: float  # SEC + FINRA + exchange fees
    total_cost: float       # commission + regulatory
    cost_bps: float         # total cost in basis points of trade value


# IBKR Pro Tiered rate schedule
TIERED_RATES = [
    (300_000,      0.0035),  # <= 300K shares/month
    (3_000_000,    0.0020),  # 300K - 3M
    (20_000_000,   0.0015),  # 3M - 20M
    (100_000_000,  0.0010),  # 20M - 100M
    (float("inf"), 0.0005),  # > 100M
]

# Regulatory fee rates
SEC_FEE_RATE = 0.0000278     # per $ sold (SEC fee, sell side only)
FINRA_TAF_RATE = 0.000145    # per share sold (FINRA TAF, sell side only)
FINRA_TAF_CAP = 7.27         # max FINRA TAF per trade
EXCHANGE_FEE_RATE = 0.0003   # per share (both sides, MOC on NYSE)


def ib_commission(
    shares: int,
    price: float,
    monthly_shares: int = 0,
) -> float:
    """
    Calculate IBKR Pro Tiered commission for a US equity trade.

    Args:
        shares: Number of shares traded.
        price: Price per share.
        monthly_shares: Cumulative monthly share volume (for tier determination).
                       If 0, uses the lowest tier ($0.0035/share).

    Returns:
        Commission in dollars.
    """
    if shares <= 0 or price <= 0:
        return 0.0

    # Determine per-share rate based on monthly volume tier
    per_share_rate = TIERED_RATES[0][1]  # default: $0.0035
    for threshold, rate in TIERED_RATES:
        if monthly_shares <= threshold:
            per_share_rate = rate
            break

    # Calculate: per-share rate, with $0.35 min and 1% of trade value max
    trade_value = shares * price
    calculated = shares * per_share_rate
    commission = max(0.35, min(calculated, 0.01 * trade_value))

    return round(commission, 4)


def regulatory_fees(
    shares: int,
    price: float,
    is_sell: bool = False,
) -> float:
    """
    Calculate regulatory pass-through fees for a US equity trade.

    SEC fee and FINRA TAF only apply to sell-side trades.
    Exchange fee applies to both sides.

    Args:
        shares: Number of shares.
        price: Price per share.
        is_sell: Whether this is a sell order.

    Returns:
        Regulatory fees in dollars.
    """
    if shares <= 0 or price <= 0:
        return 0.0

    trade_value = shares * price
    fees = 0.0

    # Exchange fee (both sides)
    fees += shares * EXCHANGE_FEE_RATE

    if is_sell:
        # SEC fee (sell side only)
        fees += trade_value * SEC_FEE_RATE
        # FINRA TAF (sell side only, capped)
        fees += min(shares * FINRA_TAF_RATE, FINRA_TAF_CAP)

    return round(fees, 4)


def total_trade_cost(
    shares: int,
    price: float,
    is_sell: bool = False,
    monthly_shares: int = 0,
) -> IBCostResult:
    """
    Calculate total trading cost for a single trade.

    This is the COMPLETE cost model for MOC (Market-on-Close) orders.
    There is NO spread component because the closing auction is a single-price
    auction — all participants receive the same official closing price.

    Args:
        shares: Number of shares traded.
        price: Price per share.
        is_sell: Whether this is a sell order.
        monthly_shares: Cumulative monthly volume for tier determination.

    Returns:
        IBCostResult with itemized and total costs.
    """
    comm = ib_commission(shares, price, monthly_shares)
    reg = regulatory_fees(shares, price, is_sell)
    total = comm + reg
    trade_value = shares * price
    bps = (total / trade_value * 10000) if trade_value > 0 else 0

    return IBCostResult(
        commission=comm,
        regulatory_fees=reg,
        total_cost=total,
        cost_bps=round(bps, 2),
    )


def estimate_daily_costs(
    n_positions: int,
    avg_position_size: float,
    avg_price: float,
    daily_turnover: float,
    monthly_shares: int = 0,
) -> dict:
    """
    Estimate daily trading costs for the portfolio.

    Args:
        n_positions: Total number of positions (long + short).
        avg_position_size: Average dollar size per position.
        avg_price: Average share price in the universe.
        daily_turnover: Fraction of portfolio traded daily (e.g. 0.25 = 25%).
        monthly_shares: Monthly share volume for tier determination.

    Returns:
        Dict with daily cost estimates.
    """
    avg_shares = int(avg_position_size / avg_price) if avg_price > 0 else 0
    n_trades = int(n_positions * daily_turnover * 2)  # buy + sell

    # Half of trades are sells (for regulatory fee estimation)
    n_sells = n_trades // 2
    n_buys = n_trades - n_sells

    total_commission = 0.0
    total_reg = 0.0

    for _ in range(n_buys):
        r = total_trade_cost(avg_shares, avg_price, is_sell=False, monthly_shares=monthly_shares)
        total_commission += r.commission
        total_reg += r.regulatory_fees

    for _ in range(n_sells):
        r = total_trade_cost(avg_shares, avg_price, is_sell=True, monthly_shares=monthly_shares)
        total_commission += r.commission
        total_reg += r.regulatory_fees

    total = total_commission + total_reg
    book_size = n_positions * avg_position_size

    return {
        "n_trades": n_trades,
        "avg_shares_per_trade": avg_shares,
        "daily_commission": round(total_commission, 2),
        "daily_regulatory": round(total_reg, 2),
        "daily_total": round(total, 2),
        "daily_bps": round(total / book_size * 10000, 2) if book_size > 0 else 0,
        "annual_total": round(total * 252, 2),
        "annual_pct_of_book": round(total * 252 / book_size * 100, 2) if book_size > 0 else 0,
    }


def portfolio_cost_bps(
    positions_df,
    prices_df,
    prev_positions_df=None,
    monthly_shares: int = 0,
) -> float:
    """
    Calculate total portfolio rebalance cost in basis points.

    This is for use in the vectorized sim when running portfolio-level
    backtests with realistic IB costs.

    Args:
        positions_df: Current target positions (dollar amounts, dates x tickers).
        prices_df: Price matrix (dates x tickers).
        prev_positions_df: Previous positions (for computing trades).
        monthly_shares: Monthly volume for tier determination.

    Returns:
        Total cost in bps of gross book value.
    """
    if prev_positions_df is None:
        return 0.0

    # Compute trades (change in position)
    trades = positions_df - prev_positions_df
    gross_book = positions_df.abs().sum()

    if gross_book == 0:
        return 0.0

    total_cost = 0.0
    for ticker in trades.index:
        trade_dollars = abs(trades[ticker])
        if trade_dollars < 1:  # skip tiny trades
            continue
        price = prices_df.get(ticker, 25.0)  # fallback to $25
        shares = max(1, int(trade_dollars / price))
        is_sell = trades[ticker] < 0
        cost = total_trade_cost(shares, price, is_sell, monthly_shares)
        total_cost += cost.total_cost

    return total_cost / gross_book * 10000
