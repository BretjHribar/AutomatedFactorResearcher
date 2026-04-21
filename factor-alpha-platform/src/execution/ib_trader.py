"""
ib_trader.py — IB Closing Auction MOC Order Execution Module.

Submits Market-on-Close (MOC) orders via ib_insync at 3:45 PM ET
for the closing auction strategy. All fills occur at the official
closing price (single-price auction, zero spread).

Daily schedule (ET):
    3:30 PM — Fetch intraday snapshot from IB TWS
    3:35 PM — Compute delay-0 alpha signals
    3:40 PM — Generate target portfolio (sector-neutral, 4x gross)
    3:45 PM — Submit MOC orders (5-min buffer before 3:50 cutoff)
    3:50 PM — NYSE cutoff (no modifications allowed)
    4:00 PM — Closing auction executes
    4:05 PM — Reconcile fills, log to DB

Requirements:
    pip install ib_insync
    IB Gateway or TWS must be running with API enabled
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# IB connection defaults
IB_HOST = "127.0.0.1"
IB_PORT = 7497          # 7497 = TWS Paper, 7496 = TWS Live, 4002 = Gateway Paper, 4001 = Gateway Live
IB_CLIENT_ID = 10

# Portfolio parameters matching implementation plan
EQUITY = 110_000
GROSS_LEVERAGE = 4.0
GROSS_EXPOSURE = EQUITY * GROSS_LEVERAGE  # $440,000
PER_SIDE = GROSS_EXPOSURE / 2              # $220,000

# Position constraints
MIN_POSITIONS = 80
MAX_POSITIONS = 150
MIN_POSITION_DOLLARS = 500     # Skip if too small (commission > 7 bps)
MAX_POSITION_WEIGHT = 0.05     # 5% of per-side allocation

# Timing (ET)
SIGNAL_COMPUTE_TIME = time(15, 35)  # 3:35 PM
MOC_SUBMIT_TIME = time(15, 45)      # 3:45 PM (5 min buffer)
NYSE_CUTOFF_TIME = time(15, 50)     # 3:50 PM (hard cutoff)

# Database for tracking
DB_PATH = "data/ib_alphas.db"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TargetPosition:
    """A target position for the closing auction."""
    symbol: str
    shares: int           # positive = long, negative = short
    side: str             # "BUY" or "SELL"
    notional: float       # dollar notional
    weight: float         # portfolio weight

    @property
    def is_long(self) -> bool:
        return self.shares > 0


@dataclass
class OrderResult:
    """Result of a single MOC order submission."""
    symbol: str
    order_id: int
    shares: int
    side: str
    status: str           # "Submitted", "Filled", "Cancelled", "Error"
    fill_price: float = 0.0
    fill_time: str = ""
    commission: float = 0.0
    error: str = ""


@dataclass
class DailyTradeReport:
    """Summary of daily trading activity."""
    date: str
    n_orders: int = 0
    n_fills: int = 0
    n_errors: int = 0
    total_commission: float = 0.0
    gross_traded: float = 0.0
    long_positions: int = 0
    short_positions: int = 0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    orders: list = field(default_factory=list)


# ============================================================================
# PORTFOLIO CONSTRUCTION
# ============================================================================

def compute_target_portfolio(
    alpha_weights: pd.Series,
    prices: pd.Series,
    per_side: float = PER_SIDE,
    min_position: float = MIN_POSITION_DOLLARS,
    max_weight: float = MAX_POSITION_WEIGHT,
) -> list[TargetPosition]:
    """
    Convert normalized alpha weights to target share positions.

    Args:
        alpha_weights: Sector-neutralized, normalized weights (sum abs = 1).
                       Positive = long, negative = short.
        prices: Current prices per symbol.
        per_side: Dollar allocation per side (long/short).
        min_position: Minimum dollar position size (skip if below).
        max_weight: Maximum weight per position.

    Returns:
        List of TargetPosition objects.
    """
    # Clip weights
    weights = alpha_weights.clip(-max_weight, max_weight)

    # Separate longs and shorts
    long_weights = weights[weights > 0]
    short_weights = weights[weights < 0]

    # Normalize each side to sum to 1
    if long_weights.sum() > 0:
        long_weights = long_weights / long_weights.sum()
    if short_weights.sum() < 0:
        short_weights = short_weights / short_weights.abs().sum()

    positions = []

    # Longs
    for sym, wt in long_weights.items():
        notional = wt * per_side
        if notional < min_position:
            continue
        price = prices.get(sym, 0)
        if price <= 0:
            continue
        shares = int(round(notional / price))  # Round to nearest whole share
        if shares <= 0:
            continue
        positions.append(TargetPosition(
            symbol=sym,
            shares=shares,
            side="BUY",
            notional=shares * price,
            weight=wt,
        ))

    # Shorts
    for sym, wt in short_weights.items():
        notional = abs(wt) * per_side
        if notional < min_position:
            continue
        price = prices.get(sym, 0)
        if price <= 0:
            continue
        shares = int(round(notional / price))
        if shares <= 0:
            continue
        positions.append(TargetPosition(
            symbol=sym,
            shares=-shares,  # negative for shorts
            side="SELL",
            notional=shares * price,
            weight=wt,
        ))

    return positions


def compute_trades(
    target: list[TargetPosition],
    current: dict[str, int],
) -> list[TargetPosition]:
    """
    Compute trades needed to move from current to target positions.

    Args:
        target: Target positions from portfolio construction.
        current: Current positions {symbol: shares} from IB.

    Returns:
        List of TargetPosition objects representing trades (not positions).
    """
    target_map = {p.symbol: p.shares for p in target}
    all_symbols = set(target_map.keys()) | set(current.keys())

    trades = []
    for sym in all_symbols:
        target_shares = target_map.get(sym, 0)
        current_shares = current.get(sym, 0)
        delta = target_shares - current_shares

        if delta == 0:
            continue

        side = "BUY" if delta > 0 else "SELL"
        trades.append(TargetPosition(
            symbol=sym,
            shares=delta,
            side=side,
            notional=0,  # Will be filled at execution
            weight=0,
        ))

    return trades


# ============================================================================
# IB CONNECTION & ORDER SUBMISSION
# ============================================================================

class IBTrader:
    """
    Interactive Brokers MOC order trader.

    Uses ib_insync for API communication.
    Supports paper trading mode for validation.
    """

    def __init__(
        self,
        host: str = IB_HOST,
        port: int = IB_PORT,
        client_id: int = IB_CLIENT_ID,
        paper: bool = True,
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.paper = paper
        self.ib = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to IB TWS/Gateway."""
        try:
            from ib_insync import IB
            self.ib = IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self._connected = True
            account = "Paper" if self.paper else "Live"
            logger.info(f"Connected to IB ({account}) at {self.host}:{self.port}")
            return True
        except ImportError:
            logger.error("ib_insync not installed. Run: pip install ib_insync")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            return False

    def disconnect(self):
        """Disconnect from IB."""
        if self.ib and self._connected:
            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IB")

    def get_current_positions(self) -> dict[str, int]:
        """Get current positions from IB account."""
        if not self._connected:
            return {}

        positions = {}
        for pos in self.ib.positions():
            sym = pos.contract.symbol
            shares = int(pos.position)
            if shares != 0:
                positions[sym] = shares
        return positions

    def get_current_prices(self, symbols: list[str]) -> pd.Series:
        """Get current market prices for symbols."""
        if not self._connected:
            return pd.Series(dtype=float)

        from ib_insync import Stock
        prices = {}
        for sym in symbols:
            try:
                contract = Stock(sym, "SMART", "USD")
                self.ib.qualifyContracts(contract)
                ticker = self.ib.reqMktData(contract, snapshot=True)
                self.ib.sleep(0.5)  # Wait for data
                if ticker.last and ticker.last > 0:
                    prices[sym] = ticker.last
                elif ticker.close and ticker.close > 0:
                    prices[sym] = ticker.close
            except Exception as e:
                logger.warning(f"Price fetch failed for {sym}: {e}")

        return pd.Series(prices)

    def submit_moc_orders(
        self,
        trades: list[TargetPosition],
    ) -> list[OrderResult]:
        """
        Submit Market-on-Close orders for all trades.

        MOC orders execute at the official closing price in the
        NYSE closing auction. All participants get the same price.

        Args:
            trades: List of trades to execute.

        Returns:
            List of OrderResult objects.
        """
        if not self._connected:
            logger.error("Not connected to IB")
            return []

        from ib_insync import Stock, MarketOrder

        results = []
        for trade in trades:
            try:
                contract = Stock(trade.symbol, "SMART", "USD")
                self.ib.qualifyContracts(contract)

                # Create MOC order
                action = "BUY" if trade.shares > 0 else "SELL"
                qty = abs(trade.shares)

                order = MarketOrder(action, qty)
                order.tif = "DAY"
                order.orderType = "MOC"  # Market-on-Close

                ib_trade = self.ib.placeOrder(contract, order)

                results.append(OrderResult(
                    symbol=trade.symbol,
                    order_id=ib_trade.order.orderId,
                    shares=trade.shares,
                    side=action,
                    status="Submitted",
                ))
                logger.info(f"  MOC {action} {qty} {trade.symbol} submitted")

            except Exception as e:
                results.append(OrderResult(
                    symbol=trade.symbol,
                    order_id=0,
                    shares=trade.shares,
                    side=trade.side,
                    status="Error",
                    error=str(e),
                ))
                logger.error(f"  Order failed for {trade.symbol}: {e}")

        return results

    def get_account_summary(self) -> dict:
        """Get account summary from IB."""
        if not self._connected:
            return {}

        summary = {}
        for item in self.ib.accountSummary():
            summary[item.tag] = item.value
        return summary


# ============================================================================
# TRADE LOGGING
# ============================================================================

def log_daily_report(report: DailyTradeReport, db_path: str = DB_PATH):
    """Log daily trade report to SQLite."""
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE IF NOT EXISTS daily_trades (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        date        TEXT NOT NULL,
        n_orders    INTEGER,
        n_fills     INTEGER,
        n_errors    INTEGER,
        total_commission REAL,
        gross_traded    REAL,
        long_positions  INTEGER,
        short_positions INTEGER,
        gross_exposure  REAL,
        net_exposure    REAL,
        created_at  TEXT DEFAULT (datetime('now'))
    )""")
    conn.execute("""INSERT INTO daily_trades
        (date, n_orders, n_fills, n_errors, total_commission,
         gross_traded, long_positions, short_positions,
         gross_exposure, net_exposure)
        VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (report.date, report.n_orders, report.n_fills, report.n_errors,
         report.total_commission, report.gross_traded,
         report.long_positions, report.short_positions,
         report.gross_exposure, report.net_exposure))
    conn.commit()
    conn.close()
