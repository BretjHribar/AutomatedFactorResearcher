"""
prod/moc_trader.py — Production MOC Trading System for IBKR

Architecture:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Daily Pipeline (3:30 PM ET)                                          │
  │  1. Load FMP matrices + classifications                               │
  │  2. Evaluate 31 alpha signals                                         │
  │  3. Pre-neutralize each alpha by subindustry (group_demean)           │
  │  4. Equal-weight combine → process_signal → target weights            │
  │  5. Check short borrow availability via IB                            │
  │  6. Convert to target shares → compute order diffs                    │
  │  7. Pre-trade risk checks (concentration, GMV, staleness)             │
  │  8. Submit MOC orders                                                 │
  │  9. Log everything (trades, fills, performance, borrow)               │
  └─────────────────────────────────────────────────────────────────────────┘

Modes:
  python prod/moc_trader.py                   # dry-run (default)
  python prod/moc_trader.py --live            # paper trading (port 7497)
  python prod/moc_trader.py --live --port 7496  # LIVE TRADING
  python prod/moc_trader.py --check-borrow    # borrow availability only

Designed for scale — this will grow into a multi-strategy hedge fund.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sqlite3
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

# Local imports
from live_bar import ENABLE_IB_LIVE_VWAP, append_live_bar

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# CONFIGURATION — loaded from strategy.json
# ============================================================================

CONFIG_PATH = Path(__file__).parent / "config" / "strategy.json"

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)

CFG = load_config()

def _cfg_time(value: str) -> dt.time:
    return dt.time(*[int(x) for x in value.split(":")])


def now_eastern() -> dt.datetime:
    try:
        return dt.datetime.now(ZoneInfo("America/New_York"))
    except ZoneInfoNotFoundError:
        return pd.Timestamp.now(tz="America/New_York").to_pydatetime()


# Extract commonly used values
UNIVERSE       = CFG["strategy"]["universe"]
NEUTRALIZE     = CFG["strategy"]["neutralization"]
PRE_NEUTRALIZE = CFG["strategy"]["pre_neutralize_alphas"]
MIN_ALPHA_SR   = float(CFG["strategy"].get("min_alpha_sharpe", 5.0))
MAX_WEIGHT     = CFG["strategy"]["max_stock_weight"]
BOOKSIZE       = CFG["account"]["booksize"]
TARGET_GMV     = CFG["account"]["target_gmv"]
ACCOUNT_EQUITY = CFG["account"]["equity_seed"]
MAX_DD_HALT    = CFG["account"]["max_drawdown_halt_pct"]
MIN_ORDER_VAL  = CFG["execution"].get("min_order_value", 0)  # $0 = no filter

IB_HOST        = os.environ.get("IB_HOST", CFG["ibkr"]["host"])
IB_PORT_PAPER  = int(os.environ.get("IB_PORT_PAPER", os.environ.get("IB_PORT", CFG["ibkr"]["port_paper"])))
IB_PORT_LIVE   = int(os.environ.get("IB_PORT_LIVE", CFG["ibkr"]["port_live"]))
IB_CLIENT_ID   = int(os.environ.get("IB_CLIENT_ID", CFG["ibkr"]["client_id"]))
IB_CLIENT_ID_ORDER_ENTRY = int(os.environ.get("IB_CLIENT_ID_ORDER_ENTRY", CFG["ibkr"].get("client_id_order_entry", IB_CLIENT_ID)))
IB_CLIENT_ID_LIVE_BAR = int(os.environ.get("IB_CLIENT_ID_LIVE_BAR", CFG["ibkr"].get("client_id_live_bar", IB_CLIENT_ID + 1)))
IB_CLIENT_ID_POSITION_PROBE = int(os.environ.get("IB_CLIENT_ID_POSITION_PROBE", CFG["ibkr"].get("client_id_position_probe", IB_CLIENT_ID + 2)))

def validate_ib_client_ids() -> None:
    ids = {
        "order_entry": IB_CLIENT_ID_ORDER_ENTRY,
        "live_bar": IB_CLIENT_ID_LIVE_BAR,
        "position_probe": IB_CLIENT_ID_POSITION_PROBE,
    }
    if len(set(ids.values())) != len(ids):
        raise ValueError(f"IB client ids must be unique per connection role: {ids}")

validate_ib_client_ids()

DB_PATH        = PROJECT_ROOT / CFG["paths"]["db"]
MATRICES_DIR   = PROJECT_ROOT / CFG["paths"]["matrices"]
UNIVERSES_DIR  = PROJECT_ROOT / CFG["paths"]["universes"]
CLASS_PATH     = PROJECT_ROOT / CFG["paths"]["classifications"]
TRADE_LOG_DIR  = PROJECT_ROOT / CFG["paths"]["trade_logs"]
FILL_LOG_DIR   = PROJECT_ROOT / CFG["paths"]["fill_logs"]
PERF_LOG_DIR   = PROJECT_ROOT / CFG["paths"]["performance_logs"]
BORROW_LOG_DIR = PROJECT_ROOT / CFG["paths"]["borrow_logs"]

SIGNAL_COMPUTE_TIME = _cfg_time(CFG["execution"]["signal_compute_time_et"])
SUBMIT_DEADLINE = _cfg_time(CFG["execution"].get(
    "submit_deadline_et", CFG["execution"]["moc_deadline_et"]
))
MOC_DEADLINE = _cfg_time(CFG["execution"]["moc_deadline_et"])

# Create log dirs
for d in [TRADE_LOG_DIR, FILL_LOG_DIR, PERF_LOG_DIR, BORROW_LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(TRADE_LOG_DIR / f"moc_{dt.date.today().isoformat()}.log"),
    ]
)
log = logging.getLogger("moc_prod")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_fmp_data():
    """Load all FMP matrices, universe, and classifications."""
    import eval_alpha_ib
    eval_alpha_ib.UNIVERSE = UNIVERSE
    eval_alpha_ib.NEUTRALIZE = "market"  # We override in signal processing
    matrices, universe, classifications = eval_alpha_ib.load_data("full")
    valid_tickers = universe.columns.tolist()

    log.info(f"FMP data: {len(matrices)} fields, {len(valid_tickers)} tickers")
    log.info(f"Date range: {matrices['close'].index[0].date()} -> {matrices['close'].index[-1].date()}")
    return matrices, universe, classifications, valid_tickers


def load_alphas():
    """Load the configured production alpha set from the canonical database."""
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute("""
        SELECT a.id, a.expression, COALESCE(e.ic_mean, 0), COALESCE(e.sharpe_is, 0)
        FROM alphas a JOIN evaluations e ON e.alpha_id = a.id
        WHERE a.archived = 0
          AND a.asset_class = 'equities'
          AND e.universe = ?
          AND e.sharpe_is >= ?
          AND COALESCE(e.neutralization, ?) = ?
        ORDER BY e.sharpe_is DESC
    """, (UNIVERSE, MIN_ALPHA_SR, NEUTRALIZE, NEUTRALIZE)).fetchall()
    conn.close()
    log.info(f"Loaded {len(rows)} alphas from DB "
             f"(universe={UNIVERSE}, Sharpe>={MIN_ALPHA_SR:.2f}, "
             f"neutralization={NEUTRALIZE})")
    return rows


# ============================================================================
# SIGNAL COMPUTATION — with pre-neutralization
# ============================================================================

def evaluate_expression(expression, matrices):
    from src.operators.fastexpression import FastExpressionEngine
    engine = FastExpressionEngine(data_fields=matrices)
    return engine.evaluate(expression)


def group_demean(signal_df: pd.DataFrame, groups_series: pd.Series) -> pd.DataFrame:
    """
    Apply group-level demeaning (WQ group_neutralize equivalent).
    This removes within-group (subindustry) common factor exposure
    BEFORE combining alphas — the winning approach from our sweep.
    """
    result = signal_df.copy()
    common_cols = signal_df.columns.intersection(groups_series.index)
    grp = groups_series.reindex(common_cols)
    for g in grp.dropna().unique():
        col_mask = (grp == g).values
        cols_in_grp = common_cols[col_mask]
        if len(cols_in_grp) < 2:
            continue  # Need at least 2 stocks to demean
        block = result[cols_in_grp]
        grp_mean = block.mean(axis=1)
        result[cols_in_grp] = block.sub(grp_mean, axis=0)
    return result


def process_signal(alpha_df, universe_df, max_wt=MAX_WEIGHT):
    """Normalize signal: universe mask -> market demean -> unit-scale -> clip."""
    signal = alpha_df.copy()
    uni_mask = universe_df.reindex(
        index=signal.index, columns=signal.columns
    ).fillna(False).astype(bool)
    signal = signal.where(uni_mask, np.nan)

    # Market demean
    mean_val = signal.mean(axis=1)
    signal = signal.sub(mean_val, axis=0)

    # Scale to unit exposure
    abs_sum = signal.abs().sum(axis=1).replace(0, np.nan)
    signal = signal.div(abs_sum, axis=0)
    signal = signal.clip(lower=-max_wt, upper=max_wt)
    return signal.fillna(0.0)


def compute_combined_signal(alpha_signals, matrices, universe_df, classifications):
    """
    Build the production combined signal using the winning approach:
    pre-neutralize each alpha by subindustry -> equal-weight combine -> process.

    Returns: (signal_row pd.Series, signal_date pd.Timestamp)
    """
    ref_idx  = matrices["close"].index
    ref_cols = matrices["close"].columns.tolist()

    # Get subindustry groups for pre-neutralization
    sub_groups = classifications.get("subindustry")

    # Pre-neutralize each alpha and stack
    stack = np.zeros((len(ref_idx), len(ref_cols)), dtype=np.float64)
    count = np.zeros_like(stack)

    uni_mask = universe_df.reindex(
        index=ref_idx, columns=ref_cols
    ).fillna(False).astype(bool)

    for aid, raw in alpha_signals.items():
        s = raw.reindex(index=ref_idx, columns=ref_cols)

        if PRE_NEUTRALIZE and sub_groups is not None:
            # Apply universe mask then group demean per-alpha
            s = s.where(uni_mask, np.nan)
            s = group_demean(s, sub_groups)

        vals = s.values.copy()
        mask = ~np.isnan(vals)
        stack[mask] += vals[mask]
        count[mask] += 1.0

    count[count == 0] = np.nan
    agg_raw = pd.DataFrame(stack / count, index=ref_idx, columns=ref_cols)

    # Final signal processing (market demean + scale + clip)
    combined = process_signal(agg_raw, universe_df, max_wt=MAX_WEIGHT)

    # Return last row
    signal_row = combined.iloc[-1]
    signal_date = ref_idx[-1]
    return signal_row, signal_date


def signal_to_target_shares(signal_row, close_prices, booksize=BOOKSIZE):
    """Convert normalized weights to integer target share counts."""
    dollar_positions = signal_row * booksize
    valid_prices = close_prices.reindex(signal_row.index).replace(0, np.nan)
    target_shares = (dollar_positions / valid_prices).fillna(0).round().astype(int)
    return target_shares[target_shares != 0]


def apply_qp_optimization(
    signal_row: pd.Series,
    matrices: dict,
    current_positions: dict,
    *,
    booksize: float,
    max_w: float = 0.02,
    lambda_risk: float = 5.0,
    kappa_tc: float = 30.0,
    commission_per_share: float = 0.0045,
    impact_bps: float = 0.5,
    vol_window: int = 60,
) -> pd.Series:
    """Single-day QP solve over the equal-weight composite signal.

    Adds:
      • per-name vol-aware risk penalty (½λ σ²_i w_i²)
      • L1 t-cost penalty against current positions (κ_i |w_i − w_prev,i|)
      • dollar-neutral constraint (sum(w) == 0)
      • per-name cap |w_i| ≤ max_w, gross leverage ‖w‖₁ ≤ 1

    On failure, falls back to the input signal_row.

    Args:
        signal_row: pd.Series of (ticker → equal-weight composite alpha forecast).
        matrices:    must contain 'close'.
        current_positions: {ticker: dollar_position}; empty dict ⇒ w_prev=0
                           (first day / dry-run).
        booksize:    used to convert current_positions → w_prev fractions.
        max_w / lambda_risk / kappa_tc / commission_per_share / impact_bps:
                     match research_equity.json defaults.
        vol_window:  rolling stddev window in bars for diagonal risk model.
    """
    from src.portfolio.qp import solve_qp
    from src.portfolio.risk_model import build_diagonal

    close_df = matrices["close"]
    rets = close_df.pct_change(fill_method=None)
    vol = rets.rolling(vol_window, min_periods=20).std().shift(1).bfill().fillna(0.02)
    last_close = close_df.iloc[-1]
    last_vol = vol.iloc[-1]

    # Active = ticker has a forecast, a positive price, and a positive vol.
    sig_idx = signal_row.index
    active_mask = (
        last_close.reindex(sig_idx).fillna(0) > 0
    ) & (
        last_vol.reindex(sig_idx).fillna(0) > 0
    )
    active = sig_idx[active_mask]
    if len(active) < 10:
        log.warning(f"  QP: only {len(active)} active tickers — skipping, returning input signal")
        return signal_row

    alpha = signal_row.reindex(active).fillna(0).values.astype(float)
    prices = last_close.reindex(active).values.astype(float)
    sig = last_vol.reindex(active).values.astype(float)

    # w_prev from current $ positions (or zeros for cold-start / dry-run)
    if current_positions:
        wp_dollars = pd.Series(current_positions, dtype=float).reindex(active).fillna(0.0)
        w_prev = (wp_dollars / float(booksize)).values
    else:
        w_prev = np.zeros(len(active))

    # Diagonal risk model — single-day, no factor model in prod (yet)
    L_list, s2 = build_diagonal(sig)

    w_new = solve_qp(
        alpha, w_prev, prices, L_list, s2,
        lambda_risk=lambda_risk,
        kappa_tc=kappa_tc,
        max_w=max_w,
        commission_per_share=commission_per_share,
        impact_bps=impact_bps,
        dollar_neutral=True,
        max_gross_leverage=1.0,
    )
    if w_new is None:
        log.warning("  QP: solver returned None — falling back to input signal_row")
        return signal_row

    out = pd.Series(0.0, index=signal_row.index)
    out.loc[active] = w_new
    log.info(f"  QP optimized: {(out > 1e-6).sum()}L / {(out < -1e-6).sum()}S, "
             f"|w|.sum() = {out.abs().sum():.4f}, w_prev L1 = {abs(w_prev).sum():.4f}")
    return out


# ============================================================================
# IB CONNECTION
# ============================================================================

class IBConnection:
    """Production-grade IBKR connection wrapper using ib_insync."""

    def __init__(self, host=IB_HOST, port=IB_PORT_PAPER, client_id=IB_CLIENT_ID):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None
        self.connected = False

    def connect(self) -> bool:
        try:
            from ib_insync import IB
            self.ib = IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.connected = True
            accts = self.ib.managedAccounts()
            is_paper = "DU" in str(accts) or self.port == IB_PORT_PAPER
            log.info(f"Connected to IB {self.host}:{self.port} (clientId={self.client_id})")
            log.info(f"  Accounts: {accts} | Paper: {is_paper}")
            return True
        except Exception as e:
            log.warning(f"IB connect failed (clientId={self.client_id}): {e}")
            self.connected = False
            return False

    def disconnect(self):
        if self.ib and self.connected:
            self.ib.disconnect()
            self.connected = False
            log.info("Disconnected from IB")

    def get_account_summary(self) -> dict:
        if not self.connected:
            return {}
        summary = {}
        for item in self.ib.accountSummary():
            if item.tag in ("NetLiquidation", "GrossPositionValue", "BuyingPower",
                            "TotalCashValue", "MaintMarginReq", "AvailableFunds"):
                summary[item.tag] = float(item.value)
        return summary

    def get_positions(self) -> dict[str, int]:
        if not self.connected:
            return {}
        positions = {}
        for pos in self.ib.positions():
            sym = pos.contract.symbol
            qty = int(pos.position)
            if qty != 0:
                positions[sym] = qty
        return positions

    # ── SHORT BORROW AVAILABILITY ──────────────────────────────────
    def check_borrow_availability(self, tickers: list[str]) -> dict:
        """
        Check short-sell borrow availability for each ticker via IB.

        Uses reqMktData with genericTickList='236' (shortable shares)
        and tick type 46 (shortableShares).

        Returns: {ticker: {"shortable": bool, "shares": int, "fee_rate": float}}
        """
        if not self.connected:
            return {}

        from ib_insync import Stock
        results = {}
        batch_size = 50  # IB rate limit: ~50 concurrent market data requests

        # Paper accounts (and many live accounts without realtime entitlement)
        # return error 10089 on default reqMktData — no shortable data is
        # delivered, so every ticker falls through to HARD_TO_BORROW. Switch to
        # delayed data (type 3) so shortableShares actually populates.
        try:
            self.ib.reqMarketDataType(3)
        except Exception as e:
            log.warning(f"  reqMarketDataType(3) failed (will use realtime): {e}")

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            contracts = []
            for sym in batch:
                c = Stock(sym, "SMART", "USD")
                try:
                    self.ib.qualifyContracts(c)
                    contracts.append((sym, c))
                except Exception:
                    results[sym] = {"shortable": False, "shares": 0,
                                    "fee_rate": None, "status": "UNQUALIFIED"}

            # Request shortable data (genericTickList 236)
            reqs = []
            for sym, c in contracts:
                try:
                    ticker_data = self.ib.reqMktData(c, genericTickList="236")
                    reqs.append((sym, c, ticker_data))
                except Exception as e:
                    results[sym] = {"shortable": False, "shares": 0,
                                    "fee_rate": None, "status": f"REQ_FAILED: {e}"}

            # Wait for data to arrive (delayed data takes longer)
            self.ib.sleep(5)

            for sym, c, td in reqs:
                shares_avail = getattr(td, "shortableShares", None)
                # IB shortable INDICATOR: >2.5 easy, 1.5-2.5 limited, <1.5 hard.
                # This is informational; the binding signal for order placement
                # is shares_avail > 0.
                shortable_val = getattr(td, "shortable", None)

                if shares_avail is not None and shares_avail > 0:
                    if shortable_val is not None:
                        if shortable_val > 2.5:
                            status = "EASY"
                        elif shortable_val > 1.5:
                            status = "LIMITED"
                        else:
                            status = "HARD"
                    else:
                        status = "AVAILABLE"
                    results[sym] = {
                        "shortable": True,
                        "shares": int(shares_avail),
                        "shortable_indicator": float(shortable_val) if shortable_val is not None else None,
                        "fee_rate": getattr(td, "shortFeeRate", None),
                        "status": status,
                    }
                else:
                    results[sym] = {
                        "shortable": False,
                        "shares": 0,
                        "shortable_indicator": float(shortable_val) if shortable_val is not None else None,
                        "fee_rate": None,
                        "status": "NO_DATA" if shares_avail is None else "HARD_TO_BORROW",
                    }

                # Cancel market data to free slot
                self.ib.cancelMktData(c)

            log.info(f"  Borrow check batch {i//batch_size + 1}: "
                     f"{sum(1 for r in results.values() if r.get('shortable'))} shortable / "
                     f"{len(results)} checked")

        return results

    def get_last_close_prices(self, tickers: list[str]) -> dict[str, float]:
        """Get yesterday's close from IB for cross-validation."""
        if not self.connected:
            return {}
        from ib_insync import Stock
        prices = {}
        for sym in tickers[:50]:
            try:
                contract = Stock(sym, "SMART", "USD")
                bars = self.ib.reqHistoricalData(
                    contract, endDateTime="",
                    durationStr="2 D", barSizeSetting="1 day",
                    whatToShow="ADJUSTED_LAST", useRTH=True
                )
                if bars:
                    prices[sym] = bars[-1].close
                self.ib.sleep(0.1)
            except Exception:
                pass
        return prices

    def submit_moc_orders(self, order_diffs: dict[str, int]) -> list[dict]:
        """Submit MOC orders. Returns list of order records."""
        if not self.connected:
            return []

        from ib_insync import Stock, Order
        records = []
        submitted = []

        for sym, shares in order_diffs.items():
            if shares == 0:
                continue
            try:
                contract = Stock(sym, "SMART", "USD")
                qualified = self.ib.qualifyContracts(contract)
                if not qualified:
                    raise RuntimeError("QUALIFY_FAILED")
                contract = qualified[0]
                action = "BUY" if shares > 0 else "SELL"
                order = Order(
                    action=action,
                    totalQuantity=abs(shares),
                    orderType="MOC",
                    tif="DAY",
                )
                trade = self.ib.placeOrder(contract, order)
                record = {
                    "symbol": sym, "action": action, "quantity": abs(shares),
                    "order_type": "MOC", "order_id": trade.order.orderId,
                    "status": "PendingSubmit", "timestamp": dt.datetime.now().isoformat(),
                }
                records.append(record)
                submitted.append((record, trade))
                log.info(f"  ORDER: {action} {abs(shares)} {sym} MOC (id={trade.order.orderId})")
            except Exception as e:
                log.error(f"  FAILED {sym}: {e}")
                records.append({
                    "symbol": sym, "action": "BUY" if shares > 0 else "SELL",
                    "quantity": abs(shares), "order_type": "MOC",
                    "status": f"FAILED: {e}", "timestamp": dt.datetime.now().isoformat(),
                })

        if submitted:
            self.ib.sleep(1)
            for record, trade in submitted:
                status = trade.orderStatus.status or record["status"]
                last_msg = trade.log[-1].message if trade.log else ""
                if status in {"Cancelled", "Inactive"}:
                    record["status"] = f"FAILED: {status} {last_msg}".strip()
                else:
                    record["status"] = status
                record["filled"] = float(trade.orderStatus.filled or 0)
                record["remaining"] = float(trade.orderStatus.remaining or 0)
                perm_id = trade.orderStatus.permId or trade.order.permId
                if perm_id:
                    record["perm_id"] = int(perm_id)

        return records

    # ── PRE-TRADE MARKET SNAPSHOT ───────────────────────────────────
    def capture_pretrade_snapshot(self, tickers: list[str]) -> dict:
        """
        Capture bid/ask/last/volume for all target tickers at signal time.
        This is the benchmark for measuring execution quality:
        - Slippage = fill_price - mid_at_signal_time
        - Spread = ask - bid at signal time (liquidity indicator)
        """
        if not self.connected:
            return {}

        from ib_insync import Stock
        snapshot = {}
        batch_size = 50

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            reqs = []

            for sym in batch:
                try:
                    c = Stock(sym, "SMART", "USD")
                    self.ib.qualifyContracts(c)
                    td = self.ib.reqMktData(c)
                    reqs.append((sym, c, td))
                except Exception:
                    snapshot[sym] = {"error": "QUALIFY_FAILED"}

            self.ib.sleep(2)  # Let quotes arrive

            for sym, c, td in reqs:
                bid = getattr(td, "bid", None)
                ask = getattr(td, "ask", None)
                last = getattr(td, "last", None)
                vol = getattr(td, "volume", None)

                # Compute mid and spread
                mid = None
                spread_bps = None
                if bid and ask and bid > 0 and ask > 0:
                    mid = (bid + ask) / 2.0
                    spread_bps = (ask - bid) / mid * 10000

                snapshot[sym] = {
                    "capture_time": dt.datetime.now().isoformat(),
                    "bid": bid if bid and bid > 0 else None,
                    "ask": ask if ask and ask > 0 else None,
                    "last": last if last and last > 0 else None,
                    "mid": round(mid, 4) if mid else None,
                    "spread_bps": round(spread_bps, 1) if spread_bps else None,
                    "volume": int(vol) if vol and vol > 0 else None,
                }
                self.ib.cancelMktData(c)

            log.info(f"  Snapshot batch {i//batch_size + 1}: {len(reqs)} tickers")

        n_with_quotes = sum(1 for s in snapshot.values() if s.get("bid"))
        log.info(f"  Snapshot complete: {n_with_quotes}/{len(snapshot)} with live quotes")
        return snapshot

    # ── FILL COLLECTION ────────────────────────────────────────────
    def collect_fills(self) -> list[dict]:
        """
        Collect all execution reports for today's session.
        Call after market close (4:01 PM).
        """
        if not self.connected:
            return []

        fills = []
        for fill in self.ib.fills():
            exec_info = fill.execution
            comm_info = fill.commissionReport
            fills.append({
                "symbol": fill.contract.symbol,
                "exec_id": exec_info.execId,
                "order_id": exec_info.orderId,
                "action": exec_info.side,
                "quantity": int(exec_info.shares),
                "fill_price": exec_info.price,
                "fill_time": exec_info.time,
                "exchange": exec_info.exchange,
                "commission": comm_info.commission if comm_info else None,
                "realized_pnl": comm_info.realizedPNL if comm_info else None,
            })
        return fills


# ============================================================================
# PRE-TRADE RISK CHECKS
# ============================================================================

def run_risk_checks(signal_row, close_prices, classifications, borrow_results,
                    target_gmv, account_summary) -> tuple[bool, list[str]]:
    """
    Pre-trade risk checks. Returns (pass: bool, warnings: list[str]).
    Any FAIL stops order submission. WARNINGs are logged but allowed.
    """
    warnings = []
    fails = []

    # 1. Max single position concentration
    max_wt = signal_row.abs().max()
    if max_wt > MAX_WEIGHT * 1.5:
        fails.append(f"FAIL: Max position weight {max_wt:.4f} exceeds 1.5x limit ({MAX_WEIGHT*1.5:.4f})")

    # 2. Long/short balance (should be near dollar-neutral)
    longs  = signal_row[signal_row > 0].sum()
    shorts = signal_row[signal_row < 0].sum()
    net_exposure = abs(longs + shorts)
    if net_exposure > 0.05:
        warnings.append(f"WARN: Net exposure {net_exposure:.4f} (>5% of book)")

    # 3. Sector concentration
    if "sector" in classifications:
        sector_groups = classifications["sector"]
        for g in sector_groups.unique():
            tickers_in_group = sector_groups[sector_groups == g].index
            group_exposure = signal_row.reindex(tickers_in_group).fillna(0).abs().sum()
            if group_exposure > CFG["risk"]["max_sector_exposure_pct"] / 100:
                warnings.append(f"WARN: Sector {g} exposure {group_exposure:.1%}")

    # 4. Unborrowed shorts
    if borrow_results:
        short_tickers = signal_row[signal_row < -1e-8].index
        hard_to_borrow = [t for t in short_tickers
                         if t in borrow_results and not borrow_results[t].get("shortable", True)]
        if hard_to_borrow:
            warnings.append(f"WARN: {len(hard_to_borrow)} hard-to-borrow shorts: "
                           f"{hard_to_borrow[:10]}")

    # 5. Account equity check (if connected)
    if account_summary:
        nlv = account_summary.get("NetLiquidation", 0)
        if nlv > 0 and target_gmv / nlv > CFG["account"]["leverage_limit"]:
            fails.append(f"FAIL: Leverage {target_gmv/nlv:.1f}x exceeds limit "
                        f"{CFG['account']['leverage_limit']}x")

    # 6. Data staleness
    # (checked in main workflow via FMP date)

    for w in warnings:
        log.warning(f"  {w}")
    for f_msg in fails:
        log.error(f"  {f_msg}")

    passed = len(fails) == 0
    return passed, warnings + fails


# ============================================================================
# DATA INTEGRITY CHECKS (Phase 1)
# ============================================================================

def run_data_integrity_checks(matrices, universe_df, classifications,
                               ib_conn=None, fmp_last_close=None) -> tuple[bool, list[str]]:
    """
    Comprehensive data integrity validation.
    ALL checks must pass before signal computation begins.
    """
    warnings = []
    fails = []

    # A. FMP data freshness — TWO checks. The hours-based check is a coarse
    # belt; the trading-day check is the real gate. The previous trading day's
    # close MUST be the last row in matrices, otherwise time-series operators
    # read across data holes and the live-bar appended for today is preceded
    # by stale rows.
    last_date = matrices["close"].index[-1]
    stale_hours = (dt.datetime.now() - last_date).total_seconds() / 3600
    if stale_hours > CFG["risk"]["stale_data_halt_hours"]:
        fails.append(f"FAIL: FMP data {stale_hours:.0f}h stale "
                     f"(threshold {CFG['risk']['stale_data_halt_hours']}h, last: {last_date.date()})")
    else:
        log.info(f"  [OK] FMP freshness: {last_date.date()} ({stale_hours:.0f}h ago)")

    # A2. Trading-day-aware staleness — the binding check.
    if CFG["risk"].get("require_last_bar_is_prev_trading_day", True):
        try:
            import pandas_market_calendars as mcal
            nyse = mcal.get_calendar("NYSE")
            today = dt.date.today()
            sched = nyse.schedule(start_date=str(today - dt.timedelta(days=10)),
                                   end_date=str(today))
            sched_dates = [d.date() for d in sched.index]
            # Most-recent COMPLETED trading day = the last sched date strictly
            # before today (today's session may not yet be closed when trader
            # runs at 14:30 CDT).
            prior_days = [d for d in sched_dates if d < today]
            if not prior_days:
                fails.append(f"FAIL: NYSE calendar returned no prior trading day before {today}")
            else:
                expected_last = prior_days[-1]
                if last_date.date() != expected_last:
                    fails.append(
                        f"FAIL: matrices end at {last_date.date()} but the "
                        f"previous trading day is {expected_last}. Cache is "
                        f"missing {(expected_last - last_date.date()).days} "
                        f"calendar days; halting before trading on stale data."
                    )
                else:
                    log.info(f"  [OK] Last bar = previous trading day ({expected_last})")
        except ImportError:
            log.warning("  pandas_market_calendars not installed; skipping "
                        "trading-day-aware staleness check (HOURS gate ONLY)")
        except Exception as e:
            fails.append(f"FAIL: trading-day staleness check raised {type(e).__name__}: {e}")

    # B. Universe coverage
    n_active = int(universe_df.iloc[-1].sum())
    if n_active < 100:
        fails.append(f"FAIL: Only {n_active} active tickers in universe (min 100)")
    else:
        log.info(f"  [OK] Universe: {n_active} active tickers")

    # C. Classification hierarchy — sector should refine to industry, industry
    # to subindustry. FMP often has industry == subindustry for many names, so
    # allow EQUALITY (not strict less-than). The real failure mode we're
    # guarding against is a degenerate case where subindustry collapses to
    # << industry, which means the subindustry classifier broke.
    n_sector = len(classifications.get("sector", pd.Series()).unique())
    n_industry = len(classifications.get("industry", pd.Series()).unique())
    n_subindustry = len(classifications.get("subindustry", pd.Series()).unique())
    hierarchy_ok = (n_sector <= n_industry <= n_subindustry) and n_subindustry >= 20
    if not hierarchy_ok:
        fails.append(f"FAIL: Classification hierarchy degenerate: "
                    f"sector={n_sector}, industry={n_industry}, subindustry={n_subindustry} "
                    f"(need sector<=industry<=subindustry, subindustry>=20)")
    else:
        log.info(f"  [OK] Classifications: {n_sector} sectors, {n_industry} industries, {n_subindustry} subindustries")

    # D. Price reasonableness
    last_close = matrices["close"].iloc[-1]
    median_price = last_close.median()
    if median_price < 1.0 or median_price > 500:
        fails.append(f"FAIL: Median price ${median_price:.2f} out of range")
    else:
        log.info(f"  [OK] Median price: ${median_price:.2f}")

    # E. Returns sanity (check for unhandled splits)
    last_returns = matrices["returns"].iloc[-1]
    extreme_tickers = last_returns[last_returns.abs() > 0.5].index.tolist()
    if extreme_tickers:
        warnings.append(f"WARN: {len(extreme_tickers)} tickers with >50% daily return "
                       f"(possible split): {extreme_tickers[:5]}")

    # F. Cross-validate FMP vs IB prices (if connected)
    if ib_conn and ib_conn.connected and fmp_last_close is not None:
        sample = fmp_last_close.dropna().index[:50].tolist()
        ib_prices = ib_conn.get_last_close_prices(sample)
        if ib_prices:
            mismatches = 0
            for sym, ib_p in ib_prices.items():
                fmp_p = fmp_last_close.get(sym, np.nan)
                if fmp_p > 0 and abs(fmp_p - ib_p) / fmp_p > 0.02:
                    mismatches += 1
            if mismatches > 5:
                fails.append(f"FAIL: {mismatches}/50 price mismatches (FMP vs IB >2%)")
            else:
                log.info(f"  [OK] Price cross-validation: {mismatches}/{len(ib_prices)} mismatches")

    for w in warnings:
        log.warning(f"  {w}")
    for f_msg in fails:
        log.error(f"  {f_msg}")

    return len(fails) == 0, warnings + fails


# ============================================================================
# TRADE LOGGING
# ============================================================================

def save_trade_log(date, target_portfolio, current_positions, order_diffs,
                   order_records, account_summary, signal_date, mode,
                   borrow_summary, risk_messages, target_gmv):
    """Save comprehensive daily trade log."""
    mode_suffix = "" if mode == "live" else f"_{mode.replace('-', '_')}"
    log_path = TRADE_LOG_DIR / f"trade_{date.isoformat()}{mode_suffix}.json"

    trade_log = {
        "date": date.isoformat(),
        "mode": mode,
        "signal_date": str(signal_date.date()) if hasattr(signal_date, 'date') else str(signal_date),
        "timestamp": dt.datetime.now().isoformat(),
        "strategy_version": CFG["strategy"]["version"],

        "config": {
            "target_gmv": target_gmv,
            "universe": UNIVERSE,
            "neutralization": NEUTRALIZE,
            "pre_neutralize": PRE_NEUTRALIZE,
            "combiner": CFG["strategy"]["combiner"],
        },
        "account": account_summary,

        "portfolio_summary": {
            "n_long":  int((target_portfolio > 0).sum()),
            "n_short": int((target_portfolio < 0).sum()),
            "gross_shares": int(target_portfolio.abs().sum()),
            "net_shares": int(target_portfolio.sum()),
        },

        "n_orders": len(order_diffs),
        "risk_messages": risk_messages,
        "borrow_summary": borrow_summary,

        "target_portfolio": {k: int(v) for k, v in target_portfolio.items()},
        "current_positions": current_positions,
        "order_diffs": {k: int(v) for k, v in order_diffs.items()},
        "order_records": order_records,
    }

    with open(log_path, "w") as f:
        json.dump(trade_log, f, indent=2)
    log.info(f"Trade log: {log_path}")
    return log_path


def save_borrow_log(date, borrow_results):
    """Save daily borrow availability snapshot."""
    if not borrow_results:
        return
    log_path = BORROW_LOG_DIR / f"borrow_{date.isoformat()}.json"
    with open(log_path, "w") as f:
        json.dump({
            "date": date.isoformat(),
            "timestamp": dt.datetime.now().isoformat(),
            "n_checked": len(borrow_results),
            "n_shortable": sum(1 for r in borrow_results.values() if r.get("shortable")),
            "n_hard_to_borrow": sum(1 for r in borrow_results.values() if not r.get("shortable")),
            "results": borrow_results,
        }, f, indent=2)
    log.info(f"Borrow log: {log_path}")


def save_pretrade_snapshot(date, snapshot):
    """Save pre-trade market snapshot (bid/ask/last/volume)."""
    if not snapshot:
        return
    log_path = TRADE_LOG_DIR / f"pretrade_{date.isoformat()}.json"
    with open(log_path, "w") as f:
        json.dump({
            "date": date.isoformat(),
            "timestamp": dt.datetime.now().isoformat(),
            "n_tickers": len(snapshot),
            "n_with_quotes": sum(1 for s in snapshot.values() if s.get("bid")),
            "tickers": snapshot,
        }, f, indent=2)
    log.info(f"Pre-trade snapshot: {log_path}")


def save_fills_log(date, fills, pretrade_snapshot):
    """Save fill execution data and compute slippage vs pre-trade snapshot."""
    if not fills:
        return
    log_path = FILL_LOG_DIR / f"fills_{date.isoformat()}.json"

    enriched_fills = []
    total_commission = 0
    slippages = []

    for fill in fills:
        sym = fill["symbol"]
        fill_price = fill["fill_price"]
        pre = pretrade_snapshot.get(sym, {})

        # Compute slippage vs pre-trade mid
        slippage_vs_mid_bps = None
        if pre.get("mid") and pre["mid"] > 0:
            slippage_vs_mid_bps = (fill_price - pre["mid"]) / pre["mid"] * 10000
            slippages.append(slippage_vs_mid_bps)

        enriched = {
            **fill,
            "pretrade_bid": pre.get("bid"),
            "pretrade_ask": pre.get("ask"),
            "pretrade_mid": pre.get("mid"),
            "pretrade_spread_bps": pre.get("spread_bps"),
            "pretrade_volume": pre.get("volume"),
            "slippage_vs_mid_bps": round(slippage_vs_mid_bps, 2) if slippage_vs_mid_bps is not None else None,
        }
        enriched_fills.append(enriched)
        if fill.get("commission"):
            total_commission += fill["commission"]

    # Aggregate execution metrics
    aggregate = {
        "n_fills": len(fills),
        "total_commission": round(total_commission, 2),
        "mean_slippage_vs_mid_bps": round(np.mean(slippages), 2) if slippages else None,
        "median_slippage_vs_mid_bps": round(np.median(slippages), 2) if slippages else None,
        "std_slippage_bps": round(np.std(slippages), 2) if slippages else None,
        "max_slippage_bps": round(max(slippages), 2) if slippages else None,
        "min_slippage_bps": round(min(slippages), 2) if slippages else None,
    }

    with open(log_path, "w") as f:
        json.dump({
            "date": date.isoformat(),
            "timestamp": dt.datetime.now().isoformat(),
            "aggregate": aggregate,
            "fills": enriched_fills,
        }, f, indent=2)
    log.info(f"Fills log: {log_path} ({len(fills)} fills, ${total_commission:.2f} commission)")
    if slippages:
        log.info(f"  Slippage vs mid: mean={aggregate['mean_slippage_vs_mid_bps']:.1f}bps, "
                f"median={aggregate['median_slippage_vs_mid_bps']:.1f}bps")


def save_reconciliation(date, target_shares, fills, account_summary_start,
                        account_summary_end):
    """Save daily reconciliation: expected vs actual positions, PnL."""
    log_path = PERF_LOG_DIR / f"recon_{date.isoformat()}.json"

    # Build fills map
    fills_map = {}
    for f in fills:
        sym = f["symbol"]
        if sym not in fills_map:
            fills_map[sym] = {"total_qty": 0, "avg_price": 0, "total_commission": 0}
        fills_map[sym]["total_qty"] += f["quantity"]
        fills_map[sym]["avg_price"] = f["fill_price"]  # Last fill price
        if f.get("commission"):
            fills_map[sym]["total_commission"] += f["commission"]

    # Fill rate
    expected_orders = set(target_shares.index)
    filled_orders = set(fills_map.keys())
    fill_rate = len(filled_orders & expected_orders) / max(len(expected_orders), 1)

    # PnL from account values
    nlv_start = account_summary_start.get("NetLiquidation", 0) if account_summary_start else 0
    nlv_end = account_summary_end.get("NetLiquidation", 0) if account_summary_end else 0
    daily_pnl = nlv_end - nlv_start if nlv_start > 0 and nlv_end > 0 else None

    recon = {
        "date": date.isoformat(),
        "timestamp": dt.datetime.now().isoformat(),
        "fill_rate": round(fill_rate, 4),
        "n_expected": len(expected_orders),
        "n_filled": len(filled_orders),
        "missing_fills": list(expected_orders - filled_orders),
        "unexpected_fills": list(filled_orders - expected_orders),
        "pnl": {
            "nlv_start": nlv_start,
            "nlv_end": nlv_end,
            "daily_pnl": daily_pnl,
            "daily_return_pct": round(daily_pnl / nlv_start * 100, 4) if daily_pnl and nlv_start > 0 else None,
        },
        "total_commission": sum(f.get("total_commission", 0) for f in fills_map.values()),
    }

    with open(log_path, "w") as f:
        json.dump(recon, f, indent=2)
    log.info(f"Reconciliation: {log_path} (fill rate: {fill_rate:.1%})")
    if daily_pnl is not None:
        log.info(f"  Daily PnL: ${daily_pnl:+,.0f} ({recon['pnl']['daily_return_pct']:+.2f}%)")


# ============================================================================
# MAIN TRADING WORKFLOW
# ============================================================================

def is_us_equity_trading_day(d: dt.date) -> bool:
    """True if `d` is a regular NYSE trading day (not weekend, not holiday).

    Uses pandas_market_calendars for the authoritative NYSE schedule.
    Falls back to weekday-only check if the lib is unavailable.
    """
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        sched = nyse.schedule(start_date=str(d), end_date=str(d))
        return len(sched) > 0
    except ImportError:
        # Fallback: weekday only (will incorrectly run on holidays — log warning)
        log.warning("pandas_market_calendars not installed; falling back to weekday-only check")
        return d.weekday() < 5  # Mon=0..Fri=4


def run_trading_workflow(mode="dry-run", port=IB_PORT_PAPER, gmv_override=None,
                         check_borrow_only=False, no_qp=False, force=False):
    """Main daily trading pipeline."""
    today = dt.date.today()
    t0 = time.time()
    booksize = gmv_override or TARGET_GMV

    # Phase 0: market-day gate — exit early on weekends + NYSE holidays.
    # `--force` overrides for manual / smoke-test runs.
    if not is_us_equity_trading_day(today) and not force:
        log.info("=" * 90)
        log.info(f"  IB MOC TRADER — {today.isoformat()} ({today.strftime('%A')}) — "
                 f"NOT A NYSE TRADING DAY")
        log.info(f"  Skipping daily run. Use --force to override.")
        log.info("=" * 90)
        return

    log.info("=" * 90)
    log.info(f"  IB MOC TRADER v{CFG['strategy']['version']} — {today.isoformat()} — {mode.upper()}")
    log.info(f"  GMV: ${booksize:,.0f} | Universe: {UNIVERSE} | Neut: {NEUTRALIZE}")
    log.info(f"  Pre-neutralize: {PRE_NEUTRALIZE} | Combiner: {CFG['strategy']['combiner']}")
    log.info("=" * 90)

    # ── Phase 1: Data Integrity ───────────────────────────────────────
    if mode == "live" and not check_borrow_only and not force:
        now_et = now_eastern()
        if now_et.time() < SIGNAL_COMPUTE_TIME:
            log.error(f"  TOO EARLY FOR LIVE SIGNAL RUN: now={now_et.time().replace(microsecond=0)} ET, "
                      f"signal_compute_time_et={SIGNAL_COMPUTE_TIME}")
            log.error("  Today's delay=0 live bar should be built close to the closing auction; aborting.")
            return

    log.info("\nPhase 1: Data integrity checks...")
    matrices, universe_df, classifications, valid_tickers = load_fmp_data()
    close_df = matrices["close"]
    fmp_last_close = close_df.iloc[-1]
    last_fmp_date = close_df.index[-1]

    # Data integrity pre-flight
    integrity_ok, integrity_msgs = run_data_integrity_checks(
        matrices, universe_df, classifications
    )
    if not integrity_ok:
        log.error("  DATA INTEGRITY FAILED -- HALTING")
        return

    # ── Phase 1b: Construct live bar (TODAY's estimated OHLCV) ────────
    log.info("\nPhase 1b: Constructing today's live bar...")
    log.info("  delay=0 requires today's OHLCV for alpha evaluation")
    active_vwap_tickers = universe_df.columns[
        universe_df.iloc[-1].fillna(False).astype(bool)
    ].tolist()
    log.info(f"  Active VWAP universe: {len(active_vwap_tickers)} tickers")

    # IB streaming VWAP is opt-in because delayed/no-subscription paper feeds
    # can return no values while consuming most of the MOC deadline window.
    early_ib = None
    early_probe = None
    if mode == "live" and not check_borrow_only and ENABLE_IB_LIVE_VWAP:
        try:
            early_probe = IBConnection(host=IB_HOST, port=port, client_id=IB_CLIENT_ID_LIVE_BAR)
            if early_probe.connect():
                early_ib = early_probe.ib
                log.info(f"  IB early-connect OK for live VWAP (clientId={IB_CLIENT_ID_LIVE_BAR})")
            else:
                log.warning("  IB early-connect failed; live_bar will use non-IB VWAP sources")
        except Exception as e:
            log.warning(f"  IB early-connect raised {type(e).__name__}; "
                        f"live_bar will use non-IB VWAP sources")
    elif mode == "live" and not check_borrow_only:
        log.info("  IB streaming VWAP disabled; using quote tape/FMP intraday VWAP")

    try:
        matrices, live_quotes, flagged_tickers = append_live_bar(
            matrices,
            ib=early_ib,
            vwap_tickers=active_vwap_tickers,
        )
    finally:
        if early_probe and early_probe.connected:
            early_probe.disconnect()
            log.info("  IB early-connect closed after live-bar construction")

    # Update close reference to use LIVE prices (today's estimated close)
    close_df = matrices["close"]
    fmp_last_close = close_df.iloc[-1]  # Now includes today's live bar
    last_fmp_date = close_df.index[-1]
    log.info(f"  Signal will use data through: {last_fmp_date.date()}")
    if flagged_tickers:
        log.warning(f"  {len(flagged_tickers)} tickers excluded (corp actions)")

    # Extend universe_df to cover today (carry forward last day's membership)
    if last_fmp_date not in universe_df.index:
        today_uni = universe_df.iloc[[-1]].copy()
        today_uni.index = [last_fmp_date]
        universe_df = pd.concat([universe_df, today_uni])
        log.info(f"  Universe extended to {last_fmp_date.date()} ({int(today_uni.iloc[0].sum())} active)")

    # ── Phase 2: Evaluate alphas ──────────────────────────────────────
    log.info("\nPhase 2: Evaluating alpha signals...")
    alphas = load_alphas()
    alpha_signals = {}
    for aid, expr, ic, sr in alphas:
        try:
            raw = evaluate_expression(expr, matrices)
            if raw is not None and not raw.empty:
                alpha_signals[aid] = raw
        except Exception as e:
            log.warning(f"  Alpha #{aid}: FAILED ({e})")
    log.info(f"  {len(alpha_signals)}/{len(alphas)} signals loaded")
    if len(alpha_signals) < 25:
        log.error(f"  Only {len(alpha_signals)} alphas loaded (min 25) -- HALTING")
        return

    # ── Phase 3: Compute combined signal ──────────────────────────────
    log.info("\nPhase 3: Signal construction (pre-sub -> equal-weight -> subindustry)...")
    t_sig = time.time()
    signal_row, signal_date = compute_combined_signal(
        alpha_signals, matrices, universe_df, classifications
    )
    log.info(f"  Signal computed in {time.time()-t_sig:.1f}s")
    log.info(f"  Signal date: {signal_date.date()}")

    n_long  = (signal_row > 1e-6).sum()
    n_short = (signal_row < -1e-6).sum()
    log.info(f"  Positions: {n_long} long, {n_short} short")

    # ── Phase 3.5: QP optimization (per-name risk + t-cost) ──────────
    # Single-day QP over the equal-weight composite. Replaces the
    # signal-as-portfolio handoff that previously went straight to
    # signal_to_target_shares — adds risk-aware sizing and a κ·|w−w_prev|
    # t-cost penalty against current positions.
    if not no_qp:
        # Probe IB for current positions BEFORE the QP so w_prev is realistic.
        # In dry-run / first-day, this is just an empty dict (w_prev = 0).
        early_positions = {}
        if mode == "live" and not check_borrow_only:
            log.info("\nPhase 3.5a: Querying IB for current positions (for QP w_prev)...")
            _probe_ib = None
            try:
                _probe_ib = IBConnection(host=IB_HOST, port=port, client_id=IB_CLIENT_ID_POSITION_PROBE)
                if _probe_ib.connect():
                    early_positions = _probe_ib.get_positions() or {}
                    log.info(f"  Current positions for w_prev: {len(early_positions)}")
            except Exception as e:
                log.warning(f"  Could not probe IB for positions: {e} -- using w_prev=0")
            finally:
                if _probe_ib and _probe_ib.connected:
                    _probe_ib.disconnect()

        log.info("\nPhase 3.5b: Single-day QP optimization (equal-weight + diag risk + t-cost)...")
        signal_row = apply_qp_optimization(
            signal_row, matrices,
            current_positions=early_positions,
            booksize=booksize,
            max_w=MAX_WEIGHT,
        )
        n_long  = (signal_row > 1e-6).sum()
        n_short = (signal_row < -1e-6).sum()
        log.info(f"  Post-QP positions: {n_long} long, {n_short} short")
    else:
        log.info("\nPhase 3.5: QP DISABLED (no_qp=True) — using equal-weight signal-as-portfolio")

    # ── Phase 4: Target shares ────────────────────────────────────────
    log.info("\nPhase 4: Converting to target shares...")
    # Use TODAY's live prices for share sizing (not stale T-1)
    target_shares = signal_to_target_shares(signal_row, fmp_last_close, booksize=booksize)

    target_dollar = (target_shares.abs() * fmp_last_close.reindex(target_shares.index).fillna(0)).sum()
    long_dollar  = (target_shares[target_shares > 0] *
                    fmp_last_close.reindex(target_shares[target_shares > 0].index).fillna(0)).sum()
    short_dollar = (target_shares[target_shares < 0].abs() *
                    fmp_last_close.reindex(target_shares[target_shares < 0].index).fillna(0)).sum()

    log.info(f"  Target GMV:  ${target_dollar:,.0f}")
    log.info(f"  Long:        ${long_dollar:,.0f} ({(target_shares > 0).sum()} pos)")
    log.info(f"  Short:       ${short_dollar:,.0f} ({(target_shares < 0).sum()} pos)")
    log.info(f"  Positions:   {len(target_shares)} total")

    # ── Phase 5: IB Connection & Borrow Check ─────────────────────────
    ib_conn = IBConnection(host=IB_HOST, port=port, client_id=IB_CLIENT_ID_ORDER_ENTRY)
    current_positions = {}
    account_summary = {}
    account_summary_start = {}  # For end-of-day PnL calculation
    borrow_results = {}
    pretrade_snapshot = {}
    order_records = []

    if mode == "live" or check_borrow_only:
        log.info("\nPhase 5: Connecting to IB...")
        connected = ib_conn.connect()

        if connected:
            account_summary = ib_conn.get_account_summary()
            account_summary_start = dict(account_summary)  # Snapshot for recon
            if account_summary:
                log.info(f"  NLV: ${account_summary.get('NetLiquidation', 0):,.0f}")
                log.info(f"  GPV: ${account_summary.get('GrossPositionValue', 0):,.0f}")

            if not check_borrow_only:
                current_positions = ib_conn.get_positions()
                log.info(f"  Current positions: {len(current_positions)}")

            # Cross-validate FMP vs IB prices
            log.info("\nPhase 5b: Price cross-validation (FMP vs IB)...")
            ib_prices = ib_conn.get_last_close_prices(valid_tickers[:50])
            if ib_prices:
                mismatches = 0
                for sym, ib_p in ib_prices.items():
                    fmp_p = fmp_last_close.get(sym, np.nan)
                    if pd.notna(fmp_p) and fmp_p > 0 and abs(fmp_p - ib_p) / fmp_p > 0.02:
                        mismatches += 1
                        log.warning(f"    MISMATCH: {sym} FMP=${fmp_p:.2f} vs IB=${ib_p:.2f}")
                log.info(f"  Price check: {mismatches}/{len(ib_prices)} mismatches")
                if mismatches > 5:
                    log.error("  HIGH MISMATCH RATE -- possible splits. Review before trading.")

            # Check borrow availability for SHORT positions
            short_tickers = target_shares[target_shares < 0].index.tolist()
            if short_tickers:
                log.info(f"\nStep 5b: Checking borrow availability for {len(short_tickers)} shorts...")
                borrow_results = ib_conn.check_borrow_availability(short_tickers)
                n_borrowable = sum(1 for r in borrow_results.values() if r.get("shortable"))
                n_hard = len(borrow_results) - n_borrowable
                log.info(f"  Borrowable: {n_borrowable} | Hard-to-borrow: {n_hard}")

                # Save borrow log
                save_borrow_log(today, borrow_results)

                if check_borrow_only:
                    # Print borrow report and exit
                    log.info(f"\n{'='*60}")
                    log.info("  BORROW AVAILABILITY REPORT")
                    log.info(f"{'='*60}")
                    log.info(f"  {'Symbol':<8} {'Status':<15} {'Shares':>10} {'Fee Rate':>10}")
                    log.info(f"  {'-'*48}")
                    for sym in sorted(borrow_results.keys()):
                        r = borrow_results[sym]
                        fee = f"{r['fee_rate']:.2f}%" if r.get('fee_rate') else "N/A"
                        log.info(f"  {sym:<8} {r['status']:<15} {r['shares']:>10,} {fee:>10}")
                    ib_conn.disconnect()
                    return
        else:
            if mode == "live":
                log.error("  IB unavailable in LIVE mode -- HALTING instead of falling back to dry-run")
                save_trade_log(today, target_shares, current_positions, {},
                               [], account_summary, signal_date, "halted_ib_unavailable",
                               {}, ["IB unavailable in LIVE mode"], booksize)
                return
            log.warning("  IB unavailable -- falling back to dry-run")
            mode = "dry-run"
    else:
        log.info("\nPhase 5: DRY-RUN -- skipping IB connection")

    # ── Phase 6: Risk checks ──────────────────────────────────────────
    log.info("\nPhase 6: Pre-trade risk checks...")
    risk_passed, risk_messages = run_risk_checks(
        signal_row, fmp_last_close, classifications, borrow_results,
        booksize, account_summary
    )

    if not risk_passed:
        log.error("  RISK CHECKS FAILED — HALTING ORDER SUBMISSION")
        save_trade_log(today, target_shares, current_positions, {},
                       [], account_summary, signal_date, "HALTED",
                       {}, risk_messages, booksize)
        if ib_conn.connected:
            ib_conn.disconnect()
        return

    log.info(f"  Risk checks: PASS ({len(risk_messages)} warnings)")

    # ── Phase 7: Pre-trade snapshot & order diffs ─────────────────────
    log.info("\nPhase 7: Computing order diffs...")
    order_diffs = {}
    for sym, target_qty in target_shares.items():
        current_qty = current_positions.get(sym, 0)
        diff = target_qty - current_qty
        if diff != 0:
            order_diffs[sym] = diff

    # Close positions we no longer want
    for sym, current_qty in current_positions.items():
        if sym not in target_shares.index:
            order_diffs[sym] = -current_qty

    # Apply minimum order value filter (reduce commission on tiny trades)
    if MIN_ORDER_VAL > 0:
        filtered_diffs = {}
        skipped_count = 0
        skipped_value = 0.0
        for sym, qty in order_diffs.items():
            price = fmp_last_close.get(sym, 0)
            order_value = abs(qty) * price if price > 0 else 0
            if order_value >= MIN_ORDER_VAL:
                filtered_diffs[sym] = qty
            else:
                skipped_count += 1
                skipped_value += order_value
        log.info(f"  Min order filter (${MIN_ORDER_VAL:,}): skipped {skipped_count} orders (${skipped_value:,.0f})")
        order_diffs = filtered_diffs

    n_buys  = sum(1 for q in order_diffs.values() if q > 0)
    n_sells = sum(1 for q in order_diffs.values() if q < 0)
    total_shares = sum(abs(q) for q in order_diffs.values())
    log.info(f"  Orders: {len(order_diffs)} ({n_buys} buys, {n_sells} sells)")
    log.info(f"  Total shares: {total_shares:,}")

    # Capture pre-trade market snapshot (bid/ask/last/volume)
    if mode == "live" and ib_conn.connected:
        log.info("\n  Capturing pre-trade market snapshot (bid/ask/last/volume)...")
        all_order_tickers = list(order_diffs.keys())
        pretrade_snapshot = ib_conn.capture_pretrade_snapshot(all_order_tickers)
        save_pretrade_snapshot(today, pretrade_snapshot)

    # ── Phase 8: Submit MOC orders ────────────────────────────────────
    log.info("")
    if mode == "live" and ib_conn.connected:
        # Deadline check
        now_et = now_eastern()
        if now_et.time() > SUBMIT_DEADLINE and not force:
            log.error(f"  PAST SUBMIT DEADLINE ({SUBMIT_DEADLINE} ET; "
                      f"now={now_et.time().replace(microsecond=0)} ET) — ABORTING")
            ib_conn.disconnect()
            return
        if now_et.time() > SUBMIT_DEADLINE and force:
            log.warning(f"  PAST SUBMIT DEADLINE ({SUBMIT_DEADLINE} ET; "
                        f"now={now_et.time().replace(microsecond=0)} ET) -- "
                        "attempting because --force was supplied")

        log.info(f"Phase 8: Submitting {len(order_diffs)} MOC orders...")
        order_records = ib_conn.submit_moc_orders(order_diffs)
        accepted = sum(1 for r in order_records if not str(r.get("status", "")).startswith("FAILED"))
        log.info(f"  Accepted {accepted}/{len(order_records)} orders")
        ib_conn.ib.sleep(2)
    else:
        log.info("Phase 8: DRY-RUN -- Orders that WOULD be placed:")
        log.info(f"  {'Symbol':<8s} {'Action':<5s} {'Qty':>8s} {'$Value':>10s}")
        log.info(f"  {'-'*35}")

        sorted_diffs = sorted(order_diffs.items(),
                              key=lambda x: abs(x[1] * fmp_last_close.get(x[0], 0)),
                              reverse=True)
        for sym, qty in sorted_diffs[:30]:
            price = fmp_last_close.get(sym, 0)
            dollar = abs(qty * price)
            action = "BUY" if qty > 0 else "SELL"
            log.info(f"  {sym:<8s} {action:<5s} {abs(qty):>8,d} ${dollar:>9,.0f}")
        if len(sorted_diffs) > 30:
            log.info(f"  ... and {len(sorted_diffs) - 30} more")

    # ── Phase 9: Save trade log ───────────────────────────────────────
    log.info("\nPhase 9: Saving logs...")
    borrow_summary = {
        "n_checked": len(borrow_results),
        "n_shortable": sum(1 for r in borrow_results.values() if r.get("shortable")),
        "hard_to_borrow": [s for s, r in borrow_results.items() if not r.get("shortable")],
    } if borrow_results else {}

    save_trade_log(today, target_shares, current_positions, order_diffs,
                   order_records, account_summary, signal_date, mode,
                   borrow_summary, risk_messages, booksize)

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    log.info("")
    log.info("=" * 90)
    log.info(f"  COMPLETE in {elapsed:.1f}s | {mode.upper()}")
    log.info(f"  Signal:   {signal_date.date()} | GMV: ${target_dollar:,.0f}")
    log.info(f"  L/S:      ${long_dollar:,.0f} / ${short_dollar:,.0f}")
    log.info(f"  Positions: {len(target_shares)} ({n_long}L / {n_short}S)")
    log.info(f"  Orders:   {len(order_diffs)} ({n_buys} buy / {n_sells} sell)")
    if borrow_results:
        log.info(f"  Borrow:   {borrow_summary.get('n_shortable', 0)} OK, "
                f"{len(borrow_summary.get('hard_to_borrow', []))} HTB")
    log.info("=" * 90)

    if ib_conn.connected:
        ib_conn.disconnect()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Production IB MOC Trader — Closing Auction L/S Equity"
    )
    parser.add_argument("--live", action="store_true",
                        help="Connect to IB and submit MOC orders")
    parser.add_argument("--port", type=int, default=IB_PORT_PAPER,
                        help=f"IB port (default: {IB_PORT_PAPER}=paper, {IB_PORT_LIVE}=live)")
    parser.add_argument("--gmv", type=float, default=None,
                        help=f"Override target GMV (default: ${TARGET_GMV:,.0f})")
    parser.add_argument("--check-borrow", action="store_true",
                        help="Check borrow availability only (no trading)")
    parser.add_argument("--no-qp", action="store_true",
                        help="Disable QP optimization layer (use equal-weight "
                             "signal-as-portfolio, the pre-2026-05-01 behaviour)")
    parser.add_argument("--force", action="store_true",
                        help="Bypass the NYSE trading-day check (use for "
                             "manual / smoke tests on weekends or holidays)")
    args = parser.parse_args()

    mode = "live" if args.live else "dry-run"
    run_trading_workflow(
        mode=mode, port=args.port, gmv_override=args.gmv,
        check_borrow_only=args.check_borrow,
        no_qp=args.no_qp,
        force=args.force,
    )


if __name__ == "__main__":
    main()
