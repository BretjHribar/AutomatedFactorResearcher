"""
ib_moc_trader.py — IB Closing Auction MOC Trading System

Daily workflow (schedule at 3:40 PM ET):
  1. Load FMP data (all history through T-1 close)
  2. Compute 13 alpha signals → Billions combiner → target weights
  3. Scale to target GMV → target shares per ticker
  4. Connect to IB (or dry-run if unavailable)
  5. Cross-validate FMP vs IB prices for T-1 close
  6. Query current IB positions
  7. Compute order diffs (target - current)
  8. Submit MOC orders
  9. Log everything to daily trade file

Modes:
  --dry-run    : No IB connection, output what WOULD be traded
  --live       : Connect to IB paper trading (port 7497) and execute
  --port PORT  : Override IB connection port (7496=live, 7497=paper)

Usage:
  python ib_moc_trader.py                    # dry-run (default)
  python ib_moc_trader.py --live             # paper trading via TWS
  python ib_moc_trader.py --live --port 7496 # live trading (DANGER)
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

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# CONFIGURATION
# ============================================================================

UNIVERSE       = "TOP2000TOP3000"
DB_PATH        = "data/ib_alphas.db"
NEUTRALIZE     = "market"
MAX_WEIGHT     = 0.01
BARS_PER_DAY   = 1
BILLIONS_LOOKBACK = 60

# Account sizing
ACCOUNT_EQUITY = 110_000.0      # Seed capital
TARGET_GMV     = 500_000.0      # Gross Market Value ($250k long + $250k short)
BOOKSIZE       = TARGET_GMV     # Scales signal to this GMV

# IB connection defaults
IB_HOST       = "127.0.0.1"
IB_PORT_PAPER = 7497
IB_PORT_LIVE  = 7496
IB_CLIENT_ID  = 10          # Unique to avoid collisions with other tools

# Directories
MATRICES_DIR  = Path("data/fmp_cache/matrices")
UNIVERSES_DIR = Path("data/fmp_cache/universes")
LOG_DIR       = Path("logs/trades")

# Order type
ORDER_TYPE_MOC = "MOC"
ORDER_TYPE_LOC = "LOC"
DEFAULT_ORDER_TYPE = ORDER_TYPE_MOC

# MOC cutoff deadlines (safety buffer)
MOC_DEADLINE_ET = dt.time(15, 48, 0)   # 3:48 PM ET — hard stop, 2 min before NYSE cutoff

# Set up logging
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"moc_trader_{dt.date.today().isoformat()}.log"),
    ]
)
log = logging.getLogger("moc_trader")


# ============================================================================
# DATA LOADING (reuses eval_alpha_ib pipeline)
# ============================================================================

def load_fmp_data():
    """Load all FMP matrices, universe, and classifications."""
    import eval_alpha_ib
    eval_alpha_ib.UNIVERSE = UNIVERSE
    eval_alpha_ib.NEUTRALIZE = NEUTRALIZE
    matrices, universe, classifications = eval_alpha_ib.load_data("full")
    valid_tickers = universe.columns.tolist()

    log.info(f"FMP data: {len(matrices)} fields, {len(valid_tickers)} tickers")
    log.info(f"Date range: {matrices['close'].index[0].date()} to {matrices['close'].index[-1].date()}")
    return matrices, universe, classifications, valid_tickers


def load_alphas_from_db():
    """Load all non-archived IB alphas from database."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT a.id, a.expression, COALESCE(e.ic_mean, 0), COALESCE(e.sharpe_is, 0)
        FROM alphas a LEFT JOIN evaluations e ON e.alpha_id = a.id
        WHERE a.archived = 0 AND a.asset_class = 'equities_ib'
        ORDER BY COALESCE(e.sharpe_is, 0) DESC
    """).fetchall()
    conn.close()
    log.info(f"Loaded {len(rows)} alphas from {DB_PATH}")
    return rows


# ============================================================================
# SIGNAL COMPUTATION (mirrors run_ib_portfolio exactly)
# ============================================================================

def evaluate_expression(expression, matrices):
    from src.operators.fastexpression import FastExpressionEngine
    engine = FastExpressionEngine(data_fields=matrices)
    return engine.evaluate(expression)


def process_signal(alpha_df, universe_df=None, max_wt=MAX_WEIGHT):
    """Normalize signal: universe mask → demean → unit-scale → clip."""
    signal = alpha_df.copy()
    if universe_df is not None:
        uni_mask = universe_df.reindex(index=signal.index, columns=signal.columns).fillna(False)
        signal = signal.where(uni_mask, np.nan)
    mean_val = signal.mean(axis=1)
    signal = signal.sub(mean_val, axis=0)
    abs_sum = signal.abs().sum(axis=1).replace(0, np.nan)
    signal = signal.div(abs_sum, axis=0)
    signal = signal.clip(lower=-max_wt, upper=max_wt)
    return signal.fillna(0.0)


def compute_billions_signal(alpha_signals, matrices, universe_df, returns_df):
    """
    Run Billions walk-forward combiner and return the LAST ROW of combined signal.
    Returns: pd.Series (ticker → weight), date of signal
    """
    from sklearn import linear_model

    close_df = matrices["close"]
    dates    = close_df.index
    tickers  = close_df.columns.tolist()
    n_bars   = len(dates)
    aid_list = list(alpha_signals.keys())
    n_alphas = len(aid_list)
    optim_lookback = BILLIONS_LOOKBACK

    # Normalize each alpha
    normed_signals = {
        aid: process_signal(raw, universe_df=universe_df, max_wt=MAX_WEIGHT)
        for aid, raw in alpha_signals.items()
    }

    # Factor returns
    ret_df = returns_df.reindex(index=dates, columns=tickers)
    fr_data = {}
    for aid, norm in normed_signals.items():
        lagged = norm.shift(1)
        ab = lagged.abs().sum(axis=1).replace(0, np.nan)
        n  = lagged.div(ab, axis=0)
        fr_data[aid] = (n * ret_df).sum(axis=1)

    fr_df = pd.DataFrame(fr_data, index=dates)

    # Expected returns
    alphas_exp_ret = (
        fr_df.rolling(window=optim_lookback, min_periods=max(1, optim_lookback // 2))
             .mean()
             .shift(1)
    )
    alphas_exp_ret = alphas_exp_ret.clip(lower=0)

    # Walk-forward
    alpha_weights_ts = pd.DataFrame(1.0 / n_alphas, index=dates, columns=aid_list)
    reg = linear_model.LinearRegression(fit_intercept=False)

    for test_start in range(1, n_bars - optim_lookback - 2):
        optim_end = test_start + optim_lookback
        if optim_end + 1 >= n_bars:
            break
        try:
            bil_alphas_df = fr_df.iloc[test_start:optim_end].copy()
            bil_demeaned = bil_alphas_df - bil_alphas_df.mean(axis=0)
            sample_std = bil_demeaned.std(axis=0).replace(0, np.nan)
            normalized = bil_demeaned.divide(sample_std)

            Y_is = normalized.iloc[:, :optim_lookback]
            A_is = Y_is

            sub_exp_ret = alphas_exp_ret.iloc[test_start:optim_end].copy()
            sub_exp_ret = sub_exp_ret.divide(sample_std).fillna(0.0)

            X_train = A_is.fillna(0.0).values
            Y_train = sub_exp_ret.values
            reg.fit(X_train, Y_train)

            residuals_vals = reg.predict(X_train) - Y_train
            residuals = pd.DataFrame(residuals_vals, index=sub_exp_ret.index, columns=sub_exp_ret.columns)

            opt_weights = residuals.divide(sample_std)
            row_sums = opt_weights.sum(axis=1).replace(0, np.nan)
            opt_weights = opt_weights.div(row_sums, axis=0)

            final_weights = opt_weights.iloc[-1]
            alpha_weights_ts.iloc[optim_end + 1] = final_weights.values
        except Exception:
            pass

    # Combine using last row of weights
    combined = pd.DataFrame(0.0, index=dates, columns=tickers)
    for aid in aid_list:
        w = alpha_weights_ts[aid]
        combined = combined.add(normed_signals[aid].mul(w, axis=0))

    # Return last row as Series
    last_date = dates[-1]
    signal_row = combined.iloc[-1]
    return signal_row, last_date


def signal_to_target_shares(signal_row, close_prices, booksize=BOOKSIZE):
    """
    Convert normalized signal weights to target share counts.

    signal_row: pd.Series (ticker → weight, sums to ~0 for L/S)
    close_prices: pd.Series (ticker → last close price)
    booksize: total GMV

    Returns: pd.Series (ticker → target_shares, int)
    """
    # Scale weights to dollar positions
    dollar_positions = signal_row * booksize

    # Convert to shares
    valid_prices = close_prices.reindex(signal_row.index).replace(0, np.nan)
    target_shares = (dollar_positions / valid_prices).fillna(0)

    # Round to integers
    target_shares = target_shares.round().astype(int)

    # Drop zeros
    target_shares = target_shares[target_shares != 0]

    return target_shares


# ============================================================================
# IB CONNECTION & TRADING
# ============================================================================

class IBConnection:
    """Wrapper for IB connection via ib_insync."""

    def __init__(self, host=IB_HOST, port=IB_PORT_PAPER, client_id=IB_CLIENT_ID):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None
        self.connected = False

    def connect(self) -> bool:
        """Attempt to connect to TWS/Gateway. Returns True if successful."""
        try:
            from ib_insync import IB
            self.ib = IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.connected = True
            accts = self.ib.managedAccounts()
            log.info(f"Connected to IB on {self.host}:{self.port}")
            log.info(f"  Accounts: {accts}")
            log.info(f"  Paper trading: {'DU' in str(accts) or self.port == 7497}")
            return True
        except Exception as e:
            log.warning(f"Failed to connect to IB: {e}")
            self.connected = False
            return False

    def disconnect(self):
        if self.ib and self.connected:
            self.ib.disconnect()
            self.connected = False
            log.info("Disconnected from IB")

    def get_account_summary(self) -> dict:
        """Get key account metrics."""
        if not self.connected:
            return {}
        summary = {}
        for item in self.ib.accountSummary():
            if item.tag in ("NetLiquidation", "GrossPositionValue", "BuyingPower",
                            "TotalCashValue", "MaintMarginReq", "AvailableFunds"):
                summary[item.tag] = float(item.value)
        return summary

    def get_positions(self) -> dict[str, int]:
        """Get current positions as {ticker: shares}."""
        if not self.connected:
            return {}
        positions = {}
        for pos in self.ib.positions():
            sym = pos.contract.symbol
            qty = int(pos.position)
            if qty != 0:
                positions[sym] = qty
        return positions

    def get_last_close_prices(self, tickers: list[str]) -> dict[str, float]:
        """
        Get yesterday's close prices from IB for cross-validation with FMP.
        Uses reqHistoricalData for daily bar.
        """
        if not self.connected:
            return {}

        from ib_insync import Stock
        prices = {}
        for sym in tickers[:50]:  # Sample 50 for validation (avoid rate limits)
            try:
                contract = Stock(sym, "SMART", "USD")
                bars = self.ib.reqHistoricalData(
                    contract, endDateTime="",
                    durationStr="2 D", barSizeSetting="1 day",
                    whatToShow="ADJUSTED_LAST", useRTH=True
                )
                if bars and len(bars) > 0:
                    prices[sym] = bars[-1].close
                self.ib.sleep(0.1)
            except Exception as e:
                log.debug(f"  Price fetch failed for {sym}: {e}")
        return prices

    def submit_moc_orders(self, order_diffs: dict[str, int],
                          order_type: str = DEFAULT_ORDER_TYPE) -> list[dict]:
        """
        Submit MOC (or LOC) orders for the given diffs.

        order_diffs: {ticker: shares_to_trade} (positive=buy, negative=sell)
        Returns: list of order records
        """
        if not self.connected:
            return []

        from ib_insync import Stock, MarketOrder, Order

        order_records = []
        for sym, shares in order_diffs.items():
            if shares == 0:
                continue

            try:
                contract = Stock(sym, "SMART", "USD")
                self.ib.qualifyContracts(contract)

                action = "BUY" if shares > 0 else "SELL"
                qty = abs(shares)

                if order_type == ORDER_TYPE_MOC:
                    order = Order(
                        action=action,
                        totalQuantity=qty,
                        orderType="MOC",
                        tif="DAY",
                    )
                elif order_type == ORDER_TYPE_LOC:
                    # LOC needs a limit price — use a wide limit for now
                    order = Order(
                        action=action,
                        totalQuantity=qty,
                        orderType="LOC",
                        lmtPrice=0,  # Will be set by market
                        tif="DAY",
                    )
                else:
                    order = Order(
                        action=action,
                        totalQuantity=qty,
                        orderType="MOC",
                        tif="DAY",
                    )

                trade = self.ib.placeOrder(contract, order)
                record = {
                    "symbol": sym,
                    "action": action,
                    "quantity": qty,
                    "order_type": order_type,
                    "order_id": trade.order.orderId,
                    "status": "SUBMITTED",
                    "timestamp": dt.datetime.now().isoformat(),
                }
                order_records.append(record)
                log.info(f"  ORDER: {action} {qty} {sym} {order_type} (id={trade.order.orderId})")

            except Exception as e:
                log.error(f"  FAILED to place order for {sym}: {e}")
                order_records.append({
                    "symbol": sym, "action": "BUY" if shares > 0 else "SELL",
                    "quantity": abs(shares), "order_type": order_type,
                    "status": f"FAILED: {e}",
                    "timestamp": dt.datetime.now().isoformat(),
                })

        return order_records


# ============================================================================
# CROSS-VALIDATION: FMP vs IB PRICES
# ============================================================================

def cross_validate_prices(fmp_close: pd.Series, ib_prices: dict[str, float],
                          tolerance_pct: float = 2.0) -> dict:
    """
    Compare FMP T-1 close prices vs IB T-1 close prices.
    Flags mismatches > tolerance_pct which may indicate split/div differences.

    Returns: {matched: int, mismatched: int, details: [...]}
    """
    results = {"matched": 0, "mismatched": 0, "details": []}

    for sym, ib_price in ib_prices.items():
        fmp_price = fmp_close.get(sym, np.nan)
        if np.isnan(fmp_price) or fmp_price == 0 or ib_price == 0:
            continue

        pct_diff = abs(fmp_price - ib_price) / fmp_price * 100

        if pct_diff > tolerance_pct:
            results["mismatched"] += 1
            results["details"].append({
                "symbol": sym,
                "fmp_price": round(fmp_price, 2),
                "ib_price": round(ib_price, 2),
                "pct_diff": round(pct_diff, 2),
                "likely_cause": "SPLIT" if pct_diff > 30 else "DIV_ADJUSTMENT",
            })
            log.warning(f"  PRICE MISMATCH: {sym} FMP=${fmp_price:.2f} vs IB=${ib_price:.2f} "
                        f"({pct_diff:.1f}% diff)")
        else:
            results["matched"] += 1

    log.info(f"Price validation: {results['matched']} matched, "
             f"{results['mismatched']} mismatched (>{tolerance_pct}%)")
    return results


# ============================================================================
# TRADE LOGGING
# ============================================================================

def save_trade_log(date: dt.date, target_portfolio: pd.Series,
                   current_positions: dict, order_diffs: dict,
                   order_records: list, account_summary: dict,
                   signal_date: pd.Timestamp, mode: str,
                   target_gmv: float = TARGET_GMV):
    """Save comprehensive daily trade log as JSON."""
    log_path = LOG_DIR / f"trade_{date.isoformat()}.json"

    trade_log = {
        "date": date.isoformat(),
        "mode": mode,
        "signal_date": str(signal_date.date()),
        "timestamp": dt.datetime.now().isoformat(),

        "account": account_summary,
        "config": {
            "target_gmv": target_gmv,
            "account_equity": ACCOUNT_EQUITY,
            "universe": UNIVERSE,
            "combiner": "Billions",
            "lookback": BILLIONS_LOOKBACK,
            "order_type": DEFAULT_ORDER_TYPE,
        },

        "portfolio_summary": {
            "n_long":  int((target_portfolio > 0).sum()),
            "n_short": int((target_portfolio < 0).sum()),
            "gross_shares": int(target_portfolio.abs().sum()),
            "net_shares": int(target_portfolio.sum()),
        },

        "n_orders": len(order_diffs),
        "n_new_positions": sum(1 for s in order_diffs if s not in current_positions),
        "n_closes": sum(1 for s, q in order_diffs.items()
                        if s in current_positions and
                        current_positions[s] + q == 0),

        "target_portfolio": {k: int(v) for k, v in target_portfolio.items()},
        "current_positions": current_positions,
        "order_diffs": order_diffs,
        "order_records": order_records,
    }

    with open(log_path, "w") as f:
        json.dump(trade_log, f, indent=2)
    log.info(f"Trade log saved: {log_path}")
    return log_path


# ============================================================================
# MAIN TRADING WORKFLOW
# ============================================================================

def run_trading_workflow(mode: str = "dry-run", port: int = IB_PORT_PAPER,
                         order_type: str = DEFAULT_ORDER_TYPE,
                         gmv_override: float = None):
    """
    Main daily trading workflow.

    Modes:
      'dry-run'  — compute everything, log orders, but don't connect to IB
      'live'     — connect to IB and submit orders
    """
    today = dt.date.today()
    t0 = time.time()

    booksize = gmv_override if gmv_override else TARGET_GMV
    target_gmv = booksize

    log.info("=" * 80)
    log.info(f"IB MOC TRADER — {today.isoformat()} — Mode: {mode.upper()}")
    log.info(f"GMV: ${target_gmv:,.0f} | Universe: {UNIVERSE} | Order type: {order_type}")
    log.info("=" * 80)

    # ── Step 1: Load FMP data ────────────────────────────────────────────
    log.info("")
    log.info("Step 1: Loading FMP data...")
    matrices, universe_df, classifications, valid_tickers = load_fmp_data()
    close_df = matrices["close"]
    returns_df = matrices["returns"]

    last_fmp_date = close_df.index[-1]
    fmp_last_close = close_df.iloc[-1]
    log.info(f"  Last FMP date: {last_fmp_date.date()}")
    log.info(f"  Valid tickers: {len(valid_tickers)}")

    # ── Step 2: Load alphas and compute signals ──────────────────────────
    log.info("")
    log.info("Step 2: Loading alpha signals...")
    alphas = load_alphas_from_db()

    alpha_signals = {}
    for aid, expr, ic, sr in alphas:
        try:
            raw = evaluate_expression(expr, matrices)
            if raw is not None and not raw.empty:
                alpha_signals[aid] = raw
                log.info(f"  Alpha #{aid}: IS SR={sr:+.2f}, IC={ic:+.4f}")
        except Exception as e:
            log.warning(f"  Alpha #{aid}: FAILED ({e})")

    log.info(f"  Loaded {len(alpha_signals)}/{len(alphas)} alpha signals")

    # ── Step 3: Billions combiner → target weights ───────────────────────
    log.info("")
    log.info("Step 3: Computing Billions combined signal...")
    t_signal = time.time()
    signal_row, signal_date = compute_billions_signal(
        alpha_signals, matrices, universe_df, returns_df
    )
    log.info(f"  Booksize: ${booksize:,.0f}")
    log.info(f"  Signal computed in {time.time() - t_signal:.1f}s")
    log.info(f"  Signal date: {signal_date.date()}")

    # Signal stats
    n_long  = (signal_row > 1e-6).sum()
    n_short = (signal_row < -1e-6).sum()
    log.info(f"  Positions: {n_long} long, {n_short} short")
    log.info(f"  Top 5 longs:  {signal_row.nlargest(5).to_dict()}")
    log.info(f"  Top 5 shorts: {signal_row.nsmallest(5).to_dict()}")

    # ── Step 4: Convert to target shares ─────────────────────────────────
    log.info("")
    log.info("Step 4: Converting to target shares...")
    target_shares = signal_to_target_shares(signal_row, fmp_last_close, booksize=booksize)

    # Compute actual GMV
    target_dollar = (target_shares.abs() * fmp_last_close.reindex(target_shares.index)).sum()
    long_dollar = (target_shares[target_shares > 0] *
                   fmp_last_close.reindex(target_shares[target_shares > 0].index)).sum()
    short_dollar = (target_shares[target_shares < 0].abs() *
                    fmp_last_close.reindex(target_shares[target_shares < 0].index)).sum()

    log.info(f"  Target GMV:  ${target_dollar:,.0f}")
    log.info(f"  Long:        ${long_dollar:,.0f} ({(target_shares > 0).sum()} positions)")
    log.info(f"  Short:       ${short_dollar:,.0f} ({(target_shares < 0).sum()} positions)")
    log.info(f"  Total positions: {len(target_shares)}")
    log.info(f"  Avg position: ${target_dollar / max(len(target_shares), 1):,.0f}")

    # ── Step 5: IB connection ────────────────────────────────────────────
    ib_conn = IBConnection(host=IB_HOST, port=port, client_id=IB_CLIENT_ID)
    current_positions = {}
    account_summary = {}
    order_records = []
    price_validation = {}

    if mode == "live":
        log.info("")
        log.info("Step 5: Connecting to IB...")
        connected = ib_conn.connect()

        if connected:
            # Account info
            account_summary = ib_conn.get_account_summary()
            if account_summary:
                log.info(f"  Net Liquidation: ${account_summary.get('NetLiquidation', 0):,.0f}")
                log.info(f"  Gross Position:  ${account_summary.get('GrossPositionValue', 0):,.0f}")
                log.info(f"  Buying Power:    ${account_summary.get('BuyingPower', 0):,.0f}")

            # Current positions
            current_positions = ib_conn.get_positions()
            log.info(f"  Current positions: {len(current_positions)} symbols")

            # Cross-validate prices (sample 50 tickers)
            log.info("")
            log.info("Step 5b: Cross-validating FMP vs IB prices...")
            sample_tickers = list(target_shares.index[:50])
            ib_prices = ib_conn.get_last_close_prices(sample_tickers)
            if ib_prices:
                price_validation = cross_validate_prices(fmp_last_close, ib_prices)
                if price_validation["mismatched"] > 5:
                    log.error(f"  HIGH MISMATCH RATE: {price_validation['mismatched']} tickers "
                              f"— possible split/div issue. Review before proceeding.")
        else:
            log.warning("  IB not available — falling back to dry-run mode")
            mode = "dry-run"
    else:
        log.info("")
        log.info("Step 5: DRY-RUN mode — skipping IB connection")

    # ── Step 6: Compute order diffs ──────────────────────────────────────
    log.info("")
    log.info("Step 6: Computing order diffs...")
    order_diffs = {}
    for sym, target_qty in target_shares.items():
        current_qty = current_positions.get(sym, 0)
        diff = target_qty - current_qty
        if diff != 0:
            order_diffs[sym] = diff

    # Also close any positions we no longer want
    for sym, current_qty in current_positions.items():
        if sym not in target_shares.index:
            order_diffs[sym] = -current_qty

    n_buys  = sum(1 for q in order_diffs.values() if q > 0)
    n_sells = sum(1 for q in order_diffs.values() if q < 0)
    total_shares = sum(abs(q) for q in order_diffs.values())

    log.info(f"  Orders to place: {len(order_diffs)} ({n_buys} buys, {n_sells} sells)")
    log.info(f"  Total shares to trade: {total_shares:,}")

    # Estimate commission
    est_comm = total_shares * 0.0015  # Tiered 3M-20M rate
    log.info(f"  Est. commission: ${est_comm:,.2f}")

    # ── Step 7: Submit orders (or log for dry-run) ───────────────────────
    log.info("")
    if mode == "live" and ib_conn.connected:
        # Check MOC deadline
        now_et = dt.datetime.now()  # Assumes running in ET timezone
        if now_et.time() > MOC_DEADLINE_ET:
            log.error(f"  PAST MOC DEADLINE ({MOC_DEADLINE_ET}) — ABORTING")
            ib_conn.disconnect()
            return

        log.info(f"Step 7: Submitting {len(order_diffs)} MOC orders...")
        order_records = ib_conn.submit_moc_orders(order_diffs, order_type=order_type)
        log.info(f"  Submitted {len(order_records)} orders")

        # Wait briefly for order status updates
        ib_conn.ib.sleep(2)
    else:
        log.info("Step 7: DRY-RUN — Orders that WOULD be placed:")
        log.info(f"  {'Symbol':<8s} {'Action':<5s} {'Qty':>8s} {'$Value':>10s}")
        log.info(f"  {'-'*35}")

        # Sort by absolute dollar value
        sorted_diffs = sorted(order_diffs.items(),
                              key=lambda x: abs(x[1] * fmp_last_close.get(x[0], 0)),
                              reverse=True)

        for sym, qty in sorted_diffs[:30]:  # Show top 30
            price = fmp_last_close.get(sym, 0)
            dollar = abs(qty * price)
            action = "BUY" if qty > 0 else "SELL"
            log.info(f"  {sym:<8s} {action:<5s} {abs(qty):>8,d} ${dollar:>9,.0f}")

        if len(sorted_diffs) > 30:
            log.info(f"  ... and {len(sorted_diffs) - 30} more orders")

    # ── Step 8: Save trade log ───────────────────────────────────────────
    log.info("")
    log.info("Step 8: Saving trade log...")
    save_trade_log(
        date=today,
        target_portfolio=target_shares,
        current_positions=current_positions,
        order_diffs=order_diffs,
        order_records=order_records,
        account_summary=account_summary,
        signal_date=signal_date,
        mode=mode,
        target_gmv=target_gmv,
    )

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    log.info("")
    log.info("=" * 80)
    log.info(f"  COMPLETE in {elapsed:.1f}s")
    log.info(f"  Mode:           {mode}")
    log.info(f"  Signal date:    {signal_date.date()}")
    log.info(f"  Target GMV:     ${target_dollar:,.0f}")
    log.info(f"  Long/Short:     ${long_dollar:,.0f} / ${short_dollar:,.0f}")
    log.info(f"  Positions:      {len(target_shares)} ({n_long} L / {n_short} S)")
    log.info(f"  Orders:         {len(order_diffs)} ({n_buys} buy / {n_sells} sell)")
    log.info(f"  Shares:         {total_shares:,}")
    log.info(f"  Est. commission: ${est_comm:,.2f}")
    log.info("=" * 80)

    if ib_conn.connected:
        ib_conn.disconnect()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="IB MOC Trader — Daily Closing Auction Trading System"
    )
    parser.add_argument("--live", action="store_true",
                        help="Connect to IB and submit real orders (default: dry-run)")
    parser.add_argument("--port", type=int, default=IB_PORT_PAPER,
                        help=f"IB connection port (default: {IB_PORT_PAPER} = paper trading)")
    parser.add_argument("--order-type", choices=["MOC", "LOC"], default="MOC",
                        help="Order type (default: MOC)")
    parser.add_argument("--gmv", type=float, default=TARGET_GMV,
                        help=f"Target Gross Market Value (default: ${TARGET_GMV:,.0f})")
    args = parser.parse_args()

    mode = "live" if args.live else "dry-run"
    run_trading_workflow(mode=mode, port=args.port, order_type=args.order_type,
                         gmv_override=args.gmv)


if __name__ == "__main__":
    main()
