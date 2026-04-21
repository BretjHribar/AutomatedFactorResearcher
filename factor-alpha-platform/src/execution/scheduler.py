"""
scheduler.py — Daily Job Scheduler for IB Closing Auction Strategy.

Runs the daily alpha computation + MOC order submission pipeline
at 3:30 PM ET, Monday through Friday.

Usage:
    python -m src.execution.scheduler              # Run scheduler
    python -m src.execution.scheduler --once        # Run once immediately
    python -m src.execution.scheduler --paper       # Paper trading mode
"""

from __future__ import annotations

import argparse
import logging
import sys
import os
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# SCHEDULE CONFIGURATION
# ============================================================================

# Daily execution times (ET)
SCHEDULE_TIME = time(15, 30)    # 3:30 PM ET — start daily pipeline
MOC_SUBMIT_TIME = time(15, 45)  # 3:45 PM ET — submit MOC orders
MARKET_CLOSE_TIME = time(16, 0) # 4:00 PM ET — auction executes
RECONCILE_TIME = time(16, 5)    # 4:05 PM ET — reconcile fills


def is_market_day(dt: datetime) -> bool:
    """Check if the given date is a US market trading day.
    Simple weekday check — does not account for holidays.
    """
    return dt.weekday() < 5  # Mon-Fri


def run_daily_pipeline(paper: bool = True):
    """
    Execute the full daily closing auction pipeline.

    Steps:
        1. Load latest alpha signals
        2. Compute target portfolio
        3. Submit MOC orders at 3:45 PM ET
        4. Wait for auction (4:00 PM)
        5. Reconcile fills at 4:05 PM
    """
    from src.execution.ib_trader import (
        IBTrader,
        compute_target_portfolio,
        compute_trades,
        log_daily_report,
        DailyTradeReport,
    )

    today = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"{'='*60}")
    logger.info(f"IB CLOSING AUCTION PIPELINE — {today}")
    logger.info(f"Mode: {'PAPER' if paper else 'LIVE'}")
    logger.info(f"{'='*60}")

    # 1. Connect to IB
    port = 7497 if paper else 7496
    trader = IBTrader(port=port, paper=paper)
    if not trader.connect():
        logger.error("Failed to connect to IB. Aborting.")
        return

    try:
        # 2. Get account info
        summary = trader.get_account_summary()
        nlv = float(summary.get("NetLiquidation", 0))
        logger.info(f"Account NLV: ${nlv:,.0f}")

        # 3. Get current positions
        current = trader.get_current_positions()
        logger.info(f"Current positions: {len(current)}")

        # 4. Load alpha signals (from most recent evaluation)
        # In production, this would compute fresh signals from
        # the 3:30 PM intraday snapshot. For now, load from DB.
        logger.info("Loading alpha signals...")
        alpha_weights, prices = _load_latest_signals(trader)

        if alpha_weights is None or len(alpha_weights) == 0:
            logger.warning("No alpha signals available. Skipping trading.")
            return

        # 5. Compute target portfolio
        targets = compute_target_portfolio(alpha_weights, prices)
        logger.info(f"Target portfolio: {len(targets)} positions "
                     f"({sum(1 for t in targets if t.is_long)} long, "
                     f"{sum(1 for t in targets if not t.is_long)} short)")

        # 6. Compute trades
        trades = compute_trades(targets, current)
        logger.info(f"Trades needed: {len(trades)}")

        if not trades:
            logger.info("No trades needed. Portfolio is on target.")
            return

        # 7. Submit MOC orders
        logger.info("Submitting MOC orders...")
        results = trader.submit_moc_orders(trades)

        # 8. Create report
        report = DailyTradeReport(
            date=today,
            n_orders=len(results),
            n_fills=sum(1 for r in results if r.status == "Filled"),
            n_errors=sum(1 for r in results if r.status == "Error"),
            total_commission=sum(r.commission for r in results),
            gross_traded=sum(abs(r.shares * r.fill_price) for r in results if r.fill_price > 0),
            long_positions=sum(1 for t in targets if t.is_long),
            short_positions=sum(1 for t in targets if not t.is_long),
            gross_exposure=sum(t.notional for t in targets),
            net_exposure=sum(t.notional * (1 if t.is_long else -1) for t in targets),
            orders=results,
        )

        # 9. Log report
        log_daily_report(report)
        logger.info(f"Orders submitted: {report.n_orders}")
        logger.info(f"Errors: {report.n_errors}")
        logger.info(f"Gross exposure: ${report.gross_exposure:,.0f}")

    finally:
        trader.disconnect()


def _load_latest_signals(trader) -> tuple:
    """
    Load the latest alpha signals.

    In production, this computes fresh signals from the 3:30 PM
    intraday snapshot via IB TWS data feed. For initial deployment,
    it loads pre-computed signals from the alpha database.

    Returns:
        (alpha_weights: pd.Series, prices: pd.Series)
    """
    import pandas as pd

    # TODO: Replace with live signal computation from IB data feed
    # For now, return empty to safely skip trading
    logger.warning("Live signal computation not yet implemented.")
    logger.warning("Implement _load_latest_signals() with IB TWS data feed.")
    return None, pd.Series(dtype=float)


# ============================================================================
# MAIN
# ============================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="IB Closing Auction Scheduler")
    parser.add_argument("--once", action="store_true",
                        help="Run pipeline once immediately")
    parser.add_argument("--paper", action="store_true", default=True,
                        help="Use paper trading (default)")
    parser.add_argument("--live", action="store_true",
                        help="Use live trading (CAUTION)")
    args = parser.parse_args()

    paper = not args.live

    if args.once:
        logger.info("Running pipeline once...")
        run_daily_pipeline(paper=paper)
        return

    # Scheduled mode
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error("apscheduler not installed. Run: pip install apscheduler")
        logger.info("Falling back to --once mode.")
        run_daily_pipeline(paper=paper)
        return

    scheduler = BlockingScheduler()
    scheduler.add_job(
        run_daily_pipeline,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour=SCHEDULE_TIME.hour,
            minute=SCHEDULE_TIME.minute,
            timezone="US/Eastern",
        ),
        kwargs={"paper": paper},
        id="daily_moc_pipeline",
        name="IB Closing Auction MOC Pipeline",
    )

    mode = "PAPER" if paper else "LIVE"
    logger.info(f"Scheduler started ({mode} mode)")
    logger.info(f"Daily execution at {SCHEDULE_TIME.strftime('%I:%M %p')} ET, Mon-Fri")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
