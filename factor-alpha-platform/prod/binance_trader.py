"""
prod/binance_trader.py — Binance Futures Paper Trader

Architecture:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  4h Pipeline (every 4 hours UTC: 00,04,08,12,16,20)                   │
  │  1. Load cached 4h matrices                                           │
  │  2. Fetch latest 4h bar from Binance API                              │
  │  3. Build TOP50/TOP100 universe by ADV20                              │
  │  4. Evaluate hardcoded alpha signals (skip if field missing)          │
  │  5. Equal-weight combine -> market demean -> scale -> clip               │
  │  6. Convert to target notional positions                               │
  │  7. Compute order diffs vs current paper positions                     │
  │  8. Simulate fills at current market price (paper mode)                │
  │  9. Log everything in unified JSON format                              │
  └─────────────────────────────────────────────────────────────────────────┘

Modes:
  python prod/binance_trader.py              # paper trade (default)
  python prod/binance_trader.py --live       # live trading (NOT IMPLEMENTED YET)

Unified log format shared across IB, Binance, KuCoin.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG_PATH = Path(__file__).parent / "config" / "binance.json"

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)

CFG = load_config()

# Paths
MATRICES_DIR = PROJECT_ROOT / CFG["paths"]["matrices"]
TICK_SIZES_PATH = PROJECT_ROOT / CFG["paths"]["tick_sizes"]
TRADE_LOG_DIR = PROJECT_ROOT / CFG["paths"]["trade_logs"]
PERF_LOG_DIR = PROJECT_ROOT / CFG["paths"]["performance_logs"]

for d in [TRADE_LOG_DIR, PERF_LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Strategy params
UNIVERSE_SIZE = int(CFG["strategy"]["universe"].replace("TOP", ""))
TARGET_GMV = CFG["account"]["target_gmv"]
MAX_WEIGHT = CFG["strategy"]["max_position_weight"]
TAKER_BPS = CFG["fees"]["taker_bps"]
MIN_ORDER_VALUE = CFG["execution"]["min_order_value"]

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging():
    log_file = TRADE_LOG_DIR / f"binance_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ]
    )
    return logging.getLogger("binance_trader")

log = setup_logging()

# ============================================================================
# DATA LOADING
# ============================================================================

def load_matrices() -> dict:
    """Load all cached 4h matrices."""
    matrices = {}
    for f in MATRICES_DIR.glob("*.parquet"):
        matrices[f.stem] = pd.read_parquet(f)
    return matrices


def fetch_latest_bar(symbols: list[str]) -> Optional[pd.DataFrame]:
    """Fetch the latest 4h bar from Binance Futures API.
    Returns DataFrame with columns [open, high, low, close, volume, quote_volume].
    """
    try:
        rows = []
        # Use batch — fetch all at once via /fapi/v1/ticker/24hr
        r = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr", timeout=10)
        if r.status_code != 200:
            log.warning(f"Binance ticker API returned {r.status_code}")
            return None
        
        tickers = {t["symbol"]: t for t in r.json()}
        data = {}
        for sym in symbols:
            t = tickers.get(sym)
            if t:
                data[sym] = {
                    "close": float(t["lastPrice"]),
                    "volume": float(t["volume"]),
                    "quote_volume": float(t["quoteVolume"]),
                }
        
        log.info(f"  Fetched live prices for {len(data)}/{len(symbols)} symbols")
        return data
        
    except Exception as e:
        log.error(f"Failed to fetch Binance prices: {e}")
        return None


# ============================================================================
# ALPHA EVALUATION
# ============================================================================

def evaluate_alphas(matrices: dict, universe_mask: pd.DataFrame) -> pd.DataFrame:
    """Evaluate all hardcoded alphas and combine using configured combiner method.
    
    Uses the shared combiner library (src/portfolio/combiners.py) — same code
    as the eval script and IB pipeline.
    """
    from src.operators.fastexpression import FastExpressionEngine
    from src.portfolio.combiners import (
        combiner_equal, combiner_adaptive, combiner_risk_parity, combiner_billions
    )
    
    engine = FastExpressionEngine(data_fields=matrices)
    
    available_fields = set(matrices.keys())
    alphas = CFG["alphas"]
    
    # Step 1: Evaluate all raw alpha signals
    raw_signals = {}
    for alpha_cfg in alphas:
        aid = alpha_cfg["id"]
        expr = alpha_cfg["expression"]
        
        try:
            raw = engine.evaluate(expr)
            if raw is None or raw.empty:
                log.warning(f"  Alpha {aid}: EMPTY")
                continue
            
            raw_signals[aid] = raw
            log.info(f"  Alpha {aid}: OK ({raw.iloc[-1].notna().sum()} active)")
            
        except Exception as e:
            err_msg = str(e)
            if "Unknown identifier" in err_msg:
                field = err_msg.split("'")[1] if "'" in err_msg else "?"
                log.info(f"  Alpha {aid}: SKIP (missing field: {field})")
            else:
                log.warning(f"  Alpha {aid}: ERROR - {err_msg[:60]}")
    
    if not raw_signals:
        log.error("No alphas evaluated successfully!")
        return pd.DataFrame()
    
    log.info(f"  {len(raw_signals)}/{len(alphas)} alphas evaluated")
    
    # Step 2: Compute returns for combiners
    close = matrices["close"]
    returns_df = close.pct_change()
    uni_df = universe_mask.astype(bool)
    
    # Step 3: Run configured combiner
    combiner_name = CFG["strategy"].get("combiner", "risk_parity")
    log.info(f"  Combiner: {combiner_name} (max_wt={MAX_WEIGHT})")
    
    COMBINERS = {
        "equal_weight": lambda: combiner_equal(raw_signals, matrices, uni_df, returns_df, max_wt=MAX_WEIGHT),
        "risk_parity":  lambda: combiner_risk_parity(raw_signals, matrices, uni_df, returns_df, lookback=504, max_wt=MAX_WEIGHT),
        "adaptive":     lambda: combiner_adaptive(raw_signals, matrices, uni_df, returns_df, lookback=504, max_wt=MAX_WEIGHT),
        "billions":     lambda: combiner_billions(raw_signals, matrices, uni_df, returns_df, optim_lookback=60, max_wt=MAX_WEIGHT),
    }
    
    if combiner_name not in COMBINERS:
        log.warning(f"  Unknown combiner '{combiner_name}', falling back to risk_parity")
        combiner_name = "risk_parity"
    
    combined = COMBINERS[combiner_name]()
    
    # Final normalization (ensure unit exposure)
    sig_abs = combined.abs().sum(axis=1).replace(0, np.nan)
    combined = combined.div(sig_abs, axis=0)
    
    return combined


# ============================================================================
# PAPER POSITION TRACKER
# ============================================================================

POSITIONS_FILE = TRADE_LOG_DIR / "paper_positions.json"

def load_paper_positions() -> dict:
    """Load current paper positions from disk."""
    if POSITIONS_FILE.exists():
        with open(POSITIONS_FILE) as f:
            return json.load(f)
    return {}

def save_paper_positions(positions: dict):
    """Save current paper positions to disk."""
    with open(POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2)


# ============================================================================
# UNIFIED TRADE LOG
# ============================================================================

def save_unified_trade_log(
    exchange: str,
    timestamp: str,
    bar_time: str,
    mode: str,
    config_snapshot: dict,
    n_alphas_evaluated: int,
    target_portfolio: dict,
    current_positions: dict,
    order_diffs: dict,
    fills: dict,
    portfolio_summary: dict,
    costs: dict,
):
    """Save trade log in unified format readable by dashboard scripts."""
    log_path = TRADE_LOG_DIR / f"trade_{timestamp.replace(':', '-')}.json"
    
    trade_log = {
        "exchange": exchange,
        "timestamp": timestamp,
        "bar_time": bar_time,
        "mode": mode,
        "version": CFG["strategy"]["version"],
        
        "config": config_snapshot,
        
        "signal": {
            "n_alphas_evaluated": n_alphas_evaluated,
            "n_alphas_total": len(CFG["alphas"]),
            "universe": CFG["strategy"]["universe"],
            "neutralization": CFG["strategy"]["neutralization"],
        },
        
        "portfolio": portfolio_summary,
        
        "orders": {
            "n_orders": len(order_diffs),
            "n_buys": sum(1 for v in order_diffs.values() if v > 0),
            "n_sells": sum(1 for v in order_diffs.values() if v < 0),
            "diffs": order_diffs,
        },
        
        "fills": fills,
        "costs": costs,
        
        "positions": {
            "before": current_positions,
            "after": target_portfolio,
        },
    }
    
    with open(log_path, "w") as f:
        json.dump(trade_log, f, indent=2)
    log.info(f"  Trade log: {log_path}")
    return log_path


def save_performance_snapshot(
    exchange: str,
    timestamp: str,
    portfolio: dict,
    prices: dict,
    gmv: float,
    pnl_since_last: float,
):
    """Append one line to daily performance CSV for equity curve tracking."""
    perf_file = PERF_LOG_DIR / f"equity_{exchange}.csv"
    
    # Create header if new file
    if not perf_file.exists():
        with open(perf_file, "w") as f:
            f.write("timestamp,exchange,gmv,n_long,n_short,n_positions,pnl_bar,cumulative_pnl\n")
    
    # Read last cumulative PnL
    try:
        existing = pd.read_csv(perf_file)
        last_cum = existing["cumulative_pnl"].iloc[-1] if len(existing) > 0 else 0.0
    except Exception:
        last_cum = 0.0
    
    n_long = sum(1 for v in portfolio.values() if v > 0)
    n_short = sum(1 for v in portfolio.values() if v < 0)
    
    with open(perf_file, "a") as f:
        f.write(f"{timestamp},{exchange},{gmv:.2f},{n_long},{n_short},"
                f"{len(portfolio)},{pnl_since_last:.2f},{last_cum + pnl_since_last:.2f}\n")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(live: bool = False):
    """Run the full Binance paper trading pipeline."""
    t0 = time.time()
    now = dt.datetime.utcnow()
    timestamp = now.isoformat()
    
    log.info("=" * 80)
    log.info(f"  BINANCE TRADER v{CFG['strategy']['version']} — {now.strftime('%Y-%m-%d %H:%M')} UTC")
    log.info(f"  Mode: {'LIVE' if live else 'PAPER'} | Universe: {CFG['strategy']['universe']}")
    log.info(f"  GMV: ${TARGET_GMV:,.0f} | Fee: {TAKER_BPS} bps taker")
    log.info("=" * 80)
    
    # Phase 0: Refresh data (fetch latest bars from API + rebuild matrices)
    log.info("\nPhase 0: Refreshing data from Binance API...")
    try:
        from prod.data_refresh import refresh_binance
        latest_bar = refresh_binance()
        log.info(f"  Data refreshed. Latest bar: {latest_bar}")
    except Exception as e:
        log.warning(f"  [!] Data refresh failed: {e}")
        log.warning(f"  Continuing with cached data...")
    
    # Phase 1: Load matrices
    log.info("\nPhase 1: Loading data matrices...")
    matrices = load_matrices()
    log.info(f"  {len(matrices)} matrices loaded")
    close = matrices["close"]
    log.info(f"  Date range: {close.index[0]} -> {close.index[-1]}")
    log.info(f"  Tickers: {close.shape[1]}")
    
    # Phase 2: Build universe
    log.info("\nPhase 2: Building universe...")
    qv = matrices.get("quote_volume", matrices.get("dollars_traded", matrices["volume"]))
    adv20 = qv.rolling(120, min_periods=60).mean()
    rank = adv20.rank(axis=1, ascending=False)
    universe_mask = rank <= UNIVERSE_SIZE
    active_count = universe_mask.iloc[-1].sum()
    log.info(f"  {CFG['strategy']['universe']}: {active_count} active tickers")
    
    # Phase 3: Evaluate alphas
    log.info("\nPhase 3: Evaluating alpha signals...")
    signal_df = evaluate_alphas(matrices, universe_mask)
    
    if signal_df.empty:
        log.error("ABORT: No valid signal")
        return
    
    # Get last signal row
    signal_row = signal_df.iloc[-1].dropna()
    signal_time = signal_df.index[-1]
    log.info(f"  Signal date: {signal_time}")
    log.info(f"  Active positions: {len(signal_row)} ({(signal_row > 0).sum()}L / {(signal_row < 0).sum()}S)")
    
    # Phase 4: Convert to notional positions
    log.info("\nPhase 4: Target portfolio...")
    last_close = close.iloc[-1]
    target_notional = (signal_row * TARGET_GMV).to_dict()
    
    # Filter to min order value
    target_notional = {k: v for k, v in target_notional.items() if abs(v) >= MIN_ORDER_VALUE}
    
    gmv_long = sum(v for v in target_notional.values() if v > 0)
    gmv_short = sum(abs(v) for v in target_notional.values() if v < 0)
    log.info(f"  Target GMV: ${gmv_long + gmv_short:,.0f} (L: ${gmv_long:,.0f} / S: ${gmv_short:,.0f})")
    log.info(f"  Positions: {sum(1 for v in target_notional.values() if v > 0)}L / "
             f"{sum(1 for v in target_notional.values() if v < 0)}S")
    
    # Phase 5: Compute diffs
    log.info("\nPhase 5: Order diffs...")
    current_positions = load_paper_positions()
    
    all_symbols = set(list(target_notional.keys()) + list(current_positions.keys()))
    order_diffs = {}
    for sym in all_symbols:
        target = target_notional.get(sym, 0)
        current = current_positions.get(sym, 0)
        diff = target - current
        if abs(diff) >= MIN_ORDER_VALUE:
            order_diffs[sym] = round(diff, 2)
    
    n_buys = sum(1 for v in order_diffs.values() if v > 0)
    n_sells = sum(1 for v in order_diffs.values() if v < 0)
    total_traded = sum(abs(v) for v in order_diffs.values())
    log.info(f"  Orders: {len(order_diffs)} ({n_buys} buys / {n_sells} sells)")
    log.info(f"  Total traded: ${total_traded:,.0f}")
    
    # Phase 6: Simulate fills (paper mode)
    if live:
        log.info("\nPhase 6: LIVE ORDER SUBMISSION")
        log.error("Live Binance trading not implemented yet!")
        return
    else:
        log.info("\nPhase 6: Paper fill simulation...")
        fills = {}
        total_cost = 0
        
        # Load tick sizes for slippage
        tick_sizes = {}
        if TICK_SIZES_PATH.exists():
            with open(TICK_SIZES_PATH) as f:
                tick_sizes = json.load(f)
        
        for sym, notional_diff in order_diffs.items():
            price = last_close.get(sym, 0)
            if price <= 0:
                continue
            
            tick = tick_sizes.get(sym, 0)
            tick_bps = tick / price * 10000 if price > 0 and tick > 0 else 2.8
            
            # Cost = taker fee + 1 tick slippage
            cost_bps = TAKER_BPS + tick_bps
            cost = abs(notional_diff) * cost_bps / 10000
            total_cost += cost
            
            fill_price = price * (1 + (cost_bps / 10000 if notional_diff > 0 else -cost_bps / 10000))
            
            fills[sym] = {
                "side": "BUY" if notional_diff > 0 else "SELL",
                "notional": round(abs(notional_diff), 2),
                "price": round(price, 6),
                "fill_price": round(fill_price, 6),
                "cost_bps": round(cost_bps, 2),
                "cost_usd": round(cost, 4),
            }
        
        log.info(f"  Fills simulated: {len(fills)}")
        log.info(f"  Total cost: ${total_cost:,.2f} ({total_cost / total_traded * 10000:.1f} bps)" if total_traded > 0 else "")
        
        # Update paper positions
        save_paper_positions(target_notional)
        log.info(f"  Paper positions updated: {len(target_notional)} symbols")
    
    # Phase 7: Compute PnL (if we have previous positions)
    pnl = 0.0
    if current_positions:
        for sym, prev_notional in current_positions.items():
            prev_price_data = close[sym].dropna() if sym in close.columns else pd.Series()
            if len(prev_price_data) >= 2:
                ret = prev_price_data.iloc[-1] / prev_price_data.iloc[-2] - 1
                pnl += prev_notional * ret
    
    log.info(f"\n  Bar PnL (gross): ${pnl:,.2f}")
    log.info(f"  Bar PnL (net):   ${pnl - total_cost:,.2f}")
    
    # Phase 8: Save logs
    log.info("\nPhase 8: Saving logs...")
    
    n_alphas_ok = len([a for a in CFG["alphas"]])  # simplified
    
    save_unified_trade_log(
        exchange="binance",
        timestamp=timestamp,
        bar_time=str(signal_time),
        mode="PAPER" if not live else "LIVE",
        config_snapshot={
            "target_gmv": TARGET_GMV,
            "universe": CFG["strategy"]["universe"],
            "taker_bps": TAKER_BPS,
        },
        n_alphas_evaluated=n_alphas_ok,
        target_portfolio=target_notional,
        current_positions=current_positions,
        order_diffs=order_diffs,
        fills=fills,
        portfolio_summary={
            "n_long": sum(1 for v in target_notional.values() if v > 0),
            "n_short": sum(1 for v in target_notional.values() if v < 0),
            "gmv": round(gmv_long + gmv_short, 2),
            "gmv_long": round(gmv_long, 2),
            "gmv_short": round(gmv_short, 2),
            "net_exposure": round(gmv_long - gmv_short, 2),
        },
        costs={
            "total_traded": round(total_traded, 2),
            "total_cost": round(total_cost, 4),
            "avg_cost_bps": round(total_cost / total_traded * 10000, 2) if total_traded > 0 else 0,
        },
    )
    
    save_performance_snapshot(
        exchange="binance",
        timestamp=timestamp,
        portfolio=target_notional,
        prices=last_close.to_dict(),
        gmv=gmv_long + gmv_short,
        pnl_since_last=pnl - total_cost,
    )
    
    elapsed = time.time() - t0
    log.info(f"\n{'='*80}")
    log.info(f"  COMPLETE in {elapsed:.1f}s | {'LIVE' if live else 'PAPER'}")
    log.info(f"  Signal: {signal_time} | GMV: ${gmv_long + gmv_short:,.0f}")
    log.info(f"  L/S: ${gmv_long:,.0f} / ${gmv_short:,.0f}")
    log.info(f"  Positions: {len(target_notional)} | Orders: {len(order_diffs)}")
    log.info(f"  PnL: ${pnl - total_cost:,.2f} (net) | Cost: ${total_cost:,.2f}")
    log.info(f"{'='*80}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binance Futures Trader")
    parser.add_argument("--live", action="store_true", help="Live trading (NOT IMPLEMENTED)")
    args = parser.parse_args()
    
    try:
        run_pipeline(live=args.live)
    except Exception as e:
        log.error(f"FATAL: {e}", exc_info=True)
        sys.exit(1)
