"""
prod/kucoin_trader.py — KuCoin Futures Paper Trader

Same architecture as binance_trader.py but adapted for KuCoin:
  - Different API endpoints
  - KuCoin VIP12 fees (1.5 bps taker, -0.5 bps maker rebate)
  - Same unified log format

Modes:
  python prod/kucoin_trader.py               # paper trade (default)
  python prod/kucoin_trader.py --live         # live trading (NOT IMPLEMENTED)
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

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG_PATH = Path(__file__).parent / "config" / "kucoin.json"

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)

CFG = load_config()

MATRICES_DIR = PROJECT_ROOT / CFG["paths"]["matrices"]
TICK_SIZES_PATH = PROJECT_ROOT / CFG["paths"]["tick_sizes"]
TRADE_LOG_DIR = PROJECT_ROOT / CFG["paths"]["trade_logs"]
PERF_LOG_DIR = PROJECT_ROOT / CFG["paths"]["performance_logs"]

for d in [TRADE_LOG_DIR, PERF_LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

UNIVERSE_SIZE = int(CFG["strategy"]["universe"].replace("TOP", ""))
TARGET_GMV = CFG["account"]["target_gmv"]
MAX_WEIGHT = CFG["strategy"]["max_position_weight"]
TAKER_BPS = CFG["fees"]["taker_bps"]
MIN_ORDER_VALUE = CFG["execution"]["min_order_value"]

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging():
    log_file = TRADE_LOG_DIR / f"kucoin_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ]
    )
    return logging.getLogger("kucoin_trader")

log = setup_logging()

# ============================================================================
# DATA / ALPHA / PORTFOLIO (identical to binance_trader.py)
# ============================================================================

def load_matrices() -> dict:
    matrices = {}
    for f in MATRICES_DIR.glob("*.parquet"):
        matrices[f.stem] = pd.read_parquet(f)
    return matrices

def evaluate_alphas(matrices: dict, universe_mask: pd.DataFrame) -> pd.DataFrame:
    """Evaluate all hardcoded alphas and combine using configured combiner method.
    
    Uses the shared combiner library (src/portfolio/combiners.py) — same code
    as the eval script, Binance trader, and IB pipeline.
    """
    from src.operators.fastexpression import FastExpressionEngine
    from src.portfolio.combiners import (
        combiner_equal, combiner_adaptive, combiner_risk_parity, combiner_billions
    )
    
    engine = FastExpressionEngine(data_fields=matrices)
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
            if "Unknown identifier" in err_msg or "not found" in err_msg.lower():
                field = err_msg.split("'")[1] if "'" in err_msg else "?"
                log.info(f"  Alpha {aid}: SKIP (missing field: {field})")
            else:
                log.warning(f"  Alpha {aid}: ERROR - {err_msg[:60]}")
    
    if not raw_signals:
        log.error("No alphas evaluated!")
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
# PAPER POSITIONS
# ============================================================================

POSITIONS_FILE = TRADE_LOG_DIR / "paper_positions.json"

def load_paper_positions() -> dict:
    if POSITIONS_FILE.exists():
        with open(POSITIONS_FILE) as f:
            return json.load(f)
    return {}

def save_paper_positions(positions: dict):
    with open(POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2)


def processed_bar_log(bar_time) -> Path | None:
    """Return an existing trade log for bar_time, if this bar already ran."""
    target = pd.Timestamp(bar_time)
    for path in sorted(TRADE_LOG_DIR.glob("trade_*.json"), reverse=True):
        try:
            with open(path) as f:
                data = json.load(f)
            logged_bar = data.get("bar_time")
            if logged_bar and pd.Timestamp(logged_bar) == target:
                return path
        except Exception:
            continue
    return None

# ============================================================================
# UNIFIED LOG (same format as Binance/IB)
# ============================================================================

def save_unified_trade_log(timestamp, bar_time, mode, target_portfolio,
                            current_positions, order_diffs, fills,
                            portfolio_summary, costs, n_alphas_evaluated):
    log_path = TRADE_LOG_DIR / f"trade_{timestamp.replace(':', '-')}.json"
    
    trade_log = {
        "exchange": "kucoin",
        "timestamp": timestamp,
        "bar_time": bar_time,
        "mode": mode,
        "version": CFG["strategy"]["version"],
        "config": {
            "target_gmv": TARGET_GMV,
            "universe": CFG["strategy"]["universe"],
            "taker_bps": TAKER_BPS,
        },
        "signal": {
            "n_alphas_evaluated": n_alphas_evaluated,
            "n_alphas_total": len(CFG["alphas"]),
        },
        "portfolio": portfolio_summary,
        "orders": {
            "n_orders": len(order_diffs),
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


def save_performance_snapshot(timestamp, portfolio, gmv, pnl):
    perf_file = PERF_LOG_DIR / "equity_kucoin.csv"
    if not perf_file.exists():
        with open(perf_file, "w") as f:
            f.write("timestamp,exchange,gmv,n_long,n_short,n_positions,pnl_bar,cumulative_pnl\n")
    
    try:
        existing = pd.read_csv(perf_file)
        last_cum = existing["cumulative_pnl"].iloc[-1] if len(existing) > 0 else 0.0
    except Exception:
        last_cum = 0.0
    
    n_long = sum(1 for v in portfolio.values() if v > 0)
    n_short = sum(1 for v in portfolio.values() if v < 0)
    
    with open(perf_file, "a") as f:
        f.write(f"{timestamp},kucoin,{gmv:.2f},{n_long},{n_short},"
                f"{len(portfolio)},{pnl:.2f},{last_cum + pnl:.2f}\n")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(live: bool = False):
    t0 = time.time()
    now = dt.datetime.utcnow()
    timestamp = now.isoformat()
    
    log.info("=" * 80)
    log.info(f"  KUCOIN TRADER v{CFG['strategy']['version']} — {now.strftime('%Y-%m-%d %H:%M')} UTC")
    log.info(f"  Mode: {'LIVE' if live else 'PAPER'} | Universe: {CFG['strategy']['universe']}")
    log.info(f"  GMV: ${TARGET_GMV:,.0f} | Fee: {TAKER_BPS} bps taker")
    log.info("=" * 80)
    
    # Phase 0: Refresh data (fetch latest bars from API + rebuild matrices)
    log.info("\nPhase 0: Refreshing data from KuCoin API...")
    try:
        from prod.data_refresh import refresh_kucoin
        latest_bar = refresh_kucoin()
        log.info(f"  Data refreshed. Latest bar: {latest_bar}")
    except Exception as e:
        log.warning(f"  [!] Data refresh failed: {e}")
        log.warning(f"  Continuing with cached data...")
    
    # Phase 1: Load data
    log.info("\nPhase 1: Loading matrices...")
    matrices = load_matrices()
    if not matrices:
        log.error("No matrices found! Run download_kucoin.py first.")
        return
    close = matrices["close"]
    log.info(f"  {len(matrices)} matrices, {close.shape[1]} tickers, {len(close)} bars")
    
    # Phase 2: Universe
    log.info("\nPhase 2: Building universe...")
    qv = matrices.get("quote_volume", matrices.get("turnover", matrices["volume"]))
    adv20 = qv.rolling(120, min_periods=60).mean()
    rank = adv20.rank(axis=1, ascending=False)
    universe_mask = rank <= UNIVERSE_SIZE
    log.info(f"  {universe_mask.iloc[-1].sum()} active tickers in {CFG['strategy']['universe']}")
    
    # Phase 3: Alphas
    log.info("\nPhase 3: Evaluating alphas...")
    signal_df = evaluate_alphas(matrices, universe_mask)
    if signal_df.empty:
        log.error("ABORT: No valid signal")
        return
    
    signal_row = signal_df.iloc[-1].dropna()
    signal_time = signal_df.index[-1]
    log.info(f"  Signal: {signal_time} | {(signal_row > 0).sum()}L / {(signal_row < 0).sum()}S")
    existing_log = processed_bar_log(signal_time)
    if existing_log is not None:
        log.warning(f"ABORT: bar {signal_time} already processed in {existing_log}")
        return
    
    # Phase 4: Target portfolio
    log.info("\nPhase 4: Target portfolio...")
    target_notional = (signal_row * TARGET_GMV).to_dict()
    target_notional = {k: v for k, v in target_notional.items() if abs(v) >= MIN_ORDER_VALUE}
    
    gmv_long = sum(v for v in target_notional.values() if v > 0)
    gmv_short = sum(abs(v) for v in target_notional.values() if v < 0)
    
    # Phase 5: Diffs
    current_positions = load_paper_positions()
    all_symbols = set(list(target_notional.keys()) + list(current_positions.keys()))
    order_diffs = {}
    for sym in all_symbols:
        diff = target_notional.get(sym, 0) - current_positions.get(sym, 0)
        if abs(diff) >= MIN_ORDER_VALUE:
            order_diffs[sym] = round(diff, 2)
    
    total_traded = sum(abs(v) for v in order_diffs.values())
    
    # Phase 6: Paper fills
    log.info("\nPhase 6: Paper fills...")
    fills = {}
    total_cost = 0
    tick_sizes = {}
    if TICK_SIZES_PATH.exists():
        with open(TICK_SIZES_PATH) as f:
            tick_sizes = json.load(f)
    
    last_close = close.iloc[-1]
    for sym, notional_diff in order_diffs.items():
        price = last_close.get(sym, 0)
        if price <= 0:
            continue
        tick = tick_sizes.get(sym, 0)
        tick_bps = tick / price * 10000 if price > 0 and tick > 0 else 3.0
        cost_bps = TAKER_BPS + tick_bps
        cost = abs(notional_diff) * cost_bps / 10000
        total_cost += cost
        fills[sym] = {
            "side": "BUY" if notional_diff > 0 else "SELL",
            "notional": round(abs(notional_diff), 2),
            "price": round(price, 6),
            "cost_bps": round(cost_bps, 2),
            "cost_usd": round(cost, 4),
        }
    
    save_paper_positions(target_notional)
    
    # PnL
    pnl = 0.0
    if current_positions:
        for sym, prev in current_positions.items():
            if sym in close.columns:
                prices = close[sym].dropna()
                if len(prices) >= 2:
                    pnl += prev * (prices.iloc[-1] / prices.iloc[-2] - 1)
    
    # Save logs
    log.info("\nPhase 7: Saving logs...")
    save_unified_trade_log(
        timestamp=timestamp, bar_time=str(signal_time), mode="PAPER",
        target_portfolio=target_notional, current_positions=current_positions,
        order_diffs=order_diffs, fills=fills,
        portfolio_summary={
            "n_long": sum(1 for v in target_notional.values() if v > 0),
            "n_short": sum(1 for v in target_notional.values() if v < 0),
            "gmv": round(gmv_long + gmv_short, 2),
        },
        costs={"total_traded": round(total_traded, 2), "total_cost": round(total_cost, 4)},
        n_alphas_evaluated=len([a for a in CFG["alphas"]]),
    )
    
    save_performance_snapshot(timestamp, target_notional, gmv_long + gmv_short, pnl - total_cost)
    
    elapsed = time.time() - t0
    log.info(f"\n  COMPLETE in {elapsed:.1f}s | {'LIVE' if live else 'PAPER'}")
    log.info(f"  GMV: ${gmv_long + gmv_short:,.0f} | PnL: ${pnl - total_cost:,.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KuCoin Futures Trader")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    
    try:
        run_pipeline(live=args.live)
    except Exception as e:
        log.error(f"FATAL: {e}", exc_info=True)
        sys.exit(1)
