"""
paper_trader.py — Main async WebSocket loop for V9b paper trading.

Connects to Binance Futures WebSocket, receives 15m klines for 5 symbols,
resamples to 1H/2H/4H/8H/12H, computes signals, and logs all trades.

Usage:
    python paper_trader.py
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import traceback
import sys
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import websockets
except ImportError:
    print("pip install websockets")
    sys.exit(1)

from config import SYMBOLS, WS_URL, LOG_DIR, FEE_BPS, FEE_FRAC
from kline_buffer import KlineBuffer
from signal_engine import SignalEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "paper_trader.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("paper_trader")

SIGNAL_LOG = LOG_DIR / "signals.jsonl"
TRADE_LOG = LOG_DIR / "trades.jsonl"
PNL_LOG = LOG_DIR / "pnl.csv"


# ---------------------------------------------------------------------------
# Position Tracker
# ---------------------------------------------------------------------------

class PositionTracker:
    """Tracks paper position, PnL, and trade history for one symbol."""

    def __init__(self, symbol):
        self.symbol = symbol
        self.direction = 0  # +1 long, -1 short, 0 flat
        self.entry_price = None
        self.entry_time = None
        self.cumulative_pnl_bps = 0.0
        self.n_trades = 0
        self.bars_in_position = 0

    def on_signal(self, timestamp, new_direction, close_price, signal_value, diag):
        """Process new signal. Returns trade record if position changed, else None."""
        trade_record = None

        if new_direction != self.direction:
            # Position change — log trade
            realized_pnl = 0.0
            if self.direction != 0 and self.entry_price:
                # Close previous position
                if self.direction == 1:
                    realized_pnl = (close_price / self.entry_price - 1) * 10000
                else:
                    realized_pnl = (1 - close_price / self.entry_price) * 10000

            fee = FEE_BPS * abs(new_direction - self.direction)
            net_pnl = realized_pnl - fee
            self.cumulative_pnl_bps += net_pnl
            self.n_trades += 1

            trade_record = {
                'timestamp': timestamp.isoformat(),
                'symbol': self.symbol,
                'old_pos': self.direction,
                'new_pos': new_direction,
                'close_price': close_price,
                'realized_pnl_bps': round(realized_pnl, 2),
                'fee_bps': fee,
                'net_pnl_bps': round(net_pnl, 2),
                'cumulative_pnl_bps': round(self.cumulative_pnl_bps, 2),
                'bars_in_position': self.bars_in_position,
                'signal_value': round(signal_value, 6),
            }

            self.direction = new_direction
            self.entry_price = close_price if new_direction != 0 else None
            self.entry_time = timestamp
            self.bars_in_position = 0
        else:
            self.bars_in_position += 1

        # Log signal regardless
        signal_record = {
            'timestamp': timestamp.isoformat(),
            'symbol': self.symbol,
            'close_price': close_price,
            'direction': new_direction,
            'signal_value': round(signal_value, 6),
            'position_changed': trade_record is not None,
            'cumulative_pnl_bps': round(self.cumulative_pnl_bps, 2),
            **{k: v for k, v in diag.items() if k not in ('top_alphas', 'top_signals')},
        }
        # Save top alphas as abbreviated
        if 'n_active_alphas' in diag:
            signal_record['n_active'] = diag['n_active_alphas']

        with open(SIGNAL_LOG, 'a') as f:
            f.write(json.dumps(signal_record) + '\n')

        if trade_record:
            with open(TRADE_LOG, 'a') as f:
                f.write(json.dumps(trade_record) + '\n')
            log.info(f"  TRADE {self.symbol}: {trade_record['old_pos']:+d} → "
                     f"{trade_record['new_pos']:+d} @ {close_price:.2f} | "
                     f"PnL={net_pnl:+.1f}bps | Cum={self.cumulative_pnl_bps:+.1f}bps")

        return trade_record


# ---------------------------------------------------------------------------
# Main Paper Trader
# ---------------------------------------------------------------------------

class PaperTrader:
    """Main orchestrator: WS connection, signal computation, trade logging."""

    def __init__(self):
        self.buffers = {}     # symbol → KlineBuffer
        self.engines = {}     # symbol → SignalEngine
        self.positions = {}   # symbol → PositionTracker
        self.running = True
        self._executor = ThreadPoolExecutor(max_workers=1)  # For CPU-heavy signal computation
        self._pending_1h = []  # Symbols waiting for 1H computation

    def initialize(self):
        """Load frozen params, create buffers, bootstrap from REST."""
        log.info("=" * 60)
        log.info("V9b PAPER TRADER — INITIALIZING")
        log.info("=" * 60)

        # Load frozen parameters
        frozen = SignalEngine.load_frozen_params()
        log.info(f"Loaded frozen params (version={frozen.get('version', '?')}, "
                 f"frozen_at={frozen.get('frozen_at', '?')})")

        for sym in SYMBOLS:
            if sym not in frozen.get('symbols', {}):
                log.warning(f"  {sym}: No frozen params, skipping!")
                continue

            # Create buffer
            buf = KlineBuffer(sym)
            buf.bootstrap_from_rest(n_1h_bars=720)
            self.buffers[sym] = buf

            # Create engine
            config = frozen['symbols'][sym]
            engine = SignalEngine(sym, config)
            self.engines[sym] = engine

            # Create position tracker
            self.positions[sym] = PositionTracker(sym)

            log.info(f"  {sym}: {buf.n_1h_bars} 1H bars | "
                     f"{config['n_alphas']} alphas | lb={config['lookback']} "
                     f"phl={config['phl']}")

        # Warmup adaptive weights from historical data
        log.info("Warming up adaptive weights from historical data...")
        all_1h_data = {s: b.get_1h() for s, b in self.buffers.items()}
        for sym in list(self.engines.keys()):
            buf = self.buffers[sym]
            engine = self.engines[sym]
            engine.warmup_from_history(
                buf.get_1h(), buf.get_2h(), buf.get_4h(),
                buf.get_8h(), buf.get_12h(), all_1h_data
            )

        log.info(f"\nInitialized {len(self.buffers)} symbols. Starting WS...")

    async def run(self):
        """Main async loop: connect WS, process klines, compute signals."""
        self.initialize()

        while self.running:
            try:
                async with websockets.connect(WS_URL, ping_interval=30,
                                              ping_timeout=60) as ws:
                    log.info(f"Connected to {WS_URL[:60]}...")
                    await self._process_messages(ws)
            except websockets.exceptions.ConnectionClosed as e:
                log.warning(f"WS disconnected: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                log.error(f"WS error: {e}\n{traceback.format_exc()}")
                log.info("Reconnecting in 10s...")
                await asyncio.sleep(10)

    async def _process_messages(self, ws):
        """Process incoming WebSocket messages."""
        async for raw_msg in ws:
            try:
                msg = json.loads(raw_msg)
                data = msg.get('data', msg)

                if data.get('e') != 'kline':
                    continue

                kline = data['k']
                symbol = kline['s']
                is_final = kline['x']

                if not is_final or symbol not in self.buffers:
                    continue

                # Process closed 15m bar
                buf = self.buffers[symbol]
                event = buf.on_kline_closed(kline)

                if event == '1h':
                    # 1H bar just completed — compute signals in thread
                    self._pending_1h.append(symbol)
                    # Check if all symbols for this hour are collected
                    # (they arrive within seconds of each other)
                    # Process after a short delay to batch them
                    asyncio.ensure_future(self._process_1h_batch())

            except Exception as e:
                log.error(f"Message processing error: {e}\n{traceback.format_exc()}")

    async def _process_1h_batch(self):
        """Wait briefly for all symbols' :45 bars to arrive, then compute."""
        await asyncio.sleep(2)  # Wait for all symbols' bars to arrive
        
        symbols_to_process = list(set(self._pending_1h))
        self._pending_1h.clear()
        
        if not symbols_to_process:
            return
        
        log.info(f"Computing signals for {len(symbols_to_process)} symbols...")
        
        # Run heavy computation in thread to avoid blocking event loop
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(self._executor, 
                                       self._compute_all_signals, symbols_to_process)
        except Exception as e:
            log.error(f"Signal computation error: {e}\n{traceback.format_exc()}")

    def _compute_all_signals(self, symbols):
        """Compute signals for all symbols (runs in thread pool)."""
        # Pre-build cross-asset data once
        all_1h_data = {s: b.get_1h() for s, b in self.buffers.items()}
        
        for symbol in symbols:
            try:
                self._compute_one_signal(symbol, all_1h_data)
            except Exception as e:
                log.error(f"Error computing {symbol}: {e}\n{traceback.format_exc()}")

    def _compute_one_signal(self, symbol, all_1h_data):
        """Compute signal for one symbol."""
        buf = self.buffers[symbol]
        engine = self.engines[symbol]
        pos = self.positions[symbol]

        # Get all timeframes
        df_1h = buf.get_1h()
        df_2h = buf.get_2h()
        df_4h = buf.get_4h()
        df_8h = buf.get_8h()
        df_12h = buf.get_12h()

        if len(df_1h) < 100:
            return

        # Compute signal
        t0 = time.time()
        direction, signal_val, diag = engine.compute_signal(
            df_1h, df_2h, df_4h, df_8h, df_12h, all_1h_data
        )
        elapsed = time.time() - t0

        close_price = buf.latest_close
        timestamp = df_1h.index[-1]

        # Log and track
        pos.on_signal(timestamp, direction, close_price, signal_val, diag)

        log.info(f"  {symbol}: dir={direction:+d} sig={signal_val:+.4f} "
                 f"price={close_price:.2f} ({elapsed:.1f}s)")

        # Periodic summary
        if engine.bar_count % 24 == 0:  # Every 24 hours
            log.info(f"  {symbol} 24h summary: pos={pos.direction:+d} "
                     f"cum_pnl={pos.cumulative_pnl_bps:+.1f}bps "
                     f"trades={pos.n_trades}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    trader = PaperTrader()
    try:
        asyncio.run(trader.run())
    except KeyboardInterrupt:
        log.info("Paper trader stopped by user.")
        # Print final summary
        log.info("\n" + "=" * 60)
        log.info("FINAL SUMMARY")
        log.info("=" * 60)
        for sym, pos in trader.positions.items():
            log.info(f"  {sym}: pos={pos.direction:+d} "
                     f"pnl={pos.cumulative_pnl_bps:+.1f}bps "
                     f"trades={pos.n_trades}")


if __name__ == '__main__':
    main()
