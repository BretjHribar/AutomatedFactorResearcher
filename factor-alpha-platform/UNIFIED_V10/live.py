"""
Live Paper Trader using the Unified Streaming Engine.

SAME ENGINE as backtest — by construction, no divergence possible.

Architecture:
  1. Connects to Binance Futures WebSocket for 15m klines
  2. Aggregates 15m bars into 1H bars
  3. Feeds closed 1H bars to StreamingEngine.on_bar()
  4. Logs signals to JSON files (read by MONITOR dashboard)

Usage:
    python -m UNIFIED_V10.live
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import json
import time
import logging
import asyncio
import websockets
import urllib.request
from pathlib import Path
from datetime import datetime, timezone

from .engine import StreamingEngine
from .config import DATA_DIR, AGG_RULES, FEE_FRAC, PARAMS_FILE


# Directories
LOG_DIR = Path(__file__).parent / 'logs'
SIGNAL_DIR = LOG_DIR / 'signals'

# Setup
LOG_DIR.mkdir(exist_ok=True)
SIGNAL_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_DIR / 'paper_trader.log'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger('V10_LIVE')


def fetch_historical_klines(symbol, interval='15m', limit=1500):
    """Fetch historical klines from Binance for warmup.

    Binance Futures limit is 1500 per request. Paginate if more needed.
    """
    all_rows = []
    remaining = limit
    end_time = None

    while remaining > 0:
        batch = min(remaining, 1500)
        url = (f"https://fapi.binance.com/fapi/v1/klines"
               f"?symbol={symbol}&interval={interval}&limit={batch}")
        if end_time:
            url += f"&endTime={end_time}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        if not data:
            break

        for k in data:
            all_rows.append({
                'datetime': pd.Timestamp(k[0], unit='ms', tz='UTC'),
                'open': float(k[1]), 'high': float(k[2]),
                'low': float(k[3]), 'close': float(k[4]),
                'volume': float(k[5]),
                'quote_volume': float(k[7]),
                'taker_buy_volume': float(k[9]),
                'taker_buy_quote_volume': float(k[10]),
            })

        # Set end_time for next batch (before first kline of this batch)
        end_time = data[0][0] - 1
        remaining -= len(data)

        if len(data) < batch:
            break  # No more data

    # Sort chronologically and deduplicate
    df = pd.DataFrame(all_rows)
    if len(df) > 0:
        df = df.drop_duplicates(subset='datetime').sort_values('datetime')
    return df


class LiveTrader:
    """Live paper trader using the unified StreamingEngine."""

    def __init__(self, params_file=None):
        params_path = Path(params_file) if params_file else PARAMS_FILE
        with open(params_path) as f:
            self.frozen = json.load(f)

        self.symbols = list(self.frozen['symbols'].keys())
        self.engines = {}      # symbol → StreamingEngine
        self.kline_buffers = {}  # symbol → list of 15m bars for current hour
        self.last_1h_close = {} # symbol → last completed 1H bar timestamp
        self.positions = {}     # symbol → current direction
        self.pnl_history = {}   # symbol → list of PnL entries

    def initialize(self):
        """Bootstrap all engines with historical data."""
        log.info("=" * 60)
        log.info("V10 UNIFIED PAPER TRADER — INITIALIZING")
        log.info("=" * 60)
        log.info(f"Version: {self.frozen['version']}")
        log.info(f"Frozen at: {self.frozen['frozen_at']}")
        log.info(f"Engine: UNIFIED_V10 StreamingEngine (Dubno-compliant)")

        for sym in self.symbols:
            cfg = self.frozen['symbols'][sym]
            selected = cfg['selected_alphas']
            lookback = cfg['lookback']
            phl = cfg['phl']

            log.info(f"  [{sym}] Bootstrapping {len(selected)} alphas, "
                     f"lb={lookback}, phl={phl}")

            # Fetch historical 15m data
            df15 = fetch_historical_klines(sym, '15m', limit=1500)
            if 'datetime' in df15.columns:
                df15 = df15.set_index('datetime')
            if hasattr(df15.index, 'tz') and df15.index.tz is not None:
                df15.index = df15.index.tz_localize(None)

            # Resample to 1H
            agg = {k: v for k, v in AGG_RULES.items() if k in df15.columns}
            df_1h = df15.resample('1h').agg(agg).dropna()

            log.info(f"  [{sym}] {len(df_1h)} 1H bars. "
                     f"Range: {df_1h.index[0]} to {df_1h.index[-1]}")

            # Create engine
            engine = StreamingEngine(
                selected_alphas=selected,
                lookback=lookback,
                phl=phl,
            )

            # Warm up by replaying historical bars (SAME as backtest!)
            for ts, row in df_1h.iterrows():
                bar = {
                    'datetime': ts,
                    'open': row['open'], 'high': row['high'],
                    'low': row['low'], 'close': row['close'],
                    'volume': row.get('volume', 0),
                    'quote_volume': row.get('quote_volume', 0),
                    'taker_buy_volume': row.get('taker_buy_volume', 0),
                    'taker_buy_quote_volume': row.get('taker_buy_quote_volume', 0),
                }
                result = engine.on_bar(bar)

            state = engine.get_state()
            log.info(f"  [{sym}] Warmup complete: {state['n_factor_returns']} "
                     f"factor return history, dir={state['prev_direction']}")

            self.engines[sym] = engine
            self.kline_buffers[sym] = []
            self.last_1h_close[sym] = df_1h.index[-1]
            self.positions[sym] = state['prev_direction']
            self.pnl_history[sym] = []

        log.info(f"\nInitialized {len(self.engines)} symbols. Starting WS...")

    def _on_kline(self, sym, kline_data):
        """Process a 15m kline event."""
        is_closed = kline_data.get('x', False)
        if not is_closed:
            return  # Only process closed bars

        bar_15m = {
            'datetime': pd.Timestamp(kline_data['t'], unit='ms'),
            'open': float(kline_data['o']),
            'high': float(kline_data['h']),
            'low': float(kline_data['l']),
            'close': float(kline_data['c']),
            'volume': float(kline_data['v']),
            'quote_volume': float(kline_data['q']),
            'taker_buy_volume': float(kline_data['V']),
            'taker_buy_quote_volume': float(kline_data['Q']),
        }

        self.kline_buffers[sym].append(bar_15m)

        # Check if we have a complete 1H bar
        bar_ts = bar_15m['datetime']
        current_hour = bar_ts.floor('1h')
        next_hour = current_hour + pd.Timedelta(hours=1)

        # 1H bar closes when we get the :45 15m bar
        if bar_ts + pd.Timedelta(minutes=15) >= next_hour:
            # Aggregate 15m bars into 1H
            hour_bars = [b for b in self.kline_buffers[sym]
                         if pd.Timestamp(b['datetime']).floor('1h') == current_hour]

            if len(hour_bars) >= 3:  # At least 3 of 4 bars
                bar_1h = {
                    'datetime': current_hour,
                    'open': hour_bars[0]['open'],
                    'high': max(b['high'] for b in hour_bars),
                    'low': min(b['low'] for b in hour_bars),
                    'close': hour_bars[-1]['close'],
                    'volume': sum(b['volume'] for b in hour_bars),
                    'quote_volume': sum(b['quote_volume'] for b in hour_bars),
                    'taker_buy_volume': sum(b['taker_buy_volume'] for b in hour_bars),
                    'taker_buy_quote_volume': sum(b['taker_buy_quote_volume'] for b in hour_bars),
                }

                # Skip if we already processed this hour
                if current_hour <= self.last_1h_close.get(sym, pd.Timestamp.min):
                    return

                self.last_1h_close[sym] = current_hour
                self._on_1h_bar(sym, bar_1h)

                # Clean old 15m bars
                self.kline_buffers[sym] = [
                    b for b in self.kline_buffers[sym]
                    if pd.Timestamp(b['datetime']) > current_hour
                ]

    def _on_1h_bar(self, sym, bar_1h):
        """Process a completed 1H bar through the StreamingEngine."""
        engine = self.engines[sym]
        result = engine.on_bar(bar_1h)

        old_dir = self.positions.get(sym, 0)
        new_dir = result['direction']
        self.positions[sym] = new_dir

        # Log signal
        signal_data = {
            'timestamp': str(bar_1h['datetime']),
            'symbol': sym,
            'direction': new_dir,
            'signal_value': result['signal_value'],
            'close_price': result['close_price'],
            'realized_pnl': result['realized_pnl'],
            'fee': result['fee'],
            'net_pnl': result['net_pnl'],
            'cumulative_pnl': result['cumulative_pnl'],
            'bar_count': result['bar_count'],
            'reason': result['reason'],
        }

        # Save to signal log file
        sig_file = SIGNAL_DIR / f'{sym}.json'
        signals = []
        if sig_file.exists():
            try:
                signals = json.loads(sig_file.read_text())
            except Exception:
                signals = []
        signals.append(signal_data)
        sig_file.write_text(json.dumps(signals, indent=2, default=str))

        # Log position change
        pnl_bps = result['net_pnl'] * 10000
        cum_bps = result['cumulative_pnl'] * 10000
        dir_str = {1: 'LONG', -1: 'SHORT', 0: 'FLAT'}.get(new_dir, '???')

        if old_dir != new_dir:
            old_str = {1: 'LONG', -1: 'SHORT', 0: 'FLAT'}.get(old_dir, '???')
            log.info(f"  [{sym}] SIGNAL: {old_str}→{dir_str} | "
                     f"pnl={pnl_bps:+.1f}bps cum={cum_bps:+.1f}bps | "
                     f"close={result['close_price']:.2f}")
        else:
            log.info(f"  [{sym}] HOLD {dir_str} | "
                     f"pnl={pnl_bps:+.1f}bps cum={cum_bps:+.1f}bps")

    async def run(self):
        """Connect to WebSocket and process klines."""
        self.initialize()

        streams = '/'.join(f"{sym.lower()}@kline_15m" for sym in self.symbols)
        ws_url = f"wss://fstream.binance.com/stream?streams={streams}"

        log.info(f"Connected to {ws_url[:80]}...")

        try:
            async for ws in websockets.connect(ws_url, ping_interval=20,
                                                ping_timeout=60):
                try:
                    async for msg in ws:
                        data = json.loads(msg)
                        if 'data' in data:
                            event = data['data']
                            if event.get('e') == 'kline':
                                sym = event['s']
                                if sym in self.engines:
                                    self._on_kline(sym, event['k'])
                except websockets.ConnectionClosed:
                    log.warning("WebSocket disconnected, reconnecting...")
                    await asyncio.sleep(5)
        except KeyboardInterrupt:
            log.info("Paper trader stopped by user.")
        except Exception as e:
            log.error(f"Fatal error: {e}")
            raise
        finally:
            self._print_summary()

    def _print_summary(self):
        """Print final trading summary."""
        log.info("\n" + "=" * 60)
        log.info("FINAL SUMMARY")
        log.info("=" * 60)
        for sym in self.symbols:
            state = self.engines[sym].get_state()
            dir_str = {1: 'LONG', -1: 'SHORT', 0: 'FLAT'}.get(
                state['prev_direction'], '???')
            cum_pnl = state['cumulative_pnl'] * 10000
            log.info(f"  {sym}: pos={dir_str} pnl={cum_pnl:+.1f}bps")


def main():
    trader = LiveTrader()
    asyncio.run(trader.run())


if __name__ == '__main__':
    main()
