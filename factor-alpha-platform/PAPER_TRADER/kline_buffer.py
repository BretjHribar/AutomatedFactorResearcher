"""
kline_buffer.py — Maintains rolling 15m bars per symbol, resamples to multi-TF.

Bootstraps from REST API on startup, then receives live bars from WebSocket.
"""
import time
import requests
import numpy as np
import pandas as pd
from collections import deque
from config import REST_URL, MAX_15M_BARS, WARMUP_1H_BARS

OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume',
              'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']

AGG_DICT = {
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
    'volume': 'sum', 'quote_volume': 'sum',
    'taker_buy_volume': 'sum', 'taker_buy_quote_volume': 'sum',
}


class KlineBuffer:
    """Maintains a rolling buffer of 15m bars and resamples to higher TFs."""

    def __init__(self, symbol, max_bars=MAX_15M_BARS):
        self.symbol = symbol
        self.max_bars = max_bars
        self.bars_15m = deque(maxlen=max_bars)
        self._df_cache = None
        self._cache_valid = False
        self._last_1h_boundary = None  # Track last completed 1H bar

    # ---------------------------------------------------------------
    # REST Bootstrap
    # ---------------------------------------------------------------

    def bootstrap_from_rest(self, n_1h_bars=WARMUP_1H_BARS):
        """Download historical 15m klines from fapi.binance.com for warmup."""
        n_15m = n_1h_bars * 4
        all_rows = []
        end_ms = int(time.time() * 1000)
        interval_ms = 900_000  # 15m

        print(f"  [{self.symbol}] Bootstrapping {n_15m} 15m bars...", flush=True)

        while len(all_rows) < n_15m:
            start_ms = end_ms - (1500 * interval_ms)
            params = {
                "symbol": self.symbol,
                "interval": "15m",
                "startTime": start_ms,
                "endTime": end_ms - 1,
                "limit": 1500,
            }
            try:
                resp = requests.get(REST_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"  [{self.symbol}] REST error: {e}, retrying...")
                time.sleep(2)
                continue

            if not data:
                break

            all_rows = data + all_rows  # Prepend (earlier bars first)
            end_ms = data[0][0]  # Move window back
            time.sleep(0.15)  # Rate limit

        # Parse into bars
        for row in all_rows[-n_15m:]:
            bar = {
                'datetime': pd.Timestamp(row[0], unit='ms', tz='UTC'),
                'open': float(row[1]),
                'high': float(row[2]),
                'low': float(row[3]),
                'close': float(row[4]),
                'volume': float(row[5]),
                'quote_volume': float(row[7]),
                'taker_buy_volume': float(row[9]),
                'taker_buy_quote_volume': float(row[10]),
            }
            self.bars_15m.append(bar)

        self._cache_valid = False
        print(f"  [{self.symbol}] Loaded {len(self.bars_15m)} 15m bars. "
              f"Range: {self.bars_15m[0]['datetime']} to {self.bars_15m[-1]['datetime']}")

    def bootstrap_from_parquet(self, path):
        """Bootstrap from a local parquet file (if available)."""
        df = pd.read_parquet(path)
        if 'datetime' in df.columns:
            df = df.set_index('datetime')
        df = df.sort_index()
        for col in OHLCV_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        for idx, row in df.iterrows():
            bar = {
                'datetime': pd.Timestamp(idx, tz='UTC') if idx.tzinfo is None else idx,
                'open': row['open'], 'high': row['high'],
                'low': row['low'], 'close': row['close'],
                'volume': row['volume'],
                'quote_volume': row.get('quote_volume', 0),
                'taker_buy_volume': row.get('taker_buy_volume', 0),
                'taker_buy_quote_volume': row.get('taker_buy_quote_volume', 0),
            }
            self.bars_15m.append(bar)

        self._cache_valid = False
        print(f"  [{self.symbol}] Loaded {len(self.bars_15m)} bars from parquet")

    # ---------------------------------------------------------------
    # Live bar handling
    # ---------------------------------------------------------------

    def on_kline_closed(self, kline_msg):
        """Process a closed 15m bar from WebSocket.
        
        Args:
            kline_msg: dict with keys t, o, h, l, c, v, q, V, Q (Binance WS format)
            
        Returns:
            '1h' if a 1H bar just completed, None otherwise.
        """
        bar = {
            'datetime': pd.Timestamp(kline_msg['t'], unit='ms', tz='UTC'),
            'open': float(kline_msg['o']),
            'high': float(kline_msg['h']),
            'low': float(kline_msg['l']),
            'close': float(kline_msg['c']),
            'volume': float(kline_msg['v']),
            'quote_volume': float(kline_msg['q']),
            'taker_buy_volume': float(kline_msg['V']),
            'taker_buy_quote_volume': float(kline_msg['Q']),
        }
        self.bars_15m.append(bar)
        self._cache_valid = False

        # Check if a 1H bar just completed
        # 15m bars within a 1H boundary: :00, :15, :30, :45
        # The :45 bar closing means the hour is complete
        bar_minute = bar['datetime'].minute
        if bar_minute == 45:
            return '1h'
        return None

    # ---------------------------------------------------------------
    # DataFrame access
    # ---------------------------------------------------------------

    def get_15m_df(self):
        """Return DataFrame of all 15m bars."""
        if self._cache_valid and self._df_cache is not None:
            return self._df_cache
        if not self.bars_15m:
            return pd.DataFrame()
        df = pd.DataFrame(list(self.bars_15m))
        df = df.set_index('datetime').sort_index()
        df = df[~df.index.duplicated(keep='last')]
        self._df_cache = df
        self._cache_valid = True
        return df

    def get_resampled(self, period):
        """Resample 15m data to any higher period (1h, 2h, 4h, 8h, 12h)."""
        df15 = self.get_15m_df()
        if df15.empty:
            return df15
        return df15.resample(period).agg(AGG_DICT).dropna()

    def get_1h(self): return self.get_resampled('1h')
    def get_2h(self): return self.get_resampled('2h')
    def get_4h(self): return self.get_resampled('4h')
    def get_8h(self): return self.get_resampled('8h')
    def get_12h(self): return self.get_resampled('12h')

    @property
    def latest_close(self):
        """Return the most recent close price."""
        if self.bars_15m:
            return self.bars_15m[-1]['close']
        return None

    @property
    def n_1h_bars(self):
        """Approximate number of 1H bars in buffer."""
        return len(self.bars_15m) // 4
