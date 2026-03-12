"""Configuration constants for V10 unified engine."""
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data' / 'binance_futures_15m'
PARAMS_FILE = Path(__file__).parent.parent / 'PAPER_TRADER' / 'frozen_params.json'

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'DOGEUSDT',
           'XRPUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'LTCUSDT']

FEE_BPS = 3
FEE_FRAC = FEE_BPS / 10000.0

# Minimum bars needed before producing signals
MIN_WARMUP_BARS = 200

# OHLCV aggregation rules for resampling
AGG_RULES = {
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
    'volume': 'sum', 'quote_volume': 'sum',
    'taker_buy_volume': 'sum', 'taker_buy_quote_volume': 'sum',
}
