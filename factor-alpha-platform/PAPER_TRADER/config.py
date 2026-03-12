"""Paper Trader Configuration."""

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'DOGEUSDT']

# Binance Futures WebSocket (combined stream)
WS_URL = (
    "wss://fstream.binance.com/stream?streams="
    + "/".join(f"{s.lower()}@kline_15m" for s in SYMBOLS)
)

# Binance Futures REST (for bootstrap)
REST_URL = "https://fapi.binance.com/fapi/v1/klines"

# Data directories
import pathlib
BASE_DIR = pathlib.Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
PARAMS_FILE = BASE_DIR / "frozen_params.json"
DATA_DIR = BASE_DIR.parent / "data" / "binance_futures_15m"

# Strategy
FEE_BPS = 3
FEE_FRAC = FEE_BPS / 10000.0
WARMUP_1H_BARS = 720  # 30 days
MAX_15M_BARS = 3000   # ~31 days of 15m bars

# Resampling periods
RESAMPLE_PERIODS = {
    '1h': '1h',
    '2h': '2h',
    '4h': '4h',
    '8h': '8h',
    '12h': '12h',
}
