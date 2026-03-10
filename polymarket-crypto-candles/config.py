"""
config.py — Central configuration for Polymarket Crypto Candle Trading
"""
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
DB_PATH = PROJECT_DIR / "db" / "signals.db"

# ============================================================================
# SYMBOLS & INTERVALS
# ============================================================================
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
INTERVALS = ["5m", "15m", "1h"]

# Friendly names
SYMBOL_NAMES = {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "SOLUSDT": "SOL"}

# ============================================================================
# DATA PERIODS
# ============================================================================
DATA_START = "2024-01-01"
DATA_END = "2026-03-09"

# Train / Holdout split
TRAIN_START = "2024-03-01"   # Skip first 2 months for signal warmup
TRAIN_END = "2025-09-01"    # 18 months train

HOLDOUT_START = "2025-09-01"
HOLDOUT_END = "2026-03-09"  # 6 months holdout

# ============================================================================
# POLYMARKET FEE MODEL
# ============================================================================
def polymarket_taker_fee(probability: float) -> float:
    """
    Polymarket taker fee as a function of market probability.
    Fee peaks at ~1.56% near p=0.50, drops to ~0% at extremes.
    Modeled as: fee = 4 * MAX_FEE * p * (1 - p)
    """
    MAX_FEE = 0.0156  # 1.56% peak
    return 4 * MAX_FEE * probability * (1 - probability)

# Simplified blended fee for backtesting
BLENDED_TAKER_FEE = 0.015  # 1.5% average (entry near 50%)

# ============================================================================
# SIGNAL PARAMETERS
# ============================================================================
MIN_WIN_RATE = 0.515       # Minimum profitable win rate at 50% entry with 1.5% fees
SIGNAL_THRESHOLD = 0.02    # Minimum edge required to trade (P(UP) - 0.50 > this)
KELLY_FRACTION = 0.25      # Fractional Kelly for sizing

# ============================================================================
# BACKTEST PARAMETERS
# ============================================================================
INITIAL_CAPITAL = 50_000.0
BASE_TRADE_SIZE = 250.0     # USDC per contract
MAX_CONCURRENT_POSITIONS = 10
