import os
import time
import Constants  # Make sure this import is at the top of the file

# Paths
ROOT = "C:\Equities\AV_Russell2000_DATA"
LOG_DATA_PATH = 'logDataFiles/'
MODEL_DATA_PATH = 'ModelOutputs/'

# Trading parameters
BOOK_SIZE = 20000000.0
MAX_STOCK_WEIGHT = 0.01
FEES_BSP = 0.0000
HEDGE_VOL = False
RANK_HEDGE = False
FUNC_LOOKBACK_LENGTH = 90
LINEAR_DECAY = 0
EXP_DECAY = 0.0
TOP_N = 4000
TARGET_DELAY = -1
TARGET_FUTURE = 0
PORT_TAIL = 0.00
UNIVERSE_BLOCKING = False
MIN_PRICE = 0.0
MAX_PRICE = 10000000.0
USE_LAMBDA_TRANSACTION_MODEL = False
EXP_FACTOR_DECAY = 0.0
VOLUME_MEAN_RANKING_WINDOW = 252
POST_PORTFOLIO_OPTIM_RE_SCALE_RISK_MODEL = True
USE_TOP_BOTTOM_100 = True
NUM_LONG_SHORT = 10

# Risk model parameters
RISK_MODEL_TYPE = Constants.GLOBAL_RISK_MODEL  # You can change this to other types as needed
RISK_MODEL_NUM_FACTORS = 150
PCA_MA = 0.0

# Database configuration
DB_CONFIG = {
    'host': 'alphasdatabase1.cysvmgsjf7ox.us-east-1.rds.amazonaws.com',
    'user': 'admin',
    'password': 'SALMON44',
    'db': 'quantschema',
    'charset': 'utf8mb4',
}

# Other constants
TIME_RATE_MIN = 1440
RUN_NAME = 'DAY_STRATEGY_1_A'
TEST_START_DATE = "2024-05-02"
OPTIM_END_DATE = "2024-05-15"
MODEL_NAME = f"M2_{int(time.time() * 1000)}"
