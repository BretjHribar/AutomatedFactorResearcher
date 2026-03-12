"""
live_trade_real.py — REAL MONEY live trading on Polymarket 5m candle contracts.

FIXES from V1:
  1. Contract discovery: slug timestamp = candle START, not end.
     Use (now // 300) * 300 to get the contract starting NOW.
  2. Outcome check: Query Polymarket resolution via Gamma API instead of
     using Binance OHLCV (Chainlink vs Binance disagree ~3% of the time).
  3. Auto-redemption: Redeem winning conditional tokens back to USDC on-chain
     after every resolved contract.

Trade size: $5/trade (~1x Kelly at 52% WR with $194 bankroll)
"""
import sys, os, time, json, asyncio, traceback
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from collections import deque

# Set up dual logging: console + file
LOG_FILE = Path(__file__).parent / "real_trader.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("real_trader")

# Redirect print to also log
_orig_print = print
def print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    logger.info(msg)


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SYMBOLS, SYMBOL_NAMES, DATA_DIR
from polymarket_api import (
    PolymarketClient, CandleContract, Orderbook,
    get_current_candle_end, get_next_candle_end, compute_polymarket_fee
)

try:
    import websockets
except ImportError:
    print("pip install websockets"); sys.exit(1)

from dotenv import load_dotenv
load_dotenv()

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import MarketOrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY

import requests as http_requests
try:
    from web3 import Web3
    HAS_WEB3 = True
except ImportError:
    HAS_WEB3 = False
    print("  ⚠ web3 not installed — auto-redemption disabled")

# ============================================================================
# CONFIGURATION
# ============================================================================

VERSION = "REAL_V2"
INTERVAL = "5m"
INTERVAL_SECONDS = 300
BINANCE_WS = "wss://stream.binance.com:9443/ws"
MAX_HISTORY = 500

TRADE_LOG   = Path(__file__).parent / "real_trades.jsonl"
STATE_FILE  = Path(__file__).parent / "real_state.json"
BOOK_LOG    = Path(__file__).parent / "real_book_snapshots.jsonl"
FILL_LOG    = Path(__file__).parent / "real_fill_quality.csv"

# V4 LGB engine configs (walk-forward validated 55%+ WR on filtered trades)
# ETHUSDT/BTCUSDT: LightGBM model with probability threshold
# SOLUSDT: MR ensemble with percentile threshold (LGB doesn't help SOL)
CONFIGS = {
    "BTCUSDT": {"engine": "lgb", "prob_threshold": 0.535, "pctile": 85, "corr_cutoff": 0.90, "max_alphas": 10, "lookback": 1440, "phl": 1},
    "ETHUSDT": {"engine": "lgb", "prob_threshold": 0.530, "pctile": 65, "corr_cutoff": 0.80, "max_alphas": 12, "lookback": 5760, "phl": 1},
    "SOLUSDT": {"engine": "mr", "prob_threshold": 0.530, "pctile": 92, "corr_cutoff": 0.80, "max_alphas": 15, "lookback": 1440, "phl": 1},
}

TRADE_SIZE_USD = 2.0     # Base trade size
MAX_TRADE_SIZE = 8.0     # Max single trade ($2 * 3x = $6, with vol boost up to $8)
MAX_FILL_PRICE = 0.53    # FILL GUARD: refuse to trade if ask > this (protects edge)
MIN_CAPITAL = 150.0
STARTING_CAPITAL = 217.86  # Actual balance as of 2026-03-10 19:44
MIN_SIGNAL_STRENGTH = 0.0  # Threshold now handled by percentile filter

def compute_trade_size(signal_strength, vol_expanded=False, engine_type="mr"):
    """Compute trade size based on signal strength and volatility regime.
    Walk-forward validated: strong signals have higher WR."""
    base = TRADE_SIZE_USD
    abs_s = abs(signal_strength)
    
    if engine_type == "lgb":
        # LGB signals are probabilities centered at 0.5, so signal = proba - 0.5
        # Typical range: 0.03 to 0.15
        if abs_s > 0.08:
            multiplier = 3.0
        elif abs_s > 0.05:
            multiplier = 2.0
        elif abs_s > 0.03:
            multiplier = 1.0
        else:
            multiplier = 0.5
    else:
        # MR signals: typical range 0.5 to 15
        if abs_s > 1.5:
            multiplier = 3.0
        elif abs_s > 1.0:
            multiplier = 2.0 
        elif abs_s > 0.5:
            multiplier = 1.0
        else:
            multiplier = 0.5  # Half size on weak signals
    
    # Boost if vol expanded (MR works better after big candles)
    if vol_expanded and abs_s > (0.03 if engine_type == "lgb" else 0.5):
        multiplier *= 1.5
    
    return min(base * multiplier, MAX_TRADE_SIZE)

FEE_PER_TRADE_BPS = 50
BARS_PER_DAY = 288

BINANCE_TO_PM = {"BTCUSDT": "btc", "ETHUSDT": "eth", "SOLUSDT": "sol"}

TRADE_DELAY_SECONDS = 0  # Fire immediately when signal generated

# On-chain addresses for auto-redemption
POLYGON_RPC = "https://polygon-bor-rpc.publicnode.com"
CT_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

# ============================================================================
# ALPHA PRIMITIVES (identical to paper trader)
# ============================================================================

def sma(s, w): return s.rolling(w, min_periods=1).mean()
def ema(s, w): return s.ewm(halflife=w, min_periods=1).mean()
def stddev(s, w): return s.rolling(w, min_periods=2).std()
def ts_zscore(s, w):
    m = s.rolling(w, min_periods=2).mean()
    sd = s.rolling(w, min_periods=2).std()
    return (s - m) / sd.replace(0, np.nan)
def delta(s, p): return s - s.shift(p)
def ts_sum(s, w): return s.rolling(w, min_periods=1).sum()
def safe_div(a, b):
    r = a / b; return r.replace([np.inf, -np.inf], 0).fillna(0)


def build_alpha_signals(df, live_mode=False):
    close = df["close"]; volume = df["volume"]; high = df["high"]; low = df["low"]
    opn = df["open"]; taker_buy = df["taker_buy_base"]
    qv = df["quote_volume"]
    ret = close.pct_change()
    log_ret = np.log(close / close.shift(1))
    vwap = safe_div(qv, volume)
    taker_ratio = safe_div(taker_buy, volume)
    taker_sell = volume - taker_buy
    taker_imbalance = safe_div(taker_buy - taker_sell, volume)
    obv = (np.sign(ret) * volume).cumsum()
    alphas = {}
    for w in [5, 8, 10, 12, 15, 20, 24, 30, 36, 48]:
        alphas[f"mr_{w}"] = -ts_zscore(close, w)
    for w in [3, 5, 8, 10, 12, 15, 20]:
        alphas[f"logrev_{w}"] = -ts_sum(log_ret, w)
    for w in [3, 5, 8, 10, 12, 15, 20]:
        alphas[f"dstd_{w}"] = -safe_div(delta(close, w), stddev(close, w))
    for w in [5, 10, 15, 20, 30]:
        alphas[f"vwap_mr_{w}"] = -ts_zscore(vwap, w)
    for w in [5, 10, 20]:
        alphas[f"ema_mr_{w}"] = -(close - ema(close, w)) / stddev(close, w*2)
    for w in [10, 20, 30]:
        alphas[f"obv_{w}"] = -ts_zscore(obv, w)
    for w in [10, 20]:
        alphas[f"vp_div_{w}"] = ts_zscore(volume, w) - ts_zscore(close, w)
    for w in [5, 10, 20]:
        alphas[f"tbr_{w}"] = ts_zscore(taker_ratio, w)
        alphas[f"timb_{w}"] = ts_zscore(taker_imbalance, w)
    alpha_df = pd.DataFrame(alphas, index=df.index)
    if not live_mode:
        alpha_df = alpha_df.shift(1)
    return alpha_df


# ============================================================================
# ALPHA SELECTION
# ============================================================================

def evaluate_alpha_nofee(signal, target):
    common = signal.dropna().index.intersection(target.dropna().index)
    if len(common) < 500: return None
    s, t = signal.loc[common], target.loc[common]
    direction = np.sign(s)
    correct = (direction == (2*t - 1))
    wr = correct.mean()
    if wr < 0.505: return None
    daily = (direction * (2*t.astype(float) - 1)).resample("1D").sum()
    daily = daily[daily != 0]
    if len(daily) < 20 or daily.std() == 0: return None
    sharpe = daily.mean() / daily.std() * np.sqrt(365)
    ic = s.corr(t.astype(float))
    return {"nofee_sharpe": sharpe, "win_rate": wr, "ic": ic}


def select_alphas_from_history(symbol):
    cfg = CONFIGS[symbol]
    parquet = DATA_DIR / f"{symbol}_{INTERVAL}.parquet"
    if not parquet.exists(): return [], []
    df = pd.read_parquet(parquet)
    train_df = df.loc["2024-03-01":"2025-05-01"]
    if len(train_df) < 10000: return [], []
    target = (train_df["close"] >= train_df["open"]).astype(int)
    alpha_matrix = build_alpha_signals(train_df, live_mode=False)
    results = []
    for col in alpha_matrix.columns:
        m = evaluate_alpha_nofee(alpha_matrix[col], target)
        if m: results.append({"name": col, **m})
    results.sort(key=lambda x: x["nofee_sharpe"], reverse=True)
    selected = []
    for r in results:
        sig = alpha_matrix[r["name"]]
        too_corr = False
        for sel in selected:
            if abs(sig.corr(alpha_matrix[sel["name"]])) > cfg["corr_cutoff"]:
                too_corr = True; break
        if not too_corr:
            selected.append(r)
        if len(selected) >= cfg["max_alphas"]: break
    return [s["name"] for s in selected], selected


# ============================================================================
# KLINE BUFFER
# ============================================================================

class KlineBuffer:
    def __init__(self, symbol, max_size=MAX_HISTORY):
        self.symbol = symbol; self.max_size = max_size
        self.klines = deque(maxlen=max_size)
        self._df_cache = None; self._cache_valid = False

    def add_kline(self, kline):
        bar = {
            "open_time": pd.Timestamp(kline["t"], unit="ms", tz="UTC"),
            "open": float(kline["o"]), "high": float(kline["h"]),
            "low": float(kline["l"]), "close": float(kline["c"]),
            "volume": float(kline["v"]), "close_time": int(kline["T"]),
            "quote_volume": float(kline["q"]), "trades": int(kline["n"]),
            "taker_buy_base": float(kline["V"]),
            "taker_buy_quote": float(kline["Q"]),
        }
        self.klines.append(bar); self._cache_valid = False

    def seed_from_parquet(self, path, n_bars=400):
        df = pd.read_parquet(path)
        for _, row in df.tail(n_bars).iterrows():
            self.klines.append({
                "open_time": row.name, "open": row["open"], "high": row["high"],
                "low": row["low"], "close": row["close"], "volume": row["volume"],
                "quote_volume": row["quote_volume"], "trades": row["trades"],
                "taker_buy_base": row["taker_buy_base"],
            })
        self._cache_valid = False

    def to_dataframe(self):
        if self._cache_valid: return self._df_cache
        if not self.klines: return pd.DataFrame()
        df = pd.DataFrame(list(self.klines))
        df.index = pd.DatetimeIndex(df["open_time"])
        for c in ["open","high","low","close","volume","quote_volume","taker_buy_base"]:
            if c in df.columns: df[c] = df[c].astype(float)
        if "trades" in df.columns: df["trades"] = df["trades"].astype(float)
        self._df_cache = df; self._cache_valid = True
        return df


# ============================================================================
# MR ENSEMBLE SIGNAL ENGINE (v3 — walk-forward validated 54%+ WR on ETH)
# ============================================================================

def build_mr_ensemble(df):
    """Build 25-indicator mean-reversion ensemble.
    
    Walk-forward validated on 230k bars (2yr+) with no look-ahead:
    - ETH p65: 54.2% WR (every quarter > 54%)
    - BTC p75: 53.8% WR
    - SOL p80: 53.2% WR
    
    All signals use data up to the CURRENT bar (no shift applied here;
    the live engine uses the latest value to predict the NEXT bar).
    """
    close = df["close"]; high = df["high"]; low = df["low"]
    opn = df["open"]; volume = df["volume"]
    qv = df["quote_volume"]
    ret = close.pct_change()
    vwap = qv / volume.replace(0, np.nan)
    
    alphas = []
    
    # 1. Price MR z-score (9 alphas)
    for w in [3, 5, 6, 8, 10, 12, 15, 20, 30]:
        m = close.rolling(w, min_periods=2).mean()
        s = close.rolling(w, min_periods=2).std().replace(0, np.nan)
        alphas.append((-(close - m) / s).replace([np.inf, -np.inf], np.nan))
    
    # 2. VWAP MR (3 alphas)
    for w in [10, 20, 30]:
        m = vwap.rolling(w, min_periods=2).mean()
        s = vwap.rolling(w, min_periods=2).std().replace(0, np.nan)
        alphas.append((-(close - m) / s).replace([np.inf, -np.inf], np.nan))
    
    # 3. EMA MR (3 alphas)
    for w in [5, 10, 20]:
        e = close.ewm(halflife=w).mean()
        s = close.rolling(w*2, min_periods=2).std().replace(0, np.nan)
        alphas.append((-(close - e) / s).replace([np.inf, -np.inf], np.nan))
    
    # 4. RSI MR (2 alphas)
    for w in [7, 14]:
        gain = ret.clip(lower=0).rolling(w).mean()
        loss = (-ret.clip(upper=0)).rolling(w).mean()
        rsi = gain / (gain + loss + 1e-10)
        alphas.append(-(rsi - 0.5))
    
    # 5. Bollinger MR (2 alphas)
    for w in [10, 20]:
        m = close.rolling(w).mean()
        s = close.rolling(w).std()
        alphas.append(-((close - m) / (2*s + 1e-10)))
    
    # 6. Keltner MR (2 alphas)
    for w in [10, 20]:
        atr = (high - low).rolling(w).mean()
        e = close.ewm(halflife=w).mean()
        alphas.append((-(close - e) / (atr + 1e-10)).replace([np.inf, -np.inf], np.nan))
    
    # 7. Stochastic MR (2 alphas)
    for w in [5, 14]:
        lowest = low.rolling(w).min()
        highest = high.rolling(w).max()
        k = (close - lowest) / (highest - lowest + 1e-10)
        alphas.append(-(k - 0.5))
    
    # 8. CCI MR (2 alphas)
    for w in [10, 20]:
        tp = (high + low + close) / 3
        m = tp.rolling(w).mean()
        s = tp.rolling(w).std()
        alphas.append(-((tp - m) / (0.015 * s + 1e-10)).replace([np.inf, -np.inf], np.nan))
    
    # Simple average of all 25 indicators
    combo = pd.concat(alphas, axis=1).mean(axis=1)
    return combo


class MREnsembleEngine:
    """V3 MR Ensemble Engine — replaces AdaptiveNetEngine.
    
    Key differences from v2:
    1. 25 MR indicators instead of momentum+MR mix
    2. Simple average instead of adaptive weighting
    3. Signal magnitude filter using expanding percentile threshold
    4. Per-coin optimized percentile (ETH=65, BTC=75, SOL=80)
    """
    def __init__(self, symbol, cfg):
        self.symbol = symbol
        self.pctile = cfg.get("pctile", 70)
        self.bar_count = 0
        self.threshold = 0.5  # Initial threshold, updated on first bar
        self.threshold_history = []  # Track threshold evolution
    
    def update(self, df):
        """Generate signal from MR ensemble.
        
        Returns: (direction, signal_val, info_dict)
        - direction: +1 (UP), -1 (DOWN), 0 (no trade)
        - signal_val: raw signal magnitude
        - info_dict: diagnostic info
        """
        self.bar_count += 1
        
        if len(df) < 50:
            return 0, 0.0, {}
        
        # Build MR ensemble signal
        combo = build_mr_ensemble(df)
        
        # Latest signal value (uses data up to the just-closed bar)
        signal_val = float(combo.iloc[-1])
        if np.isnan(signal_val):
            return 0, 0.0, {}
        
        # Update threshold from historical signals (expanding window)
        # Only use past signals — no look-ahead
        past_signals = combo.iloc[:-1].abs().dropna()
        if len(past_signals) > 100:
            self.threshold = float(np.percentile(past_signals, self.pctile))
        
        # Apply magnitude filter
        if abs(signal_val) < self.threshold:
            return 0, signal_val, {"threshold": self.threshold, "filtered": True}
        
        direction = int(np.sign(signal_val))
        info = {
            "threshold": self.threshold,
            "signal_abs": abs(signal_val),
            "n_indicators": 25,
            "pctile": self.pctile,
            "filtered": False,
        }
        return direction, signal_val, info


def build_lgb_features(df):
    """Build feature matrix for LightGBM engine."""
    close = df["close"]; high = df["high"]; low = df["low"]; opn = df["open"]
    volume = df["volume"]; qv = df["quote_volume"]; taker_buy = df["taker_buy_base"]
    ret = close.pct_change()
    vwap = qv / volume.replace(0, np.nan)
    taker_imb = (2*taker_buy - volume) / volume.replace(0, np.nan)
    vol_ratio = volume / volume.rolling(20).mean()
    atr = high - low

    feats = {}
    # MR ensemble (lead feature)
    feats["mr_ensemble"] = build_mr_ensemble(df)
    feats["mr_ensemble_abs"] = feats["mr_ensemble"].abs()

    # Individual MR z-scores
    for w in [3, 5, 8, 10, 15, 20, 30]:
        m = close.rolling(w, min_periods=2).mean()
        s = close.rolling(w, min_periods=2).std().replace(0, np.nan)
        feats[f"mr_{w}"] = (-(close - m) / s).replace([np.inf, -np.inf], np.nan)

    # VWAP MR
    for w in [10, 20]:
        m = vwap.rolling(w, min_periods=2).mean()
        s = vwap.rolling(w, min_periods=2).std().replace(0, np.nan)
        feats[f"vmr_{w}"] = (-(close - m) / s).replace([np.inf, -np.inf], np.nan)

    # RSI
    for w in [7, 14]:
        gain = ret.clip(lower=0).rolling(w).mean()
        loss = (-ret.clip(upper=0)).rolling(w).mean()
        feats[f"rsi_{w}"] = gain / (gain + loss + 1e-10)

    # Stochastic
    for w in [5, 14]:
        feats[f"stoch_{w}"] = (close - low.rolling(w).min()) / (
            high.rolling(w).max() - low.rolling(w).min() + 1e-10)

    # Volume
    feats["vol_ratio"] = vol_ratio.replace([np.inf, -np.inf], np.nan)
    feats["taker_imb"] = taker_imb
    for w in [5, 10]:
        feats[f"timb_{w}"] = taker_imb.rolling(w).mean()

    # Candle
    feats["body"] = (close - opn) / (atr + 1e-10)
    feats["cpos"] = (close - low) / (atr + 1e-10)
    feats["body_5"] = ((close - opn) / (atr + 1e-10)).rolling(5).mean()

    # Volatility
    feats["atr_ratio"] = (atr / atr.rolling(20).mean()).replace([np.inf, -np.inf], np.nan)
    feats["rvol_10"] = ret.rolling(10).std()

    # Returns
    for w in [1, 3, 5, 10]:
        feats[f"ret_{w}"] = close.pct_change(w)

    # Interactions
    feats["ens_x_vol"] = feats["mr_ensemble"] * feats["vol_ratio"]
    feats["ens_x_timb"] = feats["mr_ensemble"] * feats["taker_imb"]
    feats["ens_x_atr"] = feats["mr_ensemble"] * feats["atr_ratio"]

    # Time
    if hasattr(df.index, "hour"):
        feats["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        feats["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)

    return pd.DataFrame(feats, index=df.index)


class LGBEngine:
    """V4 LightGBM Engine — 55%+ WR at 25% of bars.

    Walk-forward validated on 200k+ OOS predictions:
    - ETH thresh=0.530: 54.7% WR on 23% of bars
    - ETH thresh=0.540: 55.7% WR on 12% of bars
    - BTC thresh=0.530: 54.5% WR on 16% of bars
    """
    LGB_PARAMS = {
        "objective": "binary", "metric": "binary_logloss",
        "num_leaves": 7, "learning_rate": 0.01,
        "feature_fraction": 0.5, "bagging_fraction": 0.6,
        "bagging_freq": 3, "min_child_samples": 100,
        "verbose": -1, "n_jobs": -1,
    }

    def __init__(self, symbol, cfg):
        self.symbol = symbol
        self.prob_threshold = cfg.get("prob_threshold", 0.535)
        self.model = None
        self.bar_count = 0
        self.retrain_interval = 2000  # Retrain every 2000 bars (~7 days)
        self.last_train_bar = -9999  # Force training on first call
        self.train_window = 80000
        self.min_train = 2000  # Reduced from 5000 — rolling features drop many rows

    def _train(self, df):
        """Train LGB model on historical data. No look-ahead."""
        import lightgbm as lgb

        target = (df["close"] >= df["open"]).astype(int)
        X_all = build_lgb_features(df)
        y_all = target.shift(-1)  # Predict NEXT bar

        # Use last train_window bars
        start = max(0, len(df) - self.train_window)
        X_tr = X_all.iloc[start:]
        y_tr = y_all.iloc[start:]

        valid = X_tr.dropna().index.intersection(y_tr.dropna().index)
        if len(valid) < self.min_train:
            return False

        val_split = int(len(valid) * 0.8)
        train_idx = valid[:val_split]
        val_idx = valid[val_split:]

        dtrain = lgb.Dataset(X_tr.loc[train_idx], y_tr.loc[train_idx])
        dval = lgb.Dataset(X_tr.loc[val_idx], y_tr.loc[val_idx], reference=dtrain)

        self.model = lgb.train(
            self.LGB_PARAMS, dtrain, num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        self.last_train_bar = self.bar_count
        return True

    def update(self, df):
        """Generate signal from LGB model.

        Returns: (direction, signal_val, info_dict)
        """
        self.bar_count += 1

        if len(df) < 200:
            return 0, 0.0, {}

        # Retrain periodically
        if self.model is None or (self.bar_count - self.last_train_bar) >= self.retrain_interval:
            try:
                success = self._train(df)
                if success:
                    print(f"    [LGB] {self.symbol} trained (bars={len(df)}, bar_count={self.bar_count})")
                else:
                    print(f"    [LGB] {self.symbol} training failed (not enough data)")
            except Exception as e:
                import traceback
                print(f"    [!] LGB train error for {self.symbol}: {e}")
                traceback.print_exc()
                if self.model is None:
                    return 0, 0.0, {}

        if self.model is None:
            return 0, 0.0, {}

        # Build features for latest bar
        X_all = build_lgb_features(df)
        latest = X_all.iloc[[-1]]

        if latest.isna().any(axis=1).iloc[0]:
            return 0, 0.0, {}

        proba = float(self.model.predict(latest)[0])

        # Apply probability threshold
        if proba >= self.prob_threshold:
            direction = 1  # UP
            signal_val = proba - 0.5
        elif proba <= (1.0 - self.prob_threshold):
            direction = -1  # DOWN
            signal_val = proba - 0.5
        else:
            return 0, proba - 0.5, {"proba": proba, "threshold": self.prob_threshold, "filtered": True}

        info = {
            "proba": proba,
            "threshold": self.prob_threshold,
            "bars_since_train": self.bar_count - self.last_train_bar,
            "filtered": False,
        }
        return direction, signal_val, info


# ============================================================================
# CONTRACT DISCOVERY (FIXED: slug timestamp = candle START)
# ============================================================================

def get_current_contract_start_ts():
    """Get the slug timestamp for the contract whose 5m window is happening NOW.

    Polymarket slug format: coin-updown-5m-{START_TIMESTAMP}
    E.g. at 15:07 UTC, the active contract started at 15:05 → ts=15:05.
    """
    now = int(time.time())
    return (now // INTERVAL_SECONDS) * INTERVAL_SECONDS


# ============================================================================
# POLYMARKET RESOLUTION CHECK (via Gamma API, not Binance)
# ============================================================================

def check_polymarket_resolution(slug):
    """Check how a Polymarket contract resolved. Returns 'UP', 'DOWN', or None."""
    try:
        resp = http_requests.get(
            f"https://gamma-api.polymarket.com/markets",
            params={"slug": slug}, timeout=5
        )
        markets = resp.json()
        if not markets:
            return None
        m = markets[0]
        if not m.get("closed"):
            return None
        prices = json.loads(m.get("outcomePrices", '["0.5","0.5"]'))
        up_price = float(prices[0])
        if up_price == 1.0:
            return "UP"
        elif up_price == 0.0:
            return "DOWN"
        return None  # Not yet resolved
    except Exception:
        return None


# ============================================================================
# AUTO-REDEMPTION (on-chain via CTF contract)
# ============================================================================

def try_redeem_position(condition_id):
    """Try to redeem a resolved position on-chain. Returns USDC gained."""
    if not HAS_WEB3:
        return 0.0

    try:
        w3 = Web3(Web3.HTTPProvider(POLYGON_RPC, request_kwargs={"timeout": 15}))
        if not w3.is_connected():
            return 0.0

        pk = os.getenv("POLYGON_PRIVATE_KEY")
        acct = w3.eth.account.from_key(pk)

        CT = Web3.to_checksum_address(CT_ADDRESS)
        USDC = Web3.to_checksum_address(USDC_ADDRESS)

        CT_ABI = [
            {"constant": False, "inputs": [
                {"name": "collateralToken", "type": "address"},
                {"name": "parentCollectionId", "type": "bytes32"},
                {"name": "conditionId", "type": "bytes32"},
                {"name": "indexSets", "type": "uint256[]"}
            ], "name": "redeemPositions", "outputs": [], "type": "function"},
            {"constant": True, "inputs": [
                {"name": "conditionId", "type": "bytes32"}
            ], "name": "payoutDenominator", "outputs": [
                {"name": "", "type": "uint256"}
            ], "type": "function"},
        ]

        USDC_ABI = [
            {"constant": True, "inputs": [{"name": "_owner", "type": "address"}],
             "name": "balanceOf", "outputs": [{"name": "balance", "type": "uint256"}],
             "type": "function"},
        ]

        ct = w3.eth.contract(address=CT, abi=CT_ABI)
        usdc = w3.eth.contract(address=Web3.to_checksum_address(USDC_ADDRESS), abi=USDC_ABI)

        # Check if resolved on-chain
        cond_bytes = Web3.to_bytes(hexstr=condition_id)
        payout_denom = ct.functions.payoutDenominator(cond_bytes).call()
        if payout_denom == 0:
            return 0.0  # Not resolved on-chain yet

        # Get USDC before
        bal_before = usdc.functions.balanceOf(acct.address).call()

        # Redeem
        nonce = w3.eth.get_transaction_count(acct.address)
        gas_price = w3.eth.gas_price

        tx = ct.functions.redeemPositions(
            Web3.to_checksum_address(USDC_ADDRESS),
            bytes(32),  # parentCollectionId
            cond_bytes,
            [1, 2]  # Both YES and NO index sets
        ).build_transaction({
            "from": acct.address,
            "nonce": nonce,
            "gasPrice": gas_price,
            "gas": 200000,
        })
        signed = w3.eth.account.sign_transaction(tx, pk)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)

        if receipt["status"] == 1:
            bal_after = usdc.functions.balanceOf(acct.address).call()
            gained = (bal_after - bal_before) / 1e6
            return gained
        return 0.0

    except Exception as e:
        print(f"    Redeem error: {e}")
        return 0.0


# ============================================================================
# BOOK SNAPSHOT + ORDER EXECUTION
# ============================================================================

def snapshot_book(pm_client, contract, side="yes"):
    token_id = contract.yes_token_id if side == "yes" else contract.no_token_id
    url = f"https://clob.polymarket.com/book?token_id={token_id}"
    try:
        data = http_requests.get(url, timeout=5).json()
        asks = sorted(data.get("asks", []), key=lambda x: float(x["price"]))
        bids = sorted(data.get("bids", []), key=lambda x: float(x["price"]), reverse=True)
        tob_ask = float(asks[0]["price"]) if asks else 0
        tob_ask_sz = float(asks[0]["size"]) if asks else 0
        tob_bid = float(bids[0]["price"]) if bids else 0
        tob_bid_sz = float(bids[0]["size"]) if bids else 0
        depth_3c = sum(float(a["price"])*float(a["size"]) for a in asks if float(a["price"]) <= tob_ask + 0.03)
        return {
            "side": side,
            "tob_ask": tob_ask, "tob_ask_size": tob_ask_sz,
            "tob_bid": tob_bid, "tob_bid_size": tob_bid_sz,
            "spread": tob_ask - tob_bid if tob_ask and tob_bid else 0,
            "depth_3c_usd": depth_3c,
            "levels": [(float(a["price"]), float(a["size"])) for a in asks[:5]],
        }
    except Exception as e:
        return {"error": str(e)}


def execute_market_order(clob_client, token_id, size_usd, neg_risk=False):
    """Execute a market order (fallback only)."""
    t0 = time.time()
    try:
        mo = MarketOrderArgs(token_id=token_id, amount=size_usd, side=BUY)
        if neg_risk:
            signed = clob_client.create_market_order(mo, options={"neg_risk": True})
        else:
            signed = clob_client.create_market_order(mo)
        resp = clob_client.post_order(signed, OrderType.FOK)
        latency_ms = (time.time() - t0) * 1000
        success = resp.get("success", False) or resp.get("status") == "matched"
        return {"success": success, "response": resp, "latency_ms": latency_ms}
    except Exception as e:
        return {
            "success": False, "error": str(e),
            "traceback": traceback.format_exc(),
            "latency_ms": (time.time() - t0) * 1000,
        }


LIMIT_PRICE = 0.53       # Data: WR=64% at $0.52-0.53, vs 38% at ≤$0.51 (adverse selection!)
SNIPE_TIMEOUT = 5        # 5s max wait — don't snipe too long (adverse selection)
MIN_SIGNAL_ABS = 0.7     # Only trade when |signal| >= 0.7 (WR=49% vs 46% for all)


class BookSniper:
    """Streams Polymarket order book via WebSocket, fires when price is right.
    
    Architecture:
    - Persistent WS + background reader task continuously updates latest_asks
    - Multiple concurrent wait_for_price() calls just poll the dict
    - No WS read contention between coins
    """
    
    WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    
    def __init__(self):
        self.latest_asks = {}  # token_id -> latest best ask
        self.ws = None
        self._connected = False
        self._reader_task = None
    
    async def connect(self):
        """Establish persistent WS connection + start background reader."""
        try:
            self.ws = await websockets.connect(self.WS_URL, ping_interval=20, ping_timeout=10)
            self._connected = True
            # Start background reader
            self._reader_task = asyncio.create_task(self._background_reader())
            print("  [OK] Polymarket book stream connected")
        except Exception as e:
            print(f"  [!] Book stream connect failed: {e}")
            self._connected = False
    
    async def _background_reader(self):
        """Continuously read WS messages and update latest_asks dict."""
        while self._connected and self.ws:
            try:
                msg = await asyncio.wait_for(self.ws.recv(), timeout=30)
                data = json.loads(msg)
                
                if isinstance(data, list):
                    for item in data:
                        if item.get("event_type") == "book":
                            asset_id = item.get("asset_id", "")
                            asks = item.get("asks", [])
                            if asks and asset_id:
                                best_ask = min(float(a["price"]) for a in asks)
                                self.latest_asks[asset_id] = best_ask
                
                elif isinstance(data, dict):
                    if data.get("event_type") == "price_change":
                        for change in data.get("price_changes", []):
                            asset_id = change.get("asset_id", "")
                            price = float(change.get("price", 0))
                            side = change.get("side", "")
                            if asset_id and price > 0 and side == "SELL":
                                cur = self.latest_asks.get(asset_id, 999)
                                self.latest_asks[asset_id] = min(cur, price)
            
            except asyncio.TimeoutError:
                continue  # Keep alive
            except websockets.exceptions.ConnectionClosed:
                self._connected = False
                print("  [!] Book stream disconnected, will reconnect")
                break
            except Exception:
                continue
    
    async def subscribe(self, token_id):
        """Subscribe to book updates for a token. Call BEFORE model runs."""
        if not self._connected:
            await self.connect()
        if not self._connected:
            return
        try:
            msg = json.dumps({
                "auth": {},
                "type": "subscribe",
                "channel": "market",
                "assets_ids": [token_id],
            })
            await self.ws.send(msg)
            self.latest_asks[token_id] = 0  # Will be updated by background reader
        except Exception as e:
            self._connected = False
    
    async def wait_for_price(self, token_id, max_price, timeout_s):
        """Wait for ask to drop to max_price. Just polls the dict (fast)."""
        t0 = time.time()
        best_seen = 999
        
        while time.time() - t0 < timeout_s:
            ask = self.latest_asks.get(token_id, 0)
            if ask > 0:
                best_seen = min(best_seen, ask)
                if ask <= max_price:
                    return ask  # Price is right!
            
            await asyncio.sleep(0.1)  # Check every 100ms
        
        return 0  # Timeout
    
    async def unsubscribe(self, token_id):
        """Unsubscribe from a token."""
        if not self._connected or not self.ws:
            return        
        try:
            msg = json.dumps({
                "type": "unsubscribe", 
                "channel": "market",
                "assets_ids": [token_id],
            })
            await self.ws.send(msg)
            self.latest_asks.pop(token_id, None)
        except Exception:
            pass


async def execute_snipe_order(clob_client, pm_client, contract, token_id, token_side,
                              size_usd, neg_risk, book_sniper):
    """Sniper execution using WebSocket book stream.
    
    The book_sniper is already subscribed and streaming — just wait for price.
    When ask ≤ $0.51, fire FOK market order instantly.
    """
    t0 = time.time()
    
    # Wait for good price (book is already streaming from subscribe() call)
    ask = await book_sniper.wait_for_price(token_id, LIMIT_PRICE, SNIPE_TIMEOUT)
    
    if ask <= 0:
        # Timeout — try one REST snapshot as fallback
        wait_s = round(time.time() - t0, 1)
        book = snapshot_book(pm_client, contract, side=token_side)
        rest_ask = book.get("tob_ask", 0) or 0
        if rest_ask <= 0 or rest_ask > LIMIT_PRICE:
            return {
                "success": False,
                "response": {"status": "SNIPE_TIMEOUT"},
                "latency_ms": (time.time() - t0) * 1000,
                "fill_type": "SNIPE_TIMEOUT",
                "snipe_wait_s": wait_s,
                "snipe_ask": rest_ask,
                "best_ask_seen": rest_ask,
                "error": f"No fill at ≤${LIMIT_PRICE} in {SNIPE_TIMEOUT}s (REST ask=${rest_ask})",
            }
        ask = rest_ask  # REST says price is good, try FOK
    
    # Price is right — FIRE FOK immediately
    try:
        mo = MarketOrderArgs(token_id=token_id, amount=size_usd, side=BUY)
        if neg_risk:
            signed = clob_client.create_market_order(mo, options={"neg_risk": True})
        else:
            signed = clob_client.create_market_order(mo)
        resp = clob_client.post_order(signed, OrderType.FOK)
        latency_ms = (time.time() - t0) * 1000
        success = resp.get("success", False) or resp.get("status") == "matched"
        
        return {
            "success": success,
            "response": resp,
            "latency_ms": latency_ms,
            "fill_type": "SNIPE",
            "snipe_wait_s": round(time.time() - t0, 1),
            "snipe_ask": ask,
        }
    except Exception as e:
        return {
            "success": False, "error": str(e),
            "traceback": traceback.format_exc(),
            "latency_ms": (time.time() - t0) * 1000,
        }


# ============================================================================
# REAL TRADER
# ============================================================================

class RealTrader:
    def __init__(self, clob_client, pm_client):
        self.clob = clob_client
        self.pm = pm_client
        self.capital = STARTING_CAPITAL
        self.trades = []
        self.pending = {}  # trade_id -> trade dict
        self.total_real_pnl = 0.0
        self.consecutive_errors = 0
        self.redeemed_conditions = set()  # Successfully redeemed
        self.traded_slugs = set()  # DEDUP: slugs we've already traded
        self.session_start_ts = int(time.time())  # Track when this session started
        # Redemption queue: condition_id -> {slug, added_ts, attempts}
        self.redemption_queue = {}
        self.in_flight_capital = 0.0  # Capital locked in unsettled positions
        self.load_state()

    def load_state(self):
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as f:
                    state = json.load(f)
                self.capital = state.get("capital", STARTING_CAPITAL)
                self.trades = state.get("trades", [])
                self.total_real_pnl = state.get("total_real_pnl", 0.0)
                self.redeemed_conditions = set(state.get("redeemed_conditions", []))
                self.redemption_queue = state.get("redemption_queue", {})
                self.traded_slugs = set(state.get("traded_slugs", []))
                n_queue = len(self.redemption_queue)
                print(f"  Loaded state: ${self.capital:,.2f} capital, {len(self.trades)} trades, "
                      f"PnL=${self.total_real_pnl:+,.2f}, {n_queue} pending redemptions, "
                      f"{len(self.traded_slugs)} traded slugs")
            except Exception:
                pass
        # Rebuild traded_slugs from trade log if empty
        if not self.traded_slugs and TRADE_LOG.exists():
            with open(TRADE_LOG) as f:
                for line in f:
                    t = json.loads(line)
                    slug = t.get('slug', '')
                    if slug:
                        self.traded_slugs.add(slug)
            print(f"  Rebuilt {len(self.traded_slugs)} traded slugs from trade log")

    def save_state(self):
        state = {
            "capital": self.capital,
            "trades": self.trades[-2000:],
            "total_real_pnl": self.total_real_pnl,
            "redeemed_conditions": list(self.redeemed_conditions),
            "redemption_queue": self.redemption_queue,
            "traded_slugs": list(self.traded_slugs),
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def is_capital_available(self, trade_size):
        """Check if we have enough free capital (not locked in-flight)."""
        max_in_flight = self.capital * 0.40  # Max 40% of capital in-flight
        if self.in_flight_capital + trade_size > max_in_flight:
            return False
        return True

    def sweep_redemptions(self):
        """Integrated redemption sweep — called every bar, no daemon needed."""
        if not self.redemption_queue:
            return
        try:
            from web3 import Web3
            w3 = Web3(Web3.HTTPProvider('https://polygon-bor-rpc.publicnode.com',
                                         request_kwargs={'timeout': 10}))
            pk = os.environ.get('POLYGON_PRIVATE_KEY', '')
            if not pk:
                return
            acct = w3.eth.account.from_key(pk)
            CT = Web3.to_checksum_address('0x4D97DCd97eC945f40cF65F87097ACe5EA0476045')
            USDC = Web3.to_checksum_address('0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174')
            CT_ABI = [
                {'constant':True,'inputs':[{'name':'account','type':'address'},{'name':'id','type':'uint256'}],
                 'name':'balanceOf','outputs':[{'name':'','type':'uint256'}],'type':'function'},
                {'constant':False,'inputs':[
                    {'name':'collateralToken','type':'address'},{'name':'parentCollectionId','type':'bytes32'},
                    {'name':'conditionId','type':'bytes32'},{'name':'indexSets','type':'uint256[]'}
                ],'name':'redeemPositions','outputs':[],'type':'function'},
                {'constant':True,'inputs':[{'name':'conditionId','type':'bytes32'}],
                 'name':'payoutDenominator','outputs':[{'name':'','type':'uint256'}],'type':'function'},
            ]
            ct = w3.eth.contract(address=CT, abi=CT_ABI)
            nonce = w3.eth.get_transaction_count(acct.address)
            settled = 0
            to_remove = []
            for cond_hex, info in list(self.redemption_queue.items()):
                try:
                    token_id = int(info.get('token_id', 0))
                    if token_id == 0:
                        to_remove.append(cond_hex)
                        continue
                    bal = ct.functions.balanceOf(acct.address, token_id).call()
                    if bal == 0:
                        to_remove.append(cond_hex)
                        self.in_flight_capital = max(0, self.in_flight_capital - info.get('size', 1))
                        continue
                    cond_bytes = Web3.to_bytes(hexstr=cond_hex)
                    payout = ct.functions.payoutDenominator(cond_bytes).call()
                    if payout == 0:
                        continue  # Not settled yet
                    tx = ct.functions.redeemPositions(USDC, bytes(32), cond_bytes, [1, 2]).build_transaction({
                        'from': acct.address, 'nonce': nonce,
                        'gasPrice': w3.eth.gas_price, 'gas': 200000
                    })
                    signed = w3.eth.account.sign_transaction(tx, pk)
                    w3.eth.send_raw_transaction(signed.raw_transaction)
                    nonce += 1
                    settled += 1
                    to_remove.append(cond_hex)
                    self.in_flight_capital = max(0, self.in_flight_capital - info.get('size', 1))
                except Exception as e:
                    if 'nonce' in str(e).lower():
                        nonce = w3.eth.get_transaction_count(acct.address)
            for c in to_remove:
                self.redemption_queue.pop(c, None)
                self.redeemed_conditions.add(c)
            if settled > 0:
                now_str = time.strftime('%H:%M:%S')
                print(f"    Redemption sweep: {settled} settled, "
                      f"{len(self.redemption_queue)} still pending, "
                      f"in-flight=${self.in_flight_capital:.0f}", flush=True)
        except Exception as e:
            pass  # Non-critical, will retry next bar

    def should_stop(self):
        if self.capital < MIN_CAPITAL:
            print(f"\n  CAPITAL BELOW ${MIN_CAPITAL} -- AUTO-STOPPING")
            return True
        if self.consecutive_errors >= 5:
            print(f"\n  {self.consecutive_errors} CONSECUTIVE ERRORS -- PAUSING")
            return True
        return False

    async def open_position(self, symbol, candle_open_time, direction,
                      signal_value, weights, contract=None, book_sniper=None,
                      engine_type="mr"):
        trade_start = time.time()
        now_ts = datetime.now(timezone.utc).isoformat()
        trade_id = f"{symbol}_5m_{candle_open_time}"

        if not contract or not contract.yes_token_id:
            print(f"    [!] No contract found, skipping trade")
            return None

        # DEDUP: Don't trade the same contract twice
        slug = getattr(contract, 'slug', '') or ''
        if slug and slug in self.traded_slugs:
            print(f"    [!] Already traded {slug}, skipping duplicate")
            return None

        # SIGNAL FILTER: Weak signals have much lower WR
        # LGB engine already filters by probability threshold, so skip this check
        if engine_type != "lgb" and abs(signal_value) < MIN_SIGNAL_ABS:
            print(f"    [!] Signal too weak ({signal_value:+.3f}), need |sig|>={MIN_SIGNAL_ABS}")
            return None

        # CAPITAL CHECK: Don't trade if too much is in-flight
        planned_size = min(compute_trade_size(signal_value, vol_expanded=False, engine_type=engine_type),
                           MAX_TRADE_SIZE, self.capital * 0.05)
        if not self.is_capital_available(planned_size):
            print(f"    [!] Capital locked: ${self.in_flight_capital:.0f} in-flight, skipping")
            return None

        # Determine token
        if direction == "UP":
            token_side, token_id = "yes", contract.yes_token_id
        else:
            token_side, token_id = "no", contract.no_token_id
        
        # Pre-book snapshot for logging
        pre_book = snapshot_book(self.pm, contract, side=token_side)
        tob_ask = pre_book.get("tob_ask", 0) or 0

        # Execute — tiered sizing
        size = min(compute_trade_size(signal_value, vol_expanded=False),
                    MAX_TRADE_SIZE, self.capital * 0.05)
        try:
            neg_risk = self.clob.get_neg_risk(token_id)
        except:
            neg_risk = False

        # EXECUTION: Strong signals → immediate FOK (adverse selection at cheap prices!)
        # Data: fills at $0.51 have 38% WR, fills at $0.52-0.53 have 64% WR
        if abs(signal_value) >= 1.0:
            # Very strong signal (56% WR) → fire immediately, don't snipe
            fill = execute_market_order(self.clob, token_id, size, neg_risk=neg_risk)
            if book_sniper:
                await book_sniper.unsubscribe(token_id)
        elif book_sniper:
            # Moderate signal → use sniper but with short timeout
            fill = await execute_snipe_order(
                self.clob, self.pm, contract, token_id, token_side,
                size, neg_risk, book_sniper)
            await book_sniper.unsubscribe(token_id)
        else:
            fill = execute_market_order(self.clob, token_id, size, neg_risk=neg_risk)
        
        if not fill.get("success"):
            coin_name = BINANCE_TO_PM.get(symbol, symbol)
            wait_s = fill.get("snipe_wait_s", 0)
            print(f"    [SNIPE] {coin_name} no fill at ≤${LIMIT_PRICE} "
                  f"(book=${tob_ask:.3f}, waited {wait_s}s)")
            with open(TRADE_LOG, "a") as f:
                f.write(json.dumps({
                    "event": "FILL_SKIP", "symbol": symbol,
                    "direction": direction, "signal_value": signal_value,
                    "planned_size": size, "tob_ask": tob_ask,
                    "slug": slug, "time": now_ts,
                    "reason": fill.get("fill_type", "NO_FILL"),
                    "snipe_wait_s": wait_s,
                }, default=str) + "\n")
            return None
        else:
            wait_s = fill.get("snipe_wait_s", 0)
            snipe_ask = fill.get("snipe_ask", tob_ask)
            print(f"    [SNIPE] Filled at ≤${LIMIT_PRICE} "
                  f"(book=${snipe_ask:.3f}, waited {wait_s}s)")

        # Book snapshot AFTER
        post_book = snapshot_book(self.pm, contract, side=token_side)

        expected_price = pre_book.get("tob_ask", 0.50) if pre_book.get("tob_ask") else 0.50
        order_accepted = fill.get("success", False)
        fill_response = fill.get("response", {})
        fee_rate = compute_polymarket_fee(expected_price)
        fee_usd = fee_rate * size

        # Extract actual fill info from response
        taking_amount = float(fill_response.get("takingAmount", 0)) if isinstance(fill_response, dict) else 0
        making_amount = float(fill_response.get("makingAmount", 0)) if isinstance(fill_response, dict) else 0
        actual_price = making_amount / taking_amount if taking_amount > 0 else expected_price

        trade = {
            "id": trade_id,
            "version": VERSION,
            "symbol": symbol,
            "coin": BINANCE_TO_PM.get(symbol, ""),
            "direction": direction,
            "signal": signal_value,
            "top_weights": dict(sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]),
            "size_usd": size,
            "expected_price": expected_price,
            "actual_price": actual_price,
            "shares_received": taking_amount,
            "fee_rate": fee_rate,
            "fee_usd": fee_usd,
            "token_side": token_side,
            "token_id_short": token_id[:20] + "...",
            "full_token_id": token_id,
            "order_accepted": order_accepted,
            "fill_response": str(fill_response)[:500],
            "api_latency_ms": fill.get("latency_ms", 0),
            "fill_error": fill.get("error", None),
            "pre_book": pre_book,
            "post_book": post_book,
            "slug": contract.slug,
            "question": contract.question,
            "condition_id": contract.condition_id,
            "time_remaining_s": contract.time_remaining,
            "time_since_roll_s": INTERVAL_SECONDS - contract.time_remaining,
            "status": "OPEN" if order_accepted else "FAILED",
            "opened_at": now_ts,
        }

        if order_accepted:
            self.pending[trade_id] = trade
            self.consecutive_errors = 0
            # DEDUP: Mark this slug as traded
            if slug:
                self.traded_slugs.add(slug)
            # IN-FLIGHT tracking
            self.in_flight_capital += size
            # REDEMPTION QUEUE: Add for integrated sweep
            cond_id = contract.condition_id
            if cond_id:
                self.redemption_queue[cond_id] = {
                    'slug': slug,
                    'token_id': token_id,
                    'size': size,
                    'added_ts': int(time.time()),
                }
        else:
            self.consecutive_errors += 1

        with open(TRADE_LOG, "a") as f:
            f.write(json.dumps({"event": "OPEN", **trade}, default=str) + "\n")
        with open(BOOK_LOG, "a") as f:
            f.write(json.dumps({
                "event": "TRADE", "time": now_ts, "coin": trade["coin"],
                "direction": direction, "pre": pre_book, "post": post_book,
            }, default=str) + "\n")

        self.save_state()
        return trade

    def close_position(self, trade_id, pm_outcome):
        """Close position using Polymarket resolution (not Binance).
        pm_outcome: 'UP' or 'DOWN' from check_polymarket_resolution()
        """
        if trade_id not in self.pending:
            return None

        trade = self.pending.pop(trade_id)
        if not trade.get("order_accepted", False):
            return None

        direction = trade["direction"]
        size = trade["size_usd"]
        entry_price = trade["actual_price"] or trade["expected_price"]
        fee = trade["fee_usd"]
        shares = trade.get("shares_received", size / entry_price)

        # Binary payoff based on POLYMARKET resolution
        won = (direction == pm_outcome)

        if won:
            # Shares pay $1 each, cost was entry_price per share
            pnl = shares * (1.0 - entry_price) - fee  # Approximation
            trade["result"] = "WIN"
        else:
            # Shares worth $0
            pnl = -(size + fee)
            trade["result"] = "LOSS"

        trade["pnl"] = pnl
        trade["pm_outcome"] = pm_outcome
        trade["status"] = "CLOSED"
        trade["closed_at"] = datetime.now(timezone.utc).isoformat()

        self.capital += pnl
        self.total_real_pnl += pnl
        self.trades.append(trade)
        self.save_state()

        with open(TRADE_LOG, "a") as f:
            f.write(json.dumps({"event": "CLOSE", **trade}, default=str) + "\n")

        # Fill quality CSV
        if not FILL_LOG.exists():
            with open(FILL_LOG, "w") as f:
                f.write("time,coin,direction,expected_price,actual_price,size,fee,result,pnl,"
                        "tob_ask,spread,depth_3c,latency_ms,pm_outcome\n")
        with open(FILL_LOG, "a") as f:
            pre = trade.get("pre_book", {})
            f.write(f"{trade['opened_at']},{trade['coin']},{trade['direction']},"
                    f"{trade['expected_price']:.3f},{entry_price:.3f},{size:.2f},"
                    f"{fee:.3f},{trade['result']},{pnl:.2f},"
                    f"{pre.get('tob_ask',0):.3f},{pre.get('spread',0):.3f},"
                    f"{pre.get('depth_3c_usd',0):.1f},{trade.get('api_latency_ms',0):.0f},"
                    f"{pm_outcome}\n")

        # Add to redemption queue (will retry until on-chain settlement)
        if trade.get("condition_id"):
            cond = trade["condition_id"]
            if cond not in self.redeemed_conditions and cond not in self.redemption_queue:
                self.redemption_queue[cond] = {
                    "slug": trade.get("slug", ""),
                    "added_ts": time.time(),
                    "attempts": 0,
                }
                self.save_state()

        return trade

    def sweep_redemption_queue(self):
        """Legacy wrapper — delegates to integrated sweep_redemptions."""
        self.sweep_redemptions()

    def rebuild_redemption_queue(self):
        """Scan all trades and ensure every condition_id is either redeemed or in queue."""
        added = 0
        for trade in self.trades:
            cond = trade.get("condition_id", "")
            if not cond:
                continue
            if cond in self.redeemed_conditions or cond in self.redemption_queue:
                continue
            self.redemption_queue[cond] = {
                "slug": trade.get("slug", ""),
                "added_ts": time.time(),
                "attempts": 0,
            }
            added += 1
        if added:
            print(f"    Rebuilt queue: added {added} missing redemptions")
            self.save_state()

    def get_stats(self):
        if not self.trades: return {}
        recent = self.trades[-100:]
        wins = sum(1 for t in recent if t.get("result") == "WIN")
        total = len(recent)
        return {
            "capital": self.capital,
            "total_trades": len(self.trades),
            "recent_wr": wins / max(total, 1),
            "recent_pnl": sum(t.get("pnl", 0) for t in recent),
            "all_time_pnl": self.total_real_pnl,
            "pending": len(self.pending),
        }


def update_parquet_data():
    """Download latest candles from Binance and update parquet files.
    Called at startup to ensure data is current before trading."""
    print("  [DATA] Checking data freshness...")
    for symbol in SYMBOLS:
        parquet_path = DATA_DIR / f"{symbol}_{INTERVAL}.parquet"
        if not parquet_path.exists():
            print(f"    [!] {symbol}: parquet not found!")
            continue

        old_df = pd.read_parquet(parquet_path)
        last_bar = old_df.index[-1]
        now = pd.Timestamp.now(tz="UTC")
        staleness = (now - last_bar).total_seconds() / 60

        if staleness > 10:  # More than 10 minutes stale
            print(f"    {SYMBOL_NAMES[symbol]}: stale by {staleness:.0f}min, updating...")
            try:
                url = "https://api.binance.com/api/v3/klines"
                resp = http_requests.get(url, params={
                    "symbol": symbol, "interval": INTERVAL, "limit": 1000
                }, timeout=10)
                data = resp.json()
                rows = []
                for k in data:
                    rows.append({
                        "open_time": pd.Timestamp(k[0], unit="ms", tz="UTC"),
                        "open": float(k[1]), "high": float(k[2]),
                        "low": float(k[3]), "close": float(k[4]),
                        "volume": float(k[5]), "close_time": int(k[6]),
                        "quote_volume": float(k[7]), "trades": int(k[8]),
                        "taker_buy_base": float(k[9]),
                        "taker_buy_quote": float(k[10]),
                    })
                new_df = pd.DataFrame(rows)
                new_df.index = pd.DatetimeIndex(new_df["open_time"])
                new_df = new_df.drop(columns=["open_time"])

                combined = pd.concat([old_df, new_df])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
                combined.to_parquet(parquet_path)

                new_last = combined.index[-1]
                gap_check = combined.index.to_series().diff().dt.total_seconds()
                max_gap = gap_check.max()
                n_gaps = (gap_check > 310).sum()  # > 5min + 10s tolerance

                print(f"    {SYMBOL_NAMES[symbol]}: updated to {new_last} "
                      f"({len(combined)} bars, {n_gaps} gaps, max_gap={max_gap:.0f}s)")
            except Exception as e:
                print(f"    {SYMBOL_NAMES[symbol]}: update failed: {e}")
        else:
            print(f"    {SYMBOL_NAMES[symbol]}: fresh ({staleness:.0f}min old, {len(old_df)} bars)")

    print("  [DATA] Check complete\n")


async def run_live_feed():
    active_symbols = list(SYMBOLS)

    print(f"\n{'='*72}")
    print(f"  POLYMARKET 5m CANDLE -- REAL MONEY TRADING V2")
    print(f"  Trade Size: ${TRADE_SIZE_USD}/trade")
    print(f"  Fixes: Correct contract alignment + PM resolution + auto-redeem")
    print(f"{'='*72}")
    print(f"  Symbols:  {', '.join(SYMBOL_NAMES[s] for s in active_symbols)}")
    print()

    # Step 0: Ensure data is fresh
    update_parquet_data()

    # CLOB client
    clob = ClobClient(
        "https://clob.polymarket.com",
        key=os.getenv("POLYGON_PRIVATE_KEY"),
        chain_id=137, signature_type=0,
        funder=os.getenv("POLYGON_WALLET_ADDRESS"),
    )
    creds = clob.create_or_derive_api_creds()
    clob.set_api_creds(creds)
    print(f"  [OK] CLOB API authenticated")

    pm_client = PolymarketClient()
    print(f"  [OK] Polymarket API initialized")

    # Test
    try:
        contracts = pm_client.discover_all_current_contracts(INTERVAL)
        for c in (contracts or []):
            book = pm_client.get_orderbook(c, side="yes")
            print(f"    {c.coin.upper()}: Bid/Ask ${book.best_bid:.3f}/${book.best_ask:.3f}")
    except Exception as e:
        print(f"  [!] Contract test: {e}")

    # V4 Engines — LGB for ETH/BTC, MR for SOL
    engines = {}
    for symbol in active_symbols:
        cfg = CONFIGS[symbol]
        engine_type = cfg.get("engine", "mr")
        if engine_type == "lgb":
            print(f"\n  Initializing {SYMBOL_NAMES[symbol]} LGB engine (thresh={cfg.get('prob_threshold', 0.535)})...")
            engines[symbol] = LGBEngine(symbol, cfg)
            print(f"    LightGBM + MR features, prob_threshold={cfg.get('prob_threshold', 0.535)}")
        else:
            print(f"\n  Initializing {SYMBOL_NAMES[symbol]} MR ensemble engine (p{cfg.get('pctile', 70)})...")
            engines[symbol] = MREnsembleEngine(symbol, cfg)
            print(f"    25 MR indicator ensemble, pctile={cfg.get('pctile', 70)}")

    # Buffers — seed with enough bars for LGB training
    buffers = {}
    for symbol in active_symbols:
        cfg = CONFIGS[symbol]
        engine_type = cfg.get("engine", "mr")
        n_seed = 10000 if engine_type == "lgb" else 2000  # LGB needs more for training
        buf = KlineBuffer(symbol, max_size=n_seed + 500)
        buf.seed_from_parquet(DATA_DIR / f"{symbol}_{INTERVAL}.parquet", n_bars=n_seed)
        buffers[symbol] = buf
        df = buf.to_dataframe()
        if len(df) > 100:
            # Warm up: train initial model / compute initial threshold
            engines[symbol].update(df)
            engine = engines[symbol]
            if hasattr(engine, "threshold"):
                print(f"    [OK] {SYMBOL_NAMES[symbol]} warm ({len(df)} bars, threshold={engine.threshold:.3f})")
            elif hasattr(engine, "model") and engine.model is not None:
                print(f"    [OK] {SYMBOL_NAMES[symbol]} warm ({len(df)} bars, LGB trained)")
            else:
                print(f"    [OK] {SYMBOL_NAMES[symbol]} warm ({len(df)} bars)")

    # Trader + Book Sniper
    trader = RealTrader(clob, pm_client)
    book_sniper = BookSniper()
    await book_sniper.connect()

    # Rebuild redemption queue from trade history (catches anything missed)
    print("\n  Rebuilding redemption queue...")
    trader.rebuild_redemption_queue()
    trader.sweep_redemption_queue()

    # WebSocket
    streams = [f"{s.lower()}@kline_{INTERVAL}" for s in active_symbols]
    ws_url = f"{BINANCE_WS}/{'/'.join(streams)}"
    last_trade_id = {s: None for s in active_symbols}
    last_trade_slug = {s: None for s in active_symbols}

    print(f"\n  Connecting to Binance WebSocket...")
    print(f"  Trade delay: {TRADE_DELAY_SECONDS}s after bar close")
    print(f"\n  {'Time':<10} {'Asset':<5} {'Dir':>4} {'Signal':>8} {'Entry$':>8} "
          f"{'Result':>8} {'PnL':>10} {'Capital':>12}")
    print(f"  {'-'*80}")

    while True:
        if trader.should_stop():
            print("\n  TRADING STOPPED -- check logs")
            break

        try:
            async with websockets.connect(ws_url, ping_interval=20) as ws:
                print(f"  [OK] Connected to Binance WebSocket")

                # Collect bar closes into batches and process concurrently
                pending_bars = {}  # symbol -> kline data
                
                async for msg in ws:
                    try:
                        data = json.loads(msg)
                        kline = data.get("k", {})
                        symbol = kline.get("s", "")
                        is_closed = kline.get("x", False)

                        if symbol not in active_symbols or not is_closed:
                            continue

                        # Collect this bar close
                        pending_bars[symbol] = kline
                        
                        # If we haven't collected all symbols yet, check for more
                        # (give 200ms for other bars to arrive — they close simultaneously)
                        if len(pending_bars) < len(active_symbols):
                            try:
                                while True:
                                    extra = await asyncio.wait_for(ws.recv(), timeout=0.2)
                                    d2 = json.loads(extra)
                                    k2 = d2.get("k", {})
                                    s2 = k2.get("s", "")
                                    if s2 in active_symbols and k2.get("x", False):
                                        pending_bars[s2] = k2
                                    if len(pending_bars) >= len(active_symbols):
                                        break
                            except asyncio.TimeoutError:
                                pass  # Done collecting, process what we have
                        
                        # ===== PROCESS ALL BAR CLOSES =====
                        now_str = datetime.now().strftime('%H:%M:%S')
                        
                        # Phase 1: Close positions + add bars + discover contracts (fast)
                        trade_tasks = {}  # symbol -> (contract, direction, signal, weights, kline)
                        
                        for sym, kl in pending_bars.items():
                            name = SYMBOL_NAMES[sym]
                            pm_coin = BINANCE_TO_PM.get(sym, "")
                            
                            # 1) Close pending position
                            prev_id = last_trade_id[sym]
                            prev_slug = last_trade_slug[sym]
                            if prev_id and prev_id in trader.pending and prev_slug:
                                pm_outcome = check_polymarket_resolution(prev_slug)
                                if pm_outcome is None:
                                    binance_up = float(kl["c"]) >= float(kl["o"])
                                    pm_outcome = "UP" if binance_up else "DOWN"
                                    outcome_source = "BINANCE_FALLBACK"
                                else:
                                    outcome_source = "POLYMARKET"
                                closed = trader.close_position(prev_id, pm_outcome)
                                if closed:
                                    r = "WIN " if closed["result"] == "WIN" else "LOSS"
                                    print(f"  {now_str:<10} {name:<5} {'':>4} {'':>8} "
                                          f"{'':>8} {r:>8} ${closed['pnl']:>+9.2f} "
                                          f"${trader.capital:>11.2f} [{outcome_source}]")
                            
                            # 1b) Redemption sweep
                            trader.sweep_redemptions()
                            
                            # 2) Add bar + compute signal
                            buffers[sym].add_kline(kl)
                            df = buffers[sym].to_dataframe()
                            if len(df) < 60:
                                continue
                            
                            # 3) Discover contract + subscribe to book WS
                            contract = None
                            if pm_coin:
                                try:
                                    start_ts = get_current_contract_start_ts()
                                    contract = pm_client.discover_contract(
                                        pm_coin, INTERVAL, start_ts)
                                    if contract and contract.yes_token_id:
                                        await book_sniper.subscribe(contract.yes_token_id)
                                        if contract.no_token_id:
                                            await book_sniper.subscribe(contract.no_token_id)
                                except Exception as e:
                                    print(f"    [!] {name} contract discovery failed: {e}")
                            
                            # 4) Compute signal
                            direction, signal_val, weights = engines[sym].update(df)
                            proba_str = f" p={weights.get('proba', 0):.3f}" if isinstance(weights, dict) and 'proba' in weights else ""
                            if direction == 0:
                                cfg = CONFIGS[sym]
                                engine_type = cfg.get("engine", "mr")
                                if engine_type == "lgb":
                                    filt = weights.get("filtered", False) if isinstance(weights, dict) else False
                                    if filt:
                                        print(f"    {name}: SKIP (filtered{proba_str})")
                                last_trade_id[sym] = None
                                last_trade_slug[sym] = None
                                continue
                            
                            dir_str = "UP" if direction > 0 else "DOWN"
                            print(f"    >>> {name}: {dir_str} sig={signal_val:+.4f}{proba_str}")
                            trade_tasks[sym] = (contract, dir_str, signal_val, weights, kl)
                        
                        # Phase 2: Fire all snipe orders CONCURRENTLY
                        async def process_trade(sym, contract, dir_str, signal_val, weights, kl):
                            cfg = CONFIGS[sym]
                            trade = await trader.open_position(
                                symbol=sym,
                                candle_open_time=kl["T"],
                                direction=dir_str,
                                signal_value=signal_val,
                                weights=weights,
                                contract=contract,
                                book_sniper=book_sniper,
                                engine_type=cfg.get("engine", "mr"),
                            )
                            return sym, trade
                        
                        if trade_tasks:
                            tasks = [
                                process_trade(sym, *args) 
                                for sym, args in trade_tasks.items()
                            ]
                            results = await asyncio.gather(*tasks, return_exceptions=True)
                            
                            for result in results:
                                if isinstance(result, Exception):
                                    print(f"    [!] Trade error: {result}")
                                    continue
                                sym, trade = result
                                name = SYMBOL_NAMES[sym]
                                if trade:
                                    last_trade_id[sym] = trade["id"]
                                    last_trade_slug[sym] = trade.get("slug")
                                    dir_str = trade.get("direction", "?")
                                    arrow = "UP" if dir_str == "UP" else "DN"
                                    status = "LIVE" if trade["order_accepted"] else "FAIL"
                                    entry = trade.get("expected_price", 0.50)
                                    latency = trade.get("api_latency_ms", 0)
                                    shares = trade.get("shares_received", 0)
                                    signal_val = trade.get("signal", 0)
                                    print(f"  {now_str:<10} {name:<5} {arrow:>4} "
                                          f"{signal_val:>+8.4f} ${entry:>7.3f} "
                                          f"{status:>8} {'':>10} "
                                          f"{'':>12} [{latency:.0f}ms {shares:.1f}sh]")
                                    if contract:
                                        print(f"    -> {trade.get('question', '')}")
                                else:
                                    last_trade_id[sym] = None
                                    last_trade_slug[sym] = None
                        
                        # Phase 3: Stats
                        stats = trader.get_stats()
                        n = stats.get("total_trades", 0)
                        if n > 0 and n % 15 == 0:
                            print(f"\n  === REAL STATS ({n} trades) ===")
                            print(f"  Capital: ${stats['capital']:.2f}  "
                                  f"WR: {stats['recent_wr']:.1%}  "
                                  f"PnL: ${stats['all_time_pnl']:+.2f}")
                            print()
                        
                        # Phase 4: Sweep redemptions
                        trader.sweep_redemption_queue()
                        
                        # Clear batch
                        pending_bars.clear()

                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"  Error: {e}")
                        traceback.print_exc()

        except websockets.exceptions.ConnectionClosed:
            print(f"\n  Connection lost, reconnecting in 5s...")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"\n  WebSocket error: {e}, reconnecting in 10s...")
            await asyncio.sleep(10)


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Polymarket 5m -- REAL MONEY V2")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--redeem", action="store_true", help="Redeem all pending wins")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.stats:
        pm = PolymarketClient()
        trader = RealTrader(ClobClient("https://clob.polymarket.com"), pm)
        print(json.dumps(trader.get_stats(), indent=2))
        return

    if args.redeem:
        pm = PolymarketClient()
        trader = RealTrader(ClobClient("https://clob.polymarket.com"), pm)
        print("Sweeping redemption queue...")
        trader.rebuild_redemption_queue()
        trader.sweep_redemption_queue()
        return

    pk = os.getenv("POLYGON_PRIVATE_KEY")
    addr = os.getenv("POLYGON_WALLET_ADDRESS")
    if not pk or not addr:
        print("ERROR: Set POLYGON_PRIVATE_KEY and POLYGON_WALLET_ADDRESS in .env")
        sys.exit(1)

    print(f"  Wallet: {addr}")
    print(f"  Trade size: ${TRADE_SIZE_USD}")
    print(f"  Safety stop: ${MIN_CAPITAL}")

    if args.dry_run:
        print("\n  DRY RUN mode")
        return

    try:
        asyncio.run(run_live_feed())
    except KeyboardInterrupt:
        print("\n\n  Trading stopped (Ctrl+C). State saved.")


if __name__ == "__main__":
    main()
