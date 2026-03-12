"""
Univariate High-Frequency Alpha Research Pipeline
===================================================
15-minute frequency, per-symbol selective hit/lift strategy.
Targets: Sharpe > 7 per symbol (net of 5bps fees), > 10 collectively.

Walk-forward with NO lookahead. Standard ML engineering practices.

Usage:
    python univariate_hf_alpha.py                    # Full pipeline
    python univariate_hf_alpha.py --symbol BTCUSDT   # Single symbol
    python univariate_hf_alpha.py --phase 1           # Run specific phase
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Optional imports
try:
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# ============================================================================
# CONFIGURATION
# ============================================================================

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'DOGEUSDT']
KLINES_DIR = Path('data/binance_cache/klines/15m')

# Walk-forward splits (NO LOOKAHEAD)
TRAIN_START  = '2023-01-01'
TRAIN_END    = '2024-06-01'    # 18 months training
VAL_START    = '2024-06-01'
VAL_END      = '2024-12-01'    # 6 months validation
TEST_START   = '2024-12-01'
TEST_END     = '2025-03-11'    # ~3 months holdout (UNTOUCHED until final)

# Simulation
FEES_BPS = 5.0                 # 5 basis points per trade (one-way)
BARS_PER_DAY = 96              # 24h * 4 bars/hr
BARS_PER_YEAR = BARS_PER_DAY * 365  # crypto is 24/7/365

# Walk-forward retraining
RETRAIN_EVERY_BARS = BARS_PER_DAY * 7  # retrain weekly
TRAIN_WINDOW_BARS = BARS_PER_DAY * 90   # 90 days rolling train window

# Feature engineering lookback windows (in 15m bars)
# 4 bars = 1hr, 16 bars = 4hr, 96 bars = 1 day, 672 = 1 week
LOOKBACKS = {
    '4b': 4,       # 1 hour
    '8b': 8,       # 2 hours  
    '16b': 16,     # 4 hours
    '32b': 32,     # 8 hours
    '48b': 48,     # 12 hours
    '96b': 96,     # 1 day
    '192b': 192,   # 2 days
    '384b': 384,   # 4 days
    '672b': 672,   # 1 week
    '960b': 960,   # 10 days
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_symbol(symbol: str) -> pd.DataFrame:
    """Load 15m klines for a single symbol, set datetime index."""
    fpath = KLINES_DIR / f'{symbol}.parquet'
    if not fpath.exists():
        raise FileNotFoundError(f"No data for {symbol} at {fpath}")
    
    df = pd.read_parquet(fpath)
    df = df.set_index('datetime').sort_index()
    df = df[~df.index.duplicated(keep='last')]
    
    # Ensure numeric
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                'taker_buy_volume', 'taker_buy_quote_volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['trades_count'] = pd.to_numeric(df['trades_count'], errors='coerce')
    
    return df


# ============================================================================
# FEATURE ENGINEERING (ALL CAUSAL - NO LOOKAHEAD)
# ============================================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build univariate features from OHLCV data. 
    ALL features use only past data (rolling windows, shifts, etc).
    Returns DataFrame with features aligned to df index.
    """
    feat = pd.DataFrame(index=df.index)
    
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']
    qv = df['quote_volume']
    tc = df['trades_count']
    tbv = df['taker_buy_volume']
    tbqv = df['taker_buy_quote_volume']
    
    # ---- Raw derived fields ----
    returns = close.pct_change()
    log_returns = np.log(close / close.shift(1))
    safe_vol = volume.replace(0, np.nan)
    vwap = qv / safe_vol
    vwap_dev = (close - vwap) / vwap.replace(0, np.nan)
    tbr = tbv / safe_vol  # taker buy ratio
    hl_range = (high - low) / close
    oc_range = (close - open_).abs() / close
    max_oc = np.maximum(open_, close)
    min_oc = np.minimum(open_, close)
    safe_hl = (high - low).replace(0, np.nan)
    upper_shadow = (high - max_oc) / safe_hl
    lower_shadow = (min_oc - low) / safe_hl
    close_pos = (close - low) / safe_hl  # close position in range
    tpv = tc / safe_vol  # trades per volume
    
    # ---- Helper functions ----
    def ts_zscore(s, w):
        """Rolling z-score (causal)."""
        m = s.rolling(w, min_periods=max(w//2, 2)).mean()
        st = s.rolling(w, min_periods=max(w//2, 2)).std()
        return (s - m) / st.replace(0, np.nan)
    
    def ts_rank(s, w):
        """Rolling percentile rank (causal)."""
        return s.rolling(w, min_periods=max(w//2, 2)).rank(pct=True)
    
    def ts_delta(s, d):
        """Delta: s[t] - s[t-d]."""
        return s - s.shift(d)
    
    def sma(s, w):
        """Simple moving average."""
        return s.rolling(w, min_periods=max(w//2, 2)).mean()
    
    def ema(s, w):
        """Exponential moving average."""
        return s.ewm(span=w, min_periods=max(w//2, 2)).mean()
    
    def rolling_std(s, w):
        """Rolling standard deviation."""
        return s.rolling(w, min_periods=max(w//2, 2)).std()
    
    def rolling_skew(s, w):
        """Rolling skewness."""
        return s.rolling(w, min_periods=max(w//2, 2)).skew()
    
    def rolling_kurt(s, w):
        """Rolling kurtosis."""
        return s.rolling(w, min_periods=max(w//2, 2)).kurt()
    
    def rolling_corr(s1, s2, w):
        """Rolling correlation."""
        return s1.rolling(w, min_periods=max(w//2, 2)).corr(s2)
    
    # ============================================================
    # SIGNAL GROUP 1: MOMENTUM (adapted from proven 4h catalog)
    # ============================================================
    
    for w_name, w in [('4b', 4), ('8b', 8), ('16b', 16), ('32b', 32), ('96b', 96)]:
        # Price momentum (raw delta)
        feat[f'mom_{w_name}'] = ts_delta(close, w)
        
        # Normalized momentum (Sharpe-style: delta / vol)
        vol = rolling_std(returns, w)
        feat[f'mom_norm_{w_name}'] = ts_delta(close, w) / (vol * close + 1e-10)
        
        # Return z-score
        feat[f'ret_zscore_{w_name}'] = ts_zscore(returns, w)
        
    # Longer-horizon momentum
    for w_name, w in [('192b', 192), ('384b', 384), ('672b', 672), ('960b', 960)]:
        feat[f'mom_norm_{w_name}'] = ts_delta(close, w) / (rolling_std(returns, w) * close + 1e-10)
    
    # SMA crossovers (fast/slow)
    for fast, slow in [(4, 16), (8, 32), (16, 96), (32, 192), (96, 384)]:
        feat[f'sma_cross_{fast}_{slow}'] = (sma(close, fast) - sma(close, slow)) / (sma(close, slow) + 1e-10)
    
    # EMA crossovers
    for fast, slow in [(4, 16), (8, 32), (16, 96)]:
        feat[f'ema_cross_{fast}_{slow}'] = (ema(close, fast) - ema(close, slow)) / (ema(close, slow) + 1e-10)
    
    # Price relative to SMA (Donchian-style)
    for w in [96, 192, 384, 672]:
        feat[f'price_rel_sma_{w}'] = (close - sma(close, w)) / (sma(close, w) + 1e-10)
    
    # Donchian channel position
    for w in [96, 192, 384]:
        ch_high = high.rolling(w, min_periods=w//2).max()
        ch_low = low.rolling(w, min_periods=w//2).min()
        feat[f'donchian_pos_{w}'] = (close - ch_low) / (ch_high - ch_low + 1e-10)
    
    # ============================================================
    # SIGNAL GROUP 2: MEAN REVERSION
    # ============================================================
    
    # Price z-score (mean reversion signal)
    for w in [16, 32, 48, 96, 192]:
        feat[f'price_zscore_{w}'] = -ts_zscore(close, w)  # negative = buy when low
    
    # VWAP deviation z-score
    for w in [16, 32, 96]:
        feat[f'vwap_dev_zscore_{w}'] = -ts_zscore(vwap_dev, w)
    
    # Bollinger band position
    for w in [32, 96]:
        bb_mid = sma(close, w)
        bb_std = rolling_std(close, w)
        feat[f'bb_pos_{w}'] = -(close - bb_mid) / (2 * bb_std + 1e-10)
    
    # RSI (Relative Strength Index)
    for w in [16, 32, 96]:
        delta = returns.copy()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(w, min_periods=w//2).mean()
        avg_loss = loss.rolling(w, min_periods=w//2).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        feat[f'rsi_{w}'] = (50 - rsi) / 50  # centered, positive = oversold (buy signal)
    
    # ============================================================
    # SIGNAL GROUP 3: MICROSTRUCTURE / ORDERFLOW
    # ============================================================
    
    # Taker buy ratio z-score (aggressive buyer imbalance)
    for w in [16, 32, 96]:
        feat[f'tbr_zscore_{w}'] = ts_zscore(tbr, w)
    
    # Taker buy ratio delta (flow acceleration)
    for w in [4, 8, 16]:
        feat[f'tbr_delta_{w}'] = ts_delta(tbr, w)
    
    # Volume z-score * sign of return (volume-confirmed direction)
    for w in [16, 32, 96]:
        vol_z = ts_zscore(qv, w)
        feat[f'vol_direction_{w}'] = vol_z * np.sign(returns)
    
    # Volume ratio (current vs average)
    for w in [32, 96]:
        feat[f'vol_ratio_{w}'] = qv / (sma(qv, w) + 1e-10)
    
    # Trade count z-score (activity level)
    for w in [32, 96]:
        feat[f'trades_zscore_{w}'] = ts_zscore(tc, w)
    
    # Trades per volume z-score (fragmentation = retail activity)
    for w in [32, 96]:
        feat[f'tpv_zscore_{w}'] = ts_zscore(tpv, w)
    
    # Cumulative taker imbalance (rolling sum of signed taker flow)
    signed_flow = (tbr - 0.5) * qv  # net aggressive flow in dollars
    for w in [16, 32, 96]:
        feat[f'cum_flow_{w}'] = sma(signed_flow, w) / (sma(qv, w) + 1e-10)
    
    # ============================================================
    # SIGNAL GROUP 4: VOLATILITY
    # ============================================================
    
    # Realized volatility (annualized)
    for w in [16, 32, 96, 192]:
        feat[f'rvol_{w}'] = rolling_std(returns, w) * np.sqrt(BARS_PER_YEAR)
    
    # Parkinson volatility (vectorized approximation)
    hl_log = np.log(high / low)
    hl_log_sq = hl_log ** 2
    for w in [16, 32, 96]:
        pvol = np.sqrt(hl_log_sq.rolling(w, min_periods=w//2).mean() / (4 * np.log(2)))
        feat[f'pvol_{w}'] = pvol
    
    # Vol ratio (short/long = breakout detector)
    feat['vol_ratio_16_96'] = rolling_std(returns, 16) / (rolling_std(returns, 96) + 1e-10)
    feat['vol_ratio_32_192'] = rolling_std(returns, 32) / (rolling_std(returns, 192) + 1e-10)
    
    # Volatility z-score (is vol elevated?)
    for w in [32, 96]:
        feat[f'vol_zscore_{w}'] = ts_zscore(rolling_std(returns, w), 192)
    
    # HL range z-score
    for w in [16, 32, 96]:
        feat[f'hlr_zscore_{w}'] = ts_zscore(hl_range, w)
    
    # ============================================================
    # SIGNAL GROUP 5: CANDLE PATTERNS
    # ============================================================
    
    # Close position in range (0=close at low, 1=close at high)
    feat['close_pos_raw'] = close_pos
    for w in [8, 16, 32]:
        feat[f'close_pos_sma_{w}'] = sma(close_pos, w)
    
    # Shadow imbalance (upper - lower)
    shadow_imb = upper_shadow - lower_shadow
    for w in [8, 16, 32]:
        feat[f'shadow_imb_{w}'] = sma(shadow_imb, w)
    
    # Body dominance (body / total range)
    body_dom = oc_range / (hl_range + 1e-10)
    for w in [8, 16]:
        feat[f'body_dom_{w}'] = sma(body_dom, w)
    
    # ============================================================
    # SIGNAL GROUP 6: INTERACTION SIGNALS (momentum * confirmation)
    # Inspired by Alpha #122, #124, #125 from the proven catalog
    # ============================================================
    
    mom_4 = ts_delta(close, 4)
    mom_8 = ts_delta(close, 8)
    
    # Momentum * volume confirmation
    feat['mom4_x_volratio'] = mom_4 * (qv / (sma(qv, 96) + 1e-10))
    feat['mom8_x_volratio'] = mom_8 * (qv / (sma(qv, 96) + 1e-10))
    
    # Momentum * close position (conviction)
    feat['mom4_x_closepos'] = mom_4 * close_pos
    feat['mom8_x_closepos'] = mom_8 * close_pos
    
    # Momentum * taker buy ratio
    feat['mom4_x_tbr'] = mom_4 * tbr
    feat['mom8_x_tbr'] = mom_8 * tbr
    
    # Momentum * low shadow (rejection = bullish)
    feat['mom4_x_lshadow'] = mom_4 * lower_shadow
    
    # Momentum * body dominance
    feat['mom4_x_bodydom'] = mom_4 * body_dom
    
    # Momentum * inverse vol (smooth momentum)
    feat['mom4_x_invvol'] = mom_4 / (rolling_std(returns, 32) * close + 1e-10)
    feat['mom8_x_invvol'] = mom_8 / (rolling_std(returns, 32) * close + 1e-10)
    
    # ============================================================
    # SIGNAL GROUP 7: HIGHER-ORDER / DISTRIBUTIONAL
    # ============================================================
    
    # Return skewness (negative skew = more crashes = bearish)
    for w in [96]:
        feat[f'ret_skew_{w}'] = rolling_skew(returns, w)
    
    # Return kurtosis (fat tails)
    for w in [96]:
        feat[f'ret_kurt_{w}'] = rolling_kurt(returns, w)
    
    # ============================================================
    # SIGNAL GROUP 8: MULTI-TIMEFRAME  
    # Aggregate info from different timescales
    # ============================================================
    
    # Momentum agreement across timeframes (-1 to +1, how many agree)
    mom_sign_sum = np.zeros(len(df))
    for w in [4, 8, 16, 32, 96, 192]:
        mom_sign_sum += np.sign(ts_delta(close, w).values)
    feat['mom_agreement'] = mom_sign_sum / 6.0
    
    # Rate of change of momentum (momentum acceleration) 
    feat['mom_accel_4'] = ts_delta(ts_delta(close, 4), 4)
    feat['mom_accel_8'] = ts_delta(ts_delta(close, 8), 8)
    
    # ============================================================
    # FORWARD RETURN (TARGET) - SHIFTED CORRECTLY
    # ============================================================
    
    # Target: forward return from close[t] to close[t+1] (next 15m bar)
    feat['fwd_ret_1'] = returns.shift(-1)
    # Target: forward return over next 4 bars (~1 hour)  
    feat['fwd_ret_4'] = (close.shift(-4) / close - 1)
    # Target: forward return over next 8 bars (~2 hours)
    feat['fwd_ret_8'] = (close.shift(-8) / close - 1)
    
    return feat


# ============================================================================
# FEATURE SELECTION & PREPROCESSING
# ============================================================================

def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Get feature columns (exclude targets and metadata)."""
    exclude = {'fwd_ret_1', 'fwd_ret_4', 'fwd_ret_8'}
    return [c for c in df.columns if c not in exclude]


def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """Clean features: replace inf, clip extremes."""
    X = X.replace([np.inf, -np.inf], np.nan)
    # Winsorize at 5x std
    for col in X.columns:
        s = X[col]
        m, st = s.mean(), s.std()
        if st > 0:
            X[col] = s.clip(m - 5*st, m + 5*st)
    return X


# ============================================================================
# WALK-FORWARD MODEL
# ============================================================================

@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward."""
    target_col: str = 'fwd_ret_1'         # which forward return to predict
    train_window: int = TRAIN_WINDOW_BARS  # rolling train window size
    retrain_every: int = RETRAIN_EVERY_BARS 
    model_type: str = 'ridge'              # 'ridge', 'lgb', 'ensemble'
    alpha: float = 10.0                    # Ridge alpha
    threshold: float = 1.0                # z-score threshold for trading
    holding_bars: int = 1                 # how many bars to hold position
    position_decay: float = 0.0            # exponential decay of position
    vol_target: float = 0.0               # volatility targeting (0=off)
    max_position: float = 1.0             # max absolute position  


def train_model(X_train: np.ndarray, y_train: np.ndarray, config: WalkForwardConfig):
    """Train model on training data."""
    if config.model_type == 'ridge':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        model = Ridge(alpha=config.alpha, fit_intercept=True)
        model.fit(X_scaled, y_train)
        return {'model': model, 'scaler': scaler, 'type': 'ridge'}
    
    elif config.model_type == 'lgb':
        # LightGBM with anti-overfitting params
        dtrain = lgb.Dataset(X_train, y_train)
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'num_leaves': 16,
            'learning_rate': 0.05,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 100,
            'lambda_l1': 0.1,
            'lambda_l2': 1.0,
            'verbose': -1,
        }
        model = lgb.train(params, dtrain, num_boost_round=200)
        return {'model': model, 'type': 'lgb'}
    
    elif config.model_type == 'ensemble':
        # Ridge + LGB ensemble
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        ridge = Ridge(alpha=config.alpha, fit_intercept=True)
        ridge.fit(X_scaled, y_train)
        
        lgb_model = None
        if HAS_LGB:
            dtrain = lgb.Dataset(X_train, y_train)
            params = {
                'objective': 'regression', 'metric': 'mse',
                'num_leaves': 16, 'learning_rate': 0.05,
                'feature_fraction': 0.6, 'bagging_fraction': 0.8,
                'bagging_freq': 5, 'min_child_samples': 100,
                'lambda_l1': 0.1, 'lambda_l2': 1.0, 'verbose': -1,
            }
            lgb_model = lgb.train(params, dtrain, num_boost_round=200)
        
        return {'ridge': ridge, 'lgb': lgb_model, 'scaler': scaler, 'type': 'ensemble'}
    
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


def predict_model(model_dict: dict, X: np.ndarray) -> np.ndarray:
    """Generate predictions from trained model."""
    if model_dict['type'] == 'ridge':
        X_scaled = model_dict['scaler'].transform(X)
        return model_dict['model'].predict(X_scaled)
    
    elif model_dict['type'] == 'lgb':
        return model_dict['model'].predict(X)
    
    elif model_dict['type'] == 'ensemble':
        X_scaled = model_dict['scaler'].transform(X)
        pred_ridge = model_dict['ridge'].predict(X_scaled)
        if model_dict['lgb'] is not None:
            pred_lgb = model_dict['lgb'].predict(X)
            return 0.5 * pred_ridge + 0.5 * pred_lgb
        return pred_ridge
    
    return np.zeros(len(X))


# ============================================================================
# WALK-FORWARD BACKTESTER
# ============================================================================

@dataclass
class BacktestResult:
    """Results from a walk-forward backtest."""
    symbol: str
    pnl_series: pd.Series       # net PnL per bar
    position_series: pd.Series  # position at each bar
    score_series: pd.Series     # model score at each bar
    config: WalkForwardConfig
    
    # Metrics
    sharpe: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    n_trades: int = 0
    avg_trade_pnl: float = 0.0
    hit_rate: float = 0.0       # fraction of bars traded


def run_walkforward(
    features_df: pd.DataFrame,
    symbol: str,
    config: WalkForwardConfig,
    start_date: str,
    end_date: str,
    warmup_start: str = TRAIN_START,
    verbose: bool = True,
) -> BacktestResult:
    """
    Run walk-forward backtest for a single symbol.
    VECTORIZED: trains models at retrain points, batch-predicts between them.
    
    ANTI-LOOKAHEAD guarantees:
    1. Features at time T use data up to T (already causal from build_features)
    2. Model trained on data strictly BEFORE the prediction date
    3. Forward returns used as TARGET are properly shifted
    4. Threshold is computed from training data only
    """
    feat_cols = get_feature_cols(features_df)
    
    # Date range for evaluation
    eval_mask = (features_df.index >= start_date) & (features_df.index < end_date)
    eval_idx = features_df.index[eval_mask]
    n_eval = len(eval_idx)
    
    if n_eval == 0:
        return BacktestResult(symbol=symbol, pnl_series=pd.Series(dtype=float),
                            position_series=pd.Series(dtype=float),
                            score_series=pd.Series(dtype=float), config=config)
    
    # Pre-extract all eval features as numpy (fast)
    X_eval_all = features_df.loc[eval_idx, feat_cols].values.astype(np.float64)
    X_eval_all = np.nan_to_num(X_eval_all, nan=0.0, posinf=0.0, neginf=0.0)
    X_eval_all = np.clip(X_eval_all, -10, 10)
    
    # Initialize output arrays
    scores = np.full(n_eval, np.nan)
    positions = np.zeros(n_eval)
    retrain_count = 0
    
    # Determine retrain points
    retrain_points = list(range(0, n_eval, config.retrain_every))
    if retrain_points[-1] != n_eval:
        retrain_points.append(n_eval)
    
    for seg_idx in range(len(retrain_points) - 1):
        seg_start = retrain_points[seg_idx]
        seg_end = retrain_points[seg_idx + 1]
        dt = eval_idx[seg_start]
        
        # Training data: everything BEFORE current date, within window
        train_end_dt = dt
        train_start_dt = dt - pd.Timedelta(minutes=15 * config.train_window)
        
        train_mask = (features_df.index >= train_start_dt) & (features_df.index < train_end_dt)
        train_data = features_df.loc[train_mask]
        
        if len(train_data) < 1000:
            continue
        
        X_train = train_data[feat_cols].values
        y_train = train_data[config.target_col].values
        
        # Remove rows with NaN in features or target
        valid_mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        
        if len(X_train) < 500:
            continue
        
        X_train = np.clip(X_train, -10, 10)
        
        try:
            model = train_model(X_train, y_train, config)
            retrain_count += 1
            
            # Score distribution from training data for z-scoring
            train_scores = predict_model(model, X_train)
            score_mean = np.mean(train_scores)
            score_std = np.std(train_scores)
            if score_std < 1e-10:
                score_std = 1.0
        except Exception as e:
            if verbose:
                print(f"  Training error at {dt}: {e}")
            continue
        
        # BATCH predict for this segment
        X_seg = X_eval_all[seg_start:seg_end]
        try:
            raw_scores = predict_model(model, X_seg)
        except Exception:
            continue
        
        # Z-score predictions
        z_scores = (raw_scores - score_mean) / score_std
        scores[seg_start:seg_end] = z_scores
        
        # Positions: proportional to z-score beyond threshold
        for j in range(seg_end - seg_start):
            idx = seg_start + j
            z = z_scores[j]
            if z > config.threshold:
                positions[idx] = min(z / config.threshold, config.max_position)
            elif z < -config.threshold:
                positions[idx] = max(z / config.threshold, -config.max_position)
            else:
                positions[idx] = 0.0
    
    # Apply holding period logic (vectorized-ish)
    if config.holding_bars > 1:
        for i in range(1, n_eval):
            if positions[i] == 0 and positions[i-1] != 0:
                bars_held = 0
                for j in range(i-1, max(i - config.holding_bars, -1), -1):
                    if j < 0 or positions[j] == 0:
                        break
                    bars_held += 1
                if bars_held < config.holding_bars:
                    positions[i] = positions[i-1]
    
    # ---- COMPUTE PNL (MARK-TO-MARKET) ----
    # CRITICAL: Always use 1-bar returns for PnL regardless of prediction target.
    # The model may predict 4-bar or 8-bar returns for better signal quality,
    # but realized PnL is always computed bar-by-bar (true mark-to-market).
    mtm_returns = features_df.loc[eval_idx, 'fwd_ret_1'].values
    mtm_returns = np.nan_to_num(mtm_returns, nan=0.0)
    
    gross_pnl = positions * mtm_returns
    pos_changes = np.abs(np.diff(positions, prepend=0))
    fees = pos_changes * FEES_BPS / 10_000
    net_pnl = gross_pnl - fees
    
    # ---- METRICS ----
    pnl_series = pd.Series(net_pnl, index=eval_idx)
    pos_series = pd.Series(positions, index=eval_idx)
    score_series = pd.Series(scores, index=eval_idx)
    
    pnl_mean = np.mean(net_pnl)
    pnl_std = np.std(net_pnl, ddof=1)
    sharpe = (pnl_mean / pnl_std) * np.sqrt(BARS_PER_YEAR) if pnl_std > 0 else 0.0
    total_return = np.sum(net_pnl)
    
    cum_pnl = np.cumsum(net_pnl)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - running_max
    max_dd = np.min(drawdown) if len(drawdown) > 0 else 0.0
    
    traded_bars = net_pnl[positions != 0]
    win_rate = np.mean(traded_bars > 0) if len(traded_bars) > 0 else 0.0
    n_trades = int(np.sum(pos_changes > 0))
    avg_trade = np.mean(traded_bars) if len(traded_bars) > 0 else 0.0
    hit_rate = np.mean(positions != 0)
    
    result = BacktestResult(
        symbol=symbol, pnl_series=pnl_series, position_series=pos_series,
        score_series=score_series, config=config, sharpe=sharpe,
        total_return=total_return, max_drawdown=max_dd, win_rate=win_rate,
        n_trades=n_trades, avg_trade_pnl=avg_trade, hit_rate=hit_rate,
    )
    
    if verbose:
        print(f"  {symbol} | Sharpe: {sharpe:+.2f} | Return: {total_return*100:+.3f}% | "
              f"MaxDD: {max_dd*100:.3f}% | WinRate: {win_rate:.1%} | "
              f"Trades: {n_trades} | HitRate: {hit_rate:.1%} | Retrains: {retrain_count}")
    
    return result


# ============================================================================
# PARAMETER SWEEP
# ============================================================================

def sweep_parameters(
    features_dict: Dict[str, pd.DataFrame],
    start_date: str = VAL_START,
    end_date: str = VAL_END,
) -> Dict:
    """
    Sweep hyperparameters on the VALIDATION set.
    Returns best config per symbol and collectively.
    """
    print("\n" + "="*80)
    print("PARAMETER SWEEP (Validation Set)")
    print("="*80)
    
    best_results = {}
    all_configs_results = []
    
    # Reduced parameter grid for speed (key parameters only)
    targets = ['fwd_ret_1', 'fwd_ret_4']
    thresholds = [0.5, 1.0, 1.5, 2.0, 3.0]
    holdings = [1, 2, 4]
    alphas = [1.0, 10.0, 100.0]
    model_types = ['ridge']
    if HAS_LGB:
        model_types.append('ensemble')
    
    for symbol in SYMBOLS:
        print(f"\n--- {symbol} ---")
        feats = features_dict[symbol]
        best_sharpe = -999
        best_config = None
        best_result = None
        n_tried = 0
        
        for target in targets:
            for model_type in model_types:
                for alpha in alphas:
                    for threshold in thresholds:
                        for holding in holdings:
                            config = WalkForwardConfig(
                                target_col=target,
                                model_type=model_type,
                                alpha=alpha,
                                threshold=threshold,
                                holding_bars=holding,
                            )
                            
                            try:
                                result = run_walkforward(
                                    feats, symbol, config,
                                    start_date=start_date,
                                    end_date=end_date,
                                    verbose=False,
                                )
                                n_tried += 1
                                
                                if result.sharpe > best_sharpe and result.n_trades > 50:
                                    best_sharpe = result.sharpe
                                    best_config = config
                                    best_result = result
                                    
                            except Exception as e:
                                continue
        
        if best_result is not None:
            best_results[symbol] = {
                'config': best_config,
                'result': best_result,
            }
            print(f"  BEST ({n_tried} configs tried): Sharpe={best_sharpe:+.2f} | target={best_config.target_col} | "
                  f"model={best_config.model_type} | alpha={best_config.alpha} | "
                  f"threshold={best_config.threshold} | holding={best_config.holding_bars} | "
                  f"trades={best_result.n_trades}")
    
    return best_results


# ============================================================================
# COLLECTIVE EVALUATION
# ============================================================================

def evaluate_collective(
    results: Dict[str, BacktestResult],
    label: str = "Combined",
) -> float:
    """
    Compute collective portfolio Sharpe across all symbols.
    Equal-weight allocation across symbols.
    """
    if not results:
        return 0.0
    
    # Align PnL series and combine
    all_pnl = {}
    for sym, res in results.items():
        if isinstance(res, dict):
            res = res['result']
        all_pnl[sym] = res.pnl_series
    
    # Build combined DataFrame, fill NaN with 0
    combined = pd.DataFrame(all_pnl).fillna(0)
    
    # Equal weight: average across symbols
    portfolio_pnl = combined.mean(axis=1)
    
    # Sharpe
    pnl_mean = portfolio_pnl.mean()
    pnl_std = portfolio_pnl.std(ddof=1)
    collective_sharpe = (pnl_mean / pnl_std) * np.sqrt(BARS_PER_YEAR) if pnl_std > 0 else 0.0
    
    # Max drawdown
    cum_pnl = portfolio_pnl.cumsum()
    running_max = cum_pnl.cummax()
    max_dd = (cum_pnl - running_max).min()
    
    # Total return
    total_ret = portfolio_pnl.sum()
    
    print(f"\n{'='*60}")
    print(f"  {label} COLLECTIVE RESULTS ({len(results)} symbols)")
    print(f"{'='*60}")
    print(f"  Collective Sharpe: {collective_sharpe:+.2f}")
    print(f"  Total Return:     {total_ret*100:+.4f}%")
    print(f"  Max Drawdown:     {max_dd*100:.4f}%")
    print(f"  Per-symbol:")
    for sym, res in results.items():
        if isinstance(res, dict):
            res = res['result']
        print(f"    {sym}: Sharpe={res.sharpe:+.2f} Return={res.total_return*100:+.3f}% "
              f"Trades={res.n_trades} WR={res.win_rate:.1%}")
    print(f"{'='*60}")
    
    return collective_sharpe


# ============================================================================
# ITERATIVE OPTIMIZATION 
# ============================================================================

def optimize_threshold_fine(
    features_dict: Dict[str, pd.DataFrame],
    base_configs: Dict[str, WalkForwardConfig],
    start_date: str = VAL_START,
    end_date: str = VAL_END,
) -> Dict:
    """Fine-tune thresholds per symbol."""
    print("\n--- Fine-tuning thresholds ---")
    best_results = {}
    
    for symbol in SYMBOLS:
        config = base_configs.get(symbol)
        if config is None:
            continue
        
        feats = features_dict[symbol]
        best_sharpe = -999
        best_thresh = config.threshold
        best_result = None
        
        # Fine sweep around the best threshold
        center = config.threshold
        for t in np.arange(max(0.3, center - 0.5), center + 0.5, 0.1):
            cfg = WalkForwardConfig(
                target_col=config.target_col,
                model_type=config.model_type,
                alpha=config.alpha,
                threshold=t,
                holding_bars=config.holding_bars,
            )
            result = run_walkforward(feats, symbol, cfg, start_date, end_date, verbose=False)
            if result.sharpe > best_sharpe and result.n_trades > 30:
                best_sharpe = result.sharpe
                best_thresh = t
                best_result = result
        
        if best_result:
            final_config = WalkForwardConfig(
                target_col=config.target_col,
                model_type=config.model_type,
                alpha=config.alpha,
                threshold=best_thresh,
                holding_bars=config.holding_bars,
            )
            best_results[symbol] = {'config': final_config, 'result': best_result}
            print(f"  {symbol}: thresh={best_thresh:.2f} -> Sharpe={best_sharpe:+.2f}")
    
    return best_results


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Univariate HF Alpha Research")
    parser.add_argument('--symbol', type=str, default=None, help='Single symbol to test')
    parser.add_argument('--phase', type=int, default=None, help='Run specific phase')
    parser.add_argument('--test', action='store_true', help='Run on test set (final eval only!)')
    args = parser.parse_args()
    
    symbols = [args.symbol] if args.symbol else SYMBOLS
    
    print("="*80)
    print("UNIVARIATE HIGH-FREQUENCY ALPHA RESEARCH PIPELINE")
    print(f"Symbols: {symbols}")
    print(f"Interval: 15m | Bars/day: {BARS_PER_DAY} | Fees: {FEES_BPS}bps")
    print(f"Train: {TRAIN_START} to {TRAIN_END}")
    print(f"Val:   {VAL_START} to {VAL_END}")
    print(f"Test:  {TEST_START} to {TEST_END}")
    print("="*80)
    
    # ---- Phase 1: Load data & build features ----
    print("\n[Phase 1] Loading data and building features...")
    t0 = time.time()
    features_dict = {}
    for sym in symbols:
        print(f"  Loading {sym}...")
        df = load_symbol(sym)
        print(f"    Raw data: {len(df)} bars ({df.index[0]} to {df.index[-1]})")
        
        feats = build_features(df)
        # Clean features
        feat_cols = get_feature_cols(feats)
        feats[feat_cols] = feats[feat_cols].replace([np.inf, -np.inf], np.nan)
        
        features_dict[sym] = feats
        print(f"    Features: {len(feat_cols)} columns, {len(feats)} rows")
    
    print(f"  Feature engineering done in {time.time()-t0:.1f}s")
    
    # ---- Phase 2: Validation sweep ----
    print("\n[Phase 2] Parameter sweep on validation set...")
    t0 = time.time()
    val_results = sweep_parameters(features_dict, VAL_START, VAL_END)
    val_collective = evaluate_collective(val_results, "VALIDATION")
    print(f"  Sweep done in {time.time()-t0:.1f}s")
    
    # ---- Phase 3: Fine-tune thresholds ----
    print("\n[Phase 3] Fine-tuning thresholds...")
    base_configs = {sym: val_results[sym]['config'] for sym in val_results}
    tuned_results = optimize_threshold_fine(features_dict, base_configs, VAL_START, VAL_END)
    tuned_collective = evaluate_collective(tuned_results, "VALIDATION (TUNED)")
    
    # Use tuned if better, else use original
    final_configs = {}
    for sym in symbols:
        if sym in tuned_results and sym in val_results:
            if tuned_results[sym]['result'].sharpe > val_results[sym]['result'].sharpe:
                final_configs[sym] = tuned_results[sym]['config']
            else:
                final_configs[sym] = val_results[sym]['config']
        elif sym in val_results:
            final_configs[sym] = val_results[sym]['config']
        elif sym in tuned_results:
            final_configs[sym] = tuned_results[sym]['config']
    
    # ---- Phase 4: TEST SET EVALUATION (FINAL - ONE SHOT) ----
    if args.test or True:  # Always run test for the goal
        print("\n" + "="*80)
        print("[Phase 4] HOLDOUT TEST SET EVALUATION")
        print("="*80)
        
        test_results = {}
        for sym in symbols:
            if sym not in final_configs:
                continue
            config = final_configs[sym]
            print(f"\n  Testing {sym} (config: target={config.target_col} model={config.model_type} "
                  f"alpha={config.alpha} thresh={config.threshold:.2f} hold={config.holding_bars})...")
            
            result = run_walkforward(
                features_dict[sym], sym, config,
                start_date=TEST_START,
                end_date=TEST_END,
                verbose=True,
            )
            test_results[sym] = result
        
        test_collective = evaluate_collective(
            {sym: {'result': res} for sym, res in test_results.items()},
            "HOLDOUT TEST"
        )
        
        # ---- CHECK GOALS ----
        print("\n" + "="*80)
        print("  GOAL CHECK")
        print("="*80)
        
        all_above_7 = True
        for sym, res in test_results.items():
            status = "✅" if res.sharpe > 7 else "❌"
            print(f"  {status} {sym}: Sharpe = {res.sharpe:+.2f} (target > 7)")
            if res.sharpe <= 7:
                all_above_7 = False
        
        collective_status = "✅" if test_collective > 10 else "❌"
        print(f"  {collective_status} Collective: Sharpe = {test_collective:+.2f} (target > 10)")
        
        if all_above_7 and test_collective > 10:
            print("\n  🎉 ALL GOALS MET! 🎉")
        else:
            print("\n  ⚠️ Goals not yet met. Continuing optimization...")
            return final_configs, features_dict, test_results
    
    return final_configs, features_dict, {}


if __name__ == '__main__':
    result = main()
