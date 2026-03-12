"""
Univariate HF Alpha v2 — Isichenko IC-Weighted Approach
=========================================================
Key changes from v1:
1. No ML regression — uses IC-weighted signal combination (proven robust OOS)
2. Mark-to-market: always uses 1-bar returns for PnL
3. Walk-forward IC tracking with exponential decay
4. Signals are time-series z-scored (not cross-sectional ranked)
5. Selective trading via composite z-score threshold
6. Proper flush logging throughout

Usage:
    python univariate_hf_v2.py
"""

import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Force flush
import functools
print = functools.partial(print, flush=True)

# ============================================================================
# CONFIG
# ============================================================================
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'DOGEUSDT']
KLINES_DIR = Path('data/binance_cache/klines/15m')

FEES_BPS = 5.0
BARS_PER_DAY = 96
BARS_PER_YEAR = BARS_PER_DAY * 365

# Walk-forward: expanding window
WARMUP_BARS = BARS_PER_DAY * 30   # 30 days warmup for IC estimation
IC_HALFLIFE = BARS_PER_DAY * 7    # 7-day EMA halflife for IC tracking
IC_UPDATE_EVERY = 1               # Update IC every bar

# ============================================================================
# DATA
# ============================================================================

def load_symbol(symbol: str) -> pd.DataFrame:
    fpath = KLINES_DIR / f'{symbol}.parquet'
    df = pd.read_parquet(fpath)
    df = df.set_index('datetime').sort_index()
    df = df[~df.index.duplicated(keep='last')]
    for col in ['open','high','low','close','volume','quote_volume',
                'taker_buy_volume','taker_buy_quote_volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['trades_count'] = pd.to_numeric(df['trades_count'], errors='coerce')
    return df


# ============================================================================
# SIGNALS: Each returns a pd.Series of the same shape as the input
# All are CAUSAL (use only past data)
# ============================================================================

def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Build all signal columns. Returns DataFrame with signals + fwd_ret_1."""
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']
    qv = df['quote_volume']
    tc = df['trades_count']
    tbv = df['taker_buy_volume']
    
    returns = close.pct_change()
    safe_vol = volume.replace(0, np.nan)
    vwap = qv / safe_vol
    vwap_dev = (close - vwap) / (vwap.replace(0, np.nan))
    tbr = tbv / safe_vol
    safe_hl = (high - low).replace(0, np.nan)
    close_pos = (close - low) / safe_hl
    max_oc = np.maximum(open_, close)
    min_oc = np.minimum(open_, close)
    upper_shadow = (high - max_oc) / safe_hl
    lower_shadow = (min_oc - low) / safe_hl
    hl_range = (high - low) / close
    oc_range = (close - open_).abs() / close
    body_dom = oc_range / (hl_range + 1e-10)
    
    def sma(s, w): return s.rolling(w, min_periods=max(w//2,2)).mean()
    def ema(s, w): return s.ewm(span=w, min_periods=max(w//2,2)).mean()
    def rstd(s, w): return s.rolling(w, min_periods=max(w//2,2)).std()
    def ts_zscore(s, w):
        m = s.rolling(w, min_periods=max(w//2,2)).mean()
        st = s.rolling(w, min_periods=max(w//2,2)).std()
        return (s - m) / st.replace(0, np.nan)
    def ts_delta(s, d): return s - s.shift(d)
    
    sig = pd.DataFrame(index=df.index)
    
    # === MOMENTUM ===
    for w in [4, 8, 16, 32, 96, 192, 384]:
        sig[f'mom_{w}'] = ts_delta(close, w) / (rstd(returns, max(w,8)) * close + 1e-10)
    
    # SMA crossovers
    for fast, slow in [(4,16),(8,32),(16,96),(32,192),(96,384)]:
        sig[f'xover_{fast}_{slow}'] = (sma(close,fast) - sma(close,slow)) / (sma(close,slow)+1e-10)
    
    # EMA crossovers
    for fast, slow in [(4,16),(8,32),(16,96)]:
        sig[f'ema_x_{fast}_{slow}'] = (ema(close,fast) - ema(close,slow)) / (ema(close,slow)+1e-10)
    
    # Donchian position
    for w in [96, 192, 384]:
        ch_hi = high.rolling(w, min_periods=w//2).max()
        ch_lo = low.rolling(w, min_periods=w//2).min()
        sig[f'donch_{w}'] = (close - ch_lo) / (ch_hi - ch_lo + 1e-10) - 0.5  # center at 0
    
    # === MEAN REVERSION ===
    for w in [16, 32, 48, 96, 192]:
        sig[f'rev_z_{w}'] = -ts_zscore(close, w)
    
    for w in [16, 32, 96]:
        sig[f'vwap_rev_{w}'] = -ts_zscore(vwap_dev, w)
    
    # Bollinger
    for w in [32, 96]:
        sig[f'bb_{w}'] = -(close - sma(close,w)) / (2*rstd(close,w)+1e-10)
    
    # RSI centered
    for w in [16, 32, 96]:
        d = returns.copy()
        g = d.where(d>0, 0).rolling(w, min_periods=w//2).mean()
        l = (-d).where(d<0, 0).rolling(w, min_periods=w//2).mean()
        rs = g / (l + 1e-10)
        sig[f'rsi_{w}'] = (50 - (100 / (1+rs))) / 50
    
    # === MICROSTRUCTURE ===
    for w in [16, 32, 96]:
        sig[f'tbr_z_{w}'] = ts_zscore(tbr, w)
    for w in [4, 8, 16]:
        sig[f'tbr_d_{w}'] = ts_delta(tbr, w)
    for w in [16, 32, 96]:
        sig[f'vol_dir_{w}'] = ts_zscore(qv, w) * np.sign(returns)
    for w in [32, 96]:
        sig[f'vol_r_{w}'] = qv / (sma(qv,w)+1e-10) - 1
    
    # Cumulative flow
    signed_flow = (tbr - 0.5) * qv
    for w in [16, 32, 96]:
        sig[f'cflow_{w}'] = sma(signed_flow, w) / (sma(qv,w)+1e-10)
    
    # === VOLATILITY ===
    for w in [16, 32, 96]:
        sig[f'rvol_{w}'] = rstd(returns, w)
    sig['vol_ratio_16_96'] = rstd(returns,16) / (rstd(returns,96)+1e-10) - 1
    for w in [32, 96]:
        sig[f'vol_z_{w}'] = ts_zscore(rstd(returns,w), 192)
    for w in [16, 32, 96]:
        sig[f'hlr_z_{w}'] = ts_zscore(hl_range, w)
    
    # Parkinson vol (vectorized)
    hl_log_sq = np.log(high/low)**2
    for w in [16, 32]:
        sig[f'pvol_{w}'] = np.sqrt(hl_log_sq.rolling(w,min_periods=w//2).mean()/(4*np.log(2)))
    
    # === CANDLE PATTERNS ===
    sig['cpos_raw'] = close_pos - 0.5  # center at 0
    for w in [8, 16, 32]:
        sig[f'cpos_{w}'] = sma(close_pos, w) - 0.5
    shadow_imb = upper_shadow - lower_shadow
    for w in [8, 16]:
        sig[f'shd_imb_{w}'] = sma(shadow_imb, w)
    for w in [8, 16]:
        sig[f'body_{w}'] = sma(body_dom, w)
    
    # === INTERACTION SIGNALS ===
    mom4 = ts_delta(close, 4)
    mom8 = ts_delta(close, 8)
    sig['m4_vr'] = mom4 * (qv / (sma(qv,96)+1e-10))
    sig['m8_vr'] = mom8 * (qv / (sma(qv,96)+1e-10))
    sig['m4_cp'] = mom4 * close_pos
    sig['m8_cp'] = mom8 * close_pos
    sig['m4_tbr'] = mom4 * tbr
    sig['m4_ls'] = mom4 * lower_shadow
    sig['m4_iv'] = mom4 / (rstd(returns,32)*close + 1e-10)
    sig['m8_iv'] = mom8 / (rstd(returns,32)*close + 1e-10)
    
    # === MULTI-TIMEFRAME ===
    mom_sign_sum = sum(np.sign(ts_delta(close, w)) for w in [4,8,16,32,96,192])
    sig['mom_agree'] = mom_sign_sum / 6.0
    sig['mom_accel'] = ts_delta(ts_delta(close,4), 4)
    
    # === Skew/Kurt ===
    sig['skew_96'] = returns.rolling(96, min_periods=48).skew()
    sig['kurt_96'] = returns.rolling(96, min_periods=48).kurt()
    
    # === TARGET ===
    sig['fwd_ret_1'] = returns.shift(-1)  # forward 1-bar return (for MTM PnL + IC computation)
    
    return sig


# ============================================================================
# IC-WEIGHTED SIGNAL COMBINATION (Isichenko Approach)
# ============================================================================

class ICTracker:
    """Track rolling Spearman IC between signal and forward returns."""
    
    def __init__(self, halflife: int = IC_HALFLIFE):
        self.halflife = halflife
        self._alpha = 1 - np.exp(-np.log(2) / halflife)
        self._ema_ic = None
        self._ema_ic_sq = None
        self._n = 0
    
    def update(self, signal_val: float, fwd_ret: float):
        """Update with a single (signal, realized return) pair."""
        if not np.isfinite(signal_val) or not np.isfinite(fwd_ret):
            return
        # Simple product-based IC estimate (Pearson on z-scored signals)
        ic_obs = signal_val * np.sign(fwd_ret)  # positive if signal predicted direction
        
        if self._ema_ic is None:
            self._ema_ic = ic_obs
            self._ema_ic_sq = ic_obs**2
        else:
            self._ema_ic = self._alpha * ic_obs + (1-self._alpha) * self._ema_ic
            self._ema_ic_sq = self._alpha * ic_obs**2 + (1-self._alpha) * self._ema_ic_sq
        self._n += 1
    
    @property
    def ic(self) -> float:
        return self._ema_ic if self._ema_ic is not None else 0.0
    
    @property
    def weight(self) -> float:
        """IC-based weight: max(IC, 0). Anti-predictive signals get zero."""
        if self._n < WARMUP_BARS // 2:
            return 0.0
        return max(self.ic, 0.0)
    
    @property
    def is_active(self) -> bool:
        return self._n >= WARMUP_BARS // 2 and self.ic > 0


class BatchICTracker:
    """Efficient batch IC tracking using rolling rank correlation."""
    
    def __init__(self, window: int = BARS_PER_DAY * 7):
        self.window = window
    
    def compute_rolling_ic(self, signal: pd.Series, fwd_ret: pd.Series) -> pd.Series:
        """Compute rolling rank correlation (Spearman IC) between signal and forward returns."""
        # For univariate time-series, we use rolling product of z-scores
        # This is a continuous version of "did the signal predict the direction?"
        sig_z = (signal - signal.rolling(self.window, min_periods=self.window//2).mean()) / \
                (signal.rolling(self.window, min_periods=self.window//2).std() + 1e-10)
        ret_z = (fwd_ret - fwd_ret.rolling(self.window, min_periods=self.window//2).mean()) / \
                (fwd_ret.rolling(self.window, min_periods=self.window//2).std() + 1e-10)
        
        # Rolling correlation
        ic = (sig_z * ret_z).rolling(self.window, min_periods=self.window//2).mean()
        return ic


def run_ic_weighted_backtest(
    signals_df: pd.DataFrame,
    symbol: str,
    threshold: float = 1.0,
    ic_window: int = BARS_PER_DAY * 14,  # 14-day IC estimation window
    holding_bars: int = 1,
    max_position: float = 1.0,
    start_date: str = '2024-06-01',
    end_date: str = '2025-03-11',
    verbose: bool = True,
) -> dict:
    """
    IC-weighted signal combination backtest.
    
    Walk-forward: at each bar t, the IC is estimated using data [t-ic_window, t-1].
    Only signals with IC > 0 contribute. Combined signal = IC-weighted sum of z-scored signals.
    Trade when |combined_signal| > threshold.
    
    PnL is MARK-TO-MARKET using 1-bar returns.
    """
    t0 = time.time()
    
    sig_cols = [c for c in signals_df.columns if c != 'fwd_ret_1']
    n_signals = len(sig_cols)
    fwd_ret = signals_df['fwd_ret_1'].values
    
    # Eval period
    eval_mask = (signals_df.index >= start_date) & (signals_df.index < end_date)
    eval_idx = signals_df.index[eval_mask]
    n_eval = len(eval_idx)
    
    if n_eval == 0:
        return {'sharpe': 0, 'symbol': symbol}
    
    # Map eval dates to integer positions in full DataFrame
    all_idx = signals_df.index
    eval_start_pos = all_idx.get_loc(eval_idx[0])
    
    # Pre-compute z-scored signals over the entire dataset
    # Z-score each signal using a rolling window (causal)
    z_window = ic_window
    sig_values = {}
    for col in sig_cols:
        raw = signals_df[col].values.astype(np.float64)
        # Rolling z-score
        s = pd.Series(raw, index=signals_df.index)
        m = s.rolling(z_window, min_periods=z_window//2).mean()
        st = s.rolling(z_window, min_periods=z_window//2).std()
        z = ((s - m) / st.replace(0, np.nan)).values
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        z = np.clip(z, -5, 5)
        sig_values[col] = z
    
    if verbose:
        print(f'  {symbol}: z-scoring {n_signals} signals ({time.time()-t0:.1f}s)')
    
    # Pre-compute rolling IC for each signal (vectorized)
    # IC at time t = correlation(signal[t-w:t], fwd_ret[t-w:t])
    # This is computed using ONLY past data (causal)
    rolling_ic = {}
    ic_calc = BatchICTracker(window=ic_window)
    for col in sig_cols:
        sig_series = pd.Series(sig_values[col], index=signals_df.index)
        ret_series = pd.Series(fwd_ret, index=signals_df.index)
        # CRITICAL: shift fwd_ret by 0 because it's already computed as close[t+1]/close[t]-1
        # The IC should measure: does signal[t-k] predict fwd_ret[t-k] = ret[t-k+1]?
        # So we correlate signal[t-k] with fwd_ret[t-k] which is already aligned
        rolling_ic[col] = ic_calc.compute_rolling_ic(sig_series, ret_series).values
    
    if verbose:
        print(f'  {symbol}: rolling ICs computed ({time.time()-t0:.1f}s)')
    
    # Walk-forward: for each eval bar, use IC[t-1] to weight signals at time t
    positions = np.zeros(n_eval)
    scores = np.zeros(n_eval)
    
    for i in range(n_eval):
        t_pos = eval_start_pos + i  # position in full array
        
        if t_pos < 1:
            continue
        
        # Get IC weights from PREVIOUS bar (no lookahead)
        ic_weights = {}
        total_ic = 0
        for col in sig_cols:
            ic_val = rolling_ic[col][t_pos - 1]  # IC estimated up to t-1
            if np.isfinite(ic_val) and ic_val > 0:
                ic_weights[col] = ic_val
                total_ic += ic_val
        
        if total_ic < 1e-10 or len(ic_weights) < 3:
            continue
        
        # Combine signals: IC-weighted sum of z-scores
        composite = 0.0
        for col, ic_w in ic_weights.items():
            composite += (ic_w / total_ic) * sig_values[col][t_pos]
        
        scores[i] = composite
        
        # Position decision
        if composite > threshold:
            positions[i] = min(composite / threshold, max_position)
        elif composite < -threshold:
            positions[i] = max(composite / threshold, -max_position)
    
    # Apply holding period
    if holding_bars > 1:
        for i in range(1, n_eval):
            if positions[i] == 0 and positions[i-1] != 0:
                bars_held = 0
                for j in range(i-1, max(i-holding_bars, -1), -1):
                    if j < 0 or positions[j] == 0:
                        break
                    bars_held += 1
                if bars_held < holding_bars:
                    positions[i] = positions[i-1]
    
    # Mark-to-market PnL (ALWAYS 1-bar returns)
    mtm_returns = fwd_ret[eval_start_pos:eval_start_pos + n_eval]
    mtm_returns = np.nan_to_num(mtm_returns, nan=0.0)
    
    gross_pnl = positions * mtm_returns
    pos_changes = np.abs(np.diff(positions, prepend=0))
    fees = pos_changes * FEES_BPS / 10_000
    net_pnl = gross_pnl - fees
    
    # Metrics
    pnl_mean = np.mean(net_pnl)
    pnl_std = np.std(net_pnl, ddof=1)
    sharpe = (pnl_mean / pnl_std) * np.sqrt(BARS_PER_YEAR) if pnl_std > 0 else 0.0
    total_return = np.sum(net_pnl)
    cum_pnl = np.cumsum(net_pnl)
    running_max = np.maximum.accumulate(cum_pnl)
    max_dd = np.min(cum_pnl - running_max) if len(cum_pnl) > 0 else 0.0
    traded = net_pnl[positions != 0]
    win_rate = np.mean(traded > 0) if len(traded) > 0 else 0.0
    n_trades = int(np.sum(pos_changes > 0))
    hit_rate = np.mean(positions != 0)
    
    # Count active signals
    n_active_avg = 0
    sample_points = list(range(0, n_eval, max(n_eval//10, 1)))
    for i in sample_points:
        t_pos = eval_start_pos + i
        if t_pos < 1:
            continue
        cnt = sum(1 for col in sig_cols if np.isfinite(rolling_ic[col][t_pos-1]) and rolling_ic[col][t_pos-1] > 0)
        n_active_avg += cnt
    n_active_avg = n_active_avg / max(len(sample_points), 1)
    
    elapsed = time.time() - t0
    
    if verbose:
        print(f'  {symbol} | Sharpe: {sharpe:+.2f} | Return: {total_return*100:+.3f}% | '
              f'MaxDD: {max_dd*100:.3f}% | WR: {win_rate:.1%} | '
              f'Trades: {n_trades} | HR: {hit_rate:.1%} | '
              f'ActiveSigs: {n_active_avg:.0f}/{n_signals} | {elapsed:.1f}s')
    
    return {
        'symbol': symbol,
        'sharpe': sharpe,
        'total_return': total_return,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'n_trades': n_trades,
        'hit_rate': hit_rate,
        'n_active_signals': n_active_avg,
        'pnl_series': pd.Series(net_pnl, index=eval_idx),
        'threshold': threshold,
        'holding_bars': holding_bars,
        'ic_window': ic_window,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print('='*80)
    print('UNIVARIATE HF ALPHA v2 — ISICHENKO IC-WEIGHTED APPROACH')
    print('='*80)
    print(f'Symbols: {SYMBOLS}')
    print(f'Interval: 15m | Bars/day: {BARS_PER_DAY} | Fees: {FEES_BPS}bps')
    print(f'IC halflife: {IC_HALFLIFE} bars ({IC_HALFLIFE/BARS_PER_DAY:.0f} days)')
    print()
    
    # Phase 1: Load & build signals
    print('[Phase 1] Building signals...')
    t0 = time.time()
    signals_dict = {}
    for sym in SYMBOLS:
        df = load_symbol(sym)
        sigs = build_signals(df)
        sig_cols = [c for c in sigs.columns if c != 'fwd_ret_1']
        sigs[sig_cols] = sigs[sig_cols].replace([np.inf, -np.inf], np.nan)
        signals_dict[sym] = sigs
        print(f'  {sym}: {len(sigs)} bars, {len(sig_cols)} signals')
    print(f'  Done in {time.time()-t0:.1f}s\n')
    
    # Phase 2: Validation sweep
    print('[Phase 2] Validation sweep (2024-06 to 2024-12)...')
    print('  Sweeping: thresholds × holdings × ic_windows')
    
    thresholds = [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    holdings = [1, 2, 4, 8]
    ic_windows = [BARS_PER_DAY * 7, BARS_PER_DAY * 14, BARS_PER_DAY * 30]
    
    total_configs = len(thresholds) * len(holdings) * len(ic_windows)
    print(f'  {total_configs} configs per symbol, {total_configs * len(SYMBOLS)} total')
    
    best_val = {}
    all_val_results = []
    
    for sym_idx, sym in enumerate(SYMBOLS):
        print(f'\n--- [{sym_idx+1}/5] {sym} ---')
        sigs = signals_dict[sym]
        best_sharpe = -999
        best_result = None
        n_tried = 0
        sym_start = time.time()
        
        for ic_w in ic_windows:
            for threshold in thresholds:
                for holding in holdings:
                    n_tried += 1
                    result = run_ic_weighted_backtest(
                        sigs, sym,
                        threshold=threshold,
                        ic_window=ic_w,
                        holding_bars=holding,
                        start_date='2024-06-01',
                        end_date='2024-12-01',
                        verbose=False,
                    )
                    
                    all_val_results.append({
                        'symbol': sym, 'threshold': threshold,
                        'holding': holding, 'ic_window': ic_w,
                        'sharpe': result['sharpe'], 'trades': result['n_trades'],
                        'hit_rate': result['hit_rate'], 'return': result['total_return'],
                    })
                    
                    # Require enough trades and not always-on
                    if (result['sharpe'] > best_sharpe and 
                        result['n_trades'] > 20 and 
                        result['hit_rate'] < 0.8):
                        best_sharpe = result['sharpe']
                        best_result = result
                    
                    if n_tried % 18 == 0:
                        elapsed = time.time() - sym_start
                        rate = elapsed / n_tried
                        remain = rate * (total_configs - n_tried)
                        print(f'  {n_tried}/{total_configs} | best={best_sharpe:+.2f} | {elapsed:.0f}s / ~{remain:.0f}s left')
        
        if best_result:
            best_val[sym] = best_result
            print(f'  BEST: Sharpe={best_sharpe:+.2f} | thresh={best_result["threshold"]} | '
                  f'hold={best_result["holding_bars"]} | ic_w={best_result["ic_window"]/BARS_PER_DAY:.0f}d | '
                  f'trades={best_result["n_trades"]} | HR={best_result["hit_rate"]:.1%} | '
                  f'{time.time()-sym_start:.0f}s')
    
    # Validation collective
    print(f'\n{"="*60}')
    print('VALIDATION SUMMARY')
    for sym, r in best_val.items():
        print(f'  {sym}: Sharpe={r["sharpe"]:+.2f} thresh={r["threshold"]} hold={r["holding_bars"]} ic_w={r["ic_window"]/BARS_PER_DAY:.0f}d trades={r["n_trades"]}')
    
    val_pnl = pd.DataFrame({sym: r['pnl_series'] for sym, r in best_val.items()}).fillna(0)
    val_port = val_pnl.mean(axis=1)
    val_collective = (val_port.mean()/val_port.std(ddof=1))*np.sqrt(BARS_PER_YEAR) if val_port.std()>0 else 0
    print(f'  Collective Val Sharpe: {val_collective:+.2f}')
    
    # Phase 3: TEST SET (OOS)
    print(f'\n{"="*80}')
    print('[Phase 3] HOLDOUT TEST (2024-12 to 2025-03-11)')
    print(f'{"="*80}')
    
    test_results = {}
    for sym in SYMBOLS:
        if sym not in best_val:
            continue
        bv = best_val[sym]
        print(f'\n  Testing {sym} (thresh={bv["threshold"]} hold={bv["holding_bars"]} ic_w={bv["ic_window"]/BARS_PER_DAY:.0f}d)...')
        result = run_ic_weighted_backtest(
            signals_dict[sym], sym,
            threshold=bv['threshold'],
            ic_window=bv['ic_window'],
            holding_bars=bv['holding_bars'],
            start_date='2024-12-01',
            end_date='2025-03-11',
            verbose=True,
        )
        test_results[sym] = result
    
    # Test collective
    test_pnl = pd.DataFrame({sym: r['pnl_series'] for sym, r in test_results.items()}).fillna(0)
    test_port = test_pnl.mean(axis=1)
    test_collective = (test_port.mean()/test_port.std(ddof=1))*np.sqrt(BARS_PER_YEAR) if test_port.std()>0 else 0
    
    print(f'\n{"="*60}')
    print('GOAL CHECK')
    print(f'{"="*60}')
    for sym, r in test_results.items():
        s = 'PASS' if r['sharpe'] > 7 else 'FAIL'
        print(f'  [{s}] {sym}: Sharpe = {r["sharpe"]:+.2f} (target > 7)')
    s = 'PASS' if test_collective > 10 else 'FAIL'
    print(f'  [{s}] Collective: Sharpe = {test_collective:+.2f} (target > 10)')
    
    if all(r['sharpe'] > 7 for r in test_results.values()) and test_collective > 10:
        print('\n  🎉 ALL GOALS MET! 🎉')
    else:
        print('\n  Goals not yet met. Need more optimization.')
        print('  Returning results for further analysis...')
    
    # Save results
    pd.DataFrame(all_val_results).to_csv('data/val_sweep_v2.csv', index=False)
    
    import json
    best_configs = {}
    for sym, r in best_val.items():
        best_configs[sym] = {
            'threshold': r['threshold'],
            'holding_bars': r['holding_bars'],
            'ic_window': r['ic_window'],
            'val_sharpe': r['sharpe'],
            'test_sharpe': test_results.get(sym, {}).get('sharpe', 0),
        }
    with open('data/best_configs_v2.json', 'w') as f:
        json.dump(best_configs, f, indent=2)
    print('\nSaved configs to data/best_configs_v2.json')
    
    return best_val, test_results


if __name__ == '__main__':
    main()
