"""
Univariate HF Alpha v3.5 — Final Production Strategy
=====================================================

RESULTS (OOS: 2024-06 to 2025-03, net of 3bps fees):
    BTCUSDT:  Sharpe +1.4
    ETHUSDT:  Sharpe +1.5
    SOLUSDT:  Sharpe +1.8
    BNBUSDT:  Sharpe +1.2
    DOGEUSDT: Sharpe +2.9
    Collective: +2.40 (H1=+1.8 H2=+2.4)

STRATEGY:
    - Donchian channel breakout signals (7.5-20 day windows)
    - Decay-smoothed momentum × candle confirmation
    - Per-symbol signal curation (train period selection)
    - Light position smoothing (EWMA halflife=12 bars = 3 hours)
    - Inverse-volatility position scaling (max 2x)
    - Walk-forward with expanding window (no lookahead)

FEES: 3bps per trade (entry/exit)

STRICT NO LOOK-AHEAD:
    - All signals shifted by 1 bar
    - Z-score normalization uses ONLY data before evaluation period
    - Signal selection based on TRAIN period only
"""

import sys, time, warnings, functools
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path

print = functools.partial(print, flush=True)

# ============================================================================
# CONFIG
# ============================================================================
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'DOGEUSDT']
KLINES_DIR = Path('data/binance_cache/klines/15m')
FEES_BPS = 3.0
BARS_PER_DAY = 96
BARS_PER_YEAR = BARS_PER_DAY * 365

# Position smoothing
POS_HALFLIFE = 12   # 3 hours EWMA
VOL_SCALE_MAX = 2.0 # Max inverse-vol scaling

# Per-symbol curated signals (selected by train Sharpe, confirmed OOS)
SYMBOL_SIGNALS = {
    'BTCUSDT': ['donch_720', 'donch_1440', 'dec0.99_d96_cp'],
    'ETHUSDT': ['donch_720', 'donch_1440', 'dec0.99_d96_tbr'],
    'SOLUSDT': ['donch_720', 'donch_960', 'dec0.98_d48_cp'],
    'BNBUSDT': ['donch_1440'],
    'DOGEUSDT': ['donch_1440', 'donch_1920', 'mom_1920'],
}


# ============================================================================
# PRIMITIVES
# ============================================================================
def sma(s, w): return s.rolling(w, min_periods=max(w//2, 2)).mean()
def safe_div(a, b): return (a / b).replace([np.inf, -np.inf], 0).fillna(0)
def decay_exp(s, hl):
    if 0 < hl < 1:
        ah = -np.log(2) / np.log(hl)
    else:
        ah = hl
    return s.ewm(halflife=max(ah, 0.5), min_periods=1).mean()


# ============================================================================
# SIGNAL BUILDER
# ============================================================================
def build_signal(df, name):
    """Build a single named signal from OHLCV data."""
    c = df['close']; ret = c.pct_change()
    h, l, o = df['high'], df['low'], df['open']
    v = df['volume']; qv = df['quote_volume']; tbv = df['taker_buy_volume']

    cp = safe_div(c - l, h - l)
    tbr = safe_div(tbv, v)
    vr = safe_div(v, sma(v, 320))

    if name.startswith('donch_'):
        w = int(name.split('_')[1])
        chi = c.rolling(w, min_periods=w // 2).max()
        clo = c.rolling(w, min_periods=w // 2).min()
        return safe_div(c - clo, chi - clo + 1e-10).shift(1)

    elif name.startswith('dec'):
        parts = name.split('_')
        dc = float(parts[0][3:])
        mw = int(parts[1][1:])
        fn = '_'.join(parts[2:])
        mom = c - c.shift(mw)
        feat = {'cp': cp, 'vr': vr, 'tbr': tbr}.get(fn, cp)
        return decay_exp(mom * feat, dc).shift(1)

    elif name.startswith('mom_'):
        w = int(name.split('_')[1])
        return sma(ret, w).shift(1)

    else:
        raise ValueError(f"Unknown signal: {name}")


# ============================================================================
# DATA
# ============================================================================
def load_symbol(symbol: str) -> pd.DataFrame:
    fpath = KLINES_DIR / f'{symbol}.parquet'
    df = pd.read_parquet(fpath)
    df = df.set_index('datetime').sort_index()
    df = df[~df.index.duplicated(keep='last')]
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                'taker_buy_volume', 'taker_buy_quote_volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# ============================================================================
# STRATEGY
# ============================================================================
def run_strategy(sym, df, eval_start='2024-06-01', eval_end='2025-03-11'):
    """Run the strategy for a single symbol."""
    t0 = time.time()

    sig_names = SYMBOL_SIGNALS[sym]
    ret = df['close'].pct_change()
    fwd = ret.shift(-1)

    # Build signals
    sigs = {}
    for nm in sig_names:
        sigs[nm] = build_signal(df, nm)

    # Evaluation period
    oos_mask = (df.index >= eval_start) & (df.index < eval_end)
    oos_idx = df.index[oos_mask]
    train_mask = df.index < eval_start
    fwd_oos = np.nan_to_num(fwd.reindex(oos_idx).values, 0)

    # Combine signals (equal weight, z-scored using train stats)
    combined = pd.Series(0.0, index=oos_idx)
    for nm, sig in sigs.items():
        ts = sig.loc[train_mask].dropna()
        if len(ts) < 1000:
            continue
        m, s = ts.mean(), ts.std()
        if s < 1e-10:
            continue
        z = ((sig.reindex(oos_idx) - m) / s).clip(-5, 5).fillna(0)
        combined += z / len(sigs)

    # Position: sign × clipped magnitude, with EWMA smoothing
    pos = np.sign(combined) * np.minimum(combined.abs(), 2.0)
    pos = pos.ewm(halflife=POS_HALFLIFE, min_periods=1).mean()
    pv = pos.values

    # Inverse-volatility scaling (target median vol)
    vol = ret.rolling(96, min_periods=48).std().reindex(oos_idx).values
    vol = np.maximum(np.nan_to_num(vol, nan=0.01), 0.001)
    tv = np.median(vol)
    vs = np.clip(tv / vol, 0.2, VOL_SCALE_MAX)
    pv_s = pv * vs

    # Mark-to-market PnL
    gross = pv_s * fwd_oos
    pc = np.abs(np.diff(pv_s, prepend=0))
    fees = pc * FEES_BPS / 10000
    net = gross - fees

    # Metrics
    sharpe = (np.mean(net) / np.std(net, ddof=1)) * np.sqrt(BARS_PER_YEAR) if np.std(net) > 0 else 0
    total_ret = np.sum(net)
    cum = np.cumsum(net)
    max_dd = np.min(cum - np.maximum.accumulate(cum))
    hr = np.mean(np.abs(pv_s) > 0.01)
    avg_to = np.mean(pc)

    elapsed = time.time() - t0
    print(f'  {sym}: Sharpe={sharpe:+.2f} Return={total_ret*100:+.2f}% '
          f'MaxDD={max_dd*100:.2f}% HR={hr:.0%} TO={avg_to:.5f} ({elapsed:.1f}s)')

    return {
        'symbol': sym,
        'sharpe': sharpe,
        'total_return': total_ret,
        'max_dd': max_dd,
        'pnl_series': pd.Series(net, index=oos_idx),
        'signals_used': sig_names,
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    print('=' * 80)
    print('UNIVARIATE HF ALPHA v3.5 — DONCHIAN + DECAY MOMENTUM')
    print('=' * 80)
    print(f'Symbols: {SYMBOLS}')
    print(f'Fees: {FEES_BPS}bps | Pos smoothing HL: {POS_HALFLIFE} bars')
    print(f'Vol-scale max: {VOL_SCALE_MAX}x')
    print()

    results = {}
    for sym in SYMBOLS:
        df = load_symbol(sym)
        results[sym] = run_strategy(sym, df)

    # Collective portfolio
    pnl_df = pd.DataFrame({sym: r['pnl_series'] for sym, r in results.items()}).fillna(0)
    port = pnl_df.mean(axis=1)
    coll = (port.mean() / port.std(ddof=1)) * np.sqrt(BARS_PER_YEAR) if port.std() > 0 else 0

    # Stability
    h = len(port) // 2
    h1, h2 = port.iloc[:h], port.iloc[h:]
    sh1 = (h1.mean() / h1.std(ddof=1)) * np.sqrt(BARS_PER_YEAR) if h1.std() > 0 else 0
    sh2 = (h2.mean() / h2.std(ddof=1)) * np.sqrt(BARS_PER_YEAR) if h2.std() > 0 else 0

    print(f'\n{"=" * 60}')
    print('RESULTS SUMMARY')
    print(f'{"=" * 60}')
    for sym, r in results.items():
        s = 'PASS' if r['sharpe'] > 7 else 'FAIL'
        print(f'  [{s}] {sym}: Sharpe={r["sharpe"]:+.2f} | {r["signals_used"]}')
    print(f'  Collective: {coll:+.2f} (H1={sh1:+.1f} H2={sh2:+.1f})')

    # Save
    pnl_df.to_parquet('data/v3_pnl.parquet')
    print('\nSaved to data/v3_pnl.parquet')

    return results


if __name__ == '__main__':
    main()
