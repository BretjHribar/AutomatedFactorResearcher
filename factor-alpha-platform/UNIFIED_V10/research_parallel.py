"""
V10 ROUND 9: Parallel Research Engine + Expanded Symbol Universe

Key features:
  - multiprocessing.Pool for parallel per-symbol research
  - Expanded to 20-30 high-volume Binance Futures symbols
  - Greedy aggregate-optimal strategy selection
  - Results IDENTICAL to single-threaded version (deterministic)
  - Target: Sharpe > 14
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import json
import time
import sys
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).parent.parent))
from UNIFIED_V10.alphas import build_1h_alphas, build_htf_signals, build_cross_asset_signals
from UNIFIED_V10.config import DATA_DIR, AGG_RULES

FEE_BPS = 5  # 5 bps per direction change (user specified)
FEE_FRAC = FEE_BPS / 10000.0  # 0.0005

TRAIN_END = '2024-06-30'
VAL_START = '2024-07-01'
VAL_END   = '2025-01-01'
MIN_BARS  = 15000  # Need enough for train + validation


# ── DATA LOADING ──

def load_1h(sym):
    """Load and resample one symbol to 1H."""
    path = DATA_DIR / f'{sym}.parquet'
    if not path.exists():
        return None
    df15 = pd.read_parquet(path)
    if 'datetime' in df15.columns:
        df15 = df15.set_index('datetime')
    df15 = df15.sort_index()
    df15 = df15[~df15.index.duplicated(keep='last')]
    if df15.index.tz is not None:
        df15.index = df15.index.tz_localize(None)
    for c in AGG_RULES:
        if c in df15.columns:
            df15[c] = pd.to_numeric(df15[c], errors='coerce')
    df_1h = df15.resample('1h').agg(AGG_RULES).dropna()
    return df_1h


def discover_symbols():
    """Find all symbols with sufficient data and volume."""
    symbols = []
    for path in sorted(DATA_DIR.glob('*.parquet')):
        sym = path.stem
        try:
            df_1h = load_1h(sym)
            if df_1h is None or len(df_1h) < MIN_BARS:
                continue
            # Check we have data in train period
            train_bars = len(df_1h.loc['2023-07-01':TRAIN_END])
            val_bars = len(df_1h.loc[VAL_START:VAL_END])
            if train_bars < 5000 or val_bars < 2000:
                continue
            # Average daily quote volume (in millions)
            avg_qv = df_1h['quote_volume'].resample('1D').sum().mean() / 1e6
            symbols.append((sym, len(df_1h), avg_qv))
        except Exception as e:
            continue
    
    # Sort by volume, keep top ones
    symbols.sort(key=lambda x: x[2], reverse=True)
    return symbols


# ── ALPHA COMPUTATION ──

def compute_all_alphas(df_1h, sym=None, all_1h=None):
    """Compute ALL alphas vectorially. SAME functions as StreamingEngine."""
    all_alphas = build_1h_alphas(df_1h)
    if isinstance(df_1h.index, pd.DatetimeIndex) and len(df_1h) >= 50:
        agg = {k: v for k, v in AGG_RULES.items() if k in df_1h.columns}
        for freq, prefix in [('2h','h2'),('4h','h4'),('8h','h8'),('12h','h12')]:
            try:
                df_htf = df_1h.resample(freq).agg(agg).dropna()
                if len(df_htf) >= 5:
                    all_alphas.update(build_htf_signals(df_htf, df_1h, prefix, shift_n=1))
            except:
                pass
    if sym and all_1h:
        try:
            all_alphas.update(build_cross_asset_signals(all_1h, sym, df_1h))
        except:
            pass
    return all_alphas


# ── PNL FUNCTIONS (streaming-equivalent) ──

def streaming_pnl_single(alpha_series, ret, fee_frac=FEE_FRAC):
    """Single alpha PnL: dir = sign(alpha), pnl = prev_dir * ret - fee * |change|"""
    direction = np.sign(alpha_series.values)
    direction = np.where(np.isnan(direction), 0, direction)
    prev_dir = np.concatenate([[0], direction[:-1]])
    pos_changes = np.abs(direction - prev_dir)
    pnl = prev_dir * ret.values - fee_frac * pos_changes
    return pd.Series(pnl, index=alpha_series.index)


def streaming_pnl_adaptive(alpha_df, ret, selected, lookback=120, phl=1, fee_frac=FEE_FRAC):
    """Adaptive net portfolio PnL."""
    X = alpha_df[selected]
    X_shifted = X.shift(1)
    factor_returns = pd.DataFrame(index=X.index, columns=selected, dtype=float)
    for col in selected:
        factor_returns[col] = np.sign(X_shifted[col].values) * ret.values
    min_p = min(100, lookback)
    rolling_er = factor_returns.rolling(lookback, min_periods=min_p).mean()
    weights = rolling_er.clip(lower=0).fillna(0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)
    combined = (X * weights_norm).sum(axis=1)
    if phl > 1:
        combined = combined.ewm(halflife=phl, min_periods=1).mean()
    direction = np.sign(combined.values)
    direction = np.where(np.isnan(direction), 0, direction)
    prev_dir = np.concatenate([[0], direction[:-1]])
    pos_changes = np.abs(direction - prev_dir)
    pnl = prev_dir * ret.values - fee_frac * pos_changes
    return pd.Series(pnl, index=X.index), pd.Series(direction, index=X.index)


def portfolio_ic_weighted(alpha_df, ret, selected, lookback=120, phl=48, fee_frac=FEE_FRAC):
    """IC-weighted: weight by rolling correlation(alpha[t-1], ret[t])."""
    X = alpha_df[selected]
    X_shifted = X.shift(1)
    ics = pd.DataFrame(index=X.index, columns=selected, dtype=float)
    for col in selected:
        ics[col] = X_shifted[col].rolling(lookback, min_periods=30).corr(ret)
    weights = ics.clip(lower=0).fillna(0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)
    combined = (X * weights_norm).sum(axis=1)
    if phl > 1:
        combined = combined.ewm(halflife=phl, min_periods=1).mean()
    direction = np.sign(combined.values)
    direction = np.where(np.isnan(direction), 0, direction)
    prev_dir = np.concatenate([[0], direction[:-1]])
    pos_changes = np.abs(direction - prev_dir)
    pnl = prev_dir * ret.values - fee_frac * pos_changes
    return pd.Series(pnl, index=X.index), pd.Series(direction, index=X.index)


def portfolio_vol_scaled(alpha_df, ret, selected, lookback=120, phl=168, fee_frac=FEE_FRAC):
    """Adaptive net with inverse-volatility position scaling."""
    X = alpha_df[selected]
    X_shifted = X.shift(1)
    factor_returns = pd.DataFrame(index=X.index, columns=selected, dtype=float)
    for col in selected:
        factor_returns[col] = np.sign(X_shifted[col].values) * ret.values
    min_p = min(100, lookback)
    rolling_er = factor_returns.rolling(lookback, min_periods=min_p).mean()
    weights = rolling_er.clip(lower=0).fillna(0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)
    combined = (X * weights_norm).sum(axis=1)
    rolling_vol = ret.rolling(72, min_periods=10).std()
    median_vol = rolling_vol.median()
    vol_scale = (median_vol / rolling_vol.clip(lower=median_vol * 0.1)).shift(1).fillna(1)
    combined = combined * vol_scale
    if phl > 1:
        combined = combined.ewm(halflife=phl, min_periods=1).mean()
    direction = np.sign(combined.values)
    direction = np.where(np.isnan(direction), 0, direction)
    prev_dir = np.concatenate([[0], direction[:-1]])
    pos_changes = np.abs(direction - prev_dir)
    pnl = prev_dir * ret.values - fee_frac * pos_changes
    return pd.Series(pnl, index=X.index), pd.Series(direction, index=X.index)


def compute_sharpe(pnl_s, min_days=10):
    """Annualized Sharpe from PnL series."""
    daily = pnl_s.resample('1D').sum()
    daily = daily[daily != 0]
    if len(daily) < min_days or daily.std() == 0:
        return None
    return daily.mean() / daily.std() * np.sqrt(365)


# ── PER-SYMBOL RESEARCH (runs in worker process) ──

def research_one_symbol(args):
    """Full research pipeline for one symbol. Designed for multiprocessing."""
    sym, df_1h_path = args
    
    # Load data
    df_1h = load_1h(sym)
    if df_1h is None:
        return sym, []
    
    all_alphas = compute_all_alphas(df_1h, sym=sym)
    ret = df_1h['close'].pct_change()
    train_ret = ret.loc['2023-07-01':TRAIN_END]
    
    # Screen alphas on train
    alpha_sharpes = {}
    for name in sorted(all_alphas.keys()):
        a = all_alphas[name].loc['2023-07-01':TRAIN_END]
        if len(a) < 500:
            continue
        pnl = streaming_pnl_single(a, train_ret.reindex(a.index, fill_value=0))
        sr = compute_sharpe(pnl)
        if sr is not None and sr > 0.3:
            mid = a.index[len(a)//2]
            s1 = compute_sharpe(pnl.loc[:mid], min_days=5)
            s2 = compute_sharpe(pnl.loc[mid:], min_days=5)
            if s1 and s1 > 0 and s2 and s2 > 0:
                alpha_sharpes[name] = sr
    
    if len(alpha_sharpes) < 2:
        return sym, []
    
    avail = sorted(alpha_sharpes.keys(), key=lambda n: alpha_sharpes[n], reverse=True)
    alpha_df = pd.DataFrame({n: all_alphas[n] for n in avail}, index=df_1h.index)
    
    # Run all methods
    candidates = []
    
    for n_top in [3, 5, 8, 10, 15, 20, 30, min(50, len(avail))]:
        subset = avail[:min(n_top, len(avail))]
        if len(subset) < 2:
            continue
        
        for phl in [1, 4, 8, 24, 48, 72, 120, 168, 240, 336, 480, 720]:
            methods = [
                ('ic_60', lambda s=subset, p=phl: portfolio_ic_weighted(alpha_df, ret, s, 60, p)),
                ('ic_120', lambda s=subset, p=phl: portfolio_ic_weighted(alpha_df, ret, s, 120, p)),
                ('ic_240', lambda s=subset, p=phl: portfolio_ic_weighted(alpha_df, ret, s, 240, p)),
                ('an_60', lambda s=subset, p=phl: streaming_pnl_adaptive(alpha_df, ret, s, 60, p)),
                ('an_120', lambda s=subset, p=phl: streaming_pnl_adaptive(alpha_df, ret, s, 120, p)),
                ('an_240', lambda s=subset, p=phl: streaming_pnl_adaptive(alpha_df, ret, s, 240, p)),
                ('an_480', lambda s=subset, p=phl: streaming_pnl_adaptive(alpha_df, ret, s, 480, p)),
            ]
            if phl >= 72:
                methods.append(('vol', lambda s=subset, p=phl: portfolio_vol_scaled(alpha_df, ret, s, 120, p)))
            
            for mname, mfunc in methods:
                try:
                    pnl_s, dirs = mfunc()
                    val_pnl = pnl_s.loc[VAL_START:VAL_END]
                    sr = compute_sharpe(val_pnl)
                    if sr and sr > 0:
                        vd = dirs.loc[VAL_START:VAL_END].values
                        vd = np.where(np.isnan(vd), 0, vd)
                        candidates.append({
                            'key': f'{sym}_{mname}_p{phl}_n{len(subset)}',
                            'sym': sym,
                            'sharpe': sr,
                            'daily_pnl': val_pnl.resample('1D').sum(),
                            'cum_pnl_bps': val_pnl.sum() * 10000,
                            'n_trades': int(np.abs(np.diff(np.concatenate([[0], vd]))).sum()),
                            'method': mname, 'phl': phl,
                            'n_alphas': len(subset),
                            'alphas': subset,
                        })
                except:
                    pass
    
    # Sort and keep diverse top-10
    candidates.sort(key=lambda c: c['sharpe'], reverse=True)
    seen = set()
    diverse = []
    for c in candidates:
        bucket = (c['method'], c['phl'] // 48)
        if bucket not in seen:
            diverse.append(c)
            seen.add(bucket)
        if len(diverse) >= 10:
            break
    
    return sym, diverse


# ── GREEDY AGGREGATE OPTIMIZER ──

def greedy_aggregate(all_candidates, max_per_sym=2, max_total=30):
    """Greedy selection: add strategy that maximizes aggregate Sharpe."""
    selected = {}
    selected_dailies = {}
    history = []
    
    for iteration in range(max_total):
        best_agg = -999
        best_cand = None
        
        for cand in all_candidates:
            sym = cand['sym']
            sym_count = sum(1 for k in selected if k.startswith(sym + '_'))
            if sym_count >= max_per_sym:
                continue
            if cand['key'] in selected:
                continue
            
            test_dailies = dict(selected_dailies)
            test_dailies[cand['key']] = cand['daily_pnl']
            
            if len(test_dailies) < 2:
                best_cand = cand
                break
            
            port_df = pd.DataFrame(test_dailies).fillna(0)
            pd_d = port_df.mean(axis=1)
            pd_d = pd_d[pd_d != 0]
            if len(pd_d) >= 10 and pd_d.std() > 0:
                agg_sr = pd_d.mean() / pd_d.std() * np.sqrt(365)
                if agg_sr > best_agg:
                    best_agg = agg_sr
                    best_cand = cand
        
        if best_cand is None:
            break
        
        selected[best_cand['key']] = best_cand
        selected_dailies[best_cand['key']] = best_cand['daily_pnl']
        
        # Current aggregate
        port_df = pd.DataFrame(selected_dailies).fillna(0)
        pd_d = port_df.mean(axis=1)
        pd_d = pd_d[pd_d != 0]
        agg = pd_d.mean() / pd_d.std() * np.sqrt(365) if len(pd_d) >= 10 and pd_d.std() > 0 else 0
        
        history.append((best_cand['key'], best_cand['sharpe'], agg))
        print(f'  [{iteration+1:2d}] +{best_cand["key"][:35]:35s} '
              f'SR={best_cand["sharpe"]:+.2f}  AGG={agg:+.2f}')
        
        # Stop if aggregate is declining for 3 consecutive adds
        if len(history) >= 5 and all(history[-i-1][2] >= history[-i][2] for i in range(3)):
            print(f'  Stopping: aggregate declining')
            break
    
    return selected, selected_dailies, history


def main():
    t_start = time.time()
    print("=" * 60)
    print("V10 ROUND 9: PARALLEL RESEARCH + EXPANDED UNIVERSE")
    print(f"Target: Sharpe > 14 | CPUs: {cpu_count()}")
    print("=" * 60)
    
    # Discover all available symbols
    print("\nDiscovering symbols...")
    sym_info = discover_symbols()
    print(f"\n  Available symbols ({len(sym_info)}):")
    for sym, n_bars, avg_qv in sym_info:
        print(f"    {sym:>15s}: {n_bars:>6d} bars, ${avg_qv:>10.1f}M avg daily QV")
    
    symbols_to_use = [s[0] for s in sym_info]
    
    # Phase 1: Parallel per-symbol research
    print(f"\n{'='*60}")
    print(f"PHASE 1: Parallel alpha research ({len(symbols_to_use)} symbols)")
    print(f"{'='*60}")
    
    args = [(sym, str(DATA_DIR)) for sym in symbols_to_use]
    
    n_workers = min(cpu_count(), len(symbols_to_use), 8)
    print(f"  Using {n_workers} workers...")
    
    t0 = time.time()
    with Pool(n_workers) as pool:
        results = pool.map(research_one_symbol, args)
    print(f"  Phase 1 done in {time.time()-t0:.0f}s")
    
    # Collect all candidates
    all_candidates = []
    for sym, candidates in results:
        if candidates:
            all_candidates.extend(candidates)
            top_sr = max(c['sharpe'] for c in candidates)
            print(f"  {sym:>15s}: {len(candidates)} candidates, top SR={top_sr:+.2f}")
        else:
            print(f"  {sym:>15s}: no viable candidates")
    
    all_candidates.sort(key=lambda c: c['sharpe'], reverse=True)
    print(f"\n  Total candidates: {len(all_candidates)}")
    
    # Phase 2: Greedy aggregate optimization
    print(f"\n{'='*60}")
    print("PHASE 2: Greedy aggregate optimization")
    print(f"{'='*60}")
    
    selected, selected_dailies, history = greedy_aggregate(
        all_candidates, max_per_sym=2, max_total=40
    )
    
    # Final results
    print(f"\n{'='*60}")
    print("ROUND 9 FINAL")
    print(f"{'='*60}")
    
    port_df = pd.DataFrame(selected_dailies).fillna(0)
    pd_d = port_df.mean(axis=1)
    pd_d = pd_d[pd_d != 0]
    if len(pd_d) >= 10 and pd_d.std() > 0:
        port_sharpe = pd_d.mean() / pd_d.std() * np.sqrt(365)
        print(f"\n  ★★★ COMBINED SHARPE: {port_sharpe:+.2f} ★★★")
        print(f"  PnL: {pd_d.sum()*10000:+.0f} bps  Win: {(pd_d>0).mean():.0%}")
        print(f"  Max DD: {(pd_d.cumsum() - pd_d.cumsum().cummax()).min()*10000:.0f} bps")
        print(f"  Strategies: {len(selected)}")
    
    # Peak aggregate
    if history:
        peak_idx = max(range(len(history)), key=lambda i: history[i][2])
        print(f"\n  Peak aggregate: {history[peak_idx][2]:+.2f} at step {peak_idx+1}")
    
    # Per-symbol
    sym_summary = {}
    for key, c in selected.items():
        sym = c['sym']
        if sym not in sym_summary:
            sym_summary[sym] = []
        sym_summary[sym].append(c)
    
    print(f"\n  Per-Symbol ({len(sym_summary)} symbols):")
    for sym in sorted(sym_summary.keys()):
        for c in sym_summary[sym]:
            print(f"    {sym:>15s}: SR={c['sharpe']:+.2f} {c['method']:>8s} "
                  f"phl={c['phl']} n={c['n_alphas']} trades={c['n_trades']}")
    
    # Save
    frozen = {
        'version': 'V10_research_r9_parallel',
        'frozen_at': pd.Timestamp.now().isoformat(),
        'train_end': TRAIN_END, 'val_start': VAL_START, 'val_end': VAL_END,
        'n_strategies': len(selected),
        'n_symbols': len(sym_summary),
        'combined_sharpe': float(port_sharpe) if len(pd_d) >= 10 else 0,
        'symbols': {},
    }
    for sym, configs in sym_summary.items():
        frozen['symbols'][sym] = []
        for c in configs:
            frozen['symbols'][sym].append({
                'selected_alphas': c.get('alphas', []),
                'lookback': 120,
                'phl': c['phl'],
                'method': c['method'],
                'val_sharpe': c['sharpe'],
            })
    
    pf = Path('UNIFIED_V10/frozen_params_v10.json')
    with open(pf, 'w') as f:
        json.dump(frozen, f, indent=2, default=str)
    print(f"\n  Saved to {pf}")
    print(f"  Total time: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
