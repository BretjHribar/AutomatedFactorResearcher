"""
V10 ROUND 4: Push to Sharpe > 9

Building on Round 3 (7.43), targeting 9+ via:
  1. Extended smoothing (phl up to 720)
  2. Blended IC (multiple IC lookback periods)  
  3. Inverse vol position sizing
  4. Broader alpha pool (lower screening threshold)
  5. Multi-method ensemble (IC + adaptive combined)
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from UNIFIED_V10.research import (
    load_all_1h, compute_all_alphas, streaming_pnl_single,
    compute_sharpe, streaming_pnl_adaptive,
    TRAIN_END, VAL_START, VAL_END, TEST_START, ALL_SYMBOLS, FEE_FRAC
)
from UNIFIED_V10.research_r3 import (
    net_factor_returns, orthogonal_filter, qp_fixed_weights,
    portfolio_qp_fixed, portfolio_ic_weighted, portfolio_risk_parity
)


def portfolio_blended_ic(alpha_df, ret, selected, ic_lookbacks=[60, 120, 240],
                          phl=168, fee_frac=FEE_FRAC):
    """Blended IC: average IC from multiple lookback periods."""
    X = alpha_df[selected]
    X_shifted = X.shift(1)

    # Compute ICs at multiple lookbacks
    all_ics = []
    for lb in ic_lookbacks:
        ics = pd.DataFrame(index=X.index, columns=selected, dtype=float)
        for col in selected:
            ics[col] = X_shifted[col].rolling(lb, min_periods=max(20, lb//3)).corr(ret)
        all_ics.append(ics)

    # Average ICs across lookbacks
    avg_ic = sum(all_ics) / len(all_ics)

    weights = avg_ic.clip(lower=0).fillna(0)
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


def portfolio_vol_scaled(alpha_df, ret, selected, lookback=120, phl=168,
                          vol_lookback=72, fee_frac=FEE_FRAC):
    """Adaptive net + inverse volatility scaling of position."""
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

    # Inverse volatility scaling: scale down when vol is high
    rolling_vol = ret.rolling(vol_lookback, min_periods=10).std()
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


def portfolio_ensemble(alpha_df, ret, selected, phl=168, fee_frac=FEE_FRAC):
    """Ensemble: average signals from IC-weighted and adaptive net."""
    X = alpha_df[selected]
    X_shifted = X.shift(1)

    # IC-weighted signal
    ics = pd.DataFrame(index=X.index, columns=selected, dtype=float)
    for col in selected:
        ics[col] = X_shifted[col].rolling(120, min_periods=30).corr(ret)
    ic_weights = ics.clip(lower=0).fillna(0)
    ic_wsum = ic_weights.sum(axis=1).replace(0, np.nan)
    ic_wnorm = ic_weights.div(ic_wsum, axis=0).fillna(0)
    ic_signal = (X * ic_wnorm).sum(axis=1)

    # Adaptive net signal
    factor_returns = pd.DataFrame(index=X.index, columns=selected, dtype=float)
    for col in selected:
        factor_returns[col] = np.sign(X_shifted[col].values) * ret.values
    rolling_er = factor_returns.rolling(120, min_periods=60).mean()
    an_weights = rolling_er.clip(lower=0).fillna(0)
    an_wsum = an_weights.sum(axis=1).replace(0, np.nan)
    an_wnorm = an_weights.div(an_wsum, axis=0).fillna(0)
    an_signal = (X * an_wnorm).sum(axis=1)

    # Ensemble: average
    combined = (ic_signal + an_signal) / 2

    if phl > 1:
        combined = combined.ewm(halflife=phl, min_periods=1).mean()

    direction = np.sign(combined.values)
    direction = np.where(np.isnan(direction), 0, direction)
    prev_dir = np.concatenate([[0], direction[:-1]])
    pos_changes = np.abs(direction - prev_dir)
    pnl = prev_dir * ret.values - fee_frac * pos_changes

    return pd.Series(pnl, index=X.index), pd.Series(direction, index=X.index)


def research_symbol(sym, df_1h, all_1h):
    print(f"\n{'='*60}")
    print(f"  {sym}")
    print(f"{'='*60}")
    t0 = time.time()

    all_alphas = compute_all_alphas(df_1h, sym, all_1h)
    ret = df_1h['close'].pct_change()

    # TRAIN screening — lower threshold for broader pool
    train_ret = ret.loc['2023-07-01':TRAIN_END]
    alpha_sharpes = {}
    for name in sorted(all_alphas.keys()):
        a = all_alphas[name].loc['2023-07-01':TRAIN_END]
        if len(a) < 500:
            continue
        pnl = streaming_pnl_single(a, train_ret.reindex(a.index, fill_value=0))
        sr = compute_sharpe(pnl)
        if sr is not None and sr > -0.2:  # Very low threshold
            mid = a.index[len(a)//2]
            s1 = compute_sharpe(pnl.loc[:mid], min_days=5)
            s2 = compute_sharpe(pnl.loc[mid:], min_days=5)
            if s1 and s1 > -1.0 and s2 and s2 > -1.0:
                alpha_sharpes[name] = sr

    print(f"  {len(alpha_sharpes)} alphas passed screening")

    if len(alpha_sharpes) < 2:
        return None

    avail = [n for n in alpha_sharpes.keys() if n in all_alphas]
    alpha_df = pd.DataFrame({n: all_alphas[n] for n in avail}, index=df_1h.index)

    # Filter to positive sharpe for top-ranked pool
    pos_sharpes = pd.Series({k: v for k, v in alpha_sharpes.items() if v > 0})
    all_good = pos_sharpes.sort_values(ascending=False).index.tolist()

    best = None
    best_cfg = None
    tested = 0

    # Strategy 1: IC-weighted with extended smoothing
    for n_top in [3, 5, 8, 10, 15, 20, 30, 50]:
        subset = all_good[:min(n_top, len(all_good))]
        if len(subset) < 2:
            continue
        for phl in [72, 120, 168, 240, 336, 480, 720]:
            for ic_lbs in [[60, 120, 240], [30, 60, 120], [120, 240, 480],
                           [60], [120], [240], [480]]:
                pnl_s, dirs = portfolio_blended_ic(
                    alpha_df, ret, subset, ic_lookbacks=ic_lbs, phl=phl
                )
                val_pnl = pnl_s.loc[VAL_START:VAL_END]
                sr = compute_sharpe(val_pnl)
                tested += 1
                if sr is not None and (best is None or sr > best['sharpe']):
                    vd = dirs.loc[VAL_START:VAL_END].values
                    vd = np.where(np.isnan(vd), 0, vd)
                    best = {
                        'sharpe': sr,
                        'cum_pnl_bps': val_pnl.sum() * 10000,
                        'n_trades': int(np.abs(np.diff(np.concatenate([[0], vd]))).sum()),
                        'daily_pnl': val_pnl.resample('1D').sum(),
                    }
                    best_cfg = {
                        'alphas': subset, 'n_alphas': len(subset),
                        'phl': phl, 'method': 'blended_ic',
                        'ic_lookbacks': ic_lbs,
                    }

    # Strategy 2: Adaptive net with extended smoothing
    for n_top in [3, 5, 8, 10, 15, 20, 30, 50]:
        subset = all_good[:min(n_top, len(all_good))]
        if len(subset) < 2:
            continue
        for lb in [60, 120, 240, 480]:
            for phl in [72, 120, 168, 240, 336, 480, 720]:
                pnl_s, dirs = streaming_pnl_adaptive(
                    alpha_df, ret, subset, lb, phl
                )
                val_pnl = pnl_s.loc[VAL_START:VAL_END]
                sr = compute_sharpe(val_pnl)
                tested += 1
                if sr is not None and (best is None or sr > best['sharpe']):
                    vd = dirs.loc[VAL_START:VAL_END].values
                    vd = np.where(np.isnan(vd), 0, vd)
                    best = {
                        'sharpe': sr,
                        'cum_pnl_bps': val_pnl.sum() * 10000,
                        'n_trades': int(np.abs(np.diff(np.concatenate([[0], vd]))).sum()),
                        'daily_pnl': val_pnl.resample('1D').sum(),
                    }
                    best_cfg = {
                        'alphas': subset, 'n_alphas': len(subset),
                        'lookback': lb, 'phl': phl,
                        'method': 'adaptive_net',
                    }

    # Strategy 3: Vol-scaled adaptive
    for n_top in [5, 10, 15, 20]:
        subset = all_good[:min(n_top, len(all_good))]
        if len(subset) < 2:
            continue
        for lb in [120, 240]:
            for phl in [120, 168, 240, 336, 480]:
                pnl_s, dirs = portfolio_vol_scaled(
                    alpha_df, ret, subset, lb, phl
                )
                val_pnl = pnl_s.loc[VAL_START:VAL_END]
                sr = compute_sharpe(val_pnl)
                tested += 1
                if sr is not None and (best is None or sr > best['sharpe']):
                    vd = dirs.loc[VAL_START:VAL_END].values
                    vd = np.where(np.isnan(vd), 0, vd)
                    best = {
                        'sharpe': sr,
                        'cum_pnl_bps': val_pnl.sum() * 10000,
                        'n_trades': int(np.abs(np.diff(np.concatenate([[0], vd]))).sum()),
                        'daily_pnl': val_pnl.resample('1D').sum(),
                    }
                    best_cfg = {
                        'alphas': subset, 'n_alphas': len(subset),
                        'lookback': lb, 'phl': phl,
                        'method': 'vol_scaled',
                    }

    # Strategy 4: Ensemble
    for n_top in [5, 10, 15, 20]:
        subset = all_good[:min(n_top, len(all_good))]
        if len(subset) < 2:
            continue
        for phl in [120, 168, 240, 336, 480]:
            pnl_s, dirs = portfolio_ensemble(
                alpha_df, ret, subset, phl=phl
            )
            val_pnl = pnl_s.loc[VAL_START:VAL_END]
            sr = compute_sharpe(val_pnl)
            tested += 1
            if sr is not None and (best is None or sr > best['sharpe']):
                vd = dirs.loc[VAL_START:VAL_END].values
                vd = np.where(np.isnan(vd), 0, vd)
                best = {
                    'sharpe': sr,
                    'cum_pnl_bps': val_pnl.sum() * 10000,
                    'n_trades': int(np.abs(np.diff(np.concatenate([[0], vd]))).sum()),
                    'daily_pnl': val_pnl.resample('1D').sum(),
                }
                best_cfg = {
                    'alphas': subset, 'n_alphas': len(subset),
                    'phl': phl, 'method': 'ensemble',
                }

    # Strategy 5: Orthogonal + IC/risk_parity/qp
    for max_n in [5, 8, 12, 16]:
        for corr_thresh in [0.40, 0.50, 0.60, 0.70]:
            selected = orthogonal_filter(
                alpha_df.loc['2023-07-01':TRAIN_END],
                pos_sharpes, corr_threshold=corr_thresh, max_n=max_n
            )
            if len(selected) < 2:
                continue
            for phl in [120, 168, 240, 336, 480, 720]:
                # IC-weighted
                pnl_s, dirs = portfolio_ic_weighted(
                    alpha_df, ret, selected, lookback=120, phl=phl
                )
                val_pnl = pnl_s.loc[VAL_START:VAL_END]
                sr = compute_sharpe(val_pnl)
                tested += 1
                if sr is not None and (best is None or sr > best['sharpe']):
                    vd = dirs.loc[VAL_START:VAL_END].values
                    vd = np.where(np.isnan(vd), 0, vd)
                    best = {
                        'sharpe': sr,
                        'cum_pnl_bps': val_pnl.sum() * 10000,
                        'n_trades': int(np.abs(np.diff(np.concatenate([[0], vd]))).sum()),
                        'daily_pnl': val_pnl.resample('1D').sum(),
                    }
                    best_cfg = {
                        'alphas': selected, 'n_alphas': len(selected),
                        'lookback': 120, 'phl': phl,
                        'method': f'ortho_ic_c{corr_thresh}',
                    }

    print(f"  Tested {tested} in {time.time()-t0:.0f}s")

    if best and best['sharpe'] > 0:
        best['config'] = best_cfg
        print(f"  ★ SR={best['sharpe']:+.2f} PnL={best['cum_pnl_bps']:+.0f}bps "
              f"trades={best['n_trades']} {best_cfg.get('method','')} "
              f"phl={best_cfg.get('phl',0)} n={best_cfg['n_alphas']}")
    else:
        print(f"  No profitable portfolio")

    return best


def main():
    print("=" * 60)
    print("V10 ROUND 4: PUSHING TO SHARPE > 9")
    print("=" * 60)

    all_1h = load_all_1h()
    print(f"Loaded {len(all_1h)} symbols\n")

    all_results = {}
    all_val_daily = {}

    for sym in ALL_SYMBOLS:
        if sym not in all_1h:
            continue
        result = research_symbol(sym, all_1h[sym], all_1h)
        if result and result['sharpe'] > 0:
            all_results[sym] = result
            all_val_daily[sym] = result['daily_pnl']

        if len(all_val_daily) >= 2:
            port_df = pd.DataFrame(all_val_daily).fillna(0)
            pd_daily = port_df.mean(axis=1)
            pd_daily = pd_daily[pd_daily != 0]
            if len(pd_daily) >= 10 and pd_daily.std() > 0:
                agg = pd_daily.mean() / pd_daily.std() * np.sqrt(365)
                print(f"\n  >>> AGGREGATE: {agg:+.2f} ({len(all_val_daily)} syms)")

    print(f"\n{'='*60}")
    print("ROUND 4 FINAL")
    print(f"{'='*60}")

    if len(all_val_daily) >= 2:
        port_df = pd.DataFrame(all_val_daily).fillna(0)
        pd_daily = port_df.mean(axis=1)
        pd_daily = pd_daily[pd_daily != 0]
        if len(pd_daily) >= 10 and pd_daily.std() > 0:
            port_sharpe = pd_daily.mean() / pd_daily.std() * np.sqrt(365)
            print(f"\n  ★★★ COMBINED SHARPE: {port_sharpe:+.2f} ★★★")
            print(f"  PnL: {pd_daily.sum()*10000:+.0f} bps  Win: {(pd_daily>0).mean():.0%}")
            print(f"  Max DD: {(pd_daily.cumsum() - pd_daily.cumsum().cummax()).min()*10000:.0f} bps")

    for sym in sorted(all_results.keys(), key=lambda s: all_results[s]['sharpe'], reverse=True):
        r = all_results[sym]
        cfg = r.get('config', {})
        print(f"  {sym:>10s}: SR={r['sharpe']:+.2f} PnL={r['cum_pnl_bps']:+.0f}bps "
              f"{cfg.get('method',''):>18s} phl={cfg.get('phl',0)} n={cfg.get('n_alphas',0)}")

    # Save
    frozen = {
        'version': 'V10_research_r4',
        'frozen_at': pd.Timestamp.now().isoformat(),
        'train_end': TRAIN_END, 'val_start': VAL_START, 'val_end': VAL_END,
        'symbols': {},
    }
    for sym, r in all_results.items():
        cfg = r.get('config', {})
        frozen['symbols'][sym] = {
            'selected_alphas': cfg.get('alphas', []),
            'lookback': cfg.get('lookback', 120),
            'phl': cfg.get('phl', 1),
            'method': cfg.get('method', ''),
            'ic_lookbacks': cfg.get('ic_lookbacks', [120]),
            'val_sharpe': r['sharpe'],
        }
    pf = Path('UNIFIED_V10/frozen_params_v10.json')
    with open(pf, 'w') as f:
        json.dump(frozen, f, indent=2, default=str)
    print(f"\n  Saved to {pf}")


if __name__ == '__main__':
    main()
