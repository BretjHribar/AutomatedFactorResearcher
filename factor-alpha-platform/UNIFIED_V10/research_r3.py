"""
V10 ROUND 3: Advanced Portfolio Construction (FAST)

Isichenko techniques, vectorized for speed:
  1. Net Factor Returns (fee-aware)
  2. Orthogonal Filter (decorrelation)
  3. QP weights computed ONCE on train FR (not rolling)
  4. Very heavy position smoothing
  5. Extended phl grid (up to 336)

Target: Sharpe > 9 aggregate on validation
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


def net_factor_returns(alpha_df, ret, fee_frac=FEE_FRAC):
    """Net factor returns = gross FR - turnover * fee."""
    dirs = np.sign(alpha_df).fillna(0)
    dirs_shifted = dirs.shift(1).fillna(0)
    gross_fr = dirs_shifted.multiply(ret, axis=0)
    turnover = dirs.diff().abs()
    net_fr = gross_fr - turnover * fee_frac
    return net_fr


def orthogonal_filter(alpha_df, sharpes, corr_threshold=0.60, max_n=20):
    """Greedy: add alpha only if corr < threshold with pool."""
    sorted_names = sharpes.sort_values(ascending=False).index.tolist()
    selected = []
    for name in sorted_names:
        if len(selected) >= max_n:
            break
        if name not in alpha_df.columns:
            continue
        if len(selected) == 0:
            selected.append(name)
            continue
        max_corr = max(abs(alpha_df[name].corr(alpha_df[e])) for e in selected)
        if max_corr < corr_threshold:
            selected.append(name)
    return selected


def qp_fixed_weights(train_fr_df):
    """Compute QP weights from full train factor returns.

    w = Sigma^-1 * mu, clipped to non-negative.
    """
    mu = train_fr_df.mean().values
    cov = train_fr_df.cov().values + 0.02 * np.eye(len(mu))
    try:
        raw_w = np.linalg.solve(cov, mu)
    except:
        raw_w = mu
    raw_w = np.clip(raw_w, 0, None)
    wsum = raw_w.sum()
    if wsum <= 0:
        return pd.Series(1.0 / len(mu), index=train_fr_df.columns)
    return pd.Series(raw_w / wsum, index=train_fr_df.columns)


def portfolio_qp_fixed(alpha_df, ret, selected, train_end, phl=48, fee_frac=FEE_FRAC):
    """Portfolio using fixed QP weights from train + heavy smoothing."""
    X = alpha_df[selected]

    # Compute net factor returns on entire series
    net_fr = net_factor_returns(X, ret, fee_frac)

    # QP weights from TRAIN period only
    train_fr = net_fr.loc[:train_end]
    # Need enough data
    if len(train_fr) < 100:
        return None, None

    w = qp_fixed_weights(train_fr)

    # Combined signal using fixed QP weights
    combined = (X * w).sum(axis=1)

    if phl > 1:
        combined = combined.ewm(halflife=phl, min_periods=1).mean()

    direction = np.sign(combined.values)
    direction = np.where(np.isnan(direction), 0, direction)

    prev_dir = np.concatenate([[0], direction[:-1]])
    pos_changes = np.abs(direction - prev_dir)
    pnl = prev_dir * ret.values - fee_frac * pos_changes

    return pd.Series(pnl, index=X.index), pd.Series(direction, index=X.index)


def portfolio_ic_weighted(alpha_df, ret, selected, lookback=120, phl=48, fee_frac=FEE_FRAC):
    """IC-weighted portfolio: weight by rolling IC (correlation with returns)."""
    X = alpha_df[selected]
    X_shifted = X.shift(1)

    # Rolling IC: correlation between alpha[t-1] and return[t]
    ics = pd.DataFrame(index=X.index, columns=selected, dtype=float)
    for col in selected:
        ics[col] = X_shifted[col].rolling(lookback, min_periods=30).corr(ret)

    # Weights: positive IC only, normalized
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


def portfolio_risk_parity(alpha_df, ret, selected, lookback=120, phl=48, fee_frac=FEE_FRAC):
    """Risk parity: weight inversely proportional to factor volatility."""
    X = alpha_df[selected]
    X_shifted = X.shift(1)

    # Factor returns
    fr = pd.DataFrame(index=X.index, columns=selected, dtype=float)
    for col in selected:
        fr[col] = np.sign(X_shifted[col].values) * ret.values

    # Inverse volatility weights
    vol = fr.rolling(lookback, min_periods=30).std()
    inv_vol = (1.0 / vol.clip(lower=1e-8)).fillna(0)
    wsum = inv_vol.sum(axis=1).replace(0, np.nan)
    weights_norm = inv_vol.div(wsum, axis=0).fillna(0)

    combined = (X * weights_norm).sum(axis=1)
    if phl > 1:
        combined = combined.ewm(halflife=phl, min_periods=1).mean()

    direction = np.sign(combined.values)
    direction = np.where(np.isnan(direction), 0, direction)
    prev_dir = np.concatenate([[0], direction[:-1]])
    pos_changes = np.abs(direction - prev_dir)
    pnl = prev_dir * ret.values - fee_frac * pos_changes

    return pd.Series(pnl, index=X.index), pd.Series(direction, index=X.index)


def research_symbol(sym, df_1h, all_1h):
    """Full research for one symbol."""
    print(f"\n{'='*60}")
    print(f"  {sym}")
    print(f"{'='*60}")
    t0 = time.time()

    all_alphas = compute_all_alphas(df_1h, sym, all_1h)
    ret = df_1h['close'].pct_change()

    # ── TRAIN: Screen alphas ──
    train_ret = ret.loc['2023-07-01':TRAIN_END]
    alpha_sharpes = {}
    for name in sorted(all_alphas.keys()):
        a = all_alphas[name].loc['2023-07-01':TRAIN_END]
        if len(a) < 500:
            continue
        pnl = streaming_pnl_single(a, train_ret.reindex(a.index, fill_value=0))
        sr = compute_sharpe(pnl)
        if sr is not None and sr > 0.0:
            mid = a.index[len(a)//2]
            s1 = compute_sharpe(pnl.loc[:mid], min_days=5)
            s2 = compute_sharpe(pnl.loc[mid:], min_days=5)
            if s1 and s1 > -0.5 and s2 and s2 > -0.5:
                alpha_sharpes[name] = sr

    print(f"  {len(alpha_sharpes)} alphas with positive train Sharpe")

    if len(alpha_sharpes) < 2:
        return None

    avail = [n for n in alpha_sharpes.keys() if n in all_alphas]
    alpha_df = pd.DataFrame({n: all_alphas[n] for n in avail}, index=df_1h.index)

    # Net sharpes for orthogonal filter
    net_fr = net_factor_returns(alpha_df.loc['2023-07-01':TRAIN_END],
                                 train_ret.reindex(alpha_df.loc['2023-07-01':TRAIN_END].index, fill_value=0))
    net_sharpes = {}
    for name in avail:
        daily = net_fr[name].resample('1D').sum()
        daily = daily[daily != 0]
        if len(daily) >= 10 and daily.std() > 0:
            net_sharpes[name] = daily.mean() / daily.std() * np.sqrt(365)
    net_sharpes_s = pd.Series(net_sharpes)

    # ── VAlIDATION: Test ALL portfolio methods ──
    best = None
    best_cfg = None
    tested = 0

    # For each orthogonal pool size
    pos_sharpes = net_sharpes_s[net_sharpes_s > 0]
    if len(pos_sharpes) < 2:
        # Fall back to regular sharpes
        pos_sharpes = pd.Series(alpha_sharpes)

    for max_n in [5, 8, 12, 16, 20, 30]:
        for corr_thresh in [0.40, 0.50, 0.60, 0.70, 0.80]:
            selected = orthogonal_filter(
                alpha_df.loc['2023-07-01':TRAIN_END],
                pos_sharpes,
                corr_threshold=corr_thresh, max_n=max_n
            )
            if len(selected) < 2:
                continue

            for phl in [24, 48, 72, 120, 168, 240, 336]:
                # Method 1: Adaptive Net (baseline)
                pnl_s, dirs = streaming_pnl_adaptive(
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
                        'method': 'adaptive_net', 'corr_thresh': corr_thresh,
                    }

                # Method 2: QP Fixed Weights
                pnl_s, dirs = portfolio_qp_fixed(
                    alpha_df, ret, selected, TRAIN_END, phl=phl
                )
                if pnl_s is not None:
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
                            'method': 'qp_fixed', 'corr_thresh': corr_thresh,
                        }

                # Method 3: IC-Weighted
                for ic_lb in [60, 120, 240]:
                    pnl_s, dirs = portfolio_ic_weighted(
                        alpha_df, ret, selected, lookback=ic_lb, phl=phl
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
                            'lookback': ic_lb, 'phl': phl,
                            'method': 'ic_weighted', 'corr_thresh': corr_thresh,
                        }

                # Method 4: Risk Parity
                pnl_s, dirs = portfolio_risk_parity(
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
                        'method': 'risk_parity', 'corr_thresh': corr_thresh,
                    }

    # Also try broader adaptive net with ALL good alphas (not orthogonal)
    all_good = list(alpha_sharpes.keys())
    all_good.sort(key=lambda n: alpha_sharpes.get(n, 0), reverse=True)
    for n_top in [3, 5, 8, 10, 15, 20, 30, 50, len(all_good)]:
        subset = all_good[:min(n_top, len(all_good))]
        if len(subset) < 2:
            continue
        for lb in [60, 120, 240, 480]:
            for phl in [48, 72, 120, 168, 240, 336]:
                pnl_s, dirs = streaming_pnl_adaptive(
                    alpha_df, ret, subset, lb, phl
                )
                val_pnl = pnl_s.loc[VAL_START:VAL_END]
                sr = compute_sharpe(val_pnl)
                tested += 1
                if sr is None:
                    continue
                vd = dirs.loc[VAL_START:VAL_END].values
                vd = np.where(np.isnan(vd), 0, vd)
                n_trades = int(np.abs(np.diff(np.concatenate([[0], vd]))).sum())
                if best is None or sr > best['sharpe']:
                    best = {
                        'sharpe': sr,
                        'cum_pnl_bps': val_pnl.sum() * 10000,
                        'n_trades': n_trades,
                        'daily_pnl': val_pnl.resample('1D').sum(),
                    }
                    best_cfg = {
                        'alphas': subset, 'n_alphas': len(subset),
                        'lookback': lb, 'phl': phl,
                        'method': 'adaptive_net_full',
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
    print("V10 ROUND 3: ADVANCED PORTFOLIO CONSTRUCTION")
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
    print("ROUND 3 FINAL")
    print(f"{'='*60}")

    if len(all_val_daily) >= 2:
        port_df = pd.DataFrame(all_val_daily).fillna(0)
        pd_daily = port_df.mean(axis=1)
        pd_daily = pd_daily[pd_daily != 0]
        if len(pd_daily) >= 10 and pd_daily.std() > 0:
            port_sharpe = pd_daily.mean() / pd_daily.std() * np.sqrt(365)
            print(f"\n  ★★★ COMBINED SHARPE: {port_sharpe:+.2f} ★★★")
            print(f"  PnL: {pd_daily.sum()*10000:+.0f} bps  Win: {(pd_daily>0).mean():.0%}")

    for sym in sorted(all_results.keys(), key=lambda s: all_results[s]['sharpe'], reverse=True):
        r = all_results[sym]
        cfg = r.get('config', {})
        print(f"  {sym:>10s}: SR={r['sharpe']:+.2f} PnL={r['cum_pnl_bps']:+.0f}bps "
              f"{cfg.get('method',''):>18s} phl={cfg.get('phl',0)} n={cfg.get('n_alphas',0)}")

    # Save
    frozen = {
        'version': 'V10_research_r3',
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
            'val_sharpe': r['sharpe'],
        }
    pf = Path('UNIFIED_V10/frozen_params_v10.json')
    with open(pf, 'w') as f:
        json.dump(frozen, f, indent=2, default=str)
    print(f"\n  Saved to {pf}")


if __name__ == '__main__':
    main()
