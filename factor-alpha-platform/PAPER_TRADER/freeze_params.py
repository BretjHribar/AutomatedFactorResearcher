"""
freeze_params.py — Run one walk-forward fold to extract frozen parameters for paper trading.

Uses the most recent data as train window, selects best config, and saves to JSON.
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, json, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'FINAL_v2', 'scripts'))
from univariate_hf_v9b_mtf import (
    build_1h_alphas, build_htf_signals, build_cross_asset_signals,
    eval_alpha, select_orthogonal, strategy_adaptive_net,
)
from config import SYMBOLS, DATA_DIR, PARAMS_FILE


def freeze():
    """Run parameter selection on latest data and save to JSON."""
    print("=" * 60)
    print("FREEZING PARAMETERS FOR PAPER TRADING")
    print("=" * 60)

    # Load data
    all_1h = {}
    all_data = {}
    for sym in SYMBOLS:
        parquet_path = DATA_DIR / f"{sym}.parquet"
        if not parquet_path.exists():
            print(f"  WARNING: {parquet_path} not found, skipping {sym}")
            continue
        df15 = pd.read_parquet(parquet_path)
        if 'datetime' in df15.columns:
            df15 = df15.set_index('datetime')
        df15 = df15.sort_index()
        df15 = df15[~df15.index.duplicated(keep='last')]
        for c in ['open','high','low','close','volume','quote_volume',
                   'taker_buy_volume','taker_buy_quote_volume']:
            if c in df15.columns:
                df15[c] = pd.to_numeric(df15[c], errors='coerce')
        all_data[sym] = df15
        all_1h[sym] = df15.resample('1h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
            'volume': 'sum', 'quote_volume': 'sum',
            'taker_buy_volume': 'sum', 'taker_buy_quote_volume': 'sum'
        }).dropna()

    frozen = {}

    for sym in SYMBOLS:
        if sym not in all_data:
            continue
        t0 = time.time()
        print(f"\n{sym}:")

        df15 = all_data[sym]
        df_1h = all_1h[sym]
        df_2h = df15.resample('2h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
            'volume': 'sum', 'quote_volume': 'sum',
            'taker_buy_volume': 'sum', 'taker_buy_quote_volume': 'sum'
        }).dropna()
        df_4h = df15.resample('4h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
            'volume': 'sum', 'quote_volume': 'sum',
            'taker_buy_volume': 'sum', 'taker_buy_quote_volume': 'sum'
        }).dropna()
        df_8h = df15.resample('8h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
            'volume': 'sum', 'quote_volume': 'sum',
            'taker_buy_volume': 'sum', 'taker_buy_quote_volume': 'sum'
        }).dropna()
        df_12h = df15.resample('12h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
            'volume': 'sum', 'quote_volume': 'sum',
            'taker_buy_volume': 'sum', 'taker_buy_quote_volume': 'sum'
        }).dropna()

        returns = df_1h['close'].pct_change()

        # Use the most recent 6/8/10 months as train
        best_sr = -999
        best_cfg = None
        best_sel = None

        for train_months in [6, 8, 10]:
            end = df_1h.index[-1]
            start = end - pd.DateOffset(months=train_months)
            train_1h = df_1h.loc[str(start):]
            if len(train_1h) < 1500:
                continue
            train_ret = returns.loc[train_1h.index]

            # Build all alphas
            alphas_1h = build_1h_alphas(train_1h)
            tr_2h = df_2h.loc[str(start):]
            tr_4h = df_4h.loc[str(start):]
            tr_8h = df_8h.loc[str(start):]
            tr_12h = df_12h.loc[str(start):]
            a2h = build_htf_signals(tr_2h, train_1h, 'h2', shift_n=1)
            a4h = build_htf_signals(tr_4h, train_1h, 'h4', shift_n=1)
            a8h = build_htf_signals(tr_8h, train_1h, 'h8', shift_n=1)
            a12h = build_htf_signals(tr_12h, train_1h, 'h12', shift_n=1)
            across = build_cross_asset_signals(all_1h, sym, train_1h)

            all_a = {**alphas_1h, **a2h, **a4h, **a8h, **a12h, **across}
            alpha_tr = pd.DataFrame(all_a, index=train_1h.index).shift(1)

            results = []
            for col in alpha_tr.columns:
                m = eval_alpha(alpha_tr[col], train_ret)
                if m and m['nofee_sharpe'] > 0:
                    results.append({'name': col, **m})
            results.sort(key=lambda x: x['nofee_sharpe'], reverse=True)
            if len(results) < 3:
                continue

            for cc in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
                for mn in [8, 10, 12, 15, 20, 25, 30]:
                    sel = select_orthogonal(results, alpha_tr, max_n=mn, corr_cutoff=cc)
                    if len(sel) < 3:
                        continue
                    for lb in [120, 180, 240, 360, 480, 720, 1440]:
                        for phl in [1, 2, 3, 6]:
                            mt = strategy_adaptive_net(
                                alpha_tr, train_ret, sel, lookback=lb, phl=phl)
                            if mt and mt['sharpe'] > best_sr:
                                best_sr = mt['sharpe']
                                best_cfg = {
                                    'cc': cc, 'mn': mn, 'lb': lb,
                                    'phl': phl, 'tw': train_months
                                }
                                best_sel = sel

        if best_cfg and best_sel:
            frozen[sym] = {
                'selected_alphas': [s['name'] for s in best_sel],
                'lookback': best_cfg['lb'],
                'phl': best_cfg['phl'],
                'train_sharpe': float(best_sr),
                'config': best_cfg,
                'n_alphas': len(best_sel),
            }
            elapsed = time.time() - t0
            print(f"  SR={best_sr:+.2f} | {len(best_sel)} alphas | "
                  f"lb={best_cfg['lb']} phl={best_cfg['phl']} tw={best_cfg['tw']}m | "
                  f"({elapsed:.0f}s)")
        else:
            print(f"  WARNING: No valid configuration found!")

    # Save to JSON
    output = {
        'version': 'V9b',
        'frozen_at': pd.Timestamp.now().isoformat(),
        'fee_bps': 3,
        'symbols': frozen,
    }

    with open(PARAMS_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Frozen params saved to {PARAMS_FILE}")
    print(f"   Symbols: {list(frozen.keys())}")


if __name__ == '__main__':
    freeze()
