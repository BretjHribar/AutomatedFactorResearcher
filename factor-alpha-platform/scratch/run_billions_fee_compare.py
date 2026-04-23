import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn import linear_model

from eval_portfolio import (
    load_raw_alpha_signals, simulate, proper_normalize_alpha,
    VAL_START, VAL_END, MAX_WEIGHT
)

raw_signals, returns_pct, close, universe = load_raw_alpha_signals()
print(f'Loaded {len(raw_signals)} alphas — {VAL_START} to {VAL_END}')
print()
print(f'  {"Lookback":<12}  {"0bps SR":>9}  {"5bps SR":>9}  {"Fee Drag":>9}  {"TO":>6}')
print(f'  {"-"*55}')

def run_billions(raw_signals, returns_pct, close, universe, ol, fees_bps):
    normed = {aid: proper_normalize_alpha(raw, universe) for aid, raw in raw_signals.items()}
    dates     = close.index
    tickers   = close.columns.tolist()
    n_bars    = len(dates)
    aid_list  = list(normed.keys())
    n_alphas  = len(aid_list)
    ret_df    = returns_pct.reindex(index=dates, columns=tickers)

    # Factor returns — lagged 1 bar (no lookahead)
    fr_data = {}
    for aid, norm in normed.items():
        lagged = norm.shift(1)
        ab = lagged.abs().sum(axis=1).replace(0, np.nan)
        n  = lagged.div(ab, axis=0)
        fr_data[aid] = (n * ret_df).sum(axis=1)
    fr_df = pd.DataFrame(fr_data, index=dates)

    # Expected returns — rolling mean shifted 1 bar (no lookahead)
    min_periods = max(1, ol // 2)
    alphas_exp_ret = (
        fr_df.rolling(window=ol, min_periods=min_periods)
             .mean()
             .shift(1)
             .clip(lower=0)
    )

    # Walk-forward regression (weights stored at optim_end+1 — no lookahead)
    reg = linear_model.LinearRegression(fit_intercept=False)
    alpha_weights_ts = pd.DataFrame(1.0 / n_alphas, index=dates, columns=aid_list)

    for test_start in range(1, n_bars - ol - 2):
        optim_end = test_start + ol
        if optim_end + 1 >= n_bars:
            break
        try:
            bil_df   = fr_df.iloc[test_start:optim_end].copy()
            demeaned = bil_df - bil_df.mean(axis=0)
            sample_std = demeaned.std(axis=0).replace(0, np.nan)
            if sample_std.isna().any():
                continue
            normalized = demeaned.divide(sample_std)
            A_is    = normalized.fillna(0.0)
            sub_exp = alphas_exp_ret.iloc[test_start:optim_end].divide(sample_std).fillna(0.0)
            reg.fit(A_is.values, sub_exp.values)
            residuals = pd.DataFrame(
                reg.predict(A_is.values) - sub_exp.values,
                index=sub_exp.index, columns=sub_exp.columns
            )
            opt_w    = residuals.divide(sample_std)
            row_sums = opt_w.sum(axis=1).replace(0, np.nan)
            opt_w    = opt_w.div(row_sums, axis=0)
            alpha_weights_ts.iloc[optim_end + 1] = opt_w.iloc[-1].values
        except Exception:
            pass

    # Combine signals
    combined = None
    for aid in aid_list:
        w  = alpha_weights_ts[aid]
        ws = normed[aid].mul(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    return simulate(combined, returns_pct, close, universe, fees_bps=fees_bps)


for ol in [5, 10, 30, 60, 120]:
    try:
        r_gross = run_billions(raw_signals, returns_pct, close, universe, ol=ol, fees_bps=0.0)
        r_net   = run_billions(raw_signals, returns_pct, close, universe, ol=ol, fees_bps=5.0)
        drag    = r_gross.sharpe - r_net.sharpe
        print(f'  ol={ol:<9}  {r_gross.sharpe:>+8.3f}   {r_net.sharpe:>+8.3f}   {drag:>+8.3f}   {r_net.turnover:>5.3f}')
    except Exception as e:
        print(f'  ol={ol} FAILED: {e}')
