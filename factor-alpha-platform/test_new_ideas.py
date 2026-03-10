import numpy as np
import pandas as pd
from eval_portfolio import load_raw_alpha_signals, compute_factor_returns, proper_normalize_alpha, simulate

def strategy_regime_sharpe(raw_signals, returns_pct, close, universe,
                           max_wt=0.02, lookback=240, decay=2, ema_halflife=60, **kwargs):
    """
    Like RegimeScaled, but allocates to factors based on their rolling Sharpe ratio
    (mean/std) rather than just raw mean return.
    """
    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    fr_df = compute_factor_returns(normed_signals, returns_pct)
    rolling_er = fr_df.rolling(window=lookback, min_periods=20).mean()
    rolling_std = fr_df.rolling(window=lookback, min_periods=20).std()
    
    rolling_sharpe = rolling_er / rolling_std.replace(0, np.nan)
    
    weights = rolling_sharpe.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)

    smoothed = weights_norm.ewm(halflife=ema_halflife, min_periods=1).mean()
    wsum2 = smoothed.sum(axis=1).replace(0, np.nan)
    smoothed_norm = smoothed.div(wsum2, axis=0).fillna(0)

    combined = None
    for alpha_id, normed in normed_signals.items():
        w = smoothed_norm[alpha_id].values
        ws = normed.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    # Volatility Modulator
    market_returns = returns_pct.mean(axis=1)
    market_vol = market_returns.rolling(window=84, min_periods=20).std()
    market_vol_smooth = market_vol.ewm(halflife=42).mean()

    median_vol = market_vol_smooth.expanding().median().clip(lower=1e-4) 
    vol_scalar = (median_vol / market_vol_smooth).clip(lower=0.2, upper=2.5)
    vol_scalar_smooth = vol_scalar.ewm(halflife=30, min_periods=1).mean()
    
    combined_scaled = combined.multiply(vol_scalar_smooth, axis=0)
    
    res = simulate(combined_scaled, returns_pct, close, universe, max_wt=max_wt, decay=decay)
    print(f"RegimeSharpe(mw={max_wt}, lb={lookback}, hl={ema_halflife}): Sharpe {res.sharpe:.3f}, TO {res.turnover:.3f}, DD {res.max_drawdown:.3f}")
    return res

if __name__ == '__main__':
    raw, ret, cl, uv = load_raw_alpha_signals()
    for lb in [180, 240, 280, 360]:
        for hl in [30, 45, 60, 90]:
            try:
                strategy_regime_sharpe(raw, ret, cl, uv, max_wt=0.02, lookback=lb, ema_halflife=hl, decay=2)
            except Exception as e:
                print("Failed:", e)
