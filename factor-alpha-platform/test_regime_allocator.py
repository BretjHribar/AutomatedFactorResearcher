import numpy as np
import pandas as pd
from eval_portfolio import load_raw_alpha_signals, compute_factor_returns, proper_normalize_alpha, simulate

def strategy_regime_allocator(raw_signals, returns_pct, close, universe, max_wt=0.03, lookback=280, ema_halflife=60, decay=2):
    """
    Sub-divides the alpha pool into Mean-Reversion and Trend clusters based on autocorrelation.
    Allocates between the two clusters dynamically using the Volatility Regime indicator.
    """
    # 1. Normalize and compute basic stats
    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    fr_df = compute_factor_returns(normed_signals, returns_pct)

    # 2. Cluster by Autocorrelation (1-lag)
    # A positive autocorrelation implies Trend, negative implies Mean-Reversion
    autocorr = {}
    for col in fr_df.columns:
        valid_rets = fr_df[col].dropna()
        if len(valid_rets) > 10:
            autocorr[col] = valid_rets.autocorr(lag=1)
        else:
            autocorr[col] = 0.0

    mr_cols = [c for c, ac in autocorr.items() if ac < 0.0]
    trend_cols = [c for c, ac in autocorr.items() if ac >= 0.0]
    
    # Give it at least some cols if one is empty
    if not mr_cols: mr_cols = list(autocorr.keys())
    if not trend_cols: trend_cols = list(autocorr.keys())

    # 3. Compute inner base weights using proper_decay logic
    rolling_er = fr_df.rolling(window=lookback, min_periods=20).mean()
    weights = rolling_er.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)

    smoothed = weights_norm.ewm(halflife=ema_halflife, min_periods=1).mean()
    wsum2 = smoothed.sum(axis=1).replace(0, np.nan)
    smoothed_norm = smoothed.div(wsum2, axis=0).fillna(0)

    # 4. Construct two separate portfolios!
    combined_mr = None
    combined_tr = None
    
    for alpha_id, normed in normed_signals.items():
        w = smoothed_norm[alpha_id].values
        ws = normed.multiply(w, axis=0)
        
        if alpha_id in mr_cols:
            combined_mr = ws if combined_mr is None else combined_mr.add(ws, fill_value=0)
        if alpha_id in trend_cols:
            combined_tr = ws if combined_tr is None else combined_tr.add(ws, fill_value=0)

    # 5. Volatility Modulator -> weight allocations
    market_returns = returns_pct.mean(axis=1)
    market_vol = market_returns.rolling(window=84, min_periods=20).std()
    market_vol_smooth = market_vol.ewm(halflife=42).mean()

    median_vol = market_vol_smooth.expanding().median().clip(lower=1e-4)
    # Vol stress = current / median. High stress = > 1.0
    vol_stress = (market_vol_smooth / median_vol).clip(lower=0.5, upper=2.0)
    vol_stress_smooth = vol_stress.ewm(halflife=30).mean()

    # Dynamic Weights:
    # High vol stress -> Trend does well, MR breaks down.
    # Low vol stress (<1.0) -> MR dominates, Trend bleeds.
    w_trend = ((vol_stress_smooth - 0.5) / 1.5).clip(lower=0, upper=1.0) # Scale 0.5->0%, 2.0->100%
    w_mr = 1.0 - w_trend

    # 6. Apply regime weights
    if combined_mr is None: combined_mr = combined_tr.copy() * 0.0
    if combined_tr is None: combined_tr = combined_mr.copy() * 0.0

    final_portfolio = combined_mr.multiply(w_mr, axis=0).add(combined_tr.multiply(w_trend, axis=0), fill_value=0)

    res = simulate(final_portfolio, returns_pct, close, universe, max_wt=max_wt, decay=decay)
    
    print(f"MR Alphas: {len(mr_cols)}, Trend Alphas: {len(trend_cols)}")
    print(f"RegimeAllocator(mw={max_wt},lb={lookback}): Sharpe {res.sharpe:.3f}, TO {res.turnover:.3f}, DD {res.max_drawdown:.3f}")
    return res

if __name__ == '__main__':
    raw, ret, cl, uv = load_raw_alpha_signals()
    for lb in [180, 240, 280, 360]:
        for hl in [30, 60, 90]:
            try:
                strategy_regime_allocator(raw, ret, cl, uv, max_wt=0.02, lookback=lb, ema_halflife=hl, decay=2)
            except Exception as e:
                print("Failed:", e)
