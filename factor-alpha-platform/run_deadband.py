import numpy as np
import pandas as pd
from eval_portfolio import load_raw_alpha_signals, strategy_proper_decay, compute_factor_returns, proper_normalize_alpha, simulate

def strategy_deadband_decay(raw_signals, returns_pct, close, universe, max_wt=0.04, lookback=280, ema_halflife=30, deadband=0.01):
    # 1. Base proper decay logic to get combined signal
    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    fr_df = compute_factor_returns(normed_signals, returns_pct)
    rolling_er = fr_df.rolling(window=lookback, min_periods=20).mean()
    weights = rolling_er.clip(lower=0)
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

    # 2. Deadband Filter at the final ticker level
    combined_np = combined.values
    filtered_np = np.zeros_like(combined_np)
    prev = np.zeros(combined_np.shape[1])
    
    for i in range(len(combined_np)):
        target = combined_np[i]
        diff = np.abs(target - prev)
        
        mask_nan = np.isnan(target)
        update_mask = (diff > deadband) & ~mask_nan
        
        prev = np.where(update_mask, target, prev)
        prev[mask_nan] = np.nan
        
        filtered_np[i] = prev
        
    filtered_df = pd.DataFrame(filtered_np, index=combined.index, columns=combined.columns)
    
    res = simulate(filtered_df, returns_pct, close, universe, max_wt=max_wt, decay=2)
    print(f"Deadband ({deadband}): Sharpe {res.sharpe:.3f}, TO {res.turnover:.3f}, Return {res.fitness:.3f}")
    return res

if __name__ == '__main__':
    raw, ret, cl, uv = load_raw_alpha_signals()
    for db in [0.005, 0.01, 0.02, 0.03]:
        strategy_deadband_decay(raw, ret, cl, uv, max_wt=0.04, lookback=280, ema_halflife=30, deadband=db)
