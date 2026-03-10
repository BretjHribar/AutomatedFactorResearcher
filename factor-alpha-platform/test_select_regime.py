from eval_portfolio import load_raw_alpha_signals, compute_factor_returns, proper_normalize_alpha, simulate
import numpy as np
import pandas as pd

def strategy_select_regime_scaled(raw_signals, returns_pct, close, universe, 
                                  max_wt=0.03, lookback=280, ema_halflife=60, top_n=10, decay=2):
    
    normed_signals = {}
    for alpha_id, raw_signal in raw_signals.items():
        normed_signals[alpha_id] = proper_normalize_alpha(raw_signal, universe, max_wt=max_wt)

    fr_df = compute_factor_returns(normed_signals, returns_pct)
    rolling_er = fr_df.rolling(window=lookback, min_periods=20).mean()
    
    # Selection logic step:
    # Only keep the top_n alphas with highest rolling ER at time t
    
    weights_np = rolling_er.values
    selected_weights = np.zeros_like(weights_np)
    
    for i in range(len(weights_np)):
        row = weights_np[i]
        if np.isnan(row).all():
            continue
        # Get indices of top_n positive ERs
        valid_indices = np.where(~np.isnan(row) & (row > 0))[0]
        if len(valid_indices) == 0:
            continue
        
        valid_vals = row[valid_indices]
        
        # Sort indices by value descending
        sort_order = np.argsort(valid_vals)[::-1]
        top_indices = valid_indices[sort_order[:top_n]]
        
        # Set selected weights
        selected_weights[i, top_indices] = row[top_indices]
        
    selected_weights_df = pd.DataFrame(selected_weights, index=rolling_er.index, columns=rolling_er.columns)

    wsum = selected_weights_df.sum(axis=1).replace(0, np.nan)
    weights_norm = selected_weights_df.div(wsum, axis=0).fillna(0)

    smoothed = weights_norm.ewm(halflife=ema_halflife, min_periods=1).mean()
    wsum2 = smoothed.sum(axis=1).replace(0, np.nan)
    smoothed_norm = smoothed.div(wsum2, axis=0).fillna(0)

    combined = None
    for alpha_id, normed in normed_signals.items():
        w = smoothed_norm[alpha_id].values
        ws = normed.multiply(w, axis=0)
        combined = ws if combined is None else combined.add(ws, fill_value=0)

    # 2. Market Regime Volatility Modulator
    market_returns = returns_pct.mean(axis=1)
    market_vol = market_returns.rolling(window=84, min_periods=20).std()
    market_vol_smooth = market_vol.ewm(halflife=42).mean()

    median_vol = market_vol_smooth.expanding().median().clip(lower=1e-4) 
    vol_scalar = (median_vol / market_vol_smooth).clip(lower=0.2, upper=2.5)
    vol_scalar_smooth = vol_scalar.ewm(halflife=30, min_periods=1).mean()
    
    combined_scaled = combined.multiply(vol_scalar_smooth, axis=0)
    
    res = simulate(combined_scaled, returns_pct, close, universe, max_wt=max_wt, decay=decay)
    print(f"Top-{top_n} RegimeScaled: Sharpe {res.sharpe:.3f}, TO {res.turnover:.3f}, DD {res.max_drawdown:.3f}")
    return res

if __name__ == '__main__':
    raw, ret, cl, uv = load_raw_alpha_signals()
    for n in [5, 10, 15, 20]:
        for mw in [0.03, 0.05, 0.08]:
            strategy_select_regime_scaled(raw, ret, cl, uv, top_n=n, max_wt=mw, lookback=240, ema_halflife=60)
