import numpy as np
import pandas as pd
from eval_portfolio import load_raw_alpha_signals, strategy_regime_net_smooth, strategy_orthogonal_regime_scaled, proper_normalize_alpha, compute_net_factor_returns, simulate

if __name__ == '__main__':
    raw, ret, cl, uv = load_raw_alpha_signals()
    print(f"Loaded {len(raw)} alphas.")
    
    # Sweep standard PosSmooth at 7bps
    for mw in [0.02, 0.025, 0.03]:
        for phl in [60, 72, 90, 120]:
            try:
                res, label = strategy_regime_net_smooth(raw, ret, cl, uv, max_wt=mw, lookback=280, pos_halflife=phl, fee_bps=7.0)
                print(f"PosSmooth(mw={mw},phl={phl}): Sharpe {res.sharpe:.3f}, TO {res.turnover:.3f}, DD {res.max_drawdown:.3f}")
            except Exception as e:
                pass
