import numpy as np
import pandas as pd
from eval_portfolio import load_raw_alpha_signals, proper_normalize_alpha, simulate, compute_net_factor_returns

raw_signals, returns_pct, close, universe = load_raw_alpha_signals()
results = []
for aid, raw_sig in raw_signals.items():
    try:
        normed = proper_normalize_alpha(raw_sig, universe, max_wt=0.04)
        res = simulate(normed, returns_pct, close, universe, max_wt=0.04, fees_bps=0.0, decay=2)
        results.append((aid, res.sharpe, res.turnover))
    except Exception as e:
        pass

results.sort(key=lambda x: x[1], reverse=True)
print("Top 20 Alphas at 0bps (GROSS PRE-FEE):")
for aid, sh, to in results[:20]:
    print(f"Alpha {aid}: SR {sh:.3f}, TO {to:.3f}")
