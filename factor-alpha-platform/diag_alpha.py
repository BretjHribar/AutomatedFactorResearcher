"""Diagnostic: check alpha quality on different windows."""
import eval_alpha_5m as ea

# Top alpha
expr = 'df_min(df_max(add(add(rank(sma(ts_rank(s_log_1p(adv20), 36), 72)),rank(sma(ts_rank(s_log_1p(taker_buy_volume), 12), 36))),rank(negative(sma(vwap_deviation, 12)))), -1.5), 1.5)'

r = ea.eval_single(expr, split='train', fees_bps=0)
print("Alpha #76 on FULL train (Feb25-Feb26):")
print("Keys:", list(r.keys()))
for k, v in sorted(r.items()):
    if isinstance(v, (int, float)):
        print(f"  {k} = {v}")
