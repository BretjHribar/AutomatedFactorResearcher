from eval_portfolio import load_raw_alpha_signals, proper_normalize_alpha, simulate

def strategy_consensus_super_alpha(raw_signals, returns_pct, close, universe, max_wt=0.04, decay=2, ema_halflife=30):
    consensus = None
    for alpha_id, raw in raw_signals.items():
        if consensus is None:
            consensus = raw.copy()
        else:
            consensus = consensus.add(raw, fill_value=0)
            
    normed = proper_normalize_alpha(consensus, universe, max_wt=max_wt)
    if ema_halflife > 0:
        normed = normed.ewm(halflife=ema_halflife, min_periods=1).mean()
        
    res = simulate(normed, returns_pct, close, universe, max_wt=max_wt, decay=decay)
    print(f"Consensus Super Alpha (hl={ema_halflife}): Sharpe {res.sharpe:.3f}, TO {res.turnover:.3f}")
    return res

if __name__ == '__main__':
    raw, ret, cl, uv = load_raw_alpha_signals()
    for hl in [0, 6, 15, 30, 45, 60]:
        strategy_consensus_super_alpha(raw, ret, cl, uv, ema_halflife=hl, decay=2)
