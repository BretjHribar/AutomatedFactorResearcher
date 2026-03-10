from eval_portfolio import load_raw_alpha_signals, strategy_proper_decay

def explore():
    raw_signals, returns_pct, close, universe = load_raw_alpha_signals()
    if not raw_signals:
        return
        
    res, _ = strategy_proper_decay(raw_signals, returns_pct, close, universe, max_wt=0.04, lookback=280, decay=0, ema_halflife=30)
    print(f"Base ProperDecay (Decay 0): Sharpe {res.sharpe:.3f}, Fitness {res.fitness:.3f}, TO {res.turnover:.3f}")

if __name__ == '__main__':
    explore()
