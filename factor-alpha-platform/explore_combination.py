import sqlite3
import pandas as pd
import numpy as np
from eval_portfolio import load_raw_alpha_signals, strategy_proper_decay

def load_data():
    raw_signals, returns_pct, close, universe = load_raw_alpha_signals()
    return raw_signals, returns_pct, close, universe

def explore():
    raw_signals, returns_pct, close, universe = load_data()
    if not raw_signals:
        return
        
    res, _ = strategy_proper_decay(raw_signals, returns_pct, close, universe, max_wt=0.04, lookback=280, decay=2, ema_halflife=30)
    print(f"Base ProperDecay All Alphas: Sharpe {res.sharpe:.3f}, Fitness {res.fitness:.3f}, TO {res.turnover:.3f}")
    
    # Calculate turnover for each alpha independently
    alpha_metrics = {}
    from eval_portfolio import simulate, proper_normalize_alpha
    for aid, raw_sig in raw_signals.items():
        normed = proper_normalize_alpha(raw_sig, universe, max_wt=0.04)
        sim_res = simulate(normed, returns_pct, close, universe, max_wt=0.04, decay=0)
        alpha_metrics[aid] = {
            'sharpe': sim_res.sharpe,
            'turnover': sim_res.turnover,
            'fitness': sim_res.fitness
        }
    
    df_metrics = pd.DataFrame.from_dict(alpha_metrics, orient='index')
    print("\nAlpha Metrics:")
    print(df_metrics.sort_values('sharpe', ascending=False).head(15))
    
    # Find combinations with strict filters
    print("\nEvaluating Combinations:")
    
    # Strategy 1: Top 5 Highest Sharpe
    top_5_sharpe = df_metrics.sort_values('sharpe', ascending=False).head(5).index
    sub_signals = {aid: raw_signals[aid] for aid in top_5_sharpe}
    res, _ = strategy_proper_decay(sub_signals, returns_pct, close, universe, max_wt=0.04, lookback=280, decay=2, ema_halflife=30)
    print(f"Top 5 Sharpe: Sharpe {res.sharpe:.3f}, TO {res.turnover:.3f}")
    
    # Strategy 2: Top 10 Lowest Turnover (with Sharpe > 0.0)
    low_to = df_metrics[df_metrics['sharpe'] > 0.0].sort_values('turnover').head(10).index
    sub_signals2 = {aid: raw_signals[aid] for aid in low_to}
    res, _ = strategy_proper_decay(sub_signals2, returns_pct, close, universe, max_wt=0.04, lookback=280, decay=2, ema_halflife=30)
    print(f"Low TO Filter: Sharpe {res.sharpe:.3f}, TO {res.turnover:.3f}")

    # Strategy 3: Best Fitness Subset
    best_fitness = df_metrics[df_metrics['fitness'] > 0.0].index
    if len(best_fitness) > 0:
        sub_signals3 = {aid: raw_signals[aid] for aid in best_fitness}
        res, _ = strategy_proper_decay(sub_signals3, returns_pct, close, universe, max_wt=0.04, lookback=280, decay=2, ema_halflife=30)
        print(f"Positive Fitness Subset: Sharpe {res.sharpe:.3f}, TO {res.turnover:.3f}")

if __name__ == '__main__':
    explore()
