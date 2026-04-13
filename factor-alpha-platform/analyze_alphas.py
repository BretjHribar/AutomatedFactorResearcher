"""
Analyze all 22 alphas for quality and duplication.
Output clean JSON results.
"""
import sqlite3, sys, os, time, json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_alpha_5m import (
    load_data, evaluate_expression, process_signal, compute_ic,
    MAX_WEIGHT, BARS_PER_DAY
)

DB_PATH = "data/alphas_5m.db"

def main():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        'SELECT id, expression, notes, created_at FROM alphas WHERE archived=0 ORDER BY id'
    ).fetchall()
    conn.close()

    matrices, universe = load_data("train")
    returns_pct = matrices["close"].pct_change()

    alpha_signals = {}
    alpha_data = {}

    for aid, expr, notes, created in rows:
        try:
            raw = evaluate_expression(expr, matrices)
            if raw is None:
                continue
            processed = process_signal(raw, universe_df=universe, max_wt=MAX_WEIGHT)
            ic_series = compute_ic(raw, returns_pct, universe_df=universe)
            ic_clean = ic_series.dropna()
            ic_mean = float(ic_clean.mean()) if len(ic_clean) > 0 else 0
            ic_std = float(ic_clean.std()) if len(ic_clean) > 1 else 1
            icir = ic_mean / ic_std if ic_std > 0 else 0

            alpha_signals[aid] = processed
            
            # Extract key fields
            fields = []
            for field in ['taker_buy_ratio', 'lower_shadow', 'upper_shadow', 'volume_momentum',
                           'beta_to_btc', 'close_position_in_range', 'log_returns', 'vwap_deviation',
                           'trades_per_volume', 'parkinson_volatility', 'historical_volatility',
                           'open_close_range', 'high_low_range', 'quote_volume', 'adv60',
                           'trades_count', 'close', 'volume']:
                if field in expr:
                    fields.append(field)
            
            alpha_data[str(aid)] = {
                'ic_mean': round(ic_mean, 5),
                'icir': round(icir, 4),
                'notes': (notes or '')[:100],
                'expr_short': expr[:100],
                'created': created,
                'fields': fields,
            }
        except Exception as e:
            alpha_data[str(aid)] = {'error': str(e)}

    # Pairwise correlations
    ids = sorted(alpha_signals.keys())
    n = len(ids)
    
    high_corrs = []
    for i in range(n):
        for j in range(i+1, n):
            a = alpha_signals[ids[i]]
            b = alpha_signals[ids[j]]
            common_idx = a.index.intersection(b.index)
            common_cols = a.columns.intersection(b.columns)
            va = a.loc[common_idx, common_cols].values.flatten()
            vb = b.loc[common_idx, common_cols].values.flatten()
            mask = np.isfinite(va) & np.isfinite(vb)
            if mask.sum() < 100:
                continue
            c = float(np.corrcoef(va[mask], vb[mask])[0, 1])
            if abs(c) > 0.40:
                high_corrs.append({
                    'alpha_a': ids[i],
                    'alpha_b': ids[j],
                    'corr': round(c, 3),
                    'duplicate': abs(c) > 0.70
                })
    
    high_corrs.sort(key=lambda x: abs(x['corr']), reverse=True)

    ics = [alpha_data[str(a)].get('ic_mean', 0) for a in ids if 'ic_mean' in alpha_data.get(str(a), {})]
    
    result = {
        'summary': {
            'total_alphas': n,
            'ic_range': [round(min(ics), 5), round(max(ics), 5)],
            'mean_ic': round(float(np.mean(ics)), 5),
            'alphas_ic_positive': sum(1 for x in ics if x > 0),
            'alphas_ic_gt_001': sum(1 for x in ics if x > 0.001),
            'high_corr_pairs_gt_050': len(high_corrs),
            'duplicate_pairs_gt_070': sum(1 for x in high_corrs if x['duplicate']),
        },
        'alphas': alpha_data,
        'correlations': high_corrs,
    }

    with open('alpha_analysis.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print("Done. Results in alpha_analysis.json")

if __name__ == "__main__":
    main()
