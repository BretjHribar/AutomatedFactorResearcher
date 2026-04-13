import subprocess
import os

candidates = [
    'true_divide(lower_shadow, df_max(high_low_range, 0.001))',
    'taker_buy_ratio',
    'close_position_in_range',
    'volume_ratio_20d',
    'ts_delta(trades_per_volume, 12)',
    'ts_delta(s_log_1p(adv60), 36)',
    'historical_volatility_20'
]

results = []
for c in candidates:
    expr = f'negative(subtract(rank(sma(ts_corr(log_returns, {c}, 144), 1440)), 0.5))'
    print(f'Evaluating: {c}')
    try:
        out = subprocess.check_output(['python', 'eval_alpha_5m.py', '--expr', expr], text=True)
        sr = [line for line in out.split('\n') if 'SR=' in line]
        if sr:
            print('  ->', sr[0].strip())
    except Exception as e:
        print('Error:', e)
