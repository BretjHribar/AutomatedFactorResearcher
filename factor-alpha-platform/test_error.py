import sys, os
sys.path.insert(0, os.path.abspath('.'))
import pandas as pd
from compare_pipelines import run_walkforward_qp, load_gp_alphas, features, alpha_signals

results = run_walkforward_qp(features, alpha_signals, fee_bps=0.0, train_bars=720, val_bars=360, reeval_interval=6, start_bar_override=8800)
print(results.tail())
