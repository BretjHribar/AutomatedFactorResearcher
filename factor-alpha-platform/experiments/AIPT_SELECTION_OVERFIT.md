# AIPT Selection Overfit Report

Generated: `2026-05-12T01:35:09.253262`

For each result folder and scenario/layer, this compares the spec selected by TRAIN Sharpe to the spec selected by VAL+TEST/FULL Sharpe.

## aipt_asset_signal_top3000_d0_longtrain_smoke

- Class: unclassified AIPT output
- Status: completed_or_smoke

### equity_top3000_d0

- TRAIN-selected: TRAIN=4.191, VAL=1.914, TEST=-0.862, VAL+TEST=-0.430, FULL=-0.230, TRAIN-VAL+TEST=+4.620, TRAIN-TEST=+5.053
- TRAIN-selected spec: `scenario=equity_top3000_d0, source_set=all, projected_sources=False, dynamic_universe=True, P=256, z=0.001, activation=sincos, seed=1, weight_mode=subindustry_gross, train_window=1500, rebalance_every=5, start_override=2016-01-01, demean_features=False`
- VAL+TEST-selected: TRAIN=3.000, VAL=2.573, TEST=-0.613, VAL+TEST=-0.026, FULL=0.225, TRAIN-VAL+TEST=+3.026, TRAIN-TEST=+3.613
- VAL+TEST-selected spec: `scenario=equity_top3000_d0, source_set=all, projected_sources=False, dynamic_universe=True, P=256, z=0.01, activation=sincos, seed=1, weight_mode=raw_gross, train_window=1500, rebalance_every=5, start_override=2016-01-01, demean_features=False`

## aipt_asset_signal_top3000_d0_price_longtrain

- Class: unclassified AIPT output
- Status: completed_or_smoke

### equity_top3000_d0

- TRAIN-selected: TRAIN=5.538, VAL=2.025, TEST=7.794, VAL+TEST=2.222, FULL=2.097, TRAIN-VAL+TEST=+3.316, TRAIN-TEST=-2.257
- TRAIN-selected spec: `scenario=equity_top3000_d0, source_set=price, projected_sources=False, dynamic_universe=True, P=256, z=0.001, activation=sincos, seed=1, weight_mode=subindustry_gross, train_window=1500, rebalance_every=5, start_override=2016-01-01, demean_features=False`
- VAL+TEST-selected: TRAIN=3.254, VAL=2.561, TEST=3.485, VAL+TEST=2.715, FULL=2.571, TRAIN-VAL+TEST=+0.540, TRAIN-TEST=-0.231
- VAL+TEST-selected spec: `scenario=equity_top3000_d0, source_set=price, projected_sources=False, dynamic_universe=True, P=1024, z=0.01, activation=sincos, seed=1, weight_mode=demean_gross, train_window=1500, rebalance_every=5, start_override=2016-01-01, demean_features=False`

## aipt_asset_signal_top3000_d0_price_longtrain_seed_sweep

- Class: unclassified AIPT output
- Status: completed_or_smoke

### equity_top3000_d0

- TRAIN-selected: TRAIN=5.274, VAL=4.063, TEST=2.809, VAL+TEST=3.372, FULL=3.767, TRAIN-VAL+TEST=+1.902, TRAIN-TEST=+2.465
- TRAIN-selected spec: `scenario=equity_top3000_d0, source_set=price, projected_sources=False, dynamic_universe=True, P=256, z=0.001, activation=sincos, seed=5, weight_mode=subindustry_gross, train_window=1500, rebalance_every=5, start_override=2016-01-01, demean_features=False`
- VAL+TEST-selected: TRAIN=3.454, VAL=7.507, TEST=4.658, VAL+TEST=5.649, FULL=4.520, TRAIN-VAL+TEST=-2.195, TRAIN-TEST=-1.204
- VAL+TEST-selected spec: `scenario=equity_top3000_d0, source_set=price, projected_sources=False, dynamic_universe=True, P=256, z=0.01, activation=sincos, seed=5, weight_mode=subindustry_rank_gross, train_window=1500, rebalance_every=5, start_override=2016-01-01, demean_features=False`

## aipt_asset_signal_top3000_d0_price_seed_postprocess

- Class: unclassified AIPT output
- Status: completed_or_smoke

### equity_top3000_d0

- TRAIN-selected: TRAIN=5.709, VAL=2.169, TEST=5.232, VAL+TEST=2.392, FULL=2.342, TRAIN-VAL+TEST=+3.317, TRAIN-TEST=+0.476
- TRAIN-selected spec: `scenario=equity_top3000_d0, source_set=price, dynamic_universe=True, P=256, z=0.001, activation=sincos, n_seeds=5, weight_mode=subindustry_gross, train_window=1500, rebalance_every=5, start_override=2016-01-01`
- VAL+TEST-selected: TRAIN=3.291, VAL=2.413, TEST=4.432, VAL+TEST=2.725, FULL=2.599, TRAIN-VAL+TEST=+0.567, TRAIN-TEST=-1.141
- VAL+TEST-selected spec: `scenario=equity_top3000_d0, source_set=price, dynamic_universe=True, P=256, z=0.01, activation=sincos, n_seeds=5, weight_mode=demean_gross, train_window=1500, rebalance_every=5, start_override=2016-01-01`

## aipt_asset_signal_top3000_d1_longtrain_smoke

- Class: unclassified AIPT output
- Status: completed_or_smoke

### equity_top3000_d1

- TRAIN-selected: TRAIN=3.271, VAL=-0.694, TEST=3.025, VAL+TEST=-0.352, FULL=-0.035, TRAIN-VAL+TEST=+3.623, TRAIN-TEST=+0.246
- TRAIN-selected spec: `scenario=equity_top3000_d1, source_set=all, projected_sources=False, dynamic_universe=True, P=256, z=0.001, activation=sincos, seed=1, weight_mode=subindustry_gross, train_window=1500, rebalance_every=5, start_override=2016-01-01, demean_features=False`
- VAL+TEST-selected: TRAIN=2.623, VAL=-0.585, TEST=4.666, VAL+TEST=-0.095, FULL=0.216, TRAIN-VAL+TEST=+2.718, TRAIN-TEST=-2.042
- VAL+TEST-selected spec: `scenario=equity_top3000_d1, source_set=all, projected_sources=False, dynamic_universe=True, P=256, z=0.01, activation=sincos, seed=1, weight_mode=raw_gross, train_window=1500, rebalance_every=5, start_override=2016-01-01, demean_features=False`

## aipt_asset_signal_top3000_d1_price_longtrain

- Class: unclassified AIPT output
- Status: completed_or_smoke

### equity_top3000_d1

- TRAIN-selected: TRAIN=3.908, VAL=-0.747, TEST=4.478, VAL+TEST=-0.421, FULL=-0.190, TRAIN-VAL+TEST=+4.329, TRAIN-TEST=-0.570
- TRAIN-selected spec: `scenario=equity_top3000_d1, source_set=price, projected_sources=False, dynamic_universe=True, P=1024, z=0.001, activation=sincos, seed=1, weight_mode=subindustry_gross, train_window=1500, rebalance_every=5, start_override=2016-01-01, demean_features=False`
- VAL+TEST-selected: TRAIN=2.491, VAL=-0.527, TEST=2.051, VAL+TEST=-0.132, FULL=0.299, TRAIN-VAL+TEST=+2.623, TRAIN-TEST=+0.440
- VAL+TEST-selected spec: `scenario=equity_top3000_d1, source_set=price, projected_sources=False, dynamic_universe=True, P=256, z=0.01, activation=sincos, seed=1, weight_mode=demean_gross, train_window=1500, rebalance_every=5, start_override=2016-01-01, demean_features=False`

## aipt_extended

- Class: diagnostic constrained/cost sweep
- Status: completed_or_smoke

### equity_smallcap_d0

- TRAIN-selected: TRAIN=1.651, VAL=1.703, TEST=1.005, VAL+TEST=1.321, FULL=1.549, TRAIN-VAL+TEST=+0.331, TRAIN-TEST=+0.646
- TRAIN-selected spec: `scenario=equity_smallcap_d0, source_set=all, projected=False, P=64, z=0.001, activation=sincos, seed=1, cost_tau=0.0`
- VAL+TEST-selected: TRAIN=1.651, VAL=1.703, TEST=1.005, VAL+TEST=1.321, FULL=1.549, TRAIN-VAL+TEST=+0.331, TRAIN-TEST=+0.646
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, source_set=all, projected=False, P=64, z=0.001, activation=sincos, seed=1, cost_tau=0.0`

## aipt_extended2

- Class: diagnostic constrained/cost sweep
- Status: completed_or_smoke

### equity_smallcap_d0

- TRAIN-selected: TRAIN=2.603, VAL=0.639, TEST=0.954, VAL+TEST=0.765, FULL=2.057, TRAIN-VAL+TEST=+1.839, TRAIN-TEST=+1.649
- TRAIN-selected spec: `scenario=equity_smallcap_d0, source_set=all, projected=False, P=256, z=0.001, activation=sincos, seed=1, cost_tau=0.0`
- VAL+TEST-selected: TRAIN=0.781, VAL=0.878, TEST=2.060, VAL+TEST=1.470, FULL=0.994, TRAIN-VAL+TEST=-0.688, TRAIN-TEST=-1.279
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, source_set=all, projected=False, P=64, z=0.1, activation=sincos, seed=1, cost_tau=0.0`

### equity_smallcap_d1

- TRAIN-selected: TRAIN=2.239, VAL=1.104, TEST=0.267, VAL+TEST=0.666, FULL=1.749, TRAIN-VAL+TEST=+1.573, TRAIN-TEST=+1.972
- TRAIN-selected spec: `scenario=equity_smallcap_d1, source_set=all, projected=False, P=256, z=0.001, activation=sincos, seed=1, cost_tau=0.0`
- VAL+TEST-selected: TRAIN=0.777, VAL=1.105, TEST=1.929, VAL+TEST=1.542, FULL=1.014, TRAIN-VAL+TEST=-0.765, TRAIN-TEST=-1.153
- VAL+TEST-selected spec: `scenario=equity_smallcap_d1, source_set=all, projected=False, P=64, z=0.1, activation=sincos, seed=1, cost_tau=0.0`

### equity_top1000_d0

- TRAIN-selected: TRAIN=0.843, VAL=0.352, TEST=1.210, VAL+TEST=0.800, FULL=0.831, TRAIN-VAL+TEST=+0.043, TRAIN-TEST=-0.368
- TRAIN-selected spec: `scenario=equity_top1000_d0, source_set=all, projected=True, P=256, z=0.001, activation=sincos, seed=1, cost_tau=0.0`
- VAL+TEST-selected: TRAIN=0.707, VAL=0.762, TEST=1.500, VAL+TEST=1.147, FULL=0.816, TRAIN-VAL+TEST=-0.440, TRAIN-TEST=-0.793
- VAL+TEST-selected spec: `scenario=equity_top1000_d0, source_set=all, projected=False, P=64, z=0.001, activation=sincos, seed=1, cost_tau=0.0`

### equity_top1000_d1

- TRAIN-selected: TRAIN=0.868, VAL=0.298, TEST=1.104, VAL+TEST=0.712, FULL=0.826, TRAIN-VAL+TEST=+0.157, TRAIN-TEST=-0.236
- TRAIN-selected spec: `scenario=equity_top1000_d1, source_set=all, projected=True, P=256, z=0.001, activation=sincos, seed=1, cost_tau=25.0`
- VAL+TEST-selected: TRAIN=0.477, VAL=0.824, TEST=1.697, VAL+TEST=1.239, FULL=0.665, TRAIN-VAL+TEST=-0.761, TRAIN-TEST=-1.219
- VAL+TEST-selected spec: `scenario=equity_top1000_d1, source_set=all, projected=False, P=64, z=0.001, activation=sincos, seed=1, cost_tau=0.0`

### kucoin_top100

- TRAIN-selected: TRAIN=1.067, VAL=-2.520, TEST=-0.058, VAL+TEST=-1.152, FULL=-0.247, TRAIN-VAL+TEST=+2.219, TRAIN-TEST=+1.124
- TRAIN-selected spec: `scenario=kucoin_top100, source_set=all, projected=True, P=256, z=0.1, activation=sincos, seed=1, cost_tau=0.0`
- VAL+TEST-selected: TRAIN=1.013, VAL=-2.468, TEST=1.494, VAL+TEST=-0.294, FULL=0.265, TRAIN-VAL+TEST=+1.307, TRAIN-TEST=-0.482
- VAL+TEST-selected spec: `scenario=kucoin_top100, source_set=all, projected=True, P=64, z=0.1, activation=sincos, seed=1, cost_tau=25.0`

## aipt_smoke

- Class: diagnostic constrained/cost smoke
- Status: completed_or_smoke

### kucoin_top100

- TRAIN-selected: TRAIN=-2.858, VAL=-2.393, TEST=-0.902, VAL+TEST=-1.359, FULL=-1.852, TRAIN-VAL+TEST=-1.499, TRAIN-TEST=-1.956
- TRAIN-selected spec: `scenario=kucoin_top100, source_set=all, projected=False, P=16, z=0.01, activation=sincos, seed=1, cost_tau=0.0`
- VAL+TEST-selected: TRAIN=-2.858, VAL=-2.393, TEST=-0.902, VAL+TEST=-1.359, FULL=-1.852, TRAIN-VAL+TEST=-1.499, TRAIN-TEST=-1.956
- VAL+TEST-selected spec: `scenario=kucoin_top100, source_set=all, projected=False, P=16, z=0.01, activation=sincos, seed=1, cost_tau=0.0`

## aipt_smoke_eq

- Class: diagnostic constrained/cost smoke
- Status: completed_or_smoke

### equity_smallcap_d0

- TRAIN-selected: TRAIN=0.879, VAL=0.578, TEST=1.673, VAL+TEST=1.110, FULL=0.950, TRAIN-VAL+TEST=-0.231, TRAIN-TEST=-0.794
- TRAIN-selected spec: `scenario=equity_smallcap_d0, source_set=all, projected=False, P=16, z=0.01, activation=sincos, seed=1, cost_tau=0.0`
- VAL+TEST-selected: TRAIN=0.879, VAL=0.578, TEST=1.673, VAL+TEST=1.110, FULL=0.950, TRAIN-VAL+TEST=-0.231, TRAIN-TEST=-0.794
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, source_set=all, projected=False, P=16, z=0.01, activation=sincos, seed=1, cost_tau=0.0`

## aipt_smoke_p1024

- Class: diagnostic constrained/cost smoke
- Status: completed_or_smoke

### equity_smallcap_d0

- TRAIN-selected: TRAIN=2.443, VAL=2.015, TEST=1.459, VAL+TEST=1.661, FULL=2.194, TRAIN-VAL+TEST=+0.781, TRAIN-TEST=+0.984
- TRAIN-selected spec: `scenario=equity_smallcap_d0, source_set=all, projected=False, P=1024, z=0.01, activation=sincos, seed=1, cost_tau=0.0`
- VAL+TEST-selected: TRAIN=2.443, VAL=2.015, TEST=1.459, VAL+TEST=1.661, FULL=2.194, TRAIN-VAL+TEST=+0.781, TRAIN-TEST=+0.984
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, source_set=all, projected=False, P=1024, z=0.01, activation=sincos, seed=1, cost_tau=0.0`

## aipt_smoke_p256

- Class: diagnostic constrained/cost smoke
- Status: completed_or_smoke

### equity_smallcap_d0

- TRAIN-selected: TRAIN=1.836, VAL=0.822, TEST=1.252, VAL+TEST=1.026, FULL=1.541, TRAIN-VAL+TEST=+0.811, TRAIN-TEST=+0.584
- TRAIN-selected spec: `scenario=equity_smallcap_d0, source_set=all, projected=False, P=256, z=0.01, activation=sincos, seed=1, cost_tau=25.0`
- VAL+TEST-selected: TRAIN=1.836, VAL=0.822, TEST=1.252, VAL+TEST=1.026, FULL=1.541, TRAIN-VAL+TEST=+0.811, TRAIN-TEST=+0.584
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, source_set=all, projected=False, P=256, z=0.01, activation=sincos, seed=1, cost_tau=25.0`

## aipt_stepwise_smoke

- Class: stepwise constraints/cost smoke
- Status: completed_or_smoke

### equity_smallcap_d0 / gross1_cap_fee

- TRAIN-selected: TRAIN=2.898, VAL=3.047, TEST=1.345, VAL+TEST=2.170, FULL=2.675, TRAIN-VAL+TEST=+0.728, TRAIN-TEST=+1.553
- TRAIN-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=64, z=0.001, activation=sincos, seed=1, cost_tau=0.0, layer=gross1_cap_fee, max_weight=0.02`
- VAL+TEST-selected: TRAIN=2.898, VAL=3.047, TEST=1.345, VAL+TEST=2.170, FULL=2.675, TRAIN-VAL+TEST=+0.728, TRAIN-TEST=+1.553
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=64, z=0.001, activation=sincos, seed=1, cost_tau=0.0, layer=gross1_cap_fee, max_weight=0.02`

### equity_smallcap_d0 / raw_sdf

- TRAIN-selected: TRAIN=4.909, VAL=5.151, TEST=3.225, VAL+TEST=4.203, FULL=4.715, TRAIN-VAL+TEST=+0.707, TRAIN-TEST=+1.684
- TRAIN-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=64, z=0.001, activation=sincos, seed=1, cost_tau=0.0, layer=raw_sdf, max_weight=0.02`
- VAL+TEST-selected: TRAIN=4.909, VAL=5.151, TEST=3.225, VAL+TEST=4.203, FULL=4.715, TRAIN-VAL+TEST=+0.707, TRAIN-TEST=+1.684
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=64, z=0.001, activation=sincos, seed=1, cost_tau=0.0, layer=raw_sdf, max_weight=0.02`

## aipt_stepwise_strict_kucoin_kernel_qp_turnover_p1024_pilot

- Class: strict cost-kernel plus project-native QP execution pilot
- Status: stopped_or_completed

### kucoin_top100 / kernel_qp_gross1_cap_fee

- TRAIN-selected: TRAIN=0.713, VAL=-3.850, TEST=2.853, VAL+TEST=-0.178, FULL=0.046, TRAIN-VAL+TEST=+0.892, TRAIN-TEST=-2.140
- TRAIN-selected spec: `scenario=kucoin_top100, source_set=all, P=1024, z=1e-05, activation=sincos, seed=2, cost_tau=10.0, turnover_cap=0.1, blend=1.0, qp_alpha_scale=10.0, qp_risk_lambda=0.1, layer=kernel_qp_gross1_cap_fee, max_weight=0.1`
- VAL+TEST-selected: TRAIN=-2.211, VAL=2.147, TEST=3.183, VAL+TEST=2.655, FULL=0.286, TRAIN-VAL+TEST=-4.866, TRAIN-TEST=-5.394
- VAL+TEST-selected spec: `scenario=kucoin_top100, source_set=all, P=1024, z=1e-05, activation=sincos, seed=3, cost_tau=10.0, turnover_cap=0.05, blend=1.0, qp_alpha_scale=10.0, qp_risk_lambda=0.1, layer=kernel_qp_gross1_cap_fee, max_weight=0.1`

## aipt_stepwise_strict_kucoin_p1024

- Class: strict stepwise execution-cost sweep
- Status: stopped_or_completed

### kucoin_top100 / gross1

- TRAIN-selected: TRAIN=1.827, VAL=-2.859, TEST=5.575, VAL+TEST=2.162, FULL=2.049, TRAIN-VAL+TEST=-0.335, TRAIN-TEST=-3.749
- TRAIN-selected spec: `scenario=kucoin_top100, source_set=all, P=1024, z=1e-05, activation=sincos, seed=2, cost_tau=0.0, layer=gross1, max_weight=0.1`
- VAL+TEST-selected: TRAIN=0.780, VAL=2.762, TEST=6.245, VAL+TEST=4.318, FULL=2.793, TRAIN-VAL+TEST=-3.538, TRAIN-TEST=-5.465
- VAL+TEST-selected spec: `scenario=kucoin_top100, source_set=all, P=1024, z=1e-05, activation=sincos, seed=1, cost_tau=0.0, layer=gross1, max_weight=0.1`

### kucoin_top100 / gross1_cap

- TRAIN-selected: TRAIN=1.829, VAL=-2.858, TEST=5.569, VAL+TEST=2.158, FULL=2.048, TRAIN-VAL+TEST=-0.329, TRAIN-TEST=-3.740
- TRAIN-selected spec: `scenario=kucoin_top100, source_set=all, P=1024, z=1e-05, activation=sincos, seed=2, cost_tau=0.0, layer=gross1_cap, max_weight=0.1`
- VAL+TEST-selected: TRAIN=0.775, VAL=2.762, TEST=6.245, VAL+TEST=4.318, FULL=2.791, TRAIN-VAL+TEST=-3.543, TRAIN-TEST=-5.470
- VAL+TEST-selected spec: `scenario=kucoin_top100, source_set=all, P=1024, z=1e-05, activation=sincos, seed=1, cost_tau=0.0, layer=gross1_cap, max_weight=0.1`

### kucoin_top100 / gross1_cap_fee

- TRAIN-selected: TRAIN=-0.886, VAL=-8.531, TEST=1.816, VAL+TEST=-2.268, FULL=-1.601, TRAIN-VAL+TEST=+1.382, TRAIN-TEST=-2.701
- TRAIN-selected spec: `scenario=kucoin_top100, source_set=all, P=1024, z=1e-05, activation=sincos, seed=2, cost_tau=0.0, layer=gross1_cap_fee, max_weight=0.1`
- VAL+TEST-selected: TRAIN=-2.199, VAL=-1.688, TEST=1.922, VAL+TEST=-0.063, FULL=-0.947, TRAIN-VAL+TEST=-2.136, TRAIN-TEST=-4.121
- VAL+TEST-selected spec: `scenario=kucoin_top100, source_set=all, P=1024, z=1e-05, activation=sincos, seed=1, cost_tau=0.0, layer=gross1_cap_fee, max_weight=0.1`

### kucoin_top100 / gross1_fee

- TRAIN-selected: TRAIN=-0.888, VAL=-8.531, TEST=1.821, VAL+TEST=-2.265, FULL=-1.600, TRAIN-VAL+TEST=+1.377, TRAIN-TEST=-2.709
- TRAIN-selected spec: `scenario=kucoin_top100, source_set=all, P=1024, z=1e-05, activation=sincos, seed=2, cost_tau=0.0, layer=gross1_fee, max_weight=0.1`
- VAL+TEST-selected: TRAIN=-2.195, VAL=-1.688, TEST=1.922, VAL+TEST=-0.063, FULL=-0.945, TRAIN-VAL+TEST=-2.131, TRAIN-TEST=-4.117
- VAL+TEST-selected spec: `scenario=kucoin_top100, source_set=all, P=1024, z=1e-05, activation=sincos, seed=1, cost_tau=0.0, layer=gross1_fee, max_weight=0.1`

### kucoin_top100 / kernel_gross1_cap_fee

- TRAIN-selected: TRAIN=-1.251, VAL=-6.198, TEST=2.053, VAL+TEST=-1.468, FULL=-1.247, TRAIN-VAL+TEST=+0.216, TRAIN-TEST=-3.304
- TRAIN-selected spec: `scenario=kucoin_top100, source_set=all, P=1024, z=0.0001, activation=sincos, seed=2, cost_tau=0.1, layer=kernel_gross1_cap_fee, max_weight=0.1`
- VAL+TEST-selected: TRAIN=-2.647, VAL=-7.188, TEST=4.056, VAL+TEST=-0.204, FULL=-1.129, TRAIN-VAL+TEST=-2.443, TRAIN-TEST=-6.704
- VAL+TEST-selected spec: `scenario=kucoin_top100, source_set=all, P=1024, z=1e-05, activation=sincos, seed=2, cost_tau=10.0, layer=kernel_gross1_cap_fee, max_weight=0.1`

### kucoin_top100 / raw_sdf

- TRAIN-selected: TRAIN=1.983, VAL=-2.839, TEST=5.378, VAL+TEST=1.832, FULL=1.916, TRAIN-VAL+TEST=+0.151, TRAIN-TEST=-3.395
- TRAIN-selected spec: `scenario=kucoin_top100, source_set=all, P=1024, z=1e-05, activation=sincos, seed=2, cost_tau=0.0, layer=raw_sdf, max_weight=0.1`
- VAL+TEST-selected: TRAIN=0.725, VAL=2.837, TEST=6.471, VAL+TEST=4.294, FULL=2.876, TRAIN-VAL+TEST=-3.569, TRAIN-TEST=-5.745
- VAL+TEST-selected spec: `scenario=kucoin_top100, source_set=all, P=1024, z=1e-05, activation=sincos, seed=1, cost_tau=0.0, layer=raw_sdf, max_weight=0.1`

## aipt_stepwise_strict_kucoin_pilot

- Class: strict stepwise execution-cost pilot
- Status: stopped_or_completed

### kucoin_top100 / gross1

- TRAIN-selected: TRAIN=2.098, VAL=0.292, TEST=-3.709, VAL+TEST=-1.965, FULL=-0.372, TRAIN-VAL+TEST=+4.063, TRAIN-TEST=+5.807
- TRAIN-selected spec: `scenario=kucoin_top100, source_set=all, P=64, z=0.001, activation=sincos, seed=2, cost_tau=0.0, layer=gross1, max_weight=0.1`
- VAL+TEST-selected: TRAIN=1.065, VAL=2.481, TEST=2.514, VAL+TEST=2.359, FULL=1.931, TRAIN-VAL+TEST=-1.294, TRAIN-TEST=-1.450
- VAL+TEST-selected spec: `scenario=kucoin_top100, source_set=all, P=64, z=0.1, activation=sincos, seed=1, cost_tau=0.0, layer=gross1, max_weight=0.1`

### kucoin_top100 / gross1_cap

- TRAIN-selected: TRAIN=2.098, VAL=0.292, TEST=-3.709, VAL+TEST=-1.965, FULL=-0.372, TRAIN-VAL+TEST=+4.063, TRAIN-TEST=+5.807
- TRAIN-selected spec: `scenario=kucoin_top100, source_set=all, P=64, z=0.001, activation=sincos, seed=2, cost_tau=0.0, layer=gross1_cap, max_weight=0.1`
- VAL+TEST-selected: TRAIN=1.065, VAL=2.481, TEST=2.514, VAL+TEST=2.359, FULL=1.931, TRAIN-VAL+TEST=-1.294, TRAIN-TEST=-1.450
- VAL+TEST-selected spec: `scenario=kucoin_top100, source_set=all, P=64, z=0.1, activation=sincos, seed=1, cost_tau=0.0, layer=gross1_cap, max_weight=0.1`

### kucoin_top100 / gross1_cap_fee

- TRAIN-selected: TRAIN=0.881, VAL=2.332, TEST=0.472, VAL+TEST=1.408, FULL=1.327, TRAIN-VAL+TEST=-0.528, TRAIN-TEST=+0.408
- TRAIN-selected spec: `scenario=kucoin_top100, source_set=all, P=64, z=0.1, activation=sincos, seed=2, cost_tau=0.0, layer=gross1_cap_fee, max_weight=0.1`
- VAL+TEST-selected: TRAIN=0.759, VAL=1.945, TEST=2.075, VAL+TEST=1.873, FULL=1.532, TRAIN-VAL+TEST=-1.114, TRAIN-TEST=-1.317
- VAL+TEST-selected spec: `scenario=kucoin_top100, source_set=all, P=64, z=0.1, activation=sincos, seed=1, cost_tau=0.0, layer=gross1_cap_fee, max_weight=0.1`

### kucoin_top100 / gross1_fee

- TRAIN-selected: TRAIN=0.881, VAL=2.332, TEST=0.472, VAL+TEST=1.408, FULL=1.327, TRAIN-VAL+TEST=-0.528, TRAIN-TEST=+0.408
- TRAIN-selected spec: `scenario=kucoin_top100, source_set=all, P=64, z=0.1, activation=sincos, seed=2, cost_tau=0.0, layer=gross1_fee, max_weight=0.1`
- VAL+TEST-selected: TRAIN=0.759, VAL=1.945, TEST=2.075, VAL+TEST=1.873, FULL=1.532, TRAIN-VAL+TEST=-1.114, TRAIN-TEST=-1.317
- VAL+TEST-selected spec: `scenario=kucoin_top100, source_set=all, P=64, z=0.1, activation=sincos, seed=1, cost_tau=0.0, layer=gross1_fee, max_weight=0.1`

### kucoin_top100 / kernel_gross1_cap_fee

- TRAIN-selected: TRAIN=0.886, VAL=2.295, TEST=0.430, VAL+TEST=1.381, FULL=1.315, TRAIN-VAL+TEST=-0.495, TRAIN-TEST=+0.456
- TRAIN-selected spec: `scenario=kucoin_top100, source_set=all, P=64, z=0.1, activation=sincos, seed=2, cost_tau=10.0, layer=kernel_gross1_cap_fee, max_weight=0.1`
- VAL+TEST-selected: TRAIN=0.759, VAL=1.946, TEST=2.069, VAL+TEST=1.870, FULL=1.531, TRAIN-VAL+TEST=-1.112, TRAIN-TEST=-1.311
- VAL+TEST-selected spec: `scenario=kucoin_top100, source_set=all, P=64, z=0.1, activation=sincos, seed=1, cost_tau=0.1, layer=kernel_gross1_cap_fee, max_weight=0.1`

### kucoin_top100 / raw_sdf

- TRAIN-selected: TRAIN=1.724, VAL=0.135, TEST=-3.687, VAL+TEST=-1.776, FULL=-0.204, TRAIN-VAL+TEST=+3.500, TRAIN-TEST=+5.411
- TRAIN-selected spec: `scenario=kucoin_top100, source_set=all, P=64, z=0.001, activation=sincos, seed=2, cost_tau=0.0, layer=raw_sdf, max_weight=0.1`
- VAL+TEST-selected: TRAIN=-0.502, VAL=1.896, TEST=1.550, VAL+TEST=1.517, FULL=0.938, TRAIN-VAL+TEST=-2.019, TRAIN-TEST=-2.051
- VAL+TEST-selected spec: `scenario=kucoin_top100, source_set=all, P=256, z=0.001, activation=sincos, seed=1, cost_tau=0.0, layer=raw_sdf, max_weight=0.1`

## aipt_stepwise_strict_kucoin_qp_p1024

- Class: strict project-native QP execution-cost sweep
- Status: stopped_or_completed

### kucoin_top100 / qp_gross1_cap_fee

- TRAIN-selected: TRAIN=-1.402, VAL=-8.223, TEST=1.878, VAL+TEST=-2.485, FULL=-1.955, TRAIN-VAL+TEST=+1.082, TRAIN-TEST=-3.280
- TRAIN-selected spec: `scenario=kucoin_top100, source_set=all, P=1024, z=1e-05, activation=sincos, seed=2, cost_tau=0.1, turnover_cap=0.0, blend=1.0, qp_alpha_scale=10.0, qp_risk_lambda=0.1, layer=qp_gross1_cap_fee, max_weight=0.1`
- VAL+TEST-selected: TRAIN=-2.074, VAL=-0.529, TEST=3.878, VAL+TEST=1.499, FULL=-0.036, TRAIN-VAL+TEST=-3.572, TRAIN-TEST=-5.952
- VAL+TEST-selected spec: `scenario=kucoin_top100, source_set=all, P=1024, z=1e-05, activation=sincos, seed=1, cost_tau=10.0, turnover_cap=0.0, blend=1.0, qp_alpha_scale=100.0, qp_risk_lambda=0.1, layer=qp_gross1_cap_fee, max_weight=0.1`

## aipt_stepwise_strict_kucoin_qp_turnover_p1024

- Class: strict project-native QP plus turnover-control execution-cost sweep
- Status: stopped_or_completed

### kucoin_top100 / qp_gross1_cap_fee

- TRAIN-selected: TRAIN=1.321, VAL=-3.902, TEST=2.290, VAL+TEST=-0.588, FULL=0.247, TRAIN-VAL+TEST=+1.909, TRAIN-TEST=-0.969
- TRAIN-selected spec: `scenario=kucoin_top100, source_set=all, P=1024, z=1e-05, activation=sincos, seed=2, cost_tau=10.0, turnover_cap=0.1, blend=1.0, qp_alpha_scale=1.0, qp_risk_lambda=0.1, layer=qp_gross1_cap_fee, max_weight=0.1`
- VAL+TEST-selected: TRAIN=-0.422, VAL=0.296, TEST=4.949, VAL+TEST=2.937, FULL=1.329, TRAIN-VAL+TEST=-3.359, TRAIN-TEST=-5.371
- VAL+TEST-selected spec: `scenario=kucoin_top100, source_set=all, P=1024, z=1e-05, activation=sincos, seed=1, cost_tau=1000.0, turnover_cap=0.25, blend=1.0, qp_alpha_scale=10.0, qp_risk_lambda=0.1, layer=qp_gross1_cap_fee, max_weight=0.1`

## aipt_stepwise_strict_kucoin_turnover_p1024

- Class: strict stepwise turnover-control execution-cost sweep
- Status: stopped_or_completed

### kucoin_top100 / gross1_cap_fee

- TRAIN-selected: TRAIN=1.989, VAL=-4.739, TEST=3.680, VAL+TEST=-0.053, FULL=0.818, TRAIN-VAL+TEST=+2.042, TRAIN-TEST=-1.691
- TRAIN-selected spec: `scenario=kucoin_top100, source_set=all, P=1024, z=1e-05, activation=sincos, seed=2, cost_tau=0.0, turnover_cap=0.1, blend=0.25, layer=gross1_cap_fee, max_weight=0.1`
- VAL+TEST-selected: TRAIN=-0.561, VAL=-0.844, TEST=6.067, VAL+TEST=2.992, FULL=1.306, TRAIN-VAL+TEST=-3.553, TRAIN-TEST=-6.628
- VAL+TEST-selected spec: `scenario=kucoin_top100, source_set=all, P=1024, z=1e-05, activation=sincos, seed=1, cost_tau=0.0, turnover_cap=0.25, blend=0.25, layer=gross1_cap_fee, max_weight=0.1`

### kucoin_top100 / kernel_gross1_cap_fee

- TRAIN-selected: TRAIN=1.811, VAL=-3.383, TEST=1.589, VAL+TEST=-0.577, FULL=0.376, TRAIN-VAL+TEST=+2.389, TRAIN-TEST=+0.223
- TRAIN-selected spec: `scenario=kucoin_top100, source_set=all, P=1024, z=1e-05, activation=sincos, seed=2, cost_tau=0.1, turnover_cap=0.05, blend=0.5, layer=kernel_gross1_cap_fee, max_weight=0.1`
- VAL+TEST-selected: TRAIN=-2.275, VAL=1.861, TEST=3.296, VAL+TEST=2.537, FULL=0.295, TRAIN-VAL+TEST=-4.812, TRAIN-TEST=-5.572
- VAL+TEST-selected spec: `scenario=kucoin_top100, source_set=all, P=1024, z=1e-05, activation=sincos, seed=3, cost_tau=0.1, turnover_cap=0.05, blend=0.25, layer=kernel_gross1_cap_fee, max_weight=0.1`

## aipt_stepwise_strict_smallcap_d0_p1024

- Class: strict stepwise execution-cost sweep
- Status: running

### equity_smallcap_d0 / gross1

- TRAIN-selected: TRAIN=9.015, VAL=8.211, TEST=6.771, VAL+TEST=7.481, FULL=8.561, TRAIN-VAL+TEST=+1.533, TRAIN-TEST=+2.244
- TRAIN-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=1024, z=0.001, activation=sincos, seed=2, cost_tau=0.0, layer=gross1, max_weight=0.02`
- VAL+TEST-selected: TRAIN=8.928, VAL=9.132, TEST=7.642, VAL+TEST=8.339, FULL=8.766, TRAIN-VAL+TEST=+0.589, TRAIN-TEST=+1.286
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=1024, z=0.001, activation=sincos, seed=1, cost_tau=0.0, layer=gross1, max_weight=0.02`

### equity_smallcap_d0 / gross1_cap

- TRAIN-selected: TRAIN=9.015, VAL=8.198, TEST=6.732, VAL+TEST=7.464, FULL=8.559, TRAIN-VAL+TEST=+1.551, TRAIN-TEST=+2.283
- TRAIN-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=1024, z=0.001, activation=sincos, seed=2, cost_tau=0.0, layer=gross1_cap, max_weight=0.02`
- VAL+TEST-selected: TRAIN=8.928, VAL=9.121, TEST=7.564, VAL+TEST=8.306, FULL=8.757, TRAIN-VAL+TEST=+0.621, TRAIN-TEST=+1.364
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=1024, z=0.001, activation=sincos, seed=1, cost_tau=0.0, layer=gross1_cap, max_weight=0.02`

### equity_smallcap_d0 / gross1_cap_fee

- TRAIN-selected: TRAIN=5.970, VAL=4.778, TEST=4.293, VAL+TEST=4.523, FULL=5.538, TRAIN-VAL+TEST=+1.447, TRAIN-TEST=+1.677
- TRAIN-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=1024, z=0.001, activation=sincos, seed=2, cost_tau=0.0, layer=gross1_cap_fee, max_weight=0.02`
- VAL+TEST-selected: TRAIN=5.801, VAL=4.876, TEST=4.947, VAL+TEST=4.877, FULL=5.544, TRAIN-VAL+TEST=+0.924, TRAIN-TEST=+0.854
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=1024, z=0.001, activation=sincos, seed=1, cost_tau=0.0, layer=gross1_cap_fee, max_weight=0.02`

### equity_smallcap_d0 / gross1_fee

- TRAIN-selected: TRAIN=5.970, VAL=4.789, TEST=4.345, VAL+TEST=4.549, FULL=5.543, TRAIN-VAL+TEST=+1.421, TRAIN-TEST=+1.625
- TRAIN-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=1024, z=0.001, activation=sincos, seed=2, cost_tau=0.0, layer=gross1_fee, max_weight=0.02`
- VAL+TEST-selected: TRAIN=5.801, VAL=4.886, TEST=5.042, VAL+TEST=4.926, FULL=5.556, TRAIN-VAL+TEST=+0.875, TRAIN-TEST=+0.760
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=1024, z=0.001, activation=sincos, seed=1, cost_tau=0.0, layer=gross1_fee, max_weight=0.02`

### equity_smallcap_d0 / kernel_gross1_cap_fee

- TRAIN-selected: TRAIN=6.032, VAL=4.860, TEST=4.316, VAL+TEST=4.582, FULL=5.600, TRAIN-VAL+TEST=+1.449, TRAIN-TEST=+1.715
- TRAIN-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=1024, z=0.001, activation=sincos, seed=2, cost_tau=0.1, layer=kernel_gross1_cap_fee, max_weight=0.02`
- VAL+TEST-selected: TRAIN=5.835, VAL=4.992, TEST=5.121, VAL+TEST=5.026, FULL=5.611, TRAIN-VAL+TEST=+0.810, TRAIN-TEST=+0.715
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=1024, z=0.001, activation=sincos, seed=1, cost_tau=0.1, layer=kernel_gross1_cap_fee, max_weight=0.02`

### equity_smallcap_d0 / raw_sdf

- TRAIN-selected: TRAIN=8.918, VAL=8.233, TEST=6.924, VAL+TEST=7.615, FULL=8.557, TRAIN-VAL+TEST=+1.302, TRAIN-TEST=+1.994
- TRAIN-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=1024, z=0.001, activation=sincos, seed=2, cost_tau=0.0, layer=raw_sdf, max_weight=0.02`
- VAL+TEST-selected: TRAIN=8.824, VAL=9.100, TEST=7.765, VAL+TEST=8.461, FULL=8.690, TRAIN-VAL+TEST=+0.364, TRAIN-TEST=+1.059
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=1024, z=0.001, activation=sincos, seed=1, cost_tau=0.0, layer=raw_sdf, max_weight=0.02`

## aipt_stepwise_strict_smallcap_d0_qp_p1024_pilot

- Class: strict project-native QP execution-cost pilot
- Status: stopped_or_completed

### equity_smallcap_d0 / qp_gross1_cap_fee

- TRAIN-selected: TRAIN=4.719, VAL=4.214, TEST=4.918, VAL+TEST=4.528, FULL=4.655, TRAIN-VAL+TEST=+0.192, TRAIN-TEST=-0.199
- TRAIN-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=1024, z=0.001, activation=sincos, seed=1, cost_tau=10.0, turnover_cap=0.0, blend=1.0, qp_alpha_scale=10.0, qp_risk_lambda=1.0, layer=qp_gross1_cap_fee, max_weight=0.02`
- VAL+TEST-selected: TRAIN=4.713, VAL=4.236, TEST=4.901, VAL+TEST=4.530, FULL=4.651, TRAIN-VAL+TEST=+0.183, TRAIN-TEST=-0.188
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=1024, z=0.001, activation=sincos, seed=1, cost_tau=10.0, turnover_cap=0.0, blend=1.0, qp_alpha_scale=10.0, qp_risk_lambda=5.0, layer=qp_gross1_cap_fee, max_weight=0.02`

## aipt_stepwise_strict_smallcap_d0_turnover_p1024

- Class: strict turnover-control execution-cost sweep
- Status: stopped_or_completed

### equity_smallcap_d0 / gross1_cap_fee

- TRAIN-selected: TRAIN=5.898, VAL=5.093, TEST=4.091, VAL+TEST=4.597, FULL=5.505, TRAIN-VAL+TEST=+1.301, TRAIN-TEST=+1.807
- TRAIN-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=1024, z=0.001, activation=sincos, seed=2, cost_tau=0.0, turnover_cap=0.75, blend=1.0, qp_alpha_scale=1.0, qp_risk_lambda=0.0, layer=gross1_cap_fee, max_weight=0.02`
- VAL+TEST-selected: TRAIN=5.758, VAL=5.098, TEST=5.014, VAL+TEST=5.038, FULL=5.551, TRAIN-VAL+TEST=+0.720, TRAIN-TEST=+0.744
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=1024, z=0.001, activation=sincos, seed=1, cost_tau=0.0, turnover_cap=0.75, blend=1.0, qp_alpha_scale=1.0, qp_risk_lambda=0.0, layer=gross1_cap_fee, max_weight=0.02`

### equity_smallcap_d0 / kernel_gross1_cap_fee

- TRAIN-selected: TRAIN=5.954, VAL=5.191, TEST=4.119, VAL+TEST=4.669, FULL=5.566, TRAIN-VAL+TEST=+1.285, TRAIN-TEST=+1.835
- TRAIN-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=1024, z=0.001, activation=sincos, seed=2, cost_tau=0.1, turnover_cap=0.75, blend=1.0, qp_alpha_scale=1.0, qp_risk_lambda=0.0, layer=kernel_gross1_cap_fee, max_weight=0.02`
- VAL+TEST-selected: TRAIN=5.794, VAL=5.204, TEST=5.188, VAL+TEST=5.180, FULL=5.619, TRAIN-VAL+TEST=+0.613, TRAIN-TEST=+0.606
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=1024, z=0.001, activation=sincos, seed=1, cost_tau=0.1, turnover_cap=0.75, blend=1.0, qp_alpha_scale=1.0, qp_risk_lambda=0.0, layer=kernel_gross1_cap_fee, max_weight=0.02`

## aipt_stepwise_strict_smallcap_d1_p1024

- Class: strict stepwise execution-cost sweep
- Status: stopped_or_completed

### equity_smallcap_d1 / gross1

- TRAIN-selected: TRAIN=4.682, VAL=4.918, TEST=4.274, VAL+TEST=4.601, FULL=4.636, TRAIN-VAL+TEST=+0.081, TRAIN-TEST=+0.409
- TRAIN-selected spec: `scenario=equity_smallcap_d1, source_set=all, P=1024, z=0.01, activation=sincos, seed=2, cost_tau=0.0, layer=gross1, max_weight=0.02`
- VAL+TEST-selected: TRAIN=4.642, VAL=4.712, TEST=4.989, VAL+TEST=4.889, FULL=4.703, TRAIN-VAL+TEST=-0.247, TRAIN-TEST=-0.348
- VAL+TEST-selected spec: `scenario=equity_smallcap_d1, source_set=all, P=1024, z=0.01, activation=sincos, seed=3, cost_tau=0.0, layer=gross1, max_weight=0.02`

### equity_smallcap_d1 / gross1_cap

- TRAIN-selected: TRAIN=4.683, VAL=4.908, TEST=4.288, VAL+TEST=4.607, FULL=4.639, TRAIN-VAL+TEST=+0.076, TRAIN-TEST=+0.394
- TRAIN-selected spec: `scenario=equity_smallcap_d1, source_set=all, P=1024, z=0.01, activation=sincos, seed=2, cost_tau=0.0, layer=gross1_cap, max_weight=0.02`
- VAL+TEST-selected: TRAIN=4.641, VAL=4.705, TEST=5.000, VAL+TEST=4.891, FULL=4.704, TRAIN-VAL+TEST=-0.249, TRAIN-TEST=-0.359
- VAL+TEST-selected spec: `scenario=equity_smallcap_d1, source_set=all, P=1024, z=0.01, activation=sincos, seed=3, cost_tau=0.0, layer=gross1_cap, max_weight=0.02`

### equity_smallcap_d1 / gross1_cap_fee

- TRAIN-selected: TRAIN=2.124, VAL=2.438, TEST=2.168, VAL+TEST=2.320, FULL=2.177, TRAIN-VAL+TEST=-0.196, TRAIN-TEST=-0.044
- TRAIN-selected spec: `scenario=equity_smallcap_d1, source_set=all, P=1024, z=0.01, activation=sincos, seed=2, cost_tau=0.0, layer=gross1_cap_fee, max_weight=0.02`
- VAL+TEST-selected: TRAIN=2.016, VAL=2.546, TEST=2.841, VAL+TEST=2.701, FULL=2.224, TRAIN-VAL+TEST=-0.685, TRAIN-TEST=-0.825
- VAL+TEST-selected spec: `scenario=equity_smallcap_d1, source_set=all, P=1024, z=0.01, activation=sincos, seed=1, cost_tau=0.0, layer=gross1_cap_fee, max_weight=0.02`

### equity_smallcap_d1 / gross1_fee

- TRAIN-selected: TRAIN=2.124, VAL=2.449, TEST=2.155, VAL+TEST=2.317, FULL=2.176, TRAIN-VAL+TEST=-0.194, TRAIN-TEST=-0.031
- TRAIN-selected spec: `scenario=equity_smallcap_d1, source_set=all, P=1024, z=0.01, activation=sincos, seed=2, cost_tau=0.0, layer=gross1_fee, max_weight=0.02`
- VAL+TEST-selected: TRAIN=2.016, VAL=2.560, TEST=2.850, VAL+TEST=2.711, FULL=2.228, TRAIN-VAL+TEST=-0.695, TRAIN-TEST=-0.834
- VAL+TEST-selected spec: `scenario=equity_smallcap_d1, source_set=all, P=1024, z=0.01, activation=sincos, seed=1, cost_tau=0.0, layer=gross1_fee, max_weight=0.02`

### equity_smallcap_d1 / kernel_gross1_cap_fee

- TRAIN-selected: TRAIN=2.138, VAL=2.474, TEST=2.187, VAL+TEST=2.347, FULL=2.195, TRAIN-VAL+TEST=-0.210, TRAIN-TEST=-0.049
- TRAIN-selected spec: `scenario=equity_smallcap_d1, source_set=all, P=1024, z=0.01, activation=sincos, seed=2, cost_tau=0.1, layer=kernel_gross1_cap_fee, max_weight=0.02`
- VAL+TEST-selected: TRAIN=2.044, VAL=2.460, TEST=3.101, VAL+TEST=2.783, FULL=2.267, TRAIN-VAL+TEST=-0.739, TRAIN-TEST=-1.058
- VAL+TEST-selected spec: `scenario=equity_smallcap_d1, source_set=all, P=1024, z=0.01, activation=sincos, seed=1, cost_tau=1.0, layer=kernel_gross1_cap_fee, max_weight=0.02`

### equity_smallcap_d1 / raw_sdf

- TRAIN-selected: TRAIN=4.797, VAL=4.918, TEST=4.068, VAL+TEST=4.507, FULL=4.709, TRAIN-VAL+TEST=+0.289, TRAIN-TEST=+0.728
- TRAIN-selected spec: `scenario=equity_smallcap_d1, source_set=all, P=1024, z=0.01, activation=sincos, seed=2, cost_tau=0.0, layer=raw_sdf, max_weight=0.02`
- VAL+TEST-selected: TRAIN=4.757, VAL=4.834, TEST=4.795, VAL+TEST=4.868, FULL=4.767, TRAIN-VAL+TEST=-0.111, TRAIN-TEST=-0.038
- VAL+TEST-selected spec: `scenario=equity_smallcap_d1, source_set=all, P=1024, z=0.01, activation=sincos, seed=3, cost_tau=0.0, layer=raw_sdf, max_weight=0.02`

## aipt_stepwise_strict_smallcap_d1_turnover_p1024

- Class: strict turnover-control execution-cost sweep
- Status: stopped_or_completed

### equity_smallcap_d1 / gross1_cap_fee

- TRAIN-selected: TRAIN=2.279, VAL=2.725, TEST=2.201, VAL+TEST=2.531, FULL=2.352, TRAIN-VAL+TEST=-0.252, TRAIN-TEST=+0.078
- TRAIN-selected spec: `scenario=equity_smallcap_d1, source_set=all, P=1024, z=0.01, activation=sincos, seed=2, cost_tau=0.0, turnover_cap=0.5, blend=1.0, qp_alpha_scale=1.0, qp_risk_lambda=0.0, layer=gross1_cap_fee, max_weight=0.02`
- VAL+TEST-selected: TRAIN=2.260, VAL=2.625, TEST=3.064, VAL+TEST=2.901, FULL=2.449, TRAIN-VAL+TEST=-0.642, TRAIN-TEST=-0.804
- VAL+TEST-selected spec: `scenario=equity_smallcap_d1, source_set=all, P=1024, z=0.01, activation=sincos, seed=3, cost_tau=0.0, turnover_cap=0.5, blend=1.0, qp_alpha_scale=1.0, qp_risk_lambda=0.0, layer=gross1_cap_fee, max_weight=0.02`

### equity_smallcap_d1 / kernel_gross1_cap_fee

- TRAIN-selected: TRAIN=2.287, VAL=2.762, TEST=2.215, VAL+TEST=2.556, FULL=2.365, TRAIN-VAL+TEST=-0.269, TRAIN-TEST=+0.072
- TRAIN-selected spec: `scenario=equity_smallcap_d1, source_set=all, P=1024, z=0.01, activation=sincos, seed=2, cost_tau=0.1, turnover_cap=0.5, blend=1.0, qp_alpha_scale=1.0, qp_risk_lambda=0.0, layer=kernel_gross1_cap_fee, max_weight=0.02`
- VAL+TEST-selected: TRAIN=2.258, VAL=2.831, TEST=3.066, VAL+TEST=2.977, FULL=2.470, TRAIN-VAL+TEST=-0.719, TRAIN-TEST=-0.807
- VAL+TEST-selected spec: `scenario=equity_smallcap_d1, source_set=all, P=1024, z=0.01, activation=sincos, seed=1, cost_tau=1.0, turnover_cap=0.5, blend=1.0, qp_alpha_scale=1.0, qp_risk_lambda=0.0, layer=kernel_gross1_cap_fee, max_weight=0.02`

## aipt_stepwise_strict_top1000_d0_p1024

- Class: strict stepwise execution-cost sweep
- Status: stopped_or_completed

### equity_top1000_d0 / gross1

- TRAIN-selected: TRAIN=0.607, VAL=0.329, TEST=1.854, VAL+TEST=1.096, FULL=0.726, TRAIN-VAL+TEST=-0.489, TRAIN-TEST=-1.247
- TRAIN-selected spec: `scenario=equity_top1000_d0, source_set=all, P=1024, z=0.0001, activation=sincos, seed=1, cost_tau=0.0, layer=gross1, max_weight=0.01`
- VAL+TEST-selected: TRAIN=0.403, VAL=0.077, TEST=2.401, VAL+TEST=1.268, FULL=0.607, TRAIN-VAL+TEST=-0.865, TRAIN-TEST=-1.998
- VAL+TEST-selected spec: `scenario=equity_top1000_d0, source_set=all, P=1024, z=0.0001, activation=sincos, seed=2, cost_tau=0.0, layer=gross1, max_weight=0.01`

### equity_top1000_d0 / gross1_cap

- TRAIN-selected: TRAIN=0.607, VAL=0.329, TEST=1.854, VAL+TEST=1.096, FULL=0.726, TRAIN-VAL+TEST=-0.489, TRAIN-TEST=-1.247
- TRAIN-selected spec: `scenario=equity_top1000_d0, source_set=all, P=1024, z=0.0001, activation=sincos, seed=1, cost_tau=0.0, layer=gross1_cap, max_weight=0.01`
- VAL+TEST-selected: TRAIN=0.403, VAL=0.077, TEST=2.401, VAL+TEST=1.268, FULL=0.607, TRAIN-VAL+TEST=-0.865, TRAIN-TEST=-1.998
- VAL+TEST-selected spec: `scenario=equity_top1000_d0, source_set=all, P=1024, z=0.0001, activation=sincos, seed=2, cost_tau=0.0, layer=gross1_cap, max_weight=0.01`

### equity_top1000_d0 / gross1_cap_fee

- TRAIN-selected: TRAIN=-0.906, VAL=-1.670, TEST=0.226, VAL+TEST=-0.710, FULL=-0.855, TRAIN-VAL+TEST=-0.196, TRAIN-TEST=-1.132
- TRAIN-selected spec: `scenario=equity_top1000_d0, source_set=all, P=1024, z=0.0001, activation=sincos, seed=1, cost_tau=0.0, layer=gross1_cap_fee, max_weight=0.01`
- VAL+TEST-selected: TRAIN=-1.071, VAL=-2.042, TEST=0.765, VAL+TEST=-0.599, FULL=-0.955, TRAIN-VAL+TEST=-0.471, TRAIN-TEST=-1.835
- VAL+TEST-selected spec: `scenario=equity_top1000_d0, source_set=all, P=1024, z=0.0001, activation=sincos, seed=2, cost_tau=0.0, layer=gross1_cap_fee, max_weight=0.01`

### equity_top1000_d0 / gross1_fee

- TRAIN-selected: TRAIN=-0.906, VAL=-1.670, TEST=0.226, VAL+TEST=-0.710, FULL=-0.855, TRAIN-VAL+TEST=-0.196, TRAIN-TEST=-1.132
- TRAIN-selected spec: `scenario=equity_top1000_d0, source_set=all, P=1024, z=0.0001, activation=sincos, seed=1, cost_tau=0.0, layer=gross1_fee, max_weight=0.01`
- VAL+TEST-selected: TRAIN=-1.071, VAL=-2.042, TEST=0.765, VAL+TEST=-0.599, FULL=-0.955, TRAIN-VAL+TEST=-0.471, TRAIN-TEST=-1.835
- VAL+TEST-selected spec: `scenario=equity_top1000_d0, source_set=all, P=1024, z=0.0001, activation=sincos, seed=2, cost_tau=0.0, layer=gross1_fee, max_weight=0.01`

### equity_top1000_d0 / kernel_gross1_cap_fee

- TRAIN-selected: TRAIN=-0.738, VAL=-1.374, TEST=0.560, VAL+TEST=-0.400, FULL=-0.659, TRAIN-VAL+TEST=-0.338, TRAIN-TEST=-1.299
- TRAIN-selected spec: `scenario=equity_top1000_d0, source_set=all, P=1024, z=0.0001, activation=sincos, seed=3, cost_tau=10.0, layer=kernel_gross1_cap_fee, max_weight=0.01`
- VAL+TEST-selected: TRAIN=-0.937, VAL=-0.776, TEST=0.645, VAL+TEST=-0.043, FULL=-0.729, TRAIN-VAL+TEST=-0.894, TRAIN-TEST=-1.581
- VAL+TEST-selected spec: `scenario=equity_top1000_d0, source_set=all, P=1024, z=0.0001, activation=sincos, seed=2, cost_tau=10.0, layer=kernel_gross1_cap_fee, max_weight=0.01`

### equity_top1000_d0 / raw_sdf

- TRAIN-selected: TRAIN=0.614, VAL=0.383, TEST=1.810, VAL+TEST=1.092, FULL=0.731, TRAIN-VAL+TEST=-0.478, TRAIN-TEST=-1.195
- TRAIN-selected spec: `scenario=equity_top1000_d0, source_set=all, P=1024, z=0.0001, activation=sincos, seed=1, cost_tau=0.0, layer=raw_sdf, max_weight=0.01`
- VAL+TEST-selected: TRAIN=0.409, VAL=0.134, TEST=2.455, VAL+TEST=1.302, FULL=0.612, TRAIN-VAL+TEST=-0.893, TRAIN-TEST=-2.046
- VAL+TEST-selected spec: `scenario=equity_top1000_d0, source_set=all, P=1024, z=0.0001, activation=sincos, seed=2, cost_tau=0.0, layer=raw_sdf, max_weight=0.01`

## aipt_stepwise_strict_top1000_d1_p1024

- Class: strict stepwise execution-cost sweep
- Status: stopped_or_completed

### equity_top1000_d1 / gross1

- TRAIN-selected: TRAIN=0.975, VAL=0.961, TEST=1.063, VAL+TEST=0.995, FULL=0.980, TRAIN-VAL+TEST=-0.020, TRAIN-TEST=-0.088
- TRAIN-selected spec: `scenario=equity_top1000_d1, source_set=all, P=1024, z=1e-05, activation=sincos, seed=1, cost_tau=0.0, layer=gross1, max_weight=0.01`
- VAL+TEST-selected: TRAIN=0.512, VAL=1.593, TEST=1.263, VAL+TEST=1.423, FULL=0.758, TRAIN-VAL+TEST=-0.911, TRAIN-TEST=-0.752
- VAL+TEST-selected spec: `scenario=equity_top1000_d1, source_set=all, P=1024, z=1e-05, activation=sincos, seed=3, cost_tau=0.0, layer=gross1, max_weight=0.01`

### equity_top1000_d1 / gross1_cap

- TRAIN-selected: TRAIN=0.975, VAL=0.961, TEST=1.063, VAL+TEST=0.995, FULL=0.980, TRAIN-VAL+TEST=-0.020, TRAIN-TEST=-0.088
- TRAIN-selected spec: `scenario=equity_top1000_d1, source_set=all, P=1024, z=1e-05, activation=sincos, seed=1, cost_tau=0.0, layer=gross1_cap, max_weight=0.01`
- VAL+TEST-selected: TRAIN=0.512, VAL=1.593, TEST=1.263, VAL+TEST=1.423, FULL=0.758, TRAIN-VAL+TEST=-0.911, TRAIN-TEST=-0.752
- VAL+TEST-selected spec: `scenario=equity_top1000_d1, source_set=all, P=1024, z=1e-05, activation=sincos, seed=3, cost_tau=0.0, layer=gross1_cap, max_weight=0.01`

### equity_top1000_d1 / gross1_cap_fee

- TRAIN-selected: TRAIN=-0.686, VAL=-1.197, TEST=-0.550, VAL+TEST=-0.867, FULL=-0.733, TRAIN-VAL+TEST=+0.181, TRAIN-TEST=-0.136
- TRAIN-selected spec: `scenario=equity_top1000_d1, source_set=all, P=1024, z=1e-05, activation=sincos, seed=1, cost_tau=0.0, layer=gross1_cap_fee, max_weight=0.01`
- VAL+TEST-selected: TRAIN=-1.217, VAL=-0.399, TEST=-0.411, VAL+TEST=-0.410, FULL=-0.999, TRAIN-VAL+TEST=-0.807, TRAIN-TEST=-0.807
- VAL+TEST-selected spec: `scenario=equity_top1000_d1, source_set=all, P=1024, z=1e-05, activation=sincos, seed=3, cost_tau=0.0, layer=gross1_cap_fee, max_weight=0.01`

### equity_top1000_d1 / gross1_fee

- TRAIN-selected: TRAIN=-0.686, VAL=-1.197, TEST=-0.550, VAL+TEST=-0.867, FULL=-0.733, TRAIN-VAL+TEST=+0.181, TRAIN-TEST=-0.136
- TRAIN-selected spec: `scenario=equity_top1000_d1, source_set=all, P=1024, z=1e-05, activation=sincos, seed=1, cost_tau=0.0, layer=gross1_fee, max_weight=0.01`
- VAL+TEST-selected: TRAIN=-1.217, VAL=-0.399, TEST=-0.410, VAL+TEST=-0.410, FULL=-0.999, TRAIN-VAL+TEST=-0.807, TRAIN-TEST=-0.807
- VAL+TEST-selected spec: `scenario=equity_top1000_d1, source_set=all, P=1024, z=1e-05, activation=sincos, seed=3, cost_tau=0.0, layer=gross1_fee, max_weight=0.01`

### equity_top1000_d1 / kernel_gross1_cap_fee

- TRAIN-selected: TRAIN=-0.948, VAL=-0.926, TEST=-1.034, VAL+TEST=-0.966, FULL=-0.952, TRAIN-VAL+TEST=+0.019, TRAIN-TEST=+0.086
- TRAIN-selected spec: `scenario=equity_top1000_d1, source_set=all, P=1024, z=1e-05, activation=sincos, seed=1, cost_tau=0.1, layer=kernel_gross1_cap_fee, max_weight=0.01`
- VAL+TEST-selected: TRAIN=-1.254, VAL=-1.310, TEST=0.029, VAL+TEST=-0.617, FULL=-1.089, TRAIN-VAL+TEST=-0.636, TRAIN-TEST=-1.282
- VAL+TEST-selected spec: `scenario=equity_top1000_d1, source_set=all, P=1024, z=1e-05, activation=sincos, seed=2, cost_tau=0.1, layer=kernel_gross1_cap_fee, max_weight=0.01`

### equity_top1000_d1 / raw_sdf

- TRAIN-selected: TRAIN=0.933, VAL=0.960, TEST=1.045, VAL+TEST=0.985, FULL=0.948, TRAIN-VAL+TEST=-0.052, TRAIN-TEST=-0.112
- TRAIN-selected spec: `scenario=equity_top1000_d1, source_set=all, P=1024, z=1e-05, activation=sincos, seed=1, cost_tau=0.0, layer=raw_sdf, max_weight=0.01`
- VAL+TEST-selected: TRAIN=0.538, VAL=1.591, TEST=1.263, VAL+TEST=1.426, FULL=0.792, TRAIN-VAL+TEST=-0.888, TRAIN-TEST=-0.725
- VAL+TEST-selected spec: `scenario=equity_top1000_d1, source_set=all, P=1024, z=1e-05, activation=sincos, seed=3, cost_tau=0.0, layer=raw_sdf, max_weight=0.01`

## aipt_top3000_factor_postprocess

- Class: strict TOP3000 factor post-analysis
- Status: completed_or_smoke

### equity_top3000_d0

- TRAIN-selected: TRAIN=-0.349, VAL=1.951, TEST=1.180, VAL+TEST=1.627, FULL=-0.273, TRAIN-VAL+TEST=-1.977, TRAIN-TEST=-1.529
- TRAIN-selected spec: `scenario=equity_top3000_d0, dynamic_universe=True, P=64, z=0.0001, variant=vol_target_252_seed_ensemble, ensemble_n=3`
- VAL+TEST-selected: TRAIN=-0.383, VAL=2.200, TEST=1.125, VAL+TEST=1.663, FULL=-0.294, TRAIN-VAL+TEST=-2.046, TRAIN-TEST=-1.508
- VAL+TEST-selected spec: `scenario=equity_top3000_d0, dynamic_universe=True, P=256, z=0.01, variant=vol_target_252_seed_ensemble, ensemble_n=3`

### equity_top3000_d1

- TRAIN-selected: TRAIN=0.571, VAL=-0.522, TEST=1.897, VAL+TEST=-0.373, FULL=0.447, TRAIN-VAL+TEST=+0.945, TRAIN-TEST=-1.325
- TRAIN-selected spec: `scenario=equity_top3000_d1, dynamic_universe=True, P=256, z=0.001, variant=vol_target_252_seed_ensemble, ensemble_n=3`
- VAL+TEST-selected: TRAIN=0.391, VAL=-0.220, TEST=4.217, VAL+TEST=0.443, FULL=0.334, TRAIN-VAL+TEST=-0.052, TRAIN-TEST=-3.827
- VAL+TEST-selected spec: `scenario=equity_top3000_d1, dynamic_universe=True, P=64, z=0.0001, variant=raw_seed_ensemble, ensemble_n=3`

## aipt_top3000_fixed_seed_ensemble

- Class: strict TOP3000 factor post-analysis
- Status: completed_or_smoke

### equity_top3000_d0

- TRAIN-selected: TRAIN=-0.402, VAL=1.553, TEST=1.115, VAL+TEST=0.978, FULL=-0.335, TRAIN-VAL+TEST=-1.380, TRAIN-TEST=-1.517
- TRAIN-selected spec: `scenario=equity_top3000_d0, dynamic_universe=True, P=64, z=0.001, activation=sincos, ensemble_n=3`
- VAL+TEST-selected: TRAIN=-0.402, VAL=1.630, TEST=1.194, VAL+TEST=1.129, FULL=-0.337, TRAIN-VAL+TEST=-1.531, TRAIN-TEST=-1.596
- VAL+TEST-selected spec: `scenario=equity_top3000_d0, dynamic_universe=True, P=64, z=0.0001, activation=sincos, ensemble_n=3`

### equity_top3000_d1

- TRAIN-selected: TRAIN=0.409, VAL=-0.648, TEST=2.910, VAL+TEST=-0.256, FULL=0.346, TRAIN-VAL+TEST=+0.665, TRAIN-TEST=-2.501
- TRAIN-selected spec: `scenario=equity_top3000_d1, dynamic_universe=True, P=256, z=0.01, activation=sincos, ensemble_n=3`
- VAL+TEST-selected: TRAIN=0.391, VAL=-0.220, TEST=4.217, VAL+TEST=0.443, FULL=0.334, TRAIN-VAL+TEST=-0.052, TRAIN-TEST=-3.827
- VAL+TEST-selected spec: `scenario=equity_top3000_d1, dynamic_universe=True, P=64, z=0.0001, activation=sincos, ensemble_n=3`

## aipt_unconstrained_dynamic_smallcap_d0_smoke

- Class: strict dynamic-universe unconstrained smoke
- Status: completed_or_smoke

### equity_smallcap_d0

- TRAIN-selected: TRAIN=0.033, VAL=5.990, TEST=5.782, VAL+TEST=5.872, FULL=0.061, TRAIN-VAL+TEST=-5.839, TRAIN-TEST=-5.749
- TRAIN-selected spec: `scenario=equity_smallcap_d0, source_set=all, projected_sources=False, dynamic_universe=True, P=64, z=0.001, activation=sincos, seed=1, demean_features=False`
- VAL+TEST-selected: TRAIN=0.033, VAL=5.990, TEST=5.782, VAL+TEST=5.872, FULL=0.061, TRAIN-VAL+TEST=-5.839, TRAIN-TEST=-5.749
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, source_set=all, projected_sources=False, dynamic_universe=True, P=64, z=0.001, activation=sincos, seed=1, demean_features=False`

## aipt_unconstrained_dynamic_smallcap_d0d1_p256

- Class: strict dynamic-universe unconstrained sweep
- Status: stopped_or_completed

### equity_smallcap_d0

- TRAIN-selected: TRAIN=0.561, VAL=6.910, TEST=5.927, VAL+TEST=6.394, FULL=0.500, TRAIN-VAL+TEST=-5.833, TRAIN-TEST=-5.366
- TRAIN-selected spec: `scenario=equity_smallcap_d0, source_set=all, projected_sources=False, dynamic_universe=True, P=256, z=0.01, activation=sincos, seed=2, demean_features=False`
- VAL+TEST-selected: TRAIN=0.373, VAL=6.504, TEST=7.978, VAL+TEST=7.164, FULL=0.371, TRAIN-VAL+TEST=-6.791, TRAIN-TEST=-7.605
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, source_set=all, projected_sources=False, dynamic_universe=True, P=256, z=0.001, activation=sincos, seed=1, demean_features=False`

### equity_smallcap_d1

- TRAIN-selected: TRAIN=1.158, VAL=3.199, TEST=5.623, VAL+TEST=4.410, FULL=1.180, TRAIN-VAL+TEST=-3.252, TRAIN-TEST=-4.465
- TRAIN-selected spec: `scenario=equity_smallcap_d1, source_set=all, projected_sources=False, dynamic_universe=True, P=256, z=0.01, activation=sincos, seed=3, demean_features=False`
- VAL+TEST-selected: TRAIN=1.158, VAL=3.199, TEST=5.623, VAL+TEST=4.410, FULL=1.180, TRAIN-VAL+TEST=-3.252, TRAIN-TEST=-4.465
- VAL+TEST-selected spec: `scenario=equity_smallcap_d1, source_set=all, projected_sources=False, dynamic_universe=True, P=256, z=0.01, activation=sincos, seed=3, demean_features=False`

## aipt_unconstrained_main

- Class: unconstrained SDF sweep
- Status: stopped_or_completed

### equity_smallcap_d0

- TRAIN-selected: TRAIN=5.473, VAL=4.122, TEST=2.936, VAL+TEST=3.530, FULL=4.955, TRAIN-VAL+TEST=+1.943, TRAIN-TEST=+2.537
- TRAIN-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=1024, z=0.001, activation=sincos, seed=3, demean_features=False`
- VAL+TEST-selected: TRAIN=5.289, VAL=4.374, TEST=3.286, VAL+TEST=3.828, FULL=4.889, TRAIN-VAL+TEST=+1.460, TRAIN-TEST=+2.002
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=1024, z=0.001, activation=sincos, seed=1, demean_features=False`

### equity_smallcap_d1

- TRAIN-selected: TRAIN=3.545, VAL=3.541, TEST=0.551, VAL+TEST=2.132, FULL=3.145, TRAIN-VAL+TEST=+1.414, TRAIN-TEST=+2.994
- TRAIN-selected spec: `scenario=equity_smallcap_d1, source_set=all, P=1024, z=0.001, activation=sincos, seed=2, demean_features=False`
- VAL+TEST-selected: TRAIN=3.346, VAL=3.287, TEST=1.406, VAL+TEST=2.408, FULL=3.082, TRAIN-VAL+TEST=+0.938, TRAIN-TEST=+1.940
- VAL+TEST-selected spec: `scenario=equity_smallcap_d1, source_set=all, P=1024, z=0.001, activation=sincos, seed=3, demean_features=False`

### equity_top1000_d0

- TRAIN-selected: TRAIN=1.294, VAL=1.051, TEST=2.182, VAL+TEST=1.578, FULL=1.367, TRAIN-VAL+TEST=-0.284, TRAIN-TEST=-0.888
- TRAIN-selected spec: `scenario=equity_top1000_d0, source_set=all, P=1024, z=0.01, activation=sincos, seed=2, demean_features=False`
- VAL+TEST-selected: TRAIN=0.993, VAL=1.194, TEST=2.265, VAL+TEST=1.673, FULL=1.170, TRAIN-VAL+TEST=-0.679, TRAIN-TEST=-1.271
- VAL+TEST-selected spec: `scenario=equity_top1000_d0, source_set=all, P=1024, z=0.01, activation=sincos, seed=1, demean_features=False`

### equity_top1000_d1

- TRAIN-selected: TRAIN=1.163, VAL=0.949, TEST=1.218, VAL+TEST=1.038, FULL=1.126, TRAIN-VAL+TEST=+0.125, TRAIN-TEST=-0.056
- TRAIN-selected spec: `scenario=equity_top1000_d1, source_set=all, P=64, z=1e-05, activation=sincos, seed=3, demean_features=False`
- VAL+TEST-selected: TRAIN=0.982, VAL=1.382, TEST=1.487, VAL+TEST=1.440, FULL=1.111, TRAIN-VAL+TEST=-0.457, TRAIN-TEST=-0.505
- VAL+TEST-selected spec: `scenario=equity_top1000_d1, source_set=all, P=256, z=0.001, activation=sincos, seed=2, demean_features=False`

## aipt_unconstrained_pit_main

- Class: unconstrained SDF sweep
- Status: stopped_or_completed

### equity_smallcap_d0

- TRAIN-selected: TRAIN=6.626, VAL=5.918, TEST=5.944, VAL+TEST=5.899, FULL=6.379, TRAIN-VAL+TEST=+0.727, TRAIN-TEST=+0.681
- TRAIN-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=1024, z=0.01, activation=sincos, seed=3, demean_features=False`
- VAL+TEST-selected: TRAIN=6.436, VAL=6.050, TEST=7.308, VAL+TEST=6.579, FULL=6.442, TRAIN-VAL+TEST=-0.143, TRAIN-TEST=-0.872
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=1024, z=0.001, activation=sincos, seed=2, demean_features=False`

## aipt_unconstrained_pit_smoke

- Class: unconstrained SDF smoke
- Status: completed_or_smoke

### equity_smallcap_d0

- TRAIN-selected: TRAIN=4.909, VAL=5.151, TEST=3.225, VAL+TEST=4.203, FULL=4.715, TRAIN-VAL+TEST=+0.707, TRAIN-TEST=+1.684
- TRAIN-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=64, z=0.001, activation=sincos, seed=1, demean_features=False`
- VAL+TEST-selected: TRAIN=4.909, VAL=5.151, TEST=3.225, VAL+TEST=4.203, FULL=4.715, TRAIN-VAL+TEST=+0.707, TRAIN-TEST=+1.684
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=64, z=0.001, activation=sincos, seed=1, demean_features=False`

## aipt_unconstrained_projection_smoke

- Class: strict datasource smoke
- Status: completed_or_smoke

### equity_smallcap_d0

- TRAIN-selected: TRAIN=7.414, VAL=7.316, TEST=5.151, VAL+TEST=6.257, FULL=7.094, TRAIN-VAL+TEST=+1.157, TRAIN-TEST=+2.263
- TRAIN-selected spec: `scenario=equity_smallcap_d0, source_set=price, projected_sources=False, P=64, z=0.001, activation=sincos, seed=1, demean_features=False`
- VAL+TEST-selected: TRAIN=7.414, VAL=7.316, TEST=5.151, VAL+TEST=6.257, FULL=7.094, TRAIN-VAL+TEST=+1.157, TRAIN-TEST=+2.263
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, source_set=price, projected_sources=False, P=64, z=0.001, activation=sincos, seed=1, demean_features=False`

## aipt_unconstrained_projection_smoke_proj

- Class: strict datasource projection smoke
- Status: completed_or_smoke

### equity_smallcap_d0

- TRAIN-selected: TRAIN=3.087, VAL=4.488, TEST=2.585, VAL+TEST=3.661, FULL=3.229, TRAIN-VAL+TEST=-0.574, TRAIN-TEST=+0.503
- TRAIN-selected spec: `scenario=equity_smallcap_d0, source_set=all, projected_sources=True, P=64, z=0.001, activation=sincos, seed=1, demean_features=False`
- VAL+TEST-selected: TRAIN=3.087, VAL=4.488, TEST=2.585, VAL+TEST=3.661, FULL=3.229, TRAIN-VAL+TEST=-0.574, TRAIN-TEST=+0.503
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, source_set=all, projected_sources=True, P=64, z=0.001, activation=sincos, seed=1, demean_features=False`

## aipt_unconstrained_smoke

- Class: unconstrained SDF smoke
- Status: completed_or_smoke

### equity_smallcap_d0

- TRAIN-selected: TRAIN=3.020, VAL=3.238, TEST=2.396, VAL+TEST=2.774, FULL=2.952, TRAIN-VAL+TEST=+0.246, TRAIN-TEST=+0.624
- TRAIN-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=64, z=0.001, activation=sincos, seed=1, demean_features=False`
- VAL+TEST-selected: TRAIN=3.020, VAL=3.238, TEST=2.396, VAL+TEST=2.774, FULL=2.952, TRAIN-VAL+TEST=+0.246, TRAIN-TEST=+0.624
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=64, z=0.001, activation=sincos, seed=1, demean_features=False`

## aipt_unconstrained_strict_main

- Class: strict unconstrained SDF sweep
- Status: stopped_or_completed

### equity_smallcap_d0

- TRAIN-selected: TRAIN=8.918, VAL=8.233, TEST=6.924, VAL+TEST=7.615, FULL=8.557, TRAIN-VAL+TEST=+1.302, TRAIN-TEST=+1.994
- TRAIN-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=1024, z=0.001, activation=sincos, seed=2, demean_features=False`
- VAL+TEST-selected: TRAIN=8.824, VAL=9.100, TEST=7.765, VAL+TEST=8.461, FULL=8.690, TRAIN-VAL+TEST=+0.364, TRAIN-TEST=+1.059
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=1024, z=0.001, activation=sincos, seed=1, demean_features=False`

### equity_smallcap_d1

- TRAIN-selected: TRAIN=4.797, VAL=4.918, TEST=4.068, VAL+TEST=4.507, FULL=4.709, TRAIN-VAL+TEST=+0.289, TRAIN-TEST=+0.728
- TRAIN-selected spec: `scenario=equity_smallcap_d1, source_set=all, P=1024, z=0.01, activation=sincos, seed=2, demean_features=False`
- VAL+TEST-selected: TRAIN=4.757, VAL=4.834, TEST=4.795, VAL+TEST=4.868, FULL=4.767, TRAIN-VAL+TEST=-0.111, TRAIN-TEST=-0.038
- VAL+TEST-selected spec: `scenario=equity_smallcap_d1, source_set=all, P=1024, z=0.01, activation=sincos, seed=3, demean_features=False`

### equity_top1000_d0

- TRAIN-selected: TRAIN=1.263, VAL=0.217, TEST=1.272, VAL+TEST=0.740, FULL=1.129, TRAIN-VAL+TEST=+0.523, TRAIN-TEST=-0.009
- TRAIN-selected spec: `scenario=equity_top1000_d0, source_set=all, P=256, z=0.001, activation=sincos, seed=2, demean_features=False`
- VAL+TEST-selected: TRAIN=0.409, VAL=0.134, TEST=2.455, VAL+TEST=1.302, FULL=0.612, TRAIN-VAL+TEST=-0.893, TRAIN-TEST=-2.046
- VAL+TEST-selected spec: `scenario=equity_top1000_d0, source_set=all, P=1024, z=0.0001, activation=sincos, seed=2, demean_features=False`

### equity_top1000_d1

- TRAIN-selected: TRAIN=0.982, VAL=1.094, TEST=0.567, VAL+TEST=0.821, FULL=0.939, TRAIN-VAL+TEST=+0.161, TRAIN-TEST=+0.415
- TRAIN-selected spec: `scenario=equity_top1000_d1, source_set=all, P=1024, z=0.0001, activation=sincos, seed=1, demean_features=False`
- VAL+TEST-selected: TRAIN=0.538, VAL=1.591, TEST=1.263, VAL+TEST=1.426, FULL=0.792, TRAIN-VAL+TEST=-0.888, TRAIN-TEST=-0.725
- VAL+TEST-selected spec: `scenario=equity_top1000_d1, source_set=all, P=1024, z=1e-05, activation=sincos, seed=3, demean_features=False`

### kucoin_top100

- TRAIN-selected: TRAIN=2.267, VAL=-1.116, TEST=-4.214, VAL+TEST=-2.787, FULL=-0.706, TRAIN-VAL+TEST=+5.055, TRAIN-TEST=+6.481
- TRAIN-selected spec: `scenario=kucoin_top100, source_set=all, P=64, z=0.0001, activation=sincos, seed=2, demean_features=False`
- VAL+TEST-selected: TRAIN=0.725, VAL=2.837, TEST=6.471, VAL+TEST=4.294, FULL=2.876, TRAIN-VAL+TEST=-3.569, TRAIN-TEST=-5.745
- VAL+TEST-selected spec: `scenario=kucoin_top100, source_set=all, P=1024, z=1e-05, activation=sincos, seed=1, demean_features=False`

## aipt_unconstrained_strict_smoke

- Class: strict unconstrained SDF smoke
- Status: completed_or_smoke

### equity_smallcap_d0

- TRAIN-selected: TRAIN=6.189, VAL=6.734, TEST=3.534, VAL+TEST=5.137, FULL=5.896, TRAIN-VAL+TEST=+1.052, TRAIN-TEST=+2.655
- TRAIN-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=64, z=0.001, activation=sincos, seed=1, demean_features=False`
- VAL+TEST-selected: TRAIN=6.189, VAL=6.734, TEST=3.534, VAL+TEST=5.137, FULL=5.896, TRAIN-VAL+TEST=+1.052, TRAIN-TEST=+2.655
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, source_set=all, P=64, z=0.001, activation=sincos, seed=1, demean_features=False`

## aipt_unconstrained_top3000_dynamic_d0_p64p256

- Class: strict dynamic-universe unconstrained sweep
- Status: stopped_or_completed

### equity_top3000_d0

- TRAIN-selected: TRAIN=0.470, VAL=1.156, TEST=1.033, VAL+TEST=0.817, FULL=0.515, TRAIN-VAL+TEST=-0.346, TRAIN-TEST=-0.563
- TRAIN-selected spec: `scenario=equity_top3000_d0, source_set=all, projected_sources=False, dynamic_universe=True, P=64, z=0.01, activation=sincos, seed=1, demean_features=False`
- VAL+TEST-selected: TRAIN=-0.404, VAL=2.354, TEST=1.744, VAL+TEST=1.759, FULL=-0.342, TRAIN-VAL+TEST=-2.163, TRAIN-TEST=-2.149
- VAL+TEST-selected spec: `scenario=equity_top3000_d0, source_set=all, projected_sources=False, dynamic_universe=True, P=64, z=0.01, activation=sincos, seed=3, demean_features=False`

## aipt_unconstrained_top3000_dynamic_d1_p64p256

- Class: strict dynamic-universe unconstrained sweep
- Status: stopped_or_completed

### equity_top3000_d1

- TRAIN-selected: TRAIN=0.415, VAL=-0.611, TEST=2.976, VAL+TEST=-0.279, FULL=0.352, TRAIN-VAL+TEST=+0.695, TRAIN-TEST=-2.560
- TRAIN-selected spec: `scenario=equity_top3000_d1, source_set=all, projected_sources=False, dynamic_universe=True, P=256, z=0.001, activation=sincos, seed=3, demean_features=False`
- VAL+TEST-selected: TRAIN=0.405, VAL=2.375, TEST=2.320, VAL+TEST=2.365, FULL=0.345, TRAIN-VAL+TEST=-1.960, TRAIN-TEST=-1.915
- VAL+TEST-selected spec: `scenario=equity_top3000_d1, source_set=all, projected_sources=False, dynamic_universe=True, P=64, z=0.001, activation=sincos, seed=2, demean_features=False`

## aipt_unconstrained_top3000_dynamic_smoke

- Class: strict dynamic-universe unconstrained smoke
- Status: completed_or_smoke

### equity_top3000_d0

- TRAIN-selected: TRAIN=-0.388, VAL=1.253, TEST=1.030, VAL+TEST=0.809, FULL=-0.271, TRAIN-VAL+TEST=-1.197, TRAIN-TEST=-1.418
- TRAIN-selected spec: `scenario=equity_top3000_d0, source_set=all, projected_sources=False, dynamic_universe=True, P=64, z=0.001, activation=sincos, seed=1, demean_features=False`
- VAL+TEST-selected: TRAIN=-0.388, VAL=1.253, TEST=1.030, VAL+TEST=0.809, FULL=-0.271, TRAIN-VAL+TEST=-1.197, TRAIN-TEST=-1.418
- VAL+TEST-selected spec: `scenario=equity_top3000_d0, source_set=all, projected_sources=False, dynamic_universe=True, P=64, z=0.001, activation=sincos, seed=1, demean_features=False`

## aipt_walkforward_asset_signal_top3000_d0_price_p256_seedens4_12m_washout7

- Class: unclassified AIPT output
- Status: completed_or_smoke

### equity_top3000_d0

- TRAIN-selected: TRAIN=6.433, VAL+TEST=2.426, FULL=2.426, TRAIN-VAL+TEST=+4.008
- TRAIN-selected spec: `scenario=equity_top3000_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=6.433, VAL+TEST=2.426, FULL=2.426, TRAIN-VAL+TEST=+4.008
- VAL+TEST-selected spec: `scenario=equity_top3000_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

## aipt_walkforward_asset_signal_top3000_d0_price_seedens_12m_washout7

- Class: unclassified AIPT output
- Status: completed_or_smoke

### equity_top3000_d0

- TRAIN-selected: TRAIN=6.628, VAL+TEST=2.287, FULL=2.287, TRAIN-VAL+TEST=+4.341
- TRAIN-selected spec: `scenario=equity_top3000_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=6.628, VAL+TEST=2.287, FULL=2.287, TRAIN-VAL+TEST=+4.341
- VAL+TEST-selected spec: `scenario=equity_top3000_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

## aipt_walkforward_cost_equity_all_exec_12m_washout7

- Class: strict costed walk-forward selector
- Status: completed_or_smoke

### equity_smallcap_d0

- TRAIN-selected: TRAIN=6.669, VAL+TEST=5.934, FULL=5.934, TRAIN-VAL+TEST=+0.735
- TRAIN-selected spec: `scenario=equity_smallcap_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=6.669, VAL+TEST=5.934, FULL=5.934, TRAIN-VAL+TEST=+0.735
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

### equity_smallcap_d1

- TRAIN-selected: TRAIN=2.907, VAL+TEST=2.319, FULL=2.319, TRAIN-VAL+TEST=+0.587
- TRAIN-selected spec: `scenario=equity_smallcap_d1, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=2.907, VAL+TEST=2.319, FULL=2.319, TRAIN-VAL+TEST=+0.587
- VAL+TEST-selected spec: `scenario=equity_smallcap_d1, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

### equity_top1000_d0

- TRAIN-selected: TRAIN=-0.479, VAL+TEST=-0.879, FULL=-0.879, TRAIN-VAL+TEST=+0.400
- TRAIN-selected spec: `scenario=equity_top1000_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=-0.479, VAL+TEST=-0.879, FULL=-0.879, TRAIN-VAL+TEST=+0.400
- VAL+TEST-selected spec: `scenario=equity_top1000_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

### equity_top1000_d1

- TRAIN-selected: TRAIN=-0.513, VAL+TEST=-1.336, FULL=-1.336, TRAIN-VAL+TEST=+0.823
- TRAIN-selected spec: `scenario=equity_top1000_d1, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=-0.513, VAL+TEST=-1.336, FULL=-1.336, TRAIN-VAL+TEST=+0.823
- VAL+TEST-selected spec: `scenario=equity_top1000_d1, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

## aipt_walkforward_cost_equity_base_12m_washout7

- Class: strict costed walk-forward selector
- Status: completed_or_smoke

### equity_smallcap_d0

- TRAIN-selected: TRAIN=6.527, VAL+TEST=5.960, FULL=5.960, TRAIN-VAL+TEST=+0.567
- TRAIN-selected spec: `scenario=equity_smallcap_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=6.527, VAL+TEST=5.960, FULL=5.960, TRAIN-VAL+TEST=+0.567
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

### equity_smallcap_d1

- TRAIN-selected: TRAIN=2.532, VAL+TEST=2.201, FULL=2.201, TRAIN-VAL+TEST=+0.331
- TRAIN-selected spec: `scenario=equity_smallcap_d1, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=2.532, VAL+TEST=2.201, FULL=2.201, TRAIN-VAL+TEST=+0.331
- VAL+TEST-selected spec: `scenario=equity_smallcap_d1, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

### equity_top1000_d0

- TRAIN-selected: TRAIN=-0.479, VAL+TEST=-0.879, FULL=-0.879, TRAIN-VAL+TEST=+0.400
- TRAIN-selected spec: `scenario=equity_top1000_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=-0.479, VAL+TEST=-0.879, FULL=-0.879, TRAIN-VAL+TEST=+0.400
- VAL+TEST-selected spec: `scenario=equity_top1000_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

### equity_top1000_d1

- TRAIN-selected: TRAIN=-0.513, VAL+TEST=-1.336, FULL=-1.336, TRAIN-VAL+TEST=+0.823
- TRAIN-selected spec: `scenario=equity_top1000_d1, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=-0.513, VAL+TEST=-1.336, FULL=-1.336, TRAIN-VAL+TEST=+0.823
- VAL+TEST-selected spec: `scenario=equity_top1000_d1, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

## aipt_walkforward_cost_equity_turnover_12m_washout7

- Class: strict costed walk-forward selector
- Status: completed_or_smoke

### equity_smallcap_d0

- TRAIN-selected: TRAIN=6.669, VAL+TEST=5.934, FULL=5.934, TRAIN-VAL+TEST=+0.735
- TRAIN-selected spec: `scenario=equity_smallcap_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=6.669, VAL+TEST=5.934, FULL=5.934, TRAIN-VAL+TEST=+0.735
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

### equity_smallcap_d1

- TRAIN-selected: TRAIN=2.907, VAL+TEST=2.319, FULL=2.319, TRAIN-VAL+TEST=+0.587
- TRAIN-selected spec: `scenario=equity_smallcap_d1, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=2.907, VAL+TEST=2.319, FULL=2.319, TRAIN-VAL+TEST=+0.587
- VAL+TEST-selected spec: `scenario=equity_smallcap_d1, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

### equity_top1000_d0

- TRAIN-selected: TRAIN=-0.479, VAL+TEST=-0.879, FULL=-0.879, TRAIN-VAL+TEST=+0.400
- TRAIN-selected spec: `scenario=equity_top1000_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=-0.479, VAL+TEST=-0.879, FULL=-0.879, TRAIN-VAL+TEST=+0.400
- VAL+TEST-selected spec: `scenario=equity_top1000_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

### equity_top1000_d1

- TRAIN-selected: TRAIN=-0.513, VAL+TEST=-1.336, FULL=-1.336, TRAIN-VAL+TEST=+0.823
- TRAIN-selected spec: `scenario=equity_top1000_d1, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=-0.513, VAL+TEST=-1.336, FULL=-1.336, TRAIN-VAL+TEST=+0.823
- VAL+TEST-selected spec: `scenario=equity_top1000_d1, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

## aipt_walkforward_cost_kucoin_all_exec_seedens_3m_washout7

- Class: strict costed walk-forward seed-ensemble selector
- Status: completed_or_smoke

### kucoin_top100

- TRAIN-selected: TRAIN=1.370, VAL+TEST=0.789, FULL=0.789, TRAIN-VAL+TEST=+0.581
- TRAIN-selected spec: `scenario=kucoin_top100, train_months=3, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=1.370, VAL+TEST=0.789, FULL=0.789, TRAIN-VAL+TEST=+0.581
- VAL+TEST-selected spec: `scenario=kucoin_top100, train_months=3, live_months=1, washout_days=7, selection_metric=train_SR`

## aipt_walkforward_cost_kucoin_completed_3m_washout7

- Class: strict costed walk-forward selector
- Status: completed_or_smoke

### kucoin_top100

- TRAIN-selected: TRAIN=2.280, VAL+TEST=-0.128, FULL=-0.128, TRAIN-VAL+TEST=+2.408
- TRAIN-selected spec: `scenario=kucoin_top100, train_months=3, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=2.280, VAL+TEST=-0.128, FULL=-0.128, TRAIN-VAL+TEST=+2.408
- VAL+TEST-selected spec: `scenario=kucoin_top100, train_months=3, live_months=1, washout_days=7, selection_metric=train_SR`

## aipt_walkforward_cost_kucoin_qp_seedens_3m_washout7

- Class: strict costed walk-forward seed-ensemble selector
- Status: completed_or_smoke

### kucoin_top100

- TRAIN-selected: TRAIN=1.332, VAL+TEST=1.468, FULL=1.468, TRAIN-VAL+TEST=-0.136
- TRAIN-selected spec: `scenario=kucoin_top100, train_months=3, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=1.332, VAL+TEST=1.468, FULL=1.468, TRAIN-VAL+TEST=-0.136
- VAL+TEST-selected spec: `scenario=kucoin_top100, train_months=3, live_months=1, washout_days=7, selection_metric=train_SR`

## aipt_walkforward_cost_kucoin_seedens_3m_washout7

- Class: strict costed walk-forward seed-ensemble selector
- Status: completed_or_smoke

### kucoin_top100

- TRAIN-selected: TRAIN=1.332, VAL+TEST=1.468, FULL=1.468, TRAIN-VAL+TEST=-0.136
- TRAIN-selected spec: `scenario=kucoin_top100, train_months=3, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=1.332, VAL+TEST=1.468, FULL=1.468, TRAIN-VAL+TEST=-0.136
- VAL+TEST-selected spec: `scenario=kucoin_top100, train_months=3, live_months=1, washout_days=7, selection_metric=train_SR`

## aipt_walkforward_unconstrained_dynamic_smallcap_p256_12m_washout7

- Class: strict dynamic-universe walk-forward selector
- Status: completed_or_smoke

### equity_smallcap_d0

- TRAIN-selected: TRAIN=6.031, VAL+TEST=0.489, FULL=0.489, TRAIN-VAL+TEST=+5.542
- TRAIN-selected spec: `scenario=equity_smallcap_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=6.031, VAL+TEST=0.489, FULL=0.489, TRAIN-VAL+TEST=+5.542
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

### equity_smallcap_d1

- TRAIN-selected: TRAIN=4.391, VAL+TEST=0.654, FULL=0.654, TRAIN-VAL+TEST=+3.737
- TRAIN-selected spec: `scenario=equity_smallcap_d1, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=4.391, VAL+TEST=0.654, FULL=0.654, TRAIN-VAL+TEST=+3.737
- VAL+TEST-selected spec: `scenario=equity_smallcap_d1, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

## aipt_walkforward_unconstrained_dynamic_smallcap_p256_seedens_12m_washout7

- Class: strict dynamic-universe walk-forward seed-ensemble selector
- Status: completed_or_smoke

### equity_smallcap_d0

- TRAIN-selected: TRAIN=6.154, VAL+TEST=0.564, FULL=0.564, TRAIN-VAL+TEST=+5.590
- TRAIN-selected spec: `scenario=equity_smallcap_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=6.154, VAL+TEST=0.564, FULL=0.564, TRAIN-VAL+TEST=+5.590
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

### equity_smallcap_d1

- TRAIN-selected: TRAIN=4.206, VAL+TEST=-0.113, FULL=-0.113, TRAIN-VAL+TEST=+4.319
- TRAIN-selected spec: `scenario=equity_smallcap_d1, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=4.206, VAL+TEST=-0.113, FULL=-0.113, TRAIN-VAL+TEST=+4.319
- VAL+TEST-selected spec: `scenario=equity_smallcap_d1, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

## aipt_walkforward_unconstrained_equity_12m_washout7

- Class: strict walk-forward selector
- Status: completed_or_smoke

### equity_smallcap_d0

- TRAIN-selected: TRAIN=10.136, VAL+TEST=8.456, FULL=8.456, TRAIN-VAL+TEST=+1.680
- TRAIN-selected spec: `scenario=equity_smallcap_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=10.136, VAL+TEST=8.456, FULL=8.456, TRAIN-VAL+TEST=+1.680
- VAL+TEST-selected spec: `scenario=equity_smallcap_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

### equity_smallcap_d1

- TRAIN-selected: TRAIN=5.547, VAL+TEST=4.129, FULL=4.129, TRAIN-VAL+TEST=+1.419
- TRAIN-selected spec: `scenario=equity_smallcap_d1, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=5.547, VAL+TEST=4.129, FULL=4.129, TRAIN-VAL+TEST=+1.419
- VAL+TEST-selected spec: `scenario=equity_smallcap_d1, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

### equity_top1000_d0

- TRAIN-selected: TRAIN=1.841, VAL+TEST=0.500, FULL=0.500, TRAIN-VAL+TEST=+1.342
- TRAIN-selected spec: `scenario=equity_top1000_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=1.841, VAL+TEST=0.500, FULL=0.500, TRAIN-VAL+TEST=+1.342
- VAL+TEST-selected spec: `scenario=equity_top1000_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

### equity_top1000_d1

- TRAIN-selected: TRAIN=1.775, VAL+TEST=0.610, FULL=0.610, TRAIN-VAL+TEST=+1.165
- TRAIN-selected spec: `scenario=equity_top1000_d1, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=1.775, VAL+TEST=0.610, FULL=0.610, TRAIN-VAL+TEST=+1.165
- VAL+TEST-selected spec: `scenario=equity_top1000_d1, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

## aipt_walkforward_unconstrained_kucoin_3m_washout7

- Class: strict walk-forward selector
- Status: completed_or_smoke

### kucoin_top100

- TRAIN-selected: TRAIN=2.723, VAL+TEST=1.511, FULL=1.511, TRAIN-VAL+TEST=+1.212
- TRAIN-selected spec: `scenario=kucoin_top100, train_months=3, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=2.723, VAL+TEST=1.511, FULL=1.511, TRAIN-VAL+TEST=+1.212
- VAL+TEST-selected spec: `scenario=kucoin_top100, train_months=3, live_months=1, washout_days=7, selection_metric=train_SR`

## aipt_walkforward_unconstrained_kucoin_seedens_3m_washout7

- Class: strict walk-forward seed-ensemble selector
- Status: completed_or_smoke

### kucoin_top100

- TRAIN-selected: TRAIN=1.877, VAL+TEST=1.754, FULL=1.754, TRAIN-VAL+TEST=+0.122
- TRAIN-selected spec: `scenario=kucoin_top100, train_months=3, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=1.877, VAL+TEST=1.754, FULL=1.754, TRAIN-VAL+TEST=+0.122
- VAL+TEST-selected spec: `scenario=kucoin_top100, train_months=3, live_months=1, washout_days=7, selection_metric=train_SR`

## aipt_walkforward_unconstrained_top3000_dynamic_p64p256_12m_washout7

- Class: strict TOP3000 no-cost walk-forward selector
- Status: completed_or_smoke

### equity_top3000_d0

- TRAIN-selected: TRAIN=5.003, VAL+TEST=-0.380, FULL=-0.380, TRAIN-VAL+TEST=+5.384
- TRAIN-selected spec: `scenario=equity_top3000_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=5.003, VAL+TEST=-0.380, FULL=-0.380, TRAIN-VAL+TEST=+5.384
- VAL+TEST-selected spec: `scenario=equity_top3000_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

### equity_top3000_d1

- TRAIN-selected: TRAIN=3.922, VAL+TEST=0.393, FULL=0.393, TRAIN-VAL+TEST=+3.529
- TRAIN-selected spec: `scenario=equity_top3000_d1, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=3.922, VAL+TEST=0.393, FULL=0.393, TRAIN-VAL+TEST=+3.529
- VAL+TEST-selected spec: `scenario=equity_top3000_d1, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

## aipt_walkforward_unconstrained_top3000_dynamic_p64p256_2024live_12m_washout7

- Class: strict TOP3000 recent no-cost walk-forward selector
- Status: completed_or_smoke

### equity_top3000_d0

- TRAIN-selected: TRAIN=5.762, VAL+TEST=1.064, FULL=1.064, TRAIN-VAL+TEST=+4.698
- TRAIN-selected spec: `scenario=equity_top3000_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=5.762, VAL+TEST=1.064, FULL=1.064, TRAIN-VAL+TEST=+4.698
- VAL+TEST-selected spec: `scenario=equity_top3000_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

### equity_top3000_d1

- TRAIN-selected: TRAIN=4.488, VAL+TEST=-0.153, FULL=-0.153, TRAIN-VAL+TEST=+4.641
- TRAIN-selected spec: `scenario=equity_top3000_d1, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=4.488, VAL+TEST=-0.153, FULL=-0.153, TRAIN-VAL+TEST=+4.641
- VAL+TEST-selected spec: `scenario=equity_top3000_d1, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

## aipt_walkforward_unconstrained_top3000_dynamic_p64p256_seedens_12m_washout7

- Class: strict TOP3000 no-cost walk-forward seed-ensemble selector
- Status: completed_or_smoke

### equity_top3000_d0

- TRAIN-selected: TRAIN=5.041, VAL+TEST=-0.373, FULL=-0.373, TRAIN-VAL+TEST=+5.414
- TRAIN-selected spec: `scenario=equity_top3000_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=5.041, VAL+TEST=-0.373, FULL=-0.373, TRAIN-VAL+TEST=+5.414
- VAL+TEST-selected spec: `scenario=equity_top3000_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

### equity_top3000_d1

- TRAIN-selected: TRAIN=3.784, VAL+TEST=0.390, FULL=0.390, TRAIN-VAL+TEST=+3.394
- TRAIN-selected spec: `scenario=equity_top3000_d1, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=3.784, VAL+TEST=0.390, FULL=0.390, TRAIN-VAL+TEST=+3.394
- VAL+TEST-selected spec: `scenario=equity_top3000_d1, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

## aipt_walkforward_unconstrained_top3000_dynamic_p64p256_seedens_2024live_12m_washout7

- Class: strict TOP3000 recent no-cost walk-forward seed-ensemble selector
- Status: completed_or_smoke

### equity_top3000_d0

- TRAIN-selected: TRAIN=6.433, VAL+TEST=0.837, FULL=0.837, TRAIN-VAL+TEST=+5.596
- TRAIN-selected spec: `scenario=equity_top3000_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=6.433, VAL+TEST=0.837, FULL=0.837, TRAIN-VAL+TEST=+5.596
- VAL+TEST-selected spec: `scenario=equity_top3000_d0, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`

### equity_top3000_d1

- TRAIN-selected: TRAIN=3.407, VAL+TEST=-0.375, FULL=-0.375, TRAIN-VAL+TEST=+3.782
- TRAIN-selected spec: `scenario=equity_top3000_d1, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
- VAL+TEST-selected: TRAIN=3.407, VAL+TEST=-0.375, FULL=-0.375, TRAIN-VAL+TEST=+3.782
- VAL+TEST-selected spec: `scenario=equity_top3000_d1, train_months=12, live_months=1, washout_days=7, selection_metric=train_SR`
