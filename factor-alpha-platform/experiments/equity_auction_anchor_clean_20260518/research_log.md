# Anchored MOC auction-dislocation reversal

Label: `AUCT_ANCHOR_MCAP90_550_D0_CLEAN_S5_F5_20260518`
Status: saturated_6_saved_of_10_target

Fixed universe:
- `AUCT_ANCHOR_MCAP90M_550M_DAILY` copied before the run from the dynamic PIT cap-band universe.
- Rules: market cap 90MM-550MM, price 1.5-80, requires PIT membership, VWAP, volume, and subindustry.
- The universe is daily dynamic, not a static train-date membership snapshot.

Pre-registered strategy hypothesis:
- Small-cap MOC closing-auction dislocations should reverse when the weak close is paired with an independent accounting/valuation/liquidity anchor.
- The auction trigger supplies timing; the anchor filters cases where the dislocation is more likely liquidity pressure than permanent information.
- Each alpha changes the anchor or the auction observable once. There are no window/decay/lookback grids.

Workflow controls:
- Train-only discovery: 2016-01-01 through 2023-01-01.
- No validation/test data is read by this script.
- Existing libraries used: `eval_alpha_ib` loading, expression engine, simulator, preprocessing, persistence, and DB diversity check.
- Neutralization: subindustry through the shared preprocessing path.
- Gates: train Sharpe > 5, train fitness > 5, turnover <= 1.0, IC > 0, PnL kurtosis <= 25, rolling SR std <= 1.50, skew >= -1.00.
- Correlation cutoff: 0.70 through the IB DB save gate.
- Exact expression blocklist includes seed alphas, `data/alpha_results.db`, `data/ib_alphas.db`, and prior experiment saved-alpha CSVs.

DB cleanup:
- Deleted 0 prior rows for `manual_final_update` before this clean run.

Passing alphas saved:
- #38: SR=+5.489, Fit=5.343, TO=0.526, IC=+0.0174, H1=+4.743, H2=+6.176, expr=`rank(decay_linear(multiply(multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(negative(true_divide(close, add(vwap, 0.001))))), rank(true_divide(cashflow_op, add(assets, 1.0)))), 3))`
- #39: SR=+5.073, Fit=5.080, TO=0.451, IC=+0.0159, H1=+4.433, H2=+5.674, expr=`rank(decay_linear(multiply(multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(true_divide(subtract(ts_max(high, 5), close), add(subtract(ts_max(high, 5), ts_min(low, 5)), 0.001)))), rank(roic)), 3))`
- #40: SR=+5.248, Fit=5.207, TO=0.525, IC=+0.0180, H1=+4.111, H2=+6.232, expr=`rank(decay_linear(multiply(multiply(rank(negative(true_divide(close, add(vwap, 0.001)))), rank(negative(ts_delta(close, 3)))), rank(true_divide(cashflow_op, add(assets, 1.0)))), 3))`
- #41: SR=+5.474, Fit=5.172, TO=0.636, IC=+0.0187, H1=+4.662, H2=+6.224, expr=`rank(decay_linear(add(add(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(negative(true_divide(close, add(vwap, 0.001))))), rank(negative(ts_delta(close, 3)))), 3))`
- #42: SR=+6.293, Fit=5.421, TO=0.801, IC=+0.0201, H1=+5.137, H2=+7.352, expr=`rank(decay_linear(rank(true_divide(subtract(high, close), add(subtract(high, low), 0.001))), 3))`
- #43: SR=+5.151, Fit=5.244, TO=0.424, IC=+0.0161, H1=+4.242, H2=+4.556, expr=`rank(decay_linear(add(add(rank(true_divide(subtract(high, close), add(subtract(high, low), 0.001))), rank(true_divide(subtract(ts_max(high, 5), close), add(subtract(ts_max(high, 5), ts_min(low, 5)), 0.001)))), add(multiply(rank(roic), rank(negative(true_divide(ts_delta(assets, 252), add(delay(assets, 252), 1.0))))), rank(negative(net_stock_issuance)))), 3))`

Saved count: 6/10

Data/workflow notes:
- Revisions 1-11 were train-only. No validation/test data was used for discovery.
- Final result did not reach the requested 10 DB-saved alphas: strict Sharpe/Fitness gates plus 0.70 corr cutoff saturated at 6 unique saves in this universe.
- The main blockers were correlation rejections against active IB alphas and the newly saved close-location/pressure states; several rejected candidates had full-gate Sharpe/Fitness above 5.
- The initial anchor-only hypothesis failed Sharpe; flow confirmation fixed Sharpe but failed fitness; fixed persistence fixed fitness but created a correlation wall; close-location plus capital discipline added two more saves.
- The workflow/library problems encountered and fixed: missing active-IB fields in the minimal matrix folder, slow diversity checks, slow subindustry group demeaning, and slow decay_linear rolling callbacks.
- No validation or test outputs were generated in this pass because the requested minimum of 10 saved train-passing alphas was not met.
- Individual alpha discovery remains fee-free by project convention; portfolio validation must apply IB/MOC fees with impact_bps=0.0.
- The harness recomputes returns from close and clips absolute daily returns above 50 percent.

## Validation Data Audit

Run timestamp UTC: 2026-05-18 validation follow-up.

- The low validation table came from the final QP execution layer, not from alpha validation decay.
- No-QP library combiner validation gross Sharpe is about 4.8, versus train gross Sharpe about 6.4; this is the expected ~25 percent train-to-validation drop.
- With IB fees on a 500k book, no-QP validation net Sharpe is about 2.3 because turnover is about 0.65/day and annualized fee drag is about 12.7-13.0 percent.
- With the QP layer enabled, validation gross Sharpe drops to about 1.1-1.7 while turnover falls to about 0.21-0.23/day and annualized fees fall to about 2.7-3.5 percent. The QP is therefore the source of the bad table.
- The validation runner loaded the train+validation panel from 2016-01-04 through 2024-07-01 before computing expressions, so time-series operators had train-history warmup at validation start.
- `eval_alpha_ib.eval_single(split="val")` should not be used as the canonical validation diagnostic for expressions with long lookbacks because it starts the matrix at the validation boundary and loses pre-validation history. Use the config pipeline or a warmup window, then slice metrics.
- Subindustry labels were static across train, train+validation, and full panels: 0 differences across 2,514 tickers.
- The validation column universe had 860 names versus 965 in the full all-period universe; the 105 excluded names were future/test-only universe members. On active validation names, per-alpha preprocessed signal row correlations versus the full-column calculation were above 0.9996, so this overlap difference is not material.
- Simulator normalization and shared `apply_preprocess` matched to machine precision on a representative alpha: max absolute position difference 8.7e-18.

Audit outputs:
- `outputs/validation_noqp_cached_audit.csv`
- `outputs/validation_per_alpha_noqp_audit.csv`
- `outputs/validation_combiner_results.csv`

## Validation Combiner Sweep

Run timestamp UTC: 2026-05-18T19:32:16.794151+00:00
Validation window: 2023-01-01 through 2024-07-01; no rows after validation end loaded for this sweep.
Implementation: shared `src.pipeline.runner` with library combiners, subindustry preprocessing, QP enabled, IB per-share MOC fees, `impact_bps=0.0`, book `$500,000`.
Saved outputs: `experiments/equity_auction_anchor_clean_20260518/outputs/validation_combiner_results.csv`, `experiments/equity_auction_anchor_clean_20260518/outputs/validation_reference_comparison.csv`, `experiments/equity_auction_anchor_clean_20260518/outputs/validation_returns.parquet`.

Best validation combiner: `adaptive` with net SR=+1.215, turnover=0.228, cost_ann=0.035.

Reference rows:
- ib_moc_equity: VAL net SR=+3.127
- aipt_smallcap_d0_prox_l1_tau5: VAL net SR=+4.776

## Original Alpha Execution/QP Turnover Tuning

Run timestamp local: 2026-05-18.

Purpose:
- Lower execution cost/turnover on the original six saved alphas (#38-#43) before changing alpha definitions.
- Use existing project libraries only: expression engine, library combiners, subindustry preprocessing, `src.pipeline.runner._build_risk_model_fn`, `src.portfolio.qp.run_walkforward`, `src.pipeline.fees`, and the AIPT L1 turnover-control helper.
- Fee model: IB per-share MOC model on `$500,000` GMV with `impact_bps=0.0`.

Outputs:
- `outputs/qp_turnover_control_original_alphas.csv`
- `outputs/qp_turnover_control_original_returns.parquet`
- `outputs/alpha_turnover_decay_diagnostic.csv`
- Scripts:
  - `run_qp_tuning_original_alphas.py`
  - `run_qp_turnover_control_original_alphas.py`
  - `run_alpha_turnover_decay_diagnostic.py`

Finding:
- The original QP problem was not the alpha signals; it was the generic QP scaling/cost setting.
- Passing already normalized portfolio weights into a linear-alpha QP without sufficient alpha scaling made the old `lambda=5, kappa=30` setting suppress the signal too much.
- The stable setting across equal/adaptive/ic-weighted combiners was:
  - risk model: `style+pca`
  - `alpha_scale=10`
  - `lambda_risk=2`
  - `kappa_tc=50`
  - no impact, IB per-share commissions

Focused validation table:

| combiner | cell | val_SR_gross | val_SR_net | turnover_val | val_cost_ann | corr_vs_noqp_net |
|:--|:--|--:|--:|--:|--:|--:|
| adaptive | no_qp | 4.816 | 2.337 | 0.701 | 0.130 | 1.000 |
| adaptive | no_qp_turncap0.50 | 4.506 | 2.337 | 0.500 | 0.102 | 0.974 |
| adaptive | qp_style+pca_s10_l2_k50 | 4.621 | 4.077 | 0.592 | 0.045 | 0.583 |
| adaptive | qp_style+pca_s10_l2_k50_turncap0.50 | 3.990 | 3.332 | 0.500 | 0.049 | 0.552 |
| equal | no_qp | 4.761 | 2.331 | 0.687 | 0.127 | 1.000 |
| equal | no_qp_turncap0.50 | 4.449 | 2.316 | 0.500 | 0.102 | 0.978 |
| equal | qp_style+pca_s10_l2_k50 | 4.523 | 4.012 | 0.577 | 0.044 | 0.562 |
| equal | qp_style+pca_s10_l2_k50_turncap0.50 | 4.015 | 3.405 | 0.498 | 0.048 | 0.538 |
| ic_wt | no_qp | 4.793 | 2.321 | 0.700 | 0.130 | 1.000 |
| ic_wt | no_qp_turncap0.50 | 4.482 | 2.314 | 0.500 | 0.102 | 0.974 |
| ic_wt | qp_style+pca_s10_l2_k50 | 4.708 | 4.173 | 0.601 | 0.046 | 0.572 |
| ic_wt | qp_style+pca_s10_l2_k50_turncap0.50 | 3.960 | 3.313 | 0.499 | 0.049 | 0.535 |

Decision:
- Preferred QP setting for validation is `style+pca / alpha_scale=10 / lambda_risk=2 / kappa_tc=50`.
- If the live constraint is a hard turnover cap near 0.50/day, use the same QP setting plus `turnover_cap=0.50`; this sacrifices net Sharpe but still beats the original no-QP net materially.
- Do not use `alpha_scale=30, kappa=100`; it was consistently worse than `alpha_scale=10, kappa=50`.

Per-alpha turnover smoothing diagnostic:
- Tested existing `Decay_lin` as a post-signal smoothing layer on each original alpha.
- Post-decay reduced turnover but usually destroyed too much signal. Best lower-turnover choice by validation net was original for 5 of 6 alphas.
- Alpha #42 was the only case where post-decay helped validation net while lowering turnover:
  - original: VAL net SR=1.102, turnover=0.893, cost_ann=0.159
  - post_decay_5: VAL net SR=1.215, turnover=0.488, cost_ann=0.102
- No DB alpha expressions were changed in this pass; portfolio-level QP execution is the robust fix.

Workflow/data notes:
- `style+pca` QP is much slower than diagonal QP because the library rebuilds a factor covariance slice each day. This is acceptable for validation but should be cached for larger sweeps.
- CVXPY emitted two "solution may be inaccurate" warnings, while `run_walkforward` reported all 626 daily solves completed for each cell. The QP library should expose solver status in outputs.
- Added `qp.alpha_scale` support to `src.pipeline.runner` with default `1.0`, so the successful setting can be run through config instead of experiment-side scaling.

## QP Candidates vs Paper-Production Reference Correlation

Run timestamp local: 2026-05-18.

References:
- `paper_ib_moc_equity`: `prod/config/research_equity.json` recomputed on train+validation only.
- `paper_aipt_smallcap_d0`: `experiments/results/aipt_live_refresh_20260514/...returns.parquet`.

Selected candidates:
- Best uncapped AUCT QP: `ic_wt_qp_styleppca_s10_l2_k50`
- Best hard-cap AUCT QP: `equal_qp_styleppca_s10_l2_k50_turncap0p5_blend1`
- Same-combiner hard-cap sanity row: `ic_wt_qp_styleppca_s10_l2_k50_turncap0p5_blend1`

Return-correlation matrix over validation window 2023-01-01 through 2024-07-01, 375 overlapping bars:

| series | paper_ib_moc | paper_aipt | auct_best_ic_wt | auct_hardcap_equal | auct_hardcap_ic_wt |
|:--|--:|--:|--:|--:|--:|
| paper_ib_moc | 1.000 | -0.059 | 0.163 | 0.179 | 0.189 |
| paper_aipt | -0.059 | 1.000 | -0.037 | -0.071 | -0.058 |
| auct_best_ic_wt | 0.163 | -0.037 | 1.000 | 0.899 | 0.958 |
| auct_hardcap_equal | 0.179 | -0.071 | 0.899 | 1.000 | 0.933 |
| auct_hardcap_ic_wt | 0.189 | -0.058 | 0.958 | 0.933 | 1.000 |

Interpretation:
- The AUCT QP candidates are mildly correlated to the IB MOC paper reference (`~0.16-0.19`) and essentially uncorrelated/slightly negative to AIPT (`~-0.04 to -0.07`).
- The capped and uncapped AUCT variants are highly correlated with each other (`0.90+`), as expected because the cap is an execution wrapper over the same signal family.

Saved outputs:
- `outputs/qp_turnover_paper_prod_correlation_matrix.csv`
- `outputs/qp_turnover_paper_prod_correlation_summary.csv`
- `outputs/qp_turnover_paper_prod_correlation_pairs.csv`
- `outputs/qp_turnover_paper_prod_correlation_returns.parquet`

## Frozen VAL-Selected Strategy Test Run

Run timestamp local: 2026-05-18.

Important correction:
- The test run used only the frozen validation-selected setting. No grid was run.
- Setting: `ic_wt` combiner, `style+pca` risk model, `alpha_scale=10`, `lambda_risk=2`, `kappa_tc=50`, `max_w=0.02`, IB per-share MOC fees, `impact_bps=0.0`.
- Test window: 2024-07-02 through 2026-05-14, 469 overlapping bars.
- AUCT QP solve: 1,095 daily solves, 0 failures.

Test comparison:

| series | test_SR_gross | test_SR_net | ret_ann_net | vol_ann_net | max_dd_net | cost_ann | turnover | corr_vs_ib_moc | corr_vs_aipt |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| paper_aipt_smallcap_d0 | 5.879 | 4.560 | 0.256 | 0.056 | -0.022 | 0.074 | n/a | 0.137 | 1.000 |
| auct_ic_wt_qp_styleppca_s10_l2_k50 | 2.215 | 1.726 | 0.182 | 0.106 | -0.075 | 0.052 | 0.674 | 0.316 | 0.155 |
| paper_ib_moc_equity | 1.843 | 1.607 | 0.157 | 0.098 | -0.098 | 0.023 | 0.267 | 1.000 | 0.137 |

Interpretation:
- The AUCT strategy beat the existing IB MOC paper strategy on this test window, but only slightly after fees: net SR 1.726 vs 1.607.
- It did not come close to AIPT on the same dates: net SR 1.726 vs 4.560.
- Test degradation is large versus validation net SR 4.173, so this is not production-ready as a standalone replacement.
- Return correlation stayed moderate to IB MOC (`0.316`) and low to AIPT (`0.155`), so it is still differentiated, but the standalone test edge is weak.

Saved outputs:
- `outputs/test_selected_strategy_summary.csv`
- `outputs/test_selected_strategy_correlation_matrix.csv`
- `outputs/test_selected_strategy_returns.parquet`
- `outputs/test_selected_strategy_weights.parquet`

## Correction: Paper IB MOC Test Reference

Run timestamp local: 2026-05-18.

The previous test comparison used the wrong IB MOC reference. I overrode the
production universe coverage gate to `threshold=0.0` and used the AUCT split
window beginning `2024-07-02`. That is not the running paper MOC dashboard
setting.

Actual dashboard snapshot:
- Source: `prod/stats/output/ops_dashboard.json`
- Strategy: `ib_moc_equity`
- Source execution: `dagster_ib_paper_moc`
- Universe: `MCAP_100M_500M`
- Universe size: 220
- Alpha count: 45
- Max lookback bars: 400
- TEST: 282 bars, `SR_gross=4.158`, `SR_net=3.555`

Reproduction check:
- Using the dashboard alpha-set shape (`archived=0 AND notes LIKE
  SMALLCAP_D0_v2/v3`), the production universe coverage gate (`0.5`), and
  IB MOC fees with `impact_bps=0.0`, the recomputed TEST window
  `2025-04-01` through `2026-05-14` gives `SR_gross=4.077`,
  `SR_net=3.540`.
- The current edited `prod/config/research_equity.json` adds an evaluation
  subquery and drops the same production reference to 36 alphas. That
  recomputes to `SR_net=3.076` on the same window, so it is not the dashboard
  reference.

Corrected production-window test comparison:

| series | test_SR_gross | test_SR_net | ret_ann_net | max_dd_net | cost_ann | turnover |
|:--|--:|--:|--:|--:|--:|--:|
| paper_aipt_smallcap_d0 | 5.852 | 4.544 | 0.266 | -0.022 | 0.077 | n/a |
| paper_ib_moc_equity_dashboard_snapshot | 4.158 | 3.555 | 0.194 | -0.024 | n/a | 0.438 |
| paper_ib_moc_equity_45alpha_recomputed_noimpact | 4.077 | 3.540 | 0.200 | -0.027 | 0.030 | 0.527 |
| paper_ib_moc_previous_eval_gate_36alpha_config | 3.645 | 3.076 | 0.172 | -0.025 | 0.032 | 0.546 |
| auct_ic_wt_qp_styleppca_s10_l2_k50 | 1.690 | 1.221 | 0.141 | -0.075 | 0.054 | 0.694 |

Corrected return correlations over `2025-04-01` through `2026-05-14`:

| series | corr_vs_ib_moc_45alpha | corr_vs_aipt |
|:--|--:|--:|
| paper_ib_moc_45alpha_recomputed_noimpact | 1.000 | 0.118 |
| paper_aipt_smallcap_d0 | 0.118 | 1.000 |
| auct_ic_wt_qp_styleppca_s10_l2_k50 | 0.196 | 0.199 |
| paper_ib_moc_previous_eval_gate_36alpha_config | 0.950 | 0.083 |

Interpretation:
- The user was right: the IB MOC paper strategy was not a 1.6 SR strategy on
  the running test definition. My earlier row was a bad recompute.
- The AUCT selected strategy is much weaker than both existing paper
  strategies on the production TEST window: net SR `1.221` versus IB MOC
  `~3.54-3.55` and AIPT `4.544`.
- The previous dirty production config change should not be used as the live
  reference; it changed the paper MOC alpha set from 45 to 36.
- I restored `prod/config/research_equity.json` to the 45-signal dashboard
  alpha-source shape while preserving the MOC no-impact fee correction.

Saved corrected outputs:
- `outputs/test_prod_window_corrected_summary.csv`
- `outputs/test_prod_window_corrected_correlation_matrix.csv`
- `outputs/test_prod_window_corrected_correlation_overlap.csv`
- `outputs/test_prod_window_corrected_returns.parquet`

Workflow improvement:
- Paper-production comparisons must read the dashboard/signal snapshot metadata
  first (alpha count, universe size, max lookback, split window) and assert that
  the recompute matches those values before presenting performance.

## Test Decay Diagnostic

Run timestamp local: 2026-05-18.

Question: why does the AUCT strategy look bad on TEST when low-volume equity
signals should not disappear immediately?

Key decomposition:
- It was not a uniform alpha cliff. The pure auction-pressure alphas still
  have positive gross edge in production TEST:
  - alpha #41: PROD_TEST gross SR `2.787`, net SR `0.959`
  - alpha #42: PROD_TEST gross SR `3.152`, net SR `0.756`
- The fundamental-anchor alphas are the drag:
  - alpha #38: PROD_TEST gross SR `-0.653`, net SR `-1.968`
  - alpha #39: PROD_TEST gross SR `-0.604`, net SR `-1.868`
  - alpha #40: PROD_TEST gross SR `-1.253`, net SR `-2.439`
  - alpha #43: PROD_TEST gross SR `0.531`, net SR `-0.925`
- Full six-alpha `ic_wt_no_qp` still has positive gross edge, but too much
  cost: PROD_TEST gross SR `2.376`, net SR `0.452`, cost_ann `13.3%`,
  turnover `77.8%/day`.
- Frozen VAL-selected QP improves net but lowers gross and raises realized
  volatility: PROD_TEST gross SR `1.690`, net SR `1.221`, cost_ann `5.4%`,
  turnover `69.4%/day`.
- Half-year behavior shows the core did not die immediately:
  - `ic_wt_no_qp` net SR: 2024H2 `3.329`, 2025H1 `2.169`,
    2025H2 `1.895`, 2026YTD `-0.926`
  - selected QP net SR: 2024H2 `2.203`, 2025H1 `3.112`,
    2025H2 `1.551`, 2026YTD `-0.116`

Diagnostic ablation:
- Keeping only the two surviving pure pressure alphas (#41/#42), without QP,
  gives:
  - AUCT_TEST gross/net SR `4.565 / 2.384`
  - PROD_TEST gross/net SR `3.189 / 1.152`
- This confirms the broad "fundamental anchor + auction pressure" strategy
  idea is the bad part. The close-location/auction-pressure subtheme is the
  part that survived.

Data checks:
- Active-universe price/HLOC/VWAP sanity did not show a split-adjustment or
  price-field alignment break:
  - close coverage inside active universe: `100%` in TRAIN/VAL/TEST
  - HLOC violation rate: `0%`
  - VWAP far-outside-high-low rate: `0%`
  - VWAP/close 1%-99% range widened only modestly from TRAIN
    `[0.962, 1.041]` to PROD_TEST `[0.945, 1.055]`
- Subindustry data is present for all active names, but breadth is thin:
  PROD_TEST averages `349` active names across `135` subindustries, with
  about `85` singleton names/day. Singleton subindustry buckets are neutralized
  to zero, reducing effective breadth.

Interpretation:
- This is mainly a strategy-construction and gate problem, not an obvious
  split/HLOC/VWAP data-corruption problem.
- I let the hypothesis sprawl from pure auction pressure into stale
  fundamental anchors. Those anchored variants passed train gates but were
  weak net-of-fees in validation and negative in production TEST.
- The validation QP result masked that by making the portfolio look much
  better than the underlying no-QP alpha set. The QP helped cost but became a
  large part of the apparent edge.
- The AUCT universe also pays much more IB per-share fee drag than the paper
  MOC strategy. Similar turnover is much more expensive here because the book
  trades lower-priced names and more shares.

Saved diagnostic outputs:
- `outputs/test_decay_per_alpha_metrics.csv`
- `outputs/test_decay_layer_metrics.csv`
- `outputs/test_decay_halfyear_metrics.csv`
- `outputs/test_decay_data_quality.csv`
- `outputs/test_decay_ic_combiner_weights.csv`
- `outputs/test_decay_pressure_41_42_ablation.csv`
- `outputs/test_decay_subindustry_breadth.csv`

## Production Execution Share/Min-Order Replay

Run timestamp local: 2026-05-18.

Question: is the production IB MOC path using a minimum share/order filter or
share rounding that explains why the research AUCT test result is much worse
than the two paper strategies?

Production code path found:
- `prod/config/strategy.json` sets `execution.min_order_value = 200`.
- `prod/moc_trader.py` converts normalized target weights to integer shares
  with `round(weight * booksize / price)`.
- It then computes order diffs versus current IB share positions and skips
  orders where `abs(diff_shares) * price < 200`.
- The shared multi-strategy execution library has the same delta/min-order
  behavior in `src.execution.netting.build_child_orders`, so the experiment
  replay now uses that instead of local delta/filter code.

Post-QP executable replay:
- Script: `apply_prod_execution_filter_auct.py`
- This applies production share rounding and the `$200` child-order filter to
  the saved VAL-selected AUCT QP weights.
- PROD_TEST `2025-04-01` to `2026-05-14`:
  - continuous QP net SR: `1.221`, cost_ann `5.41%`, turnover `69.36%`
  - share-round + min-order net SR: `1.204`, cost_ann `5.62%`,
    turnover `70.84%`
- The filter skipped many tickets, about `17.5` per day, but only
  `0.32%` of book turnover per day. The share rounding increased realized
  turnover more than the skipped tiny orders saved.

Execution-feedback QP replay:
- Script: `run_exec_feedback_qp_auct.py`
- This reruns the selected QP with the production execution state in the loop:
  solve QP, convert to integer target shares, apply `$200` min order filter,
  carry actual held shares, and feed those executable dollar weights into the
  next day's QP as `w_prev`.
- It used the frozen validation-selected settings:
  `ic_wt`, `style+pca`, `alpha_scale=10`, `lambda_risk=2`, `kappa_tc=50`,
  `impact_bps=0`, `book=500000`.
- QP loop solved `1095/1095` bars with zero reported failures. CVXPY emitted
  one "solution may be inaccurate" warning.
- PROD_TEST result:
  - continuous QP net SR: `1.221`, cost_ann `5.41%`, turnover `69.36%`
  - execution-feedback QP net SR: `1.189`, cost_ann `5.54%`,
    turnover `70.64%`
- The full executable-feedback net return series is `0.996` correlated with
  the original continuous QP net series.

Interpretation:
- The production share/min-order mechanics are real, but they are not the
  missing edge. They slightly worsen AUCT net performance.
- The skipped orders are high in count but tiny in dollars, so they cannot
  explain the gap versus the paper strategies.
- The more important code issue found is a production QP unit mismatch:
  `IBConnection.get_positions()` returns share counts, while
  `apply_qp_optimization()` documents and treats `current_positions` as dollar
  positions when building `w_prev`. This experiment used the logically correct
  executable dollar weights for research. Production code should be patched or
  explicitly documented before relying on live QP turnover penalties.

Saved outputs:
- `outputs/auct_prod_execution_filter_summary.csv`
- `outputs/auct_prod_execution_filter_daily_stats.csv`
- `outputs/auct_prod_execution_filter_weights.parquet`
- `outputs/auct_exec_feedback_qp_summary.csv`
- `outputs/auct_exec_feedback_qp_prod_window_summary.csv`
- `outputs/auct_exec_feedback_qp_daily_stats.csv`
- `outputs/auct_exec_feedback_qp_weights.parquet`
- `outputs/auct_exec_feedback_qp_returns.parquet`

## AUCT vs Production Alpha Attribution

Run timestamp local: 2026-05-19.

Question: why does the AUCT strategy underperform the two production paper
strategies?

AIPT paper/shadow strategy stats:
- Active production strategy config:
  `prod/config/strategies/aipt_highest_sharpe_ib_paper.json`
- Strategy id: `aipt_smallcap_d0_prox_l1_tau5`
- Dashboard status: enabled/current artifact, but `submit_enabled=false`,
  `include_in_totals=false`, and PnL source is "current AIPT research
  artifact; not broker truth".
- On `500000` book, the source artifact reports:
  - TRAIN net SR `5.060`, net ann return `25.64%`, cost_ann `6.93%`
  - VAL net SR `4.776`, net ann return `26.32%`, cost_ann `7.30%`
  - TEST net SR `4.544`, net ann return `26.57%`, cost_ann `7.70%`
  - VAL+TEST net SR `4.631`, net ann return `26.23%`, cost_ann `7.49%`
- Dollar interpretation on `500000` book:
  - VAL+TEST expected net annual PnL about `$131k`
  - expected average net daily PnL about `$520`
  - annualized net vol about `$28k`
  - historical max drawdown about `$10.9k`
  - average daily turnover about `$370k`

Cross-universe alpha diagnostic:
- Script: `compare_auct_vs_prod_alphas.py`
- It evaluated:
  - production 45 IB MOC alphas on production `MCAP_100M_500M`
  - production 45 IB MOC alphas on AUCT universe
  - AUCT six alphas on AUCT universe
  - AUCT six alphas on production universe
- It uses existing project libraries:
  `FastExpressionEngine`, `apply_preprocess`, library combiners, and
  `src.pipeline.fees`.

Core finding:
- The AUCT universe is not the main problem. The production alpha set still
  works on the AUCT universe.
- On `PROD_TEST`:
  - production alphas on AUCT universe: median gross SR `2.351`, 45/45
    positive gross, 29/45 positive net
  - AUCT alphas on AUCT universe: median gross SR `-0.036`, 3/6 positive
    gross, 2/6 positive net
  - production equal no-QP on AUCT universe: gross/net SR `3.258 / 1.023`
  - AUCT equal no-QP on AUCT universe: gross/net SR `0.750 / -0.848`
  - AUCT IC-weight no-QP on AUCT universe: gross/net SR `2.376 / 0.452`
  - AUCT selected QP: gross/net SR `1.690 / 1.221`

Production reference comparison on the same `2025-04-01` to `2026-05-14`
window:
- IB MOC 45-alpha recompute: gross/net SR `4.077 / 3.540`, net ann return
  `20.04%`, cost_ann `3.05%`, turnover `52.71%`
- AIPT artifact: gross/net SR `5.852 / 4.544`, net ann return `26.57%`,
  cost_ann `7.70%`
- AUCT selected QP: gross/net SR `1.690 / 1.221`, net ann return `14.08%`,
  cost_ann `5.41%`, turnover `69.36%`

Why AUCT underperforms:
1. The hypothesis drifted into weak fundamental anchors.
   Four of six AUCT alphas use `cashflow_op/assets`, `roic`, `assets`, or
   `net_stock_issuance`. Those variants were the main drag in production
   test. The two pure pressure/close-location alphas kept positive gross edge,
   but their fee drag is too high as standalone deployable alphas.
2. The production IB MOC alpha set is a dense family of simple price/volume
   close-dislocation variants. Field summary shows production alphas are
   almost entirely `close`, `vwap`, `volume`, `dollars_traded`, `high`, `low`,
   `adv20`, and `adv60`; AUCT added slow fundamental conditioning that did not
   improve OOS behavior.
3. The AUCT alpha set is too small. Six alphas with return correlations around
   `0.55` leave the combined book exposed to a few bad sub-hypotheses. The
   production set has 45 alphas, also correlated, but many more variants of
   the same working close/volume mechanism.
4. Gross edge is the primary failure, not just costs. In `PROD_TEST`, AUCT
   raw alpha median gross SR is near zero on its own universe. Production
   alphas have positive gross SR across the board on both universes.
5. QP helped AUCT more than it should have. AUCT IC-weight no-QP gross/net was
   `2.376 / 0.452`; the selected QP improved net to `1.221`, but that means a
   lot of the apparent deployability came from optimizer/risk-model behavior,
   not from a robust alpha stack.
6. The universe is mostly equivalent to production. `PROD_TEST` active-name
   overlap is high: mean overlap `306` names/day, Jaccard `0.794`,
   production names in AUCT `89.5%`, AUCT names in production `87.6%`.

Data checks:
- AUCT price/HLOC/VWAP checks still look clean: close coverage `100%`, HLOC
  violation `0%`, VWAP far-outside-H/L `0%`.
- Subindustry data is available, but AUCT breadth is thin: `PROD_TEST` has
  about `349` active names across `135` subindustries, with about `85`
  singleton names/day. This reduces usable breadth after subindustry
  neutralization, but it affects production-style alphas too, and those still
  work on the AUCT universe.

Conclusion:
- The underperformance is not explained by min-order execution, share rounding,
  fee mismatch, bad price data, or the AUCT universe.
- The real issue is bad strategy research: I forced a "fundamental anchor plus
  auction pressure" theme after the evidence favored simpler close-location /
  pressure-state price-volume signals. The correct response would be to
  abandon the fundamental-anchor AUCT family and restart with a narrow,
  pre-registered close-auction pressure hypothesis, higher OOS-style train
  gates, and enough distinct variants to survive the 0.70 correlation cutoff.

Saved outputs:
- `outputs/auct_vs_prod_per_alpha_cross_universe.csv`
- `outputs/auct_vs_prod_group_summary.csv`
- `outputs/auct_vs_prod_combiner_cross_universe.csv`
- `outputs/auct_vs_prod_alpha_return_corr_summary.csv`
- `outputs/auct_vs_prod_universe_diagnostics.csv`
- `outputs/auct_vs_prod_universe_overlap.csv`
- `outputs/auct_vs_prod_field_summary.csv`
- `outputs/auct_vs_prod_prodtest_equity_curves.png`
- `outputs/auct_vs_prod_prodtest_rolling63_sr.png`
