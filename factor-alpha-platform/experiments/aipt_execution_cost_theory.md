# AIPT Replication And Cost-Aware Extension

This note records the experiment design used in:

- `experiments/aipt_unconstrained.py`
- `experiments/aipt_stepwise_constraints.py`
- `experiments/aipt_no_lookahead_audit.py`

## Paper Core

The paper models the SDF asset weight function as

```text
w_t = S(Z_t) lambda
F_{t+1} = S(Z_t)' R_{t+1}
lambda(z) = (E[F F'] + z I)^(-1) E[F]
```

where `S(Z_t)` is a large random-feature expansion of point-in-time characteristics. The replication uses:

- cross-sectional rank standardization to `[-0.5, 0.5]`;
- random Fourier features `sin(gamma Z omega), cos(gamma Z omega)`;
- rolling ridge SDF fits;
- factor complexity grids over `P/T`;
- raw SDF metrics first, before any neutrality, cap, gross, or fee constraint;
- tradability layers only after the raw paper object matches.

## No-Lookahead Timing

For signal date `t`, random features are built only from matrices at `t`.

- Equity delay 0: signal at close `t`, earn close `t` to close `t+1`.
- Equity delay 1: signal at close `t`, execute at next open, earn open `t+1` to open `t+2`.
- KuCoin: 4h close-to-next-close.

The rolling SDF fit excludes the current forward return:

```text
delay 0 fit at t uses factor rows < t
delay 1 fit at t uses factor rows < t - 1
```

Datasource projection is trained only on the initial training window and is then frozen.

The strict equity run uses:

- `data/fmp_cache/matrices_pit_v2`, whose manifest records `pit_lag_days = 1`;
- experiment-local PIT universes in `experiments/data/aipt_universes`;
- small-cap membership from same-date PIT membership, close, and market cap;
- top1000 membership from same-date PIT membership, close, and `adv60`;
- no backfills in the AIPT experiment scripts.

The audit report is written to `experiments/results/aipt_no_lookahead_audit_strict.json`.

## Execution Costs In The Kernel

The paper estimator is a frictionless ridge Markowitz problem in random-feature factor space:

```text
max_lambda mu'lambda - 0.5 lambda' G lambda - z lambda'lambda
```

where `G = E[F F']`. To include trading costs before forming the asset portfolio, use a local quadratic turnover approximation around the previous asset weights `w_prev`:

```text
max_lambda mu'lambda
  - 0.5 lambda' G lambda
  - z lambda'lambda
  - tau/2 * || C_t^(1/2) (S_t lambda / sqrt(N_t) - w_prev) ||^2
```

This yields the cost-aware ridge system:

```text
(G + z I + tau S_t' C_t S_t / N_t) lambda
  = mu + tau S_t' C_t w_prev / sqrt(N_t)
```

`experiments/aipt_stepwise_constraints.py` implements the full local kernel for the stepwise cost experiments:

```text
A_t = S_t / sqrt(N_t)
(G + z I + tau A_t' C_t A_t) lambda
  = mu + tau A_t' C_t w_prev
```

This is the clean no-neutrality generalization. The effective random-feature inner product is no longer only return covariance, but return covariance plus an execution-cost geometry. High-cost names reduce the value of features that load on them, while the carry term rewards features that preserve existing positions.

For a gross-normalized or capped portfolio the kernel is a local approximation, because the final mapping from `lambda` to weights is nonlinear. That is why the stepwise experiments include both the frictionless SDF Sharpe and the realized asset-level net Sharpe: the drop from one layer to the next is the object of study, not something hidden inside one combined backtest.

The implementation also supports explicit execution controls after target weights are formed:

```text
w_trade = w_prev + blend * (w_target - w_prev)
if ||w_trade - w_prev||_1 > turnover_cap:
    w_trade = w_prev + turnover_cap * (w_trade - w_prev) / ||w_trade - w_prev||_1
```

This is a simple proximal/no-trade-region approximation for linear execution costs. It is especially important for high-frequency crypto, where a local quadratic kernel can still leave realized L1 turnover too high after gross normalization.

The stepwise runner also exposes the project-native name-level QP from `src.portfolio.qp.solve_qp`. For a raw AIPT signal `a_t = S_t lambda / sqrt(N_t)`, the QP layer solves:

```text
max_w alpha_scale * a_t'w
  - 0.5 * lambda_risk * risk(w)
  - kappa_tc * sum_i c_i |w_i - w_{prev,i}|

s.t. ||w||_1 <= 1
     |w_i| <= min(max_weight, ADV_capacity_i)
     dollar_neutral = false
```

The risk term is currently diagonal and estimated only from return rows inside the rolling training window. This preserves the no-lookahead convention while testing the project library's execution optimizer against the kernel-only construction.

## Datasource Projection

A datasource is a block of fields, such as price/liquidity, fundamentals, or BTC-relative crypto terms. The generalized selection problem is:

```text
choose projection P over datasource blocks before random features,
then build S(P Z_t)
```

The implemented diagnostic projects fields by an initial-window, train-only score:

```text
score(field) = positive_no-neutrality_univariate_SR * sqrt(coverage) / liquidity_cost_proxy
```

Selected fields receive a scale in the input projection before random features are drawn. This is deliberately train-only; validation and test metrics are never used to choose fields, source sets, `P`, ridge, activation, or cost strength.

For a more general datasource projection, replace field-level scoring with a train-only block operator:

```text
Z_t = [Z_t^price, Z_t^fundamental, Z_t^liquidity, Z_t^crypto_btc, ...]
P = argmin_P train_loss(P) + eta ||P||_* + kappa cost_proxy(P)
S_t = phi(P Z_t Omega)
```

The important rule is temporal: estimate `P` only inside the training window, freeze it before validation/test, and treat every datasource block as unavailable until its own timestamp and publication lag allow it.
