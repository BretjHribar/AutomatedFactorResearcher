# AIPT Data Integrity & Fixes — 2026-04-24

## Executive summary

On 2026-04-24 the AIPT paper-trading strategies (P=100 and P=1000 on KuCoin perps) had produced **9 consecutive losing bars** against a backtest that claimed **annualised Sharpe ≈ 26**. The gap was far beyond what noise or regime change could explain. A deep audit uncovered three compounding bugs, all data-integrity issues. This document records what was wrong, what was fixed, and the test suite installed so the class of bug cannot recur silently.

**Post-fix headline:** corrected backtest Sharpe drops to ~4–6 (P=100 through P=1000). The Sharpe-26 result is retracted; it was an artefact of corrupted OHLC data, not real alpha. Live results before 2026-04-24 14:00 UTC are likewise invalid and have been archived.

---

## Bug #1 — Feature-set mismatch between backfill and live (the killer)

**Symptom.** The lambda trained by backfill was applied to a feature space it had never seen, so live portfolio weights were essentially random projections of real signals.

**Root cause.**
- Backfill (`aipt_trader.py::run_backfill`) reads from `data/kucoin_cache/matrices/4h/` (research matrices). This directory contained `beta_to_btc.parquet` and `historical_volatility_120.parquet`. Combined with `CHAR_NAMES`, backfill saw **D=24** features.
- Live (`aipt_trader.py::run_pipeline`) reads from `data/kucoin_cache/matrices/4h/prod/` (prod matrices). The prod builder `prod/data_refresh.py::_build_kucoin_matrices` **did not compute** `beta_to_btc` or `historical_volatility_120`. Live therefore saw **D=22**.
- The RFF projection `theta = rng.standard_normal((P//2, D))` is deterministic given `(D, P, seed)`. When `D` differs between backfill and live, `theta` is a completely different matrix. The signal `S_t = sin/cos(Z_t @ theta * gamma)` lives in a different feature space on each side.
- Lambda trained on backfill's `F_t = S_t'·R_t/√N` (D=24) was saved to state and reloaded by live. Live's `F_t` (D=22) is a different basis — applying the trained lambda to live signals is equivalent to dotting with a random vector.

**How it manifested.** Live signal quality was approximately zero for the first ~12 bars (until `REBAL_EVERY=12` would have re-estimated lambda using live factor returns). Observed: 7 consecutive losing bars starting at 04-23 08:00 UTC, cumulative net drag of ~−1.3%.

**Fix.** Added `beta_to_btc` and `historical_volatility_120` to `_build_kucoin_matrices` in [prod/data_refresh.py](../data_refresh.py) so prod builds all 24 features. Also mirrored the computation in a new rebuild script [prod/rebuild_research_matrices.py](../rebuild_research_matrices.py) for the research directory so both sides are regenerated from the same logic.

**Test guards.**
- `TestFeatureParity::test_research_prod_char_names_parity` — fails if the two directories' `CHAR_NAMES` ∩ available ever diverge.
- `TestRFFDeterminism::test_backfill_and_live_use_same_d` — fails immediately if `D` drops.
- `TestRFFDeterminism::test_d22_d24_produce_different_theta_shapes` — regression assertion that the invariant is *necessary* (D mismatch means theta mismatch).

---

## Bug #2 — KuCoin Futures API column-order mismatch

**Symptom.** ~99.5% of kline parquet rows violated the OHLC invariant (`low ≤ min(O,C) ≤ max(O,C) ≤ high`). The "high" column frequently contained values lower than open and close.

**Root cause.** Both `download_kucoin.py` and `prod/data_refresh.py` assumed the KuCoin Futures v1 kline endpoint (`/api/v1/kline/query`) returns candles in column order:

```
[time, open, close, high, low, volume, turnover]     # KuCoin's published docs
```

**This is wrong.** Empirically the API returns:

```
[time, open, HIGH, LOW, CLOSE, volume, turnover]     # verified 2026-04-24
```

Verified against three independent consistency checks:
1. **OHLC invariant.** Under `[O, H, L, C]`, the invariant `H ≥ max(O,C)` and `L ≤ min(O,C)` is satisfied for sealed bars (XBT, ETH, SOL). Under `[O, C, H, L]` it is violated 99%+ of the time.
2. **Cross-check vs KuCoin Spot API.** The spot endpoint (`/api/v1/market/candles`) does return `[time, open, close, high, low, ...]` as documented — spot's close for a given 4h bar matches futures' position-4 value (close), not position-2.
3. **Live current-bar behaviour.** Position 4 of a still-forming bar changes as new trades print; position 2 only moves when a higher price prints (monotonic until the bar closes). That's close and high behaviour respectively.

The spot API is correct; the futures API is not, and the public docs on the futures side are wrong.

**Downstream impact.** Kline parquets on disk had column labels `close`, `high`, `low` populated with real `H`, `L`, `C` values respectively (the `open` label was fine). Every derived feature that uses `close`, `high`, or `low` — which is most of them — was being trained on the wrong column. `returns`, `log_returns`, `momentum_Nd`, `historical_volatility_N` all used the *high series* as if it were the close series. `high_low_range`, `close_position_in_range`, `parkinson_volatility_*`, `vwap_deviation` were built from shuffled H/L/C values.

To compound it, the kline-update code (`_kucoin_update_klines`) evolved over time and at some point appended new bars with a slightly different wrong mapping (`high`↔`low` swapped relative to what download_kucoin.py used). So the historical kline parquets had **two different corruption regimes** — old bars wrong one way, new bars wrong another way.

**Fix.**
1. Corrected `columns=` in both `_kucoin_fetch_klines` (data_refresh.py) and `download_kucoin.py` to the true API order: `["time", "open", "high", "low", "close", "volume", "turnover"]`.
2. Repaired the 551 existing kline parquets via a value-based heuristic: for each bar, `open` is always correct; `parquet.close` always held the real `H`; the lower of `{parquet.high, parquet.low}` is real `L`, the higher is real `C`. Applied across all symbols, this reduced residual OHLC violations from 1,709,392 → 0.
3. Rebuilt prod and research matrices from the repaired klines.

**Test guards.**
- `TestOHLCInvariants::test_kline_ohlc_invariant_major_pairs[XBTUSDTM|ETHUSDTM|SOLUSDTM]` — per-symbol invariant ≥ 99% on major pairs.
- `TestOHLCInvariants::test_all_klines_ohlc_invariant_global` — aggregate < 0.5% violations across all 554 klines.
- `TestOHLCInvariants::test_research_matrix_ohlc_invariant` and `test_prod_matrix_ohlc_invariant` — matrix-level invariant < 0.1%.
- `TestAPISchema::test_kucoin_futures_api_ohlc_schema` — live API regression test that fails if KuCoin ever changes the futures v1 schema.

---

## Bug #3 — Research matrices had high/low populated as close/open copies

**Symptom.** In research `high.parquet`, 78% of values equalled the same bar's `close.parquet` value; in research `low.parquet`, ~99% of XBT values equalled the same bar's `open.parquet` value.

**Root cause.** Almost certainly an old version of a research-matrix builder that used `max(open, close)` as a fallback for "high" and `min(open, close)` as a fallback for "low" when it didn't trust the raw high/low values coming through. This produced a self-consistent but meaningless "range" — the H-L range collapsed to the absolute open-to-close move. Characteristics like `parkinson_volatility_*`, `high_low_range`, `close_position_in_range` became nearly deterministic functions of intrabar return.

**Why this catastrophically inflated the backtest.** When H/L are deterministic functions of O and C, the rank-transformed `Z_t` columns for H/L-derived features become highly correlated with the `Z_t` column for `log_returns` — they're all essentially proxies for the same signal. Cross-sectionally ranking a redundant feature set and letting Markowitz optimise over its RFF projections produced a lambda that over-fit to the current-bar intrabar return pattern. This pattern is cross-sectional noise on a 4h bar — it doesn't predict the *next* bar — but because the "training" signal `F_{t+1} = S_t'·R_{t+1}` still had coefficients that happened to fit on the 2024-09 → 2026-04 OOS window, the backtest looked like Sharpe 26. It was a fit of RFF noise to a lucky regime.

**Fix.** Rebuilt research matrices from the repaired klines using the same computation as the prod builder (see [prod/rebuild_research_matrices.py](../rebuild_research_matrices.py)). Research `high.parquet` now matches prod `high.parquet` bit-for-bit for the universe tickers on overlapping bars.

**Test guards.**
- `TestFeatureValueParity::test_research_vs_prod_exact[high|low|close|...]` — bit parity between research and prod matrices for core OHLC and any pure function of klines.
- `TestFeatureValueParity::test_log_returns_parity_excluding_nan_boundary` — same check for log_returns, tolerating the one-bar boundary where prod's tail truncation differs from research's full-history coverage.

---

## Lock-file robustness (parallel fix delivered during the same audit)

The AIPT P=100 and P=1000 scheduler tasks fire simultaneously. Two bugs were addressed on the lock around `refresh_kucoin`:

1. **Concurrent parquet writes** → corrupted matrix files. Fixed by introducing a lock file `data/kucoin_cache/matrices/4h/.kucoin_refresh.lock` that the second process waits on.
2. **10-minute timeout was too short** — after the Windows machine resumed from sleep, both schedulers fired and each mistakenly declared the other's lock stale after 10 min, stealing it mid-refresh. Replaced the mtime-based staleness check with a **PID-liveness check via psutil**: if the lock holder's PID is still running, respect the lock up to 1 hour; if the PID is dead, the lock is abandoned and can be taken.

See [prod/data_refresh.py::refresh_kucoin](../data_refresh.py).

---

## Stale-data guard (fixed 2026-04-23)

`_kucoin_update_klines` previously skipped any symbol whose latest bar was within 6 hours of "now". That incorrectly excluded symbols sitting at the 16:00 UTC bar when the 20:00 UTC bar was already available. Replaced with `now.floor('4h')` comparison: skip only if the symbol already has the most recent completed 4h bar.

---

## Honest backtest numbers (post-fix)

Ran `run_single_config(P, z=1e-3, seed)` against the corrected research matrices for a small (P, seed) grid:

| P | seed=42 | seed=1 | mean/bar (bps) | std/bar (bps) |
|---|---|---|---|---|
| 100  | +4.16 | +4.04 | +56  | 628  |
| 500  | +5.53 | +5.29 | +120 | 1,053 |
| 1000 | +6.17 | +5.95 | +170 | 1,286 |

Previously the same grid reported Sharpe 25–31. The annualised Sharpe numbers here (4–6) are still very high by industry benchmarks — top quant funds are typically Sharpe 2–3 — so the honest out-of-sample number is probably lower again once residual issues are shaken out (feature selection bias, slippage, funding costs, universe survivorship, and the "too-new" OOS window of only 19 months). But at least the numbers now reflect computations on valid data.

---

## Test suite

New file: [tests/unit/test_aipt_integrity.py](../../tests/unit/test_aipt_integrity.py). 23 tests grouped as:

- **TestOHLCInvariants** — 6 tests covering individual major pairs, aggregate kline parquets, research matrix, and prod matrix.
- **TestFeatureParity** — 3 tests ensuring `CHAR_NAMES` coverage parity between research and prod.
- **TestFeatureValueParity** — 8 tests on exact bit parity of pure-function features (close, open, high, low, momentum, hvol, returns, log_returns).
- **TestNoLookahead** — 1 test that perturbs every bar after index t and asserts characteristics at indices ≤ t are unchanged, across all 24 features.
- **TestRFFDeterminism** — 3 tests that `theta/gamma` are bit-identical given same `(D, P, seed)` and that the pipeline's `D` is consistent end-to-end.
- **TestAPISchema** — 1 network-marked test that verifies KuCoin Futures v1 kline endpoint still returns `[O, H, L, C, V, tv]`.
- **TestReplayMatchesLive** — 1 test that replays the latest equity row from prod matrices and asserts it matches the logged value within 0.5 bps. Skipped when no equity rows exist.

Run with:

```
pytest tests/unit/test_aipt_integrity.py -v
```

Current status: **22 passed, 1 skipped** (replay test will activate once live produces a new bar).

---

## What was archived

All pre-fix live state has been archived to `prod/archive_pre_fix_20260424_1519/`:
- equity CSVs (P=100, P=1000)
- factor-return history (`.npz` + `.npy`)
- state JSONs (`aipt_state.json`, `weights.npz`)
- trade JSON directories

Archived material should be treated as diagnostic-only. Any bar-level statistics from the archived equity CSVs were computed on corrupted features and are not valid performance records.

---

## What to watch next

1. **First live bar after the fix.** Scheduled next for 2026-04-24 16:05 UTC (bar 16:00 UTC). That bar should produce sensible turnover (~50%) and a return whose sign is uncorrelated with noise — if we still see a large negative bar, remaining issues go beyond data integrity.
2. **Drift between research and prod matrices.** Every live run writes prod matrices. Research is regenerated only on demand via `prod/rebuild_research_matrices.py`. Re-run this script any time the kline data is manually modified, or set up a scheduled recompute before the next full backtest.
3. **Residual Sharpe 4–6 plausibility.** Still high. Candidates for further investigation:
   - Feature-list selection bias (the 24 characteristics were chosen at some point; try ablation without `log_returns` or `high_low_range`).
   - OOS window is only 19 months in the backtest — mostly a trendy crypto regime.
   - Slippage / funding / impact not modelled.
   - Universe survivorship — `coverage > 0.3` on the full sample retrospectively excludes coins that delisted.
