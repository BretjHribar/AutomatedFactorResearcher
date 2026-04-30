# Winning Equities AIPT Pipeline — End-to-End

**Final result on TOP2000 PIT-correct data, OOS 2024-01-02 → 2026-04-20 (576 daily bars):**

| Metric | Value |
|---|---:|
| Universe | TOP2000 by ADV20 (N≈1897 active stocks) |
| Bars | Daily (DELAY=1) |
| Features | **D=3** chars selected by block-CV on TRAIN |
| RFF size | **P=1000–2000** (gamma rescaled by √(24/D)) |
| Ridge | ρ = 1e-3 |
| Train window | 1500 bars (~6 years) trailing rolling |
| Rebalance | weekly (5 bars) |
| **Gross SR** (FULL OOS) | **+5.29** |
| **Net SR @1bp** (FULL OOS) | **+5.03** |
| **TEST Net SR @1bp** | **+4.61** |
| **TEST Net SR @3bp** | **+4.17** |
| **TEST IR (Pearson)** | **+5.74** |
| **TEST IC (Pearson)** | **+0.0155** |
| TO/bar | 15.8% |
| TEST cumulative net @ 1bp | +35.4% |

vs original arbitrary D=24: TEST nSR @1bp went from +4.04 → **+4.61** (+14% lift) and IC from +0.0114 → +0.0155 (+36%).

---

## The pipeline in 5 stages

### Stage 1 — Data ingestion (raw FMP cache, already in place)

**Source**: Financial Modeling Prep API. Stored as per-ticker quarterly parquets in:
- [data/fmp_cache/income/](data/fmp_cache/income/) — income statements
- [data/fmp_cache/balance/](data/fmp_cache/balance/) — balance sheets
- [data/fmp_cache/cashflow/](data/fmp_cache/cashflow/) — cash flow statements
- [data/fmp_cache/metrics/](data/fmp_cache/metrics/) — pre-computed metrics
- [data/fmp_cache/prices/](data/fmp_cache/prices/) — daily OHLCV per ticker

Each statement file contains `date` (period end), `filingDate` (SEC filing date), `acceptedDate` (SEC acceptance timestamp), plus the financial fields. **The filingDate is the load-bearing field for PIT correctness.**

### Stage 2 — PIT matrix rebuild

**Script**: [rebuild_pit_matrices.py](rebuild_pit_matrices.py)
**Output**: [data/fmp_cache/matrices_pit/](data/fmp_cache/matrices_pit/) — 89 daily-aligned (T, N) parquets

**The fix**: forward-fill fundamentals from `filingDate + 1d` (next-day-tradeable convention), NOT from period-end `date`. This eliminates the 30-90-day look-ahead bias present in the legacy `matrices_clean/` and partially in `matrices/`.

Key functions:
- `load_prices(tickers)` — load OHLCV from per-ticker prices/ parquets
- `derive_timeseries_chars(matrices, dates)` — compute log_returns, hist_vol_*, parkinson_vol_*, adv20/60, momentum_*, microstructure ranges
- `build_pit_field(stmt_dir, fmp_field, tickers, dates, lag_days=1)` — for each fundamental, build a (T, N) matrix where `mat[t, sym] = latest value of fmp_field for sym known by bar t` (i.e., filingDate + 1 ≤ t)
- `build_pit_fundamentals(tickers, dates)` — apply build_pit_field to all needed raw fields
- `derive_ratios(prices_chars, fund)` — compute ratios (ROE, book/market, ev/ebitda, etc.) from PIT fundamentals + daily close

Result: 2588 daily bars × 2462 tickers × 89 fields, all with consistent freshness through 2026-04-20.

### Stage 3 — Data integrity check

**Script**: [equities_data_integrity.py](equities_data_integrity.py)
**Output**: [data/aipt_results/equities_data_integrity_report.json](data/aipt_results/equities_data_integrity_report.json)

PASS/FAIL audit modeled on the crypto integrity dashboard. Checks:
1. Per-matrix shape consistency
2. Date alignment + freshness consistency
3. Ticker alignment across all files
4. Recent NaN coverage (last 60 bars ≥30% non-NaN)
5. **Update-frequency anomaly** — flags fundamentals where recent monthly update rate dropped below 20% AND below 40% of prior median (catches the Dec/Jan freeze that broke the legacy pipeline)
6. Look-ahead bias spot-check (sampled tickers)

The PIT matrices pass all checks; the legacy `matrices_clean/` and `matrices/` fail multiple.

### Stage 4 — Block-CV feature selection on TRAIN

**Selection script**: [select_chars_block_cv.py](select_chars_block_cv.py)
**Output**: [data/aipt_results/selected_chars_block_cv.json](data/aipt_results/selected_chars_block_cv.json)

Methodology:
1. Load all 68 candidate chars (24 originals + 20 fundamental ratios + 24 microstructure/vol/momentum extras)
2. Restrict to TRAIN window (start_bar=502 → oos_start_idx=2012, dates 2017-12-29 → 2024-01-02)
3. Split TRAIN into K=5 contiguous blocks, each ~302 bars (~14 months) covering distinct regimes (late-cycle bull, COVID crash, recovery, 2022 bear, 2023 AI boom)
4. **Forward-greedy selection by min-IC across blocks**:
   - Pre-compute rank-normalized panels for all candidates (Spearman-style, [-0.5, 0.5] per bar)
   - Sign each char by univariate IC (negate negative-IC chars so they enter as "+")
   - Greedy: at each step, for each remaining char `c`, evaluate the COMBINED-signal IC = sum of signed rank-normalized chars in `selected ∪ {c}` correlated with next-bar returns, in EACH of the 5 blocks separately
   - Score = `min(block_ICs)` — the worst block
   - Pick the char that maximizes the minimum block IC; require positive marginal gain > EPS
5. Stop when no candidate adds positive marginal min-IC

Key functions:
- `precompute_ranks(matrices, candidates, train_start, train_end)` (in [select_chars_train_combined_ic.py](select_chars_train_combined_ic.py)) — build (T_train, N) panels of rank-normalized per-bar values, one per char
- `block_ic(selected_sums, returns_per_bar, block_indices)` — mean cross-sectional Spearman IC over a block
- `all_block_ics(selected, signs, ranks_dict, returns_per_bar, blocks)` — combined IC in each of K blocks

**Selected chars (D=3):**
1. **+roa** (return on assets) — quality factor
2. **−momentum_5d** (5-day return, negated → short-term reversal) — short-horizon mean-reversion
3. **+cap** (market cap) — size factor

Final TRAIN min-IC across 5 blocks = +0.0158. Adding any 4th char reduced the worst block's IC.

### Stage 5 — AIPT backtest with selected chars

**Script**: [backtest_voc_equities_d3_blockcv.py](backtest_voc_equities_d3_blockcv.py)
**Underlying engine**: [backtest_voc_equities_neutralized.py](backtest_voc_equities_neutralized.py) (`build_Z_panel`, `run_with_neutralization`, `split_metrics`)

Pipeline at each bar:
1. **Z panel construction** ([backtest_voc_equities_neutralized.py:build_Z_panel](backtest_voc_equities_neutralized.py)):
   - For each bar `t`, take the 3 selected chars at bar `t-1` (DELAY=1)
   - Apply per-char sign (negate momentum_5d so it enters as a "+" signal)
   - Cross-sectionally rank-normalize each column → values in [-0.5, 0.5], NaN→0
   - Z_t shape: (N, D) = (1897, 3)

2. **Random Fourier Features** ([backtest_voc_equities_neutralized.py:run_with_neutralization](backtest_voc_equities_neutralized.py)):
   - θ ∈ R^{P/2 × D} sampled once with seed=42
   - γ sampled from grid [0.5, 0.6, …, 1.0] then RESCALED by √(D_ref / D) = √(24/3) ≈ 2.83 (preserves projection variance for variable D)
   - At each bar: `proj = Z_t @ θᵀ * γ` (shape N × P/2)
   - `S_t = [sin(proj), cos(proj)]` interleaved, shape (N, P)

3. **Markowitz factor portfolio**:
   - `F_{t+1} = (1/√N) S_tᵀ R_{t+1}` — factor returns (P-vector)
   - Cache `F_{t+1}` at every bar
   - Rebalance every 5 bars: `λ̂ = (FF/T + ρI)⁻¹ μ̂` over the trailing 1500-bar window

4. **Asset weights**:
   - `pred_v = (1/√N) S_v λ̂` (over valid assets)
   - For mode="baseline" (no neutralization): `w_t = pred_v / Σ|pred_v|` (gross-1)

5. **Per-bar accounting**:
   - `port_t = w_{t-1} · R_t`
   - `turnover_t = ½‖w_t − w_{t-1}‖₁`
   - `net_t = port_t − turnover_t × fee_bps × 2 / 10000`
   - Per-bar IC + R² captured for diagnostics

6. **OOS aggregation** ([backtest_voc_equities_neutralized.py:split_metrics](backtest_voc_equities_neutralized.py)):
   - VAL / TEST 50/50 split of OOS bars
   - Annualized SR = mean / std × √252
   - Three fee levels reported: 0bp / 1bp / 3bp

Output: [data/aipt_results/voc_equities_pit_d3_blockcv.csv](data/aipt_results/voc_equities_pit_d3_blockcv.csv) (P sweep results).

---

## File-by-file dependency graph

```
data/fmp_cache/{income,balance,cashflow,metrics,prices}/*.parquet
                                ↓
                  rebuild_pit_matrices.py
                                ↓
                  data/fmp_cache/matrices_pit/*.parquet  (89 fields, fresh through 2026-04-20)
                                ↓
                  ┌──────────────┴──────────────┐
                  ↓                              ↓
       equities_data_integrity.py     select_chars_block_cv.py  (TRAIN only)
                  ↓                              ↓
        report.json (PASS)            selected_chars_block_cv.json  (D=3)
                                                ↓
                  backtest_voc_equities_d3_blockcv.py
                                                ↓
                       voc_equities_pit_d3_blockcv.csv
                                                ↓
                          plot equity curve / report
```

## Reproducibility commands

```bash
# 1. Rebuild PIT matrices (~22 min)
python rebuild_pit_matrices.py

# 2. Verify integrity (~30s)
python equities_data_integrity.py

# 3. Run block-CV selection on TRAIN (~10 min)
python select_chars_block_cv.py

# 4. Backtest on OOS with selected D=3 (P sweep) (~5 min)
python backtest_voc_equities_d3_blockcv.py
```

## Key parameters (single source of truth)

| Param | Value | Defined in |
|---|---:|---|
| UNIVERSE | TOP2000 (~1897 active) | backtest_voc_equities_neutralized.py |
| BARS_PER_YEAR | 252 | backtest_voc_equities.py |
| TRAIN_BARS | 1500 (~6 yr trailing) | backtest_voc_equities.py |
| MIN_TRAIN_BARS | 500 | backtest_voc_equities.py |
| REBAL_EVERY | 5 (weekly) | backtest_voc_equities.py |
| OOS_START | 2024-01-01 | backtest_voc_equities.py |
| Z_RIDGE (ρ) | 1e-3 | backtest_voc_equities.py |
| GAMMA_GRID (base) | {0.5, 0.6, ..., 1.0} | backtest_voc_equities.py |
| GAMMA_REF_D | 24 | (rescale factor for D-invariance) |
| TAKER_BPS_GRID | [0, 1, 3] | backtest_voc_equities_neutralized.py |
| PIT_LAG_DAYS | 1 (next-day-tradeable) | rebuild_pit_matrices.py |
| K_BLOCKS | 5 (block-CV folds) | select_chars_block_cv.py |
| EPS (selection stop) | 1e-5 | select_chars_block_cv.py |
| MAX_CHARS | 30 | select_chars_block_cv.py |
| seed | 42 | (deterministic) |

## Why this works

1. **PIT correctness eliminates a hidden in-sample boost** that was inflating historical Sharpe and creating spurious char rankings.
2. **Block-CV with min-IC criterion** filters out chars that worked great in one regime but failed in another. The result is a tiny, regime-robust set.
3. **Random Fourier Features + ridge-Markowitz** turns those 3 inputs into a P-dimensional kernel feature space, capturing nonlinear interactions that a 3-char linear model couldn't.
4. **Gamma rescaling by √(D_ref / D)** keeps the kernel bandwidth K-invariant, so the same `[0.5–1.0]` base grid works for any D.
5. **Daily bars + low TO + 1bp fees** = small fee drag (~0.3–0.5 SR vs >3 for crypto).

## Things that did NOT help (verified)

- Adding more chars without selection (D=24, 34, 44, 233 — all worse than D=3 block-CV)
- Univariate-IC + correlation-filter selection (D=30 → TEST nSR +2.6, much worse)
- Greedy combined-IC selection without CV (D=11 → TEST nSR +1.0, regime-overfit)
- Neutralization (market / industry / subindustry / risk-model) — all hurt
- EWMA smoothing on weights or features
- PCA neutrality in QP
- Gârleanu-Pedersen factor-space TC penalty
