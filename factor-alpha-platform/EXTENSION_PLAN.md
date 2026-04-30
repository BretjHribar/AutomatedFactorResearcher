# Equities Backtest Extension Plan — Survivorship-Corrected, 2010+

**Goal**: Extend our equity AIPT backtest to use 16 years of data (2010-2026) with **proper survivorship correction** (include delisted tickers), addressing the largest open Isichenko violation.

**Reference**: Isichenko §2.1.2 — "running quant research on assets alive today but ignoring those no longer active" is "another kind of buggy lookahead".

---

## Status legend
- [ ] not started · [~] in progress · [x] done · [!] blocked / failed

## Steps

### Phase A — Discovery & Validation (cheap, ~1 hour)

- [x] **Step 1** — Enumerate FMP delisted universe → filter US common stocks delisted 2010+
  - Script: `extension/01_enumerate_delisted.py`
  - Output: `data/fmp_cache/delisted_universe.json` (4,202 entries)
  - **Result**: 8,942 total delisted; **4,202 US common stocks delisted 2010+**
    - NASDAQ 2,857 · NYSE 1,244 · AMEX 101
    - Heavy concentration 2021-2025 (3,500+); pre-2016 sparse (~10/year — FMP's delist DB starts thin)
    - 2022-2023 peak: 748 + 1,058 (likely SPAC unwinds + rate cycle bankruptcies)
  - **Implication**: full universe = ~12K (8K active + 4K delisted). Our TOP2000 backtest likely misses 100-300 failed names per year in 2022-2024 OOS

- [x] **Step 6 (early)** — Restate-detection probe
  - Script: `extension/02_restate_probe.py`
  - Cache age: 58.5 days
  - **Result**: 75/10400 = **0.72% of (ticker, quarter, field) values changed**
    - `totalDebt`: **5.62%** ← high (vendor reclassifies debt components)
    - `ebitda`: 1.75% (computed metric, formula drift)
    - All other fields: < 0.5% (effectively PIT-acceptable)
  - **Mitigation**: avoid `totalDebt`-derived ratios; prefer `long_term_debt + short_term_debt` directly. Or add 30-60 day extra lag on filingDate.

### Phase B — Bulk Data Fetch (long, ~6-12 hours)

- [x] **Step 2** — Bulk fetch prices for augmented universe (2010-01-01 →)
  - Script: `extension/03_bulk_prices.py`
  - **Result**: 69.1 min, 6,694 fetched, 6 failed (PyLong int64 overflow on MULN — known FMP quirk; rest empty/non-existent)
  - Total `prices/` dir now has **12,121 parquets** (we had ~7,500 from prior fetches plus the new delisted)
  - Throughput: ~98/min steady (no 429s — your plan handles 100/min easily)

- [ ] **Step 3** — Bulk fetch fundamentals for augmented universe
  - Script: `extension/04_bulk_fundamentals.py`
  - For each ticker: income, balance, cashflow, metrics — quarterly, limit=80 (~20yr)
  - Cache to `data/fmp_cache/{income,balance,cashflow,metrics}/`

### Phase C — Build PIT-Correct Panel

- [ ] **Step 4** — Build PIT universe membership
  - Script: `extension/05_pit_universe.py`
  - Per-bar membership: `ticker is investable iff ipoDate ≤ t AND (delistedDate is null OR delistedDate > t)`
  - Output: `data/fmp_cache/universes_pit/membership.parquet` (T × N boolean)

- [ ] **Step 5** — Re-run PIT matrix rebuild on extended date range
  - Use existing `rebuild_pit_matrices.py`, change start to 2010-01-01
  - Output: `data/fmp_cache/matrices_pit_v2/` (won't overwrite v1)

### Phase D — Validation Gates (must pass before backtest)

- [ ] **D.1** — Integrity dashboard PASSes on extended data
  - Run `equities_data_integrity.py` against `matrices_pit_v2/`
- [ ] **D.2** — Spot-check: known bankruptcy (e.g., Sears 2018, GameStop saga, Toys R Us)
  - Verify panel shows declining price + zero terminal value
- [ ] **D.3** — Spot-check: known acquisition (e.g., LinkedIn 2016, T-Mobile/Sprint 2020)
  - Verify ticker disappears at acquisition date, not before
- [ ] **D.4** — Restate diff repeated after 1 week → quantify silent restate rate

### Phase E — Re-evaluate Strategies

- [ ] **Step 7** — Re-run `proper_evaluation.py` on extended history
  - 12-16 years of OOS instead of 6
  - Sharpe SE ~1.4× tighter → CIs 1.7→1.2 wide
  - Pairwise differences ~0.6 SR → ~0.4 SR resolvable
- [ ] **E.2** — Walk-forward COVID stress test (2020-Q1 isolated window)
- [ ] **E.3** — Compare extended D=24 vs D=3 vs D=44 — does the picture change?

---

## Risk / cost estimates

| Phase | API calls | Wall time | Disk |
|---|---:|---:|---:|
| A | ~600 | ~30 min | <10 MB |
| B | ~10K-20K | ~3-6 hr | ~5 GB |
| C | 0 | ~30 min | ~5 GB (matrices) |
| D | 0 | ~30 min | n/a |
| E | 0 | ~2-3 hr (recompute) | n/a |

## Known limitations even after this work

1. **Restatement bias** — FMP returns CURRENT financial values; 2010 ROE may have been restated. Mitigation: extra 90-day filing-date lag.
2. **OTC selection bias** — failed micro-caps may still be underrepresented even with delisted enumeration.
3. **Acquired vs bankrupt** — both look "delisted" but have very different terminal returns. Need EDGAR Form 25 codes for distinction (later).

## Output documents

- This file (`EXTENSION_PLAN.md`) — checklist + status
- `extension/run_log.md` — append-only timestamped log
- `data/fmp_cache/delisted_universe.json` — filtered delisted list
- `data/fmp_cache/matrices_pit_v2/` — extended PIT matrices
- `data/aipt_results/proper_eval_extended/` — re-evaluation results
