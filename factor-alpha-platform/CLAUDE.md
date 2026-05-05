# Project conventions

## File organization

- **`prod/`** — live trading code (moc_trader, kucoin_trader, configs, state, logs). Production paths only.
- **`src/`** — reusable library code (pipeline, operators, simulation, portfolio).
- **`tools/`** — utilities and ad-hoc scripts that fit a clear category (`tools/diagnostics/`, `tools/plot/`, etc.).
- **`data/`** — cached matrices, universes, and durable result artifacts (PNGs, CSVs) that the team will reference.
- **`archive/`** — superseded but kept-for-reference scripts. Never imported by active code.
- **`experiments/`** — **NEW: any new file that doesn't have an obvious home goes here**, including:
  - one-off analysis scripts
  - hypothesis-test scripts
  - exploratory results (CSV, PNG, MD writeups)
  - research notes
  - prototype tools that haven't earned a permanent place yet
  - anything we'd otherwise junk-drawer into the repo root or `data/`

If you're about to create a new `.py` or `.md` and you don't know where it goes, default to `experiments/`. Promote to `tools/` or `src/` only after it proves durable and reusable.

## Universe / data conventions

- KuCoin universe parquets at `data/kucoin_cache/universes/KUCOIN_TOP{N}_4h.parquet` are **constant top-N by ADV with 1-year minimum-history filter**, rebalanced every 20 days (per Isichenko-style construction). Legacy `cov_full > 0.3` versions saved as `*.legacy_cov.parquet`.
- `eval_alpha.py` has `COVERAGE_CUTOFF = 0.0` — disabled because the proper universe parquet handles per-bar membership directly.
- Splits: TRAIN 2023-09-01 → 2025-09-01, VAL 2025-09-01 → 2026-01-01, TEST 2026-01-01 → present.

## Turnover convention

- All TO computed as `Σ|Δw|` (full two-sided): a 50/50 long-short book that flips sign = 200% TO.
- Fees applied as `bps × Σ|Δw|` directly (no `× 2` correction anywhere).
