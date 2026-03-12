# FINAL_v1 — Binance Futures 1H Multi-Timeframe Adaptive Net Strategy

**Date**: 2026-03-11  
**Version**: V8 (Production Candidate)  
**Status**: Validated on fresh Binance Futures data

---

## Performance Summary

| Symbol | Annualized Sharpe | Negative Months | Total PnL (bps) |
|--------|-------------------|-----------------|-----------------|
| BTCUSDT | +5.31 | 0 / 10 | +16,608 |
| ETHUSDT | +6.06 | 0 / 10 | +23,548 |
| SOLUSDT | +6.05 | 0 / 10 | +36,464 |
| BNBUSDT | +4.77 | 2 / 10 | +18,374 |
| DOGEUSDT | +7.06 | 0 / 10 | +48,000 |
| **Equal-Weighted Portfolio** | **+9.93** | — | — |

- **OOS Period**: June 2024 – March 2025 (10 monthly walk-forward folds)
- **Fee Model**: 3 bps on position changes only (Binance USDT-M perpetual futures taker)
- **Data Source**: `fapi.binance.com/fapi/v1/klines` (Binance Futures — freshly downloaded 2026-03-11)
- **Annualization**: `mean(daily_pnl) / std(daily_pnl) * sqrt(365)`

---

## Directory Contents

```
FINAL_v1/
├── README.md                          ← This file
├── v8_production_writeup.md           ← Full technical write-up for quant/dev review
├── v8_performance_chart.png           ← Cumulative PnL chart (individual + collective)
│
├── scripts/
│   ├── univariate_hf_v8_mtf.py        ← PRODUCTION STRATEGY CODE (main deliverable)
│   ├── download_futures_15m.py         ← Data download script (fapi.binance.com)
│   └── v8_rerun_futures.py             ← Validation run script (fresh data)
│
├── results/
│   ├── v8_rerun_output.txt             ← Raw output from validation run
│   └── v8_performance_chart.png        ← Performance visualization
│
└── data/
    └── v8_fresh_futures_pnl.parquet    ← Bar-level PnL series (5 symbols × ~7200 bars)
```

---

## File Descriptions

### `v8_production_writeup.md`
Comprehensive technical document for review by Head Quant and Head Dev. Contains:
- Architecture overview with data flow diagram
- Signal taxonomy (10 categories, ~240 signals across 1H + 4H timeframes)
- Portfolio construction methodology (Isichenko-style adaptive net weighting)
- Walk-forward validation design (8-month train, 1-month OOS test)
- Fee model deep-dive with mathematical verification
- **11 identified bugs/concerns** with severity ratings (CRITICAL → INFO)
- Anti-lookahead audit (all components verified ✅)
- Production deployment recommendations

### `scripts/univariate_hf_v8_mtf.py`
The main production strategy code. Self-contained Python module (~480 lines) with:
- `build_mtf_alphas_safe()` — Generates ~240 alpha signals from 1H and 4H data
- `eval_alpha()` — Evaluates individual alpha signal quality
- `select_orthogonal()` — Greedy orthogonal alpha selection
- `strategy_adaptive_net()` — Core portfolio construction (Isichenko-style)
- `walk_forward()` — Complete walk-forward backtest engine

### `scripts/download_futures_15m.py`
Downloads 15m klines from Binance Futures API (`fapi.binance.com`) for 5 symbols.
Run with: `python download_futures_15m.py`

### `scripts/v8_rerun_futures.py`
Validation script that runs the full V8 strategy on freshly downloaded data.
Run with: `python v8_rerun_futures.py`

### `data/v8_fresh_futures_pnl.parquet`
Bar-level PnL time series for all 5 symbols. DataFrame with DatetimeIndex and 5 columns (one per symbol). Values are fractional returns (e.g., 0.001 = 10 bps). Can be loaded with:
```python
import pandas as pd
pnl = pd.read_parquet('data/v8_fresh_futures_pnl.parquet')
```

---

## How to Reproduce

```bash
# 1. Download fresh 15m futures data
python scripts/download_futures_15m.py

# 2. Run the full walk-forward backtest
python scripts/v8_rerun_futures.py

# 3. Or run the production module directly
python scripts/univariate_hf_v8_mtf.py
```

---

## Key Design Decisions

1. **1H frequency** — 3 bps fees are negligible vs average 1H moves (~50 bps); 15m was too fee-intensive
2. **Multi-timeframe** — 4H signals provide higher-timeframe context; shifted by 1 period to prevent lookahead
3. **Adaptive net weighting** — Only positive-return alphas get weight; auto-adapts to regime changes
4. **Sign-based positions** — `sign(combined_signal)` not magnitude; magnitude-weighted (V7) degraded performance
5. **Fee on position changes only** — Correct for perpetual futures (holding costs are via funding, not commissions)
6. **Walk-forward** — All parameters selected on train; test is truly out-of-sample

---

## Known Limitations

See Section 5 of `v8_production_writeup.md` for full details. Key items:
- No funding rate model (est. neutral for high-frequency direction changes)
- No slippage model (est. <1 bps for BTC/ETH/SOL/BNB; 1-2 bps for DOGE)
- Test-fold adaptive weights have cold-start warmup period
- 480 parameter configurations per fold carry mild overfitting risk

---

*Prepared for audit review, 2026-03-11*
