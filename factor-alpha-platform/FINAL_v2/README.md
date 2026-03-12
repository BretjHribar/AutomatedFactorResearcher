# FINAL_v2 — V9b: 5-Timeframe + Cross-Asset Adaptive Net Strategy

**Date**: 2026-03-11  
**Version**: V9b (Production Candidate)  
**Status**: Validated on fresh Binance Futures data  
**Predecessor**: V8 (FINAL_v1, Sharpe +9.93) → V9b (Sharpe +14.02)

---

## Performance Summary

| Symbol | Annualized Sharpe | Negative Months | Total PnL (bps) |
|--------|-------------------|-----------------|-----------------|
| BTCUSDT | +8.08 | 0 / 10 | +22,071 |
| ETHUSDT | +8.98 | 0 / 10 | +30,924 |
| SOLUSDT | +9.65 | 0 / 10 | +53,124 |
| BNBUSDT | +7.42 | 0 / 10 | +24,431 |
| DOGEUSDT | +9.09 | 0 / 10 | +52,854 |
| **Equal-Weighted Portfolio** | **+14.02** | **0 / 50** | **+183,404** |

- **OOS Period**: June 2024 – March 2025 (10 monthly walk-forward folds)
- **Fee Model**: 3 bps on position changes only (Binance USDT-M perpetual futures taker)
- **Data Source**: `fapi.binance.com/fapi/v1/klines` (Binance Futures)
- **Annualization**: `mean(daily_pnl) / std(daily_pnl) * sqrt(365)`

---

## What Changed from V8 (FINAL_v1)

| Feature | V8 | V9b |
|---------|----|----|
| Timeframes | 1H + 4H | **1H + 2H + 4H + 8H + 12H** |
| Cross-asset signals | None | **BTC + ETH + SOL as factors** |
| Train window | Fixed 8 months | **Optimized per fold (6/8/10 months)** |
| Signal count | ~240 | **~340+** |
| Parameter grid | 480 configs | **~5,000 configs (3 train × wider sweep)** |
| Collective Sharpe | +9.93 | **+14.02** |
| Negative months | 2 (BNB) | **0 across all 50 fold×symbol** |
| Audit fixes | — | ✅ Boundary fix, param logging |

---

## Directory Contents

```
FINAL_v2/
├── README.md                          ← This file
│
├── scripts/
│   ├── univariate_hf_v9b_mtf.py       ← PRODUCTION STRATEGY CODE
│   ├── download_futures_15m.py         ← Data download script (fapi.binance.com)
│   └── v9b_rerun.py                    ← Validation run script
│
├── results/
│   └── v9b_output.txt                  ← Raw output from validation run
│
└── data/
    └── v9b_pnl.parquet                 ← Bar-level PnL series (5 symbols)
```

---

## Key Design Decisions

1. **5 timeframes** — 2H and 8H fill resolution gaps; 12H captures daily/session patterns
2. **Cross-asset signals** — BTC/ETH/SOL momentum + mean reversion + relative strength as features for all symbols
3. **Per-fold train window** — 6m preferred in volatile regimes (BTC), 10m in stable (SOL later folds)
4. **Audit Fix #2** — Train/test boundary is now exclusive (no 1-bar overlap)
5. **Audit Fix #3** — Selected parameters and alpha names logged per fold

---

## How to Reproduce

```bash
# 1. Download fresh 15m futures data
python scripts/download_futures_15m.py

# 2. Run the full walk-forward backtest
python scripts/v9b_rerun.py

# 3. Or run the production module directly
python scripts/univariate_hf_v9b_mtf.py
```

---

*Prepared for audit review, 2026-03-11*
