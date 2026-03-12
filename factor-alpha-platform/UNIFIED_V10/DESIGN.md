# UNIFIED_V10 — Design Document

## Motivation

The previous system had **two separate code paths**: a vectorized pandas backtest and an incremental streaming live trader. Despite using the same alpha functions, they diverged due to subtle differences in NaN handling, warmup timing, and weight computation. This violated the Dubno/Goldman principle:

> *"Backtesting, paper trading, and live trading are three worlds to place the trading system in. The system should not be able to tell which world it is in. Backtesting on recorded data should produce the same results as the live run."*
> — Michael Dubno, CTO of Goldman Sachs

**UNIFIED_V10 enforces this principle: one code path, one engine, identical results.**

---

## Architecture

```
UNIFIED_V10/
├── engine.py          # StreamingEngine — THE core (single code path)
├── alphas.py          # Alpha computation functions (shared, no divergence)
├── backtest.py        # Replay bars through engine for comparison testing
├── walk_forward.py    # Walk-forward backtest using frozen params
├── live.py            # Live paper trader using the same engine
├── config.py          # Constants and paths
├── DESIGN.md          # This file
└── tests/
    ├── test_no_lookahead.py    # Lookahead bias detection (4 tests)
    ├── test_equivalence.py     # Streaming vs vectorized equivalence
    └── test_data_integrity.py  # Data quality validation
```

---

## Core Design: `StreamingEngine.on_bar(bar)`

### The Only Entry Point

Every bar — whether replayed from history or received live from Binance WebSocket — goes through `on_bar()`. There is no other way to generate signals.

### Signal Timing

```
on_bar(bar_T) is called when bar_T is FULLY CLOSED

Timeline:
  close[T-1] ──────── close[T] ──────── close[T+1]
                         │                    │
                    on_bar(T)            on_bar(T+1)
                         │                    │
                    compute alpha[T]     compute alpha[T+1]
                    direction = sign(    PnL realized:
                      combined(alpha[T]    prev_dir * return[T+1]
                      * weights[T]))
                    hold direction
                    from close[T]→close[T+1]
```

### PnL Accounting

At `on_bar(T)`:
```
current_return = close[T] / close[T-1] - 1
realized_pnl = prev_direction × current_return
fee = FEE_FRAC × |new_direction - prev_direction|
net_pnl = realized_pnl - fee
```

### Factor Returns (Adaptive Weights)

```
factor_return[T] = sign(alpha[T-1]) × return[T]
```

Uses **previous bar's** alpha value (stored in `self._prev_alpha_values`). This is equivalent to the vectorized backtest's `.shift(1)` on the alpha matrix.

---

## Critical Design Decisions

### 1. Alpha/Factor Returns Computed On EVERY Bar

Even during warmup (before producing signals), the engine computes alpha values and accumulates factor returns. This ensures the factor return history is identical to the vectorized version from bar 1.

**Why this matters:** The old streaming engine skipped alpha computation during warmup, causing the factor return deque to have fewer entries than the vectorized rolling window. This led to different weights → different directions → different PnL.

### 2. NaN Handling Matches Vectorized Exactly

| Scenario | Behavior |
|----------|----------|
| Alpha value is NaN (insufficient data) | Preserved as NaN in `_prev_alpha_values` |
| NaN alpha in factor return | `sign(NaN) × return = NaN` → excluded from rolling mean |
| NaN weight (rolling min_periods not met) | Treated as 0 in combined signal |
| NaN combined signal | Direction = 0 (flat) |

**Why this matters:** The vectorized `rolling(120, min_periods=100).mean()` skips NaN entries when counting non-NaN values toward `min_periods`. If we converted NaN to 0, the rolling mean would include zeros, producing different weight values.

### 3. First Factor Return at Bar Index 2

In the vectorized backtest:
- `pct_change()` produces NaN at index 0
- `X.shift(1)` produces NaN at index 0 (and index 1 has alpha from index 0, which may be NaN for some alphas)
- First valid factor return is at index 2

The streaming engine skips factor return computation for `bar_count < 3` to match this timing exactly.

### 4. Rolling Weight Computation Uses Pandas

The weight computation uses `pd.DataFrame(deque).rolling(lookback, min_periods=min(100, lookback)).mean()` — **exactly** the same pandas operation as the vectorized backtest. Not a simple `mean()` over the deque.

**Why this matters:** `deque.mean()` includes all entries equally. `rolling.mean()` with `min_periods` may return NaN if there aren't enough non-NaN entries. This subtle difference caused the initial 37.7% direction disagreement.

### 5. Buffer Rebuilds DataFrame Each Bar

The engine maintains a growing list of bar dicts. At each `on_bar()`, it creates `pd.DataFrame(self._bars)` and runs `build_1h_alphas()` on it. This is O(N) per bar and O(N²) total for a backtest.

**Why this is acceptable:**
- For live trading: only called once per hour → negligible
- For backtesting: ~1min per 300 bars on synthetic data. Slow but correct.
- The vectorized backtest exists for fast backtesting; the streaming engine is the **reference implementation** for correctness verification.

### 6. Fee Timing

Fees are charged at the moment of position change:
```
fee = 3bps × |new_direction - prev_direction|
```

Possible values: 0 (no change), 3bps (flat→long or reverse from short→long is 6bps).

---

## Verification Results

### Equivalence Test (CRITICAL)
```
Direction agreement:       100.000000% (199 bars)
Full direction match:      100.000000%
Max cumulative PnL diff:   0.000000 bps
```

### Lookahead Bias Tests
```
1H alpha truncation:       1,224 combos tested, 0 violations ✓
HTF alpha truncation:      156 combos tested, 0 violations ✓
Engine truncation:         4 time points, 0 violations ✓
Future data corruption:    300 bars verified, 0 violations ✓
```

### Determinism
```
Same input → same output:  400 bars × 2 runs, 0 mismatches ✓
```

### Data Integrity
```
10 symbols × 7 checks:    70/70 passed ✓
```

---

## Alpha Library (~340+ signals)

### 1H Alphas (~220)
| Category | Count | Examples |
|----------|-------|---------|
| Mean Reversion | 65+ | `mr_10`, `logrev_5`, `dstd_8`, `vwap_mr_20`, `ema_mr_10` |
| Momentum | 27+ | `mom_12`, `emax_5_20`, `breakout_48` |
| Trend Direction | 4 | `trend_6`, `trend_12`, `trend_24`, `trend_48` |
| Decay Momentum | 50 | `dec0.9_d6_cp`, `dec0.95_d12_tbr` |
| Vol-Conditioned | 12 | `lovol_mr_10`, `hivol_mom_8`, `vs_mr_20` |
| Volume/Microstructure | 15 | `obv_20`, `tbr_10`, `timb_5`, `vwret_10` |
| Candle | 8 | `body_10`, `reject_5`, `atr_mr_10` |
| Regime | 8 | `regime_mom_24`, `regime_mr_48` |
| Cross-Timeframe | 3 | `mtf_agree`, `trend_pullback`, `trend_pullback_20` |
| Technical | 13 | `bb_20`, `rsi_14`, `stoch_14`, `cci_14` |
| Acceleration | 6 | `accel_5`, `accel_8`, `vol_accel` |

### HTF Alphas (~100)
Computed on 2H, 4H, 8H, 12H timeframes with `shift_n=1` (no lookahead):
- Mean reversion, momentum, breakout, decay momentum, volume z-score

### Cross-Asset Alphas (~24)
Using BTC, ETH, SOL as market factors:
- Factor momentum, factor mean reversion, relative strength

---

## Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `FEE_BPS` | 3 | Binance futures commission (bps) |
| `MIN_WARMUP_BARS` | 200 | (legacy, warmup now handled by rolling min_periods) |
| `lookback` | 120 | Rolling window for adaptive weight computation |
| `phl` | 1 | Position halflife (1 = no smoothing) |
| `buffer_size` | 2000 | Max bars in rolling buffer |

---

## References

- **Dubno/Goldman:** "Backtesting on recorded data should produce the same results as the live run." (Maxdama, p.27)
- **Dubno:** "Even in backtesting, data is only fed to the system one at a time."
- **Dubno:** "No function can query external stats."
- **Isichenko:** Adaptive net factor return weighting with non-negative constraints
