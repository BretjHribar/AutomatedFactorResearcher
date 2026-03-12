# Paper Trader: V9b 5-Timeframe Adaptive Net — Production Design & Implementation Guide

## Overview

Live paper trading system for the V9b strategy on Binance USDT-M Perpetual Futures.
Connects to `fstream.binance.com` via WebSocket, receives 15m klines, resamples to
1H/2H/4H/8H/12H, computes ~340 alpha signals + cross-asset features, and logs all
signals, trades, and PnL to JSONL/CSV.

---

## Architecture

```
┌─────────────────────────────────────┐
│    Binance Futures WebSocket         │
│  wss://fstream.binance.com/stream   │
│  5 symbols × 15m kline streams      │
└──────────────┬──────────────────────┘
               │ on x=true (bar closed)
               ▼
┌─────────────────────────────────────┐
│    KlineBuffer (per symbol)          │
│  - 15m deque (2000 bars)            │
│  - Resamples: 1H, 2H, 4H, 8H, 12H │
│  - Bootstraps from REST on startup  │
└──────────────┬──────────────────────┘
               │ every 1H bar close
               ▼
┌─────────────────────────────────────┐
│    SignalEngine (per symbol)         │
│  build_1h_alphas()  (220 signals)   │
│  build_htf_signals() (4 HTFs × 25) │
│  build_cross_asset_signals()  (24)  │
│  → adaptive_net_weights()           │
│  → direction = sign(combined)       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│    PositionManager + Logger          │
│  - Paper trade tracking             │  
│  - PnL computation                  │
│  - JSONL signal & trade logs        │
└─────────────────────────────────────┘
```

---

## Shift/Lookahead Semantics

### Backtest vs Live

| Component | Backtest | Live |
|-----------|----------|------|
| 1H alpha matrix | `.shift(1)` applied | **NO shift** — implicit from event timing |
| HTF (2H/4H/8H/12H) close | `.shift(1)` on HTF data | **SAME** — only use previous complete HTF bar |
| Execution | signal[t] → PnL[t] | signal at bar close → execute at next bar open |

### Why No shift(1) in Live Mode

In the backtest, `alpha_df.shift(1)` ensures signal at index `t` was computed from
data up to `t-1`. In live mode, this shift is **natural**:

```
14:00 UTC — 1H bar [13:00–14:00) closes
         → Compute signals from all data up to 13:00-close
         → Signal predicts what to do for [14:00–15:00)
         → No shift needed — we physically can't use future data
```

### HTF Signal Safety

HTF signals STILL use `.shift(1)` even in live mode, ensuring we only use the
**previous** complete HTF bar (matching backtest behavior).

---

## Warmup Requirements

| Component | Bars Needed | Time |
|-----------|-------------|------|
| breakout_360 (1H) | 360 | 15 days |
| h12_brk_120 (12H) | 120×12 = 1440h | 60 days |
| Adaptive weights (lookback up to 1440) | 1440 | 60 days |
| **Minimum recommended** | **720 1H bars** | **30 days** |

On startup, download 720×4 = 2880 15m bars per symbol from REST API.

---

## File Structure

```
PAPER_TRADER/
├── README.md              ← This file
├── paper_trader.py        ← Main async event loop
├── kline_buffer.py        ← 15m buffer + multi-TF resampling
├── signal_engine.py       ← Alpha computation + adaptive net (live)
├── config.py              ← Constants, URLs, symbols
├── freeze_params.py       ← Extract frozen params from walk-forward
└── logs/                  ← Runtime output (created automatically)
```

---

## Running

```bash
# 1. Freeze parameters (run once, or monthly)
python freeze_params.py

# 2. Start paper trader
python paper_trader.py
```

---

*V9b Paper Trader — 2026-03-11*
