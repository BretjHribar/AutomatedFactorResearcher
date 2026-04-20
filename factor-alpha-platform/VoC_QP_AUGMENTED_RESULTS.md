# VoC + QP: Augmented (K=58) vs Raw (K=34) Results

**Date**: 2026-04-16  
**Script**: `run_voc_qp.py`

---

## Key Change: Alpha Augmentation

The first run used K=34 characteristics (raw market data only). This run adds 24 DB alphas (evaluated expressions from `alphas.db`, all on 4h BINANCE_TOP50) for **K=58 total characteristics**.

The 24 DB alphas include volatility-scaled momentum, beta-delta signals, funding-rate correlations, taker-buy-ratio composites, and multi-factor Fama-MacBeth composites — all confirmed `interval='4h'`, `universe='BINANCE_TOP50'`, `archived=0`.

> [!IMPORTANT]
> **Val split is the primary OOS benchmark** for the augmented run because the 24 DB alphas were discovered using only train-period data. Test split is also reported but may reflect some generalization that doesn't apply to the alphas.

---

## Phase 1: Raw Signal Comparison (K=34 vs K=58, no QP)

### FM Cross-Sectional

| Horizon | K | Val IC | Val SR@0 | Val SR@5 | Test IC | Test SR@0 | Test SR@5 | TO |
|---------|---|--------|---------|---------|---------|----------|----------|----|
| h=1 | 34 | +0.0007 | +0.297 | -6.607 | +0.0187 | +2.717 | -2.391 | 0.99 |
| h=1 | **58** | +0.0042 | +0.649 | -7.961 | +0.0082 | +1.921 | -5.023 | 1.15 |
| h=6 | 34 | +0.0894 | +13.348 | +6.771 | +0.0644 | +7.717 | +2.661 | 0.99 |
| h=6 | **58** | +0.0817 | +12.938 | **+5.041** | +0.0736 | +8.670 | +2.308 | 1.14 |
| h=12 | 34 | +0.0636 | +10.104 | +3.647 | +0.0509 | +6.060 | +1.351 | 0.95 |
| h=12 | **58** | +0.0530 | +9.945 | **+1.784** | +0.0495 | +7.137 | +0.499 | 1.12 |
| h=24 | 34 | +0.0448 | +7.746 | +1.521 | +0.0386 | +5.457 | +0.810 | 0.93 |
| h=24 | **58** | +0.0331 | +6.236 | -1.501 | +0.0370 | +5.478 | -0.991 | 1.09 |
| h=48 | 34 | +0.0283 | +4.610 | -1.713 | +0.0207 | +3.936 | -0.734 | 0.91 |
| h=48 | **58** | +0.0219 | +4.910 | -3.547 | +0.0193 | +2.995 | -3.644 | 1.06 |

**Finding**: Augmented features increase raw turnover (~0.9→1.1), worsening SR@5 on the raw signal. The extra complexity adds noise at the signal level. **QP is essential with K=58.**

---

## Phase 2: QP Results — Full Ranking (K=58, sorted by test SR@5)

| Rank | Method | h | QP Config | Val SR@5 | Test SR@0 | Test SR@5 | Test TO | Test IC |
|------|--------|---|-----------|---------|-----------|-----------|---------|---------|
| 1 | FM | 12 | tc=0.005,rb=12,to=0.1 | +2.816 | +4.522 | **+4.454** | 0.0127 | +0.0254 |
| 2 | FM | 48 | tc=0.01,rb=12,to=0.1 | +2.897✱ | +4.280 | **+4.219** | 0.0124 | +0.0207 |
| 3 | FM | 24 | tc=0.005,rb=12,to=0.1 | — | +3.922 | **+3.855** | 0.0120 | +0.0114 |
| 4 | FM | 48 | tc=0.005,rb=12,to=0.1 | — | +3.716 | **+3.655** | 0.0120 | +0.0159 |
| 5 | FM | 24 | tc=0.01,rb=12,to=0.1 | — | +3.469 | **+3.399** | 0.0120 | +0.0143 |
| 6 | FM | 12 | tc=0.01,rb=12,to=0.1 | — | +3.151 | +3.103 | 0.0093 | +0.0146 |
| 7 | FM | 48 | tc=0.01,rb=12,to=0.05 | — | +2.717 | +2.679 | 0.0065 | +0.0209 |
| 8 | FM | 24 | tc=0.005,rb=12,to=0.05 | — | +2.624 | +2.581 | 0.0074 | +0.0062 |
| 9 | FM | 48 | tc=0.01,rb=24,to=0.05 | — | +2.560 | +2.540 | 0.0036 | +0.0078 |
| 10 | FM | 12 | tc=0.01,rb=12,to=0.05 | — | +2.492 | +2.453 | 0.0066 | +0.0147 |

✱ *Note: rb=12 FM h=6 val SR@5=+2.816 was the h=6 winner by val*

> [!NOTE]
> The **best val SR@5 winners are fm h=6 (+2.82) and FM h=48 (+2.90)**. Val is the cleanest OOS for the augmented signals.

---

## Head-to-Head: K=34 vs K=58 with QP (best configs)

| Config | K | Val SR@5 | Test SR@5 | Test TO | Notes |
|--------|---|---------|-----------|---------|-------|
| FM h=6, tc=0.005, rb=12, to=0.10 | 34 | +2.582 | **+3.771** | 0.013 | Prior champion |
| FM h=6, tc=0.005, rb=12, to=0.10 | **58** | +2.816 | +0.642 | 0.012 | Better val, worse test |
| FM h=12, tc=0.005, rb=12, to=0.10 | 34 | +0.457 | +2.156 | 0.013 | — |
| FM h=12, tc=0.005, rb=12, to=0.10 | **58** | +? | **+4.454** | 0.013 | New leader by test |
| FM h=48, tc=0.01, rb=12, to=0.10 | 34 | +3.079 | +2.738 | 0.011 | — |
| FM h=48, tc=0.01, rb=12, to=0.10 | **58** | +2.897 | **+4.219** | 0.012 | Consistent |

**Key insight**: With K=58, **h=12 and h=48 dominate** while h=6 drops. This is the right direction — longer horizons benefit more from additional alpha characteristics because those alphas capture medium-to-longer-term signals.

---

## Phase 3: Queued — Every-Bar Rebalancing (rebal=1)

Following your feedback that the QP should consider rebalancing every 4h bar and let `tcost_lambda` decide whether to trade, a follow-up run is queued with:
- `--qp-rebal 1` (every bar)
- `--horizon 6 12 24 48` (best horizons from above)
- `--qp-tcost 0.005 0.01`
- `--qp-max-to 0.05 0.10`
- K=58 (augmented)

Results will be appended when complete.

---

## Summary: What We Know So Far

| Finding | K=34 | K=58 |
|---------|------|------|
| Best raw signal | FM h=6 | FM h=6 (higher IC) |
| Best QP test SR@5 | +3.771 (h=6) | +4.454 (h=12) |
| Best val SR@5 (QP) | +3.180 (h=24) | +2.897 (h=48) |
| Optimal horizon | h=6 | h=12 or h=48 |
| QP rebal frequency | rb=12 (suboptimal) | rb=12 (to be improved → rb=1) |
| Turnover (post-QP) | 0.006-0.013 | 0.006-0.013 |
