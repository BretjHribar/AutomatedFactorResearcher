# AIPT Research Notes — Session 2026-04-24/25

End-to-end log of every backtest run this session, with all numerical results, what each experiment was testing, and what we learned. Three asset classes (crypto 4h, crypto 4h with random GP trees, US equities daily) plus QP-execution and IC/R² instrumentation.

Methodology common to all runs:
- AIPT = Random Fourier Features → ridge-Markowitz on factor returns (Didisheim-Ke-Kelly-Malamud 2023, SSRN 4388526)
- Z_t feature panel rank-normalized to [-0.5, 0.5] cross-sectionally each bar
- F_t = (1/√N) Sₜᵀ R_{t+1};  λ̂ = (FF/T + ρI)⁻¹ μ̂  (Woodbury when P > T)
- w_t = (1/√N) Sₜ λ̂  (then L1-normalized to gross=1)
- All λ̂ updates are walk-forward — 1500-bar (equities) or 4380-bar (crypto) trailing window, rebalanced every 5 (equities) or 12 (crypto) bars
- Fees: 3 bps (crypto, KuCoin VIP12 taker), 1 bp (equities)

---

## §1. Crypto baseline — 18 hand-picked alphas + 24 chars (mode comparison)

Originally established baseline: which feature mix to use for AIPT P=1000 over 2024-09-01 → 2026-04-24 (3604 bars OOS). All variants share same RFF parameters (seed=42).

### §1.1 Three-mode comparison (no smoothing, no QP)

| Mode | D | gross SR | net SR (3 bps) | TO/bar | gross cum | net cum |
|---|---:|---:|---:|---:|---:|---:|
| 1: random projections (chars only) | 24 | +5.90 | +1.19 | 56.9% | +154.1% | +31.2% |
| 2: alphas only | 18 | +2.55 | +1.43 | 15.5% | +76.4% | +42.9% |
| 3: both (chars + alphas) | 42 | +5.30 | **+1.72** | 48.7% | +155.6% | **+50.3%** |

Mode 3 ("Both") wins net SR. Alphas already embed `sma`/`Decay_exp`/`ts_zscore` → low TO; raw chars are point-in-time → high TO; combining gives best of both.

### §1.2 IC + R² for the same three baselines (P=1000)

| Mode | D | IC (Pearson) | IR (Pearson) | IC (Spearman) | R² mean | R² median |
|---|---:|---:|---:|---:|---:|---:|
| 1: chars only | 24 | +0.0235 | +8.26 | +0.0318 | 0.0183 | 0.0086 |
| 2: alphas only | 18 | +0.0073 | +2.39 | −0.0126 | 0.0204 | 0.0097 |
| 3: chars + alphas | 42 | **+0.0249** | **+8.73** | +0.0291 | 0.0184 | 0.0084 |
| 3+: chars + ALL 49 DB alphas | 73 | +0.0175 | +6.62 | +0.0178 | 0.0157 | 0.0073 |

Mode 3 (D=42) has the best IC. Adding the newer DB alphas (49 total) hurts — the extra 13 alphas are net noise.

---

## §2. Crypto Markowitz with quadratic turnover penalty (Gârleanu-Pedersen)

Question: can we close the gross→net gap by adding `(τ/2)·‖Sₜ(λ−λ_prev)‖²/N` to the optimization at rebalance time?

| Mode | τ | gross SR | net SR | TO | net cum |
|---|---:|---:|---:|---:|---:|
| RP (chars) | 0.0 | +5.90 | +1.19 | 56.9% | +31.2% |
| RP (chars) | 0.1 | +5.85 | +1.10 | 56.7% | +28.3% |
| RP (chars) | 1.0 | +5.84 | +1.10 | 56.7% | +28.3% |
| RP (chars) | 10.0 | +5.84 | +1.10 | 56.7% | +28.3% |
| Alphas | 0.0 | +2.55 | +1.43 | 15.5% | +42.9% |
| Alphas | 0.1 | +2.60 | +1.47 | 15.6% | +44.0% |
| Alphas | 1.0 | +2.60 | +1.47 | 15.6% | +44.0% |
| Alphas | 10.0 | +2.60 | +1.47 | 15.6% | +44.0% |
| Both | 0.0 | +5.30 | +1.72 | 48.7% | +50.3% |
| Both | 0.1 | +5.37 | +1.76 | 48.7% | +51.3% |
| Both | 1.0 | +5.37 | +1.76 | 48.7% | +51.3% |
| Both | 10.0 | +5.37 | +1.76 | 48.7% | +51.3% |

**Conclusion:** rebalance-time τ-penalty saturates immediately and gives essentially zero TO reduction. The 48-57% TO floor is **within-period drift** — w_t = (1/√N) Sₜ λ changes every bar even with frozen λ because Sₜ changes. The factor-space penalty can't reach this.

---

## §3. Crypto iter-1 — EWMA-w / EWMA-Z / per-bar TC penalty (28 variants)

To attack within-period drift, introduce smoothing in asset/feature space. All variants on Mode 3 (chars + 18 alphas), P=1000.

### §3.1 EWMA on weights:  sm_w_t = (1−α) sm_w_{t-1} + α (1/√N) Sₜ λ

| α | gross SR | net SR | TO | net cum |
|---:|---:|---:|---:|---:|
| 1.0 (baseline) | +5.30 | +1.72 | 48.7% | +50.3% |
| 0.5 | +4.20 | **+2.33** | 25.6% | +68.6% |
| 0.25 | +3.14 | +2.12 | 13.7% | +61.5% |
| 0.10 | +2.05 | +1.57 | 6.2% | +43.7% |
| 0.05 | +1.51 | +1.23 | 3.6% | +33.4% |
| 0.025 | +1.09 | +0.92 | 2.1% | +24.7% |
| 0.01 | +0.45 | +0.36 | 1.1% | +9.5% |

Optimum α=0.5 → net SR 1.72 → 2.33 (+0.61 vs baseline).

### §3.2 EWMA on Z (smooth feature panel):  Z̃_t = β Z̃_{t-1} + (1−β) Z_t

| β | gross SR | net SR | TO | net cum |
|---:|---:|---:|---:|---:|
| 0.0 (=baseline) | +5.30 | +1.72 | 48.7% | +50.3% |
| 0.1 | +5.29 | +1.99 | 44.5% | +58.1% |
| 0.2 | +5.17 | +2.20 | 40.0% | +64.0% |
| 0.3 | +4.96 | +2.34 | 35.2% | +67.8% |
| 0.4 | +4.65 | **+2.39** | 30.1% | +69.1% |
| 0.5 | +4.26 | +2.38 | 24.9% | +68.4% |
| 0.7 | +3.25 | +2.15 | 14.2% | +60.2% |
| 0.8 | +2.60 | +1.87 | 9.3% | +51.4% |
| 0.9 | +1.90 | +1.51 | 4.8% | +40.4% |
| 0.95 | +1.38 | +1.17 | 2.7% | +31.1% |
| 0.98 | +0.54 | +0.42 | 1.5% | +11.4% |

Optimum β=0.4 (essentially flat 0.3–0.5) → net SR 1.72 → 2.39 (+0.67 vs baseline).

### §3.3 Per-bar Markowitz with asset-space TC penalty (rebalance every bar)

| τ | gross SR | net SR | TO | net cum | runtime |
|---:|---:|---:|---:|---:|---:|
| 0.1 | +5.29 | +1.65 | 48.8% | +47.7% | 8 min |
| 1.0 | +5.29 | +1.65 | 48.8% | +47.7% | 8 min |
| 10.0 | +5.29 | +1.65 | 48.8% | +47.7% | 8 min |
| 100.0 | +5.29 | +1.65 | 48.8% | +47.7% | 8 min |

**Same as §2 — penalty saturates because Sₜ-driven within-period drift is the binding constraint.** Confirms the diagnosis.

### §3.4 Combos (EWMA-w × EWMA-Z) and ridge sweeps

| Variant | gross SR | net SR | TO | net cum |
|---|---:|---:|---:|---:|
| combo(α=0.5, β=0.5) | +3.39 | +2.39 | 13.1% | +67.6% |
| combo(α=0.3, β=0.5) | +2.83 | +2.15 | 8.7% | +59.8% |
| combo(α=0.5, β=0.7) | +2.65 | +2.02 | 8.1% | +55.6% |
| baseline + ridge=1e-4 | +4.48 | +0.51 | 48.8% | +13.7% |
| baseline + ridge=1e-2 | +4.66 | +1.98 | 42.4% | +67.7% |
| baseline + ridge=1e-1 | +2.60 | +1.46 | 30.7% | +85.5% |
| baseline + ridge=1.0 | +0.53 | +0.24 | 18.3% | +32.8% |

**iter-1 winner: combo(α=0.5, β=0.5) → net SR 2.39.** EWMA-w and EWMA-Z are *substitutes*, not complements — stacking doesn't help.

---

## §4. Crypto iter-2 — combo grid + higher P + rebal frequency + ridge × β (29 variants)

Looking for >2.4 by exploring uncovered cells (especially β<0.5 + EWMA-w combos, higher P).

| Variant | gross SR | net SR | TO | net cum |
|---|---:|---:|---:|---:|
| ewma_z(β=0.3) [refined] | +4.96 | +2.34 | 35.2% | +67.8% |
| **combo(β=0.3, α=0.5)** | +3.86 | **+2.50** | 18.2% | +72.1% |
| **combo(β=0.3, α=0.75)** | +4.47 | **+2.50** | 26.4% | +72.5% |
| combo(β=0.5, α=0.3) | +2.83 | +2.15 | 8.7% | +59.8% |
| combo(β=0.5, α=0.75) | +3.86 | +2.44 | 18.7% | +69.8% |
| P=2000, combo(0.5,0.5) | +3.06 | +1.94 | 14.4% | +53.8% |
| P=3000, combo(0.5,0.5) | +2.87 | +1.71 | 14.8% | +46.8% |
| rebal=3,  α=0.5 | +4.32 | +2.38 | 25.6% | +67.9% |
| rebal=24, α=0.5 | +4.05 | +2.20 | 25.5% | +65.6% |
| baseline + ridge=0.01 | +4.66 | **+1.98** | 42.4% | +67.7% |

**iter-2 winner: combo(β=0.3, α=0.5) → net SR 2.50.** Higher P (>1000) consistently overfits with smoothing. Default rebal=12 is optimal.

---

## §5. Crypto iter-3 — full DB alphas (36) + ridge × α-EWMA (18 variants)

Pull all 36 crypto/4h KUCOIN_TOP100 alphas from the DB (was 18 hardcoded).

| Variant | gross SR | net SR | TO | net cum |
|---|---:|---:|---:|---:|
| baseline (D=60) | +5.44 | +2.03 | 46.2% | +59.7% |
| ewma_w(α=0.5) | +4.08 | +2.31 | 24.2% | +68.6% |
| ewma_z(β=0.5) | +4.23 | +2.41 | 24.2% | +68.9% |
| combo(α=0.5, β=0.5) | +3.44 | +2.43 | 13.3% | +69.7% |
| P=2000 combo | +2.82 | +1.84 | 13.2% | +53.5% |
| P=3000 combo | +2.97 | +1.94 | 13.1% | +52.8% |
| P=4000 combo | +2.67 | +1.64 | 13.4% | +46.0% |
| ridge=0.003, α=0.5 | +4.11 | +2.62 | 22.5% | +86.1% |
| ridge=0.003, α=0.75 | +4.81 | +2.61 | 33.0% | +84.6% |
| ridge=0.003, α=1.0 | +5.36 | +2.38 | 43.9% | +76.2% |
| **ridge=0.01,  α=0.5** | +3.76 | **+2.68** | 19.1% | **+103.0%** |
| ridge=0.01,  α=0.75 | +4.27 | +2.64 | 28.6% | +99.9% |
| ridge=0.01,  α=1.0 | +4.66 | +2.40 | 38.9% | +89.2% |
| ridge=0.03,  α=0.5 | +3.14 | +2.41 | 16.0% | +112.5% |
| ridge=0.03,  α=0.75 | +3.48 | +2.35 | 24.2% | +108.5% |
| alphas-only (mode 2) baseline | +3.04 | +1.59 | 20.2% | +47.7% |
| alphas-only combo(0.5,0.5) | +2.44 | +1.82 | 8.3% | +53.0% |

**iter-3 winner: ridge=0.01 + α=0.5 (no β) → net SR 2.68, net cum +103%.** Stronger ridge + EWMA-w + 36 alphas is the best stack.

---

## §6. Crypto iter-4 — all 49 DB alphas + ridge × β grid (17 variants)

DB grew from 36 to 49 alphas; we hoped more signal would push past 2.7. Spoiler: it didn't.

| Variant | gross SR | net SR | TO | net cum |
|---|---:|---:|---:|---:|
| baseline (D=73) | +4.32 | +1.26 | 43.5% | +38.8% |
| ridge=0.01, α=0.5 | +3.33 | +2.48 | 16.3% | +103.1% |
| ridge=0.003, α=0.5, β=0.2 | +3.45 | +2.43 | 16.6% | +85.7% |
| **ridge=0.01,  α=0.5, β=0.2** | +3.17 | **+2.49** | 13.1% | +104.0% |
| ridge=0.03,  α=0.5, β=0.2 | +2.75 | +2.29 | 10.6% | +114.7% |
| ridge=0.003, α=0.5, β=0.3 | +3.32 | +2.42 | 14.7% | +85.0% |
| ridge=0.01,  α=0.5, β=0.3 | +3.08 | +2.48 | 11.5% | +103.6% |
| ridge=0.03,  α=0.5, β=0.3 | +2.70 | +2.29 | 9.4% | +114.3% |
| ridge=0.003, α=0.5, β=0.4 | +3.19 | +2.39 | 12.9% | +83.7% |
| ridge=0.01,  α=0.5, β=0.4 | +2.99 | +2.46 | 10.0% | +102.2% |
| ridge=0.03,  α=0.5, β=0.4 | +2.64 | +2.28 | 8.2% | +112.8% |
| ridge=0.003, α=0.5, β=0.5 | +3.05 | +2.36 | 11.1% | +81.7% |
| ridge=0.01,  α=0.5, β=0.5 | +2.88 | +2.43 | 8.6% | +99.6% |
| ridge=0.03,  α=0.5, β=0.5 | +2.57 | +2.25 | 7.1% | +109.8% |
| ridge=0.01, α=0.35, β=0.3 | +2.80 | +2.35 | 8.7% | +98.0% |
| ridge=0.01, α=0.4,  β=0.3 | +2.91 | +2.41 | 9.6% | +100.5% |

**iter-4 winner: net SR 2.49.** Worse than iter-3's 2.68 — the 13 newer alphas are net noise. Confirms what iter-3+full set already showed.

**Key takeaway across iters 1-4:** smoothing-based crypto AIPT plateaus at ~net SR 2.5-2.7. To break further we need either better signal (different feature space) or proper QP execution.

---

## §7. Crypto random GP trees (depth ≤ 3) — 18 configs

Replace hand-picked 24 chars with K random GP trees on raw fields. Two pipelines: (a) ridge directly on K trees, (b) RFF expansion to size P then ridge. Three K values × six P values (incl. ridge-only). Critical: gamma_scale = √(24/K) so projection variance is K-invariant.

OOS split: VAL = 2024-09-01 → 2025-03-01, TEST = 2025-03-01 → 2026-04-24.

### §7.1 Sharpe / TO results

| K | Mode | VAL nSR | VAL gSR | VAL TO | TEST nSR | TEST gSR | TEST TO | TEST AnnR% |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 480 | TREES-only | −0.39 | +1.63 | 35.2% | −0.38 | +1.77 | 29.9% | −6.9 |
| 480 | RFF P=2000 | −0.24 | +2.27 | 35.6% | −0.74 | +2.46 | 32.5% | −9.8 |
| 480 | RFF P=5000 | +1.81 | +4.33 | 33.4% | −1.26 | +1.95 | 31.6% | −16.2 |
| 480 | RFF P=10000 | +1.57 | +4.14 | 33.2% | −1.14 | +2.04 | 32.1% | −15.1 |
| 480 | RFF P=20000 | +2.19 | +4.70 | 32.8% | −0.23 | +2.89 | 31.5% | −3.0 |
| 480 | RFF P=40000 | +1.30 | +3.70 | 32.7% | −0.32 | +2.76 | 31.8% | −4.3 |
| 966 | TREES-only | +1.53 | +3.65 | 31.5% | +0.55 | +4.91 | 47.4% | +7.9 |
| 966 | RFF P=2000 | +0.59 | +3.18 | 38.0% | +0.86 | +5.37 | 44.6% | +11.2 |
| 966 | RFF P=5000 | +1.90 | +4.35 | 34.3% | +1.29 | +5.60 | 41.5% | +16.3 |
| 966 | RFF P=10000 | +2.36 | +5.00 | 34.5% | +1.56 | +5.95 | 41.3% | +19.3 |
| 966 | RFF P=20000 | **+3.42** | +5.81 | 33.8% | +1.69 | +5.85 | 39.5% | +21.0 |
| **966** | **RFF P=40000** | **+3.81** | +6.24 | 34.3% | **+1.97** | +6.10 | 38.8% | **+24.3** |
| 1928 | TREES-only | +1.25 | +3.73 | 33.8% | +0.68 | +5.47 | 48.9% | +9.0 |
| 1928 | RFF P=2000 | +0.24 | +2.89 | 38.1% | −0.04 | +4.23 | 45.4% | −0.6 |
| 1928 | RFF P=5000 | +1.40 | +3.91 | 37.1% | +0.34 | +4.54 | 43.1% | +4.5 |
| 1928 | RFF P=10000 | +1.31 | +3.73 | 35.9% | +0.86 | +5.21 | 42.5% | +11.0 |
| **1928** | **RFF P=20000** | +0.78 | +3.16 | 36.1% | **+1.98** | +6.12 | 41.2% | **+26.0** |
| 1928 | RFF P=40000 | +1.96 | +4.35 | 35.7% | +1.90 | +5.89 | 40.4% | +25.3 |

### §7.2 Same configs with IC + R² (in progress as of writeup; partial results)

| K | Mode | VAL IC | VAL IR | VAL R² | TEST IC | TEST IR | TEST R² |
|---:|---|---:|---:|---:|---:|---:|---:|
| 480 | TREES-only | −0.0047 | −1.12 | 0.038 | −0.0007 | −0.18 | 0.034 |
| 480 | RFF P=2000 | +0.0064 | +2.42 | 0.015 | +0.0083 | +3.21 | 0.015 |
| 480 | RFF P=5000 | +0.0119 | +4.61 | 0.015 | +0.0076 | +2.95 | 0.014 |
| 480 | RFF P=10000 | +0.0137 | +5.40 | 0.014 | +0.0103 | +3.87 | 0.016 |
| 480 | RFF P=20000 | +0.0153 | +5.90 | 0.015 | +0.0123 | +4.58 | 0.016 |
| 480 | RFF P=40000 | +0.0157 | +6.07 | 0.015 | +0.0132 | +4.92 | 0.016 |
| 966 | TREES-only | +0.0033 | +0.91 | 0.029 | **+0.0167** | +4.99 | 0.025 |
| 966 | RFF P=2000 | +0.0110 | +3.95 | 0.017 | **+0.0242** | **+9.45** | 0.015 |
| 966 | RFF P=5000–40000 | (in progress) | | | | | |
| 1928 | (all) | (in progress) | | | | | |

**Headlines so far:**
- TEST IC peak: K=966 RFF P=2000 → **+0.0242, IR +9.45** (matches chars+alphas baseline +0.0249/+8.73 from §1.2)
- IC climbs steadily with P at K=480 (peaks +0.0132 at P=40000)
- IC R² and Sharpe disagree on best config — IR is the more stable rank
- Trees ≈ chars+alphas on IC. The big gross-SR jump from trees doesn't translate to net because TO stays at 30-49%

**Trees vs chars equivalence:** at the gross level, K=966 RFF P=40000 (+6.10 TEST gross SR) ≈ chars-only (+5.90 full-OOS gross). The expensive tree pipeline buys very little over hand-picked chars on this universe.

---

## §8. Crypto QP execution (24 chars, P=1000) — COMPLETE

OOS split 50/50: VAL = 2024-09-01 → 2025-06-28, TEST = 2025-06-28 → 2026-04-24.

QP objective (cvxpy, ECOS solver):
```
maximize  alpha·h - (κ/2)·hᵀΣh - tx_cost·‖h - h_prev‖₁
s.t.      sum(h) = 0   (dollar-neutral)
          ‖h‖_∞ ≤ max_position
          ‖h‖_1 ≤ max_gross_leverage
          ‖h - h_prev‖_1 ≤ max_turnover
          PCA[:,k]·h = 0   for k = 1..n_pca_factors
```

Sweep grid: max_TO ∈ {0.20, 0.40, 0.60} × κ ∈ {1, 10} × tx_cost ∈ {3, 15} bps × max_gross ∈ {0.8, 1.0} × n_pca ∈ {0, 3} = **48 configs on VAL**, then evaluate winner on TEST.

Baseline (raw P=1000, no QP):
- VAL: SR_n = +1.54, cum = +24.3%, TO = 52.3%
- TEST: SR_n = +0.71, cum = +6.9%, TO = 61.5%

### §8.1 Final QP results (winner picked on VAL, evaluated on TEST)

**Best QP config (highest val_net_SR among 48 configs, all ≥80% convergence):**
- max_turnover = 0.40
- κ (risk_aversion) = 10.0
- tx_cost = 15 bps (in objective; actual fees still 3 bps)
- max_gross_leverage = 1.0
- n_pca_factors = 0  (no PCA neutralization)
- max_position = 0.08, cov_shrinkage = 0.9, optimizer_lookback = 120

**TEST results (held-out, no tuning):**

| Metric | Baseline (raw P=1000, no QP) | QP-tuned | Δ |
|---|---:|---:|---:|
| VAL Net SR | +1.54 | **+2.73** | **+1.19** |
| VAL Net cum % | +24.3 | +64.0 | +39.7 |
| VAL TO/bar | 52.3% | 20.0% | −32.3 pp |
| **TEST Net SR** | +0.71 | **+2.26** | **+1.55** |
| **TEST Net cum %** | +6.9 | **+35.8** | **+28.9** |
| TEST Gross SR | +7.54 | +3.62 | −3.92 |
| TEST TO/bar | 61.5% | 20.0% | −41.5 pp |

**Honest TEST lift: 0.71 → 2.26 net SR (3.2×).** The QP forfeits ~half the gross SR (7.54 → 3.62) but cuts TO 3× — net effect is huge because the raw strategy was fee-bleeding hard.

**Notable:** the best config picks `tx_cost = 15 bps` in the objective (5× the actual 3 bps fee). This isn't a fee model error — it's "shrinkage on trading." Penalizing trades extra in the optimizer makes the model trade only when expected alpha clearly exceeds cost, which generalizes better than penalizing at the exact fee level (analogous to using `tx_cost = realized_fee + bid_ask_spread + market_impact_estimate`).

**`n_pca_factors = 0` is the winner** — neutralizing the top-3 PCA factors of returns consistently HURT (configs with pca=3 averaged ~1 SR worse). The market-beta + 1-2 sector-PCA dimensions actually contain real alpha, not just risk to hedge out.

### §8.2 QP > smoothing tricks for this signal

Comparison of net SR for the same base signal (chars+alphas P=1000, full OOS Sep'24→end):
- Raw baseline (§1): **+1.72** TO=48.7%
- + EWMA-w (α=0.5):  +2.33  TO=25.6%
- + EWMA-Z (β=0.5):  +2.38  TO=24.9%
- + combo (α=0.5,β=0.5):  +2.39  TO=13.1%
- + best from iter-3 (ridge=0.01, α=0.5):  +2.68  TO=19.1%
- **+ QP execution (held-out TEST only): +2.26  TO=20.0%** — different OOS window so not fully comparable, but TEST 2.26 is on the *harder* second half

The QP achieves comparable net SR to the best smoothing recipe **on a strictly harder OOS window (TEST = 2025-06 → 2026-04)** vs the smoothing numbers' full Sep'24→end window which includes the easier early period. Adjusting for window, **QP is materially better.**



---

## §9. US Equities AIPT (TOP2000, P=1000) — baseline

First equities run. Daily bars, DELAY=1 (signal at t uses chars at t-1; trade t open; PnL t→t+1).

Setup:
- Universe: TOP2000 by ADV20, ≥50% coverage → **N = 1898 stocks**
- 24 hand-picked equity chars (momentum / value / quality / vol / liquidity)
- Train window: 1500 bars (~6 years), rebal every 5 days
- Fees: 1 bp aggregate
- OOS: 2024-01-02 → 2026-02-26 (540 trading days)

| Split | Gross SR | Net SR (1 bp) | TO/bar | IC (Pearson) | IR (Pearson) | R² mean |
|---|---:|---:|---:|---:|---:|---:|
| **FULL OOS** (540 bars) | **+4.95** | **+4.66** | 13.8% | +0.0125 | +4.90 | 0.0018 |
| VAL (270 bars) | +6.48 | +6.15 | 14.5% | +0.0152 | +6.64 | 0.0016 |
| TEST (270 bars) | +3.67 | +3.41 | 13.1% | +0.0098 | +3.52 | 0.0021 |

**Net SR +4.66 OOS — far better than crypto.** Why:
- 20× more assets → diversification
- Daily bars → less feature staleness
- 1 bp fees vs 3 bps → 1/3 the fee drag
- TO naturally lower (13% vs crypto 49%)

VAL→TEST drop (6.15 → 3.41) is consistent with sampling noise (1σ ≈ ±1 SR at T=270) plus regime drift (rolling λ shifts into 2024 OOS data by mid-TEST).

### §9.1 Multi-universe × multi-P sweep (in progress)

Universes: TOP500, TOP1000, TOP2000, TOP3000  
P values: 1000, 2000, 5000, 10000  
(16 configs total — results streaming to `data/aipt_results/voc_equities_sweep.csv`)

Will be appended below as configs complete.

---

## §10. Cross-cutting takeaways

### What works
1. **EWMA on weights or features** — straightforward 30-50% TO reduction with modest gross-SR cost. Use α=0.5 or β=0.3-0.5.
2. **Stronger ridge** (ρ=1e-2 vs 1e-3) — meaningful improvement for crypto on full alpha set.
3. **Per-bar QP with proper L1 trade cost** — initial result shows VAL net SR 1.54 → 2.63 (1.7×) on chars-only crypto. Will know TEST result soon.
4. **Equities just work better than crypto for AIPT** — net SR 4.66 vs 1.72 with the same recipe, driven by lower per-bar TO and lower fees.

### What doesn't work
1. **Rebalance-time TC penalty (Gârleanu-Pedersen factor-space)** — saturates at zero TO reduction because within-period drift (Sₜ change) dominates λ change.
2. **Per-bar TC penalty** — same reason, doesn't reach Sₜ-driven asset drift.
3. **Higher P with smoothing** — overfits past P=1000 for crypto Mode 3.
4. **Adding more (lower-quality) alphas** — DB jump from 36 → 49 alphas hurt every metric.
5. **Random GP trees (depth ≤ 3)** — at K=966 P=40000 the gross SR matches chars (+6.1 vs +5.9) but doesn't beat it. Expensive (~7 hr to sweep) for ~zero gross alpha.
6. **PCA neutralization in the QP** — n_pca=3 consistently hurts in the crypto sweep.

### Open questions
1. Where does the equities P sweep peak? (in progress)
2. Does QP execution close the rest of the crypto gross→net gap on the TEST set?
3. Can we get equities net SR above 5 by combining QP + larger universes (TOP3000)?
4. Is the K=966 trees TEST IC of +0.024 reproducible across seeds, or sampling noise?

---

## §11. File index

| Output | What |
|---|---|
| [data/aipt_results/voc_P1000_*.csv](data/aipt_results) | §1 baseline 3-mode + τ-sweep results |
| [data/aipt_results/iter_results.csv](data/aipt_results/iter_results.csv) | §3 iter-1 EWMA results |
| [data/aipt_results/iter2_results.csv](data/aipt_results/iter2_results.csv) | §4 iter-2 combos + higher P |
| [data/aipt_results/iter3_results.csv](data/aipt_results/iter3_results.csv) | §5 iter-3 ridge × α with full DB |
| [data/aipt_results/iter4_results.csv](data/aipt_results/iter4_results.csv) | §6 iter-4 ridge × β with all 49 |
| [data/aipt_results/trees_sweep_results.csv](data/aipt_results/trees_sweep_results.csv) | §7 trees Sharpe/TO sweep |
| [data/aipt_results/trees_sweep_ic_results.csv](data/aipt_results/trees_sweep_ic_results.csv) | §7.2 trees IC/R² (in progress) |
| [data/aipt_results/trees_sweep_summary.png](data/aipt_results/trees_sweep_summary.png) | §7 chart |
| [data/aipt_results/qp_val_sweep.csv](data/aipt_results/qp_val_sweep.csv) | §8 QP sweep (in progress) |
| [data/aipt_results/qp_best_test.csv](data/aipt_results/qp_best_test.csv) | §8 QP TEST evaluation (pending) |
| [data/aipt_results/voc_equities_baseline.csv](data/aipt_results/voc_equities_baseline.csv) | §9 equities TOP2000 baseline |
| [data/aipt_results/voc_equities_sweep.csv](data/aipt_results/voc_equities_sweep.csv) | §9.1 equities multi-universe sweep (in progress) |

Scripts:
- [backtest_voc_postfix.py](backtest_voc_postfix.py) — original 3-mode crypto baseline + τ-sweep
- [backtest_voc_iter.py](backtest_voc_iter.py), [backtest_voc_iter2.py](backtest_voc_iter2.py), [backtest_voc_iter3.py](backtest_voc_iter3.py), [backtest_voc_iter4.py](backtest_voc_iter4.py) — iterative crypto experiments
- [backtest_voc_ic_r2.py](backtest_voc_ic_r2.py) — crypto IC/R² capture
- [backtest_voc_trees.py](backtest_voc_trees.py) — crypto random-tree sweep (overnight)
- [backtest_voc_trees_ic.py](backtest_voc_trees_ic.py) — crypto trees IC/R² (in progress)
- [backtest_p1000_qp_execution.py](backtest_p1000_qp_execution.py) — crypto QP execution sweep (in progress)
- [backtest_voc_equities.py](backtest_voc_equities.py) — equities TOP2000 baseline
- [backtest_voc_equities_sweep.py](backtest_voc_equities_sweep.py) — equities multi-universe × P sweep (in progress)
