"""
Fast batch tester — TRAIN-only selection.

For each candidate expression we compute on TRAIN window (2020-2024) ONLY:
  - gross Sharpe
  - max correlation to existing saved alphas

We DO NOT peek at VAL/TEST. The combiner gets to weight signals and
the held-out splits are reserved for evaluating the final composite.

PASS = TRAIN_gross_SR >= MIN_TRAIN_SR AND max|corr_train| < MAX_CORR.

Usage:
  python eval_smallcap_d0_batch.py
  (edit CANDIDATES list at top)
"""
from __future__ import annotations
import sys, sqlite3
from pathlib import Path
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.operators.fastexpression import FastExpressionEngine

UNIV_NAME = "MCAP_100M_500M"
MAX_W     = 0.02
TRAIN_START = "2020-01-01"
TRAIN_END   = "2024-01-01"
VAL_END     = "2025-04-01"
BOOK        = 500_000
MIN_TRAIN_SR     = 5.0      # user-set gate
MIN_TRAIN_FITNESS = 6.0     # fitness = SR * sqrt(|ret_ann| / max(TO, 0.125))
MAX_CORR          = 0.70    # max |corr| to any existing alpha on TRAIN (user-set strict — DO NOT TOUCH)

# Realistic per-share IB MOC fees (matches eval_smallcap_d0_final.py)
COMMISSION_PER_SHARE = 0.0045
PER_ORDER_MIN        = 0.35
SEC_FEE_PER_DOLLAR   = 27.80e-6
SELL_FRACTION        = 0.50
IMPACT_BPS           = 0.5
BORROW_BPS_ANNUAL    = 50

DATA_DIR = ROOT / "data/fmp_cache/matrices"
UNIV_DIR = ROOT / "data/fmp_cache/universes"
DB       = ROOT / "data/alpha_results.db"


# Candidates to try - edit here  (BATCH C — novel composites)
CANDIDATES_OLD = [
    # ---- VWAP-deviation × volume ----
    ("CMP_VolSurge_x_VWAPdev",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))))"),
    ("CMP_DollarSurge_x_VWAPdev",
     "rank(multiply(rank(true_divide(dollars_traded, sma(dollars_traded, 20))), rank(negative(true_divide(close, vwap)))))"),
    # Vol surge × Bollinger
    ("CMP_VolSurge_x_BB20",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(subtract(close, sma(close, 20)), df_max(stddev(close, 20), 0.001))))))"),
    ("CMP_VolSurge_x_BB5",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(subtract(close, sma(close, 5)), df_max(stddev(close, 5), 0.001))))))"),

    # ---- Range expansion × reversal ----
    ("CMP_RangeExp_x_Rev5",
     "rank(multiply(rank(true_divide(subtract(high, low), sma(subtract(high, low), 20))), rank(negative(ts_delta(close, 5)))))"),
    # high-low Bollinger (reversal of high-low position)
    ("BB_High_5",
     "rank(negative(true_divide(subtract(high, sma(high, 5)), df_max(stddev(high, 5), 0.001))))"),
    ("BB_Low_5",
     "rank(negative(true_divide(subtract(low, sma(low, 5)), df_max(stddev(low, 5), 0.001))))"),

    # ---- Tail-return reversal ----
    ("REV_MinRet5_neg",         "rank(negative(ts_min(returns, 5)))"),
    ("REV_MinRet10_neg",        "rank(negative(ts_min(returns, 10)))"),

    # ---- Acceleration (delta of delta) ----
    ("ACC_5x5_neg",             "rank(negative(ts_delta(ts_delta(close, 5), 5)))"),

    # ---- Decay-linear smoothed reversal (low TO, similar SR?) ----
    ("REV_DecayLin5_5d",        "rank(negative(decay_linear(ts_delta(close, 5), 5)))"),
    ("REV_DecayLin5_10d",       "rank(negative(decay_linear(ts_delta(close, 5), 10)))"),

    # ---- Sum of returns reversal at non-overlapping windows ----
    ("REV_SumRet14_neg",        "rank(negative(ts_sum(returns, 14)))"),
    ("REV_SumRet42_neg",        "rank(negative(ts_sum(returns, 42)))"),

    # ---- Open-relative-to-MA reversal ----
    ("REV_OpenVsSMA20",         "rank(negative(true_divide(open, sma(close, 20))))"),

    # ---- Cross: VolSurge × Lottery-MAX5 (pure non-reversal) ----
    ("CMP_VolSurge_x_LotMAX5",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(ts_max(returns, 5)))))"),
]

# BATCH C — novel composites (different horizons + triple composites)
CANDIDATES_C = [
    # ---- Triple composites: vol × VWAP × short-rev ----
    ("CMP_VolXVWAPxRev3",
     "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 3)))))"),
    ("CMP_VolXVWAPxRev5",
     "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 5)))))"),

    # ---- Vol surge × different reversal horizons ----
    ("CMP_VolSurge_x_Rev10",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(ts_delta(close, 10)))))"),
    ("CMP_VolSurge_x_Rev21",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(ts_delta(close, 21)))))"),
    ("CMP_DollarSurge_x_Rev21",
     "rank(multiply(rank(true_divide(dollars_traded, sma(dollars_traded, 20))), rank(negative(ts_delta(close, 21)))))"),

    # ---- Vol surge × distance-from-high (orthogonal angle) ----
    ("CMP_VolSurge_x_Dist63",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, ts_max(high, 63))))))"),

    # ---- Range expansion × VWAP-dev (no volume) ----
    ("CMP_RangeExp_x_VWAPdev",
     "rank(multiply(rank(true_divide(subtract(high, low), sma(subtract(high, low), 20))), rank(negative(true_divide(close, vwap)))))"),

    # ---- Inverse-vol scaled reversal (low-vol = stronger reversal) ----
    ("RVS_InvVol_x_Rev5",
     "rank(multiply(rank(true_divide(1.0, df_max(historical_volatility_20, 0.005))), rank(negative(ts_delta(close, 5)))))"),

    # ---- Reversal of midpoint and typical price (different reference) ----
    ("REV_MidpointRev5",
     "rank(negative(ts_delta(true_divide(add(high, low), 2.0), 5)))"),
    ("REV_TypicalPriceRev5",
     "rank(negative(ts_delta(true_divide(add(add(high, low), close), 3.0), 5)))"),
    ("CMP_VolSurge_x_TypicalRev5",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(ts_delta(true_divide(add(add(high, low), close), 3.0), 5)))))"),

    # ---- Decay-linear smoothed composites (lower TO) ----
    ("CMP_VolSurge_x_DecayRev5",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(decay_linear(ts_delta(close, 5), 5)))))"),

    # ---- Body-to-range × VWAP-dev (no volume) ----
    ("CMP_Body_x_VWAPdev",
     "rank(multiply(rank(negative(true_divide(subtract(close, open), df_max(subtract(high, low), 0.01)))), rank(negative(true_divide(close, vwap)))))"),

    # ---- Volume × BB60 (longer-window, fewer trades) ----
    ("CMP_VolSurge_x_BB60",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(subtract(close, sma(close, 60)), df_max(stddev(close, 60), 0.001))))))"),

    # ---- Bollinger 60-day standalone (already at SR 4.60 in batch 3) ----
    ("BB_60",
     "rank(negative(true_divide(subtract(close, sma(close, 60)), df_max(stddev(close, 60), 0.001))))"),
]

# BATCH D — final 3, more novel angles
CANDIDATES_D = [
    # ---- Range position vol-confirmed (Williams %R reversal but vol-gated) ----
    ("CMP_VolSurge_x_RangePos5",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(true_divide(subtract(ts_max(high, 5), close), df_max(subtract(ts_max(high, 5), ts_min(low, 5)), 0.01)))))"),
    ("CMP_VolSurge_x_RangePos14",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))"),

    # ---- VWAP-dev with longer VWAP smoothing × volume ----
    ("CMP_VolSurge_x_VWAP5dev",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, sma(vwap, 5))))))"),
    ("CMP_VolSurge_x_VWAP21dev",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, sma(vwap, 21))))))"),

    # ---- Vol surge × delta-of-high (intraday-high reversal) ----
    ("CMP_VolSurge_x_HighRev5",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(ts_delta(high, 5)))))"),
    ("CMP_VolSurge_x_LowRev5",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(ts_delta(low, 5)))))"),

    # ---- Dollar-surge × range expansion (no reversal) ----
    ("CMP_DollarSurge_x_RangeExp",
     "rank(multiply(rank(true_divide(dollars_traded, sma(dollars_traded, 20))), rank(true_divide(subtract(high, low), sma(subtract(high, low), 20)))))"),

    # ---- Vol × overnight gap reversal ----
    ("CMP_VolSurge_x_GapON",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(open, ts_delay(close, 1))))))"),

    # ---- Vol surge × log-return reversal (different scaling) ----
    ("CMP_VolSurge_x_LogRetRev5",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(ts_delta(log(close), 5)))))"),

    # ---- Vol × negative price-volume corr ----
    ("CMP_VolSurge_x_PVCorrNeg",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(ts_corr(close, volume, 20)))))"),

    # ---- Range expansion alone (test as standalone) ----
    ("RANGE_Exp20",
     "rank(true_divide(subtract(high, low), sma(subtract(high, low), 20)))"),

    # ---- Vol surge × VWAP-dev × Rev21 (triple, slower) ----
    ("CMP_VolxVWAPxRev21",
     "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 21)))))"),

    # ---- Vol surge × BB14 (between 5 and 20) ----
    ("CMP_VolSurge_x_BB14",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(subtract(close, sma(close, 14)), df_max(stddev(close, 14), 0.001))))))"),
]

# BATCH E — novel formulations targeting TRAIN SR >= 6 with corr < 0.75
CANDIDATES_E = [
    # ---- Vol-surge with LONGER baseline (vs existing 20d) ----
    ("CMP_VolSurge60_x_Rev5",
     "rank(multiply(rank(true_divide(volume, sma(volume, 60))), rank(negative(ts_delta(close, 5)))))"),
    ("CMP_VolSurge60_x_VWAPdev",
     "rank(multiply(rank(true_divide(volume, sma(volume, 60))), rank(negative(true_divide(close, vwap)))))"),
    ("CMP_DollarSurge60_x_Rev5",
     "rank(multiply(rank(true_divide(dollars_traded, sma(dollars_traded, 60))), rank(negative(ts_delta(close, 5)))))"),

    # ---- Decay-exp smoothed composites (different smoothing kernel) ----
    ("CMP_DecayExp_VolxRev5",
     "rank(decay_linear(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(ts_delta(close, 5)))), 5))"),
    ("CMP_DecayExp_VolxVWAPdev",
     "rank(decay_linear(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), 5))"),

    # ---- Decay smoothed VWAP-deviation alone ----
    ("VWAPdev_DecayLin5",
     "rank(decay_linear(negative(true_divide(close, vwap)), 5))"),
    ("VWAPdev_DecayLin10",
     "rank(decay_linear(negative(true_divide(close, vwap)), 10))"),

    # ---- 4-way composites: vol × VWAP × range-expansion × Rev3 ----
    ("CMP_4way_VolxVWAPxRangexRev3",
     "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), multiply(rank(true_divide(subtract(high, low), sma(subtract(high, low), 20))), rank(negative(ts_delta(close, 3))))))"),

    # ---- Open-vwap deviation (mid-day mean reversion) ----
    ("CMP_VolSurge_x_OpenVWAPdev",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(open, vwap)))))"),

    # ---- Vol-surge × conditioned reversal (trade_when high vol) ----
    # When today's volume > 1.5x avg, take reversal signal; else 0
    ("CMP_VolSurge_x_CondRev5",
     "rank(trade_when(true_divide(volume, sma(volume, 20)), negative(ts_delta(close, 5)), 0.0))"),

    # ---- Sum-returns based composite (different base — sum vs delta) ----
    ("CMP_VolSurge_x_NegSumRet5",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(ts_sum(returns, 5)))))"),
    ("CMP_VolSurge_x_NegSumRet10",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(ts_sum(returns, 10)))))"),
    ("CMP_VolSurge_x_NegSumRet21",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(ts_sum(returns, 21)))))"),

    # ---- Triple: vol × VWAP × MidpointRev5 (different rev observable) ----
    ("CMP_VolxVWAPxMidRev5",
     "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(true_divide(add(high, low), 2.0), 5)))))"),

    # ---- Vol-surge × log_returns sum (log scale reduces outliers) ----
    ("CMP_VolSurge_x_NegLogRet5",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(ts_sum(log_returns, 5)))))"),

    # ---- Vol-surge × Rev5 (close based) but with ts_zscore normalization ----
    ("CMP_VolSurge_x_Rev5_zs",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(ts_zscore(close, 5)))))"),

    # ---- Vol-surge × Bollinger 30 (between 20 and 60) ----
    ("CMP_VolSurge_x_BB30",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(subtract(close, sma(close, 30)), df_max(stddev(close, 30), 0.001))))))"),
]

# BATCH F — different aggregation/normalization strategies for orthogonality
CANDIDATES_F = [
    # ---- ts_rank (time-series rank) instead of cross-sectional — different beast ----
    ("TSrank_VWAPdev_60",
     "rank(negative(ts_rank(true_divide(close, vwap), 60)))"),
    ("TSrank_Rev5_60",
     "rank(negative(ts_rank(ts_delta(close, 5), 60)))"),
    ("TSrank_VolSurge_x_TSrank_VWAPdev",
     "rank(multiply(rank(ts_rank(true_divide(volume, sma(volume, 20)), 60)), rank(negative(ts_rank(true_divide(close, vwap), 60)))))"),

    # ---- Additive composites (linear combo of ranks) ----
    ("ADD_VolSurge_p_Rev5",
     "rank(add(multiply(rank(true_divide(volume, sma(volume, 20))), 1.0), multiply(rank(negative(ts_delta(close, 5))), 2.0)))"),
    ("ADD_VolSurge_p_VWAPdev_p_Rev5",
     "rank(add(add(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 5)))))"),

    # ---- Decay-smoothed VWAP composites ----
    ("Decay_VolSurge_x_VWAPdev",
     "rank(decay_linear(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), 10))"),
    ("Decay_TripleVolxVWAPxRev3",
     "rank(decay_linear(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 3)))), 5))"),

    # ---- Delta of VWAP-dev (change in microstructure positioning) ----
    ("CMP_VolSurge_x_DeltaVWAPdev5",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(ts_delta(true_divide(close, vwap), 5)))))"),

    # ---- Returns squared as risk-adjusted reversal ----
    ("CMP_VolSurge_x_VarRev5",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(ts_delta(square(returns), 5)))))"),

    # ---- Two-stage selectivity: trade only on extreme volume names ----
    ("CMP_VolExtreme_Rev5",
     "rank(multiply(power(rank(true_divide(volume, sma(volume, 20))), 2.0), rank(negative(ts_delta(close, 5)))))"),

    # ---- VWAP-dev × VOL × different reversal observable: DELTA of body-to-range ----
    ("CMP_Vol_x_DeltaBodyToRange",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(ts_delta(true_divide(subtract(close, open), df_max(subtract(high, low), 0.01)), 3)))))"),

    # ---- Pure microstructure: vol × intraday return × delta ----
    ("CMP_Vol_x_IntradayRet_NegSum5",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(ts_sum(subtract(true_divide(close, open), 1.0), 5)))))"),

    # ---- Sign × magnitude split ----
    ("CMP_VolSurge_x_SignedSqrtRev5",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), negative(signed_power(ts_delta(close, 5), 0.5))))"),

    # ---- Bollinger range expansion: stddev change ----
    ("CMP_VolSurge_x_StdRev5",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(ts_delta(stddev(close, 5), 3)))))"),

    # ---- Decay smoothed of NEW combinations ----
    ("Decay_VolSurge_x_Dist63",
     "rank(decay_linear(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, ts_max(high, 63))))), 5))"),

    # ---- Vol-surge with CHANGE in volume (acceleration) ----
    ("CMP_VolAccel_x_Rev5",
     "rank(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 3)), rank(negative(ts_delta(close, 5)))))"),
]

# BATCH G — short-decay variants + asymmetric additive (target SR>=6 corr<0.85)
CANDIDATES_G = [
    # ---- Short decay (3-day) variants of strong existing patterns ----
    ("Decay3_VolxVWAPxRev3",
     "rank(decay_linear(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 3)))), 3))"),
    ("Decay3_VolxVWAPxRev5",
     "rank(decay_linear(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 5)))), 3))"),
    ("Decay3_VolxRev3",
     "rank(decay_linear(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(ts_delta(close, 3)))), 3))"),
    ("Decay3_VolxVWAPdev",
     "rank(decay_linear(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), 3))"),

    # ---- Decay 7 (between 5 and 10) ----
    ("Decay7_VolxVWAPxRev5",
     "rank(decay_linear(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 5)))), 7))"),

    # ---- Asymmetric additive composites ----
    ("ADD_3v1V3R_Rev5",
     "rank(add(add(multiply(rank(true_divide(volume, sma(volume, 20))), 1.0), multiply(rank(negative(true_divide(close, vwap))), 1.0)), multiply(rank(negative(ts_delta(close, 5))), 3.0)))"),
    ("ADD_1v3V1R_Rev5",
     "rank(add(add(multiply(rank(true_divide(volume, sma(volume, 20))), 1.0), multiply(rank(negative(true_divide(close, vwap))), 3.0)), multiply(rank(negative(ts_delta(close, 5))), 1.0)))"),
    ("ADD_3v3V1R_Rev5",
     "rank(add(add(multiply(rank(true_divide(volume, sma(volume, 20))), 3.0), multiply(rank(negative(true_divide(close, vwap))), 3.0)), multiply(rank(negative(ts_delta(close, 5))), 1.0)))"),
    ("ADD_1v1V1R_Rev3",
     "rank(add(add(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 3)))))"),
    ("ADD_VWAP_p_Rev5",
     "rank(add(rank(negative(true_divide(close, vwap))), rank(negative(ts_delta(close, 5)))))"),

    # ---- Vol-baseline-10 (between 20 and 60) ----
    ("CMP_VolSurge10_x_Rev5",
     "rank(multiply(rank(true_divide(volume, sma(volume, 10))), rank(negative(ts_delta(close, 5)))))"),
    ("CMP_VolSurge10_x_VWAPdev",
     "rank(multiply(rank(true_divide(volume, sma(volume, 10))), rank(negative(true_divide(close, vwap)))))"),

    # ---- DELTA of VWAP-dev with vol confirmation ----
    ("CMP_VolxDeltaVWAPdev3",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(ts_delta(true_divide(close, vwap), 3)))))"),

    # ---- VWAP-dev with shorter decay 3 ----
    ("VWAPdev_DecayLin3",
     "rank(decay_linear(negative(true_divide(close, vwap)), 3))"),
    ("VWAPdev_DecayLin7",
     "rank(decay_linear(negative(true_divide(close, vwap)), 7))"),

    # ---- 4-way variants with different combos ----
    ("CMP_4way_VolxVWAPxRangexRev5",
     "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), multiply(rank(true_divide(subtract(high, low), sma(subtract(high, low), 20))), rank(negative(ts_delta(close, 5))))))"),
    ("CMP_4way_VolxVWAPxBB5xRev5",
     "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), multiply(rank(negative(true_divide(subtract(close, sma(close, 5)), df_max(stddev(close, 5), 0.001)))), rank(negative(ts_delta(close, 5))))))"),
]

# MINI-batch H: targeted candidates for 4th pick of batch F (SR>=6 corr<0.85)
CANDIDATES_H = [
    # 4-way with smaller windows
    ("CMP_4way_VolxVWAPdevxBB3xRev3",
     "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), multiply(rank(negative(true_divide(subtract(close, sma(close, 3)), df_max(stddev(close, 3), 0.001)))), rank(negative(ts_delta(close, 3))))))"),
    # Triple with delta-VWAP-dev (microstructure change × reversal)
    ("CMP_VolxVWAPxDeltaVWAP",
     "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(true_divide(close, vwap), 3)))))"),
    # Vol × BB5 with decay
    ("Decay3_VolxBB5",
     "rank(decay_linear(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(subtract(close, sma(close, 5)), df_max(stddev(close, 5), 0.001))))), 3))"),
    # Vol × delta-decayed-VWAP (decay then delta)
    ("CMP_VolxDeltaDecayVWAP",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(ts_delta(decay_linear(true_divide(close, vwap), 5), 3)))))"),
    # Pure VWAP × range × Rev3
    ("CMP_VWAPxRangexRev3",
     "rank(multiply(multiply(rank(negative(true_divide(close, vwap))), rank(true_divide(subtract(high, low), sma(subtract(high, low), 20)))), rank(negative(ts_delta(close, 3)))))"),
    # Range × Rev3 × Vol
    ("CMP_VolxRangexRev3",
     "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(true_divide(subtract(high, low), sma(subtract(high, low), 20)))), rank(negative(ts_delta(close, 3)))))"),
    # Triple with delta-volume baseline (vol-acceleration confirmed VWAP×Rev)
    ("CMP_VolAccelxVWAPxRev3",
     "rank(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 3)), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 3)))))"),
]

# BATCH G — final 4. New angles: vol-acceleration variants, conditional, hump-filtered
CANDIDATES_GFINAL = [
    # Vol-acceleration variants (build on the orthogonal #67)
    ("CMP_VolAccel5_x_VWAPxRev5",
     "rank(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 5)), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 5)))))"),
    ("CMP_VolAccel_x_VWAPdev",
     "rank(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 3)), rank(negative(true_divide(close, vwap)))))"),
    ("CMP_VolAccel_x_Rev3",
     "rank(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 3)), rank(negative(ts_delta(close, 3)))))"),
    ("CMP_VolAccel_x_Rev5",
     "rank(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 3)), rank(negative(ts_delta(close, 5)))))"),
    ("Decay3_VolAccelxVWAPxRev3",
     "rank(decay_linear(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 3)), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 3)))), 3))"),

    # Triple with BB instead of Rev: vol × VWAP × BB-N
    ("CMP_VolxVWAPxBB10",
     "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(true_divide(subtract(close, sma(close, 10)), df_max(stddev(close, 10), 0.001))))))"),
    ("CMP_VolxVWAPxBB5",
     "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(true_divide(subtract(close, sma(close, 5)), df_max(stddev(close, 5), 0.001))))))"),

    # Vol-accel × range-expansion × Rev3 (replace VWAP with range)
    ("CMP_VolAccelxRangexRev3",
     "rank(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 3)), rank(true_divide(subtract(high, low), sma(subtract(high, low), 20)))), rank(negative(ts_delta(close, 3)))))"),

    # Conditional triple: only trade when vol surge AND vwap-dev both extreme
    ("CMP_TWVolxVWAPxRev3",
     "rank(trade_when(true_divide(volume, sma(volume, 20)), multiply(rank(negative(true_divide(close, vwap))), rank(negative(ts_delta(close, 3)))), 0.0))"),

    # Decay7 of vol-accel pattern (longer smoothing of new orthogonal signal)
    ("Decay7_VolAccelxVWAPxRev3",
     "rank(decay_linear(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 3)), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 3)))), 7))"),
]

# BATCH H — 8 more with novel angles (SR>=6, corr<=0.85)
CANDIDATES_H2 = [
    # ---- VWAP at HIGH or LOW (different reference price) ----
    ("CMP_VolxHighVWAPdev",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(high, vwap)))))"),
    ("CMP_VolxLowVWAPdev",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(low, vwap)))))"),

    # ---- Volume z-score normalization (different from vol/sma ratio) ----
    ("CMP_VolZS_x_VWAPdev",
     "rank(multiply(rank(ts_zscore(volume, 20)), rank(negative(true_divide(close, vwap)))))"),
    ("CMP_VolZS_x_Rev5",
     "rank(multiply(rank(ts_zscore(volume, 20)), rank(negative(ts_delta(close, 5)))))"),

    # ---- VWAP self-spread (vwap vs its own SMA) — pure VWAP momentum reversal ----
    ("CMP_VolxVWAPselfRev5",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(vwap, sma(vwap, 5))))))"),
    ("CMP_VolxVWAPselfRev10",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(vwap, sma(vwap, 10))))))"),

    # ---- Long-decay smoothed VWAP-deviation × volume (slower) ----
    ("Decay10_VolxVWAPdev",
     "rank(decay_linear(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), 10))"),
    ("Decay10_VolxVWAPxRev5",
     "rank(decay_linear(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 5)))), 10))"),

    # ---- Triple with Rev10 (horizon gap between 5 and 21) ----
    ("CMP_VolxVWAPxRev10",
     "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 10)))))"),

    # ---- Vol-acceleration with different windows (filling 1d, 7d horizon) ----
    ("CMP_VolAccel1xVWAPxRev3",
     "rank(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 1)), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 3)))))"),
    ("CMP_VolAccel7xVWAPxRev5",
     "rank(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 7)), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 5)))))"),

    # ---- Dollars-traded acceleration variants ----
    ("CMP_DollarAccel_x_VWAPxRev3",
     "rank(multiply(multiply(rank(ts_delta(true_divide(dollars_traded, sma(dollars_traded, 20)), 3)), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 3)))))"),

    # ---- Combine vol×VWAP×Rev with delta of body-to-range ----
    ("CMP_VolxVWAPxDeltaBody",
     "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(true_divide(subtract(close, open), df_max(subtract(high, low), 0.01)), 3)))))"),

    # ---- Decay-linear of pure VWAP-self-spread ----
    ("VWAPselfRev5_DecayLin5",
     "rank(decay_linear(negative(true_divide(vwap, sma(vwap, 5))), 5))"),
    ("VWAPselfRev10_DecayLin5",
     "rank(decay_linear(negative(true_divide(vwap, sma(vwap, 10))), 5))"),

    # ---- 5-way composite (vol × VWAP × range × Rev3 × VolAccel) ----
    ("CMP_5way_VolxVWAPxRangexRev3xVolAccel",
     "rank(multiply(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(true_divide(subtract(high, low), sma(subtract(high, low), 20)))), multiply(rank(negative(ts_delta(close, 3))), rank(ts_delta(true_divide(volume, sma(volume, 20)), 3)))))"),
]

# BATCH I — 4 final candidates with new angles
CANDIDATES_I = [
    # ---- Triple with VWAP-3d (very short VWAP smoothing) ----
    ("CMP_VolxVWAP3dev_x_Rev5",
     "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, sma(vwap, 3))))))"),
    # ---- Triple with range-position-5 in third slot ----
    ("CMP_VolxVWAPxRangePos5",
     "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(true_divide(subtract(ts_max(high, 5), close), df_max(subtract(ts_max(high, 5), ts_min(low, 5)), 0.01)))))"),
    # ---- Vol × VWAP × ts_zscore-Rev (z-scored reversal in triple) ----
    ("CMP_VolxVWAPxZSRev5",
     "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_zscore(close, 5)))))"),
    # ---- Vol-accel × VWAP × Bollinger-5 (vol-accel triple with BB instead of Rev) ----
    ("CMP_VolAccelxVWAPxBB5",
     "rank(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 3)), rank(negative(true_divide(close, vwap)))), rank(negative(true_divide(subtract(close, sma(close, 5)), df_max(stddev(close, 5), 0.001))))))"),
    # ---- Decay-exp variants ----
    ("CMP_DecayExp03_VolxVWAPxRev3",
     "rank(decay_exp(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 3)))), 0.3))"),
    # ---- Vol × VWAP × Rev3 × VolAccel (4-way) ----
    ("CMP_4way_VolxVWAPxRev3xVolAccel",
     "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), multiply(rank(negative(ts_delta(close, 3))), rank(ts_delta(true_divide(volume, sma(volume, 20)), 3)))))"),
    # ---- Triple with delayed reversal (lagged) ----
    ("CMP_VolxVWAPxDelayedRev3",
     "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), ts_delay(rank(negative(ts_delta(close, 3))), 1)))"),
    # ---- Triple with ts_corr (price-volume correlation reversal) ----
    ("CMP_VolxVWAPxNegPVCorr5",
     "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_corr(close, volume, 5)))))"),
]

# BATCH J — final 2 picks. Mix of highest-SR + most-novel
CANDIDATES_J = [
    # ---- Triple variants combining VolAccel × VWAP × different rev observables ----
    ("CMP_VolAccel5_x_VWAPxBB10",
     "rank(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 5)), rank(negative(true_divide(close, vwap)))), rank(negative(true_divide(subtract(close, sma(close, 10)), df_max(stddev(close, 10), 0.001))))))"),
    ("CMP_VolAccel3_x_VWAPxRev10",
     "rank(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 3)), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 10)))))"),

    # ---- 4-way with different ingredients ----
    ("CMP_4way_VolxVWAPxRev5xVolAccel5",
     "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), multiply(rank(negative(ts_delta(close, 5))), rank(ts_delta(true_divide(volume, sma(volume, 20)), 5)))))"),

    # ---- Vol-acceleration ALONE × VWAP × range ----
    ("CMP_VolAccelxVWAPxRange",
     "rank(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 3)), rank(negative(true_divide(close, vwap)))), rank(true_divide(subtract(high, low), sma(subtract(high, low), 20)))))"),

    # ---- Decayed Vol-Accel composites (lower TO) ----
    ("Decay5_VolAccelxVWAPxRev3",
     "rank(decay_linear(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 3)), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 3)))), 5))"),
    ("Decay7_VolAccel5xVWAPxRev5",
     "rank(decay_linear(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 5)), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 5)))), 7))"),

    # ---- Triple with VWAP-7d (between 1d and 21d) ----
    ("CMP_VolxVWAP7dev_x_Rev5",
     "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, sma(vwap, 7))))), rank(negative(ts_delta(close, 5)))))"),
]

# BATCH K — 10 candidates targeting SR >= 6 AND fitness >= 7
# Strategy: decay-smooth strong patterns to reduce TO while keeping SR high
CANDIDATES_K = [
    # Decay 5 of strongest existing triples (target lower TO than originals)
    ("Decay5_VolxVWAPxRev21",
     "rank(decay_linear(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 21)))), 5))"),
    ("Decay5_VolxVWAPxRangePos5",
     "rank(decay_linear(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(true_divide(subtract(ts_max(high, 5), close), df_max(subtract(ts_max(high, 5), ts_min(low, 5)), 0.01)))), 5))"),
    ("Decay5_VolxVWAPxRev10",
     "rank(decay_linear(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 10)))), 5))"),
    ("Decay7_VolxVWAPxRev3",
     "rank(decay_linear(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 3)))), 7))"),
    ("Decay7_VolxVWAPdev",
     "rank(decay_linear(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), 7))"),
    ("Decay10_VolxVWAPxRev3",
     "rank(decay_linear(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 3)))), 10))"),
    # Decay-smoothed VolAccel triples (lower TO)
    ("Decay5_VolAccel5xVWAPxRev5",
     "rank(decay_linear(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 5)), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 5)))), 5))"),
    ("Decay7_VolAccelxVWAPxRev3",
     "rank(decay_linear(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 3)), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 3)))), 7))"),
    # Decay-smoothed 4-way (highest SR triple from batch I)
    ("Decay5_4way_VolxVWAPxRev3xVolAccel",
     "rank(decay_linear(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), multiply(rank(negative(ts_delta(close, 3))), rank(ts_delta(true_divide(volume, sma(volume, 20)), 3)))), 5))"),
    # Decayed VWAPdev ALONE at longer windows (pure low-TO microstructure)
    ("VWAPdev_DecayLin14",
     "rank(decay_linear(negative(true_divide(close, vwap)), 14))"),
    ("VWAPdev_DecayLin21",
     "rank(decay_linear(negative(true_divide(close, vwap)), 21))"),
    # Decay-exp variants (different smoothing kernel)
    ("DecayExp02_VolxVWAPxRev5",
     "rank(decay_exp(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 5)))), 0.2))"),
    ("DecayExp01_VolxVWAPxRev3",
     "rank(decay_exp(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 3)))), 0.1))"),
]

# BATCH L — light decay (2-3) on the highest-SR alphas (target SR>=6 + fit>=7 + corr<0.85)
CANDIDATES_L = [
    # Decay 2 of #76 (CMP_VolxVWAPxRangePos5 SR 7.13)
    ("Decay2_VolxVWAPxRangePos5",
     "rank(decay_linear(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(true_divide(subtract(ts_max(high, 5), close), df_max(subtract(ts_max(high, 5), ts_min(low, 5)), 0.01)))), 2))"),
    # Decay 3 of #76
    ("Decay3_VolxVWAPxRangePos5",
     "rank(decay_linear(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(true_divide(subtract(ts_max(high, 5), close), df_max(subtract(ts_max(high, 5), ts_min(low, 5)), 0.01)))), 3))"),
    # Decay 2 of #67 (CMP_VolAccelxVWAPxRev3 SR 6.50)
    ("Decay2_VolAccelxVWAPxRev3",
     "rank(decay_linear(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 3)), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 3)))), 2))"),
    # Decay 3 of #58 (CMP_VolSurge_x_RangePos5 SR 6.96)
    ("Decay3_VolxRangePos5",
     "rank(decay_linear(multiply(rank(true_divide(volume, sma(volume, 20))), rank(true_divide(subtract(ts_max(high, 5), close), df_max(subtract(ts_max(high, 5), ts_min(low, 5)), 0.01)))), 3))"),
    # Decay 2 of #57 (CMP_VolxVWAPxRev21 SR 6.33)
    ("Decay2_VolxVWAPxRev21",
     "rank(decay_linear(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 21)))), 2))"),
    # Decay 3 of additive (#61 ADD_VolSurge_p_VWAPdev_p_Rev5 SR 7.31)
    ("Decay3_ADD_VolSurge_p_VWAPdev_p_Rev5",
     "rank(decay_linear(add(add(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 5)))), 3))"),
    # Decay 2 of #75 (CMP_VolxVWAPxRev10 SR 6.72)
    ("Decay2_VolxVWAPxRev10",
     "rank(decay_linear(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 10)))), 2))"),
    # Decay 3 of #75
    ("Decay3_VolxVWAPxRev10",
     "rank(decay_linear(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 10)))), 3))"),
    # Decay 2 of #77 (CMP_4way_VolxVWAPxRev3xVolAccel SR 6.66)
    ("Decay2_4way_VolxVWAPxRev3xVolAccel",
     "rank(decay_linear(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), multiply(rank(negative(ts_delta(close, 3))), rank(ts_delta(true_divide(volume, sma(volume, 20)), 3)))), 2))"),
    # Decay 2 of #58 (CMP_VolSurge_x_RangePos5)
    ("Decay2_VolxRangePos5",
     "rank(decay_linear(multiply(rank(true_divide(volume, sma(volume, 20))), rank(true_divide(subtract(ts_max(high, 5), close), df_max(subtract(ts_max(high, 5), ts_min(low, 5)), 0.01)))), 2))"),
    # Higher-decay of strongest additive
    ("Decay5_ADD_VolSurge_p_VWAPdev_p_Rev3",
     "rank(decay_linear(add(add(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 3)))), 5))"),
    # Decay 2 of #79 (CMP_VolAccel3_x_VWAPxRev10)
    ("Decay2_VolAccel3xVWAPxRev10",
     "rank(decay_linear(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 3)), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 10)))), 2))"),
]

# BATCH M — find the 10th: try genuinely new angles for SR>=6 fit>=7 corr<=0.85
CANDIDATES_M_OLD = [
    # Decay-exp variants of high-SR composites (different decay kernel)
    ("DecayExp03_VolxVWAPxRangePos5",
     "rank(decay_exp(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(true_divide(subtract(ts_max(high, 5), close), df_max(subtract(ts_max(high, 5), ts_min(low, 5)), 0.01)))), 0.3))"),
    ("DecayExp04_VolxVWAPxRev3",
     "rank(decay_exp(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 3)))), 0.4))"),
    # Mild decay 2 of additive
    ("Decay2_ADD_VolSurge_p_VWAPdev_p_Rev3",
     "rank(decay_linear(add(add(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 3)))), 2))"),
    # Decay 7 of additive (lower TO version)
    ("Decay7_ADD_VolSurge_p_VWAPdev_p_Rev5",
     "rank(decay_linear(add(add(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 5)))), 7))"),
    # Decay 2 of #57 VolxVWAPxRev21 (long-rev variant)
    ("Decay3_VolxVWAPxRev21",
     "rank(decay_linear(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 21)))), 3))"),
    # Range-position 14 (different range window)
    ("Decay3_VolxVWAPxRangePos14",
     "rank(decay_linear(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))), 3))"),
    # Pure decay-linear of non-composite signals
    ("Decay3_RangePos5",
     "rank(decay_linear(true_divide(subtract(ts_max(high, 5), close), df_max(subtract(ts_max(high, 5), ts_min(low, 5)), 0.01)), 3))"),
]

# BATCH N — find 5 more (SR>=5, fit>=6, corr<0.7) — NOVEL signal sources only
CANDIDATES_N = [
    # Decay-linear of pure RangePos at different windows (no vol/VWAP coupling)
    ("Decay5_RangePos5",
     "rank(decay_linear(true_divide(subtract(ts_max(high, 5), close), df_max(subtract(ts_max(high, 5), ts_min(low, 5)), 0.01)), 5))"),
    ("Decay7_RangePos5",
     "rank(decay_linear(true_divide(subtract(ts_max(high, 5), close), df_max(subtract(ts_max(high, 5), ts_min(low, 5)), 0.01)), 7))"),
    ("Decay3_RangePos14",
     "rank(decay_linear(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)), 3))"),
    ("Decay5_RangePos14",
     "rank(decay_linear(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)), 5))"),
    # Pure decay of body-to-range (intraday signal — different observable than close)
    ("Decay5_BodyToRange_neg",
     "rank(decay_linear(negative(true_divide(subtract(close, open), df_max(subtract(high, low), 0.01))), 5))"),
    ("Decay3_BodyToRange_neg",
     "rank(decay_linear(negative(true_divide(subtract(close, open), df_max(subtract(high, low), 0.01))), 3))"),
    # Pure decay of distance-from-high (no vol/VWAP)
    ("Decay5_Dist63_neg",
     "rank(decay_linear(negative(true_divide(close, ts_max(high, 63))), 5))"),
    ("Decay7_Dist63_neg",
     "rank(decay_linear(negative(true_divide(close, ts_max(high, 63))), 7))"),
    ("Decay10_Dist63_neg",
     "rank(decay_linear(negative(true_divide(close, ts_max(high, 63))), 10))"),
    # Pure decay of lottery signal (Bali-Cakici: high MAX returns sells off)
    ("Decay5_LotMAX5_neg",
     "rank(decay_linear(negative(ts_max(returns, 5)), 5))"),
    ("Decay7_LotMAX5_neg",
     "rank(decay_linear(negative(ts_max(returns, 5)), 7))"),
    # Decay of skewness signals
    ("Decay5_NegSkew60",
     "rank(decay_linear(negative(ts_skewness(returns, 60)), 5))"),
    # Decay-exp variants of pure microstructure (no composite)
    ("DecayExp02_RangePos5",
     "rank(decay_exp(true_divide(subtract(ts_max(high, 5), close), df_max(subtract(ts_max(high, 5), ts_min(low, 5)), 0.01)), 0.2))"),
    # Pure decay of volume ratio with reversal sign
    ("Decay7_BodyToRange_neg",
     "rank(decay_linear(negative(true_divide(subtract(close, open), df_max(subtract(high, low), 0.01))), 7))"),
    # Vol-confirmed lottery (orthogonal to reversal)
    ("Decay5_VolxLotMAX5",
     "rank(decay_linear(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(ts_max(returns, 5)))), 5))"),
    # Vol-confirmed gap (orthogonal to reversal)
    ("Decay5_VolxGapON_neg",
     "rank(decay_linear(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(open, ts_delay(close, 1))))), 5))"),
    # Decay of pure 60d Bollinger (might be more orthogonal at decay)
    ("Decay5_BB60",
     "rank(decay_linear(negative(true_divide(subtract(close, sma(close, 60)), df_max(stddev(close, 60), 0.001))), 5))"),
]

# BATCH O — try industry-relative + vol-conditional + longer decays
CANDIDATES_O = [
    # Industry-rank reversal
    ("group_rank_Rev5",
     "negative(group_rank(ts_delta(close, 5), subindustry))"),
    ("group_rank_Rev3",
     "negative(group_rank(ts_delta(close, 3), subindustry))"),
    ("group_rank_VWAPdev",
     "negative(group_rank(true_divide(close, vwap), subindustry))"),
    # Industry-mean-residualized signals
    ("group_neutralize_Rev5",
     "rank(negative(group_neutralize(ts_delta(close, 5), subindustry)))"),
    ("group_neutralize_VWAPdev",
     "rank(negative(group_neutralize(true_divide(close, vwap), subindustry)))"),
    # Vol-conditional via trade_when at higher threshold
    ("VolSurge2x_Rev5",
     "rank(trade_when(subtract(true_divide(volume, sma(volume, 20)), 2.0), negative(ts_delta(close, 5)), 0.0))"),
    # Decay 10 of pure rev signals (very low TO)
    ("Decay10_Rev5",
     "rank(decay_linear(negative(ts_delta(close, 5)), 10))"),
    ("Decay14_Rev5",
     "rank(decay_linear(negative(ts_delta(close, 5)), 14))"),
    # Decay 10 of vol×rev
    ("Decay10_VolxRev5",
     "rank(decay_linear(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(ts_delta(close, 5)))), 10))"),
    # Decay-exp on body-to-range (very orthogonal observable)
    ("DecayExp02_BodyToRange_neg",
     "rank(decay_exp(negative(true_divide(subtract(close, open), df_max(subtract(high, low), 0.01))), 0.2))"),
    # Pure decay BB30 (between 20 and 60)
    ("Decay5_BB30",
     "rank(decay_linear(negative(true_divide(subtract(close, sma(close, 30)), df_max(stddev(close, 30), 0.001))), 5))"),
    # Hump-filtered Rev signal (selectivity)
    ("Hump_Rev5",
     "rank(hump(negative(ts_delta(close, 5)), 0.005))"),
    # Decay of vol-z-score × VWAPdev (different vol normalization)
    ("Decay5_VolZS_x_VWAPdev",
     "rank(decay_linear(multiply(rank(ts_zscore(volume, 20)), rank(negative(true_divide(close, vwap)))), 5))"),
    # Higher-decay BodyToRange (very orthogonal)
    ("Decay10_BodyToRange_neg",
     "rank(decay_linear(negative(true_divide(subtract(close, open), df_max(subtract(high, low), 0.01))), 10))"),
    ("Decay14_BodyToRange_neg",
     "rank(decay_linear(negative(true_divide(subtract(close, open), df_max(subtract(high, low), 0.01))), 14))"),
]

# === HYPOTHESIS-DRIVEN ALPHA TESTING (one at a time) ===
# H1: Order-flow imbalance (signed volume aggregated over 5d) reversal.
# Mechanism: sustained sell-pressure (mostly DOWN-day volume) creates oversold
# conditions distinct from one-day vol×rev pattern.
# Result: SR 1.85, fit 1.24, corr 0.45 — orthogonal but too weak (raw vol scales dominate)
#
# H1.1: NORMALIZED OFI — net signed flow / total flow over 5d.
# Result: SR 2.85 fit 2.15 corr 0.57 — better than H1 but still weak.
# OFI mechanism real but signal too noisy for this universe.
#
# H2: ts_argmin recency. Result: SR -1.90 corr 0.17.
# Sign was wrong: recent worst day → continued decline (not bounce).
# Flipped: SR +1.90 corr 0.17 — extreme orthogonality but too weak alone.
#
# H3: close vs day midpoint. Result: SR 7.19 fit 6.10 corr 0.84.
# Strong signal but same-family-as-existing range-position alphas.
# Lesson: different ratio of same OHLC data won't break the corr ceiling.
#
# H4: argmax-divergence. Result: SR 0.18 corr 0.24. Mechanism doesn't predict.
# But operator confirmed orthogonal — argmax timing differences live in fresh space.
#
# H5: ts_corr(returns, volume, 10). SR 1.35 corr 0.31. Mechanism right, weak alone.
#
# H6 triple composite: SR 2.57 corr 0.47. Composite weaker than strongest part (H1.1 at 2.85).
# Lesson: multiplying weak orthogonal signals caps at harmonic-mean strength.
#
# H7: ts_quantile. SR 4.11 corr 0.74. Closest to gate yet — just 0.89 short SR, 0.04 over corr.
#
# H7.1 ts_zscore(close, 21): SR 5.07 fit 6.15 corr 0.99 — IDENTICAL to Bollinger formula.
# Critical lesson: zscore of close ≡ (close - sma)/stddev. Same math.
#
# H7.2 ts_zscore(returns, 21): SR 4.81, fit 3.39, corr 0.67 ✓ — BREAKTHROUGH on corr!
# 69% TO killed fitness. Decay smoothing should fix without losing the corr win.
#
# H7.3 decay-smooth: SR 4.97, fit 4.76, corr 0.78 — decay pulled corr UP into existing family.
# Wrong fitness fix.
#
# H7.4 zscore returns 60d: SR 5.16, fit 3.79, corr 0.70 (right at cutoff!).
# SR passes! Corr right at edge. TO=69% kills fit. Need to drop TO without raising corr.
#
# H7.5 zscore sum5 returns: 4.52/0.83 — sum input pulled corr up. End H7 family.
# Best from H7 family: H7.2 SR 4.81 corr 0.67 (SR short 0.19, corr passes).
#
# H8 free-float turnover × rev5: SR 5.71 corr 0.86. × rev pulls into family.
#
# H9 pure 1d rev: 4.55/0.73. Correlates with multi-day rev.
# Lesson confirmed: H7.2 corr 0.67 came from vol-scaling, not from 1d itself.
#
# H10-H14 done. Best near miss: VWAPdev_DecayLin21 SR 4.62 fit 6.84 corr 0.66 (SR short 0.38).
#
# H15-H19 done.
# H17 DecayLin19: SR 4.75 fit 6.96 corr 0.69 ✓ — SR short 0.25
# H18 DecayExp(0.1): SR 5.01 ✓ fit 7.20 ✓ corr 0.73 — corr fails by 0.03
# Sweet spot is between these two — narrow but exists.
#
# H20-H24 done. VWAPdev decay caps at SR 4.92 corr 0.70.
CANDIDATES_OLD = [
    ("H20_VWAPdev_DecayExp008",
     "rank(decay_exp(negative(true_divide(close, vwap)), 0.08))"),
    ("H21_VWAPdev_DecayExp009",
     "rank(decay_exp(negative(true_divide(close, vwap)), 0.09))"),
    ("H22_VWAPdev_DecayLin17",
     "rank(decay_linear(negative(true_divide(close, vwap)), 17))"),
    # Mixed observable: close/sma(vwap, 5) — slightly slower VWAP reference
    ("H23_close_smaVWAP5_DecayLin14",
     "rank(decay_linear(negative(true_divide(close, sma(vwap, 5))), 14))"),
    # Inverse: vwap/close (positive when close < vwap, so direction flipped)
    ("H24_inv_VWAPdev_DecayLin18",
     "rank(decay_linear(subtract(true_divide(vwap, close), 1.0), 18))"),
]

# H25-H29 done. 3 PASS SR+corr but fit short.
CANDIDATES_OLD2 = [
    # H25: H17 (corr 0.69) × H7.4 (SR 5.16) — both close to passing different gates
    ("H25_H17_x_H74",
     "rank(multiply(rank(decay_linear(negative(true_divide(close, vwap)), 19)), rank(negative(ts_zscore(returns, 60)))))"),
    # H26: H17 × H7.2 (both pass corr; combination might keep corr low + boost SR)
    ("H26_H17_x_H72",
     "rank(multiply(rank(decay_linear(negative(true_divide(close, vwap)), 19)), rank(negative(ts_zscore(returns, 21)))))"),
    # H27: H17 × H2_flipped (extreme orthogonality of H2 should drag corr WAY down)
    ("H27_H17_x_H2flip",
     "rank(multiply(rank(decay_linear(negative(true_divide(close, vwap)), 19)), rank(ts_argmin(returns, 21))))"),
    # H28: H7.4 × H2_flipped (strong SR + ortho signal)
    ("H28_H74_x_H2flip",
     "rank(multiply(rank(negative(ts_zscore(returns, 60))), rank(ts_argmin(returns, 21))))"),
    # H29: VWAPdev DecayExp(0.08) × H7.2 (combine the two near-pass families)
    ("H29_DecayExp008_x_H72",
     "rank(multiply(rank(decay_exp(negative(true_divide(close, vwap)), 0.08)), rank(negative(ts_zscore(returns, 21)))))"),
]

# H30-H34 done. STRICT PASSES: H32 (decay3) and H34 (decay5) of H29 family.
CANDIDATES_OLD3 = [
    # H30: H25 with outer decay 3
    ("H30_H25_outerdecay3",
     "rank(decay_linear(multiply(rank(decay_linear(negative(true_divide(close, vwap)), 19)), rank(negative(ts_zscore(returns, 60)))), 3))"),
    # H31: H26 with outer decay 3
    ("H31_H26_outerdecay3",
     "rank(decay_linear(multiply(rank(decay_linear(negative(true_divide(close, vwap)), 19)), rank(negative(ts_zscore(returns, 21)))), 3))"),
    # H32: H29 with outer decay 3
    ("H32_H29_outerdecay3",
     "rank(decay_linear(multiply(rank(decay_exp(negative(true_divide(close, vwap)), 0.08)), rank(negative(ts_zscore(returns, 21)))), 3))"),
    # H33: H25 with outer decay 5 (more smoothing if 3 isn't enough)
    ("H33_H25_outerdecay5",
     "rank(decay_linear(multiply(rank(decay_linear(negative(true_divide(close, vwap)), 19)), rank(negative(ts_zscore(returns, 60)))), 5))"),
    # H34: H29 with outer decay 5
    ("H34_H29_outerdecay5",
     "rank(decay_linear(multiply(rank(decay_exp(negative(true_divide(close, vwap)), 0.08)), rank(negative(ts_zscore(returns, 21)))), 5))"),
]

# H35-H39: variants of the H29 family for a 3rd strict pass
CANDIDATES_OLD4 = [
    # H35: outer decay 4 (between 3 and 5)
    ("H35_H29_outerdecay4",
     "rank(decay_linear(multiply(rank(decay_exp(negative(true_divide(close, vwap)), 0.08)), rank(negative(ts_zscore(returns, 21)))), 4))"),
    # H36: replace H7.2 (zscore 21d) with H7.4 (zscore 60d) and outer decay 3
    ("H36_DecayExp008_x_H74_outerdecay3",
     "rank(decay_linear(multiply(rank(decay_exp(negative(true_divide(close, vwap)), 0.08)), rank(negative(ts_zscore(returns, 60)))), 3))"),
    # H37: H32 with decay-exp inner = 0.06 (more smoothing on inner)
    ("H37_DecayExp006_x_H72_outerdecay3",
     "rank(decay_linear(multiply(rank(decay_exp(negative(true_divide(close, vwap)), 0.06)), rank(negative(ts_zscore(returns, 21)))), 3))"),
    # H38: H32 + decay-exp inner = 0.10
    ("H38_DecayExp010_x_H72_outerdecay3",
     "rank(decay_linear(multiply(rank(decay_exp(negative(true_divide(close, vwap)), 0.10)), rank(negative(ts_zscore(returns, 21)))), 3))"),
    # H39: outer decay 6 with deeper smoothing
    ("H39_H29_outerdecay6",
     "rank(decay_linear(multiply(rank(decay_exp(negative(true_divide(close, vwap)), 0.08)), rank(negative(ts_zscore(returns, 21)))), 6))"),
]

# === New hypothesis batch (H40-H44) — 5 distinct economic mechanisms ===
# Goal: identify which mechanisms have potential. Refine winners one-at-a-time after.
CANDIDATES_PHASE1 = [
    # H40: VOL-OF-VOL REGIME — when 5d stddev is unusually high vs 60d, vol mean-reverts
    # (Heston/GARCH literature). Test direction first; refine if signal exists.
    ("H40_zscore_vol5_in_60d",
     "rank(ts_zscore(stddev(returns, 5), 60))"),
    # H41: RETURN AUTOCORRELATION REGIME — high 1-lag autocorr = trending
    # (continuation), low/negative = mean-reverting. Direction signals which to fade.
    ("H41_returns_autocorr_21d",
     "rank(ts_corr(returns, ts_delay(returns, 1), 21))"),
    # H42: LOG-RANGE Z-SCORE — direction-agnostic regime via intraday range expansion
    # vs own history. Wide ranges absorb liquidity; revert.
    ("H42_zscore_log_range_21d",
     "rank(ts_zscore(log(true_divide(high, low)), 21))"),
    # H43: FAT-TAIL REGIME — high kurtosis stocks are prone to extreme moves
    # which usually overshoot → revert.
    ("H43_kurtosis_returns_21d",
     "rank(ts_kurtosis(returns, 21))"),
    # H44: DOLLAR-VOL Z-SCORE — captures dollar-flow regime distinct from share-vol
    # ratios used in existing alphas.
    ("H44_zscore_dollarvol_60d",
     "rank(ts_zscore(multiply(volume, close), 60))"),
]

# Phase 2: pair regime detectors with directional signals.
# H45-H49: each multiplies a NEW regime signal × a corr-passing directional signal.
CANDIDATES_PHASE2 = [
    # H45: dollar-vol regime × zscore-returns 60d (H44 × H7.4)
    ("H45_dollarvol_zs_x_zscoreReturns60",
     "rank(multiply(rank(ts_zscore(multiply(volume, close), 60)), rank(negative(ts_zscore(returns, 60)))))"),
    # H46: vol-of-vol regime × zscore-returns 21d (H40 × H7.2)
    ("H46_volofvol_x_zscoreReturns21",
     "rank(multiply(rank(ts_zscore(stddev(returns, 5), 60)), rank(negative(ts_zscore(returns, 21)))))"),
    # H47: log-range regime × decay-exp VWAPdev
    ("H47_logrange_x_DecayExp008VWAPdev",
     "rank(multiply(rank(ts_zscore(log(true_divide(high, low)), 21)), rank(decay_exp(negative(true_divide(close, vwap)), 0.08))))"),
    # H48: dollar-vol regime × decay-exp VWAPdev
    ("H48_dollarvol_x_DecayExp008VWAPdev",
     "rank(multiply(rank(ts_zscore(multiply(volume, close), 60)), rank(decay_exp(negative(true_divide(close, vwap)), 0.08))))"),
    # H49: vol-of-vol × decay-exp VWAPdev
    ("H49_volofvol_x_DecayExp008VWAPdev",
     "rank(multiply(rank(ts_zscore(stddev(returns, 5), 60)), rank(decay_exp(negative(true_divide(close, vwap)), 0.08))))"),
]

# Phase 3: lift fit by adding outer decay-3 to the SR+corr passers.
CANDIDATES_PHASE3 = [
    # H50: H48 + outer-decay 3 (dollar-vol regime × VWAPdev directional)
    ("H50_H48_outerdecay3",
     "rank(decay_linear(multiply(rank(ts_zscore(multiply(volume, close), 60)), rank(decay_exp(negative(true_divide(close, vwap)), 0.08))), 3))"),
    # H51: H45 + outer-decay 3 (dollar-vol regime × zscore-returns directional)
    ("H51_H45_outerdecay3",
     "rank(decay_linear(multiply(rank(ts_zscore(multiply(volume, close), 60)), rank(negative(ts_zscore(returns, 60)))), 3))"),
    # H52: H47 + outer-decay 3 (log-range regime × VWAPdev directional)
    ("H52_H47_outerdecay3",
     "rank(decay_linear(multiply(rank(ts_zscore(log(true_divide(high, low)), 21)), rank(decay_exp(negative(true_divide(close, vwap)), 0.08))), 3))"),
    # H53: NEW pairing — log-range × zscore-returns 21d (untested combination)
    ("H53_logrange_x_zscoreReturns21",
     "rank(multiply(rank(ts_zscore(log(true_divide(high, low)), 21)), rank(negative(ts_zscore(returns, 21)))))"),
    # H54: NEW pairing — vol-of-vol × decay-exp VWAPdev WITH outer decay 3
    ("H54_volofvol_x_DecayExpVWAP_outerdecay3",
     "rank(decay_linear(multiply(rank(ts_zscore(stddev(returns, 5), 60)), rank(decay_exp(negative(true_divide(close, vwap)), 0.08))), 3))"),
]

# Phase 4: H48 needs lighter smoothing (decay-2 not 3). Plus new triple composites.
CANDIDATES_PHASE4 = [
    # H55: H48 + outer-decay 2 (lighter than 3 to preserve SR)
    ("H55_H48_outerdecay2",
     "rank(decay_linear(multiply(rank(ts_zscore(multiply(volume, close), 60)), rank(decay_exp(negative(true_divide(close, vwap)), 0.08))), 2))"),
    # H56: TRIPLE — dollar-vol × DecayExp VWAPdev × ts_zscore returns 21d
    # Three orthogonal axes: flow regime + microstructure + return regime
    ("H56_dollarvol_x_DecayExpVWAP_x_zscoreReturns21",
     "rank(multiply(multiply(rank(ts_zscore(multiply(volume, close), 60)), rank(decay_exp(negative(true_divide(close, vwap)), 0.08))), rank(negative(ts_zscore(returns, 21)))))"),
    # H57: log-range × DecayExp VWAPdev × ts_zscore returns 21d (substitute log-range for dollar-vol)
    ("H57_logrange_x_DecayExpVWAP_x_zscoreReturns21",
     "rank(multiply(multiply(rank(ts_zscore(log(true_divide(high, low)), 21)), rank(decay_exp(negative(true_divide(close, vwap)), 0.08))), rank(negative(ts_zscore(returns, 21)))))"),
    # H58: H48 with DecayLin19 VWAPdev instead of DecayExp (different smoothing kernel)
    ("H58_dollarvol_x_DecayLin19VWAP",
     "rank(multiply(rank(ts_zscore(multiply(volume, close), 60)), rank(decay_linear(negative(true_divide(close, vwap)), 19))))"),
    # H59: H56 with outer decay 2 (light smoothing of triple)
    ("H59_H56_outerdecay2",
     "rank(decay_linear(multiply(multiply(rank(ts_zscore(multiply(volume, close), 60)), rank(decay_exp(negative(true_divide(close, vwap)), 0.08))), rank(negative(ts_zscore(returns, 21)))), 2))"),
]

# Phase 5: H56 needs HEAVIER decay to drop corr. H55/H58 need fit lift.
CANDIDATES_PHASE5 = [
    # H60: H56 + outer-decay 5 (heavier, drop corr 0.74 → ~0.65)
    ("H60_H56_outerdecay5",
     "rank(decay_linear(multiply(multiply(rank(ts_zscore(multiply(volume, close), 60)), rank(decay_exp(negative(true_divide(close, vwap)), 0.08))), rank(negative(ts_zscore(returns, 21)))), 5))"),
    # H61: H56 + outer-decay 4 (between 3 and 5)
    ("H61_H56_outerdecay4",
     "rank(decay_linear(multiply(multiply(rank(ts_zscore(multiply(volume, close), 60)), rank(decay_exp(negative(true_divide(close, vwap)), 0.08))), rank(negative(ts_zscore(returns, 21)))), 4))"),
    # H62: H58 + outer-decay 3 (lift fit)
    ("H62_H58_outerdecay3",
     "rank(decay_linear(multiply(rank(ts_zscore(multiply(volume, close), 60)), rank(decay_linear(negative(true_divide(close, vwap)), 19))), 3))"),
    # H63: H56 with zscore-returns 60d (slower directional) for natural lower TO
    ("H63_dollarvol_x_DecayExpVWAP_x_zscoreReturns60",
     "rank(multiply(multiply(rank(ts_zscore(multiply(volume, close), 60)), rank(decay_exp(negative(true_divide(close, vwap)), 0.08))), rank(negative(ts_zscore(returns, 60)))))"),
    # H64: H63 + outer-decay 3
    ("H64_H63_outerdecay3",
     "rank(decay_linear(multiply(multiply(rank(ts_zscore(multiply(volume, close), 60)), rank(decay_exp(negative(true_divide(close, vwap)), 0.08))), rank(negative(ts_zscore(returns, 60)))), 3))"),
]

# Phase 6: most-orthogonal regime detectors × VWAPdev directional + outer decay
CANDIDATES_PHASE6 = [
    # H65: argmin recency × DecayExpVWAPdev (corr 0.17 × corr 0.67)
    ("H65_argmin_x_DecayExpVWAP_outerdecay3",
     "rank(decay_linear(multiply(rank(ts_argmin(returns, 21)), rank(decay_exp(negative(true_divide(close, vwap)), 0.08))), 3))"),
    # H66: vol-of-vol × DecayLin19 VWAPdev × outer decay 2
    ("H66_volofvol_x_DecayLin19VWAP_outerdecay2",
     "rank(decay_linear(multiply(rank(ts_zscore(stddev(returns, 5), 60)), rank(decay_linear(negative(true_divide(close, vwap)), 19))), 2))"),
    # H67: log-range × DecayExpVWAPdev × outer decay 4
    ("H67_logrange_x_DecayExpVWAP_outerdecay4",
     "rank(decay_linear(multiply(rank(ts_zscore(log(true_divide(high, low)), 21)), rank(decay_exp(negative(true_divide(close, vwap)), 0.08))), 4))"),
    # H68: market_cap regime (NEW field never used) × DecayExpVWAPdev — low caps revert harder
    ("H68_invMarketCap_x_DecayExpVWAP",
     "rank(multiply(rank(negative(market_cap)), rank(decay_exp(negative(true_divide(close, vwap)), 0.08))))"),
    # H69: ts_zscore returns 90d (longer than tested 21/60) × DecayExpVWAPdev
    ("H69_zscoreReturns90_x_DecayExpVWAP",
     "rank(multiply(rank(negative(ts_zscore(returns, 90))), rank(decay_exp(negative(true_divide(close, vwap)), 0.08))))"),
]

# Phase 7: H68 BREAKTHROUGH — market_cap (unused field) breaks corr.
CANDIDATES_PHASE7 = [
    # H70: low-ADV (small-stock proxy) × DecayExpVWAPdev
    ("H70_invADV20_x_DecayExpVWAP",
     "rank(multiply(rank(negative(adv20)), rank(decay_exp(negative(true_divide(close, vwap)), 0.08))))"),
    # H71: book-to-market (value tilt) × DecayExpVWAPdev
    ("H71_BookToMarket_x_DecayExpVWAP",
     "rank(multiply(rank(book_to_market), rank(decay_exp(negative(true_divide(close, vwap)), 0.08))))"),
    # H72: inverse-pe (value) × DecayExpVWAPdev
    ("H72_invPE_x_DecayExpVWAP",
     "rank(multiply(rank(negative(pe_ratio)), rank(decay_exp(negative(true_divide(close, vwap)), 0.08))))"),
    # H73: ROE (quality) × DecayExpVWAPdev
    ("H73_ROE_x_DecayExpVWAP",
     "rank(multiply(rank(roe), rank(decay_exp(negative(true_divide(close, vwap)), 0.08))))"),
    # H74: invADV60 (slow-stock proxy) × DecayExpVWAPdev
    ("H74_invADV60_x_DecayExpVWAP",
     "rank(multiply(rank(negative(adv60)), rank(decay_exp(negative(true_divide(close, vwap)), 0.08))))"),
]

# Phase 8: fundamental × ts_zscore returns 21d
CANDIDATES_PHASE8 = [
    # H75: invADV20 × zscoreReturns21
    ("H75_invADV20_x_zscoreReturns21",
     "rank(multiply(rank(negative(adv20)), rank(negative(ts_zscore(returns, 21)))))"),
    # H76: BookToMarket × zscoreReturns21
    ("H76_BookToMarket_x_zscoreReturns21",
     "rank(multiply(rank(book_to_market), rank(negative(ts_zscore(returns, 21)))))"),
    # H77: ROE × zscoreReturns21
    ("H77_ROE_x_zscoreReturns21",
     "rank(multiply(rank(roe), rank(negative(ts_zscore(returns, 21)))))"),
    # H78: invADV60 × zscoreReturns21
    ("H78_invADV60_x_zscoreReturns21",
     "rank(multiply(rank(negative(adv60)), rank(negative(ts_zscore(returns, 21)))))"),
    # H79: invMarketCap × zscoreReturns21 (variation of H68)
    ("H79_invMarketCap_x_zscoreReturns21",
     "rank(multiply(rank(negative(market_cap)), rank(negative(ts_zscore(returns, 21)))))"),
]

# Phase 9: bump SR on fund × VWAPdev family with faster inner OR triple
CANDIDATES_PHASE9 = [
    # H80: invADV20 × DecayExp(0.10) VWAPdev — faster inner
    ("H80_invADV20_x_DecayExp010VWAP",
     "rank(multiply(rank(negative(adv20)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    # H81: BookToMarket × DecayExp(0.10) VWAPdev
    ("H81_BookToMarket_x_DecayExp010VWAP",
     "rank(multiply(rank(book_to_market), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    # H82: invADV60 × DecayExp(0.10) VWAPdev
    ("H82_invADV60_x_DecayExp010VWAP",
     "rank(multiply(rank(negative(adv60)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    # H83: ROE × DecayExp(0.10) VWAPdev
    ("H83_ROE_x_DecayExp010VWAP",
     "rank(multiply(rank(roe), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    # H84: TRIPLE — invMarketCap × DecayExpVWAPdev × ts_zscore returns 21d
    ("H84_invMarketCap_x_DecayExpVWAP_x_zscoreReturns21",
     "rank(multiply(multiply(rank(negative(market_cap)), rank(decay_exp(negative(true_divide(close, vwap)), 0.08))), rank(negative(ts_zscore(returns, 21)))))"),
]

# === Autonomous overnight queue: 30 distinct fundamental-field hypotheses ===
# Recipe (proven by H68/H72/H80-82): rank(FUND_FIELD) × DecayExp(0.10) VWAPdev
# Each FUND_FIELD is a different economic mechanism — not a sweep.
CANDIDATES = [
    # === VALUE family (different valuation denominators) ===
    ("H85_FCFYield_x_DecayExp010VWAP",
     "rank(multiply(rank(free_cashflow_yield), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    ("H86_invEVEBITDA_x_DecayExp010VWAP",
     "rank(multiply(rank(negative(ev_to_ebitda)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    ("H87_invEVFCF_x_DecayExp010VWAP",
     "rank(multiply(rank(negative(ev_to_fcf)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    ("H88_invEVOCF_x_DecayExp010VWAP",
     "rank(multiply(rank(negative(ev_to_ocf)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    ("H89_invEVRev_x_DecayExp010VWAP",
     "rank(multiply(rank(negative(ev_to_revenue)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    ("H90_DividendYield_x_DecayExp010VWAP",
     "rank(multiply(rank(dividend_yield), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    ("H91_EarningsYield_x_DecayExp010VWAP",
     "rank(multiply(rank(earnings_yield), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    # === QUALITY family (different profitability ratios) ===
    ("H92_ROA_x_DecayExp010VWAP",
     "rank(multiply(rank(roa), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    ("H93_ROIC_x_DecayExp010VWAP",
     "rank(multiply(rank(roic), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    ("H94_ROCE_x_DecayExp010VWAP",
     "rank(multiply(rank(roce), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    ("H95_GrossMargin_x_DecayExp010VWAP",
     "rank(multiply(rank(gross_margin), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    ("H96_OperatingMargin_x_DecayExp010VWAP",
     "rank(multiply(rank(operating_margin), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    ("H97_NetMargin_x_DecayExp010VWAP",
     "rank(multiply(rank(net_margin), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    ("H98_EBITDAMargin_x_DecayExp010VWAP",
     "rank(multiply(rank(ebitda_margin), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    # === SAFETY/LEVERAGE family ===
    ("H99_invDE_x_DecayExp010VWAP",
     "rank(multiply(rank(negative(debt_to_equity)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    ("H100_invDebtAssets_x_DecayExp010VWAP",
     "rank(multiply(rank(negative(debt_to_assets)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    ("H101_CurrentRatio_x_DecayExp010VWAP",
     "rank(multiply(rank(current_ratio), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    ("H102_QuickRatio_x_DecayExp010VWAP",
     "rank(multiply(rank(quick_ratio), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    ("H103_CashRatio_x_DecayExp010VWAP",
     "rank(multiply(rank(cash_ratio), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    ("H104_invNetDebt_x_DecayExp010VWAP",
     "rank(multiply(rank(negative(net_debt)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    ("H105_InterestCoverage_x_DecayExp010VWAP",
     "rank(multiply(rank(interest_coverage), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    # === EFFICIENCY family ===
    ("H106_AssetTurnover_x_DecayExp010VWAP",
     "rank(multiply(rank(asset_turnover), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    ("H107_InventoryTurnover_x_DecayExp010VWAP",
     "rank(multiply(rank(inventory_turnover), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    # === GROWTH change-based (year-over-year fundamental change) ===
    ("H108_ROEgrowth_x_DecayExp010VWAP",
     "rank(multiply(rank(ts_delta(roe, 252)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    ("H109_BMgrowth_x_DecayExp010VWAP",
     "rank(multiply(rank(ts_delta(book_to_market, 63)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    ("H110_MarketCapGrowth_x_DecayExp010VWAP",
     "rank(multiply(rank(negative(ts_delta(market_cap, 252))), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    # === MISC unused fields ===
    ("H111_BookPS_x_DecayExp010VWAP",
     "rank(multiply(rank(book_value_per_share), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    ("H112_RevenuePS_x_DecayExp010VWAP",
     "rank(multiply(rank(revenue_per_share), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    ("H113_GrahamNumber_x_DecayExp010VWAP",
     "rank(multiply(rank(graham_number), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    ("H114_invInterestExp_x_DecayExp010VWAP",
     "rank(multiply(rank(negative(interest_expense)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
]


def proc_signal(s, uni, cls):
    s = s.astype(float).where(uni, np.nan)
    for g in cls.dropna().unique():
        m = (cls == g).values
        if m.any():
            sub = s.iloc[:, m]; s.iloc[:, m] = sub.sub(sub.mean(axis=1), axis=0)
    ab = s.abs().sum(axis=1).replace(0, np.nan)
    return s.div(ab, axis=0).clip(-MAX_W, MAX_W).fillna(0)


def realistic_cost(w, close, book):
    pos = w * book
    trd = pos.diff().abs()
    safe = close.where(close > 0)
    shares = trd / safe
    pn_comm = (shares * COMMISSION_PER_SHARE).clip(lower=0)
    has = trd > 1.0
    pn_comm = pn_comm.where(~has, np.maximum(pn_comm, PER_ORDER_MIN)).where(has, 0)
    cost = (pn_comm.sum(axis=1)
            + (trd * SEC_FEE_PER_DOLLAR * SELL_FRACTION).sum(axis=1)
            + (trd * IMPACT_BPS / 1e4).sum(axis=1)
            + (-pos.clip(upper=0)).sum(axis=1) * (BORROW_BPS_ANNUAL / 1e4) / 252.0
           ) / book
    return cost


def main():
    print(f"=== Loading universe + matrices ({UNIV_NAME}) ===")
    uni = pd.read_parquet(UNIV_DIR / f"{UNIV_NAME}.parquet").astype(bool)
    if not isinstance(uni.index, pd.DatetimeIndex):
        uni.index = pd.to_datetime(uni.index)
    cov = uni.sum(axis=0)/len(uni); valid = sorted(cov[cov>0.5].index.tolist())
    uni = uni[valid]; dates = uni.index; tickers = uni.columns.tolist()
    print(f"  {len(tickers)} tickers, {len(dates)} dates")

    mats = {}
    for fp in sorted(DATA_DIR.glob("*.parquet")):
        if fp.stem.startswith("_"): continue
        try: df = pd.read_parquet(fp)
        except: continue
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce"); df = df[df.index.notna()]
        cc = [c for c in df.columns if c in tickers]
        if cc: mats[fp.stem] = df.loc[df.index.isin(dates), cc].reindex(index=dates, columns=tickers)

    close = mats["close"]
    ret = close.pct_change(fill_method=None)
    cls = pd.read_parquet(DATA_DIR / "subindustry.parquet").iloc[-1].reindex(tickers)
    engine = FastExpressionEngine(data_fields=mats)
    print(f"  Loaded {len(mats)} fields")

    # Existing alphas for orthogonality
    print("\n=== Loading existing SMALLCAP_D0 alphas for correlation ===")
    rows = sqlite3.connect(DB).execute("""
        SELECT a.id, a.expression FROM alphas a
         WHERE a.archived=0 AND a.notes LIKE '%SMALLCAP_D0%'
         ORDER BY a.id""").fetchall()
    existing = {}
    for aid, expr in rows:
        try:
            sig = proc_signal(engine.evaluate(expr), uni, cls)
            existing[f"#{aid}"] = sig
        except Exception as e:
            print(f"  skip #{aid}: {e}")
    print(f"  {len(existing)} existing alphas loaded")

    # TRAIN window only — VAL/TEST are not consulted for selection
    train_mask = (dates >= TRAIN_START) & (dates < TRAIN_END)
    nx = ret.shift(-1)
    ann = np.sqrt(252)

    # Restrict existing alpha signals to TRAIN window for orthogonality comp
    existing_train = {k: v.loc[train_mask] for k, v in existing.items()}

    print("\n=== Testing candidates (TRAIN-only — no VAL/TEST peek) ===\n")
    results = []
    for name, expr in CANDIDATES:
        try:
            raw = engine.evaluate(expr)
        except Exception as e:
            print(f"[ERR ] {name:30s}  parse/eval failed: {e}")
            continue

        sig = proc_signal(raw, uni, cls)
        sig_train = sig.loc[train_mask]

        # gross daily PnL on TRAIN
        g_tr = (sig_train * nx.loc[train_mask]).sum(axis=1).fillna(0)
        cost_tr = realistic_cost(sig_train, close.loc[train_mask], BOOK)
        n_tr = g_tr - cost_tr

        sr_train = g_tr.mean()/g_tr.std()*ann if g_tr.std()>0 else float('nan')
        nsr_train = n_tr.mean()/n_tr.std()*ann if n_tr.std()>0 else float('nan')
        to = sig_train.diff().abs().sum(axis=1).mean() / 2
        ret_ann = g_tr.mean() * 252
        fitness = sr_train * np.sqrt(abs(ret_ann) / max(to, 0.125)) if not np.isnan(sr_train) else float('nan')

        # corr on TRAIN window only
        a_flat = sig_train.values.flatten()
        a_msk = ~np.isnan(a_flat) & (a_flat != 0)
        max_corr = 0.0
        for k, e in existing_train.items():
            b_flat = e.values.flatten()
            mm = a_msk & ~np.isnan(b_flat) & (b_flat != 0)
            if mm.sum() < 1000: continue
            c = abs(float(np.corrcoef(a_flat[mm], b_flat[mm])[0,1]))
            if c > max_corr: max_corr = c

        flag = "PASS" if (sr_train >= MIN_TRAIN_SR
                          and fitness >= MIN_TRAIN_FITNESS
                          and max_corr < MAX_CORR) else "----"
        print(f"[{flag}] {name:30s}  SR_g_train={sr_train:+5.2f}  fit={fitness:5.2f}  "
              f"ret={ret_ann*100:+5.1f}%  TO={to*100:.1f}%/d  max|corr|={max_corr:.2f}")
        results.append((name, expr, sr_train, nsr_train, to, max_corr, flag, fitness, ret_ann))

    print("\n=== Summary: PASS list (TRAIN only) ===")
    for r in results:
        if r[-1] == "PASS":
            print(f"  {r[0]:30s}  SR={r[2]:.2f}  fit={r[7]:.2f}  ret={r[8]*100:+.1f}%  "
                  f"TO={r[4]*100:.1f}%/d  max|corr|={r[5]:.2f}")
            print(f"      expr: {r[1]}")
    print(f"\n{sum(1 for r in results if r[-1]=='PASS')}/{len(results)} candidates pass "
          f"(gate SR>={MIN_TRAIN_SR}, fit>={MIN_TRAIN_FITNESS}, corr<{MAX_CORR})")


if __name__ == "__main__":
    main()
