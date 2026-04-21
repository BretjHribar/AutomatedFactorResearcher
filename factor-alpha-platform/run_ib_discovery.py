"""
run_ib_discovery.py -- Autonomous Wave-2 Alpha Discovery for IB Closing Auction

Evaluates every candidate in seed_alphas_ib.py (Wave 2 section), then
auto-generates new variations until DB has 10 alphas (beyond the 1 already saved).

Gates (from eval_alpha_ib.py):
  IS Sharpe >= 3.0, Fitness >= 1.0, IC > 0, Sub-period > 0.3,
  Turnover <= 0.50, Kurtosis <= 25, Rolling SR std <= 0.30, Skew >= -0.5

Orthogonality: corr < 0.65 with existing DB alphas.

Runs endlessly until TARGET_NEW = 10 new alphas are saved.
"""
import sys, os, time, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONIOENCODING"] = "utf-8"

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

# ---- Import eval harness ----
import eval_alpha_ib as ib

TARGET_NEW    = 10   # stop when this many new alphas accepted
MIN_IS_SHARPE = ib.MIN_IS_SHARPE   # 3.0

# ============================================================================
# EXTRA CANDIDATES beyond seed_alphas_ib.py wave 2
# Generated on-the-fly when wave 2 isn't enough.
# Each entry: (expr, name, category, reasoning)
# ============================================================================

EXTRA_CANDIDATES = [
    # TIER A: DECAY-SMOOTHED BORDERLINE SIGNALS (SR 2.9-3.0 that almost passed)
    ("rank(Decay_exp(-(close - ts_min(low, 5)) / (ts_max(high, 5) - ts_min(low, 5) + 0.001), 0.7))",
     "w4_5d_range_exhaustion_decay07", "candle",
     "5-day range exhaustion with Decay_exp(0.7). Raw version SR=+2.921, just missed gate. "
     "5-day window has lower correlation with alpha #1 (today-only) than 3-day version."),

    ("rank(Decay_lin(-(close - ts_min(low, 5)) / (ts_max(high, 5) - ts_min(low, 5) + 0.001), 3))",
     "w4_5d_range_exhaustion_lin3", "candle",
     "5-day range exhaustion with 3-day linear decay. Weights recent days more — more "
     "responsive than exponential. Another approach to push SR over 3.0 while keeping "
     "the 5-day context that differentiates it from alpha #1."),

    ("rank(Decay_exp((low - close) / (high - low + 0.001), 0.8) + Decay_exp(-returns, 0.8))",
     "w4_exhaustion_plus_return_decay", "composite",
     "Decayed exhaustion PLUS decayed return reversal (additive, not product). "
     "w3_exhaustion_times_return_rev hit SR=+2.987. Additive combination is less "
     "noisy than product. Two decayed components sum to push over SR=3.0."),

    # TIER B: TIME-SERIES ZSCORE SIGNALS (own-history normalization, different from CS rank)
    ("rank(-ts_zscore(close, 20))",
     "w4_ts_zscore_close_20d", "reversal",
     "Rank of negative 20-day time-series z-score. Each stock's own deviation from "
     "its 20-day mean/std. Pure time-series reversal — orthogonal to cross-sectional "
     "close_near_low because it measures vs own history, not today's H-L range."),

    ("rank(-ts_zscore(returns, 10))",
     "w4_ts_zscore_returns_10d", "reversal",
     "Negative 10-day ts_zscore of returns. Each stock's return vs its own 10-day "
     "return distribution. High rank = unusually bad return day by own standards. "
     "Vol-adjusted, scale-free, different information from price-level signals."),

    ("rank(Decay_exp(-ts_zscore(close, 15), 0.85))",
     "w4_ts_zscore_decay_15d", "reversal",
     "Decayed 15-day ts_zscore reversal. Smoothing reduces flip noise. "
     "15-day window = 3-week regime, between weekly (5d) and monthly (20d)."),

    # TIER C: PRICE VELOCITY/ACCELERATION (2nd-order dynamics, orthogonal to level signals)
    ("rank(-ts_delta(returns, 1))",
     "w4_return_deceleration", "reversal",
     "Negative first-difference of returns = return deceleration. Buy stocks where "
     "today's return > yesterday's (momentum slowing). Second-derivative of price — "
     "fundamentally different from all level-based candle signals."),

    ("rank(ts_delta(ts_delta(close, 3), 3))",
     "w4_price_acc_3d", "reversal",
     "3-day price acceleration: ts_delta of ts_delta(close,3). Positive = 3-day losses "
     "are decelerating = selling is running out of steam. Different timescale and "
     "mathematical structure from any level-based reversal."),

    # TIER D: TIME-SERIES RANK REVERSALS (different from cross-sectional rank)
    ("rank(-ts_rank(close, 10))",
     "w4_tsrank_10d_reversal", "reversal",
     "Rank of negative 10-day time-series percentile. -ts_rank(close,10) is high when "
     "today's close is near its 10-day low. Different from alpha #1: uses ranking "
     "within 10-day window (multi-day context), not absolute H-L distance."),

    ("rank(-ts_rank(close, 20))",
     "w4_tsrank_20d_reversal", "reversal",
     "Stocks at 20-day lows tend to revert. Monthly horizon reversal vs alpha #1's "
     "intraday horizon. Lower correlation expected due to different timescale."),

    # TIER E: VOLUME-DRIVEN SIGNALS (orthogonal data source)
    ("rank(ts_rank(volume, 20))",
     "w4_volume_persistence", "volume",
     "High relative volume stocks trend. vol_rank_reversal (-ts_rank) had SR=-2.34 "
     "so its POSITIVE form should be ~+2.34. Tests volume momentum as independent factor."),

    ("rank(-ts_delta(close, 1) - ts_delta(close, 3) - ts_delta(close, 5))",
     "w4_multi_reversal_sum", "reversal",
     "Sum of 1+3+5 day reversals. Stocks weak across all three horizons are more "
     "thoroughly exhausted. Lower TO than 1-day alone. Less correlated with alpha #1 "
     "(uses close-to-close not H-L range)."),

    ("rank(Decay_lin(-ts_delta(close, 1) - ts_delta(close, 3) - ts_delta(close, 5), 5))",
     "w4_multi_reversal_lin5", "reversal",
     "5-day linear decay of multi-horizon reversal. Further reduces turnover. "
     "Decaying the sum preserves recency while smoothing daily oscillations."),

    # TIER F: INTRADAY Z-SCORE (self-normalized intraday move)
    ("rank(-(close - open) / (ts_std_dev(close - open, 20) + 0.001))",
     "w4_intraday_z_reversal", "candle",
     "Intraday open-to-close decline normalized by 20-day std of intraday moves. "
     "Each stock's OWN intraday volatility as denominator. A stock dropping 2x its "
     "normal daily range is a stronger buy than one dropping within normal range."),

    ("rank(Decay_exp(-(close - open) / (ts_std_dev(close - open, 20) + 0.001), 0.85))",
     "w4_intraday_z_decay", "candle",
     "Decayed intraday z-score reversal. Expected corr with alpha #1: ~0.4-0.5 "
     "(same direction, different normalization: own-vol vs H-L range)."),

    # TIER G: VWAP SMOOTHED
    ("rank(Decay_exp(vwap - close, 0.85))",
     "w4_vwap_gap_decayed", "reversal",
     "Decayed (vwap - close). w3_close_below_vwap had SR=+3.375 but corr=0.810 with #1. "
     "Smoothing changes the signal profile — may reduce daily alignment with #1 "
     "while preserving directional content."),

    ("rank(Decay_exp((vwap - close) / (high - low + 0.001), 0.85))",
     "w4_vwap_range_norm_decayed", "reversal",
     "Decayed VWAP-to-range ratio. vwap_deviation_range_norm had SR=+4.525 but corr=0.937. "
     "Smoothing may reduce corr enough (target <0.65) since daily alignment is what "
     "drives correlation structure in cross-sectional signals."),

    # TIER H: ADV-RELATIVE REVERSAL (liquidity-adjusted)
    ("rank(-ts_delta(close, 1) / (ts_std_dev(close, 20) + 0.001))",
     "w4_reversal_vol_adj", "reversal",
     "1-day reversal normalized by 20-day price volatility. Same direction as reversal_1d "
     "but weights larger drops relative to the stock's own recent price vol. "
     "Different cross-sectional ordering: a 2-sigma 1-day drop ranks higher than same-$ drop "
     "for a high-vol name."),

    # TIER I: CROSS-SECTIONAL CORRELATION SIGNALS (entirely different mechanism)
    ("rank(-ts_corr(close, volume, 5))",
     "w4_pv_anticorr_5d", "volume",
     "Negative 5-day price-volume correlation. Low/negative corr = price moved without "
     "volume confirmation = likely unwind. Completely different signal family."),

    ("rank(-ts_corr(returns, volume, 10))",
     "w4_ret_vol_anticorr_10d", "volume",
     "Negative 10-day return-volume correlation. Returns vs volume (not price levels). "
     "Stocks where returns are negatively correlated with volume over 10 days tend to "
     "mean-revert. Orthogonal to all single-day candle signals by construction."),

    # TIER J: ABSOLUTE EXHAUSTION MAGNITUDE
    ("rank(Decay_exp(-(close - ts_min(low, 3)) / (ts_max(high, 3) - ts_min(low, 3) + 0.001), 0.5))",
     "w4_3d_exhaustion_decay05", "candle",
     "3-day range exhaustion with heavy decay (alpha=0.5, halflife~1 day). "
     "Faster smoothing than alpha #5 (decay 0.7). Tests if a harder-smoothed version "
     "gets different corr profile vs alpha #1."),

    ("rank(Decay_exp(-(close - ts_min(low, 3)) / (ts_max(high, 3) - ts_min(low, 3) + 0.001), 0.85))",
     "w4_3d_exhaustion_decay085", "candle",
     "3-day range exhaustion with lighter decay (alpha=0.85, halflife~4 days). "
     "Less aggressive smoothing than alpha #5. Different balance of responsiveness "
     "vs noise reduction."),

    # =========================================================================
    # WAVE 23: NON-EXHAUSTION ORTHOGONAL SIGNALS
    # DB: 15 alphas. FCF_yield + exhaustion = #18 (SR=3.54). 
    # Exhaustion core is saturated (corr > 0.70 with #1 on all variants).
    # ts_zscore(close) saturated by #17. 
    # Need new PRIMITIVES: vwap, multi-day returns, lagged signals, fundamental-only.
    # =========================================================================

    # W23-1: VWAP distance as standalone (not decayed)
    # vwap - close = how far below VWAP the close is
    # Large positive = stock closed well below VWAP = seller exhaustion
    ("rank(vwap - close)",
     "w23_vwap_distance", "reversal",
     "VWAP minus close price (raw, no decay). "
     "Large positive = closed below VWAP = institutional selling pressure. "
     "Different from exhaustion (which uses H-L range, not VWAP). "
     "Previously: Decay_exp(vwap-close,0.85) had SR=4.198 but corr=0.727 with #1."),

    # W23-2: ts_zscore(vwap - close, 20) — normalized VWAP deviation
    # How anomalous is today's VWAP distance vs last 20 days?
    ("rank(-ts_zscore(vwap - close, 20))",
     "w23_vwap_zscore", "reversal",
     "Negative 20d z-score of (vwap - close). "
     "Measures how anomalous today's VWAP distance is vs recent history. "
     "Different from #17 (which z-scores close itself, not VWAP deviation)."),

    # W23-3: Multi-day return reversal (5d) — pure price momentum reversal
    # 5-day cumulative return reversal (slower than 1-day)
    ("rank(Decay_exp(-ts_delta(close, 5), 0.85))",
     "w23_5d_return_decay", "reversal",
     "Decay_exp(0.85) of negative 5-day return. "
     "5-day reversal captures weekly oversold. "
     "Different timescale from 1-day (alpha #14) and 20d (alpha #17)."),

    # W23-4: FCF yield + ts_zscore(close, 20) product
    # rank(FCF) * rank(reversal) — conditional: only cash-cheap stocks oversold
    ("rank(ts_rank(free_cashflow_yield, 60)) * rank(-ts_zscore(close, 20))",
     "w23_fcf_times_zscore", "composite",
     "rank(FCF yield ts_rank) × rank(negative 20d close z-score). "
     "Product: only cash-cheap stocks at z-score lows pass. "
     "Low turnover by construction (both components are slow-moving)."),

    # W23-5: Earnings yield + ts_zscore(close, 20) product
    # Value × mean-reversion conditional
    ("rank(ts_rank(earnings_yield, 60)) * rank(-ts_zscore(close, 20))",
     "w23_earnings_times_zscore", "composite",
     "rank(earnings yield ts_rank) × rank(negative 20d close z-score). "
     "Cheap-earnings stocks at z-score lows = deep value reversal. "
     "Product conditional: both must be extreme to rank high."),

    # W23-6: Operating margin + ts_zscore (quality × mean-reversion)
    ("rank(ts_rank(operating_margin, 60)) * rank(-ts_zscore(close, 20))",
     "w23_op_margin_times_zscore", "composite",
     "rank(op margin ts_rank) × rank(negative 20d close z-score). "
     "High-margin companies at 20d z-score lows = quality reversal. "
     "Different from #22-9 (which used net_margin, SR=2.198)."),

    # W23-7: Decay-wrapped multi-reversal with longer decay
    # ts_delta(close,1) + ts_delta(close,3) with Decay_exp(0.7)
    ("rank(Decay_exp(-ts_delta(close, 1) - ts_delta(close, 3), 0.7))",
     "w23_multi_rev_decay07", "composite",
     "Decay_exp(0.7) of negative 1d+3d returns. "
     "Alpha #10 adds vol_rank. This is pure multi-day reversal with decay. "
     "Decay(0.7) gives ~3 day half-life = moderate smoothing for lower TO."),

    # W23-8: FCF yield + volume (additive revisit)
    # W20 had SR=2.836. Now with different universe, may pass.
    ("rank(ts_rank(free_cashflow_yield, 60) + ts_rank(volume, 20))",
     "w23_fcf_yield_plus_volume", "composite",
     "60d ts_rank of FCF yield + volume rank. "
     "FCF is cash-based value signal. Alpha #15 uses earnings_yield. "
     "Previously SR=2.836 in old universe. May be different now."),

    # W23-9: Debt-to-equity inverse + volume (financial quality + activity)
    ("rank(-ts_rank(debt_to_equity, 60) + ts_rank(volume, 20))",
     "w23_low_debt_equity_plus_volume", "composite",
     "Negative debt-to-equity ts_rank + volume rank. "
     "Low leverage stocks with trading activity = quality catalyst. "
     "Different from debt_to_assets (leverage measured vs equity not assets)."),

    # W23-10: Interest coverage + volume (financial strength + activity)
    ("rank(ts_rank(interest_coverage, 60) + ts_rank(volume, 20))",
     "w23_interest_coverage_plus_volume", "composite",
     "60d ts_rank of interest coverage + volume rank. "
     "Interest coverage = EBIT/interest expense = ability to service debt. "
     "High coverage = financially strong. With volume = quality catalyst."),

    # W23-11: ts_zscore(close,30) — 30-day z-score (between #17's 20 and W22's 60)
    ("rank(-ts_zscore(close, 30))",
     "w23_zscore_close_30d", "reversal",
     "Negative 30d z-score of close. "
     "Between alpha #17's 20d and W22's 60d. "
     "6-week mean reversion = different investor horizon."),

    # W23-12: FCF yield + ts_delta(close,5) reversal (value + 5d reversal)
    ("rank(ts_rank(free_cashflow_yield, 60) + (-ts_delta(close, 5)))",
     "w23_fcf_plus_5d_reversal", "composite",
     "FCF yield ts_rank + negative 5-day return. "
     "Cash-cheap stocks that fell over the past week = value reversal. "
     "5-day window is between daily (#18 uses exhaustion) and #17 (20d zscore)."),

    # W23-13: Net margin + FCF yield product (quality × value, no price)
    ("rank(ts_rank(net_margin, 60)) * rank(ts_rank(free_cashflow_yield, 60))",
     "w23_net_margin_times_fcf", "composite",
     "rank(net margin) × rank(FCF yield) — pure fundamental product. "
     "Profitable AND cash-cheap stocks. No price signal at all. "
     "Extremely low turnover expected (both fundamentals move slowly)."),

    # W23-14: Earnings yield + operating margin product (value × quality)
    ("rank(ts_rank(earnings_yield, 60)) * rank(ts_rank(operating_margin, 60))",
     "w23_earnings_times_op_margin", "composite",
     "rank(earnings yield) × rank(operating margin). "
     "Value AND quality simultaneously. No price/volume component. "
     "Pure fundamental conditional signal."),

    # W23-15: Decay_exp(vwap - close, 0.5) — slow-decayed VWAP distance
    ("rank(Decay_exp(vwap - close, 0.5))",
     "w23_vwap_distance_decay05", "reversal",
     "Decay_exp(0.5) of VWAP distance. "
     "More moderate decay than the W4's 0.85 version (which had corr=0.727 with #1). "
     "Slower decay may create different enough cross-section from both #1 and VWAP raw."),
]

def evaluate_one(expr, name, category, reasoning, conn):
    """Run eval_alpha_ib.eval_full on a single expression. Returns full result dict."""
    print(f"\n  [{name}]", flush=True)
    print(f"  Expr: {expr}", flush=True)
    t0 = time.time()
    try:
        result = ib.eval_full(expr, conn)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None, False
    elapsed = time.time() - t0

    if not result["success"]:
        print(f"  FAILED: {result['error']}")
        return result, False

    sr = result["is_sharpe"]
    fit = result["is_fitness"]
    ic = result["ic_mean"]
    to_ = result["turnover"]
    h1 = result["stability_h1"]
    h2 = result["stability_h2"]
    kurt = result["pnl_kurtosis"]
    skew = result["pnl_skew"]
    rsr = result["rolling_sr_std"]

    print(f"  SR={sr:+.3f}  Fit={fit:.3f}  IC={ic:+.4f}  TO={to_:.3f}  "
          f"H1={h1:+.2f} H2={h2:+.2f}  Kurt={kurt:.1f}  [{elapsed:.1f}s]")

    # Check gates
    gates = {
        "SR":      sr >= ib.MIN_IS_SHARPE,
        "Fitness": fit >= ib.MIN_FITNESS,
        "IC":      ic > ib.MIN_IC_MEAN,
        "H1":      h1 > ib.MIN_SUB_SHARPE,
        "H2":      h2 > ib.MIN_SUB_SHARPE,
        "TO":      to_ <= ib.MAX_TURNOVER,
        "Kurt":    kurt <= ib.MAX_PNL_KURTOSIS,
        "Skew":    skew >= ib.MIN_PNL_SKEW,
        "RSR":     rsr <= ib.MAX_ROLLING_SR_STD,
    }
    fails = [k for k, v in gates.items() if not v]
    if fails:
        print(f"  GATE FAIL: {', '.join(fails)}")
        ib.log_trial(conn, expr, sr, saved=False)
        conn.commit()
        return result, False

    # Orthogonality check
    saved = ib.save_alpha(conn, expr, reasoning, result, category)
    if saved:
        conn.execute(
            "UPDATE trial_log SET saved=1 WHERE trial_id=(SELECT MAX(trial_id) FROM trial_log)"
        )
        conn.commit()
        print(f"  *** SAVED (SR={sr:+.3f}) ***")
        return result, True
    else:
        print(f"  REJECTED: correlation too high with existing alpha")
        return result, False


def count_saved(conn):
    return conn.execute("SELECT COUNT(*) FROM alphas WHERE archived=0").fetchone()[0]


def main():
    conn = ib.get_conn()
    start_count = count_saved(conn)
    accepted = 0
    target = TARGET_NEW

    print(f"\n{'='*80}")
    print(f"IB ALPHA DISCOVERY — TARGET: {target} NEW ALPHAS (gate SR >= {MIN_IS_SHARPE})")
    print(f"Starting DB count: {start_count} alphas")
    print(f"{'='*80}")

    # ---- Phase 1: Wave 2 from seed_alphas_ib.py ----
    from seed_alphas_ib import SEED_ALPHAS
    wave2 = [a for a in SEED_ALPHAS if a["name"].startswith("w2")]
    
    print(f"\n--- PHASE 1: Wave 2 Candidates ({len(wave2)} alphas) ---")
    for alpha in wave2:
        if accepted >= target:
            break
        _, saved = evaluate_one(
            alpha["expr"], alpha["name"], alpha["category"], alpha["reasoning"], conn
        )
        if saved:
            accepted += 1
            print(f"  Progress: {accepted}/{target} accepted")

    # ---- Phase 2: Extra candidates ----
    if accepted < target:
        print(f"\n--- PHASE 2: Extended Candidates ({len(EXTRA_CANDIDATES)} alphas) ---")
        for expr, name, category, reasoning in EXTRA_CANDIDATES:
            if accepted >= target:
                break
            _, saved = evaluate_one(expr, name, category, reasoning, conn)
            if saved:
                accepted += 1
                print(f"  Progress: {accepted}/{target} accepted")

    # ---- Final report ----
    final_count = count_saved(conn)
    print(f"\n{'='*80}")
    print(f"DISCOVERY COMPLETE")
    print(f"  Started with: {start_count} alphas")
    print(f"  Accepted:     {accepted} new alphas")
    print(f"  Final DB:     {final_count} alphas")
    print(f"{'='*80}")

    # Print scoreboard
    rows = conn.execute("""
        SELECT a.id, a.expression, ROUND(e.sharpe_is,3), ROUND(e.ic_mean,4),
               ROUND(e.turnover,3), ROUND(e.fitness,3)
        FROM alphas a LEFT JOIN evaluations e ON e.alpha_id = a.id
        WHERE a.archived = 0
        ORDER BY e.sharpe_is DESC
    """).fetchall()
    print(f"\n  {'ID':<4} {'SR_IS':>7} {'IC':>7} {'TO':>6} {'Fit':>6}  Expression")
    print(f"  {'-'*90}")
    for r in rows:
        aid, expr, sr, ic, to_, fit = r
        print(f"  #{aid:<3} {sr:>7.3f} {ic:>7.4f} {to_:>6.3f} {fit:>6.3f}  {expr[:60]}")

    conn.close()


if __name__ == "__main__":
    main()
