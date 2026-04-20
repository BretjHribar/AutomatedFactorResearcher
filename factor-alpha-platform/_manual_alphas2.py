"""
Manual alpha design - Round 2.
Focus: structurally different from existing beta-momentum family.
Uses ts_kurtosis, Decay_exp on cross-sectional signals, and combined momentum structures.
"""
import sys, os, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import eval_alpha as ea
ea.UNIVERSE = "BINANCE_TOP50"
ea.INTERVAL = "4h"
ea.TRAIN_START = "2021-01-01"
ea.TRAIN_END   = "2025-01-01"
ea.SUBPERIODS = [
    ("2021-01-01", "2023-01-01", "H1"),
    ("2023-01-01", "2025-01-01", "H2"),
]
ea.BARS_PER_DAY = 6
ea.MAX_WEIGHT = 0.10
ea.COVERAGE_CUTOFF = 0.3
ea.MIN_IS_SHARPE = 1.5
ea.CORR_CUTOFF = 0.70
ea.MAX_TURNOVER = 0.10
ea.MIN_SUB_SHARPE = 1.0
ea.MAX_ROLLING_SR_STD = 0.05
ea.MIN_PNL_SKEW = -0.5
ea._DATA_CACHE.clear()

MIN_SR = 1.5
MAX_TO = 0.10
MIN_FITNESS = 5.0

conn = ea.get_conn()
ea.ensure_trial_log(conn)

ALPHAS = [
    # ── Volume × kurtosis (alpha #10 variant) ──
    # #10: multiply(ts_delta(close, 6), ts_kurtosis(volume, 60))
    # Hypothesis: fat-tailed volume (kurtosis) = unusual activity →
    # price move in direction of fat tail. Use longer windows for less churn.
    (
        "close_delta_vol_kurtosis_long",
        "multiply(ts_delta(close, 24), ts_kurtosis(volume, 120))"
    ),
    (
        "close_delta_vol_kurtosis_mid",
        "multiply(ts_delta(close, 12), ts_kurtosis(volume, 90))"
    ),

    # ── Returns skewness (tail risk) ──
    # Hypothesis: positive returns skewness = lottery-like asset → fade (overpriced).
    # Negative skewness = crash risk but mean-reverts → also fade (short).
    # Bet against positive skew, with vol.
    (
        "returns_skew_vol_fade",
        "multiply(ts_delta(close, 12), ts_skewness(log_returns, 60))"
    ),

    # ── Open-close range × close position ──
    # Hypothesis: strong close in range (close near high) AND large body = momentum bar.
    # Similar structure to #15 which worked, but with different normalization.
    (
        "body_close_position",
        "sma(multiply(true_divide(open_close_range, df_max(high_low_range, 0.0001)), close_position_in_range), 6)"
    ),

    # ── Momentum × vol-normalized (Sharpe of momentum) ──
    # Hypothesis: momentum that's large relative to volatility = conviction move.
    # Like a realized Sharpe ratio over 20 days.
    (
        "momentum_sharpe_20d",
        "sma(multiply(ts_delta(close, 20), true_divide(ts_sum(log_returns, 30), df_max(historical_volatility_20, 0.0001))), 3)"
    ),
    (
        "momentum_sharpe_60d",
        "sma(multiply(ts_delta(close, 60), true_divide(ts_sum(log_returns, 60), df_max(historical_volatility_60, 0.0001))), 3)"
    ),

    # ── Parkinson vol × beta-delta: low-vol, decoupling alpha ──
    # Hypothesis: Low historical range (Parkinson vol), AND price decoupling from BTC
    # = idiosyncratic (uncorrelated) alpha. High conviction signal.
    (
        "parkinson_beta_decoupling",
        "Decay_exp(multiply(ts_delta(close, 12), true_divide(subtract(beta_to_btc, sma(beta_to_btc, 60)), df_max(parkinson_volatility_60, 0.0001))), 0.05)"
    ),

    # ── Cross-sectional: Volume z-score + price momentum ──
    # Hypothesis: tokens that are attracting unusual volume (high CS z-score)
    # AND moving up = breakout with conviction.
    (
        "vol_zscore_momentum",
        "df_min(df_max(add(add(zscore_cs(sma(volume_ratio_20d, 60)), zscore_cs(sma(ts_delta(s_log_1p(adv60), 30), 120))), zscore_cs(sma(taker_buy_ratio, 120))), -1.5), 1.5)"
    ),
    (
        "vol_zscore_funding_taker",
        "df_min(df_max(add(add(zscore_cs(sma(volume_ratio_20d, 60)), zscore_cs(sma(funding_rate_zscore, 60))), zscore_cs(sma(ts_delta(trades_per_volume, 30), 60))), -1.5), 1.5)"
    ),

    # ── ts_regression residual (alpha orthogonal to vol) ──
    # Hypothesis: return component NOT explained by realized vol = pure alpha.
    # Captures idiosyncratic return signal.
    (
        "regression_residual_vol_taker",
        "df_min(df_max(add(add(zscore_cs(sma(ts_regression(log_returns, historical_volatility_20, 60, 0, 2), 60)), zscore_cs(sma(ts_regression(log_returns, funding_rate_zscore, 60, 0, 2), 60))), zscore_cs(sma(taker_buy_ratio, 120))), -1.5), 1.5)"
    ),

    # ── ts_zscore of close_position + ts_skewness + taker ──
    # Proven components from #21, #28 but combined differently
    (
        "close_pos_skew_taker",
        "df_min(df_max(add(add(zscore_cs(sma(ts_zscore(close_position_in_range, 60), 60)), zscore_cs(sma(ts_skewness(log_returns, 60), 60))), zscore_cs(sma(taker_buy_ratio, 120))), -1.5), 1.5)"
    ),

    # ── Corr of returns vs taker_buy_ratio ──
    # Hypothesis: when taker buying aligns WITH returns (positive corr), trend is real.
    # When diverging (corr negative), regime is uncertain.
    (
        "returns_taker_corr_signal",
        "df_min(df_max(add(zscore_cs(sma(ts_corr(log_returns, taker_buy_ratio, 60), 60)), zscore_cs(sma(taker_buy_ratio, 120))), -1.5), 1.5)"
    ),

    # ── ADV acceleration + overnight gap + taker ──
    # Proven components from #19, #20 but with different mix
    (
        "adv_gap_taker_v2",
        "df_min(df_max(add(add(zscore_cs(sma(ts_delta(s_log_1p(adv20), 30), 120)), zscore_cs(sma(overnight_gap, 120))), zscore_cs(sma(taker_buy_ratio, 120))), -1.5), 1.5)"
    ),
]


print(f"\n{'='*90}", flush=True)
print(f"  MANUAL ALPHA DESIGN ROUND 2 -- {len(ALPHAS)} hypotheses", flush=True)
print(f"  Gates: SR>={MIN_SR}, TO<{MAX_TO}, Fitness>{MIN_FITNESS}", flush=True)
print(f"{'='*90}\n", flush=True)

found = 0
for name, expr in ALPHAS:
    t0 = time.time()
    print(f"  [{name}]", flush=True)

    r = ea.eval_single(expr, split="train", fees_bps=0)
    if not r["success"]:
        print(f"  FAILED stage-1\n", flush=True)
        continue

    sr, to, fit = r["sharpe"], r["turnover"], r["fitness"]
    elapsed = time.time() - t0
    print(f"  S1: SR={sr:+.3f} TO={to:.4f} Fit={fit:.2f}", flush=True)

    if sr < MIN_SR or to >= MAX_TO or fit < MIN_FITNESS:
        reasons = []
        if sr < MIN_SR: reasons.append(f"SR={sr:+.2f}")
        if to >= MAX_TO: reasons.append(f"TO={to:.3f}")
        if fit < MIN_FITNESS: reasons.append(f"Fit={fit:.1f}")
        print(f"  FAIL: {', '.join(reasons)}\n", flush=True)
        ea.log_trial(conn, expr, sr, saved=False)
        continue

    full = ea.eval_full(expr, conn)
    if not full["success"]:
        print(f"  Full FAILED: {full['error']}\n", flush=True)
        ea.log_trial(conn, expr, sr, saved=False)
        continue

    h1, h2 = full["stability_h1"], full["stability_h2"]
    roll_sr = full.get("rolling_sr_std", 999)
    skew = full.get("pnl_skew", 0)
    full_fit = full.get("is_fitness", 0)
    print(f"  Full: H1={h1:+.2f} H2={h2:+.2f} skew={skew:+.2f} rollSR={roll_sr:.4f} fit={full_fit:.2f}", flush=True)

    both_pos = h1 > 0 and h2 > 0
    min_sub = min(h1, h2)
    passes = both_pos and min_sub >= ea.MIN_SUB_SHARPE and roll_sr <= ea.MAX_ROLLING_SR_STD and skew >= ea.MIN_PNL_SKEW

    ea.log_trial(conn, expr, sr, saved=False)

    if passes:
        saved = ea.save_alpha(conn, expr, f"manual_{name}", full)
        if saved:
            found += 1
            print(f"  >>> SAVED! (session total: {found})\n", flush=True)
        else:
            print(f"  >>> Rejected by diversity check\n", flush=True)
    else:
        reasons = []
        if not both_pos: reasons.append("sub-periods")
        if min_sub < ea.MIN_SUB_SHARPE: reasons.append(f"min_sub={min_sub:.2f}")
        if roll_sr > ea.MAX_ROLLING_SR_STD: reasons.append(f"rollSR={roll_sr:.4f}")
        if skew < ea.MIN_PNL_SKEW: reasons.append(f"skew={skew:.2f}")
        print(f"  FULL FAIL: {', '.join(reasons)}\n", flush=True)

print(f"\n{'='*90}", flush=True)
print(f"  DONE. Saved {found}/{len(ALPHAS)} alphas.", flush=True)
n_total = conn.execute("SELECT COUNT(*) FROM alphas WHERE archived=0 AND interval='4h' AND universe='BINANCE_TOP50'").fetchone()[0]
print(f"  Total active 4h alphas: {n_total}", flush=True)
conn.close()
