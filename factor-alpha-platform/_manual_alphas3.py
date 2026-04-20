"""
Manual alpha design - Round 3.
Strategy: Avoid ALL price-delta and beta signals. 
Target: pure microstructure, funding, and cross-sectional flow signals
that have ZERO overlap with the existing beta-momentum family.
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
    # ── Taker buy persistence ──
    # Hypothesis: Persistent (slow-SMA) taker buying = institutional accumulation.
    # 120-bar SMA captures multi-week trends in buyer aggression.
    # Keep very simple — the simpler the better at 4h.
    (
        "taker_persistence_solo",
        "sma(taker_buy_ratio, 120)"
    ),

    # ── Overnight gap accumulation ──
    # Hypothesis: sustained overnight gaps (crypto = 24h, so this is open vs prev close)
    # measure persistent off-hours demand.
    (
        "overnight_gap_sma",
        "sma(overnight_gap, 120)"
    ),

    # ── Taker × overnight gap ──
    # Hypothesis: aggressive buying AND overnight gaps = both dimensions aligned.
    (
        "taker_overnight_product",
        "sma(multiply(taker_buy_ratio, overnight_gap), 60)"
    ),

    # ── VWAP deviation persistence ──
    # Hypothesis: Trading persistently above VWAP = demand constantly absorbing supply.
    # VWAP deviation SMA = medium-term demand imbalance.
    (
        "vwap_deviation_sma",
        "sma(vwap_deviation, 120)"
    ),

    # ── Taker-to-volume ratio × ADV growth ──
    # Hypothesis: Rising ADV (more $$ volume) + buyers dominating = capital inflow.
    # Use both slow (120-bar) taker and ADV delta.
    (
        "taker_adv_inflow",
        "sma(multiply(taker_buy_ratio, ts_delta(s_log_1p(adv60), 30)), 60)"
    ),

    # ── trades_per_volume (mean-reversion signal) ──
    # Hypothesis: many small trades per $ volume = retail noise, mean-reverts.
    # Fewer trades per volume = block trades = informed, trends.
    # Bet against high trades_per_volume (fade retail),
    # confirmed by slow taker signal.
    (
        "informed_vs_retail",
        "sma(negative(multiply(ts_delta(trades_per_volume, 30), taker_buy_ratio)), 60)"
    ),

    # ── CS-rank of taker + overnight + adv (three-way flow) ──
    # All three proven CS atoms from #19, #20 but use a clean 3-way combo
    (
        "three_way_flow",
        "df_min(df_max(add(add(zscore_cs(sma(taker_buy_ratio, 120)), zscore_cs(sma(overnight_gap, 120))), zscore_cs(sma(ts_delta(s_log_1p(adv60), 30), 90))), -1.5), 1.5)"
    ),

    # ── Regression of returns on funding only (funding-adjusted return) ──
    # Hypothesis: the portion of return explained by funding = crowded trade.
    # The residual (unexplained by funding) = true demand signal.
    (
        "return_orthogonal_to_funding",
        "df_min(df_max(add(zscore_cs(sma(ts_regression(log_returns, funding_rate_zscore, 60, 0, 2), 60)), zscore_cs(sma(taker_buy_ratio, 120))), -1.5), 1.5)"
    ),

    # ── Autocorrelation of taker_buy_ratio ──
    # Hypothesis: when taker buying is autocorrelated (persistent), it's real demand.
    # When random = noise. Use ts_corr of taker with its lag.
    (
        "taker_autocorr",
        "df_min(df_max(add(zscore_cs(sma(ts_corr(taker_buy_ratio, delay(taker_buy_ratio, 6), 30), 60)), zscore_cs(sma(taker_buy_ratio, 120))), -1.5), 1.5)"
    ),

    # ── Funding-adjusted taker (taker net of funding) ──
    # Hypothesis: high taker buying AND negative funding = real demand, not just
    # shorts being squeezed. Subtract out funding impact.
    (
        "taker_net_of_funding",
        "sma(subtract(taker_buy_ratio, funding_rate_zscore), 120)"
    ),

    # ── Close position in range × volume ratio (candle + volume confirmation) ──
    # Hypothesis: bullish candle (close near high) AND above-average dollar volume = conviction.
    (
        "candle_vol_confirm",
        "sma(multiply(close_position_in_range, volume_ratio_20d), 60)"
    ),

    # ── ts_kurtosis of taker_buy_ratio ──
    # Hypothesis: fat-tailed distribution of taker buying = extreme buyer aggression
    # in some bars = potential for trend continuation.
    (
        "taker_kurtosis_signal",
        "sma(ts_kurtosis(taker_buy_ratio, 60), 60)"
    ),
]


print(f"\n{'='*90}", flush=True)
print(f"  MANUAL ALPHA DESIGN ROUND 3 -- {len(ALPHAS)} hypotheses (no price-delta/beta)", flush=True)
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
