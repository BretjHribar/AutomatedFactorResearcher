"""
Manual alpha design for 4h BINANCE_TOP50.
Each alpha is a deliberate hypothesis, not a search.

Gates: SR >= 1.5, TO < 0.10, fitness > 5.0, sub-period stable
Train: 2021-01-01 to 2025-01-01
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
    # ── Family 1: Beta-anomaly momentum ──
    # Hypothesis: When price rises AND beta drops (decoupling from BTC), it's idiosyncratic strength.
    # Use 20-bar (3.3d) delta - medium term signal, 20d beta SMA
    (
        "beta_anom_20d_close",
        "Decay_exp(multiply(ts_delta(close, 20), subtract(beta_to_btc, sma(beta_to_btc, 120))), 0.05)"
    ),
    # Same but short-term: 3-bar (12h) close delta × shorter beta window
    (
        "beta_anom_3bar",
        "Decay_exp(multiply(ts_delta(close, 3), subtract(beta_to_btc, sma(beta_to_btc, 30))), 0.05)"
    ),
    # Beta anomaly with vol-adjusted close delta
    (
        "beta_anom_voladj",
        "Decay_exp(multiply(true_divide(ts_delta(close, 12), df_max(historical_volatility_20, 0.0001)), subtract(beta_to_btc, sma(beta_to_btc, 60))), 0.05)"
    ),

    # ── Family 2: Volatility regime × sign-of-trend ──
    # Hypothesis: Go with momentum only when vol is LOW (trending regime).
    # When vol is high, momentum is noise. Use vol-inverse weighting.
    (
        "vol_regime_momentum",
        "multiply(Sign(momentum_20d), true_divide(ts_sum(log_returns, 30), df_max(historical_volatility_20, 0.0001)))"
    ),
    # Longer horizon version
    (
        "vol_regime_momentum_60d",
        "multiply(Sign(momentum_60d), true_divide(ts_sum(log_returns, 60), df_max(historical_volatility_60, 0.0001)))"
    ),
    # With taker buy confirmation: only buy when buyers are also dominant
    (
        "vol_regime_taker_confirm",
        "multiply(Sign(momentum_20d), true_divide(taker_buy_ratio, df_max(historical_volatility_20, 0.0001)))"
    ),

    # ── Family 3: Funding rate mean-reversion + volume confirmation ──
    # Hypothesis: Extreme negative funding → crowded shorts → likely squeeze.
    # Confirm with rising volume (buyers absorbing supply).
    (
        "funding_squeeze_vol",
        "df_min(df_max(add(zscore_cs(sma(funding_rate_zscore, 60)), zscore_cs(sma(ts_delta(s_log_1p(adv20), 30), 60))), -1.5), 1.5)"
    ),
    # Funding + taker buy: negative funding + aggressive buyers = strong long signal
    (
        "funding_taker_combo",
        "df_min(df_max(add(zscore_cs(sma(funding_rate_zscore, 60)), zscore_cs(sma(taker_buy_ratio, 120))), -1.5), 1.5)"
    ),

    # ── Family 4: Candle structure (lower shadow = accumulation) ──
    # Hypothesis: Long lower shadow means buyers stepped in at lows = demand.
    # Combined with close near high (bullish candle body) even stronger.
    (
        "lower_shadow_accumulation",
        "df_min(df_max(add(add(zscore_cs(sma(true_divide(lower_shadow, df_max(high_low_range, 0.001)), 120)), zscore_cs(sma(close_position_in_range, 60))), zscore_cs(sma(taker_buy_ratio, 120))), -1.5), 1.5)"
    ),
    # Lower shadow + funding (buyers at lows AND negative funding = squeeze setup)
    (
        "lower_shadow_funding",
        "df_min(df_max(add(add(zscore_cs(sma(true_divide(lower_shadow, df_max(high_low_range, 0.001)), 120)), zscore_cs(sma(funding_rate_zscore, 60))), zscore_cs(sma(ts_delta(trades_per_volume, 30), 60))), -1.5), 1.5)"
    ),

    # ── Family 5: Log-return autocorrelation (4h reversal) ──
    # Hypothesis: 4h returns show mild negative autocorrelation (short-term mean-reversion).
    # Bet against the last bar's direction.
    (
        "log_return_reversal",
        "df_min(df_max(add(add(zscore_cs(Decay_exp(ts_corr(log_returns, delay(log_returns, 1), 30), 0.05)), zscore_cs(sma(ts_zscore(log_returns, 120), 120))), zscore_cs(sma(taker_buy_ratio, 120))), -1.5), 1.5)"
    ),

    # ── Family 6: ADV momentum (liquidity flow) ──
    # Hypothesis: Rising dollar volume (new money flowing in) is bullish signal.
    # Use 30-bar delta of log(ADV) as a flow measure.
    (
        "adv_flow_taker",
        "df_min(df_max(add(add(zscore_cs(sma(ts_delta(s_log_1p(adv60), 30), 120)), zscore_cs(sma(taker_buy_ratio, 120))), zscore_cs(sma(overnight_gap, 120))), -1.5), 1.5)"
    ),
    # ADV flow + lower shadow + overnight gap
    (
        "adv_shadow_gap",
        "df_min(df_max(add(add(zscore_cs(sma(ts_delta(s_log_1p(adv60), 30), 90)), zscore_cs(sma(true_divide(lower_shadow, df_max(high_low_range, 0.001)), 120))), zscore_cs(sma(overnight_gap, 120))), -1.5), 1.5)"
    ),

    # ── Family 7: Intraday structure (trades per volume = informed trading) ──
    # Hypothesis: High trades-per-volume = many small trades = retail noise. 
    # Low tpv = large block trades = informed. Bet with informed flow.
    (
        "informed_flow",
        "df_min(df_max(add(add(zscore_cs(sma(ts_delta(trades_per_volume, 30), 60)), zscore_cs(sma(ts_regression(log_returns, historical_volatility_20, 60, 0, 2), 60))), zscore_cs(sma(taker_buy_ratio, 120))), -1.5), 1.5)"
    ),
]


print(f"\n{'='*90}", flush=True)
print(f"  MANUAL ALPHA DESIGN -- {len(ALPHAS)} hypotheses", flush=True)
print(f"  Gates: SR>={MIN_SR}, TO<{MAX_TO}, Fitness>{MIN_FITNESS}", flush=True)
print(f"  Train: {ea.TRAIN_START} to {ea.TRAIN_END}", flush=True)
print(f"{'='*90}\n", flush=True)

found = 0
for name, expr in ALPHAS:
    t0 = time.time()
    print(f"  [{name}]", flush=True)
    print(f"  {expr[:80]}...", flush=True) if len(expr) > 80 else print(f"  {expr}", flush=True)

    r = ea.eval_single(expr, split="train", fees_bps=0)
    if not r["success"]:
        print(f"  FAILED stage-1\n", flush=True)
        continue

    sr, to, fit = r["sharpe"], r["turnover"], r["fitness"]
    elapsed = time.time() - t0
    print(f"  S1: SR={sr:+.3f} TO={to:.4f} Fit={fit:.2f} ({elapsed:.1f}s)", flush=True)

    if sr < MIN_SR or to >= MAX_TO or fit < MIN_FITNESS:
        reasons = []
        if sr < MIN_SR: reasons.append(f"SR={sr:+.2f}<{MIN_SR}")
        if to >= MAX_TO: reasons.append(f"TO={to:.3f}>={MAX_TO}")
        if fit < MIN_FITNESS: reasons.append(f"Fit={fit:.1f}<{MIN_FITNESS}")
        print(f"  FAIL: {', '.join(reasons)}\n", flush=True)
        ea.log_trial(conn, expr, sr, saved=False)
        continue

    # Full eval
    full = ea.eval_full(expr, conn)
    if not full["success"]:
        print(f"  Full eval FAILED: {full['error']}\n", flush=True)
        ea.log_trial(conn, expr, sr, saved=False)
        continue

    ic = full["ic_mean"]
    h1, h2 = full["stability_h1"], full["stability_h2"]
    roll_sr = full.get("rolling_sr_std", 999)
    skew = full.get("pnl_skew", 0)
    full_fit = full.get("is_fitness", 0)
    print(f"  Full: IC={ic:+.5f} H1={h1:+.2f} H2={h2:+.2f} skew={skew:+.2f} rollSR={roll_sr:.4f} fit={full_fit:.2f}", flush=True)

    both_pos = h1 > 0 and h2 > 0
    min_sub = min(h1, h2)
    passes = both_pos and min_sub >= ea.MIN_SUB_SHARPE and roll_sr <= ea.MAX_ROLLING_SR_STD and skew >= ea.MIN_PNL_SKEW

    ea.log_trial(conn, expr, sr, saved=False)

    if passes:
        print(f"  ALL GATES PASS -- saving...", flush=True)
        saved = ea.save_alpha(conn, expr, f"manual_{name}", full)
        if saved:
            found += 1
            print(f"  >>> SAVED as new alpha! (total saved this session: {found})\n", flush=True)
        else:
            print(f"  >>> Rejected by diversity check\n", flush=True)
    else:
        reasons = []
        if not both_pos: reasons.append("sub-periods not both pos")
        if min_sub < ea.MIN_SUB_SHARPE: reasons.append(f"min_sub={min_sub:.2f}")
        if roll_sr > ea.MAX_ROLLING_SR_STD: reasons.append(f"rollSR={roll_sr:.4f}")
        if skew < ea.MIN_PNL_SKEW: reasons.append(f"skew={skew:.2f}")
        print(f"  FULL FAIL: {', '.join(reasons)}\n", flush=True)

print(f"\n{'='*90}", flush=True)
print(f"  DONE. Saved {found}/{len(ALPHAS)} alphas.", flush=True)
n_total = conn.execute("SELECT COUNT(*) FROM alphas WHERE archived=0 AND interval='4h' AND universe='BINANCE_TOP50'").fetchone()[0]
print(f"  Total active 4h alphas: {n_total}", flush=True)
conn.close()
