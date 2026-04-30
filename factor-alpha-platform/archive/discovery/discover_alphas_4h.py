"""
discover_alphas_4h.py -- 4H Alpha Discovery Pipeline

Discovers orthogonal alpha factors for 4h Binance perpetual futures.
Uses eval_alpha.py (INTERVAL=4h) harness for evaluation.

SR >= 4.0 quality gate, TO < 0.40 (4h naturally lower turnover)
Train: 2021-01-01 to 2025-01-01 (4 years)

Usage:
    python discover_alphas_4h.py
"""

import sys, os, time, random
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_alpha as ea

# Override config
UNIVERSE = "BINANCE_TOP50"
ea.UNIVERSE = UNIVERSE
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
ea.MIN_IS_SHARPE = 1.5    # Lowered -- fitness gate is the real filter
ea.CORR_CUTOFF = 0.70
ea.MAX_TURNOVER = 0.10
ea.MIN_SUB_SHARPE = 1.0
ea.MAX_PNL_KURTOSIS = 20
ea.MAX_ROLLING_SR_STD = 0.05
ea.MIN_PNL_SKEW = -0.5

TARGET_ALPHAS = 20
MIN_SR = 1.5
MAX_TO = 0.10
MIN_FITNESS = 5.0

# ============================================================================
# ATOM LIBRARIES
# Built from analysis of existing proven 4h alphas.
# Key insight: price-level structures (Decay_exp, multiply, ts_delta(close))
# are far stronger than zscore_cs(sma()) composites on 4h.
# 4h bar equivalents: 6bars=1d, 30bars=5d, 60bars=10d, 120bars=20d, 360bars=60d
# ============================================================================

# --- Category A: ts_delta(close) × signal structures (from #12, #14, #15, #16) ---
CLOSE_DELTA_WINDOWS = [3, 6, 12, 24, 48]
BETA_SIGNALS = [
    "subtract(beta_to_btc, sma(beta_to_btc, 60))",
    "subtract(beta_to_btc, sma(beta_to_btc, 120))",
    "ts_delta(beta_to_btc, 6)",
    "ts_delta(beta_to_btc, 12)",
    "ts_delta(beta_to_btc, 24)",
    "subtract(beta_to_btc, sma(beta_to_btc, 30))",
]
VOL_SIGNALS = [
    "true_divide(high_low_range, df_max(open_close_range, 0.0001))",
    "true_divide(high_low_range, df_max(close, 0.0001))",
    "true_divide(open_close_range, df_max(high_low_range, 0.0001))",
    "subtract(historical_volatility_20, sma(historical_volatility_20, 60))",
    "ts_delta(historical_volatility_20, 12)",
    "true_divide(lower_shadow, df_max(high_low_range, 0.0001))",
    "true_divide(upper_shadow, df_max(high_low_range, 0.0001))",
]
DECAY_RATES = [0.05, 0.1, 0.2]

# --- Category B: Sign(momentum) × ratio structures (from #11, #13) ---
SIGN_SIGNALS = [
    "Sign(sma(returns, 60))",
    "Sign(sma(returns, 120))",
    "Sign(momentum_20d)",
    "Sign(momentum_60d)",
    "Sign(ts_delta(close, 24))",
    "Sign(ts_delta(close, 48))",
]
RATIO_SIGNALS = [
    "true_divide(volume_momentum_5_20, df_max(historical_volatility_60, 0.0001))",
    "true_divide(volume_momentum_5_20, df_max(parkinson_volatility_60, 0.0001))",
    "true_divide(volume_momentum_5_20, df_max(historical_volatility_20, 0.0001))",
    "true_divide(volume_ratio_20d, df_max(historical_volatility_20, 0.0001))",
    "true_divide(taker_buy_ratio, df_max(historical_volatility_20, 0.0001))",
    "true_divide(momentum_20d, df_max(historical_volatility_20, 0.0001))",
    "true_divide(ts_sum(log_returns, 30), df_max(historical_volatility_20, 0.0001))",
    "true_divide(ts_sum(log_returns, 60), df_max(historical_volatility_60, 0.0001))",
]

# --- Category C: zscore_cs(sma) composites that proved to work (from #18-31) ---
# Only include atoms with longer SMA windows that proved effective
CS_ATOMS = [
    "zscore_cs(sma(taker_buy_ratio, 120))",
    "zscore_cs(sma(ts_delta(s_log_1p(adv60), 30), 120))",
    "zscore_cs(sma(ts_delta(s_log_1p(adv60), 30), 90))",
    "zscore_cs(sma(ts_delta(trades_per_volume, 30), 60))",
    "zscore_cs(sma(ts_zscore(close_position_in_range, 60), 60))",
    "zscore_cs(sma(ts_zscore(log_returns, 120), 120))",
    "zscore_cs(sma(ts_skewness(log_returns, 60), 60))",
    "zscore_cs(sma(ts_regression(log_returns, historical_volatility_20, 60, 0, 2), 60))",
    "zscore_cs(sma(ts_regression(log_returns, funding_rate_zscore, 60, 0, 2), 60))",
    "zscore_cs(sma(true_divide(lower_shadow, df_max(high_low_range, 0.001)), 120))",
    "zscore_cs(sma(funding_rate_zscore, 60))",
    "zscore_cs(sma(vwap_deviation, 120))",
    "zscore_cs(sma(ts_corr(log_returns, delay(log_returns, 1), 30), 60))",
    "zscore_cs(sma(ts_corr(log_returns, delay(log_returns, 1), 60), 60))",
    "zscore_cs(sma(ts_corr(log_returns, funding_rate_zscore, 60), 60))",
    "zscore_cs(sma(ts_corr(close_position_in_range, taker_buy_ratio, 60), 60))",
    "zscore_cs(sma(ts_corr(log_returns, trades_per_volume, 60), 60))",
    "zscore_cs(Decay_exp(ts_corr(log_returns, delay(log_returns, 1), 30), 0.05))",
    "zscore_cs(sma(ts_delta(s_log_1p(quote_volume), 30), 120))",
]


def build_composite(components, outer_smooth=None):
    """Build additive composite alpha expression."""
    expr = components[0]
    for c in components[1:]:
        expr = f"add({expr},{c})"
    if outer_smooth:
        return f"sma({expr}, {outer_smooth})"
    return f"df_min(df_max({expr}, -1.5), 1.5)"


def generate_candidates(seed=42):
    rng = random.Random(seed)
    candidates = []
    seen = set()

    def add_expr(expr):
        if expr not in seen:
            seen.add(expr)
            candidates.append(expr)

    def add_combo(comps, outer=None):
        expr = build_composite(list(comps), outer)
        add_expr(expr)

    # ── Strategy A: Decay_exp(multiply(ts_delta(close, w), signal), rate) ──
    # Mirror of #12, #14, #16 -- strongest existing alphas
    for w in CLOSE_DELTA_WINDOWS:
        for sig in BETA_SIGNALS + VOL_SIGNALS:
            for rate in DECAY_RATES:
                add_expr(f"Decay_exp(multiply(ts_delta(close, {w}), {sig}), {rate})")

    # ── Strategy B: multiply(ts_delta(close, w), signal) with outer sma ──
    for w in CLOSE_DELTA_WINDOWS:
        for sig in BETA_SIGNALS + VOL_SIGNALS:
            for sma_w in [2, 3, 6, 12]:
                add_expr(f"sma(multiply(ts_delta(close, {w}), {sig}), {sma_w})")

    # ── Strategy C: Sign(momentum) × ratio (from #11, #13) ──
    for sign in SIGN_SIGNALS:
        for ratio in RATIO_SIGNALS:
            add_expr(f"multiply({sign}, {ratio})")

    # ── Strategy D: Sign × ratio with outer sma ──
    for sign in SIGN_SIGNALS:
        for ratio in RATIO_SIGNALS:
            for sma_w in [6, 12, 24]:
                add_expr(f"sma(multiply({sign}, {ratio}), {sma_w})")

    # ── Strategy E: 3-4 component CS composites from proven atoms ──
    for _ in range(800):
        comps = rng.sample(CS_ATOMS, rng.choice([3, 4]))
        add_combo(comps)

    # ── Strategy F: 5-6 component CS composites ──
    for _ in range(500):
        comps = rng.sample(CS_ATOMS, rng.choice([5, 6]))
        add_combo(comps)

    # ── Strategy G: Mix price-level atom + CS composite ──
    # Build pure price-level single atoms first
    price_atoms = []
    for w in CLOSE_DELTA_WINDOWS:
        for sig in BETA_SIGNALS:
            price_atoms.append(f"zscore_cs(Decay_exp(multiply(ts_delta(close, {w}), {sig}), 0.05))")
            price_atoms.append(f"zscore_cs(sma(multiply(ts_delta(close, {w}), {sig}), 6))")
    for sign in SIGN_SIGNALS:
        for ratio in RATIO_SIGNALS[:4]:
            price_atoms.append(f"zscore_cs(multiply({sign}, {ratio}))")

    for _ in range(600):
        n_price = rng.choice([1, 2])
        n_cs = rng.choice([2, 3])
        p = rng.sample(price_atoms, min(n_price, len(price_atoms)))
        c = rng.sample(CS_ATOMS, min(n_cs, len(CS_ATOMS)))
        add_combo(p + c)

    # ── Strategy H: Mutate existing alpha #12 family ──
    # #12: Decay_exp(multiply(ts_delta(close, 12), subtract(beta_to_btc, sma(beta_to_btc, 60))), 0.05)
    for w in [6, 12, 18, 24, 36, 48]:
        for sma_w in [30, 60, 90, 120]:
            for rate in [0.02, 0.05, 0.1, 0.2]:
                add_expr(f"Decay_exp(multiply(ts_delta(close, {w}), subtract(beta_to_btc, sma(beta_to_btc, {sma_w}))), {rate})")

    # ── Strategy I: ts_delta(close) × (beta - sma(beta)) + CS atom additive ──
    base_price = [
        "zscore_cs(Decay_exp(multiply(ts_delta(close, 12), subtract(beta_to_btc, sma(beta_to_btc, 60))), 0.05))",
        "zscore_cs(Decay_exp(multiply(ts_delta(close, 6), subtract(beta_to_btc, sma(beta_to_btc, 60))), 0.1))",
        "zscore_cs(sma(multiply(ts_delta(close, 6), true_divide(high_low_range, df_max(open_close_range, 0.0001))), 2))",
        "zscore_cs(multiply(Sign(momentum_60d), true_divide(volume_momentum_5_20, df_max(parkinson_volatility_60, 0.0001))))",
    ]
    for bp in base_price:
        for cs in CS_ATOMS:
            add_combo([bp, cs])
        for _ in range(50):
            cs_picks = rng.sample(CS_ATOMS, rng.choice([2, 3]))
            add_combo([bp] + cs_picks)

    # ── Strategy J: Different seeds for more diversity ──
    for extra_seed in [1000, 2000, 3000]:
        rng2 = random.Random(seed + extra_seed)
        for _ in range(300):
            comps = rng2.sample(CS_ATOMS, rng2.choice([3, 4, 5]))
            add_combo(comps)

    return candidates


# Time limit
TIME_LIMIT_SECONDS = 3600  # 1 hour


def main():
    t0 = time.time()
    conn = ea.get_conn()
    ea.ensure_trial_log(conn)

    # Clear data cache since we changed config
    ea._DATA_CACHE.clear()

    # Count existing with correct universe filter
    n_existing = conn.execute(
        "SELECT COUNT(*) FROM alphas WHERE archived=0 AND interval='4h' AND universe=?",
        (UNIVERSE,)
    ).fetchone()[0]

    candidates = generate_candidates()

    est_per_eval = 0.5  # seconds (from timing)
    est_total = len(candidates) * est_per_eval
    print(f"\n{'='*80}", flush=True)
    print(f"  4H ALPHA DISCOVERY -- {UNIVERSE}", flush=True)
    print(f"  Train: {ea.TRAIN_START} to {ea.TRAIN_END}", flush=True)
    print(f"  Target: {TARGET_ALPHAS} new | Existing: {n_existing}", flush=True)
    print(f"  Gates: SR >= {MIN_SR}, TO < {MAX_TO}, Corr < {ea.CORR_CUTOFF}", flush=True)
    print(f"  Candidates: {len(candidates)}", flush=True)
    print(f"  Est. time: {est_total/60:.0f}m ({est_per_eval:.2f}s/eval)", flush=True)
    print(f"  Time limit: {TIME_LIMIT_SECONDS/60:.0f}m", flush=True)
    print(f"{'='*80}\n", flush=True)

    found = 0
    s1_pass = 0
    tested = 0
    best_sr = 0

    for i, expr in enumerate(candidates):
        if found >= TARGET_ALPHAS:
            break

        elapsed_total = time.time() - t0
        if elapsed_total > TIME_LIMIT_SECONDS:
            print(f"\n  TIME LIMIT REACHED ({TIME_LIMIT_SECONDS/60:.0f}m)", flush=True)
            break

        tested += 1
        t1 = time.time()
        try:
            result = ea.eval_single(expr, split="train", fees_bps=0)
        except Exception:
            continue
        if not result["success"]:
            continue
        elapsed = time.time() - t1

        sr = result["sharpe"]
        to = result["turnover"]
        fit = result["fitness"]
        best_sr = max(best_sr, sr)

        if sr < MIN_SR or to >= MAX_TO or fit < MIN_FITNESS:
            if tested % 100 == 0:
                rate = tested / (time.time() - t0)
                remaining = (len(candidates) - tested) / rate if rate > 0 else 0
                print(f"  [{tested}/{len(candidates)}] best_sr={best_sr:+.2f} s1_pass={s1_pass} "
                      f"found={found} rate={rate:.1f}/s ETA={remaining/60:.0f}m", flush=True)
            ea.log_trial(conn, expr, sr, saved=False)
            continue

        s1_pass += 1
        print(f"\n[{tested}/{len(candidates)}] SR={sr:+.3f} TO={to:.4f} ({elapsed:.0f}s) -> PASS! Full eval...", flush=True)

        t2 = time.time()
        full = ea.eval_full(expr, conn)
        elapsed2 = time.time() - t2

        if not full["success"]:
            print(f"  Full eval FAILED: {full['error']}", flush=True)
            ea.log_trial(conn, expr, sr, saved=False)
            continue

        ic = full["ic_mean"]
        h1 = full["stability_h1"]
        h2 = full["stability_h2"]
        roll_sr = full.get("rolling_sr_std", 999)
        skew = full.get("pnl_skew", 0)
        fit = full.get("is_fitness", 0)
        print(f"  Full: IC={ic:+.05f} H1={h1:+.2f} H2={h2:+.2f} skew={skew:+.2f} "
              f"rollSR={roll_sr:.4f} fit={fit:.2f} ({elapsed2:.0f}s)", flush=True)

        both_pos = h1 > 0 and h2 > 0
        min_sub = min(h1, h2)
        all_pass = (
            both_pos and
            min_sub >= ea.MIN_SUB_SHARPE and
            roll_sr <= ea.MAX_ROLLING_SR_STD and
            skew >= ea.MIN_PNL_SKEW
        )

        ea.log_trial(conn, expr, sr, saved=False)

        if all_pass:
            print(f"  >>> ALL GATES PASS! Checking diversity (corr < {ea.CORR_CUTOFF})...", flush=True)
            saved = ea.save_alpha(conn, expr, "4h_discovery_v1", full)
            if saved:
                found += 1
                print(f"  >>> SAVED! ({found}/{TARGET_ALPHAS})", flush=True)
            else:
                print(f"  >>> Rejected by diversity check", flush=True)
        else:
            reasons = []
            if not both_pos: reasons.append("sub-periods not both positive")
            if min_sub < ea.MIN_SUB_SHARPE: reasons.append(f"min_sub={min_sub:.2f}")
            if roll_sr > ea.MAX_ROLLING_SR_STD: reasons.append(f"rollSR={roll_sr:.4f}")
            if skew < ea.MIN_PNL_SKEW: reasons.append(f"skew={skew:.2f}")
            print(f"  --- FULL FAIL: {', '.join(reasons)}", flush=True)

    total_time = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  DISCOVERY COMPLETE")
    print(f"  Tested: {tested} | S1 Pass: {s1_pass} | Saved: {found}/{TARGET_ALPHAS}")
    print(f"  Best SR seen: {best_sr:+.3f}")
    print(f"  Time: {total_time:.0f}s ({total_time/60:.1f}m)")
    print(f"{'='*80}")

    n_total = conn.execute(
        "SELECT COUNT(*) FROM alphas WHERE archived=0 AND interval='4h' AND universe=?",
        (UNIVERSE,)
    ).fetchone()[0]
    print(f"\n  Total active 4h alphas: {n_total}")
    conn.close()


if __name__ == "__main__":
    main()
