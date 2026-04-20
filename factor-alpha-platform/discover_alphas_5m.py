"""
discover_alphas_5m.py — Batch 4 discovery (aggressive diversity)
Target: 5 more alphas on top of existing 14
Strategy: Heavy 5-6 component swaps + full-random + 5-component composites
"""

import sys, os, time, random
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_alpha_5m as ea

UNIVERSE = "BINANCE_TOP50"
ea.UNIVERSE = UNIVERSE
ucfg = ea.UNIVERSE_CONFIG[UNIVERSE]
ea.MAX_WEIGHT = ucfg["max_weight"]
ea.MIN_IS_SHARPE = ucfg["min_is_sharpe"]

TARGET_ALPHAS = 100
MIN_SR = 2.5
MAX_TO = 0.05

# ============================================================================
# ATOM LIBRARIES
# ============================================================================

# Template A: #92-family base atoms (level signals, short SMA)
BASE_A = [
    "zscore_cs(sma(ts_sum(returns, 12), 36))",
    "zscore_cs(negative(sma(vwap_deviation, 36)))",
    "zscore_cs(sma(taker_buy_ratio, 36))",
    "zscore_cs(sma(volume_momentum_1, 36))",
    "zscore_cs(sma(close_position_in_range, 36))",
    "zscore_cs(sma(open_close_range, 36))",
    "zscore_cs(sma(overnight_gap, 72))",
]

# Template B: #99-family base atoms (ts_delta signals, long SMA)
BASE_B = [
    "zscore_cs(negative(sma(ts_delta(vwap_deviation, 24), 96)))",
    "zscore_cs(sma(ts_delta(close_position_in_range, 24), 96))",
    "zscore_cs(sma(ts_delta(taker_buy_ratio, 24), 96))",
    "zscore_cs(sma(ts_delta(overnight_gap, 24), 96))",
    "zscore_cs(negative(sma(ts_corr(historical_volatility_60, volume, 72), 96)))",
    "zscore_cs(sma(momentum_20d, 96))",
    "zscore_cs(negative(sma(trades_count, 96)))",
]

# Expanded replacement library — maximum diversity
REPLACEMENTS = [
    # Beta (all variants)
    "zscore_cs(negative(sma(beta_to_btc, 72)))",
    "zscore_cs(sma(beta_to_btc, 72))",
    "zscore_cs(sma(ts_delta(beta_to_btc, 24), 72))",
    "zscore_cs(negative(sma(ts_delta(beta_to_btc, 12), 36)))",
    "zscore_cs(sma(ts_delta(beta_to_btc, 48), 96))",
    "zscore_cs(negative(sma(beta_to_btc, 36)))",
    # Volatility (all horizons)
    "zscore_cs(negative(sma(historical_volatility_20, 72)))",
    "zscore_cs(sma(historical_volatility_20, 72))",
    "zscore_cs(negative(sma(historical_volatility_10, 36)))",
    "zscore_cs(sma(div(historical_volatility_10, historical_volatility_60), 36))",
    "zscore_cs(negative(sma(parkinson_volatility_20, 72)))",
    "zscore_cs(sma(parkinson_volatility_20, 72))",
    "zscore_cs(negative(sma(historical_volatility_120, 72)))",
    "zscore_cs(sma(ts_delta(historical_volatility_20, 12), 36))",
    "zscore_cs(sma(historical_volatility_60, 96))",
    "zscore_cs(negative(sma(historical_volatility_60, 96)))",
    "zscore_cs(sma(div(parkinson_volatility_20, historical_volatility_20), 72))",
    "zscore_cs(negative(sma(ts_delta(parkinson_volatility_20, 24), 72)))",
    "zscore_cs(sma(ts_delta(historical_volatility_60, 24), 96))",
    # Higher moments (all windows)
    "zscore_cs(negative(sma(ts_skewness(returns, 144), 72)))",
    "zscore_cs(sma(ts_skewness(returns, 144), 72))",
    "zscore_cs(negative(sma(ts_kurtosis(returns, 144), 72)))",
    "zscore_cs(sma(ts_kurtosis(returns, 144), 72))",
    "zscore_cs(sma(ts_skewness(returns, 288), 72))",
    "zscore_cs(negative(sma(ts_kurtosis(returns, 288), 72)))",
    "zscore_cs(sma(ts_skewness(returns, 72), 36))",
    "zscore_cs(negative(sma(ts_skewness(returns, 72), 36)))",
    "zscore_cs(sma(ts_kurtosis(returns, 72), 36))",
    "zscore_cs(negative(sma(ts_kurtosis(returns, 72), 36)))",
    # Shadows & candle structure
    "zscore_cs(sma(upper_shadow, 72))",
    "zscore_cs(negative(sma(upper_shadow, 72)))",
    "zscore_cs(sma(lower_shadow, 72))",
    "zscore_cs(negative(sma(lower_shadow, 72)))",
    "zscore_cs(sma(high_low_range, 72))",
    "zscore_cs(negative(sma(high_low_range, 72)))",
    "zscore_cs(sma(upper_shadow, 36))",
    "zscore_cs(negative(sma(lower_shadow, 36)))",
    "zscore_cs(sma(ts_delta(upper_shadow, 24), 72))",
    "zscore_cs(sma(ts_delta(lower_shadow, 24), 72))",
    # Volume dynamics
    "zscore_cs(sma(volume_ratio_20d, 72))",
    "zscore_cs(negative(sma(volume_ratio_20d, 72)))",
    "zscore_cs(sma(trades_per_volume, 72))",
    "zscore_cs(negative(sma(trades_per_volume, 72)))",
    "zscore_cs(negative(sma(div(volume, adv20), 72)))",
    "zscore_cs(sma(div(volume, adv20), 72))",
    "zscore_cs(sma(div(dollars_traded, adv60), 72))",
    "zscore_cs(negative(sma(div(dollars_traded, adv60), 72)))",
    "zscore_cs(sma(volume_momentum_5_20, 72))",
    "zscore_cs(negative(sma(volume_momentum_5_20, 72)))",
    "zscore_cs(sma(div(adv20, adv60), 72))",
    "zscore_cs(negative(sma(div(adv20, adv60), 72)))",
    "zscore_cs(sma(trades_count, 72))",
    "zscore_cs(negative(sma(trades_count, 72)))",
    "zscore_cs(sma(ts_delta(trades_count, 24), 72))",
    "zscore_cs(sma(ts_delta(volume_ratio_20d, 24), 72))",
    # Momentum variants
    "zscore_cs(sma(div(momentum_20d, historical_volatility_20), 96))",
    "zscore_cs(negative(sma(div(momentum_20d, historical_volatility_20), 96)))",
    "zscore_cs(sma(div(momentum_5d, historical_volatility_10), 36))",
    "zscore_cs(negative(sma(div(momentum_5d, historical_volatility_10), 36)))",
    "zscore_cs(sma(momentum_60d, 72))",
    "zscore_cs(negative(sma(momentum_60d, 72)))",
    "zscore_cs(sma(momentum_5d, 36))",
    "zscore_cs(negative(sma(momentum_5d, 36)))",
    "zscore_cs(sma(ts_delta(momentum_20d, 24), 72))",
    "zscore_cs(negative(sma(ts_delta(momentum_20d, 24), 72)))",
    # Correlations
    "zscore_cs(negative(sma(ts_corr(returns, volume, 72), 36)))",
    "zscore_cs(sma(ts_corr(returns, volume, 72), 36))",
    "zscore_cs(sma(ts_corr(taker_buy_ratio, returns, 72), 96))",
    "zscore_cs(negative(sma(ts_corr(returns, volume, 144), 72)))",
    "zscore_cs(sma(ts_corr(returns, volume, 144), 72))",
    "zscore_cs(negative(sma(ts_corr(taker_buy_ratio, volume, 72), 96)))",
    "zscore_cs(sma(ts_corr(close_position_in_range, returns, 72), 96))",
    # Log returns
    "zscore_cs(sma(ts_sum(log_returns, 36), 72))",
    "zscore_cs(negative(sma(ts_sum(log_returns, 36), 72)))",
    "zscore_cs(sma(ts_sum(log_returns, 72), 72))",
    "zscore_cs(negative(sma(ts_zscore(log_returns, 144), 72)))",
    "zscore_cs(sma(ts_zscore(log_returns, 144), 72))",
    "zscore_cs(sma(ts_sum(log_returns, 144), 96))",
    # ts_delta variants (different lookbacks / SMA windows)
    "zscore_cs(sma(ts_delta(vwap_deviation, 12), 36))",
    "zscore_cs(negative(sma(ts_delta(vwap_deviation, 12), 36)))",
    "zscore_cs(sma(ts_delta(close_position_in_range, 12), 36))",
    "zscore_cs(sma(ts_delta(taker_buy_ratio, 12), 36))",
    "zscore_cs(sma(ts_delta(overnight_gap, 12), 36))",
    "zscore_cs(negative(sma(ts_delta(overnight_gap, 12), 36)))",
    "zscore_cs(sma(ts_delta(open_close_range, 12), 36))",
    "zscore_cs(negative(sma(ts_delta(open_close_range, 12), 36)))",
    "zscore_cs(sma(ts_delta(vwap_deviation, 48), 96))",
    "zscore_cs(negative(sma(ts_delta(vwap_deviation, 48), 96)))",
    # Level base atoms with different SMA windows
    "zscore_cs(sma(vwap_deviation, 72))",
    "zscore_cs(negative(sma(vwap_deviation, 72)))",
    "zscore_cs(sma(taker_buy_ratio, 72))",
    "zscore_cs(sma(close_position_in_range, 72))",
    "zscore_cs(sma(open_close_range, 72))",
    "zscore_cs(sma(overnight_gap, 96))",
    "zscore_cs(sma(volume_momentum_1, 72))",
    "zscore_cs(sma(ts_sum(returns, 12), 72))",
    "zscore_cs(sma(ts_sum(returns, 24), 72))",
    "zscore_cs(sma(ts_sum(returns, 48), 96))",
    "zscore_cs(negative(sma(ts_sum(returns, 48), 96)))",
    # Interaction terms (products/ratios of different signal families)
    "zscore_cs(sma(div(taker_buy_ratio, volume_ratio_20d), 72))",
    "zscore_cs(sma(div(upper_shadow, high_low_range), 72))",
    "zscore_cs(sma(div(lower_shadow, high_low_range), 72))",
    "zscore_cs(sma(div(open_close_range, high_low_range), 72))",
]


def build_composite(components, outer_smooth):
    expr = components[0]
    for c in components[1:]:
        expr = f"add({expr},{c})"
    return f"sma({expr}, {outer_smooth})"


def generate_candidates(seed=9876):
    rng = random.Random(seed)
    candidates = []
    seen = set()

    def add(comps, outer):
        key = (tuple(sorted(comps)), outer)
        if key not in seen:
            seen.add(key)
            candidates.append(build_composite(comps, outer))

    # ── Strategy 1: 5-swap from #92 template (most components replaced) ──
    for _ in range(200):
        comps = list(BASE_A)
        for idx in rng.sample(range(7), 5):
            comps[idx] = rng.choice(REPLACEMENTS)
        add(comps, rng.choice([96, 120, 144, 168]))

    # ── Strategy 2: 6-swap from #92 template (nearly full replacement) ──
    for _ in range(200):
        comps = list(BASE_A)
        for idx in rng.sample(range(7), 6):
            comps[idx] = rng.choice(REPLACEMENTS)
        add(comps, rng.choice([96, 120, 144, 168]))

    # ── Strategy 3: 5-swap from #99 template ──
    for _ in range(200):
        comps = list(BASE_B)
        for idx in rng.sample(range(7), 5):
            comps[idx] = rng.choice(REPLACEMENTS)
        add(comps, rng.choice([96, 120, 144, 168]))

    # ── Strategy 4: 6-swap from #99 template ──
    for _ in range(200):
        comps = list(BASE_B)
        for idx in rng.sample(range(7), 6):
            comps[idx] = rng.choice(REPLACEMENTS)
        add(comps, rng.choice([96, 120, 144, 168]))

    # ── Strategy 5: Full random 7-component from replacements (massive diversity) ──
    for _ in range(400):
        comps = rng.sample(REPLACEMENTS, min(7, len(REPLACEMENTS)))
        add(comps, rng.choice([96, 120, 144, 168]))

    # ── Strategy 6: 5-component composites (smaller, different structure) ──
    for _ in range(200):
        comps = rng.sample(REPLACEMENTS, 5)
        add(comps, rng.choice([72, 96, 120]))

    # ── Strategy 7: Cross-template (2 from A + 5 replacements) ──
    for _ in range(150):
        a_picks = rng.sample(BASE_A, 2)
        r_picks = rng.sample(REPLACEMENTS, 5)
        add(a_picks + r_picks, rng.choice([96, 120, 144]))

    # ── Strategy 8: Cross-template (2 from B + 5 replacements) ──
    for _ in range(150):
        b_picks = rng.sample(BASE_B, 2)
        r_picks = rng.sample(REPLACEMENTS, 5)
        add(b_picks + r_picks, rng.choice([96, 120, 144]))

    return candidates


def main():
    t0 = time.time()
    conn = ea.get_conn()
    ea.ensure_tables(conn)

    n_existing = conn.execute(
        "SELECT COUNT(*) FROM alphas WHERE archived=0 AND universe=?", (UNIVERSE,)
    ).fetchone()[0]

    candidates = generate_candidates()

    print(f"\n{'='*80}")
    print(f"  ALPHA DISCOVERY BATCH 4 — {UNIVERSE}")
    print(f"  Target: {TARGET_ALPHAS} new | Existing: {n_existing}")
    print(f"  Gates: SR >= {MIN_SR}, TO < {MAX_TO}, Corr < {ea.CORR_CUTOFF}")
    print(f"  Candidates: {len(candidates)}")
    print(f"{'='*80}\n")

    found = 0
    s1_pass = 0
    tested = 0

    for i, expr in enumerate(candidates):
        if found >= TARGET_ALPHAS:
            break

        tested += 1
        t1 = time.time()
        try:
            result = ea.eval_single(expr, split="train", fees_bps=0)
        except Exception as e:
            continue
        if not result["success"]:
            continue
        elapsed = time.time() - t1

        sr = result["sharpe"]
        to = result["turnover"]

        if sr < MIN_SR or to >= MAX_TO:
            if tested % 100 == 0:
                print(f"  [{tested}/{len(candidates)}] s1_pass={s1_pass} found={found}", flush=True)
            ea.log_trial(conn, expr, sr, 0, saved=False)
            continue

        s1_pass += 1
        print(f"\n[{tested}/{len(candidates)}] SR={sr:+.3f} TO={to:.4f} ({elapsed:.0f}s) -> PASS! Full eval...", flush=True)

        t2 = time.time()
        full = ea.eval_full(expr, conn)
        elapsed2 = time.time() - t2

        if not full["success"]:
            print(f"  Full eval FAILED: {full['error']}", flush=True)
            ea.log_trial(conn, expr, sr, 0, saved=False)
            continue

        ic = full["ic_mean"]
        h1 = full["stability_h1"]
        h2 = full["stability_h2"]
        roll_sr = full.get("rolling_sr_std", 999)
        skew = full.get("pnl_skew", 0)
        print(f"  Full: IC={ic:+.05f} H1={h1:+.2f} H2={h2:+.2f} skew={skew:+.2f} rollSR={roll_sr:.4f} ({elapsed2:.0f}s)", flush=True)

        both_pos = h1 > 0 and h2 > 0
        min_sub = min(h1, h2)
        all_pass = (
            both_pos and
            min_sub >= ea.MIN_SUB_SHARPE and
            roll_sr <= ea.MAX_ROLLING_SR_STD and
            skew >= ea.MIN_PNL_SKEW
        )

        ea.log_trial(conn, expr, sr, ic, saved=False)

        if all_pass:
            print(f"  >>> ALL GATES PASS! Checking diversity (corr < {ea.CORR_CUTOFF})...", flush=True)
            saved = ea.save_alpha(conn, expr, "batch4_discovery", full)
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
    print(f"  Time: {total_time:.0f}s ({total_time/60:.1f}m)")
    print(f"{'='*80}")

    n_total = conn.execute(
        "SELECT COUNT(*) FROM alphas WHERE archived=0 AND universe=?", (UNIVERSE,)
    ).fetchone()[0]
    print(f"\n  Total active alphas: {n_total}")
    conn.close()


if __name__ == "__main__":
    main()
