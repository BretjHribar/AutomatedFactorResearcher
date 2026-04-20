"""Fast 5th alpha finder - 3-component swaps from #92 + pre-screening correlation."""
import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import eval_alpha_5m as ea

ea.UNIVERSE = "BINANCE_TOP50"
ucfg = ea.UNIVERSE_CONFIG["BINANCE_TOP50"]
ea.MAX_WEIGHT = ucfg["max_weight"]
ea.MIN_IS_SHARPE = ucfg["min_is_sharpe"]

rng = random.Random(123)

alpha92 = [
    "zscore_cs(sma(ts_sum(returns, 12), 36))",
    "zscore_cs(negative(sma(vwap_deviation, 36)))",
    "zscore_cs(sma(taker_buy_ratio, 36))",
    "zscore_cs(sma(volume_momentum_1, 36))",
    "zscore_cs(sma(close_position_in_range, 36))",
    "zscore_cs(sma(open_close_range, 36))",
    "zscore_cs(sma(overnight_gap, 72))",
]

alpha99 = [
    "zscore_cs(negative(sma(ts_delta(vwap_deviation, 24), 96)))",
    "zscore_cs(sma(ts_delta(close_position_in_range, 24), 96))",
    "zscore_cs(sma(ts_delta(taker_buy_ratio, 24), 96))",
    "zscore_cs(sma(ts_delta(overnight_gap, 24), 96))",
    "zscore_cs(negative(sma(ts_corr(historical_volatility_60, volume, 72), 96)))",
    "zscore_cs(sma(momentum_20d, 96))",
    "zscore_cs(negative(sma(trades_count, 96)))",
]

replacements = [
    "zscore_cs(negative(sma(beta_to_btc, 72)))",
    "zscore_cs(sma(beta_to_btc, 72))",
    "zscore_cs(negative(sma(historical_volatility_20, 72)))",
    "zscore_cs(sma(historical_volatility_20, 72))",
    "zscore_cs(negative(sma(ts_skewness(returns, 144), 72)))",
    "zscore_cs(negative(sma(ts_kurtosis(returns, 144), 72)))",
    "zscore_cs(sma(ts_skewness(returns, 288), 72))",
    "zscore_cs(sma(upper_shadow, 72))",
    "zscore_cs(negative(sma(upper_shadow, 72)))",
    "zscore_cs(sma(lower_shadow, 72))",
    "zscore_cs(negative(sma(lower_shadow, 72)))",
    "zscore_cs(sma(volume_ratio_20d, 72))",
    "zscore_cs(negative(sma(volume_ratio_20d, 72)))",
    "zscore_cs(sma(trades_per_volume, 72))",
    "zscore_cs(negative(sma(trades_per_volume, 72)))",
    "zscore_cs(sma(div(momentum_20d, historical_volatility_20), 96))",
    "zscore_cs(negative(sma(div(momentum_20d, historical_volatility_20), 96)))",
    "zscore_cs(negative(sma(div(volume, adv20), 72)))",
    "zscore_cs(sma(div(adv20, adv60), 72))",
    "zscore_cs(negative(sma(high_low_range, 72)))",
    "zscore_cs(sma(high_low_range, 72))",
    "zscore_cs(sma(momentum_60d, 72))",
    "zscore_cs(negative(sma(momentum_60d, 72)))",
    "zscore_cs(sma(ts_sum(log_returns, 36), 72))",
    "zscore_cs(negative(sma(ts_sum(log_returns, 36), 72)))",
    "zscore_cs(negative(sma(ts_corr(returns, volume, 72), 36)))",
    "zscore_cs(sma(ts_corr(taker_buy_ratio, returns, 72), 96))",
    "zscore_cs(sma(div(dollars_traded, adv60), 72))",
    "zscore_cs(negative(sma(parkinson_volatility_20, 72)))",
    "zscore_cs(sma(div(historical_volatility_10, historical_volatility_60), 36))",
    "zscore_cs(sma(volume_momentum_5_20, 72))",
    "zscore_cs(sma(ts_delta(beta_to_btc, 24), 72))",
]

def build(components, outer):
    expr = components[0]
    for c in components[1:]:
        expr = f"add({expr},{c})"
    return f"sma({expr}, {outer})"

candidates = []
seen = set()

# 3-swap from #92
for _ in range(200):
    comps = list(alpha92)
    indices = rng.sample(range(7), 3)
    for idx in indices:
        comps[idx] = rng.choice(replacements)
    outer = rng.choice([96, 120, 144])
    key = (tuple(sorted(comps)), outer)
    if key not in seen:
        seen.add(key)
        candidates.append(build(comps, outer))

# 3-swap from #99
for _ in range(200):
    comps = list(alpha99)
    indices = rng.sample(range(7), 3)
    for idx in indices:
        comps[idx] = rng.choice(replacements)
    outer = rng.choice([96, 120, 144])
    key = (tuple(sorted(comps)), outer)
    if key not in seen:
        seen.add(key)
        candidates.append(build(comps, outer))

print(f"Generated {len(candidates)} candidates")

conn = ea.get_conn()
ea.ensure_tables(conn)

found = False
tested = 0
s1_pass = 0
for i, expr in enumerate(candidates):
    if found:
        break
    tested += 1
    result = ea.eval_single(expr, split="train", fees_bps=0)
    sr = result["sharpe"]
    to = result["turnover"]
    
    if sr >= 2.5 and to < 0.05:
        s1_pass += 1
        print(f"\n[{tested}] SR={sr:+.3f} TO={to:.4f} -> PASS! Full eval...", flush=True)
        full = ea.eval_full(expr, conn)
        if full["success"]:
            h1 = full["stability_h1"]
            h2 = full["stability_h2"]
            skew = full.get("pnl_skew", 0)
            roll_sr = full.get("rolling_sr_std", 999)
            both_pos = h1 > 0 and h2 > 0
            min_sub = min(h1, h2)
            if (both_pos and min_sub >= ea.MIN_SUB_SHARPE and 
                roll_sr <= ea.MAX_ROLLING_SR_STD and skew >= ea.MIN_PNL_SKEW):
                saved = ea.save_alpha(conn, expr, "curated_3swap", full)
                if saved:
                    print(f"  >>> SAVED! IC={full['ic_mean']:+.05f} H1={h1:+.2f} H2={h2:+.2f}", flush=True)
                    found = True
                else:
                    print(f"  Diversity FAIL", flush=True)
            else:
                reasons = []
                if not both_pos: reasons.append("sub-periods")
                if min_sub < ea.MIN_SUB_SHARPE: reasons.append(f"min_sub={min_sub:.2f}")
                if roll_sr > ea.MAX_ROLLING_SR_STD: reasons.append(f"rollSR={roll_sr:.4f}")
                if skew < ea.MIN_PNL_SKEW: reasons.append(f"skew={skew:.2f}")
                print(f"  Full FAIL: {', '.join(reasons)}", flush=True)
    elif tested % 50 == 0:
        print(f"  [{tested}/{len(candidates)}] best so far, s1_pass={s1_pass}", flush=True)

print(f"\nTested: {tested} | S1 Pass: {s1_pass} | Found: {found}")
conn.close()
