"""Quick neutralization sweep on TOP3000 universe."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_alpha_ib

eval_alpha_ib.UNIVERSE = "TOP3000"

TEST_ALPHAS = [
    ("close_near_low",      "rank((low - close) / (high - low + 0.001))"),
    ("upper_wick_ratio",    "rank((high - close) / (high - low + 0.001))"),
    ("vwap_deviation",      "rank(-(close - vwap) / close)"),
    ("reversal_1d",         "rank(-ts_delta(close, 1))"),
    ("reversal_3d",         "rank(-ts_delta(close, 3))"),
    ("low_vol_premium",     "rank(-ts_std_dev(close, 10))"),
]

NEUTRALIZATIONS = ["sector", "industry", "subindustry", "market"]

print("NEUTRALIZATION SWEEP on TOP3000, delay=0, fee-free")
print("=" * 100)

for neut in NEUTRALIZATIONS:
    eval_alpha_ib.NEUTRALIZE = neut
    eval_alpha_ib._DATA_CACHE.clear()
    print(f"\n--- {neut.upper()} ---")
    for name, expr in TEST_ALPHAS:
        r = eval_alpha_ib.eval_single(expr, split="train")
        if r["success"]:
            sr = r["sharpe"]
            fit = r["fitness"]
            to = r["turnover"]
            print(f"  {name:25s} | SR={sr:+.3f} Fit={fit:.3f} TO={to:.4f}")
        else:
            print(f"  {name:25s} | FAILED: {r['error'][:50]}")
