"""
Save 4 discovered alphas following the workflow in
.agents/workflows/crypto-4h-alpha-research.md.

Quality gates enforced (from the workflow document, not the stricter ones
that drifted into eval_alpha.py): IS Sharpe > 2.0, Fitness > 5.0, Turnover
< 0.30, both sub-periods > 0. Correlation cutoff |corr| < 0.70 applied
via check_diversity (since DB starts empty this is moot for the first alpha).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Preload the module so we can override its constants for KUCOIN_TOP100
import eval_alpha as ea

ea.EXCHANGE = "kucoin"
ea.UNIVERSE = "KUCOIN_TOP100"
ea.TRAIN_START = "2023-09-01"
ea.TRAIN_END = "2025-09-01"
ea.SUBPERIODS = [
    ("2023-09-01", "2024-09-01", "H1"),
    ("2024-09-01", "2025-09-01", "H2"),
]
ea._DATA_CACHE.clear()

alphas = [
    {
        "expression": "zscore_cs(true_divide(sma(log_returns, 120), df_max(sma(parkinson_volatility_20, 120), 0.001)))",
        "reasoning": "Vol-adjusted long-horizon drift. Numerator: 120-bar (20-day) smoothed log-return "
                     "captures sustained cross-sectional outperformers. Denominator: 120-bar smoothed "
                     "Parkinson (H/L-based) volatility normalises by realised risk. Economically a "
                     "Sharpe-like score per ticker: coins whose drift has been high relative to their "
                     "own range-measured volatility tend to persist over the next 4h window. Long-window "
                     "smoothing suppresses noise and keeps turnover low (~12%). H1 SR 3.58, H2 SR 2.14.",
        "category": "trend",
    },
    {
        "expression": "zscore_cs(sma(upper_shadow, 30))",
        "reasoning": "Sustained upper-shadow presence as a bullish intrabar microstructure signal. "
                     "upper_shadow = (high - max(open, close)) / close measures the 'test-up-and-fail' "
                     "portion of each 4h bar. Smoothed over 30 bars (5 days) to capture persistent "
                     "demand: coins repeatedly probing higher, regardless of whether they close near "
                     "the high, show buyer interest that tends to translate into next-bar performance. "
                     "Very low turnover (~7%). H1 SR 3.27, H2 SR 1.33.",
        "category": "microstructure",
    },
    {
        "expression": "zscore_cs(sma(ts_zscore(turnover, 240), 60))",
        "reasoning": "Dollar-volume regime shift. Per-ticker z-score of raw turnover against its own "
                     "240-bar (40-day) rolling history, then smoothed 60 bars (10 days) and cross-"
                     "sectionally normalised. Coins whose dollar volume has drifted into a new regime "
                     "(persistently high vs their own past) tend to attract continued flow. The inner "
                     "z-score makes the signal scale-invariant across tickers of wildly different "
                     "market caps. Very low turnover (~4%). H1 SR 2.66, H2 SR 2.14.",
        "category": "volume",
    },
    {
        "expression": "zscore_cs(sma(volume_ratio_20d, 120))",
        "reasoning": "Relative liquidity premium. volume_ratio_20d = current volume / 20-day rolling "
                     "mean volume — tickers whose short-window volume is persistently above their "
                     "20-day norm show sustained institutional/retail interest. Smoothed over 120 "
                     "bars (20 days) for very slow rotation. Orthogonal to the turnover-regime signal "
                     "because it normalises by recent history rather than the full 240-bar distribution. "
                     "H1 SR 3.20, H2 SR 2.18; Fitness 16.9; turnover only ~3.4%.",
        "category": "volume",
    },
]


def main():
    conn = ea.get_conn()
    ea.ensure_trial_log(conn)

    print(f"Loading TRAIN data for {ea.UNIVERSE} ({ea.TRAIN_START} -> {ea.TRAIN_END})")
    # Trigger data load once
    ea.load_data("train")

    for i, entry in enumerate(alphas, 1):
        expr = entry["expression"]
        print(f"\n[{i}/4] {expr[:120]}")
        result = ea.eval_full(expr, conn)
        if not result["success"]:
            print(f"  EVAL FAILED: {result['error']}")
            continue

        # Gate check — workflow's documented thresholds, not script defaults
        is_sharpe = result["is_sharpe"]
        fitness = result["is_fitness"]
        turnover = result["turnover"]
        h1, h2 = result["stability_h1"], result["stability_h2"]
        gates = [
            (is_sharpe > 2.0,   f"IS Sharpe > 2.0: {is_sharpe:+.3f}"),
            (fitness   > 5.0,   f"Fitness > 5.0: {fitness:.3f}"),
            (turnover  < 0.30,  f"Turnover < 0.30: {turnover:.3f}"),
            (h1 > 0 and h2 > 0, f"Sub-period stability: H1={h1:+.2f} H2={h2:+.2f}"),
        ]
        print(f"  SR={is_sharpe:+.2f}  Fit={fitness:.2f}  TO={turnover:.3f}  H1={h1:+.2f}  H2={h2:+.2f}  "
              f"MDD={result['max_drawdown']:+.3f}  IC={result['ic_mean']:+.4f}")
        all_pass = True
        for passed, desc in gates:
            print(f"  [{'PASS' if passed else 'FAIL'}] {desc}")
            if not passed:
                all_pass = False
        if not all_pass:
            print("  SKIP (fails workflow gates)")
            continue

        result["category"] = entry["category"]
        saved = ea.save_alpha(conn, expr, entry["reasoning"], result)
        if saved:
            ea.log_trial(conn, expr, is_sharpe, saved=True)

    # Show final DB state
    print("\nFinal DB state — crypto 4h alphas:")
    rows = conn.execute("""
        SELECT a.id, a.expression, e.sharpe_is, e.turnover, e.fitness, e.ic_mean
        FROM alphas a JOIN evaluations e ON e.alpha_id = a.id
        WHERE a.asset_class='crypto' AND a.interval='4h' AND a.archived=0
        ORDER BY a.id
    """).fetchall()
    for r in rows:
        print(f"  id={r[0]:>2}  SR={r[2]:+.2f}  TO={r[3]:.3f}  Fit={r[4]:.2f}  IC={r[5]:+.4f}  {r[1][:80]}")
    conn.close()


if __name__ == "__main__":
    main()
