"""
H6-H8 deeper experiments based on H1-H5 findings.

H6 — Combined best config: TOP30, min_hist=365d, rebal=60d, exclude memes
     vs all individual best-of-each axis. Identify true Pareto-optimal config.

H7 — ADV-weighted vs equal-weighted within universe (Russell methodology).
     Top-30-by-ADV with positions weighted by liquidity rank instead of equal.

H8 — Multi-alpha test: run all 18 DB alphas on best universe. Confirm findings
     generalize beyond the v10 benchmark.
"""
from __future__ import annotations
import sys, json, time, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse functions from main experiments
sys.path.insert(0, str(ROOT/"experiments"))
from universe_experiments import (
    build_universe_topn_rebal, eval_signal, signal_to_portfolio,
    split_metrics, load_all,
    BENCHMARK_EXPR, BARS_PER_DAY, BARS_PER_YEAR,
    TRAIN_START, TRAIN_END, VAL_END,
    OUT, log,
)

DB_PATH = ROOT/"data/alphas.db"


def main():
    matrices = load_all()
    close = matrices["close"]
    adv = matrices["adv20"]
    rets = matrices["returns"]

    log("Evaluating benchmark alpha...")
    sig = eval_signal(BENCHMARK_EXPR, matrices)

    # ─────────────────────────────────────────────────────────────────
    # H6 — Combined-best config sweep
    # ─────────────────────────────────────────────────────────────────
    log("=== H6: combined best configs (TOP30 x min_hist=365 x rebal=60 baseline) ===")
    rows = []
    MEMES = ["DOGE", "SHIB", "PEPE", "BONK", "WIF", "FLOKI", "MEME",
             "MOG", "MOODENG", "TRUMP", "FART"]
    STABLES = ["USDC", "DAI", "TUSD", "BUSD", "FDUSD"]

    configs = [
        # (name, top_n, min_hist_days, rebal_days, exclusions)
        ("baseline_TOP30_20d_365d_no_excl", 30, 365, 20, []),
        ("rebal_60d_only",                  30, 365, 60, []),
        ("no_memes_only",                   30, 365, 20, MEMES),
        ("no_stables_only",                 30, 365, 20, STABLES),
        ("60d_no_memes",                    30, 365, 60, MEMES),
        ("60d_no_stables",                  30, 365, 60, STABLES),
        ("60d_no_memes_no_stables",         30, 365, 60, MEMES+STABLES),
        ("60d_TOP20_no_memes",              20, 365, 60, MEMES),
        ("60d_TOP50_no_memes",              50, 365, 60, MEMES),
        ("60d_TOP30_180d",                  30, 180, 60, MEMES),
    ]
    for name, top_n, min_hist, rebal_days, excl in configs:
        rb = rebal_days * BARS_PER_DAY
        uni = build_universe_topn_rebal(adv, close, top_n=top_n, rebal_bars=rb,
                                         min_history_days=min_hist, exclusions=excl)
        if (uni.sum(axis=1) > 0).sum() < 100:
            print(f"  {name}: skipped (insufficient eligible)"); continue
        w = signal_to_portfolio(sig, uni)
        m = split_metrics(w, rets, fee_bps=3.0)
        m.update({"name": name, "top_n": top_n, "min_hist": min_hist,
                  "rebal_days": rebal_days, "n_excl": len(excl),
                  "avg_active": float(uni.sum(axis=1).mean()),
                  "unique_tickers": int(uni.any(axis=0).sum())})
        rows.append(m)
        print(f"  {name:35s}  TR={m['TRAIN_SR_n']:+.2f} VAL={m['VAL_SR_n']:+.2f} "
              f"TEST={m['TEST_SR_n']:+.2f} VT={m['VT_SR_n']:+.2f} DD={m['VT_dd_n']*100:+.0f}% "
              f"TO={m['to_per_bar']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT/"universe_h6_combined.csv", index=False, float_format="%.4f")
    log(f"  saved {OUT/'universe_h6_combined.csv'}")

    # ─────────────────────────────────────────────────────────────────
    # H8 — Multi-alpha sanity check (do findings generalize?)
    # ─────────────────────────────────────────────────────────────────
    log("=== H8: multi-alpha test on best (TOP30, 60d rebal, no memes, 365d hist) ===")
    rb = 60 * BARS_PER_DAY
    best_uni = build_universe_topn_rebal(adv, close, top_n=30, rebal_bars=rb,
                                         min_history_days=365, exclusions=MEMES)
    baseline_uni = build_universe_topn_rebal(adv, close, top_n=30, rebal_bars=20*BARS_PER_DAY,
                                              min_history_days=365, exclusions=[])

    con = sqlite3.connect(str(DB_PATH))
    alphas = con.execute("""SELECT id, expression, name FROM alphas
                             WHERE archived=0 AND asset_class='crypto' AND interval='4h'
                             ORDER BY id""").fetchall()
    print(f"  found {len(alphas)} alphas in DB")
    rows = []
    for aid, expr, name in alphas:
        try:
            sig_a = eval_signal(expr, matrices)
        except Exception as e:
            print(f"  a{aid}: FAIL {type(e).__name__}: {str(e)[:60]}")
            continue
        # Baseline universe
        w_base = signal_to_portfolio(sig_a, baseline_uni)
        m_base = split_metrics(w_base, rets, fee_bps=3.0)
        # Best universe
        w_best = signal_to_portfolio(sig_a, best_uni)
        m_best = split_metrics(w_best, rets, fee_bps=3.0)

        rows.append({"alpha_id": aid,
                     "base_TRAIN": m_base["TRAIN_SR_n"],
                     "base_VT":    m_base["VT_SR_n"],
                     "best_TRAIN": m_best["TRAIN_SR_n"],
                     "best_VT":    m_best["VT_SR_n"],
                     "delta_TRAIN": m_best["TRAIN_SR_n"] - m_base["TRAIN_SR_n"],
                     "delta_VT":    m_best["VT_SR_n"] - m_base["VT_SR_n"],
                     "name": (name or "")[:50]})
        print(f"  a{aid:>3}  base TR/VT={m_base['TRAIN_SR_n']:+.2f}/{m_base['VT_SR_n']:+.2f}  "
              f"best={m_best['TRAIN_SR_n']:+.2f}/{m_best['VT_SR_n']:+.2f}  "
              f"d_VT={m_best['VT_SR_n']-m_base['VT_SR_n']:+.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT/"universe_h8_multi_alpha.csv", index=False, float_format="%.3f")
    print(f"\n  Summary across {len(df)} alphas:")
    print(f"  base_TRAIN  mean {df['base_TRAIN'].mean():+.2f}  median {df['base_TRAIN'].median():+.2f}")
    print(f"  base_VT     mean {df['base_VT'].mean():+.2f}  median {df['base_VT'].median():+.2f}")
    print(f"  best_TRAIN  mean {df['best_TRAIN'].mean():+.2f}  median {df['best_TRAIN'].median():+.2f}")
    print(f"  best_VT     mean {df['best_VT'].mean():+.2f}  median {df['best_VT'].median():+.2f}")
    print(f"  delta_TRAIN mean {df['delta_TRAIN'].mean():+.2f} (n improved: {(df['delta_TRAIN']>0).sum()}/{len(df)})")
    print(f"  delta_VT    mean {df['delta_VT'].mean():+.2f} (n improved: {(df['delta_VT']>0).sum()}/{len(df)})")
    log(f"  saved {OUT/'universe_h8_multi_alpha.csv'}")

    log("ALL DONE")


if __name__ == "__main__":
    main()
