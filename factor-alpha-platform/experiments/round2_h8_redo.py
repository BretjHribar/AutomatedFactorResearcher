"""
Round 2 — H8 redo with the ACTUAL H6 winner (TOP20+60d+memes-excluded).

The first H8 used TOP30+60d+no-memes (a +1.20 config). The H6 winner was
TOP20+60d+no-memes (a +2.50 config). Re-running.
"""
from __future__ import annotations
import sys, sqlite3
from pathlib import Path
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"experiments"))
from universe_experiments import (
    build_universe_topn_rebal, eval_signal, signal_to_portfolio,
    split_metrics, load_all,
    BARS_PER_DAY, OUT, log,
)

MEMES = ["DOGE", "SHIB", "PEPE", "BONK", "WIF", "FLOKI", "MEME",
         "MOG", "MOODENG", "TRUMP", "FART"]
DB_PATH = ROOT/"data/alphas.db"


def main():
    matrices = load_all()
    log("H8 REDO: testing actual H6 winner TOP20+60d+memes-excluded")
    base = build_universe_topn_rebal(matrices["adv20"], matrices["close"],
                                      top_n=30, rebal_bars=20*BARS_PER_DAY,
                                      min_history_days=365)
    best = build_universe_topn_rebal(matrices["adv20"], matrices["close"],
                                      top_n=20, rebal_bars=60*BARS_PER_DAY,
                                      min_history_days=365, exclusions=MEMES)
    print(f"  baseline avg_active={float(base.sum(axis=1).mean()):.1f}")
    print(f"  best     avg_active={float(best.sum(axis=1).mean()):.1f}")

    con = sqlite3.connect(str(DB_PATH))
    alphas = con.execute("""SELECT id, expression FROM alphas
                             WHERE archived=0 AND asset_class='crypto' AND interval='4h'
                             ORDER BY id""").fetchall()
    rows = []
    for aid, expr in alphas:
        try:
            sig_a = eval_signal(expr, matrices)
        except Exception as e:
            print(f"  a{aid}: FAIL {type(e).__name__}"); continue
        w_b = signal_to_portfolio(sig_a, base)
        m_b = split_metrics(w_b, matrices["returns"], fee_bps=3.0)
        w_t = signal_to_portfolio(sig_a, best)
        m_t = split_metrics(w_t, matrices["returns"], fee_bps=3.0)
        rows.append({"alpha_id": aid,
                     "base_TR": m_b["TRAIN_SR_n"], "base_VT": m_b["VT_SR_n"],
                     "best_TR": m_t["TRAIN_SR_n"], "best_VT": m_t["VT_SR_n"],
                     "d_VT":    m_t["VT_SR_n"] - m_b["VT_SR_n"]})
        print(f"  a{aid:>3}  base TR/VT={m_b['TRAIN_SR_n']:+.2f}/{m_b['VT_SR_n']:+.2f}  "
              f"best={m_t['TRAIN_SR_n']:+.2f}/{m_t['VT_SR_n']:+.2f}  "
              f"d_VT={m_t['VT_SR_n']-m_b['VT_SR_n']:+.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT/"round2_h8_redo.csv", index=False, float_format="%.3f")
    print(f"\nSummary: mean_d_VT={df['d_VT'].mean():+.2f}, n_improved={(df['d_VT']>0).sum()}/{len(df)}")
    print(f"  best_VT mean {df['best_VT'].mean():+.2f}  (vs base mean {df['base_VT'].mean():+.2f})")
    print(f"  best_TRAIN mean {df['best_TR'].mean():+.2f}  (vs base mean {df['base_TR'].mean():+.2f})")
    log("H8 REDO DONE")


if __name__ == "__main__":
    main()
