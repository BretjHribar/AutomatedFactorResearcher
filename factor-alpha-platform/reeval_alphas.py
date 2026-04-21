"""
reeval_alphas.py — Re-evaluate all saved IB alphas against the current universe.
Prints a comparison table (old stored metrics vs new evaluation).
Does NOT modify or delete any alphas from the DB.
"""
import sqlite3
import time
import os, sys
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_alpha_ib as ib

DB_PATH = "data/ib_alphas.db"

def reeval_all():
    conn = sqlite3.connect(DB_PATH)

    # Load all non-archived alphas with their stored evaluation metrics
    rows = conn.execute("""
        SELECT a.id, a.name, a.expression,
               COALESCE(e.sharpe_is, 0) as old_sr,
               COALESCE(e.turnover, 0) as old_to,
               COALESCE(e.sharpe_h1, 0) as old_h1,
               COALESCE(e.sharpe_h2, 0) as old_h2
        FROM alphas a
        LEFT JOIN evaluations e ON e.alpha_id = a.id
        WHERE a.archived = 0
        ORDER BY COALESCE(e.sharpe_is, 0) DESC
    """).fetchall()

    print(f"\n{'='*90}")
    print(f"RE-EVALUATING {len(rows)} ALPHAS AGAINST CURRENT UNIVERSE")
    print(f"{'='*90}")
    print(f"{'ID':>4}  {'Name':<40}  {'OLD SR':>7}  {'NEW SR':>7}  {'OLD H1':>7}  {'NEW H1':>7}  {'NEW H2':>7}  {'NEW TO':>7}  Status")
    print(f"{'-'*100}")

    results = []
    for alpha_id, name, expr, old_sr, old_to, old_h1, old_h2 in rows:
        t0 = time.time()
        try:
            result = ib.eval_full(expr, conn)
        except Exception as e:
            print(f"  #{alpha_id:>2} {name:<40}  ERROR: {e}")
            continue

        elapsed = time.time() - t0

        if not result["success"]:
            print(f"  #{alpha_id:>2} {name:<40}  FAILED: {result['error']}")
            continue

        new_sr = result["is_sharpe"]
        new_h1 = result["stability_h1"]
        new_h2 = result["stability_h2"]
        new_to = result["turnover"]
        new_fit = result["is_fitness"]
        new_kurt = result["pnl_kurtosis"]
        new_skew = result["pnl_skew"]

        # Status — use gate thresholds
        passes_gate = (
            new_sr >= ib.MIN_IS_SHARPE and
            new_h1 > ib.MIN_SUB_SHARPE and
            new_h2 > ib.MIN_SUB_SHARPE and
            new_kurt <= ib.MAX_PNL_KURTOSIS and
            new_skew >= ib.MIN_PNL_SKEW
        )
        status = "PASS" if passes_gate else "FAIL"

        delta = new_sr - old_sr
        delta_str = f"({delta:+.2f})"

        print(f"  #{alpha_id:>2} {name:<35} {old_sr:>+7.3f}  {new_sr:>+7.3f}{delta_str:<8} {old_h1:>+6.2f}  {new_h1:>+6.2f}  {new_h2:>+6.2f}  {new_to:>6.3f}  {status}  [{elapsed:.0f}s]")

        results.append({
            "id": alpha_id, "name": name, "expr": expr,
            "old_sr": old_sr, "new_sr": new_sr,
            "new_h1": new_h1, "new_h2": new_h2,
            "new_to": new_to, "new_fit": new_fit,
            "passes": passes_gate
        })

    conn.close()

    # Summary
    print(f"\n{'='*90}")
    passing = [r for r in results if r["passes"]]
    failing = [r for r in results if not r["passes"]]
    print(f"SUMMARY: {len(passing)}/{len(results)} alphas pass gates under new universe")
    if failing:
        print(f"\nFailing alphas (may need review):")
        for r in failing:
            print(f"  #{r['id']} {r['name']}: SR={r['new_sr']:+.3f} H1={r['new_h1']:+.2f} H2={r['new_h2']:+.2f}")

    if passing:
        print(f"\nPassing alphas sorted by new SR:")
        for r in sorted(passing, key=lambda x: -x["new_sr"]):
            print(f"  #{r['id']:>2} {r['name']:<40}  SR={r['new_sr']:+.3f}  Fit={r['new_fit']:.3f}  TO={r['new_to']:.3f}")

    print(f"{'='*90}")

if __name__ == "__main__":
    reeval_all()
