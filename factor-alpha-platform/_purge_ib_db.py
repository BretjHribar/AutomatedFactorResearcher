import sqlite3

DB_PATH = "data/ib_alphas.db"
MIN_IS_SHARPE = 3.0

conn = sqlite3.connect(DB_PATH)

# Find alphas below the gate
failing = conn.execute("""
    SELECT a.id, a.expression, ROUND(e.sharpe_is, 3) as sr
    FROM alphas a LEFT JOIN evaluations e ON e.alpha_id = a.id
    WHERE a.archived = 0 AND (e.sharpe_is IS NULL OR e.sharpe_is < ?)
""", (MIN_IS_SHARPE,)).fetchall()

if not failing:
    print("No alphas below SR gate — nothing to remove.")
else:
    print(f"Removing {len(failing)} alphas below SR={MIN_IS_SHARPE}:")
    for aid, expr, sr in failing:
        print(f"  #{aid} SR={sr}  {expr}")

    ids = [r[0] for r in failing]
    placeholders = ",".join("?" * len(ids))

    # Delete evaluations first (FK)
    n_evals = conn.execute(
        f"DELETE FROM evaluations WHERE alpha_id IN ({placeholders})", ids
    ).rowcount
    print(f"\nDeleted {n_evals} evaluation rows")

    # Then delete alphas
    n_alphas = conn.execute(
        f"DELETE FROM alphas WHERE id IN ({placeholders})", ids
    ).rowcount
    print(f"Deleted {n_alphas} alpha rows")

conn.commit()

# Show remaining
remaining = conn.execute("""
    SELECT a.id, ROUND(e.sharpe_is,3), ROUND(e.ic_mean,4), ROUND(e.turnover,3), a.expression
    FROM alphas a LEFT JOIN evaluations e ON e.alpha_id = a.id
    WHERE a.archived = 0
    ORDER BY e.sharpe_is DESC
""").fetchall()
conn.close()

print(f"\nRemaining DB ({len(remaining)} alphas):")
print(f"  {'ID':<4} {'SR_IS':>7} {'IC':>7} {'TO':>6}  Expression")
print(f"  {'-'*80}")
for r in remaining:
    aid, sr, ic, to_, expr = r
    print(f"  #{aid:<3} {sr:>7.3f} {ic:>7.4f} {to_:>6.3f}  {expr[:60]}")
