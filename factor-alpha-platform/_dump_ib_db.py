import sqlite3
conn = sqlite3.connect('data/ib_alphas.db')
rows = conn.execute("""
    SELECT a.id, a.name, a.expression, 
           ROUND(e.sharpe_is,3) as sr_is, ROUND(e.ic_mean,4) as ic,
           ROUND(e.turnover,3) as to_, ROUND(e.fitness,3) as fitness,
           a.created_at
    FROM alphas a LEFT JOIN evaluations e ON e.alpha_id = a.id
    WHERE a.archived = 0
    ORDER BY e.sharpe_is DESC
""").fetchall()
print(f"IB Alpha DB -- {len(rows)} alphas")
print()
print(f"  {'ID':<4} {'SR_IS':>7} {'IC':>7} {'TO':>6} {'Fitness':>7}  Expression")
print(f"  {'-'*95}")
for r in rows:
    aid, name, expr, sr, ic, to_, fit, ts = r
    print(f"  #{aid:<3} {sr:>7.3f} {ic:>7.4f} {to_:>6.3f} {fit:>7.3f}  {expr[:65]}")
conn.close()
