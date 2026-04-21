import sqlite3
conn = sqlite3.connect('data/ib_alphas.db')
rows = conn.execute("""
    SELECT trial_id, expression, is_sharpe, saved, created_at 
    FROM trial_log 
    ORDER BY created_at DESC 
    LIMIT 100
""").fetchall()
print("Recent trials -- SR > 2.5, unsaved (rejected on corr or gate):")
for r in rows:
    tid, expr, sr, saved, ts = r
    if sr and sr > 2.5 and not saved:
        print(f"  SR={sr:+.3f}  {expr[:75]}")
conn.close()
