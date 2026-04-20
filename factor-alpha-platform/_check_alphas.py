import sqlite3
conn = sqlite3.connect('data/alphas.db')
cur = conn.cursor()
cur.execute("SELECT id, expression, interval, universe, archived FROM alphas WHERE interval='4h' AND archived=0 ORDER BY id")
rows = cur.fetchall()
print(f"4h non-archived alphas: {len(rows)}")
for r in rows:
    print(f"  id={r[0]}, interval={r[2]}, universe={r[3]}, expr={r[1][:100]}")
conn.close()
