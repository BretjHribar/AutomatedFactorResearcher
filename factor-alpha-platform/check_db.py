import sqlite3
conn = sqlite3.connect("data/alphas.db")
cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Tables:", [t[0] for t in tables])
for t in tables[:5]:
    cursor = conn.execute(f"PRAGMA table_info('{t[0]}')")
    cols = cursor.fetchall()
    print(f"  {t[0]} columns:", [c[1] for c in cols])
conn.close()
