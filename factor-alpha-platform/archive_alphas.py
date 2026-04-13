"""Mark all existing alphas in data/alphas_5m.db as archived=1 (inactive)."""
import sqlite3
from pathlib import Path

DB_PATH = "data/alphas_5m.db"
if not Path(DB_PATH).exists():
    print(f"DB not found: {DB_PATH}")
else:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT id, universe, expression, archived FROM alphas")
    rows = cur.fetchall()
    print(f"Found {len(rows)} alphas:")
    for r in rows:
        status = "ALREADY INACTIVE" if r[3] else "ACTIVE -> will archive"
        print(f"  #{r[0]} [{r[1]}] archived={r[3]} | {status} | {r[2][:60]}")
    
    updated = cur.execute("UPDATE alphas SET archived=1 WHERE archived=0").rowcount
    con.commit()
    print(f"\nArchived {updated} alphas (marked inactive).")
    
    # Verify
    active = cur.execute("SELECT COUNT(*) FROM alphas WHERE archived=0").fetchone()[0]
    inactive = cur.execute("SELECT COUNT(*) FROM alphas WHERE archived=1").fetchone()[0]
    print(f"Active: {active} | Inactive: {inactive}")
    con.close()
