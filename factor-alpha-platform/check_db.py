import sqlite3
c = sqlite3.connect("data/alphas.db")

# Schema
for row in c.execute("SELECT sql FROM sqlite_master WHERE type='table'").fetchall():
    print(row[0])
    print()

# Count
print(f"Total alphas: {c.execute('SELECT COUNT(*) FROM alphas').fetchone()[0]}")
print(f"Active alphas: {c.execute('SELECT COUNT(*) FROM alphas WHERE archived=0').fetchone()[0]}")

# Sample
print("\nSample active alphas:")
for row in c.execute("SELECT id, expression, category FROM alphas WHERE archived=0 LIMIT 10").fetchall():
    print(f"  #{row[0]}: [{row[2]}] {row[1][:80]}")

# All active
print("\nAll active alphas:")
for row in c.execute("SELECT id, expression, category FROM alphas WHERE archived=0").fetchall():
    print(f"  #{row[0]}: [{row[2]}] {row[1][:80]}")
