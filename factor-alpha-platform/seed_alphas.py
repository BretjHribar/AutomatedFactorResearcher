"""Seed the alphas.db with proven GP v2 alphas from CSV."""
import sqlite3
import csv

DB_PATH = "data/alphas.db"
CSV_PATH = "gp_v2_good_alphas.csv"

conn = sqlite3.connect(DB_PATH)

# Read CSV
with open(CSV_PATH, 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Found {len(rows)} alphas in CSV")

# Check which are already in DB
existing = set(r[0] for r in conn.execute("SELECT expression FROM alphas").fetchall())
print(f"Already in DB: {len(existing)}")

added = 0
for row in rows:
    expr = row['expression']
    if expr in existing:
        print(f"  SKIP (exists): {expr[:60]}")
        continue
    
    is_sharpe = float(row['is_sharpe'])
    oos_sharpe = float(row['oos_sharpe'])
    oos_turnover = float(row['oos_turnover'])
    oos_max_dd = float(row['oos_max_dd'])
    
    c = conn.cursor()
    c.execute("""INSERT INTO alphas (expression, name, category, asset_class, interval, source, notes)
                 VALUES (?, ?, ?, ?, ?, ?, ?)""",
              (expr,
               expr[:80],
               'gp_v2',
               'crypto',
               '4h',
               'gp_v2_research',
               f'IS_SR={is_sharpe:.3f} OOS_SR={oos_sharpe:.3f}'))
    alpha_id = c.lastrowid
    
    # Save evaluation metrics
    c.execute("""INSERT INTO evaluations (alpha_id, sharpe_is, sharpe_train, return_ann,
                 max_drawdown, turnover, fitness, ic_mean, ic_ir, psr,
                 train_start, train_end, n_bars, evaluated_at)
                 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
              (alpha_id,
               is_sharpe, is_sharpe,
               0,
               oos_max_dd, oos_turnover,
               oos_sharpe,
               0, 0, 0,
               '2022-09-01', '2024-09-01',
               0))
    
    print(f"  ADDED #{alpha_id}: OOS_SR={oos_sharpe:+.3f} | {expr[:60]}")
    added += 1

conn.commit()
conn.close()
print(f"\nDone: {added} alphas added to database")
