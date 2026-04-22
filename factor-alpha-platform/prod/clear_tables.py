import sqlite3

# Clear crypto alphas
conn = sqlite3.connect('data/alphas.db')
count = conn.execute('SELECT COUNT(*) FROM alphas').fetchone()[0]
conn.execute('DELETE FROM alphas')
conn.commit()
print(f'data/alphas.db: deleted {count} rows')

tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
for t in tables:
    c = conn.execute(f'SELECT COUNT(*) FROM {t[0]}').fetchone()[0]
    if c > 0:
        print(f'  Table {t[0]}: {c} rows remaining')
conn.close()

# Clear IB alphas
conn2 = sqlite3.connect('data/ib_alphas.db')
count2 = conn2.execute('SELECT COUNT(*) FROM alphas').fetchone()[0]
conn2.execute('DELETE FROM alphas')
conn2.commit()
print(f'data/ib_alphas.db: deleted {count2} rows')
conn2.close()

print('All alpha tables cleared.')
