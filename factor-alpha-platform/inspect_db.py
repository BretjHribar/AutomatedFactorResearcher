import sqlite3

for db_name in ['alphas.db', 'alpha_results.db']:
    print(f'\n=== {db_name} ===')
    try:
        con = sqlite3.connect(db_name)
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        print('Tables:', tables)
        for t in tables:
            cur.execute(f'PRAGMA table_info({t})')
            cols = [(r[1], r[2]) for r in cur.fetchall()]
            print(f'  {t} columns: {cols}')
            cur.execute(f'SELECT COUNT(*) FROM {t}')
            print(f'  {t} row count:', cur.fetchone()[0])
        # Show all rows for small tables
        for t in tables:
            cur.execute(f'SELECT COUNT(*) FROM {t}')
            count = cur.fetchone()[0]
            if count <= 20:
                cur.execute(f'SELECT * FROM {t}')
                rows = cur.fetchall()
                print(f'  {t} rows:')
                for row in rows:
                    print(f'    {row}')
        con.close()
    except Exception as e:
        print(f'Error: {e}')
