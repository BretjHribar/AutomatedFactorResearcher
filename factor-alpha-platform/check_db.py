import sqlite3
conn = sqlite3.connect('data/alpha_gp_crypto_4h.db')
cur = conn.cursor()
cur.execute('''SELECT a.alpha_id, a.expression, e.sharpe, e.fitness, e.turnover
               FROM alphas a JOIN evaluations e ON a.alpha_id = e.alpha_id
               ORDER BY e.sharpe DESC''')
rows = cur.fetchall()
print(f'Total: {len(rows)} alphas')
print(f'{"#":>3} | {"Sharpe":>8} | {"Fitness":>8} | {"TO":>7} | Expression')
print('-' * 120)
for r in rows:
    to_str = f'{r[4]*100:.1f}%'
    print(f'{r[0]:3d} | {r[2]:+8.3f} | {r[3]:8.3f} | {to_str:>7s} | {r[1]}')
conn.close()
