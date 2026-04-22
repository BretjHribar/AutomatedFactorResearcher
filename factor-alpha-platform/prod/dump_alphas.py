import sqlite3
conn = sqlite3.connect('data/ib_alphas.db')
rows = conn.execute("SELECT id, expression FROM alphas WHERE archived=0 AND asset_class='equities_ib'").fetchall()
conn.close()

fields_used = set()
for aid, expr in rows:
    print(f"{aid:3d}: {expr}")
    for field in ['close', 'open', 'high', 'low', 'volume', 'returns', 'vwap', 'adv20']:
        if field in expr.lower():
            fields_used.add(field)

print(f"\n--- Fields referenced across all 31 alphas: {sorted(fields_used)}")
