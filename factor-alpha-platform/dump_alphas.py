import sqlite3, json

conn = sqlite3.connect('data/alphas_5m.db')
rows = conn.execute(
    'SELECT id, expression, name, category, source, notes, created_at FROM alphas ORDER BY id'
).fetchall()

data = []
for r in rows:
    data.append({
        'id': r[0],
        'expression': r[1],
        'name': r[2],
        'category': r[3],
        'source': r[4],
        'notes': r[5],
        'created_at': r[6]
    })

with open('alpha_listing.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"Dumped {len(data)} alphas to alpha_listing.json")
conn.close()
