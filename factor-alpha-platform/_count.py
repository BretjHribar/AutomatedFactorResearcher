import sqlite3
conn = sqlite3.connect("data/alphas.db")
n = conn.execute("SELECT COUNT(*) FROM alphas WHERE archived=0 AND interval='4h' AND universe='BINANCE_TOP50'").fetchone()[0]
print(n)
conn.close()
