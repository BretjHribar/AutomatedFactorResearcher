import sqlite3
import os

db_path = 'c:/Users/breth/PycharmProjects/AutomatedFactorResearcher/factor-alpha-platform/data/alphas.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("--- ALPHAS ---")
cursor.execute("SELECT id, expression FROM alphas")
for row in cursor.fetchall():
    print(f"ID: {row[0]}, Expr: {row[1]}")

conn.close()
