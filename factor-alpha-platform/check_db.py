import sqlite3
import pandas as pd

conn = sqlite3.connect('c:/Users/breth/PycharmProjects/AutomatedFactorResearcher/factor-alpha-platform/data/alphas_5m.db')
df = pd.read_sql_query("SELECT id, expression, notes FROM alphas WHERE source='agent1_research' ORDER BY id DESC LIMIT 10", conn)
print(df.to_string())
