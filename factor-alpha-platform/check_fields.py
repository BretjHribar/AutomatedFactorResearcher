import os
fields = ['invested_capital', 'sales', 'return_equity', 'liabilities_curr', 'assets']
for f in fields:
    p = f'data/fmp_cache/matrices/{f}.parquet'
    exists = os.path.exists(p)
    tag = "YES" if exists else "NO"
    print(f'{f:25s} {tag}')
