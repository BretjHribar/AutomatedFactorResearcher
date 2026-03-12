"""
Download 15m Binance Futures data for additional high-volume symbols.
Uses the Binance public API (no auth required for kline data).
"""
import requests
import pandas as pd
import time
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data' / 'binance_futures_15m'

# Top Binance Futures by volume (excluding existing 10)
NEW_SYMBOLS = [
    # Already downloaded: TRXUSDT, DOTUSDT, MATICUSDT
    'UNIUSDT', 'APTUSDT', 'NEARUSDT', 'FILUSDT', 'ATOMUSDT',
    'OPUSDT', 'ARBUSDT', 'SUIUSDT', '1000SHIBUSDT', '1000PEPEUSDT',
    'INJUSDT', 'FETUSDT', 'AAVEUSDT', 'WLDUSDT',
    'XLMUSDT', 'ICPUSDT', 'EOSUSDT', 'ETCUSDT', 'MKRUSDT',
    'RNDRUSDT', 'SEIUSDT', 'JUPUSDT', 'TONUSDT', 'ENAUSDT',
]


def download_klines(symbol, interval='15m', start='2022-01-01', limit=1500):
    """Download klines from Binance Futures API."""
    url = 'https://fapi.binance.com/fapi/v1/klines'
    start_ts = int(pd.Timestamp(start).timestamp() * 1000)
    end_ts = int(pd.Timestamp.now().timestamp() * 1000)
    
    all_data = []
    current = start_ts
    
    while current < end_ts:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current,
            'limit': limit,
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f'  Error: {e}')
            time.sleep(5)
            continue
        
        if not data:
            break
        
        all_data.extend(data)
        current = data[-1][0] + 1  # Next candle after last
        
        if len(data) < limit:
            break
        
        time.sleep(0.1)  # Rate limit
    
    if not all_data:
        return None
    
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades',
        'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    
    df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume',
                 'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume',
             'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']]
    df = df.drop_duplicates(subset='datetime').sort_values('datetime')
    
    return df


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check which symbols already exist
    existing = {p.stem for p in DATA_DIR.glob('*.parquet')}
    to_download = [s for s in NEW_SYMBOLS if s not in existing]
    
    print(f'Existing: {len(existing)} symbols')
    print(f'To download: {len(to_download)} symbols')
    
    for sym in to_download:
        print(f'\n  Downloading {sym}...', end='', flush=True)
        t0 = time.time()
        df = download_klines(sym, start='2022-01-01')
        if df is not None and len(df) > 1000:
            path = DATA_DIR / f'{sym}.parquet'
            df.to_parquet(path, index=False)
            print(f' {len(df)} bars ({time.time()-t0:.1f}s) → {path.name}')
        else:
            print(f' SKIP (only {len(df) if df is not None else 0} bars)')
    
    # Summary
    all_files = list(DATA_DIR.glob('*.parquet'))
    print(f'\n  Total: {len(all_files)} symbol data files')


if __name__ == '__main__':
    main()
