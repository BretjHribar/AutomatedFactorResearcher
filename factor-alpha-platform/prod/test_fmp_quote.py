"""Test FMP batch quote for our universe tickers."""
import requests, json

API_KEY = "C6T2KGmSbbsDL3sM7gjx680hmUTiEXfy"
tickers = ["AAPL", "NC", "WLKP", "AFGC", "BNTC", "OOMA", "XOMA", "VGZ"]

url = f"https://financialmodelingprep.com/stable/quote?symbol={','.join(tickers)}&apikey={API_KEY}"
resp = requests.get(url)
data = resp.json()
print(f"Got {len(data)} quotes\n")
print(f"{'Sym':<8} {'Open':>8} {'High':>8} {'Low':>8} {'Price':>8} {'PrevCl':>8} {'Volume':>10}")
print("-" * 70)
for q in data:
    print(f"{q['symbol']:<8} {q.get('open',0):>8.2f} {q.get('dayHigh',0):>8.2f} "
          f"{q.get('dayLow',0):>8.2f} {q.get('price',0):>8.2f} {q.get('previousClose',0):>8.2f} "
          f"{q.get('volume',0):>10,}")
