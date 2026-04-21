import json
d = json.load(open("logs/trades/trade_2026-04-20.json"))
print(f"Date: {d['date']}")
print(f"Mode: {d['mode']}")
print(f"Signal date: {d['signal_date']}")
print(f"Portfolio: {d['portfolio_summary']}")
print(f"Orders: {d['n_orders']}")
print(f"Config: {json.dumps(d['config'], indent=2)}")
print(f"Top 5 positions (by shares):")
tp = d["target_portfolio"]
for sym in sorted(tp, key=lambda s: abs(tp[s]), reverse=True)[:10]:
    print(f"  {sym}: {tp[sym]} shares")
