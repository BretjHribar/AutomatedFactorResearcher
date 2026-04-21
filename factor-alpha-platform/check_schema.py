import eval_alpha_ib as ib

# Test eval_single on H1 split directly
expr = "rank((low - close) / (high - low + 0.001))"
r = ib.eval_single(expr, split="h1")
print("H1 success:", r["success"])
if r["success"]:
    print("H1 sharpe:", r["sharpe"])
    print("H1 n_bars:", r["n_bars"])
    print("H1 turnover:", r["turnover"])
    print("H1 pnl sum:", r["pnl_vec"].sum())
    print("H1 pnl len:", len(r["pnl_vec"]))
else:
    print("H1 error:", r["error"])
