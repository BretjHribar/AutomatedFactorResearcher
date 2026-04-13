"""
Model: What if we had alphas with turnover < 0.05 but similar gross Sharpe?

Two approaches:
1. Analytical: hold gross PnL constant, vary fee drag
2. Empirical: for each EWMA span, what gross SR would we NEED to break even?
"""
import numpy as np, time, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import eval_alpha_5m as ea
import eval_portfolio_5m as ep

BOOKSIZE = ep.BOOKSIZE
MAX_WT = ea.MAX_WEIGHT
BARS_PER_DAY = ep.BARS_PER_DAY

print("Loading data...", flush=True); t0 = time.time()
matrices, universe = ea.load_data("train")
close = matrices["close"]; returns = close.pct_change()
alphas = ep.load_alphas(universe="TOP100")

raw_sum = None; n = 0
for i, (aid, expr, ic, sr) in enumerate(alphas):
    print(f"  Alpha {i+1}/{len(alphas)}...", end='', flush=True)
    try:
        raw = ep.evaluate_expression(expr, matrices)
        if raw is None: print(" SKIP"); continue
        r = raw.fillna(0)
        if raw_sum is None: raw_sum = r.copy()
        else: raw_sum = raw_sum.add(r, fill_value=0)
        n += 1; print(" OK", flush=True)
    except Exception as e: print(f" ERR")

composite = ea.process_signal(raw_sum / n, universe_df=universe, max_wt=MAX_WT)
common = composite.columns.intersection(returns.columns).tolist()
idx = composite.index.intersection(returns.index)
sig = np.nan_to_num(composite.loc[idx, common].values.astype(np.float64), nan=0)
ret = np.nan_to_num(returns.loc[idx, common].values.astype(np.float64), nan=0)
n_bars, n_tickers = sig.shape
print(f"Ready: {n_bars} bars, {n_tickers} tickers ({time.time()-t0:.0f}s)\n", flush=True)

# ===== Get ACTUAL gross PnL + turnover at each EWMA span =====
print("Computing gross stats at each EWMA span...", flush=True)
spans = [1, 3, 6, 12, 24, 36, 48, 72, 96, 144, 288, 576]
ema_stats = []

for sp in spans:
    s = sig.copy()
    if sp > 1:
        a = 2.0/(sp+1)
        for t in range(1, n_bars): s[t] = a*s[t]+(1-a)*s[t-1]
    
    pos = np.zeros(n_tickers); gross_pnl = np.zeros(n_bars); to_arr = np.zeros(n_bars)
    for t in range(1, n_bars):
        gross_pnl[t] = np.sum(pos*ret[t])*BOOKSIZE
        asum = np.abs(s[t]).sum()
        if asum > 1e-10:
            new = np.clip(s[t]/asum, -MAX_WT, MAX_WT)
            a2 = np.abs(new).sum()
            if a2 > 1e-10: new /= a2
        else: new = pos.copy()
        to_arr[t] = np.abs(new-pos).sum(); pos = new
    
    avg_to = np.mean(to_arr[1:])
    daily_gross = [np.sum(gross_pnl[d:d+BARS_PER_DAY]) for d in range(0, n_bars, BARS_PER_DAY)]
    daily_gross = np.array(daily_gross)
    gross_sr = np.mean(daily_gross)/np.std(daily_gross)*np.sqrt(365) if np.std(daily_gross)>0 else 0
    gross_ret = np.sum(gross_pnl)/BOOKSIZE
    
    ema_stats.append({
        'span': sp, 'to': avg_to, 'gross_sr': gross_sr, 'gross_ret': gross_ret,
        'daily_mean': np.mean(daily_gross), 'daily_std': np.std(daily_gross),
        'gross_pnl': gross_pnl
    })
    
    hl = f"{sp*5/60:.0f}h" if sp < 288 else f"{sp*5/60/24:.0f}d"
    print(f"  EMA={sp:>4} ({hl:>4}): TO={avg_to:.4f}  Gross SR={gross_sr:+.2f}  Gross Ret={gross_ret:+.1%}  "
          f"Daily mean=${np.mean(daily_gross):,.0f}  std=${np.std(daily_gross):,.0f}", flush=True)


# ===== MODEL 1: Analytical — What net SR would a hypothetical alpha give? =====
# If an alpha has gross_SR = X and turnover = TO, what is its net SR at fee F?
# Net daily mean = gross_daily_mean - TO * bars_per_day * fee_rate * booksize
# Net daily std ≈ gross_daily_std (fee is constant, doesn't add much vol)

print(f"\n{'='*90}", flush=True)
print("MODEL 1: Net Sharpe as f(turnover, gross_sharpe, fees)", flush=True)
print(f"{'='*90}", flush=True)

# Use the actual gross stats from EMA=1 (highest gross alpha) as the "base case"
base = ema_stats[0]  # EMA=1
base_gross_daily_mean = base['daily_mean']
base_gross_daily_std = base['daily_std']
base_gross_sr = base['gross_sr']

print(f"\nBase case (EMA=1, current signal):", flush=True)
print(f"  Gross daily mean PnL: ${base_gross_daily_mean:,.0f}", flush=True)
print(f"  Gross daily std PnL:  ${base_gross_daily_std:,.0f}", flush=True)
print(f"  Gross annual SR:      {base_gross_sr:+.2f}", flush=True)
print(f"  Actual TO:            {base['to']:.4f}/bar", flush=True)

# For hypothetical alphas with similar gross stats but different TO:
print(f"\n{'TO/bar':>8} {'TO/day':>8} {'Fee/day':>10} {'Fee/day':>10} {'Net SR':>8} {'Net SR':>8} {'Net SR':>8} {'Net Ret':>10} {'Net Ret':>10}", flush=True)
print(f"{'':>8} {'':>8} {'@1bp':>10} {'@3bp':>10} {'@1bp':>8} {'@3bp':>8} {'@7bp':>8} {'@1bp':>10} {'@3bp':>10}", flush=True)
print("-"*100, flush=True)

hyp_tos = [0.40, 0.30, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001]
model1_data = []

for to in hyp_tos:
    to_day = to * BARS_PER_DAY
    fee_1bp = to_day * (1/10000) * BOOKSIZE
    fee_3bp = to_day * (3/10000) * BOOKSIZE
    fee_7bp = to_day * (7/10000) * BOOKSIZE
    
    net_mean_1 = base_gross_daily_mean - fee_1bp
    net_mean_3 = base_gross_daily_mean - fee_3bp
    net_mean_7 = base_gross_daily_mean - fee_7bp
    
    sr_1 = net_mean_1/base_gross_daily_std*np.sqrt(365)
    sr_3 = net_mean_3/base_gross_daily_std*np.sqrt(365)
    sr_7 = net_mean_7/base_gross_daily_std*np.sqrt(365)
    
    n_days = n_bars / BARS_PER_DAY
    ret_1 = net_mean_1 * n_days / BOOKSIZE
    ret_3 = net_mean_3 * n_days / BOOKSIZE
    
    model1_data.append((to, sr_1, sr_3, sr_7, ret_1, ret_3))
    
    print(f"{to:8.4f} {to_day:8.1f} ${fee_1bp:9,.0f} ${fee_3bp:9,.0f} {sr_1:+8.2f} {sr_3:+8.2f} {sr_7:+8.2f} {ret_1:+10.1%} {ret_3:+10.1%}", flush=True)


# ===== MODEL 2: At each ACTUAL EWMA smoothing level, what gross SR would break even? =====
print(f"\n{'='*90}", flush=True)
print("MODEL 2: Required gross SR to break even at each turnover level", flush=True)
print(f"{'='*90}", flush=True)
print(f"\nThis answers: 'If I find a NATIVELY low-turnover alpha, what gross SR do I need?'", flush=True)
print(f"\n{'TO/bar':>8} {'TO/day':>8} {'Fee/day@3bp':>12} {'Break-even':>12} {'Break-even':>12} {'Actual':>8}", flush=True)
print(f"{'':>8} {'':>8} {'':>12} {'daily mean':>12} {'annual SR':>12} {'gross SR':>8}", flush=True)
print("-"*90, flush=True)

for es in ema_stats:
    to = es['to']; to_day = to * BARS_PER_DAY
    fee_3 = to_day * (3/10000) * BOOKSIZE
    fee_7 = to_day * (7/10000) * BOOKSIZE
    # Break even: net_mean = 0 => gross_mean = fee/day
    # Required SR = gross_mean / gross_std * sqrt(365)
    be_sr_3 = fee_3 / es['daily_std'] * np.sqrt(365) if es['daily_std'] > 0 else 0
    be_sr_7 = fee_7 / es['daily_std'] * np.sqrt(365) if es['daily_std'] > 0 else 0
    surplus_3 = es['gross_sr'] - be_sr_3
    surplus_7 = es['gross_sr'] - be_sr_7
    
    hl = f"{es['span']*5/60:.0f}h" if es['span'] < 288 else f"{es['span']*5/60/24:.0f}d"
    print(f"{to:8.4f} {to_day:8.1f} ${fee_3:11,.0f} ${fee_3:11,.0f}  SR={be_sr_3:+5.1f}  SR={be_sr_7:+5.1f}  | Actual={es['gross_sr']:+.1f}  Surplus@3bp={surplus_3:+.1f}  Surplus@7bp={surplus_7:+.1f}", flush=True)


# ===== MODEL 3: Direct answer — alpha with TO=0.05 and similar gross SR =====
print(f"\n{'='*90}", flush=True)
print("MODEL 3: HYPOTHETICAL ALPHA — TO=0.05 with various gross Sharpe levels", flush=True)
print(f"{'='*90}", flush=True)

# At TO=0.05, what daily fee cost?
to_target = 0.05
to_day = to_target * BARS_PER_DAY
fee_per_day_1bp = to_day * (1/10000) * BOOKSIZE
fee_per_day_3bp = to_day * (3/10000) * BOOKSIZE
fee_per_day_7bp = to_day * (7/10000) * BOOKSIZE

print(f"\nTO = {to_target}/bar = {to_day:.0f}/day", flush=True)
print(f"Fee/day: @1bp=${fee_per_day_1bp:,.0f}  @3bp=${fee_per_day_3bp:,.0f}  @7bp=${fee_per_day_7bp:,.0f}", flush=True)

# Use the vol from EMA~96 (which has TO~0.05) as representative
vol_ref = ema_stats[7]  # EMA=96, TO=0.052
ref_std = vol_ref['daily_std']
print(f"Reference daily PnL std (from EMA=96, similar TO): ${ref_std:,.0f}", flush=True)

print(f"\n{'Gross SR':>10} {'Daily mean':>12} {'Net@1bp':>10} {'Net@3bp':>10} {'Net@7bp':>10} {'Ret@3bp':>10} {'Ret@7bp':>10}", flush=True)
print("-"*85, flush=True)

for gross_sr in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]:
    daily_mean = gross_sr * ref_std / np.sqrt(365)
    net_1 = (daily_mean - fee_per_day_1bp) / ref_std * np.sqrt(365)
    net_3 = (daily_mean - fee_per_day_3bp) / ref_std * np.sqrt(365)
    net_7 = (daily_mean - fee_per_day_7bp) / ref_std * np.sqrt(365)
    n_days = n_bars / BARS_PER_DAY
    ret_3 = (daily_mean - fee_per_day_3bp) * n_days / BOOKSIZE
    ret_7 = (daily_mean - fee_per_day_7bp) * n_days / BOOKSIZE
    print(f"{gross_sr:10.0f} ${daily_mean:11,.0f} {net_1:+10.2f} {net_3:+10.2f} {net_7:+10.2f} {ret_3:+10.1%} {ret_7:+10.1%}", flush=True)


# ===== PLOT =====
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Net SR vs Turnover (holding gross SR constant at current level)
ax = axes[0, 0]
tos_plot = [d[0] for d in model1_data]
for fee, idx, color, label in [(1, 1, 'green', '1 bps'), (3, 2, 'orange', '3 bps'), (7, 3, 'red', '7 bps')]:
    srs = [d[idx] for d in model1_data]
    ax.semilogx(tos_plot, srs, 'o-', color=color, linewidth=2, markersize=8, label=label)
ax.axhline(y=0, color='gray', linestyle='--')
ax.axhline(y=2, color='gray', linestyle=':', alpha=0.5)
ax.axvline(x=0.05, color='blue', linestyle='--', alpha=0.5, label='TO=0.05 target')
ax.set_xlabel('Turnover per Bar')
ax.set_ylabel('Net Sharpe Ratio')
ax.set_title(f'Model 1: Net SR vs Turnover\n(Holding gross SR={base_gross_sr:.1f} constant)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-15, 15)

# 2. Required gross SR to break even vs turnover
ax = axes[0, 1]
tos_be = [es['to'] for es in ema_stats]
for fee, color, label in [(3, 'orange', '3 bps'), (7, 'red', '7 bps')]:
    be_srs = []
    for es in ema_stats:
        to_d = es['to'] * BARS_PER_DAY
        fee_d = to_d * (fee/10000) * BOOKSIZE
        be_sr = fee_d / es['daily_std'] * np.sqrt(365) if es['daily_std'] > 0 else 0
        be_srs.append(be_sr)
    ax.semilogx(tos_be, be_srs, 's-', color=color, linewidth=2, markersize=8, label=f'Need (@ {fee}bp)')
actual_srs = [es['gross_sr'] for es in ema_stats]
ax.semilogx(tos_be, actual_srs, 'bo-', linewidth=2, markersize=8, label='Actual gross SR')
ax.axvline(x=0.05, color='blue', linestyle='--', alpha=0.5, label='TO=0.05 target')
ax.set_xlabel('Turnover per Bar')
ax.set_ylabel('Sharpe Ratio')
ax.set_title('Model 2: Required vs Actual Gross SR')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Net SR vs Gross SR at TO=0.05
ax = axes[1, 0]
gross_srs = np.arange(0, 25, 0.5)
for fee, color, label in [(1, 'green', '1 bps'), (3, 'orange', '3 bps'), (7, 'red', '7 bps')]:
    fee_d = to_target * BARS_PER_DAY * (fee/10000) * BOOKSIZE
    net_srs = [(g * ref_std / np.sqrt(365) - fee_d) / ref_std * np.sqrt(365) for g in gross_srs]
    ax.plot(gross_srs, net_srs, '-', color=color, linewidth=2, label=label)
ax.axhline(y=0, color='gray', linestyle='--')
ax.axhline(y=2, color='gray', linestyle=':', alpha=0.5, label='Target SR=2')
ax.fill_between([6, 10], [-5, -5], [25, 25], alpha=0.1, color='green', label='Current alpha range')
ax.set_xlabel('Gross Sharpe Ratio')
ax.set_ylabel('Net Sharpe Ratio')
ax.set_title(f'Model 3: Net SR at TO=0.05 vs Gross SR')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 25)
ax.set_ylim(-5, 15)

# 4. Breakeven surface: TO vs Fee level, for gross SR=8
ax = axes[1, 1]
to_range = np.logspace(-3, -0.3, 50)
fee_range = np.arange(0.5, 10, 0.5)
for gross in [4, 6, 8, 10]:
    be_fees = []
    for to_val in to_range:
        # At what fee does net SR = 0?
        # net_mean = 0 => gross_mean = fee_cost
        # gross_mean = gross_SR * std / sqrt(365)
        gross_mean = gross * ref_std / np.sqrt(365)
        to_d = to_val * BARS_PER_DAY
        # gross_mean = to_d * (fee/10000) * booksize
        # fee = gross_mean / (to_d * booksize) * 10000
        be_fee = gross_mean / (to_d * BOOKSIZE) * 10000 if to_d > 0 else 999
        be_fees.append(min(be_fee, 15))
    ax.loglog(to_range, be_fees, '-', linewidth=2, label=f'Gross SR={gross}')
ax.axhline(y=3, color='orange', linestyle='--', alpha=0.7, label='3 bps')
ax.axhline(y=7, color='red', linestyle='--', alpha=0.7, label='7 bps')
ax.axvline(x=0.05, color='blue', linestyle='--', alpha=0.5, label='TO=0.05')
ax.set_xlabel('Turnover per Bar')
ax.set_ylabel('Breakeven Fee (bps)')
ax.set_title('Breakeven Fee vs Turnover (for each Gross SR)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(0.001, 0.5)
ax.set_ylim(0.5, 15)

plt.tight_layout()
plt.savefig('diag_turnover_model.png', dpi=150)
print(f"\nChart saved: diag_turnover_model.png", flush=True)
plt.close()
print(f"Done ({time.time()-t0:.0f}s)", flush=True)
