"""Precise fitness cutoff calculation"""
import math

booksize = 2_000_000
bars_per_year = 252 * 288  # 72576
bars_per_day = 288

# From our actual data at EMA=96 (TO=0.052):
# daily_std = $20,524
bar_std = 20524 / math.sqrt(288)

print("="*70)
print("BREAKEVEN TABLE")
print("="*70)
header = f"  {'TO':>6} {'FD@3bp':>8} {'BE@3bp':>8} {'FD@7bp':>8} {'BE@7bp':>8}"
print(header)
print("-"*50)
for to in [0.40, 0.22, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01]:
    fd3 = to * booksize * 3/10000 / bar_std * math.sqrt(bars_per_year)
    fd7 = to * booksize * 7/10000 / bar_std * math.sqrt(bars_per_year)
    mark = " <--" if to == 0.05 else ""
    print(f"  {to:6.3f} {fd3:8.1f} {fd3:8.1f} {fd7:8.1f} {fd7:8.1f}{mark}")

# Key numbers at TO=0.05
fee_drag_3 = 0.05 * booksize * 3/10000 / bar_std * math.sqrt(bars_per_year)
fee_drag_7 = 0.05 * booksize * 7/10000 / bar_std * math.sqrt(bars_per_year)

print(f"\nAt TO=0.05:")
print(f"  Fee drag @ 3bp = {fee_drag_3:.1f} Sharpe units")
print(f"  Fee drag @ 7bp = {fee_drag_7:.1f} Sharpe units")
print(f"  Need gross SR >= {fee_drag_3:.1f} to break even at 3bp")
print(f"  Need gross SR >= {fee_drag_3+2:.1f} for net SR=2 at 3bp")
print(f"  Need gross SR >= {fee_drag_7:.1f} to break even at 7bp")

print(f"\nAt TO=0.03:")
fd3_03 = 0.03 * booksize * 3/10000 / bar_std * math.sqrt(bars_per_year)
fd7_03 = 0.03 * booksize * 7/10000 / bar_std * math.sqrt(bars_per_year)
print(f"  Fee drag @ 3bp = {fd3_03:.1f} Sharpe units")
print(f"  Need gross SR >= {fd3_03:.1f} to break even at 3bp")

# FITNESS calculation for each scenario
print(f"\n{'='*70}")
print("FITNESS VALUES")
print(f"{'='*70}")

for label, sr, to in [
    ("Current typical (SR=8 TO=0.30)", 8, 0.30),
    ("Current best (SR=10 TO=0.22)", 10, 0.22),
    ("Target: SR=8 TO=0.05", 8, 0.05),
    ("Target: SR=6 TO=0.05", 6, 0.05),
    ("Target: SR=10 TO=0.05", 10, 0.05),
    ("Target: SR=10 TO=0.03", 10, 0.03),
    ("Breakeven@3bp: SR=8.3 TO=0.05", fee_drag_3, 0.05),
    ("NetSR=2@3bp: SR=10.3 TO=0.05", fee_drag_3+2, 0.05),
]:
    bar_mean = sr * bar_std / math.sqrt(bars_per_year)
    ret_ann = bar_mean * bars_per_year / (booksize * 0.5)
    denom = max(to, 0.125)
    fitness = sr * math.sqrt(abs(ret_ann) / denom)
    print(f"  {label:42s} fitness={fitness:7.1f}  retAnn={ret_ann:.2f}")

print(f"\n{'='*70}")
print("RECOMMENDED SETTINGS FOR ALPHA DISCOVERY WORKFLOW")
print(f"{'='*70}")
print()
print("  MAX_TURNOVER = 0.05          # CRITICAL: this is the key gate")
print(f"  MIN_IS_SHARPE = 6.0          # Keep current (fee-drag@3bp={fee_drag_3:.1f})")
print()
print("  At these settings, fitness would be ~20+")
print("  But FITNESS alone is NOT sufficient because it floors TO at 0.125")
print("  You MUST gate on turnover separately")
print()
print("  The fitness values you'd see for qualifying alphas:")
print(f"    SR=6  TO=0.05 => fitness ~{6*math.sqrt(abs(6*bar_std/math.sqrt(bars_per_year)*bars_per_year/(booksize*0.5))/0.125):.0f}")
print(f"    SR=8  TO=0.05 => fitness ~{8*math.sqrt(abs(8*bar_std/math.sqrt(bars_per_year)*bars_per_year/(booksize*0.5))/0.125):.0f}")
print(f"    SR=10 TO=0.05 => fitness ~{10*math.sqrt(abs(10*bar_std/math.sqrt(bars_per_year)*bars_per_year/(booksize*0.5))/0.125):.0f}")
