import os
import sys
import io
import time
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Insert path to allow importing aipt_kucoin
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import aipt_kucoin as aipt

ps_to_test = [10, 100, 1000, 5000, 10000]
results = []

for p in ps_to_test:
    print(f"Running P={p}...")
    # Capture stdout
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    
    t0 = time.time()
    try:
        aipt.run_production(P=p)
    except Exception as e:
        sys.stdout = old_stdout
        print(f"Failed on P={p}: {e}")
        continue
    
    # Restore stdout
    sys.stdout = old_stdout
    output = new_stdout.getvalue()
    
    # Parse output metrics using regex
    gross_sr_m = re.search(r"Annualized Sharpe \(0bps fees\):\s+\+?([\d\.]+)", output)
    net_sr_m = re.search(r"Annualized Sharpe \(3bps fees\):\s+\+?([\d\.]+)", output)
    turnover_m = re.search(r"Mean turnover per bar:\s+([\d\.]+)", output)
    mean_ret_m = re.search(r"Mean port return per bar:\s+([\d\.]+)", output)
    std_ret_m = re.search(r"Std  port return per bar:\s+([\d\.]+)", output)
    
    gross_sr = float(gross_sr_m.group(1)) if gross_sr_m else np.nan
    net_sr = float(net_sr_m.group(1)) if net_sr_m else np.nan
    turnover = float(turnover_m.group(1)) if turnover_m else np.nan
    mean_ret = float(mean_ret_m.group(1)) if mean_ret_m else np.nan
    std_ret = float(std_ret_m.group(1)) if std_ret_m else np.nan
    elapsed = time.time() - t0
    
    res = {
        "P": p,
        "Gross_SR": gross_sr,
        "Net_SR": net_sr,
        "Turnover": turnover,
        "Volatility": std_ret,
        "Time": elapsed
    }
    results.append(res)
    print(f"  Gross SR:  {gross_sr}")
    print(f"  Net SR:    {net_sr}")
    print(f"  Turnover:  {turnover}")
    print(f"  Time:      {elapsed:.1f}s")


# Save plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("AIPT 'Virtue of Complexity'\nImpact of Random Fourier Dimension (P)", fontsize=16)

ps = [r['P'] for r in results]

# Plot 1: Sharpe Ratios
ax1 = axes[0, 0]
ax1.plot(ps, [r['Gross_SR'] for r in results], marker='o', label='Gross SR (0bps)', color='teal', linewidth=2)
ax1.plot(ps, [r['Net_SR'] for r in results], marker='o', label='Net SR (3bps)', color='darkred', linewidth=2)
ax1.set_xscale('log')
ax1.set_title('Annualized Sharpe Ratio vs P')
ax1.set_xlabel('Number of Random Features (P) - Log Scale')
ax1.set_ylabel('Sharpe Ratio')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Volatility
ax2 = axes[0, 1]
ax2.plot(ps, [r['Volatility'] * 10000 for r in results], marker='o', color='purple', linewidth=2)
ax2.set_xscale('log')
ax2.set_title('Portfolio Volatility (bps per bar)')
ax2.set_xlabel('Number of Random Features (P) - Log Scale')
ax2.set_ylabel('Volatility (bps)')
ax2.grid(True, alpha=0.3)

# Plot 3: Turnover
ax3 = axes[1, 0]
ax3.plot(ps, [r['Turnover'] for r in results], marker='o', color='orange', linewidth=2)
ax3.set_xscale('log')
ax3.set_title('Mean Turnover per 4h Bar')
ax3.set_xlabel('Number of Random Features (P) - Log Scale')
ax3.set_ylabel('Turnover Fraction')
ax3.grid(True, alpha=0.3)

# Plot 4: Compute Time
ax4 = axes[1, 1]
ax4.plot(ps, [r['Time'] for r in results], marker='o', color='gray', linewidth=2)
ax4.set_xscale('log')
ax4.set_title('Computation Time (seconds)')
ax4.set_xlabel('Number of Random Features (P) - Log Scale')
ax4.set_ylabel('Execution Time (s)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(PROJECT_ROOT, "data", "aipt_results", "P_complexity_sweep.png")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\nSaved summary chart to {save_path}")
