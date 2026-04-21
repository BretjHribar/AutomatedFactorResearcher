"""
collect_results.py — Parse ib_portfolio_results.txt for Billions Test SR at 0.50 bps
Run after each experiment, or after all complete.

Usage: python collect_results.py > results_summary.txt
"""
import re, glob, os, sys
from pathlib import Path

TARGET_LABEL = "IBKR Tiered 20M-100M  $0.0010/sh"   # ~0.50 bps tier

results_txt = Path("ib_portfolio_results.txt")
if not results_txt.exists():
    print("No results file found.")
    sys.exit(1)

content = results_txt.read_text(encoding='utf-8', errors='ignore')

# Find universe
univ_match = re.search(r'Universe: (\S+) \|', content)
universe = univ_match.group(1) if univ_match else "UNKNOWN"

# Find the correct fee section
fee_sections = re.split(r'---- (.+?) ----', content)
target_section = None
for i, s in enumerate(fee_sections):
    if TARGET_LABEL in s:
        target_section = fee_sections[i+1] if i+1 < len(fee_sections) else ""
        break

if not target_section:
    # Fall back to first fee block after header
    target_section = content

# Parse combiner rows: Method | Train SR | Train TO | Trn$/day | Val SR | Val TO | Val$/day | Test SR | Test TO | Tst$/day
row_pat = re.compile(
    r'([\w\s\[\]]+?)\s+\|\s+([-+]?\d+\.\d+)\s+(\d+\.\d+)\s+\$\s*([\d,k]+)\s+\|\s+([-+]?\d+\.\d+)\s+(\d+\.\d+)\s+\$\s*([\d,k]+)\s+\|\s+([-+]?\d+\.\d+)\s+(\d+\.\d+)'
)

print(f"\n{'='*80}")
print(f"  Universe: {universe}")
print(f"  Fee tier: {TARGET_LABEL} (~0.50 bps eff.)")
print(f"{'='*80}")
print(f"  {'Method':<35} | {'Train SR':>9} | {'Val SR':>8} | {'Test SR':>9} | {'Test TO':>8}")
print(f"  {'-'*75}")

for m in row_pat.finditer(target_section):
    method = m.group(1).strip()
    train_sr = float(m.group(2))
    val_sr = float(m.group(5))
    test_sr = float(m.group(8))
    test_to = float(m.group(9))
    if 'QP' in method:
        continue  # skip QP in fast runs
    print(f"  {method:<35} | {train_sr:>+9.3f} | {val_sr:>+8.3f} | {test_sr:>+9.3f} | {test_to:>8.4f}")
