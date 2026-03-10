import subprocess
import re
import time
import sys

def run_loop():
    print("Running eval_portfolio.py --compare ...")
    res = subprocess.run(["python", "eval_portfolio.py", "--compare"], capture_output=True, text=True)
    top_sharpe = 0
    lines = res.stdout.split('\n')
    top_strategy_line = ""
    for line in lines:
        if "RegimeScaled" in line or "ProperDecay" in line or "ProperAdaptive" in line:
            parts = line.split()
            if len(parts) > 1:
                val = parts[1].replace('+', '')
                try:
                    sharpe = float(val)
                    if sharpe > top_sharpe:
                        top_sharpe = sharpe
                        top_strategy_line = line
                except:
                    pass
    print(f"Top Sharpe: {top_sharpe}")
    print(f"Top Strategy line: {top_strategy_line}")
    if top_sharpe >= 2.5:
        print("Target reached!")
    else:
        print("Target not reached yet.")

if __name__ == "__main__":
    run_loop()
