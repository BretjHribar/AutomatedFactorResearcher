import subprocess
import shutil

print("Running portfolio comparison sweeps...")
subprocess.run(["python", "eval_portfolio.py", "--compare"])

try:
    shutil.copy("data/portfolio_results.csv", "portfolio_trials.csv")
    print("Successfully copied to portfolio_trials.csv!")
except Exception as e:
    print(f"Could not copy to portfolio_trials.csv: {e}")
