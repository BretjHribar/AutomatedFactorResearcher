"""Run a subprocess identical in shape to recorders._run_subprocess to verify
the rebuilt user_code container can actually import eval_alpha_ib + friends in
a fresh subprocess (not just in-process).
"""
import os, subprocess, sys

env = dict(os.environ)
env["PYTHONPATH"] = "/opt/dagster/app:/opt/dagster/app/prod"
res = subprocess.run(
    [sys.executable, "-c",
     "import eval_alpha_ib, seed_alphas_ib, live_bar, moc_trader; print('ALL_OK')"],
    cwd="/opt/dagster/app",
    env=env,
    capture_output=True, text=True,
)
print("STDOUT:", (res.stdout or "").strip())
print("STDERR:", (res.stderr or "").strip()[:400])
print("RC:", res.returncode)
