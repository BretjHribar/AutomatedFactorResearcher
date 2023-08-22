import sys
import subprocess

procs = []
for i in range(2):
    proc = subprocess.Popen([sys.executable, 'AWSgenProgWorkerV2.py'])
    procs.append(proc)

for proc in procs:
    proc.wait()