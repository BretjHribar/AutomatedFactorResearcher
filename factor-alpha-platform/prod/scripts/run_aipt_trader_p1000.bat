@echo off
cd /d "C:\Users\breth\PycharmProjects\AutomatedFactorResearcher\factor-alpha-platform"
"C:\Users\breth\AppData\Local\Programs\Python\Python312\python.exe" -u "prod\aipt_trader_p1000.py" >> "prod\logs\kucoin\aipt_p1000\scheduler_stdout.log" 2>&1
