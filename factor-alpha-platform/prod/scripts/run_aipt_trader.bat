@echo off
cd /d "C:\Users\breth\PycharmProjects\AutomatedFactorResearcher\factor-alpha-platform"
"C:\Users\breth\AppData\Local\Programs\Python\Python312\python.exe" -u "prod\aipt_trader.py" >> "prod\logs\kucoin\aipt\scheduler_stdout.log" 2>&1
