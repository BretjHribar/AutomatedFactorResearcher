@echo off
REM ============================================================================
REM  AIPT Trader - Windows Task Scheduler Setup
REM
REM  Creates a scheduled task that runs every 4h at :05 past the hour
REM  (5 min after bar close to ensure data is available on KuCoin API)
REM
REM  Schedule: 00:05, 04:05, 08:05, 12:05, 16:05, 20:05 UTC
REM
REM  To install:  Run this script as Administrator
REM  To remove:   schtasks /delete /tn "AIPT_KuCoin_Trader" /f
REM  To run now:  schtasks /run /tn "AIPT_KuCoin_Trader"
REM  To check:    schtasks /query /tn "AIPT_KuCoin_Trader" /v
REM ============================================================================

set PYTHON=C:\Users\breth\AppData\Local\Programs\Python\Python312\python.exe
set SCRIPT=C:\Users\breth\PycharmProjects\AutomatedFactorResearcher\factor-alpha-platform\prod\aipt_trader.py
set WORKDIR=C:\Users\breth\PycharmProjects\AutomatedFactorResearcher\factor-alpha-platform

echo Creating AIPT KuCoin Trader scheduled task...

REM Create task with 4h repetition interval
schtasks /create ^
    /tn "AIPT_KuCoin_Trader" ^
    /tr "\"%PYTHON%\" -u \"%SCRIPT%\"" ^
    /sc daily ^
    /st 00:05 ^
    /ri 240 ^
    /du 24:00 ^
    /sd 2026/04/23 ^
    /f ^
    /rl HIGHEST

echo.
echo Task created. Verify with:
echo   schtasks /query /tn "AIPT_KuCoin_Trader" /v
echo.
echo To run manually:
echo   schtasks /run /tn "AIPT_KuCoin_Trader"
echo.
echo To delete:
echo   schtasks /delete /tn "AIPT_KuCoin_Trader" /f
pause
