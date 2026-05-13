# Install (or replace) the Windows Scheduled Task that runs the IB Gateway
# watchdog every 10 minutes. Triggers at user login and every 10 minutes
# thereafter indefinitely.
#
# Run from an ELEVATED PowerShell. (Right-click PowerShell -> Run as administrator.)
#
#   .\tools\install_ib_gateway_watchdog_task.ps1
#
# To uninstall:
#   Unregister-ScheduledTask -TaskName "IB Gateway Watchdog" -Confirm:$false

$ErrorActionPreference = "Stop"

$TaskName = "IB Gateway Watchdog"
$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$Script = Join-Path $RepoRoot "tools\ib_gateway_watchdog.py"

# Watchdog uses only stdlib (socket, subprocess, json) -- no venv-installed
# packages needed. Prefer a real Python install on disk over the repo venv
# because the repo's venv\Scripts\python.exe launcher is broken on this host
# (embedded base-interpreter path points to C:\Users\B\...\Python311 from
# the original project clone). Search common locations + PATH.
$PythonCandidates = @(
    "C:\Users\$env:UserName\AppData\Local\Programs\Python\Python312\python.exe",
    "C:\Users\$env:UserName\AppData\Local\Programs\Python\Python311\python.exe",
    "C:\Users\$env:UserName\AppData\Local\Programs\Python\Python313\python.exe",
    "C:\Users\$env:UserName\AppData\Local\Programs\Python\Python314\python.exe",
    "C:\Python312\python.exe",
    "C:\Python311\python.exe"
)
$PythonExe = $null
foreach ($p in $PythonCandidates) {
    if (Test-Path $p) { $PythonExe = $p; break }
}
if (-not $PythonExe) {
    # Last resort: ask PATH
    $w = Get-Command python -ErrorAction SilentlyContinue
    if ($w) { $PythonExe = $w.Source }
}
if (-not $PythonExe -or -not (Test-Path $PythonExe)) {
    throw "no working python.exe found -- install Python 3.10+ or fix venv"
}
Write-Host "using python: $PythonExe"

if (-not (Test-Path $Script)) {
    throw "watchdog script not found at $Script"
}

$Action  = New-ScheduledTaskAction -Execute $PythonExe -Argument "`"$Script`" --once" -WorkingDirectory $RepoRoot
$AtLogon = New-ScheduledTaskTrigger -AtLogOn
$Repeat  = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 10) `
    -RepetitionDuration (New-TimeSpan -Days 365)
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 2) `
    -MultipleInstances IgnoreNew
$Principal = New-ScheduledTaskPrincipal -UserId $env:UserName -LogonType Interactive -RunLevel Limited

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Write-Host "Removing existing task: $TaskName"
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask `
    -TaskName $TaskName `
    -Description "Polls IB Gateway port 4002; relaunches ibgateway.exe if down. Runs every 10 min." `
    -Action $Action `
    -Trigger @($AtLogon, $Repeat) `
    -Settings $Settings `
    -Principal $Principal | Out-Null

Write-Host "Installed scheduled task '$TaskName'."
Write-Host "  Runs:    $PythonExe `"$Script`" --once"
Write-Host "  Trigger: at log on + every 10 min for 365 days"
Write-Host "  Log:     $RepoRoot\prod\logs\ib_gateway_watchdog.log"
Write-Host ""
Write-Host "Verify:"
Write-Host "  Get-ScheduledTask -TaskName '$TaskName' | Format-List"
Write-Host "  Get-ScheduledTaskInfo -TaskName '$TaskName'"
