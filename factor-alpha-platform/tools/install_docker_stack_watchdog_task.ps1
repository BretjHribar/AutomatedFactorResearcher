# Install (or replace) the Windows Scheduled Task that runs the Docker stack
# watchdog every 5 minutes.
#
# Run from an ELEVATED PowerShell (right-click PowerShell -> Run as administrator):
#
#   PowerShell -ExecutionPolicy Bypass -NoProfile -File .\tools\install_docker_stack_watchdog_task.ps1
#
# What it does: every 5 min, the watchdog verifies (a) Docker is up,
# (b) all 4 Dagster containers are running, (c) the user_code container has
# its required env vars. If anything is wrong, it heals by re-running
# `docker compose --env-file ... up -d --force-recreate ...`. Logs to
# prod\logs\docker_stack_watchdog.log; state in prod\state\.
#
# To uninstall:
#   Unregister-ScheduledTask -TaskName "Docker Stack Watchdog" -Confirm:$false

$ErrorActionPreference = "Stop"

$TaskName = "Docker Stack Watchdog"
$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$Script = Join-Path $RepoRoot "tools\docker_stack_watchdog.py"

# Use a real Python install on disk (not the repo venv, since that launcher is
# broken on this host -- see ib_gateway_watchdog install script for the same
# logic).
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
    $w = Get-Command python -ErrorAction SilentlyContinue
    if ($w) { $PythonExe = $w.Source }
}
if (-not $PythonExe -or -not (Test-Path $PythonExe)) {
    throw "no working python.exe found"
}
Write-Host "using python: $PythonExe"

if (-not (Test-Path $Script)) {
    throw "watchdog script not found at $Script"
}

$Action  = New-ScheduledTaskAction -Execute $PythonExe -Argument "`"$Script`" --once" -WorkingDirectory $RepoRoot
$AtLogon = New-ScheduledTaskTrigger -AtLogOn
$Repeat  = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 5) `
    -RepetitionDuration (New-TimeSpan -Days 365)
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5) `
    -MultipleInstances IgnoreNew

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Write-Host "Removing existing task: $TaskName"
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask `
    -TaskName $TaskName `
    -Description "Polls Docker daemon + dagster containers + user_code env every 5 min; heals via docker compose --env-file ... up -d." `
    -Action $Action `
    -Trigger @($AtLogon, $Repeat) `
    -Settings $Settings | Out-Null

Write-Host "Installed scheduled task '$TaskName'."
Write-Host "  Runs:    $PythonExe `"$Script`" --once"
Write-Host "  Trigger: at log on + every 5 min for 365 days"
Write-Host "  Log:     $RepoRoot\prod\logs\docker_stack_watchdog.log"
Write-Host ""
Write-Host "Verify:"
Write-Host "  Get-ScheduledTask -TaskName '$TaskName' | Format-List"
Write-Host "  Get-ScheduledTaskInfo -TaskName '$TaskName'"
