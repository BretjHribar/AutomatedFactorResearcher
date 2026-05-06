param(
    [switch]$SkipWslFeatures,
    [switch]$NoRestart
)

$ErrorActionPreference = "Stop"

function Test-IsAdmin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-IsAdmin)) {
    throw "Run this script from an elevated PowerShell session."
}

$logDir = Join-Path $PSScriptRoot "..\logs\install"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logPath = Join-Path $logDir ("docker_desktop_install_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))
Start-Transcript -Path $logPath -Append | Out-Null

try {
    Write-Host "Installing Docker Desktop prerequisites..."

    if (-not $SkipWslFeatures) {
        Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -NoRestart -All
        Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -NoRestart -All
    }

    $dockerProgramData = "C:\ProgramData\DockerDesktop"
    if (Test-Path -LiteralPath $dockerProgramData) {
        $children = @(Get-ChildItem -LiteralPath $dockerProgramData -Force -ErrorAction SilentlyContinue)
        if ($children.Count -eq 0) {
            Write-Host "Removing empty failed-install directory: $dockerProgramData"
            Remove-Item -LiteralPath $dockerProgramData -Force
        }
        else {
            Write-Host "Repairing ownership on existing Docker Desktop data directory: $dockerProgramData"
            & icacls.exe $dockerProgramData /setowner "BUILTIN\Administrators" /T /C
        }
    }

    Write-Host "Installing Docker Desktop with winget..."
    winget install -e --id Docker.DockerDesktop --accept-package-agreements --accept-source-agreements --silent

    $dockerBin = "C:\Program Files\Docker\Docker\resources\bin"
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($userPath -notlike "*$dockerBin*") {
        Write-Host "Adding Docker CLI to the user PATH..."
        $newPath = if ([string]::IsNullOrWhiteSpace($userPath)) { $dockerBin } else { "$userPath;$dockerBin" }
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    }

    $dockerDesktop = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (Test-Path -LiteralPath $dockerDesktop) {
        Write-Host "Starting Docker Desktop..."
        Start-Process -FilePath $dockerDesktop
    }

    Write-Host "Docker install command completed. A reboot may be required before Docker starts cleanly."
    Write-Host "After reboot/startup, verify with: docker version; docker compose version"
}
finally {
    Stop-Transcript | Out-Null
    Write-Host "Install log: $logPath"
}
