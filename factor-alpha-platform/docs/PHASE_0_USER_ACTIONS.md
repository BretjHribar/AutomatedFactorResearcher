# Phase 0 — actions you need to do (15 minutes total)

The code-side Phase 0 changes are deployed (the `[ ]` items in
`docs/RELIABILITY_PLAN.md` §7). These three actions need elevated PowerShell
on your machine and can't be done from the dev session.

## 1. Install the Docker stack watchdog (~2 min)

This auto-recovers Docker + the user_code container when Windows Update or
Hyper-V churn knocks them out (the root cause of today's missed equity trade).

From elevated PowerShell:

```powershell
cd C:\Users\breth\PycharmProjects\AutomatedFactorResearcher\factor-alpha-platform
PowerShell -ExecutionPolicy Bypass -NoProfile -File .\tools\install_docker_stack_watchdog_task.ps1
```

Verify:
```powershell
Get-ScheduledTaskInfo -TaskName "Docker Stack Watchdog"
```
Should see `LastTaskResult = 0` after the first 5-min tick, log at
`prod\logs\docker_stack_watchdog.log` if any recovery action fired.

## 2. Disable Windows Update auto-install (~1 min)

The root cause of today's failure: Windows Update ran 3 times during the
trading day (9:57 AM, 10:07 AM, 12:55 PM) and each install restarted Hyper-V,
killing the Docker daemon link. **Active hours don't help** — they only
prevent the post-install REBOOT, not the install itself.

From elevated PowerShell:

```powershell
# Disable automatic Windows Update installs (you can still manually check + install)
New-Item -Path "HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU" -Force | Out-Null
Set-ItemProperty -Path "HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU" -Name "NoAutoUpdate" -Value 1 -Type DWord

# Verify
Get-ItemProperty -Path "HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU" -Name "NoAutoUpdate"
```

You'll need to **manually install updates on a weekend or overnight** going
forward. Set a calendar reminder for the 2nd Saturday of each month
(post-Patch-Tuesday).

To re-enable auto-update later:
```powershell
Set-ItemProperty -Path "HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU" -Name "NoAutoUpdate" -Value 0 -Type DWord
```

## 3. Restart WSL2 once to pick up the RAM cap (~30 sec)

I already wrote `C:\Users\breth\.wslconfig` with `memory=12GB` and
`autoMemoryReclaim=gradual`. WSL won't apply the new config until WSL is
shut down once:

From PowerShell (admin OR regular both work):
```powershell
wsl --shutdown
# Wait 8 seconds for WSL to fully stop:
Start-Sleep -Seconds 8
# Docker Desktop will auto-restart its WSL VM with the new config.
# Optionally restart Docker Desktop UI from the tray for clean re-init.
```

This caps the WSL2 VM's RAM at 12GB so AIPT research workloads on the host
can't starve it. Defensive even if AIPT isn't running today — keeps headroom
for everything Docker Desktop wants.

## 4. (Optional) Discord webhook for paging (~3 min)

The critical-alert pager Dagster schedule fires every minute and POSTs to the
webhook in `DISCORD_ALERT_WEBHOOK_URL`. Without a URL it's a no-op, but with
one you'll get a phone notification within ~60 seconds of any critical alert.

1. In any Discord server: pick a channel (e.g., `#prod-alerts`), click the
   gear → **Integrations** → **Webhooks** → **New Webhook** → name it
   `factor-alpha-pager` → copy the URL.

2. Add the URL to `deploy\dagster\.env` (replacing the empty placeholder):
   ```
   DISCORD_ALERT_WEBHOOK_URL='https://discord.com/api/webhooks/XXXXX/YYYYY'
   ```

3. Re-bounce user_code so it picks up the env var:
   ```powershell
   docker compose --env-file deploy\dagster\.env -f deploy\dagster\docker-compose.yml up -d --force-recreate dagster_user_code dagster_daemon
   ```

The next critical alert (or just inject a test row into `data\prod_state.db`)
will be posted to your channel within a minute.

## What's already deployed (no action needed)

- `submit_deadline_et` moved from 15:45 → 15:47 ET in `prod\config\strategy.json`.
- New Dagster asset `equity_execution_env_sentinel` runs as part of
  `equity_integrity_job` (15:15 ET preflight). Raises a critical paging alert
  if `ALLOW_IB_PAPER_ORDERS` is not "1". Today's miss would have surfaced at
  15:15 ET instead of after the 15:38 MOC failed silently.
- Market-hours guard on `equity_eod_data_refresh_result` — refuses FMP fetches
  during open session when cache is already current. Prevents the 5/13
  intraday-leak incident from recurring.
- Critical-alert pager schedule (`critical_alert_pager_every_minute`,
  `* * * * *` UTC). No-op until `DISCORD_ALERT_WEBHOOK_URL` is set.
- `~/.wslconfig` written with `memory=12GB`.

## Status verification (run once after the above)

```powershell
# All four containers running with the right env?
docker ps --format "{{.Names}}: {{.Status}}"
docker exec factor_alpha_dagster_user_code env | findstr "ALLOW_IB IB_HOST FMP_API"

# Sentinel passes?
docker exec factor_alpha_dagster_user_code python -c "
import sqlite3
db = sqlite3.connect('/opt/dagster/app/data/prod_state.db')
for r in db.execute('SELECT status, message FROM data_integrity_checks WHERE check_name=\"equity_execution_env_sentinel\" ORDER BY checked_at DESC LIMIT 1'):
    print(r)
"

# Schedules all RUNNING?
docker exec factor_alpha_dagster_webserver dagster schedule list -w /opt/dagster/dagster_home/workspace.yaml -l factor_alpha_platform | findstr "Schedule:"
```

Expected after Phase 0: tomorrow's trading day should fire equity MOC cleanly
even if Windows Update misbehaves. The watchdog catches container/env loss
within 5 min; the env-sentinel catches it at 15:15 ET preflight and pages
Discord; both will fire before 15:38 ET MOC.
