# CI/CD and Docker Deployment Runbook

This runbook describes the intended path from local development to UAT paper trading and PROD trading for the factor alpha platform.

## Goals

- Build the same deployable artifacts for research, data checks, paper execution, and live execution.
- Keep UAT and PROD behavior identical except for secrets, account IDs, schedule enablement, and execution guardrails.
- Make every running strategy traceable to a git commit, Docker image digest, config hash, and Dagster run ID.
- Never rebuild code during promotion. CI builds immutable images once; UAT/PROD pull a pinned tag or digest.
- Keep broker and exchange credentials out of images, logs, git, and CI artifacts.

## Deployment Artifacts

The platform should publish two primary images:

- `factor-alpha-user-code`
  - Contains `definitions.py`, `src/`, `prod/`, runtime package dependencies, and Dagster code-server entrypoint.
  - Owns Dagster assets, checks, jobs, schedules, execution wrappers, data validation, and trading scripts.

- `factor-alpha-dagster`
  - Contains Dagster webserver/daemon dependencies and `deploy/dagster/dagster.yaml` plus `workspace.yaml`.
  - Runs the Dagster UI, scheduler daemon, run coordinator, and Docker run launcher.

- `ib-gateway`
  - Cloud-only sidecar service from the `deploy/dagster/docker-compose.ib-gateway.yml` override.
  - Runs IB Gateway plus IBC automation in its own container and exposes API only to the Docker network plus optional host-local ports.

Do not bake local market data, `prod/logs`, `.env`, `.dagster_home`, broker credentials, API keys, or IB Gateway credentials into the app images.

## Image Tags

Publish every image with all of these tags:

- `sha-<git_sha>`: immutable commit tag, required for UAT/PROD.
- `branch-<branch_name>`: convenient dev/UAT trace tag.
- `uat`: only after a UAT deploy promotion.
- `prod`: only after an approved PROD promotion.

Recommended registry naming:

```text
ghcr.io/<owner>/factor-alpha-user-code:sha-<git_sha>
ghcr.io/<owner>/factor-alpha-dagster:sha-<git_sha>
```

PROD should deploy by SHA or digest, not by `latest`.

## CI Pipeline

Run on every pull request and every push to `main`.

1. Checkout and install Python 3.12.
2. Install the package with dev/runtime extras:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,orchestration,connectors,optimization,monitoring]"
```

3. Run static and structural gates:

```bash
python -m ruff check .
python -m py_compile definitions.py src/execution/recorders.py src/orchestration/dagster_defs.py prod/live_bar.py prod/moc_trader.py
dagster definitions validate -f definitions.py
```

4. Run unit tests:

```bash
python -m pytest tests/unit tests/test_dagster_definitions.py tests/test_live_bar_vwap.py -q
```

5. Run acceptance tests on scheduled or labeled builds:

```bash
python tools/test_pipeline_acceptance.py
```

6. Build Docker images:

```bash
docker build -f deploy/dagster/Dockerfile_user_code -t factor-alpha-user-code:sha-<git_sha> .
docker build -f deploy/dagster/Dockerfile_dagster -t factor-alpha-dagster:sha-<git_sha> .
```

7. Smoke-test the images:

```bash
docker run --rm factor-alpha-user-code:sha-<git_sha> dagster definitions validate -f definitions.py
docker compose -f deploy/dagster/docker-compose.yml config
docker compose \
  --env-file deploy/dagster/.env.example \
  --env-file deploy/dagster/.env.ib-gateway.example \
  -f deploy/dagster/docker-compose.yml \
  -f deploy/dagster/docker-compose.ib-gateway.yml \
  config
```

8. Push images only after all gates pass.

The GitHub Actions implementation lives at:

```text
.github/workflows/factor-alpha-ci-cd.yml
```

On Windows dev boxes, Docker Desktop requires administrator elevation. Use the repo helper from an elevated PowerShell session:

```powershell
.\tools\install_docker_desktop.ps1
```

## UAT Deployment Procedure

UAT is paper trading only. It should run continuously before PROD is enabled.

1. Pull the selected SHA images.
2. Create `deploy/dagster/.env` from `deploy/dagster/.env.example` and `deploy/dagster/.env.ib-gateway.example`, then update it for paper trading. When the cloud box runs its own containerized Gateway, use `4004`, not `4002`, because `4004` is the paper API port exposed inside the Docker network:

```text
FMP_API_KEY=<uat_secret>
TRADING_MODE=paper
IB_GATEWAY_CONTAINER_API_PORT=4004
IB_HOST=ib-gateway
IB_PORT=4004
ENABLE_IB_LIVE_VWAP=0
ALLOW_IB_PAPER_ORDERS=1
FORCE_IB_PAPER_MOC=0
TWS_USERID=<paper_or_ib_user>
TWS_PASSWORD=<paper_or_ib_password>
```

3. Start or update services. Local Windows-Gateway development uses only `docker-compose.yml`; cloud UAT uses both compose files:

```bash
docker compose \
  --env-file deploy/dagster/.env \
  -f deploy/dagster/docker-compose.yml \
  -f deploy/dagster/docker-compose.ib-gateway.yml \
  pull

docker compose \
  --env-file deploy/dagster/.env \
  -f deploy/dagster/docker-compose.yml \
  -f deploy/dagster/docker-compose.ib-gateway.yml \
  up -d
```

4. Verify:

```bash
docker compose --env-file deploy/dagster/.env -f deploy/dagster/docker-compose.yml -f deploy/dagster/docker-compose.ib-gateway.yml ps
docker compose --env-file deploy/dagster/.env -f deploy/dagster/docker-compose.yml -f deploy/dagster/docker-compose.ib-gateway.yml logs --tail=100 ib-gateway
docker compose --env-file deploy/dagster/.env -f deploy/dagster/docker-compose.yml -f deploy/dagster/docker-compose.ib-gateway.yml logs --tail=100 dagster_daemon
```

5. In Dagster, enable UAT schedules:
   - Integrity checks.
   - Equity EOD refresh checks.
   - Live quote collector.
   - Equity research signal.
   - Crypto research signal.
   - Crypto paper recorder.
   - IB paper MOC execution.

6. Required UAT evidence before PROD promotion:
   - Passing Dagster asset checks for equity and crypto data integrity.
   - Passing `ib_gateway_connectivity_status` and dashboard `ib_gateway_connectivity` check.
   - At least one successful equity paper MOC run.
   - At least one successful KuCoin and Binance paper recorder run.
   - Dashboard state includes current execution summaries.
   - Decision dataset config hash and git SHA match the deployed image.

## Equity EOD Refresh Schedule

The equity production cache is refreshed by Dagster, not Windows Task Scheduler.
The refresh starts 30 minutes after the NYSE close and repeats hourly so late FMP
bars, vendor fills, and revisions are detected before the next MOC trading day.

Enabled schedules:

```text
equity_eod_refresh_hourly_after_close_et      30 16-23 * * 1-5  America/New_York
equity_eod_refresh_hourly_overnight_catchup_et 30 0-8 * * 2-6   America/New_York
```

Operational contract:

- At 16:30 ET and every hour afterward, fetch the recent FMP EOD overlap window.
- If the target NYSE bar is missing, return `waiting_for_vendor` and try again next hour.
- If any recent bar changed, rebuild only price-derived matrices, repair classification matrices, rebuild MCAP universes, and run active-universe coverage checks.
- Keep the production universe pinned to `data/fmp_cache/metadata.json` tickers. Extra cached price files must not expand the live/research universe.
- Keep `factor_alpha/eod_refresh=equity_eod` tag concurrency at limit 1 so hourly EOD runs cannot write the same parquet cache concurrently.
- Equity integrity, research signal, and IB paper MOC jobs all depend on `equity_eod_data_refresh_result`; the MOC job must halt if the required full historical bar is missing.

Manual local check:

```powershell
venv\Scripts\python.exe -m src.data.equity_refresh --recheck-recent --workers 5 --overlap-days 7 --min-active-coverage 0.99 --fail-if-incomplete
```

Manual Docker/Dagster checks:

```powershell
docker exec factor_alpha_dagster_webserver sh -lc "cd /opt/dagster/dagster_home && dagster schedule list -w workspace.yaml -l factor_alpha_platform"
docker exec factor_alpha_dagster_user_code python -c "from pathlib import Path; import pandas as pd; c=pd.read_parquet('/opt/dagster/app/data/fmp_cache/matrices/close.parquet'); print(c.shape, c.index.min(), c.index.max(), c.index.is_monotonic_increasing)"
```

Recovery if the cache is stale:

1. Verify `FMP_API_KEY` is set in the Dagster environment.
2. Run the manual local EOD refresh command above.
3. Run `run_equity_integrity` for `MCAP_100M_500M`.
4. Rebuild and restart Docker Dagster services.
5. Confirm both EOD schedules are `RUNNING`.
6. Keep the legacy Windows `MOC_Trader_Daily` task disabled so Dagster is the single execution scheduler.

## PROD Deployment Procedure

PROD should promote the exact image SHA already validated in UAT.

1. Freeze schedules in UAT and record the validated image SHA.
2. Create a release tag, for example `prod-YYYYMMDD-<short_sha>`.
3. Pull the same image SHA on PROD.
4. Create or update PROD `deploy/dagster/.env` with live-only secrets and guardrails. The containerized live Gateway API port is `4003` inside the Docker network:

```text
TRADING_MODE=live
IB_GATEWAY_CONTAINER_API_PORT=4003
IB_HOST=ib-gateway
IB_PORT=4003
ENABLE_IB_LIVE_VWAP=0
ALLOW_IB_PAPER_ORDERS=0
FORCE_IB_PAPER_MOC=0
TWS_USERID=<live_ib_user>
TWS_PASSWORD=<live_ib_password>
```

5. Start services and run read-only validation first:

```bash
docker compose --env-file deploy/dagster/.env -f deploy/dagster/docker-compose.yml -f deploy/dagster/docker-compose.ib-gateway.yml up -d
docker compose --env-file deploy/dagster/.env -f deploy/dagster/docker-compose.yml -f deploy/dagster/docker-compose.ib-gateway.yml logs --tail=100 ib-gateway
docker compose --env-file deploy/dagster/.env -f deploy/dagster/docker-compose.yml -f deploy/dagster/docker-compose.ib-gateway.yml logs --tail=100 dagster_daemon
```

6. Enable schedules in phases:
   - Phase 1: data integrity and dashboard state only.
   - Phase 2: research signal generation.
   - Phase 3: paper/shadow execution recording.
   - Phase 4: live execution only after manual approval and broker connectivity checks.

The Docker Gateway override can log into a live account, but live equity order submission should still get its own explicit Dagster asset, environment guard, account allow-list, and manual enablement path before real money is enabled. Live crypto order submission is not currently implemented; crypto should remain recorder/shadow mode until a dedicated live execution adapter, kill switch, and exchange-specific reconciliation checks exist.

## IB Gateway Cloud Service

Use `deploy/dagster/docker-compose.ib-gateway.yml` when the target machine should run IB Gateway itself. It keeps Gateway out of the Dagster images while still making the deployment self-contained on a cloud VM.

Recommended layout:

```text
UAT cloud VM:
  TRADING_MODE=paper
  IB_HOST=ib-gateway
  IB_PORT=4004
  ALLOW_IB_PAPER_ORDERS=1

PROD cloud VM:
  TRADING_MODE=live
  IB_HOST=ib-gateway
  IB_PORT=4003
  ALLOW_IB_PAPER_ORDERS=0
```

Operational notes:

- `ib_gateway_connectivity_status` runs as part of equity preflight and writes `ib_gateway_connectivity` into the dashboard integrity checks.
- The Gateway container has a Docker healthcheck against its configured internal API port.
- Host API/VNC ports are bound to `127.0.0.1` only. Use SSH forwarding for remote debugging, for example `ssh -L 5900:127.0.0.1:5900 <box>`.
- If testing the Gateway container on the Windows dev box while desktop IB Gateway is already listening on `4002`, set `IB_GATEWAY_HOST_PAPER_PORT=14002`; Dagster still connects inside Docker to `IB_PORT=4004`.
- Do not expose the IB API port to the public network. The IB API socket is not an authenticated HTTPS API.
- IBKR two-factor authentication cannot be bypassed by Docker. IBC can automate the Gateway UI and restart/relogin flows, but initial or periodic two-factor approval may still require operator action depending on account settings.
- Prefer `TWS_PASSWORD_FILE`, `TWS_USERID_PAPER`, and `TWS_PASSWORD_PAPER_FILE` with mounted secret files on long-lived cloud hosts.

## Rollback

Rollback must be image-based and fast.

1. Disable execution schedules in Dagster.
2. Set compose image tags back to the previous validated SHA.
3. Restart user-code, daemon, and webserver.
4. Re-run data integrity checks.
5. Re-enable schedules only after checks pass.

Keep at least the last three validated image SHAs and their config hashes in the release notes.

## Secrets and Config

- Store CI registry tokens in GitHub Actions secrets.
- Store UAT/PROD `.env` files on the target box only, never in git.
- Store IB Gateway credentials outside app images. Prefer the separate IB Gateway/IBC compose override or a managed host-level service.
- Log account type, port, client IDs, and order IDs, but never passwords, API secrets, private keys, or session tokens.
- Use distinct UAT and PROD broker/exchange credentials.

## Required Improvements Before Production

- Replace `latest` in `docker-compose.yml` and `dagster.yaml` with environment-driven image tags.
- Add `.dockerignore` to keep data, logs, caches, `.dagster_home`, and virtualenvs out of build contexts.
- Add GitHub Actions workflows with path filters for this subproject.
- Add image vulnerability scanning and dependency auditing.
- Persist Dagster Postgres backups and application logs off-box.
- Add deployment notes that record git SHA, image digest, config hash, enabled schedules, and operator approval.
- Add a dedicated live-equity execution Dagster asset with live-account allow-listing, kill switch, and explicit approval workflow.
