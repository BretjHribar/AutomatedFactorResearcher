# Cloud Migration Plan — Hetzner Linux + CI/CD

**Drafted:** 2026-05-19
**Goal:** Move production paper trading off the Windows dev box onto a managed
Linux cloud host with full CI/CD. Get to 99% scheduled-trade success rate
(2 nines). Pre-requisite for going to real money on the equity book.

This document supersedes the Phase 1+ sections of `docs/RELIABILITY_PLAN.md`.

---

## TL;DR

- **Target host:** 1× Hetzner CCX23 in US-East (Ashburn). $30/month.
- **IB Gateway path:** `gnzsnz/ib-gateway:stable` container, **not** a standalone
  IBC install on Windows. The image bundles IBC + IB Gateway + auto-restart.
  This is what we attempted on 2026-05-13 and it only failed because of a
  Windows-Docker-Desktop volume-permissions bug. **That bug does not exist
  on Linux native dockerd.**
- **Code portability:** the codebase is already Linux-compatible. The one
  Windows-specific path in `src/execution/recorders.py:_kill_process_tree`
  already has a POSIX branch. Hardcoded `C:\...` paths only appear in
  per-run log artifacts, not in source.
- **Cutover effort:** 3 phases, ~3 weeks elapsed. Most of that is
  shadow-running for a week to confirm parity with the Windows trader.
- **CI/CD:** GitHub Actions → ghcr.io image registry → SSH-deploy to Hetzner.
  Two image artifacts (user_code + dagster), tagged by git SHA, immutable.

---

## 1. Why on a Linux container, not IBC standalone on the current Windows box

You asked specifically about IBC. Quick answer first: **don't install IBC on
the Windows host. IBC is already inside `gnzsnz/ib-gateway`.** Installing it
standalone on Windows would solve auto-login on the current box but doesn't
fix the deeper problems (Docker Desktop instability, Windows Update Hyper-V
churn, no CI/CD, single-host risk).

Comparison:

| Approach | Auto-login | Solves Docker Desktop crashes | Solves Hyper-V churn | Auditable deploys | Cost |
|---|---|---|---|---|---|
| Standalone IBC on this Windows box | yes | **no** | **no** | **no** | $0 |
| `gnzsnz/ib-gateway` on Hetzner Linux | yes (built-in) | n/a (Linux dockerd doesn't have this class of bug) | n/a | yes (CI/CD) | $30/mo |

The container path is strictly more durable for the same end-state behavior.
IBC is still doing the work — just inside the container, configured
declaratively from `TWS_USERID_PAPER` + `TWS_PASSWORD_PAPER` env vars, with
auto-restart and 2FA-timeout policies built in.

---

## 2. What stays the same, what changes

### Code (no change needed)

- All Python source code under `src/`, `prod/`, `experiments/`.
- `src/execution/recorders.py:_kill_process_tree` already branches POSIX vs
  Windows on `sys.platform`. POSIX path uses SIGTERM/SIGKILL — runs as-is.
- Dagster asset graph, jobs, schedules — all already designed for Linux
  containers; the Windows host was just hosting Linux containers via WSL2.
- Trader scripts (`prod/moc_trader.py`, `prod/kucoin_trader.py`) — pure
  Python, `ib_insync`/`requests`/`pandas`. No OS-specific calls.
- The Phase 0 hardening from 2026-05-15: env-var sentinel, market-hours
  guard on EOD refresh, deep gRPC healthcheck for user_code, critical-alert
  Discord pager — all Linux-compatible.

### Config (small surgical changes)

- `deploy/dagster/.env` — adjust paths (`FACTOR_ALPHA_DATA_DIR` becomes
  `/opt/factor-alpha-platform/data`), add TWS_USERID_PAPER / TWS_PASSWORD_PAPER,
  switch `IB_HOST` to `ib-gateway` and `IB_PORT` to `4004` (containerized
  paper API port per the existing override file).
- `prod/config/strategy.json` — no change needed; the trader reads
  `IB_PORT` env which the runtime sets.

### Infra that goes away

- `tools/ib_gateway_watchdog.py` (Windows-only, killed by gnzsnz/ib-gateway
  container's IBC + restart policy)
- `tools/docker_stack_watchdog.py` (Windows-only, replaced by systemd unit
  that restarts the compose stack — though `restart: unless-stopped` in
  docker-compose.yml already covers most of it)
- `tools/install_*_task.ps1` scripts (PowerShell, Windows-only)
- The local `.wslconfig`, Docker Desktop, Windows Update fight

### Infra that's new

- Hetzner cloud account + 1 server
- DNS (optional, can use Hetzner IP) — or Tailscale for SSH
- `ghcr.io` image hosting (free for public, low cost for private)
- GitHub Actions workflows for CI + CD
- `deploy/systemd/` directory with units for the compose stack
- A small shell script `tools/linux_health_probe.sh` for the existing
  Dagster integrity assertions on the host (replaces what the Windows
  watchdogs did)

---

## 3. Architecture diagram

```
        ┌───────────────────────────────────────────────────┐
        │              GitHub repo (this one)               │
        │     - main branch (auto-deploy to UAT)            │
        │     - feature branches (CI only)                  │
        └───────────────────────┬───────────────────────────┘
                                │ push / PR
                ┌───────────────▼────────────────┐
                │       GitHub Actions           │
                │  CI:  ruff + pytest + dagster  │
                │       validate + build image   │
                │  CD:  ssh deploy on main push  │
                └───┬────────────────────────┬───┘
                    │ push                   │ ssh + secrets
                    │                        │
        ┌───────────▼────────┐    ┌──────────▼────────────┐
        │     ghcr.io        │    │   Hetzner CCX23 box   │
        │  user_code:sha-X   │◄───┤  docker compose pull  │
        │  dagster:sha-X     │    │  up -d                │
        └────────────────────┘    │                        │
                                  │  Containers running:   │
                                  │   - postgres           │
                                  │   - dagster_user_code  │
                                  │   - dagster_webserver  │
                                  │   - dagster_daemon     │
                                  │   - ib-gateway (IBC)   │
                                  │                        │
                                  │  Bind mounts:          │
                                  │   - /opt/.../data      │
                                  │   - /opt/.../logs      │
                                  │                        │
                                  │  Exposed:              │
                                  │   - :3000 Dagster UI   │
                                  │   - :22 SSH (firewall) │
                                  └──────────┬─────────────┘
                                             │
                                             ▼
                                  ┌─────────────────────┐
                                  │  IBKR via IB API    │
                                  │  + KuCoin REST      │
                                  │  + FMP REST         │
                                  └─────────────────────┘
```

---

## 4. The Hetzner box — specs and setup

### Sizing rationale

| Workload | RAM | CPU | Disk |
|---|---|---|---|
| Postgres (Dagster runs DB) | ~1 GB | <0.5 | ~1 GB |
| user_code (definitions.py loaded, asset graph) | ~600 MB | low | 0 |
| dagster_webserver + daemon | ~500 MB | low | 0 |
| ib-gateway (Java + IBC) | ~1.5–2 GB | low except at startup | ~0.5 GB |
| KuCoin trader subprocess (peak when refreshing 554 symbols) | ~1.5 GB | 1 vCPU bursting | 0 |
| Equity trader subprocess (peak when computing 45 alphas × 200 names) | ~1 GB | 1 vCPU bursting | 0 |
| EOD refresh subprocess (FMP fan-out 5 workers) | ~1 GB | 1 vCPU sustained | network bound |
| FMP cache (active) | n/a | n/a | ~15 GB |
| KuCoin cache | n/a | n/a | ~4 GB |
| Logs + state + headroom | n/a | n/a | ~5 GB |
| **Totals at peak concurrency** | **~8 GB** | **2–4 vCPU** | **~25 GB** |

**Hetzner CCX23 (dedicated AMD EPYC):** 4 vCPU, 16 GB RAM, 80 GB NVMe SSD,
20 TB egress — **$30/month**.

Comfortable 2× headroom on RAM. Lots of disk room for growth (binance
cache if reactivated, additional research outputs, etc.).

### Provisioning checklist

```bash
# On hetzner cloud console:
#   - Create project: "factor-alpha"
#   - SSH key: upload your public key
#   - Create server:
#       Location: Ashburn (US-East), Type: CCX23
#       Image: Ubuntu 24.04 LTS
#       Networking: IPv4 enabled, firewall: "factor-alpha-fw"
#       SSH key: your uploaded key
#       Hostname: factor-alpha-prod-1

# Firewall rules (factor-alpha-fw):
#   - Inbound: 22 (SSH) from your IP only -- or use Tailscale and close 22
#   - Inbound: 3000 (Dagster UI) from your IP only -- or close and SSH-tunnel
#   - Outbound: everything (default)
```

After SSH'ing in:

```bash
# Install Docker + Compose plugin
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER  # need to logout/login

# Install other essentials
sudo apt update && sudo apt install -y \
    git rsync htop jq sqlite3 ca-certificates curl \
    unattended-upgrades fail2ban

# Enable unattended security upgrades (NOT full upgrades -- only security)
sudo dpkg-reconfigure -plow unattended-upgrades

# fail2ban for SSH
sudo systemctl enable --now fail2ban

# Tailscale (optional but recommended -- closes SSH to public internet)
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
# Then close port 22 in Hetzner firewall, use `tailscale ssh root@<box>` from laptop

# Clone the repo
sudo mkdir -p /opt
sudo chown $USER:$USER /opt
cd /opt
git clone https://github.com/<you>/factor-alpha-platform.git
cd factor-alpha-platform
```

---

## 5. IB Gateway via `gnzsnz/ib-gateway` (the IBC question, answered)

The existing `deploy/dagster/docker-compose.ib-gateway.yml` already wires this
up. On Linux it works without modification — the volume permissions issue
that hit us on Windows is a Docker-Desktop-Windows quirk.

The relevant env vars (set in `deploy/dagster/.env` on the Hetzner box):

```bash
TRADING_MODE=paper
TWS_USERID_PAPER=jogxzg790
TWS_PASSWORD_PAPER=<value from root .env>

IB_GATEWAY_CONTAINER_API_PORT=4004    # paper API exposed inside Docker net
IB_HOST=ib-gateway                    # service hostname, not host.docker.internal
IB_PORT=4004
ALLOW_IB_PAPER_ORDERS=1

# IBC behavior tuning (already wired in the override compose):
TWOFA_TIMEOUT_ACTION=restart      # if 2FA times out, restart instead of dying
TWOFA_EXIT_INTERVAL=180           # how many seconds to wait for 2FA tap
RELOGIN_AFTER_TWOFA_TIMEOUT=yes
EXISTING_SESSION_DETECTED_ACTION=primary  # take over an existing session
BYPASS_WARNING=yes
ALLOW_BLIND_TRADING=yes
AUTO_RESTART_TIME='11:45 PM'      # gateway self-restarts daily
TIME_ZONE=America/New_York
```

What happens at runtime:

1. Compose starts ib-gateway container. Container entrypoint launches IBC,
   IBC launches IB Gateway, IB Gateway shows login dialog.
2. IBC sees the dialog, reads TWS_USERID_PAPER + TWS_PASSWORD_PAPER from
   the environment, types them in, clicks "Paper Log In".
3. IB Gateway connects to IBKR's paper servers, opens API port 4004 inside
   the Docker network.
4. Containers in the dagster_network can reach `ib-gateway:4004`.
5. At 11:45 PM ET each night the gateway auto-restarts (IBKR's
   recommended hygiene), IBC re-logs it in immediately.
6. If IBKR ever prompts 2FA (paper rarely does), IBC waits 180s for the
   tap, then restarts and re-logs in.

**No standalone IBC install needed.** IBC is already inside the image,
configured by environment variables, with no GUI for you to click.

### IBKR paper account session conflict

IBKR allows only one active session per username. If the Windows host is
still running its paper trader while the Hetzner box also tries to log in
with `jogxzg790`, one will get kicked.

**Solution:** request a second paper account from IBKR (free) for the
Hetzner box. Run the two in parallel for a week (Phase 2), confirm parity,
then either keep both or shut down the Windows trader.

To request: IBKR Account Management → Settings → Paper Trading → "Reset"
or "Create new paper account". Returns a new `DU...` account ID and new
username.

---

## 6. CI/CD

### Image registry

Use `ghcr.io/<owner>/factor-alpha-user-code` and `.../factor-alpha-dagster`.
- Free for public repos
- For private: $0 for 500 MB storage, $0 for read traffic; pay-as-you-go
  beyond. Both our images are ~600 MB each, ~10 build/day max for active
  development = well under any tier you'd notice.

### Workflows

Two GitHub Actions workflows:

**`.github/workflows/ci.yml`** — runs on every PR + every push to main:

```yaml
name: CI
on:
  pull_request:
  push:
    branches: [main]

jobs:
  static:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }
      - run: pip install -e ".[dev,orchestration,connectors,optimization,monitoring]"
      - run: python -m ruff check .
      - run: python -m py_compile definitions.py src/execution/recorders.py src/orchestration/dagster_defs.py prod/live_bar.py prod/moc_trader.py
      - run: dagster definitions validate -f definitions.py
      - run: python -m pytest tests/unit tests/test_dagster_definitions.py -q

  build:
    needs: static
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Build + push user_code
        uses: docker/build-push-action@v6
        with:
          context: .
          file: deploy/dagster/Dockerfile_user_code
          push: true
          tags: |
            ghcr.io/${{ github.repository_owner }}/factor-alpha-user-code:sha-${{ github.sha }}
            ghcr.io/${{ github.repository_owner }}/factor-alpha-user-code:${{ github.ref_name == 'main' && 'uat' || format('branch-{0}', github.ref_name) }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
      - name: Build + push dagster
        uses: docker/build-push-action@v6
        with:
          context: .
          file: deploy/dagster/Dockerfile_dagster
          push: true
          tags: |
            ghcr.io/${{ github.repository_owner }}/factor-alpha-dagster:sha-${{ github.sha }}
            ghcr.io/${{ github.repository_owner }}/factor-alpha-dagster:${{ github.ref_name == 'main' && 'uat' || format('branch-{0}', github.ref_name) }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
      - name: Smoke-test the user_code image
        run: |
          docker run --rm ghcr.io/${{ github.repository_owner }}/factor-alpha-user-code:sha-${{ github.sha }} \
            dagster definitions validate -f definitions.py
```

**`.github/workflows/deploy.yml`** — runs on push to main (auto-deploy UAT) +
manual trigger (promote to prod):

```yaml
name: Deploy
on:
  push:
    branches: [main]
  workflow_dispatch:
    inputs:
      target:
        description: 'uat or prod'
        required: true
        default: 'uat'
        type: choice
        options: [uat, prod]
      image_sha:
        description: 'Git SHA to deploy (must already be built by CI)'
        required: true

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: ${{ github.event_name == 'push' && 'uat' || inputs.target }}
    steps:
      - uses: actions/checkout@v4
      - name: Set SHA
        id: sha
        run: |
          if [ -n "${{ inputs.image_sha }}" ]; then
            echo "value=${{ inputs.image_sha }}" >> $GITHUB_OUTPUT
          else
            echo "value=${{ github.sha }}" >> $GITHUB_OUTPUT
          fi
      - name: Deploy via SSH
        uses: appleboy/ssh-action@v1
        with:
          host: ${{ secrets.HETZNER_HOST }}
          username: ${{ secrets.HETZNER_USER }}
          key: ${{ secrets.HETZNER_SSH_KEY }}
          script: |
            cd /opt/factor-alpha-platform
            git fetch --all
            git checkout ${{ steps.sha.outputs.value }}
            export FACTOR_ALPHA_USER_CODE_IMAGE=ghcr.io/${{ github.repository_owner }}/factor-alpha-user-code:sha-${{ steps.sha.outputs.value }}
            export FACTOR_ALPHA_DAGSTER_IMAGE=ghcr.io/${{ github.repository_owner }}/factor-alpha-dagster:sha-${{ steps.sha.outputs.value }}
            docker compose --env-file deploy/dagster/.env \
              -f deploy/dagster/docker-compose.yml \
              -f deploy/dagster/docker-compose.ib-gateway.yml \
              pull
            docker compose --env-file deploy/dagster/.env \
              -f deploy/dagster/docker-compose.yml \
              -f deploy/dagster/docker-compose.ib-gateway.yml \
              up -d
            # Wait for user_code health
            for i in {1..30}; do
              s=$(docker inspect --format '{{.State.Health.Status}}' factor_alpha_dagster_user_code)
              [ "$s" = "healthy" ] && break
              sleep 5
            done
      - name: Post-deploy smoke test
        uses: appleboy/ssh-action@v1
        with:
          host: ${{ secrets.HETZNER_HOST }}
          username: ${{ secrets.HETZNER_USER }}
          key: ${{ secrets.HETZNER_SSH_KEY }}
          script: |
            cd /opt/factor-alpha-platform
            docker exec factor_alpha_dagster_user_code python -c "
            import sqlite3
            db = sqlite3.connect('/opt/dagster/app/data/prod_state.db')
            for r in db.execute('SELECT status,message FROM data_integrity_checks ORDER BY checked_at DESC LIMIT 5'):
                print(r)
            "
```

### GitHub Actions secrets to create

| Secret | Value |
|---|---|
| `HETZNER_HOST` | The box's public IP or Tailscale name |
| `HETZNER_USER` | `root` or whoever has docker access |
| `HETZNER_SSH_KEY` | Private SSH key matching the public key on the box |

The `.env` itself stays on the Hetzner box (not in git, not in Actions
secrets). Use Actions secrets only for SSH access and any tokens.

### Rollback

```bash
# On the box, in /opt/factor-alpha-platform
git log --oneline -10                 # find the previous good SHA
git checkout <previous-sha>
export FACTOR_ALPHA_USER_CODE_IMAGE=ghcr.io/.../factor-alpha-user-code:sha-<previous>
export FACTOR_ALPHA_DAGSTER_IMAGE=ghcr.io/.../factor-alpha-dagster:sha-<previous>
docker compose --env-file deploy/dagster/.env \
  -f deploy/dagster/docker-compose.yml \
  -f deploy/dagster/docker-compose.ib-gateway.yml \
  up -d
```

A one-liner `tools/rollback.sh` would wrap this. Add it after first
successful deploy.

---

## 7. Data migration

### One-shot transfer (initial seed)

From the Windows host (via Git Bash or PowerShell with rsync):

```bash
# From the Windows host:
rsync -avz --progress \
  /c/Users/breth/PycharmProjects/AutomatedFactorResearcher/factor-alpha-platform/data/ \
  root@<hetzner-ip>:/opt/factor-alpha-platform/data/
```

About 34 GB total; ~15-30 min depending on bandwidth. Initial state files:

- `data/fmp_cache/` (~15 GB) — equity matrices
- `data/kucoin_cache/` (~4 GB) — crypto matrices
- `data/alpha_results.db` — alpha library
- `data/prod_state.db` — copy this LAST and only if you want to preserve
  alert + check history; OK to start fresh too

You can also skip `binance_cache` (~15 GB) since Binance is disabled.

### Steady-state

After cutover, the Hetzner box runs `equity_eod_refresh_*` and
`kucoin_paper_execution` schedules on its own; data updates in place. No
ongoing sync between the two hosts.

---

## 8. Migration phases

### Phase 1 — provision + shadow (week 1)

Goal: Hetzner box running the full stack in parallel with Windows; both
trading paper independently; daily P&L diff goes to your inbox.

- [ ] **Day 1 (you):** Create Hetzner account, provision CCX23 in Ashburn,
      copy your SSH public key, install Docker via the script above.
- [ ] **Day 1 (you):** Request a second IBKR paper account for the box.
      This avoids the one-session-per-username conflict with the Windows
      paper trader.
- [ ] **Day 1 (me):** Write `.env.uat.template` with the new paper account
      placeholders; help you populate it on the box (over Tailscale SSH).
- [ ] **Day 2 (me):** First manual deploy: clone repo on box, `rsync` data,
      `docker compose up`. Verify all 5 containers + ib-gateway come up
      healthy. Verify Dagster UI at `:3000`.
- [ ] **Day 3-7:** Both trading systems run in parallel. Daily I compare
      the prod_state.db on Windows vs the new one on Hetzner; signal
      computations should be byte-identical (same alphas, same source data
      after the rsync; trade execution will diverge because they're
      different paper accounts).

### Phase 2 — CI/CD live (week 2)

Goal: every push to main builds + auto-deploys to the Hetzner box.

- [ ] Write `.github/workflows/ci.yml` per §6.2.
- [ ] Write `.github/workflows/deploy.yml` per §6.2.
- [ ] Create `ghcr.io` PAT (or use the workflow's auto-injected GITHUB_TOKEN
      with `packages: write` scope; preferred).
- [ ] Add the 3 secrets to GitHub: `HETZNER_HOST`, `HETZNER_USER`,
      `HETZNER_SSH_KEY`.
- [ ] First end-to-end test: a no-op PR → CI passes → merge → auto-deploys
      to box → post-deploy smoke test green.
- [ ] Add `tools/rollback.sh` for one-line revert by SHA.

### Phase 3 — cutover (week 3)

Goal: Hetzner box becomes primary; Windows becomes research-only.

- [ ] **Day 1:** Confirm 5 trading days of clean parallel runs in Phase 1.
- [ ] **Day 2:** Re-point the IBKR account in the Hetzner `.env` from the
      shadow paper account to the original `jogxzg790`.
- [ ] **Day 2:** Stop the Windows Dagster stack (`docker compose down`)
      and disable the Windows scheduled tasks (`Unregister-ScheduledTask`).
      Windows host now has no production responsibility.
- [ ] **Day 2:** Trigger a manual MOC fire on the Hetzner box (or wait for
      15:38 ET). Confirm orders submit, fills land, dashboard updates.
- [ ] **Day 3-5:** Monitor. Day 5 success = cutover declared complete.
- [ ] **Day 5:** Remove `tools/ib_gateway_watchdog.py`, `tools/docker_stack_watchdog.py`,
      `tools/install_*_task.ps1`, the `.wslconfig` file, and any other
      Windows-only artifacts. Update `docs/RELIABILITY_PLAN.md` to mark
      Phase 0 items as superseded.

### Phase 4 (optional, month 2) — second box for UAT separation

When you want a real UAT/PROD gate:

- [ ] Provision a second CCX23 ($30/month = $60 total).
- [ ] CI deploys to UAT box; UAT must pass a 24-hour clean run before
      promotion to PROD.
- [ ] PROD deploys gated on manual `workflow_dispatch` button click.

Skip Phase 4 until Phase 3 has been clean for a month. One box is enough
for now.

---

## 9. Codebase changes I'll make pre-migration

A small PR before Phase 1 starts, deployable to the existing Windows stack
without behavior change but ready for Linux:

1. **`deploy/dagster/.env.example`** — add the 3 new ib-gateway env vars
   (`TWS_USERID_PAPER`, `TWS_PASSWORD_PAPER`, `IB_GATEWAY_CONTAINER_API_PORT`)
   with placeholders + comment explaining each.

2. **`deploy/dagster/docker-compose.yml`** — add `restart: unless-stopped`
   to every service (it's there now for some, missing for others). Already
   verified each container has it; this is a confirmation step.

3. **`docs/CICD_DEPLOYMENT_RUNBOOK.md`** — update with the concrete workflow
   files from §6.2 (currently it's a skeleton). Mark Phase 4 cloud UAT as
   the canonical path.

4. **`tools/rollback.sh`** (new) — one-line image-SHA rollback for the box.

5. **`tests/test_dagster_definitions.py`** — make sure this exists and
   passes; CI gates on it.

6. **`prod/kucoin_trader.py`, `prod/moc_trader.py`** — fix the
   `datetime.utcnow()` deprecation warnings spotted in logs. Cosmetic
   but should land before CI starts flagging them.

7. **`.dockerignore`** — add `data/`, `prod/logs/`, `venv/`, `.dagster_home/`,
   `__pycache__/`, `experiments/results/` so they don't bloat the Docker
   build context.

8. **`docs/CLOUD_MIGRATION_PLAN.md`** — this document; merged at the same time.

---

## 10. Costs

| Item | Monthly | Annual |
|---|---|---|
| Hetzner CCX23 (single box) | $30 | $360 |
| Hetzner CCX23 × 2 (prod + UAT, optional Phase 4) | $60 | $720 |
| ghcr.io image storage | $0 (under free tier for any reasonable cadence) | $0 |
| GitHub Actions minutes | $0 (2000/mo free for private; we'd use <100/mo) | $0 |
| Tailscale (free personal tier) | $0 | $0 |
| BetterUptime (free tier, 10 monitors, 3-min checks) | $0 | $0 |
| Discord webhook | $0 | $0 |
| **Total (Phase 3)** | **$30** | **$360** |
| **Total (Phase 4)** | **$60** | **$720** |

Compared to one missed equity trade day (5/13, 5/15, 5/18 all happened
in the last 6 trading days), this pays for itself in week one.

---

## 11. Risks and what we do about them

| Risk | Mitigation |
|---|---|
| IBKR rejects the auto-login because they updated their login form | gnzsnz/ib-gateway is on a rolling release with active maintenance. Pin to a specific image tag (e.g., `stable-v10.45`) to avoid surprise updates. Also: their image gets a maintenance release within a day or two when IB makes UI changes. |
| 2FA gets enabled on the paper account by IBKR | Container's `TWOFA_TIMEOUT_ACTION=restart` keeps retrying; you tap the phone notification once when it fires. For paper, this is rare. For real-money live, treat 2FA as a manual-approval gate. |
| Hetzner box dies | Hourly Hetzner snapshots ($0.012/GB/month = ~$1/mo for 80GB), can rebuild on a new IP in 15 min. Restore script: `tools/restore_from_snapshot.sh`. |
| ghcr.io down during a deploy | `docker compose pull` retries; CI doesn't gate on it. If you push during an outage, the deploy just delays. Older images stay available for rollback. |
| Bad code merged to main | UAT gate (Phase 4) catches it. Until Phase 4, the rollback is one SSH command + `docker compose up` with the previous image SHA — typically <2 min. |
| Network partition between box and IBKR | IB Gateway will auto-reconnect; orders pending submission will retry. We've seen <30s blips that the trader handles transparently. Anything longer is identical to today's Windows behavior. |
| You lose your laptop's SSH key | Hetzner web console has a rescue/reset mechanism; ~10 min to get back in. Backup the SSH key to your password manager. |

---

## 12. What I need from you to start

Three small decisions, then I'll start Phase 1 work:

1. **Hetzner account** — sign up at console.hetzner.com, send me the
   project name once it's created. (Or I can write a Terraform script
   if you want infra-as-code; honestly overkill for one box.)
2. **Second paper account** — request from IBKR; takes ~10 min in their
   web portal. New username + password go into a 1Password / Bitwarden
   entry I can read when configuring the Hetzner `.env`.
3. **Public GitHub repo or private?** Free private ghcr.io storage is
   generous either way; just affects who can browse the source. Recommend
   private for paper trading code.

Once those are in place, I'll do the pre-migration PR (§9) on Monday and
spin up the Hetzner box mid-week. End-state: cleanly running on Linux with
CI/CD by ~3 weeks from start.

---

## 13. What this does NOT solve (deliberately scoped out)

- **Live trading.** This plan keeps you on paper. Going live needs:
  position-reconciliation pass after each fill, a kill-switch, account
  allow-list, separate Dagster job with explicit manual approval.
  Address after 2 weeks of clean Phase 3.
- **Real-money disaster recovery.** A multi-region setup or warm-standby
  cluster — not needed for paper, definitely needed for real money. Punt.
- **Production-grade observability (metrics, traces, log aggregation).**
  The existing Dagster UI + ops_dashboard + Discord pager covers what
  matters. Prometheus/Grafana can come later if/when needed.
- **Research workload migration.** AIPT / alpha-research stays on the
  Windows box. The Hetzner box runs prod only. Hard separation between
  research and prod is itself a reliability feature.
