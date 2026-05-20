# Reliability Plan — getting to 99% (2 nines)

**Drafted:** 2026-05-15
**Target SLO:** the daily MOC trade and the 6× daily KuCoin paper trade fire and submit cleanly **on at least 99% of scheduled days** — i.e. miss no more than ~3.65 days per year. Current run rate is closer to ~70% (alerts on 9 of last 10 days, two complete missed equity trade days in the last 4).

This document audits every distinct failure mode observed in the past 10 days, identifies root causes, and lays out a phased migration to durable infrastructure. The dominant common factor is **Docker Desktop on a Windows dev box hosting production**.

---

## 1. Audit — every failure mode observed 2026-05-05 → 2026-05-15

Reconstructed from `data/prod_state.db.alerts`, `prod/logs/execution/`, `prod/logs/kucoin/runlogs/`, and live debugging notes.

| # | Date | Symptom | Root cause | Time to recover | Trade impact |
|---|---|---|---|---|---|
| F1 | 5/5–5/6 | `crypto_latest_bar_freshness` 271h old | KuCoin universe rebuild script not run; research universe deliberately frozen | manual | none (research path) |
| F2 | 5/6 | `equity latest_bar_freshness` ends 5/4 instead of 5/5 | FMP late publish + missed retry | hourly retry fixed | none |
| F3 | 5/7 | `ib_paper_moc_execution: subprocess failed with returncode 1` | (cause not captured — likely IB Gateway not up) | next day | 1 missed equity rebal |
| F4 | 5/8–5/9 | `ib_gateway_connectivity` alerts | IB Gateway native process died, no watchdog | manual restart | possible missed runs |
| F5 | **5/9 12:08 → 5/14 14:00** | KuCoin schedule silent for **5 days**, 30 missed `kucoin_paper_execution` fires | **Dagster daemon gRPC ghosting** — parent gRPC server stayed "healthy" but child `code-server` died; daemon swallowed `DagsterUserCodeUnreachableError` and skipped every schedule tick. Shallow `grpc-health-check` didn't catch it. | manual gRPC restart + deep healthcheck deployed 5/14 | **~30 KuCoin paper trades lost** |
| F6 | 5/11–5/13 | `ib_gateway_connectivity` reopened multiple times | IB Gateway dropping silently mid-day, no auto-recover | manual | 1 missed equity rebal (5/13) |
| F7 | 5/13 | Docker Desktop didn't auto-start after Windows restart | Docker Desktop service not set to auto-start with elevated rights; user has to launch manually | manual | tight on equity 15:38 ET deadline |
| F8 | 5/13 | Equity rebalance aborted at 15:48:26 ET with `PAST SUBMIT DEADLINE` | Compound of F6 + F7 + my own delay; trader hit hard guard at 15:45 ET | force-fire missed full window | **1 full missed equity day** |
| F9 | 5/13 | After force-fire, all 121 production matrix parquets contaminated with intraday-as-of-close row | I fired `research_signal_job` manually during market hours; the job's asset selection includes `equity_eod_data_refresh_result`, which pulled FMP's live intraday quote and stamped it as 2026-05-13 close | `tools/strip_intraday_row.py` cleaned 125 parquets | no live trade impact, blocked next-day attempt |
| F10 | 5/14 | Found F5 was still in effect (KuCoin silent since 5/9) | Same as F5; not caught until I went looking | manual fix | already counted |
| F11 | 5/15 | Both equity MOC fires today (15:12 + 19:38 UTC) `status=blocked` | user_code container lost its `ALLOW_IB_PAPER_ORDERS` env var — most likely a `docker restart` or auto-recover bounce that didn't re-apply `--env-file deploy/dagster/.env` | manual restart with env-file | **1 full missed equity day** |
| F12 | 5/15 ~20:08–22:00 UTC | Docker Desktop dies again silently. 21:30 UTC EOD refresh missed. | Docker Desktop on Windows instability — third occurrence in 4 days | manual restart | data refresh delayed 1+h |
| F13 | 5/15 | IB Gateway died at ~15:00 ET | Native gateway process drop (no logs as to why) | **WATCHDOG CAUGHT IT** at 10:01 CDT, relaunched | none — first watchdog save |

### Failure mode classes

- **A. Docker Desktop / WSL2 instability** (F7, F11 root cause, F12): the host runtime under everything. 3+ unplanned restarts in 4 days.
- **B. Silent gRPC ghosting** (F5, F10): daemon-↔-user_code link can rot while both containers report healthy. Caused the worst incident (5 days dark). Fixed by deep healthcheck on 5/14.
- **C. Env-var loss on container bounce** (F11): docker compose without `--env-file` produces a running container with no IB credentials/flags. No alert; recorder silently `blocked`s.
- **D. IB Gateway native process drops** (F4, F6, F13): undocumented cause; gateway just exits or wedges. Watchdog as of 5/14 catches at 10-min cadence.
- **E. Operator-initiated cache contamination** (F9): firing the wrong job during market hours pulled intraday data.
- **F. Late or absent FMP data** (F2): vendor-side; partial mitigation via hourly retries.
- **G. Missed deadlines** (F8): combined effect when A+B+D align with market open.

### Why this dataset says we're nowhere near 2 nines

- 10 distinct production failures in 10 days (1.0 fail/day average).
- 2 full missed equity trade days in 10 (F8, F11) → ~20% miss rate, vs. 1% target.
- ~30 missed KuCoin 4h fires in F5 alone → over 60% miss rate in the worst week.
- **The current dev-box-as-prod setup cannot reach 99%.** Every fix we've added has been firefighting on top of a fundamentally unstable substrate.

---

## 2. Why Docker Desktop dies on this host

Three observed Docker Desktop deaths in 4 days. Likely causes, ranked by typical incidence on Windows hosts:

1. **WSL2 / Hyper-V memory pressure** — Docker Desktop runs Linux containers in a WSL2 VM. WSL2 grabs RAM aggressively and gives it back slowly. With AIPT/research workloads spawning multi-GB Python processes on the same host (saw 5.9GB and 854MB processes in our session), WSL2 hits a cliff and the daemon kills itself. **Most likely culprit here.**
2. **Windows Update background interventions** — Windows Update can pause Hyper-V services briefly; if Docker Desktop service's restart policy isn't aggressive enough it stays down.
3. **WSL2 distro disk corruption** — long-running ext4 inside the WSL2 vhdx file occasionally needs `fsck`; Docker Desktop sometimes can't bring the daemon back without one.
4. **Crash in Docker Desktop GUI auto-update** — Desktop pulls auto-updates and restarts the daemon, sometimes failing.
5. **Docker Desktop not configured to auto-start with admin rights** — confirmed cause of F7.

**Why "make sure it never happens again" is hard on Windows:** Docker Desktop is not a production runtime. Its uptime expectations and architecture (license-fenced Hyper-V VM + a GUI app + a service + WSL2 + auto-updates) are inherently more fragile than `dockerd` on Linux. Microsoft and Docker do not target multi-9 uptime for this product. The deepest fix is **don't put production on Docker Desktop.**

What we can do *now* on Windows (incremental hardening, but won't get to 2 nines):
- Set Docker Desktop to "Start when you sign in" + autostart elevated via Task Scheduler trigger
- Cap WSL2 RAM in `%USERPROFILE%\.wslconfig` (e.g., `memory=12GB`) to leave room for AIPT research
- Move AIPT/research workloads off the prod host or run them under a job-object quota so they can't starve Docker
- Auto-restart Docker Desktop via Task Scheduler when `docker info` fails (mirror of the IB Gateway watchdog pattern)
- Configure each Compose service with explicit `restart: unless-stopped` (already done)
- Pin Docker Desktop to a known-good version + disable auto-updates

What truly fixes it: **a Linux host.** See Section 3.

---

## 3. Recommended infrastructure — Linux UAT box, prod on it

### Why move

- `dockerd` on Linux has 4–5 nines uptime in normal use. No GUI to crash, no WSL2 indirection.
- The repo already has `deploy/dagster/docker-compose.ib-gateway.yml` set up for cloud — IB Gateway as a managed container with auto-login. The 5/13 attempt failed on Windows due to a known Docker-Desktop-Windows named-volume permissions bug. **That bug doesn't exist on Linux** (gnzsnz/ib-gateway is built and CI-tested on Linux).
- IBKR API restricts to **one running session per account**. With a cloud box we move the session there; the Windows dev box stays a research environment with no production responsibilities. End of contention.
- CI/CD becomes possible: GitHub Actions → build images → push to a registry → SSH/SSM deploy to the box, all automated.

### Cost comparison

For the workload we need: 2–4 vCPU, 8GB RAM, 40GB SSD, dedicated bandwidth, IBKR-friendly latency (US East), Linux. Containers: postgres + dagster_webserver + dagster_daemon + user_code + ib-gateway + watchdog.

| Provider | Plan | vCPU/RAM/SSD | $/month | Notes |
|---|---|---|---|---|
| **Hetzner Cloud (US-East ASH)** | **CCX23** | **4 dedicated / 16GB / 80GB NVMe** | **~$30** | **Best $/performance. Dedicated AMD vCPU. Recommended.** |
| Hetzner Cloud US | CCX13 | 2 / 8GB / 80GB NVMe | ~$15 | Tight on RAM with AIPT-style workloads; OK for prod only |
| DigitalOcean | Premium Intel 4 vCPU / 8GB | 4 / 8GB / 160GB | ~$48 | Easy, mature |
| Linode | Dedicated 4 vCPU / 8GB | 4 / 8GB / 160GB | ~$48 | Similar |
| Vultr | High Performance 4 vCPU / 8GB | 4 / 8GB / 128GB | ~$48 | Similar |
| AWS | t3.large reserved | 2 / 8GB / 30GB EBS | ~$60 | Most $/$, but flexible egress; reserved instance helps |
| AWS | c7i.large reserved | 2 / 4GB | ~$72 | Too tight on RAM |

**Recommendation: Hetzner CCX23 in US-East (Ashburn), Ubuntu 24.04 LTS.** ~$30/month, dedicated 4 AMD EPYC vCPUs, 16GB RAM, 80GB NVMe. Plenty of headroom for the current workload, room to add Binance live + extra strategies later. Hetzner has datacenter in Virginia which is ~5ms to NYC4 (IBKR's primary US gateway). Mature dashboard, snapshots, hourly billing.

**Latency check**: from Hetzner ASH to IBKR's NYC4 endpoints averages ~5–8ms. IB Gateway is more sensitive to *jitter* than absolute latency; either is fine for MOC orders. Live order routing was already paper-mode and this stays the same.

### Annual cost: ~$360. Hard to argue with that vs. one missed trade day.

---

## 4. CI/CD plan

The repo already has `docs/CICD_DEPLOYMENT_RUNBOOK.md` outlining the intended pipeline. It's never been operationalized. Here's the concrete buildout.

### 4.1 Stages

```
[dev push to feature branch]
        ↓
[ GitHub Actions: factor-alpha-ci.yml ]
   - ruff check
   - py_compile critical files
   - dagster definitions validate
   - pytest tests/unit + critical specs
   - build user_code + dagster images (multi-arch linux/amd64)
   - smoke-run images: dagster definitions validate inside the image
   - push to ghcr.io with sha-<git_sha> and branch-<name> tags
        ↓
[ merge to main ]
   - same as above
   - also tag images :uat
        ↓
[ deploy to UAT cloud box ]
   - SSH or GitHub Actions self-hosted runner on the box
   - `docker compose --env-file ... pull && up -d --no-recreate-non-essentials`
   - run `dagster definitions validate` inside the pulled image as a gate
   - if any integrity check fails post-deploy: rollback to previous image SHA
        ↓
[ smoke tests on UAT ]
   - confirm all schedules show RUNNING
   - confirm `ib_gateway_connectivity_status` passes
   - confirm `latest_bar_freshness` for equity and crypto pass
   - confirm at least one `kucoin_paper_execution` cycle completes after deploy
        ↓
[ promote to prod ]
   - manual approval step in Actions (button click)
   - re-tag the same SHA as :prod
   - same compose path on prod box, same restart pattern
```

### 4.2 Concrete first deliverables

1. `.github/workflows/factor-alpha-ci.yml` — runs on every PR + main push. Replaces the existing skeleton with the real pipeline.
2. `.github/workflows/factor-alpha-deploy-uat.yml` — manual or post-merge trigger; SSHs to the UAT box and `docker compose up -d`.
3. `deploy/dagster/.env.uat` template — committed sans secrets, secrets in GH Actions encrypted secrets.
4. Image registry: GitHub Container Registry (`ghcr.io/<owner>/factor-alpha-user-code`, `…/factor-alpha-dagster`). Free for private repos within free tier; cheap beyond.
5. Pinned base images: `python:3.12-slim-bookworm@sha256:<digest>` and `postgres:16-alpine@sha256:<digest>` so a Docker Hub outage doesn't break builds.

### 4.3 Manual interventions go away

Today: every code change is `vim` on the prod host. Every config change risks the same. There is no audit trail beyond `git log`. With CI/CD:
- Every change PR + merged + image tagged
- Every deploy is `docker compose pull && up -d` against an immutable SHA
- Rollback is `git revert` + redeploy → known-good SHA
- No more "I edited strategy.json on Friday and the trader broke Monday"

---

## 5. Monitoring + alerting upgrades

The existing alert engine writes to SQLite. It correctly detected most of this week's incidents — but **nothing paged**. We learned about the 5-day KuCoin outage because the user happened to look at the dashboard.

Required additions:

1. **Real paging on open critical alerts.** A Dagster sensor that fires whenever an open critical alert is older than 30 min, posts to:
   - Discord webhook (free, simplest), OR
   - PagerDuty (~$25/user/month, on-call rotations, escalation)
   - Email via SES/Mailgun (~free, no on-call escalation)
   - **Recommendation: Discord webhook + a Telegram bot. Both free.**
2. **Liveness watchdog at the Dagster daemon level.** A sensor that checks "did `equity_research_signal_1530_et` have a tick in the last 25 hours on a weekday?" — if not, scream. This catches the F5/F10 gRPC ghosting class even if the deep healthcheck somehow misses it.
3. **Heartbeat ping out of the cloud box** every 5 min to a free uptime service (UptimeRobot, BetterUptime). If the box goes silent, you get a push. This catches host-level failures the box itself can't report.
4. **Cron-job that verifies the schedule expectations** every Mon-Fri at 21:30 UTC: "did equity_eod_refresh fire today? did all 6 KuCoin schedules fire today?". Bayesian-mean stale-detection from the alert history; weekly summary.
5. **Operational dashboard** (already mostly built): add "last 7 days schedule fire matrix" panel — green dot per expected fire, red dot per miss, makes silent-failure obvious at a glance.

---

## 6. Specific hardening for the architecture we keep

The list below is what the audit identified as the gap behind each failure mode; each maps to a concrete change.

| Failure | Fix |
|---|---|
| F1 KuCoin universe stale | move `tools/build_kucoin_universe_20d.py` to a 7-day Dagster schedule, not manual |
| F2 FMP late publish | already mitigated by hourly retry; add a 04:30 UTC final-retry as last gasp |
| F3 IB returncode=1 with no captured detail | enforce stderr_tail capture in `ExecutionResult`; fail the alert with actionable detail |
| F4/F6/F13 IB Gateway native drops | **on cloud box: replace with gnzsnz/ib-gateway container + auto-login from secrets**. Windows watchdog stays as the dev-box fallback. |
| F5/F10 gRPC ghosting | **deep healthcheck shipped 5/14**. Add the daemon-side liveness check (item 2 in §5) as belt + suspenders. |
| F7 Docker Desktop didn't start | **Linux: not applicable**. Windows: Task Scheduler with `RunLevel=Highest` + `RunAtLogon`. |
| F8 missed deadline | move `submit_deadline_et` from 15:45 to 15:47 (still gives 1 min margin vs NYSE MOC 15:50, 8 min vs NASDAQ 15:55). Add a 15:25 ET preflight alert that PAGES if IB Gateway isn't reachable. |
| F9 cache contamination | gate `equity_eod_data_refresh_result` asset with a market-hours check; refuse to fetch FMP if `now < 16:30 ET on a trading day`. Manual fire returns `status=blocked_market_open`. |
| F11 env-var loss | **never** call `docker restart` or `docker recreate` without `--env-file`. Wrap every bounce in a Makefile target or PowerShell helper that always passes the env file. Add a Dagster sensor that reads `ALLOW_IB_PAPER_ORDERS` on start and aborts the run with a paging alert if it's not "1" or "0" (i.e. missing). |
| F12 Docker died | covered by Linux migration |

---

## 7. Phased migration plan

### Phase 0 — this week (no migration yet, immediate hardening)

- [x] Deep gRPC healthcheck on user_code (shipped 5/14)
- [x] IB Gateway watchdog (shipped 5/14)
- [x] Strip-intraday tool for cache cleanup (shipped 5/13)
- [ ] **Market-hours guard on `equity_eod_data_refresh_result`** — refuse FMP fetches during open session. **Critical, do today.**
- [ ] **Env-var sentinel sensor** — Dagster sensor that pages if `ALLOW_IB_PAPER_ORDERS != "1"` 1 hour before MOC. **Critical, do today.**
- [ ] **Move `submit_deadline_et` from 15:45 to 15:47 ET** in `prod/config/strategy.json`.
- [ ] **Configure Docker Desktop autostart with admin rights**; cap WSL2 RAM at 12GB in `~/.wslconfig`.
- [ ] **Discord webhook alerting** for open critical alerts older than 30 min.

### Phase 1 — next 1 week: spin up UAT cloud box

- [ ] Provision Hetzner CCX23 in US-East
- [ ] Install Docker, Tailscale (for secure SSH from your laptop), unattended-upgrades
- [ ] Mount data volumes (with hourly Hetzner snapshots)
- [ ] Run `docker compose -f docker-compose.yml -f docker-compose.ib-gateway.yml up -d` with credentials from the existing root `.env`
- [ ] Validate: IB Gateway auto-logs in, schedules show RUNNING, integrity all pass
- [ ] **Run a week in shadow mode** — UAT box does its own KuCoin paper trading and IB MOC; compare daily P&L vs. the Windows box. Should be identical.

### Phase 2 — next 2 weeks: CI/CD pipeline live

- [ ] Build out `.github/workflows/factor-alpha-ci.yml` per §4.2
- [ ] Push to `ghcr.io`, pull on UAT box
- [ ] Add the deploy workflow with manual UAT promotion
- [ ] After 1 week of clean UAT runs, do a single manual promote to prod tag

### Phase 3 — month 1: cut over

- [ ] Stop the Windows-box Dagster stack (kill containers; leave research env intact)
- [ ] UAT box becomes prod
- [ ] Provision a second Hetzner box for new UAT, same image SHA path. Migration of UAT/PROD lives there from now on.
- [ ] Windows box reverts to a pure research host

### Phase 4 — month 2: add observability

- [ ] Heartbeat ping → BetterUptime (free tier)
- [ ] Weekly schedule-fire matrix report posted to Discord
- [ ] Grafana panel via the existing dashboard JSON (or use Dagster's own UI charts)

---

## 8. Open questions for you

1. **Two boxes or one?** Production-only ($30/mo, no UAT separation) vs. one prod + one UAT ($60/mo, real promotion testing). Recommend two boxes once we're past the migration cutover.
2. **Pager channel?** Discord, Telegram, PagerDuty, plain SMS (Twilio ~$1/mo + $0.0079/SMS)?
3. **Live mode timeline?** This plan keeps you on paper-mode. Going live needs a separate kill-switch + position-reconciliation pass on each fill. Worth scheduling separately once we're at 99% on paper.
4. **GitHub Container Registry or Docker Hub?** ghcr.io is free for private repos under reasonable limits; Docker Hub free tier is more limited but conventional. Recommend ghcr.io.

---

## 9. Expected reliability after each phase

| Phase | Estimated equity-trade success rate | Estimated KuCoin-fire success rate |
|---|---|---|
| Today | ~80% | ~70% (5 days dark in 10) |
| Phase 0 (after market-hours guard + env sentinel + Discord paging) | ~93% | ~95% |
| Phase 1 (Hetzner shadow box) | UAT box at ~98%, Windows stays where it is | both at ~98% |
| Phase 2 (CI/CD active) | ~99% deployable changes | ~99% |
| Phase 3 (cutover, single Linux box owning prod) | **99–99.5%** | **99–99.5%** |

2 nines is reachable. **It is not reachable on Docker Desktop on this Windows host without moving prod off it.** The migration is the single biggest lever.

---

## 10. Today's recommended immediate actions

1. **Apply the Phase 0 hardening NOW** — particularly the market-hours guard on `equity_eod_data_refresh_result` (prevents another F9) and the env-var sentinel (prevents another F11). These are both ~30-minute changes.
2. **Provision the Hetzner box this weekend.** It's $30/month. Migration takes a few days. Even running it as a shadow gives me a parallel signal to confirm everything works before cutover.
3. **Don't fire `research_signal_job` manually during market hours** until the guard is in place.
4. **Stop using the Windows venv's `python.exe` for the watchdog and any other prod-adjacent task** — it's broken (Section unrelated to this plan; see prior debug session). Already fixed in `install_ib_gateway_watchdog_task.ps1`.
