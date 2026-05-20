"""Watchdog that keeps the Dagster Docker stack alive on this Windows host.

Why this exists
---------------
On 2026-05-15, three Windows Update cycles during the trading day (9:57 AM,
10:07 AM, 12:55 PM CDT) restarted Hyper-V, which killed Docker Desktop's
daemon link AND bounced the user_code container WITHOUT its --env-file. The
container came back alive but with no IB env vars, so both equity MOC fires
returned `status=blocked`. No equity trade today.

Polled actions
--------------
Every run:
  1. Verify `docker info` returns 0. If not, start Docker Desktop and wait.
  2. Verify all four containers (postgres, user_code, webserver, daemon) are
     Up. If any is missing OR has its IB env vars empty, do `docker compose up
     -d --force-recreate <service>` with the SAME --env-file path the
     human-operated stack uses. This guarantees env reload.
  3. If any of step (1) or (2) fired a recovery, log the action.

This script uses ONLY stdlib + the Docker CLI on PATH (no docker SDK).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = ROOT / "prod/logs/docker_stack_watchdog.log"
DEFAULT_STATE = ROOT / "prod/state/docker_stack_watchdog.json"

COMPOSE_FILE = ROOT / "deploy/dagster/docker-compose.yml"
ENV_FILE = ROOT / "deploy/dagster/.env"

DEFAULT_DOCKER_EXE = Path(r"C:\Program Files\Docker\Docker\Docker Desktop.exe")
REQUIRED_CONTAINERS = (
    "factor_alpha_dagster_postgres",
    "factor_alpha_dagster_user_code",
    "factor_alpha_dagster_webserver",
    "factor_alpha_dagster_daemon",
)

# user_code containers MUST have all these env vars set for paper trading to
# work. If any is missing/empty, the container needs a re-create with
# --env-file. See F11 in docs/RELIABILITY_PLAN.md.
REQUIRED_USER_CODE_ENV = (
    "ALLOW_IB_PAPER_ORDERS",
    "IB_HOST",
    "IB_PORT",
    "FMP_API_KEY",
)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _log(log_path: Path, msg: str) -> None:
    line = f"{_now_iso()}  {msg}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _read_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def _run(cmd: list[str], timeout: int = 60) -> tuple[int, str, str]:
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except Exception as exc:
        return 125, "", f"{type(exc).__name__}: {exc}"


def docker_alive() -> bool:
    rc, _, _ = _run(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=10)
    return rc == 0


def start_docker_desktop(exe: Path, log_path: Path) -> bool:
    if not exe.exists():
        _log(log_path, f"ERROR  Docker Desktop exe not found at {exe}")
        return False
    try:
        creationflags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | NEW_PROCESS_GROUP
        subprocess.Popen(
            [str(exe)],
            cwd=str(exe.parent),
            creationflags=creationflags,
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _log(log_path, f"LAUNCH  spawned Docker Desktop ({exe})")
        return True
    except Exception as exc:
        _log(log_path, f"ERROR  failed to spawn Docker Desktop: {type(exc).__name__}: {exc}")
        return False


def wait_for_docker(log_path: Path, max_sec: int = 120) -> bool:
    """Poll docker info up to max_sec; return True when alive."""
    deadline = time.time() + max_sec
    while time.time() < deadline:
        if docker_alive():
            return True
        time.sleep(5)
    _log(log_path, f"ERROR  Docker did not come back within {max_sec}s")
    return False


def container_running(name: str) -> bool:
    rc, out, _ = _run(["docker", "inspect", "-f", "{{.State.Running}}", name], timeout=10)
    return rc == 0 and out.strip().lower() == "true"


def user_code_env_missing(name: str = "factor_alpha_dagster_user_code") -> list[str]:
    """Return list of required env var names that are missing or empty inside
    the user_code container. Empty list = all good."""
    missing: list[str] = []
    for key in REQUIRED_USER_CODE_ENV:
        rc, out, _ = _run(["docker", "exec", name, "sh", "-c", f"printenv {key}"], timeout=10)
        if rc != 0 or not out.strip():
            missing.append(key)
    return missing


def compose_recreate(service: str, log_path: Path) -> bool:
    """`docker compose --env-file ... up -d --force-recreate <service>`.
    The --env-file is the load-bearing piece: without it, the recreated
    container loses every IB / FMP env var (root cause of 2026-05-15)."""
    cmd = [
        "docker", "compose",
        "--env-file", str(ENV_FILE),
        "-f", str(COMPOSE_FILE),
        "up", "-d", "--force-recreate", service,
    ]
    _log(log_path, f"RECREATE  {service}: {' '.join(cmd)}")
    rc, _, err = _run(cmd, timeout=180)
    if rc != 0:
        _log(log_path, f"ERROR  recreate {service} rc={rc} stderr={err[-400:]}")
        return False
    return True


def check_and_recover(*, log_path: Path, state_path: Path, docker_exe: Path) -> dict:
    state = _read_state(state_path)
    actions: list[str] = []
    now_ts = time.time()

    # 1. Docker daemon up?
    if not docker_alive():
        actions.append("docker_down")
        _log(log_path, "DOWN   docker daemon unreachable; starting Docker Desktop")
        start_docker_desktop(docker_exe, log_path)
        if not wait_for_docker(log_path):
            state["last_check_ts"] = now_ts
            state["last_status"] = "docker_down_after_relaunch"
            state["last_actions"] = actions
            _write_state(state_path, state)
            return {"ts": _now_iso(), "status": "docker_down_after_relaunch", "actions": actions}
        _log(log_path, "RECOVER  Docker is back")

    # 2. All four containers running?
    missing_containers = [c for c in REQUIRED_CONTAINERS if not container_running(c)]
    if missing_containers:
        actions.append(f"containers_down:{','.join(missing_containers)}")
        _log(log_path, f"DOWN   containers not running: {missing_containers}")
        # Bring the whole stack up; this is idempotent.
        cmd = [
            "docker", "compose",
            "--env-file", str(ENV_FILE),
            "-f", str(COMPOSE_FILE),
            "up", "-d",
        ]
        rc, _, err = _run(cmd, timeout=180)
        if rc != 0:
            _log(log_path, f"ERROR  compose up failed rc={rc} {err[-300:]}")

    # 3. user_code has all required env vars?
    if container_running("factor_alpha_dagster_user_code"):
        missing_env = user_code_env_missing()
        if missing_env:
            actions.append(f"user_code_env_missing:{','.join(missing_env)}")
            _log(log_path, f"DOWN   user_code missing env: {missing_env}; recreating with --env-file")
            compose_recreate("dagster_user_code", log_path)
            # Quick re-check
            time.sleep(8)
            still_missing = user_code_env_missing()
            if still_missing:
                _log(log_path, f"ERROR  user_code env still missing after recreate: {still_missing}")
                actions.append(f"user_code_env_still_missing:{','.join(still_missing)}")

    status = "healthy" if not actions else "recovered" if "docker_down" in actions[0] else "partial_recovery"
    state.update({
        "last_check_ts": now_ts,
        "last_check_iso": _now_iso(),
        "last_status": status,
        "last_actions": actions,
    })
    _write_state(state_path, state)
    return {"ts": _now_iso(), "status": status, "actions": actions}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", default=str(DEFAULT_LOG))
    p.add_argument("--state", default=str(DEFAULT_STATE))
    p.add_argument("--docker-exe", default=str(DEFAULT_DOCKER_EXE))
    p.add_argument("--once", action="store_true", help="Single check + exit.")
    p.add_argument("--poll-sec", type=int, default=300,
                   help="Polling interval when running continuously.")
    args = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    log_path = Path(args.log)
    state_path = Path(args.state)
    docker_exe = Path(args.docker_exe)

    if args.once:
        report = check_and_recover(
            log_path=log_path, state_path=state_path, docker_exe=docker_exe,
        )
        print(json.dumps(report, indent=2))
        return 0

    _log(log_path, f"START  docker stack watchdog poll every {args.poll_sec}s")
    try:
        while True:
            check_and_recover(
                log_path=log_path, state_path=state_path, docker_exe=docker_exe,
            )
            time.sleep(args.poll_sec)
    except KeyboardInterrupt:
        _log(log_path, "STOP   docker stack watchdog interrupted")
        return 0


if __name__ == "__main__":
    sys.exit(main())
