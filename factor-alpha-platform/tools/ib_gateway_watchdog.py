"""Watchdog that keeps the native Windows IB Gateway alive.

Polls the paper-trading API port (default 4002) at a fixed interval. When the
port is not listening, launches `ibgateway.exe`. Logs every action so the user
can see what happened and when.

Pairs with the IBKR "Memorize Password" / "Auto-restart" gateway preferences
to give effective auto-login:
  1. Open IB Gateway once manually, log in, enable:
       - "Memorize Password"  (Configure -> Settings -> Lock and Exit)
       - "Auto-restart" at e.g. 11:45 PM (Configure -> Settings -> Lock and Exit -> Auto restart)
  2. Once set, ibgateway.exe re-launches will auto-login using the cached
     credentials (subject to IBKR session/2FA policy on the account).

This watchdog only LAUNCHES the gateway; the gateway itself does the auth.
Two-factor: if the account ever asks for 2FA, the gateway will show a prompt
and stay un-connected; the watchdog will detect this (port still down) and
will NOT keep spawning new processes — the cooldown prevents that.

Run via Windows Task Scheduler:
  Task name:   "IB Gateway Watchdog"
  Trigger:     At log on  + Repeat every 1 minute indefinitely
  Action:      C:\\Users\\breth\\PycharmProjects\\AutomatedFactorResearcher\\
               factor-alpha-platform\\venv\\Scripts\\python.exe
               C:\\Users\\breth\\PycharmProjects\\AutomatedFactorResearcher\\
               factor-alpha-platform\\tools\\ib_gateway_watchdog.py --once

Or run continuously:
  python tools\\ib_gateway_watchdog.py
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = ROOT / "prod/logs/ib_gateway_watchdog.log"
DEFAULT_STATE = ROOT / "prod/state/ib_gateway_watchdog.json"
DEFAULT_GATEWAY_EXE = Path(r"C:\Jts\ibgateway\1045\ibgateway.exe")
DEFAULT_PORT = 4002
DEFAULT_POLL_SEC = 30
# Base cooldown between launch attempts. Effective cooldown grows
# exponentially with consecutive failed launches (capped at MAX_COOLDOWN_SEC)
# so a gateway stuck in a login-wait state doesn't get a fresh process every
# polling tick. See `_effective_cooldown_sec()` for the exact schedule.
DEFAULT_LAUNCH_COOLDOWN_SEC = 600   # 10 min base
MAX_COOLDOWN_SEC = 3600 * 4         # 4 hour ceiling
MAX_CONSECUTIVE_FAILS_BEFORE_GIVEUP = 6  # ~ after this many consecutive misses, just log
GATEWAY_PROCESS_NAME = "ibgateway.exe"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _log(log_path: Path, msg: str) -> None:
    line = f"{_now_iso()}  {msg}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _port_listening(host: str, port: int, timeout_sec: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True
    except OSError:
        return False


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


def _running_gateway_pids() -> list[int]:
    """Return PIDs of any running ibgateway.exe processes on this host.

    Uses `tasklist /FI "IMAGENAME eq ibgateway.exe" /FO CSV /NH` which is
    available on every Windows install with no extra dependencies. Returns
    an empty list on any failure (defensive: a tasklist hiccup must NOT make
    us spawn a fresh process).
    """
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", f"IMAGENAME eq {GATEWAY_PROCESS_NAME}", "/FO", "CSV", "/NH"],
            timeout=10,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return []
    pids: list[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line or "ibgateway" not in line.lower():
            continue
        # CSV row: "ibgateway.exe","1234","Console","1","132,368 K"
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) < 2:
            continue
        pid_raw = parts[1].strip('"').replace(",", "").strip()
        try:
            pids.append(int(pid_raw))
        except ValueError:
            continue
    return pids


def _kill_pids(pids: list[int], log_path: Path) -> int:
    """Best-effort terminate of the given PIDs via `taskkill /F`.

    Returns the count of successful kills.
    """
    killed = 0
    for pid in pids:
        try:
            r = subprocess.run(
                ["taskkill", "/F", "/PID", str(pid), "/T"],
                capture_output=True,
                timeout=5,
                check=False,
            )
            if r.returncode == 0:
                killed += 1
        except Exception as exc:
            _log(log_path, f"KILL_ERR pid={pid} {type(exc).__name__}: {exc}")
    return killed


def _effective_cooldown_sec(base: int, consecutive_fails: int) -> int:
    """Exponential backoff capped at MAX_COOLDOWN_SEC.

    Schedule (base=600s, 10 min):
      0 fails -> 600s (10 min)
      1 fail  -> 1200s (20 min)
      2 fails -> 2400s (40 min)
      3 fails -> 4800s (80 min)
      4 fails -> 9600s (160 min) -> capped at 4h (14400)
      5+ fails -> 4h
    """
    if consecutive_fails <= 0:
        return base
    return min(base * (2 ** consecutive_fails), MAX_COOLDOWN_SEC)


def _spawn_gateway(exe: Path, log_path: Path) -> bool:
    if not exe.exists():
        _log(log_path, f"ERROR  IB Gateway exe not found at {exe}")
        return False
    try:
        # DETACHED_PROCESS|CREATE_NEW_PROCESS_GROUP so the GUI keeps running
        # after the watchdog exits (when run via --once).
        creationflags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        subprocess.Popen(
            [str(exe)],
            cwd=str(exe.parent),
            creationflags=creationflags,
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _log(log_path, f"LAUNCH  spawned {exe}")
        return True
    except Exception as exc:
        _log(log_path, f"ERROR  failed to spawn IB Gateway: {type(exc).__name__}: {exc}")
        return False


def check_and_recover(
    *,
    host: str,
    port: int,
    exe: Path,
    log_path: Path,
    state_path: Path,
    cooldown_sec: int,
) -> dict:
    """Single check + optional launch. Returns a small status dict.

    Key invariants this enforces:
      1. NEVER spawn a new ibgateway.exe if one is already running unless the
         existing one has been alive for longer than the (backoff-adjusted)
         cooldown without listening on the API port. In that case the existing
         process is stuck (login-wait, 2FA-pending, frozen) and we kill it
         first, THEN spawn one fresh.
      2. Use exponential backoff on consecutive-fail count so a gateway stuck
         on a credential prompt doesn't get a fresh process every 10 minutes.
         After 6+ consecutive failures, just log and wait for human action.
      3. Reset consecutive-fail count to 0 the moment we see the port listen.
    """
    state = _read_state(state_path)
    last_launch_iso = state.get("last_launch_iso")
    last_launch_ts = float(state.get("last_launch_ts") or 0.0)
    consecutive_fails = int(state.get("consecutive_fails") or 0)
    now_ts = time.time()

    listening = _port_listening(host, port)
    existing_pids = _running_gateway_pids()

    status: dict = {
        "ts": _now_iso(),
        "host": host,
        "port": port,
        "listening": listening,
        "existing_gateway_pids": existing_pids,
        "consecutive_fails": consecutive_fails,
        "action": "none",
        "last_launch_iso": last_launch_iso,
    }

    if listening:
        # Healthy. Reset failure counter so the next outage starts at base
        # cooldown rather than at a backed-off value from a stale streak.
        if consecutive_fails > 0:
            state["consecutive_fails"] = 0
            state["last_listening_iso"] = _now_iso()
            _write_state(state_path, state)
            _log(log_path, f"RECOVER port {host}:{port} listening again; cleared {consecutive_fails} consecutive_fails")
        elif state.get("last_listening_iso") != _now_iso()[:13]:  # hourly write
            state["last_listening_iso"] = _now_iso()
            _write_state(state_path, state)
        return status

    # Port NOT listening. Three sub-cases:
    #   (a) No ibgateway.exe processes -> spawn one (subject to cooldown).
    #   (b) ibgateway.exe processes alive but recent launch -> they may
    #       still be starting up; wait for cooldown to elapse before
    #       declaring them stuck.
    #   (c) ibgateway.exe processes alive AND we're past cooldown -> they
    #       are stuck (login-wait / 2FA / frozen). Kill them all, then spawn
    #       one fresh if not in deep-backoff.
    eff_cooldown = _effective_cooldown_sec(cooldown_sec, consecutive_fails)
    since_last_launch = now_ts - last_launch_ts
    in_cooldown = since_last_launch < eff_cooldown

    if existing_pids and in_cooldown:
        # Recently-spawned gateway is still trying to start. Don't spawn again.
        status["action"] = "cooldown_existing_starting"
        status["seconds_until_relaunch"] = int(eff_cooldown - since_last_launch)
        _log(
            log_path,
            f"WAIT   port {host}:{port} not listening; "
            f"{len(existing_pids)} ibgateway.exe alive (pids={existing_pids}); "
            f"cooldown {int(since_last_launch)}s/{eff_cooldown}s; "
            f"consecutive_fails={consecutive_fails}",
        )
        return status

    if consecutive_fails >= MAX_CONSECUTIVE_FAILS_BEFORE_GIVEUP and in_cooldown:
        # Deep-backoff: even if no process exists, don't spawn again until
        # the (very long) cooldown elapses. Operator needs to look at this.
        status["action"] = "giveup_until_cooldown_elapses"
        status["seconds_until_relaunch"] = int(eff_cooldown - since_last_launch)
        _log(
            log_path,
            f"GIVEUP port {host}:{port} not listening after {consecutive_fails} "
            f"consecutive fails; cooldown {int(since_last_launch)}s/{eff_cooldown}s; "
            "operator should inspect IB Gateway (likely needs Memorize Password or 2FA approval)",
        )
        return status

    # Past cooldown. Kill any existing stuck processes before spawning fresh,
    # so we don't accumulate (the 60-instances bug from 2026-05-19).
    if existing_pids:
        killed = _kill_pids(existing_pids, log_path)
        _log(
            log_path,
            f"KILL   {killed}/{len(existing_pids)} stuck ibgateway.exe (pids={existing_pids}) "
            f"-- port {host}:{port} not listening past {eff_cooldown}s cooldown",
        )
        time.sleep(2)  # let Windows release the lock on the EXE

    ok = _spawn_gateway(exe, log_path)
    state["last_launch_iso"] = _now_iso()
    state["last_launch_ts"] = now_ts
    state["last_launch_ok"] = bool(ok)
    state["consecutive_fails"] = consecutive_fails + 1
    _write_state(state_path, state)
    status["action"] = "launched_after_kill" if existing_pids else "launched"
    status["consecutive_fails"] = state["consecutive_fails"]
    status["next_cooldown_sec"] = _effective_cooldown_sec(cooldown_sec, state["consecutive_fails"])
    return status


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--exe", default=str(DEFAULT_GATEWAY_EXE))
    p.add_argument("--log", default=str(DEFAULT_LOG))
    p.add_argument("--state", default=str(DEFAULT_STATE))
    p.add_argument("--poll-sec", type=int, default=DEFAULT_POLL_SEC,
                   help="Polling interval when running continuously (ignored with --once).")
    p.add_argument("--cooldown-sec", type=int, default=DEFAULT_LAUNCH_COOLDOWN_SEC,
                   help="Min seconds between successive launch attempts.")
    p.add_argument("--once", action="store_true",
                   help="Single check + exit (use with Windows Task Scheduler).")
    args = p.parse_args(argv)
    exe = Path(args.exe)
    log_path = Path(args.log)
    state_path = Path(args.state)
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    if args.once:
        status = check_and_recover(
            host=args.host, port=args.port, exe=exe,
            log_path=log_path, state_path=state_path,
            cooldown_sec=args.cooldown_sec,
        )
        print(json.dumps(status, indent=2))
        return 0

    _log(log_path, f"START  watchdog polling {args.host}:{args.port} every {args.poll_sec}s; exe={exe}")
    try:
        while True:
            check_and_recover(
                host=args.host, port=args.port, exe=exe,
                log_path=log_path, state_path=state_path,
                cooldown_sec=args.cooldown_sec,
            )
            time.sleep(args.poll_sec)
    except KeyboardInterrupt:
        _log(log_path, "STOP   watchdog interrupted")
        return 0


if __name__ == "__main__":
    sys.exit(main())
