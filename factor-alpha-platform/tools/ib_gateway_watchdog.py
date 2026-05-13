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
DEFAULT_LAUNCH_COOLDOWN_SEC = 180  # Don't relaunch more than once per 3 min.


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
    """Single check + optional launch. Returns a small status dict."""
    state = _read_state(state_path)
    last_launch_iso = state.get("last_launch_iso")
    last_launch_ts = state.get("last_launch_ts", 0.0)
    now_ts = time.time()

    listening = _port_listening(host, port)
    status = {
        "ts": _now_iso(),
        "host": host,
        "port": port,
        "listening": listening,
        "action": "none",
        "last_launch_iso": last_launch_iso,
    }
    if listening:
        # Mark healthy. Don't churn state writes if we've been healthy a while.
        if state.get("last_listening_iso") != _now_iso()[:13]:  # hourly granularity
            state["last_listening_iso"] = _now_iso()
            _write_state(state_path, state)
        return status

    # Not listening. Check cooldown.
    since_last_launch = now_ts - float(last_launch_ts or 0.0)
    if since_last_launch < cooldown_sec:
        status["action"] = "cooldown"
        status["seconds_until_relaunch"] = int(cooldown_sec - since_last_launch)
        _log(log_path, f"DOWN  port {host}:{port} not listening; cooldown {int(since_last_launch)}s/{cooldown_sec}s — not spawning")
        return status

    # Spawn.
    ok = _spawn_gateway(exe, log_path)
    state["last_launch_iso"] = _now_iso()
    state["last_launch_ts"] = now_ts
    state["last_launch_ok"] = bool(ok)
    _write_state(state_path, state)
    status["action"] = "launched" if ok else "launch_failed"
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
