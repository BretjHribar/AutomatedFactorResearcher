"""First-touch bootstrap for the Hetzner box.

Handles the awkward Hetzner-Ubuntu first-login flow where the email-provided
root password is marked expired and SSH forces a password change before
shell access. After the change, installs our SSH key + a strong replacement
password, then runs an idempotent system bootstrap (Docker, swap, UFW,
unattended security upgrades).

Reads:
  .secrets/hetzner_root_pw.txt   -- the NEW root password we want to set
  .secrets/hetzner_ed25519.pub   -- our public key to install for future logins

Args:
  --host 5.78.223.66
  --initial-pw '<from-the-hetzner-email>'

Idempotent: safe to re-run.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import paramiko


ROOT = Path(__file__).resolve().parent.parent
PW_PATH = ROOT / ".secrets" / "hetzner_root_pw.txt"
KEY_PATH = ROOT / ".secrets" / "hetzner_ed25519"
PUB_PATH = ROOT / ".secrets" / "hetzner_ed25519.pub"


def _log(msg: str) -> None:
    print(f"[bootstrap] {msg}", flush=True)


def _read_secret(p: Path) -> str:
    return p.read_text(encoding="utf-8").strip()


def _shell_read_until(chan: paramiko.Channel, patterns: list[str], timeout: float = 30.0) -> str:
    """Read from the channel until one of the regex patterns matches OR timeout."""
    buf = ""
    deadline = time.time() + timeout
    compiled = [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in patterns]
    while time.time() < deadline:
        if chan.recv_ready():
            data = chan.recv(8192)
            if not data:
                break
            buf += data.decode("utf-8", errors="replace")
            for c in compiled:
                if c.search(buf):
                    return buf
        elif chan.exit_status_ready():
            break
        else:
            time.sleep(0.2)
    return buf


def _try_password_login(host: str, password: str, timeout: float = 20.0) -> paramiko.SSHClient | str:
    """Try to log in with password. Returns connected client OR a string describing failure mode."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            host,
            username="root",
            password=password,
            timeout=timeout,
            allow_agent=False,
            look_for_keys=False,
        )
        return client
    except paramiko.AuthenticationException as e:
        return f"auth_failed: {e}"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def _drive_forced_password_change(host: str, current_pw: str, new_pw: str) -> None:
    """When sshd forces a password change before shell, paramiko's connect()
    raises AuthenticationException (with the 'New password' prompt embedded
    in the message) because there's no interactive channel. We work around
    this by using a Transport directly with auth_password() returning a
    challenge handler that drives the chpassword flow."""
    _log("attempting first-login forced password change via Transport...")
    sock = paramiko.Transport((host, 22))
    sock.connect()

    # paramiko's keyboard-interactive auth lets us answer prompts. Some sshd
    # configurations fire passwd via PAM during keyboard-interactive, others
    # via the post-auth shell. Try keyboard-interactive first.
    answered = {"count": 0}

    def handler(title, instructions, prompt_list):
        # prompt_list is list[(prompt_str, echo_bool)]
        responses = []
        for prompt, _echo in prompt_list:
            p = prompt.lower()
            answered["count"] += 1
            _log(f"  prompt: {prompt!r}")
            if "current" in p or "old" in p or answered["count"] == 1:
                responses.append(current_pw)
            elif "new" in p or "retype" in p or "again" in p:
                responses.append(new_pw)
            else:
                # First-ever prompt is usually "Password:" -> current.
                responses.append(current_pw)
        return responses

    try:
        sock.auth_interactive("root", handler)
        if sock.is_authenticated():
            _log("authenticated via keyboard-interactive; password may or may not have been rotated -- will verify")
            sock.close()
            return
    except paramiko.BadAuthenticationType:
        pass
    except paramiko.SSHException as e:
        _log(f"keyboard-interactive returned: {e}")
    finally:
        try:
            sock.close()
        except Exception:
            pass


def _connect_with_key(host: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    pk = paramiko.Ed25519Key.from_private_key_file(str(KEY_PATH))
    client.connect(
        host,
        username="root",
        pkey=pk,
        timeout=20,
        allow_agent=False,
        look_for_keys=False,
    )
    return client


def _exec(client: paramiko.SSHClient, cmd: str, *, sudo_pw: str | None = None, timeout: int = 120) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout, get_pty=False)
    if sudo_pw:
        stdin.write(sudo_pw + "\n")
        stdin.flush()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    rc = stdout.channel.recv_exit_status()
    return rc, out, err


def _drive_expired_password_shell(host: str, current_pw: str, new_pw: str, pub_key: str) -> bool:
    """Login forces a password change before any shell. Open an interactive
    channel WITH a PTY, drive the passwd prompts, then run setup commands
    in the same shell while we have it.
    """
    _log("opening interactive PTY channel to drive forced password change")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username="root", password=current_pw, timeout=20,
                   allow_agent=False, look_for_keys=False)
    chan = client.invoke_shell(term="xterm", width=200, height=50)
    try:
        # Wait for the first prompt. Could be "Current password:" or directly
        # "New password:" depending on PAM config. Hetzner's Ubuntu jumps
        # straight to "New password:" because SSH auth already proved the
        # current pw.
        out = _shell_read_until(chan, [r"current.*password", r"new\s*(unix\s*)?password", r"\$\s*$", r"#\s*$"], timeout=20)
        _log(f"  initial shell output (tail):\n{out[-400:]}")
        low = out.lower()

        sent_current = False
        if "current" in low and "password" in low:
            _log("  sending CURRENT password")
            chan.send(current_pw + "\n")
            sent_current = True
            out_next = _shell_read_until(chan, [r"new\s*(unix\s*)?password", r"failure"], timeout=15)
            _log(f"  after current pw:\n{out_next[-300:]}")
            low_next = out_next.lower()
        else:
            low_next = low

        if "new" in low_next and "password" in low_next:
            _log("  sending NEW password (first)")
            chan.send(new_pw + "\n")
            out_retype = _shell_read_until(
                chan,
                [r"retype|again|reenter|re-enter|new password",
                 r"too short|too simple|too long|manipulation|failure|bad password"],
                timeout=15,
            )
            _log(f"  after new pw:\n{out_retype[-300:]}")
            low_retype = out_retype.lower()
            if any(w in low_retype for w in ("too short", "too simple", "manipulation", "failure", "bad password")):
                _log("  PAM rejected new password; aborting")
                return False
            _log("  sending NEW password (retype)")
            chan.send(new_pw + "\n")
            out_done = _shell_read_until(
                chan,
                [r"successfully updated", r"updated successfully",
                 r"\$\s*$", r"#\s*$", r"connection closed", r"connection to .* closed"],
                timeout=20,
            )
            _log(f"  after retype:\n{out_done[-400:]}")
        else:
            _log(f"  unexpected initial state; did not see 'New password' prompt")
            _log(f"  full buffer:\n{out}")
            return False

        # Connection often closes after password change on Hetzner.
        time.sleep(3)
    finally:
        try:
            chan.close()
        except Exception:
            pass
        client.close()

    # Reconnect with the NEW password to install key + finalize
    _log("reconnecting with new password...")
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(host, username="root", password=new_pw, timeout=20,
                       allow_agent=False, look_for_keys=False)
    except Exception as e:
        _log(f"reconnect with new pw failed: {e}")
        return False
    try:
        _log("installing SSH public key + chage -d 99999 (no expiry)")
        safe_pub = pub_key.strip().replace("'", "")
        cmds = [
            "mkdir -p /root/.ssh && chmod 700 /root/.ssh",
            f"touch /root/.ssh/authorized_keys && grep -qxF '{safe_pub}' /root/.ssh/authorized_keys || echo '{safe_pub}' >> /root/.ssh/authorized_keys",
            "chmod 600 /root/.ssh/authorized_keys",
            "chage -d 99999 root || true",
        ]
        for c in cmds:
            rc, out, err = _exec(client, c)
            if rc != 0:
                _log(f"  WARN cmd rc={rc}: {c}\n  stderr: {err.strip()[:200]}")
    finally:
        client.close()
    return True


def install_key_and_rotate_pw(host: str, current_pw: str, new_pw: str, pub_key: str) -> bool:
    """Connect with password, install SSH key, rotate password to new value.

    Three scenarios:
      A. Password works AND not expired -> exec_command flow.
      B. Password works in connect() but exec_command fails because PAM
         says password is expired -> use PTY shell to drive the chpasswd
         prompt, then reconnect to install key.
      C. Connect itself fails -> log + return False.
    """
    result = _try_password_login(host, current_pw)
    if isinstance(result, str):
        _log(f"password login result: {result}")
        return False

    client = result
    _log("password login succeeded -- probing whether exec_command works (i.e., whether pw is expired)")
    rc, out, err = _exec(client, "echo OK", timeout=10)
    if rc == 0 and "OK" in out:
        _log("exec_command works; password not expired. Standard flow.")
        try:
            safe_pub = pub_key.strip().replace("'", "")
            _exec(client, "mkdir -p /root/.ssh && chmod 700 /root/.ssh")
            _exec(client, f"touch /root/.ssh/authorized_keys && grep -qxF '{safe_pub}' /root/.ssh/authorized_keys || echo '{safe_pub}' >> /root/.ssh/authorized_keys")
            _exec(client, "chmod 600 /root/.ssh/authorized_keys")
            import base64
            b64 = base64.b64encode(f"root:{new_pw}".encode()).decode()
            rc, out, err = _exec(client, f"echo '{b64}' | base64 -d | chpasswd && chage -d 99999 root")
            _log(f"  chpasswd rc={rc} err={err.strip()[:200]}")
        finally:
            client.close()
    else:
        _log(f"exec_command failed (rc={rc}): {err.strip()[:300]}")
        _log("falling back to PTY-driven forced password change")
        client.close()
        if not _drive_expired_password_shell(host, current_pw, new_pw, pub_key):
            return False

    # Now connect with the new key to confirm
    try:
        kc = _connect_with_key(host)
        rc, out, _ = _exec(kc, "whoami && hostname && uptime")
        _log(f"key-auth verified:\n{out.strip()}")
        kc.close()
        return True
    except Exception as e:
        _log(f"KEY AUTH FAILED after install: {type(e).__name__}: {e}")
        return False


BOOTSTRAP_SCRIPT = r"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "=== [1/7] apt update + essentials ==="
apt-get update -qq
apt-get install -y -qq \
    ca-certificates curl gnupg lsb-release \
    git rsync htop jq sqlite3 tmux nano \
    unattended-upgrades fail2ban ufw

echo "=== [2/7] Docker (official repo) ==="
if ! command -v docker >/dev/null; then
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list
    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    systemctl enable --now docker
fi
docker --version
docker compose version

echo "=== [3/7] 4 GB swap (safety net for tight RAM) ==="
if ! swapon --show | grep -q /swapfile; then
    fallocate -l 4G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
echo 'vm.swappiness=10' > /etc/sysctl.d/99-swappiness.conf
sysctl -p /etc/sysctl.d/99-swappiness.conf >/dev/null
free -h | head -3

echo "=== [4/7] UFW firewall ==="
ufw --force reset >/dev/null
ufw default deny incoming >/dev/null
ufw default allow outgoing >/dev/null
ufw allow 22/tcp comment 'SSH' >/dev/null
ufw allow 3000/tcp comment 'Dagster UI' >/dev/null
ufw --force enable >/dev/null
ufw status | head -10

echo "=== [5/7] unattended security-only upgrades ==="
cat > /etc/apt/apt.conf.d/50unattended-upgrades.local <<'EOF'
Unattended-Upgrade::Allowed-Origins {
    "${distro_id}:${distro_codename}-security";
    "${distro_id}ESMApps:${distro_codename}-apps-security";
};
Unattended-Upgrade::Automatic-Reboot "false";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
EOF
systemctl enable --now unattended-upgrades

echo "=== [6/7] fail2ban for SSH ==="
systemctl enable --now fail2ban
fail2ban-client status sshd 2>/dev/null | head -5 || true

echo "=== [7/7] directory layout ==="
mkdir -p /opt
chown root:root /opt
mkdir -p /opt/factor-alpha-platform/data/fmp_cache
mkdir -p /opt/factor-alpha-platform/data/kucoin_cache
mkdir -p /opt/factor-alpha-platform/prod/logs
mkdir -p /opt/factor-alpha-platform/prod/stats/output
mkdir -p /opt/factor-alpha-platform/prod/state
mkdir -p /opt/factor-alpha-platform/deploy/dagster

echo ""
echo "============================================================"
echo "BOOTSTRAP COMPLETE"
echo "============================================================"
echo "  Hostname:     $(hostname)"
echo "  Public IP:    $(curl -s -4 ifconfig.me 2>/dev/null || echo unknown)"
echo "  Docker:       $(docker --version)"
echo "  Compose:      $(docker compose version | head -1)"
echo "  RAM:          $(free -h | awk '/^Mem:/ {print $2}')"
echo "  Swap:         $(free -h | awk '/^Swap:/ {print $2}')"
echo "  Disk free:    $(df -h / | awk 'NR==2 {print $4}')"
echo "  UFW:          $(ufw status | head -1)"
echo "  Time:         $(date -u)"
echo "============================================================"
"""


def run_bootstrap(host: str) -> bool:
    client = _connect_with_key(host)
    try:
        _log("running bootstrap script on remote (~3 min)...")
        # Use exec_command with a single bash invocation
        cmd = f"bash -lc {shell_quote(BOOTSTRAP_SCRIPT)}"
        stdin, stdout, stderr = client.exec_command(cmd, timeout=900, get_pty=True)
        # Stream output so user sees progress
        for line in iter(stdout.readline, ""):
            sys.stdout.write(f"[remote] {line}")
            sys.stdout.flush()
        rc = stdout.channel.recv_exit_status()
        err = stderr.read().decode("utf-8", errors="replace")
        if err.strip():
            _log(f"stderr (tail):\n{err[-2000:]}")
        _log(f"bootstrap exit rc={rc}")
        return rc == 0
    finally:
        client.close()


def shell_quote(s: str) -> str:
    """Single-quote shell-safe."""
    return "'" + s.replace("'", "'\"'\"'") + "'"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--initial-pw", required=True, help="Initial root password from Hetzner email")
    ap.add_argument("--skip-key-install", action="store_true",
                    help="Skip password+key install (use when key is already on the box)")
    args = ap.parse_args()

    new_pw = _read_secret(PW_PATH)
    pub_key = _read_secret(PUB_PATH)

    if not args.skip_key_install:
        ok = install_key_and_rotate_pw(args.host, args.initial_pw, new_pw, pub_key)
        if not ok:
            _log("FAILED to install key + rotate password")
            return 1
    else:
        _log("--skip-key-install: assuming SSH key already in place")
        try:
            kc = _connect_with_key(args.host)
            rc, out, _ = _exec(kc, "whoami && hostname")
            _log(f"key-auth OK:\n{out.strip()}")
            kc.close()
        except Exception as e:
            _log(f"key-auth pre-check FAILED: {e}")
            return 1

    if not run_bootstrap(args.host):
        _log("bootstrap FAILED")
        return 1

    _log("DONE — box is ready for repo clone + .env build + docker compose up")
    return 0


if __name__ == "__main__":
    sys.exit(main())
