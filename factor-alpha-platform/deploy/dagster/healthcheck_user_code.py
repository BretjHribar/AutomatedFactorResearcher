"""Deep liveness probe for the user_code gRPC server.

The built-in `dagster api grpc-health-check` only pings the parent gRPC server
process (port 4000). The parent can stay alive while the child code-server
process — which actually loads `definitions.py` and serves repository data —
dies or its internal unix socket gets cleaned up. When that happens, the
daemon's repository-list requests fail with:

    DagsterUserCodeUnreachableError: ... unix:/tmp/tmpXXXXX error: No such file
    or directory

This script calls `list_repositories` over the TCP port, which transitively
forwards through the parent to the child code-server. If the child is dead,
this raises and the healthcheck fails; Docker then restarts the container per
its `restart: unless-stopped` policy.

Exit code 0 on success, 1 on any error. Timeout 8s to avoid hanging Docker.
"""
from __future__ import annotations

import sys

TIMEOUT_SEC = 8


def main() -> int:
    try:
        from dagster._grpc.client import DagsterGrpcClient
    except Exception as exc:
        print(f"healthcheck: cannot import dagster grpc client: {exc}", file=sys.stderr)
        return 1
    try:
        client = DagsterGrpcClient(host="127.0.0.1", port=4000)
        # get_server_id transitively forwards from the parent gRPC server to the
        # child code-server (this is the exact RPC the daemon uses on every
        # workspace tick). It accepts a timeout kwarg; list_repositories does
        # not. Both will fail if the child unix socket is gone.
        server_id = client.get_server_id(timeout=TIMEOUT_SEC)
        if not server_id:
            print("healthcheck: get_server_id returned empty", file=sys.stderr)
            return 1
        return 0
    except Exception as exc:
        print(f"healthcheck: get_server_id failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
