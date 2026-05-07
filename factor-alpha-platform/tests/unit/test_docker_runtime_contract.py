from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_user_code_dockerfile_includes_ib_runtime_root_modules():
    dockerfile = PROJECT_ROOT / "deploy/dagster/Dockerfile_user_code"
    text = dockerfile.read_text(encoding="utf-8")

    assert "COPY eval_alpha_ib.py seed_alphas_ib.py ./" in text


def test_user_code_dockerfile_smokes_ib_runtime_imports_at_build_time():
    dockerfile = PROJECT_ROOT / "deploy/dagster/Dockerfile_user_code"
    text = dockerfile.read_text(encoding="utf-8")

    assert "ENV PYTHONPATH=/opt/dagster/app:/opt/dagster/app/prod" in text
    # Build smoke must cover the same module set as the runtime probe in
    # `_ib_runtime_dependency_payload`. If the runtime probe gains a new
    # required module, add it here too — otherwise a regression slips past
    # the build and only fails at the next scheduled integrity tick.
    assert "import eval_alpha_ib" in text
    assert "import seed_alphas_ib" in text
    assert "import live_bar" in text
    assert "import moc_trader" in text


def test_dockerfile_build_smoke_covers_runtime_probe_modules():
    """Lock the build-smoke module list to the runtime probe's required set."""
    dockerfile = (PROJECT_ROOT / "deploy/dagster/Dockerfile_user_code").read_text(encoding="utf-8")
    # Pull the required module list from the runtime probe's source so the
    # two stay in sync. If someone adds a module to the probe but forgets
    # the Dockerfile, this test fails.
    import inspect

    from src.orchestration.dagster_defs import _ib_runtime_dependency_payload

    probe_src = inspect.getsource(_ib_runtime_dependency_payload)
    expected_modules = {"eval_alpha_ib", "seed_alphas_ib", "live_bar", "moc_trader"}
    for module in expected_modules:
        assert f'"{module}"' in probe_src, f"runtime probe missing {module}"
        assert f"import {module}" in dockerfile, f"Dockerfile build smoke missing {module}"
