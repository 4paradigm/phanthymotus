"""Guard the Docker build against baking QQ credentials or user runtime data."""
from pathlib import Path


def test_core_image_copies_only_bundled_resource_defaults():
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(
        encoding="utf-8"
    )
    # A dirty local resource directory may contain live secrets and user data.
    assert "COPY resource/ /work/resource/" not in dockerfile
    assert "ADD resource/" not in dockerfile
    assert (
        "COPY resource/memory/defaults/ /work/resource/memory/defaults/"
        in dockerfile
    )
    assert "COPY resource/memory/defaults/ /opt/defaults/memory/" in dockerfile


def test_build_context_excludes_mutable_qq_runtime_data():
    ignore = (Path(__file__).resolve().parents[1] / ".dockerignore").read_text(
        encoding="utf-8"
    ).splitlines()
    # Runtime data must not even enter the Docker build context.
    for pattern in (
        "resource/*", "!resource/memory/", "resource/memory/*",
        "!resource/memory/defaults/", "!resource/memory/defaults/**",
        "**/*.secret", "**/*.db", "**/.env",
    ):
        assert pattern in ignore


def test_core_build_reuses_reviewed_lockfile():
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(
        encoding="utf-8"
    )
    assert "COPY pyproject.toml uv.lock /work/" in dockerfile
    assert "uv sync --frozen --no-dev --no-install-project" in dockerfile
    assert "RUN uv lock" not in dockerfile
