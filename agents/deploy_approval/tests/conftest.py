"""Shared fixtures (final alignment). All external systems are mocked."""
from __future__ import annotations

import subprocess

import pytest

from ..config import Config
from ..policy import Policy


@pytest.fixture(scope="session")
def offline_ca_pem(tmp_path_factory):
    """Generate one test-only CA per session; never provide a default runtime CA."""
    directory = tmp_path_factory.mktemp("offline-agent-core-ca")
    (directory / "ca.cnf").write_text(
        "[req]\nprompt = no\ndistinguished_name = dn\n"
        "[dn]\nCN = Offline Agent Core test CA\n"
        "[ca_extensions]\n"
        "basicConstraints = critical, CA:true\n"
        "keyUsage = critical, keyCertSign, cRLSign\n"
        "subjectKeyIdentifier = hash\n"
        "subjectAltName = IP:192.0.2.1\n",
        encoding="ascii",
    )
    # Config-file extensions also support macOS's existing LibreSSL CLI.
    subprocess.run(
        [
            "openssl", "req", "-new", "-x509", "-newkey", "rsa:2048",
            "-nodes", "-sha256", "-days", "2", "-config", "ca.cnf",
            "-extensions", "ca_extensions", "-keyout", "ca.key", "-out", "ca.pem",
        ],
        cwd=directory, check=True, capture_output=True, text=True, timeout=30,
    )
    (directory / "ca.key").chmod(0o600)
    return (directory / "ca.pem").read_text(encoding="ascii")


def make_config(**overrides):
    defaults = dict(
        allow_private_http=False,
        http_allowed_cidrs=[],
        github_api_url="https://api.github.com",
        webhook_enabled=True,
        github_webhook_secret="test-secret",
        machine_owners_file="/dev/null",
        github_repos=["4paradigm/phanthymotus"],
        poll_enabled=True,
        poll_interval_seconds=30,
        review_comment_author_id="7950763",
        review_comment_author_login="review-agent-bot",
        agent_core_tokens={"test-machine": "test-token"},
    )
    defaults.update(overrides)
    # Offline fixtures model an already authorized installation. Negative
    # authorization tests explicitly override these fields.
    defaults.setdefault("active_repos", list(defaults["github_repos"]))
    defaults.setdefault("auth_valid", True)
    return Config(**defaults)


@pytest.fixture
def config():
    return make_config()


@pytest.fixture
def policy(config):
    return Policy(config)
