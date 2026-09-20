"""Shared fixtures (final alignment). All external systems are mocked."""
from __future__ import annotations

from pathlib import Path

import pytest

from ..config import Config
from ..policy import Policy

TEST_CERT_PEM = """-----BEGIN CERTIFICATE-----
MIIDFTCCAf2gAwIBAgIUUKyyFjPTNi+m5bqRiWlcVw4pZLkwDQYJKoZIhvcNAQEL
BQAwGjEYMBYGA1UEAwwPdGVzdC1hZ2VudC1jb3JlMB4XDTI2MDkxMTA0MTUyNVoX
DTI2MDkxMjA0MTUyNVowGjEYMBYGA1UEAwwPdGVzdC1hZ2VudC1jb3JlMIIBIjAN
BgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAstG79cp6N3f8dal/rvVfeQSly9hN
IkcySU1o7/nCe5DKB2MaI7QfzcUujooCzVnBpWHqQXoofyGFw0ot0eWbuUspdAmf
Z5wZz6W4b2cmsnn8eXFIgkjaPpOtH5nLjJbcMZvu7FE1ppWRxIT6kTCKskYunF9m
oF1ukWuJ/rv8Q7yUshPOx16AWAKjQmxdR1c2wIZOMbptuO1TQJ540WdfxHBT/YLD
yC4SbAXBB8VbmYcEVW6XcFpV3btwkcwa0DcaXkV8UeIAmwvZf8w7/+Mg9XqDy1JO
BZ62JaMJDIOBn/J6ZPBdPUYWeqf14YCC+fMqjS1Qb7JqP93RSlzC2LW20wIDAQAB
o1MwUTAdBgNVHQ4EFgQU8fKSUIPhQTxnEAj+cpb8bBmWRYUwHwYDVR0jBBgwFoAU
8fKSUIPhQTxnEAj+cpb8bBmWRYUwDwYDVR0TAQH/BAUwAwEB/zANBgkqhkiG9w0B
AQsFAAOCAQEAQjl8niYIGVXQJAnWi7eFUs4cFRLmUs753bEM0075gC98Ad9yL+IL
4W9Yh3wFNbSFpBf55Mgtao/XFVTShTMqel0kjMRWWD96dou9aVfQt6V046wsGLBZ
rADpRbb9KCjQttSpiBPCl3fYNy7/mfjuzu1HQkMQ8BGRQl19jV48wIKYCS08IStz
wskVpOgIMw6xO8rtcIw7sziltuTR+q2DDTR8e9VwS9AYo+eFBeEdip7hRzrAL1E2
OSbX1hAEkxGgpyAhYff8jl7as29733MEI6VaYPWgydh43Ub1pKkKu/ErgYDMLB+D
3Xhd37fnQQjb91lxluwJ6q44WQO7WCGHRA==
-----END CERTIFICATE-----
"""


@pytest.fixture(autouse=True)
def _map_run_certs_to_tmp(tmp_path, monkeypatch):
    cert_dir = tmp_path / "run-certs"
    cert_dir.mkdir()
    cert_path = cert_dir / "test-agent-core.pem"
    cert_path.write_text(TEST_CERT_PEM, encoding="ascii")

    def _mapped_cert_path(cert_file: str) -> Path:
        prefix = "/run/deploy-approval/certs/"
        if cert_file.startswith(prefix):
            return cert_dir / cert_file[len(prefix):]
        return Path(cert_file)

    monkeypatch.setattr("agents.deploy_approval.policy._cert_filesystem_path", _mapped_cert_path)


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
        registry="ccr.ccs.tencentyun.com",
        review_comment_author_id="7950763",
        review_comment_author_login="review-agent-bot",
    )
    defaults.update(overrides)
    return Config(**defaults)


@pytest.fixture
def config():
    return make_config()


@pytest.fixture
def policy(config):
    return Policy(config)
