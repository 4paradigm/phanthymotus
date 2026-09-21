"""Shared fixtures (final alignment). All external systems are mocked."""
from __future__ import annotations


import pytest

from ..config import Config
from ..policy import Policy


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
        agent_core_tokens={"test-machine": "test-token"},
    )
    defaults.update(overrides)
    return Config(**defaults)


@pytest.fixture
def config():
    return make_config()


@pytest.fixture
def policy(config):
    return Policy(config)
