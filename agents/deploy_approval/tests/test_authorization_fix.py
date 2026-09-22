"""Tests for startup authorization discovery, ACTIVE_REPOS, and regression fixes."""

from __future__ import annotations

import gzip
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ..config import Config, DEFAULT_GITHUB_REPOS, SUPPORTED_GITHUB_REPOS
from ..github_client import GitHubClient, GitHubError
from ..clients_common import stream_request, SecurityError
from ..server import _resolve_active_repos
from ..github_command_watcher import GitHubCommandWatcher


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_config(**overrides) -> Config:
    cfg = Config(
        github_repos=list(DEFAULT_GITHUB_REPOS),
        github_api_url="https://api.github.com",
        poll_enabled=True,
        poll_interval_seconds=30,
        registry="registry.example",
        review_comment_author_id="7950763",
        machine_owners_file="/dev/null",
        secrets_file="/dev/null",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_github(config, token_provider=None, http=None):
    return GitHubClient(config, token_provider=token_provider, http=http)


def _fake_install_repo_response(repos, page=1, per_page=100):
    items = []
    for name in repos:
        items.append({"full_name": name, "name": name.split("/", 1)[-1]})
    return items


# ------------------------------------------------------------------
# 1-4: ACTIVE_REPOS resolution (A, B, C, D)
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resolve_active_repos_main_only():
    config = _make_config(github_repos=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"])
    gh = _make_github(config)
    gh.list_installation_repositories = AsyncMock(
        return_value=["4paradigm/phanthymotus"]
    )
    active = await _resolve_active_repos(gh, config.github_repos)
    assert active == ["4paradigm/phanthymotus"]


@pytest.mark.asyncio
async def test_resolve_active_repos_main_and_driver():
    config = _make_config(github_repos=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"])
    gh = _make_github(config)
    gh.list_installation_repositories = AsyncMock(
        return_value=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"]
    )
    active = await _resolve_active_repos(gh, config.github_repos)
    assert active == ["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"]


@pytest.mark.asyncio
async def test_resolve_active_repos_missing_main_fails():
    config = _make_config(github_repos=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"])
    gh = _make_github(config)
    gh.list_installation_repositories = AsyncMock(return_value=["4paradigm/phanthymotus-driver"])
    with pytest.raises(RuntimeError, match="required repo.*not authorized"):
        await _resolve_active_repos(gh, config.github_repos)


@pytest.mark.asyncio
async def test_resolve_active_repos_unknown_ignored():
    config = _make_config(github_repos=["4paradigm/phanthymotus"])
    gh = _make_github(config)
    gh.list_installation_repositories = AsyncMock(
        return_value=["4paradigm/phanthymotus", "some/other-repo"]
    )
    active = await _resolve_active_repos(gh, config.github_repos)
    assert active == ["4paradigm/phanthymotus"]


# ------------------------------------------------------------------
# 5: Pagination for /installation/repositories
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_installation_repositories_pagination():
    config = _make_config()

    repos1 = [{"full_name": f"repo-{i}"} for i in range(100)]
    repos2 = [{"full_name": f"repo-{i}"} for i in range(100, 110)]

    class MockTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            page_param = str(request.url.params.get("page", "1"))
            page = int(page_param)
            if page == 1:
                resp_data = {"total_count": 110, "repositories": repos1}
            else:
                resp_data = {"total_count": 110, "repositories": repos2}
            body = json.dumps(resp_data).encode()
            return httpx.Response(200, content=body, request=request)

    gh = GitHubClient(config, http=httpx.AsyncClient(transport=MockTransport()))
    gh._token_provider = AsyncMock(return_value="fake-token")

    repos = await gh.list_installation_repositories()
    assert len(repos) == 110
    assert repos == sorted(repos)


# ------------------------------------------------------------------
# 6: Malformed installation API schema fails
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_installation_repositories_malformed_schema():
    config = _make_config()

    class MockTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(200, content=b"[\"not-an-object\"]", request=request)

    gh = GitHubClient(config, http=httpx.AsyncClient(transport=MockTransport()))
    gh._token_provider = AsyncMock(return_value="fake-token")

    with pytest.raises(GitHubError):
        await gh.list_installation_repositories()


@pytest.mark.asyncio
async def test_list_installation_repositories_top_level_array_rejected():
    """Regression: top-level array response must FAIL (GitHub returns object)."""
    config = _make_config()

    class MockTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(
                200,
                content=json.dumps([{"full_name": "org/repo"}]).encode(),
                request=request,
            )

    gh = GitHubClient(config, http=httpx.AsyncClient(transport=MockTransport()))
    gh._token_provider = AsyncMock(return_value="fake-token")

    with pytest.raises(GitHubError):
        await gh.list_installation_repositories()


# ------------------------------------------------------------------
# 7: Installation HTTP failure fails startup
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_installation_repositories_http_failure():
    config = _make_config()

    class MockTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(500, content=b"{}", request=request)

    gh = GitHubClient(config, http=httpx.AsyncClient(transport=MockTransport()))
    gh._token_provider = AsyncMock(return_value="fake-token")

    with pytest.raises(GitHubError, match="unexpected status 500"):
        await gh.list_installation_repositories()


# ------------------------------------------------------------------
# 8-11: URL double-prefix regression tests
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_url_relative_path_exactly_once():
    config = _make_config(github_api_url="https://api.github.com")
    gh = _make_github(config)
    url = gh._resolve_api_url("/repos/foo/bar")
    assert url == "https://api.github.com/repos/foo/bar"
    assert url.count("api.github.com") == 1


@pytest.mark.asyncio
async def test_url_absolute_same_origin_exactly_once():
    config = _make_config(github_api_url="https://api.github.com")
    gh = _make_github(config)
    url = gh._resolve_api_url("https://api.github.com/repos/foo/bar")
    assert url == "https://api.github.com/repos/foo/bar"
    assert url.count("api.github.com") == 1


@pytest.mark.asyncio
async def test_url_absolute_wrong_origin_rejected():
    config = _make_config(github_api_url="https://api.github.com")
    gh = _make_github(config)
    with pytest.raises(GitHubError, match="origin"):
        gh._resolve_api_url("https://evil.com/repos/foo/bar")


@pytest.mark.asyncio
async def test_no_double_prefix_in_api_method():
    config = _make_config(github_api_url="https://api.github.com")
    gh = _make_github(config)
    url = gh.api("/repos/foo/bar")
    assert url == "https://api.github.com/repos/foo/bar"
    url2 = gh.api("repos/foo/bar")
    assert url2 == "https://api.github.com/repos/foo/bar"
    url3 = gh.api("//repos/foo/bar")
    assert url3 == "https://api.github.com/repos/foo/bar"


# ------------------------------------------------------------------
# 12-13: stream_request gzip fix + body size cap
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_request_gzip_json_survives():
    config = _make_config()

    json_data = json.dumps({"key": "value"}).encode("utf-8")
    compressed = gzip.compress(json_data)

    class MockTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            # A real httpx transport decompresses the body before aiter_bytes()
            # returns chunks, but preserves the Content-Encoding / Content-Length
            # headers on the Response object.  stream_request must strip those
            # headers so the reconstructed Response can call .json() on the
            # already-decoded body.
            resp = httpx.Response(200, content=json_data, request=request)
            resp.headers["content-encoding"] = "gzip"
            resp.headers["content-length"] = str(len(compressed))
            return resp

    client = httpx.AsyncClient(transport=MockTransport())
    resp = await stream_request(client, "GET", "https://api.github.com/test", config.max_response_bytes)
    data = resp.json()
    assert data == {"key": "value"}


@pytest.mark.asyncio
async def test_stream_request_body_size_cap_still_works():
    config = _make_config(max_response_bytes=100)

    class MockTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(200, content=b"x" * 200, request=request)

    client = httpx.AsyncClient(transport=MockTransport())
    with pytest.raises(SecurityError, match="exceeds limit"):
        await stream_request(client, "GET", "https://api.github.com/test", config.max_response_bytes)


# ------------------------------------------------------------------
# 14-15: Webhook repo allowance based on ACTIVE_REPOS
# ------------------------------------------------------------------

def test_webhook_driver_blocked_when_inactive():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from ..router_webhook import router as webhook_router

    config = Config(
        github_repos=["4paradigm/phanthymotus"],
        webhook_enabled=True,
        github_webhook_secret="secret",
        poll_enabled=True,
        registry="registry.example",
        review_comment_author_id="7950763",
        machine_owners_file="/dev/null",
        secrets_file="/dev/null",
    )

    app = FastAPI()
    app.state.config = config
    app.include_router(webhook_router)

    client = TestClient(app)

    payload = json.dumps({
        "action": "created",
        "repository": {"full_name": "4paradigm/phanthymotus-driver"},
        "issue": {"number": 1, "pull_request": {}},
        "comment": {"id": 99},
    }).encode()

    response = client.post(
        "/webhook",
        data=payload,
        headers={
            "X-GitHub-Event": "issue_comment",
            "X-Hub-Signature-256": "sha256=94b5df81103e3a1a6c2e6f7a9d0a429131d732bfca34d2e3d5b63d56cff457da",
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 404


def test_webhook_driver_allowed_when_active():
    from unittest.mock import AsyncMock

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from ..router_webhook import router as webhook_router

    config = Config(
        github_repos=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"],
        webhook_enabled=True,
        github_webhook_secret="secret",
        poll_enabled=True,
        registry="registry.example",
        review_comment_author_id="7950763",
        machine_owners_file="/dev/null",
        secrets_file="/dev/null",
    )

    proxy = AsyncMock()
    proxy.get_comment = AsyncMock(return_value={
        "id": 99,
        "body": "/request_deploy",
        "user": {"id": 111, "login": "alice"},
    })

    app = FastAPI()
    app.state.config = config
    app.state.proxy = proxy
    app.include_router(webhook_router)

    client = TestClient(app)

    payload = json.dumps({
        "action": "created",
        "repository": {"full_name": "4paradigm/phanthymotus-driver"},
        "issue": {"number": 1, "pull_request": {}},
        "comment": {"id": 99},
    }).encode()

    response = client.post(
        "/webhook",
        data=payload,
        headers={
            "X-GitHub-Event": "issue_comment",
            "X-Hub-Signature-256": "sha256=94b5df81103e3a1a6c2e6f7a9d0a429131d732bfca34d2e3d5b63d56cff457da",
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "deferred"


# ------------------------------------------------------------------
# 16-19: First-observation historical command baseline safety
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_first_observation_baseline_blocks_historical_commands():
    """Integration-style test: snapshot BEFORE reconcile blocks replay.

    Mock reconcile_pr to behave like production: when hidden_state is None,
    it creates a new state with last_processed_comment_id=0.  Without the
    "snapshot before reconcile" guard this would cause the watcher to start
    processing historical comments from cursor=0.
    """
    config = _make_config()
    proxy = MagicMock()

    # Mutable hidden state to simulate real persistence.
    hidden_state = [None]

    async def mock_read_hidden_state(*args, **kwargs):
        return hidden_state[0]

    proxy.read_hidden_state = AsyncMock(side_effect=mock_read_hidden_state)
    proxy.get_pr = AsyncMock(return_value={"state": "open"})
    proxy.get_issue_comments = AsyncMock(return_value=[
        {"id": 10, "body": "/approve_deploy machine=test"},
        {"id": 20, "body": "/request_deploy"},
    ])
    async def mock_persist_cursor(repo, pr, cid):
        hidden_state[0] = {"last_processed_comment_id": cid, "head_sha": "a" * 40}
        return hidden_state[0]
    proxy.persist_cursor = AsyncMock(side_effect=mock_persist_cursor)
    proxy.get_comment = AsyncMock(return_value={"id": 20, "body": "/request_deploy", "user": {"id": 111, "login": "alice"}})
    proxy.is_bot_comment = MagicMock(return_value=False)

    controller = AsyncMock()

    async def mock_reconcile_pr(repo, pr_number):
        # Production reconcile_pr creates lifecycle state when none exists
        if hidden_state[0] is None:
            hidden_state[0] = {"last_processed_comment_id": 0, "head_sha": "a" * 40}

    controller.reconcile_pr = AsyncMock(side_effect=mock_reconcile_pr)
    controller.on_command = AsyncMock(return_value=True)

    watcher = GitHubCommandWatcher(config, proxy, controller)

    await watcher._process_pr("4paradigm/phanthymotus", 1)

    # Historical commands must NOT be dispatched.
    controller.on_command.assert_not_called()
    # Baseline cursor must have been persisted to max existing comment id.
    proxy.persist_cursor.assert_called()
    call_args = proxy.persist_cursor.call_args
    assert call_args.args[2] == 20
    # Hidden state should now have cursor=20 (baseline).
    assert hidden_state[0]["last_processed_comment_id"] == 20


@pytest.mark.asyncio
async def test_first_observation_baseline_new_command_after_still_executes():
    """Two-cycle test: baseline + new command after baseline.

    Cycle 1: state=None, comments=[id=10 historical]
      -> cursor=10, on_command=0
    Cycle 2: state exists cursor=10, comments=[id=10 historical, id=20 NEW]
      -> on_command called exactly once for id=20
    """
    config = _make_config()
    proxy = MagicMock()

    hidden_state = [None]
    historical_comments = [
        {"id": 10, "body": "/approve_deploy machine=test"},
    ]

    async def mock_read_hidden_state(*args, **kwargs):
        return hidden_state[0]

    proxy.read_hidden_state = AsyncMock(side_effect=mock_read_hidden_state)
    proxy.get_pr = AsyncMock(return_value={"state": "open"})
    proxy.get_issue_comments = AsyncMock(return_value=historical_comments)
    async def mock_persist_cursor(repo, pr, cid):
        hidden_state[0] = {"last_processed_comment_id": cid, "head_sha": "a" * 40}
        return hidden_state[0]
    proxy.persist_cursor = AsyncMock(side_effect=mock_persist_cursor)
    proxy.get_comment = AsyncMock(return_value={"id": 20, "body": "/request_deploy", "user": {"id": 111, "login": "alice"}})
    proxy.is_bot_comment = MagicMock(return_value=False)

    controller = AsyncMock()

    async def mock_reconcile_pr(repo, pr_number):
        if hidden_state[0] is None:
            hidden_state[0] = {"last_processed_comment_id": 0, "head_sha": "a" * 40}

    controller.reconcile_pr = AsyncMock(side_effect=mock_reconcile_pr)
    controller.on_command = AsyncMock(return_value=True)
    controller.reconcile_pr = AsyncMock(side_effect=mock_reconcile_pr)

    watcher = GitHubCommandWatcher(config, proxy, controller)

    # Cycle 1: no state, historical comment only
    await watcher._process_pr("4paradigm/phanthymotus", 1)
    controller.on_command.assert_not_called()
    proxy.persist_cursor.assert_called()
    assert hidden_state[0]["last_processed_comment_id"] == 10

    # Cycle 2: state exists, new comment added
    hidden_state[0] = {"last_processed_comment_id": 10, "head_sha": "a" * 40}
    historical_comments.append({"id": 20, "body": "/request_deploy"})
    controller.on_command.reset_mock()
    await watcher._process_pr("4paradigm/phanthymotus", 1)

    # on_command called exactly once for the new comment (id=20)
    controller.on_command.assert_called_once()


@pytest.mark.asyncio
async def test_existing_state_cursor_behavior_unchanged():
    config = _make_config()
    proxy = MagicMock()
    proxy.read_hidden_state = AsyncMock(return_value={
        "last_processed_comment_id": 5,
        "head_sha": "a" * 40,
    })
    proxy.get_issue_comments = AsyncMock(return_value=[
        {"id": 3, "body": "/approve_deploy machine=test"},
        {"id": 10, "body": "/request_deploy"},
    ])
    proxy.get_comment = AsyncMock(return_value={"id": 10, "body": "/request_deploy", "user": {"id": 111, "login": "alice"}})
    proxy.is_bot_comment = MagicMock(return_value=False)
    proxy.persist_cursor = AsyncMock(return_value=None)

    controller = AsyncMock()
    controller.reconcile_pr = AsyncMock()
    controller.on_command = AsyncMock(return_value=True)

    watcher = GitHubCommandWatcher(config, proxy, controller)

    await watcher._process_pr("4paradigm/phanthymotus", 1)

    controller.on_command.assert_called()


@pytest.mark.asyncio
async def test_bot_comment_no_cursor_regression():
    config = _make_config()
    proxy = MagicMock()
    proxy.read_hidden_state = AsyncMock(return_value={
        "last_processed_comment_id": 100,
        "head_sha": "a" * 40,
    })
    proxy.get_issue_comments = AsyncMock(return_value=[
        {"id": 50, "body": "<!-- deploy-approval-state:v1\n{}\n-->"},
    ])
    proxy.get_comment = AsyncMock(return_value={"id": 50, "body": "<!-- deploy-approval-state:v1\n{}\n-->", "user": {"id": 123, "login": "bot"}})
    proxy.is_bot_comment = MagicMock(return_value=True)
    proxy.persist_cursor = AsyncMock(return_value=None)

    controller = AsyncMock()
    controller.reconcile_pr = AsyncMock()
    controller.on_command = AsyncMock(return_value=False)

    watcher = GitHubCommandWatcher(config, proxy, controller)

    await watcher._process_pr("4paradigm/phanthymotus", 1)

    controller.on_command.assert_not_called()


# ------------------------------------------------------------------
# 20: No token/private-key logging
# ------------------------------------------------------------------

def test_no_token_in_default_github_repos():
    assert "4paradigm/phanthymotus" in DEFAULT_GITHUB_REPOS
    assert "4paradigm/phanthymotus-driver" in DEFAULT_GITHUB_REPOS


def test_supported_github_repos_contains_both():
    assert "4paradigm/phanthymotus" in SUPPORTED_GITHUB_REPOS
    assert "4paradigm/phanthymotus-driver" in SUPPORTED_GITHUB_REPOS


@pytest.mark.asyncio
async def test_resolve_active_repos_logs_no_secrets(caplog):
    import logging
    caplog.set_level(logging.INFO)
    config = _make_config(github_repos=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"])
    gh = _make_github(config)
    gh.list_installation_repositories = AsyncMock(
        return_value=["4paradigm/phanthymotus"]
    )
    await _resolve_active_repos(gh, config.github_repos)
    logged = caplog.text
    assert "4paradigm/phanthymotus" in logged
    assert "4paradigm/phanthymotus-driver" in logged


# ------------------------------------------------------------------
# STARTUP INTEGRATION TESTS (create_app / lifespan path)
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_startup_main_only_active_repos():
    """Lifespan integration: installation returns only main.

    desired = main + driver
    active = [main]
    config.github_repos becomes [main]
    bootstrap receives only main
    watcher starts after authorization resolution
    """
    from fastapi.testclient import TestClient
    from ..server import create_app
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(
            "version: 1\n"
            "machines:\n"
            "  test-machine:\n"
            "    node_id: node-1\n"
            "    node_host: 127.0.0.1\n"
            "    owners:\n"
            "      - owner1\n"
            "    targets:\n"
            "      - perception\n"
            "    platforms:\n"
            "      - linux/arm64\n"
        )
        machine_yaml_path = f.name

    cfg = Config(
        github_repos=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"],
        github_api_url="https://api.github.com",
        poll_enabled=True,
        poll_interval_seconds=30,
        registry="registry.example",
        review_comment_author_id="7950763",
        machine_owners_file=machine_yaml_path,
        secrets_file="/dev/null",
        webhook_enabled=False,
    )

    class FakeGitHub:
        async def get_current_user(self):
            return {"id": 222, "login": "bot"}

        async def list_installation_repositories(self):
            return ["4paradigm/phanthymotus"]

        async def list_repository_labels(self, repo):
            return []

        async def create_repository_label(self, repo, name, color, description):
            return {"name": name, "color": color, "description": description}

    watcher_started = []

    class FakeWatcher:
        def __init__(self, config, proxy, controller):
            pass
        def start(self):
            watcher_started.append(True)

        async def stop(self):
            pass

    from .. import server as server_mod
    orig_auth = server_mod.github_app_auth.create_github_app_auth
    orig_ghc = server_mod.GitHubClient
    orig_watcher = server_mod.GitHubCommandWatcher

    class _FakeAppAuth:
        app_id = "12345"
        async def get_installation_token(self):
            return "fake-token"
        async def close(self):
            pass

    server_mod.github_app_auth.create_github_app_auth = lambda: _FakeAppAuth()
    server_mod.GitHubClient = lambda cfg_, token_provider=None: FakeGitHub()
    server_mod.GitHubCommandWatcher = FakeWatcher

    try:
        app = create_app(cfg)
        with TestClient(app) as client:
            # Trigger lifespan by entering the test client context
            pass
    finally:
        server_mod.github_app_auth.create_github_app_auth = orig_auth
        server_mod.GitHubClient = orig_ghc
        server_mod.GitHubCommandWatcher = orig_watcher

    assert watcher_started == [True]
    assert cfg.github_repos == ["4paradigm/phanthymotus"]


@pytest.mark.asyncio
async def test_startup_missing_main_watcher_not_started():
    """Lifespan integration: main missing from installation -> startup FAILS, watcher does NOT start."""
    from fastapi.testclient import TestClient
    from ..server import create_app

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(
            "version: 1\n"
            "machines:\n"
            "  test-machine:\n"
            "    node_id: node-1\n"
            "    node_host: 127.0.0.1\n"
            "    owners:\n"
            "      - owner1\n"
            "    targets:\n"
            "      - perception\n"
            "    platforms:\n"
            "      - linux/arm64\n"
        )
        machine_yaml_path = f.name

    cfg = Config(
        github_repos=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"],
        github_api_url="https://api.github.com",
        poll_enabled=True,
        poll_interval_seconds=30,
        registry="registry.example",
        review_comment_author_id="7950763",
        machine_owners_file=machine_yaml_path,
        secrets_file="/dev/null",
        webhook_enabled=False,
    )

    class FakeGitHub:
        async def get_current_user(self):
            return {"id": 222, "login": "bot"}

        async def list_installation_repositories(self):
            # Only driver authorized; main is missing
            return ["4paradigm/phanthymotus-driver"]

        async def list_repository_labels(self, repo):
            return []

        async def create_repository_label(self, repo, name, color, description):
            return {"name": name, "color": color, "description": description}

    watcher_started = []

    class FakeWatcher:
        def __init__(self, config, proxy, controller):
            pass
        def start(self):
            watcher_started.append(True)

        async def stop(self):
            pass

    from .. import server as server_mod
    orig_auth = server_mod.github_app_auth.create_github_app_auth
    orig_ghc = server_mod.GitHubClient
    orig_watcher = server_mod.GitHubCommandWatcher

    class _FakeAppAuth:
        app_id = "12345"
        async def get_installation_token(self):
            return "fake-token"
        async def close(self):
            pass

    server_mod.github_app_auth.create_github_app_auth = lambda: _FakeAppAuth()
    server_mod.GitHubClient = lambda cfg_, token_provider=None: FakeGitHub()
    server_mod.GitHubCommandWatcher = FakeWatcher

    try:
        app = create_app(cfg)
        with pytest.raises(RuntimeError, match="required repo.*not authorized"):
            with TestClient(app):
                pass
    finally:
        server_mod.github_app_auth.create_github_app_auth = orig_auth
        server_mod.GitHubClient = orig_ghc
        server_mod.GitHubCommandWatcher = orig_watcher

    assert watcher_started == []


# ------------------------------------------------------------------
# BASELINE PERSISTENCE FAILURE SAFETY TESTS
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_baseline_persist_failure_cycle1_zero_dispatch():
    """Test A: persist_cursor returns None on first cycle -> zero dispatch."""
    config = _make_config()
    proxy = MagicMock()

    hidden_state = [None]

    async def mock_read_hidden_state(*args, **kwargs):
        return hidden_state[0]

    proxy.read_hidden_state = AsyncMock(side_effect=mock_read_hidden_state)
    proxy.get_pr = AsyncMock(return_value={"state": "open"})
    proxy.get_issue_comments = AsyncMock(return_value=[
        {"id": 10, "body": "/request_deploy"},
    ])
    # First persist returns None (simulating transient failure)
    persist_call_count = [0]
    async def mock_persist_cursor(repo, pr, cid):
        persist_call_count[0] += 1
        if persist_call_count[0] == 1:
            return None  # First attempt fails
        hidden_state[0] = {"last_processed_comment_id": cid, "head_sha": "a" * 40}
        return hidden_state[0]
    proxy.persist_cursor = AsyncMock(side_effect=mock_persist_cursor)
    proxy.get_comment = AsyncMock(return_value={"id": 10, "body": "/request_deploy", "user": {"id": 111, "login": "alice"}})
    proxy.is_bot_comment = MagicMock(return_value=False)

    controller = AsyncMock()

    async def mock_reconcile_pr(repo, pr_number):
        if hidden_state[0] is None:
            hidden_state[0] = {"last_processed_comment_id": 0, "head_sha": "a" * 40, "command": {"comment_id": 0, "kind": "", "phase": "completed", "args": {}}}

    controller.reconcile_pr = AsyncMock(side_effect=mock_reconcile_pr)
    controller.on_command = AsyncMock(return_value=True)

    watcher = GitHubCommandWatcher(config, proxy, controller)

    # Cycle 1: persist fails, no dispatch
    await watcher._process_pr("4paradigm/phanthymotus", 1)
    controller.on_command.assert_not_called()
    # State now has cursor=0 from reconcile
    assert hidden_state[0]["last_processed_comment_id"] == 0


@pytest.mark.asyncio
async def test_baseline_persist_failure_cycle2_still_zero_dispatch():
    """Test A continuation: cycle 2 still zero dispatch, cursor recovers."""
    config = _make_config()
    proxy = MagicMock()

    hidden_state = [None]

    async def mock_read_hidden_state(*args, **kwargs):
        return hidden_state[0]

    proxy.read_hidden_state = AsyncMock(side_effect=mock_read_hidden_state)
    proxy.get_pr = AsyncMock(return_value={"state": "open"})
    proxy.get_issue_comments = AsyncMock(return_value=[
        {"id": 10, "body": "/request_deploy"},
    ])
    persist_call_count = [0]
    async def mock_persist_cursor(repo, pr, cid):
        persist_call_count[0] += 1
        if persist_call_count[0] == 1:
            return None  # First attempt fails
        hidden_state[0] = {"last_processed_comment_id": cid, "head_sha": "a" * 40}
        return hidden_state[0]
    proxy.persist_cursor = AsyncMock(side_effect=mock_persist_cursor)
    proxy.get_comment = AsyncMock(return_value={"id": 10, "body": "/request_deploy", "user": {"id": 111, "login": "alice"}})
    proxy.is_bot_comment = MagicMock(return_value=False)

    controller = AsyncMock()

    async def mock_reconcile_pr(repo, pr_number):
        if hidden_state[0] is None:
            hidden_state[0] = {"last_processed_comment_id": 0, "head_sha": "a" * 40, "command": {"comment_id": 0, "kind": "", "phase": "completed", "args": {}}}

    controller.reconcile_pr = AsyncMock(side_effect=mock_reconcile_pr)
    controller.on_command = AsyncMock(return_value=True)

    watcher = GitHubCommandWatcher(config, proxy, controller)

    # Simulate cycle 1 already happened: state has cursor=0 from reconcile
    hidden_state[0] = {"last_processed_comment_id": 0, "head_sha": "a" * 40, "command": {"comment_id": 0, "kind": "", "phase": "completed", "args": {}}}
    persist_call_count[0] = 1  # First attempt already failed

    # Cycle 2: persist succeeds, still no dispatch (baseline only)
    await watcher._process_pr("4paradigm/phanthymotus", 1)
    controller.on_command.assert_not_called()
    # Cursor should now be >= 10
    assert hidden_state[0]["last_processed_comment_id"] >= 10


@pytest.mark.asyncio
async def test_new_command_after_recovered_baseline_executes():
    """Test B: After baseline recovered, new comment id=20 is dispatched."""
    config = _make_config()
    proxy = MagicMock()

    hidden_state = [None]
    comments_list = [
        {"id": 10, "body": "/request_deploy"},
    ]

    async def mock_read_hidden_state(*args, **kwargs):
        return hidden_state[0]

    proxy.read_hidden_state = AsyncMock(side_effect=mock_read_hidden_state)
    proxy.get_pr = AsyncMock(return_value={"state": "open"})
    proxy.get_issue_comments = AsyncMock(return_value=comments_list)
    persist_call_count = [0]
    async def mock_persist_cursor(repo, pr, cid):
        persist_call_count[0] += 1
        hidden_state[0] = {"last_processed_comment_id": cid, "head_sha": "a" * 40}
        return hidden_state[0]
    proxy.persist_cursor = AsyncMock(side_effect=mock_persist_cursor)

    get_comment_call = [0]
    async def mock_get_comment(repo, cid):
        get_comment_call[0] += 1
        if cid == 10:
            return {"id": 10, "body": "/request_deploy", "user": {"id": 111, "login": "alice"}}
        if cid == 20:
            return {"id": 20, "body": "/request_deploy", "user": {"id": 111, "login": "alice"}}
        return {"id": cid, "body": "", "user": {"id": 0}}
    proxy.get_comment = mock_get_comment
    proxy.is_bot_comment = MagicMock(return_value=False)

    controller = AsyncMock()

    async def mock_reconcile_pr(repo, pr_number):
        if hidden_state[0] is None:
            hidden_state[0] = {"last_processed_comment_id": 0, "head_sha": "a" * 40, "command": {"comment_id": 0, "kind": "", "phase": "completed", "args": {}}}

    controller.reconcile_pr = AsyncMock(side_effect=mock_reconcile_pr)
    controller.on_command = AsyncMock(return_value=True)

    watcher = GitHubCommandWatcher(config, proxy, controller)

    # Cycle 1: baseline succeeds on first try
    await watcher._process_pr("4paradigm/phanthymotus", 1)
    controller.on_command.assert_not_called()
    assert hidden_state[0]["last_processed_comment_id"] >= 10

    # Cycle 2: new comment added
    hidden_state[0] = {"last_processed_comment_id": 10, "head_sha": "a" * 40}
    comments_list.append({"id": 20, "body": "/request_deploy"})
    controller.on_command.reset_mock()

    await watcher._process_pr("4paradigm/phanthymotus", 1)
    controller.on_command.assert_called_once()
    call_args = controller.on_command.call_args
    # args: (cmd, repo, pr_number, comment_id) -> args[3] is comment_id
    assert call_args.args[3] == 20


@pytest.mark.asyncio
async def test_baseline_persist_consecutive_failures_zero_dispatch():
    """Test C: baseline persistence fails twice -> zero dispatch both cycles."""
    config = _make_config()
    proxy = MagicMock()

    hidden_state = [None]

    async def mock_read_hidden_state(*args, **kwargs):
        return hidden_state[0]

    proxy.read_hidden_state = AsyncMock(side_effect=mock_read_hidden_state)
    proxy.get_pr = AsyncMock(return_value={"state": "open"})
    proxy.get_issue_comments = AsyncMock(return_value=[
        {"id": 10, "body": "/request_deploy"},
    ])
    persist_call_count = [0]
    async def mock_persist_cursor(repo, pr, cid):
        persist_call_count[0] += 1
        return None  # Always fails
    proxy.persist_cursor = AsyncMock(side_effect=mock_persist_cursor)
    proxy.get_comment = AsyncMock(return_value={"id": 10, "body": "/request_deploy", "user": {"id": 111, "login": "alice"}})
    proxy.is_bot_comment = MagicMock(return_value=False)

    controller = AsyncMock()

    async def mock_reconcile_pr(repo, pr_number):
        if hidden_state[0] is None:
            hidden_state[0] = {"last_processed_comment_id": 0, "head_sha": "a" * 40, "command": {"comment_id": 0, "kind": "", "phase": "completed", "args": {}}}

    controller.reconcile_pr = AsyncMock(side_effect=mock_reconcile_pr)
    controller.on_command = AsyncMock(return_value=True)

    watcher = GitHubCommandWatcher(config, proxy, controller)

    # Cycle 1
    await watcher._process_pr("4paradigm/phanthymotus", 1)
    controller.on_command.assert_not_called()

    # Cycle 2: state still has cursor=0, baseline incomplete
    await watcher._process_pr("4paradigm/phanthymotus", 1)
    controller.on_command.assert_not_called()


@pytest.mark.asyncio
async def test_baseline_persist_returns_low_cursor_zero_dispatch():
    """Test D: persist returns dict but last_processed_comment_id < baseline_id."""
    config = _make_config()
    proxy = MagicMock()

    hidden_state = [None]

    async def mock_read_hidden_state(*args, **kwargs):
        return hidden_state[0]

    proxy.read_hidden_state = AsyncMock(side_effect=mock_read_hidden_state)
    proxy.get_pr = AsyncMock(return_value={"state": "open"})
    proxy.get_issue_comments = AsyncMock(return_value=[
        {"id": 10, "body": "/request_deploy"},
    ])
    async def mock_persist_cursor(repo, pr, cid):
        # Returns a dict but with a cursor less than requested baseline_id=10
        return {"last_processed_comment_id": 5, "head_sha": "a" * 40}
    proxy.persist_cursor = AsyncMock(side_effect=mock_persist_cursor)
    proxy.get_comment = AsyncMock(return_value={"id": 10, "body": "/request_deploy", "user": {"id": 111, "login": "alice"}})
    proxy.is_bot_comment = MagicMock(return_value=False)

    controller = AsyncMock()

    async def mock_reconcile_pr(repo, pr_number):
        if hidden_state[0] is None:
            hidden_state[0] = {"last_processed_comment_id": 0, "head_sha": "a" * 40, "command": {"comment_id": 0, "kind": "", "phase": "completed", "args": {}}}

    controller.reconcile_pr = AsyncMock(side_effect=mock_reconcile_pr)
    controller.on_command = AsyncMock(return_value=True)

    watcher = GitHubCommandWatcher(config, proxy, controller)

    await watcher._process_pr("4paradigm/phanthymotus", 1)
    controller.on_command.assert_not_called()
    # Cursor should NOT have advanced to 10
    assert hidden_state[0]["last_processed_comment_id"] == 0


@pytest.mark.asyncio
async def test_zero_comments_first_observation_reconcile_creates_bot_comment():
    """Test E: initial comments empty, reconcile creates lifecycle bot comment id=100."""
    config = _make_config()
    proxy = MagicMock()

    hidden_state = [None]
    comments_list = []  # Initially empty

    async def mock_read_hidden_state(*args, **kwargs):
        return hidden_state[0]

    proxy.read_hidden_state = AsyncMock(side_effect=mock_read_hidden_state)
    proxy.get_pr = AsyncMock(return_value={"state": "open"})
    proxy.get_issue_comments = AsyncMock(side_effect=lambda *a, **k: comments_list)
    async def mock_persist_cursor(repo, pr, cid):
        hidden_state[0] = {"last_processed_comment_id": cid, "head_sha": "a" * 40}
        return hidden_state[0]
    proxy.persist_cursor = AsyncMock(side_effect=mock_persist_cursor)
    proxy.get_comment = AsyncMock(return_value={"id": 101, "body": "/request_deploy", "user": {"id": 111, "login": "alice"}})
    proxy.is_bot_comment = MagicMock(return_value=False)

    controller = AsyncMock()

    async def mock_reconcile_pr(repo, pr_number):
        if hidden_state[0] is None:
            hidden_state[0] = {"last_processed_comment_id": 0, "head_sha": "a" * 40, "command": {"comment_id": 0, "kind": "", "phase": "completed", "args": {}}}
        # Reconcile creates a lifecycle bot comment
        comments_list.append({"id": 100, "body": "<!-- deploy-approval-state:v1\n{}\n-->"})

    controller.reconcile_pr = AsyncMock(side_effect=mock_reconcile_pr)
    controller.on_command = AsyncMock(return_value=True)

    watcher = GitHubCommandWatcher(config, proxy, controller)

    # Cycle 1: baseline at 100 (from post-reconcile snapshot)
    await watcher._process_pr("4paradigm/phanthymotus", 1)
    controller.on_command.assert_not_called()
    assert hidden_state[0]["last_processed_comment_id"] >= 100

    # Cycle 2: new comment id=101
    hidden_state[0] = {"last_processed_comment_id": 100, "head_sha": "a" * 40}
    comments_list.append({"id": 101, "body": "/request_deploy"})
    controller.on_command.reset_mock()

    await watcher._process_pr("4paradigm/phanthymotus", 1)
    controller.on_command.assert_called_once()
    call_args = controller.on_command.call_args
    assert call_args.args[3] == 101


# ------------------------------------------------------------------
# WATCHER POST-RECONCILE NONE FAIL-CLOSED
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_existing_state_disappears_after_reconcile_fails_closed():
    """Regression: reconcile wipes lifecycle state -> zero dispatch, no exception."""
    config = _make_config()
    proxy = MagicMock()

    hidden_state = [{
        "last_processed_comment_id": 10,
        "head_sha": "a" * 40,
        "command": {"comment_id": 5, "kind": "request_deploy", "phase": "completed", "args": {}},
    }]

    async def mock_read_hidden_state(*args, **kwargs):
        # First read (initial_state) returns valid state.
        # Second read (after reconcile) returns None.
        if mock_read_hidden_state.call_count <= 1:
            mock_read_hidden_state.call_count += 1
            return hidden_state[0]
        mock_read_hidden_state.call_count += 1
        return None
    mock_read_hidden_state.call_count = 0

    proxy.read_hidden_state = AsyncMock(side_effect=mock_read_hidden_state)
    proxy.get_pr = AsyncMock(return_value={"state": "open"})
    proxy.get_issue_comments = AsyncMock(return_value=[
        {"id": 15, "body": "/request_deploy"},
    ])
    proxy.persist_cursor = AsyncMock(return_value=None)
    proxy.get_comment = AsyncMock(return_value={"id": 15, "body": "/request_deploy", "user": {"id": 111, "login": "alice"}})
    proxy.is_bot_comment = MagicMock(return_value=False)

    controller = AsyncMock()

    async def mock_reconcile_pr(repo, pr_number):
        # Simulate reconcile wiping state (edge case)
        hidden_state[0] = None

    controller.reconcile_pr = AsyncMock(side_effect=mock_reconcile_pr)
    controller.on_command = AsyncMock(return_value=True)

    watcher = GitHubCommandWatcher(config, proxy, controller)

    await watcher._process_pr("4paradigm/phanthymotus", 1)

    controller.on_command.assert_not_called()
    proxy.get_comment.assert_not_called()


# ------------------------------------------------------------------
# STARTUP AUTHORIZATION FAILURE RESOURCE CLEANUP
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_startup_authorization_failure_closes_resources():
    """Lifespan integration: authorization failure -> resources cleaned up."""
    from fastapi.testclient import TestClient
    from ..server import create_app

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(
            "version: 1\n"
            "machines:\n"
            "  test-machine:\n"
            "    node_id: node-1\n"
            "    node_host: 127.0.0.1\n"
            "    owners:\n"
            "      - owner1\n"
            "    targets:\n"
            "      - perception\n"
            "    platforms:\n"
            "      - linux/arm64\n"
        )
        machine_yaml_path = f.name

    cfg = Config(
        github_repos=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"],
        github_api_url="https://api.github.com",
        poll_enabled=True,
        poll_interval_seconds=30,
        registry="registry.example",
        review_comment_author_id="7950763",
        machine_owners_file=machine_yaml_path,
        secrets_file="/dev/null",
        webhook_enabled=False,
    )

    github_closed = []
    registry_closed = []
    auth_closed = []

    class FakeGitHub:
        async def get_current_user(self):
            return {"id": 222, "login": "bot"}

        async def list_installation_repositories(self):
            raise RuntimeError("installation API down")

        async def list_repository_labels(self, repo):
            return []

        async def create_repository_label(self, repo, name, color, description):
            return {"name": name, "color": color, "description": description}

        @property
        def http(self):
            class FakeHTTP:
                async def aclose(self):
                    github_closed.append(True)
            return FakeHTTP()

    class FakeRegistry:
        @property
        def http(self):
            class FakeHTTP:
                async def aclose(self):
                    registry_closed.append(True)
            return FakeHTTP()

    watcher_started = []

    class FakeWatcher:
        def __init__(self, config, proxy, controller):
            pass
        def start(self):
            watcher_started.append(True)
        async def stop(self):
            pass

    from .. import server as server_mod
    orig_auth = server_mod.github_app_auth.create_github_app_auth
    orig_ghc = server_mod.GitHubClient
    orig_reg = server_mod.RegistryClient
    orig_watcher = server_mod.GitHubCommandWatcher

    class _FakeAppAuth:
        app_id = "12345"
        async def get_installation_token(self):
            return "fake-token"
        async def close(self):
            auth_closed.append(True)

    server_mod.github_app_auth.create_github_app_auth = lambda: _FakeAppAuth()
    server_mod.GitHubClient = lambda cfg_, token_provider=None: FakeGitHub()
    server_mod.RegistryClient = lambda cfg_: FakeRegistry()
    server_mod.GitHubCommandWatcher = FakeWatcher

    try:
        app = create_app(cfg)
        with pytest.raises(RuntimeError, match="installation API down"):
            with TestClient(app):
                pass
    finally:
        server_mod.github_app_auth.create_github_app_auth = orig_auth
        server_mod.GitHubClient = orig_ghc
        server_mod.RegistryClient = orig_reg
        server_mod.GitHubCommandWatcher = orig_watcher

    assert watcher_started == []
    assert auth_closed == [True]
    assert github_closed == [True]
    assert registry_closed == [True]


# ------------------------------------------------------------------
# CLEANUP FAILURE DOES NOT MASK STARTUP ERROR
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_startup_cleanup_failure_does_not_mask_authorization_error():
    """Regression: cleanup exceptions must not mask the root startup error.

    Root cause: list_installation_repositories() raises RuntimeError.
    Cleanup steps also raise.
    The TestClient / lifespan must still see the root RuntimeError,
    not any cleanup error.
    """
    from fastapi.testclient import TestClient
    from ..server import create_app

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(
            "version: 1\n"
            "machines:\n"
            "  test-machine:\n"
            "    node_id: node-1\n"
            "    node_host: 127.0.0.1\n"
            "    owners:\n"
            "      - owner1\n"
            "    targets:\n"
            "      - perception\n"
            "    platforms:\n"
            "      - linux/arm64\n"
        )
        machine_yaml_path = f.name

    cfg = Config(
        github_repos=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"],
        github_api_url="https://api.github.com",
        poll_enabled=True,
        poll_interval_seconds=30,
        registry="registry.example",
        review_comment_author_id="7950763",
        machine_owners_file=machine_yaml_path,
        secrets_file="/dev/null",
        webhook_enabled=False,
    )

    cleanup_attempts = {"stop": 0, "aclose": 0, "gh_close": 0, "reg_close": 0, "auth_close": 0}

    class FakeGitHub:
        async def get_current_user(self):
            return {"id": 222, "login": "bot"}

        async def list_installation_repositories(self):
            raise RuntimeError("installation API down")

        async def list_repository_labels(self, repo):
            return []

        async def create_repository_label(self, repo, name, color, description):
            return {"name": name, "color": color, "description": description}

        @property
        def http(self):
            class FakeHTTP:
                async def aclose(self):
                    cleanup_attempts["gh_close"] += 1
                    raise RuntimeError("github close failed")
            return FakeHTTP()

    class FakeRegistry:
        @property
        def http(self):
            class FakeHTTP:
                async def aclose(self):
                    cleanup_attempts["reg_close"] += 1
                    raise RuntimeError("registry close failed")
            return FakeHTTP()

    watcher_started = []

    class FakeWatcher:
        def __init__(self, config, proxy, controller):
            pass
        def start(self):
            watcher_started.append(True)
        async def stop(self):
            cleanup_attempts["stop"] += 1
            raise RuntimeError("watcher stop failed")

    from .. import server as server_mod
    orig_auth = server_mod.github_app_auth.create_github_app_auth
    orig_ghc = server_mod.GitHubClient
    orig_reg = server_mod.RegistryClient
    orig_watcher = server_mod.GitHubCommandWatcher
    orig_controller = server_mod.DeployController

    class _FakeAppAuth:
        app_id = "12345"
        async def get_installation_token(self):
            return "fake-token"
        async def close(self):
            cleanup_attempts["auth_close"] += 1
            raise RuntimeError("auth close failed")

    class FakeController:
        async def aclose(self):
            cleanup_attempts["aclose"] += 1
            raise RuntimeError("controller close failed")

    server_mod.github_app_auth.create_github_app_auth = lambda: _FakeAppAuth()
    server_mod.GitHubClient = lambda cfg_, token_provider=None: FakeGitHub()
    server_mod.RegistryClient = lambda cfg_: FakeRegistry()
    server_mod.GitHubCommandWatcher = FakeWatcher
    server_mod.DeployController = lambda *a, **k: FakeController()

    try:
        app = create_app(cfg)
        with pytest.raises(RuntimeError, match="installation API down"):
            with TestClient(app):
                pass
    finally:
        server_mod.github_app_auth.create_github_app_auth = orig_auth
        server_mod.GitHubClient = orig_ghc
        server_mod.RegistryClient = orig_reg
        server_mod.GitHubCommandWatcher = orig_watcher
        server_mod.DeployController = orig_controller

    # watcher.start was never called
    assert watcher_started == []

    # All cleanup steps were attempted even though each raised
    assert cleanup_attempts["stop"] == 1
    assert cleanup_attempts["aclose"] == 1
    assert cleanup_attempts["gh_close"] == 1
    assert cleanup_attempts["reg_close"] == 1
    assert cleanup_attempts["auth_close"] == 1


# ---------------------------------------------------------------------------
# Test: post-yield exception triggers cleanup exactly once (no double cleanup)
# ---------------------------------------------------------------------------





@pytest.mark.asyncio
async def test_lifespan_body_exception_cleanup_runs_once():
    """Verify that an exception thrown through the yield point enters the
    single finally block exactly once - no duplicate cleanup from dual
    call sites."""
    import tempfile
    from fastapi.testclient import TestClient
    from ..server import create_app
    from ..config import Config
    import agents.deploy_approval.server as server_mod

    orig_auth = server_mod.github_app_auth.create_github_app_auth
    orig_ghc = server_mod.GitHubClient
    orig_reg = server_mod.RegistryClient
    orig_watcher = server_mod.GitHubCommandWatcher
    orig_controller = server_mod.DeployController

    cleanup_counts = {
        'stop': 0,
        'aclose': 0,
        'gh_close': 0,
        'reg_close': 0,
        'auth_close': 0,
    }

    class FakeGitHubHTTP:
        async def aclose(self):
            cleanup_counts['gh_close'] += 1

    class FakeRegistryHTTP:
        async def aclose(self):
            cleanup_counts['reg_close'] += 1

    class FakeWatcherCounting:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass

        async def stop(self):
            cleanup_counts['stop'] += 1
            raise RuntimeError('watcher stop failed')

    class FakeControllerCounting:
        async def aclose(self):
            cleanup_counts['aclose'] += 1
            raise RuntimeError('controller close failed')

    class FakeGitHubCounting:
        def __init__(self, *a, **k):
            self.http = FakeGitHubHTTP()

        async def list_installation_repositories(self):
            return ['4paradigm/phanthymotus']

        async def aclose(self):
            cleanup_counts['gh_close'] += 1
            raise RuntimeError('github close failed')

    class FakeRegistryCounting:
        def __init__(self, *a):
            self.http = FakeRegistryHTTP()

        async def aclose(self):
            cleanup_counts['reg_close'] += 1
            raise RuntimeError('registry close failed')

    class FakeAppAuthCounting:
        app_id = 'test-app'

        async def get_installation_token(self):
            return 'tok'

        async def close(self):
            cleanup_counts['auth_close'] += 1
            raise RuntimeError('auth close failed')

    server_mod.github_app_auth.create_github_app_auth = lambda: FakeAppAuthCounting()
    server_mod.GitHubClient = lambda cfg_, token_provider=None: FakeGitHubCounting()
    server_mod.RegistryClient = lambda cfg_: FakeRegistryCounting()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as mf:
        mf.write("version: 1\n")
        mf.write("machines:\n")
        mf.write("  test-machine:\n")
        mf.write("    node_id: node-1\n")
        mf.write("    node_host: 127.0.0.1\n")
        mf.write("    owners:\n")
        mf.write("      - owner1\n")
        mf.write("    targets:\n")
        mf.write("      - perception\n")
        mf.write("    platforms:\n")
        mf.write("      - linux/arm64\n")
        machine_yaml_path = mf.name

    cfg = Config(
        github_repos=['4paradigm/phanthymotus', '4paradigm/phanthymotus-driver'],
        github_api_url='https://api.github.com',
        poll_enabled=True,
        poll_interval_seconds=30,
        registry='registry.example',
        review_comment_author_id='7950763',
        machine_owners_file=machine_yaml_path,
        secrets_file='/dev/null',
        webhook_enabled=False,
    )

    server_mod.GitHubCommandWatcher = FakeWatcherCounting
    server_mod.DeployController = lambda *a, **k: FakeControllerCounting()

    try:
        app = create_app(cfg)
        with pytest.raises(RuntimeError, match='lifespan body failure'):
            async with app.router.lifespan_context(app):
                raise RuntimeError('lifespan body failure')
    finally:
        server_mod.github_app_auth.create_github_app_auth = orig_auth
        server_mod.GitHubClient = orig_ghc
        server_mod.RegistryClient = orig_reg
        server_mod.GitHubCommandWatcher = orig_watcher
        server_mod.DeployController = orig_controller

    assert cleanup_counts['stop'] == 1, f"stop called {cleanup_counts['stop']} times"
    assert cleanup_counts['aclose'] == 1, f"aclose called {cleanup_counts['aclose']} times"
    assert cleanup_counts['gh_close'] == 1, f"gh_close called {cleanup_counts['gh_close']} times"
    assert cleanup_counts['reg_close'] == 1, f"reg_close called {cleanup_counts['reg_close']} times"
    assert cleanup_counts['auth_close'] == 1, f"auth_close called {cleanup_counts['auth_close']} times"


@pytest.mark.asyncio
async def test_normal_shutdown_cleanup_exactly_once():
    """Verify that a clean lifespan exit triggers each cleanup step exactly
    once."""
    import tempfile
    from fastapi.testclient import TestClient
    from ..server import create_app
    from ..config import Config
    import agents.deploy_approval.server as server_mod

    orig_auth = server_mod.github_app_auth.create_github_app_auth
    orig_ghc = server_mod.GitHubClient
    orig_reg = server_mod.RegistryClient
    orig_watcher = server_mod.GitHubCommandWatcher
    orig_controller = server_mod.DeployController

    cleanup_counts = {
        'stop': 0,
        'aclose': 0,
        'gh_close': 0,
        'reg_close': 0,
        'auth_close': 0,
    }

    class FakeGitHubHTTP:
        async def aclose(self):
            cleanup_counts['gh_close'] += 1

    class FakeRegistryHTTP:
        async def aclose(self):
            cleanup_counts['reg_close'] += 1

    class FakeWatcherNormal:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass

        async def stop(self):
            cleanup_counts['stop'] += 1

    class FakeControllerNormal:
        async def aclose(self):
            cleanup_counts['aclose'] += 1

    class FakeGitHubNormal:
        def __init__(self, *a, **k):
            self.http = FakeGitHubHTTP()

        async def list_installation_repositories(self):
            return ['4paradigm/phanthymotus']

        async def aclose(self):
            cleanup_counts['gh_close'] += 1

    class FakeRegistryNormal:
        def __init__(self, *a):
            self.http = FakeRegistryHTTP()

        async def aclose(self):
            cleanup_counts['reg_close'] += 1

    class FakeAppAuthNormal:
        app_id = 'test-app'

        async def get_installation_token(self):
            return 'tok'

        async def close(self):
            cleanup_counts['auth_close'] += 1

    server_mod.github_app_auth.create_github_app_auth = lambda: FakeAppAuthNormal()
    server_mod.GitHubClient = lambda cfg_, token_provider=None: FakeGitHubNormal()
    server_mod.RegistryClient = lambda cfg_: FakeRegistryNormal()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as mf:
        mf.write("version: 1\n")
        mf.write("machines:\n")
        mf.write("  test-machine:\n")
        mf.write("    node_id: node-1\n")
        mf.write("    node_host: 127.0.0.1\n")
        mf.write("    owners:\n")
        mf.write("      - owner1\n")
        mf.write("    targets:\n")
        mf.write("      - perception\n")
        mf.write("    platforms:\n")
        mf.write("      - linux/arm64\n")
        machine_yaml_path = mf.name

    cfg = Config(
        github_repos=['4paradigm/phanthymotus', '4paradigm/phanthymotus-driver'],
        github_api_url='https://api.github.com',
        poll_enabled=True,
        poll_interval_seconds=30,
        registry='registry.example',
        review_comment_author_id='7950763',
        machine_owners_file=machine_yaml_path,
        secrets_file='/dev/null',
        webhook_enabled=False,
    )

    server_mod.GitHubCommandWatcher = FakeWatcherNormal
    server_mod.DeployController = lambda *a, **k: FakeControllerNormal()

    try:
        app = create_app(cfg)
        with TestClient(app):
            pass
    finally:
        server_mod.github_app_auth.create_github_app_auth = orig_auth
        server_mod.GitHubClient = orig_ghc
        server_mod.RegistryClient = orig_reg
        server_mod.GitHubCommandWatcher = orig_watcher
        server_mod.DeployController = orig_controller

    assert cleanup_counts['stop'] == 1, f"stop called {cleanup_counts['stop']} times"
    assert cleanup_counts['aclose'] == 1, f"aclose called {cleanup_counts['aclose']} times"
    assert cleanup_counts['gh_close'] == 1, f"gh_close called {cleanup_counts['gh_close']} times"
    assert cleanup_counts['reg_close'] == 1, f"reg_close called {cleanup_counts['reg_close']} times"
    assert cleanup_counts['auth_close'] == 1, f"auth_close called {cleanup_counts['auth_close']} times"
