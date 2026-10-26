from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from starlette.exceptions import HTTPException

from ..github_client import (
    GitHubClient, GitHubPermissionError, GitHubRateLimitError,
    GitHubTransientError,
)
from ..github_command_watcher import GitHubCommandWatcher
from ..router_webhook import webhook


def _request(body: bytes, *, event: str, delivery: str = "delivery-1", app=None):
    class FakeRequest:
        headers = {
            "X-GitHub-Event": event,
            "X-GitHub-Delivery": delivery,
            "X-Hub-Signature-256": "sha256=valid",
        }

        def __init__(self):
            self.app = app

        async def stream(self):
            yield body

    return FakeRequest()


@pytest.mark.asyncio
async def test_webhook_enqueues_once_without_comment_api_read(config):
    config.webhook_enabled = True
    config.github_webhook_secret = "offline-secret"
    config.github_repos = ["4paradigm/phanthymotus-driver"]
    config.active_repos = list(config.github_repos)
    config.auth_valid = True
    watcher = SimpleNamespace(enqueue_pr=MagicMock(return_value=True))
    app = SimpleNamespace(state=SimpleNamespace(config=config, watcher=watcher))
    payload = {
        "action": "created",
        "repository": {"full_name": "4paradigm/phanthymotus-driver"},
        "issue": {"number": 442, "pull_request": {"url": "ignored"}},
        "comment": {"id": 6406856003, "body": "/request_deploy"},
    }
    request = _request(json.dumps(payload).encode(), event="issue_comment", app=app)
    with patch("agents.deploy_approval.router_webhook._verify_signature_impl", return_value=True):
        assert await webhook(request) == {"status": "queued"}
        assert await webhook(request) == {"status": "duplicate"}
    watcher.enqueue_pr.assert_called_once_with("4paradigm/phanthymotus-driver", 442)


@pytest.mark.asyncio
async def test_new_pull_request_event_enqueues_for_initialization(config):
    repo = "4paradigm/phanthymotus-driver"
    config.webhook_enabled = True
    config.github_repos = [repo]
    config.active_repos = [repo]
    config.auth_valid = True
    watcher = SimpleNamespace(enqueue_pr=MagicMock(return_value=True))
    app = SimpleNamespace(state=SimpleNamespace(config=config, watcher=watcher))
    payload = {"action": "opened", "repository": {"full_name": repo}, "pull_request": {"number": 501}}
    with patch("agents.deploy_approval.router_webhook._verify_signature_impl", return_value=True):
        assert await webhook(_request(json.dumps(payload).encode(), event="pull_request", app=app)) == {
            "status": "queued"
        }
    watcher.enqueue_pr.assert_called_once_with(repo, 501)


@pytest.mark.asyncio
async def test_review_agent_review_event_queues_only(config):
    repo = "4paradigm/phanthymotus-driver"
    config.webhook_enabled = True
    config.github_repos = [repo]
    config.active_repos = [repo]
    config.auth_valid = True
    watcher = SimpleNamespace(enqueue_pr=MagicMock(return_value=True))
    app = SimpleNamespace(state=SimpleNamespace(config=config, watcher=watcher))
    payload = {
        "action": "submitted", "repository": {"full_name": repo},
        "pull_request": {"number": 501}, "review": {"body": "ignored by webhook"},
    }
    with patch("agents.deploy_approval.router_webhook._verify_signature_impl", return_value=True):
        assert await webhook(_request(json.dumps(payload).encode(), event="pull_request_review", app=app)) == {
            "status": "queued"
        }
    watcher.enqueue_pr.assert_called_once_with(repo, 501)


@pytest.mark.asyncio
async def test_priority_pr_is_fresh_read_before_open_pr_scan(config):
    repo = "4paradigm/phanthymotus-driver"
    config.github_repos = [repo]
    config.active_repos = [repo]
    events = []
    proxy = MagicMock()

    async def get_pr(_repo, _number):
        events.append("get_pr")
        return {"state": "open", "merged": False, "draft": False}

    async def list_prs(_repo):
        events.append("list_open_prs")
        return []

    proxy.get_pr = AsyncMock(side_effect=get_pr)
    proxy.get_open_prs = AsyncMock(side_effect=list_prs)
    proxy.ensure_visible_evidence_archive_comments = AsyncMock()
    proxy.retire_legacy_evidence_link_replies = AsyncMock()
    proxy.read_hidden_state = AsyncMock(return_value={"head_sha": "a" * 40})
    proxy.find_trusted_lifecycle_comment = AsyncMock(return_value=None)
    controller = SimpleNamespace(project_evidence=AsyncMock(side_effect=lambda *_: events.append("project")))
    watcher = GitHubCommandWatcher(config, proxy, controller)
    watcher._process_pr = AsyncMock(side_effect=lambda *_: events.append("process"))
    watcher._validated_current_summary_cos_links = AsyncMock(return_value=set())
    assert watcher.enqueue_pr(repo, 442)

    await watcher._poll_once()

    assert events.index("process") < events.index("list_open_prs")
    assert not watcher._priority_pr_queue
    watcher._process_pr.assert_awaited_once_with(repo, 442)


@pytest.mark.asyncio
async def test_unchanged_pr_skips_repeated_evidence_projection(config):
    repo = "4paradigm/phanthymotus-driver"
    config.github_repos = [repo]
    config.active_repos = [repo]
    proxy = MagicMock()
    proxy.get_pr = AsyncMock(return_value={
        "state": "open", "merged": False, "draft": False,
        "head": {"sha": "a" * 40}, "updated_at": "2026-10-26T00:00:00Z",
    })
    proxy.ensure_visible_evidence_archive_comments = AsyncMock()
    proxy.retire_legacy_evidence_link_replies = AsyncMock()
    controller = SimpleNamespace(project_evidence=AsyncMock())
    watcher = GitHubCommandWatcher(config, proxy, controller)
    watcher._process_pr = AsyncMock()
    watcher._validated_current_summary_cos_links = AsyncMock(return_value=set())

    await watcher._process_scheduled_pr(repo, 442)
    await watcher._process_scheduled_pr(repo, 442)
    proxy.get_pr.return_value["updated_at"] = "2026-10-26T00:01:00Z"
    await watcher._process_scheduled_pr(repo, 442)

    assert controller.project_evidence.await_count == 2
    assert proxy.ensure_visible_evidence_archive_comments.await_count == 2


def test_priority_queue_prefers_newest_unique_pr_and_deduplicates(config):
    repo = "4paradigm/phanthymotus-driver"
    config.github_repos = [repo]
    config.active_repos = [repo]
    watcher = GitHubCommandWatcher(config, MagicMock(), MagicMock())

    assert watcher.enqueue_pr(repo, 442)
    assert watcher.enqueue_pr(repo, 443)
    assert watcher.enqueue_pr(repo, 442)

    assert list(watcher._priority_pr_queue) == [(repo, 442), (repo, 443)]
    assert len(watcher._priority_pr_set) == 2


@pytest.mark.asyncio
async def test_large_repo_fallback_rotation_is_bounded_and_repo_fair(config):
    main = "4paradigm/phanthymotus"
    driver = "4paradigm/phanthymotus-driver"
    config.github_repos = [main, driver]
    config.active_repos = [main, driver]
    prs = {
        main: [{"number": number, "state": "open", "updated_at": "2026-10-26T00:00:00Z"}
               for number in range(1, 27)],
        driver: [{"number": number, "state": "open", "updated_at": "2026-10-26T00:00:00Z"}
                 for number in range(1, 130)],
    }
    proxy = MagicMock()
    proxy.get_open_prs = AsyncMock(side_effect=lambda repo: prs[repo])
    proxy.get_pr = AsyncMock(return_value={"state": "open", "merged": False, "draft": False})
    proxy.ensure_visible_evidence_archive_comments = AsyncMock()
    proxy.retire_legacy_evidence_link_replies = AsyncMock()
    proxy.read_hidden_state = AsyncMock(return_value={"head_sha": "a" * 40})
    proxy.find_trusted_lifecycle_comment = AsyncMock(return_value=None)
    seen = {main: [], driver: []}
    controller = SimpleNamespace(project_evidence=AsyncMock())
    watcher = GitHubCommandWatcher(config, proxy, controller)
    async def process(repo, number):
        seen[repo].append(number)
    watcher._process_pr = AsyncMock(side_effect=process)
    watcher._validated_current_summary_cos_links = AsyncMock(return_value=set())

    for _ in range(70):
        before_main, before_driver = len(seen[main]), len(seen[driver])
        await watcher._poll_once()
        assert len(seen[main]) - before_main <= 8
        assert len(seen[driver]) - before_driver <= 8
        if 26 in seen[main] and 129 in seen[driver]:
            break

    assert 26 in seen[main]
    assert 129 in seen[driver]


@pytest.mark.asyncio
async def test_closed_priority_pr_is_not_written(config):
    repo = "4paradigm/phanthymotus-driver"
    config.github_repos = [repo]
    config.active_repos = [repo]
    proxy = MagicMock()
    proxy.get_pr = AsyncMock(return_value={"state": "closed", "merged": False, "draft": False})
    proxy.get_open_prs = AsyncMock(return_value=[])
    controller = SimpleNamespace(project_evidence=AsyncMock())
    watcher = GitHubCommandWatcher(config, proxy, controller)
    watcher._process_pr = AsyncMock()
    assert watcher.enqueue_pr(repo, 442)

    await watcher._poll_once()

    watcher._process_pr.assert_not_awaited()
    controller.project_evidence.assert_not_awaited()
    assert not watcher._priority_pr_queue


@pytest.mark.asyncio
async def test_priority_baseline_requeues_for_new_comments_next_turn(config):
    repo = "4paradigm/phanthymotus-driver"
    config.github_repos = [repo]
    config.active_repos = [repo]
    proxy = MagicMock()
    proxy.get_pr = AsyncMock(return_value={"state": "open", "merged": False, "draft": False})
    proxy.get_open_prs = AsyncMock(return_value=[])
    proxy.ensure_visible_evidence_archive_comments = AsyncMock()
    proxy.retire_legacy_evidence_link_replies = AsyncMock()
    proxy.read_hidden_state = AsyncMock(return_value={"head_sha": "a" * 40})
    proxy.find_trusted_lifecycle_comment = AsyncMock(return_value=None)
    controller = SimpleNamespace(project_evidence=AsyncMock())
    watcher = GitHubCommandWatcher(config, proxy, controller)
    watcher._pending_baseline_repos.add(repo)
    watcher._baseline_progress[repo] = set()
    watcher._baseline_pr = AsyncMock(return_value=True)
    watcher._process_pr = AsyncMock()
    watcher._validated_current_summary_cos_links = AsyncMock(return_value=set())
    assert watcher.enqueue_pr(repo, 442)

    await watcher._poll_once()

    watcher._baseline_pr.assert_awaited_once_with(repo, 442)
    watcher._process_pr.assert_not_awaited()
    assert list(watcher._priority_pr_queue) == [(repo, 442)]

    await watcher._poll_once()

    watcher._process_pr.assert_awaited_once_with(repo, 442)
    assert not watcher._priority_pr_queue


@pytest.mark.asyncio
async def test_secondary_rate_limit_keeps_authorization_and_backs_off(config):
    calls = []

    async def handler(request):
        calls.append(request)
        return httpx.Response(
            403,
            headers={"Retry-After": "17", "X-RateLimit-Remaining": "0"},
            json={"message": "You have exceeded a secondary rate limit."},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = GitHubClient(config, http=http, token_provider=AsyncMock(return_value="offline-token"))
        with pytest.raises(GitHubRateLimitError) as caught:
            await client._request("GET", "/repos/4paradigm/phanthymotus/pulls/1")
        assert caught.value.retry_after_seconds >= 17
        assert config.auth_valid is True
        with pytest.raises(GitHubRateLimitError):
            await client._request("GET", "/repos/4paradigm/phanthymotus/pulls/1")
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_rate_limited_authorization_refresh_does_not_start_new_baseline(config):
    from ..github_command_watcher import GitHubCommandWatcher

    repo = "4paradigm/phanthymotus-driver"
    config.github_repos = [repo]
    config.active_repos = [repo]
    config.auth_valid = True
    github = MagicMock()
    github.list_installation_repositories = AsyncMock(
        side_effect=GitHubRateLimitError(status_code=403, retry_after_seconds=30)
    )
    auth = MagicMock()
    auth.refresh_installation_token = AsyncMock(return_value="offline-token")
    watcher = GitHubCommandWatcher(config, MagicMock(), MagicMock(), github=github, github_auth=auth)

    await watcher._refresh_active_repos()

    assert config.auth_valid is True
    assert config.active_repos == [repo]
    assert repo not in watcher._pending_baseline_repos


@pytest.mark.asyncio
async def test_permanent_permission_403_closes_authorization(config):
    async def handler(request):
        return httpx.Response(
            403,
            json={"message": "Resource not accessible by integration"},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = GitHubClient(config, http=http, token_provider=AsyncMock(return_value="offline-token"))
        with pytest.raises(GitHubPermissionError):
            await client._request("GET", "/repos/4paradigm/phanthymotus/pulls/1")
    assert config.auth_valid is False


@pytest.mark.asyncio
async def test_401_closes_authorization(config):
    async def handler(request):
        return httpx.Response(401, json={"message": "Bad credentials"}, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = GitHubClient(config, http=http, token_provider=AsyncMock(return_value="offline-token"))
        with pytest.raises(GitHubPermissionError):
            await client._request("GET", "/repos/4paradigm/phanthymotus/pulls/1")
    assert config.auth_valid is False


@pytest.mark.asyncio
async def test_429_retries_after_without_closing_authorization(config):
    async def handler(request):
        return httpx.Response(429, headers={"Retry-After": "9"}, json={}, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = GitHubClient(config, http=http, token_provider=AsyncMock(return_value="offline-token"))
        with pytest.raises(GitHubRateLimitError) as caught:
            await client._request("GET", "/repos/4paradigm/phanthymotus/pulls/1")
    assert caught.value.retry_after_seconds >= 9
    assert config.auth_valid is True


@pytest.mark.asyncio
async def test_5xx_uses_bounded_backoff_without_closing_authorization(config):
    calls = []

    async def handler(request):
        calls.append(request)
        return httpx.Response(503, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = GitHubClient(config, http=http, token_provider=AsyncMock(return_value="offline-token"))
        with pytest.raises(GitHubTransientError):
            await client._request("GET", "/repos/4paradigm/phanthymotus/pulls/1")
        with pytest.raises(GitHubTransientError):
            await client._request("GET", "/repos/4paradigm/phanthymotus/pulls/1")
    assert len(calls) == 1
    assert config.auth_valid is True


@pytest.mark.asyncio
async def test_transport_timeout_uses_backoff(config):
    async def handler(request):
        raise httpx.ConnectTimeout("offline", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = GitHubClient(config, http=http, token_provider=AsyncMock(return_value="offline-token"))
        with pytest.raises(GitHubTransientError):
            await client._request("GET", "/repos/4paradigm/phanthymotus/pulls/1")
    assert client.diagnostics()["backing_off"] is True
    assert config.auth_valid is True


@pytest.mark.asyncio
async def test_open_pr_listing_reuses_authorized_304_snapshot(config):
    calls = []
    pr_page = [{"number": 3, "state": "open", "updated_at": "2026-10-26T00:00:00Z"}]

    async def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(200, headers={"ETag": '"open-prs-v1"'}, json=pr_page, request=request)
        assert request.headers.get("if-none-match") == '"open-prs-v1"'
        return httpx.Response(304, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = GitHubClient(config, http=http, token_provider=AsyncMock(return_value="offline-token"))
        first = await client.list_open_prs("4paradigm/phanthymotus")
        second = await client.list_open_prs("4paradigm/phanthymotus")
    assert first == second == pr_page
    assert len(calls) == 2
