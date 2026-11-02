"""Regressions for existing-cursor Baseline recovery after HEAD drift.

These tests call the production ``GitHubCommandWatcher._baseline_pr`` method;
GitHub and the Controller's durable boundary are represented by async fakes.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ..github_command_watcher import GitHubCommandWatcher
from .conftest import make_config


REPO = "4paradigm/phanthymotus"
DRIVER = "4paradigm/phanthymotus-driver"
OLD_HEAD = "1" * 40
NEW_HEAD = "2" * 40


def _setup(*, auth=True, repo=REPO, old_head=OLD_HEAD, new_head=NEW_HEAD,
           comments=None, cutoff=None):
    config = make_config()
    config.auth_valid = auth
    config.active_repos = [REPO, DRIVER]
    config.github_repos = [REPO, DRIVER]
    current = {"head_sha": old_head, "status": "reviewing",
               "last_processed_comment_id": 20,
               "command": {"comment_id": 19, "kind": "approve_deploy",
                           "phase": "completed", "args": {}}}
    store = {"state": current}
    pr = {"state": "open", "merged": False, "draft": False,
          "head": {"sha": new_head}, "updated_at": "2026-10-01T00:00:00Z"}
    proxy = SimpleNamespace()
    async def read_state(r, n):
        return store["state"]
    async def get_pr(r, n):
        return pr
    async def comments_read(r, n):
        return list(comments or [])
    async def persist(r, n, cursor):
        store["state"] = {**store["state"], "last_processed_comment_id": cursor}
        return {"last_processed_comment_id": cursor}
    proxy.read_hidden_state = AsyncMock(side_effect=read_state)
    proxy.get_pr = AsyncMock(side_effect=get_pr)
    proxy.get_issue_comments = AsyncMock(side_effect=comments_read)
    proxy.persist_cursor = AsyncMock(side_effect=persist)
    controller = SimpleNamespace()
    async def reconcile(r, n):
        if store["state"]["head_sha"] != pr["head"]["sha"]:
            store["state"] = {**store["state"], "head_sha": pr["head"]["sha"],
                               "status": "review-required", "review_evidence": {},
                               "components": [], "deployments": [],
                               "command": {"comment_id": 20, "kind": "", "phase": "completed", "args": {}},
                               "round_archive": {"round_id": old_head + "-19", "sha256": "a" * 64}}
    controller.reconcile_pr = AsyncMock(side_effect=reconcile)
    watcher = GitHubCommandWatcher(config, proxy, controller)
    watcher._repo_baseline_cutoff_utc[repo] = cutoff or (
        datetime.now(timezone.utc) + timedelta(minutes=1))
    return watcher, proxy, controller, config, store, pr


@pytest.mark.asyncio
async def test_existing_cursor_head_drift_baseline_recovers():
    watcher, proxy, controller, _, store, _ = _setup()
    assert await watcher._baseline_pr(REPO, 347)
    controller.reconcile_pr.assert_awaited_once_with(REPO, 347)
    assert store["state"]["head_sha"] == NEW_HEAD
    assert store["state"]["last_processed_comment_id"] >= 20


@pytest.mark.asyncio
async def test_restart_with_persisted_old_state_recovers():
    first, _, _, cfg, store, pr = _setup()
    watcher, proxy, controller, _, _, _ = _setup()
    watcher.config = cfg
    watcher.proxy = first.proxy
    watcher.controller = first.controller
    assert await watcher._baseline_pr(REPO, 347)
    assert store["state"]["head_sha"] == NEW_HEAD


@pytest.mark.asyncio
async def test_reauthorized_repo_recovers_existing_cursor():
    watcher, _, controller, config, _, _ = _setup(auth=True)
    config.auth_valid = True
    assert await watcher._baseline_pr(REPO, 347)
    controller.reconcile_pr.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconcile_archives_old_round_before_baseline_write():
    watcher, proxy, controller, _, store, _ = _setup()
    await watcher._baseline_pr(REPO, 347)
    assert store["state"]["round_archive"] == {"round_id": OLD_HEAD + "-19", "sha256": "a" * 64}
    assert proxy.persist_cursor.await_count == 1


@pytest.mark.asyncio
async def test_archive_failure_does_not_advance_cursor():
    watcher, proxy, controller, _, store, _ = _setup()
    async def fail(*args):
        raise RuntimeError("archive failed")
    controller.reconcile_pr.side_effect = fail
    assert not await watcher._baseline_pr(REPO, 347)
    proxy.persist_cursor.assert_not_awaited()
    assert store["state"]["last_processed_comment_id"] == 20


@pytest.mark.asyncio
async def test_lifecycle_write_failure_does_not_advance_cursor():
    watcher, proxy, controller, _, store, _ = _setup()
    async def failed_write(*args):
        store["state"] = {**store["state"], "head_sha": NEW_HEAD}
        # Simulates archive success followed by failed Lifecycle write.
    controller.reconcile_pr.side_effect = failed_write
    assert not await watcher._baseline_pr(REPO, 347)
    proxy.persist_cursor.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_is_followed_by_fresh_pr_and_state_reads():
    watcher, proxy, controller, _, _, _ = _setup()
    await watcher._baseline_pr(REPO, 347)
    assert proxy.read_hidden_state.await_count >= 4
    assert proxy.get_pr.await_count >= 3


@pytest.mark.asyncio
async def test_second_head_change_fails_closed():
    watcher, proxy, controller, _, _, pr = _setup()
    calls = 0
    async def moving_pr(r, n):
        nonlocal calls
        calls += 1
        if calls >= 3:
            pr["head"]["sha"] = "3" * 40
        return pr
    proxy.get_pr.side_effect = moving_pr
    assert not await watcher._baseline_pr(REPO, 347)
    proxy.persist_cursor.assert_not_awaited()


@pytest.mark.asyncio
async def test_cursor_never_moves_backwards():
    watcher, proxy, _, _, store, _ = _setup()
    assert await watcher._baseline_pr(REPO, 347)
    assert store["state"]["last_processed_comment_id"] >= 20


@pytest.mark.asyncio
async def test_command_after_authorization_cutoff_remains_unconsumed():
    cutoff = datetime.now(timezone.utc)
    watcher, proxy, _, _, store, _ = _setup(cutoff=cutoff, comments=[
        {"id": 21, "created_at": (cutoff + timedelta(seconds=5)).isoformat(),
         "body": "/approve_deploy machine=x"},
    ])
    assert await watcher._baseline_pr(REPO, 347)
    assert store["state"]["last_processed_comment_id"] == 20


@pytest.mark.asyncio
async def test_old_approval_is_not_dispatched_during_baseline():
    watcher, _, controller, _, _, _ = _setup()
    assert await watcher._baseline_pr(REPO, 347)
    assert not hasattr(controller, "on_command")


@pytest.mark.asyncio
async def test_same_head_does_not_reconcile_or_archive_again():
    watcher, _, controller, _, _, _ = _setup(old_head=NEW_HEAD)
    assert await watcher._baseline_pr(REPO, 347)
    controller.reconcile_pr.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [PermissionError("403"), RuntimeError("429"), TimeoutError()])
async def test_github_read_failures_fail_closed(error):
    watcher, proxy, _, _, _, _ = _setup()
    proxy.get_pr.side_effect = error
    assert not await watcher._baseline_pr(REPO, 347)
    proxy.persist_cursor.assert_not_awaited()


@pytest.mark.asyncio
async def test_priority_baseline_retry_can_recover_on_next_attempt():
    watcher, proxy, controller, _, _, _ = _setup()
    recover = controller.reconcile_pr.side_effect
    controller.reconcile_pr.side_effect = RuntimeError("transient archive read")
    assert not await watcher._baseline_pr(REPO, 347)
    controller.reconcile_pr.side_effect = recover
    assert await watcher._baseline_pr(REPO, 347)


@pytest.mark.asyncio
@pytest.mark.parametrize("archive", [None, {"round_id": "f" * 40 + "-19", "sha256": "a" * 64},
                                     {"round_id": OLD_HEAD + "-19", "sha256": "bad"}])
async def test_missing_or_invalid_archive_fails_closed(archive):
    watcher, proxy, controller, _, store, _ = _setup()
    reconcile = controller.reconcile_pr.side_effect
    async def incomplete(*args):
        await reconcile(*args)
        store["state"]["round_archive"] = archive
    controller.reconcile_pr.side_effect = incomplete
    assert not await watcher._baseline_pr(REPO, 347)
    proxy.persist_cursor.assert_not_awaited()


@pytest.mark.asyncio
async def test_swallowed_reconcile_error_does_not_advance_cursor():
    watcher, proxy, controller, _, _, _ = _setup()
    controller.reconcile_pr.side_effect = None
    assert not await watcher._baseline_pr(REPO, 347)
    proxy.persist_cursor.assert_not_awaited()


@pytest.mark.asyncio
async def test_core_recovery_does_not_touch_driver_lifecycle():
    watcher, _, _, _, store, _ = _setup()
    driver_state = {"head_sha": OLD_HEAD, "last_processed_comment_id": 8}
    assert await watcher._baseline_pr(REPO, 347)
    assert driver_state == {"head_sha": OLD_HEAD, "last_processed_comment_id": 8}
    assert store["state"]["head_sha"] == NEW_HEAD


@pytest.mark.asyncio
async def test_revoked_authorization_prevents_reconcile_and_cursor_write():
    watcher, proxy, controller, config, store, _ = _setup()
    config.auth_valid = False
    assert not await watcher._baseline_pr(REPO, 347)
    controller.reconcile_pr.assert_not_awaited()
    proxy.persist_cursor.assert_not_awaited()
    assert store["state"]["last_processed_comment_id"] == 20


@pytest.mark.asyncio
async def test_cursor_write_failure_does_not_report_baseline_success():
    watcher, proxy, _, _, store, _ = _setup()
    proxy.persist_cursor.side_effect = TimeoutError("GitHub write timeout")
    assert not await watcher._baseline_pr(REPO, 347)
    assert store["state"]["last_processed_comment_id"] == 20


@pytest.mark.asyncio
async def test_missing_build_context_remains_review_required():
    watcher, _, _, _, store, _ = _setup()
    assert await watcher._baseline_pr(REPO, 347)
    assert store["state"]["status"] == "review-required"
