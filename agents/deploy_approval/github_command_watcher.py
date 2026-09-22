"""GitHub command watcher — polls PR comments using `POLL_INTERVAL_SECONDS`.

This is the ONLY active poller for Deploy Approval. It polls configured
repositories/open PRs, reads comments, identifies commands newer than the
hidden-state cursor, and dispatches them one at a time to the Deploy Controller.

No SQLite, no Redis, no local cursor file, no comment lease, no DB persistence.
Single serial worker — no asyncio.gather for mutating PR work.

Safety rules:
- Never write stale local hidden state over Controller output.
- Cursor-only mutation must fresh-read hidden state from GitHub and preserve
  exact visible markdown.
- After dispatching one valid command, stop processing that PR for this cycle.
- Next cycle starts with fresh GitHub state.
- Transient command failure -> no cursor advance and no later command processing
  in same PR cycle.
"""

from __future__ import annotations

import asyncio
import logging

from . import commands as commands_mod
from .config import Config
from .github_client import GitHubError
from .github_state_proxy import GitHubStateProxy, GitHubStateProxyError

logger = logging.getLogger(__name__)


class DeployCommandError(Exception):
    """Raised when a command cannot be processed."""


def _needs_initial_comment_baseline(state: dict | None) -> bool:
    """Determine if a PR needs an initial comment baseline cycle.

    Returns True when:
    1. state is None (never observed)
    2. state exists but is the "empty reconcile" state:
       last_processed_comment_id == 0
       command.comment_id == 0
       command.kind == ""
       command.phase == "completed"
       command.args == {}

    This allows recovery when baseline persistence failed silently.
    """
    if state is None:
        return True
    if not isinstance(state, dict):
        return False
    if state.get("last_processed_comment_id") != 0:
        return False
    cmd = state.get("command")
    if not isinstance(cmd, dict):
        return False
    if cmd.get("comment_id") != 0:
        return False
    if cmd.get("kind") != "":
        return False
    if cmd.get("phase") != "completed":
        return False
    if cmd.get("args") != {}:
        return False
    return True


class GitHubCommandWatcher:
    """Polls PR comments using `POLL_INTERVAL_SECONDS` and dispatches commands.

    Exactly one serial command worker. Processes comments in ascending order.
    """

    def __init__(
        self,
        config: Config,
        proxy: GitHubStateProxy,
        controller: object,
    ):
        self.config = config
        self.proxy = proxy
        self.controller = controller
        self._task: asyncio.Task | None = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self._poll_once()
            except Exception as e:
                logger.error("command watcher poll cycle failed: %s", e)
            await asyncio.sleep(self.config.poll_interval_seconds)

    async def _poll_once(self) -> None:
        """One poll cycle over all configured repos and open PRs.

        Processes repos/PRs/comments deterministically and sequentially.
        No asyncio.gather for mutating PR work.
        """
        for repo in self.config.github_repos:
            try:
                prs = await self.proxy.get_open_prs(repo)
            except GitHubError as e:
                logger.warning("watcher list PRs %s: %s", repo, e)
                continue
            for pr in prs:
                pr_number = int(pr.get("number") or 0)
                if not pr_number:
                    continue
                try:
                    await self._process_pr(repo, pr_number)
                except Exception as e:
                    logger.error(
                        "watcher process %s#%s: %s", repo, pr_number, e
                    )

    async def _process_pr(self, repo: str, pr_number: int) -> None:
        """Process commands for one PR.

        Reads hidden state, fetches comments, processes in ascending id order.
        Transient failures stop processing later commands in this PR for this cycle.

        After one recognized command is dispatched, stops processing that PR
        for the current poll cycle.

        First-observation safety: when no Deploy Approval hidden state exists
        yet, baseline the cursor to the maximum comment id observed so GitHub
        will not replay historical commands on a future restart.

        Baseline-incomplete recovery: when hidden state exists but still
        represents an un-consumed reconcile-initial state (cursor=0, empty
        command), the watcher re-runs the baseline cycle.  This prevents
        historical commands from being replayed after a transient
        persist_cursor failure.
        """
        initial_state = await self.proxy.read_hidden_state(repo, pr_number)

        if _needs_initial_comment_baseline(initial_state):
            # --- First-observation / baseline-incomplete path ---
            pr_info = await self.proxy.get_pr(repo, pr_number)
            if not pr_info or pr_info.get("state") != "open":
                # Closed / merged PR without usable state — reconcile if
                # appropriate but do NOT dispatch any commands.
                if initial_state is None:
                    await self.controller.reconcile_pr(repo, pr_number)
                return

            # BEFORE reconcile: snapshot current comments.
            try:
                comments_before = await self.proxy.get_issue_comments(repo, pr_number)
            except Exception as e:
                logger.warning(
                    "watcher baseline snapshot-before %s#%s: %s — fail closed",
                    repo, pr_number, e,
                )
                return

            max_cid_before = 0
            for c in comments_before:
                cid = c.get("id")
                if isinstance(cid, int) and not isinstance(cid, bool) and cid > max_cid_before:
                    max_cid_before = cid

            # Reconcile (may create lifecycle hidden state).
            try:
                await self.controller.reconcile_pr(repo, pr_number)
            except Exception as e:
                logger.warning(
                    "watcher reconcile %s#%s: %s — skip baseline this cycle",
                    repo, pr_number, e,
                )
                return

            # After reconcile: fresh read state.
            state_after = await self.proxy.read_hidden_state(repo, pr_number)
            if state_after is None:
                # reconcile did not create state — abort this cycle
                return

            # AFTER reconcile: second snapshot to capture lifecycle bot
            # comments created by reconcile, plus any race new comments.
            try:
                comments_after = await self.proxy.get_issue_comments(repo, pr_number)
            except Exception as e:
                logger.warning(
                    "watcher baseline snapshot-after %s#%s: %s — fail closed",
                    repo, pr_number, e,
                )
                return

            max_cid_after = 0
            for c in comments_after:
                cid = c.get("id")
                if isinstance(cid, int) and not isinstance(cid, bool) and cid > max_cid_after:
                    max_cid_after = cid

            baseline_id = max(max_cid_before, max_cid_after)

            # Persist baseline cursor.
            try:
                persisted = await self.proxy.persist_cursor(repo, pr_number, baseline_id)
            except Exception as e:
                logger.warning(
                    "watcher persist baseline %s#%s: %s — fail closed",
                    repo, pr_number, e,
                )
                return

            # Verify persistence succeeded at least as far as requested.
            if not isinstance(persisted, dict):
                logger.warning(
                    "watcher persist baseline %s#%s returned %s — fail closed",
                    repo, pr_number, type(persisted).__name__,
                )
                return
            persisted_cid = persisted.get("last_processed_comment_id")
            if not isinstance(persisted_cid, int) or isinstance(persisted_cid, bool):
                logger.warning(
                    "watcher persist baseline %s#%s invalid cursor — fail closed",
                    repo, pr_number,
                )
                return
            if persisted_cid < baseline_id:
                logger.warning(
                    "watcher persist baseline %s#%s cursor %s < requested %s — fail closed",
                    repo, pr_number, persisted_cid, baseline_id,
                )
                return

            # Baseline persisted successfully — but DO NOT dispatch commands
            # in this cycle. Next cycle will process comments > baseline_id.
            return

        # Existing state PR — normal processing path.
        await self.controller.reconcile_pr(repo, pr_number)
        state = await self.proxy.read_hidden_state(repo, pr_number)
        if state is None:
            return
        cursor = state.get("last_processed_comment_id", 0)

        # Fetch comments
        try:
            comments = await self.proxy.get_issue_comments(repo, pr_number)
        except GitHubStateProxyError as e:
            logger.warning("watcher get comments %s#%s: %s", repo, pr_number, e)
            return

        # Filter new comments, sort ascending
        new_comments = [
            c for c in comments
            if isinstance(c.get("id"), int) and not isinstance(c.get("id"), bool) and c["id"] > cursor
        ]
        new_comments.sort(key=lambda c: c["id"])

        for c in new_comments:
            comment_id = c["id"]

            # Skip bot's own lifecycle comments — fresh-read body to avoid stale-body TOCTOU
            try:
                fresh_comment_obj = await self.proxy.get_comment(repo, comment_id)
            except Exception as e:
                logger.warning(
                    "watcher fresh-read comment %s#%s #%s: %s — skip this cycle",
                    repo, pr_number, comment_id, e,
                )
                return

            if fresh_comment_obj is None:
                # Comment was deleted between list and get — skip this cycle
                return

            if self.proxy.is_bot_comment(fresh_comment_obj):
                # Use cursor-only persistence to preserve lifecycle markdown
                if state.get("head_sha"):
                    state = await self.proxy.persist_cursor(
                        repo, pr_number, comment_id,
                    ) or state
                continue

            # Validate fresh comment ID matches candidate
            fresh_id = fresh_comment_obj.get("id")
            if not isinstance(fresh_id, int) or isinstance(fresh_id, bool) or fresh_id != comment_id:
                logger.warning(
                    "watcher fresh comment id %s (type=%s) != candidate %s — skip this cycle",
                    fresh_id, type(fresh_id).__name__, comment_id,
                )
                return

            fresh_body = fresh_comment_obj.get("body", "")
            if not isinstance(fresh_body, str) or not fresh_body.strip():
                if state.get("head_sha"):
                    state = await self.proxy.persist_cursor(
                        repo, pr_number, comment_id,
                    ) or state
                continue

            if not commands_mod.command_starts_line_any(fresh_body):
                # Non-command comment — use cursor-only persistence
                if state.get("head_sha"):
                    state = await self.proxy.persist_cursor(
                        repo, pr_number, comment_id,
                    ) or state
                continue

            cmd = commands_mod.parse_command(fresh_body)
            if not cmd.is_command:
                if state.get("head_sha"):
                    state = await self.proxy.persist_cursor(
                        repo, pr_number, comment_id,
                    ) or state
                continue

            # Dispatch to controller — transient failure must NOT advance cursor
            try:
                handled = await self.controller.on_command(
                    cmd, repo, pr_number, comment_id,
                )
                if handled:
                    await self._consume_comment_cursor(repo, pr_number, comment_id)
                    # After one command is dispatched, stop processing this PR
                    # for this cycle. Next cycle starts with fresh GitHub state.
                    return
            except DeployCommandError as e:
                logger.warning(
                    "command %s comment %s in %s#%s: %s",
                    cmd.kind, comment_id, repo, pr_number, e,
                )
                # Invalid/unauthorized commands are terminally handled — advance cursor
                # only when an authoritative lifecycle already exists.
                await self._consume_comment_cursor(repo, pr_number, comment_id)
                return
            except Exception as e:
                logger.error(
                    "unexpected error processing command %s comment %s: %s",
                    cmd.kind, comment_id, e,
                )
                # Transient error — do NOT advance cursor, stop processing this PR
                return

    async def _consume_comment_cursor(self, repo: str, pr_number: int, comment_id: int) -> None:
        """Advance the command cursor without mutating lifecycle state."""
        try:
            state = await self.proxy.read_hidden_state(repo, pr_number)
        except Exception as e:
            logger.warning("watcher consume cursor read %s#%s: %s", repo, pr_number, e)
            return
        if state is not None:
            await self.proxy.persist_cursor(repo, pr_number, comment_id)
            return
