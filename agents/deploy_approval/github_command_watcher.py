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
        """
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
