"""Poller — periodically asks GitHub for new PR comments containing the trigger.

This is the default trigger mechanism. It needs only outbound network access,
so the agent can run behind NAT with no public IP, no open ports, and no
webhook configuration.

One API call per repo per cycle:
    GET /repos/{owner}/{repo}/issues/comments?since=<ts>

That endpoint returns issue *and* PR comments (a PR is an issue underneath),
so PR comments are filtered by looking for "/pull/" in html_url.
"""

import asyncio
import json
import logging
import re
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import Config
from .github_client import GitHubClient
from .trigger import create_job_from_comment

logger = logging.getLogger(__name__)

# Extract PR number from an API issue_url, e.g.
#   https://api.github.com/repos/owner/repo/issues/123
ISSUE_URL_PATTERN = re.compile(r"/issues/(\d+)$")

# Overlap the `since` window slightly so comments landing mid-request are not
# missed. Duplicates are filtered by comment ID.
POLL_OVERLAP_SECONDS = 10

# Cap on remembered comment IDs (per agent, not per repo).
PROCESSED_IDS_MAX = 1000

# Pages of 100 comments to walk per repo per cycle.
MAX_PAGES_PER_POLL = 5

# How often to look for merge commits of PRs already reviewed. Much slower than
# the trigger poll: nothing waits on a merge id.
#
# The pass is skipped when no reviewed PR is missing one. A PR that is still open
# — or was closed without merging — never gets one, so it keeps the pass alive at
# one request per repo per interval. That is the intended cost: it is also what
# picks up a merge that happens weeks after the review.
MERGE_BACKFILL_INTERVAL_SECONDS = 300

# PRs per repo examined in one backfill pass (one request, newest-updated first).
MERGE_BACKFILL_PER_PAGE = 100


class Poller:
    """Polls GitHub for trigger comments and enqueues review jobs."""

    def __init__(
        self,
        config: Config,
        github_client: GitHubClient,
        job_queue,
        store,
    ):
        self._config = config
        self._github = github_client
        self._job_queue = job_queue
        self._store = store
        self._task: asyncio.Task | None = None
        self._state_path = Path(config.data_dir) / "poller_state.json"

        # repo_full_name -> ISO8601 timestamp of last successful check
        self._last_checked: dict[str, str] = {}
        # Bounded record of already-handled comment IDs
        self._processed_ids: deque[int] = deque(maxlen=PROCESSED_IDS_MAX)
        self._processed_set: set[int] = set()

        # Stats for /status
        self.poll_count = 0
        self.last_poll_at: str | None = None
        self.last_error: str | None = None
        self.triggers_found = 0
        # Merge-commit backfill. Monotonic, so a clock change cannot stall it;
        # 0.0 means the first cycle runs one, which catches PRs merged while the
        # agent was down.
        self._last_backfill = 0.0
        self.merge_commits_found = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self):
        """Load state and start the polling loop."""
        self._load_state()
        self._task = asyncio.create_task(self._loop(), name="poller")
        logger.info(
            f"Poller started (interval={self._config.poll_interval_seconds}s, "
            f"repos={list(self._config.repos)})"
        )

    async def stop(self):
        """Stop the polling loop and persist state."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._save_state()

    # ── Polling loop ──────────────────────────────────────────────────────────

    async def _loop(self):
        while True:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.last_error = str(e)
                logger.exception("Poll cycle failed")

            try:
                await asyncio.sleep(self._config.poll_interval_seconds)
            except asyncio.CancelledError:
                break

    async def _poll_once(self):
        """Run one poll cycle across all configured repos."""
        for repo in self._config.repos:
            try:
                await self._poll_repo(repo)
            except Exception as e:
                # One repo failing must not stop the others.
                self.last_error = f"{repo}: {e}"
                logger.error(f"Failed to poll {repo}: {e}")

        self.poll_count += 1
        self.last_poll_at = _now_iso()
        self._save_state()

        # Piggybacks on the same loop rather than owning a task: it shares the
        # repo list and the client, and runs far less often than it.
        if time.monotonic() - self._last_backfill >= MERGE_BACKFILL_INTERVAL_SECONDS:
            self._last_backfill = time.monotonic()
            for repo in self._config.repos:
                try:
                    await self._backfill_merge_commits(repo)
                except Exception as e:
                    logger.warning(f"Merge backfill failed for {repo}: {e}")

    async def _backfill_merge_commits(self, repo: str):
        """Record the merge commit of PRs that have been reviewed and merged.

        The reviewed commit and the commit that names the built image are both
        local to a throwaway worktree; this is the id that survives on the base
        branch, and the one to search for when tracing a release to a change.
        """
        pending = await self._store.prs_missing_merge_commit(repo)
        if not pending:
            return

        pulls = await self._github.list_pulls(
            repo, state="closed", per_page=MERGE_BACKFILL_PER_PAGE
        )
        found = 0
        for pr in pulls:
            number = pr.get("number")
            if number not in pending:
                continue
            # Only a merged PR has a real merge commit. On an open one the field
            # holds a throwaway test-merge sha that would be wrong to record.
            merged_at = pr.get("merged_at")
            sha = pr.get("merge_commit_sha") or ""
            if not merged_at or not sha:
                continue
            if await self._store.set_merge_commit(repo, number, sha, merged_at):
                found += 1
                logger.info(
                    f"Recorded merge commit for {repo}#{number}: {sha[:7]}")
        self.merge_commits_found += found

    async def _poll_repo(self, repo: str):
        """Poll a single repo for new trigger comments."""
        since = self._last_checked.get(repo)
        if since is None:
            # First run for this repo — look back a bounded window so we do not
            # replay the repo's entire comment history.
            since = _iso(
                datetime.now(timezone.utc)
                - timedelta(minutes=self._config.poll_initial_lookback_minutes)
            )
            logger.info(f"First poll for {repo}, looking back to {since}")

        # Record the cycle start before making requests, so comments created
        # during the request are picked up next cycle.
        cycle_start = datetime.now(timezone.utc)

        comments = await self._fetch_comments(repo, since)

        for comment in comments:
            await self._handle_comment(repo, comment)

        # Advance the watermark, with a small overlap for safety.
        self._last_checked[repo] = _iso(
            cycle_start - timedelta(seconds=POLL_OVERLAP_SECONDS)
        )

    async def _fetch_comments(self, repo: str, since: str) -> list[dict]:
        """Fetch all comments updated since `since`, following pagination."""
        all_comments = []
        for page in range(1, MAX_PAGES_PER_POLL + 1):
            batch = await self._github.list_repo_comments(
                repo, since=since, page=page, per_page=100
            )
            all_comments.extend(batch)
            if len(batch) < 100:
                break
            if page == MAX_PAGES_PER_POLL:
                logger.warning(
                    f"{repo}: hit page cap ({MAX_PAGES_PER_POLL}) — "
                    "some comments may be delayed to the next cycle"
                )
        return all_comments

    async def _handle_comment(self, repo: str, comment: dict):
        """Check one comment for the trigger and enqueue a job if it matches."""
        comment_id = comment.get("id")
        if comment_id is None or comment_id in self._processed_set:
            return

        # Only PR comments — issue comments share this endpoint.
        html_url = comment.get("html_url", "")
        if "/pull/" not in html_url:
            return

        body = comment.get("body") or ""
        if "/request_bot_review" not in body:
            return

        pr_number = _extract_pr_number(comment)
        if pr_number is None:
            logger.warning(f"Could not extract PR number from comment {comment_id}")
            return

        requester = (comment.get("user") or {}).get("login", "unknown")

        # Mark handled before dispatching, so a failure mid-flight does not
        # cause the same comment to retrigger on every subsequent cycle.
        self._mark_processed(comment_id)
        self.triggers_found += 1

        logger.info(
            f"Trigger found via poll: {repo}#{pr_number} "
            f"comment={comment_id} by {requester}"
        )

        await create_job_from_comment(
            repo_full_name=repo,
            pr_number=pr_number,
            comment_id=comment_id,
            comment_body=body,
            requester=requester,
            config=self._config,
            github_client=self._github,
            job_queue=self._job_queue,
            store=self._store,
            source="poll",
        )

    def _mark_processed(self, comment_id: int):
        if len(self._processed_ids) == self._processed_ids.maxlen:
            oldest = self._processed_ids[0]
            self._processed_set.discard(oldest)
        self._processed_ids.append(comment_id)
        self._processed_set.add(comment_id)

    # ── State persistence ─────────────────────────────────────────────────────

    def _load_state(self):
        """Restore watermarks and processed IDs so a restart neither replays
        old triggers nor misses ones that arrived while we were down."""
        if not self._state_path.exists():
            logger.info("No poller state file — starting fresh")
            return
        try:
            data = json.loads(self._state_path.read_text())
            self._last_checked = data.get("last_checked", {})
            ids = data.get("processed_ids", [])[-PROCESSED_IDS_MAX:]
            self._processed_ids = deque(ids, maxlen=PROCESSED_IDS_MAX)
            self._processed_set = set(ids)
            logger.info(
                f"Loaded poller state: {len(self._last_checked)} repos, "
                f"{len(ids)} processed comment IDs"
            )
        except Exception as e:
            logger.warning(f"Failed to load poller state ({e}) — starting fresh")

    def _save_state(self):
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({
                "last_checked": self._last_checked,
                "processed_ids": list(self._processed_ids),
            }, indent=2))
            tmp.replace(self._state_path)  # atomic
        except Exception as e:
            logger.warning(f"Failed to save poller state: {e}")

    # ── Introspection ─────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "enabled": True,
            "interval_seconds": self._config.poll_interval_seconds,
            "poll_count": self.poll_count,
            "last_poll_at": self.last_poll_at,
            "last_error": self.last_error,
            "triggers_found": self.triggers_found,
            "merge_commits_found": self.merge_commits_found,
            "watermarks": dict(self._last_checked),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_pr_number(comment: dict) -> int | None:
    """Pull the PR number out of a comment's issue_url or html_url."""
    issue_url = comment.get("issue_url", "")
    m = ISSUE_URL_PATTERN.search(issue_url)
    if m:
        return int(m.group(1))

    # Fall back to html_url: .../pull/123#issuecomment-456
    m = re.search(r"/pull/(\d+)", comment.get("html_url", ""))
    if m:
        return int(m.group(1))
    return None


def _now_iso() -> str:
    return _iso(datetime.now(timezone.utc))


def _iso(dt: datetime) -> str:
    """Format a datetime the way the GitHub API expects (ISO8601, Z suffix)."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
