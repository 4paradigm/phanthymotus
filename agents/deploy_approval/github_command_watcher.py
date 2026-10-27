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
import time
import logging
import re
from collections import deque
from datetime import datetime, timezone
from urllib.parse import urlparse

from . import commands as commands_mod
from .config import Config, resolve_authorized_repos
from .github_client import (
    GitHubClient, GitHubError, GitHubRateLimitError, GitHubTransientError,
)
from .github_state_proxy import HIDDEN_STATE_MARKER, GitHubStateProxy, GitHubStateProxyError

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


def _top_level_command(text: str, name: str) -> str:
    """Return a recognized top-level special command line, if present."""
    expected = "/" + name
    for line in commands_mod._iter_top_level_command_lines(text or ""):
        if line.split(None, 1)[0].lower() == expected:
            return line
    return ""


class GitHubCommandWatcher:
    """Polls PR comments using `POLL_INTERVAL_SECONDS` and dispatches commands.

    Exactly one serial command worker. Processes comments in ascending order.
    """

    def __init__(
        self,
        config: Config,
        proxy: GitHubStateProxy,
        controller: object,
        github: GitHubClient | None = None,
        github_auth: object | None = None,
    ):
        self.config = config
        self.proxy = proxy
        self.controller = controller
        self._github = github
        self._github_auth = github_auth
        self._auth_refresh_interval = 120  # seconds
        self._last_auth_refresh = -1.0  # never refreshed
        self._monotonic = time.monotonic  # deterministic clock source
        self._task: asyncio.Task | None = None
        self._running = False
        # Repos newly (re)activated by an auth refresh.  Each is baselined
        # (cursor = max observed comment id, no dispatch) before normal
        # command processing resumes, so preexisting comments never execute
        # as fresh approvals.
        self._pending_baseline_repos: set[str] = set()
        # Fixed UTC boundary for each active authorization generation. A slow
        # baseline must not move the boundary forward and absorb new commands.
        self._repo_baseline_cutoff_utc: dict[str, datetime] = {}
        # Per-repo set of PRs already baselined in the current
        # pending-generation. Prevents rebaselining PRs that
        # already succeeded this generation.
        self._baseline_progress: dict[str, set[int]] = {}
        # Bounded polling cursors: repository order rotates each cycle, while
        # each repository resumes its ordinary scan position in PR-number order.
        self._repo_poll_cursor = 0
        self._pr_scan_cursor: dict[str, int] = {}
        self._recent_pr_cursor: dict[str, int] = {}
        self._priority_pr_queue: deque[tuple[str, int]] = deque()
        self._priority_pr_set: set[tuple[str, int]] = set()
        self._priority_pr_enqueued_at: dict[tuple[str, int], float] = {}
        self._priority_pr_sources: dict[tuple[str, int], str] = {}
        self._priority_generation = 0
        self._wakeup = asyncio.Event()
        self._last_successful_cycle = None
        self._last_driver_poll = None
        self._repo_backlog: dict[str, dict] = {}
        self._last_pr_attempt: dict[tuple[str, int], dict] = {}
        self._recent_command_records: dict[tuple[str, int, int], dict] = {}
        self._latest_recent_command_key: tuple[str, int, int] | None = None
        self._recent_discovery_at: dict[str, str] = {}
        self._recent_discovery_monotonic: dict[str, float] = {}
        self._recent_discovery_last_failure: dict[str, str | None] = {}
        self._cycle_durations: deque[float] = deque(maxlen=100)
        self._pr_durations: deque[float] = deque(maxlen=200)
        self._last_evidence_projection: dict[tuple[str, int], tuple[str, float]] = {}

    async def _validated_current_summary_cos_links(self, repo: str, pr_number: int) -> set[str]:
        """Return one-year COS URLs present in the trusted current Summary."""
        store = getattr(getattr(self.controller, "evidence_service", None), "store", None)
        if store is None or not callable(getattr(store, "download_url_expiry", None)):
            return set()
        pr = await self.proxy.get_pr(repo, pr_number)
        state = await self.proxy.read_hidden_state(repo, pr_number)
        lifecycle = await self.proxy.find_trusted_lifecycle_comment(repo, pr_number)
        if (not isinstance(pr, dict) or pr.get("state") != "open" or pr.get("merged")
                or not isinstance(state, dict)
                or not isinstance(lifecycle, dict)
                or (pr.get("head") or {}).get("sha") != state.get("head_sha")):
            return set()
        body = lifecycle.get("body", "")
        if not isinstance(body, str) or HIDDEN_STATE_MARKER not in body:
            return set()
        visible = body.split(HIDDEN_STATE_MARKER, 1)[0]
        urls = set(re.findall(r"<(https://[^<>\s]+)>", visible))
        if not urls:
            return set()

        objects = []
        metadata = state.get("cos")
        if isinstance(metadata, dict):
            object_key = metadata.get("object_key", "")
            try:
                from .cos_client import CosClient
                CosClient.validate_object_key(repo, pr_number, state["head_sha"], object_key)
            except (KeyError, TypeError, ValueError):
                pass
            else:
                if (isinstance(metadata.get("sha256"), str)
                        and re.fullmatch(r"[0-9a-f]{64}", metadata["sha256"])
                        and type(metadata.get("size")) is int and metadata["size"] > 0):
                    objects.append((object_key, object_key.rsplit("/", 1)[-1], None))

        try:
            from .evidence_models import round_id_for_state
            from .evidence_store import build_attachment_key
            current_round = round_id_for_state(state)
            records = await self.proxy.read_evidence_records(repo, pr_number)
        except (GitHubStateProxyError, ValueError, TypeError):
            return set()
        for record in records:
            if record.get("kind") != "attachment":
                continue
            item = record.get("data")
            if (not isinstance(item, dict) or item.get("status") != "ready"
                    or item.get("actor_verified") is not True
                    or item.get("head_sha") != state.get("head_sha")
                    or item.get("round_id") != current_round
                    or not isinstance(item.get("sha256"), str)
                    or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
                    or type(item.get("size")) is not int or item["size"] <= 0):
                continue
            attachment_id = item.get("attachment_id")
            filename = item.get("filename")
            try:
                expected_key = build_attachment_key(
                    repo, pr_number, state["head_sha"], current_round,
                    attachment_id, item.get("created_at", ""), filename,
                )
            except (TypeError, ValueError):
                continue
            if item.get("object_key") == expected_key:
                objects.append((expected_key, filename, attachment_id))

        valid = set()
        for url in urls:
            for key, filename, attachment_id in objects:
                if store.download_url_expiry(
                        url, key=key, filename=filename, attachment_id=attachment_id) is not None:
                    valid.add(url)
                    break
        return valid

    def enqueue_pr(self, repo: str, pr_number: int) -> bool:
        """Prioritize an authorized PR after a webhook observes a new comment."""
        if (not isinstance(repo, str)
                or not getattr(self.config, "auth_valid", False)
                or repo not in self.config.active_repos
                or repo not in self.config.github_repos
                or type(pr_number) is not int
                or pr_number <= 0):
            return False
        item = (repo, pr_number)
        if item in self._priority_pr_set:
            if item in self._priority_pr_queue:
                self._priority_pr_queue.remove(item)
                self._priority_pr_queue.appendleft(item)
                self._priority_pr_sources[item] = "webhook"
                self._priority_pr_enqueued_at[item] = self._monotonic()
            else:
                # A webhook can arrive while the same PR is being read. Keep a
                # fresh queued pass so the new comment is not hidden by dedupe.
                self._queue_priority_pr(item, "webhook")
            self._priority_generation += 1
            self._wakeup.set()
            return True
        if len(self._priority_pr_queue) >= 128:
            evicted = self._priority_pr_queue.pop()
            self._priority_pr_set.discard(evicted)
            self._priority_pr_enqueued_at.pop(evicted, None)
            self._priority_pr_sources.pop(evicted, None)
        self._priority_pr_queue.appendleft(item)
        self._priority_pr_set.add(item)
        self._priority_pr_sources[item] = "webhook"
        self._priority_pr_enqueued_at[item] = self._monotonic()
        self._priority_generation += 1
        self._wakeup.set()
        return True

    def _queue_priority_pr(self, item: tuple[str, int], source: str) -> bool:
        """Requeue an in-flight hint without duplicates, keeping webhook priority."""
        if item in self._priority_pr_queue:
            return True
        if len(self._priority_pr_queue) >= 128:
            if source == "webhook":
                evicted = self._priority_pr_queue.pop()
                self._priority_pr_set.discard(evicted)
                self._priority_pr_enqueued_at.pop(evicted, None)
                self._priority_pr_sources.pop(evicted, None)
            else:
                evicted = next((queued for queued in reversed(self._priority_pr_queue)
                                if self._priority_pr_sources.get(queued) == "recent"), None)
                if evicted is None:
                    self._priority_pr_set.discard(item)
                    self._priority_pr_enqueued_at.pop(item, None)
                    self._priority_pr_sources.pop(item, None)
                    return False
                self._priority_pr_queue.remove(evicted)
                self._priority_pr_set.discard(evicted)
                self._priority_pr_enqueued_at.pop(evicted, None)
                self._priority_pr_sources.pop(evicted, None)
        if source == "webhook":
            self._priority_pr_queue.appendleft(item)
        else:
            self._priority_pr_queue.append(item)
        self._priority_pr_set.add(item)
        self._priority_pr_sources[item] = source
        self._priority_pr_enqueued_at[item] = self._monotonic()
        return True

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())

    def mark_repos_pending_baseline(
        self, repos: list[str], *, startup: bool = True,
    ) -> None:
        """Seed the pending-baseline gate for externally resolved repos.

        Startup and authorization recovery both establish a new generation.
        This includes the configured primary repository: commands posted during
        an authorization outage must never become fresh approvals on recovery.
        """
        cutoff_utc = datetime.now(timezone.utc)
        for repo in repos:
            if repo not in self.config.github_repos or repo not in self.config.active_repos:
                continue
            self._pending_baseline_repos.add(repo)
            self._baseline_progress[repo] = set()
            self._repo_baseline_cutoff_utc[repo] = cutoff_utc

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
                await self._refresh_active_repos_if_needed()
                await self._poll_once()
            except Exception as e:
                logger.error("command watcher poll cycle failed: %s", e)
            try:
                await asyncio.wait_for(
                    self._wakeup.wait(), timeout=self.config.poll_interval_seconds,
                )
                self._wakeup.clear()
            except asyncio.TimeoutError:
                pass

    def diagnostics(self) -> dict:
        now = self._monotonic()
        oldest_age = max(
            (now - self._priority_pr_enqueued_at.get(item, now)
             for item in self._priority_pr_queue),
            default=0.0,
        )
        return {
            "last_successful_cycle": self._last_successful_cycle,
            "last_driver_poll": self._last_driver_poll,
            "per_repo_backlog": {repo: dict(value) for repo, value in self._repo_backlog.items()},
            "priority_queue_length": len(self._priority_pr_queue),
            "oldest_queued_age_seconds": round(oldest_age, 1),
            "cycle_duration_seconds": round(self._cycle_durations[-1], 1) if self._cycle_durations else None,
            "cycle_duration_average_seconds": round(sum(self._cycle_durations) / len(self._cycle_durations), 1) if self._cycle_durations else None,
            "cycle_duration_p95_seconds": round(sorted(self._cycle_durations)[int((len(self._cycle_durations) - 1) * 0.95)], 1) if self._cycle_durations else None,
            "pr_duration_average_seconds": round(sum(self._pr_durations) / len(self._pr_durations), 1) if self._pr_durations else None,
            "pr_duration_p95_seconds": round(sorted(self._pr_durations)[int((len(self._pr_durations) - 1) * 0.95)], 1) if self._pr_durations else None,
            "recent_command": self._public_recent_command_metric(),
            "recent_command_discovery_at": dict(self._recent_discovery_at),
            "recent_command_discovery_last_failure": dict(self._recent_discovery_last_failure),
            "pr_442_last_attempt": dict(self._last_pr_attempt.get(("4paradigm/phanthymotus-driver", 442), {})),
            "github_api": self._github.diagnostics() if callable(getattr(self._github, "diagnostics", None)) else {},
            "webhook_enabled": bool(getattr(self.config, "webhook_enabled", False)),
        }

    def _github_backing_off(self) -> bool:
        diagnostics = self._github.diagnostics() if callable(
            getattr(self._github, "diagnostics", None)
        ) else None
        return isinstance(diagnostics, dict) and diagnostics.get("backing_off") is True

    def _public_recent_command_metric(self) -> dict | None:
        key = self._latest_recent_command_key
        item = self._recent_command_records.get(key) if key is not None else None
        if not isinstance(item, dict):
            return None
        return {name: value for name, value in item.items()
                if not name.startswith("_")}

    @staticmethod
    def _parse_comment_time(value) -> datetime | None:
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(timezone.utc)

    def _recent_comment_pr_number(self, repo: str, comment: dict) -> int | None:
        issue_url = comment.get("issue_url")
        if not isinstance(issue_url, str):
            return None
        parsed = urlparse(issue_url)
        api = urlparse(self.config.github_api_url)
        if (parsed.scheme.lower() != api.scheme.lower()
                or parsed.netloc.lower() != api.netloc.lower()):
            return None
        match = re.search(
            r"/repos/" + re.escape(repo) + r"/issues/([1-9][0-9]*)$",
            parsed.path,
        )
        if match is None:
            return None
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None

    def _observe_recent_command(self, repo, pr_number, comment, created_at):
        comment_id = comment.get("id")
        if type(comment_id) is not int or comment_id <= 0:
            return None
        key = (repo, pr_number, comment_id)
        if key not in self._recent_command_records:
            record = {
                "repo": repo, "pr_number": pr_number, "comment_id": comment_id,
                "comment_created_at": comment.get("created_at"),
                "discovered_at": datetime.now(timezone.utc).isoformat(),
                "discovery_delay_seconds": round(max(
                    0.0, (datetime.now(timezone.utc) - created_at).total_seconds(),
                ), 3),
                "processing_started_at": None, "discovery_to_processing_seconds": None,
                "last_failure_reason": None, "status": "discovered",
                "_discovered_monotonic": self._monotonic(),
                "_created_sort": created_at.timestamp(),
            }
            self._recent_command_records[key] = record
            latest_key = self._latest_recent_command_key
            latest = self._recent_command_records.get(latest_key) if latest_key else None
            if latest is None or record["_created_sort"] >= latest.get("_created_sort", 0):
                self._latest_recent_command_key = key
            logger.info(
                "watcher recent command discovered repo=%s pr=%s comment_id=%s created_at=%s discovered_at=%s delay_seconds=%.3f",
                repo, pr_number, comment_id, record["comment_created_at"], record["discovered_at"],
                record["discovery_delay_seconds"],
            )
            if len(self._recent_command_records) > 512:
                oldest = min(self._recent_command_records,
                             key=lambda item: self._recent_command_records[item].get(
                                 "_discovered_monotonic", 0.0))
                self._recent_command_records.pop(oldest, None)
                if oldest == self._latest_recent_command_key:
                    self._latest_recent_command_key = max(
                        self._recent_command_records,
                        key=lambda item: self._recent_command_records[item].get("_created_sort", 0.0),
                        default=None,
                    )
        return key

    def _mark_recent_command_processing(self, repo, pr_number, comment_id):
        key = (repo, pr_number, comment_id)
        record = self._recent_command_records.get(key)
        if not isinstance(record, dict):
            return
        started_at = datetime.now(timezone.utc).isoformat()
        elapsed = max(0.0, self._monotonic() - record.get("_discovered_monotonic", self._monotonic()))
        record.update(
            processing_started_at=started_at,
            discovery_to_processing_seconds=round(elapsed, 3),
            status="processing",
        )
        logger.info(
            "watcher recent command processing started repo=%s pr=%s comment_id=%s started_at=%s delay_seconds=%.3f",
            repo, pr_number, comment_id, started_at, elapsed,
        )

    def _finish_recent_command(
        self, repo, pr_number, comment_id, *, failure_reason=None, retryable=True,
    ):
        record = self._recent_command_records.get((repo, pr_number, comment_id))
        if not isinstance(record, dict):
            return
        if failure_reason is not None:
            record["last_failure_reason"] = failure_reason
        else:
            record["last_failure_reason"] = None
        record["status"] = "retry" if failure_reason and retryable else "handled"
        logger.info(
            "watcher recent command finished repo=%s pr=%s comment_id=%s status=%s delay_seconds=%s last_failure_reason=%s",
            repo, pr_number, comment_id, record["status"],
            record.get("discovery_to_processing_seconds"), failure_reason,
        )

    def _fail_recent_commands_for_pr(self, repo, pr_number, reason):
        for (observed_repo, observed_pr, comment_id), record in list(
                self._recent_command_records.items()):
            if (observed_repo == repo and observed_pr == pr_number
                    and record.get("status") != "handled"):
                self._finish_recent_command(
                    repo, pr_number, comment_id, failure_reason=reason,
                )

    async def _discover_recent_command_prs(self, repos: list[str]) -> None:
        """Queue a small set of PRs referenced by the newest command comments."""
        list_recent = getattr(self._github, "list_recent_issue_comments", None)
        if not callable(list_recent):
            return
        candidates = {}
        for repo in repos:
            if (not self.config.auth_valid or repo not in self.config.active_repos
                    or repo not in self.config.github_repos):
                continue
            now = self._monotonic()
            if now - self._recent_discovery_monotonic.get(repo, float("-inf")) < 30.0:
                continue
            self._recent_discovery_monotonic[repo] = now
            self._recent_discovery_at[repo] = datetime.now(timezone.utc).isoformat()
            try:
                comments = await asyncio.wait_for(
                    list_recent(repo, limit=100), timeout=2.0,
                )
            except (GitHubRateLimitError, GitHubTransientError) as exc:
                self._recent_discovery_last_failure[repo] = type(exc).__name__
                logger.warning(
                    "watcher recent command discovery deferred repo=%s reason=%s",
                    repo, type(exc).__name__,
                )
                if isinstance(exc, GitHubRateLimitError):
                    break
                if self._github_backing_off():
                    break
                continue
            except GitHubError as exc:
                if exc.status_code == 401:
                    self._recent_discovery_last_failure[repo] = "http_401_authentication_rejected"
                    logger.error(
                        "watcher recent command discovery authentication rejected repo=%s; authorization gate remains closed",
                        repo,
                    )
                    return
                # This optional endpoint may be unavailable under a valid App
                # installation. Keep the ordinary open-PR scan as the fallback.
                self._recent_discovery_last_failure[repo] = (
                    f"http_{exc.status_code}" if exc.status_code in (403, 404)
                    else type(exc).__name__
                )
                logger.warning(
                    "watcher recent command discovery unavailable repo=%s status=%s; falling back to PR scan",
                    repo, getattr(exc, "status_code", None),
                )
                continue
            except Exception as exc:
                self._recent_discovery_last_failure[repo] = type(exc).__name__
                logger.warning(
                    "watcher recent command discovery failed repo=%s reason=%s; falling back to PR scan",
                    repo, type(exc).__name__,
                )
                continue
            self._recent_discovery_at[repo] = datetime.now(timezone.utc).isoformat()
            self._recent_discovery_last_failure[repo] = None
            if not isinstance(comments, list):
                self._recent_discovery_last_failure[repo] = "invalid_response"
                continue
            per_pr = {}
            cutoff = self._repo_baseline_cutoff_utc.get(repo)
            for comment in comments[:100]:
                if not isinstance(comment, dict):
                    continue
                number = self._recent_comment_pr_number(repo, comment)
                created_at = self._parse_comment_time(comment.get("created_at"))
                if (number is None or created_at is None or not isinstance(cutoff, datetime)
                        or cutoff.tzinfo is None or created_at <= cutoff.astimezone(timezone.utc)):
                    continue
                body = comment.get("body", "")
                if not isinstance(body, str) or not commands_mod.parse_command(body).is_command:
                    continue
                key = self._observe_recent_command(repo, number, comment, created_at)
                if key is None:
                    continue
                observed = self._recent_command_records[key]
                if observed.get("status") == "handled":
                    continue
                previous = per_pr.get(number)
                if previous is None or created_at > previous[0]:
                    per_pr[number] = (created_at, key)
            for number, (created_at, key) in per_pr.items():
                if (repo, number) in self._priority_pr_set:
                    continue
                candidates[(repo, number)] = (created_at, key)

        # Insert oldest first at the fixed position after webhooks; each newer
        # item is then placed ahead of older discovery hints in the final queue.
        ordered = sorted(candidates.items(), key=lambda item: item[1][0])
        for (repo, number), (_created_at, _key) in ordered:
            item = (repo, number)
            if item in self._priority_pr_set:
                continue
            if len(self._priority_pr_queue) >= 128:
                oldest_recent = next((queued for queued in reversed(self._priority_pr_queue)
                                      if self._priority_pr_sources.get(queued) == "recent"), None)
                if oldest_recent is None:
                    continue
                self._priority_pr_queue.remove(oldest_recent)
                self._priority_pr_set.discard(oldest_recent)
                self._priority_pr_enqueued_at.pop(oldest_recent, None)
                self._priority_pr_sources.pop(oldest_recent, None)
            webhook_count = sum(
                self._priority_pr_sources.get(queued) == "webhook"
                for queued in self._priority_pr_queue
            )
            self._priority_pr_queue.insert(webhook_count, item)
            self._priority_pr_set.add(item)
            self._priority_pr_sources[item] = "recent"
            self._priority_pr_enqueued_at[item] = self._monotonic()

    def _record_pr_attempt(
        self, repo: str, pr_number: int, reason: str, duration_seconds: float | None = None,
    ) -> None:
        self._last_pr_attempt[(repo, pr_number)] = {
            "at": datetime.now(timezone.utc).isoformat(), "reason": reason,
        }
        if duration_seconds is not None:
            duration = max(0.0, duration_seconds)
            self._last_pr_attempt[(repo, pr_number)]["duration_seconds"] = round(duration, 1)
            self._pr_durations.append(duration)
        if len(self._last_pr_attempt) > 512:
            protected = ("4paradigm/phanthymotus-driver", 442)
            candidates = [key for key in self._last_pr_attempt if key != protected]
            oldest = min(candidates, key=lambda key: self._last_pr_attempt[key].get("at", ""))
            self._last_pr_attempt.pop(oldest, None)

    async def _refresh_active_repos_if_needed(self) -> None:
        """Periodically refresh runtime authorization before polling."""
        if self._github is None or self._github_auth is None:
            return
        if self._github_backing_off():
            return
        now = self._monotonic()
        if self._last_auth_refresh >= 0.0 and (
            (now - self._last_auth_refresh) < self._auth_refresh_interval
        ):
            return
        # Timestamp set BEFORE the attempt: a failed refresh still occupies
        # this interval slot, so the next cycle retries instead of hot-looping.
        self._last_auth_refresh = now
        await self._refresh_active_repos()

    async def _refresh_active_repos(self) -> None:
        """Refresh ACTIVE_REPOS from GitHub App installation authorization.

        CORRECT AUTH SEQUENCE (fail-closed):
        1. Force fresh installation token under existing single-flight lock.
        2. GET /installation/repositories with the refreshed token.
        3. Validate complete paginated response.
        4. Intersect config.github_repos with authorized_names.
        5. Ensure required repo is present.
        6. Atomically publish new_active + auth_valid together.

        On any failure: auth_valid=False, do NOT publish new repos,
        and do NOT proceed with _poll_once() that cycle.
        """
        if self._github is None or self._github_auth is None:
            return
        old_active = list(self.config.active_repos)
        old_auth_valid = getattr(self.config, "auth_valid", True)
        try:
            # Step 1: FORCE fresh installation token FIRST
            try:
                await self._github_auth.refresh_installation_token()
            except Exception as tok_exc:
                logger.warning(
                    "auth refresh: forced token refresh failed \u2014 "
                    "keeping previous active repos: %s", old_active,
                )
                self.config.auth_valid = False
                return

            # Step 2: Query repos with refreshed token
            authorized_names = await self._github.list_installation_repositories()
            try:
                new_active = resolve_authorized_repos(self.config.github_repos, authorized_names)
            except RuntimeError:
                self.config.auth_valid = False
                self.config.active_repos = []
                logger.warning("auth refresh: configured primary repository is not authorized")
                return

            # Step 6: Atomically publish new_active + auth_valid
            self.config.active_repos = new_active
            self.config.auth_valid = True

            old_active_set = set(old_active)
            new_active_set = set(new_active)
            removed = old_active_set - new_active_set
            for repo in removed:
                self._pending_baseline_repos.discard(repo)
                self._baseline_progress.pop(repo, None)
            newly_active = new_active_set if not old_auth_valid else new_active_set - old_active_set
            self.mark_repos_pending_baseline(sorted(newly_active), startup=False)
        except (GitHubRateLimitError, GitHubTransientError) as e:
            # API pressure or transport failure is not evidence that the
            # installation lost authorization; preserve the current snapshot
            # and avoid triggering a generation-wide baseline on recovery.
            self.config.auth_valid = old_auth_valid
            logger.warning(
                "auth refresh deferred reason=%s retry_after=%.1f",
                "rate_limited" if isinstance(e, GitHubRateLimitError) else "transient_api_error",
                e.retry_after_seconds,
            )
        except Exception as e:
            logger.warning(
                "auth refresh failed: %s \u2014 failing closed",
                type(e).__name__,
            )
            self.config.auth_valid = False

    async def _process_scheduled_pr(self, repo: str, pr_number: int) -> bool:
        """Run one PR through the existing serial writer and evidence path."""
        baseline_pending = repo in self._pending_baseline_repos
        baseline_done = self._baseline_progress.setdefault(repo, set())
        baseline_succeeded = True
        if baseline_pending and pr_number not in baseline_done:
            if await self._baseline_pr(repo, pr_number):
                baseline_done.add(pr_number)
            else:
                baseline_succeeded = False
        else:
            await self._process_pr(repo, pr_number)
        if (self.config.auth_valid is not True or repo not in self.config.active_repos
                or repo not in self.config.github_repos):
            return False
        current_pr = await self.proxy.get_pr(repo, pr_number)
        if (isinstance(current_pr, dict) and current_pr.get("state") == "open"
                and not current_pr.get("merged") and not current_pr.get("draft")):
            updated_at = current_pr.get("updated_at")
            head_sha = current_pr.get("head", {}).get("sha")
            projection_version = (
                f"{head_sha}:{updated_at}"
                if isinstance(head_sha, str) and isinstance(updated_at, str) else ""
            )
            key = (repo, pr_number)
            previous = self._last_evidence_projection.get(key)
            now = self._monotonic()
            refresh_due = previous is None or now - previous[1] >= 6 * 24 * 60 * 60
            changed = not projection_version or previous is None or previous[0] != projection_version
            if changed or refresh_due:
                await self.controller.project_evidence(repo, pr_number)
                await self.proxy.ensure_visible_evidence_archive_comments(
                    repo, pr_number, limit=2,
                )
                valid_cos_links = await self._validated_current_summary_cos_links(repo, pr_number)
                await self.proxy.retire_legacy_evidence_link_replies(
                    repo, pr_number, valid_cos_links, limit=2,
                )
                if projection_version:
                    self._last_evidence_projection.pop(key, None)
                    self._last_evidence_projection[key] = (projection_version, now)
                    if len(self._last_evidence_projection) > 4096:
                        self._last_evidence_projection.pop(next(iter(self._last_evidence_projection)))
        return baseline_succeeded

    async def _drain_priority_prs(
        self, priority_done, priority_attempted, recent_attempted,
        *, allowed_sources,
    ) -> bool:
        """Process a bounded priority tier through the existing serial PR path."""
        cycle_ok = True
        queued_items = [item for item in self._priority_pr_queue
                        if self._priority_pr_sources.get(item, "webhook") in allowed_sources]
        for item in queued_items:
            if item not in self._priority_pr_queue:
                continue
            self._priority_pr_queue.remove(item)
            queued_repo, pr_number = item
            source = self._priority_pr_sources.get(item, "webhook")
            if self.config.auth_valid is not True:
                self._queue_priority_pr(item, source)
                cycle_ok = False
                break
            if (queued_repo not in self.config.active_repos
                    or queued_repo not in self.config.github_repos):
                self._priority_pr_set.discard(item)
                self._priority_pr_enqueued_at.pop(item, None)
                self._priority_pr_sources.pop(item, None)
                continue
            if pr_number in priority_attempted.setdefault(queued_repo, []):
                self._queue_priority_pr(item, source)
                continue
            source_attempted = (priority_attempted.setdefault(queued_repo, [])
                                if source == "webhook"
                                else recent_attempted.setdefault(queued_repo, []))
            if len(source_attempted) >= (2 if source == "webhook" else 1):
                self._queue_priority_pr(item, source)
                continue
            source_attempted.append(pr_number)
            if source != "webhook":
                priority_attempted.setdefault(queued_repo, []).append(pr_number)
            try:
                current_pr = await self.proxy.get_pr(queued_repo, pr_number)
            except GitHubRateLimitError as exc:
                self._queue_priority_pr(item, source)
                self._fail_recent_commands_for_pr(queued_repo, pr_number, "github_rate_limited")
                self._record_pr_attempt(queued_repo, pr_number, "github_rate_limited")
                logger.warning("watcher rate limited before priority PR; retry_after=%.1f", exc.retry_after_seconds)
                cycle_ok = False
                break
            except Exception as exc:
                self._queue_priority_pr(item, source)
                self._fail_recent_commands_for_pr(queued_repo, pr_number, type(exc).__name__)
                self._record_pr_attempt(queued_repo, pr_number, type(exc).__name__)
                logger.warning("watcher priority PR read %s#%s failed: %s", queued_repo, pr_number, type(exc).__name__)
                cycle_ok = False
                if self._github_backing_off():
                    break
                continue
            if (not isinstance(current_pr, dict) or current_pr.get("state") != "open"
                    or current_pr.get("merged") or current_pr.get("draft")):
                for observed_repo, observed_pr, observed_comment_id in list(
                        self._recent_command_records):
                    if observed_repo == queued_repo and observed_pr == pr_number:
                        self._finish_recent_command(
                            queued_repo, pr_number, observed_comment_id,
                            failure_reason="pr_not_open", retryable=False,
                        )
                priority_attempted[queued_repo].remove(pr_number)
                while item in self._priority_pr_queue:
                    self._priority_pr_queue.remove(item)
                self._priority_pr_set.discard(item)
                self._priority_pr_enqueued_at.pop(item, None)
                self._priority_pr_sources.pop(item, None)
                continue
            if item not in self._priority_pr_queue:
                self._priority_pr_set.discard(item)
                self._priority_pr_enqueued_at.pop(item, None)
                self._priority_pr_sources.pop(item, None)
            baseline_only = (
                queued_repo in self._pending_baseline_repos
                and pr_number not in self._baseline_progress.get(queued_repo, set())
            )
            attempt_started = self._monotonic()
            try:
                succeeded = await self._process_scheduled_pr(queued_repo, pr_number)
                if not succeeded:
                    self._fail_recent_commands_for_pr(queued_repo, pr_number, "baseline_incomplete")
                    self._queue_priority_pr(item, source)
                    self._record_pr_attempt(
                        queued_repo, pr_number, "baseline_retry",
                        self._monotonic() - attempt_started,
                    )
                    cycle_ok = False
                    continue
                priority_done[queued_repo].append(pr_number)
                if (baseline_only
                        and pr_number in self._baseline_progress.get(queued_repo, set())):
                    self.enqueue_pr(queued_repo, pr_number)
                if queued_repo.endswith("phanthymotus-driver"):
                    self._last_driver_poll = datetime.now(timezone.utc).isoformat()
                self._record_pr_attempt(
                    queued_repo, pr_number, "webhook_priority_processed",
                    self._monotonic() - attempt_started,
                )
            except GitHubRateLimitError as exc:
                self._queue_priority_pr(item, source)
                self._fail_recent_commands_for_pr(queued_repo, pr_number, "github_rate_limited")
                self._record_pr_attempt(
                    queued_repo, pr_number, "github_rate_limited",
                    self._monotonic() - attempt_started,
                )
                logger.warning("watcher rate limited during priority PR; retry_after=%.1f", exc.retry_after_seconds)
                cycle_ok = False
                break
            except Exception as exc:
                self._queue_priority_pr(item, source)
                self._fail_recent_commands_for_pr(queued_repo, pr_number, type(exc).__name__)
                self._record_pr_attempt(
                    queued_repo, pr_number, type(exc).__name__, self._monotonic() - attempt_started,
                )
                logger.warning("watcher priority PR %s#%s failed: %s", queued_repo, pr_number, type(exc).__name__)
                cycle_ok = False
        return cycle_ok

    async def _poll_once(self) -> None:
        """Drain webhook work, discover recent commands, then run fair scans."""
        if not self.config.active_repos or self.config.auth_valid is not True:
            return
        cycle_started = self._monotonic()
        repos = list(dict.fromkeys(self.config.active_repos))
        if repos:
            repo_start = self._repo_poll_cursor % len(repos)
            repos = repos[repo_start:] + repos[:repo_start]
        priority_done: dict[str, list[int]] = {repo: [] for repo in repos}
        priority_attempted: dict[str, list[int]] = {repo: [] for repo in repos}
        recent_attempted: dict[str, list[int]] = {repo: [] for repo in repos}
        cycle_ok = True
        queue_generation_at_start = self._priority_generation
        interrupted_for_new_priority = False

        # Drain already queued webhook work before awaiting the optional
        # repo-wide discovery endpoint.
        cycle_ok = await self._drain_priority_prs(
            priority_done, priority_attempted, recent_attempted,
            allowed_sources={"webhook"},
        ) and cycle_ok
        if not self._github_backing_off():
            await self._discover_recent_command_prs(repos)
        if self.config.auth_valid is not True:
            cycle_ok = False
        if self.config.auth_valid is True and not self._github_backing_off():
            cycle_ok = await self._drain_priority_prs(
                priority_done, priority_attempted, recent_attempted,
                allowed_sources={"webhook", "recent"},
            ) and cycle_ok

        # A rate limit is shared across endpoints. Do not burn more calls in
        # the same cycle after the client has entered its backoff window.
        if self._github_backing_off():
            cycle_ok = False
        else:
            for repo in repos:
                if interrupted_for_new_priority:
                    break
                has_new_webhook = any(
                    queued_repo in self.config.active_repos
                    and queued_repo in self.config.github_repos
                    and self._priority_pr_sources.get((queued_repo, queued_number)) == "webhook"
                    for queued_repo, queued_number in self._priority_pr_queue
                )
                if (self._priority_generation != queue_generation_at_start
                        and has_new_webhook):
                    interrupted_for_new_priority = True
                    self._wakeup.set()
                    break
                if (self.config.auth_valid is not True or repo not in self.config.active_repos
                        or repo not in self.config.github_repos):
                    continue
                try:
                    prs = await self.proxy.get_open_prs(repo)
                except GitHubRateLimitError as exc:
                    logger.warning("watcher rate limited listing %s; retry_after=%.1f", repo, exc.retry_after_seconds)
                    cycle_ok = False
                    break
                except GitHubError as exc:
                    logger.warning("watcher list PRs %s: %s", repo, exc)
                    cycle_ok = False
                    continue
                except Exception as exc:
                    logger.warning("watcher list PRs %s failed: %s", repo, type(exc).__name__)
                    cycle_ok = False
                    if self._github_backing_off():
                        break
                    continue

                valid_prs = []
                for pr in prs:
                    if not isinstance(pr, dict):
                        continue
                    try:
                        number = int(pr.get("number") or 0)
                    except (TypeError, ValueError):
                        continue
                    if (number > 0 and pr.get("state", "open") == "open"
                            and not pr.get("merged") and not pr.get("draft")):
                        valid_prs.append((number, pr))
                valid_prs.sort(key=lambda item: item[0])
                open_numbers = [number for number, _ in valid_prs]
                baseline_all = repo in self._pending_baseline_repos
                pr_work_budget, fast_lane_size = 8, 6
                selected = list(priority_attempted.get(repo, []))
                selected_set = set(selected)
                selected_set.update(
                    queued_number for queued_repo, queued_number in self._priority_pr_queue
                    if queued_repo == repo
                )
                recent_pool = sorted(
                    valid_prs,
                    key=lambda item: str(item[1].get("updated_at") or ""),
                    reverse=True,
                )[:12]
                if len(recent_pool) > 1:
                    rotating = recent_pool[1:]
                    recent_start = self._recent_pr_cursor.get(repo, 0) % len(rotating)
                    recent_order = rotating[recent_start:] + rotating[:recent_start]
                    fast_lane = [recent_pool[0], *recent_order[:fast_lane_size - 1]]
                    self._recent_pr_cursor[repo] = (recent_start + 1) % len(rotating)
                else:
                    fast_lane = list(recent_pool)
                reserve_ordinary = min(2, max(0, pr_work_budget - len(selected)))
                recent_budget = min(
                    fast_lane_size, max(0, pr_work_budget - len(selected) - reserve_ordinary),
                )
                for number, _ in fast_lane:
                    if len(selected) >= pr_work_budget or recent_budget <= 0:
                        break
                    if number in selected_set:
                        continue
                    selected.append(number)
                    selected_set.add(number)
                    recent_budget -= 1

                scan_start = self._pr_scan_cursor.get(repo, 0) % len(open_numbers) if open_numbers else 0
                ordinary = open_numbers[scan_start:] + open_numbers[:scan_start]
                ordinary_slots = max(0, pr_work_budget - len(selected))
                cursor_after = {}
                scanned = 0
                for number in ordinary:
                    scanned += 1
                    if number in selected_set:
                        continue
                    selected.append(number)
                    selected_set.add(number)
                    cursor_after[number] = (scan_start + scanned) % len(open_numbers)
                    if len(selected) >= pr_work_budget or len(cursor_after) >= ordinary_slots:
                        break

                processed = []
                for number in selected[:pr_work_budget]:
                    has_new_webhook = any(
                        queued_repo in self.config.active_repos
                        and queued_repo in self.config.github_repos
                        and self._priority_pr_sources.get((queued_repo, queued_number)) == "webhook"
                        for queued_repo, queued_number in self._priority_pr_queue
                    )
                    if (self._priority_generation != queue_generation_at_start
                            and has_new_webhook):
                        interrupted_for_new_priority = True
                        self._wakeup.set()
                        break
                    if number in priority_attempted.get(repo, []):
                        if number in priority_done.get(repo, []):
                            processed.append(number)
                        continue
                    if (self.config.auth_valid is not True or repo not in self.config.active_repos
                            or repo not in self.config.github_repos):
                        break
                    attempt_started = self._monotonic()
                    try:
                        succeeded = await self._process_scheduled_pr(repo, number)
                        processed.append(number)
                        self._record_pr_attempt(
                            repo, number,
                            "fallback_processed" if succeeded else "baseline_retry",
                            self._monotonic() - attempt_started,
                        )
                        if not succeeded:
                            cycle_ok = False
                    except GitHubRateLimitError as exc:
                        logger.warning("watcher rate limited processing %s#%s; retry_after=%.1f", repo, number, exc.retry_after_seconds)
                        cycle_ok = False
                        break
                    except Exception as exc:
                        self._record_pr_attempt(
                            repo, number, type(exc).__name__, self._monotonic() - attempt_started,
                        )
                        logger.warning("watcher process %s#%s failed: %s", repo, number, type(exc).__name__)
                        cycle_ok = False
                    if number in cursor_after:
                        self._pr_scan_cursor[repo] = cursor_after[number]
                    if self._github_backing_off():
                        cycle_ok = False
                        break

                if baseline_all:
                    progress = self._baseline_progress.setdefault(repo, set())
                    if all(number in progress for number in open_numbers):
                        self._pending_baseline_repos.discard(repo)
                remaining = (
                    sum(number not in self._baseline_progress.get(repo, set()) for number in open_numbers)
                    if repo in self._pending_baseline_repos else 0
                )
                self._repo_backlog[repo] = {
                    "open_prs": len(open_numbers), "remaining_baselines": remaining,
                    "queued_prs": sum(1 for queued_repo, _ in self._priority_pr_queue if queued_repo == repo),
                    "oldest_queued_age_seconds": round(max((
                        self._monotonic() - self._priority_pr_enqueued_at.get((queued_repo, number), self._monotonic())
                        for queued_repo, number in self._priority_pr_queue if queued_repo == repo
                    ), default=0.0), 1),
                }
                if repo.endswith("phanthymotus-driver"):
                    self._last_driver_poll = datetime.now(timezone.utc).isoformat()
                logger.info(
                    "watcher poll %s selected_prs=%s priority_prs=%s processed_prs=%s remaining_baselines=%s scan_cursor=%s",
                    repo, selected[:pr_work_budget], priority_attempted.get(repo, []), processed,
                    remaining, self._pr_scan_cursor.get(repo, 0),
                )
                if interrupted_for_new_priority:
                    break
        if repos:
            self._repo_poll_cursor = (repo_start + 1) % len(repos)
        if cycle_ok:
            self._last_successful_cycle = datetime.now(timezone.utc).isoformat()
        self._cycle_durations.append(max(0.0, self._monotonic() - cycle_started))
        metrics = self.diagnostics()
        logger.info(
            "watcher metrics last_successful_cycle=%s last_driver_poll=%s backlog=%s queued=%s oldest_queued_age_seconds=%s cycle_seconds=%s cycle_avg_seconds=%s cycle_p95_seconds=%s pr_avg_seconds=%s pr_p95_seconds=%s pr442_last_attempt=%s github_api=%s webhook_enabled=%s",
            metrics["last_successful_cycle"], metrics["last_driver_poll"],
            metrics["per_repo_backlog"], metrics["priority_queue_length"],
            metrics["oldest_queued_age_seconds"], metrics["cycle_duration_seconds"],
            metrics["cycle_duration_average_seconds"], metrics["cycle_duration_p95_seconds"],
            metrics["pr_duration_average_seconds"], metrics["pr_duration_p95_seconds"],
            metrics["pr_442_last_attempt"],
            metrics["github_api"], metrics["webhook_enabled"],
        )

    async def _baseline_pr(self, repo: str, pr_number: int) -> bool:
        """Baseline old commands without replaying historical business commands.

        Used on first observation AND when a repo is newly (re)activated by an
        authorization refresh: the cursor is advanced to the maximum comment
        id observed on GitHub so preexisting comments never execute as fresh
        approvals. Commands posted AFTER the baseline persist are processed
        normally by subsequent cycles. Historical commands are never replayed.

        For PRs that already have a baselined cursor (existing lifecycle state
        from a previous authorized period), advance the cursor to the maximum
        observed comment id using ``persist_cursor`` so that any commands
        posted while the repo was unauthorized are skipped on reactivation.
        """
        from .evidence_models import EvidenceError, round_id_for_state

        initial_state = await self.proxy.read_hidden_state(repo, pr_number)
        expected_state = initial_state

        baseline_id = 0
        if _needs_initial_comment_baseline(initial_state):
            # --- First-observation / baseline-incomplete path ---
            pr_info = await self.proxy.get_pr(repo, pr_number)
            if not pr_info or pr_info.get("state") != "open":
                if initial_state is None:
                    await self.controller.reconcile_pr(repo, pr_number)
                return False

            # BEFORE reconcile: snapshot current comments.
            try:
                comments_before = await self.proxy.get_issue_comments(repo, pr_number)
            except Exception as e:
                logger.warning(
                    "watcher baseline snapshot-before %s#%s: %s \u2014 fail closed",
                    repo, pr_number, e,
                )
                return False

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
                    "watcher reconcile %s#%s: %s \u2014 skip baseline this cycle",
                    repo, pr_number, e,
                )
                return False

            # After reconcile: fresh read state.
            state_after = await self.proxy.read_hidden_state(repo, pr_number)
            if state_after is None:
                return False
            expected_state = state_after

            # AFTER reconcile: second snapshot.
            try:
                comments_after = await self.proxy.get_issue_comments(repo, pr_number)
            except Exception as e:
                logger.warning(
                    "watcher baseline snapshot-after %s#%s: %s \u2014 fail closed",
                    repo, pr_number, e,
                )
                return False

            max_cid_after = 0
            for c in comments_after:
                cid = c.get("id")
                if isinstance(cid, int) and not isinstance(cid, bool) and cid > max_cid_after:
                    max_cid_after = cid

            baseline_id = max(max_cid_before, max_cid_after)
        else:
            # --- Existing baselined PR \u2014 reactivation cursor advance ---
            # The PR has a real cursor state. Advance cursor to max(old, observed)
            # without resetting business fields.
            pr_info = await self.proxy.get_pr(repo, pr_number)
            if not pr_info or pr_info.get("state") != "open":
                if initial_state is None:
                    await self.controller.reconcile_pr(repo, pr_number)
                return False

            try:
                comments = await self.proxy.get_issue_comments(repo, pr_number)
            except Exception as e:
                logger.warning(
                    "watcher baseline snapshot-before %s#%s: %s \u2014 fail closed",
                    repo, pr_number, e,
                )
                return False

            old_cursor = initial_state.get("last_processed_comment_id", 0)
            if (not isinstance(old_cursor, int) or isinstance(old_cursor, bool)
                    or old_cursor < 0):
                return False
            cutoff = self._repo_baseline_cutoff_utc.get(repo)
            if not isinstance(cutoff, datetime) or cutoff.tzinfo is None:
                logger.warning("watcher baseline cutoff missing %s#%s — fail closed", repo, pr_number)
                return False
            baseline_id = old_cursor
            try:
                ordered_comments = sorted(comments, key=lambda c: c["id"])
                for comment in ordered_comments:
                    cid = comment.get("id")
                    if not isinstance(cid, int) or isinstance(cid, bool) or cid < 0:
                        return False
                    if cid <= old_cursor:
                        continue
                    created_at = comment.get("created_at")
                    if not isinstance(created_at, str) or not created_at.strip():
                        break
                    try:
                        created_utc = datetime.fromisoformat(
                            created_at.strip().replace("Z", "+00:00")
                        )
                    except ValueError:
                        break
                    if created_utc.tzinfo is None or created_utc.utcoffset() is None:
                        break
                    if created_utc.astimezone(timezone.utc) > cutoff.astimezone(timezone.utc):
                        break
                    baseline_id = cid
            except (TypeError, KeyError):
                return False

            # A restart/authorization baseline must not jump over unarchived
            # evidence from the active testing round. Recover only ordinary
            # comments newer than round_started_at; command comments remain
            # governed by the baseline and are never dispatched here.
            if initial_state.get("status") == "testing":
                head_sha = initial_state.get("head_sha")
                try:
                    round_id = round_id_for_state(initial_state)
                    round_started_at = initial_state.get("round_started_at")
                    round_started = datetime.fromisoformat(
                        str(round_started_at).replace("Z", "+00:00")
                    )
                    if (round_started.tzinfo is None
                            or not isinstance(head_sha, str)
                            or len(head_sha) != 40):
                        raise ValueError
                except (TypeError, ValueError):
                    logger.warning("watcher evidence baseline context invalid %s#%s", repo, pr_number)
                    return False
                pr_head = (pr_info.get("head") or {}).get("sha")
                if (pr_info.get("state") != "open" or pr_info.get("merged")
                        or pr_info.get("draft") or pr_head != head_sha):
                    return False
                evidence_service = getattr(self.controller, "evidence_service", None)
                for candidate in ordered_comments:
                    comment_id = candidate.get("id")
                    if comment_id <= old_cursor:
                        continue
                    body = candidate.get("body", "")
                    if (not isinstance(body, str) or not body.strip()
                            or commands_mod.command_starts_line_any(body)
                            or _top_level_command(body, "authorize_evidence")
                            or _top_level_command(body, "submit_cos_evidence")):
                        continue
                    try:
                        created = datetime.fromisoformat(
                            str(candidate.get("created_at", "")).replace("Z", "+00:00")
                        )
                    except ValueError:
                        continue
                    if created.tzinfo is None or created <= round_started:
                        continue
                    if evidence_service is None:
                        return False
                    try:
                        fresh_comment = await self.proxy.get_comment(repo, comment_id)
                    except Exception as e:
                        logger.warning(
                            "watcher evidence recovery comment %s#%s #%s: %s",
                            repo, pr_number, comment_id, type(e).__name__,
                        )
                        return False
                    if (not isinstance(fresh_comment, dict)
                            or fresh_comment.get("id") != comment_id):
                        return False
                    if self.proxy.is_bot_comment(fresh_comment):
                        continue
                    fresh_body = fresh_comment.get("body", "")
                    if (not isinstance(fresh_body, str) or not fresh_body.strip()
                            or commands_mod.command_starts_line_any(fresh_body)
                            or _top_level_command(fresh_body, "authorize_evidence")
                            or _top_level_command(fresh_body, "submit_cos_evidence")):
                        continue
                    try:
                        fresh_created = datetime.fromisoformat(
                            str(fresh_comment.get("created_at", "")).replace("Z", "+00:00")
                        )
                    except ValueError:
                        return False
                    if (fresh_created.tzinfo is None or fresh_created <= round_started
                            or fresh_comment.get("created_at") != candidate.get("created_at")):
                        return False
                    if (not self.config.auth_valid or repo not in self.config.active_repos
                            or repo not in self.config.github_repos):
                        return False
                    try:
                        current_pr = await self.proxy.get_pr(repo, pr_number)
                        current_state = await self.proxy.read_hidden_state(repo, pr_number)
                    except Exception as e:
                        logger.warning(
                            "watcher evidence recovery state %s#%s: %s",
                            repo, pr_number, type(e).__name__,
                        )
                        return False
                    if (not isinstance(current_pr, dict) or current_pr.get("state") != "open"
                            or current_pr.get("merged") or current_pr.get("draft")
                            or (current_pr.get("head") or {}).get("sha") != head_sha
                            or not isinstance(current_state, dict)
                            or current_state.get("status") != "testing"
                            or current_state.get("head_sha") != head_sha):
                        return False
                    try:
                        if round_id_for_state(current_state) != round_id:
                            return False
                    except ValueError:
                        return False
                    user = fresh_comment.get("user", {})
                    actor_id = user.get("id") if isinstance(user, dict) else None
                    actor_login = user.get("login") if isinstance(user, dict) else None
                    if (isinstance(actor_id, bool) or not str(actor_id or "").isdigit()
                            or not isinstance(actor_login, str) or not actor_login.strip()):
                        return False
                    try:
                        await evidence_service.handle_native_comment(
                            repo=repo, pr_number=pr_number, comment_id=comment_id,
                            actor_id=int(actor_id), actor_login=actor_login,
                            action="submit_evidence", arguments={}, automatic=True,
                        )
                    except EvidenceError as e:
                        if (e.status_code in {400, 403, 404, 410, 413}
                                and e.code not in {"app_access_revoked", "repository_not_authorized"}):
                            logger.info(
                                "watcher evidence recovery skipped %s#%s comment=%s reason=%s",
                                repo, pr_number, comment_id, e.code,
                            )
                            continue
                        logger.warning(
                            "watcher evidence recovery retry %s#%s comment=%s reason=%s",
                            repo, pr_number, comment_id, e.code,
                        )
                        return False
                    except Exception as e:
                        logger.warning(
                            "watcher evidence recovery retry %s#%s comment=%s error=%s",
                            repo, pr_number, comment_id, type(e).__name__,
                        )
                        return False
        # Revalidate the PR, authorization generation, and lifecycle identity
        # after all reads/recovery work and immediately before cursor mutation.
        if (not self.config.auth_valid or repo not in self.config.active_repos
                or repo not in self.config.github_repos):
            return False
        try:
            latest_pr = await self.proxy.get_pr(repo, pr_number)
            latest_state = await self.proxy.read_hidden_state(repo, pr_number)
        except Exception as e:
            logger.warning("watcher baseline recheck %s#%s: %s — fail closed", repo, pr_number, type(e).__name__)
            return False
        expected_head = expected_state.get("head_sha") if isinstance(expected_state, dict) else None
        expected_status = expected_state.get("status") if isinstance(expected_state, dict) else None
        if (not isinstance(latest_pr, dict) or latest_pr.get("state") != "open"
                or latest_pr.get("merged")
                or (latest_pr.get("head") or {}).get("sha") != expected_head
                or not isinstance(latest_state, dict)
                or latest_state.get("head_sha") != expected_head
                or latest_state.get("status") != expected_status):
            return False

        # Persist baseline cursor (shared by both paths).
        try:
            persisted = await self.proxy.persist_cursor(repo, pr_number, baseline_id)
        except Exception as e:
            logger.warning(
                "watcher persist baseline %s#%s: %s \u2014 fail closed",
                repo, pr_number, e,
            )
            return False

        if not isinstance(persisted, dict):
            logger.warning(
                "watcher persist baseline %s#%s returned %s \u2014 fail closed",
                repo, pr_number, type(persisted).__name__,
            )
            return False
        persisted_cid = persisted.get("last_processed_comment_id")
        if not isinstance(persisted_cid, int) or isinstance(persisted_cid, bool):
            logger.warning(
                "watcher persist baseline %s#%s invalid cursor \u2014 fail closed",
                repo, pr_number,
            )
            return False
        if persisted_cid < baseline_id:
            logger.warning(
                "watcher persist baseline %s#%s cursor %s < requested %s \u2014 fail closed",
                repo, pr_number, persisted_cid, baseline_id,
            )
            return False
        return True

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
        from .evidence_models import EvidenceError

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
        for (observed_repo, observed_pr, observed_comment_id), observed in list(
                self._recent_command_records.items()):
            if (observed_repo == repo and observed_pr == pr_number
                    and observed_comment_id <= cursor and observed.get("status") != "handled"):
                self._finish_recent_command(repo, pr_number, observed_comment_id)

        # Fetch comments
        try:
            comments = await self.proxy.get_issue_comments(repo, pr_number)
        except GitHubStateProxyError as e:
            logger.warning("watcher get comments %s#%s: %s", repo, pr_number, e)
            self._fail_recent_commands_for_pr(repo, pr_number, type(e).__name__)
            return

        # Filter new comments, sort ascending
        new_comments = [
            c for c in comments
            if isinstance(c.get("id"), int) and not isinstance(c.get("id"), bool) and c["id"] > cursor
        ]
        new_comments.sort(key=lambda c: c["id"])
        native_comment_budget = 3
        native_comments_processed = 0

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
                self._finish_recent_command(
                    repo, pr_number, comment_id, failure_reason=type(e).__name__,
                )
                return

            if fresh_comment_obj is None:
                # Comment was deleted between list and get — skip this cycle
                self._finish_recent_command(
                    repo, pr_number, comment_id,
                    failure_reason="comment_unavailable", retryable=False,
                )
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

            if _top_level_command(fresh_body, "authorize_evidence"):
                # This comment proves browser identity; it is not test evidence.
                await self._consume_comment_cursor(repo, pr_number, comment_id)
                continue

            if _top_level_command(fresh_body, "submit_cos_evidence"):
                # Retire the former COS receipt command without treating its
                # payload as evidence or sending it to a second writer.
                await self._consume_comment_cursor(repo, pr_number, comment_id)
                continue

            if not commands_mod.command_starts_line_any(fresh_body):
                if state.get("status") == "testing":
                    if native_comments_processed >= native_comment_budget:
                        return
                    if (not self.config.auth_valid or repo not in self.config.active_repos
                            or repo not in self.config.github_repos):
                        return
                    try:
                        pr_info = await self.proxy.get_pr(repo, pr_number)
                    except Exception as e:
                        logger.warning(
                            "watcher native evidence PR read %s#%s: %s",
                            repo, pr_number, type(e).__name__,
                        )
                        return
                    if (isinstance(pr_info, dict) and pr_info.get("state") == "open"
                            and not pr_info.get("merged") and not pr_info.get("draft")):
                        pr_head = (pr_info.get("head") or {}).get("sha")
                        if pr_head != state.get("head_sha"):
                            return
                        evidence_service = getattr(self.controller, "evidence_service", None)
                        if evidence_service is None:
                            logger.warning(
                                "watcher native evidence unavailable %s#%s comment=%s",
                                repo, pr_number, comment_id,
                            )
                            return
                        user = fresh_comment_obj.get("user", {})
                        actor_id = user.get("id") if isinstance(user, dict) else None
                        actor_login = user.get("login") if isinstance(user, dict) else None
                        if (isinstance(actor_id, bool)
                                or not str(actor_id or "").isdigit()
                                or not isinstance(actor_login, str) or not actor_login.strip()):
                            logger.info(
                                "watcher native evidence skipped %s#%s comment=%s reason=identity_missing",
                                repo, pr_number, comment_id,
                            )
                            if state.get("head_sha"):
                                await self._consume_comment_cursor(repo, pr_number, comment_id)
                            return
                        try:
                            await evidence_service.handle_native_comment(
                                repo=repo, pr_number=pr_number, comment_id=comment_id,
                                actor_id=int(actor_id), actor_login=actor_login,
                                action="submit_evidence", arguments={}, automatic=True,
                            )
                        except EvidenceError as e:
                            terminal_skip = (
                                e.status_code in {400, 401, 403, 404, 409, 410, 413}
                                and e.code not in {
                                    "attachment_retry_conflict", "comment_unavailable",
                                    "object_not_verified",
                                }
                            )
                            if terminal_skip:
                                logger.info(
                                    "watcher native evidence skipped %s#%s comment=%s reason=%s",
                                    repo, pr_number, comment_id, e.code,
                                )
                                await self._consume_comment_cursor(repo, pr_number, comment_id)
                                return
                            logger.warning(
                                "watcher native evidence retry %s#%s comment=%s reason=%s",
                                repo, pr_number, comment_id, e.code,
                            )
                            return
                        except Exception as e:
                            logger.warning(
                                "watcher native evidence retry %s#%s comment=%s error=%s",
                                repo, pr_number, comment_id, type(e).__name__,
                            )
                            return
                        await self._consume_comment_cursor(repo, pr_number, comment_id)
                        try:
                            state = await self.proxy.read_hidden_state(repo, pr_number)
                        except Exception:
                            return
                        cursor_after = (state.get("last_processed_comment_id", 0)
                                        if isinstance(state, dict) else 0)
                        if (not isinstance(cursor_after, int) or isinstance(cursor_after, bool)
                                or cursor_after < comment_id):
                            return
                        native_comments_processed += 1
                        continue
                # Non-command comment — use cursor-only persistence
                if state.get("head_sha"):
                    state = await self.proxy.persist_cursor(
                        repo, pr_number, comment_id,
                    ) or state
                continue

            cmd = commands_mod.parse_command(fresh_body)
            if not cmd.is_command:
                self._finish_recent_command(
                    repo, pr_number, comment_id,
                    failure_reason="recognized_command_rejected_by_parser",
                    retryable=False,
                )
                if state.get("head_sha"):
                    state = await self.proxy.persist_cursor(
                        repo, pr_number, comment_id,
                    ) or state
                continue

            self._mark_recent_command_processing(repo, pr_number, comment_id)

            # Keep the existing one-business-action-per-PR-cycle behavior:
            # comments after this cycle's recovered evidence wait for the next
            # poll before command dispatch.
            if native_comments_processed:
                return

            cutoff = self._repo_baseline_cutoff_utc.get(repo)
            created_at = fresh_comment_obj.get("created_at")
            if (not isinstance(cutoff, datetime) or cutoff.tzinfo is None
                    or not isinstance(created_at, str) or not created_at.strip()):
                self._finish_recent_command(
                    repo, pr_number, comment_id, failure_reason="comment_timestamp_unavailable",
                )
                logger.warning(
                    "watcher command timestamp unavailable %s#%s #%s — fail closed",
                    repo, pr_number, comment_id,
                )
                return
            try:
                created_utc = datetime.fromisoformat(
                    created_at.strip().replace("Z", "+00:00")
                )
            except ValueError:
                self._finish_recent_command(
                    repo, pr_number, comment_id, failure_reason="comment_timestamp_invalid",
                )
                return
            if created_utc.tzinfo is None or created_utc.utcoffset() is None:
                self._finish_recent_command(
                    repo, pr_number, comment_id, failure_reason="comment_timestamp_timezone_missing",
                )
                return
            if created_utc.astimezone(timezone.utc) <= cutoff.astimezone(timezone.utc):
                # A pre-generation command can be safely consumed as history,
                # but must never be dispatched as a fresh approval.
                await self._consume_comment_cursor(repo, pr_number, comment_id)
                return
            updated_at = fresh_comment_obj.get("updated_at")
            if not isinstance(updated_at, str) or not updated_at.strip():
                self._finish_recent_command(
                    repo, pr_number, comment_id, failure_reason="comment_update_time_unavailable",
                )
                return
            try:
                updated_utc = datetime.fromisoformat(
                    updated_at.strip().replace("Z", "+00:00")
                )
            except ValueError:
                self._finish_recent_command(
                    repo, pr_number, comment_id, failure_reason="comment_update_time_invalid",
                )
                return
            if updated_utc.tzinfo is None or updated_utc.utcoffset() is None:
                self._finish_recent_command(
                    repo, pr_number, comment_id, failure_reason="comment_update_timezone_missing",
                )
                return
            if updated_utc.astimezone(timezone.utc) != created_utc.astimezone(timezone.utc):
                # GitHub edits retain created_at; an edited command is not a
                # newly authored approval and is terminally skipped.
                await self._consume_comment_cursor(repo, pr_number, comment_id)
                return

            # TOCTOU recheck: the auth snapshot was validated at cycle start,
            # but awaits have run since. Fail closed if authorization was
            # revoked while we were reading GitHub state.
            if (not self.config.auth_valid or repo not in self.config.active_repos
                    or repo not in self.config.github_repos):
                self._finish_recent_command(
                    repo, pr_number, comment_id, failure_reason="authorization_changed",
                )
                logger.warning(
                    "watcher auth snapshot invalid before dispatch %s#%s \u2014 fail closed",
                    repo, pr_number,
                )
                return

            # Dispatch to controller — transient failure must NOT advance cursor
            try:
                handled = await self.controller.on_command(
                    cmd, repo, pr_number, comment_id,
                )
                if handled:
                    self._finish_recent_command(repo, pr_number, comment_id)
                    await self._consume_comment_cursor(repo, pr_number, comment_id)
                else:
                    self._finish_recent_command(
                        repo, pr_number, comment_id,
                        failure_reason="command_handler_returned_unfinished",
                    )
                # False means unfinished work (including COS failure). Never
                # consume a later comment past it; retry from fresh state next poll.
                return
            except DeployCommandError as e:
                self._finish_recent_command(
                    repo, pr_number, comment_id,
                    failure_reason=type(e).__name__, retryable=False,
                )
                logger.warning(
                    "command %s comment %s in %s#%s: %s",
                    cmd.kind, comment_id, repo, pr_number, e,
                )
                # Invalid/unauthorized commands are terminally handled — advance cursor
                # only when an authoritative lifecycle already exists.
                await self._consume_comment_cursor(repo, pr_number, comment_id)
                return
            except Exception as e:
                self._finish_recent_command(
                    repo, pr_number, comment_id, failure_reason=type(e).__name__,
                )
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
