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
from collections import deque
from datetime import datetime, timezone

from . import commands as commands_mod
from .config import Config, resolve_authorized_repos
from .github_client import GitHubClient, GitHubError
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

    def enqueue_pr(self, repo: str, pr_number: int) -> None:
        """Prioritize an authorized PR after a webhook observes a new comment."""
        if (not isinstance(repo, str)
                or not getattr(self.config, "auth_valid", False)
                or repo not in self.config.active_repos
                or repo not in self.config.github_repos
                or type(pr_number) is not int
                or pr_number <= 0):
            return
        item = (repo, pr_number)
        if item in self._priority_pr_set:
            return
        if len(self._priority_pr_queue) >= 128:
            evicted = self._priority_pr_queue.popleft()
            self._priority_pr_set.discard(evicted)
        self._priority_pr_queue.append(item)
        self._priority_pr_set.add(item)

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
            await asyncio.sleep(self.config.poll_interval_seconds)

    async def _refresh_active_repos_if_needed(self) -> None:
        """Periodically refresh runtime authorization before polling."""
        if self._github is None or self._github_auth is None:
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
        except Exception as e:
            logger.warning(
                "auth refresh failed: %s \u2014 failing closed",
                type(e).__name__,
            )
            self.config.auth_valid = False

    async def _poll_once(self) -> None:
        """One poll cycle over all configured repos and open PRs.

        Processes repos/PRs/comments deterministically and sequentially.
        No asyncio.gather for mutating PR work.

        Uses only runtime-authorized active_repos. Never falls back to
        DESIRED_REPOS (config.github_repos). If active_repos is empty or
        auth snapshot is invalid, skip polling this cycle.
        """
        if not self.config.active_repos or not getattr(self.config, "auth_valid", True):
            return
        repos = list(dict.fromkeys(self.config.active_repos))
        if repos:
            repo_start = self._repo_poll_cursor % len(repos)
            repos = repos[repo_start:] + repos[:repo_start]
        for repo in repos:
            if (not self.config.auth_valid or repo not in self.config.active_repos
                    or repo not in self.config.github_repos):
                continue
            try:
                prs = await self.proxy.get_open_prs(repo)
            except GitHubError as e:
                logger.warning("watcher list PRs %s: %s", repo, e)
                continue
            except Exception as e:
                logger.warning("watcher list PRs %s failed: %s", repo, type(e).__name__)
                continue
            baseline_all = repo in self._pending_baseline_repos
            baseline_done = self._baseline_progress.get(repo, set())
            valid_prs = []
            for pr in prs:
                try:
                    pr_number = int(pr.get("number") or 0)
                except (TypeError, ValueError):
                    continue
                if pr_number > 0:
                    valid_prs.append((pr_number, pr))
            valid_prs.sort(key=lambda item: item[0])
            open_pr_numbers = [number for number, _ in valid_prs]

            # Each repo gets a hard cap for all PR work, including command
            # processing, baseline attempts and evidence projection.
            pr_work_budget = 8
            fast_lane_size = 6
            by_number = {number: pr for number, pr in valid_prs}
            selected_numbers: list[int] = []
            selected_set: set[int] = set()
            priority_numbers: list[int] = []

            # Drain this repo's queued work first. The Open PR response above
            # is the authority: stale, closed, merged, or missing entries are
            # discarded without invoking any PR writer.
            queued_count = len(self._priority_pr_queue)
            for _ in range(queued_count):
                queued_repo, queued_number = self._priority_pr_queue.popleft()
                queued_item = (queued_repo, queued_number)
                if (queued_repo not in self.config.active_repos
                        or queued_repo not in self.config.github_repos):
                    self._priority_pr_set.discard(queued_item)
                    continue
                if queued_repo != repo:
                    self._priority_pr_queue.append(queued_item)
                    continue
                queued_pr = by_number.get(queued_number)
                if (queued_pr is None
                        or queued_pr.get("state", "open") != "open"
                        or queued_pr.get("merged")):
                    self._priority_pr_set.discard(queued_item)
                    continue
                if len(selected_numbers) >= pr_work_budget:
                    self._priority_pr_queue.append(queued_item)
                    continue
                self._priority_pr_set.discard(queued_item)
                selected_numbers.append(queued_number)
                selected_set.add(queued_number)
                priority_numbers.append(queued_number)

            recent_pool = sorted(
                valid_prs,
                key=lambda item: str(item[1].get("updated_at") or ""),
                reverse=True,
            )[:12]
            if len(recent_pool) > 1:
                rotating_recent = recent_pool[1:]
                recent_start = self._recent_pr_cursor.get(repo, 0) % len(rotating_recent)
                recent_order = rotating_recent[recent_start:] + rotating_recent[:recent_start]
                fast_lane = [recent_pool[0], *recent_order[:fast_lane_size - 1]]
                self._recent_pr_cursor[repo] = (recent_start + 1) % len(rotating_recent)
            else:
                fast_lane = list(recent_pool)
            recent_added = 0
            for number, _ in fast_lane:
                if len(selected_numbers) >= pr_work_budget:
                    break
                if number in selected_set:
                    continue
                selected_numbers.append(number)
                selected_set.add(number)
                recent_added += 1
                if recent_added >= fast_lane_size or len(selected_numbers) >= pr_work_budget:
                    break

            # Sort by PR number above, then rotate from the saved position.
            # Modulo adjustment handles PR additions/removals; API ordering
            # changes do not affect this deterministic scan order.
            scan_start = (
                self._pr_scan_cursor.get(repo, 0) % len(open_pr_numbers)
                if open_pr_numbers else 0
            )
            ordinary_order = open_pr_numbers[scan_start:] + open_pr_numbers[:scan_start]
            ordinary_slots = max(0, pr_work_budget - len(selected_numbers))
            ordinary_added = 0
            ordinary_scanned = 0
            ordinary_cursor_after = {}
            for pr_number in ordinary_order:
                ordinary_scanned += 1
                if pr_number in selected_set:
                    continue
                selected_numbers.append(pr_number)
                selected_set.add(pr_number)
                ordinary_added += 1
                ordinary_cursor_after[pr_number] = (
                    scan_start + ordinary_scanned
                ) % len(open_pr_numbers)
                if ordinary_added >= ordinary_slots:
                    break

            processed_prs: list[int] = []
            for pr_number in selected_numbers[:pr_work_budget]:
                if (not self.config.auth_valid or repo not in self.config.active_repos
                        or repo not in self.config.github_repos):
                    break
                pr = by_number[pr_number]
                processed_prs.append(pr_number)
                try:
                    if baseline_all and pr_number not in baseline_done:
                        # Baseline only. Never dispatch historical comments in
                        # the same operation that establishes the cursor.
                        success = await self._baseline_pr(repo, pr_number)
                        if success:
                            self._baseline_progress.setdefault(repo, set()).add(pr_number)
                    else:
                        await self._process_pr(repo, pr_number)
                    # The existing serial watcher remains the sole writer of
                    # durable attachment projections.
                    await self.controller.project_evidence(repo, pr_number)
                except Exception as e:
                    # A failed PR consumes only its own slot; later selected
                    # PRs still get their serial attempt this cycle.
                    logger.error("watcher process %s#%s: %s", repo, pr_number, e)
                try:
                    current_pr = None
                    if (self.config.auth_valid and repo in self.config.active_repos
                            and repo in self.config.github_repos):
                        current_pr = await self.proxy.get_pr(repo, pr_number)
                    if (isinstance(current_pr, dict)
                            and current_pr.get("state") == "open"
                            and not current_pr.get("merged")
                            and not current_pr.get("draft")):
                        await self.proxy.ensure_visible_evidence_archive_comments(
                            repo, pr_number, limit=2,
                        )
                except Exception as e:
                    logger.warning(
                        "watcher evidence archive visibility %s#%s: %s",
                        repo, pr_number, type(e).__name__,
                    )
                if pr_number in ordinary_cursor_after:
                    self._pr_scan_cursor[repo] = ordinary_cursor_after[pr_number]
            # Baselining uses per-PR progress tracking: only discard from
            # _pending_baseline_repos when ALL open PRs for this repo have
            # been successfully baselined this generation.  PRs whose
            # baseline persist failed remain unbaselined and will be retried
            # on the next cycle — commands are NOT dispatched while pending.
            if baseline_all:
                self._baseline_progress.setdefault(repo, set())
                all_baselined = all(
                    pr_number in self._baseline_progress[repo]
                    for pr_number in open_pr_numbers
                )
                if all_baselined:
                    self._pending_baseline_repos.discard(repo)
            remaining = (
                sum(n not in self._baseline_progress.get(repo, set()) for n in open_pr_numbers)
                if repo in self._pending_baseline_repos else 0
            )
            logger.info(
                "watcher poll %s selected_prs=%s priority_prs=%s processed_prs=%s remaining_baselines=%s scan_cursor=%s",
                repo, selected_numbers[:pr_work_budget], priority_numbers,
                processed_prs, remaining, self._pr_scan_cursor.get(repo, 0),
            )
        if repos:
            self._repo_poll_cursor = (repo_start + 1) % len(repos)

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
                if state.get("head_sha"):
                    state = await self.proxy.persist_cursor(
                        repo, pr_number, comment_id,
                    ) or state
                continue

            # Keep the existing one-business-action-per-PR-cycle behavior:
            # comments after this cycle's recovered evidence wait for the next
            # poll before command dispatch.
            if native_comments_processed:
                return

            cutoff = self._repo_baseline_cutoff_utc.get(repo)
            created_at = fresh_comment_obj.get("created_at")
            if (not isinstance(cutoff, datetime) or cutoff.tzinfo is None
                    or not isinstance(created_at, str) or not created_at.strip()):
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
                return
            if created_utc.tzinfo is None or created_utc.utcoffset() is None:
                return
            if created_utc.astimezone(timezone.utc) <= cutoff.astimezone(timezone.utc):
                # A pre-generation command can be safely consumed as history,
                # but must never be dispatched as a fresh approval.
                await self._consume_comment_cursor(repo, pr_number, comment_id)
                return
            updated_at = fresh_comment_obj.get("updated_at")
            if not isinstance(updated_at, str) or not updated_at.strip():
                return
            try:
                updated_utc = datetime.fromisoformat(
                    updated_at.strip().replace("Z", "+00:00")
                )
            except ValueError:
                return
            if updated_utc.tzinfo is None or updated_utc.utcoffset() is None:
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
                    await self._consume_comment_cursor(repo, pr_number, comment_id)
                # False means unfinished work (including COS failure). Never
                # consume a later comment past it; retry from fresh state next poll.
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
