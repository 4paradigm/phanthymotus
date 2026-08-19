"""Shared trigger logic — creates jobs from PR comments.

Used by both the webhook receiver and the polling loop, so the two entry points
behave identically and share dedup.

Repeat-trigger policy, keyed on the commit rather than the PR:

- **New commit** → new review. Pushing a fix and re-triggering is the normal
  workflow, so it must work.
- **Same commit, still in flight** → skipped, with a comment saying so. Running
  two builds of one commit wastes a worker for 20 minutes to print the same
  answer twice.
- **Same commit, already reviewed** → skipped, with a comment pointing at the
  earlier result and mentioning `force`. Silently redoing it is worse: the
  requester waits, unsure whether anything is happening.
- **Same commit, previous attempt produced no result** (cancelled by a restart,
  timed out, or errored) → allowed. Those delivered nothing, so refusing would
  leave the commit permanently un-reviewable.
- **`/request_bot_review force`** → re-review regardless.

The check reads SQLite, not the in-memory queue, because the queue is empty
after a restart and would let a completed review be redone unnoticed.
"""

import logging

from . import comments
from .config import Config
from .github_client import GitHubClient
from .models import (
    CONCLUSIVE_STATUSES,
    TERMINAL_STATUSES,
    ReviewJob,
    parse_trigger_command,
)

logger = logging.getLogger(__name__)


async def create_job_from_comment(
    repo_full_name: str,
    pr_number: int,
    comment_id: int,
    comment_body: str,
    requester: str,
    config: Config,
    github_client: GitHubClient,
    job_queue,
    store,
    source: str = "webhook",
) -> ReviewJob | None:
    """Parse a PR comment and enqueue a review job if it contains the trigger.

    Returns the enqueued job, or None if the comment was ignored or skipped.
    """
    trigger = parse_trigger_command(comment_body)
    if trigger is None:
        return None

    if repo_full_name not in config.repos:
        logger.warning(f"Ignoring trigger for unconfigured repo: {repo_full_name}")
        return None

    try:
        pr_info = await github_client.get_pr(repo_full_name, pr_number)
    except Exception as e:
        logger.error(f"Failed to fetch PR {repo_full_name}#{pr_number}: {e}")
        return None

    if pr_info.get("state") != "open":
        logger.info(f"Skipping {repo_full_name}#{pr_number}: PR is not open")
        return None

    head_sha = pr_info["head"]["sha"]

    # Acknowledge receipt before any skip decision, so the requester can always
    # tell the comment was seen.
    await github_client.add_reaction(repo_full_name, comment_id, "eyes")

    if not trigger["force"]:
        skip = await _should_skip(
            repo_full_name, pr_number, head_sha, job_queue, store
        )
        if skip is not None:
            logger.info(
                f"Skipping {repo_full_name}#{pr_number}@{head_sha[:7]}: {skip[0]}"
            )
            try:
                await github_client.post_comment(
                    repo_full_name, pr_number, skip[1]
                )
            except Exception as e:
                logger.warning(f"Failed to post skip notice: {e}")
            return None

    job = ReviewJob(
        repo_full_name=repo_full_name,
        pr_number=pr_number,
        pr_head_sha=head_sha,
        pr_head_ref=pr_info["head"]["ref"],
        pr_base_ref=pr_info["base"]["ref"],
        pr_title=pr_info.get("title") or "",
        pr_body=pr_info.get("body") or "",
        pr_author=(pr_info.get("user") or {}).get("login", ""),
        comment_id=comment_id,
        requester=requester,
        source=source,
        skip_build=trigger["skip_build"],
        build_only=trigger["build_only"],
        force_targets=trigger["force_targets"],
        perception_variants=trigger["perception_variants"],
    )

    # Acknowledge with a comment immediately. With polling this matters — the
    # trigger may sit for up to one poll interval, and without a reply the
    # requester cannot tell whether the agent saw it. The worker then edits this
    # same comment through the build and result stages.
    try:
        job.progress_comment_id = await github_client.post_comment(
            repo_full_name,
            pr_number,
            comments.format_ack(
                requester=requester,
                head_sha=head_sha,
                skip_build=job.skip_build,
                build_only=job.build_only,
                source=source,
            ),
        )
    except Exception as e:
        # Not fatal — the worker will post a fresh comment when it starts.
        logger.warning(f"Failed to post acknowledgment comment: {e}")

    await job_queue.enqueue(job)
    logger.info(
        f"Enqueued job {job.id} via {source}: "
        f"{repo_full_name}#{pr_number}@{head_sha[:7]} by {requester}"
    )
    return job


async def _should_skip(
    repo: str, pr_number: int, head_sha: str, job_queue, store
) -> tuple[str, str] | None:
    """Decide whether to skip this trigger. Returns (reason, comment) or None.

    Scoped to the exact commit: a different commit is always a new review.
    """
    # In-flight in this process — authoritative and cheapest.
    if job_queue.has_pending_job(repo, pr_number, head_sha):
        return (
            "already in flight",
            comments.format_skipped_in_flight(head_sha),
        )

    # Recorded in SQLite: covers both a job running under a previous process
    # and a review that already completed for this commit.
    try:
        prior = await store.find_jobs_for_commit(repo, pr_number, head_sha)
    except Exception as e:
        # Never block a review because the dedup lookup failed.
        logger.warning(f"Dedup lookup failed, allowing trigger: {e}")
        return None

    if not prior:
        return None

    active = [j for j in prior if j["status"] not in TERMINAL_STATUSES]
    if active:
        return (
            "already in flight (persisted)",
            comments.format_skipped_in_flight(head_sha),
        )

    # Only a review that produced an answer blocks a repeat. A job that was
    # cancelled, timed out, or errored delivered nothing, so re-triggering is
    # the right response — otherwise a restart or an infrastructure failure
    # would leave that commit permanently un-reviewable.
    conclusive = [j for j in prior if j["status"] in CONCLUSIVE_STATUSES]
    if not conclusive:
        logger.info(
            f"Prior attempt(s) for {head_sha[:7]} ended without a result "
            f"({', '.join(sorted({j['status'] for j in prior}))}) — allowing retry"
        )
        return None

    done = conclusive[0]
    return (
        f"already reviewed ({done['status']})",
        comments.format_skipped_already_reviewed(
            head_sha, done["status"], done["finished_at"]
        ),
    )
