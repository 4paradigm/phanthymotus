"""Data models and errors for PR review jobs."""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


# ── Errors ────────────────────────────────────────────────────────────────────


class ReviewError(Exception):
    """Base for pipeline errors.

    `retryable` decides whether the worker tries again. Conditions caused by
    the PR itself (merge conflict) are terminal — retrying just spends another
    hour reporting the same thing. Infrastructure problems (network, git,
    registry, timeouts) are worth another attempt.
    """

    retryable = True


class MergeConflictError(ReviewError):
    """PR cannot be merged onto main — the author must resolve it."""

    retryable = False


class JobTimeoutError(ReviewError):
    """Job exceeded its whole-job wall-clock budget and is presumed lost."""

    retryable = True


class EmptyReviewError(ReviewError):
    """The reviewer failed and left nothing written behind.

    Terminal on purpose. The review has already been re-tried in place — twice
    over, once per model call and once over the whole loop — so a third pass
    would only rebuild every image again to ask the same broken gateway the same
    question. An error comment says what happened; the placeholder review this
    replaces read as "the reviewer had no comments", which is worse than
    silence.
    """

    retryable = False


# ── Enums ─────────────────────────────────────────────────────────────────────


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    RETRYING = "retrying"
    BUILD_SUCCESS = "build_success"
    BUILD_FAILED = "build_failed"
    REVIEW_DONE = "review_done"
    TIMEOUT = "timeout"
    ERROR = "error"
    CANCELLED = "cancelled"


class BuildTarget(str, Enum):
    CORE = "core"
    PERCEPTION = "perception"
    ACTUCORE = "actucore"
    DRIVER = "driver"


# Perception and actucore are built for Jetson only — that is where they run, so
# neither script takes a `--variant`. What is selectable is the JetPack version,
# which picks the base image and lands in the tag as
# `release.YYMMDD.<sha>-jetson-jp<ver>`.
#
# These are the versions `deploy/build_perception.sh` and
# `deploy/build_actucore.sh` accept; they exit 1 on anything else, so an
# unrecognised one must never reach them.
SUPPORTED_JP_VERSIONS = ("5.11", "6.1")
DEFAULT_JP_VERSION = "5.11"


# Statuses from which a job will never advance. Plain strings, because the
# store hands back rows where status is a string, not an enum member.
TERMINAL_STATUSES = frozenset({
    JobStatus.BUILD_FAILED.value,
    JobStatus.REVIEW_DONE.value,
    JobStatus.TIMEOUT.value,
    JobStatus.ERROR.value,
    JobStatus.CANCELLED.value,
})

# Terminal statuses that actually produced an answer for the requester. Only
# these should block a repeat trigger on the same commit.
#
# The rest — cancelled, timeout, error — are terminal but delivered nothing, so
# re-triggering is the correct response to them, not something to refuse. A job
# killed by a restart or by an infrastructure failure must not leave a commit
# permanently un-reviewable.
CONCLUSIVE_STATUSES = frozenset({
    JobStatus.REVIEW_DONE.value,
    JobStatus.BUILD_FAILED.value,
})


class Stage(str, Enum):
    """Where in the pipeline a running job is.

    `status` alone is too coarse: a job sits at RUNNING through a fetch, a
    worktree merge, several builds, and the review. Without this the dashboard
    shows "running" with nothing else for minutes at a time.
    """

    QUEUED = "queued"
    FETCHING = "fetching refs"
    WORKTREE = "preparing worktree"
    DETECTING = "detecting changes"
    BUILDING = "building"
    RULE_CHECKS = "running rule checks"
    LLM_REVIEW = "generating review"
    POSTING = "posting results"
    DONE = "done"


# ── Data ──────────────────────────────────────────────────────────────────────


@dataclass
class BuildResult:
    target: BuildTarget
    driver_path: str | None  # e.g. "unitree/g1", only for DRIVER
    # None means "still building". The worker persists a placeholder row before
    # a build starts so the dashboard has a log pane to tail; encoding that as
    # False made an in-progress build render as FAILED.
    success: bool | None
    image_tag: str  # full image ref when successful
    log_tail: str  # last N lines, for the PR comment
    log_path: str = ""  # full log on disk, for the dashboard
    # Container the target's deploy/service.yml declares. Set only when the
    # target ships a parseable fragment, so it doubles as the signal that
    # deploy/run-pr-image.sh will work for this image.
    container_name: str = ""
    # JetPack version for a perception build, "" for core and drivers. Part of
    # the build's identity, not a detail of it: one job can produce two
    # perception images, and only this tells them apart.
    variant: str = ""
    # Wall clock for the whole script invocation, None while still building.
    # Recorded because "was it slow or was it stuck?" is the first question
    # asked of a long build, and reading it off log timestamps is tedious.
    duration_seconds: float | None = None
    # Why the agent killed this build, if it did: "idle" (went quiet) or "cap"
    # (hit the absolute wall clock). "" for a build that ended on its own,
    # including one that failed to compile — that distinction is the point.
    timeout_kind: str = ""

    def label(self) -> str:
        return build_label(self.target, self.driver_path, self.variant)


def build_label(
    target: BuildTarget, driver_path: str | None, variant: str = ""
) -> str:
    """Human name for one build: what the comment and the dashboard both show.

    In one place because a job can now contain two builds of the same target,
    and a label that omits the variant would render them as duplicates.
    """
    name = driver_path or target.value
    if variant:
        return f"{name} (jetson-jp{variant})"
    return name


@dataclass
class ReviewJob:
    repo_full_name: str  # "4paradigm/phanthymotus"
    pr_number: int
    pr_head_sha: str
    pr_head_ref: str  # branch name
    pr_base_ref: str  # e.g. "main"
    comment_id: int  # triggering comment
    requester: str  # GitHub username
    source: str = "webhook"  # "webhook" | "poll"
    # Who opened the PR. Distinct from `requester`, who merely typed the trigger
    # and is often a reviewer or the bot operator; the dashboard attributes work
    # to the author, while the acknowledgment comment still thanks the requester.
    pr_author: str = ""
    # The PR's own account of itself. Captured at trigger time off the get_pr
    # response already being fetched, so it costs no extra API call and survives
    # a retry without one.
    pr_title: str = ""
    pr_body: str = ""
    # Summary of what the filter actually fed the reviewer: how many comments
    # survived, how many were dropped, whether the description was usable. Kept
    # on the job so the dashboard does not have to wait for the trace to load.
    pr_context: dict = field(default_factory=dict)

    # Options parsed from the command
    skip_build: bool = False
    build_only: bool = False
    force_targets: list[str] = field(default_factory=list)  # e.g. ["core"]
    # JetPack versions to build perception for, in the order requested. Empty
    # means the default; two entries mean two images from one job.
    perception_variants: list[str] = field(default_factory=list)

    # Runtime state
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: JobStatus = JobStatus.QUEUED
    # Finer-grained progress within RUNNING, plus a free-text detail such as
    # "1/2 unitree/g1" so the dashboard can say what is being built.
    stage: str = Stage.QUEUED.value
    stage_detail: str = ""
    stage_started_at: datetime | None = None
    attempt: int = 0
    attempt_errors: list[str] = field(default_factory=list)
    build_results: list[BuildResult] = field(default_factory=list)
    review_text: str = ""
    # Rule-check findings as plain dicts, so they survive persistence and can
    # be rendered by the dashboard rather than only formatted into a comment.
    findings: list[dict] = field(default_factory=list)
    # Deterministic pre-review results, kept so the dashboard and the comment
    # render the same numbers the loop was told about.
    large_files: list[dict] = field(default_factory=list)
    infra_files: list[str] = field(default_factory=list)
    shared_base_files: list[str] = field(default_factory=list)
    # How the review loop ended. A review cut short must not look like a review
    # that found nothing, so this is persisted rather than inferred.
    review_rounds: int = 0
    review_stopped_reason: str = ""
    review_tool_calls: int = 0
    error: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    worktree_path: str = ""
    # The three ids a released image has to be traceable through:
    #
    #   pr_head_sha     what the author pushed
    #   build_ref_sha   worktree HEAD after the PR is merged onto base — the one
    #                   the build scripts shorten into release.YYMMDD.<7hex>, so
    #                   the only id that finds the published image
    #   merge_commit_sha  the commit on the base branch once the PR is merged;
    #                   backfilled by the poller, empty until then
    build_ref_sha: str = ""
    merge_commit_sha: str = ""
    merged_at: str = ""  # ISO8601 from GitHub, alongside merge_commit_sha
    # Acknowledgment comment, edited in place through the job's lifetime.
    progress_comment_id: int | None = None

    def elapsed_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds()

    def stage_elapsed_seconds(self) -> float | None:
        """How long the current stage has been running.

        Surfaced so a stage that is taking unusually long (a slow fetch, a long
        build) is visible as such rather than looking like a hang.
        """
        if self.stage_started_at is None:
            return None
        end = self.finished_at or datetime.now(timezone.utc)
        return (end - self.stage_started_at).total_seconds()

    def set_stage(self, stage: "Stage", detail: str = ""):
        self.stage = stage.value
        self.stage_detail = detail
        self.stage_started_at = datetime.now(timezone.utc)


# ── Command parsing ───────────────────────────────────────────────────────────

TRIGGER = "/request_bot_review"

# A JetPack version token: `jetson-5.11`, `jetson-jp6.1`, `jp5.11`, or the bare
# version. Only perception has variants, so no target prefix is required — and a
# version on its own is taken as a request to build perception.
JP_TOKEN_PATTERN = re.compile(r"^(?:jetson-)?(?:jp)?(\d+\.\d+)$", re.IGNORECASE)


def parse_trigger_command(comment_body: str) -> dict | None:
    """Parse a `/request_bot_review` command out of a comment body.

    Returns {"skip_build", "build_only", "force", "force_targets",
    "perception_variants"}, or None when the comment does not contain the
    trigger.
    """
    if not comment_body:
        return None

    for raw_line in comment_body.splitlines():
        # Tolerate markdown quote/list prefixes, but require the trigger to
        # start the line so it is not picked up from surrounding prose.
        line = raw_line.strip().lstrip(">*- \t")
        if not line.lower().startswith(TRIGGER):
            continue

        args = line[len(TRIGGER):].split()
        result = {
            "skip_build": False,
            "build_only": False,
            "force": False,
            "force_targets": [],
            "perception_variants": [],
        }
        for arg in args:
            token = arg.strip().strip("`,")
            lowered = token.lower()
            if lowered == "skip-build":
                result["skip_build"] = True
            elif lowered == "build-only":
                result["build_only"] = True
            elif lowered in ("force", "--force", "-f"):
                # Re-review a commit that was already reviewed.
                result["force"] = True
            elif lowered in ("core", "perception"):
                result["force_targets"].append(lowered)
            elif JP_TOKEN_PATTERN.match(lowered):
                version = JP_TOKEN_PATTERN.match(lowered).group(1)
                if version not in SUPPORTED_JP_VERSIONS:
                    # Dropped rather than passed through: the build script exits
                    # 1 on an unknown version, which would fail the whole job.
                    # The build-in-progress comment lists what will actually be
                    # built, so the drop is visible on the PR.
                    logger.warning(
                        f"Ignoring unsupported JetPack version {token!r} "
                        f"(supported: {', '.join(SUPPORTED_JP_VERSIONS)})"
                    )
                    continue
                if version not in result["perception_variants"]:
                    result["perception_variants"].append(version)
                # Asking for a version is asking for perception: the token means
                # nothing for any other target.
                if "perception" not in result["force_targets"]:
                    result["force_targets"].append("perception")
            elif "/" in token:
                # A driver path such as unitree/g1
                result["force_targets"].append(token)
        return result

    return None
