"""PR comment formatting.

Lives in its own module so both `trigger` (acknowledgment) and `worker`
(progress / results) can format comments without importing each other.
"""

from .builder import split_image_ref
from .models import BuildResult
from .reviewer import Finding

# Marker prefixed to every bot comment, so bot comments are identifiable.
BOT_MARKER = "<!-- pr-review-agent -->"

# GitHub rejects an issue comment body over 65536 characters with a 422. That
# would lose the entire failure report — the one comment the author most needs —
# so failed build logs are budgeted against it rather than cut to a fixed line
# count. The margin absorbs the surrounding table and the wrappers.
GITHUB_COMMENT_LIMIT = 65536
COMMENT_BUDGET = 60000
# Below this a log fragment is too small to show the error with any context, so
# many-failure jobs (a driver PR touching a dozen drivers) link to the dashboard
# instead of padding the comment with a dozen unreadable snippets.
MIN_USEFUL_LOG_CHARS = 1500

MODE_LABELS = {
    (False, False): "Build + Review",
    (True, False): "Review only (build skipped)",
    (False, True): "Build only (review skipped)",
    (True, True): "No-op",
}


def format_ack(
    requester: str,
    head_sha: str,
    skip_build: bool,
    build_only: bool,
    source: str,
) -> str:
    """Immediate acknowledgment posted when a job is accepted.

    This same comment is edited in place to show build progress and results,
    so a PR gets one comment that tracks progress rather than one per stage.
    """
    mode = MODE_LABELS.get((skip_build, build_only), "Build + Review")
    source_label = "polling" if source == "poll" else "webhook"

    return f"""{BOT_MARKER}
## PR Review Agent

Request from @{requester} accepted — starting review.

| | |
|---|---|
| Commit | `{head_sha[:7]}` |
| Mode | {mode} |
| Triggered via | {source_label} |
| Status | Queued |
"""


def format_building(
    requester: str,
    head_sha: str,
    labels: list[str],
) -> str:
    """Build-in-progress state.

    Takes the resolved build plan rather than the targets, so a perception build
    for two JetPack versions is announced as two builds — which is what will
    happen, and how long it will take.
    """
    listed = ", ".join(f"`{n}`" for n in labels) if labels else "None"
    return f"""{BOT_MARKER}
## PR Review Agent

| | |
|---|---|
| Commit | `{head_sha[:7]}` |
| Build targets | {listed} |
| Status | Building... |

Builds usually take 5–20 minutes, and longer when something compiles from source.
This comment will be updated when done.
"""


def format_build_result(head_sha: str, results: list[BuildResult]) -> str:
    """Final build state — success or failure, with logs for failures."""
    rows = []
    for r in results:
        name = r.label()
        took = _fmt_took(r.duration_seconds)
        if r.success:
            _, tag = split_image_ref(r.image_tag)
            version = f"`{tag}`" if tag else "—"
            rows.append(
                f"| {name} | :white_check_mark: Success | {version} | {took} |"
            )
        elif r.timeout_kind:
            rows.append(f"| {name} | :hourglass: Killed | — | {took} |")
        else:
            rows.append(f"| {name} | :x: Failed | — | {took} |")

    table = (
        "| Target | Status | Version | Took |\n"
        "|--------|--------|---------|------|\n" + "\n".join(rows)
    )

    all_ok = all(r.success for r in results)
    killed = [r for r in results if r.timeout_kind]
    if all_ok:
        headline = "All builds succeeded."
    elif killed and not [r for r in results if not r.success and not r.timeout_kind]:
        headline = "A build did not finish — the agent stopped it. See below."
    else:
        headline = "Build failed. See the collapsed logs below."

    body = f"""{BOT_MARKER}
## PR Review Agent — Build Result

Commit: `{head_sha[:7]}`

{headline}

{table}
"""

    # A killed build is not a failed build, and reading the log tail alone would
    # suggest the last compiler line was the error. Said before the logs so it is
    # visible without expanding them.
    for r in killed:
        took = _fmt_took(r.duration_seconds)
        if r.timeout_kind == "idle":
            body += (
                f"\n> :hourglass: `{r.label()}` was killed for going quiet, not "
                f"for failing to build: it ran for {took} and then produced no "
                f"output for long enough to be presumed stuck. The log below "
                f"ends where it went silent. If this build legitimately goes "
                f"quiet for that long, raise `BUILD_IDLE_TIMEOUT_SECONDS`.\n"
            )
        else:
            body += (
                f"\n> :hourglass: `{r.label()}` hit the absolute build time "
                f"limit after {took} and was killed while still producing "
                f"output. Raise `BUILD_TIMEOUT_SECONDS` if this build is meant "
                f"to take that long.\n"
            )

    # Full refs, so the image can be pulled or deployed without hunting through
    # the log for it.
    built = [r for r in results if r.success and r.image_tag]
    if built:
        body += "\n### Images\n\n"
        for r in built:
            name = r.label()
            body += f"**{name}**\n```\n{r.image_tag}\n```\n"
        body += _deploy_help(built)

    missing_ref = [r for r in results if r.success and not r.image_tag]
    if missing_ref:
        names = ", ".join(r.label() for r in missing_ref)
        body += (
            f"\n> Built successfully, but the image reference could not be read "
            f"from the build log for: {names}. Check the full log on the "
            f"dashboard.\n"
        )

    # Logs last, and sharing whatever the rest of the comment left over: the
    # author reads the log to fix the build, so it gets the space, but the table
    # and image refs above it must survive intact. Splitting the remainder
    # equally keeps the total inside the limit by construction, rather than
    # relying on the client's truncation backstop.
    failed = [r for r in results if not r.success and r.log_tail]
    if failed:
        share = (COMMENT_BUDGET - len(body)) // len(failed)
        if share >= MIN_USEFUL_LOG_CHARS:
            for r in failed:
                body += _log_details(r.label(), r.log_tail, share)
        else:
            names = ", ".join(f"`{r.label()}`" for r in failed)
            body += (
                f"\n> {len(failed)} builds failed — too many to include their "
                f"logs here without cutting each to a few unreadable lines. "
                f"Open the dashboard for the full log of each: {names}.\n"
            )

    if not all_ok:
        body += (
            "\nPush a fix and comment `/request_bot_review` again to retrigger.\n"
        )

    return body


def _fmt_took(seconds: float | None) -> str:
    """A build's wall clock, for the result table. "—" while still building.

    Minutes are the unit that matters here: a build is either a few minutes or
    the better part of an hour, and seconds-only would read as a raw number for
    every real build.
    """
    if seconds is None:
        return "—"
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def _log_details(name: str, log_tail: str, budget: int) -> str:
    """One collapsed build log, trimmed only if it cannot fit.

    The whole tail goes in when there is room. Sending the author to the
    dashboard for the actual error is friction at the moment they are most
    blocked, and a truncated log is worse than none: it reads as though the
    build stopped where the text stops.

    When it does not fit, the *end* is kept — that is where the failure is — and
    the omission is stated, with a pointer to the full log.
    """
    lines = log_tail.splitlines()
    # `budget` also has to cover this wrapper and the summary line.
    room = budget - 400
    dropped = 0

    if len(log_tail) > room:
        kept: list[str] = []
        total = 0
        for line in reversed(lines):
            total += len(line) + 1
            if total > room and kept:
                break
            kept.append(line)
        kept.reverse()
        dropped = len(lines) - len(kept)
        lines = kept

    text = "\n".join(lines)
    # A single line can be longer than the whole budget — a build step echoing a
    # one-line JSON blob, or a progress bar written without newlines. Trimming by
    # line cannot help there, so cut characters and keep the end.
    cut_mid_line = len(text) > room
    if cut_mid_line:
        text = text[-room:]

    where = "full log on the dashboard"
    omitted = f"{dropped} earlier line{'' if dropped == 1 else 's'} omitted"
    if dropped and cut_mid_line:
        note = f"last {len(lines)} lines, cut mid-line — {omitted}; {where}"
    elif dropped:
        note = (
            f"last {len(lines)} lines — {omitted} to stay under GitHub's "
            f"comment limit; {where}"
        )
    elif cut_mid_line:
        note = f"tail only — this log has no line breaks to cut on; {where}"
    else:
        note = f"complete log, {len(lines)} line{'' if len(lines) == 1 else 's'}"

    # A four-backtick fence, because build output can itself contain ``` — an
    # npm or docker step echoing a README would otherwise close the block early
    # and spill the rest of the log into the comment as markup.
    return f"""
<details><summary>{name} build log ({note})</summary>

````
{text}
````

</details>
"""


def _deploy_help(built: list[BuildResult]) -> str:
    """How to run a freshly built image.

    Leads with the repo's own `deploy/run-pr-image.sh`, which pulls the image,
    reads the compose service fragment out of `/deploy/service.yml`, and starts
    it from a standalone compose file. That keeps the same tool production uses
    (compose, at /opt/phanthy-motus/docker-compose.yml) instead of a
    hand-written `docker run`, and means the flags each service declares —
    privileged, host networking, device mounts — are used exactly as its author
    wrote them.
    """
    out = "\n### Try it\n"

    runnable = [r for r in built if r.container_name]
    for r in runnable:
        name = r.label()
        out += f"""
**{name}**

```bash
./deploy/run-pr-image.sh {r.image_tag}
```

Then `--logs`, `--shell`, `--down`. Starts container `{r.container_name}`, and
`--down` removes it and the generated compose file completely.
"""

    # Targets with no service fragment — in practice just core, which is the
    # agent itself and updates in place rather than running as a second copy.
    web_only = [r for r in built if not r.container_name]
    if web_only:
        names = ", ".join(r.label() for r in web_only)
        out += f"""
**{names}** — deployed by updating through the web console, not by running a
container: Agent Core pulls the image and hands over to a restart helper. Open
the dashboard's deploy panel and upgrade to this version.
"""

    if runnable:
        out += """
That script is for a throwaway test and leaves the host's real deployment
untouched. For a lasting one, deploy from the web console — it merges the same
service fragment into the host's compose file, so the container survives a
reboot.
"""

    out += """
Once this PR is approved, the version becomes installable from the web console.
"""
    return out


def format_no_build_needed(head_sha: str) -> str:
    """No buildable changes detected — proceeding straight to review."""
    return f"""{BOT_MARKER}
## PR Review Agent

| | |
|---|---|
| Commit | `{head_sha[:7]}` |
| Build targets | None (changes do not touch a buildable component) |
| Status | Generating review... |
"""


def format_review(findings: list[Finding], review_text: str, job=None) -> str:
    """The substantive code review, posted as its own comment.

    `job` carries the deterministic results (large files, infrastructure) and how
    the review loop ended. Optional so a caller with only text still works.
    """
    body = f"""{BOT_MARKER}
## PR Review Agent — Code Review

{review_text}
"""

    if job is not None:
        body += _format_large_files(job)
        body += _format_infra(job)

    if findings:
        body += "\n### Rule Checks\n\n"
        for f in findings:
            icon = {
                "error": ":x:",
                "warning": ":warning:",
                "info": ":information_source:",
            }.get(f.severity, ":grey_question:")
            body += f"- {icon} `{f.file}` — {f.message}\n"

    if job is not None:
        body += _format_review_budget(job)

    body += "\n---\n<sub>Generated automatically by PR Review Agent.</sub>\n"
    return body


def _format_large_files(job) -> str:
    """Added/modified files over the size limit, listed separately.

    Its own section because the ask is that these are impossible to miss: large
    assets belong in COS and fetched at build time, not committed.
    """
    entries = getattr(job, "large_files", None) or []
    if not entries:
        return ""
    out = "\n### Large files\n\n"
    for e in entries:
        out += f"- :x: `{e['file']}` — **{e['bytes'] / 1024:.0f}KB**\n"
    out += (
        "\nFiles this size should live in COS "
        "(`agi-phanthy-dev-1252788780.cos.ap-beijing.myqcloud.com/public/`) and "
        "be fetched at build time — see how `unitree/g1` pulls cyclonedds via a "
        "Dockerfile `ARG`. Note `.gitignore` covers only images, so nothing "
        "stops a large file being committed except this check.\n"
    )
    return out


def _format_infra(job) -> str:
    """Infrastructure files touched, shared bases called out first."""
    infra = getattr(job, "infra_files", None) or []
    if not infra:
        return ""
    shared = set(getattr(job, "shared_base_files", None) or [])
    out = "\n### Infrastructure changes\n\n"
    for f in infra:
        if f in shared:
            out += (
                f"- :rotating_light: `{f}` — **shared**: affects every component "
                "built on it, across both repositories\n"
            )
        else:
            out += f"- :warning: `{f}`\n"
    out += (
        "\nInfrastructure changes are held to a minimal-change standard: only "
        "modify when necessary, and do not grow the image.\n"
    )
    return out


def _format_review_budget(job) -> str:
    """Say when the review stopped early — silence would read as 'all clear'."""
    reason = getattr(job, "review_stopped_reason", "") or ""
    rounds = getattr(job, "review_rounds", 0) or 0
    tools = getattr(job, "review_tool_calls", 0) or 0
    if not reason:
        return ""

    note = ""
    if reason == "max_rounds":
        note = (
            f"\n> :warning: **This review was cut short** after {rounds} rounds "
            "(round limit). It may be incomplete — retrigger for another pass.\n"
        )
    elif reason == "timeout":
        note = (
            f"\n> :warning: **This review was cut short** by the time limit "
            f"after {rounds} rounds. It may be incomplete.\n"
        )
    elif reason == "error":
        note = (
            f"\n> :warning: **This review ended on an error** after {rounds} "
            "rounds, so it may be incomplete.\n"
        )
    return note + (
        f"\n<sub>Explored the checkout over {rounds} rounds, "
        f"{tools} tool calls.</sub>\n"
    )


def format_no_changes(head_sha: str) -> str:
    return f"""{BOT_MARKER}
## PR Review Agent

Commit `{head_sha[:7]}` has no file changes relative to main — nothing to
build or review.
"""


def format_interrupted(head_sha: str, was_running: bool) -> str:
    """The agent shut down before this job finished.

    Restarts are operational events the author cannot infer from a comment
    frozen at "Building...", so say plainly what happened and what to do.
    """
    what = "was interrupted mid-run" if was_running else "never started"
    return f"""{BOT_MARKER}
## PR Review Agent — Interrupted

Review of `{head_sha[:7]}` {what} because the agent was stopped or restarted.

Comment `/request_bot_review` again to retrigger.
"""


def format_superseded(old_sha: str, new_sha: str) -> str:
    """A queued job dropped because a newer request arrived for the same PR."""
    return f"""{BOT_MARKER}
## PR Review Agent — Superseded

Queued review of `{old_sha[:7]}` was dropped because a newer request arrived
for `{new_sha[:7]}`. See the newer comment for status.
"""


def format_skipped_in_flight(head_sha: str) -> str:
    """A repeat trigger arrived for a commit that is already being reviewed."""
    return f"""{BOT_MARKER}
## PR Review Agent — Already in progress

A review of `{head_sha[:7]}` is already running. This request was skipped rather
than starting a second build of the same commit.

Push a new commit to review the change, or use
`/request_bot_review force` to re-run this one.
"""


def format_skipped_already_reviewed(
    head_sha: str, status: str, finished_at: float | None
) -> str:
    """A repeat trigger arrived for a commit that was already reviewed."""
    when = ""
    if finished_at:
        from datetime import datetime, timezone
        ts = datetime.fromtimestamp(finished_at, tz=timezone.utc)
        when = f" on {ts.strftime('%Y-%m-%d %H:%M')} UTC"

    return f"""{BOT_MARKER}
## PR Review Agent — Already reviewed

`{head_sha[:7]}` was already reviewed{when} (result: `{status}`). This request
was skipped rather than rebuilding an unchanged commit.

- Pushed a fix? The new commit will be reviewed when you trigger again.
- Want this commit re-reviewed anyway? Use `/request_bot_review force`.
"""


def format_retrying(
    head_sha: str,
    attempt: int,
    max_attempts: int,
    reason: str,
    backoff_seconds: int,
) -> str:
    """Shown between attempts when a job failed for a retryable reason."""
    return f"""{BOT_MARKER}
## PR Review Agent — Retrying

| | |
|---|---|
| Commit | `{head_sha[:7]}` |
| Attempt | {attempt} of {max_attempts} failed |
| Status | Retrying in {backoff_seconds}s... |

<details><summary>Failure reason</summary>

```
{reason}
```

</details>
"""


def format_final_failure(
    head_sha: str,
    max_attempts: int,
    attempt_errors: list[str],
) -> str:
    """All attempts exhausted — a real failure the author needs to see."""
    body = f"""{BOT_MARKER}
## PR Review Agent — Failed

Commit `{head_sha[:7]}` did not complete after {max_attempts} attempts.

Likely causes: build environment problem, network failure, unreachable
registry, or a build that exceeds the per-job time limit.
"""

    for i, err in enumerate(attempt_errors, start=1):
        body += f"""
<details><summary>Attempt {i} failure reason</summary>

```
{err}
```

</details>
"""

    body += "\nOnce resolved, comment `/request_bot_review` again to retrigger.\n"
    return body


def format_error(head_sha: str, error: str) -> str:
    """A terminal, non-retryable error (e.g. merge conflict)."""
    return f"""{BOT_MARKER}
## PR Review Agent — Error

Commit: `{head_sha[:7]}`

```
{error}
```

Push a fix and comment `/request_bot_review` again to retrigger.
"""
