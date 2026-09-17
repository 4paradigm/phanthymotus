"""Run the reviewed repo's own test suites inside the images built from the PR.

Why inside the image rather than in this container, which already has Python:
`perception/tests` guards a large part of itself behind `skipif` on optional
dependencies —

    SKIPPED [11] pythainlp is not installed on this host
    SKIPPED [4]  khanaa is not installed on this host
    SKIPPED [1]  VITS2 frontend dependencies missing: jieba, pypinyin, wetext,
                 kaldifst, g2p_en, inflect, nltk

— which are installed by `requirements.thai.txt`, `requirements.ja.txt` and the
vits2 requirements *in the perception image*. Mirroring them into this agent's
own requirements.txt would mean re-deriving the image by hand, and getting it
wrong would not turn the suite red: it would quietly run ~10 fewer tests and
still report green. A green built from a silently narrower run is worse than no
run at all, so the image is the only honest place for this.

The same choice fixes two lesser things for free: the suites run against the
Python the component actually ships (3.10 for agent-core, 3.8 on jp5.11) rather
than this container's 3.12, and the dependency set stays correct with no
maintenance here.

It costs a `docker run` per suite, and it means a PR that built no image for a
component gets no test result for it — reported as `skipped`, never faked.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import re
import shlex
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from .builder import _read_tail, run_logged
from .config import Config
from .models import BuildResult, BuildTarget, TestResult

logger = logging.getLogger(__name__)

# Test logs share the job's log directory with build logs, so the dashboard's
# `/api/jobs/{id}/log/{idx}` tailing works unchanged. 100 keeps them clear of
# the build indices (0..N, N being at most the driver fan-out) and the glob in
# store.read_log is prefix-exact, so `1-*.log` never matches `100-*.log`.
TEST_IDX_BASE = 100

# Only the tail is quoted into the PR comment. 400 rather than the build's 4000
# because `-q -rf` output is dense and its short summary is always within the
# last few dozen lines — a longer tail would only crowd out a build failure's
# log when both are competing for the same comment budget.
TEST_LOG_TAIL_LINES = 400

# Tried when the configured index fails. Measured reachable from a perception
# container on Orin 6, where the repo's usual Tencent mirror is not.
PYPI_FALLBACK = "https://pypi.org/simple/"

# Suite definitions. `workdir` is relative to the worktree root and is
# load-bearing: `python -m pytest` puts the cwd on sys.path[0], and both suites
# locate their code relative to their own __file__ from there. Running from
# /work instead would resolve imports against the code baked into the image and
# report a green that describes the previous release, not this PR.
SUITES: dict[str, dict] = {
    "agent-core": {
        "workdir": "agent-core",
        "target": BuildTarget.CORE,
        # /work/.venv is where the image's uv sync put the app's dependencies.
        "python": "/work/.venv/bin/python",
        "env": {},
    },
    "perception": {
        "workdir": "perception",
        "target": BuildTarget.PERCEPTION,
        "python": "python3",
        "env": {
            # A globally-installed fugue_test plugin fails to import
            # (`No module named 'pkg_resources'`) and aborts collection before
            # any test runs, which makes a healthy suite look broken.
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            # Reproduced from the image's own CMD. Without it aarch64 hits
            # "cannot allocate memory in static TLS block" at import.
            "LD_PRELOAD": "/usr/lib/aarch64-linux-gnu/libgomp.so.1",
        },
    },
    "actucore": {
        "workdir": "actucore",
        "target": BuildTarget.ACTUCORE,
        "python": "python3",
        # Same image family as perception — same two reasons for both vars.
        "env": {
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "LD_PRELOAD": "/usr/lib/aarch64-linux-gnu/libgomp.so.1",
        },
    },
}

# pytest's exit codes, which say things the junit XML cannot.
_PYTEST_OK = 0
_PYTEST_TESTS_FAILED = 1
_PYTEST_NO_TESTS_COLLECTED = 5


class HostPathError(RuntimeError):
    """The worktree cannot be translated to a path the docker daemon can see."""


def host_path(path: Path | str, config: Config) -> str:
    """Translate a path under `data_dir` to the host path the daemon resolves.

    `docker build` never needed this: the client tars the context and ships it
    over the socket, so the client's view of a path is what counts. `docker run
    -v` is the opposite — the daemon resolves the source, and a container path
    handed to it names nothing on the host, so Docker helpfully creates an empty
    directory and mounts that. Every test then fails on "file not found" and
    lands on the PR as the author's doing.

    Raises rather than guessing when the mapping is unknown but demonstrably
    needed. A wrong mount is not a degraded run, it is a fabricated one.
    """
    text = str(path)
    if not config.data_host_dir:
        if Path("/.dockerenv").exists():
            raise HostPathError(
                "DATA_HOST_DIR is not set, but this agent is running in a "
                "container: the docker daemon cannot resolve its paths, and "
                "mounting one would silently produce an empty directory. Set "
                "DATA_HOST_DIR to the host path bind-mounted at DATA_DIR."
            )
        # Not containerised: the agent and the daemon share a filesystem.
        return text
    data_dir = str(config.data_dir).rstrip("/")
    if text == data_dir:
        return config.data_host_dir.rstrip("/")
    if text.startswith(data_dir + "/"):
        return config.data_host_dir.rstrip("/") + text[len(data_dir):]
    raise HostPathError(
        f"{text!r} is not under DATA_DIR ({data_dir!r}); refusing to guess "
        f"its host path"
    )


def test_log_filename(idx: int, component: str) -> str:
    """`{idx}-{safe-component}-tests.log`, matching builder.log_filename."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", component)
    return f"{idx}-{safe}-tests.log"


def plan_suites(
    build_results: list[BuildResult], config: Config
) -> list[tuple[str, str, str]]:
    """Decide which suites to run: `(component, image_ref, skip_reason)`.

    A component is testable only when this job built an image for it and read
    a ref back out of the build log. Everything else is `skipped` with a
    reason — never a fallback to `:latest` or to a previous tag, because a
    green from an image that does not contain this PR would be reported on the
    PR, and fed to the reviewer, with nothing saying which commit it described.
    """
    by_target: dict[BuildTarget, BuildResult] = {}
    for result in build_results:
        if result.success is not True or not result.image_tag:
            continue
        # First variant wins: a job can build perception for both jp5.11 and
        # jp6.1, and running the same Python tests twice only doubles the bill.
        by_target.setdefault(result.target, result)

    plan: list[tuple[str, str, str]] = []
    for component in config.test_components:
        suite = SUITES.get(component)
        if suite is None:
            logger.warning(f"Unknown test component {component!r}, skipping")
            continue
        built = by_target.get(suite["target"])
        if built is None:
            plan.append((
                component, "",
                f"this PR built no {component} image, so there is nothing "
                f"containing its code to run the suite in",
            ))
        else:
            plan.append((component, built.image_tag, ""))
    return plan


def _docker_cmd(
    component: str, image_ref: str, job_id: str, worktree: Path,
    junit_dir: Path, config: Config,
) -> list[str]:
    suite = SUITES[component]
    inner = _inner_script(component, config)

    cmd = [
        "docker", "run", "--rm",
        # Named so a container leaked by a cancellation is identifiable and
        # sweepable, rather than an anonymous hash holding 8 GB.
        "--name", f"pr-test-{job_id}-{component}",
        # The worktree is writable on purpose: neither suite was written
        # against a read-only tree, and a spurious PermissionError would be
        # reported to the author as their bug. It is destroyed at cleanup.
        "-v", f"{host_path(worktree, config)}:/src",
        # junit and the pytest cache go here, NOT into the worktree: the LLM
        # review walks that same tree afterwards through tools.Sandbox, and a
        # stray results.xml or .pytest_cache would show up in list_dir as part
        # of the PR.
        "-v", f"{host_path(junit_dir, config)}:/out",
        "-w", f"/src/{suite['workdir']}",
        "-e", "HOME=/tmp",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "PIP_DISABLE_PIP_VERSION_CHECK=1",
        # A runaway test must not take the build host's memory with it.
        "--memory", config.test_memory_limit,
    ]
    for key, value in suite["env"].items():
        cmd += ["-e", f"{key}={value}"]
    # The images are arm64. On an x86 build host this is qemu — slow, but the
    # binfmt handler is already registered by the buildx cross-builds.
    if platform.machine() not in ("aarch64", "arm64"):
        cmd += ["--platform", "linux/arm64"]
    cmd += [image_ref, "/bin/bash", "-lc", inner]
    return cmd


def _inner_script(component: str, config: Config) -> str:
    """The shell run inside the container: install pytest, then run the suite.

    pytest is installed at run time because neither image ships it, and adding
    it to both would grow every production image on every robot for a CI-only
    dependency. It is one package from an index, versus the whole dependency
    closure this design exists to avoid maintaining.

    `-i` is mandatory, not a preference: both images bake
    `PIP_INDEX_URL=jetson.webredirect.org`, which resolves on none of the hosts
    this runs on. The fallback to pypi.org exists because this install is the
    single most likely thing to fail — on Orin 6 the repo's usual Tencent
    mirror does not resolve at all, while inside the VPC it is the fast one.
    Trying the configured index first keeps both true.

    Exit 97 is the sentinel for "the harness failed before pytest ran" — it is
    mapped to status="error", not "failed", so an index outage never lands on
    the PR as a test failure.
    """
    suite = SUITES[component]
    python = suite["python"]
    spec = config.test_pytest_spec  # unquoted: may legitimately be two specs
    install = (
        f"{python} -m pip install --no-cache-dir "
        f"-i {shlex.quote(config.test_pypi_index)} {spec}"
    )
    if config.test_pypi_index.rstrip("/") != PYPI_FALLBACK.rstrip("/"):
        install += (
            f" || {python} -m pip install --no-cache-dir "
            f"-i {shlex.quote(PYPI_FALLBACK)} {spec}"
        )
    return (
        "set -o pipefail\n"
        f"{install} || exit 97\n"
        f"{python} -m pytest tests "
        "-p no:cacheprovider -o cache_dir=/out/.pytest_cache "
        f"--junitxml=/out/{component}.xml -q -rf --color=no\n"
    )


def _parse_junit(path: Path, max_failures: int) -> dict:
    """Counts, failing test ids and failure text from pytest's junit XML.

    junit rather than a regex over the `-q` summary line, because the summary
    is not a stable contract across pytest majors and — more importantly — the
    cases most worth reporting accurately produce no summary line at all: a
    collection error, a segfault, an OOM kill, a suite killed on the idle
    timeout. The XML is written per test as it runs, so a killed run still
    describes whatever completed, and collection errors appear as testcases
    with a nested <error> rather than vanishing.

    Any parse problem degrades to empty counts; it must never raise into the
    pipeline, which is non-blocking by design.
    """
    empty = {
        "passed": 0, "failed": 0, "skipped": 0, "errors": 0, "total": 0,
        "failing_ids": [], "failure_text": "", "parsed": False,
    }
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as error:
        logger.warning(f"Could not parse junit XML at {path}: {error}")
        return empty

    total = failures = errors = skipped = 0
    # pytest emits <testsuites><testsuite>; older tooling emits a bare
    # <testsuite>. `iter` covers both without branching on the root tag.
    for suite in root.iter("testsuite"):
        total += _int_attr(suite, "tests")
        failures += _int_attr(suite, "failures")
        errors += _int_attr(suite, "errors")
        skipped += _int_attr(suite, "skipped")

    failing_ids: list[str] = []
    fragments: list[str] = []
    for case in root.iter("testcase"):
        bad = case.find("failure")
        if bad is None:
            bad = case.find("error")
        if bad is None:
            continue
        classname = (case.get("classname") or "").replace(".", "/")
        name = case.get("name") or "?"
        failing_ids.append(f"{classname}::{name}" if classname else name)
        if len(fragments) < max_failures:
            message = (bad.get("message") or "").strip()
            body = (bad.text or "").strip()
            fragments.append(
                f"{failing_ids[-1]}\n{message[:400]}\n{body[-1200:]}".strip()
            )

    return {
        "passed": max(total - failures - errors - skipped, 0),
        "failed": failures,
        "skipped": skipped,
        "errors": errors,
        "total": total,
        "failing_ids": failing_ids,
        "failure_text": "\n\n".join(fragments),
        "parsed": True,
    }


def _int_attr(element, name: str) -> int:
    try:
        return int(element.get(name) or 0)
    except (TypeError, ValueError):
        return 0


async def run_suite(
    component: str,
    image_ref: str,
    job_id: str,
    worktree: Path,
    config: Config,
    log_path: Path,
    junit_dir: Path,
) -> TestResult:
    """Pull the image, run one suite in it, and classify the outcome."""
    junit_dir.mkdir(parents=True, exist_ok=True)
    junit_path = junit_dir / f"{component}.xml"
    # A stale XML from a previous attempt would be parsed as this run's result
    # if the container died before writing one.
    junit_path.unlink(missing_ok=True)

    result = TestResult(component=component, status=None, image_tag=image_ref,
                        log_path=str(log_path))
    started = time.monotonic()

    try:
        cmd = _docker_cmd(component, image_ref, job_id, worktree, junit_dir, config)
    except HostPathError as error:
        logger.error(f"Refusing to run {component} tests: {error}")
        result.status = "error"
        result.skip_reason = str(error)
        result.duration_seconds = 0.0
        return result

    # Pulled explicitly, as its own logged step: on the x86 build host the
    # images are `buildx --push`-ed and are not in the local store, so without
    # this the first minutes of a run are a silent multi-GB download that looks
    # like a hang before pytest.
    pull_ok, pull_kind, _ = await run_logged(
        ["docker", "pull", image_ref],
        cwd=str(worktree), env={"PATH": "/usr/local/bin:/usr/bin:/bin"},
        timeout=config.test_timeout_seconds,
        idle_timeout=config.test_idle_timeout_seconds,
        log_path=log_path, label=f"{component} image", what="Image pull",
    )
    if not pull_ok:
        result.status = "error"
        result.timeout_kind = pull_kind
        result.log_tail = _read_tail(log_path, TEST_LOG_TAIL_LINES)
        result.skip_reason = f"could not pull {image_ref}"
        result.duration_seconds = time.monotonic() - started
        return result

    success, timeout_kind, returncode = await run_logged(
        cmd,
        cwd=str(worktree),
        env={"PATH": "/usr/local/bin:/usr/bin:/bin"},
        timeout=config.test_timeout_seconds,
        idle_timeout=config.test_idle_timeout_seconds,
        log_path=log_path,
        label=f"{component} tests",
        what="Test run",
        append=True,
    )

    parsed = _parse_junit(junit_path, config.test_max_reported_failures)
    result.passed = parsed["passed"]
    result.failed = parsed["failed"]
    result.skipped = parsed["skipped"]
    result.errors = parsed["errors"]
    result.total = parsed["total"]
    result.failing_ids = parsed["failing_ids"]
    result.failure_text = parsed["failure_text"]
    result.timeout_kind = timeout_kind
    result.log_tail = _read_tail(log_path, TEST_LOG_TAIL_LINES)
    result.duration_seconds = time.monotonic() - started
    result.status = _classify(success, returncode, timeout_kind, parsed)
    if result.status == "error" and not result.skip_reason:
        result.skip_reason = _error_reason(returncode, timeout_kind)
    logger.info(
        f"{component} tests: {result.status} "
        f"({result.passed} passed, {result.failed} failed, "
        f"{result.errors} errors, {result.skipped} skipped)"
    )
    return result


def _classify(success: bool, returncode: int | None, timeout_kind: str,
              parsed: dict) -> str:
    """Map a run to one of the five statuses.

    The distinction that matters: `failed` means the PR broke tests, `error`
    means we never got a verdict. Reporting the second as the first puts an
    agent-side outage on somebody's pull request.
    """
    if success:
        return "passed"
    if timeout_kind:
        # Killed for going quiet or hitting the cap. Not the PR's verdict.
        return "error"
    if returncode == 97:
        return "error"          # pytest was never installed
    if returncode == _PYTEST_NO_TESTS_COLLECTED:
        return "error"          # a suite that collects nothing is broken wiring
    if returncode == _PYTEST_TESTS_FAILED and parsed["parsed"]:
        return "failed"
    if parsed["parsed"] and (parsed["failed"] or parsed["errors"]):
        # Exit code was something else (3 internal, 4 usage) but the XML has
        # real failures in it — report what we can see.
        return "failed"
    return "error"


def _error_reason(returncode: int | None, timeout_kind: str) -> str:
    if timeout_kind == "idle":
        return "the suite went quiet and was killed; no verdict"
    if timeout_kind == "cap":
        return "the suite hit the absolute time cap; no verdict"
    if returncode == 97:
        return "pytest could not be installed in the image (mirror unreachable?)"
    if returncode == _PYTEST_NO_TESTS_COLLECTED:
        return "pytest collected no tests at all"
    return f"pytest did not produce a usable result (exit {returncode})"


async def sweep_leaked_containers() -> int:
    """Remove containers left behind by a cancelled run. Best effort.

    `run_logged` kills the process group, which kills the `docker run` *client*
    — `--rm` does not fire for the container it left behind. Called at startup
    next to the stale-worktree sweep.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "ps", "-aq", "--filter", "name=pr-test-",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
    except OSError as error:
        logger.warning(f"Could not list leaked test containers: {error}")
        return 0
    ids = [line for line in out.decode().split() if line]
    if not ids:
        return 0
    try:
        killer = await asyncio.create_subprocess_exec(
            "docker", "rm", "-f", *ids,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await killer.wait()
    except OSError as error:
        logger.warning(f"Could not remove leaked test containers: {error}")
        return 0
    logger.info(f"Removed {len(ids)} leaked test container(s)")
    return len(ids)


def summarize_for_review(results: list[TestResult]) -> str:
    """The deterministic half of what the reviewer is told about the tests.

    Counts and failing ids only — assertion output is PR-authored text and goes
    in the fenced, untrusted part of the context instead.

    A suite with status `error` is deliberately summarised as "could not be
    run" with no counts: that is an agent-side problem, and dressing it up as a
    test signal would spend the reviewer's attention on our outage.
    """
    if not results:
        return ""
    lines: list[str] = []
    for result in results:
        if result.status == "skipped":
            lines.append(f"- {result.component}: not run — {result.skip_reason}")
            continue
        if result.status == "error":
            lines.append(
                f"- {result.component}: could not be run — {result.skip_reason}. "
                f"This is an agent-side problem, not something for this review."
            )
            continue
        if result.status is None:
            continue
        parts = [f"{result.passed} passed"]
        if result.failed:
            parts.append(f"{result.failed} failed")
        if result.errors:
            parts.append(f"{result.errors} errors")
        if result.skipped:
            parts.append(f"{result.skipped} skipped")
        lines.append(
            f"- {result.component}: {', '.join(parts)} "
            f"({result.total} total) — ran in {result.image_tag}"
        )
        for test_id in result.failing_ids[:20]:
            lines.append(f"    - {test_id}")
    return "\n".join(lines)


def failure_context(results: list[TestResult], max_chars: int) -> str:
    """Assertion output for the fenced, untrusted half of the reviewer context."""
    blocks = [
        f"### {r.component}\n{r.failure_text}"
        for r in results
        if r.status == "failed" and r.failure_text
    ]
    if not blocks:
        return ""
    text = "\n\n".join(blocks)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n… (truncated)"
    return text
