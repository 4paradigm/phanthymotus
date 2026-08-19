"""Builder — shells out to the repos' existing build scripts.

Build output is streamed to a file on disk as it is produced, rather than
buffered in memory until the process exits. That is what makes the dashboard's
live log tailing possible, and it means a build that hangs or is killed still
leaves behind everything it printed up to that point — which is exactly what is
needed to diagnose it.
"""

import asyncio
import logging
import os
import re
import signal
from pathlib import Path

from .config import Config
from .models import DEFAULT_JP_VERSION, BuildResult, BuildTarget, build_label

logger = logging.getLogger(__name__)

# Image reference markers, per build script. All three scripts print the ref in
# their own wording, so each has to be matched explicitly:
#
#   build_core.sh:36         Image : <ref>
#   build_perception.sh:87   Image  : <ref>            (two spaces)
#   both, on success         Done. Image pushed: <ref>
#                            Done. Image built locally: <ref>
#   driver build.sh:257      构建 <name>  →  <ref>
#   driver build.sh:302      完成：<ref>
#
# Completion markers come first: they mean the build actually finished, whereas
# the declared-target lines are printed before it runs.
COMPLETION_PATTERNS = (
    re.compile(r"完成\s*[:：]\s*(\S+)"),
    re.compile(r"Done\.\s*Image pushed:\s*(\S+)"),
    re.compile(r"Done\.\s*Image built locally:\s*(\S+)"),
)
DECLARED_PATTERNS = (
    re.compile(r"构建\s+.*?[→>]\s*(\S+)"),
    re.compile(r"Image\s*:\s*(\S+)"),
)
# Last-resort match on the tag shape itself: release.YYMMDD.<7-hex>. Distinctive
# enough to be a safe backstop, so a future wording change in any build script
# degrades to this instead of silently losing the ref again — which is exactly
# how the driver script's output went unnoticed.
REF_SHAPE_PATTERN = re.compile(
    r"(?<![\w./:-])"
    r"([\w.\-]+(?::\d+)?(?:/[\w.\-]+)+:release\.\d{6}\.[0-9a-f]{7,40})"
)

# How much of a failed build's log to carry to the PR comment. Generous on
# purpose: a build failure is diagnosed from the log, and 80 lines routinely cut
# off above the actual error — a failing `pip install` or `apt-get` prints
# hundreds of lines after it. `comments.format_build_result` trims to whatever
# GitHub's comment limit leaves, so these bounds only need to be larger than
# that limit can hold; reading further would always be discarded.
LOG_TAIL_LINES = 4000
READ_CHUNK = 8192
# Bytes read from each end of a large log when scanning for the image tag.
# Covers "Image :" at the head and "Done. ..." at the tail without loading a
# multi-megabyte docker build log into memory.
SCAN_WINDOW = 256 * 1024
# Bytes read from the end when building the tail for the PR comment.
TAIL_WINDOW = 512 * 1024


def _build_env(config: Config) -> dict[str, str]:
    """Construct environment variables for the build scripts."""
    env = os.environ.copy()
    env["REGISTRY"] = config.registry
    env["REGISTRY_USER"] = config.registry_user
    env["REGISTRY_PASSWORD"] = config.registry_password
    env["IMAGE_NAMESPACE"] = config.image_namespace
    env["MIRROR"] = config.mirror
    env["IMAGE_NAMESPACE_DRIVERS"] = config.image_namespace_drivers
    env["DEBIAN_FRONTEND"] = "noninteractive"
    if config.resource_center_url:
        env["RESOURCE_CENTER_URL"] = config.resource_center_url
    if config.resource_center_api_key:
        env["RESOURCE_CENTER_API_KEY"] = config.resource_center_api_key
    return env


# ── Public build entry points ─────────────────────────────────────────────────


async def build_core(worktree: Path, config: Config, log_path: Path) -> BuildResult:
    """Build the agent-core image via deploy/build_core.sh."""
    return await _build_with_script(
        target=BuildTarget.CORE,
        driver_path=None,
        script=worktree / "deploy" / "build_core.sh",
        args=["--mirror", config.mirror],
        cwd=worktree,
        config=config,
        log_path=log_path,
    )


async def build_perception(
    worktree: Path,
    config: Config,
    log_path: Path,
    jp_version: str = DEFAULT_JP_VERSION,
) -> BuildResult:
    """Build the perception image via deploy/build_perception.sh.

    Jetson-only by construction — perception runs on Jetson hardware, so the
    script takes no `--variant`. `--jp-version` picks the base image and shows
    up in the tag as `-jetson-jp<ver>`.
    """
    return await _build_with_script(
        target=BuildTarget.PERCEPTION,
        driver_path=None,
        script=worktree / "deploy" / "build_perception.sh",
        args=[
            "--mirror", config.mirror,
            "--jp-version", jp_version,
        ],
        cwd=worktree,
        config=config,
        log_path=log_path,
        variant=jp_version,
    )


async def build_actucore(
    worktree: Path,
    config: Config,
    log_path: Path,
    jp_version: str = DEFAULT_JP_VERSION,
) -> BuildResult:
    """Build the actucore image via deploy/build_actucore.sh.

    Jetson-only by construction — execution models need the GPU, so the script
    takes no `--variant`. `--jp-version` picks the base image and shows up in
    the tag as `-jetson-jp<ver>`.
    """
    return await _build_with_script(
        target=BuildTarget.ACTUCORE,
        driver_path=None,
        script=worktree / "deploy" / "build_actucore.sh",
        args=[
            "--mirror", config.mirror,
            "--jp-version", jp_version,
        ],
        cwd=worktree,
        config=config,
        log_path=log_path,
        variant=jp_version,
    )


async def build_driver(
    worktree: Path, driver_path: str, config: Config, log_path: Path
) -> BuildResult:
    """Build one driver image via build.sh in CI mode."""
    env_overrides = {"IMAGE_NAMESPACE": config.image_namespace_drivers}
    return await _build_with_script(
        target=BuildTarget.DRIVER,
        driver_path=driver_path,
        script=worktree / "build.sh",
        args=["--mirror", config.mirror, driver_path],
        cwd=worktree,
        config=config,
        log_path=log_path,
        env_overrides=env_overrides,
    )


async def _build_with_script(
    target: BuildTarget,
    driver_path: str | None,
    script: Path,
    args: list[str],
    cwd: Path,
    config: Config,
    log_path: Path,
    env_overrides: dict[str, str] | None = None,
    variant: str = "",
) -> BuildResult:
    label = build_label(target, driver_path, variant)

    if not script.exists():
        message = f"{script.name} not found in worktree"
        _write_log(log_path, message)
        return BuildResult(
            target=target,
            driver_path=driver_path,
            success=False,
            image_tag="",
            log_tail=message,
            log_path=str(log_path),
            variant=variant,
        )

    env = _build_env(config)
    if env_overrides:
        env.update(env_overrides)

    success = await _run_build(
        ["bash", str(script), *args],
        cwd=str(cwd),
        env=env,
        timeout=config.build_timeout_seconds,
        log_path=log_path,
        label=label,
    )

    return BuildResult(
        target=target,
        driver_path=driver_path,
        success=success,
        image_tag=_extract_image_tag(log_path),
        log_tail=_read_tail(log_path),
        log_path=str(log_path),
        variant=variant,
    )


# ── Process execution ─────────────────────────────────────────────────────────


async def _run_build(
    cmd: list[str],
    cwd: str,
    env: dict[str, str],
    timeout: int,
    log_path: Path,
    label: str,
) -> bool:
    """Run a build, streaming its output to `log_path`. Returns success.

    The subprocess is killed on both timeout and cancellation. Cancellation
    matters because the whole-job timeout cancels this coroutine from the
    outside — without the explicit kill, `docker build` would keep running
    orphaned, holding the build cache and CPU for the retry to contend with.
    """
    logger.info(f"Building {label}: {' '.join(cmd[:3])}... (cwd={cwd})")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # start_new_session puts the child in its own process group so the whole
    # build tree (bash -> docker -> buildx) dies with it, not just bash.
    #
    # stdin is /dev/null because nothing here can answer a prompt: inheriting the
    # agent's stdin would let a `read` in a build script block until the build
    # timeout instead of failing immediately.
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        env=env,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )

    # buffering=0 so a reader tailing this file sees output as it is produced
    # rather than one block per flush.
    log_file = open(log_path, "wb", buffering=0)

    async def pump_and_wait() -> int:
        """Copy the child's output to disk until EOF, then reap it.

        Fixed-size chunks rather than readline(): StreamReader.readline()
        raises ValueError once a line exceeds the stream limit (64 KiB by
        default), and docker build output can contain very long single lines.

        The reap is inside the timed region deliberately — a process that
        closes stdout without exiting would otherwise hang past the timeout.
        """
        while True:
            chunk = await proc.stdout.read(READ_CHUNK)
            if not chunk:
                break
            log_file.write(chunk)
        return await proc.wait()

    task = asyncio.create_task(pump_and_wait())

    try:
        # Shielded so a timeout or outer cancellation does not kill the task
        # mid-write; it is drained explicitly below so the partial log survives.
        returncode = await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    except asyncio.TimeoutError:
        await _terminate(proc)
        await _drain(task)
        log_file.write(f"\n[agent] Build timed out after {timeout}s\n".encode())
        logger.error(f"Build timed out after {timeout}s: {label}")
        return False
    except asyncio.CancelledError:
        await _terminate(proc)
        await _drain(task)
        log_file.write(b"\n[agent] Build cancelled (agent stopping)\n")
        logger.warning(f"Build cancelled, subprocess killed: {label}")
        raise
    finally:
        # Closed after the task is drained, so the partial log is flushed even
        # when the build was killed.
        log_file.close()

    success = returncode == 0
    if success:
        logger.info(f"Build succeeded: {label}")
    else:
        logger.error(f"Build failed (rc={returncode}): {label}")
    return success


async def _drain(task: asyncio.Task):
    """Let the pump finish writing whatever the child already emitted.

    Shielded and bounded: we are often already being cancelled here, and an
    unshielded await would abort immediately and lose the tail of the log.
    """
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=5)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        task.cancel()
    except Exception:
        # The pump itself failed; the log is whatever made it to disk.
        pass


async def _terminate(proc: asyncio.subprocess.Process):
    """Kill a build subprocess and its process group, then reap it."""
    if proc.returncode is not None:
        return

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except ProcessLookupError:
            return

    try:
        await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=10)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass


# ── Log file helpers ──────────────────────────────────────────────────────────


def _write_log(log_path: Path, text: str):
    """Write a one-off message as the whole log (used for pre-flight errors)."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(text + "\n")
    except OSError as e:
        logger.warning(f"Failed to write log {log_path}: {e}")


def _read_tail(log_path: Path, lines: int = LOG_TAIL_LINES) -> str:
    """Last N lines of a log, for the PR comment."""
    try:
        size = log_path.stat().st_size
        with open(log_path, "rb") as f:
            if size > TAIL_WINDOW:
                f.seek(size - TAIL_WINDOW)
                # Drop the first (probably partial) line after seeking.
                f.readline()
            data = f.read()
    except OSError:
        return ""
    text = data.decode("utf-8", errors="replace")
    parts = text.splitlines()
    return "\n".join(parts[-lines:])


def _extract_image_tag(log_path: Path) -> str:
    """Pull the built image reference out of a build log.

    Scans the head and tail rather than the whole file: the ref is printed
    before the build starts and again on completion, never only in the middle.
    """
    try:
        size = log_path.stat().st_size
        with open(log_path, "rb") as f:
            if size <= 2 * SCAN_WINDOW:
                head = f.read()
                tail = b""
            else:
                head = f.read(SCAN_WINDOW)
                f.seek(size - SCAN_WINDOW)
                tail = f.read(SCAN_WINDOW)
    except OSError:
        return ""

    head_text = head.decode("utf-8", errors="replace")
    tail_text = tail.decode("utf-8", errors="replace")

    # Completion markers first — they mean the build finished. Tail before head,
    # since that is where a completion line lands.
    for pattern in COMPLETION_PATTERNS:
        for text in (tail_text, head_text):
            m = pattern.search(text)
            if m:
                return m.group(1)

    # Then the declared target, printed before the build ran.
    for pattern in DECLARED_PATTERNS:
        for text in (head_text, tail_text):
            m = pattern.search(text)
            if m:
                return m.group(1)

    # Backstop: anything shaped like a release ref. Last match wins, because
    # later mentions ("pushing <ref>") come from further along the build.
    matches = REF_SHAPE_PATTERN.findall(tail_text) or REF_SHAPE_PATTERN.findall(
        head_text
    )
    if matches:
        logger.info("Image ref recovered by shape match — build script wording may have changed")
        return matches[-1]

    return ""


def split_image_ref(ref: str) -> tuple[str, str]:
    """Split a full image reference into (ref, tag).

    Splitting on the last colon would break on a registry port
    (`host:5000/img`), so only a colon after the final `/` counts.
    """
    if not ref:
        return "", ""
    last_slash = ref.rfind("/")
    colon = ref.rfind(":")
    if colon > last_slash:
        return ref, ref[colon + 1:]
    return ref, ""


def container_name_from_service_yaml(yaml_text: str) -> str:
    """Read the container_name a service fragment declares, or "".

    Only the name is needed now: `deploy/run-pr-image.sh` does the rest by
    reading the same fragment out of the image, so translating the whole
    fragment into `docker run` here would duplicate what compose already does.
    The name is used to tell the reviewer which container to expect.
    """
    try:
        import yaml
        doc = yaml.safe_load(yaml_text)
    except Exception as e:
        logger.warning(f"Could not parse service.yml: {e}")
        return ""

    if not isinstance(doc, dict) or not doc:
        return ""
    key = next(iter(doc))
    svc = doc[key]
    if not isinstance(svc, dict):
        return ""
    return str(svc.get("container_name") or key)


def service_yaml_path(target: BuildTarget, driver_path: str | None) -> str | None:
    """Where a target's `deploy/service.yml` lives in the checkout, if it has one.

    Drivers, perception and actucore all ship one and are deployed the same way
    (Agent Core extracts it from the image and merges it into the host compose
    file).

    Core has none: it is the agent itself, updated in place through the web
    console via `POST /api/system/update`, which pulls the image and hands over
    to a restart-helper container. A `docker run` for core would be wrong.
    """
    if target == BuildTarget.DRIVER and driver_path:
        return f"{driver_path}/deploy/service.yml"
    if target == BuildTarget.PERCEPTION:
        return "perception/deploy/service.yml"
    if target == BuildTarget.ACTUCORE:
        return "actucore/deploy/service.yml"
    return None


def read_service_yaml(worktree: Path, rel_path: str) -> str:
    """Read a `deploy/service.yml` out of the checkout, if present."""
    try:
        return (worktree / rel_path).read_text()
    except OSError:
        return ""


def log_filename(
    idx: int, target: BuildTarget, driver_path: str | None, variant: str = ""
) -> str:
    """Log filename for one build within a job: `{idx}-{safe-label}.log`.

    The variant is part of the name so two perception builds in one job are
    told apart by more than their index.
    """
    label = driver_path or target.value
    if variant:
        label = f"{label}-jetson-jp{variant}"
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", label)
    return f"{idx}-{safe}.log"
