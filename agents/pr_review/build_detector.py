"""Build target detection — analyze changed files to determine what to build."""

import logging
from pathlib import Path

from .models import BuildTarget

logger = logging.getLogger(__name__)

# Files that never justify a build on their own.
IGNORED_NAMES = {
    "README.md", "README_zh.md", "README_dev.md",
    "CONTRIBUTING.md", "LICENSE", "CODEOWNERS",
    ".env.example", ".gitignore", ".dockerignore",
}
IGNORED_DIR_PREFIXES = ("docs/", ".github/")

# A directory is a buildable driver if it has both of these — the same test
# `build.sh` applies when discovering what it can build.
DRIVER_MARKERS = ("driver.yaml", "Dockerfile")


def detect_targets(
    repo_full_name: str,
    changed_files: list[str],
    worktree: Path | None = None,
) -> tuple[list[BuildTarget], list[str]]:
    """Analyze changed files to determine build targets.

    Args:
        repo_full_name: e.g. "4paradigm/phanthymotus"
        changed_files: relative paths from `git diff --name-only`
        worktree: checkout to probe for driver markers. Required to detect
            drivers; without it driver detection returns nothing rather than
            guessing.

    Returns:
        (targets, driver_paths) where driver_paths is populated for DRIVER.
    """
    repo_name = repo_full_name.split("/")[-1]

    if repo_name == "phanthymotus":
        return _detect_motus_targets(changed_files)
    if repo_name == "phanthymotus-driver":
        return _detect_driver_targets(changed_files, worktree)

    logger.warning(f"Unknown repo, no build targets: {repo_full_name}")
    return [], []


def _detect_motus_targets(
    changed_files: list[str],
) -> tuple[list[BuildTarget], list[str]]:
    """Detect build targets for the phanthymotus repo."""
    targets = set()
    for f in changed_files:
        if _is_ignored(f):
            continue
        parts = Path(f).parts
        top = parts[0] if parts else ""
        if top == "agent-core":
            targets.add(BuildTarget.CORE)
        elif top == "perception":
            targets.add(BuildTarget.PERCEPTION)
        elif top == "actucore":
            targets.add(BuildTarget.ACTUCORE)
    # Deterministic order, so the build plan and its log indices are stable.
    order = (BuildTarget.CORE, BuildTarget.PERCEPTION, BuildTarget.ACTUCORE)
    return [t for t in order if t in targets], []


def _detect_driver_targets(
    changed_files: list[str], worktree: Path | None
) -> tuple[list[BuildTarget], list[str]]:
    """Detect which drivers changed, by probing the worktree.

    Discovery is filesystem-driven rather than a hardcoded provider list. A
    hardcoded list silently skips every newly added vendor: PR #166 adding
    `robotera/q5_bundle` produced no build target at all, so the PR would get a
    review and no image — the opposite of useful.
    """
    if worktree is None:
        logger.warning("No worktree provided — cannot detect driver targets")
        return [], []

    candidates: set[str] = set()
    for f in changed_files:
        if _is_ignored(f):
            continue
        parts = Path(f).parts
        # A driver change is always provider/model/<file>. Repo-root files
        # (build.sh, README) and single-level paths are not a driver.
        if len(parts) < 3:
            continue
        candidates.add(f"{parts[0]}/{parts[1]}")

    driver_paths = []
    for candidate in sorted(candidates):
        d = worktree / candidate
        missing = [m for m in DRIVER_MARKERS if not (d / m).is_file()]
        if missing:
            logger.info(
                f"Skipping {candidate}: not a buildable driver "
                f"(missing {', '.join(missing)})"
            )
        else:
            driver_paths.append(candidate)

    if driver_paths:
        return [BuildTarget.DRIVER], driver_paths
    return [], []


def _is_ignored(filepath: str) -> bool:
    """Whether a changed file should be disregarded for build detection."""
    if Path(filepath).name in IGNORED_NAMES:
        return True
    return filepath.startswith(IGNORED_DIR_PREFIXES)
