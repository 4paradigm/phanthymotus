"""Git workspace management — bare clones + worktrees for parallel PR processing.

Mirrors what `enter_pr_branch.sh` does by hand: fetch the PR head, create a
branch at the base, merge the PR into it. Two things it is careful about, both
learned the hard way:

- **Refs in a bare clone live at `refs/heads/*`, not `refs/remotes/origin/*`.**
  `origin/main` does not resolve, so the base is addressed by its plain branch
  name, taken from the PR's own `base.ref`.
- **Only the one PR's ref is fetched.** A wildcard `refs/pull/*/head` refspec
  makes every fetch pull every PR head ever opened — 166 of them on
  phanthymotus-driver — which stalls the first fetch for minutes.
"""

import asyncio
import logging
import os
from pathlib import Path

from .models import MergeConflictError

logger = logging.getLogger(__name__)

# Bound git network operations. Without these, only the whole-job timeout
# applies, so a hung fetch shows as "running, no output" for an hour.
FETCH_TIMEOUT = 300
GIT_LOCAL_TIMEOUT = 120

# The wildcard refspec an earlier version installed. Removed on startup so
# already-deployed clones stop paying for it without needing a re-clone.
LEGACY_PR_REFSPEC = "+refs/pull/*/head:refs/pull/*/head"

# Environment applied to every git invocation.
#
# The identity matters: merging a PR onto its base creates a merge commit, and
# git refuses to do that with no committer configured. A container has no
# ~/.gitconfig, so without this every merge fails with "Committer identity
# unknown" — which an earlier version reported to authors as a merge conflict.
# Set here rather than via `git config --global` so it travels with the process
# and cannot be lost by a rebuilt image.
_GIT_ENV = {
    "GIT_TERMINAL_PROMPT": "0",   # never block waiting for credentials
    "GIT_ASKPASS": "",
    # Non-interactive SSH. Without BatchMode, a missing or unreadable key makes
    # ssh prompt and the fetch hangs until the timeout instead of failing with a
    # usable message. Host keys are accepted on first use because the container
    # has no known_hosts of its own — the mounted key is what authenticates us,
    # and the alternative is every fresh container hanging on a host-key prompt.
    "GIT_SSH_COMMAND": os.environ.get(
        "GIT_SSH_COMMAND",
        "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "
        "-o ConnectTimeout=15",
    ),
    "GIT_AUTHOR_NAME": os.environ.get("GIT_AUTHOR_NAME", "PR Review Agent"),
    "GIT_AUTHOR_EMAIL": os.environ.get(
        "GIT_AUTHOR_EMAIL", "pr-review-agent@phanthy.local"
    ),
    "GIT_COMMITTER_NAME": os.environ.get("GIT_COMMITTER_NAME", "PR Review Agent"),
    "GIT_COMMITTER_EMAIL": os.environ.get(
        "GIT_COMMITTER_EMAIL", "pr-review-agent@phanthy.local"
    ),
}

# Network git operations are retried: reaching github.com from these hosts is
# intermittent (measured 13/15 TCP connects on a good minute), and a transient
# failure should not cost a whole job attempt.
NETWORK_RETRIES = 3
NETWORK_RETRY_BACKOFF = 5

# Phrases git uses when a merge fails specifically because of a conflict.
_CONFLICT_MARKERS = (
    "CONFLICT",
    "Automatic merge failed",
    "would be overwritten by merge",
    "Your local changes to the following files would be overwritten",
)


def _looks_like_conflict(merge_output: str) -> bool:
    """Whether a failed `git merge` failed because of an actual conflict."""
    return any(m in merge_output for m in _CONFLICT_MARKERS)


class GitWorkspaceManager:
    """Manages bare git clones and worktrees for parallel PR builds."""

    def __init__(self, data_dir: str, repos: dict[str, str]):
        """
        Args:
            data_dir: Base directory for all git data (e.g. /data/repos)
            repos: Mapping of repo full_name -> git clone URL
        """
        self._data_dir = Path(data_dir)
        self._repos = repos
        self._worktrees_dir = self._data_dir / "worktrees"
        self._fetch_locks: dict[str, asyncio.Lock] = {}

    def _bare_path(self, repo_full_name: str) -> Path:
        """Path to bare clone for a repo."""
        name = repo_full_name.split("/")[-1]
        return self._data_dir / f"{name}.git"

    def _get_fetch_lock(self, repo_full_name: str) -> asyncio.Lock:
        if repo_full_name not in self._fetch_locks:
            self._fetch_locks[repo_full_name] = asyncio.Lock()
        return self._fetch_locks[repo_full_name]

    # ── Setup ─────────────────────────────────────────────────────────────────

    async def ensure_clones(self):
        """Ensure bare clones exist, and are configured to fetch cheaply."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._worktrees_dir.mkdir(parents=True, exist_ok=True)

        for full_name, url in self._repos.items():
            bare_path = self._bare_path(full_name)
            if not bare_path.exists():
                logger.info(f"Cloning bare repo: {full_name} -> {bare_path}")
                await self._run_git_network(
                    ["git", "clone", "--bare", url, str(bare_path)],
                    cwd=str(self._data_dir),
                    label=f"clone {full_name}",
                )
            else:
                logger.info(f"Bare repo exists: {bare_path}")
                await self._sync_remote_url(bare_path, url)
            await self._drop_legacy_pr_refspec(bare_path)

    async def _sync_remote_url(self, bare_path: Path, url: str):
        """Point an existing clone's origin at the configured transport.

        The URL is only set at clone time, so a clone created under a different
        GIT_TRANSPORT keeps fetching over the old one. Switching transports
        would otherwise appear to do nothing — the setting would change while
        every fetch still used the URL baked in at clone time.
        """
        current = (
            await self._run_git(
                ["git", "remote", "get-url", "origin"],
                cwd=str(bare_path),
                check=False,
            )
        ).strip()
        if current == url:
            return
        await self._run_git(
            ["git", "remote", "set-url", "origin", url], cwd=str(bare_path)
        )
        logger.warning(
            f"Repointed {bare_path.name} origin: {current or '(unset)'} -> {url}"
        )

    async def _drop_legacy_pr_refspec(self, bare_path: Path):
        """Remove the wildcard PR refspec if a previous version installed it.

        Existing deployments already have it in their config, and it is the
        reason `git fetch` took minutes. Stripping it here means the fix applies
        on restart rather than requiring the volume to be wiped.
        """
        existing = await self._run_git(
            ["git", "config", "--get-all", "remote.origin.fetch"],
            cwd=str(bare_path),
            check=False,
        )
        if LEGACY_PR_REFSPEC not in existing:
            return
        await self._run_git(
            ["git", "config", "--unset-all", "remote.origin.fetch",
             r"^\+refs/pull/"],
            cwd=str(bare_path),
            check=False,
        )
        logger.warning(
            f"Removed wildcard PR refspec from {bare_path.name} — fetches "
            "no longer pull every PR head"
        )

    # ── Fetch ─────────────────────────────────────────────────────────────────

    async def fetch_for_pr(
        self, repo_full_name: str, pr_number: int, base_ref: str
    ) -> None:
        """Fetch just what this PR needs: its base branch and its own head ref.

        Two narrow fetches rather than one broad one. This is what
        `enter_pr_branch.sh:155,163` does, and the reason matters: a wildcard
        PR refspec turns every fetch into a download of all ~166 PR heads.
        """
        bare_path = self._bare_path(repo_full_name)
        pr_ref = f"refs/pull/{pr_number}/head"

        # Serialised per repo: concurrent jobs on the same repo would otherwise
        # contend on the same object store.
        async with self._get_fetch_lock(repo_full_name):
            logger.info(f"Fetching {repo_full_name} {base_ref} + PR #{pr_number}")
            await self._run_git_network(
                ["git", "fetch", "origin",
                 f"+refs/heads/{base_ref}:refs/heads/{base_ref}"],
                cwd=str(bare_path),
                label=f"fetch {base_ref}",
            )
            # Force-update: a force-pushed PR moves its head non-fast-forward.
            await self._run_git_network(
                ["git", "fetch", "origin", f"+{pr_ref}:{pr_ref}"],
                cwd=str(bare_path),
                label=f"fetch PR #{pr_number}",
            )

    # ── Worktrees ─────────────────────────────────────────────────────────────

    async def create_worktree(
        self,
        repo_full_name: str,
        pr_number: int,
        head_sha: str,
        base_ref: str,
    ) -> tuple[Path, str]:
        """Create an isolated worktree at `base_ref` with the PR merged in.

        `base_ref` is a plain branch name (e.g. "main"), not `origin/main`: this
        is a bare clone, where branches live at `refs/heads/*` and there are no
        remote-tracking refs for `origin/main` to resolve against.

        Returns (worktree_path, build_ref_sha). The second value is the worktree
        HEAD after the merge — the commit the build scripts turn into the image
        tag (`release.YYMMDD.<7hex>`), which is what makes a published image
        traceable back to a review. It is a *local* merge commit: this worktree
        is thrown away, so it exists nowhere else and cannot be recovered later.
        (When the PR has not diverged from base the merge fast-forwards and this
        equals `head_sha`.)

        Raises MergeConflictError when the PR cannot be merged.
        """
        bare_path = self._bare_path(repo_full_name)
        repo_short = repo_full_name.split("/")[-1]
        wt_name = f"{repo_short}-pr-{pr_number}-{head_sha[:7]}"
        wt_path = self._worktrees_dir / wt_name

        # Clean up a worktree left behind by a crashed or timed-out run.
        if wt_path.exists():
            await self._remove_worktree_force(bare_path, wt_path)

        # --detach so we never try to check out (or create) a branch that the
        # bare repo already has, and so concurrent jobs cannot collide on one.
        await self._run_git(
            ["git", "worktree", "add", "--detach", str(wt_path), base_ref],
            cwd=str(bare_path),
            timeout=GIT_LOCAL_TIMEOUT,
        )

        pr_ref = f"refs/pull/{pr_number}/head"
        rc, out = await self._run_git_status(
            ["git", "merge", pr_ref, "--no-edit"],
            cwd=str(wt_path),
            timeout=GIT_LOCAL_TIMEOUT,
        )
        if rc != 0:
            await self._run_git(
                ["git", "merge", "--abort"], cwd=str(wt_path), check=False
            )
            await self._remove_worktree_force(bare_path, wt_path)

            # Only call it a conflict when git actually says so. Anything else
            # is our problem, not the author's, and telling them to "resolve
            # conflicts" that do not exist wastes their time — a missing git
            # identity in the container was reported that way for exactly one
            # commit too long.
            if _looks_like_conflict(out):
                raise MergeConflictError(
                    f"PR #{pr_number} conflicts with {base_ref} and cannot be "
                    "merged. Please resolve the conflicts in the PR first."
                )
            raise RuntimeError(
                f"git merge of PR #{pr_number} into {base_ref} failed "
                f"(rc={rc}), and not because of a conflict:\n{out.strip()}"
            )

        logger.info(f"Worktree ready: {wt_path} ({base_ref} + PR #{pr_number})")
        return wt_path, await self._head_sha(wt_path)

    async def _head_sha(self, worktree_path: Path) -> str:
        """Worktree HEAD, or "" if it cannot be read.

        Never fatal: this is recorded for traceability, and losing the id is not
        a reason to fail a review that would otherwise have run.
        """
        rc, out = await self._run_git_status(
            ["git", "rev-parse", "HEAD"],
            cwd=str(worktree_path),
            timeout=GIT_LOCAL_TIMEOUT,
        )
        if rc != 0:
            logger.warning(f"Could not read build ref in {worktree_path}: {out.strip()}")
            return ""
        return out.strip()

    async def remove_worktree(self, repo_full_name: str, worktree_path: Path):
        """Remove a worktree after use."""
        bare_path = self._bare_path(repo_full_name)
        await self._remove_worktree_force(bare_path, worktree_path)

    # ── Diffs ─────────────────────────────────────────────────────────────────

    async def get_changed_files(
        self, worktree_path: Path, base_ref: str
    ) -> list[str]:
        """Files changed by the PR, relative to the merge base with `base_ref`."""
        stdout = await self._run_git(
            ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
            cwd=str(worktree_path),
            timeout=GIT_LOCAL_TIMEOUT,
        )
        return [f for f in stdout.strip().splitlines() if f]

    async def get_diff(
        self, worktree_path: Path, base_ref: str, max_lines: int = 3000
    ) -> str:
        """Full diff relative to `base_ref`, truncated for the LLM prompt."""
        stdout = await self._run_git(
            ["git", "diff", f"{base_ref}...HEAD"],
            cwd=str(worktree_path),
            timeout=GIT_LOCAL_TIMEOUT,
        )
        lines = stdout.splitlines()
        if len(lines) > max_lines:
            return (
                "\n".join(lines[:max_lines])
                + f"\n\n... (truncated, {len(lines)} total lines)"
            )
        return stdout

    async def get_diff_stat(self, worktree_path: Path, base_ref: str) -> str:
        """Diff stat, used for large-file detection."""
        return await self._run_git(
            ["git", "diff", "--stat", f"{base_ref}...HEAD"],
            cwd=str(worktree_path),
            timeout=GIT_LOCAL_TIMEOUT,
        )

    # ── Cleanup ───────────────────────────────────────────────────────────────

    async def cleanup_stale_worktrees(self):
        """Remove worktrees left over from crashed jobs."""
        if not self._worktrees_dir.exists():
            return
        for entry in self._worktrees_dir.iterdir():
            if not entry.is_dir():
                continue
            logger.warning(f"Cleaning up stale worktree: {entry}")
            for full_name in self._repos:
                bare_path = self._bare_path(full_name)
                if bare_path.exists():
                    await self._run_git(
                        ["git", "worktree", "remove", "--force", str(entry)],
                        cwd=str(bare_path),
                        check=False,
                    )
                    if not entry.exists():
                        break
            if entry.exists():
                await self._run_cmd(["rm", "-rf", str(entry)], check=False)

    async def _remove_worktree_force(self, bare_path: Path, wt_path: Path):
        await self._run_git(
            ["git", "worktree", "remove", "--force", str(wt_path)],
            cwd=str(bare_path),
            check=False,
        )
        if wt_path.exists():
            await self._run_cmd(["rm", "-rf", str(wt_path)], check=False)
        # Drop the administrative entry too, so a later `worktree add` at the
        # same path is not rejected as already registered.
        await self._run_git(
            ["git", "worktree", "prune"], cwd=str(bare_path), check=False
        )

    # ── Subprocess plumbing ───────────────────────────────────────────────────

    async def _run_git_network(
        self, cmd: list[str], cwd: str, label: str
    ) -> str:
        """Run a network git command, retrying transient failures.

        Connectivity to github.com from these hosts is intermittent, so a single
        failed attempt is not evidence of a real problem. Retrying here rather
        than relying on the job-level retry keeps a blip from costing a whole
        attempt (and a 60s backoff) and from surfacing as an error on the PR.
        """
        last = ""
        for attempt in range(1, NETWORK_RETRIES + 1):
            rc, out = await self._run_git_status(cmd, cwd=cwd, timeout=FETCH_TIMEOUT)
            if rc == 0:
                if attempt > 1:
                    logger.info(f"{label} succeeded on attempt {attempt}")
                return out
            last = out
            logger.warning(
                f"{label} failed (attempt {attempt}/{NETWORK_RETRIES}, rc={rc}): "
                f"{out.strip().splitlines()[-1] if out.strip() else '(no output)'}"
            )
            if attempt < NETWORK_RETRIES:
                await asyncio.sleep(NETWORK_RETRY_BACKOFF * attempt)

        raise RuntimeError(
            f"{label} failed after {NETWORK_RETRIES} attempts:\n{last.strip()}"
        )

    async def _run_git(
        self,
        cmd: list[str],
        cwd: str,
        check: bool = True,
        timeout: int | None = None,
    ) -> str:
        return await self._run_cmd(cmd, cwd=cwd, check=check, timeout=timeout)

    async def _run_git_status(
        self, cmd: list[str], cwd: str, timeout: int | None = None
    ) -> tuple[int, str]:
        """Run git and return (returncode, output) instead of raising.

        Needed where the *kind* of failure matters — a failed merge has to be
        told apart from a conflicting one.
        """
        return await self._run_cmd(
            cmd, cwd=cwd, check=False, timeout=timeout, with_status=True
        )

    async def _run_cmd(
        self,
        cmd: list[str],
        cwd: str | None = None,
        check: bool = True,
        timeout: int | None = None,
        with_status: bool = False,
    ):
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, **_GIT_ENV},
            start_new_session=True,
        )

        try:
            stdout_bytes, _ = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except (asyncio.TimeoutError, asyncio.CancelledError) as e:
            _kill(proc)
            await _reap(proc)
            if isinstance(e, asyncio.CancelledError):
                raise
            raise RuntimeError(
                f"git timed out after {timeout}s: {' '.join(cmd[:4])}"
            ) from e

        stdout = stdout_bytes.decode(errors="replace")
        rc = proc.returncode or 0

        if with_status:
            return rc, stdout
        if check and rc != 0:
            raise RuntimeError(
                f"Command failed (rc={rc}): {' '.join(cmd)}\n{stdout}"
            )
        return stdout


def _kill(proc: asyncio.subprocess.Process):
    """Kill a git process group — git spawns helpers like git-remote-https."""
    if proc.returncode is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), 9)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


async def _reap(proc: asyncio.subprocess.Process):
    try:
        await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=10)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
