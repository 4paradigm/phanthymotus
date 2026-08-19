"""Read-only tools the review loop uses to explore a PR's checkout.

Everything here operates on a worktree built from an **untrusted PR**: both the
file names and the file contents are authored by whoever opened it. The sandbox
is therefore the load-bearing part of this module, not an afterthought.

Two attacks it exists to stop:

- **Path traversal** — `read_file("../../../jobs.db")`, or an absolute path.
- **Symlink escape** — a PR that commits `notes.txt -> /proc/self/environ` and
  asks the reviewer to read it. That file holds GITHUB_TOKEN, REGISTRY_PASSWORD
  and LLM_API_KEY, and the review is posted to a **public PR comment**, so a
  successful read is credential exfiltration. `Path.resolve()` follows symlinks,
  so resolving *first* and then checking containment catches both cases with one
  test.

Confinement alone is not sufficient, because a secret can be committed *inside*
the checkout, where every path check passes. So a credential-shaped filename is
refused as well — see `Sandbox.is_sensitive`. That refusal has to happen in
`_walk` too, not just `resolve`: `grep` never calls `resolve` on the files it
visits, so it would otherwise return a committed `.env` line by line.

There is deliberately no shell/exec tool. Reviewing code needs reading; `exec`
would add an execution path and buy no review capability.
"""

import fnmatch
import logging
import re
import subprocess
from pathlib import Path

from .reviewer import is_sensitive_name

logger = logging.getLogger(__name__)

# Per-result cap. One `read_file` of a generated 5 MB file would otherwise
# consume the whole context window.
MAX_RESULT_CHARS = 4000
MAX_READ_LINES = 400
MAX_GREP_MATCHES = 60
MAX_LIST_ENTRIES = 200

# Never surfaced: .git carries remote/credential config and is pure noise for a
# review; the rest are bulky vendored trees that waste the budget.
EXCLUDED_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache"}

# Extensions read as bytes rather than text.
BINARY_SUFFIXES = {
    ".so", ".a", ".o", ".pyc", ".zip", ".gz", ".tar", ".xz", ".bz2", ".whl",
    ".pt", ".onnx", ".bin", ".jpg", ".jpeg", ".png", ".gif", ".pdf", ".urdf.bin",
}


class SandboxError(Exception):
    """A tool was asked for something it must not read."""


class Sandbox:
    """Confines every path operation to one worktree."""

    def __init__(self, worktree: Path, base_ref: str = "main"):
        self._root = worktree.resolve()
        self._base_ref = base_ref

    @property
    def root(self) -> Path:
        return self._root

    def resolve(self, rel: str) -> Path:
        """Resolve a caller-supplied path, or raise SandboxError.

        `Path.resolve()` is what makes this safe against symlinks: a link
        pointing outside the worktree resolves to its target, which then fails
        the containment check. Checking the literal path first would pass.
        """
        raw = (rel or ".").strip()
        # An absolute path is never valid — paths are worktree-relative.
        candidate = self._root / raw.lstrip("/")
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError) as e:  # RuntimeError: symlink loop
            raise SandboxError(f"cannot resolve {raw!r}: {e}") from e

        if resolved != self._root and not resolved.is_relative_to(self._root):
            raise SandboxError(
                f"{raw!r} resolves outside the PR checkout and was refused. "
                "Only paths inside the repository can be read."
            )

        parts = set(resolved.relative_to(self._root).parts)
        if parts & EXCLUDED_DIRS:
            raise SandboxError(f"{raw!r} is inside an excluded directory")

        if self.is_sensitive(resolved):
            # Confinement is not enough here: this file is *inside* the
            # checkout, so it passes every path check. But a PR that commits a
            # real .env or id_rsa would otherwise have its contents read into a
            # public PR comment and the dashboard timeline by the reviewer
            # itself. The rule checks already report the file; that is the right
            # way to review it.
            raise SandboxError(
                f"{raw!r} matches a secret-bearing filename pattern and was "
                "refused. Do not try to read it — the rule checks already "
                "flag it. Review it by name and by what the PR says about it."
            )

        return resolved

    def is_sensitive(self, p: Path) -> bool:
        """Whether a path inside the checkout may carry a credential.

        Every path segment is tested, not just the filename: a directory called
        `secrets/` makes `secrets/notes.txt` sensitive even though the file's own
        name is innocuous.
        """
        try:
            rel_parts = p.relative_to(self._root).parts
        except ValueError:
            return False
        return any(is_sensitive_name(part) for part in rel_parts)

    def rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(self._root)) or "."
        except ValueError:
            return str(p)


# ── Tool implementations ──────────────────────────────────────────────────────
#
# Each returns a string — what the model sees. Failures are returned, never
# raised, so the model can correct itself and continue.


def list_dir(sb: Sandbox, path: str = ".") -> str:
    """Entries in a directory, with type and size."""
    try:
        target = sb.resolve(path)
    except SandboxError as e:
        return f"error: {e}"

    if not target.exists():
        return f"error: {path!r} does not exist"
    if not target.is_dir():
        return f"error: {path!r} is not a directory (use read_file)"

    rows = []
    try:
        entries = sorted(
            target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
        )
    except OSError as e:
        return f"error: cannot list {path!r}: {e}"

    for e in entries[:MAX_LIST_ENTRIES]:
        if e.name in EXCLUDED_DIRS:
            continue
        # lstat, so a symlink is reported as a link rather than followed.
        try:
            st = e.lstat()
        except OSError:
            continue
        if e.is_symlink():
            try:
                dest = e.readlink()
            except OSError:
                dest = "?"
            rows.append(f"  {e.name} -> {dest}  [symlink, not followed]")
        elif e.is_dir():
            rows.append(f"  {e.name}/")
        elif sb.is_sensitive(e):
            rows.append(
                f"  {e.name}  ({_human(st.st_size)})  "
                "[possible secret — contents not readable]"
            )
        else:
            rows.append(f"  {e.name}  ({_human(st.st_size)})")

    header = f"{sb.rel(target)}/  —  {len(rows)} entr{'y' if len(rows) == 1 else 'ies'}"
    more = ""
    if len(entries) > MAX_LIST_ENTRIES:
        more = f"\n  … {len(entries) - MAX_LIST_ENTRIES} more entries not shown"
    return _cap(header + "\n" + "\n".join(rows) + more)


def read_file(
    sb: Sandbox, path: str, start_line: int = 1, max_lines: int = 200
) -> str:
    """Read a text file, line-numbered."""
    try:
        target = sb.resolve(path)
    except SandboxError as e:
        return f"error: {e}"

    if not target.exists():
        return f"error: {path!r} does not exist"
    if target.is_dir():
        return f"error: {path!r} is a directory (use list_dir)"

    if target.suffix.lower() in BINARY_SUFFIXES:
        size = _safe_size(target)
        return (
            f"{path} is a binary file ({_human(size)}); contents not shown. "
            "Binary assets normally belong in COS rather than the repo."
        )

    try:
        raw = target.read_bytes()
    except OSError as e:
        return f"error: cannot read {path!r}: {e}"

    if b"\0" in raw[:8192]:
        return f"{path} appears to be binary ({_human(len(raw))}); not shown."

    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    start = max(1, start_line)
    limit = max(1, min(max_lines, MAX_READ_LINES))
    chunk = lines[start - 1: start - 1 + limit]

    if not chunk:
        return f"{path}: no lines at {start} (file has {len(lines)})"

    body = "\n".join(f"{start + i:5d}  {ln}" for i, ln in enumerate(chunk))
    shown_to = start + len(chunk) - 1
    header = f"{path}  (lines {start}-{shown_to} of {len(lines)})"
    if shown_to < len(lines):
        header += f" — {len(lines) - shown_to} more lines below"
    return _cap(header + "\n" + body)


def grep(sb: Sandbox, pattern: str, path: str = ".", glob: str = "") -> str:
    """Search file contents by regex."""
    try:
        target = sb.resolve(path)
    except SandboxError as e:
        return f"error: {e}"

    try:
        rx = re.compile(pattern)
    except re.error as e:
        # Returned, not raised: the model can fix its own regex.
        return f"error: invalid regex {pattern!r}: {e}"

    files = [target] if target.is_file() else _walk(sb, target)
    matches = []
    for f in files:
        if glob and not fnmatch.fnmatch(f.name, glob):
            continue
        if f.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            content = f.read_text(errors="replace")
        except OSError:
            continue
        for n, line in enumerate(content.splitlines(), 1):
            if rx.search(line):
                matches.append(f"{sb.rel(f)}:{n}: {line.strip()[:200]}")
                if len(matches) >= MAX_GREP_MATCHES:
                    break
        if len(matches) >= MAX_GREP_MATCHES:
            break

    if not matches:
        return f"no matches for {pattern!r} under {path}"
    head = f"{len(matches)} match(es) for {pattern!r}"
    if len(matches) >= MAX_GREP_MATCHES:
        head += " (capped)"
    return _cap(head + "\n" + "\n".join(matches))


def file_diff(sb: Sandbox, path: str = "") -> str:
    """This PR's diff, for one file or in summary."""
    args = ["git", "diff", f"{sb._base_ref}...HEAD"]
    if path:
        try:
            target = sb.resolve(path)
        except SandboxError as e:
            return f"error: {e}"
        # Pass the path after `--` so a filename starting with `-` cannot be
        # read as a git option.
        args += ["--", sb.rel(target)]
    else:
        args.insert(2, "--stat")

    try:
        out = subprocess.run(
            args, cwd=str(sb.root), capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"error: git diff failed: {e}"

    if out.returncode != 0:
        return f"error: git diff failed: {(out.stderr or '').strip()[:300]}"
    body = out.stdout.strip()
    if not body:
        return f"no diff for {path or 'this PR'}"
    return _cap(body)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _walk(sb: Sandbox, root: Path) -> list[Path]:
    found = []
    for p in root.rglob("*"):
        if not p.is_file() or p.is_symlink():
            continue
        try:
            if set(p.relative_to(sb.root).parts) & EXCLUDED_DIRS:
                continue
        except ValueError:
            continue
        # The more important half of the secret refusal: grep walks the whole
        # tree, so without this one `grep "TOKEN"` spills a committed .env line
        # by line, never going through resolve().
        if sb.is_sensitive(p):
            continue
        found.append(p)
    return found


def _safe_size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def _human(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / 1024 / 1024:.1f}MB"
    if n >= 1024:
        return f"{n / 1024:.0f}KB"
    return f"{n}B"


def _cap(s: str) -> str:
    if len(s) <= MAX_RESULT_CHARS:
        return s
    return (
        s[:MAX_RESULT_CHARS]
        + f"\n… truncated at {MAX_RESULT_CHARS} chars — narrow the request"
    )


# ── Schemas ───────────────────────────────────────────────────────────────────
#
# Hand-written rather than reflected from type hints: agent-core's
# `_build_system_tools` only maps str/int/float/bool and would KeyError on
# anything else, and these need per-parameter prose anyway.

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": (
                "List a directory in the PR checkout. Use it to understand "
                "layout, or to compare a new component against an existing one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Repo-relative directory, e.g. 'unitree/g1'. Defaults to the repo root.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a text file from the PR checkout, line-numbered. Read the "
                "component's docs and the files this PR changes before judging them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Repo-relative file path."},
                    "start_line": {"type": "integer", "description": "First line (default 1)."},
                    "max_lines": {"type": "integer", "description": "Lines to return (default 200, max 400)."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search file contents by regex. Use it to check whether a "
                "convention is followed elsewhere, or to find a definition."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Python regex."},
                    "path": {"type": "string", "description": "File or directory to search (default repo root)."},
                    "glob": {"type": "string", "description": "Filename filter, e.g. '*.py'."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_diff",
            "description": (
                "This PR's diff. With no path, returns the --stat summary; with "
                "a path, the full diff for that file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Repo-relative file path, or omit for the summary."}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_review",
            "description": (
                "Submit the finished review. Call this exactly once, when you "
                "have read enough to judge the change."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "One or two sentences on what this PR does.",
                    },
                    "issues": {
                        "type": "string",
                        "description": (
                            "Markdown bullets, most severe first, each with a "
                            "file:line reference and the concrete consequence. "
                            "'No issues found.' if there are none."
                        ),
                    },
                    "suggestions": {
                        "type": "string",
                        "description": "Markdown bullets of non-blocking improvements, or 'No suggestions.'",
                    },
                },
                "required": ["summary", "issues"],
            },
        },
    },
]

FINISH_TOOL = "finish_review"

DISPATCH = {
    "list_dir": list_dir,
    "read_file": read_file,
    "grep": grep,
    "file_diff": file_diff,
}
