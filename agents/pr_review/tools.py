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

# Per-result char budgets, one per tool. A single shared 4000 made every one of
# the item-count caps below unreachable: 400 numbered lines of Python is ~20 KB,
# so `max_lines` was dead — every read returned ~65 lines whatever was asked
# for, and `grep` announced "60 match(es)" after delivering 30.
#
# These move together with review_agent.MAX_TOOL_RESULT and
# review_trace.MAX_FIELD_CHARS. Raising one alone accomplishes nothing: the
# transcript independently re-truncates every tool result, and the trace
# under-reports what the model actually saw — which is how a truncation bug
# stayed invisible across several passes over the traces.
MAX_READ_CHARS = 12000
MAX_DIFF_CHARS = 12000
MAX_GREP_CHARS = 8000
MAX_LIST_CHARS = 8000

# Backstop for any path that still calls `_cap` without an explicit limit. Not
# what bounds normal tool output any more — each tool below counts characters
# while it builds its result, so the size is bounded by construction.
MAX_RESULT_CHARS = 12000

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

    # Counted while accumulating. Capping the joined string truncated entries
    # the header had already counted, and the "… N more entries not shown"
    # footer was itself among the first things removed — so a half-listed
    # directory looked complete.
    shown: list[str] = []
    used = 0
    for r in rows:
        if shown and used + len(r) + 1 > MAX_LIST_CHARS:
            break
        shown.append(r)
        used += len(r) + 1

    header = (
        f"{sb.rel(target)}/  —  {len(shown)} entr"
        f"{'y' if len(shown) == 1 else 'ies'}"
    )
    hidden = (len(entries) - MAX_LIST_ENTRIES if len(entries) > MAX_LIST_ENTRIES
              else 0) + (len(rows) - len(shown))
    more = ""
    if hidden:
        more = f"\n  … {hidden} more entries not shown"
    return header + "\n" + "\n".join(shown) + more


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

    # The header counts against the budget too. Sizing only the body returned
    # MAX_READ_CHARS + header, which the transcript's own cap then trimmed —
    # putting the lie back one layer up. The allowance covers the longest header
    # this function can write: the path appears twice, once in the resume hint.
    header_allowance = 2 * len(path) + 160
    body_lines, shown_to, hit_budget = _emit_lines(
        lines, start, limit, max(MAX_READ_CHARS - header_allowance, 500)
    )
    if not body_lines:
        return f"{path}: no lines at {start} (file has {len(lines)})"

    header = f"{path}  (lines {start}-{shown_to} of {len(lines)})"
    if shown_to < len(lines):
        remaining = len(lines) - shown_to
        why = (
            "stopped at the result-size limit"
            if hit_budget
            else f"returned the {len(body_lines)} lines requested"
        )
        # The resume hint is the load-bearing half. An honest range alone is not
        # enough: the model still has to work out the next call, and getting
        # that wrong is precisely what the overlapping-read loop was.
        header += (
            f" — {why}; {remaining} lines remain. Continue with "
            f'read_file("{path}", start_line={shown_to + 1}).'
        )
    # No _cap: the budget is enforced per line above, so this is bounded by
    # construction. Wrapping it would re-introduce the post-hoc truncation this
    # function was rewritten to remove.
    return header + "\n" + "\n".join(body_lines)


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

    # Counted while accumulating, not by capping the joined string: the old
    # `_cap(head + matches)` silently dropped about half of a full result set
    # while the header still claimed all 60, so "60 match(es) … (capped)" was a
    # lie about a lie.
    shown: list[str] = []
    used = 0
    for m in matches:
        if shown and used + len(m) + 1 > MAX_GREP_CHARS:
            break
        shown.append(m)
        used += len(m) + 1

    if len(shown) < len(matches):
        head = (
            f"showing {len(shown)} of {len(matches)}"
            f"{'+' if len(matches) >= MAX_GREP_MATCHES else ''} matches for "
            f"{pattern!r} (result-size limit) — narrow with path= or glob="
        )
    else:
        head = f"{len(shown)} match(es) for {pattern!r}"
        if len(matches) >= MAX_GREP_MATCHES:
            head += f" (stopped at the first {MAX_GREP_MATCHES}; there may be more)"
    return head + "\n" + "\n".join(shown)


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

    preamble, hunks = _split_hunks(body)
    if not hunks:
        # A --stat summary, or a pure rename/mode change: nothing to cut at.
        return _cap(body, MAX_DIFF_CHARS)
    return _fit_hunks(preamble, hunks, MAX_DIFF_CHARS)


# ── Helpers ───────────────────────────────────────────────────────────────────


# `@@ -old,count +new,count @@ trailing`. The post-image start (group 2) is what
# a resume hint needs: it is the line number the reader would open the file at.
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")

# Dropped-hunk rows in the footer. A 500-hunk diff should say so, not list them.
_MAX_FOOTER_ROWS = 20

# Per-row width in that footer. A hunk header carries trailing context (the
# enclosing function), which is useful but unbounded — 20 unbounded rows
# overflowed the space reserved for them, which is the one thing the footer
# must never do.
_FOOTER_ROW_CHARS = 116


def _split_hunks(diff: str) -> tuple[str, list[dict]]:
    """Split a unified diff into its leading file headers and its hunks.

    Each hunk carries the file it belongs to and its post-image start line, so a
    dropped hunk can be named as a `read_file` call rather than silently lost.
    Any `diff --git` / `+++` header lines between two hunks are folded into the
    following hunk's text, so re-joining the pieces reproduces the input.
    """
    preamble_lines: list[str] = []
    hunks: list[dict] = []
    pending: list[str] = []      # header lines seen since the last hunk closed
    cur: dict | None = None
    cur_file = ""

    for ln in diff.split("\n"):
        if ln.startswith("+++ b/"):
            cur_file = ln[6:].strip()
        elif ln.startswith("+++ ") and not ln.startswith("+++ /dev/null"):
            cur_file = ln[4:].strip()

        m = _HUNK_RE.match(ln)
        if m:
            if cur is not None:
                hunks.append(cur)
            # `pending` holds whatever preceded this hunk. Before the first hunk
            # that is the diff's own header and becomes the preamble; between
            # hunks it is a new file's `diff --git` block and belongs to this
            # hunk, so that re-joining the pieces reproduces the input.
            if not hunks and cur is None:
                preamble_lines = pending
                head: list[str] = []
            else:
                head = pending
            cur = {
                "file": cur_file,
                "post_start": int(m.group(2)),
                "header": ln,
                "text": "\n".join(head + [ln]),
            }
            pending = []
            continue

        if cur is None:
            pending.append(ln)
        elif ln.startswith("diff --git "):
            # A new file section closes the open hunk.
            hunks.append(cur)
            cur = None
            pending = [ln]
        else:
            cur["text"] += "\n" + ln

    if cur is not None:
        hunks.append(cur)
    elif pending and hunks:
        # Trailing lines after the last hunk with nothing open — keep them.
        hunks[-1]["text"] += "\n" + "\n".join(pending)

    if not hunks:
        return diff, []
    return "\n".join(preamble_lines), hunks


def _dropped_footer(dropped: list[dict], total: int, budget: int) -> str:
    """Name the omitted hunks as the `read_file` calls that would show them."""
    rows = []
    for h in dropped[:_MAX_FOOTER_ROWS]:
        row = (
            f'  read_file("{h["file"]}", start_line={h["post_start"]})'
            f'   ({h["header"].strip()})'
        )
        rows.append(row[:_FOOTER_ROW_CHARS])
    if len(dropped) > _MAX_FOOTER_ROWS:
        rows.append(f"  … and {len(dropped) - _MAX_FOOTER_ROWS} more")
    return (
        f"… {len(dropped)} of {total} hunks omitted at the {budget}-char "
        "result limit. The omitted hunks touch these ranges — read them "
        "directly:\n" + "\n".join(rows)
    )


def _fit_hunks(preamble: str, hunks: list[dict], budget: int) -> str:
    """As many whole hunks as fit, then say which ranges were dropped.

    Cutting mid-hunk is worse than dropping a hunk: a half-hunk looks like a
    complete one. On phanthymotus-driver PR #174 a diff truncated at 4000 chars
    with no header, no line numbers and no way to ask for the rest is what sent
    the reviewer off to page through a 4741-line file by hand for 13 rounds. A
    hunk boundary plus the dropped ranges lets it read exactly what it is
    missing with `read_file`, which now reports honest ranges.
    """
    def fit(limit: int) -> tuple[list[str], int]:
        out: list[str] = []
        used = 0
        if preamble:
            out.append(preamble)
            used = len(preamble) + 1
        kept = 0
        for h in hunks:
            text = h["text"]
            if used + len(text) + 1 <= limit:
                out.append(text)
                used += len(text) + 1
                kept += 1
                continue
            if kept == 0:
                # One hunk larger than the whole budget. Returning only a footer
                # would say nothing about the change, so emit what fits and mark
                # the boundary explicitly rather than letting it pass as whole.
                room = max(limit - used - 200, 500)
                out.append(
                    text[:room]
                    + f"\n…[hunk truncated at {room} chars; "
                    + f'read_file("{h["file"]}", start_line={h["post_start"]}) '
                    + "for the rest]"
                )
                kept = 1
            break
        return out, kept

    # The footer's size depends on how many hunks are dropped, which depends on
    # the space left for it — so reserve nothing, measure, and re-fit. Converges
    # in at most a couple of passes because the footer is bounded at
    # _MAX_FOOTER_ROWS rows of _FOOTER_ROW_CHARS. Guessing a fixed reserve
    # instead both overflowed the budget on a 371-hunk diff and threw away a
    # fifth of it on diffs that had nothing to drop.
    reserve = 0
    for _ in range(4):
        out, kept = fit(budget - reserve)
        if kept >= len(hunks):
            return "\n".join(out)
        footer = _dropped_footer(hunks[kept:], len(hunks), budget)
        if len(footer) + 1 <= reserve:
            break
        reserve = len(footer) + 1
    return "\n".join(out + [footer])


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


def _cap(s: str, limit: int = MAX_RESULT_CHARS) -> str:
    """Last-resort truncation. Prefer counting while building — see `_emit_lines`.

    Truncating a finished string is what caused the bug this module was
    rewritten for: the caller had already written a header describing content
    that `_cap` then removed, and the model has no way to see where its own
    input was cut.
    """
    if len(s) <= limit:
        return s
    # The note counts against the limit. Appending it afterwards returned
    # limit+48 chars, which then tripped the transcript's own cap — the same
    # mistake one layer down.
    note = f"\n… truncated at {limit} chars — narrow the request"
    return s[: max(limit - len(note), 0)] + note


def _emit_lines(
    lines: list[str], start: int, limit: int, budget: int
) -> tuple[list[str], int, bool]:
    """Numbered lines that fit `budget` chars, the last line number reached, and
    whether the budget (rather than `limit`) is what stopped it.

    The cap has to be applied *while* building the body, never to the finished
    string. `_cap(header + body)` truncated after the header was already
    written, so a read of lines 2200-2719 announced 520 lines, delivered 65, and
    put no marker at the boundary. The model cannot see where its input was cut,
    so it guesses the next window: on phanthymotus-driver PR #174 that produced
    13 consecutive overlapping reads of one 4741-line file — 2200-2719,
    2280-2779, 2338-2777, 2700-3319, 2650-3299, … — and spent the entire round
    budget without ever calling finish_review.
    """
    out: list[str] = []
    used = 0
    stop = min(start - 1 + limit, len(lines))
    hit_budget = False
    marker = " …[single line truncated]"
    for i in range(start - 1, stop):
        row = f"{i + 1:5d}  {lines[i]}"
        # A single line longer than the whole budget — minified JS, a generated
        # header, a one-line JSON blob — would otherwise return zero lines and
        # read as an empty file, indistinguishable from a bad start_line.
        # Hard-truncate that one line and say so, marker included in the budget.
        if len(row) > budget:
            row = row[: max(budget - len(marker), 0)] + marker
        # `out and` guarantees at least one line always comes back, so the
        # resume hint the caller builds from `shown_to` always advances and
        # cannot suggest the same window forever.
        if out and used + len(row) + 1 > budget:
            hit_budget = True
            break
        out.append(row)
        used += len(row) + 1
    return out, start + len(out) - 1, hit_budget


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
                    "max_lines": {
                        "type": "integer",
                        "description": (
                            "Lines to return (default 200, max 400). If the "
                            "result does not fit the size limit, the header "
                            "names the exact line to resume from — use that "
                            "number rather than guessing a new window."
                        ),
                    },
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
                "a path, the full diff for that file. A diff too large for one "
                "result is returned as whole hunks, and the omitted hunks are "
                "listed as the read_file calls that would show them."
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
