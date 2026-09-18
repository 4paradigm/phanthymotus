"""Parsing and validation for PR-conversation deploy commands (pre-merge validation).

Commands must start a line. Only the PR's main-conversation comments are read.
Parsing is strict: unknown flags, a duplicated required arg or a stray argument
are rejected as ``unknown``.

Five commands are public:
  - ``/request_deploy`` — PR Author requests a deployment (ZERO parameters)
  - ``/approve_deploy`` — Machine Owner approves a deployment
  - ``/record_test`` — Machine Owner records test result (no machine=)
  - ``/deploy_status`` — Read-only status query
  - ``/deploy_help`` — Help for commands
"""

from __future__ import annotations

from dataclasses import dataclass
import shlex
from typing import Literal


CommandKind = Literal[
    "request_deploy",
    "approve_deploy",
    "record_test",
    "deploy_status",
    "deploy_help",
    "unknown",
]


@dataclass
class ParsedCommand:
    kind: CommandKind
    machine_alias: str = ""
    result: str = ""
    summary: str = ""
    help_topic: str = ""
    raw: str = ""
    comment_id: int = 0

    @property
    def is_command(self) -> bool:
        return self.kind != "unknown"


_KNOWN = {
    "request_deploy",
    "approve_deploy",
    "record_test",
    "deploy_status",
    "deploy_help",
}


def _split_token(t: str):
    if "=" in t:
        k, _, v = t.partition("=")
        return k.strip().lower(), v.strip().strip('"').strip("'")
    return None, t


def _parse_args(tokens: list[str]) -> tuple[list[str], dict[str, str]]:
    positional: list[str] = []
    kv: dict[str, str] = {}
    for t in tokens:
        k, v = _split_token(t)
        if k is None:
            positional.append(str(v))
        else:
            if k in kv:
                raise ValueError(f"duplicate argument {k}")
            kv[k] = v
    return positional, kv


# Allowed key=value arguments for each command.
_ALLOWED_KV = {
    "request_deploy": frozenset(),        # ZERO parameters
    "approve_deploy": frozenset({"machine"}),
    "record_test": frozenset({"result", "summary"}),
    "deploy_status": frozenset(),
    "deploy_help": frozenset(),
}


def _iter_top_level_command_lines(comment_text: str) -> list[str]:
    """Yield lines that are top-level (not inside fenced code / HTML comments).

    State machine rules:
    - Parse fence/HTML structural state BEFORE rejecting indented lines.
    - Fenced code block: allow 0..3 leading spaces. 4+ spaces is NOT a fence.
      Opening fence: >=3 consecutive ` or ~. Record fence char and length.
      Closing fence: same char, length >= opening, followed by space/tab/EOL only.
    - HTML comments: allow 0..3 leading spaces for <!-- and -->.
      Single-line <!-- ... --> completes on the same line.
      Inside HTML comment: never yield commands.
    - Only yield lines whose RAW original line truly starts with "/" at column 0.
    """
    in_fence = False
    fence_char = ""
    fence_min_len = 0
    in_html_comment = False
    result = []
    for raw_line in comment_text.splitlines():
        # Compute stripped once for structural detection
        stripped = raw_line.strip()

        # ── Structural parsing FIRST (fence / HTML comment) ──────────────
        # HTML comment: allow 0..3 leading spaces
        if "<!--" in stripped:
            if "-->" in stripped:
                # Single-line HTML comment: opens and closes on same line
                open_pos = stripped.index("<!--")
                close_pos = stripped.index("-->")
                if close_pos > open_pos:
                    # Content between <!-- and --> — stays inside comment
                    continue
                if in_html_comment:
                    # Closes an open multiline HTML comment
                    in_html_comment = False
                continue
            # If already inside an HTML comment, this nested <!-- is just content
            if not in_html_comment:
                # Check leading spaces for opening <!--
                leading = len(raw_line) - len(raw_line.lstrip(" \t"))
                if leading <= 3:
                    in_html_comment = True
                # If 4+ leading spaces, ignore as non-structural
            continue

        # Closing tag without opening (multiline end) — allow 0..3 spaces
        if "-->" in stripped and in_html_comment:
            leading = len(raw_line) - len(raw_line.lstrip(" \t"))
            if leading <= 3:
                in_html_comment = False
            continue

        if in_html_comment:
            continue

        # Closing fence check: allow 0..3 leading spaces
        if in_fence and stripped:
            lead = len(raw_line) - len(raw_line.lstrip(" \t"))
            if lead <= 3:
                # Closing fence: same char, length >= opening
                # Trailing char after fence run must be space/tab/EOL
                fence_run_len = 0
                for ch in stripped:
                    if ch == fence_char:
                        fence_run_len += 1
                    else:
                        break
                if fence_run_len >= fence_min_len:
                    # Check trailing: must be space, tab, or end of string
                    trailing = stripped[fence_run_len:]
                    if trailing == "" or trailing[0] in (" ", "\t"):
                        in_fence = False
                        fence_char = ""
                        fence_min_len = 0
                        continue
            # Not a valid closing fence — stay in fence, skip this line
            continue

        # Opening fence detection: allow 0..3 leading spaces
        lead = len(raw_line) - len(raw_line.lstrip(" \t"))
        if lead <= 3 and stripped:
            for ch in ("`", "~"):
                if stripped.startswith(ch * 3):
                    fence_len = 0
                    for c in stripped:
                        if c == ch:
                            fence_len += 1
                        else:
                            break
                    if fence_len >= 3:
                        if not in_fence:
                            in_fence = True
                            fence_char = ch
                            fence_min_len = fence_len
                        else:
                            # Closing fence via opening-detection path (same char)
                            if ch == fence_char and fence_len >= fence_min_len:
                                trailing = stripped[fence_len:]
                                if trailing == "" or trailing[0] in (" ", "\t"):
                                    in_fence = False
                                    fence_char = ""
                                    fence_min_len = 0
                        break
                # Break out of ch loop once we've checked both ` and ~
                if ch == "~":
                    break

        if in_fence:
            continue

        # Skip indented lines (tabs or spaces) — not top-level commands
        if raw_line and raw_line[0] in (" ", "\t"):
            continue
        # Skip blockquote lines
        if raw_line.startswith(">"):
            continue

        # Only top-level column-0 lines starting with "/"
        if raw_line.startswith("/") and len(raw_line) > 1 and (raw_line[1] != " "):
            result.append(raw_line)
    return result


def parse_command(comment_text: str) -> ParsedCommand:
    """Parse the first top-level command line in ``comment_text``."""
    if not comment_text:
        return ParsedCommand(kind="unknown", raw="")
    for raw_line in _iter_top_level_command_lines(comment_text):
        line = raw_line.lstrip()[1:]  # strip leading "/" and any whitespace after
        try:
            tokens = shlex.split(raw_line)
        except ValueError:
            continue
        if not tokens:
            continue
        name = tokens[0][1:].strip().lower()
        if name not in _KNOWN:
            continue
        try:
            positional, kv = _parse_args(tokens[1:])
        except ValueError:
            continue
        allowed = _ALLOWED_KV[name]
        if any(k not in allowed for k in kv):
            continue
        # request_deploy: reject any positional args or key=value args
        if name == "request_deploy":
            if positional:
                continue
            return ParsedCommand(kind="request_deploy", raw=raw_line.strip())
        return _build_command(name, positional, kv, tokens[1:])
    return ParsedCommand(kind="unknown", raw="")


def _build_command(name: str, positional: list[str], kv: dict[str, str],
                   raw_tokens: list[str]) -> ParsedCommand:
    raw = name + " " + " ".join(raw_tokens)
    if name == "request_deploy":
        # Already handled above
        return ParsedCommand(kind="request_deploy", raw=raw)
    if name == "approve_deploy":
        if positional:
            return ParsedCommand(kind="unknown", raw="")
        machine_alias = kv.get("machine", "")
        if not machine_alias:
            return ParsedCommand(kind="unknown", raw="")
        return ParsedCommand(
            kind="approve_deploy", machine_alias=machine_alias, raw=raw
        )
    if name == "record_test":
        if positional:
            return ParsedCommand(kind="unknown", raw="")
        result = kv.get("result", "")
        if result not in ("pass", "fail"):
            return ParsedCommand(kind="unknown", raw="")
        # Reject machine= parameter
        if "machine" in kv:
            return ParsedCommand(kind="unknown", raw="")
        summary = kv.get("summary", "")
        return ParsedCommand(
            kind="record_test", result=result, summary=summary, raw=raw
        )
    if name == "deploy_status":
        if positional:
            return ParsedCommand(kind="unknown", raw="")
        return ParsedCommand(kind="deploy_status", raw=raw)
    if name == "deploy_help":
        topic = " ".join(positional) if positional else ""
        return ParsedCommand(
            kind="deploy_help", help_topic=topic, raw=raw
        )
    return ParsedCommand(kind="unknown", raw="")


def command_starts_line(text: str, kind: str) -> bool:
    prefix = "/" + kind
    for line in _iter_top_level_command_lines(text):
        if line.startswith(prefix):
            return True
    return False


def command_starts_line_any(text: str) -> bool:
    for line in _iter_top_level_command_lines(text):
        name = line.lstrip()[1:].split()[0].lower()
        if name in _KNOWN:
            return True
    return False
