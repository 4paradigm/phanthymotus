"""Parsing and validation for PR-conversation deploy commands (pre-merge validation).

Commands must start a line. Only the PR's main-conversation comments are read.
Parsing is strict: unknown flags, a duplicated required arg or a stray argument
are rejected as ``unknown``.

Commands are public:
  - ``/request_deploy`` — PR Author requests a deployment (ZERO parameters)
  - ``/approve_deploy`` — Machine Owner approves a deployment
  - ``/record_test`` — Machine Owner records test result (no machine=)
  - ``/request_acceptance`` — Machine Owner requests independent acceptance
  - ``/evidence_link`` — Read-only short-lived evidence download link
  - ``/deploy_status`` — Read-only status query
  - ``/deploy_help`` — Help for commands
  - ``/submit_evidence`` — Persist native PR attachments and notes
  - ``/remove_evidence`` — Add a logical deletion record for an attachment
"""

from __future__ import annotations

from dataclasses import dataclass
import shlex
import re
from difflib import SequenceMatcher
from typing import Literal


CommandKind = Literal[
    "request_deploy",
    "approve_deploy",
    "record_test",
    "request_acceptance",
    "evidence_link",
    "deploy_status",
    "deploy_help",
    "submit_evidence",
    "remove_evidence",
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
    evidence_args: dict[str, str] | None = None
    error: str = ""
    intent: str = ""

    @property
    def is_command(self) -> bool:
        return self.kind != "unknown" or bool(self.intent)


_KNOWN = {
    "request_deploy",
    "approve_deploy",
    "record_test",
    "request_acceptance",
    "evidence_link",
    "deploy_status",
    "deploy_help",
    "submit_evidence",
    "remove_evidence",
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
    "request_acceptance": frozenset(),
    "evidence_link": frozenset({"id"}),
    "deploy_status": frozenset(),
    "deploy_help": frozenset(),
    "submit_evidence": frozenset({"head", "round", "machine", "note", "replaces"}),
    "remove_evidence": frozenset({"id", "head", "round"}),
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
        try:
            tokens = shlex.split(raw_line)
        except ValueError:
            first_token = raw_line.split(None, 1)[0]
            name = first_token[1:].strip().lower()
            if name in _KNOWN:
                return ParsedCommand(
                    kind=name, raw=raw_line.strip(), error="invalid_arguments",
                )
            continue
        if not tokens:
            continue
        name = tokens[0][1:].strip().lower()
        if name not in _KNOWN:
            continue
        try:
            positional, kv = _parse_args(tokens[1:])
        except ValueError as exc:
            return ParsedCommand(kind=name, raw=raw_line.strip(), error=str(exc))
        allowed = _ALLOWED_KV[name]
        if any(k not in allowed for k in kv):
            unknown = sorted(set(kv) - allowed)
            return ParsedCommand(
                kind=name, raw=raw_line.strip(),
                error=f"unknown_argument:{','.join(unknown)}",
            )
        # request_deploy: reject any positional args or key=value args
        if name == "request_deploy":
            if positional:
                return ParsedCommand(kind=name, raw=raw_line.strip(), error="unexpected_argument")
            return ParsedCommand(kind="request_deploy", raw=raw_line.strip())
        if name == "request_acceptance":
            if positional or kv:
                return ParsedCommand(kind=name, raw=raw_line.strip(), error="unexpected_argument")
            return ParsedCommand(kind="request_acceptance", raw=raw_line.strip())
        parsed = _build_command(name, positional, kv, tokens[1:])
        if parsed.kind == "unknown":
            parsed.kind = name
            parsed.raw = raw_line.strip()
            parsed.error = parsed.error or "invalid_arguments"
        return parsed
    return ParsedCommand(kind="unknown", raw="")


def _is_single_edit_or_transposition(name: str, candidate: str) -> bool:
    """Match one likely typo without accepting arbitrary similar slash commands."""
    if name == candidate:
        return True
    if abs(len(name) - len(candidate)) > 1:
        return False
    if (len(name) == len(candidate) + 1
            and (name.startswith(candidate) or name.endswith(candidate))):
        return False
    if (len(candidate) == len(name) + 1
            and (candidate.startswith(name) or candidate.endswith(name))):
        return False
    if len(name) == len(candidate):
        differences = [index for index, (left, right) in enumerate(zip(name, candidate))
                       if left != right]
        if len(differences) == 1:
            return True
        return (
            len(differences) == 2
            and differences[1] == differences[0] + 1
            and name[differences[0]] == candidate[differences[1]]
            and name[differences[1]] == candidate[differences[0]]
        )

    shorter, longer = (name, candidate) if len(name) < len(candidate) else (candidate, name)
    short_index = long_index = 0
    skipped = False
    while short_index < len(shorter) and long_index < len(longer):
        if shorter[short_index] == longer[long_index]:
            short_index += 1
            long_index += 1
            continue
        if skipped:
            return False
        skipped = True
        long_index += 1
    return True


def detect_deployment_command_intent(comment_text: str) -> str:
    """Return an exact or near-match deployment command name from a top-level line.

    This deliberately ignores prose, quotes, blockquotes, fenced examples, and
    unrelated slash commands so the watcher can answer only likely commands.
    """
    candidates = {"request_deploy", "approve_deploy", "request_acceptance", "record_test"}
    for line in _iter_top_level_command_lines(comment_text or ""):
        parts = line.split(None, 1)
        token = parts[0] if parts else ""
        if not token.startswith("/"):
            continue
        name = token[1:].strip().lower()
        if name in candidates:
            return name
        closest = max(candidates, key=lambda value: SequenceMatcher(None, name, value).ratio())
        if (len(name) >= 8
                and SequenceMatcher(None, name, closest).ratio() >= 0.8
                and _is_single_edit_or_transposition(name, closest)):
            return closest
    return ""


def _build_command(name: str, positional: list[str], kv: dict[str, str],
                   raw_tokens: list[str]) -> ParsedCommand:
    raw = name + " " + " ".join(raw_tokens)
    if name == "request_deploy":
        # Already handled above
        return ParsedCommand(kind="request_deploy", raw=raw)
    if name == "approve_deploy":
        if positional:
            return ParsedCommand(kind="unknown", raw="", error="unexpected_argument")
        machine_alias = kv.get("machine", "")
        if "machine" in kv and not machine_alias:
            return ParsedCommand(kind="unknown", raw="", error="missing_machine_value")
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
    if name == "evidence_link":
        if positional:
            return ParsedCommand(kind="unknown", raw="")
        attachment_id = kv.get("id", "")
        if kv and not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", attachment_id):
            return ParsedCommand(kind="unknown", raw="")
        return ParsedCommand(
            kind="evidence_link", evidence_args={"id": attachment_id} if attachment_id else {}, raw=raw,
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
    if name in {"submit_evidence", "remove_evidence"}:
        if positional:
            return ParsedCommand(kind="unknown", raw="")
        head = kv.get("head", "")
        round_id = kv.get("round", "")
        if (not re.fullmatch(r"[0-9a-fA-F]{40}", head)
                or not re.fullmatch(r"[0-9a-fA-F]{40}-[0-9]+", round_id)
                or not round_id.lower().startswith(head.lower() + "-")):
            return ParsedCommand(kind="unknown", raw="")
        if name == "submit_evidence":
            return ParsedCommand(kind=name, evidence_args={
                "head": head.lower(), "round": round_id.lower(),
                "machine": kv.get("machine", ""),
                "note": kv.get("note", ""), "replaces": kv.get("replaces", ""),
            }, raw=raw)
        attachment_id = kv.get("id", "")
        if not attachment_id or any(k in kv for k in ("machine", "note", "replaces")):
            return ParsedCommand(kind="unknown", raw="")
        return ParsedCommand(kind=name, evidence_args={
            "head": head.lower(), "round": round_id.lower(), "id": attachment_id,
        }, raw=raw)
    return ParsedCommand(kind="unknown", raw="")


def command_starts_line_any(text: str) -> bool:
    for line in _iter_top_level_command_lines(text):
        name = line.lstrip()[1:].split()[0].lower()
        if name in _KNOWN:
            return True
    return False
