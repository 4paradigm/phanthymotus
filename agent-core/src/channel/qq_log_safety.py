"""Safe, bounded logging of user-derived content in channel turns.

Private QQ messages are never printed verbatim. Other channel summaries
escape control characters to prevent forged log lines.
"""


def safe_user_log_preview(content: object, *, qq_turn: bool) -> str:
    if qq_turn:
        return "[QQ user content redacted]"
    if not isinstance(content, str):
        return "[non-text content]"
    return content.encode("unicode_escape", errors="backslashreplace").decode("ascii")[:200]
