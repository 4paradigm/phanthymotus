"""What the PR's author and reviewers said, prepared for the review loop.

Everything here is **untrusted input**: the description and every comment are
written by whoever opened or commented on the PR, and the reviewer's output is
posted publicly and read by humans deciding whether to merge. This module's job
is to make that text useful without making it dangerous:

- filter out the agent's own comments, or the reviewer reads its previous reviews
  and anchors on them instead of on the code
- strip HTML comments, which are PR-template boilerplate and the obvious place to
  hide injected instructions
- bound the whole thing, so a wall-of-text body cannot crowd the rules out of the
  context window

The remaining defence is framing, not filtering, and lives in `review_agent`:
the rules stay in the system message, this text goes in a fenced user turn
labelled as claims to verify, and an attempt to instruct the reviewer is itself
reported as a finding.
"""

import logging
import re
from dataclasses import dataclass, field

from .comments import BOT_MARKER
from .models import TRIGGER

logger = logging.getLogger(__name__)

# `<!-- ... -->`, including multi-line. PR templates are full of them, and they
# are invisible in GitHub's rendered view — so text hidden there is text the
# author may not know is being fed to a reviewer.
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

# Markdown furniture that a template contributes but an author does not: ATX
# headings, list bullets, task boxes, table pipes, rules. What is left after
# removing these is the author's actual prose.
_FURNITURE = re.compile(r"(?m)^\s*(#{1,6}\s*|[-*+]\s*(\[[ xX]\]\s*)?|>\s*|\|.*\||-{3,}|_{3,})")

# Below this many characters of real prose, a description tells a reviewer
# nothing — an unfilled template renders as headings and empty checkboxes.
MIN_MEANINGFUL_CHARS = 30


@dataclass
class PRContext:
    """The PR's own account of itself, filtered and bounded."""

    title: str = ""
    description: str = ""
    comments: list[dict] = field(default_factory=list)  # {author, body}
    description_missing: bool = False
    comments_total: int = 0
    comments_dropped: int = 0

    @property
    def has_anything(self) -> bool:
        return bool(self.title or self.description or self.comments)


def build_pr_context(
    title: str,
    body: str,
    raw_comments: list[dict] | None = None,
    max_chars: int = 4000,
    max_comments: int = 20,
) -> PRContext:
    """Filter and bound the PR's description and conversation.

    `raw_comments` is the GitHub issue-comments payload, oldest first.
    """
    description = clean_body(body)
    ctx = PRContext(
        title=(title or "").strip(),
        description=description,
        description_missing=is_effectively_empty(description),
    )

    usable = [c for c in (raw_comments or []) if _is_useful_comment(c)]
    ctx.comments_total = len(usable)

    # Newest comments are the relevant ones — a 40-comment PR has moved on from
    # its first exchange — so fill the budget from the end and report the loss.
    budget = max(0, max_chars - len(description))
    kept: list[dict] = []
    for c in reversed(usable[-max_comments:] if max_comments > 0 else []):
        body_text = clean_body(c.get("body", ""))
        if not body_text:
            continue
        cost = len(body_text) + 40  # rough per-comment overhead
        if kept and cost > budget:
            break
        budget -= cost
        kept.append({
            "author": (c.get("user") or {}).get("login", "unknown"),
            "body": body_text[:max_chars],
        })
        if budget <= 0:
            break

    kept.reverse()
    ctx.comments = kept
    ctx.comments_dropped = ctx.comments_total - len(kept)
    return ctx


def clean_body(text: str) -> str:
    """Strip HTML comments and normalise whitespace."""
    if not text:
        return ""
    cleaned = _HTML_COMMENT.sub("", text)
    # Collapse the run of blank lines a stripped comment leaves behind.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def is_effectively_empty(description: str) -> bool:
    """Whether a description carries no information a reviewer can use.

    An unfilled template is not an empty string — it is headings and empty
    checkboxes — so the test is how much prose survives removing the furniture.
    """
    if not description.strip():
        return True
    prose = _FURNITURE.sub("", description)
    prose = re.sub(r"\s+", " ", prose).strip()
    return len(prose) < MIN_MEANINGFUL_CHARS


def _is_useful_comment(comment: dict) -> bool:
    body = (comment.get("body") or "").strip()
    if not body:
        return False
    # The agent's own output. Without this the reviewer reads its previous
    # reviews and build results, and anchors on them rather than the code.
    if BOT_MARKER in body:
        return False
    # A bare trigger command carries no review information.
    stripped = clean_body(body)
    if not stripped or stripped.lower().startswith(TRIGGER):
        return False
    return True
