"""GitHub webhook receiver for the Deploy Approval Agent (stateless).

Uses GitHub's HMAC-SHA256 signature. Even when triggered by a webhook, the
comment and author are re-read from the GitHub API. The command is dispatched
to the DeployController via the watcher's on_command callback.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging

from fastapi import APIRouter, HTTPException, Request

from .config import Config
from .github_state_proxy import HIDDEN_STATE_MARKER
from .commands import command_starts_line_any, parse_command

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/webhook")
async def webhook(request: Request):
    config: Config = request.app.state.config
    if not config.webhook_enabled:
        raise HTTPException(status_code=404, detail="Webhook disabled")

    x_github_event = request.headers.get("X-GitHub-Event", "")
    x_hub_signature_256 = request.headers.get("X-Hub-Signature-256", "")

    body = await _read_body_limited(
        request, config.max_response_bytes
    )
    if not _verify_signature_impl(
        body, x_hub_signature_256, config.github_webhook_secret
    ):
        raise HTTPException(status_code=401, detail="Invalid signature")

    if x_github_event != "issue_comment":
        return {"status": "ignored", "reason": f"event={x_github_event}"}

    try:
        payload = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="Invalid JSON")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object")
    if payload.get("action") != "created":
        return {"status": "ignored", "reason": "not created"}

    repository = payload.get("repository")
    issue = payload.get("issue")
    comment = payload.get("comment")
    if (
        not isinstance(repository, dict)
        or not isinstance(issue, dict)
        or not isinstance(comment, dict)
    ):
        raise HTTPException(status_code=400, detail="Missing webhook fields")

    if "pull_request" not in issue:
        return {"status": "ignored", "reason": "not_a_pr"}

    repo = repository.get("full_name")
    pr_number = issue.get("number")
    comment_id = comment.get("id")
    if (
        not isinstance(repo, str)
        or not repo
        or not isinstance(pr_number, int)
        or isinstance(pr_number, bool)
        or not isinstance(comment_id, int)
        or isinstance(comment_id, bool)
        or pr_number <= 0
        or comment_id <= 0
    ):
        raise HTTPException(
            status_code=400,
            detail="Malformed repo/pr_number/comment.id",
        )

    if repo not in (config.github_repos or []):
        logger.warning("webhook for un-allowlisted repo %r ignored", repo)
        raise HTTPException(status_code=404, detail="Repository not allowed")

    # Re-read comment from GitHub API (don't trust webhook body)
    proxy = request.app.state.proxy
    gh_comment = await proxy.get_comment(repo, comment_id)
    body = gh_comment.get("body", "")

    if not isinstance(body, str) or not body:
        return {"status": "ignored"}

    # Skip bot's own comments
    if HIDDEN_STATE_MARKER in body:
        return {"status": "ignored", "reason": "bot comment"}

    # Check if it's a command
    if not command_starts_line_any(body):
        return {"status": "ignored", "reason": "not a command"}

    # Parse and dispatch
    cmd = parse_command(body)
    if not cmd.is_command:
        return {"status": "unknown"}

    # Webhook does NOT dispatch deploy commands - deferred to GitHubCommandWatcher
    return {"status": "deferred", "reason": "processed by GitHubCommandWatcher"}


def _verify_signature_impl(payload, signature, secret):
    if not secret:
        return False
    expected = "sha256=" + hmac.new(
        secret.encode() if isinstance(secret, str) else secret, payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature or "")


async def _read_body_limited(request, limit: int) -> bytes:
    chunks = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail="Webhook too large")
        chunks.append(chunk)
    return b"".join(chunks)
