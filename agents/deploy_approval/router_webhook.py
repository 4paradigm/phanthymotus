"""GitHub webhook receiver for the Deploy Approval Agent (stateless).

Uses GitHub's HMAC-SHA256 signature. Webhook payloads are routing hints only;
the single watcher re-reads PRs, comments, identities and lifecycle state
before dispatching any command.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
from collections import deque

from fastapi import APIRouter, HTTPException, Request

from .config import Config

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/webhook")
async def webhook(request: Request):
    config: Config = request.app.state.config
    if not config.webhook_enabled:
        raise HTTPException(status_code=404, detail="Webhook disabled")

    x_github_event = request.headers.get("X-GitHub-Event", "")
    delivery_id = request.headers.get("X-GitHub-Delivery", "")
    x_hub_signature_256 = request.headers.get("X-Hub-Signature-256", "")

    body = await _read_body_limited(
        request, config.max_response_bytes
    )
    if not _verify_signature_impl(
        body, x_hub_signature_256, config.github_webhook_secret
    ):
        raise HTTPException(status_code=401, detail="Invalid signature")

    if x_github_event not in {
        "issue_comment", "pull_request", "pull_request_review",
        "pull_request_review_comment",
    }:
        return {"status": "ignored", "reason": f"event={x_github_event}"}

    try:
        payload = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="Invalid JSON")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object")
    repository = payload.get("repository")
    if not isinstance(repository, dict):
        raise HTTPException(status_code=400, detail="Missing webhook fields")
    repo = repository.get("full_name")
    if x_github_event == "issue_comment":
        issue = payload.get("issue")
        comment = payload.get("comment")
        if not isinstance(issue, dict) or not isinstance(comment, dict):
            raise HTTPException(status_code=400, detail="Missing webhook fields")
        if payload.get("action") != "created":
            return {"status": "ignored", "reason": "not created"}
        if "pull_request" not in issue:
            return {"status": "ignored", "reason": "not_a_pr"}
        pr_number = issue.get("number")
        comment_id = comment.get("id")
        if (not isinstance(comment_id, int) or isinstance(comment_id, bool) or comment_id <= 0):
            raise HTTPException(status_code=400, detail="Malformed comment.id")
    elif x_github_event == "pull_request":
        action = payload.get("action")
        if action not in {
            "opened", "reopened", "synchronize", "ready_for_review",
            "closed", "converted_to_draft",
        }:
            return {"status": "ignored", "reason": "unsupported pull_request action"}
        pull_request = payload.get("pull_request")
        if not isinstance(pull_request, dict):
            raise HTTPException(status_code=400, detail="Missing pull_request")
        pr_number = pull_request.get("number")
        if not pr_number:
            pr_number = payload.get("number")
    else:
        expected_action = "submitted" if x_github_event == "pull_request_review" else "created"
        if payload.get("action") != expected_action:
            return {"status": "ignored", "reason": "unsupported review action"}
        pull_request = payload.get("pull_request")
        if not isinstance(pull_request, dict):
            raise HTTPException(status_code=400, detail="Missing pull_request")
        pr_number = pull_request.get("number")
    if (not isinstance(repo, str) or not repo
            or not isinstance(pr_number, int) or isinstance(pr_number, bool) or pr_number <= 0):
        raise HTTPException(status_code=400, detail="Malformed repo/pr_number")

    # Webhook mutation gate: use runtime-authorized active_repos,
    # NOT the static DESIRED_REPOS (config.github_repos).
    # Also require the auth snapshot to be valid.
    if not config.active_repos or not getattr(config, "auth_valid", True):
        logger.warning(
            "webhook for repo %r ignored: no active repos or auth invalid",
            repo,
        )
        raise HTTPException(status_code=403, detail="Authorization not available")
    if repo not in config.active_repos or repo not in config.github_repos:
        logger.warning("webhook for unauthorized repo %r ignored", repo)
        raise HTTPException(status_code=404, detail="Repository not allowed")

    # The signed event is only a routing hint. The single watcher re-fetches
    # PR, comments, identities and lifecycle state before any processing.
    if not delivery_id:
        raise HTTPException(status_code=400, detail="Missing delivery id")
    if delivery_id:
        if not re.fullmatch(r"[A-Za-z0-9-]{1,128}", delivery_id):
            raise HTTPException(status_code=400, detail="Malformed delivery id")
        deliveries = getattr(request.app.state, "webhook_deliveries", None)
        delivery_set = getattr(request.app.state, "webhook_delivery_set", None)
        if deliveries is None or delivery_set is None:
            deliveries = deque()
            delivery_set = set()
            request.app.state.webhook_deliveries = deliveries
            request.app.state.webhook_delivery_set = delivery_set
        if delivery_id in delivery_set:
            return {"status": "duplicate"}
    try:
        enqueued = request.app.state.watcher.enqueue_pr(repo, pr_number)
    except Exception as exc:
        logger.warning(
            "webhook enqueue failed repo=%s pr=%s exception=%s",
            repo, pr_number, type(exc).__name__,
        )
        raise HTTPException(status_code=503, detail="Watcher unavailable") from exc
    if not enqueued:
        raise HTTPException(status_code=403, detail="Repository authorization unavailable")
    if delivery_id:
        if len(deliveries) >= 4096:
            delivery_set.discard(deliveries.popleft())
        deliveries.append(delivery_id)
        delivery_set.add(delivery_id)
    return {"status": "queued"}


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
