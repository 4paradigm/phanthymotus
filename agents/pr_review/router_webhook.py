"""Webhook router — receives GitHub issue_comment events.

Optional. Polling is the default trigger and needs no inbound access; the
webhook exists so an instance that *can* be reached from the internet gets
instant triggering instead of waiting for the next poll. Both paths share
`trigger.create_job_from_comment`, so they dedup against each other.
"""

import hashlib
import hmac
import logging

from fastapi import APIRouter, Header, HTTPException, Request

from .config import Config
from .trigger import create_job_from_comment

logger = logging.getLogger(__name__)

router = APIRouter()


def _verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify GitHub's HMAC-SHA256 webhook signature."""
    if not secret:
        # No secret configured — accept unsigned deliveries (dev only).
        return True
    expected = "sha256=" + hmac.new(
        secret.encode(), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature or "")


@router.post("/webhook")
async def webhook(
    request: Request,
    x_github_event: str = Header(None, alias="X-GitHub-Event"),
    x_hub_signature_256: str = Header("", alias="X-Hub-Signature-256"),
):
    config: Config = request.app.state.config

    if not config.webhook_enabled:
        raise HTTPException(status_code=404, detail="Webhook disabled")

    body = await request.body()

    if not _verify_signature(body, x_hub_signature_256, config.github_webhook_secret):
        raise HTTPException(status_code=401, detail="Invalid signature")

    if x_github_event != "issue_comment":
        return {"status": "ignored", "reason": f"event={x_github_event}"}

    payload = await request.json()

    if payload.get("action") != "created":
        return {"status": "ignored", "reason": "not a new comment"}

    issue = payload.get("issue", {})
    if "pull_request" not in issue:
        return {"status": "ignored", "reason": "not a PR comment"}

    job = await create_job_from_comment(
        repo_full_name=payload["repository"]["full_name"],
        pr_number=issue["number"],
        comment_id=payload["comment"]["id"],
        comment_body=payload["comment"].get("body", ""),
        requester=payload["comment"]["user"]["login"],
        config=config,
        github_client=request.app.state.github_client,
        job_queue=request.app.state.job_queue,
        store=request.app.state.store,
        source="webhook",
    )

    if job is None:
        return {"status": "ignored", "reason": "no trigger, skipped, or PR closed"}

    return {"status": "queued", "job_id": job.id, "pr": job.pr_number}
