"""GitHub REST API client."""

import logging

import httpx

logger = logging.getLogger(__name__)

# GitHub rejects a comment body over 65536 characters with a 422. Anything the
# agent posts is derived from build output or LLM text, so a length nobody
# planned for is always possible — and losing the whole comment to it is much
# worse than losing its tail. `comments.py` budgets logs well under this; the
# clamp is the backstop for everything else.
COMMENT_BODY_LIMIT = 65536


def _clamp_body(body: str) -> str:
    """Keep a comment postable, saying so if anything had to go."""
    if len(body) <= COMMENT_BODY_LIMIT:
        return body
    note = (
        "\n\n> :warning: This comment was truncated to fit GitHub's 65536-"
        "character limit. See the dashboard for the full output.\n"
    )
    logger.warning(
        f"Comment body {len(body)} chars exceeds GitHub's limit — truncating"
    )
    return body[: COMMENT_BODY_LIMIT - len(note)] + note


class GitHubClient:
    """Thin async wrapper around GitHub REST API."""

    def __init__(self, token: str):
        self._client = httpx.AsyncClient(
            base_url="https://api.github.com",
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30.0,
        )

    async def close(self):
        await self._client.aclose()

    async def get_pr(self, repo: str, pr_number: int) -> dict:
        """Get PR metadata (head SHA, refs, etc.)."""
        resp = await self._client.get(f"/repos/{repo}/pulls/{pr_number}")
        resp.raise_for_status()
        return resp.json()

    async def get_pr_files(self, repo: str, pr_number: int) -> list[dict]:
        """Get list of files changed in a PR."""
        resp = await self._client.get(
            f"/repos/{repo}/pulls/{pr_number}/files",
            params={"per_page": 100},
        )
        resp.raise_for_status()
        return resp.json()

    async def list_pr_comments(
        self, repo: str, pr_number: int, per_page: int = 100
    ) -> list[dict]:
        """Conversation comments on one PR, oldest first.

        The repo-wide `list_repo_comments` below is the poller's endpoint; this is
        the per-PR one, used to give the reviewer the discussion as it stands when
        the review runs.

        Line-level review comments live at `/pulls/{n}/comments` and are
        deliberately not fetched — same reason the trigger is not read there.
        """
        resp = await self._client.get(
            f"/repos/{repo}/issues/{pr_number}/comments",
            params={"per_page": per_page},
        )
        resp.raise_for_status()
        return resp.json()

    async def list_repo_comments(
        self,
        repo: str,
        since: str,
        page: int = 1,
        per_page: int = 100,
    ) -> list[dict]:
        """List issue comments across a whole repo, updated since `since`.

        This is the endpoint the poller uses. A PR is an issue underneath, so
        this returns PR conversation comments too — callers filter by looking
        for "/pull/" in each comment's html_url.

        Note `since` filters on updated_at, not created_at, so editing an old
        comment resurfaces it. The poller dedups by comment ID.

        Args:
            since: ISO8601 timestamp, e.g. "2026-08-15T10:00:00Z"
        """
        resp = await self._client.get(
            f"/repos/{repo}/issues/comments",
            params={
                "since": since,
                "sort": "created",
                "direction": "asc",
                "per_page": per_page,
                "page": page,
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def list_pulls(
        self,
        repo: str,
        state: str = "closed",
        sort: str = "updated",
        direction: str = "desc",
        per_page: int = 100,
    ) -> list[dict]:
        """List a repo's PRs — one call, not one per PR.

        Used to backfill merge commits: this endpoint carries `merge_commit_sha`
        and `merged_at` for every entry, so the whole backlog is covered by a
        single request per repo.

        Caller beware: on an *open* PR, `merge_commit_sha` is GitHub's throwaway
        test-merge sha and changes whenever the base moves. It is only the real
        merge commit when `merged_at` is non-null.
        """
        resp = await self._client.get(
            f"/repos/{repo}/pulls",
            params={
                "state": state,
                "sort": sort,
                "direction": direction,
                "per_page": per_page,
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def post_comment(self, repo: str, pr_number: int, body: str) -> int:
        """Post a comment on a PR. Returns the comment ID."""
        resp = await self._client.post(
            f"/repos/{repo}/issues/{pr_number}/comments",
            json={"body": _clamp_body(body)},
        )
        resp.raise_for_status()
        return resp.json()["id"]

    async def edit_comment(self, repo: str, comment_id: int, body: str):
        """Edit an existing comment."""
        resp = await self._client.patch(
            f"/repos/{repo}/issues/comments/{comment_id}",
            json={"body": _clamp_body(body)},
        )
        resp.raise_for_status()

    async def add_reaction(self, repo: str, comment_id: int, reaction: str):
        """Add a reaction to a comment (e.g. 'eyes', 'rocket', '+1')."""
        try:
            resp = await self._client.post(
                f"/repos/{repo}/issues/comments/{comment_id}/reactions",
                json={"content": reaction},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            # Non-critical, just log
            logger.warning(f"Failed to add reaction: {e}")

    async def get_diff(self, repo: str, pr_number: int) -> str:
        """Get the PR diff as text."""
        resp = await self._client.get(
            f"/repos/{repo}/pulls/{pr_number}",
            headers={"Accept": "application/vnd.github.diff"},
        )
        resp.raise_for_status()
        return resp.text
