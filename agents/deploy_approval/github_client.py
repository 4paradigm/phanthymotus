"""GitHub API client for comments, PR state and approver identity.

Only trusts the GitHub REST API after re-reading it (webhook payloads are used
solely as a hint; comment text and author are re-fetched). Team membership is
checked via the documented API when the approval source needs it.
"""

from __future__ import annotations

from dataclasses import dataclass

import logging
import random
import re
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx

from .config import Config, validate_repository_names
from .clients_common import (
    SecurityError,
    enforce_body_size,
    require_2xx,
    require_http_policy,
    stream_request,
)

logger = logging.getLogger(__name__)

@dataclass
class PrSnapshot:
    """A strict, fail-closed snapshot of a PR's deploy-relevant fields.

    ``state`` is exactly ``"open"`` or ``"closed"``. ``merged``/``merged_at``/
    ``merge_commit_sha``/``head_sha`` are only the GitHub-verified values.
    ``author_login``/``author_id`` are the PR author's GitHub identity from the
    SAME fetched PR (``user.login`` + immutable numeric ``user.id``) and are
    used as the auto-discovered candidate's requester identity (self-approval
    gate). When any field is missing, the wrong type, or the state is unknown,
    ``pr_state`` returns None so the caller fails closed (never a candidate).
    """
    state: str
    head_sha: str
    merged: bool
    merged_at: str
    merge_commit_sha: str
    author_login: str = ""
    author_id: str = ""


class GitHubError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class GitHubRateLimitError(GitHubError):
    """GitHub primary or secondary rate limit; authorization remains valid."""

    def __init__(self, *, status_code: int, retry_after_seconds: float):
        super().__init__("GitHub API rate limited", status_code=status_code)
        self.retry_after_seconds = max(0.0, retry_after_seconds)


class GitHubPermissionError(GitHubError):
    """GitHub rejected the installation's authorization."""


class GitHubTransientError(GitHubError):
    """Retryable GitHub 5xx or network failure with bounded backoff."""

    def __init__(self, *, status_code: int | None, retry_after_seconds: float):
        super().__init__("GitHub API temporarily unavailable", status_code=status_code)
        self.retry_after_seconds = max(0.0, retry_after_seconds)


class GitHubRefConflict(GitHubError):
    """A Git reference changed between read and compare-and-swap update."""


class GitHubClient:
    def __init__(self, config: Config, http: httpx.AsyncClient | None = None, token_provider: callable = None):
        self.config = config
        self._token_provider = token_provider
        self.http = http or httpx.AsyncClient(
            timeout=httpx.Timeout(
                config.total_timeout,
                connect=config.connect_timeout,
                read=config.read_timeout,
                pool=config.connect_timeout,
            ),
            follow_redirects=False,
            trust_env=False,
        )
        self._rate_limit_until = 0.0
        self._rate_limit_failures = 0
        self._transient_until = 0.0
        self._transient_failures = 0
        self._request_count = 0
        self._request_counts_by_repo: dict[str, int] = {}
        self._status_counts: dict[str, int] = {}
        self._recent_rate_limits: list[dict] = []
        self._recent_api_errors: list[dict] = []
        self._open_pr_pages: dict[tuple[str, int], dict] = {}

    def diagnostics(self) -> dict:
        return {
            "request_count": self._request_count,
            "request_counts_by_repo": dict(self._request_counts_by_repo),
            "status_counts": dict(self._status_counts),
            "recent_rate_limits": [dict(item) for item in self._recent_rate_limits],
            "recent_api_errors": [dict(item) for item in self._recent_api_errors],
            "rate_limited": self._rate_limit_until > time.monotonic(),
            "backing_off": max(self._rate_limit_until, self._transient_until) > time.monotonic(),
        }

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Unified async per-request GitHub API call with fresh installation token.

        Each request obtains a fresh token from ``self._token_provider()``.
        No long-lived Authorization header is mutated on the client.

        Accepts both relative paths (e.g. ``/repos/...``) and absolute URLs.
        Absolute URLs are passed through verbatim (no double-prefixing).
        """
        url = self._resolve_api_url(path)
        now = time.monotonic()
        if self._rate_limit_until > now:
            raise GitHubRateLimitError(
                status_code=429,
                retry_after_seconds=self._rate_limit_until - now,
            )
        if self._transient_until > now:
            raise GitHubTransientError(
                status_code=None,
                retry_after_seconds=self._transient_until - now,
            )
        mutating = method.upper() not in ("GET", "HEAD", "OPTIONS")
        if mutating:
            self._require_write_authorization(url)
        if self._token_provider is None:
            raise GitHubError("No token provider configured for GitHubClient")
        try:
            token = await self._token_provider()
        except Exception as exc:
            self.config.auth_valid = False
            raise GitHubError("GitHub App authorization unavailable") from exc
        if not isinstance(token, str) or not token.strip():
            self.config.auth_valid = False
            raise GitHubError("GitHub App authorization unavailable")
        # Token acquisition is asynchronous; recheck the current authorization
        # snapshot immediately before any request with a side effect.
        if mutating:
            self._require_write_authorization(url)
        headers = dict(kwargs.pop("headers", {}) or {})
        headers["Authorization"] = "Bearer " + token
        headers.setdefault("Accept", "application/vnd.github+json")
        headers.setdefault("X-GitHub-Api-Version", "2026-03-10")
        require_http_policy(url, self.config, allow_private=self.config.allow_private_http)
        kwargs.setdefault("timeout", self.config.total_timeout)
        self._request_count += 1
        path_parts = urlparse(url).path.split("/")
        repo_name = "/".join(path_parts[2:4]) if len(path_parts) >= 5 and path_parts[1] == "repos" else ""
        if repo_name:
            self._request_counts_by_repo[repo_name] = self._request_counts_by_repo.get(repo_name, 0) + 1
        try:
            resp = await stream_request(
                self.http, method, url, self.config.max_response_bytes,
                headers=headers, **kwargs)
        except (httpx.HTTPError, SecurityError) as exc:
            is_timeout = isinstance(exc, httpx.TimeoutException) or (
                isinstance(exc, SecurityError) and "timeout" in str(exc).lower()
            )
            if not isinstance(exc, httpx.HTTPError) and not is_timeout:
                raise GitHubError("GitHub response rejected by transport policy") from exc
            retry = self._arm_transient_backoff()
            self._recent_api_errors.append({
                "status": "transport_error", "repo": repo_name,
                "retry_after_seconds": round(retry, 1),
                "at": datetime.now(timezone.utc).isoformat(),
            })
            self._recent_api_errors = self._recent_api_errors[-20:]
            raise GitHubTransientError(status_code=None, retry_after_seconds=retry) from exc
        self._status_counts[str(resp.status_code)] = self._status_counts.get(str(resp.status_code), 0) + 1
        # Label permissions are optional UI projection; a 403 there must not
        # disable the controller. Invalid credentials always close the gate.
        label_projection = len(path_parts) >= 5 and path_parts[1] == "repos" and (
            path_parts[4] == "labels" or (
                len(path_parts) >= 7 and path_parts[4] == "issues" and path_parts[6] == "labels"))
        recent_issue_comment_discovery = (
            method.upper() == "GET"
            and re.search(r"/repos/[^/]+/[^/]+/issues/comments$", urlparse(url).path) is not None
        )
        # Git Data API 403s commonly indicate missing repository contents
        # permission. They still fail the operation, but do not invalidate an
        # otherwise usable installation token; 401 remains an auth failure.
        repo_index = next((index for index, part in enumerate(path_parts[:-3])
                           if part == "repos"), -1)
        git_data_api = (repo_index >= 0 and len(path_parts) > repo_index + 3
                        and path_parts[repo_index + 3] == "git")
        error_message = ""
        if resp.status_code in (403, 429):
            try:
                payload = resp.json()
                if isinstance(payload, dict) and isinstance(payload.get("message"), str):
                    error_message = payload["message"].lower()
            except (ValueError, TypeError):
                pass
        rate_limited = resp.status_code == 429 or (
            resp.status_code == 403 and (
                resp.headers.get("Retry-After") is not None
                or resp.headers.get("X-RateLimit-Remaining") == "0"
                or "secondary rate limit" in error_message
                or "rate limit exceeded" in error_message
                or "abuse detection" in error_message
                or "temporarily blocked" in error_message
            )
        )
        if rate_limited:
            self._rate_limit_failures += 1
            delay = 0.0
            server_deadline = False
            retry_after = resp.headers.get("Retry-After")
            try:
                delay = max(0.0, float(retry_after))
                server_deadline = delay > 0
            except (TypeError, ValueError):
                if retry_after:
                    try:
                        retry_at = datetime.strptime(
                            retry_after, "%a, %d %b %Y %H:%M:%S %Z"
                        ).replace(tzinfo=timezone.utc).timestamp()
                        delay = max(0.0, retry_at - time.time())
                        server_deadline = delay > 0
                    except ValueError:
                        pass
            if delay <= 0:
                reset = resp.headers.get("X-RateLimit-Reset")
                try:
                    delay = max(0.0, float(reset) - time.time())
                    server_deadline = delay > 0
                except (TypeError, ValueError):
                    delay = min(900.0, 30.0 * (2 ** min(self._rate_limit_failures - 1, 5)))
            if not server_deadline:
                delay = min(900.0, max(delay, 1.0) * random.uniform(0.9, 1.1))
                delay = min(3600.0, delay)
            else:
                delay = max(delay, 1.0)
            self._rate_limit_until = max(self._rate_limit_until, time.monotonic() + delay)
            self._recent_rate_limits.append({
                "status": resp.status_code,
                "repo": repo_name,
                "retry_after_seconds": round(delay, 1),
                "at": datetime.now(timezone.utc).isoformat(),
            })
            self._recent_rate_limits = self._recent_rate_limits[-20:]
            raise GitHubRateLimitError(
                status_code=resp.status_code, retry_after_seconds=delay,
            )
        if resp.status_code >= 500:
            retry = self._arm_transient_backoff()
            self._recent_api_errors.append({
                "status": resp.status_code, "repo": repo_name,
                "retry_after_seconds": round(retry, 1),
                "at": datetime.now(timezone.utc).isoformat(),
            })
            self._recent_api_errors = self._recent_api_errors[-20:]
            raise GitHubTransientError(status_code=resp.status_code, retry_after_seconds=retry)
        if 200 <= resp.status_code < 300 or resp.status_code == 304:
            self._rate_limit_failures = 0
            self._transient_failures = 0
        if resp.status_code == 401:
            self.config.auth_valid = False
            self._recent_api_errors.append({
                "status": 401, "repo": repo_name, "reason": "authentication_rejected",
                "at": datetime.now(timezone.utc).isoformat(),
            })
            self._recent_api_errors = self._recent_api_errors[-20:]
            raise GitHubPermissionError("GitHub authentication rejected", status_code=401)
        if (resp.status_code == 403 and not label_projection and not git_data_api
                and not recent_issue_comment_discovery):
            self.config.auth_valid = False
            self._recent_api_errors.append({
                "status": 403, "repo": repo_name, "reason": "installation_permission_rejected",
                "at": datetime.now(timezone.utc).isoformat(),
            })
            self._recent_api_errors = self._recent_api_errors[-20:]
            raise GitHubPermissionError("GitHub installation permission rejected", status_code=403)
        return resp

    def _arm_transient_backoff(self) -> float:
        self._transient_failures += 1
        delay = min(120.0, 2.0 * (2 ** min(self._transient_failures - 1, 6)))
        delay *= random.uniform(0.8, 1.2)
        self._transient_until = max(self._transient_until, time.monotonic() + delay)
        return delay

    def _require_write_authorization(self, url: str) -> None:
        pieces = urlparse(url).path.split("/")
        repo = "/".join(pieces[2:4]).lower() if len(pieces) >= 5 and pieces[1] == "repos" else ""
        try:
            controlled = validate_repository_names(self.config.github_repos)
        except ValueError as exc:
            raise GitHubError("GitHub repository configuration is invalid") from exc
        if (self.config.auth_valid is not True or not repo
                or repo not in controlled or repo not in self.config.active_repos):
            raise GitHubError("GitHub write requires current installation authorization for a configured repository")

    def _resolve_api_url(self, path: str) -> str:
        """Resolve a path to a full URL without double-prefixing.

        - Relative paths (starting with /) are joined with github_api_url.
        - Absolute URLs are returned as-is after origin validation.
        """
        from urllib.parse import urlparse
        parsed = urlparse(path)
        if parsed.scheme and parsed.netloc:
            # Absolute URL — validate origin
            config_origin = self.config.github_api_url.rstrip("/")
            url_origin = parsed.scheme + "://" + parsed.netloc.rstrip("/")
            if url_origin != config_origin:
                raise GitHubError(
                    f"absolute URL origin {url_origin!r} does not match "
                    f"configured github_api_url origin {config_origin!r}"
                )
            if parsed.username or parsed.password:
                raise GitHubError("absolute URL must not contain userinfo")
            return path
        # Relative path — join with configured API URL
        return self.api(path.lstrip("/"))

    def api(self, path: str) -> str:
        return self.config.github_api_url.rstrip("/") + "/" + path.lstrip("/")

    async def _read_json(self, resp: httpx.Response) -> dict:
        resp = await enforce_body_size(resp, self.config.max_response_bytes)
        data = resp.json()
        if not isinstance(data, dict):
            raise GitHubError("github returned a non-object payload")
        return data

    async def get_public_git_data(
        self, repo: str, path: str, *, params: dict | None = None
    ) -> dict:
        """Read a narrowly allowlisted Git Data object without credentials.

        Public Git Notes are a read-only fallback. This deliberately does not
        use ``_request`` because that method always obtains and sends an App
        token, and it does not change the behavior of authenticated API calls.
        """
        if not isinstance(repo, str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo
        ):
            raise GitHubError("GitHub public data repository is invalid")
        try:
            configured = validate_repository_names(self.config.github_repos)
        except ValueError as exc:
            raise GitHubError("GitHub repository configuration is invalid") from exc
        authorized = {name.lower() for name in configured}
        active = {name.lower() for name in self.config.active_repos}
        if repo.lower() not in authorized or repo.lower() not in active:
            raise GitHubPermissionError(
                "GitHub public data repository is not authorized", status_code=403
            )

        api = urlparse(self.config.github_api_url)
        if (api.scheme != "https" or api.hostname != "api.github.com"
                or api.port not in (None, 443) or api.username or api.password
                or api.path not in ("", "/") or api.query or api.fragment):
            raise GitHubError("GitHub public data requires the official HTTPS API")

        if not isinstance(path, str) or "\\" in path or "%" in path:
            raise GitHubError("GitHub public data path is invalid")
        is_note_ref = re.fullmatch(
            r"git/ref/notes/deploy-approval-evidence-v1/pr-([1-9][0-9]*)", path
        )
        is_git_object = re.fullmatch(r"git/(commits|trees|blobs)/[0-9a-f]{40}", path)
        if not (is_note_ref or is_git_object):
            raise GitHubError("GitHub public data path is not allowlisted")
        if params is not None:
            recursive = params.get("recursive") if isinstance(params, dict) else None
            valid_recursive = (
                recursive is True or recursive is False
                or (type(recursive) is int and recursive in (0, 1))
                or recursive in ("0", "1")
            )
            if (not isinstance(params, dict) or not is_git_object
                    or is_git_object.group(1) != "trees"
                    or set(params) != {"recursive"}
                    or not valid_recursive):
                raise GitHubError("GitHub public data parameters are invalid")
            recursive_enabled = (
                recursive is True or recursive == 1 or recursive == "1"
            )
            params = {"recursive": "1" if recursive_enabled else "0"}

        # Do not let an injected client silently attach ambient credentials or
        # read proxy/TLS settings from the process environment.
        if (getattr(self.http, "_trust_env", True) is not False
                or getattr(self.http, "auth", None) is not None
                or "authorization" in getattr(self.http, "headers", {})
                or bool(getattr(self.http, "cookies", {}))):
            raise GitHubError("GitHub public data requires an anonymous HTTP client")

        base = self.config.github_api_url.rstrip("/")
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10",
        }

        async def fetch_json(url_path: str, *, query: dict | None = None) -> dict:
            url = base + url_path
            try:
                require_http_policy(url, self.config, allow_private=False)
                response = await stream_request(
                    self.http, "GET", url, self.config.max_response_bytes,
                    headers=headers, params=query,
                    timeout=self.config.total_timeout,
                )
            except (httpx.HTTPError, SecurityError) as exc:
                is_timeout = isinstance(exc, httpx.TimeoutException) or (
                    isinstance(exc, SecurityError) and "timeout" in str(exc).lower()
                )
                if isinstance(exc, httpx.HTTPError) or is_timeout:
                    retry = self._arm_transient_backoff()
                    raise GitHubTransientError(
                        status_code=None, retry_after_seconds=retry
                    ) from exc
                raise GitHubError("GitHub public response rejected by transport policy") from exc

            if response.status_code == 404:
                raise GitHubError("GitHub public data was not found", status_code=404)
            if response.status_code in (403, 429):
                if (response.status_code == 429
                        or response.headers.get("Retry-After") is not None
                        or response.headers.get("X-RateLimit-Remaining") == "0"):
                    retry = self._arm_rate_limit_backoff(response)
                    raise GitHubRateLimitError(
                        status_code=response.status_code, retry_after_seconds=retry
                    )
                raise GitHubPermissionError(
                    "GitHub denied anonymous public data access", status_code=403
                )
            if response.status_code >= 500:
                retry = self._arm_transient_backoff()
                raise GitHubTransientError(
                    status_code=response.status_code, retry_after_seconds=retry
                )
            if response.status_code != 200:
                raise GitHubError(
                    "GitHub public data request failed", status_code=response.status_code
                )
            return await self._read_json(response)

        metadata = await fetch_json(f"/repos/{repo}")
        if metadata.get("full_name") != repo or metadata.get("private") is not False:
            raise GitHubPermissionError(
                "GitHub repository is not the requested public repository", status_code=403
            )
        return await fetch_json(f"/repos/{repo}/{path}", query=params)

    def _arm_rate_limit_backoff(self, response: httpx.Response) -> float:
        """Apply the existing bounded rate-limit delay for anonymous reads."""
        retry_after = response.headers.get("Retry-After")
        try:
            delay = float(retry_after) if retry_after else 0.0
        except (TypeError, ValueError):
            delay = 0.0
        if delay <= 0:
            reset = response.headers.get("X-RateLimit-Reset")
            try:
                delay = max(0.0, float(reset) - time.time()) if reset else 0.0
            except (TypeError, ValueError):
                delay = 0.0
        if delay <= 0:
            self._rate_limit_failures += 1
            delay = min(900.0, 30.0 * (2 ** min(self._rate_limit_failures - 1, 5)))
            delay *= random.uniform(0.9, 1.1)
        delay = min(3600.0, max(1.0, delay))
        self._rate_limit_until = max(self._rate_limit_until, time.monotonic() + delay)
        return delay

    async def _read_json_list(self, resp: httpx.Response) -> list:
        resp = await enforce_body_size(resp, self.config.max_response_bytes)
        data = resp.json()
        if not isinstance(data, list):
            raise GitHubError("github returned a non-array payload")
        return data

    async def _git_data_json(self, method: str, path: str, action: str,
                             *, ref_update: bool = False, **kwargs) -> dict:
        resp = await self._request(method, path, timeout=self.config.total_timeout, **kwargs)
        if not 200 <= resp.status_code < 300:
            error = GitHubRefConflict if ref_update and resp.status_code in (409, 422) else GitHubError
            raise error(f"github {action} failed with HTTP {resp.status_code}",
                        status_code=resp.status_code)
        return await self._read_json(resp)

    async def get_git_ref(self, repo: str, ref: str) -> dict:
        if not isinstance(ref, str) or not ref or ref.startswith("refs/"):
            raise GitHubError("github git ref name is invalid")
        return await self._git_data_json("GET", f"/repos/{repo}/git/ref/{ref}",
                                         "get git ref")

    async def get_git_commit(self, repo: str, sha: str) -> dict:
        return await self._git_data_json("GET", f"/repos/{repo}/git/commits/{sha}",
                                         "get git commit")

    async def get_git_tree(self, repo: str, sha: str, *, recursive: bool = False) -> dict:
        params = {"recursive": "1"} if recursive else None
        return await self._git_data_json("GET", f"/repos/{repo}/git/trees/{sha}",
                                         "get git tree", params=params)

    async def get_git_blob(self, repo: str, sha: str) -> dict:
        return await self._git_data_json("GET", f"/repos/{repo}/git/blobs/{sha}",
                                         "get git blob")

    async def create_git_blob(self, repo: str, content: str) -> dict:
        return await self._git_data_json(
            "POST", f"/repos/{repo}/git/blobs", "create git blob",
            json={"content": content, "encoding": "utf-8"})

    async def create_git_tree(self, repo: str, tree: list[dict], *, base_tree: str | None = None) -> dict:
        body = {"tree": tree}
        if base_tree:
            body["base_tree"] = base_tree
        return await self._git_data_json("POST", f"/repos/{repo}/git/trees",
                                         "create git tree", json=body)

    async def create_git_commit(self, repo: str, message: str, tree: str,
                                parents: list[str]) -> dict:
        return await self._git_data_json(
            "POST", f"/repos/{repo}/git/commits", "create git commit",
            json={"message": message, "tree": tree, "parents": parents})

    async def create_git_ref(self, repo: str, ref: str, sha: str) -> dict:
        return await self._git_data_json(
            "POST", f"/repos/{repo}/git/refs", "create git ref", ref_update=True,
            json={"ref": "refs/" + ref, "sha": sha})

    async def update_git_ref(self, repo: str, ref: str, sha: str) -> dict:
        return await self._git_data_json(
            "PATCH", f"/repos/{repo}/git/refs/{ref}", "update git ref",
            ref_update=True, json={"sha": sha, "force": False})

    async def get_pr(self, repo: str, pr_number: int) -> dict:
        path = f"/repos/{repo}/pulls/{pr_number}"
        resp = await self._request("GET", path, timeout=self.config.total_timeout)
        try:
            require_2xx(resp.status_code, "github get PR")
        except SecurityError as e:
            raise GitHubError(str(e), status_code=resp.status_code) from e
        return await self._read_json(resp)

    async def get_issue_comments(self, repo: str, pr_number: int) -> list[dict]:
        """All comments in the PR main conversation, with a HARD bound.

        The aggregate is capped by an explicit maximum number of pages, an
        explicit maximum number of comments and a maximum accumulated JSON size
        (the raw response byte cap already applies per page). Beyond any cap the
        call is refused fail-closed (never an unlimited in-memory scan).
        """
        path = f"/repos/{repo}/issues/{pr_number}/comments"
        url = self.api(path)
        require_http_policy(
            url, self.config, allow_private=self.config.allow_private_http
        )
        comments: list[dict] = []
        page = 1
        max_pages = getattr(self.config, "github_comment_max_pages", 20)
        max_comments = getattr(self.config, "github_comment_max_comments", 500)
        max_bytes = getattr(self.config, "github_comment_max_bytes", 4 * 1024 * 1024)
        seen_bytes = 0
        while True:
            if page > max_pages:
                raise GitHubError(
                    f"github comment pagination exceeded {max_pages} pages "
                    f"for {repo}#{pr_number}"
                )
            resp = await self._request(
                "GET", url,
                params={"per_page": 100, "page": page}, timeout=self.config.total_timeout)
            try:
                require_2xx(resp.status_code, "github list comments")
            except SecurityError as e:
                raise GitHubError(str(e)) from e
            batch = await self._read_json_list(resp)
            import json as _json
            for c in batch:
                if len(comments) >= max_comments:
                    raise GitHubError(
                        f"github comment count exceeded {max_comments} for "
                        f"{repo}#{pr_number}"
                    )
                payload = _json.dumps(c, sort_keys=True).encode("utf-8")
                seen_bytes += len(payload)
                if seen_bytes > max_bytes:
                    raise GitHubError(
                        f"github comment aggregation exceeded {max_bytes} bytes "
                        f"for {repo}#{pr_number}"
                    )
                comments.append(c)
            if len(batch) < 100:
                break
            page += 1
        return comments

    async def list_recent_issue_comments(self, repo: str, *, limit: int = 100) -> list[dict]:
        """Read a bounded repo-wide window of recently-created issue comments.

        Results are discovery hints only. The watcher must re-read the PR and
        each candidate comment through its existing authorization path before
        dispatch. A 403/404 on this optional endpoint does not invalidate App
        authorization for the rest of the client.
        """
        if (not isinstance(repo, str) or not re.fullmatch(
                r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo)
                or type(limit) is not int or not 1 <= limit <= 100):
            raise GitHubError("recent issue comment request is invalid")
        path = f"/repos/{repo}/issues/comments"
        resp = await self._request(
            "GET", path,
            params={"sort": "created", "direction": "desc", "per_page": limit, "page": 1},
            timeout=self.config.total_timeout,
        )
        if resp.status_code in (403, 404):
            raise GitHubError(
                "recent issue comment discovery is unavailable",
                status_code=resp.status_code,
            )
        try:
            require_2xx(resp.status_code, "github list recent issue comments")
        except SecurityError as exc:
            raise GitHubError(str(exc)) from exc
        return await self._read_json_list(resp)

    async def list_open_prs(self, repo: str) -> list[dict]:
        """List all open PRs for a repo, newest-first, with page overlap dedupe."""
        seen: dict[int, dict] = {}
        page = 1
        while True:
            path = f"/repos/{repo}/pulls"
            url = self.api(path)
            require_http_policy(
                url, self.config,
                allow_private=self.config.allow_private_http
            )
            cached = self._open_pr_pages.get((repo, page))
            request_headers = {}
            if isinstance(cached, dict):
                if cached.get("etag"):
                    request_headers["If-None-Match"] = cached["etag"]
                elif cached.get("last_modified"):
                    request_headers["If-Modified-Since"] = cached["last_modified"]
            resp = await self._request(
                "GET", path,
                params={
                    "state": "open",
                    "sort": "updated",
                    "direction": "desc",
                    "per_page": 100,
                    "page": page,
                }, headers=request_headers, timeout=self.config.total_timeout)
            if resp.status_code == 304:
                if cached is None:
                    # A 304 without a local page is unusable; retry once without
                    # validators instead of treating it as an empty PR page.
                    resp = await self._request(
                        "GET", path,
                        params={
                            "state": "open", "sort": "updated", "direction": "desc",
                            "per_page": 100, "page": page,
                        }, timeout=self.config.total_timeout)
                else:
                    batch = cached.get("items")
                    if not isinstance(batch, list):
                        raise GitHubError("github cached open PR page is malformed")
                    if not batch:
                        break
                    for pr in batch:
                        if isinstance(pr, dict):
                            num = pr.get("number")
                            if isinstance(num, int) and not isinstance(num, bool) and num > 0:
                                seen[num] = pr
                    if len(batch) < 100:
                        break
                    page += 1
                    continue
            try:
                require_2xx(resp.status_code, "github list open PRs")
            except SecurityError as e:
                raise GitHubError(str(e)) from e
            batch = await self._read_json_list(resp)
            self._open_pr_pages[(repo, page)] = {
                "items": batch,
                "etag": resp.headers.get("ETag", ""),
                "last_modified": resp.headers.get("Last-Modified", ""),
            }
            if not batch:
                break
            for pr in batch:
                if not isinstance(pr, dict):
                    continue
                num = pr.get("number")
                if isinstance(num, bool) or not isinstance(num, int) or num <= 0:
                    continue
                seen[num] = pr
            if len(batch) < 100:
                break
            page += 1

        def _sort_key(pr: dict):
            updated = pr.get("updated_at")
            if not isinstance(updated, str):
                updated = ""
            try:
                from datetime import datetime, timezone

                ts = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                return (1, ts.timestamp(), updated)
            except (TypeError, ValueError):
                return (0, 0.0, updated)

        return sorted(seen.values(), key=_sort_key, reverse=True)

    async def get_comment(self, repo: str, comment_id: int) -> dict:
        path = f"/repos/{repo}/issues/comments/{comment_id}"
        resp = await self._request("GET", path, timeout=self.config.total_timeout)
        try:
            require_2xx(resp.status_code, "github get comment")
        except SecurityError as e:
            raise GitHubError(str(e)) from e
        return await self._read_json(resp)

    async def post_issue_comment(
        self, repo: str, pr_number: int, body: str
    ) -> dict:
        path = f"/repos/{repo}/issues/{pr_number}/comments"
        resp = await self._request("POST", path, json={"body": body}, timeout=self.config.total_timeout)
        try:
            require_2xx(resp.status_code, "github post comment")
        except SecurityError as e:
            raise GitHubError(str(e)) from e
        return await self._read_json(resp)

    async def update_comment(self, repo: str, comment_id: int, body: str):
        path = f"/repos/{repo}/issues/comments/{comment_id}"
        resp = await self._request("PATCH", path, json={"body": body}, timeout=self.config.total_timeout)
        try:
            require_2xx(resp.status_code, "github update comment")
        except SecurityError as e:
            raise GitHubError(str(e)) from e

    async def get_issue_labels(self, repo: str, issue_number: int) -> list[str]:
        """Get the current labels for an issue/PR."""
        url = self.api(f"/repos/{repo}/issues/{issue_number}/labels")
        require_http_policy(
            url, self.config, allow_private=self.config.allow_private_http
        )
        resp = await self._request(
            "GET", f"/repos/{repo}/issues/{issue_number}/labels",
            timeout=self.config.total_timeout)
        try:
            require_2xx(resp.status_code, "github get labels")
        except SecurityError as e:
            raise GitHubError(str(e)) from e
        data = await self._read_json_list(resp)
        return [l.get("name", "") for l in data if isinstance(l, dict) and l.get("name")]


    async def add_issue_label(self, repo: str, issue_number: int,
                               label: str) -> None:
        """Add a single label to an issue/PR."""
        path = f"/repos/{repo}/issues/{issue_number}/labels"
        resp = await self._request("POST", path, json={"labels": [label]},
                                    timeout=self.config.total_timeout)
        try:
            require_2xx(resp.status_code, "github add label")
        except SecurityError as e:
            raise GitHubError(str(e)) from e

    async def remove_issue_label(self, repo: str, issue_number: int,
                                  label: str) -> None:
        """Remove a single label from an issue/PR."""
        path = f"/repos/{repo}/issues/{issue_number}/labels/{label}"
        resp = await self._request("DELETE", path, timeout=self.config.total_timeout)
        # 404 is ok (label already doesn't exist)
        if resp.status_code not in (200, 204, 404):
            try:
                require_2xx(resp.status_code, "github remove label")
            except SecurityError as e:
                raise GitHubError(str(e)) from e

    async def pr_merged(self, repo: str, pr_number: int) -> tuple[bool, str, str]:
        """Return (merged, merged_at, merge_commit_sha)."""
        pr = await self.get_pr(repo, pr_number)
        merged = pr.get("merged") is True  # strict: only JSON true counts
        return merged, str(pr.get("merged_at") or ""), str(pr.get("merge_commit_sha") or "")

    async def resolve_commit_sha(self, repo: str, ref: str) -> str:
        """GET /repos/{repo}/commits/{ref} -> return full 40-hex SHA.

        Raises GitHubError on 404/422/malformed response.
        """
        if not ref or not str(ref).strip():
            raise GitHubError("resolve_commit_sha: ref is empty")
        resp = await self._request("GET", f"/repos/{repo}/commits/{ref}")
        data = await self._read_json(resp)
        sha = data.get("sha")
        if not isinstance(sha, str) or len(sha) != 40 or not all(c in "0123456789abcdef" for c in sha.lower()):
            raise GitHubError(f"resolve_commit_sha: invalid sha {sha!r}")
        return sha.lower()


    async def list_installation_repositories(self) -> list[str]:
        """GET /installation/repositories with bounded pagination.

        Returns a sorted, deduplicated list of repository full_names
        authorized for the current GitHub App installation.

        Raises GitHubError on non-2xx, malformed schema, pagination
        inconsistency, or incomplete collection.
        """
        url = "/installation/repositories"
        repos: list[str] = []
        seen: set[str] = set()
        page = 1
        max_pages = 10  # bounded pagination per spec
        expected_total: int | None = None
        total_batch_entries: int = 0
        while page <= max_pages:
            resp = await self._request(
                "GET", url,
                params={"page": page, "per_page": 100},
                timeout=self.config.total_timeout,
            )
            try:
                require_2xx(resp.status_code, "github list installation repositories")
            except SecurityError as e:
                raise GitHubError(str(e)) from e
            data = await self._read_json(resp)
            if not isinstance(data, dict):
                raise GitHubError(
                    "installation/repositories: expected top-level object"
                )
            total_count = data.get("total_count")
            if not isinstance(total_count, int) or isinstance(total_count, bool) or total_count < 0:
                raise GitHubError(
                    "installation/repositories: invalid total_count"
                )
            # total_count must be consistent across all pages
            if expected_total is None:
                expected_total = total_count
            elif total_count != expected_total:
                raise GitHubError(
                    f"installation/repositories: total_count changed from "
                    f"{expected_total} to {total_count} on page {page}"
                )
            repositories = data.get("repositories")
            if not isinstance(repositories, list):
                raise GitHubError(
                    "installation/repositories: expected repositories list"
                )
            total_batch_entries += len(repositories)
            for item in repositories:
                if not isinstance(item, dict):
                    raise GitHubError(
                        "installation/repositories: repository entry is not an object"
                    )
                full_name = item.get("full_name")
                if not isinstance(full_name, str) or not full_name.strip():
                    raise GitHubError(
                        "installation/repositories: empty or invalid full_name"
                    )
                fn = full_name.strip()
                if fn not in seen:
                    repos.append(fn)
                    seen.add(fn)
            if len(repositories) < 100:
                break
            page += 1
        if page > max_pages:
            raise GitHubError(
                f"installation repository pagination exceeded {max_pages} pages"
            )
        # Final completeness validation: the sum of all page batch lengths
        # must equal total_count (no missing pages), and the number of
        # unique repos must also equal total_count (no silent dedup).
        if total_batch_entries != expected_total:
            raise GitHubError(
                f"installation/repositories: total batch entries "
                f"{total_batch_entries} != total_count {expected_total}"
            )
        if len(seen) != expected_total:
            raise GitHubError(
                f"installation/repositories: unique repos "
                f"{len(seen)} != total_count {expected_total}"
            )
        return sorted(repos)

    async def pr_state(self, repo: str, pr_number: int) -> PrSnapshot | None:
        """Strict, fail-closed snapshot of a PR for candidate creation.

        Returns None when the PR cannot be read, the state is unknown, or any
        deploy-relevant field is missing / the wrong type (bool/int confusion,
        null). ``merged`` must be an explicit JSON boolean; ``head_sha`` must be
        a non-empty string; a closed+merged PR must also carry ``merged_at`` and
        ``merge_commit_sha``.
        """
        pr = await self.get_pr(repo, pr_number)
        state = pr.get("state")
        if not isinstance(state, str) or state not in ("open", "closed"):
            return None
        head = pr.get("head")
        if not isinstance(head, dict):
            return None
        head_sha = head.get("sha")
        if not isinstance(head_sha, str) or not head_sha:
            return None
        merged = pr.get("merged")
        if type(merged) is not bool:
            return None
        merged_at = pr.get("merged_at")
        merge_commit_sha = pr.get("merge_commit_sha")
        if not isinstance(merged_at, str):
            merged_at = ""
        if not isinstance(merge_commit_sha, str):
            merge_commit_sha = ""
        if state == "closed" and merged:
            if not merged_at or not merge_commit_sha:
                # Closed+merged but without merge identity -> fail closed.
                return None
        user = pr.get("user")
        author_login = ""
        author_id = ""
        if isinstance(user, dict):
            login = user.get("login")
            uid = user.get("id")
            if isinstance(login, str) and login.strip():
                author_login = login.strip()
            if isinstance(uid, int) and not isinstance(uid, bool) and uid:
                author_id = str(uid)
        if not author_login or not author_id:
            # The PR author identity is required for the auto-candidate's
            # requester (self-approval gate). Fail closed when missing.
            return None
        return PrSnapshot(
            state=state,
            head_sha=head_sha,
            merged=merged,
            merged_at=merged_at,
            merge_commit_sha=merge_commit_sha,
            author_login=author_login,
            author_id=author_id,
        )


    async def collaborator_permission(self, repo: str, username: str) -> str:
        """Return the collaborator permission for ``username`` in ``repo``.

        Only the REST API is authoritative (``GET /repos/{repo}/collaborators/{user}``).
        Returns the raw permission string, which the caller maps to a decision
        (write/maintain/admin pass; read/triage/none fail). Any API error, 404,
        non-2xx, or schema/type failure fails closed (returns an empty string).
        """
        path = f"/repos/{repo}/collaborators/{username}"
        resp = None
        try:
            resp = await self._request("GET", path, timeout=self.config.total_timeout)
        except httpx.HTTPError as e:
            logger.warning("collaborator permission check failed: %s", e)
            return ""
        except SecurityError as e:
            logger.warning("collaborator permission check blocked: %s", e)
            return ""
        if resp.status_code in (401, 403, 404):
            return ""
        try:
            require_2xx(resp.status_code, "github collaborator permission")
        except SecurityError as e:
            logger.warning("collaborator permission HTTP %s: %s", resp.status_code, e)
            return ""
        data = await self._read_json(resp)
        perm = data.get("permission")
        if not isinstance(perm, str) or not perm:
            return ""
        return perm

    async def list_repository_labels(self, repo: str) -> list[dict]:
        """List all labels in a repository with bounded pagination."""
        url = f"/repos/{repo}/labels"
        labels: list[dict] = []
        max_pages = 20
        for page in range(1, max_pages + 1):
            resp = await self._request(
                "GET", url,
                params={"per_page": 100, "page": page},
                timeout=self.config.total_timeout,
            )
            try:
                require_2xx(resp.status_code, "github list repository labels")
            except SecurityError as e:
                raise GitHubError(str(e)) from e
            batch = await self._read_json_list(resp)
            for item in batch:
                if not isinstance(item, dict):
                    raise GitHubError("github returned a malformed label record")
                labels.append(item)
            if len(batch) < 100:
                return labels
        raise GitHubError(
            f"github label pagination exceeded {max_pages} pages for {repo}"
        )

    async def create_repository_label(
        self, repo: str, name: str, color: str, description: str,
    ) -> dict:
        """Create a repository label using the authenticated GitHub token."""
        path = f"/repos/{repo}/labels"
        resp = await self._request("POST", path,
            json={"name": name, "color": color, "description": description},
            timeout=self.config.total_timeout,)
        try:
            require_2xx(resp.status_code, "github create repository label")
        except SecurityError as e:
            raise GitHubError(str(e)) from e
        return await self._read_json(resp)

    async def comment_identity(self, repo: str, comment_id: int) -> tuple[str, str]:
        """Re-read a comment from GitHub and return its immutable author
        ``(user_id, login)``.

        The Webhook payload's login/id is never trusted: the comment is re-fetched
        by id so a spoofed event body cannot forge the approver. Returns empty
        strings when the comment cannot be read or the user object is malformed.
        """
        try:
            c = await self.get_comment(repo, comment_id)
        except (GitHubError, httpx.HTTPError) as e:
            logger.warning("re-read comment %s failed: %s", comment_id, e)
            return "", ""
        user = c.get("user")
        if not isinstance(user, dict):
            return "", ""
        uid = user.get("id")
        login = user.get("login")
        if not isinstance(uid, int) or uid is None:
            return "", ""
        return str(uid), str(login or "")
