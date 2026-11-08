"""Repository-scoped policy snapshots sourced from each repository's main."""

from __future__ import annotations

import asyncio
import copy
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from .config import DEFAULT_GITHUB_REPOS, validate_repository_names
from .models import MachineInfo

logger = logging.getLogger(__name__)

POLICY_REFRESH_SECONDS = 30 * 60
_POLICY_FILES = ("machines.yaml", "test_reviewers.yaml")
BLOCKED_REPOSITORY_POLICY_NOT_CONFIGURED = "BLOCKED_REPOSITORY_POLICY_NOT_CONFIGURED"


@dataclass
class _RepositoryPolicy:
    machines: dict[str, MachineInfo] | None = None
    reviewers: list[str] | None = None
    source_sha: dict[str, str] = field(default_factory=dict)
    etag: dict[str, str] = field(default_factory=dict)
    last_checked: float | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class RepositoryPolicyStore:
    """Keep independently validated policy snapshots for the two controlled repos.

    The old local machine and reviewer files remain available through the
    legacy no-repository Policy methods. They are deliberately never copied
    into either repository snapshot.
    """

    def __init__(self, config, *, machine_parser: Callable, reviewer_parser: Callable):
        self._config = config
        self._machine_parser = machine_parser
        self._reviewer_parser = reviewer_parser
        self._repositories: dict[str, _RepositoryPolicy] = {}
        self._reported_unconfigured: set[tuple[str, str]] = set()
        self._reported_authorization: set[str] = set()

    @staticmethod
    def _canonical_repo(repo: str) -> str | None:
        try:
            canonical = validate_repository_names([repo])[0]
        except (TypeError, ValueError):
            return None
        targets = {name.lower() for name in DEFAULT_GITHUB_REPOS}
        return canonical if canonical in targets else None

    def _state(self, repo: str) -> _RepositoryPolicy:
        return self._repositories.setdefault(repo, _RepositoryPolicy())

    def _app_authorized(self, repo: str) -> bool:
        if self._config.auth_valid is not True:
            authorized = False
        else:
            try:
                configured = set(validate_repository_names(self._config.github_repos))
                active = set(validate_repository_names(
                    list(self._config.active_repos),
                    label="active installation repositories",
                    allow_empty=True,
                ))
            except (TypeError, ValueError):
                authorized = False
            else:
                authorized = repo in configured and repo in active
        if authorized:
            self._reported_authorization.discard(repo)
        elif repo not in self._reported_authorization:
            logger.warning(
                "repository policy access is blocked for %s: current GitHub App installation authorization is missing",
                repo,
            )
            self._reported_authorization.add(repo)
        return authorized

    def machines_for(self, repo: str) -> dict[str, MachineInfo]:
        canonical = self._canonical_repo(repo)
        if canonical is None:
            logger.warning("repository machine policy requested for an unsupported repository")
            return {}
        if not self._app_authorized(canonical):
            return {}
        machines = self._state(canonical).machines
        if machines is None:
            self._warn_unconfigured(canonical, "machines.yaml")
            return {}
        return copy.deepcopy(machines)

    def machine_for(self, alias: str, repo: str) -> MachineInfo | None:
        if not isinstance(alias, str) or not alias:
            return None
        canonical = self._canonical_repo(repo)
        if canonical is None:
            logger.warning("repository machine policy requested for an unsupported repository")
            return None
        if not self._app_authorized(canonical):
            return None
        machines = self._state(canonical).machines
        if machines is None:
            self._warn_unconfigured(canonical, "machines.yaml")
            return None
        machine = machines.get(alias)
        return copy.deepcopy(machine) if machine is not None else None

    def reviewers_for_repo(self, repo: str) -> list[str]:
        canonical = self._canonical_repo(repo)
        if canonical is None:
            logger.warning("repository acceptance policy requested for an unsupported repository")
            return []
        if not self._app_authorized(canonical):
            return []
        reviewers = self._state(canonical).reviewers
        if reviewers is None:
            self._warn_unconfigured(canonical, "test_reviewers.yaml")
            return []
        return list(reviewers)

    def status(self, repo: str) -> dict:
        canonical = self._canonical_repo(repo)
        if canonical is None:
            return {
                "status": "unsupported_repository",
                "machines_configured": False,
                "reviewers_configured": False,
                "missing_files": list(_POLICY_FILES),
                "blocking_reason": BLOCKED_REPOSITORY_POLICY_NOT_CONFIGURED,
            }
        if not self._app_authorized(canonical):
            return {
                "status": "unauthorized",
                "authorized": False,
                "machines_configured": False,
                "reviewers_configured": False,
                "missing_files": [],
                "blocking_reason": None,
            }
        state = self._repositories.get(canonical)
        machine_configured = bool(state and state.machines is not None)
        reviewers_configured = bool(state and state.reviewers is not None)
        missing_files = []
        if not machine_configured:
            missing_files.append("machines.yaml")
            self._warn_unconfigured(canonical, "machines.yaml")
        if not reviewers_configured:
            missing_files.append("test_reviewers.yaml")
            self._warn_unconfigured(canonical, "test_reviewers.yaml")
        if machine_configured and reviewers_configured:
            policy_status = "configured"
        elif machine_configured or reviewers_configured:
            policy_status = "partially_configured"
        else:
            policy_status = "unconfigured"
        return {
            "status": policy_status,
            "authorized": True,
            "machines_configured": machine_configured,
            "reviewers_configured": reviewers_configured,
            "missing_files": missing_files,
            "blocking_reason": (
                BLOCKED_REPOSITORY_POLICY_NOT_CONFIGURED if missing_files else None
            ),
            "last_check_age_seconds": (
                max(0, int(time.monotonic() - state.last_checked))
                if state and state.last_checked is not None else None
            ),
        }

    async def refresh_if_due(self, repo: str, github_client) -> dict:
        canonical = self._canonical_repo(repo)
        if canonical is None:
            logger.warning("repository policy refresh rejected for an unsupported repository")
            return {"status": "unsupported_repository", "updated": []}

        state = self._state(canonical)
        async with state.lock:
            if not self._app_authorized(canonical):
                return {"status": "unauthorized", "updated": []}
            now = time.monotonic()
            if (state.last_checked is not None
                    and now - state.last_checked < POLICY_REFRESH_SECONDS):
                return {
                    "status": "not_due",
                    "updated": [],
                    "machines_configured": state.machines is not None,
                    "reviewers_configured": state.reviewers is not None,
                }

            # A due check is one bounded pass over these two exact paths. Each
            # file is validated and committed independently so a bad update
            # cannot discard another file's last known good value.
            updated: list[str] = []
            file_results: dict[str, str] = {}
            getter = getattr(github_client, "get_repository_file_at_ref", None)
            if not callable(getter):
                state.last_checked = time.monotonic()
                logger.warning(
                    "repository policy refresh unavailable for %s: GitHub client has no policy file reader",
                    canonical,
                )
                return {"status": "unavailable", "updated": []}

            for filename in _POLICY_FILES:
                await_result = None
                try:
                    await_result = await getter(
                        canonical,
                        filename,
                        ref="main",
                        etag=state.etag.get(filename),
                    )
                except Exception as exc:
                    logger.warning(
                        "repository policy refresh failed for %s/%s on main (%s); retaining last verified policy",
                        canonical,
                        filename,
                        type(exc).__name__,
                    )
                    file_results[filename] = "error"
                    continue

                if await_result is None:
                    logger.warning(
                        "repository policy file %s/%s is absent from main; retaining last verified policy",
                        canonical,
                        filename,
                    )
                    file_results[filename] = "missing"
                    continue
                if not isinstance(await_result, dict):
                    logger.warning(
                        "repository policy file response for %s/%s is invalid; retaining last verified policy",
                        canonical,
                        filename,
                    )
                    file_results[filename] = "invalid_response"
                    continue
                if await_result.get("not_modified") is True:
                    if state.etag.get(filename):
                        file_results[filename] = "unchanged"
                    else:
                        logger.warning(
                            "repository policy file response for %s/%s was unexpectedly not modified; retaining last verified policy",
                            canonical,
                            filename,
                        )
                        file_results[filename] = "invalid_response"
                    continue

                source_path = await_result.get("path")
                source_sha = await_result.get("sha")
                content = await_result.get("content")
                source_etag = await_result.get("etag", "")
                if (source_path != filename
                        or not isinstance(source_sha, str)
                        or re.fullmatch(r"[0-9a-f]{40}", source_sha) is None
                        or not isinstance(content, bytes)
                        or not isinstance(source_etag, str)
                        or len(source_etag) > 512
                        or any(ord(char) < 32 or ord(char) == 127 for char in source_etag)):
                    logger.warning(
                        "repository policy file response for %s/%s failed validation; retaining last verified policy",
                        canonical,
                        filename,
                    )
                    file_results[filename] = "invalid_response"
                    continue

                # A valid Git blob SHA is the version comparison key. Avoid
                # YAML parsing and object replacement when the source did not
                # change, including a previously rejected but unchanged blob.
                if state.source_sha.get(filename) == source_sha:
                    if source_etag:
                        state.etag[filename] = source_etag
                    file_results[filename] = "unchanged"
                    continue

                try:
                    if filename == "machines.yaml":
                        candidate = self._machine_parser(content)
                        if not isinstance(candidate, dict) or not candidate:
                            raise ValueError("machines policy must not be empty")
                    else:
                        candidate = self._reviewer_parser(content)
                        if not isinstance(candidate, list) or not candidate:
                            raise ValueError("reviewer policy must not be empty")
                except Exception as exc:
                    # The exception may contain values from untrusted YAML.
                    # Report the failed file and exception class only.
                    state.source_sha[filename] = source_sha
                    if source_etag:
                        state.etag[filename] = source_etag
                    logger.warning(
                        "repository policy file %s/%s failed schema validation (%s); retaining last verified policy",
                        canonical,
                        filename,
                        type(exc).__name__,
                    )
                    file_results[filename] = "invalid_schema"
                    continue

                if filename == "machines.yaml":
                    state.machines = candidate
                else:
                    state.reviewers = candidate
                state.source_sha[filename] = source_sha
                if source_etag:
                    state.etag[filename] = source_etag
                self._reported_unconfigured.discard((canonical, filename))
                updated.append(filename)
                file_results[filename] = "updated"

            if state.machines is None:
                self._warn_unconfigured(canonical, "machines.yaml")
            if state.reviewers is None:
                self._warn_unconfigured(canonical, "test_reviewers.yaml")

            if updated:
                overall = "updated"
            elif any(value in {"error", "invalid_response", "invalid_schema"} for value in file_results.values()):
                overall = "partial_error"
            elif any(value == "missing" for value in file_results.values()):
                overall = "missing_source"
            else:
                overall = "unchanged"
            # Start the full 30-minute interval after the bounded fetch pass
            # completes, including failures, so slow API calls cannot shorten
            # the promised refresh period.
            state.last_checked = time.monotonic()
            return {
                "status": overall,
                "updated": updated,
                "files": file_results,
                "machines_configured": state.machines is not None,
                "reviewers_configured": state.reviewers is not None,
            }

    def _warn_unconfigured(self, repo: str, filename: str) -> None:
        marker = (repo, filename)
        if marker in self._reported_unconfigured:
            return
        logger.warning(
            "%s: repository %s has no verified %s from main; access is fail-closed. "
            "An administrator must configure %s/%s on main",
            BLOCKED_REPOSITORY_POLICY_NOT_CONFIGURED,
            repo, filename, repo, filename,
        )
        self._reported_unconfigured.add(marker)
