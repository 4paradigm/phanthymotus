"""FastAPI application entry point for the Deploy Approval Agent (stateless).

Runtime persistence is solely the GitHub lifecycle comment hidden state.
Single writer: only GitHubCommandWatcher mutates deploy state.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from .config import Config, DEFAULT_GITHUB_REPOS, load_config
from .github_state_proxy import _ALLOWED_STATUS_LABELS
from .github_client import GitHubClient
from agents import github_app_auth
from .github_state_proxy import GitHubStateProxy
from .github_command_watcher import GitHubCommandWatcher
from .policy import Policy
from .registry_client import RegistryClient
from .router_webhook import router as webhook_router
from .service import DeployController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


_STATUS_LABEL_SPECS = (
    (
        "status: review-required",
        "D93F0B",
        "Deploy Approval: current HEAD requires review",
    ),
    (
        "status: reviewing",
        "5319E7",
        "Deploy Approval: current HEAD is being reviewed",
    ),
    (
        "status: deploy-ready",
        "0E8A16",
        "Deploy Approval: reviewed HEAD is ready for deploy request",
    ),
    (
        "status: deploy-requested",
        "FBCA04",
        "Deploy Approval: Machine Owner action required",
    ),
    (
        "status: testing",
        "1D76DB",
        "Deploy Approval: deployment complete; human testing required",
    ),
    (
        "status: succeeded",
        "0E8A16",
        "Deploy Approval: human validation passed",
    ),
    (
        "status: failed",
        "B60205",
        "Deploy Approval: deployment or validation failed",
    ),
)


def _validate_label_namespace(repo: str, labels: list[dict]) -> dict[str, dict]:
    if not isinstance(labels, list):
        raise ValueError(f"repository {repo}: labels response must be a list")
    exact: dict[str, dict] = {}
    casefold_seen: dict[str, str] = {}
    for label in labels:
        if not isinstance(label, dict):
            raise ValueError(f"repository {repo}: label record must be an object")
        name = label.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"repository {repo}: label name must be a non-empty string")
        name = name.strip()
        lowered = name.casefold()
        if name in exact:
            raise ValueError(
                f"repository {repo}: duplicate label record {name!r}"
            )
        existing = casefold_seen.get(lowered)
        if existing is not None and existing != name:
            raise ValueError(
                f"repository {repo}: existing conflicting label {existing!r} "
                f"conflicts with required exact label {name!r}"
            )
        exact[name] = label
        casefold_seen[lowered] = name
    return exact


_LABEL_SPECS: dict[str, tuple[str, str]] = {
    "status: review-required": ("D93F0B", "Deploy Approval: current HEAD requires review"),
    "status: reviewing": ("5319E7", "Deploy Approval: current HEAD is being reviewed"),
    "status: deploy-ready": ("0E8A16", "Deploy Approval: reviewed HEAD is ready for deploy request"),
    "status: deploy-requested": ("FBCA04", "Deploy Approval: Machine Owner action required"),
    "status: testing": ("1D76DB", "Deploy Approval: deployment complete; human testing required"),
    "status: succeeded": ("0E8A16", "Deploy Approval: human validation passed"),
    "status: failed": ("B60205", "Deploy Approval: deployment or validation failed"),
}


async def _bootstrap_status_labels(github: GitHubClient, repos: list[str] | None = None) -> dict:
    """Best-effort bootstrap of canonical status labels.

    Label bootstrap is **optional UI projection only**.  Any label API
    permission failure (403, 404, permission denied, etc.) is logged as a
    warning and **never** raises or blocks startup.

    Returns a summary dict keyed by repo name.
    """
    if repos is None:
        repos = list(getattr(getattr(github, "config", None), "github_repos", DEFAULT_GITHUB_REPOS))
    summary: dict = {}

    for repo in repos:
        repo_summary: dict[str, list] = {"available": [], "missing": [], "errors": []}
        try:
            labels = await github.list_repository_labels(repo)
        except Exception as exc:
            logger.warning(
                "label bootstrap list failed for %s: %s",
                repo, type(exc).__name__,
            )
            repo_summary["errors"].append(type(exc).__name__)
            summary[repo] = repo_summary
            continue

        try:
            exact = _validate_label_namespace(repo, labels)
        except Exception as exc:
            logger.warning(
                "label bootstrap validate failed for %s: %s",
                repo, type(exc).__name__,
            )
            repo_summary["errors"].append(type(exc).__name__)
            summary[repo] = repo_summary
            continue

        repo_summary["available"] = list(exact.keys())
        lower_names = {name.casefold(): name for name in exact}

        # Attempt to create any missing labels.
        for name, (color, description) in _LABEL_SPECS.items():
            if name in exact:
                continue
            conflicting = lower_names.get(name.casefold())
            if conflicting is not None and conflicting != name:
                logger.warning(
                    "repository %s: skipping label %s due to conflict %s",
                    repo, name, conflicting,
                )
                repo_summary["missing"].append(name)
                continue
            try:
                await github.create_repository_label(repo, name, color, description)
                repo_summary["available"].append(name)
            except Exception as exc:
                # Race-safety: fresh check before giving up.
                try:
                    fresh_labels = await github.list_repository_labels(repo)
                    fresh_exact = _validate_label_namespace(repo, fresh_labels)
                    if name in fresh_exact:
                        repo_summary["available"].append(name)
                        continue
                except Exception:
                    pass
                logger.warning(
                    "label bootstrap create failed for %s %s: %s",
                    repo, name, type(exc).__name__,
                )
                repo_summary["errors"].append(name)

        # Final check -- missing labels are WARNING only, never raise.
        for name in _ALLOWED_STATUS_LABELS:
            if name not in exact and name not in repo_summary["available"]:
                if name not in repo_summary["missing"]:
                    repo_summary["missing"].append(name)
                logger.warning(
                    "repository %s: status label %s is missing (label bootstrap is best-effort)",
                    repo, name,
                )

        summary[repo] = repo_summary

    return summary


def create_app(config: Config | None = None):
    """Build the FastAPI app.

    Single writer: only GitHubCommandWatcher is started.
    No Poller, no second mutation loop.

    GitHub App auth is created first, then injected into GitHubClient
    and GitHubStateProxy.  No startup bot-identity lookup.
    """
    config = config or load_config()
    policy = Policy(config)
    policy.load_machines()

    # Create shared GitHub App auth provider first
    github_auth = github_app_auth.create_github_app_auth()

    # GitHubClient MUST receive async token_provider; fail closed if missing
    github = GitHubClient(
        config,
        token_provider=github_auth.get_installation_token,
    )
    registry = RegistryClient(config)

    # GitHubStateProxy receives github_app_id for lazy provenance check
    proxy = GitHubStateProxy(
        config, github, github_app_id=github_auth.app_id,
    )

    controller = DeployController(
        config, proxy, policy, github, registry,
        agent_core_factory=None,
    )

    # Single serial command watcher
    watcher = GitHubCommandWatcher(config, proxy, controller)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            await _bootstrap_status_labels(github)
        except Exception:
            logger.warning(
                "label bootstrap failed (best-effort); watcher starting without label projection",
            )
        watcher.start()
        try:
            yield
        finally:
            await watcher.stop()
            close = getattr(controller, "aclose", None)
            if close is not None:
                await close()
            for client in (github, registry):
                http = getattr(client, "http", None)
                aclose = getattr(http, "aclose", None)
                if aclose is not None:
                    await aclose()
            await github_auth.close()

    app = FastAPI(title="Deploy Approval Agent", lifespan=lifespan)
    app.state.config = config
    app.state.policy = policy
    app.state.github = github
    app.state.registry = registry
    app.state.controller = controller
    app.state.proxy = proxy
    app.state.watcher = watcher

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    app.include_router(webhook_router)
    return app


def main():
    config = load_config()
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
