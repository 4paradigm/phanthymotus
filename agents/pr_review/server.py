"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .config import load_config
from .git_workspace import GitWorkspaceManager
from .github_client import GitHubClient
from .job_queue import JobQueue
from .poller import Poller
from .router_api import router as api_router
from .router_webhook import router as webhook_router
from .store import JobStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Resolved from this file rather than the CWD. agent-core uses a relative
# './web' and relies on its launcher to guarantee the working directory;
# resolving from __file__ removes that hidden requirement so
# `python -m agents.pr_review.server` works from anywhere.
WEB_DIR = Path(__file__).parent / "web"


class _NoCacheStaticFiles(StaticFiles):
    """StaticFiles that refuses to be cached, and ignores non-HTTP scopes.

    Mirrors agent-core's `_HTTPOnlyStaticFiles` (`start.py:655`): the dashboard
    is edited in place with no build step or content hashing, so a cached JS
    file would silently serve stale code after a deploy.
    """

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return

        async def send_no_cache(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                headers[b"cache-control"] = b"no-cache, no-store, must-revalidate"
                message = {**message, "headers": list(headers.items())}
            await send(message)

        await super().__call__(scope, receive, send_no_cache)


def _check_git_transport(config):
    """Warn loudly if the configured git transport cannot possibly work.

    An unusable transport otherwise shows up much later as a fetch that hangs
    until its timeout, three attempts deep, on someone's PR.
    """
    if config.git_transport != "ssh":
        logger.info("Git transport: https")
        return

    key_dir = Path("/root/.ssh")
    keys = (
        [p for p in key_dir.glob("id_*") if not p.name.endswith(".pub")]
        if key_dir.is_dir() else []
    )
    if keys:
        logger.info(
            f"Git transport: ssh (keys: {', '.join(p.name for p in keys)})"
        )
    else:
        logger.error(
            "GIT_TRANSPORT=ssh but no SSH key found at /root/.ssh — every git "
            "fetch will fail. Mount a key (compose mounts $SSH_DIR read-only) "
            "or set GIT_TRANSPORT=https if github.com:443 is reachable here."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    app.state.config = config

    if not config.github_token:
        logger.warning("GITHUB_TOKEN is not set — GitHub API calls will fail")

    _check_git_transport(config)

    app.state.github_client = GitHubClient(config.github_token)

    # Durable job history and build logs.
    store = JobStore(config.data_dir)
    store.init()
    # Nothing resumes across a restart, so close out anything left mid-flight by
    # an unclean shutdown before the dashboard can show it as still running.
    await store.reconcile_orphans()
    pruned = await store.prune(config.job_history_days)
    if pruned:
        logger.info(f"Pruned {pruned} job(s) older than {config.job_history_days}d")
    app.state.store = store

    # Persistent bare clones — cloned once, then fetched incrementally.
    workspace_mgr = GitWorkspaceManager(config.data_dir, config.repos)
    await workspace_mgr.ensure_clones()
    await workspace_mgr.cleanup_stale_worktrees()
    app.state.workspace_mgr = workspace_mgr

    job_queue = JobQueue(
        max_workers=config.max_concurrent_jobs,
        config=config,
        github_client=app.state.github_client,
        workspace_mgr=workspace_mgr,
        store=store,
    )
    await job_queue.start()
    app.state.job_queue = job_queue

    # Polling is the default trigger: outbound-only, so the agent needs no
    # public IP, no open port, and no webhook registration.
    app.state.poller = None
    if config.poll_enabled:
        poller = Poller(config, app.state.github_client, job_queue, store)
        await poller.start()
        app.state.poller = poller

    logger.info(
        f"PR Review Agent listening on {config.host}:{config.port} "
        f"(workers={config.max_concurrent_jobs}, "
        f"poll={'on' if config.poll_enabled else 'off'}, "
        f"webhook={'on' if config.webhook_enabled else 'off'}, "
        f"job_timeout={config.job_timeout_seconds}s, "
        f"max_attempts={config.max_attempts})"
    )
    logger.info(f"Dashboard: http://localhost:{config.port}/")

    yield

    if app.state.poller is not None:
        await app.state.poller.stop()
    await job_queue.stop()
    await app.state.github_client.close()


app = FastAPI(title="PR Review Agent", lifespan=lifespan)

# Order matters: the routers are registered before the static mount so /api and
# /webhook win over the catch-all at /.
app.include_router(api_router)
app.include_router(webhook_router)

if WEB_DIR.is_dir():
    app.mount(
        "/", _NoCacheStaticFiles(directory=str(WEB_DIR), html=True), name="web"
    )
else:
    logger.warning(f"Web directory not found at {WEB_DIR} — dashboard disabled")


def main():
    config = load_config()
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
