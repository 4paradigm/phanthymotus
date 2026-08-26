"""Configuration loaded from environment variables."""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    # GitHub
    github_token: str = ""
    github_webhook_secret: str = ""

    # Which transport git uses to reach GitHub.
    #
    # "ssh" by default, and not because of authentication — both repos are
    # public. It is a reachability question: on the Tencent hosts, TCP to
    # github.com:443 completes but the TLS handshake is dropped by SNI, so git
    # over HTTPS fails every time while api.github.com works fine. SSH carries
    # no SNI and works. Requires a key mounted at /root/.ssh.
    #
    # Set to "https" on a host where github.com:443 is genuinely reachable and
    # you would rather not mount a key.
    git_transport: str = "ssh"

    # Repos to watch. Clone URLs are derived from git_transport.
    repo_names: tuple[str, ...] = (
        "4paradigm/phanthymotus",
        "4paradigm/phanthymotus-driver",
    )

    # Trigger mode — polling needs only outbound network, webhook needs inbound.
    poll_enabled: bool = True
    poll_interval_seconds: int = 30
    poll_initial_lookback_minutes: int = 10
    webhook_enabled: bool = True

    # Registry
    registry: str = ""
    registry_user: str = ""
    registry_password: str = ""
    image_namespace: str = "phanthy-motus"
    image_namespace_drivers: str = "phanthy-motus/drivers"
    mirror: str = "tencent"
    push_enabled: bool = True

    # LLM Review (OpenAI-compatible)
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "claude-sonnet-4-20250514"
    max_diff_lines: int = 3000
    llm_max_tokens: int = 4000
    llm_timeout_seconds: int = 180

    # Agentic review loop. Rounds and wall-clock are both bounded: the round cap
    # alone would allow a pathological review to run for hours at the per-request
    # timeout, which is how agent-core's main loop can.
    review_max_rounds: int = 20
    review_timeout_seconds: int = 600
    # Added files at or above this size are reported separately and should live
    # in COS instead of the repo.
    large_file_threshold_kb: int = 500
    # How much of the PR's own account of itself the reviewer is given. Bounded
    # because it is untrusted text sharing a context window with the rules.
    pr_context_max_chars: int = 4000
    pr_context_max_comments: int = 20

    # Worker
    max_concurrent_jobs: int = 2
    # What actually bounds a build: time since its last line of output. A live
    # docker build prints constantly (buildx progress, compiler lines, layer
    # exports); a wedged one prints nothing. Slowness is not a hang — bounding
    # by total wall clock killed a build that was compiling openfst one file at
    # a time, output flowing the whole way, and reported it as a build failure.
    build_idle_timeout_seconds: int = 600
    # Absolute backstop for one build, for the case the idle bound cannot see:
    # a build that prints forever without finishing.
    build_timeout_seconds: int = 7200
    # Whole-job backstop. A job exceeding this is treated as lost and retried.
    # Loose on purpose: every stage below it is already bounded on its own —
    # git by FETCH_TIMEOUT/GIT_LOCAL_TIMEOUT, the review loop by
    # review_timeout_seconds, builds by silence — so this is no longer "how long
    # may a job take" but "the agent itself is stuck".
    job_timeout_seconds: int = 14400
    # Total attempts per job including the first (3 = two retries).
    max_attempts: int = 3
    retry_backoff_seconds: int = 60

    # How long job history and build logs are retained. Pruned at startup.
    job_history_days: int = 30

    # Paths
    data_dir: str = "/data/repos"

    # Server
    host: str = "0.0.0.0"
    port: int = 25000

    # Resource Center (optional)
    resource_center_url: str = ""
    resource_center_api_key: str = ""

    @property
    def repos(self) -> dict[str, str]:
        """Repo full_name -> clone URL, built from the configured transport."""
        return {name: clone_url(name, self.git_transport) for name in self.repo_names}


def clone_url(full_name: str, transport: str) -> str:
    if transport == "https":
        return f"https://github.com/{full_name}.git"
    return f"git@github.com:{full_name}.git"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _load_repo_names() -> tuple[str, ...] | None:
    """Optionally override the repo list via GITHUB_REPOS.

    Format: comma-separated full names, e.g.
        GITHUB_REPOS=4paradigm/phanthymotus,4paradigm/phanthymotus-driver
    """
    raw = os.environ.get("GITHUB_REPOS", "").strip()
    if not raw:
        return None
    names = tuple(
        item.strip() for item in raw.split(",")
        if item.strip() and "/" in item
    )
    return names or None


def load_config() -> Config:
    overrides = {}
    names = _load_repo_names()
    if names is not None:
        overrides["repo_names"] = names

    transport = os.environ.get("GIT_TRANSPORT", "ssh").strip().lower()
    if transport not in ("ssh", "https"):
        transport = "ssh"

    return Config(
        github_token=os.environ.get("GITHUB_TOKEN", ""),
        github_webhook_secret=os.environ.get("GITHUB_WEBHOOK_SECRET", ""),
        git_transport=transport,
        poll_enabled=_env_bool("POLL_ENABLED", True),
        poll_interval_seconds=_env_int("POLL_INTERVAL_SECONDS", 30),
        poll_initial_lookback_minutes=_env_int("POLL_INITIAL_LOOKBACK_MINUTES", 10),
        webhook_enabled=_env_bool("WEBHOOK_ENABLED", True),
        registry=os.environ.get("REGISTRY", ""),
        registry_user=os.environ.get("REGISTRY_USER", ""),
        registry_password=os.environ.get("REGISTRY_PASSWORD", ""),
        image_namespace=os.environ.get("IMAGE_NAMESPACE", "phanthy-motus"),
        image_namespace_drivers=os.environ.get(
            "IMAGE_NAMESPACE_DRIVERS", "phanthy-motus/drivers"
        ),
        mirror=os.environ.get("MIRROR", "tencent"),
        push_enabled=_env_bool("PUSH_ENABLED", True),
        llm_base_url=os.environ.get("LLM_BASE_URL", ""),
        llm_api_key=os.environ.get("LLM_API_KEY", ""),
        llm_model=os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514"),
        max_diff_lines=_env_int("MAX_DIFF_LINES", 3000),
        llm_max_tokens=_env_int("LLM_MAX_TOKENS", 4000),
        llm_timeout_seconds=_env_int("LLM_TIMEOUT_SECONDS", 180),
        review_max_rounds=_env_int("REVIEW_MAX_ROUNDS", 20),
        review_timeout_seconds=_env_int("REVIEW_TIMEOUT_SECONDS", 600),
        large_file_threshold_kb=_env_int("LARGE_FILE_THRESHOLD_KB", 500),
        pr_context_max_chars=_env_int("PR_CONTEXT_MAX_CHARS", 4000),
        pr_context_max_comments=_env_int("PR_CONTEXT_MAX_COMMENTS", 20),
        max_concurrent_jobs=_env_int("MAX_CONCURRENT_JOBS", 2),
        build_idle_timeout_seconds=_env_int("BUILD_IDLE_TIMEOUT_SECONDS", 600),
        build_timeout_seconds=_env_int("BUILD_TIMEOUT_SECONDS", 7200),
        job_timeout_seconds=_env_int("JOB_TIMEOUT_SECONDS", 14400),
        max_attempts=_env_int("MAX_ATTEMPTS", 3),
        retry_backoff_seconds=_env_int("RETRY_BACKOFF_SECONDS", 60),
        job_history_days=_env_int("JOB_HISTORY_DAYS", 30),
        data_dir=os.environ.get("DATA_DIR", "/data/repos"),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=_env_int("PORT", 25000),
        resource_center_url=os.environ.get("RESOURCE_CENTER_URL", ""),
        resource_center_api_key=os.environ.get("RESOURCE_CENTER_API_KEY", ""),
        **overrides,
    )
