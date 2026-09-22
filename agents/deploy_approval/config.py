"""Configuration for the Deploy Approval Agent (pre-merge validation only).

Deploy Approval does not define a new runtime env namespace. It reuses the
existing upstream GitHub/poll/webhook/registry keys and fixed read-only files
for machine ownership and COS secrets.

Review Agent HTTP API is NOT a Deploy Approval dependency.
Deploy Approval reads Review Agent output from GitHub PR comments.
"""

from __future__ import annotations

from collections.abc import Mapping
import logging
import re
import os
import ipaddress
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


DEFAULT_GITHUB_REPOS = (
    "4paradigm/phanthymotus",
    "4paradigm/phanthymotus-driver",
)


SUPPORTED_GITHUB_REPOS = frozenset({
    "4paradigm/phanthymotus",
    "4paradigm/phanthymotus-driver",
})

# ── Authoritative production Review Agent GitHub identity ──────────────
# This is the ONLY accepted identity for Review Agent PR comments.
# It is NOT an env var. It is NOT configurable at runtime.
# Secrets.yaml review_comment_trust MUST match these exact values.
REVIEW_AGENT_GITHUB_USER_ID = "7950763"
REVIEW_AGENT_GITHUB_LOGIN = "kentcyq"


@dataclass
class Config:
    host: str = "0.0.0.0"
    port: int = 25001

    github_api_url: str = "https://api.github.com"
    github_webhook_secret: str = ""
    webhook_enabled: bool = False
    poll_enabled: bool = True
    poll_interval_seconds: int = 30
    github_repos: list[str] = field(
        default_factory=lambda: list(DEFAULT_GITHUB_REPOS)
    )
    github_comment_max_pages: int = 20
    github_comment_max_comments: int = 500
    github_comment_max_bytes: int = 4 * 1024 * 1024

    # Review comment trust configuration
    # Defaults align with authoritative production Review Agent identity.
    # Production startup validate_config() enforces exact match.
    review_comment_author_id: str = REVIEW_AGENT_GITHUB_USER_ID
    review_comment_author_login: str = REVIEW_AGENT_GITHUB_LOGIN

    # Machine owners configuration
    machine_owners_file: str = "/run/deploy-approval/machines.yaml"
    secrets_file: str = "/run/deploy-approval/secrets.yaml"

    # Security
    allow_private_http: bool = False
    http_allowed_cidrs: list[str] = field(default_factory=list)
    max_response_bytes: int = 8 * 1024 * 1024
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    total_timeout: float = 60.0

    # COS evidence storage
    cos_region: str = ""
    cos_bucket: str = ""
    cos_secret_id: str = ""
    cos_secret_key: str = ""

    # Registry trust anchor — the ONLY allowed registry host for image resolution.
    registry: str = ""

    registry_auth_host_allowlist: list[str] = field(default_factory=list)

    # Agent Core access tokens keyed by machine alias
    agent_core_tokens: dict[str, str] = field(
        default_factory=dict,
        repr=False,
    )


def _env_int(name: str, default: int, *, min_val: int = 1, max_val: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    if isinstance(raw, bool):
        raise ValueError(f"{name} must be a positive int, got bool")
    try:
        val = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive int, got {raw!r}")
    if isinstance(val, bool):
        raise ValueError(f"{name} must be a positive int, got bool")
    if val <= 0:
        raise ValueError(f"{name} must be a positive int, got {val!r}")
    if max_val is not None and val > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {val!r}")
    if min_val is not None and val < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {val!r}")
    return val


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        f = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number, got {raw!r}")
    if f <= 0 or f != f or f in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be a positive finite number, got {raw!r}")
    return f


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off", ""}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val == "":
        return default
    if val in _TRUE_VALUES:
        return True
    if val in _FALSE_VALUES:
        return False
    raise ValueError(
        f"invalid boolean value for {name}={raw!r} \u2014 expected one of "
        f"{sorted(_TRUE_VALUES | _FALSE_VALUES)}"
    )


def _env_str_list(name: str, default: list[str] | None = None) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return default or []
    if not raw.strip():
        return []
    result = [x.strip() for x in raw.split(",") if x.strip()]
    for item in result:
        if not isinstance(item, str):
            raise ValueError(f"{name} must be a comma-separated list of strings, got non-string element")
    return result


def _load_yaml_mapping(path: str) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    if not p.is_file():
        raise ValueError(f"{path} must be a regular file")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _load_secrets_config(path: str) -> dict:
    data = _load_yaml_mapping(path)
    if not data:
        return {}
    version = data.get("version")
    if version != 1:
        raise ValueError("secrets.yaml version must be 1")
    cos = data.get("cos", {})
    if cos is None:
        cos = {}
    if not isinstance(cos, dict):
        raise ValueError("secrets.yaml cos section must be a mapping")
    result = {"cos": {}, "review_comment_trust": {}}

    def _coerce_string(key: str, *, required: bool = False, allow_empty: bool = True) -> str:
        if key not in cos:
            if required:
                raise ValueError(f"secrets.yaml cos.{key} is required")
            return ""
        value = cos.get(key)
        if not isinstance(value, str):
            raise ValueError(f"secrets.yaml cos.{key} must be a string")
        if not allow_empty and not value.strip():
            raise ValueError(f"secrets.yaml cos.{key} must be a non-empty string")
        return value

    for key in ("region", "bucket", "secret_id", "secret_key"):
        result["cos"][key] = _coerce_string(key)
    # Review comment trust config
    rct = data.get("review_comment_trust", {})
    if rct is None:
        rct = {}
    if not isinstance(rct, dict):
        raise ValueError("secrets.yaml review_comment_trust section must be a mapping")
    for key in ("author_id", "author_login"):
        value = rct.get(key, "")
        if not isinstance(value, str):
            raise ValueError(f"secrets.yaml review_comment_trust.{key} must be a string")
        result["review_comment_trust"][key] = value
    # Agent Core tokens
    act = data.get("agent_core_tokens")
    if act is not None:
        if not isinstance(act, dict):
            raise ValueError("secrets.yaml agent_core_tokens must be a mapping")
        result["agent_core_tokens"] = {}
        for k, v in act.items():
            if not isinstance(k, str) or not k.strip():
                raise ValueError("secrets.yaml agent_core_tokens keys must be non-empty strings")
            if not isinstance(v, str) or not v.strip():
                raise ValueError("secrets.yaml agent_core_tokens values must be non-empty strings")
            result["agent_core_tokens"][k.strip()] = v.strip()
    else:
        result["agent_core_tokens"] = {}
    return result


def load_config() -> Config:
    secrets = _load_secrets_config("/run/deploy-approval/secrets.yaml")
    cos = secrets.get("cos", {})
    review_trust = secrets.get("review_comment_trust", {})
    cfg = Config(
        github_api_url="https://api.github.com",
        github_webhook_secret=os.getenv("GITHUB_WEBHOOK_SECRET", ""),
        webhook_enabled=_env_bool("WEBHOOK_ENABLED", False),
        poll_enabled=_env_bool("POLL_ENABLED", True),
        poll_interval_seconds=_env_int("POLL_INTERVAL_SECONDS", 30),
        github_repos=list(
            DEFAULT_GITHUB_REPOS if os.getenv("GITHUB_REPOS") is None
            else _env_str_list("GITHUB_REPOS")
        ),
        review_comment_author_id=review_trust.get("author_id", ""),
        review_comment_author_login=review_trust.get("author_login", ""),
        machine_owners_file="/run/deploy-approval/machines.yaml",
        secrets_file="/run/deploy-approval/secrets.yaml",
        allow_private_http=False,
        http_allowed_cidrs=[],
        connect_timeout=10.0,
        read_timeout=30.0,
        total_timeout=60.0,
        cos_region=cos.get("region", ""),
        cos_bucket=cos.get("bucket", ""),
        cos_secret_id=cos.get("secret_id", ""),
        cos_secret_key=cos.get("secret_key", ""),
        registry=os.getenv("REGISTRY", "").strip(),
        registry_auth_host_allowlist=[],
        agent_core_tokens=secrets.get("agent_core_tokens", {}),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    if not isinstance(cfg.github_repos, list):
        raise ValueError("GITHUB_REPOS must be a list")
    if not cfg.github_repos:
        raise ValueError("GITHUB_REPOS is required")
    if not cfg.poll_enabled:
        raise ValueError(
            "Deploy Approval requires polling; webhook does not replace the GitHubCommandWatcher"
        )
    for r in cfg.github_repos:
        if not isinstance(r, str):
            raise ValueError(f"GITHUB_REPOS member must be a string, got {r!r}")
    if len(cfg.github_repos) != len(set(cfg.github_repos)):
        raise ValueError("GITHUB_REPOS must not contain duplicates")
    required_repos = set(DEFAULT_GITHUB_REPOS)
    if (
        len(cfg.github_repos) != len(DEFAULT_GITHUB_REPOS)
        or set(cfg.github_repos) != required_repos
    ):
        raise ValueError(
            "GITHUB_REPOS must contain exactly: "
            + ", ".join(DEFAULT_GITHUB_REPOS)
        )
    if cfg.webhook_enabled and not cfg.github_webhook_secret:
        raise ValueError(
            "WEBHOOK_ENABLED is true but GITHUB_WEBHOOK_SECRET is empty"
        )
    if not isinstance(cfg.port, int) or isinstance(cfg.port, bool):
        raise ValueError(f"APP_PORT must be an int, got {cfg.port!r}")
    if cfg.port < 1 or cfg.port > 65535:
        raise ValueError(f"APP_PORT must be 1..65535, got {cfg.port}")
    if not isinstance(cfg.host, str):
        raise ValueError("APP_HOST must be a string")
    if not isinstance(cfg.http_allowed_cidrs, list):
        raise ValueError("HTTP_ALLOWED_CIDRS must be a list")
    _int_fields = {
        "POLL_INTERVAL_SECONDS": cfg.poll_interval_seconds,
        "MAX_RESPONSE_BYTES": cfg.max_response_bytes,
    }
    for name, v in _int_fields.items():
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"{name} must be an int, got {v!r}")
        if v <= 0:
            raise ValueError(f"{name} must be positive, got {v!r}")
    try:
        ipaddress.ip_address(cfg.host)
    except ValueError:
        pass
    for r in cfg.http_allowed_cidrs:
        if not isinstance(r, str):
            raise ValueError(f"HTTP_ALLOWED_CIDRS member must be a string, got {r!r}")
    for name in (
        "host", "machine_owners_file", "secrets_file",
        "github_api_url",
        "cos_region", "cos_bucket", "cos_secret_id", "cos_secret_key",
        "review_comment_author_id", "review_comment_author_login",
    ):
        val = getattr(cfg, name)
        if not isinstance(val, str):
            raise ValueError(f"{name} must be a string")
    if not isinstance(cfg.registry_auth_host_allowlist, list):
        raise ValueError("REGISTRY_AUTH_HOST_ALLOWLIST must be a list")
    # Registry trust anchor validation
    registry = cfg.registry
    if not registry:
        raise ValueError("REGISTRY is required — production Deploy Approval must configure REGISTRY")
    # Must not contain scheme
    if registry.startswith("https://") or registry.startswith("http://"):
        raise ValueError(f"REGISTRY must not include scheme, got {registry!r}")
    # Must not contain whitespace
    if any(c.isspace() for c in registry):
        raise ValueError(f"REGISTRY must not contain whitespace, got {registry!r}")
    # Must not contain userinfo
    if "@" in registry:
        raise ValueError(f"REGISTRY must not contain userinfo, got {registry!r}")
    # Must not contain path
    if "/" in registry:
        raise ValueError(f"REGISTRY must not contain path, got {registry!r}")
    # Must not contain query or fragment
    if "?" in registry or "#" in registry:
        raise ValueError(f"REGISTRY must not contain query/fragment, got {registry!r}")
    # Must have a valid host part (before any port)
    host_part = registry.split(":")[0]
    if not host_part:
        raise ValueError(f"REGISTRY host must not be empty, got {registry!r}")
    # Review comment trust: strict canonical identity check.
    # The authoritative production Review Agent identity is defined by
    # REVIEW_AGENT_GITHUB_USER_ID / REVIEW_AGENT_GITHUB_LOGIN.
    # Secrets.yaml review_comment_trust MUST match exactly.
    # This prevents deploying with a wrong trust configuration (e.g. Haohao-end).
    if cfg.review_comment_author_id != REVIEW_AGENT_GITHUB_USER_ID:
        raise ValueError(
            f"secrets.yaml review_comment_trust.author_id must be "
            f"{REVIEW_AGENT_GITHUB_USER_ID!r} (authoritative production "
            f"Review Agent GitHub user ID), got {cfg.review_comment_author_id!r}. "
            "Deploy Approval refuses to trust a Review Agent comment author "
            "that does not match the canonical identity."
        )
    if cfg.review_comment_author_login != REVIEW_AGENT_GITHUB_LOGIN:
        raise ValueError(
            f"secrets.yaml review_comment_trust.author_login must be "
            f"{REVIEW_AGENT_GITHUB_LOGIN!r} (authoritative production "
            f"Review Agent GitHub login), got {cfg.review_comment_author_login!r}. "
            "Deploy Approval refuses to trust a Review Agent comment author "
            "that does not match the canonical identity."
        )
    # Validate agent_core_tokens
    act = cfg.agent_core_tokens
    if not isinstance(act, Mapping):
        raise ValueError("agent_core_tokens must be a mapping")
    if not act:
        raise ValueError("agent_core_tokens must be non-empty")
    for k, v in act.items():
        if not isinstance(k, str) or not k.strip():
            raise ValueError("agent_core_tokens keys must be non-empty strings")
        if not isinstance(v, str) or not v.strip():
            raise ValueError("agent_core_tokens values must be non-empty strings")
