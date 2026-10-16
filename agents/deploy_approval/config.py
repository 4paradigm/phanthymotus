"""Configuration for the Deploy Approval Agent (pre-merge validation only).

Deploy Approval does not define a new runtime env namespace. It reuses the
existing upstream GitHub/poll/webhook keys and fixed read-only files for
machine ownership, Review Agent trust, Agent Core tokens, and COS secrets.

Review Agent HTTP API is NOT a Deploy Approval dependency.
Deploy Approval reads Review Agent output from GitHub PR comments.

Per-machine Agent Core CA PEM trust anchors are strictly validated: keys must be
non-empty machine-alias strings and values must be valid PEM certificate blobs.
Invalid entries raise ValueError instead of being silently ignored.
"""

from __future__ import annotations

from collections.abc import Mapping
import logging
import os
import ipaddress
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


DEFAULT_GITHUB_REPOS = (
    "4paradigm/phanthymotus",
    "4paradigm/phanthymotus-driver",
)


# Compatibility names for the legacy defaults, not runtime authorization gates.
SUPPORTED_GITHUB_REPOS = frozenset(DEFAULT_GITHUB_REPOS)
DESIRED_REPOS = DEFAULT_GITHUB_REPOS

_OWNER_RE = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?"
_REPOSITORY_RE = re.compile(rf"{_OWNER_RE}/[A-Za-z0-9_.-]{{1,100}}")
_LOGIN_RE = re.compile(rf"{_OWNER_RE}(?:\[bot\])?")


def validate_repository_names(repos: list[str], *, label: str = "GITHUB_REPOS", allow_empty: bool = False) -> list[str]:
    """Validate a controlled list and canonicalize GitHub's case-insensitive names."""
    if not isinstance(repos, list):
        raise ValueError(f"{label} must be a list")
    if not repos and not allow_empty:
        raise ValueError(f"{label} is required")
    result = []
    for repo in repos:
        if (not isinstance(repo, str) or not _REPOSITORY_RE.fullmatch(repo)
                or repo.split("/")[-1] in (".", "..")
                or "--" in repo.split("/")[0]):
            raise ValueError(f"{label} contains an invalid owner/repo")
        canonical = repo.lower()
        if canonical in result:
            raise ValueError(f"{label} must not contain duplicates")
        result.append(canonical)
    return result


def resolve_authorized_repos(desired: list[str], authorized: list) -> list[str]:
    """Intersect the configured allowlist with a complete installation snapshot."""
    desired = validate_repository_names(desired)
    if not isinstance(authorized, list):
        raise ValueError("installation repositories must be a list")
    names = [entry.get("full_name") if isinstance(entry, dict) else entry for entry in authorized]
    authorized_set = set(validate_repository_names(names, label="installation repositories", allow_empty=True))
    active = [repo for repo in desired if repo in authorized_set]
    if desired[0] not in active:
        raise RuntimeError(f"required repo {desired[0]!r} is not authorized for this GitHub App installation")
    return active


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
    active_repos: list[str] = field(default_factory=list)
    auth_valid: bool = False
    github_comment_max_pages: int = 20
    github_comment_max_comments: int = 500
    github_comment_max_bytes: int = 4 * 1024 * 1024

    # Review comment trust configuration
    # Production loads BOTH values exclusively from protected secrets.yaml.
    # There is no fallback identity or environment override.
    review_comment_author_id: str = ""
    review_comment_author_login: str = ""

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



    # Missing browser credentials disables only the optional evidence feature.
    deploy_evidence_enabled: bool = False
    evidence_public_base_url: str = ""
    evidence_oauth_client_id: str = ""
    evidence_oauth_client_secret: str = field(default="", repr=False)
    evidence_session_key: str = field(default="", repr=False)
    evidence_max_file_bytes: int = 512 * 1024 * 1024
    evidence_max_video_bytes: int = 512 * 1024 * 1024
    evidence_part_bytes: int = 8 * 1024 * 1024
    evidence_max_note_bytes: int = 64 * 1024

    # Agent Core access tokens keyed by machine alias
    agent_core_tokens: dict[str, str] = field(
        default_factory=dict,
        repr=False,
    )

    # Explicit deployment environment opt-in; strict never downgrades.
    agent_core_tls_mode: str = "strict"

    # Per-machine CA PEM trust anchors keyed by machine alias (strict mode)
    agent_core_ca_pem: dict[str, str] = field(
        default_factory=dict,
        repr=False,
    )

    @property
    def primary_repo(self) -> str:
        return validate_repository_names(self.github_repos)[0]


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
    if p.is_symlink():
        raise ValueError(f"{path} must be a protected regular file")
    if not p.exists():
        return {}
    # Trust comes from the protected server file, not a comment or environment.
    # Inspect the opened descriptor so replacing the pathname cannot change
    # the object whose ownership/permissions were checked.
    try:
        fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise ValueError(f"{path} must be a protected regular file") from exc
    with os.fdopen(fd, "r", encoding="utf-8") as fh:
        info = os.fstat(fh.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid not in (0, os.getuid())
                or stat.S_IMODE(info.st_mode) & 0o022):
            raise ValueError(f"{path} must be a protected regular file")
        try:
            data = yaml.safe_load(fh)
        except yaml.YAMLError:
            # Parser diagnostics can quote a line containing a token or key.
            raise ValueError(f"{path} must contain valid protected YAML") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def validate_agent_core_tls_mode(value: str) -> str:
    if not isinstance(value, str) or value not in ("legacy", "strict"):
        raise ValueError("AGENT_CORE_TLS_MODE must be legacy or strict")
    return value


def _load_ca_pem_config(secrets: dict) -> dict[str, str]:
    """Load per-machine CA PEM trust anchors from secrets config."""
    ca_pem = secrets.get("agent_core_ca_pem", {})
    if not isinstance(ca_pem, dict):
        raise ValueError("secrets.yaml agent_core_ca_pem must be a mapping")
    result = {}
    for alias, pem in ca_pem.items():
        if not isinstance(alias, str) or not alias.strip():
            raise ValueError("agent_core_ca_pem machine aliases must be non-empty strings")
        alias = alias.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", alias):
            raise ValueError("agent_core_ca_pem contains an invalid machine alias")
        if alias in result:
            raise ValueError("agent_core_ca_pem contains duplicate normalized machine aliases")
        if not isinstance(pem, str) or not pem.strip():
            raise ValueError(f"agent_core_ca_pem for machine {alias!r} must be non-empty PEM")
        try:
            import ssl
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.load_verify_locations(cadata=pem.strip())
        except (ssl.SSLError, ValueError, TypeError) as exc:
            raise ValueError(f"agent_core_ca_pem for machine {alias!r} is invalid") from exc
        result[alias] = pem.strip()
    return result


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

    # Browser evidence is optional. Malformed or incomplete evidence values
    # disable that feature without making the Controller configuration fail.
    evidence_fields = (
        "public_base_url", "oauth_client_id", "oauth_client_secret", "session_key",
    )
    evidence = data.get("evidence", {})
    evidence_values = {key: "" for key in evidence_fields}
    evidence_enabled = isinstance(evidence, dict)
    if evidence_enabled:
        for key in evidence_fields:
            value = evidence.get(key)
            if (not isinstance(value, str) or not value.strip()
                    or any(ord(char) < 32 or ord(char) == 127 for char in value)):
                evidence_enabled = False
                break
            evidence_values[key] = value
    if not evidence_enabled:
        evidence_values = {key: "" for key in evidence_fields}
    result["evidence"] = evidence_values
    result["deploy_evidence_enabled"] = evidence_enabled

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
    # Supplied CA material remains validated in either mode.
    result["agent_core_ca_pem"] = _load_ca_pem_config(data)
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
    evidence = secrets.get("evidence", {})
    review_trust = secrets.get("review_comment_trust", {})
    cfg = Config(
        github_api_url="https://api.github.com",
        github_webhook_secret=os.getenv("GITHUB_WEBHOOK_SECRET", ""),
        webhook_enabled=_env_bool("WEBHOOK_ENABLED", False),
        poll_enabled=_env_bool("POLL_ENABLED", True),
        poll_interval_seconds=_env_int("POLL_INTERVAL_SECONDS", 30),
        github_repos=list(
            DEFAULT_GITHUB_REPOS if os.getenv("GITHUB_REPOS") is None
            else ([] if not os.environ["GITHUB_REPOS"].strip()
                  else [repo.strip() for repo in os.environ["GITHUB_REPOS"].split(",")])
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
        deploy_evidence_enabled=secrets.get("deploy_evidence_enabled", False),
        evidence_public_base_url=evidence.get("public_base_url", "").rstrip("/"),
        evidence_oauth_client_id=evidence.get("oauth_client_id", ""),
        evidence_oauth_client_secret=evidence.get("oauth_client_secret", ""),
        evidence_session_key=evidence.get("session_key", ""),
        agent_core_tokens=secrets.get("agent_core_tokens", {}),
        agent_core_tls_mode=validate_agent_core_tls_mode(os.getenv("AGENT_CORE_TLS_MODE", "strict")),
        agent_core_ca_pem=secrets.get("agent_core_ca_pem", {}),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    for name, lower, upper in (("evidence_max_file_bytes", 1, 512 * 1024 * 1024),
                               ("evidence_max_video_bytes", 1, 512 * 1024 * 1024),
                               ("evidence_part_bytes", 1024 * 1024, 512 * 1024 * 1024),
                               ("evidence_max_note_bytes", 1, 64 * 1024)):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
            raise ValueError(f"{name} is outside the evidence size bounds")
    cfg.github_repos = validate_repository_names(cfg.github_repos)
    if not cfg.poll_enabled:
        raise ValueError(
            "Deploy Approval requires polling; webhook does not replace the GitHubCommandWatcher"
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
    validate_agent_core_tls_mode(cfg.agent_core_tls_mode)
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
    ca_map = cfg.agent_core_ca_pem
    if not isinstance(ca_map, Mapping):
        raise ValueError("agent_core_ca_pem must be a mapping")
    # Revalidate direct Config construction as well as secrets.yaml loading.
    normalized_ca_map = _load_ca_pem_config({"agent_core_ca_pem": ca_map})
    if len(normalized_ca_map) != len(ca_map):
        raise ValueError("agent_core_ca_pem contains duplicate normalized machine aliases")
    cfg.agent_core_ca_pem = normalized_ca_map
    # ── Production COS guard: reject invalid region, bucket, placeholders ──
    # This prevents the Controller from starting with obviously wrong or
    # unfilled-in COS configuration, even if someone bypasses deploy.sh.
    # No len-based bypass: all values are strictly validated.
    # No empty/whitespace-only values are allowed.
    _placeholder_re = re.compile(r"<[^>]+>", re.IGNORECASE)
    for field_name in ("cos_region", "cos_bucket", "cos_secret_id", "cos_secret_key"):
        val = getattr(cfg, field_name, "")
        if not isinstance(val, str) or not val.strip():
            raise ValueError(
                f"cos.{field_name} must be a non-empty string"
            )
        if _placeholder_re.search(val):
            raise ValueError(
                f"cos.{field_name} contains an unreplaced placeholder"
            )
    # Region: must match Tencent COS pattern (e.g. "ap-beijing")
    if not re.fullmatch(r"[a-z]+-[a-z]+[0-9]*", cfg.cos_region.strip()):
        raise ValueError(
            f"cos.region has invalid format: {cfg.cos_region!r}"
        )
    # Bucket: must be alphanumeric with hyphens
    if not re.fullmatch(r"[a-z0-9][a-z0-9\-]*", cfg.cos_bucket.strip()):
        raise ValueError(
            f"cos.bucket has invalid format: {cfg.cos_bucket!r}"
        )
    if not re.fullmatch(r"[1-9][0-9]*", cfg.review_comment_author_id):
        raise ValueError("review_comment_trust.author_id must be a positive decimal string")
    if not _LOGIN_RE.fullmatch(cfg.review_comment_author_login):
        raise ValueError("review_comment_trust.author_login must be a valid GitHub login")
