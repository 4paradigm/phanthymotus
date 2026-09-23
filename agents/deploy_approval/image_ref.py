"""Image reference validation for Deploy Approval.

Deploy Approval no longer accesses the Registry.  Image references come
from the trusted Review Agent GitHub comment.  This module provides a
single validation gate so that only syntactically valid image:tag or
legacy repo@sha256 digest references reach Agent Core or hidden state.
"""

from __future__ import annotations

import re

_DEPLOY_PLATFORM = "linux/arm64"

# --- Regexes ---

# Authority: [a-z0-9._-]+ possibly followed by :<port>
# Port, if present, must be all digits and 1..65535
_AUTHORITY_RE = re.compile(r"^[A-Za-z0-9._-]+(?::[0-9]+)?$")

# Repository path separator: only period and underscore are valid internal separators.
# Asterisk, plus, percent, and other characters are rejected.

# Full repository path: one or more segments joined by /
# Must not start or end with /, must not contain //
_REPO_PATH_RE = re.compile(
    r"^[a-z0-9]+(?:(?:[._]|__+|[-]+)[a-z0-9]+)*(?:/[a-z0-9]+(?:(?:[._]|__+|[-]+)[a-z0-9]+)*)*$"
)

# Tag: non-empty, no whitespace, limited
_TAG_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$")

# Digest: sha256:<64 lowercase hex>
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _validate_port(port_str: str) -> None:
    """Validate that a port string is digits and in range 1..65535."""
    if not port_str.isdigit():
        raise ValueError(f"invalid port {port_str!r}")
    port = int(port_str)
    if port < 1 or port > 65535:
        raise ValueError(f"port must be 1-65535, got {port}")


def _validate_authority(authority: str) -> None:
    """Validate the registry authority portion."""
    if not authority:
        raise ValueError("authority must not be empty")
    if not _AUTHORITY_RE.fullmatch(authority):
        raise ValueError(f"invalid authority {authority!r}")
    if ":" in authority:
        _, port_str = authority.rsplit(":", 1)
        _validate_port(port_str)


def _validate_repository_path(repo_path: str) -> None:
    """Validate the repository path portion (lowercase, OCI-compatible)."""
    if not repo_path:
        raise ValueError("repository path must not be empty")
    # Reject // (double slash)
    if "//" in repo_path:
        raise ValueError("repository path must not contain //")
    # Reject trailing /
    if repo_path.endswith("/"):
        raise ValueError("repository path must not end with /")
    if not _REPO_PATH_RE.fullmatch(repo_path):
        raise ValueError(f"invalid repository path {repo_path!r}")


def validate_image_ref(value: str) -> str:
    """Validate and return the exact image reference string.

    Accepts:
      - Full image:tag  (e.g. registry.example/ns/repo:tag)
      - With port       (e.g. registry.example:5000/ns/repo:tag)
      - Legacy digest   (e.g. registry.example/ns/repo@sha256:<64hex>)

    Raises ValueError on any malformed input.

    Guarantees:
      - Returns the original value unchanged (validated_image == image)
      - Rejects ALL whitespace (space, tab, newline, CR, NBSP, etc.)
      - Rejects leading/trailing whitespace
      - Rejects URL schemes, backslash, query, fragment, backslash
      - Authority is syntax-validated with optional port range
      - Repository path uses OCI-compatible lowercase rules
      - Tag must be present, non-empty, non-'latest', syntactically valid
      - Digest must be sha256:<64 lowercase hex> only
    """
    if not isinstance(value, str):
        raise ValueError("image reference must be a string")

    # Reject empty
    if not value:
        raise ValueError("image reference must not be empty")

    # Reject ANY whitespace character (space, tab, newline, CR, NBSP, etc.)
    for ch in value:
        if ch.isspace():
            raise ValueError(
                "image reference must not contain whitespace"
            )

    # No leading/trailing whitespace (redundant with above, but explicit)
    stripped = value.strip()
    if stripped != value:
        raise ValueError("image reference must not have leading/trailing whitespace")

    # No control characters (belt and braces)
    for ch in value:
        if ord(ch) < 32 or ord(ch) == 127:
            raise ValueError("image reference must not contain control characters")

    # No URL schemes
    if "http://" in value or "https://" in value:
        raise ValueError("image reference must not contain a URL scheme")

    # No backslash, query, fragment
    if "\\" in value or "?" in value or "#" in value:
        raise ValueError("image reference must not contain backslash, query, or fragment")

    # Must contain /
    if "/" not in value:
        raise ValueError("image reference must be a full repository reference (contains /)")

    # Check for digest form first
    if "@" in value:
        # Only one @ allowed
        if value.count("@") > 1:
            raise ValueError("image reference must not contain multiple @")
        at_idx = value.index("@")
        repo_part = value[:at_idx]
        digest = value[at_idx + 1:]
        if not repo_part:
            raise ValueError("repository part of digest reference must not be empty")
        if not _DIGEST_RE.fullmatch(digest):
            raise ValueError(f"invalid digest reference {value!r}")
        # Validate authority and repository path
        _validate_full_reference(repo_part)
        return value

    # Tag form: must have : after the last /
    last_slash = value.rfind("/")
    if ":" not in value[last_slash:]:
        raise ValueError("image reference must have a tag or digest (no : or @ found)")

    colon_idx = value.rindex(":")
    repo_part = value[:colon_idx]
    tag = value[colon_idx + 1:]

    if not repo_part:
        raise ValueError("repository part must not be empty")
    if not tag:
        raise ValueError("tag must not be empty")
    if tag == "latest":
        raise ValueError("tag 'latest' is not allowed")
    if not _TAG_RE.fullmatch(tag):
        raise ValueError(f"invalid tag {tag!r}")

    # Validate authority and repository path
    _validate_full_reference(repo_part)

    return value


def _validate_full_reference(ref: str) -> None:
    """Validate authority/repository portion of an image reference.

    ref is everything before :tag or @digest.
    Examples:
      registry.example/ns/repo
      registry.example:5000/ns/repo
    """
    slash_idx = ref.find("/")
    if slash_idx < 0:
        # No slash — entire ref is authority (but we already required "/" in the full ref)
        # This shouldn't happen because we checked "/" above, but be safe
        raise ValueError("image reference must contain /")

    authority = ref[:slash_idx]
    repo_path = ref[slash_idx + 1:]

    _validate_authority(authority)
    _validate_repository_path(repo_path)


def get_deploy_platform() -> str:
    """Return the platform used for new component snapshots.

    This is a policy-derived value, NOT a registry-resolved platform.
    Retained for hidden-state schema compatibility.
    """
    return _DEPLOY_PLATFORM
