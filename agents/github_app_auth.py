"""Shared GitHub App authentication provider.

Provides:
  1. App JWT (RS256) — memory-only, short-lived.
  2. Installation access token — fixed installation ID, parsed expires_at,
     refreshed ~5 min before expiry, async single-flight.

Deploy Approval uses this provider. Review Agent uses its own GITHUB_TOKEN and does NOT use this provider.
No key/token leaks into logs, repr, errors, or evidence.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# Maximum GitHub App JWT lifetime is 10 minutes.
_MAX_JWT_SECONDS = 600
# Refresh installation token this many seconds before expiry.
_REFRESH_BEFORE_SECONDS = 300


class GitHubAppAuthError(Exception):
    """Base error for GitHub App authentication failures."""


class GitHubAppAuth:
    """Single-flight GitHub App JWT + installation access token provider.

    Parameters
    ----------
    app_id : str
        The GitHub App numeric ID.
    installation_id : str
        The fixed GitHub App installation ID.
    private_key : bytes
        The RS256 private key material (never persisted).
    http : httpx.AsyncClient | None
        Optional shared HTTP client. A private client is created when None.
    """

    def __init__(
        self,
        app_id: str,
        installation_id: str,
        private_key: bytes,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        if not isinstance(app_id, str) or not app_id.strip():
            raise GitHubAppAuthError("app_id is required")
        if not isinstance(installation_id, str) or not installation_id.strip():
            raise GitHubAppAuthError("installation_id is required")
        if not isinstance(private_key, bytes) or not private_key:
            raise GitHubAppAuthError("private_key bytes are required")

        self._app_id = app_id.strip()
        self._installation_id = installation_id.strip()
        self._private_key = private_key
        self._http = http
        self._owns_http = http is None

        # JWT cache
        self._jwt_token: str = ""
        self._jwt_exp: float = 0.0

        # Installation token cache
        self._install_token: str = ""
        self._install_expires_at: float = 0.0

        # Single-flight lock for token refresh
        self._lock: asyncio.Lock = asyncio.Lock()

        # HTTP client setup
        if self._owns_http:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0, read=15.0),
                follow_redirects=False,
                trust_env=False,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_installation_token(self) -> str:
        """Return a valid installation access token.

        Returns the cached token if not expiring within the safety window,
        otherwise refreshes atomically under a single-flight lock.
        """
        now = time.time()
        if self._install_token and now < (self._install_expires_at - _REFRESH_BEFORE_SECONDS):
            return self._install_token

        async with self._lock:
            # Double-check inside lock
            now = time.time()
            if self._install_token and now < (self._install_expires_at - _REFRESH_BEFORE_SECONDS):
                return self._install_token
            return await self._refresh_installation_token()

    @property
    def app_id(self) -> str:
        """Public read-only access to the GitHub App ID."""
        return self._app_id

    @property
    def installation_id(self) -> str:
        """Public read-only access to the installation ID."""
        return self._installation_id

    def jwtBearerToken(self) -> str:
        """Return a valid App JWT (RS256).

        Regenerates when expired. Memory-only, never logged.
        """
        now = time.time()
        if self._jwt_token and now < (self._jwt_exp - 5):
            return self._jwt_token
        self._jwt_token = self._generate_jwt()
        return self._jwt_token

    async def close(self) -> None:
        """Close the owned HTTP client if present."""
        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None

    # ------------------------------------------------------------------
    # Internal: JWT
    # ------------------------------------------------------------------

    def _generate_jwt(self) -> str:
        now = int(time.time())
        payload = {
            "iat": now - 60,  # GitHub official recommendation: 60s clock skew
            "exp": now + _MAX_JWT_SECONDS,
            "iss": self._app_id,
        }
        header = {"alg": "RS256", "typ": "JWT"}

        b64_header = _b64url(json.dumps(header, separators=(",", ":")).encode())
        b64_payload = _b64url(json.dumps(payload, separators=(",", ":")).encode())
        signing_input = f"{b64_header}.{b64_payload}"

        import hashlib as _hashlib
        import struct as _struct
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding
        except ImportError:
            raise GitHubAppAuthError(
                "cryptography package is required for GitHub App JWT"
            )

        private_key = serialization.load_pem_private_key(self._private_key, password=None)
        signature = private_key.sign(signing_input.encode(), padding.PKCS1v15(), hashes.SHA256())

        return f"{signing_input}.{_b64url(signature)}"

    # ------------------------------------------------------------------
    # Internal: Installation token
    # ------------------------------------------------------------------

    async def _refresh_installation_token(self) -> str:
        jwt = self.jwtBearerToken()
        url = (
            f"https://api.github.com/app/installations/"
            f"{self._installation_id}/access_tokens"
        )

        try:
            resp = await self._http.post(
                url,
                headers={
                    "Authorization": f"Bearer {jwt}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2026-03-10",
                },
                timeout=30.0,
            )
        except httpx.HTTPError as e:
            raise GitHubAppAuthError(f"installation token refresh failed: {e}") from e

        if resp.status_code not in (200, 201):
            raise GitHubAppAuthError(
                f"installation token refresh returned {resp.status_code}"
            )

        data = resp.json()
        if not isinstance(data, dict):
            raise GitHubAppAuthError("installation token refresh returned non-object")

        token = data.get("token")
        if not isinstance(token, str) or not token:
            raise GitHubAppAuthError("installation token refresh missing token")

        expires_at_str = data.get("expires_at")
        expires_at = self._parse_expires_at(expires_at_str)

        self._install_token = token
        self._install_expires_at = expires_at
        return token

    @staticmethod
    def _parse_expires_at(expires_at_str: object) -> float:
        """Parse GitHub's ISO8601 expires_at into Unix timestamp."""
        if not isinstance(expires_at_str, str) or not expires_at_str:
            raise GitHubAppAuthError("installation token expires_at is required")
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                raise GitHubAppAuthError("installation token expires_at must be timezone-aware")
            expires_at = dt.timestamp()
        except (TypeError, ValueError):
            raise GitHubAppAuthError("installation token expires_at is malformed")
        if expires_at <= time.time():
            raise GitHubAppAuthError("installation token expires_at is expired")
        return expires_at

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Dunder / repr safety
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<GitHubAppAuth app_id={self._app_id!r} "
            f"installation_id={self._installation_id!r}>"
        )


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

def load_private_key_from_file(file_path: str) -> bytes:
    """Read and validate a GitHub App private key file.

    Checks:
      - file exists, is a regular file
      - is readable
      - is not a symlink
      - does not contain placeholder markers
    """
    path = Path(file_path)

    if not isinstance(file_path, str) or not file_path.strip():
        raise GitHubAppAuthError("GITHUB_APP_PRIVATE_KEY_FILE path is required")

    if path.is_symlink():
        raise GitHubAppAuthError(
            "GITHUB_APP_PRIVATE_KEY_FILE must not be a symlink"
        )
    if not path.exists():
        raise GitHubAppAuthError(
            f"GITHUB_APP_PRIVATE_KEY_FILE does not exist: {file_path}"
        )
    if not path.is_file():
        raise GitHubAppAuthError(
            f"GITHUB_APP_PRIVATE_KEY_FILE must be a regular file: {file_path}"
        )
    try:
        key_data = path.read_bytes()
    except OSError as e:
        raise GitHubAppAuthError(
            f"GITHUB_APP_PRIVATE_KEY_FILE not readable: {e}"
        ) from e

    if not key_data:
        raise GitHubAppAuthError(
            "GITHUB_APP_PRIVATE_KEY_FILE is empty"
        )

    # Basic sanity check — should contain PEM markers
    key_text = key_data.decode("utf-8", errors="replace")
    if "BEGIN" not in key_text or "PRIVATE KEY" not in key_text:
        raise GitHubAppAuthError(
            "GITHUB_APP_PRIVATE_KEY_FILE does not appear to contain a PEM private key"
        )

    return key_data


def create_github_app_auth() -> GitHubAppAuth:
    """Create a GitHubAppAuth instance from environment variables.

    Required ENV:
      GITHUB_APP_ID
      GITHUB_INSTALLATION_ID
      GITHUB_APP_PRIVATE_KEY_FILE
    """
    app_id = os.getenv("GITHUB_APP_ID", "").strip()
    installation_id = os.getenv("GITHUB_INSTALLATION_ID", "").strip()
    key_file = os.getenv("GITHUB_APP_PRIVATE_KEY_FILE", "").strip()

    if not app_id:
        raise GitHubAppAuthError("GITHUB_APP_ID is required")
    if not installation_id:
        raise GitHubAppAuthError("GITHUB_INSTALLATION_ID is required")
    if not key_file:
        raise GitHubAppAuthError("GITHUB_APP_PRIVATE_KEY_FILE is required")

    private_key = load_private_key_from_file(key_file)

    return GitHubAppAuth(
        app_id=app_id,
        installation_id=installation_id,
        private_key=private_key,
    )


def require_github_app_env() -> None:
    """Validate that all required GitHub App ENV vars are set."""
    for name in ("GITHUB_APP_ID", "GITHUB_INSTALLATION_ID", "GITHUB_APP_PRIVATE_KEY_FILE"):
        val = os.getenv(name)
        if not val or not val.strip():
            raise GitHubAppAuthError(f"{name} is required")




# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _b64url(data: bytes) -> str:
    """Base64url encode without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")
