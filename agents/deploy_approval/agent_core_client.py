"""Client for Agent Core's ``{code, data, message}`` HTTP API.

Covers driver/perception/actucore deploy+status, the MCP ping health check and
the core ``POST /api/system/update`` adapter. No sockets, no SSH, no shell.
The runtime credential is a per-machine Agent Core Bearer token from
``secrets.yaml``.

TLS verification is disabled: the controller connects over HTTPS directly to
each machine's Agent Core without certificate pinning.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from .clients_common import (
    SecurityError,
    enforce_body_size,
    require_2xx,
    require_http_policy,
    stream_request,
)
from .config import Config
from .image_ref import validate_image_ref

logger = logging.getLogger(__name__)

# Control character range: ASCII 0x00-0x1F and 0x7F
_CONTROL_CHARS = frozenset(chr(i) for i in range(0x00, 0x20)) | {chr(0x7F)}


def _validate_api_path_segment(value: Any, label: str) -> str:
    """Validate a dynamic HTTP path segment (driver_id / mcp_id).

    Returns a sanitized string ready for use in a URL path segment.
    Raises AgentCoreError on any invalid input — this is a PRE-REQUEST
    validation failure, not a network outcome.
    """
    if not isinstance(value, str):
        raise AgentCoreError(f"{label} must be a string")
    stripped = value.strip()
    if not stripped:
        raise AgentCoreError(f"{label} must be non-empty after strip")
    if stripped in {".", ".."}:
        raise AgentCoreError(f"{label} must not be '.' or '..'")
    if "/" in stripped:
        raise AgentCoreError(f"{label} must not contain '/'")
    if "\\" in stripped:
        raise AgentCoreError(f"{label} must not contain '\\'")
    if "?" in stripped:
        raise AgentCoreError(f"{label} must not contain '?'")
    if "#" in stripped:
        raise AgentCoreError(f"{label} must not contain '#'")
    if "%" in stripped:
        raise AgentCoreError(f"{label} must not contain '%'")
    for ch in stripped:
        if ch in _CONTROL_CHARS:
            raise AgentCoreError(
                f"{label} must not contain control characters"
            )
    return stripped


class AgentCoreError(Exception):
    pass


class AgentCoreDeployOutcomeUncertain(AgentCoreError):
    pass


class AgentCoreClient:
    def __init__(
        self,
        config: Config,
        base_url: str,
        *,
        node_host: str,
        access_token: str = "",
        http: httpx.AsyncClient | None = None,
    ):
        self.config = config
        self._access_token = access_token
        if not base_url:
            raise AgentCoreError(
                "no Agent Core endpoint: the selected machine node_host endpoint is required"
            )
        if not isinstance(base_url, str) or not base_url.strip():
            raise AgentCoreError(
                "invalid Agent Core endpoint: must be a non-empty https URL"
            )
        import ipaddress
        # Defense-in-depth: when node_host is provided, validate it as a
        # literal IP address and confirm the URL scheme/host/port are exact.
        if not isinstance(node_host, str) or not node_host.strip():
            raise AgentCoreError("node_host is required")
        try:
            parsed_ip = ipaddress.ip_address(node_host)
        except ValueError as e:
            raise AgentCoreError(
                f"node_host must be a literal IP address, got {node_host!r}"
            ) from e
        if parsed_ip.version != 4:
            raise AgentCoreError(
                f"node_host must be an IPv4 address, got {node_host!r}"
            )
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        if parsed.scheme != "https":
            raise AgentCoreError(
                f"Agent Core endpoint scheme must be https, got {parsed.scheme!r}"
            )
        if parsed.hostname != node_host:
            raise AgentCoreError(
                f"Agent Core endpoint hostname must match node_host ({node_host}), "
                f"got {parsed.hostname!r}"
            )
        if parsed.port != 15678:
            raise AgentCoreError(
                f"Agent Core endpoint port must be 15678, got {parsed.port}"
            )
        if parsed.username or parsed.password:
            raise AgentCoreError("Agent Core endpoint must not contain userinfo")
        if parsed.path not in ("", "/"):
            raise AgentCoreError("Agent Core endpoint must not contain a path")
        if parsed.query:
            raise AgentCoreError("Agent Core endpoint must not contain query")
        if parsed.fragment:
            raise AgentCoreError("Agent Core endpoint must not contain fragment")
        expected_base = f"https://{node_host}:15678"
        if base_url.rstrip("/") != expected_base:
            raise AgentCoreError(
                "Agent Core endpoint must match the selected machine node_host exactly"
            )
        self.base_url = base_url.rstrip("/")
        self.node_host = node_host
        self._owns_http = http is None
        if http is None:
            self.http = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    config.total_timeout,
                    connect=config.connect_timeout,
                    read=config.read_timeout,
                    write=config.connect_timeout,
                    pool=config.connect_timeout,
                ),
                follow_redirects=False,
                verify=False,
                trust_env=False,
            )
        else:
            self.http = http

    async def aclose(self) -> None:
        if self._owns_http:
            await self.http.aclose()

    def _headers(self) -> dict:
        h = {}
        if self._access_token:
            h["Authorization"] = "Bearer " + self._access_token
        return h

    def _check_code(self, data: dict, method: str, path: str) -> None:
        """Fail-closed envelope check: ``code`` must be a *non-boolean* integer
        and only one of 0/200. A boolean is not an int here
        (``isinstance(True, int)`` is True in Python) so a true ``code`` is
        refused. The ``data`` payload type is NOT checked here — each endpoint
        adapter owns its exact shape (list for /api/drivers and /api/mcp,
        object for driver_status/mcp_ping)."""
        code = data.get("code")
        if isinstance(code, bool) or not isinstance(code, int):
            raise AgentCoreError(
                f"agent-core unexpected code {code!r} for {method} {path}"
            )
        if code not in (0, 200):
            raise AgentCoreError(
                f"agent-core unexpected code {code!r} for {method} {path}"
            )
        # The `data` payload shape belongs to each endpoint adapter. Endpoints
        # like GET /api/drivers and GET /api/mcp return `data` as an ARRAY,
        # while /api/drivers/<id>/status and /api/mcp ping return an OBJECT.
        # Requiring an object here would break the real Agent Core contract.

    async def request(
        self,
        method: str,
        path: str,
        json: dict | None = None,
    ):
        url = self.base_url + path
        try:
            require_http_policy(
                url, self.config, allow_private=self.config.allow_private_http,
                agent_core_node=self.node_host,
            )
        except SecurityError as e:
            raise AgentCoreError(str(e)) from e
        resp, data = await self._request_impl(method, path, json, url)
        self._check_code(data, method, path)
        return data

    async def _request_impl(self, method, path, json, url):
        try:
            resp = await stream_request(
                self.http, method, url, self.config.max_response_bytes,
                headers=self._headers(), json=json, timeout=self.config.total_timeout)
        except httpx.HTTPError as e:
            raise AgentCoreError(f"agent-core request failed: {e}")
        except SecurityError as e:
            raise AgentCoreError(str(e)) from e
        try:
            require_2xx(resp.status_code, f"agent-core {method} {path}")
        except SecurityError as e:
            raise AgentCoreError(str(e)) from e
        resp = await enforce_body_size(resp, self.config.max_response_bytes)
        try:
            data = resp.json()
        except ValueError:
            raise AgentCoreError("agent-core returned non-JSON")
        if not isinstance(data, dict):
            raise AgentCoreError("agent-core returned unexpected payload")
        return resp, data

    async def verify(self) -> dict:
        """Hit the auth-verify endpoint and return its raw `data`.

        Agent Core's `/api/auth/verify` is the one endpoint that is *not*
        wrapped in the `{code, data}` envelope — it returns the verdict directly
        (`{valid, auth_required}`), and returns HTTP 401 when the token is bad.
        Accept both shapes defensively but never treat a non-2xx as success.
        `valid` / `auth_required` must be real booleans (JSON true/false).
        """
        url = self.base_url + "/api/auth/verify"
        require_http_policy(
            url, self.config, allow_private=self.config.allow_private_http,
            agent_core_node=self.node_host,
        )
        try:
            resp = await stream_request(
                self.http, "GET", url, self.config.max_response_bytes,
                headers=self._headers(), timeout=self.config.total_timeout)
        except httpx.HTTPError as e:
            raise AgentCoreError(f"agent-core auth verify request failed: {e}")
        except SecurityError as e:
            raise AgentCoreError(str(e)) from e
        try:
            require_2xx(resp.status_code, "agent-core auth verify")
        except SecurityError as e:
            raise AgentCoreError(str(e)) from e
        resp = await enforce_body_size(resp, self.config.max_response_bytes)
        try:
            data = resp.json()
        except ValueError:
            raise AgentCoreError("agent-core auth verify returned non-JSON")
        if not isinstance(data, dict):
            raise AgentCoreError("agent-core auth verify unexpected payload")
        # The verify endpoint may be reached two ways: the documented raw
        # `{valid, auth_required}` body (preferred) or a generic `{code,data}`
        # wrapper. In the wrapped form, code must be 0/200 and data must be an
        # object whose `valid` is honoured fail-closed.
        if "code" in data:
            code = data.get("code")
            if isinstance(code, bool) or not isinstance(code, int) or code not in (200, 0):
                raise AgentCoreError(
                    f"agent-core auth verify unexpected code {code!r}"
                )
            inner = data.get("data")
            if not isinstance(inner, dict):
                raise AgentCoreError(
                    "agent-core auth verify envelope data is not an object"
                )
            data = inner
        # Strict booleans: JSON false/true only; a string "false" is refused.
        for key in ("valid", "auth_required"):
            v = data.get(key)
            if not isinstance(v, bool):
                raise AgentCoreError(
                    f"agent-core auth verify {key} must be a boolean"
                )
        # Fail-closed: authentication must be required AND the token valid. A
        # disabled or invalid auth (auth_required=false, or valid=false) is
        # never acceptable for a deployment request.
        if not (data.get("valid") is True and data.get("auth_required") is True):
            raise AgentCoreError(
                "agent-core auth verify did not return "
                "valid=true and auth_required=true"
            )
        return data
    async def list_drivers(self) -> list:
        data = await self.request("GET", "/api/drivers")
        drivers = data.get("data")
        if not isinstance(drivers, list):
            raise AgentCoreError("agent-core list_drivers data must be a list")
        return drivers

    async def list_mcp(self) -> list:
        data = await self.request("GET", "/api/mcp")
        mcps = data.get("data")
        if not isinstance(mcps, list):
            raise AgentCoreError("agent-core list_mcp data must be a list")
        return mcps

    async def deploy_driver(self, driver_id: str, image: str) -> dict:
        """POST the target image to the selected Agent Core driver.

        Accepts exact image:tag or legacy repo@sha256:<64hex>.
        The image string is validated but NOT modified — passed verbatim
        as {"image": image} to Agent Core.
        """
        if not isinstance(driver_id, str):
            raise AgentCoreError("deploy_driver requires a non-empty driver id")
        if not driver_id:
            raise AgentCoreError("deploy_driver requires a non-empty driver id")
        stripped = driver_id.strip()
        if not stripped:
            raise AgentCoreError("deploy_driver driver_id must be non-empty after strip")
        if " " in driver_id or "\t" in driver_id:
            raise AgentCoreError("deploy_driver driver_id must not contain whitespace")
        driver_id = _validate_api_path_segment(driver_id, "deploy_driver driver_id")
        try:
            validated_image = validate_image_ref(image)
        except ValueError as exc:
            raise AgentCoreError(str(exc)) from exc
        # validated_image == image — passed verbatim as {"image": image}
        path = f"/api/drivers/{driver_id}/deploy"
        url = self.base_url + path
        try:
            require_http_policy(
                url, self.config, allow_private=self.config.allow_private_http,
                agent_core_node=self.node_host,
            )
        except SecurityError as e:
            raise AgentCoreError(str(e)) from e
        try:
            resp = await stream_request(
                self.http, "POST", url, self.config.max_response_bytes,
                headers=self._headers(), json={"image": image}, timeout=self.config.total_timeout)
        except httpx.HTTPError as e:
            raise AgentCoreDeployOutcomeUncertain(
                f"agent-core deploy outcome uncertain: {e}"
            ) from e
        except SecurityError as e:
            raise AgentCoreDeployOutcomeUncertain(str(e)) from e
        try:
            require_2xx(resp.status_code, f"agent-core POST {path}")
        except SecurityError as e:
            raise AgentCoreDeployOutcomeUncertain(str(e)) from e
        # Body size check
        try:
            resp = await enforce_body_size(resp, self.config.max_response_bytes)
        except SecurityError as e:
            raise AgentCoreDeployOutcomeUncertain(str(e)) from e
        # Parse JSON envelope
        try:
            data = resp.json()
        except ValueError:
            raise AgentCoreDeployOutcomeUncertain(
                "agent-core deploy returned non-JSON"
            )
        if not isinstance(data, dict):
            raise AgentCoreDeployOutcomeUncertain(
                "agent-core deploy returned unexpected payload"
            )
        # --- Application-level classification once we have a valid 2xx envelope ---
        code = data.get("code")
        if isinstance(code, bool) or not isinstance(code, int):
            # Malformed envelope => uncertain
            raise AgentCoreDeployOutcomeUncertain(
                f"agent-core deploy malformed code {code!r}"
            )
        if code not in (0, 200):
            # Explicit application error (e.g. code=500) => CONFIRMED failure
            raise AgentCoreError(
                f"agent-core deploy failed: code={code!r}, message={data.get('message')!r}"
            )
        # code is 0 or 200 — inspect data payload
        inner = data.get("data")
        if not isinstance(inner, dict):
            raise AgentCoreDeployOutcomeUncertain(
                "agent-core deploy: code=200 but data is not an object"
            )
        status = inner.get("status")
        error_msg = inner.get("error", "")
        skipped = inner.get("skipped", False)
        if inner.get("status") == "error" or (isinstance(error_msg, str) and error_msg):
            # data.status=="error" or non-empty data.error => CONFIRMED failure
            raise AgentCoreError(
                f"agent-core deploy error: status={inner.get('status')!r}, error={error_msg!r}"
            )
        if skipped is True:
            # skipped=True is NOT a known-success deployment
            raise AgentCoreError(
                "agent-core deploy skipped=true; not a confirmed success"
            )
        # Known-success: must have a non-empty status string and no error/skipped
        if not isinstance(status, str) or not status:
            raise AgentCoreDeployOutcomeUncertain(
                "agent-core deploy success response missing status"
            )
        return data

    async def driver_status(self, driver_id: str) -> dict:
        driver_id = _validate_api_path_segment(driver_id, "driver_status driver_id")
        data = await self.request("GET", f"/api/drivers/{driver_id}/status")
        inner = data.get("data")
        if not isinstance(inner, dict):
            raise AgentCoreError(
                "agent-core driver_status data must be an object"
            )
        if "error" in inner:
            raise AgentCoreError(
                f"agent-core driver_status error payload: {inner.get('error')!r}"
            )
        if "running_image" in inner:
            running = inner.get("running_image")
            if not isinstance(running, str):
                raise AgentCoreError(
                    "agent-core driver_status running_image must be a string"
                )
            result: dict[str, Any] = {"running_image": running}
            logs = inner.get("logs")
            if isinstance(logs, str) and logs:
                result["logs"] = logs
            return result
        logs = inner.get("logs")
        if "status" in inner and isinstance(logs, str):
            return {"running_image": ""}
        raise AgentCoreError(
            "agent-core driver_status missing running_image for no-container shape"
        )



    async def mcp_ping(self, mcp_id: str) -> dict:
        mcp_id = _validate_api_path_segment(mcp_id, "mcp_ping mcp_id")
        data = await self.request("POST", f"/api/mcp/{mcp_id}/ping")
        inner = data.get("data")
        if not isinstance(inner, dict):
            raise AgentCoreError("agent-core mcp_ping data must be an object")
        online = inner.get("online")
        if not isinstance(online, bool):
            raise AgentCoreError("agent-core mcp_ping online must be a boolean")
        tools = inner.get("tools")
        if tools is None:
            inner["tools"] = []
        elif isinstance(tools, list):
            # Strict schema, never coerced: a non-dict tool (number/string/null
            # /... ) is a protocol violation, not a tool with a stringified
            # name. The service defense-in-depth validates names separately.
            normalized = []
            for item in tools:
                if not isinstance(item, dict):
                    raise AgentCoreError(
                        "agent-core mcp_ping tools must be a list of objects"
                    )
                normalized.append(dict(item))
            inner["tools"] = normalized
        else:
            raise AgentCoreError(
                "agent-core mcp_ping tools must be a list or null"
            )
        return inner
