"""Client tests: agent-core envelope checks, review client, common transport."""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

import httpx

from ..agent_core_client import AgentCoreClient, AgentCoreDeployOutcomeUncertain, AgentCoreError
from ..config import Config, DEFAULT_GITHUB_REPOS, validate_config
from ..models import MachineInfo
from ..policy import Policy
from ..service import DeployController, DeployControllerError
from .conftest import make_config


class _Transport(httpx.AsyncBaseTransport):
    """In-memory httpx async transport returning a canned response."""

    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    async def handle_async_request(self, request):
        return httpx.Response(
            self.status,
            json=self.payload,
            request=request,
        )


async def fake_token_provider():
    return "fake-installation-token"


class _TransportFromScenarios(httpx.AsyncBaseTransport):
    """Routes each request path to a canned transport so one client can serve
    multiple Agent Core endpoints."""

    def __init__(self, transports):
        self.transports = transports

    async def handle_async_request(self, request):
        tr = self.transports.get(request.url.path, self.transports.get(""))
        if tr is None:
            raise AssertionError(f"no transport for {request.url.path}")
        return await tr.handle_async_request(request)


def _client():
    return make_config(allow_private_http=False)


def _agent_core_client(
    transport: httpx.AsyncBaseTransport,
    *,
    base_url: str = "https://192.0.2.1:15678",
    node_host: str = "192.0.2.1",
) -> AgentCoreClient:
    return AgentCoreClient(
        _client(),
        base_url=base_url,
        node_host=node_host,
        http=httpx.AsyncClient(transport=transport),
    )


def _open_pr(number: int, updated_at: str, *, marker: str = "") -> dict:
    pr = {"number": number, "updated_at": updated_at}
    if marker:
        pr["marker"] = marker
    return pr


@pytest.mark.parametrize(
    "github_repos, should_pass, expected_error",
    [
        ([], False, "GITHUB_REPOS is required"),
        (["4paradigm/phanthymotus"], False, "GITHUB_REPOS must contain exactly"),
        (["4paradigm/phanthymotus-driver"], False, "GITHUB_REPOS must contain exactly"),
        (["4paradigm/phanthymotus", "4paradigm/phanthymotus"], False, "GITHUB_REPOS must not contain duplicates"),
        (["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"], True, ""),
        (["4paradigm/phanthymotus-driver", "4paradigm/phanthymotus"], True, ""),
        (["4paradigm/phanthymotus", "some/other-repo"], False, "GITHUB_REPOS must contain exactly"),
        (["example/unsupported-repo"], False, "GITHUB_REPOS must contain exactly"),
    ],
)
def test_validate_config_requires_exact_runtime_repo_set(github_repos, should_pass, expected_error):
    cfg = Config(
        github_repos=github_repos,
        poll_enabled=True,
        github_webhook_secret="secret",
        review_comment_author_id="7950763",
        agent_core_tokens={"test-machine": "test-token"},
    )
    if should_pass:
        validate_config(cfg)
    else:
        with pytest.raises(ValueError, match=expected_error):
            validate_config(cfg)


def test_agent_core_endpoint_policy_requires_https_exact_ipv4():
    cfg = _client()
    ok = AgentCoreClient(
        cfg,
        base_url="https://10.0.0.1:15678",
        node_host="10.0.0.1",
        http=httpx.AsyncClient(transport=_Transport({"valid": True, "auth_required": True})),
    )
    assert ok.base_url == "https://10.0.0.1:15678"

    bad_cases = [
        ("http://10.0.0.1:15678", "10.0.0.1", "scheme"),
        ("https://10.0.0.2:15678", "10.0.0.1", "hostname"),
        ("https://robot.local:15678", "10.0.0.1", "hostname"),
        ("https://10.0.0.1:15679", "10.0.0.1", "port"),
        ("https://user:pass@10.0.0.1:15678", "10.0.0.1", "userinfo"),
        ("https://10.0.0.1:15678/api", "10.0.0.1", "path"),
        ("https://10.0.0.1:15678?x=1", "10.0.0.1", "query"),
        ("https://10.0.0.1:15678#x", "10.0.0.1", "fragment"),
        ("https://10.0.0.1:15678", "robot.local", "literal IP"),
    ]
    for base_url, node_host, message in bad_cases:
        with pytest.raises(AgentCoreError, match=message):
            AgentCoreClient(
                cfg,
                base_url=base_url,
                node_host=node_host,
                http=httpx.AsyncClient(transport=_Transport({"valid": True, "auth_required": True})),
            )


def test_agent_core_constructor_requires_node_host():
    cfg = _client()
    with pytest.raises(TypeError):
        AgentCoreClient(
            cfg,
            base_url="https://192.0.2.1:15678",
            http=httpx.AsyncClient(transport=_Transport({"code": 200, "data": {}})),
        )
    with pytest.raises(AgentCoreError, match="node_host is required"):
        AgentCoreClient(
            cfg,
            base_url="https://192.0.2.1:15678",
            node_host="",
            http=httpx.AsyncClient(transport=_Transport({"code": 200, "data": {}})),
        )


def test_agent_core_constructor_has_no_ca_file_parameter():
    sig = inspect.signature(AgentCoreClient.__init__)
    assert "ca_file" not in sig.parameters


def test_agent_core_self_created_client_disables_proxy_env(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://bad proxy")
    client = AgentCoreClient(
        _client(),
        base_url="https://192.0.2.1:15678",
        node_host="192.0.2.1",
    )
    assert client.http.trust_env is False



def test_agent_core_aclose_owns_only_self_created_http():
    owned = AgentCoreClient(
        _client(),
        base_url="https://192.0.2.1:15678",
        node_host="192.0.2.1",
    )
    asyncio.run(owned.aclose())
    asyncio.run(owned.aclose())
    assert owned.http.is_closed

    external = httpx.AsyncClient(transport=_Transport({"code": 200, "data": {}}))
    injected = AgentCoreClient(
        _client(),
        base_url="https://192.0.2.1:15678",
        node_host="192.0.2.1",
        http=external,
    )
    asyncio.run(injected.aclose())
    assert external.is_closed is False
    asyncio.run(external.aclose())


class _FakeCore:
    def __init__(self, verify_side_effect=None, close_side_effect=None):
        self.verify_calls = 0
        self.aclose_calls = 0
        self.verify_side_effect = verify_side_effect
        self.close_side_effect = close_side_effect

    async def verify(self):
        self.verify_calls += 1
        if self.verify_side_effect:
            raise self.verify_side_effect

    async def aclose(self):
        self.aclose_calls += 1
        if self.close_side_effect:
            raise self.close_side_effect


def _controller_for_core_tests(core_factory=None) -> DeployController:
    config = make_config(agent_core_tokens={"m1": "m1-token"})
    policy = Policy(config)
    policy.machines = {
        "m1": MachineInfo(
            alias="m1",
            node_id="node-1",
            owners=["owner"],
            node_host="192.0.2.1",
            targets=["driver"],
            platforms=["linux/arm64"],
            driver_paths=["vendor/driver"],
        )
    }
    return DeployController(
        config,
        MagicMock(),
        policy,
        MagicMock(),
        agent_core_factory=core_factory,
    )


def test_resolve_core_verify_agent_core_error_closes_new_client(monkeypatch):
    import agents.deploy_approval.service as service_mod
    core = _FakeCore(verify_side_effect=AgentCoreError("bad token"))
    monkeypatch.setattr(service_mod, "AgentCoreClient", lambda *a, **k: core)
    controller = _controller_for_core_tests(core_factory=lambda nid: None)
    with pytest.raises(DeployControllerError, match="unreachable"):
        asyncio.run(controller._resolve_core_client("node-1"))
    assert core.verify_calls == 1


def test_resolve_core_unexpected_verify_error_closes_new_client(monkeypatch):
    import agents.deploy_approval.service as service_mod
    core = _FakeCore(verify_side_effect=RuntimeError("boom"))
    monkeypatch.setattr(service_mod, "AgentCoreClient", lambda *a, **k: core)
    controller = _controller_for_core_tests(core_factory=lambda nid: None)
    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(controller._resolve_core_client("node-1"))
    assert core.verify_calls == 1


def test_verified_core_is_cached_and_reused(monkeypatch):
    core = _FakeCore()
    created = []

    def _factory(node_id):
        created.append(node_id)
        return core

    controller = _controller_for_core_tests(core_factory=_factory)
    first = asyncio.run(controller._core_for_node("node-1"))
    second = asyncio.run(controller._core_for_node("node-1"))
    assert first is core
    assert second is core
    assert len(created) == 1


def test_controller_aclose_logs_and_continues():
    controller = _controller_for_core_tests()
    bad = _FakeCore(close_side_effect=RuntimeError("close failed"))
    good = _FakeCore()
    controller._core_clients = {"bad": bad, "good": good}
    asyncio.run(controller.aclose())
    assert bad.aclose_calls == 1
    assert good.aclose_calls == 1
    assert controller._core_clients == {}


def test_agent_core_accepts_code_200():
    cfg = _client()
    tr = _Transport({"code": 200, "data": {"running_image": "registry/repo@sha256:" + "a" * 64}})
    c = _agent_core_client(tr)
    out = asyncio.run(c.driver_status("driver"))
    assert out == {"running_image": "registry/repo@sha256:" + "a" * 64}


def test_agent_core_rejects_missing_running_image():
    cfg = _client()
    tr = _Transport({"code": 200, "data": {"status": "running"}})
    c = AgentCoreClient(
        cfg, base_url="https://192.0.2.1:15678",
        node_host="192.0.2.1", http=httpx.AsyncClient(transport=tr),
    )
    with pytest.raises(AgentCoreError):
        asyncio.run(c.driver_status("driver"))


def test_agent_core_rejects_missing_data_envelope():
    cfg = _client()
    tr = _Transport({"message": "boom", "data": None})
    c = AgentCoreClient(
        cfg, base_url="https://192.0.2.1:15678",
        node_host="192.0.2.1", http=httpx.AsyncClient(transport=tr),
    )
    with pytest.raises(AgentCoreError):
        asyncio.run(c.driver_status("driver"))


def test_agent_core_rejects_non_200_code():
    cfg = _client()
    tr = _Transport({"code": 500, "message": "boom", "data": None})
    c = AgentCoreClient(
        cfg, base_url="https://192.0.2.1:15678",
        node_host="192.0.2.1", http=httpx.AsyncClient(transport=tr),
    )
    with pytest.raises(AgentCoreError):
        asyncio.run(c.driver_status("driver"))


def test_agent_core_auth_verify_accepts_raw_shape():
    # /api/auth/verify returns raw {valid, auth_required}, not a {code,data}
    # wrapper — the client must accept both.
    cfg = _client()
    tr = _Transport({"valid": True, "auth_required": True})
    c = AgentCoreClient(
        cfg, base_url="https://192.0.2.1:15678",
        node_host="192.0.2.1", http=httpx.AsyncClient(transport=tr),
    )
    out = asyncio.run(c.verify())
    assert out.get("valid") is True
    assert out.get("auth_required") is True


def test_agent_core_auth_disabled_fails_closed():
    # Authentication must be required AND valid: an auth-disabled Agent Core
    # (auth_required=false or valid=false) is never acceptable for a deploy.
    import pytest
    for payload in (
        {"valid": True, "auth_required": False},
        {"valid": False, "auth_required": True},
        {"valid": False, "auth_required": False},
        {"valid": "true", "auth_required": True},
    ):
        cfg = _client()
        tr = _Transport(payload)
        c = AgentCoreClient(
            cfg, base_url="https://192.0.2.1:15678",
        node_host="192.0.2.1", http=httpx.AsyncClient(transport=tr),
        )
        with pytest.raises(AgentCoreError):
            asyncio.run(c.verify())


def test_agent_core_rejects_http_401():
    cfg = _client()
    tr = _Transport({"detail": "nope"}, status=401)
    c = AgentCoreClient(
        cfg, base_url="https://192.0.2.1:15678",
        node_host="192.0.2.1", http=httpx.AsyncClient(transport=tr),
    )
    with pytest.raises(AgentCoreError):
        asyncio.run(c.verify())


class _ChunkedTransport(httpx.AsyncBaseTransport):
    """Serves a sizeable body WITHOUT a Content-Length so a true streaming byte
    cap must be enforced from the wire (not from a header)."""

    def __init__(self, body: bytes, status: int = 200):
        self.body = body
        self.status = status

    async def handle_async_request(self, request):
        # No content-length header: httpx must deliver via a stream.
        return httpx.Response(
            self.status,
            content=self.body,
            headers={},
            request=request,
        )


def test_stream_request_oversize_without_content_length():
    # A response with no Content-Length and more bytes than the limit must be
    # refused while streaming — it must never be buffered in full first.
    from ..clients_common import SecurityError, stream_request

    cfg = _client()
    cfg.max_response_bytes = 64
    big = b"x" * 4096
    tr = _ChunkedTransport(big)
    client = httpx.AsyncClient(transport=tr)
    with pytest.raises(SecurityError):
        asyncio.run(
            stream_request(
                client, "GET", "https://example.invalid/x",
                cfg.max_response_bytes,
            )
        )


def test_agent_core_oversize_response_fails_closed():
    # Even with no Content-Length header, an oversized Agent Core response is
    # refused (streaming byte cap), not parsed.
    cfg = _client()
    cfg.max_response_bytes = 64
    tr = _ChunkedTransport(b'{"code": 200, "data": "' + b"x" * 512 + b'"}')
    c = AgentCoreClient(
        cfg, base_url="https://192.0.2.1:15678",
        node_host="192.0.2.1", http=httpx.AsyncClient(transport=tr),
    )
    with pytest.raises(AgentCoreError):
        asyncio.run(c.driver_status("driver"))


def test_github_list_open_prs_reads_past_500_and_ignores_age():
    from ..github_client import GitHubClient
    from urllib.parse import parse_qs, urlparse

    cfg = make_config(allow_private_http=False)
    cfg.github_api_url = "https://api.github.com"

    old_ts = "2010-01-01T00:00:00Z"
    fresh_ts = "2026-09-03T12:00:00Z"
    page_batches = {
        1: [_open_pr(i, fresh_ts) for i in range(1, 101)],
        2: [_open_pr(i, fresh_ts) for i in range(101, 201)],
        3: [_open_pr(i, fresh_ts) for i in range(201, 301)],
        4: [_open_pr(i, fresh_ts) for i in range(301, 401)],
        5: [_open_pr(i, fresh_ts) for i in range(401, 501)],
        6: [_open_pr(i, fresh_ts) for i in range(501, 601)],
        7: [_open_pr(601, old_ts)],
    }

    requested_pages: list[int] = []
    requested_queries: list[dict[str, str]] = []

    class FakeTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            qs = parse_qs(urlparse(str(request.url)).query)
            requested_pages.append(int(qs.get("page", ["1"])[0]))
            requested_queries.append({
                "state": qs.get("state", [""])[0],
                "sort": qs.get("sort", [""])[0],
                "direction": qs.get("direction", [""])[0],
                "per_page": qs.get("per_page", [""])[0],
            })
            page = requested_pages[-1]
            return httpx.Response(200, json=page_batches.get(page, []), request=request)

    gh = GitHubClient(cfg, http=httpx.AsyncClient(transport=FakeTransport()), token_provider=fake_token_provider)
    prs = asyncio.run(gh.list_open_prs("org/repo"))

    assert len(prs) == 601
    assert any(pr["number"] == 601 and pr["updated_at"] == old_ts for pr in prs)
    assert requested_pages == [1, 2, 3, 4, 5, 6, 7]
    assert all(
        q == {"state": "open", "sort": "updated", "direction": "desc", "per_page": "100"}
        for q in requested_queries
    )


def test_github_list_open_prs_never_requests_closed():
    from ..github_client import GitHubClient
    from urllib.parse import parse_qs, urlparse

    cfg = make_config(allow_private_http=False)
    cfg.github_api_url = "https://api.github.com"

    requested_states: list[str] = []

    class FakeTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            qs = parse_qs(urlparse(str(request.url)).query)
            state = qs.get("state", [""])[0]
            requested_states.append(state)
            if state != "open":
                raise AssertionError(f"unexpected state {state!r}")
            page = int(qs.get("page", ["1"])[0])
            batch = [_open_pr(i, "2026-09-03T12:00:00Z") for i in range(1, 101)] if page == 1 else []
            return httpx.Response(200, json=batch, request=request)

    gh = GitHubClient(cfg, http=httpx.AsyncClient(transport=FakeTransport()), token_provider=fake_token_provider)
    prs = asyncio.run(gh.list_open_prs("org/repo"))

    assert len(prs) == 100
    assert requested_states == ["open", "open"]
    assert "closed" not in requested_states


def test_github_list_open_prs_deduplicates_page_overlap():
    from ..github_client import GitHubClient
    from urllib.parse import parse_qs, urlparse

    cfg = make_config(allow_private_http=False)
    cfg.github_api_url = "https://api.github.com"

    requested_pages: list[int] = []

    class FakeTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            qs = parse_qs(urlparse(str(request.url)).query)
            page = int(qs.get("page", ["1"])[0])
            requested_pages.append(page)
            if page == 1:
                batch = [_open_pr(i, f"2026-09-03T12:{i:02d}:00Z", marker="page1") for i in range(1, 101)]
            elif page == 2:
                batch = [
                    _open_pr(100, "2026-09-03T13:00:00Z", marker="page2"),
                    _open_pr(101, "2026-09-03T13:01:00Z", marker="page2"),
                ]
            else:
                batch = []
            return httpx.Response(200, json=batch, request=request)

    gh = GitHubClient(cfg, http=httpx.AsyncClient(transport=FakeTransport()), token_provider=fake_token_provider)
    prs = asyncio.run(gh.list_open_prs("org/repo"))

    assert requested_pages == [1, 2]
    assert len(prs) == 101
    assert sum(1 for pr in prs if pr["number"] == 100) == 1
    assert next(pr for pr in prs if pr["number"] == 100)["marker"] == "page2"

def test_real_agent_core_list_envelopes_accept_list_data():
    """The real Agent Core wraps /api/drivers and /api/mcp as
    ``{"code":200,"data":[...]}`` (data is an ARRAY), while
    /api/drivers/<id>/status returns ``data`` as an OBJECT. The generic
    envelope check must accept both; each adapter validates its own shape."""
    cfg = _client()

    async def _scenario():
        out = []
        cases = [
            ({"code": 200, "data": []}, "list_drivers", []),
            ({"code": 200, "data": []}, "list_mcp", []),
            ({"code": 200, "data": {"status": "stopped", "running_image": ""}},
             "driver_status", {"running_image": "", "status": "stopped"}),
            ({"code": 200, "data": {"online": True, "tools": [{"name": "x"}]}},
             "mcp_ping", {"online": True, "tools": [{"name": "x"}]}),
        ]
        for payload, kind, expect in cases:
            tr = _Transport(payload)
            c = AgentCoreClient(
                cfg, base_url="https://192.0.2.1:15678",
        node_host="192.0.2.1", http=httpx.AsyncClient(transport=tr),
            )
            if kind == "list_drivers":
                got = await c.list_drivers()
            elif kind == "list_mcp":
                got = await c.list_mcp()
            elif kind == "driver_status":
                got = await c.driver_status("driver")
            else:
                got = await c.mcp_ping("mcp1")
            assert got == expect, (kind, got, expect)
            out.append(kind)
        return out

    assert asyncio.run(_scenario()) == [
        "list_drivers", "list_mcp", "driver_status", "mcp_ping",
    ]



def test_list_drivers_rejects_object_data_fail_closed():
    """If a node wrongly wraps /api/drivers as an object the adapter fails
    closed instead of guessing."""
    cfg = _client()
    tr = _Transport({"code": 200, "data": {"id": "perception"}})
    c = AgentCoreClient(
        cfg, base_url="https://192.0.2.1:15678",
        node_host="192.0.2.1", http=httpx.AsyncClient(transport=tr),
    )
    with pytest.raises(AgentCoreError):
        asyncio.run(c.list_drivers())


def test_deploy_driver_exact_tag_passthrough():
    """Exact full image tag passed verbatim to Agent Core deploy POST."""
    import json
    cfg = _client()
    tag = "bj-warehouse.tencentcloudcr.com/phanthy-motus/perception:release.260922.4707deb-jetson-jp5.11"

    class Capture(httpx.AsyncBaseTransport):
        def __init__(self):
            self.body = None

        async def handle_async_request(self, request):
            self.body = request.content
            return httpx.Response(200, json={"code": 200, "data": {"status": "starting"}},
                                  request=request)

    tr = Capture()
    c = AgentCoreClient(
        cfg, base_url="https://192.0.2.1:15678",
        node_host="192.0.2.1", http=httpx.AsyncClient(transport=tr),
    )
    asyncio.run(c.deploy_driver("perception", tag))
    assert json.loads(tr.body) == {"image": tag}


def test_deploy_driver_legacy_digest_accepted():
    """Legacy repo@sha256:<64hex> still accepted for hidden-state compatibility."""
    import json
    cfg = _client()
    digest_ref = "registry.example/repo@sha256:" + "a" * 64

    class Capture(httpx.AsyncBaseTransport):
        def __init__(self):
            self.body = None

        async def handle_async_request(self, request):
            self.body = request.content
            return httpx.Response(200, json={"code": 200, "data": {"status": "starting"}},
                                  request=request)

    tr = Capture()
    c = AgentCoreClient(
        cfg, base_url="https://192.0.2.1:15678",
        node_host="192.0.2.1", http=httpx.AsyncClient(transport=tr),
    )
    asyncio.run(c.deploy_driver("drv", digest_ref))
    assert json.loads(tr.body) == {"image": digest_ref}


def test_deploy_driver_malformed_image_rejected():
    """Malformed image references must be rejected before any HTTP call."""
    cfg = _client()
    for bad in ("", "registry/repo", "https://registry/repo:tag",
                "registry/repo:tag\nbad",
                "registry/repo:tag?x=1", "registry/repo:tag#fragment"):
        tr = _Transport({"code": 200, "data": {"status": "starting"}})
        c = AgentCoreClient(
            cfg, base_url="https://192.0.2.1:15678",
        node_host="192.0.2.1", http=httpx.AsyncClient(transport=tr),
        )
        with pytest.raises(AgentCoreError):
            asyncio.run(c.deploy_driver("drv", bad))


# ── MCP strict schema: NO coercion of non-object tools ────────────────────

def test_mcp_health_rejects_non_object_tools_without_coercion():
    """A non-object tool (number/string/null) is a protocol violation and must
    FAIL the client's mcp_ping with AgentCoreError — never coerced into a fake
    ``{"name": "<str>"}`` tool. All of [123], ["foo"], [None] fail closed."""
    cfg = _client()
    for raw_tools in ([123], ["foo"], [None]):
        tr = _Transport({
            "code": 200,
            "data": {"online": True, "tools": raw_tools},
        })
        c = AgentCoreClient(
            cfg, base_url="https://192.0.2.1:15678",
        node_host="192.0.2.1", http=httpx.AsyncClient(transport=tr),
        )
        with pytest.raises(AgentCoreError):
            asyncio.run(c.mcp_ping("mcp1"))


# ── GitHubClient.resolve_commit_sha regression tests ────────────────────────

def test_resolve_commit_sha_returns_exact_40_hex():
    """resolve_commit_sha must return exact 40-char lowercase hex."""
    from ..github_client import GitHubClient
    import httpx

    payload = {"sha": "a" * 40}

    class Tr(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(200, json=payload, request=request)

    async def _token():
        return "fake"

    cfg = make_config()
    client = GitHubClient(cfg, token_provider=_token, http=httpx.AsyncClient(transport=Tr()))
    result = asyncio.run(client.resolve_commit_sha("org/repo", "abc1234"))
    assert result == "a" * 40


def test_resolve_commit_sha_rejects_non_40hex():
    """Non-40-character or non-hex SHA responses must be rejected."""
    from ..github_client import GitHubClient
    import httpx

    for bad in ("abc", "a" * 39, "a" * 41, "ZZZZ" * 10):
        payload = {"sha": bad}

        class Tr(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                return httpx.Response(200, json=payload, request=request)

        async def _token():
            return "fake"

        cfg = make_config()
        client = GitHubClient(cfg, token_provider=_token, http=httpx.AsyncClient(transport=Tr()))
        try:
            result = asyncio.run(client.resolve_commit_sha("org/repo", "abc1234"))
            assert False, f"should have failed for sha={bad}"
        except Exception:
            pass


def test_resolve_commit_sha_rejects_missing_sha_key():
    """Response without 'sha' key must be rejected."""
    from ..github_client import GitHubClient
    import httpx

    payload = {"other": "value"}

    class Tr(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(200, json=payload, request=request)

    async def _token():
        return "fake"

    cfg = make_config()
    client = GitHubClient(cfg, token_provider=_token, http=httpx.AsyncClient(transport=Tr()))
    try:
        result = asyncio.run(client.resolve_commit_sha("org/repo", "abc1234"))
        assert False, "should have failed"
    except Exception:
        pass


def test_resolve_commit_sha_rejects_non_2xx():
    """Non-2xx responses must propagate errors."""
    from ..github_client import GitHubClient
    import httpx

    class Tr(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(404, json={"message": "not found"}, request=request)

    async def _token():
        return "fake"

    cfg = make_config()
    client = GitHubClient(cfg, token_provider=_token, http=httpx.AsyncClient(transport=Tr()))
    try:
        result = asyncio.run(client.resolve_commit_sha("org/repo", "abc1234"))
        assert False, "should have failed"
    except Exception:
        pass



import pytest


class TestValidateApiPathSegment:
    """Blocker 3: _validate_api_path_segment rejects malicious values."""

    def _make_client(self, transport):
        return AgentCoreClient(
            make_config(allow_private_http=True),
            base_url="https://10.0.0.1:15678",
            node_host="10.0.0.1",
            http=httpx.AsyncClient(transport=transport),
        )

    def _ok_transport(self):
        return _Transport({"code": 200, "data": {}})

    def _status_transport(self):
        return _Transport(
            {"code": 200, "data": {"status": "running", "running_image": "img@sha256:" + "a" * 64}},
        )

    def _mcp_transport(self):
        return _Transport({"code": 200, "data": {"online": True, "tools": []}})

    def test_agent_core_runtime_id_allows_embedded_dot_but_rejects_dot_segments(self):
        """Real Agent Core runtime id allows embedded dots; exact . / .. rejected."""
        image = "registry/repo@sha256:" + "a" * 64
        c = self._make_client(_Transport({"code": 200, "data": {"status": "starting"}}))

        result = asyncio.run(c.deploy_driver("x-humanoid-tianyi2.0", image))
        assert result["data"]["status"] == "starting"

        for bad in (".", ".."):
            with pytest.raises(AgentCoreError):
                asyncio.run(c.deploy_driver(bad, image))

    def test_deploy_driver_rejects_slash_in_driver_id(self):
        c = self._make_client(self._ok_transport())
        with pytest.raises(AgentCoreError, match="must not contain '/'"):
            asyncio.run(c.deploy_driver("drv/evil", "registry/repo@sha256:" + "a" * 64))

    def test_deploy_driver_rejects_backslash_in_driver_id(self):
        c = self._make_client(self._ok_transport())
        with pytest.raises(AgentCoreError, match="must not contain"):
            asyncio.run(c.deploy_driver("drv\\evil", "registry/repo@sha256:" + "a" * 64))

    def test_deploy_driver_rejects_question_mark_in_driver_id(self):
        c = self._make_client(self._ok_transport())
        with pytest.raises(AgentCoreError, match="must not contain '?'"):
            asyncio.run(c.deploy_driver("drv?x=1", "registry/repo@sha256:" + "a" * 64))

    def test_deploy_driver_rejects_hash_in_driver_id(self):
        c = self._make_client(self._ok_transport())
        with pytest.raises(AgentCoreError, match="must not contain '#'"):
            asyncio.run(c.deploy_driver("drv#section", "registry/repo@sha256:" + "a" * 64))

    def test_deploy_driver_rejects_percent_in_driver_id(self):
        c = self._make_client(self._ok_transport())
        with pytest.raises(AgentCoreError, match="must not contain '%'"):
            asyncio.run(c.deploy_driver("drv%20evil", "registry/repo@sha256:" + "a" * 64))

    def test_deploy_driver_rejects_whitespace_in_driver_id(self):
        c = self._make_client(self._ok_transport())
        with pytest.raises(AgentCoreError, match="must not contain"):
            asyncio.run(c.deploy_driver("drv evil", "registry/repo@sha256:" + "a" * 64))

    def test_deploy_driver_rejects_control_characters_in_driver_id(self):
        c = self._make_client(self._ok_transport())
        with pytest.raises(AgentCoreError, match="must not contain control characters"):
            asyncio.run(c.deploy_driver("drv\x00evil", "registry/repo@sha256:" + "a" * 64))

    def test_deploy_driver_rejects_non_string_driver_id(self):
        c = self._make_client(self._ok_transport())
        with pytest.raises(AgentCoreError, match="requires a non-empty driver id"):
            asyncio.run(c.deploy_driver(123, "registry/repo@sha256:" + "a" * 64))

    def test_deploy_driver_rejects_empty_driver_id(self):
        c = self._make_client(self._ok_transport())
        with pytest.raises(AgentCoreError, match="requires a non-empty driver id"):
            asyncio.run(c.deploy_driver("", "registry/repo@sha256:" + "a" * 64))

    def test_deploy_driver_rejects_stripped_empty_driver_id(self):
        c = self._make_client(self._ok_transport())
        with pytest.raises(AgentCoreError, match="must be non-empty"):
            asyncio.run(c.deploy_driver("  ", "registry/repo@sha256:" + "a" * 64))

    def test_deploy_post_explicit_error_code_is_confirmed_failure(self):
        class Transport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                return httpx.Response(200, json={
                    "code": 500,
                    "data": {"status": "failed"},
                    "message": "deployment failed",
                }, request=request)

        c = self._make_client(Transport())
        with pytest.raises(AgentCoreError, match="deployment failed") as excinfo:
            asyncio.run(c.deploy_driver("web", "registry/repo@sha256:" + "a" * 64))
        assert isinstance(excinfo.value, AgentCoreError)
        assert not isinstance(excinfo.value, AgentCoreDeployOutcomeUncertain)

    def test_deploy_post_data_status_error_is_confirmed_failure(self):
        class Transport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                return httpx.Response(200, json={
                    "code": 0,
                    "data": {"status": "error", "error": "image pull failed"},
                    "message": "error",
                }, request=request)

        c = self._make_client(Transport())
        with pytest.raises(AgentCoreError, match="image pull failed") as excinfo:
            asyncio.run(c.deploy_driver("web", "registry/repo@sha256:" + "a" * 64))
        assert isinstance(excinfo.value, AgentCoreError)
        assert not isinstance(excinfo.value, AgentCoreDeployOutcomeUncertain)

    def test_deploy_post_skipped_response_is_confirmed_failure(self):
        class Transport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                return httpx.Response(200, json={
                    "code": 0,
                    "data": {"status": "ok", "skipped": True},
                    "message": "skipped",
                }, request=request)

        c = self._make_client(Transport())
        with pytest.raises(AgentCoreDeployOutcomeUncertain, match="skipped"):
            asyncio.run(c.deploy_driver("web", "registry/repo@sha256:" + "a" * 64))

    def test_deploy_post_transport_timeout_is_uncertain(self):
        # Scenario 1: httpx.ReadTimeout (transport-layer error)
        class ReadTimeoutTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                raise httpx.ReadTimeout("timed out", request=request)

        c = self._make_client(ReadTimeoutTransport())
        with pytest.raises(AgentCoreDeployOutcomeUncertain, match="timed out"):
            asyncio.run(c.deploy_driver("web", "registry/repo@sha256:" + "a" * 64))

        # Scenario 2: REAL stream_request total wall-clock timeout
        # A slow transport that sleeps past the total_timeout provokes SecurityError
        # from asyncio.wait_for, which must be classified as UNCERTAIN once the
        # unsafe POST has already begun.
        class SlowTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                await asyncio.sleep(10)
                raise httpx.ReadTimeout("too late", request=request)

        slow_cfg = make_config(allow_private_http=True, total_timeout=0.01)
        c2 = AgentCoreClient(
            slow_cfg,
            base_url="https://10.0.0.1:15678",
            node_host="10.0.0.1",
            http=httpx.AsyncClient(transport=SlowTransport()),
        )
        with pytest.raises(AgentCoreDeployOutcomeUncertain) as excinfo:
            asyncio.run(c2.deploy_driver("web", "registry/repo@sha256:" + "a" * 64))
        assert "total timeout" in str(excinfo.value).lower() or "timeout" in str(excinfo.value).lower()


    def test_driver_status_rejects_slash_in_driver_id(self):
        c = self._make_client(self._status_transport())
        with pytest.raises(AgentCoreError, match="must not contain"):
            asyncio.run(c.driver_status("evil/../../system"))

    def test_mcp_ping_rejects_question_mark_in_mcp_id(self):
        c = self._make_client(self._mcp_transport())
        with pytest.raises(AgentCoreError, match="must not contain '?'"):
            asyncio.run(c.mcp_ping("mcp?x=1"))


# ── FIX 2: GitHub self-created client trust_env regression ──────────────────

def test_github_self_created_client_disables_proxy_env(monkeypatch):
    """GitHubClient self-created AsyncClient must not inherit proxy env."""
    monkeypatch.setenv("HTTP_PROXY", "http://evil-proxy:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://evil-proxy:8080")
    monkeypatch.setenv("ALL_PROXY", "http://evil-proxy:8080")
    from ..github_client import GitHubClient
    from ..config import Config
    cfg = Config(github_repos=["org/repo"], poll_enabled=True)
    client = GitHubClient(cfg)
    try:
        assert client.http.trust_env is False
    finally:
        asyncio.run(client.http.aclose())


    def test_deploy_driver_skipped_running_is_known_success(self):
        """skipped=true + status=running => no exception (caller still verifies)."""
        from httpx import Response
        class Transport(httpx.BaseTransport):
            def handle_request(self, request):
                return Response(
                    200,
                    json={"code": 200, "data": {
                        "status": "running", "skipped": True,
                        "message": "already running with same image",
                    }},
                )
        c = self._make_client(Transport())
        result = asyncio.run(c.deploy_driver("web", "registry/repo@sha256:" + "a" * 64))
        assert result is not None

    def test_deploy_driver_skipped_deploying_is_uncertain(self):
        """skipped=true + status=deploying => AgentCoreDeployOutcomeUncertain."""
        from httpx import Response
        class Transport(httpx.BaseTransport):
            def handle_request(self, request):
                return Response(
                    200,
                    json={"code": 200, "data": {
                        "status": "deploying", "skipped": True,
                        "message": "another deploy in progress",
                    }},
                )
        c = self._make_client(Transport())
        with pytest.raises(AgentCoreDeployOutcomeUncertain, match="skipped"):
            asyncio.run(c.deploy_driver("web", "registry/repo@sha256:" + "a" * 64))

    def test_driver_status_preserves_status_with_running_image_strict(self):
        """status field preserved when running_image present; invalid status fails closed."""
        from httpx import Response
        class Transport(httpx.BaseTransport):
            def handle_request(self, request):
                if "valid" in str(request.url):
                    return Response(200, json={"code": 200, "data": {
                        "status": "running", "running_image": "registry/repo:tag",
                    }})
                return Response(200, json={"code": 200, "data": {
                    "status": 123, "running_image": "registry/repo:tag",
                }})
        c = self._make_client(Transport())
        result = asyncio.run(c.driver_status("valid-driver"))
        assert result == {"status": "running", "running_image": "registry/repo:tag"}
        with pytest.raises(AgentCoreError, match="status must be a non-empty string"):
            asyncio.run(c.driver_status("invalid-driver"))
