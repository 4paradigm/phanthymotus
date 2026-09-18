"""Tests for authenticated, server-side evidence delivery."""
from __future__ import annotations

import asyncio
import hashlib
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from starlette.requests import Request

from .. import evidence_download as mod
from ..config import Config
from ..cos_client import CosClient, CosError
from .conftest import make_config

HEAD = "a" * 40
SECRET = "oauth-secret"


def _request(app):
    return Request({"type": "http", "method": "GET", "path": "/evidence/download",
                    "query_string": b"", "headers": [], "client": ("127.0.0.1", 1),
                    "server": ("test", 80), "scheme": "https", "app": app})


def _app(**kwargs):
    config = make_config(github_repos=["4paradigm/phanthymotus"],
                         deploy_approval_public_base_url="https://deploy.example",
                         github_oauth_client_id="client", github_oauth_client_secret=SECRET)
    config.__dict__.update(kwargs)
    app = SimpleNamespace(state=SimpleNamespace(config=config))
    return app


def _payload(**kwargs):
    data = {"version": 1, "state": "state", "pkce_verifier": "verifier",
            "repo": "4paradigm/phanthymotus", "pr": 7, "head": HEAD,
            "issued_at": time.time(), "expires_at": time.time() + 300}
    data.update(kwargs)
    return data


@pytest.mark.parametrize("repo,pr,head", [
    ("4paradigm/phanthymotus", 0, HEAD),
    ("4paradigm/phanthymotus", 7, "A" * 40), ("4paradigm/phanthymotus", 7, "a" * 39),
    ("4paradigm/phanthymotus", 7, "not-a-sha"),
])
def test_request_identity_rejects_invalid(repo, pr, head):
    with pytest.raises(ValueError): mod._request_identity(repo, pr, head)


def test_request_identity_rejects_unsupported_repo_at_route():
    response = asyncio.run(mod.evidence_download(_request(_app()), "no/such-repo", 7, HEAD))
    assert response.status_code == 400


def test_request_identity_accepts_valid():
    assert mod._request_identity("4paradigm/phanthymotus", 7, HEAD) == ("4paradigm/phanthymotus", 7, HEAD)


def test_oauth_redirect_and_cookie_security():
    app = _app()
    response = asyncio.run(mod.evidence_download(_request(app), "4paradigm/phanthymotus", 7, HEAD))
    location = response.headers["location"]
    assert "client_id=client" in location and "redirect_uri=" in location
    assert "state=" in location and "code_challenge=" in location
    assert "code_challenge_method=S256" in location
    cookie = response.headers["set-cookie"]
    assert "Secure" in cookie and "HttpOnly" in cookie and "SameSite=lax" in cookie
    assert "Max-Age=300" in cookie and "Path=/evidence/oauth/callback" in cookie


def test_signed_cookie_round_trip_and_rejection():
    packed = mod._pack_cookie(_payload(), SECRET)
    assert mod._unpack_cookie(packed, SECRET)["head"] == HEAD
    with pytest.raises(ValueError): mod._unpack_cookie(packed[:-1] + "x", SECRET)
    with pytest.raises(ValueError): mod._unpack_cookie(mod._pack_cookie(_payload(expires_at=time.time() - 1), SECRET), SECRET)
    with pytest.raises(ValueError): mod._unpack_cookie(mod._pack_cookie(_payload(version=2), SECRET), SECRET)


def _authorized_request(hidden, *, owners=None):
    app = _app()
    app.state.proxy = SimpleNamespace(
        get_pr=AsyncMock(return_value={"state": "open", "merged": False, "head": {"sha": HEAD}, "user": {"id": 1}}),
        read_hidden_state=AsyncMock(return_value=hidden), collaborator_permission=AsyncMock(return_value="none"))
    machine = SimpleNamespace(owners=owners or [])
    app.state.policy = SimpleNamespace(get_machine=MagicMock(return_value=machine))
    return _request(app), app


def _hidden(**kwargs):
    value = {"head_sha": HEAD, "status": "succeeded",
             "cos": {"object_key": "phanthymotus_pr/phanthymotus/2026-09/2026-09-17/pr-7/evidence-" + HEAD + ".log.gz",
                     "sha256": "b" * 64, "size": 4}, "deployments": []}
    value.update(kwargs)
    return value


@pytest.mark.parametrize("user,permission", [
    ({"id": 1, "login": "author"}, "none"), ({"id": 2, "login": "owner"}, "none"),
    ({"id": 2, "login": "writer"}, "write"), ({"id": 2, "login": "maintainer"}, "maintain"),
    ({"id": 2, "login": "admin"}, "admin"),
])
def test_fresh_state_authorization_passes_author_owner_and_collaborators(user, permission):
    deployments = [{"phase": "deployed", "machine": "m"}] if user["login"] == "owner" else []
    request, app = _authorized_request(_hidden(deployments=deployments), owners=["owner"])
    app.state.proxy.collaborator_permission.return_value = permission
    assert asyncio.run(mod._fresh_authorized_state(request, _payload(), user))["head"] == HEAD


@pytest.mark.parametrize("permission", ["read", "triage", "none"])
def test_fresh_state_authorization_rejects_weak_collaborators(permission):
    request, app = _authorized_request(_hidden())
    app.state.proxy.collaborator_permission.return_value = permission
    with pytest.raises(PermissionError): asyncio.run(mod._fresh_authorized_state(request, _payload(), {"id": 2, "login": "u"}))


def test_collaborator_exception_fails_closed_unless_author_or_owner():
    request, app = _authorized_request(_hidden())
    app.state.proxy.collaborator_permission.side_effect = RuntimeError("private COS token leaked")
    with pytest.raises(PermissionError): asyncio.run(mod._fresh_authorized_state(request, _payload(), {"id": 2, "login": "u"}))


@pytest.mark.parametrize("hidden,pr,expected", [
    (_hidden(), {"state": "closed", "merged": False}, "no"),
    (_hidden(), {"state": "open", "merged": True}, "no"),
    (_hidden(head_sha="b" * 40), {"state": "open", "merged": False}, "no"),
    (_hidden(status="testing"), {"state": "open", "merged": False}, "no"),
    (_hidden(cos={"object_key": "x", "sha256": "bad", "size": 1}), {"state": "open", "merged": False}, "no"),
    (_hidden(cos={"object_key": "x", "sha256": "b" * 64, "size": 0}), {"state": "open", "merged": False}, "no"),
])
def test_fresh_state_binding_and_metadata_fail_closed(hidden, pr, expected):
    request, app = _authorized_request(hidden)
    app.state.proxy.get_pr.return_value = dict(pr, head={"sha": HEAD})
    with pytest.raises((ValueError, PermissionError)):
        asyncio.run(mod._fresh_authorized_state(request, _payload(), {"id": 1, "login": "author"}))


def test_object_key_comes_only_from_fresh_hidden_state(monkeypatch):
    request, app = _authorized_request(_hidden())
    seen = []
    monkeypatch.setattr(CosClient, "validate_object_key", staticmethod(lambda *args: seen.append(args)))
    browser = _payload(object_key="attacker-key")
    asyncio.run(mod._fresh_authorized_state(request, browser, {"id": 1, "login": "author"}))
    assert seen[0][-1] == app.state.proxy.read_hidden_state.return_value["cos"]["object_key"]


class _OAuthClient:
    def __init__(self, response, user): self.response, self.user = response, user
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    async def post(self, url, **kwargs):
        assert kwargs["data"]["code_verifier"] == "verifier"
        return self.response
    async def get(self, url, **kwargs):
        assert kwargs["headers"]["Authorization"] == "Bearer access-token"
        return self.user


def test_callback_success_downloads_bytes_and_clears_cookie(monkeypatch):
    archive = b"gzip-bytes"
    app = _app()
    state = _hidden(cos={"object_key": "phanthymotus_pr/phanthymotus/2026-09/2026-09-17/pr-7/evidence-" + HEAD + ".log.gz",
                          "sha256": hashlib.sha256(archive).hexdigest(), "size": len(archive)})
    app.state.proxy = SimpleNamespace(get_pr=AsyncMock(return_value={"state": "open", "merged": False, "head": {"sha": HEAD}, "user": {"id": 1}}), read_hidden_state=AsyncMock(return_value=state), collaborator_permission=AsyncMock(return_value="none"))
    app.state.policy = SimpleNamespace(get_machine=MagicMock(return_value=None))
    monkeypatch.setattr(mod.httpx, "AsyncClient", lambda **kwargs: _OAuthClient(httpx.Response(200, json={"access_token": "access-token"}), httpx.Response(200, json={"id": 1, "login": "author"})))
    monkeypatch.setattr(mod.CosClient, "download_evidence_archive", AsyncMock(return_value=archive))
    request = _request(app)
    request._headers = [(b"cookie", (mod._COOKIE_NAME + "=" + mod._pack_cookie(_payload(), SECRET)).encode())]
    response = asyncio.run(mod.evidence_oauth_callback(request, "code", "state"))
    assert response.body == archive and response.headers["content-type"] == "application/gzip"
    assert "attachment" in response.headers["content-disposition"] and HEAD in response.headers["content-disposition"]
    for key, value in {"cache-control": "no-store", "pragma": "no-cache", "x-content-type-options": "nosniff", "referrer-policy": "no-referrer"}.items(): assert response.headers[key] == value
    assert "Max-Age=0" in response.headers["set-cookie"]


@pytest.mark.parametrize("failure", ["size", "sha", "cos"])
def test_callback_integrity_and_cos_failures_are_generic_and_clear_cookie(monkeypatch, failure):
    app = _app()
    state = _hidden()
    app.state.proxy = SimpleNamespace(get_pr=AsyncMock(return_value={"state": "open", "merged": False, "head": {"sha": HEAD}, "user": {"id": 1}}), read_hidden_state=AsyncMock(return_value=state), collaborator_permission=AsyncMock(return_value="none"))
    app.state.policy = SimpleNamespace(get_machine=MagicMock(return_value=None))
    monkeypatch.setattr(mod.httpx, "AsyncClient", lambda **kwargs: _OAuthClient(httpx.Response(200, json={"access_token": "access-token"}), httpx.Response(200, json={"id": 1, "login": "author"})))
    if failure == "cos": monkeypatch.setattr(mod.CosClient, "download_evidence_archive", AsyncMock(side_effect=CosError("raw secret")))
    else: monkeypatch.setattr(mod.CosClient, "download_evidence_archive", AsyncMock(return_value=b"wrong"))
    request = _request(app)
    request._headers = [(b"cookie", (mod._COOKIE_NAME + "=" + mod._pack_cookie(_payload(), SECRET)).encode())]
    response = asyncio.run(mod.evidence_oauth_callback(request, "code", "state"))
    assert response.status_code == 403 and b"raw secret" not in response.body
    assert "Max-Age=0" in response.headers["set-cookie"]
