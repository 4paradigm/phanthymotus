"""Authenticated, server-side delivery of private COS evidence archives."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import time
import urllib.parse

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response

from .cos_client import CosClient, EVIDENCE_MAX_ARCHIVE_BYTES

evidence_router = APIRouter()
_COOKIE_NAME = "deploy_approval_evidence_oauth"
_COOKIE_MAX_AGE = 300
_COOKIE_DOMAIN = b"deploy-approval-evidence-oauth-state-v1"


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _unb64(value: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("invalid base64")
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _cookie_key(secret: str) -> bytes:
    return hmac.new(secret.encode("utf-8"), _COOKIE_DOMAIN, hashlib.sha256).digest()


def _pack_cookie(payload: dict, secret: str) -> str:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode()
    encoded = _b64(raw)
    signature = hmac.new(_cookie_key(secret), encoded.encode(), hashlib.sha256).digest()
    return f"{encoded}.{_b64(signature)}"


def _unpack_cookie(value: str, secret: str) -> dict:
    try:
        encoded, signature = value.split(".", 1)
        expected = hmac.new(_cookie_key(secret), encoded.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _unb64(signature)):
            raise ValueError("invalid signature")
        payload = json.loads(_unb64(encoded))
    except (ValueError, TypeError, json.JSONDecodeError):
        raise ValueError("invalid OAuth cookie")
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("invalid OAuth cookie payload")
    if not isinstance(payload.get("state"), str) or not isinstance(payload.get("pkce_verifier"), str):
        raise ValueError("invalid OAuth cookie bindings")
    if not isinstance(payload.get("repo"), str) or not isinstance(payload.get("head"), str):
        raise ValueError("invalid OAuth cookie identity")
    if isinstance(payload.get("pr"), bool) or not isinstance(payload.get("pr"), int) or payload["pr"] <= 0:
        raise ValueError("invalid OAuth cookie PR")
    issued = payload.get("issued_at")
    expires = payload.get("expires_at")
    if not isinstance(issued, (int, float)) or isinstance(issued, bool) or not isinstance(expires, (int, float)) or isinstance(expires, bool):
        raise ValueError("invalid OAuth cookie time")
    if expires <= time.time() or expires <= issued or expires - issued > _COOKIE_MAX_AGE:
        raise ValueError("expired OAuth cookie")
    return payload


def _clear(response: Response) -> Response:
    response.delete_cookie(_COOKIE_NAME, path="/evidence/oauth/callback")
    return response


def _request_identity(repo: str, pr: int, head: str) -> tuple[str, int, str]:
    if not isinstance(repo, str) or not repo:
        raise ValueError("invalid repo")
    if isinstance(pr, bool) or not isinstance(pr, int) or pr <= 0:
        raise ValueError("invalid PR")
    if not isinstance(head, str) or not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("invalid HEAD")
    return repo, pr, head


@evidence_router.get("/evidence/download")
async def evidence_download(request: Request, repo: str, pr: int, head: str):
    config = request.app.state.config
    try:
        repo, pr, head = _request_identity(repo, pr, head)
        if repo not in config.github_repos:
            raise ValueError("unsupported repo")
    except ValueError:
        return JSONResponse({"detail": "invalid evidence identity"}, status_code=400)
    state = {
        "version": 1,
        "state": secrets.token_urlsafe(32),
        "pkce_verifier": secrets.token_urlsafe(48),
        "repo": repo,
        "pr": pr,
        "head": head,
        "issued_at": time.time(),
        "expires_at": time.time() + _COOKIE_MAX_AGE,
    }
    challenge = _b64(hashlib.sha256(state["pkce_verifier"].encode()).digest())
    redirect_uri = f"{config.deploy_approval_public_base_url}/evidence/oauth/callback"
    params = urllib.parse.urlencode({
        "client_id": config.github_oauth_client_id,
        "redirect_uri": redirect_uri,
        "state": state["state"],
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    })
    response = RedirectResponse(f"https://github.com/login/oauth/authorize?{params}", status_code=307)
    response.set_cookie(
        _COOKIE_NAME,
        _pack_cookie(state, config.github_oauth_client_secret),
        max_age=_COOKIE_MAX_AGE,
        secure=True,
        httponly=True,
        samesite="lax",
        path="/evidence/oauth/callback",
    )
    return response


async def _fresh_authorized_state(request: Request, payload: dict, user: dict) -> dict:
    config = request.app.state.config
    proxy = request.app.state.proxy
    repo, pr, head = payload["repo"], payload["pr"], payload["head"]
    fresh_pr = await proxy.get_pr(repo, pr)
    fresh_head = fresh_pr.get("head", {}).get("sha", "")
    if fresh_pr.get("state") != "open" or fresh_pr.get("merged") is True:
        raise ValueError("PR is not open")
    hidden = await proxy.read_hidden_state(repo, pr)
    if not isinstance(hidden, dict) or fresh_head != head or hidden.get("head_sha") != head:
        raise ValueError("evidence binding changed")
    if hidden.get("status") not in {"succeeded", "failed"}:
        raise ValueError("evidence is not terminal")
    cos = hidden.get("cos", {})
    object_key = cos.get("object_key")
    digest = cos.get("sha256")
    size = cos.get("size")
    if not isinstance(object_key, str) or not object_key or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid evidence metadata")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0 or size > EVIDENCE_MAX_ARCHIVE_BYTES:
        raise ValueError("invalid evidence size")
    user_id = user.get("id")
    login = user.get("login")
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0 or not isinstance(login, str) or not login:
        raise ValueError("invalid GitHub user")
    author_id = fresh_pr.get("user", {}).get("id")
    authorized = isinstance(author_id, int) and not isinstance(author_id, bool) and author_id == user_id
    if not authorized:
        for deployment in hidden.get("deployments", []):
            if deployment.get("phase") != "deployed":
                continue
            machine = request.app.state.policy.get_machine(deployment.get("machine", ""))
            if machine and login.lower() in {owner.lower() for owner in machine.owners}:
                authorized = True
                break
    if not authorized:
        try:
            authorized = await proxy.collaborator_permission(repo, login) in {"write", "maintain", "admin"}
        except Exception:
            authorized = False
    if not authorized:
        raise PermissionError("not authorized")
    CosClient.validate_object_key(repo, pr, head, object_key)
    return {"object_key": object_key, "sha256": digest, "size": size, "head": head}


@evidence_router.get("/evidence/oauth/callback")
async def evidence_oauth_callback(request: Request, code: str = "", state: str = ""):
    config = request.app.state.config
    try:
        payload = _unpack_cookie(request.cookies.get(_COOKIE_NAME, ""), config.github_oauth_client_secret)
        if not code or not hmac.compare_digest(state, payload["state"]):
            raise ValueError("invalid OAuth state")
        redirect_uri = f"{config.deploy_approval_public_base_url}/evidence/oauth/callback"
        async with httpx.AsyncClient(follow_redirects=False, trust_env=False, timeout=httpx.Timeout(20.0, connect=5.0, read=15.0)) as http:
            token_response = await http.post(
                "https://github.com/login/oauth/access_token",
                data={"client_id": config.github_oauth_client_id, "client_secret": config.github_oauth_client_secret, "code": code, "redirect_uri": redirect_uri, "code_verifier": payload["pkce_verifier"]},
                headers={"Accept": "application/json"},
            )
            if token_response.status_code != 200:
                raise ValueError("OAuth exchange failed")
            token_data = token_response.json()
            access_token = token_data.get("access_token") if isinstance(token_data, dict) else None
            if not isinstance(access_token, str) or not access_token:
                raise ValueError("OAuth exchange failed")
            user_response = await http.get("https://api.github.com/user", headers={"Authorization": f"Bearer {access_token}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2026-03-10"})
            if user_response.status_code != 200:
                raise ValueError("GitHub identity failed")
            user = user_response.json()
        evidence = await _fresh_authorized_state(request, payload, user)
        archive = await CosClient(config).download_evidence_archive(evidence["object_key"], EVIDENCE_MAX_ARCHIVE_BYTES)
        if len(archive) != evidence["size"] or hashlib.sha256(archive).hexdigest() != evidence["sha256"]:
            raise ValueError("evidence integrity check failed")
        response = Response(archive, media_type="application/gzip", headers={"Content-Disposition": f'attachment; filename="evidence-{evidence["head"]}.log.gz"', "Cache-Control": "no-store", "Pragma": "no-cache", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"})
        return _clear(response)
    except PermissionError:
        return _clear(JSONResponse({"detail": "forbidden"}, status_code=403))
    except Exception:
        return _clear(JSONResponse({"detail": "evidence download unavailable"}, status_code=403))
