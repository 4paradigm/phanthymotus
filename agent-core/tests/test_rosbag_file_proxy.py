"""The browser receives bag files through the owning MCP, in bounded chunks."""
import asyncio
import hmac
import os
import sys

import fastapi
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import auth  # noqa: E402
from api import mcp_manage  # noqa: E402


BAG = "bag_20261127_120000_a1b2c3d4e5f6"
FILE = BAG + "_0.db3"


class FakeContent:
    async def iter_chunked(self, size):
        assert size == 1024 * 1024
        yield b"first"
        yield b"second"


class FakeResponse:
    status = 200
    headers = {"Content-Length": "11"}
    content = FakeContent()
    released = False

    def release(self):
        self.released = True


class FakeSession:
    def __init__(self, *args, **kwargs):
        self.response = FakeResponse()
        self.closed = False
        self.request = None

    async def get(self, url, **kwargs):
        self.request = (url, kwargs)
        return self.response

    async def close(self):
        self.closed = True


def test_download_uses_registered_service_and_streams(monkeypatch):
    target = {"id": "perception", "url": "http://localhost:15720/mcp", "transport": "http",
              "tools": [{"name": "rosbag_recorder"}]}
    monkeypatch.setattr(mcp_manage, "_get_mcp_list", lambda: [target])
    monkeypatch.setattr(auth, "get_token", lambda: "test-token")
    session = FakeSession()
    monkeypatch.setattr(mcp_manage.aiohttp, "ClientSession", lambda **_: session)

    async def scenario():
        response = await mcp_manage.mcp_rosbag_file("perception", BAG, FILE)
        chunks = [chunk async for chunk in response.body_iterator]
        return response, chunks

    response, chunks = asyncio.run(scenario())
    assert chunks == [b"first", b"second"]
    assert response.headers["content-length"] == "11"
    assert session.request[0] == "http://localhost:15720/rosbag/file"
    assert session.request[1]["params"] == {"bag": BAG, "file": FILE}
    scoped = hmac.new(b"test-token", b"rosbag-file", "sha256").hexdigest()
    assert session.request[1]["headers"] == {"X-Access-Token": scoped}
    assert session.closed and session.response.released


def test_download_rejects_paths_and_services_without_the_card(monkeypatch):
    target = {"id": "other", "url": "http://localhost:1/mcp", "transport": "http", "tools": []}
    monkeypatch.setattr(mcp_manage, "_get_mcp_list", lambda: [target])
    with pytest.raises(fastapi.HTTPException) as bad_path:
        asyncio.run(mcp_manage.mcp_rosbag_file("other", "../secrets", FILE))
    assert bad_path.value.status_code == 400
    with pytest.raises(fastapi.HTTPException) as no_card:
        asyncio.run(mcp_manage.mcp_rosbag_file("other", BAG, FILE))
    assert no_card.value.status_code == 404


@pytest.mark.parametrize("url", [
    "http://attacker.example/mcp", "http://localhost.attacker.example/mcp",
    "http://attacker.example@localhost/mcp", "http://localhost@attacker.example/mcp",
])
def test_download_never_sends_token_to_registered_remote_url(monkeypatch, url):
    target = {"id": "perception", "url": url, "tools": [{"name": "rosbag_recorder"}]}
    monkeypatch.setattr(mcp_manage, "_get_mcp_list", lambda: [target])
    monkeypatch.setattr(auth, "get_token", lambda: "test-token")
    monkeypatch.setattr(mcp_manage.aiohttp, "ClientSession", lambda **_: pytest.fail("request sent"))
    with pytest.raises(fastapi.HTTPException) as denied:
        asyncio.run(mcp_manage.mcp_rosbag_file("perception", BAG, FILE))
    assert denied.value.status_code == 400


def test_cancelled_connection_closes_upstream_session(monkeypatch):
    target = {"id": "perception", "url": "http://localhost:15720/mcp",
              "tools": [{"name": "rosbag_recorder"}]}
    monkeypatch.setattr(mcp_manage, "_get_mcp_list", lambda: [target])
    session = FakeSession()

    async def cancelled(*_args, **_kwargs):
        raise asyncio.CancelledError()

    session.get = cancelled
    monkeypatch.setattr(mcp_manage.aiohttp, "ClientSession", lambda **_: session)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(mcp_manage.mcp_rosbag_file("perception", BAG, FILE))
    assert session.closed


def test_rejected_upstream_response_is_closed(monkeypatch):
    target = {"id": "perception", "url": "http://localhost:15720/mcp",
              "tools": [{"name": "rosbag_recorder"}]}
    monkeypatch.setattr(mcp_manage, "_get_mcp_list", lambda: [target])
    session = FakeSession()
    session.response.status = 404

    async def error_json(**_kwargs):
        return {"error": "missing"}

    session.response.json = error_json
    monkeypatch.setattr(mcp_manage.aiohttp, "ClientSession", lambda **_: session)
    with pytest.raises(fastapi.HTTPException) as denied:
        asyncio.run(mcp_manage.mcp_rosbag_file("perception", BAG, FILE))
    assert denied.value.status_code == 404
    assert session.closed and session.response.released
