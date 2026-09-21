"""GitHub App auth contract tests: cache, single-flight, double-check.

Uses dummy private_key bytes (no PEM) and mock HTTP transport.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agents.github_app_auth import GitHubAppAuth


class TestGetInstallationTokenSingleFlight:
    """Public API: get_installation_token must be single-flight under async lock."""

    @pytest.mark.asyncio
    async def test_get_installation_token_single_flight(self):
        """Concurrent calls to get_installation_token must issue exactly one HTTP POST."""
        auth = GitHubAppAuth(
            app_id="123456",
            installation_id="789",
            private_key=b"dummy-private-key-bytes-not-a-pem",
        )

        # Patch jwtBearerToken to return a fixed JWT so no crypto is needed
        auth.jwtBearerToken = lambda: "test-app-jwt"

        call_count = 0
        response_token = "installation-token-abc"

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            if "/access_tokens" in str(request.url):
                call_count += 1
                return httpx.Response(
                    200,
                    json={"token": response_token, "expires_at": "2099-01-01T00:00:00Z"},
                )
            return httpx.Response(200, json={"slug": "test-app"})

        auth._http = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport), trust_env=False)

        # Fire 10 concurrent calls
        tasks = [auth.get_installation_token() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All results must be identical
        assert all(r == response_token for r in results)
        # HTTP POST to /access_tokens must be exactly 1
        assert call_count == 1, f"Expected 1 refresh request, got {call_count}"

        await auth.close()

    @pytest.mark.asyncio
    async def test_get_installation_token_cache_reuse(self):
        """Subsequent calls after first refresh must return cached token without HTTP."""
        auth = GitHubAppAuth(
            app_id="123456",
            installation_id="789",
            private_key=b"dummy-private-key-bytes-not-a-pem",
        )
        auth.jwtBearerToken = lambda: "test-app-jwt"

        call_count = 0

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            if "/access_tokens" in str(request.url):
                call_count += 1
                return httpx.Response(
                    200,
                    json={"token": "tok-cache", "expires_at": "2099-01-01T00:00:00Z"},
                )
            return httpx.Response(200, json={"slug": "test-app"})

        auth._http = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport), trust_env=False)

        # First call triggers refresh
        t1 = await auth.get_installation_token()
        assert t1 == "tok-cache"
        first_count = call_count

        # Second call must use cache
        t2 = await auth.get_installation_token()
        assert t2 == "tok-cache"
        assert call_count == first_count, "Second call should not trigger HTTP"

        await auth.close()

    @pytest.mark.asyncio
    async def test_get_installation_token_double_check_inside_lock(self):
        """When two coroutines reach the lock, only one should refresh."""
        auth = GitHubAppAuth(
            app_id="123456",
            installation_id="789",
            private_key=b"dummy-private-key-bytes-not-a-pem",
        )
        auth.jwtBearerToken = lambda: "test-app-jwt"

        call_count = 0
        entered_lock = asyncio.Event()
        block_request = asyncio.Event()

        async def slow_mock_transport(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            if "/access_tokens" in str(request.url):
                call_count += 1
                entered_lock.set()
                await block_request.wait()
                return httpx.Response(
                    200,
                    json={"token": "tok-slow", "expires_at": "2099-01-01T00:00:00Z"},
                )
            return httpx.Response(200, json={"slug": "test-app"})

        auth._http = httpx.AsyncClient(transport=httpx.MockTransport(slow_mock_transport), trust_env=False)

        task1 = asyncio.create_task(auth.get_installation_token())
        # Wait until task1 has entered the lock and blocked on the transport
        await entered_lock.wait()
        # Now task1 is inside the lock, blocked on block_request
        task2 = asyncio.create_task(auth.get_installation_token())

        block_request.set()
        results = await asyncio.gather(task1, task2)
        assert all(r == "tok-slow" for r in results)
        assert call_count == 1, f"Expected 1 refresh, got {call_count}"

        await auth.close()
