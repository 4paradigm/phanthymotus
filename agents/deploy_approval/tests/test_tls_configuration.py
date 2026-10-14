"""Offline client construction contracts and real SSLContext TLS handshakes.

The ephemeral-port probes exercise the production client's SSLContext, not a
production deployment E2E. No AgentCoreClient request or deploy API is called.
All test certificates, keys and OpenSSL CA state live under pytest's tmp_path.
"""

from __future__ import annotations

import asyncio
import ssl
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
import yaml

from ..agent_core_client import AgentCoreClient, AgentCoreError, AgentCoreDeployOutcomeUncertain
from ..config import Config, _load_secrets_config


def _config(cas: dict[str, str], mode: str = "strict") -> Config:
    return Config(
        allow_private_http=True,
        agent_core_tokens={"machine-a": "test-only-token"},
        agent_core_ca_pem=cas,
        agent_core_tls_mode=mode,
    )


def _client(config: Config) -> AgentCoreClient:
    # Construct the production client with its required fixed endpoint. Its
    # transport is never used to connect to this endpoint in these tests.
    return AgentCoreClient(
        config,
        "https://127.0.0.1:15678",
        node_host="127.0.0.1",
        machine_alias="machine-a",
        access_token="test-only-token",
    )


def _openssl(directory: Path, *args: str) -> None:
    # Config-file extensions work with macOS LibreSSL as well as OpenSSL;
    # avoid the newer req -addext option and any Python crypto dependency.
    subprocess.run(
        ["openssl", *args], cwd=directory, check=True,
        capture_output=True, text=True, timeout=30,
    )


def _test_ca(directory: Path) -> Path:
    directory.mkdir()
    (directory / "ca.cnf").write_text(
        "[req]\nprompt = no\ndistinguished_name = dn\n"
        "[dn]\nCN = Offline TLS test CA\n"
        "[ca_extensions]\n"
        "basicConstraints = critical, CA:true, pathlen:0\n"
        "keyUsage = critical, keyCertSign, cRLSign\n"
        "subjectKeyIdentifier = hash\n"
        "authorityKeyIdentifier = keyid:always\n"
        "subjectAltName = IP:127.0.0.1\n",
        encoding="ascii",
    )
    _openssl(
        directory, "req", "-new", "-x509", "-newkey", "rsa:2048",
        "-nodes", "-sha256", "-days", "2", "-config", "ca.cnf",
        "-extensions", "ca_extensions", "-keyout", "ca.key", "-out", "ca.pem",
    )
    (directory / "ca.key").chmod(0o600)
    return directory / "ca.pem"


@pytest.fixture
def test_ca(tmp_path):
    return _test_ca(tmp_path / "trusted-ca")


def _ephemeral_test_certificate(
    directory: Path, ca_cert: Path, ip: str | None = "127.0.0.1", *, expired: bool = False,
) -> tuple[Path, Path]:
    directory.mkdir()
    (directory / "issued").mkdir()
    (directory / "index.txt").write_text("", encoding="ascii")
    (directory / "serial").write_text("01\n", encoding="ascii")
    (directory / "server.cnf").write_text(
        "[req]\nprompt = no\ndistinguished_name = dn\n"
        "[dn]\nCN = 127.0.0.1\n"
        "[ca]\ndefault_ca = test_ca\n"
        "[test_ca]\ndatabase = index.txt\nserial = serial\n"
        "new_certs_dir = issued\ndefault_md = sha256\ndefault_days = 1\n"
        "policy = test_policy\n"
        "[test_policy]\ncommonName = supplied\n"
        "[server_extensions]\n"
        "basicConstraints = critical, CA:false\n"
        "keyUsage = critical, digitalSignature, keyEncipherment\n"
        "extendedKeyUsage = serverAuth\n"
        "subjectKeyIdentifier = hash\nauthorityKeyIdentifier = keyid,issuer\n"
        + (f"subjectAltName = IP:{ip}\n" if ip is not None else ""),
        encoding="ascii",
    )
    _openssl(
        directory, "req", "-new", "-newkey", "rsa:2048", "-nodes",
        "-sha256", "-config", "server.cnf", "-keyout", "server.key",
        "-out", "server.csr",
    )
    (directory / "server.key").chmod(0o600)
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=2) if expired else now - timedelta(minutes=5)
    end = now - timedelta(days=1) if expired else now + timedelta(days=1)
    _openssl(
        directory, "ca", "-batch", "-notext", "-config", "server.cnf",
        "-cert", str(ca_cert), "-keyfile", str(ca_cert.with_suffix(".key")),
        "-extensions", "server_extensions",
        "-startdate", start.strftime("%y%m%d%H%M%SZ"),
        "-enddate", end.strftime("%y%m%d%H%M%SZ"),
        "-in", "server.csr", "-out", "server.pem",
    )
    return directory / "server.pem", directory / "server.key"


@asynccontextmanager
async def _tls_test_server(cert_file: Path, key_file: Path, seen: list[bytes]):
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(str(cert_file), str(key_file))
    handlers = set()

    async def handle(reader, writer):
        task = asyncio.current_task()
        handlers.add(task)
        try:
            request = b""
            while b"\r\n\r\n" not in request:
                chunk = await asyncio.wait_for(reader.read(8192), timeout=5)
                if not chunk:
                    return
                # Record every decrypted application byte, including partial
                # headers, so a failed handshake cannot hide a leaked token.
                seen.append(chunk)
                request += chunk
            body = b'{"tls":"ok"}'
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                + body
            )
            await writer.drain()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, ssl.SSLError):
                pass
            finally:
                handlers.discard(task)

    server = await asyncio.start_server(handle, "127.0.0.1", 0, ssl=server_context)
    try:
        yield server.sockets[0].getsockname()[1]
    finally:
        server.close()
        await server.wait_closed()
        pending = tuple(handlers)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.parametrize(
    "ca_case, base_url, node_host, alias, error",
    [
        ("missing", "https://127.0.0.1:15678", "127.0.0.1", "machine-a", "no valid agent_core_ca_pem"),
        ("blank", "https://127.0.0.1:15678", "127.0.0.1", "machine-a", "no valid agent_core_ca_pem"),
        ("malformed", "https://127.0.0.1:15678", "127.0.0.1", "machine-a", "CA PEM.*invalid"),
        ("wrong-alias", "https://127.0.0.1:15678", "127.0.0.1", "machine-b", "no valid agent_core_ca_pem"),
        ("valid", "https://127.0.0.1:15678", "127.0.0.1", " ", "machine_alias is required"),
        ("valid", "https://127.0.0.1:15678", "127.0.0.2", "machine-a", "hostname must match node_host"),
        ("valid", "http://127.0.0.1:15678", "127.0.0.1", "machine-a", "scheme must be https"),
        ("valid", "https://127.0.0.1:443", "127.0.0.1", "machine-a", "port must be 15678"),
        ("valid", "https://127.0.0.1", "127.0.0.1", "machine-a", "port must be 15678"),
        ("valid", "https://localhost:15678", "localhost", "machine-a", "literal IP address"),
        ("valid", "https://[::1]:15678", "::1", "machine-a", "IPv4 address"),
        ("valid", "https://127.0.0.1:15678", "", "machine-a", "node_host is required"),
    ],
    ids=[
        "missing-ca", "blank-ca", "malformed-ca", "unmapped-machine-alias",
        "blank-machine-alias", "url-node-host-mismatch", "http-rejected",
        "wrong-port", "missing-port", "hostname-node-rejected", "ipv6-rejected",
        "missing-node-host",
    ],
)
def test_invalid_tls_configuration_is_rejected_during_client_construction(
    test_ca, ca_case, base_url, node_host, alias, error,
):
    ca_map = {"machine-a": test_ca.read_text(encoding="ascii")}
    if ca_case == "missing":
        ca_map = {}
    elif ca_case == "blank":
        ca_map["machine-a"] = " \n"
    elif ca_case == "malformed":
        ca_map["machine-a"] = "-----BEGIN CERTIFICATE-----\nnot-a-cert\n-----END CERTIFICATE-----"
    # Only construction-time contracts belong here. SAN, issuer and expiry
    # rejection are proved below by actual TLS handshakes, with no deploy mock.
    with pytest.raises(AgentCoreError, match=error):
        AgentCoreClient(
            _config(ca_map), base_url, node_host=node_host,
            machine_alias=alias, access_token="test-only-token",
        )


@pytest.mark.asyncio
async def test_valid_client_uses_https_fixed_port_and_only_selected_machine_ca(tmp_path, test_ca):
    ca_pem = test_ca.read_text(encoding="ascii")
    other_ca = _test_ca(tmp_path / "other-machine-ca")
    client = _client(_config({
        "machine-b": other_ca.read_text(encoding="ascii"),
        "machine-a": ca_pem,
    }))
    try:
        assert client.base_url == "https://127.0.0.1:15678"
        assert client.node_host == "127.0.0.1"
        assert client._machine_alias == "machine-a"
        assert client._ssl_ctx.verify_mode == ssl.CERT_REQUIRED
        assert client._ssl_ctx.check_hostname is True
        assert client._ssl_ctx.get_ca_certs(binary_form=True) == [ssl.PEM_cert_to_DER_cert(ca_pem)]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_client_ssl_context_real_handshake_with_valid_ca_and_ip_sends_bearer(tmp_path, test_ca):
    cert_file, key_file = _ephemeral_test_certificate(tmp_path / "server", test_ca)
    seen = []
    client = _client(_config({"machine-a": test_ca.read_text(encoding="ascii")}))
    try:
        async with _tls_test_server(cert_file, key_file, seen) as port:
            async with httpx.AsyncClient(
                verify=client._ssl_ctx, trust_env=False, timeout=5,
                headers=client._headers(), follow_redirects=False,
            ) as probe:
                response = await probe.get(f"https://127.0.0.1:{port}/tls-probe")
            assert response.status_code == 200
            assert response.json() == {"tls": "ok"}
            assert b"authorization: bearer test-only-token\r\n" in b"".join(seen).lower()
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure, server_ip, verify_code",
    [("wrong-ip-san", "127.0.0.2", 64), ("wrong-chain", "127.0.0.1", 20), ("expired", "127.0.0.1", 10)],
    ids=["wrong-ip-san", "wrong-certificate-chain", "expired-certificate"],
)
async def test_client_ssl_context_real_handshake_rejects_invalid_peer_without_bearer(
    tmp_path, test_ca, failure, server_ip, verify_code,
):
    server_ca = _test_ca(tmp_path / "untrusted-ca") if failure == "wrong-chain" else test_ca
    cert_file, key_file = _ephemeral_test_certificate(
        tmp_path / "server", server_ca, server_ip, expired=failure == "expired",
    )
    seen = []
    # A valid CA always permits construction, even when the server's IP SAN,
    # issuer or validity will fail verification during the subsequent handshake.
    client = _client(_config({"machine-a": test_ca.read_text(encoding="ascii")}))
    try:
        async with _tls_test_server(cert_file, key_file, seen) as port:
            async with httpx.AsyncClient(
                verify=client._ssl_ctx, trust_env=False, timeout=5,
                headers=client._headers(), follow_redirects=False,
            ) as probe:
                with pytest.raises(httpx.ConnectError) as raised:
                    await probe.get(f"https://127.0.0.1:{port}/tls-probe")
            # Require a certificate verification failure, not an unrelated
            # connection error which could otherwise make this test pass.
            cause = raised.value
            while cause is not None and not isinstance(cause, ssl.SSLCertVerificationError):
                cause = cause.__cause__ or cause.__context__
            assert isinstance(cause, ssl.SSLCertVerificationError)
            assert cause.verify_code == verify_code
        assert seen == []  # No HTTP bytes or Authorization reached the server.
    finally:
        await client.aclose()


def test_secrets_loader_rejects_bad_ca_input(tmp_path):
    secrets = tmp_path / "test-secrets.yaml"
    secrets.write_text(
        "version: 1\nagent_core_ca_pem:\n  machine-a: 'not a certificate'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="CA PEM|agent_core_ca_pem"):
        _load_secrets_config(str(secrets))


def test_secrets_loader_rejects_invalid_machine_alias(tmp_path, test_ca):
    secrets = tmp_path / "test-secrets.yaml"
    indented_pem = "\n".join("    " + line for line in test_ca.read_text(encoding="ascii").splitlines())
    secrets.write_text(
        "version: 1\nagent_core_ca_pem:\n  ' ': |\n" + indented_pem + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="machine aliases"):
        _load_secrets_config(str(secrets))


@pytest.mark.asyncio
async def test_tls_compatibility_mode_without_ca(tmp_path, caplog):
    """Legacy compatibility now requires an explicit mode and policy binding."""
    caplog.set_level("WARNING", logger="agents.deploy_approval.agent_core_client")
    config = _migration_config(tmp_path)
    seen = []

    def respond(request):
        seen.append(request)
        return httpx.Response(200, json={"valid": True, "auth_required": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False, follow_redirects=False) as http:
        client = _migration_client(config, http=http)
        assert client.legacy_tls is True
        assert await client.verify() == {"valid": True, "auth_required": True}
    assert len(seen) == 1
    assert seen[0].headers["Authorization"] == "Bearer test-only-token"
    assert "LEGACY_TLS_UNVERIFIED=YES" in caplog.text
    assert "TLS_CERT_VERIFIED=NO" in caplog.text
    assert "test-only-token" not in caplog.text


@pytest.mark.parametrize("cas", [{}, {"machine-a": " "}, {"machine-b": "wrong-alias"}])
def test_tls_strict_mode_requires_ca(cas):
    with pytest.raises(AgentCoreError, match="no valid agent_core_ca_pem"):
        _client(_config(cas, mode="strict"))


@pytest.mark.asyncio
async def test_tls_strict_mode_requires_ip_san(tmp_path, test_ca, caplog):
    caplog.set_level("INFO", logger="agents.deploy_approval.agent_core_client")
    # The CA is trusted and CN equals the IPv4, but SAN is absent.
    cert, key = _ephemeral_test_certificate(tmp_path / "no-san", test_ca, ip=None)
    client = _client(_config({"machine-a": test_ca.read_text(encoding="ascii")}, mode="strict"))
    seen = []
    try:
        assert client.legacy_tls is False
        assert client._ssl_ctx.verify_mode == ssl.CERT_REQUIRED
        assert client._ssl_ctx.check_hostname is True
        async with _tls_test_server(cert, key, seen) as port:
            async with httpx.AsyncClient(
                verify=client._ssl_ctx, trust_env=False, timeout=5,
                headers=client._headers(), follow_redirects=False,
            ) as probe:
                with pytest.raises(httpx.ConnectError) as raised:
                    await probe.get(f"https://127.0.0.1:{port}/tls-probe")
            cause = raised.value
            while cause is not None and not isinstance(cause, ssl.SSLCertVerificationError):
                cause = cause.__cause__ or cause.__context__
            assert isinstance(cause, ssl.SSLCertVerificationError)
        assert seen == []
        assert "TLS_MODE=strict" in caplog.text
        assert "test-only-token" not in caplog.text
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_tls_legacy_mode_keeps_https(tmp_path):
    config = _migration_config(tmp_path)
    for url, host, error in (
        ("http://10.23.0.5:15678", "10.23.0.5", "scheme must be https"),
        ("https://10.23.0.5:443", "10.23.0.5", "port must be 15678"),
        ("https://10.23.0.6:15678", "10.23.0.5", "hostname must match node_host"),
        ("https://user:secret@10.23.0.5:15678", "10.23.0.5", "userinfo"),
    ):
        with pytest.raises(AgentCoreError, match=error):
            AgentCoreClient(config, url, node_host=host, machine_alias="machine-a")

    seen = []
    responses = [
        httpx.Response(302, headers={"Location": "http://other-host/"}),
        httpx.Response(200, json={"valid": False, "auth_required": True}),
        httpx.Response(200, json={"valid": True, "auth_required": True}),
    ]

    def respond(request):
        seen.append(request)
        return responses.pop(0)

    # Offline HTTP contract probe: no real Agent Core request or deployment.
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False, follow_redirects=False) as http:
        client = AgentCoreClient(
            config, "https://10.23.0.5:15678", node_host="10.23.0.5",
            machine_alias="machine-a", access_token="test-only-token", http=http,
        )
        with pytest.raises(AgentCoreError, match="302"):
            await client.verify()
        assert len(seen) == 1  # The redirect target received no token or request.
        with pytest.raises(AgentCoreError, match="valid=true and auth_required=true"):
            await client.verify()
        assert await client.verify() == {"valid": True, "auth_required": True}
    assert len(seen) == 3
    for request in seen:
        assert str(request.url) == "https://10.23.0.5:15678/api/auth/verify"
        assert request.headers["Authorization"] == "Bearer test-only-token"


def _migration_config(tmp_path, *, host="10.23.0.5", mode="legacy"):
    """Explicitly controlled offline machine policy, never a production fallback."""
    machines = tmp_path / "machines.yaml"
    machines.write_text(yaml.safe_dump({"version": 1, "machines": {
        "machine-a": {
            "node_id": "offline-node", "node_host": host, "owners": ["offline-owner"],
            "targets": ["core", "perception", "actucore"],
            "platforms": ["linux/arm64"], "variants": ["5.11", "6.1"],
        },
    }}), encoding="utf-8")
    return Config(
        machine_owners_file=str(machines), agent_core_tls_mode=mode,
        agent_core_tokens={"machine-a": "test-only-token"}, agent_core_ca_pem={},
    )


def _migration_client(config, **overrides):
    arguments = dict(
        base_url="https://10.23.0.5:15678", node_host="10.23.0.5",
        machine_alias="machine-a", access_token="test-only-token",
    )
    arguments.update(overrides)
    return AgentCoreClient(config, **arguments)


def _owned_mock_transport(monkeypatch, handler):
    """Keep real owned AsyncClient construction, replacing only the wire transport."""
    original = httpx.AsyncClient
    calls = []

    def create(**kwargs):
        calls.append(kwargs.copy())
        return original(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", create)
    return calls


@pytest.mark.asyncio
async def test_tls_legacy_no_ca_client_connects(tmp_path, monkeypatch, caplog):
    caplog.set_level("WARNING", logger="agents.deploy_approval.agent_core_client")
    config = _migration_config(tmp_path)
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.path == "/api/auth/verify":
            return httpx.Response(200, json={"valid": True, "auth_required": True})
        return httpx.Response(200, json={"code": 0, "data": {"status": "updating"}})

    construction = _owned_mock_transport(monkeypatch, respond)
    client = _migration_client(config)
    try:
        assert config.agent_core_ca_pem == {}
        assert construction[0]["verify"] is False
        assert await client.verify() == {"valid": True, "auth_required": True}
        assert (await client.update_core("registry.example/core:v17"))["data"]["status"] == "updating"
        assert [r.method for r in requests] == ["GET", "GET", "POST"]
        assert all(r.headers["Authorization"] == "Bearer test-only-token" for r in requests)
        assert all(str(r.url).startswith("https://10.23.0.5:15678/") for r in requests)
        assert "LEGACY_TLS_UNVERIFIED=YES" in caplog.text
        assert "TLS_CERT_VERIFIED=NO" in caplog.text
        assert "test-only-token" not in caplog.text
    finally:
        await client.aclose()
    assert client.http.is_closed


@pytest.mark.asyncio
async def test_tls_legacy_https_exact_ipv4_port_only(tmp_path):
    config = _migration_config(tmp_path)
    for url, host in (
        ("http://10.23.0.5:15678", "10.23.0.5"),
        ("https://10.23.0.5:443", "10.23.0.5"),
        ("https://10.23.0.5", "10.23.0.5"),
        ("https://10.23.0.5:bad", "10.23.0.5"),
        ("https://10.23.0.6:15678", "10.23.0.5"),
        ("https://device.local:15678", "device.local"),
        ("https://user:secret@10.23.0.5:15678", "10.23.0.5"),
        ("https://10.23.0.5:15678/api", "10.23.0.5"),
        ("https://10.23.0.5:15678?token=secret", "10.23.0.5"),
        ("https://10.23.0.5:15678#extra", "10.23.0.5"),
    ):
        with pytest.raises(AgentCoreError):
            _migration_client(config, base_url=url, node_host=host)
    for host in ("10.23.0.5", "172.20.0.5", "192.168.10.5"):
        config = _migration_config(tmp_path, host=host)
        client = _migration_client(config, base_url=f"https://{host}:15678", node_host=host)
        await client.aclose()  # Construction only; no network request.


def test_tls_legacy_rejects_non_rfc1918_hosts(tmp_path, monkeypatch):
    construction = _owned_mock_transport(monkeypatch, lambda r: httpx.Response(200))
    for host in ("8.8.8.8", "127.0.0.1", "169.254.10.5", "100.64.0.5",
                 "192.0.2.1", "0.0.0.0", "224.0.0.1", "172.32.0.1", "192.169.0.1"):
        config = _migration_config(tmp_path, host=host)
        with pytest.raises(AgentCoreError, match="RFC1918"):
            _migration_client(config, base_url=f"https://{host}:15678", node_host=host)
    assert construction == []


@pytest.mark.asyncio
async def test_tls_legacy_disables_redirects_and_proxies(tmp_path, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://untrusted.invalid:9999")
    monkeypatch.setenv("ALL_PROXY", "http://untrusted.invalid:9999")
    requests = []

    def redirect(request):
        requests.append(request)
        return httpx.Response(302, headers={"Location": "http://untrusted.invalid/"})

    construction = _owned_mock_transport(monkeypatch, redirect)
    client = _migration_client(_migration_config(tmp_path))
    try:
        assert construction[0]["trust_env"] is False
        assert construction[0]["follow_redirects"] is False
        assert construction[0]["verify"] is False
        assert client.http.trust_env is False
        assert client.http.follow_redirects is False
        assert client.http.timeout.connect == client.config.connect_timeout
        assert client.http.timeout.read == client.config.read_timeout
        with pytest.raises(AgentCoreError, match="302"):
            await client.update_core("registry.example/core:v17")
        assert len(requests) == 1
        assert requests[0].method == "GET" and requests[0].url.path == "/api/auth/verify"
        assert requests[0].url.host == "10.23.0.5"
    finally:
        await client.aclose()


def test_tls_legacy_missing_alias_or_token_denied(tmp_path, monkeypatch):
    construction = _owned_mock_transport(monkeypatch, lambda r: httpx.Response(200))
    for overrides in ({"machine_alias": ""}, {"machine_alias": "unknown"},
                      {"access_token": ""}, {"access_token": "wrong-token"},
                      {"base_url": "https://10.23.0.6:15678", "node_host": "10.23.0.6"}):
        with pytest.raises(AgentCoreError):
            _migration_client(_migration_config(tmp_path), **overrides)
    config = _migration_config(tmp_path)
    config.agent_core_tokens = {}
    with pytest.raises(AgentCoreError, match="configured token"):
        _migration_client(config)
    config = _migration_config(tmp_path)
    config.agent_core_tokens["unknown"] = "test-only-token"
    with pytest.raises(AgentCoreError, match="match the machine policy"):
        _migration_client(config, machine_alias="unknown")
    config = _migration_config(tmp_path)
    config.agent_core_ca_pem = {"machine-a": "invalid PEM"}
    with pytest.raises(AgentCoreError, match="CA configuration is invalid"):
        _migration_client(config)
    config = _migration_config(tmp_path)
    config.machine_owners_file = str(tmp_path / "missing.yaml")
    with pytest.raises(AgentCoreError, match="machine policy"):
        _migration_client(config)
    assert construction == []


@pytest.mark.asyncio
async def test_tls_legacy_auth_invalid_rejected(tmp_path, caplog):
    config = _migration_config(tmp_path)
    failures = [
        httpx.Response(401), httpx.Response(403),
        httpx.Response(200, json={"valid": False, "auth_required": True}),
        httpx.Response(200, json={"valid": True, "auth_required": False}),
        httpx.Response(200, json={"valid": "true", "auth_required": True}),
        httpx.Response(200, json={"code": 0, "data": {"valid": False, "auth_required": True}}),
        httpx.Response(200, content=b"not-json"), httpx.Response(200, json=[]),
        "timeout",
    ]
    for failure in failures:
        requests = []

        def respond(request):
            requests.append(request)
            if failure == "timeout":
                raise httpx.ReadTimeout("offline auth timeout", request=request)
            return failure

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False, follow_redirects=False) as http:
            client = _migration_client(config, http=http)
            for operation in (
                lambda: client.update_core("registry.example/core:v17"),
                lambda: client.deploy_driver("perception", "registry.example/perception:v17"),
                lambda: client.request("POST", "/api/system/update", {"image": "registry.example/core:v17"}),
            ):
                with pytest.raises(AgentCoreError) as raised:
                    await operation()
                assert not isinstance(raised.value, AgentCoreDeployOutcomeUncertain)
            assert requests and all(r.method == "GET" and r.url.path == "/api/auth/verify" for r in requests)
            assert "test-only-token" not in caplog.text


@pytest.mark.asyncio
async def test_tls_legacy_post_uncertain_no_retry(tmp_path):
    config = _migration_config(tmp_path)
    for endpoint in ("core", "driver"):
        for failure in ("timeout", "reset", "non-json", "bad-envelope"):
            requests = []

            def respond(request):
                requests.append(request)
                if request.method == "GET":
                    return httpx.Response(200, json={"valid": True, "auth_required": True})
                if failure == "timeout":
                    raise httpx.ReadTimeout("offline POST timeout", request=request)
                if failure == "reset":
                    raise httpx.ReadError("offline connection reset", request=request)
                if failure == "non-json":
                    return httpx.Response(200, content=b"unknown outcome")
                return httpx.Response(200, json={"code": True, "data": {}})

            async with httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False, follow_redirects=False) as http:
                client = _migration_client(config, http=http)
                with pytest.raises(AgentCoreDeployOutcomeUncertain):
                    if endpoint == "core":
                        await client.update_core("registry.example/core:v17")
                    else:
                        await client.deploy_driver("perception", "registry.example/perception:v17")
            assert [r.method for r in requests] == ["GET", "POST"]


def test_tls_strict_no_ca_fails_closed(tmp_path, monkeypatch):
    construction = _owned_mock_transport(monkeypatch, lambda r: httpx.Response(200))
    for cas in ({}, {"machine-a": " "}, {"other-machine": "not-selected"}):
        config = _migration_config(tmp_path, mode="strict")
        config.agent_core_ca_pem = cas
        with pytest.raises(AgentCoreError, match="no valid agent_core_ca_pem"):
            _migration_client(config)
    assert construction == []


@pytest.mark.asyncio
async def test_tls_strict_no_fallback(tmp_path, monkeypatch, offline_ca_pem, caplog):
    config = _migration_config(tmp_path, mode="strict")
    config.agent_core_ca_pem = {"machine-a": "invalid PEM"}
    with pytest.raises(AgentCoreError, match="CA PEM.*invalid"):
        _migration_client(config)
    requests = []

    def bad_certificate(request):
        requests.append(request)
        raise httpx.ConnectError("offline certificate verification failed", request=request)

    construction = _owned_mock_transport(monkeypatch, bad_certificate)
    config.agent_core_ca_pem = {"machine-a": offline_ca_pem}
    client = _migration_client(config)
    try:
        with pytest.raises(AgentCoreError, match="certificate verification failed"):
            await client.verify()
        assert len(construction) == len(requests) == 1
        assert isinstance(construction[0]["verify"], ssl.SSLContext)
        assert construction[0]["verify"].verify_mode == ssl.CERT_REQUIRED
        assert "LEGACY_TLS_UNVERIFIED" not in caplog.text
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_tls_strict_verified_ssl_context(tmp_path, monkeypatch, offline_ca_pem):
    config = _migration_config(tmp_path, mode="strict")
    config.agent_core_ca_pem = {"machine-a": offline_ca_pem}
    construction = _owned_mock_transport(monkeypatch, lambda r: httpx.Response(200))
    client = _migration_client(config)
    try:
        assert client.legacy_tls is False
        context = construction[0]["verify"]
        assert context is client._ssl_ctx
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True
        assert context.get_ca_certs(binary_form=True) == [ssl.PEM_cert_to_DER_cert(offline_ca_pem)]
        assert construction[0]["trust_env"] is False
        assert construction[0]["follow_redirects"] is False
    finally:
        await client.aclose()


def test_tls_mode_invalid_rejected(tmp_path, monkeypatch):
    from .. import config as config_module

    secrets = tmp_path / "secrets.yaml"
    secrets.write_text(yaml.safe_dump({
        "version": 1, "agent_core_tokens": {"machine-a": "test-only-token"},
        "cos": {"region": "ap-beijing", "bucket": "offline-bucket-123",
                "secret_id": "OFFLINE_ID", "secret_key": "OFFLINE_KEY"},
        "review_comment_trust": {"author_id": "123456", "author_login": "offline-review"},
    }), encoding="utf-8")
    loader = _load_secrets_config
    monkeypatch.setattr(config_module, "_load_secrets_config", lambda path: loader(str(secrets)))
    monkeypatch.setenv("GITHUB_REPOS", "example/main,example/drivers")
    for mode in ("", "compatibility", "LEGACY", "strcit", " legacy", "legacy\n"):
        monkeypatch.setenv("AGENT_CORE_TLS_MODE", mode)
        with pytest.raises(ValueError, match="AGENT_CORE_TLS_MODE"):
            config_module.load_config()
        with pytest.raises(AgentCoreError, match="AGENT_CORE_TLS_MODE"):
            _migration_client(_migration_config(tmp_path, mode=mode))
    monkeypatch.delenv("AGENT_CORE_TLS_MODE")
    assert Config().agent_core_tls_mode == "strict"
    assert config_module.load_config().agent_core_tls_mode == "strict"
    monkeypatch.setenv("AGENT_CORE_TLS_MODE", "legacy")
    loaded = config_module.load_config()
    assert loaded.agent_core_tls_mode == "legacy" and loaded.agent_core_ca_pem == {}
    assert loaded.agent_core_tokens == {"machine-a": "test-only-token"}
    assert loaded.cos_secret_id == "OFFLINE_ID" and loaded.cos_secret_key == "OFFLINE_KEY"
    assert loaded.review_comment_author_id == "123456" and loaded.review_comment_author_login == "offline-review"
    assert loaded.github_repos == ["example/main", "example/drivers"]
    for mode in ("legacy", "strict"):
        monkeypatch.setenv("AGENT_CORE_TLS_MODE", mode)
        data = yaml.safe_load(secrets.read_text())
        data["agent_core_ca_pem"] = {"machine-a": "invalid PEM"}
        secrets.write_text(yaml.safe_dump(data), encoding="utf-8")
        with pytest.raises(ValueError, match="agent_core_ca_pem"):
            config_module.load_config()
