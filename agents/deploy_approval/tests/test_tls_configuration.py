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

from ..agent_core_client import AgentCoreClient, AgentCoreError
from ..config import Config, _load_secrets_config


def _config(cas: dict[str, str]) -> Config:
    return Config(
        allow_private_http=True,
        agent_core_tokens={"machine-a": "test-only-token"},
        agent_core_ca_pem=cas,
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
    directory: Path, ca_cert: Path, ip: str = "127.0.0.1", *, expired: bool = False,
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
        f"subjectAltName = IP:{ip}\n",
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
