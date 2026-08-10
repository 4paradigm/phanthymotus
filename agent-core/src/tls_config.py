"""Resolve the TLS certificate pair used by the Agent Core web server.

Importing this module has no filesystem or process side effects.  Certificate
generation happens only when :func:`resolve_tls_certificates` is called with no
explicit certificate configuration and neither compatibility file exists.
"""

from __future__ import annotations

import os
import pathlib
import re
import ssl
import stat
import subprocess
from collections.abc import Mapping

TLS_CERT_FILE_ENV = 'MOTUS_TLS_CERT_FILE'
TLS_KEY_FILE_ENV = 'MOTUS_TLS_KEY_FILE'
DEFAULT_CERT_DIR = pathlib.Path('./resource/certs')
MAX_PUBLIC_CERTIFICATE_CHAIN_BYTES = 32 * 1024

_CERTIFICATE_BLOCK = re.compile(
    r'-----BEGIN CERTIFICATE-----\s*'
    r'(?P<body>[A-Za-z0-9+/=\r\n]+?)\s*'
    r'-----END CERTIFICATE-----'
)


class TLSConfigurationError(RuntimeError):
    """Raised when TLS files are incomplete or unusable."""


def project_public_certificate_chain(pem: str) -> str:
    """Return a bounded, certificate-only normalized PEM chain.

    The configured server certfile can contain a full public chain, but it is
    never safe to expose that file verbatim: an operator could accidentally
    append unrelated material to it. Only complete parseable CERTIFICATE
    blocks survive this projection. Paths, comments, and all key material are
    rejected instead of being reflected into the capture bootstrap response.
    """

    if not isinstance(pem, str):
        raise TLSConfigurationError('TLS public certificate chain must be text')
    try:
        encoded = pem.encode('ascii')
    except UnicodeEncodeError as exc:
        raise TLSConfigurationError(
            'TLS public certificate chain must contain ASCII PEM only'
        ) from exc
    if not encoded or len(encoded) > MAX_PUBLIC_CERTIFICATE_CHAIN_BYTES:
        raise TLSConfigurationError(
            'TLS public certificate chain exceeds the capture bootstrap limit'
        )
    if 'PRIVATE KEY' in pem or '\x00' in pem:
        raise TLSConfigurationError(
            'TLS public certificate chain contains forbidden material'
        )

    normalized_blocks: list[str] = []
    cursor = 0
    for match in _CERTIFICATE_BLOCK.finditer(pem):
        if pem[cursor:match.start()].strip():
            raise TLSConfigurationError(
                'TLS public certificate file must contain certificates only'
            )
        body = ''.join(match.group('body').split())
        if not body:
            raise TLSConfigurationError('TLS public certificate block is empty')
        wrapped = '\n'.join(body[index:index + 64] for index in range(0, len(body), 64))
        block = (
            '-----BEGIN CERTIFICATE-----\n'
            f'{wrapped}\n'
            '-----END CERTIFICATE-----'
        )
        try:
            der = ssl.PEM_cert_to_DER_cert(block)
        except ValueError as exc:
            raise TLSConfigurationError(
                'TLS public certificate chain contains malformed PEM'
            ) from exc
        if not der or der[0] != 0x30:
            raise TLSConfigurationError(
                'TLS public certificate chain contains malformed DER'
            )
        normalized_blocks.append(block)
        cursor = match.end()

    if pem[cursor:].strip() or not normalized_blocks:
        raise TLSConfigurationError(
            'TLS public certificate file must contain certificates only'
        )
    normalized = '\n'.join(normalized_blocks) + '\n'
    if len(normalized.encode('ascii')) > MAX_PUBLIC_CERTIFICATE_CHAIN_BYTES:
        raise TLSConfigurationError(
            'TLS public certificate chain exceeds the capture bootstrap limit'
        )
    return normalized


def load_public_certificate_chain(cert_file: str | pathlib.Path) -> str:
    """Read and project the public chain used by the running HTTPS server."""

    path = pathlib.Path(cert_file)
    _require_regular_file(path, TLS_CERT_FILE_ENV)
    try:
        with path.open('rb') as certificate_stream:
            raw = certificate_stream.read(MAX_PUBLIC_CERTIFICATE_CHAIN_BYTES + 1)
    except OSError as exc:
        raise TLSConfigurationError(
            'TLS public certificate chain could not be read safely'
        ) from exc
    if len(raw) > MAX_PUBLIC_CERTIFICATE_CHAIN_BYTES:
        raise TLSConfigurationError(
            'TLS public certificate chain exceeds the capture bootstrap limit'
        )
    try:
        pem = raw.decode('ascii')
    except UnicodeDecodeError as exc:
        raise TLSConfigurationError(
            'TLS public certificate chain must contain ASCII PEM only'
        ) from exc
    return project_public_certificate_chain(pem)


def _explicit_path(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, '')
    return value.strip() if isinstance(value, str) else ''


def _require_regular_file(path: pathlib.Path, setting: str) -> None:
    if not path.is_file():
        raise TLSConfigurationError(
            f'{setting} must point to an existing regular file inside the Core runtime'
        )


def _require_distinct_files(cert_path: pathlib.Path, key_path: pathlib.Path) -> None:
    try:
        same_file = cert_path.samefile(key_path)
    except OSError as exc:
        raise TLSConfigurationError(
            'TLS certificate and private-key files could not be compared safely'
        ) from exc
    if same_file:
        raise TLSConfigurationError(
            'TLS certificate and private key must be different files'
        )


def _private_key_mode(key_path: pathlib.Path) -> int:
    try:
        return stat.S_IMODE(key_path.stat().st_mode)
    except OSError as exc:
        raise TLSConfigurationError(
            'TLS private-key permissions could not be inspected safely'
        ) from exc


def _require_private_key_permissions(key_path: pathlib.Path) -> None:
    if os.name == 'posix' and _private_key_mode(key_path) & 0o077:
        raise TLSConfigurationError(
            'TLS private key must not be accessible by group or other users; '
            'set its mode to 0600 or 0400'
        )


def _secure_compatibility_key(key_path: pathlib.Path) -> None:
    if os.name != 'posix' or not (_private_key_mode(key_path) & 0o077):
        return
    if key_path.is_symlink():
        raise TLSConfigurationError(
            'a linked compatibility TLS private key has broad permissions; '
            'fix the managed target or configure the explicit TLS pair'
        )

    flags = os.O_RDONLY
    flags |= getattr(os, 'O_CLOEXEC', 0)
    flags |= getattr(os, 'O_NOFOLLOW', 0)
    try:
        key_fd = os.open(key_path, flags)
    except OSError as exc:
        raise TLSConfigurationError(
            'the compatibility TLS private key could not be opened safely'
        ) from exc
    try:
        metadata = os.fstat(key_fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise TLSConfigurationError(
                'a multiply-linked compatibility TLS private key cannot be modified; '
                'fix the managed target or configure the explicit TLS pair'
            )
        os.fchmod(key_fd, 0o600)
    except OSError as exc:
        raise TLSConfigurationError(
            'the compatibility TLS private key could not be restricted to mode 0600'
        ) from exc
    finally:
        os.close(key_fd)
    _require_private_key_permissions(key_path)


def _validate_certificate_pair(cert_path: pathlib.Path, key_path: pathlib.Path) -> None:
    """Prove that Python's server TLS stack can load this matching pair."""

    try:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(
            certfile=str(cert_path),
            keyfile=str(key_path),
            # Core has no interactive secret-input path. An encrypted key must
            # fail immediately instead of blocking startup on an OpenSSL prompt.
            password=lambda: b'',
        )
    except (OSError, ssl.SSLError) as exc:
        raise TLSConfigurationError(
            'TLS certificate and private key must be a parseable matching server pair'
        ) from exc


def _resolve_explicit_pair(environ: Mapping[str, str]) -> tuple[str, str] | None:
    cert_value = _explicit_path(environ, TLS_CERT_FILE_ENV)
    key_value = _explicit_path(environ, TLS_KEY_FILE_ENV)

    if bool(cert_value) != bool(key_value):
        raise TLSConfigurationError(
            f'{TLS_CERT_FILE_ENV} and {TLS_KEY_FILE_ENV} must be configured together'
        )
    if not cert_value:
        return None

    cert_path = pathlib.Path(cert_value)
    key_path = pathlib.Path(key_value)
    _require_regular_file(cert_path, TLS_CERT_FILE_ENV)
    _require_regular_file(key_path, TLS_KEY_FILE_ENV)
    _require_distinct_files(cert_path, key_path)
    _require_private_key_permissions(key_path)
    _validate_certificate_pair(cert_path, key_path)
    print(
        '[tls] Using explicitly configured certificate files; '
        'browser trust is determined independently by the client.'
    )
    return str(cert_path), str(key_path)


def _ensure_compatibility_pair(cert_dir: str | pathlib.Path) -> tuple[str, str]:
    directory = pathlib.Path(cert_dir)
    cert_path = directory / 'cert.pem'
    key_path = directory / 'key.pem'

    # A broken symlink is still an occupied deployment path. Treat it as an
    # invalid existing file so openssl can never follow or replace it.
    cert_exists = cert_path.exists() or cert_path.is_symlink()
    key_exists = key_path.exists() or key_path.is_symlink()
    if cert_exists or key_exists:
        if cert_path.is_file() and key_path.is_file():
            _require_distinct_files(cert_path, key_path)
            _secure_compatibility_key(key_path)
            _validate_certificate_pair(cert_path, key_path)
            print(
                '[tls] Using the existing compatibility certificate pair; '
                'browser trust is not inferred.'
            )
            return str(cert_path), str(key_path)
        raise TLSConfigurationError(
            'the compatibility TLS certificate pair is incomplete or not made of '
            f'regular files: {cert_path} and {key_path}'
        )

    directory.mkdir(parents=True, exist_ok=True)
    try:
        # OpenSSL truncates this existing file without widening its mode.  The
        # private key therefore never has a group/world-readable creation
        # window, even when the host process umask is permissive.
        key_fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(key_fd)
    except OSError as exc:
        raise TLSConfigurationError(
            'failed to reserve the compatibility TLS private key securely'
        ) from exc
    try:
        subprocess.run(
            [
                'openssl', 'req', '-x509', '-newkey', 'rsa:2048',
                '-keyout', str(key_path), '-out', str(cert_path),
                '-days', '3650', '-nodes',
                '-subj', '/CN=phanthy-motus',
            ],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TLSConfigurationError(
            'failed to generate the self-signed compatibility TLS certificate pair'
        ) from exc

    if not cert_path.is_file() or not key_path.is_file():
        raise TLSConfigurationError(
            'openssl completed without creating both compatibility TLS files'
        )
    _require_distinct_files(cert_path, key_path)
    _require_private_key_permissions(key_path)
    _validate_certificate_pair(cert_path, key_path)

    print(
        '[tls] Generated a self-signed compatibility fallback certificate; '
        'this does not establish browser trust or Quest WebXR readiness.'
    )
    return str(cert_path), str(key_path)


def resolve_tls_certificates(
    *,
    environ: Mapping[str, str] | None = None,
    cert_dir: str | pathlib.Path = DEFAULT_CERT_DIR,
) -> tuple[str, str]:
    """Return the certificate and private-key paths for Uvicorn.

    An explicit pair is fail-closed and is never generated or modified.  With
    no explicit pair, the historical ``resource/certs`` behavior remains, but
    a partial or non-file default pair is rejected instead of overwritten.
    """

    configured = _resolve_explicit_pair(os.environ if environ is None else environ)
    if configured is not None:
        return configured
    return _ensure_compatibility_pair(cert_dir)
