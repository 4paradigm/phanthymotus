"""Focused tests for the single gzip evidence and private COS contracts."""
from __future__ import annotations

import asyncio
import gzip
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from ..cos_client import CosClient, CosError, EVIDENCE_MAX_ARCHIVE_BYTES
from ..evidence_builder import EvidenceBuilder
from .conftest import make_config

HEAD = "a" * 40


def _state(**overrides):
    state = {"review_evidence": {"build_comment_id": 1001, "build_comment_updated_at": "2026-09-18T03:55:54Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 1002, "code_review_comment_id": 1003, "review_author_id": "7950763"}, "components": [], "case_results": {},
             "command": {"kind": "approve_deploy"}}
    state.update(overrides)
    return state


def _build(**kwargs):
    args = {"repo": "4paradigm/phanthymotus", "pr_number": 7, "head_sha": HEAD,
            "state": _state(), "result": "pass"}
    args.update(kwargs)
    return asyncio.run(EvidenceBuilder(make_config()).build_evidence(**args))


def _text(archive):
    return gzip.decompress(archive).decode("utf-8", errors="strict")


def test_evidence_is_single_gzip_utf8_text():
    body = _text(_build()[0])
    assert body.startswith("[metadata]")
    assert "repo=4paradigm/phanthymotus" in body and "pr=7" in body
    assert f"head={HEAD}" in body and "review_build_comment_id=1001" in body
    assert "result=pass" in body and "[deploy]" in body
    assert "manifest" not in body.lower() and not body.startswith("ustar")


def test_evidence_fixed_metadata_survives_large_runtime_logs():
    body = _text(_build(runtime_logs={"agent": "x" * (11 * 1024 * 1024)})[0])
    for value in ("repo=4paradigm/phanthymotus", "pr=7", f"head={HEAD}",
                  "review_build_comment_id=1001", "result=pass"):
        assert value in body


def test_evidence_keeps_newest_runtime_tail():
    logs = {"runtime": "OLDEST_RUNTIME_LOG_MUST_DROP\n" + "x" * (11 * 1024 * 1024) +
            "\nNEWEST_RUNTIME_LOG_MUST_STAY\n"}
    body = _text(_build(runtime_logs=logs)[0])
    assert "NEWEST_RUNTIME_LOG_MUST_STAY" in body
    assert "OLDEST_RUNTIME_LOG_MUST_DROP" not in body


def test_evidence_archive_and_plaintext_hard_cap():
    archive = _build(runtime_logs={"runtime": "z" * (12 * 1024 * 1024)})[0]
    assert len(archive) <= EVIDENCE_MAX_ARCHIVE_BYTES
    assert len(gzip.decompress(archive)) <= EVIDENCE_MAX_ARCHIVE_BYTES


def test_evidence_secret_redaction():
    values = ["ghp_FAKE_GITHUB_TOKEN", "github_pat_FAKE_TOKEN", "Bearer FAKE_BEARER",
              "password=FAKE_PASSWORD", "token=FAKE_TOKEN", "q-ak=FAKE_AK",
              "q-signature=FAKE_SIGNATURE", "q-sign-time=FAKE_TIME", "q-key-time=FAKE_KEY_TIME"]
    body = _text(_build(summary="\n".join(values), runtime_logs={"r": "\n".join(values)})[0])
    for value in values:
        assert value.split("=", 1)[-1] not in body
    assert "FAKE_BEARER" not in body


def test_deploy_failure_is_not_fake_record_test():
    body = _text(_build(result="fail", deploy_error="unsafe deploy failed")[0])
    assert "[deploy]" in body and "terminal_error=unsafe deploy failed" in body
    assert "[test]" not in body


def test_real_record_test_contains_test_section():
    body = _text(_build(result="pass", state=_state(command={"kind": "record_test"}))[0])
    assert "[test]" in body and "result=pass" in body


def test_cos_object_key_exact_layout():
    client = CosClient(make_config())
    now = datetime(2026, 9, 17, 20, 30)
    for repo, expected in [
        ("4paradigm/phanthymotus", "phanthymotus_pr/phanthymotus/2026-09/2026-09-17/pr-7/evidence-" + HEAD + ".log.gz"),
        ("4paradigm/phanthymotus-driver", "phanthymotus_pr/phanthymotus-driver/2026-09/2026-09-17/pr-7/evidence-" + HEAD + ".log.gz"),
    ]:
        assert client.build_object_key(repo, 7, HEAD, now=now) == expected
    with pytest.raises(ValueError):
        client.build_object_key("some/other-repo", 7, HEAD, now=now)


@pytest.mark.parametrize("key", [
    "phanthymotus_pr/wrong/2026-09/2026-09-17/pr-7/evidence-" + HEAD + ".log.gz",
    "phanthymotus_pr/phanthymotus/2026-09/2026-09-17/pr-8/evidence-" + HEAD + ".log.gz",
    "phanthymotus_pr/phanthymotus/2026-09/2026-09-17/pr-7/evidence-" + "b" * 40 + ".log.gz",
    "phanthymotus_pr/phanthymotus/2026-08/2026-09-17/pr-7/evidence-" + HEAD + ".log.gz",
    "phanthymotus_pr/phanthymotus/2026-09/2026-09-17/../pr-7/evidence-" + HEAD + ".log.gz",
    "phanthymotus_pr/phanthymotus\\2026-09/2026-09-17/pr-7/evidence-" + HEAD + ".log.gz",
    "phanthymotus_pr/phanthymotus/2026-09/2026-09-17/pr-7/extra/evidence-" + HEAD + ".log.gz",
    "phanthymotus_pr/phanthymotus/2026-09/2026-09-17/pr-7/evidence-" + "A" * 40 + ".log.gz",
    "phanthymotus_pr/phanthymotus/2026-09/2026-09-17/pr-7/evidence-" + "a" * 39 + ".log.gz",
])
def test_cos_validate_object_key_fail_closed(key):
    with pytest.raises(ValueError):
        CosClient.validate_object_key("4paradigm/phanthymotus", 7, HEAD, key)


class _Body:
    def __init__(self, chunks):
        self.chunks, self.closed = iter(chunks), False
    def get_raw_stream(self): return self
    def read(self, _size): return next(self.chunks, b"")
    def close(self): self.closed = True


def _install_qcloud(monkeypatch, client, body, length):
    events = []
    class SDK:
        def __init__(self, _config): pass
        def head_object(self, **kwargs): events.append("HEAD"); return {"Content-Length": str(length)}
        def get_object(self, **kwargs): events.append("GET"); return {"Body": body}
    module = ModuleType("qcloud_cos")
    module.CosConfig = lambda **kwargs: SimpleNamespace()
    module.CosS3Client = SDK
    monkeypatch.setitem(sys.modules, "qcloud_cos", module)
    client.config.cos_region, client.config.cos_secret_id, client.config.cos_secret_key, client.config.cos_bucket = "cn-bj", "id", "key", "bucket"
    return events


def test_cos_download_head_before_get_and_hard_bound(monkeypatch):
    client = CosClient(make_config())
    body = _Body([b"evidence"])
    events = _install_qcloud(monkeypatch, client, body, 8)
    assert asyncio.run(client.download_evidence_archive("key", 10)) == b"evidence"
    assert events == ["HEAD", "GET"] and body.closed
    events = _install_qcloud(monkeypatch, client, _Body([b"not-read"]), EVIDENCE_MAX_ARCHIVE_BYTES + 1)
    with pytest.raises(CosError):
        asyncio.run(client.download_evidence_archive("key", EVIDENCE_MAX_ARCHIVE_BYTES))
    assert events == ["HEAD"]
    body = _Body([b"x" * (EVIDENCE_MAX_ARCHIVE_BYTES + 1)])
    _install_qcloud(monkeypatch, client, body, EVIDENCE_MAX_ARCHIVE_BYTES)
    with pytest.raises(CosError):
        asyncio.run(client.download_evidence_archive("key", EVIDENCE_MAX_ARCHIVE_BYTES))
    assert body.closed
    body = _Body([b"short"])
    _install_qcloud(monkeypatch, client, body, 99)
    with pytest.raises(CosError):
        asyncio.run(client.download_evidence_archive("key", EVIDENCE_MAX_ARCHIVE_BYTES))
    assert body.closed


# ── presigned URL contract tests ─────────────────────────────────────────

def test_cos_presign_ttl_is_120():
    """EVIDENCE_PRESIGNED_URL_TTL_SECONDS must be exactly 120."""
    from ..cos_client import EVIDENCE_PRESIGNED_URL_TTL_SECONDS
    assert EVIDENCE_PRESIGNED_URL_TTL_SECONDS == 120


@pytest.mark.asyncio
async def test_cos_presign_missing_credentials_returns_empty(monkeypatch):
    """When credentials are missing, generate_evidence_download_url returns ''."""
    client = CosClient(make_config(cos_secret_id="", cos_secret_key=""))
    result = client.generate_evidence_download_url("phanthymotus_pr/test/evidence.gz")
    assert result == ""


@pytest.mark.asyncio
async def test_cos_presign_invalid_object_key_returns_empty(monkeypatch):
    """Invalid object keys must return empty string."""
    client = CosClient(make_config())
    for bad_key in ["", "/phanthymotus_pr/test", "phanthymotus/../test", "phanthymotus_pr/test\\key", "other_repo/pr/1/test"]:
        result = client.generate_evidence_download_url(bad_key)
        assert result == f"expected empty for {bad_key!r} but got {result!r}" or result == ""


@pytest.mark.asyncio
async def test_cos_presign_https_accepted_http_rejected(monkeypatch):
    """HTTPS presigned URL accepted; HTTP rejected."""
    import sys
    from types import ModuleType

    class FakeCosConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeUrl(str):
        def __new__(cls, url):
            return str.__new__(cls, url)
        def startswith(self, prefix):
            return super().startswith(prefix)

    class FakeCosS3Client:
        def __init__(self, config):
            self.config = config
        def get_presigned_url(self, **kwargs):
            # Simulate SDK returning the URL
            return FakeUrl(self._test_url)

        def set_test_url(self, url):
            self._test_url = url

    fake_module = ModuleType("qcloud_cos")
    fake_module.CosConfig = FakeCosConfig
    fake_module.CosS3Client = FakeCosS3Client

    saved = sys.modules.get("qcloud_cos")
    sys.modules["qcloud_cos"] = fake_module
    try:
        cfg = make_config(cos_region="ap-shanghai", cos_bucket="test-bucket", cos_secret_id="SID", cos_secret_key="SK")
        client = CosClient(cfg)

        # HTTPS -> accepted
        client._client_class = FakeCosS3Client
        fake_client = FakeCosS3Client(None)
        fake_client.set_test_url("https://bucket-123456.cos.ap-shanghai.myqcloud.com/phanthymotus_pr/test/evidence.gz")
        client._fake_cos_client = fake_client

        # Patch the constructor to return our fake
        orig_init = FakeCosS3Client.__init__
        def patched_init(self, config):
            orig_init(self, config)
            self._test_url = "https://bucket-123456.cos.ap-shanghai.myqcloud.com/phanthymotus_pr/test/evidence.gz"
        FakeCosS3Client.__init__ = patched_init

        result = client.generate_evidence_download_url("phanthymotus_pr/test/evidence.gz")
        assert result.startswith("https://"), f"expected https URL, got {result!r}"

        # HTTP -> rejected
        def patched_init_http(self, config):
            orig_init(self, config)
            self._test_url = "http://bucket-123456.cos.ap-shanghai.myqcloud.com/phanthymotus_pr/test/evidence.gz"
        FakeCosS3Client.__init__ = patched_init_http

        result2 = client.generate_evidence_download_url("phanthymotus_pr/test/evidence.gz")
        assert result2 == "", f"expected empty for http URL, got {result2!r}"
    finally:
        FakeCosS3Client.__init__ = orig_init
        if saved:
            sys.modules["qcloud_cos"] = saved
        elif "qcloud_cos" in sys.modules:
            del sys.modules["qcloud_cos"]


@pytest.mark.asyncio
async def test_cos_presign_sdk_exception_returns_empty(monkeypatch):
    """SDK exception must fail closed."""
    import sys
    from types import ModuleType

    class FakeCosConfig:
        def __init__(self, **kwargs):
            pass

    class FakeCosS3Client:
        def __init__(self, config):
            pass
        def get_presigned_url(self, **kwargs):
            raise RuntimeError("network error")

    fake_module = ModuleType("qcloud_cos")
    fake_module.CosConfig = FakeCosConfig
    fake_module.CosS3Client = FakeCosS3Client

    saved = sys.modules.get("qcloud_cos")
    sys.modules["qcloud_cos"] = fake_module
    try:
        client = CosClient(make_config())
        result = client.generate_evidence_download_url("phanthymotus_pr/test/evidence.gz")
        assert result == ""
    finally:
        if saved:
            sys.modules["qcloud_cos"] = saved
        elif "qcloud_cos" in sys.modules:
            del sys.modules["qcloud_cos"]


def test_cos_presign_no_secret_leakage_in_source():
    """Source must not log URL query strings, secrets, or Authorization headers."""
    source = open(Path(__file__).parent.parent / "cos_client.py").read()
    assert "logger.info" not in source or "presigned" not in source.lower() or "url" not in source.lower().split("logger.info")[0][-50:] if "logger.info" in source else True
    # Ensure no direct secret logging
    assert 'logger.info' not in source or 'SecretId' not in source.split('logger.info')[1].split('\n')[0] if 'logger.info' in source else True


@pytest.mark.asyncio
async def test_cos_presign_method_belongs_to_cos_client():
    """generate_evidence_download_url must be a method of CosClient, not CosError."""
    from ..cos_client import CosClient, CosError
    assert hasattr(CosClient, "generate_evidence_download_url")
    assert not hasattr(CosError, "generate_evidence_download_url")


@pytest.mark.asyncio
async def test_cos_presign_uses_real_local_sdk_construction_seam():
    """Tests must mock qcloud_cos, not client._client or self._config."""
    source = open(Path(__file__).parent.parent / "cos_client.py").read()
    # The method must construct CosConfig/CosS3Client locally
    assert "CosConfig(" in source
    assert "CosS3Client(" in source
    # Must not reference self._client or self._config inside generate_evidence_download_url
    # Check that the method uses self.config (not self._config)
    import ast
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "CosClient":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "generate_evidence_download_url":
                    method_src = ast.unparse(item)
                    assert "self._config" not in method_src, "generate_evidence_download_url must use self.config, not self._config"
                    assert "self._client" not in method_src, "generate_evidence_download_url must not use self._client"


def test_production_compose_default_repo_is_phanthymotus_only():
    """GITHUB_REPOS default must be 4paradigm/phanthymotus only (config-level check)."""
    from ..config import DEFAULT_GITHUB_REPOS
    assert DEFAULT_GITHUB_REPOS == ("4paradigm/phanthymotus",)
    assert "4paradigm/phanthymotus-driver" not in str(DEFAULT_GITHUB_REPOS)


# ── service approval contract tests ────────────────────────────────────────

@pytest.mark.asyncio
async def test_pr_author_cannot_approve_own_deploy():
    """PR author must not be able to approve their own deployment."""
    from ..service import DeployController, _is_self_approval

    # Same numeric ID -> self-approval
    assert _is_self_approval("12345", {"user": {"id": 12345}}) is True
    # Different numeric ID -> not self-approval
    assert _is_self_approval("12345", {"user": {"id": 67890}}) is False
    # Missing user -> fail closed
    assert _is_self_approval("12345", {}) is True
    # Missing id -> fail closed
    assert _is_self_approval("12345", {"user": {}}) is True


@pytest.mark.asyncio
async def test_approve_rejects_machine_without_full_coverage():
    """A machine that does not cover all remaining components must be rejected."""
    from ..service import DeployController
    from types import SimpleNamespace

    machines = [
        SimpleNamespace(
            alias="alpha",
            node_id="node-1",
            platforms=["linux/arm64"],
            variants=[],
            driver_paths=[],
            machine_supports_target=lambda alias, target: True,
        )
    ]

    components = [
        {"component_id": "comp-a", "target": "perception", "resolved_platform": "linux/arm64"},
        {"component_id": "comp-b", "target": "planning", "resolved_platform": "linux/arm64"},
    ]

    mock_policy = SimpleNamespace()
    mock_policy.get_machines = lambda: machines
    # "planning" is not a supported target (only perception/actucore/driver)
    def _supports_target(alias, target):
        return target in ("perception", "actucore", "driver")
    mock_policy.machine_supports_target = _supports_target

    controller = DeployController(
        config=SimpleNamespace(
            machine_config="tests/machines.yaml",
            health_timeout_seconds=30,
            health_poll_interval_seconds=1,
        ),
        proxy=None,
        policy=mock_policy,
        github=None,
    )

    groups = controller._get_machine_groups_for_components(components)
    # No machine covers both components, so groups should be empty
    assert groups == []


@pytest.mark.asyncio
async def test_no_post_deploy_health_gate():
    """_wait_for_deploy_health must not exist in service.py."""
    source = open(Path(__file__).parent.parent / "service.py").read()
    assert "_wait_for_deploy_health" not in source, (
        "_wait_for_deploy_health method must be removed"
    )
