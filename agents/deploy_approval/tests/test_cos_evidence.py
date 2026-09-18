"""Focused tests for the single gzip evidence and private COS contracts."""
from __future__ import annotations

import asyncio
import gzip
import sys
from datetime import datetime
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
    expected = "phanthymotus_pr/phanthymotus/2026-09/2026-09-17/pr-7/evidence-" + HEAD + ".log.gz"
    for repo in ("4paradigm/phanthymotus", "Haohao-end/phanthymotus"):
        assert client.build_object_key(repo, 7, HEAD, now=now) == expected
    assert client.build_object_key("4paradigm/phanthymotus-driver", 7, HEAD, now=now) == expected.replace("phanthymotus/", "phanthymotus-driver/")


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
    client.config.cos_secret_id, client.config.cos_secret_key, client.config.cos_bucket = "id", "key", "bucket"
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
