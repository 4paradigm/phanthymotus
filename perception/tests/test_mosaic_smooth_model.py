"""YuNet provisioning through the demo Card's start lifecycle, with no network."""

import hashlib
import json
import threading
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import URLError

import pytest

from plugins import mosaic_smooth as mosaic
from utils import model_downloader as models
from vision_stubs import _FakeExecutor, _FakeNode, _wait_until


@pytest.fixture
def demo(tmp_path, monkeypatch):
    payload = b"pinned YuNet test fixture"
    model_dir = tmp_path / "mosaic_smooth"
    filename = models.MOSAIC_SMOOTH_MODEL_FILENAME
    monkeypatch.delenv("MOSAIC_SMOOTH_YUNET_MODEL", raising=False)
    monkeypatch.delenv("MOSAIC_SMOOTH_MODEL_BASE_URL", raising=False)
    monkeypatch.setattr(models, "MOSAIC_SMOOTH_MODEL_DIR", str(model_dir))
    require_subpath = models.require_models_subpath
    monkeypatch.setattr(models, "require_models_subpath",
                        lambda path: require_subpath(path, root=str(tmp_path)))
    monkeypatch.setitem(models.MOSAIC_SMOOTH_MODEL_FILES, filename, {
        "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
    })
    monkeypatch.setattr(models, "time", SimpleNamespace(sleep=lambda _: None))
    download = Mock(side_effect=lambda url, timeout: BytesIO(payload))
    def create_engine(path, cfg):
        mosaic.validate_model(path)
        return Mock()

    engine = Mock(side_effect=create_engine)
    monkeypatch.setattr(models, "urlopen", download)
    monkeypatch.setattr(mosaic, "SmoothMosaic", engine)
    executor = _FakeExecutor()
    plugin = mosaic.MosaicSmoothDemoPlugin({}, executor)
    try:
        yield plugin, executor, model_dir / filename, payload, download, engine
    finally:
        plugin.shutdown()


def dispatch(plugin, action, instance="demo", **kwargs):
    return plugin.dispatch(mosaic.PREFIX, {
        "action": action, "instance_id": instance, **kwargs,
    })


def start(plugin, instance="demo"):
    return dispatch(plugin, "start", instance, input_topic="/test/camera")


def consent(plugin, instance="demo"):
    assert dispatch(plugin, "config", instance, config={
        "consent_acknowledged": True,
    })["state"] == "configured"


def test_missing_model_downloads_only_after_consent_then_reuses_cache(demo):
    plugin, executor, path, payload, download, engine = demo
    assert dispatch(plugin, "info")["state"] == "idle"
    assert start(plugin)["error"] == "consent_acknowledgment_required"
    assert not path.parent.exists()
    assert not executor.nodes
    download.assert_not_called()

    consent(plugin)
    assert start(plugin)["state"] in ("loading", "running")
    assert _wait_until(lambda: dispatch(plugin, "info")["state"] == "running")
    assert dispatch(plugin, "info")["model_fetch"]["stage"] == "download_complete"
    assert path.read_bytes() == payload
    download.assert_called_once_with(
        models.MOSAIC_SMOOTH_MODEL_SOURCES[0] + "/" + path.name, timeout=120,
    )
    assert dispatch(plugin, "stop")["state"] == "idle"

    consent(plugin)
    assert start(plugin)["state"] in ("loading", "running")
    assert _wait_until(lambda: dispatch(plugin, "info")["state"] == "running")
    assert dispatch(plugin, "info")["model_fetch"]["stage"] == "cache_hit"
    assert download.call_count == 1
    assert engine.call_count == 2


def test_download_failure_publishes_no_image_and_a_new_start_retries(demo, monkeypatch):
    plugin, _, path, payload, download, engine = demo
    mirror = "https://example.invalid/approved-models"
    monkeypatch.setenv("MOSAIC_SMOOTH_MODEL_BASE_URL", mirror)
    download.side_effect = URLError("offline")
    consent(plugin)
    start(plugin)
    assert _wait_until(lambda: not plugin._nodes)
    result = dispatch(plugin, "info")
    assert result["state"] == "error"
    assert result["reason"] == "model_download_failed:RuntimeError"
    assert result["model_fetch"]["stage"] == "download_failed"
    assert result["model_fetch"]["retries"] == 2
    assert result["privacy_guaranteed"] is False
    assert result["metrics"]["published"] == 0
    assert not _FakeNode.instances[0].publishers[0].messages
    assert not path.exists()
    assert not list(path.parent.glob(".mosaic_smooth_yunet-*"))
    engine.assert_not_called()
    assert download.call_count == 3
    assert {call.args[0] for call in download.call_args_list} == {f"{mirror}/{path.name}"}

    download.side_effect = lambda url, timeout: BytesIO(payload)
    assert start(plugin)["state"] in ("loading", "running")
    assert _wait_until(lambda: dispatch(plugin, "info")["state"] == "running")
    assert dispatch(plugin, "info")["model_fetch"]["retries"] == 1
    assert path.read_bytes() == payload
    engine.assert_called_once()


def test_hash_mismatch_does_not_promote_download_and_cache_can_be_repaired(demo):
    plugin, _, path, payload, download, engine = demo
    path.parent.mkdir()
    corrupt_cache = b"x" * len(payload)
    path.write_bytes(corrupt_cache)
    download.side_effect = lambda url, timeout: BytesIO(b"y" * len(payload))
    consent(plugin)
    start(plugin)
    assert _wait_until(lambda: not plugin._nodes)
    result = dispatch(plugin, "info")
    assert result["reason"] == "model_verification_failed:RuntimeError"
    assert result["model_fetch"]["stage"] == "verification_failed"
    assert path.read_bytes() == corrupt_cache
    engine.assert_not_called()
    assert download.call_count == 3 * len(models.MOSAIC_SMOOTH_MODEL_SOURCES)

    def fallback(url, timeout):
        content = payload if url.startswith(models.MOSAIC_SMOOTH_MODEL_SOURCES[1]) else corrupt_cache
        return BytesIO(content)

    download.side_effect = fallback
    assert start(plugin)["state"] in ("loading", "running")
    assert _wait_until(lambda: dispatch(plugin, "info")["state"] == "running")
    assert path.read_bytes() == payload
    engine.assert_called_once()


def test_verified_legacy_model_is_reused(demo):
    plugin, _, path, payload, download, engine = demo
    path.parent.mkdir()
    legacy = path.with_name("yunet.onnx")
    legacy.write_bytes(payload)
    consent(plugin)
    start(plugin)
    assert _wait_until(lambda: dispatch(plugin, "info")["state"] == "running")
    assert not path.exists()
    assert engine.call_args.args[0] == str(legacy)
    assert dispatch(plugin, "info")["model_fetch"]["stage"] == "cache_hit"
    download.assert_not_called()


@pytest.mark.skipif(models.fcntl is None, reason="model volume locking requires flock")
def test_concurrent_card_starts_share_one_verified_download(demo, monkeypatch):
    plugin, _, path, payload, download, engine = demo
    fetching = threading.Event()
    release = threading.Event()
    waiting_on_lock = threading.Event()
    flock = models.fcntl.flock

    def track_flock(fd, operation):
        if operation == models.fcntl.LOCK_EX and fetching.is_set():
            waiting_on_lock.set()
        return flock(fd, operation)

    def slow_download(url, timeout):
        fetching.set()
        assert release.wait(5)
        return BytesIO(payload)

    monkeypatch.setattr(models.fcntl, "flock", track_flock)
    download.side_effect = slow_download
    consent(plugin, "first")
    consent(plugin, "second")
    try:
        assert start(plugin, "first")["state"] == "loading"
        assert fetching.wait(5)
        assert start(plugin, "second")["state"] == "loading"
        assert waiting_on_lock.wait(5)
        assert dispatch(plugin, "info", "second")["model_fetch"]["stage"] == "waiting_for_lock"
        assert not path.exists()
        engine.assert_not_called()
        assert download.call_count == 1
    finally:
        release.set()

    assert _wait_until(lambda: dispatch(plugin, "info", "first")["state"] == "running")
    assert _wait_until(lambda: dispatch(plugin, "info", "second")["state"] == "running")
    assert path.read_bytes() == payload
    assert engine.call_count == 2
    assert download.call_count == 1
    assert dispatch(plugin, "info", "first")["model_fetch"]["stage"] == "download_complete"
    assert dispatch(plugin, "info", "second")["model_fetch"]["stage"] == "cache_hit"


def test_card_exposes_live_bytes_and_verification_stage(demo, monkeypatch):
    plugin, executor, path, payload, download, engine = demo
    reading = threading.Event()
    release_read = threading.Event()
    verifying = threading.Event()
    release_verify = threading.Event()
    first_size = len(payload) // 2
    verify = models._verify_pinned_file

    class StreamingResponse(BytesIO):
        def read1(self, count):
            if self.tell() == 0:
                return super().read(first_size)
            reading.set()
            assert release_read.wait(5), "download test did not release the remaining bytes"
            return super().read(count)

    def gated_verify(file, metadata):
        verifying.set()
        assert release_verify.wait(5), "download test did not release verification"
        return verify(file, metadata)

    download.side_effect = lambda *args, **kwargs: StreamingResponse(payload)
    monkeypatch.setattr(models, "_verify_pinned_file", gated_verify)
    consent(plugin)
    try:
        start(plugin)
        assert reading.wait(5), "Card did not enter the second read"
        status = dispatch(plugin, "info")
        assert status["state"] == "loading"
        percent = first_size * 100 // len(payload)
        assert f"{percent}%" in status["reason"]
        assert status["model_fetch"] == {
            "stage": "downloading", "retries": 0, "percent": percent,
            "bytes_done": first_size, "bytes_total": len(payload),
        }
        # The ROS timer publishes the same observable state as MCP info.
        node = executor.nodes[0]
        node._publish_status()
        assert json.loads(node.publishers[1].messages[-1])["model_fetch"] == status["model_fetch"]
        engine.assert_not_called()
        assert not path.exists()
        release_read.set()
        assert verifying.wait(5), "Card did not reach model verification"
        status = dispatch(plugin, "info")
        assert status["model_fetch"]["stage"] == "verifying_download"
        assert status["model_fetch"]["percent"] == 100
        assert status["state"] == "loading" and not path.exists()
        engine.assert_not_called()
    finally:
        release_read.set()
        release_verify.set()
    assert _wait_until(lambda: dispatch(plugin, "info")["state"] == "running")
    assert dispatch(plugin, "info")["model_fetch"]["stage"] == "download_complete"


def test_stop_during_download_prevents_late_readiness_and_output(demo):
    plugin, executor, path, payload, download, engine = demo
    downloading = threading.Event()
    release = threading.Event()

    def blocked_response(*args, **kwargs):
        downloading.set()
        assert release.wait(5), "stop test did not release transfer"
        return BytesIO(payload)

    download.side_effect = blocked_response
    consent(plugin)
    start(plugin)
    assert downloading.wait(5), "Card did not begin downloading"
    node = executor.nodes[0]
    stopped = []
    stopper = threading.Thread(target=lambda: stopped.append(dispatch(plugin, "stop")))
    stopper.start()
    try:
        assert _wait_until(lambda: dispatch(plugin, "info")["state"] == "stopping")
        assert dispatch(plugin, "info")["model_fetch"]["stage"] == "cancelled"
    finally:
        release.set()
        stopper.join(5)
    assert not stopper.is_alive()
    assert stopped == [{"state": "idle", "image_publication_disabled": True}]
    assert dispatch(plugin, "info")["state"] == "idle"
    assert node.snapshot()["model_fetch"]["stage"] == "cancelled"
    assert not node.publishers[0].messages and not executor.nodes
    assert path.read_bytes() == payload
    engine.assert_not_called()
