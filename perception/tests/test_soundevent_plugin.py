"""Host-side tests for the SoundEvent model and PCM streaming contract."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import threading
import time
import types
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np
import pytest

from vision_stubs import _FakeAudioChunk, _FakeExecutor, _FakeNode, _wait_until

import plugins.soundevent as soundevent
from utils import model_downloader


def _write_metadata_model(path: Path, labels: list[str]) -> None:
    # Metadata-bearing TFLite files keep the FlatBuffer at the front and append
    # associated files as a ZIP archive.  ``zipfile`` deliberately supports
    # this self-extracting-archive layout.
    path.write_bytes(b"\x1c\x00\x00\x00TFL3" + b"\x00" * 24)
    with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("yamnet_label_list.txt", "\n".join(labels) + "\n")


def _load_model_downloader_copy(name: str):
    spec = importlib.util.spec_from_file_location(name, model_downloader.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _audio_chunk(data: bytes, timestamp: float) -> _FakeAudioChunk:
    message = _FakeAudioChunk()
    seconds = int(timestamp)
    message.header.stamp = types.SimpleNamespace(
        sec=seconds,
        nanosec=round((timestamp - seconds) * 1_000_000_000),
    )
    message.format = soundevent.AUDIO_FORMAT
    message.data = data
    return message


@pytest.fixture(autouse=True)
def _stub_destroy_subscription(monkeypatch):
    monkeypatch.setattr(
        _FakeNode,
        "destroy_subscription",
        lambda self, subscription: self.subscriptions.remove(subscription),
        raising=False,
    )


class _LifecycleModel:
    labels = ["Bark"]

    def predict(self, waveform):
        return np.asarray([0.9], dtype=np.float32)


class _ModelBuilderProbe:
    def __init__(self, release=None, failures=0):
        self.release = release
        self.failures = failures
        self.started = threading.Event()
        self.calls = 0
        self.models = []
        self.on_status = None
        self._lock = threading.Lock()

    def __call__(self, on_status=None):
        # Mirrors _build_model's signature: the plugin hands it a status sink so
        # the download's progress reaches the card. Recorded so a test can drive
        # it and assert what the card then shows.
        self.on_status = on_status
        with self._lock:
            self.calls += 1
            call = self.calls
        self.started.set()
        if self.release is not None and not self.release.wait(timeout=3.0):
            raise RuntimeError("test model loader was not released")
        if call <= self.failures:
            raise RuntimeError("download failed")
        model = _LifecycleModel()
        self.models.append(model)
        return model


def test_labels_from_metadata_bearing_tflite(tmp_path):
    labels = [f"label-{index}" for index in range(521)]
    model_path = tmp_path / "yamnet.tflite"
    _write_metadata_model(model_path, labels)

    assert model_path.read_bytes()[4:8] == b"TFL3"
    assert soundevent.labels_from_model(model_path) == labels


def test_labels_from_model_rejects_missing_associated_file(tmp_path):
    model_path = tmp_path / "yamnet.tflite"
    model_path.write_bytes(b"\x1c\x00\x00\x00TFL3" + b"\x00" * 24)
    with zipfile.ZipFile(model_path, "a") as archive:
        archive.writestr("other.txt", "not labels")

    with pytest.raises(RuntimeError, match="no embedded label list"):
        soundevent.labels_from_model(model_path)


_COS_SOURCE = f"{model_downloader.COS_BASE}/soundevent"
_MODELSCOPE_SOURCE = model_downloader.SOUNDEVENT_MODELSCOPE_BASE
_ENV_SOURCE = "https://models.example/soundevent"
_ALL_SOURCES = [_COS_SOURCE, _MODELSCOPE_SOURCE]


# The sources are assembled at import, like every other bundle's base_url in
# model_downloader, and the environment variable *replaces* the first one the
# way OCR_MODEL_BASE_URL and FACE_MODEL_BASE_URL do. So this is where the
# override is observable; patching the module attribute afterwards would only
# prove the patch took.
def test_soundevent_model_base_honors_environment(monkeypatch):
    monkeypatch.delenv("SOUNDEVENT_MODEL_BASE_URL", raising=False)
    default_module = _load_model_downloader_copy("model_downloader_default_test")
    assert default_module.SOUNDEVENT_MODEL_BASE == _COS_SOURCE
    assert default_module.SOUNDEVENT_MODEL_SOURCES == [_COS_SOURCE, _MODELSCOPE_SOURCE]

    monkeypatch.setenv("SOUNDEVENT_MODEL_BASE_URL", _ENV_SOURCE)
    configured_module = _load_model_downloader_copy("model_downloader_env_test")
    assert configured_module.SOUNDEVENT_MODEL_BASE == _ENV_SOURCE
    # The override takes COS's place rather than being added in front of it:
    # a site that redirects this variable at its own mirror is not asking for
    # the public bucket to stay in the list.
    assert configured_module.SOUNDEVENT_MODEL_SOURCES == [
        _ENV_SOURCE, _MODELSCOPE_SOURCE
    ]


def test_soundevent_model_download_uses_pinned_manifest(tmp_path, monkeypatch):
    captured = {}

    monkeypatch.setattr(model_downloader, "SOUNDEVENT_MODEL_SOURCES", _ALL_SOURCES)
    monkeypatch.setattr(
        model_downloader,
        "require_models_subpath",
        lambda model_dir: str(tmp_path),
    )

    def fake_ensure(name, model_dir, base_url, files, progress_cb=None):
        captured.update(
            name=name, model_dir=model_dir, base_url=base_url, files=files
        )
        return {
            model_downloader.SOUNDEVENT_MODEL_FILENAME: str(
                tmp_path / model_downloader.SOUNDEVENT_MODEL_FILENAME
            )
        }

    monkeypatch.setattr(model_downloader, "ensure_verified_bundle", fake_ensure)

    result = model_downloader.ensure_soundevent_model()

    assert result == str(tmp_path / "yamnet_classification.tflite")
    assert captured == {
        "name": "soundevent",
        # One bundle carrying every source, not one call per source: that is
        # what lets the downloader probe them and pick, instead of always
        # paying for the first one in the list.
        "model_dir": str(tmp_path),
        "base_url": _ALL_SOURCES,
        "files": {
            "yamnet_classification.tflite": {
                "size": 4_126_810,
                "sha256": (
                    "10c95ea3eb9a7bb4cb8bddf6feb023250381008177ac162ce169694d05c317de"
                ),
            }
        },
    }


def test_soundevent_sources_are_ordered_by_measured_speed(tmp_path, monkeypatch):
    """COS and ModelScope are both registered; the machine picks.

    The pinned size and SHA256 are what make choosing by speed a performance
    decision rather than a trust one — both hosts must serve the same bytes.
    """
    monkeypatch.setattr(
        model_downloader, "require_models_subpath", lambda path: str(tmp_path)
    )
    cos, modelscope = _COS_SOURCE, _MODELSCOPE_SOURCE
    # ModelScope measured faster here; COS is declared first and must still lose.
    rates = {cos: 500_000.0, modelscope: 5_000_000.0}
    monkeypatch.setattr(model_downloader, "_probe_source",
                        lambda source, filename: rates[source])
    fetched = []

    def fake_fetch(name, url, destination, metadata, **kwargs):
        fetched.append(url)
        Path(destination).write_bytes(b"")

    monkeypatch.setattr(model_downloader, "_fetch_pinned_file", fake_fetch)
    monkeypatch.setattr(model_downloader, "_bundle_matches",
                        lambda model_dir, files: False)

    model_downloader.ensure_soundevent_model()

    assert fetched == [f"{modelscope}/yamnet_classification.tflite"]


@pytest.fixture
def soundevent_download(tmp_path, monkeypatch):
    payload = b"verified soundevent test model"
    monkeypatch.setattr(
        model_downloader, "require_models_subpath", lambda path: str(tmp_path)
    )
    monkeypatch.setattr(model_downloader, "SOUNDEVENT_MODEL_SOURCES", _ALL_SOURCES)
    monkeypatch.setattr(model_downloader, "SOUNDEVENT_MODEL_FILES", {
        model_downloader.SOUNDEVENT_MODEL_FILENAME: {
            "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
        },
    })
    monkeypatch.setattr(model_downloader.time, "sleep", lambda seconds: None)
    return payload


@pytest.mark.parametrize("successful_source", [0, 1])
def test_soundevent_download_tries_sources_in_order(
    successful_source, soundevent_download, monkeypatch
):
    urls = [f"{base}/yamnet_classification.tflite" for base in _ALL_SOURCES]
    requests = []

    def urlopen(url, timeout):
        requests.append(url)
        if url != urls[successful_source]:
            raise HTTPError(url, 404, "Not Found", None, None)
        return io.BytesIO(soundevent_download)

    monkeypatch.setattr(model_downloader, "urlopen", urlopen)
    path = model_downloader.ensure_soundevent_model()

    assert Path(path).read_bytes() == soundevent_download
    expected = [url for url in urls[:successful_source] for _ in range(3)]
    expected.append(urls[successful_source])
    assert requests == expected
    # A validated cache is reused without contacting any source again.
    assert model_downloader.ensure_soundevent_model() == path
    assert requests == expected


@pytest.mark.parametrize("failure", ["http", "connection", "timeout", "size", "sha256"])
def test_soundevent_falls_back_when_cos_download_or_validation_fails(
    failure, soundevent_download, monkeypatch, caplog
):
    cos_url = f"{_COS_SOURCE}/yamnet_classification.tflite"
    fallback_url = f"{_MODELSCOPE_SOURCE}/yamnet_classification.tflite"
    requests = []

    def urlopen(url, timeout):
        requests.append(url)
        if url == fallback_url:
            return io.BytesIO(soundevent_download)
        assert url == cos_url
        if failure == "http":
            raise HTTPError(url, 404, "Not Found", None, None)
        if failure == "connection":
            raise URLError("connection refused")
        if failure == "timeout":
            raise TimeoutError("read timed out")
        if failure == "size":
            return io.BytesIO(b"short")
        return io.BytesIO(b"x" * len(soundevent_download))

    monkeypatch.setattr(model_downloader, "urlopen", urlopen)
    path = model_downloader.ensure_soundevent_model()

    assert requests == [cos_url] * 3 + [fallback_url]
    assert Path(path).read_bytes() == soundevent_download
    assert f"failed from {model_downloader.COS_BASE}/soundevent" in caplog.text
    assert "1 source(s) left" in caplog.text
    reason = {
        "http": "HTTPError", "connection": "URLError", "timeout": "TimeoutError",
        "size": "ValueError", "sha256": "ValueError",
    }[failure]
    assert f"({reason})" in caplog.text
    assert not list(Path(path).parent.glob(".soundevent-*/"))


def test_soundevent_contacts_each_source_once_per_retry_set(
    soundevent_download, monkeypatch
):
    """Every source is tried, each exactly three times, then the attempt fails.

    The bound is what a robot feels: an unreachable mirror costs a fixed slice
    of a cold start rather than an open-ended one, and the next `start` retries.
    """
    requests = []

    def urlopen(url, timeout):
        requests.append(url)
        raise HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr(model_downloader, "urlopen", urlopen)
    with pytest.raises(RuntimeError, match="failed to download"):
        model_downloader.ensure_soundevent_model()

    assert requests == [
        f"{base}/yamnet_classification.tflite"
        for base in _ALL_SOURCES for _ in range(3)
    ]


def test_soundevent_reports_all_sources_failed_without_caching_invalid_model(
    soundevent_download, tmp_path, monkeypatch
):
    requests = []

    def urlopen(url, timeout):
        requests.append(url)
        return io.BytesIO(b"x" * len(soundevent_download))

    monkeypatch.setattr(model_downloader, "urlopen", urlopen)
    with pytest.raises(RuntimeError, match="failed to download") as error:
        model_downloader.ensure_soundevent_model()

    # Three attempts at each of the two sources, and the corrupt bytes none of
    # them delivered correctly are not left behind as a cache.
    assert len(requests) == 3 * len(_ALL_SOURCES)
    assert "SHA256 mismatch" in str(error.value.__cause__)
    assert not (tmp_path / model_downloader.SOUNDEVENT_MODEL_FILENAME).exists()
    assert not list(tmp_path.glob(".soundevent-*/"))


def test_plugin_constructor_does_not_load_model(monkeypatch):
    def unexpected_build():
        raise AssertionError("model loading must not run during construction")

    monkeypatch.setattr(soundevent, "_build_model", unexpected_build)
    executor = _FakeExecutor()
    plugin = soundevent.SoundEventPlugin({}, executor)

    assert plugin.dispatch("soundevent", {"action": "info"})["state"] == "idle"
    assert executor.nodes == []


def test_start_loads_model_in_background_and_replays_pending(monkeypatch):
    release = threading.Event()
    builder = _ModelBuilderProbe(release=release)
    monkeypatch.setattr(soundevent, "_build_model", builder)
    executor = _FakeExecutor()
    plugin = soundevent.SoundEventPlugin({}, executor)

    try:
        started = time.monotonic()
        result = plugin.dispatch(
            "soundevent",
            {"action": "start", "input_topic": "/mic/a", "instance_id": "a"},
        )
        elapsed = time.monotonic() - started

        assert elapsed < 0.1, "start must not wait for model download or initialization"
        assert result["state"] == "loading"
        assert result["input"] == "/mic/a"
        assert result["output"] == "/mic/a/soundevent"
        assert builder.started.wait(timeout=1.0)

        info_started = time.monotonic()
        info = plugin.dispatch(
            "soundevent", {"action": "info", "instance_id": "a"}
        )
        assert time.monotonic() - info_started < 0.1
        assert info["state"] == "loading"
        assert info["topic_out"][0]["topic"] == "/mic/a/soundevent"

        release.set()
        assert _wait_until(
            lambda: plugin.dispatch(
                "soundevent", {"action": "info", "instance_id": "a"}
            )["state"]
            == "running"
        )
        assert len(executor.nodes) == 1
        assert executor.nodes[0]._model is builder.models[0]
    finally:
        release.set()
        plugin.dispatch("soundevent", {"action": "stop"})


def test_concurrent_starts_share_one_model_load(monkeypatch):
    release = threading.Event()
    builder = _ModelBuilderProbe(release=release)
    monkeypatch.setattr(soundevent, "_build_model", builder)
    executor = _FakeExecutor()
    plugin = soundevent.SoundEventPlugin({}, executor)
    results = []

    def start(index):
        results.append(
            plugin.dispatch(
                "soundevent",
                {
                    "action": "start",
                    "input_topic": "/mic/%d" % index,
                    "instance_id": "instance-%d" % index,
                },
            )
        )

    threads = [threading.Thread(target=start, args=(index,)) for index in range(8)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=1.0)

        assert len(results) == 8
        assert all(result["state"] == "loading" for result in results)
        assert builder.calls == 1

        release.set()
        assert _wait_until(
            lambda: len(executor.nodes) == 8
            and all(node.state == "running" for node in executor.nodes)
        )
        assert builder.calls == 1
        assert len({id(node._model) for node in executor.nodes}) == 1
    finally:
        release.set()
        plugin.dispatch("soundevent", {"action": "stop"})


def test_stop_during_loading_cancels_pending_instance(monkeypatch):
    release = threading.Event()
    builder = _ModelBuilderProbe(release=release)
    monkeypatch.setattr(soundevent, "_build_model", builder)
    executor = _FakeExecutor()
    plugin = soundevent.SoundEventPlugin({}, executor)

    try:
        plugin.dispatch(
            "soundevent",
            {"action": "start", "input_topic": "/mic/a", "instance_id": "a"},
        )
        plugin.dispatch(
            "soundevent",
            {"action": "start", "input_topic": "/mic/b", "instance_id": "b"},
        )
        assert builder.started.wait(timeout=1.0)
        assert plugin.dispatch(
            "soundevent", {"action": "stop", "instance_id": "a"}
        ) == {"state": "idle"}

        release.set()
        assert _wait_until(
            lambda: plugin.dispatch(
                "soundevent", {"action": "info", "instance_id": "b"}
            )["state"]
            == "running"
        )
        assert plugin.dispatch(
            "soundevent", {"action": "info", "instance_id": "a"}
        )["state"] == "idle"
        assert [node.input_topic for node in executor.nodes] == ["/mic/b"]
    finally:
        release.set()
        plugin.dispatch("soundevent", {"action": "stop"})


def test_model_load_failure_reports_error_and_next_start_retries(monkeypatch):
    builder = _ModelBuilderProbe(failures=1)
    monkeypatch.setattr(soundevent, "_build_model", builder)
    executor = _FakeExecutor()
    plugin = soundevent.SoundEventPlugin({}, executor)

    try:
        first = plugin.dispatch(
            "soundevent",
            {"action": "start", "input_topic": "/mic/a", "instance_id": "a"},
        )
        assert first["state"] == "loading"
        assert _wait_until(
            lambda: plugin.dispatch(
                "soundevent", {"action": "info", "instance_id": "a"}
            )["state"]
            == "error"
        )
        failed = plugin.dispatch(
            "soundevent", {"action": "info", "instance_id": "a"}
        )
        assert "download failed" in failed["error"]

        retry = plugin.dispatch(
            "soundevent",
            {"action": "start", "input_topic": "/mic/a", "instance_id": "a"},
        )
        assert retry["state"] == "loading"
        assert _wait_until(
            lambda: plugin.dispatch(
                "soundevent", {"action": "info", "instance_id": "a"}
            )["state"]
            == "running"
        )
        assert builder.calls == 2
    finally:
        plugin.dispatch("soundevent", {"action": "stop"})


def test_start_replaces_an_errored_node(monkeypatch):
    builder = _ModelBuilderProbe()
    monkeypatch.setattr(soundevent, "_build_model", builder)
    executor = _FakeExecutor()
    plugin = soundevent.SoundEventPlugin({}, executor)

    try:
        plugin.dispatch(
            "soundevent",
            {"action": "start", "input_topic": "/mic/a", "instance_id": "a"},
        )
        assert _wait_until(
            lambda: plugin.dispatch(
                "soundevent", {"action": "info", "instance_id": "a"}
            )["state"]
            == "running"
        )
        failed = executor.nodes[0]
        failed._stop_event.set()
        failed._queue.put_nowait(None)
        failed._thread.join(timeout=1.0)
        failed.state = "error"

        result = plugin.dispatch(
            "soundevent",
            {"action": "start", "input_topic": "/mic/a", "instance_id": "a"},
        )
        assert result["state"] == "running"
        assert len(executor.nodes) == 1
        assert executor.nodes[0] is not failed
        assert failed.destroyed
    finally:
        plugin.dispatch("soundevent", {"action": "stop"})


@pytest.mark.parametrize(
    "action", ["stop_one", "stop_all", "replace_topic", "replace_error"]
)
def test_retire_leaves_the_executor_before_destroying_subscription(
    action, monkeypatch
):
    """No subscription handle may be destroyed while its node is registered."""
    order = []
    retained_at_remove = []

    class RecordingExecutor(_FakeExecutor):
        def remove_node(self, node):
            order.append(("remove_node", node in self.nodes))
            retained_at_remove.append(node._subscription in node.subscriptions)
            super().remove_node(node)

    executor = RecordingExecutor()
    plugin = soundevent.SoundEventPlugin({}, executor)
    plugin._model = _LifecycleModel()
    plugin._model_state = "ready"
    start_args = {"action": "start", "input_topic": "/mic/a", "instance_id": "a"}

    try:
        assert plugin.dispatch("soundevent", start_args)["state"] == "running"
        node = executor.nodes[0]
        worker = node._thread
        original_destroy_subscription = node.destroy_subscription
        original_destroy_node = node.destroy_node

        def destroy_subscription(subscription):
            order.append(("destroy_subscription", node in executor.nodes))
            return original_destroy_subscription(subscription)

        def destroy_node():
            order.append(("destroy_node", node in executor.nodes))
            # The shared fake only marks the node destroyed. Model rclpy's
            # ownership here: destroying a node also releases its subscriptions.
            for subscription in list(node.subscriptions):
                node.destroy_subscription(subscription)
            original_destroy_node()

        monkeypatch.setattr(node, "destroy_subscription", destroy_subscription)
        monkeypatch.setattr(node, "destroy_node", destroy_node)

        if action == "stop_one":
            args = {"action": "stop", "instance_id": "a"}
        elif action == "stop_all":
            args = {"action": "stop"}
        elif action == "replace_topic":
            args = {**start_args, "input_topic": "/mic/b"}
        else:
            node.state = "error"
            args = start_args

        result = plugin.dispatch("soundevent", args)

        assert result["state"] == ("idle" if action.startswith("stop") else "running")
        assert order == [
            ("remove_node", True),
            ("destroy_node", False),
            ("destroy_subscription", False),
        ]
        assert retained_at_remove == [True]
        assert node.destroyed
        assert node.subscriptions == []
        assert not worker.is_alive()
        assert node not in executor.nodes

        if action.startswith("stop"):
            # A repeated stop is harmless, and a later start uses a fresh node.
            assert plugin.dispatch("soundevent", args) == {"state": "idle"}
            assert len(order) == 3
            assert plugin.dispatch("soundevent", start_args)["state"] == "running"
        assert len(executor.nodes) == 1
        assert executor.nodes[0] is not node
    finally:
        plugin.dispatch("soundevent", {"action": "stop"})


def test_stop_ignores_audio_and_results_before_disposal():
    node = soundevent._SoundEventNode("/mic/audio", _LifecycleModel(), "test")
    try:
        node.start()
        subscription = node._subscription
        generation = node._stream_generation
        assert node.stop() == {"state": "idle"}
        assert subscription in node.subscriptions

        queued = node._queue.qsize()
        subscription.callback(_audio_chunk(b"\x00\x00" * 512, 100.0))
        assert node._queue.qsize() == queued
        assert node.status()["statistics"]["chunks_received"] == 0
        assert not node._publish([{"name": "Bark", "confidence": 0.9}], 100.0, generation)
        assert node.publishers[0].messages == []
    finally:
        node.stop()
        node.destroy_node()


def test_pcm_buffering_publishes_window_end_timestamps(monkeypatch):
    class FakeModel:
        labels = ["Bark"]

        def __init__(self):
            self.waveforms = []

        def predict(self, waveform):
            self.waveforms.append(waveform.copy())
            return np.asarray([0.9], dtype=np.float32)

    model = FakeModel()
    node = soundevent._SoundEventNode("/mic/audio", model, "test")
    node.start()
    try:
        sample_count = soundevent.WINDOW_SAMPLES + soundevent.HOP_SAMPLES
        pcm = np.full(sample_count, 16_384, dtype="<i2").tobytes()
        start_timestamp = 100.25
        offsets = (0, 8_000, 20_000, len(pcm))
        for index, (begin, end) in enumerate(zip(offsets, offsets[1:])):
            node._audio_callback(
                _audio_chunk(pcm[begin:end], start_timestamp + index * 0.1)
            )

        assert _wait_until(lambda: len(node.publishers[0].messages) == 2)
        payloads = [json.loads(raw) for raw in node.publishers[0].messages]

        assert [payload["timestamp"] for payload in payloads] == pytest.approx(
            [
                start_timestamp + soundevent.WINDOW_SECONDS,
                start_timestamp + soundevent.WINDOW_SECONDS + soundevent.HOP_SECONDS,
            ]
        )
        assert [payload["events"] for payload in payloads] == [
            [{"name": "Bark", "confidence": pytest.approx(0.9)}],
            [{"name": "Bark", "confidence": pytest.approx(0.9)}],
        ]
        assert len(model.waveforms) == 2
        assert all(waveform.shape == (soundevent.WINDOW_SAMPLES,) for waveform in model.waveforms)
        assert all(np.allclose(waveform, 0.5) for waveform in model.waveforms)
    finally:
        node.stop()
        node.destroy_node()
