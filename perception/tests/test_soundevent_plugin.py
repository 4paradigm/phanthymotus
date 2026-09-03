"""Host-side tests for the SoundEvent model and PCM streaming contract."""

from __future__ import annotations

import importlib.util
import json
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

from vision_stubs import _FakeAudioChunk, _FakeNode, _wait_until

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


def test_soundevent_model_base_honors_environment(monkeypatch):
    monkeypatch.delenv("SOUNDEVENT_MODEL_BASE_URL", raising=False)
    default_module = _load_model_downloader_copy("model_downloader_default_test")
    assert default_module.SOUNDEVENT_MODEL_BASE == (
        f"{default_module.COS_BASE}/soundevent"
    )

    configured_base = "https://models.example/soundevent"
    monkeypatch.setenv("SOUNDEVENT_MODEL_BASE_URL", configured_base)
    configured_module = _load_model_downloader_copy("model_downloader_env_test")
    assert configured_module.SOUNDEVENT_MODEL_BASE == configured_base


def test_soundevent_model_download_uses_pinned_manifest(tmp_path, monkeypatch):
    configured_base = "https://models.example/soundevent"
    captured = {}

    monkeypatch.setattr(model_downloader, "SOUNDEVENT_MODEL_BASE", configured_base)
    monkeypatch.setattr(
        model_downloader,
        "require_models_subpath",
        lambda model_dir: str(tmp_path),
    )

    def fake_ensure(name, model_dir, base_url, files):
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
        "model_dir": str(tmp_path),
        "base_url": configured_base,
        "files": {
            "yamnet_classification.tflite": {
                "size": 4_126_810,
                "sha256": (
                    "10c95ea3eb9a7bb4cb8bddf6feb023250381008177ac162ce169694d05c317de"
                ),
            }
        },
    }


def test_pcm_buffering_publishes_window_end_timestamps(monkeypatch):
    monkeypatch.setattr(
        _FakeNode,
        "destroy_subscription",
        lambda self, subscription: self.subscriptions.remove(subscription),
        raising=False,
    )

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
