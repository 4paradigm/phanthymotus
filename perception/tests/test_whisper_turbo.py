"""Whisper lifecycle and verified downloads, without GPU, audio assets or network."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlsplit
import wave

import pytest

from plugins import asr
from plugins.whisper_turbo import WhisperTurboAdapter
from utils import install_whisper_turbo as install
from utils import model_downloader as downloads
from utils.model_progress import fetch_status


def metadata(data, **extra):
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(), **extra}


class Chunks(io.BytesIO):
    def read(self, size=-1):
        return super().read(min(size, 2) if size >= 0 else size)


def test_download_progress_fallback_alias_and_verified_cache(tmp_path, monkeypatch, caplog):
    data = b"01234567890123456789"
    files = {"model.bin": metadata(data, source_path="model/weights.bin")}
    calls, status = [], []
    def open_url(url, **kwargs):
        calls.append(url)
        return Chunks(b"x" * len(data) if "bad.test" in url else data)
    monkeypatch.setattr(downloads, "urlopen", open_url)
    monkeypatch.setattr(downloads.time, "sleep", lambda _: None)
    progress, _ = fetch_status(status.append, "Whisper Turbo")
    args = ("whisper-test", str(tmp_path), ["https://bad.test", "https://good.test"], files)
    downloads.ensure_verified_bundle(*args, progress_cb=progress)
    assert (tmp_path / "model.bin").read_bytes() == data
    assert calls[-1] == "https://good.test/model/weights.bin"
    assert "SHA256 mismatch" in caplog.text
    assert "(ValueError)" in caplog.text
    assert any("10%" in line for line in status) and any("100%" in line for line in status)
    calls.clear()
    status.clear()
    downloads.ensure_verified_bundle(*args, progress_cb=progress)
    assert calls == status == []


def test_bad_download_keeps_existing_cache_and_skips_valid_files(tmp_path, monkeypatch):
    good = b"verified"
    (tmp_path / "keep.bin").write_bytes(good)
    (tmp_path / "model.bin").write_bytes(b"old")
    files = {"keep.bin": metadata(good), "model.bin": metadata(b"new")}
    calls = []
    monkeypatch.setattr(downloads, "urlopen", lambda url, **kw: calls.append(url) or Chunks(b"bad"))
    monkeypatch.setattr(downloads.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="model.bin"):
        downloads.ensure_verified_bundle("whisper-test", str(tmp_path), "https://bad.test", files)
    assert all(url.endswith("/model.bin") for url in calls)
    assert (tmp_path / "keep.bin").read_bytes() == good
    assert (tmp_path / "model.bin").read_bytes() == b"old"
    assert not list(tmp_path.glob(".whisper-test-*"))


def test_source_probes_use_remote_path_for_a_cache_alias(monkeypatch):
    probes = []
    monkeypatch.setattr(downloads, "_probe_source", lambda src, name: probes.append(name) or 1000000)
    files = {"model.bin": {"size": 1000000, "source_path": "model/weights.bin"}}
    downloads._order_sources("whisper", ["first", "second"], files)
    assert probes == ["model/weights.bin", "model/weights.bin"]


@pytest.mark.parametrize("alias", ["", "../weights.bin", "/weights.bin"])
def test_invalid_remote_alias_fails_before_network(tmp_path, monkeypatch, alias):
    monkeypatch.setattr(downloads, "urlopen", lambda *a, **kw: pytest.fail("unexpected network"))
    with pytest.raises(ValueError, match="Invalid model filename"):
        downloads.ensure_verified_bundle("whisper-test", str(tmp_path), "https://release.test",
                                         {"model.bin": metadata(b"x", source_path=alias)})


@pytest.mark.parametrize("jp", ["511", "61"])
def test_installer_layout_and_warm_cache(tmp_path, monkeypatch, jp):
    blobs = {name: b"data" for name in [
        "jp511/bin/whisper_turbo_worker", "jp61/bin/whisper_turbo_worker",
        "jp511/lib/libwhisper.so.1.8.3", "jp61/lib/libwhisper.so.1.8.3",
        "model/ggml-large-v3-turbo-q5_0.bin", "lora/ja.wslr", "numeric.json",
        "numeric-itn/lexicon.json", "LICENSE.whisper", "LICENSE.whisper.cpp", "MODEL_CARD.json"]}
    manifest = {"files": [{"path": name, **metadata(data)} for name, data in blobs.items()],
                "cuda_runtime_packages": {"jp511": [
                    {"filename": "cuda.deb", "url": "https://nvidia.test/cuda.deb", **metadata(b"cuda")}]}}
    raw = json.dumps(manifest).encode()
    blobs.update({"release.json": raw, "cuda.deb": b"cuda"})
    monkeypatch.setattr(install, "MANIFEST_SIZE", len(raw))
    monkeypatch.setattr(install, "MANIFEST_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(install, "_sources", lambda: ["https://release.test"])
    calls, extractions = [], []
    def open_url(url, **kwargs):
        calls.append(url)
        return io.BytesIO(blobs[urlsplit(url).path.lstrip("/")])
    def extract(command, **kwargs):
        extractions.append(command)
        Path(command[-1]).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(downloads, "urlopen", open_url)
    monkeypatch.setattr(install.subprocess, "run", extract)
    worker = install.ensure_release(tmp_path, jp)
    assert worker == tmp_path / "runtime" / ("jp" + jp) / "bin/whisper_turbo_worker"
    assert worker.stat().st_mode & 0o111
    assert (tmp_path / "model.bin").read_bytes() == b"data"
    assert not (tmp_path / "model").exists()
    assert (worker.parent.parent / "lib/libwhisper.so").resolve().name == "libwhisper.so.1.8.3"
    assert not (tmp_path / "runtime" / ("jp61" if jp == "511" else "jp511")).exists()
    assert len(extractions) == (1 if jp == "511" else 0)
    if extractions:
        assert extractions[0][-1] == str(tmp_path / "runtime/jp511/cuda-runtime")
    calls.clear()
    install.ensure_release(tmp_path, jp)
    assert calls == []


def test_verified_mirror_override_retains_public_fallback(monkeypatch):
    monkeypatch.setenv("WHISPER_TURBO_MODEL_BASE_URL", "https://mirror.test/fixed")
    assert install._sources()[0] == "https://mirror.test/fixed"
    assert install.API_URL in install._sources()


def test_invalid_jetpack_fails_before_cache_creation(tmp_path):
    with pytest.raises(RuntimeError, match="JetPack"):
        install.ensure_release(tmp_path / "unused", "unknown")
    assert not (tmp_path / "unused").exists()


def test_constructor_info_config_and_invalid_start_are_lazy(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("backend created before valid start")
    monkeypatch.setattr(asr, "_build_asr_adapter", forbidden)
    plugin = asr.ASRPlugin({"asr_model": "whisper-turbo-q5", "device": "gpu"}, None)
    assert plugin.dispatch("asr", {"action": "info"})["state"] == "idle"
    assert plugin.dispatch("asr", {"action": "config"})["status"] == "configured"
    with pytest.raises(ValueError, match="input_topic"):
        plugin.dispatch("asr", {"action": "start"})
    assert plugin._adapter is None and not plugin._loading


def test_config_switch_does_not_start_native_worker(monkeypatch):
    builds = []
    monkeypatch.setattr(asr, "_build_asr_adapter", lambda *a, **kw: builds.append(a) or object())
    plugin = asr.ASRPlugin({"asr_model": "sensevoice-small", "device": "cpu"}, None)
    plugin.dispatch("asr", {"action": "config", "asr_model": "whisper-turbo-q5"})
    deadline = time.monotonic() + 2
    while plugin._loading and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not plugin._loading and len(builds) == 1
    assert plugin._adapter is None and plugin._device == "gpu"


@pytest.mark.parametrize("instance", ["", "card"])
def test_stop_during_preparation_cancels_start(monkeypatch, instance):
    entered, release = threading.Event(), threading.Event()
    adapters = []
    class Adapter(WhisperTurboAdapter):
        def __init__(self):
            self.closed = False
            adapters.append(self)
        def prepare(self, **kwargs):
            entered.set()
            assert release.wait(3)
        def close(self):
            self.closed = True
    monkeypatch.setattr(asr, "_build_asr_adapter", lambda *a, **kw: Adapter())
    plugin = asr.ASRPlugin({"asr_model": "whisper-turbo-q5", "device": "gpu"}, None)
    results = []
    thread = threading.Thread(target=lambda: results.append(plugin.dispatch(
        "asr", {"action": "start", "instance_id": "card", "input_topic": "/mic"})))
    thread.start()
    try:
        assert entered.wait(2)
        plugin.dispatch("asr", {"action": "stop", "instance_id": instance})
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive() and results == [{"state": "idle"}]
    assert not plugin._nodes and not plugin._pending_starts
    assert plugin._adapter is None and adapters[0].closed
    assert not plugin._lazy_load_requested and not plugin._lazy_start_requests


def test_cancel_one_start_preserves_peer_and_last_stop_releases_backend(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    adapters, nodes, results = [], [], {}
    class Adapter(WhisperTurboAdapter):
        def __init__(self):
            self.closed = False
            adapters.append(self)
        def prepare(self, **kwargs):
            entered.set()
            assert release.wait(3)
        def close(self):
            self.closed = True
    class Node:
        state = "idle"
        _direction_topic = ""
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            self.state = "running"
            return {"state": "running"}
        def request_stop(self):
            pass
        def stop(self):
            self.state = "idle"
            return {"state": "idle"}
        def destroy_node(self):
            pass
    monkeypatch.setattr(asr, "_build_asr_adapter", lambda *a, **kw: Adapter())
    monkeypatch.setattr(asr, "_ASRNode", Node)
    plugin = asr.ASRPlugin({"asr_model": "whisper-turbo-q5", "device": "gpu"},
                           SimpleNamespace(add_node=nodes.append, remove_node=nodes.remove))
    def start(key):
        results[key] = plugin.dispatch("asr", {
            "action": "start", "instance_id": key, "input_topic": "/mic/" + key})
    threads = [threading.Thread(target=start, args=(key,)) for key in ("a", "b")]
    for thread in threads:
        thread.start()
    try:
        assert entered.wait(2)
        deadline = time.monotonic() + 2
        while plugin._lazy_start_requests != {"a", "b"} and time.monotonic() < deadline:
            time.sleep(0.01)
        assert plugin._lazy_start_requests == {"a", "b"}
        plugin.dispatch("asr", {"action": "stop", "instance_id": "a"})
    finally:
        release.set()
        for thread in threads:
            thread.join(3)
    assert all(not thread.is_alive() for thread in threads)
    assert results == {"a": {"state": "idle"}, "b": {"state": "running"}}
    assert len(adapters) == 1 and not adapters[0].closed
    plugin.dispatch("asr", {"action": "stop", "instance_id": "b"})
    assert plugin._adapter is None and adapters[0].closed


def test_stale_deferred_start_cannot_revive_after_stop(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    adapters = []
    class Adapter(WhisperTurboAdapter):
        def __init__(self):
            self.closed = False
            adapters.append(self)
        def prepare(self, **kwargs):
            entered.set()
            assert release.wait(3)
        def close(self):
            self.closed = True
    monkeypatch.setattr(asr, "_build_asr_adapter", lambda *a, **kw: Adapter())
    monkeypatch.setattr(asr, "MODEL_LOAD_GRACE_S", 0)
    plugin = asr.ASRPlugin({"asr_model": "whisper-turbo-q5", "device": "gpu"}, None)
    try:
        assert plugin.dispatch("asr", {
            "action": "start", "instance_id": "card", "input_topic": "/mic"})["state"] == "loading"
        assert entered.wait(2)
        deferred = plugin._pending_starts["card"]
        plugin.dispatch("asr", {"action": "stop", "instance_id": "card"})
    finally:
        release.set()
    deadline = time.monotonic() + 2
    while plugin._loading and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not plugin._loading
    assert plugin.dispatch("asr", deferred) == {"state": "idle"}
    assert len(adapters) == 1 and adapters[0].closed and plugin._adapter is None


def test_worker_protocol_reuses_one_process_and_returns_language(tmp_path, monkeypatch):
    worker = tmp_path / "worker"
    worker.write_text("#!/usr/bin/env python3\nimport json,sys\n"
                     "print(json.dumps({'ready':True,'gpu':True}),flush=True)\n"
                     "for line in sys.stdin:\n"
                     " print(json.dumps({'raw_text':'hello','predicted_language':'ja'}),flush=True)\n")
    worker.chmod(0o755)
    (tmp_path / "model.bin").write_bytes(b"protocol fixture")
    monkeypatch.setenv("WHISPER_TURBO_WORKER", str(worker))
    monkeypatch.setattr("ctypes.util.find_library", lambda _: None)
    audio = io.BytesIO()
    with wave.open(audio, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 1600)
    adapter = WhisperTurboAdapter(str(tmp_path))
    assert adapter._process is None
    try:
        adapter.prepare()
        pid = adapter._process.pid
        for _ in range(2):
            assert adapter.transcribe_rich(audio.getvalue()) == ("hello", {"lang": "ja"})
            assert adapter._process.pid == pid
    finally:
        adapter.close()
    assert adapter._process is None
