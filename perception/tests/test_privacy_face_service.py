"""Shared ownership, strict detection, and recognition compatibility contracts."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from plugins import ort_worker as ow
from plugins.face_proxy import FaceServiceProxy
from plugins.face_service import FaceService
from plugins.privacy_runtime import encode_jpeg
from test_ort_worker import HELLO, _worker


def test_service_leases_are_atomic_and_last_release_only(monkeypatch):
    worker, _ctx, _seen = _worker(monkeypatch, [HELLO])
    worker._ensure_started()
    sent = []
    monkeypatch.setattr(
        worker, "_request", lambda command, timeout: sent.append(command) or {}
    )

    def acquire(_):
        return worker.acquire_service(
            "shared", "plugins.face_service", "build", model="same"
        )

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(acquire, range(20)))
    assert len([c for c in sent if c[0] == "service"]) == 1
    assert worker._service_refs["shared"] == 20
    with pytest.raises(ow.OrtWorkerError, match="conflict"):
        worker.acquire_service("shared", "plugins.face_service", "build", model="other")
    for _ in range(19):
        worker.release_service("shared")
    assert not any(c[0] == "drop" for c in sent)
    worker.release_service("shared")
    worker.release_service("shared")
    assert len([c for c in sent if c[0] == "drop"]) == 1
    assert worker._services == {} and worker._service_refs == {}


def test_call_timeout_includes_shared_worker_lock(monkeypatch):
    worker = ow.OrtWorker()
    entered, release = threading.Event(), threading.Event()

    def holder():
        with worker._lock:
            entered.set()
            release.wait(2)

    thread = threading.Thread(target=holder)
    thread.start()
    assert entered.wait(1)
    try:
        with pytest.raises(ow.OrtWorkerError, match="busy"):
            worker.call("shared", "detect", timeout=0.02)
    finally:
        release.set()
        thread.join(1)


def test_termination_closes_ipc_endpoints_and_reaps_feeder(monkeypatch):
    worker, ctx, _seen = _worker(monkeypatch, [HELLO])
    worker._ensure_started()
    closed, joined = [], []
    for index, channel in enumerate((ctx.cmd, ctx.res)):
        channel._reader = SimpleNamespace(
            close=lambda i=index: closed.append((i, "reader"))
        )
        channel._writer = SimpleNamespace(
            close=lambda i=index: closed.append((i, "writer"))
        )
        channel._thread = SimpleNamespace(
            join=lambda timeout, i=index: joined.append((i, timeout))
        )
    worker.close()
    assert ctx.proc.terminated
    assert closed == [(0, "reader"), (0, "writer"), (1, "reader"), (1, "writer")]
    assert joined == [(0, 1), (1, 1)]


def test_idle_worker_death_reaps_old_channels_before_restart(monkeypatch):
    import multiprocessing as mp

    worker, ctx, _seen = _worker(monkeypatch, [HELLO])
    worker._ensure_started()
    closed = []
    for index, channel in enumerate((ctx.cmd, ctx.res)):
        channel.close = lambda i=index: closed.append(i)
    replacement = type(ctx)([HELLO])
    monkeypatch.setattr(mp, "get_context", lambda method: replacement)
    ctx.proc._alive = False
    worker._ensure_started()
    assert ctx.proc.terminated and closed == [0, 1]
    assert worker._proc is replacement.proc
    worker.close()


def test_failed_rebuild_keeps_spec_and_retries(monkeypatch):
    worker, _ctx, _seen = _worker(monkeypatch, [HELLO])
    worker._services["shared"] = ("plugins.face_service", "build", {"model": "same"})
    worker._service_refs["shared"] = 2
    sent = []

    def request(command, timeout):
        sent.append(command)
        if len(sent) == 1:
            raise ow.OrtWorkerError("temporary rebuild failure")
        return {"ok": True}

    monkeypatch.setattr(worker, "_request", request)
    worker._ensure_started()
    assert "shared" in worker._services
    assert "shared" not in worker._service_ready
    assert worker.call("shared", "detect")["ok"]
    assert [c[0] for c in sent] == ["service", "service", "call"]
    assert worker._service_refs["shared"] == 2


def test_face_proxies_share_without_overwrite_and_close_independently(monkeypatch):
    import utils.model_downloader as downloader
    import utils.onnx_provider as providers

    worker, _ctx, _seen = _worker(monkeypatch, [HELLO])
    worker._ensure_started()
    sent = []

    def request(command, timeout):
        sent.append(command)
        return {
            "device": "cpu",
            "providers": ["CPUExecutionProvider"],
            "decoded": True,
            "height": 64,
            "width": 96,
            "faces": [],
        }

    monkeypatch.setattr(worker, "_request", request)
    monkeypatch.setattr(ow, "get_worker", lambda: worker)
    monkeypatch.setattr(downloader, "ensure_face_model", lambda *a, **kw: {})
    monkeypatch.setattr(
        providers, "ort_providers_for_device", lambda d: ["CPUExecutionProvider"]
    )
    monkeypatch.setattr(providers, "warn_on_parked_cores", lambda *a: None)
    privacy = FaceServiceProxy(model_dir="/models/face/buffalo_sc", device="cpu")
    recognition = FaceServiceProxy(model_dir="/models/face/buffalo_sc", device="cpu")
    third = FaceServiceProxy(
        model_dir="/models/face/buffalo_sc", device="cpu", num_threads=3
    )
    assert privacy._service_key == recognition._service_key != third._service_key
    assert len([c for c in sent if c[0] == "service"]) == 2
    assert all(c[4]["detection_only"] is True for c in sent if c[0] == "service")
    privacy.detect(b"jpeg", "frame")
    privacy.close()
    assert recognition.recognise(b"jpeg") == ((64, 96), [])
    assert not any(c == ("drop", recognition._service_key) for c in sent)
    recognition.close()
    recognition.close()
    third.close()
    assert worker._services == {} and worker._service_refs == {}
    with pytest.raises(RuntimeError, match="closed"):
        privacy.detect(b"jpeg", "late")


def test_service_detect_never_aligns_or_embeds():
    class Analyzer:
        def detect(self, image, **kwargs):
            assert kwargs == {"max_faces": 0, "det_thresh": 0.5, "strict": True}
            return [
                SimpleNamespace(bbox=(1.0, 1.0, 10.0, 10.0), det_score=0.9),
                SimpleNamespace(bbox=(20.0, 20.0, 30.0, 30.0), det_score=0.8),
            ]

        def prepare(self, *a):
            pytest.fail("detection-only must not align")

        def embed(self, *a):
            pytest.fail("detection-only must not embed")

        def enable_recognition(self):
            pytest.fail("detection-only must not load recognition")

    service = FaceService.__new__(FaceService)
    service._analyzer = Analyzer()
    data = encode_jpeg(np.zeros((64, 96, 3), np.uint8))
    result = service.detect(data, "same-frame")
    assert result["complete"] and result["truncated"] is False
    assert len(result["faces"]) == 2
    assert result["frame_token"] == "same-frame"
    assert set(result["faces"][0]) == {"bbox", "det_score"}


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "nan_score",
        "inf_bbox",
        "nan_keypoint",
        "bad_shape",
        "bad_count",
        "bad_score",
    ],
)
def test_strict_scrfd_head_distinguishes_empty_success_from_broken(fault):
    from plugins.face_runtime import FaceAnalyzer
    from utils.cv2_compat import load_cv2

    outputs = []
    for channels in (1, 4, 10):
        for stride in (8, 16, 32):
            outputs.append(np.zeros(((32 // stride) ** 2 * 2, channels), np.float32))
    if fault == "nan_score":
        outputs[0][0, 0] = np.nan
    elif fault == "inf_bbox":
        outputs[3][0, 0] = np.inf
    elif fault == "nan_keypoint":
        outputs[6][0, 0] = np.nan
    elif fault == "bad_shape":
        outputs[0] = outputs[0][:-1]
    elif fault == "bad_count":
        outputs.pop()
    elif fault == "bad_score":
        outputs[0][0, 0] = 1.5
    analyzer = FaceAnalyzer.__new__(FaceAnalyzer)
    analyzer._cv2 = load_cv2()
    analyzer._det_size = (32, 32)
    analyzer._det_thresh, analyzer._nms_thresh = 0.5, 0.4
    analyzer._det_input, analyzer._det_outputs = "input", [str(i) for i in range(9)]
    analyzer._det = SimpleNamespace(run=lambda *a: outputs)
    image = np.zeros((32, 32, 3), np.uint8)
    if fault is None:
        assert analyzer.detect(image, strict=True) == []
    else:
        with pytest.raises(ValueError, match="SCRFD"):
            analyzer.detect(image, strict=True)
