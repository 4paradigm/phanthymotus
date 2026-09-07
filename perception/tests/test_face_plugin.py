"""
tests/test_face_plugin.py — Face plugin lifecycle, failure reasons, enrolment.

ROS stubs come from vision_stubs (installed by conftest before collection). The
analyzer is faked: the real one needs two ONNX models and onnxruntime, neither
of which a host-side suite should require. The *decode* is validated separately
against the real models — see perception/README.md § Face Recognition.

Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import json
import os
import threading
import time
import zipfile

import numpy as np
import pytest

from vision_stubs import (  # noqa: F401
    _FakeCompressedImage,
    _FakeExecutor,
    _FakeNode,
    _FakeString,
    _wait_until,
)

import plugins.face as face_plugin  # noqa: E402
import plugins.face_db as face_db_module  # noqa: E402
from plugins.face_db import EMBEDDING_DIM, FaceDB  # noqa: E402
from plugins.face_runtime import DetectedFace  # noqa: E402


# ── fakes ─────────────────────────────────────────────────────────────────────

def _unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return raw / np.linalg.norm(raw)


class _FakeFrame:
    """A JPEG stand-in: the bytes encode which faces the fake analyzer reports.

    `b"<identity>:<size>:<blur>|<identity>:<size>:<blur>"` — one segment per
    face. This keeps the tests about the plugin's decisions rather than about
    image encoding.
    """

    @staticmethod
    def build(*faces: tuple[int, int, float]) -> bytes:
        return "|".join(
            f"{identity}:{size}:{blur}" for identity, size, blur in faces
        ).encode()

    @staticmethod
    def one(identity: int, size: int = 120, blur: float = 500.0) -> bytes:
        return _FakeFrame.build((identity, size, blur))


class _FakeAnalyzer:
    """Detects whatever the frame bytes say, and embeds by identity number."""

    def __init__(self, delay: float = 0.0, frame_shape=(480, 640)):
        self.delay = delay
        self.closed = False
        self.frame_shape = frame_shape
        self.seen: list[bytes] = []
        self.embed_calls = 0

    # -- the surface plugins/face.py uses --
    def decode_jpeg(self, data: bytes):
        self.seen.append(data)
        if data == b"corrupt":
            return None
        image = np.zeros((*self.frame_shape, 3), dtype=np.uint8)
        image[0, 0, 0] = 1                      # keep .size truthy
        self._spec = data.decode()
        return image

    def detect(self, image, max_faces: int = 0):
        if self.delay:
            time.sleep(self.delay)
        if not self._spec:
            return []
        faces = []
        height, width = self.frame_shape
        segments = self._spec.split("|")
        # Lay the boxes out left to right; the first one is centred, so it wins
        # the centre-weighted dominance comparison when sizes are equal.
        for index, segment in enumerate(segments):
            identity, size, blur = segment.split(":")
            side = int(size)
            centre_x = width / 2 if index == 0 else (index * width) / (len(segments) + 1)
            x1 = max(0.0, centre_x - side / 2)
            y1 = max(0.0, height / 2 - side / 2)
            face = DetectedFace(
                bbox=(x1, y1, x1 + side, y1 + side),
                det_score=0.9,
                kps=np.zeros((5, 2), dtype=np.float32),
                blur=float(blur),
            )
            face.identity = int(identity)
            faces.append(face)
        faces.sort(key=lambda f: f.area, reverse=True)
        return faces[:max_faces] if max_faces else faces

    def prepare(self, image, face):
        face.aligned = np.zeros((112, 112, 3), dtype=np.uint8)
        return face

    def embed(self, aligned):
        raise AssertionError("subclasses carry the identity; use _SpecAnalyzer")

    def close(self):
        self.closed = True


class _SpecAnalyzer(_FakeAnalyzer):
    """The analyzer the tests use: identity travels in the aligned crop.

    `prepare` stamps the face's identity into pixel [0,0,0] and `embed` reads it
    back, which is how a frame spec like `b"7:120:500"` ends up as a stable
    512-d vector for person 7 without any model.
    """

    def prepare(self, image, face):
        aligned = np.zeros((112, 112, 3), dtype=np.uint8)
        aligned[0, 0, 0] = face.identity
        face.aligned = aligned
        return face

    def embed(self, aligned):
        self.embed_calls += 1
        return _unit(int(aligned[0, 0, 0]))


@pytest.fixture(autouse=True)
def _tmp_models(monkeypatch, tmp_path):
    """Keep the DB in tmp and never touch /models or the network."""
    monkeypatch.setattr(
        face_db_module, "require_models_subpath", lambda path, root="/models": str(path)
    )
    monkeypatch.setattr(
        face_plugin, "_image_roots", lambda cfg: (str(tmp_path),)
    )


def _base_cfg(tmp_path, **overrides) -> dict:
    cfg = {
        "db_dir": os.path.join(str(tmp_path), "db"),
        "match_threshold": 0.35,
        "min_face_px": 64,
        "blur_min": 60.0,
        "det_thresh": 0.5,
        "subject_dominance": 1.6,
        "max_faces": 8,
        "enroll_window_s": 3.0,
        "enroll_max_analyzed": 8,
        "min_interval_ms": 0,
        "unknown_capacity": 500,
        "max_batch": 200,
    }
    cfg.update(overrides)
    return cfg


class _EngineProbe:
    """Counts engine builds; optional delay stands in for the model load."""

    def __init__(self, tmp_path, delay: float = 0.0, analyzer_cls=_SpecAnalyzer):
        self.tmp_path = tmp_path
        self.delay = delay
        self.analyzer_cls = analyzer_cls
        self.calls = 0
        self.built: list = []
        self.lock = threading.Lock()

    def __call__(self, cfg):
        with self.lock:
            self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        engine = face_plugin._FaceEngine(
            self.analyzer_cls(), FaceDB(db_dir=cfg["db_dir"],
                                       unknown_capacity=int(cfg.get("unknown_capacity", 500)))
        )
        self.built.append(engine)
        return engine


@pytest.fixture
def plugin(monkeypatch, tmp_path):
    probe = _EngineProbe(tmp_path)
    monkeypatch.setattr(face_plugin, "_build_engine", probe)
    executor = _FakeExecutor()
    instance = face_plugin.FaceRecognitionPlugin(_base_cfg(tmp_path), executor)
    instance._probe = probe
    instance._executor_ref = executor
    yield instance
    instance.dispatch("face_recognition", {"action": "stop"})


# ── tool definition ───────────────────────────────────────────────────────────

def test_tool_shape_matches_the_card_contract():
    tool = face_plugin.TOOLS[0]
    assert tool["name"] == "face_recognition"
    assert tool["type"] == "processor"
    assert tool["multiInstance"] is True
    assert tool["topic_in"][0]["format"] == "image/jpeg"
    assert tool["topic_out"][0]["format"] == "data/json"

    schema = tool["inputSchema"]
    actions = set(schema["properties"]["action"]["enum"])
    # Every action must be described in x-action-params, or agent-core splits
    # the tool for the LLM with a parameter list that silently omits it.
    assert actions == set(schema["x-action-params"])
    declared = set(schema["properties"])
    for action, spec in schema["x-action-params"].items():
        missing = set(spec["params"]) - declared
        assert not missing, f"{action} references undeclared params: {missing}"
        assert spec["description"]

    # unknown_capacity must be editable on the card, per the requirement.
    assert "unknown_capacity" in tool["configSchema"]["properties"]
    assert tool["configSchema"]["properties"]["unknown_capacity"]["default"] == 500


# ── subject selection / failure reasons ───────────────────────────────────────

def _faces(*specs):
    analyzer = _SpecAnalyzer()
    analyzer.decode_jpeg(_FakeFrame.build(*specs))
    faces = analyzer.detect(None)
    for face in faces:
        analyzer.prepare(None, face)
    return faces, (480, 640)


def test_no_face_reason():
    gates = face_plugin._gates(_base_cfg(""))
    subject, failure = face_plugin.select_subject([], (480, 640), gates)
    assert subject is None
    assert failure["reason"] == face_plugin.REASON_NO_FACE
    assert failure["faces"] == 0


def test_low_quality_reason_names_every_failed_gate():
    gates = face_plugin._gates(_base_cfg(""))
    faces, shape = _faces((1, 40, 10.0))          # too small AND too blurry
    subject, failure = face_plugin.select_subject(faces, shape, gates)
    assert subject is None
    assert failure["reason"] == face_plugin.REASON_LOW_QUALITY
    assert "40 px" in failure["detail"] and "need" not in failure["detail"]
    assert "move closer" in failure["detail"]
    assert "hold still" in failure["detail"]
    assert failure["candidates"][0]["min_side_px"] == 40


def test_ambiguous_subject_lists_candidates():
    gates = face_plugin._gates(_base_cfg(""))
    faces, shape = _faces((1, 120, 500.0), (2, 118, 500.0))
    subject, failure = face_plugin.select_subject(faces, shape, gates)
    assert subject is None
    assert failure["reason"] == face_plugin.REASON_AMBIGUOUS
    assert failure["faces"] == 2
    assert len(failure["candidates"]) == 2
    assert "1.6" in failure["detail"] or "1.60" in failure["detail"]


def test_a_dominant_face_wins_despite_a_bystander():
    gates = face_plugin._gates(_base_cfg(""))
    faces, shape = _faces((1, 200, 500.0), (2, 70, 500.0))
    subject, failure = face_plugin.select_subject(faces, shape, gates)
    assert failure is None
    assert subject.identity == 1


def test_a_blurred_bystander_does_not_make_it_ambiguous():
    """Only faces that pass the quality gate can compete for the subject."""
    gates = face_plugin._gates(_base_cfg(""))
    faces, shape = _faces((1, 120, 500.0), (2, 120, 5.0))
    subject, failure = face_plugin.select_subject(faces, shape, gates)
    assert failure is None
    assert subject.identity == 1


def test_worst_reason_prefers_ambiguity_over_blur_over_absence():
    failures = [
        {"reason": face_plugin.REASON_NO_FACE, "detail": "a"},
        {"reason": face_plugin.REASON_LOW_QUALITY, "detail": "b"},
        {"reason": face_plugin.REASON_AMBIGUOUS, "detail": "c"},
    ]
    picked = face_plugin._worst_reason(failures)
    assert picked["reason"] == face_plugin.REASON_AMBIGUOUS
    assert picked["frames_examined"] == 3
    assert picked["frame_reasons"] == {
        face_plugin.REASON_AMBIGUOUS: 1,
        face_plugin.REASON_LOW_QUALITY: 1,
        face_plugin.REASON_NO_FACE: 1,
    }
    assert face_plugin._worst_reason([])["reason"] == face_plugin.REASON_NO_FACE


# ── lifecycle ─────────────────────────────────────────────────────────────────

def test_first_start_reports_loading_then_runs(plugin):
    result = plugin.dispatch("face_recognition", {
        "action": "start", "input_topic": "/cam/rgb"})
    assert result["state"] == "loading"
    assert result["output"] == "/cam/rgb/face"

    assert _wait_until(lambda: plugin.dispatch(
        "face_recognition", {"action": "info"})["state"] == "running")
    info = plugin.dispatch("face_recognition", {"action": "info"})
    assert info["topic_out"][0]["topic"] == "/cam/rgb/face"
    assert info["database"]["persons"] == 0


def test_concurrent_starts_build_one_engine(monkeypatch, tmp_path):
    probe = _EngineProbe(tmp_path, delay=0.25)
    monkeypatch.setattr(face_plugin, "_build_engine", probe)
    plugin = face_plugin.FaceRecognitionPlugin(_base_cfg(tmp_path), _FakeExecutor())
    try:
        threads = [
            threading.Thread(target=plugin.dispatch, args=(
                "face_recognition",
                {"action": "start", "input_topic": f"/cam{index}", "instance_id": f"i{index}"},
            ))
            for index in range(6)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert _wait_until(lambda: len(plugin._nodes) == 6, timeout=6.0)
        assert probe.calls == 1          # one download, one load, N instances
    finally:
        plugin.dispatch("face_recognition", {"action": "stop"})


def test_start_stop_churn_leaves_no_orphan_node(monkeypatch, tmp_path):
    """The failure mode README § Plugin Concurrency documents."""
    probe = _EngineProbe(tmp_path, delay=0.05)
    monkeypatch.setattr(face_plugin, "_build_engine", probe)
    _FakeNode.instances.clear()
    plugin = face_plugin.FaceRecognitionPlugin(_base_cfg(tmp_path), _FakeExecutor())
    try:
        for _ in range(5):
            plugin.dispatch("face_recognition", {
                "action": "start", "input_topic": "/cam/rgb"})
            plugin.dispatch("face_recognition", {"action": "stop"})
            plugin.dispatch("face_recognition", {
                "action": "config", "min_interval_ms": 10})
        plugin.dispatch("face_recognition", {"action": "stop"})
        time.sleep(0.4)

        live = [
            node for node in _FakeNode.instances
            if getattr(node, "state", "idle") == "running" and not node.destroyed
        ]
        assert not live, f"{len(live)} orphaned node(s) still running"
        assert not any(
            node.worker_alive for node in _FakeNode.instances
            if hasattr(node, "worker_alive")
        )
    finally:
        plugin.dispatch("face_recognition", {"action": "stop"})


def test_stop_is_idempotent_and_reports_idle(plugin):
    assert plugin.dispatch("face_recognition", {"action": "stop"})["state"] == "idle"
    assert plugin.dispatch("face_recognition", {"action": "stop"})["state"] == "idle"


def test_start_requires_an_input_topic(plugin):
    with pytest.raises(ValueError):
        plugin.dispatch("face_recognition", {"action": "start"})


def test_unknown_action_returns_none(plugin):
    assert plugin.dispatch("face_recognition", {"action": "teleport"}) is None


def test_engine_load_failure_surfaces_in_info(monkeypatch, tmp_path):
    def explode(cfg):
        raise RuntimeError("onnxruntime is not installed")
    monkeypatch.setattr(face_plugin, "_build_engine", explode)
    plugin = face_plugin.FaceRecognitionPlugin(_base_cfg(tmp_path), _FakeExecutor())
    plugin.dispatch("face_recognition", {"action": "start", "input_topic": "/cam"})
    assert _wait_until(lambda: plugin.dispatch(
        "face_recognition", {"action": "info"})["state"] == "error")
    info = plugin.dispatch("face_recognition", {"action": "info"})
    assert "onnxruntime" in info["error"]
    assert "onnxruntime" in info["desc"]


# ── recognition stream ────────────────────────────────────────────────────────

def _run_one_frame(plugin, frame: bytes, topic: str = "/cam/rgb"):
    plugin.dispatch("face_recognition", {"action": "start", "input_topic": topic})
    assert _wait_until(lambda: plugin._nodes.get(topic) is not None, timeout=5.0)
    node = plugin._nodes[topic]
    assert _wait_until(lambda: node.state == "running", timeout=5.0)
    node._image_cb(_FakeCompressedImage(frame))
    publisher = node.publishers[0]
    assert _wait_until(lambda: publisher.messages, timeout=5.0)
    return node, [json.loads(item) for item in publisher.messages]


def test_unknown_face_gets_a_stable_id_across_frames(plugin):
    node, payloads = _run_one_frame(plugin, _FakeFrame.one(7))
    first = payloads[0]["faces"][0]
    assert first["person_id"] == "unknown-1"
    assert first["known"] is False
    assert first["quality"] == "ok"

    node._image_cb(_FakeCompressedImage(_FakeFrame.one(7)))
    assert _wait_until(lambda: len(node.publishers[0].messages) >= 2, timeout=5.0)
    second = json.loads(node.publishers[0].messages[-1])["faces"][0]
    assert second["person_id"] == "unknown-1"      # same person, same id
    assert second["score"] > 0.9


def test_a_registered_person_is_reported_with_their_profile(plugin):
    engine = plugin._require_engine()
    engine.db.add("运营部小王", [_unit(3)], meta={"team": "ops"})

    _node, payloads = _run_one_frame(plugin, _FakeFrame.one(3))
    face = payloads[0]["faces"][0]
    assert face["person_id"] == "p-1"
    assert face["profile"] == "运营部小王"
    assert face["known"] is True


def test_a_low_quality_face_is_reported_but_not_enrolled(plugin):
    """It must not burn an unknown-N slot on a face it cannot match again."""
    _node, payloads = _run_one_frame(plugin, _FakeFrame.one(9, size=30, blur=4.0))
    face = payloads[0]["faces"][0]
    assert face["person_id"] is None
    assert face["known"] is False
    assert face["quality"] == "low"
    assert face["reason"] == face_plugin.REASON_LOW_QUALITY
    assert plugin._require_engine().db.stats()["persons"] == 0


def test_a_corrupt_frame_publishes_an_error_not_a_crash(plugin):
    _node, payloads = _run_one_frame(plugin, b"corrupt")
    assert payloads[0]["error"] == "undecodable frame"
    assert payloads[0]["count"] == 0


def test_several_people_in_one_frame_are_all_reported(plugin):
    _node, payloads = _run_one_frame(
        plugin, _FakeFrame.build((1, 120, 500.0), (2, 118, 500.0))
    )
    payload = payloads[0]
    assert payload["count"] == 2
    ids = {face["person_id"] for face in payload["faces"]}
    assert ids == {"unknown-1", "unknown-2"}      # ambiguity only blocks enrolment


# ── register_user_photo ───────────────────────────────────────────────────────

def _write_photo(tmp_path, name: str, frame: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(frame)
    return str(path)


def test_register_from_a_photo_path(plugin, tmp_path):
    path = _write_photo(tmp_path, "alice.jpg", _FakeFrame.one(11))
    result = plugin.dispatch("face_recognition", {
        "action": "register_user_photo", "image_path": path,
        "profile": "Alice", "meta": {"team": "ops"},
    })
    assert result["ok"] is True
    assert result["person_id"] == "p-1"
    assert result["profile"] == "Alice"
    assert result["meta"] == {"team": "ops"}

    engine = plugin._require_engine()
    assert engine.db.match(_unit(11), 0.35)[0] == "p-1"


def test_register_from_base64(plugin):
    import base64
    encoded = base64.b64encode(_FakeFrame.one(12)).decode()
    result = plugin.dispatch("face_recognition", {
        "action": "register_user_photo", "image_b64": encoded, "profile": "Bob"})
    assert result["ok"] is True and result["person_id"] == "p-1"


def test_register_reports_each_failure_reason(plugin, tmp_path):
    cases = {
        "no_face": (b"", face_plugin.REASON_NO_FACE),
        "blurry": (_FakeFrame.one(1, size=30, blur=3.0), face_plugin.REASON_LOW_QUALITY),
        "crowd": (_FakeFrame.build((1, 120, 500.0), (2, 119, 500.0)),
                  face_plugin.REASON_AMBIGUOUS),
        "corrupt": (b"corrupt", face_plugin.REASON_BAD_INPUT),
    }
    for name, (frame, expected) in cases.items():
        path = _write_photo(tmp_path, f"{name}.jpg", frame)
        result = plugin.dispatch("face_recognition", {
            "action": "register_user_photo", "image_path": path, "profile": name})
        assert result["ok"] is False, name
        assert result["reason"] == expected, name
        assert result["detail"], name
    assert plugin._require_engine().db.stats()["persons"] == 0


def test_register_rejects_missing_and_out_of_root_paths(plugin, tmp_path):
    missing = plugin.dispatch("face_recognition", {
        "action": "register_user_photo", "image_path": str(tmp_path / "nope.jpg")})
    assert missing["reason"] == face_plugin.REASON_BAD_INPUT

    escaped = plugin.dispatch("face_recognition", {
        "action": "register_user_photo", "image_path": "/etc/hostname"})
    assert escaped["reason"] == face_plugin.REASON_BAD_INPUT
    assert "must be under" in escaped["detail"]

    nothing = plugin.dispatch("face_recognition", {
        "action": "register_user_photo", "profile": "x"})
    assert nothing["reason"] == face_plugin.REASON_BAD_INPUT


def test_registering_a_known_face_merges_instead_of_duplicating(plugin, tmp_path):
    path = _write_photo(tmp_path, "a.jpg", _FakeFrame.one(20))
    first = plugin.dispatch("face_recognition", {
        "action": "register_user_photo", "image_path": path, "profile": "Alice"})
    second = plugin.dispatch("face_recognition", {
        "action": "register_user_photo", "image_path": path, "profile": "Alice"})

    assert second["person_id"] == first["person_id"]
    assert second["merged"] is True
    assert second["score_to_existing"] > 0.9
    assert plugin._require_engine().db.stats()["persons"] == 1


def test_registering_a_tracked_stranger_promotes_their_unknown_id(plugin, tmp_path):
    """The id already on the activity stream must survive being named."""
    _node, payloads = _run_one_frame(plugin, _FakeFrame.one(21))
    unknown_id = payloads[0]["faces"][0]["person_id"]
    assert unknown_id == "unknown-1"

    path = _write_photo(tmp_path, "b.jpg", _FakeFrame.one(21))
    result = plugin.dispatch("face_recognition", {
        "action": "register_user_photo", "image_path": path, "profile": "Dave"})

    assert result["person_id"] == unknown_id
    assert result["promoted"] is True
    record = plugin._require_engine().db.get_person(unknown_id)
    assert record["named"] is True and record["profile"] == "Dave"


# ── register_current_stream ───────────────────────────────────────────────────

def test_register_from_the_stream_uses_the_whole_window(plugin):
    node, _payloads = _run_one_frame(plugin, _FakeFrame.one(30))
    for _ in range(4):
        node._image_cb(_FakeCompressedImage(_FakeFrame.one(30)))

    result = plugin.dispatch("face_recognition", {
        "action": "register_current_stream", "profile": "Erin"})
    assert result["ok"] is True
    assert result["frames_examined"] >= 2
    assert result["frames_used"] >= 2
    assert result["window_s"] == pytest.approx(3.0)


def test_register_from_the_stream_without_an_instance(plugin):
    result = plugin.dispatch("face_recognition", {
        "action": "register_current_stream", "profile": "Nobody"})
    assert result["ok"] is False
    assert result["reason"] == face_plugin.REASON_NO_FRAMES


def test_register_from_the_stream_reports_a_crowd(plugin):
    node, _ = _run_one_frame(
        plugin, _FakeFrame.build((1, 120, 500.0), (2, 119, 500.0)))
    for _ in range(3):
        node._image_cb(_FakeCompressedImage(
            _FakeFrame.build((1, 120, 500.0), (2, 119, 500.0))))

    result = plugin.dispatch("face_recognition", {
        "action": "register_current_stream", "profile": "Someone"})
    assert result["ok"] is False
    assert result["reason"] == face_plugin.REASON_AMBIGUOUS
    assert result["frame_reasons"][face_plugin.REASON_AMBIGUOUS] >= 1


def test_register_from_the_stream_refuses_an_unstable_subject(plugin):
    """Two people taking turns must not be enrolled as one identity."""
    node, _ = _run_one_frame(plugin, _FakeFrame.one(40))
    for identity in (41, 42, 43, 44):
        node._image_cb(_FakeCompressedImage(_FakeFrame.one(identity)))

    result = plugin.dispatch("face_recognition", {
        "action": "register_current_stream", "profile": "Whoever"})
    assert result["ok"] is False
    assert result["reason"] == face_plugin.REASON_AMBIGUOUS
    assert "stable subject" in result["detail"]


def test_register_stream_window_is_clamped_to_the_config(plugin):
    node, _ = _run_one_frame(plugin, _FakeFrame.one(45))
    node._image_cb(_FakeCompressedImage(_FakeFrame.one(45)))
    result = plugin.dispatch("face_recognition", {
        "action": "register_current_stream", "profile": "Frank", "window_s": 999})
    assert result["window_s"] == pytest.approx(3.0)


def test_register_stream_needs_an_instance_id_when_several_run(plugin):
    for index in (1, 2):
        plugin.dispatch("face_recognition", {
            "action": "start", "input_topic": f"/cam{index}",
            "instance_id": f"i{index}"})
    assert _wait_until(lambda: len(plugin._nodes) == 2, timeout=5.0)
    result = plugin.dispatch("face_recognition", {
        "action": "register_current_stream", "profile": "x"})
    assert result["reason"] == face_plugin.REASON_BAD_INPUT
    assert "instance_id" in result["detail"]


def test_the_window_drops_frames_older_than_enroll_window_s(monkeypatch, tmp_path):
    probe = _EngineProbe(tmp_path)
    monkeypatch.setattr(face_plugin, "_build_engine", probe)
    plugin = face_plugin.FaceRecognitionPlugin(
        _base_cfg(tmp_path, enroll_window_s=0.2), _FakeExecutor())
    try:
        node, _ = _run_one_frame(plugin, _FakeFrame.one(50))
        assert len(node.recent_frames()) >= 1
        time.sleep(0.35)
        node._image_cb(_FakeCompressedImage(_FakeFrame.one(50)))
        recent = node.recent_frames()
        assert len(recent) == 1, "stale frames were not evicted from the window"
    finally:
        plugin.dispatch("face_recognition", {"action": "stop"})


# ── register_user_photos (batch) ──────────────────────────────────────────────

def _make_package(tmp_path, entries, manifest=None) -> str:
    directory = tmp_path / "pack"
    directory.mkdir(exist_ok=True)
    for name, frame in entries.items():
        (directory / name).write_bytes(frame)
    if manifest is not None:
        (directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False))
    return str(directory)


def test_batch_reports_one_result_per_photo(plugin, tmp_path):
    package = _make_package(
        tmp_path,
        {
            "alice.jpg": _FakeFrame.one(60),
            "bob.jpg": _FakeFrame.one(61, size=30, blur=3.0),
            "team.jpg": _FakeFrame.build((62, 120, 500.0), (63, 119, 500.0)),
            "empty.jpg": b"",
        },
        manifest={
            "alice.jpg": "Alice from ops",
            "bob.jpg": "Bob",
            "team.jpg": "The team",
            "empty.jpg": "Nobody",
        },
    )
    result = plugin.dispatch("face_recognition", {
        "action": "register_user_photos", "package": package})

    assert result["ok"] is True
    assert result["total"] == 4
    assert result["registered"] == 1
    assert result["failed"] == 3

    by_file = {item["file"]: item for item in result["results"]}
    assert by_file["alice.jpg"]["ok"] is True
    assert by_file["alice.jpg"]["profile"] == "Alice from ops"
    assert by_file["bob.jpg"]["reason"] == face_plugin.REASON_LOW_QUALITY
    assert by_file["team.jpg"]["reason"] == face_plugin.REASON_AMBIGUOUS
    assert by_file["team.jpg"]["candidates"]
    assert by_file["empty.jpg"]["reason"] == face_plugin.REASON_NO_FACE
    # Every failure explains itself — that is the requirement.
    for item in result["results"]:
        assert item.get("ok") or item["detail"]


def test_batch_manifest_list_form_and_person_grouping(plugin, tmp_path):
    package = _make_package(
        tmp_path,
        {"a1.jpg": _FakeFrame.one(70), "a2.jpg": _FakeFrame.one(71)},
        manifest=[
            {"file": "a1.jpg", "profile": "Grace", "person": "grace",
             "meta": {"badge": "G1"}},
            {"file": "a2.jpg", "profile": "Grace", "person": "grace"},
        ],
    )
    result = plugin.dispatch("face_recognition", {
        "action": "register_user_photos", "package": package})
    assert result["registered"] == 2
    ids = {item["person_id"] for item in result["results"]}
    assert len(ids) == 1, "photos sharing a person key must become one identity"

    engine = plugin._require_engine()
    person = engine.db.get_person(ids.pop())
    assert person["samples"] == 2
    assert person["meta"] == {"badge": "G1"}


def test_batch_falls_back_to_sidecars_then_the_filename(plugin, tmp_path):
    directory = tmp_path / "pack2"
    directory.mkdir()
    (directory / "heidi.jpg").write_bytes(_FakeFrame.one(80))
    (directory / "heidi.json").write_text(json.dumps(
        {"profile": "Heidi from QA", "meta": {"floor": 2}}))
    (directory / "ivan.jpg").write_bytes(_FakeFrame.one(81))
    (directory / "ivan.txt").write_text("Ivan the intern\n")
    (directory / "judy.jpg").write_bytes(_FakeFrame.one(82))

    result = plugin.dispatch("face_recognition", {
        "action": "register_user_photos", "package": str(directory)})
    profiles = {item["file"]: item["profile"] for item in result["results"]}
    assert profiles["heidi.jpg"] == "Heidi from QA"
    assert profiles["ivan.jpg"] == "Ivan the intern"
    assert profiles["judy.jpg"] == "judy"          # filename stem


def test_batch_accepts_a_zip_and_refuses_traversal(plugin, tmp_path):
    archive = tmp_path / "people.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("kate.jpg", _FakeFrame.one(90))
        handle.writestr("../escape.jpg", _FakeFrame.one(91))
        handle.writestr("/abs.jpg", _FakeFrame.one(92))
        handle.writestr("manifest.json", json.dumps({"kate.jpg": "Kate"}))

    result = plugin.dispatch("face_recognition", {
        "action": "register_user_photos", "package": str(archive)})
    assert result["registered"] == 1
    assert [item["file"] for item in result["results"]] == ["kate.jpg"]
    assert not (tmp_path / "escape.jpg").exists()


def test_batch_rejects_an_empty_or_oversized_package(plugin, tmp_path, monkeypatch):
    empty = tmp_path / "empty-pack"
    empty.mkdir()
    result = plugin.dispatch("face_recognition", {
        "action": "register_user_photos", "package": str(empty)})
    assert result["reason"] == face_plugin.REASON_BAD_INPUT
    assert "no images" in result["detail"]

    plugin._plugin_cfg["max_batch"] = 1
    package = _make_package(
        tmp_path, {"a.jpg": _FakeFrame.one(1), "b.jpg": _FakeFrame.one(2)})
    result = plugin.dispatch("face_recognition", {
        "action": "register_user_photos", "package": package})
    assert result["reason"] == face_plugin.REASON_BAD_INPUT
    assert "max_batch" in result["detail"]


def test_batch_requires_a_package(plugin):
    result = plugin.dispatch("face_recognition", {"action": "register_user_photos"})
    assert result["reason"] == face_plugin.REASON_BAD_INPUT


# ── roster CRUD ───────────────────────────────────────────────────────────────

def test_roster_crud_through_dispatch(plugin, tmp_path):
    path = _write_photo(tmp_path, "l.jpg", _FakeFrame.one(100))
    created = plugin.dispatch("face_recognition", {
        "action": "register_user_photo", "image_path": path,
        "profile": "Leo", "meta": {"team": "ops", "floor": 3}})
    person_id = created["person_id"]

    listed = plugin.dispatch("face_recognition", {"action": "list_persons"})
    assert listed["ok"] is True and listed["total"] == 1

    filtered = plugin.dispatch("face_recognition", {
        "action": "list_persons", "query": "ops"})
    assert filtered["total"] == 1

    updated = plugin.dispatch("face_recognition", {
        "action": "update_person", "person_id": person_id,
        "profile": "Leo (facilities)", "meta": {"floor": 4},
        "meta_delete": ["team"]})
    assert updated["person"]["profile"] == "Leo (facilities)"
    assert updated["person"]["meta"] == {"floor": 4}

    fetched = plugin.dispatch("face_recognition", {
        "action": "get_person", "person_id": person_id})
    assert fetched["person"]["meta"] == {"floor": 4}

    forgotten = plugin.dispatch("face_recognition", {
        "action": "forget", "person_id": person_id})
    assert forgotten["ok"] is True and forgotten["forgotten"] == 1
    assert plugin.dispatch("face_recognition", {"action": "list_persons"})["total"] == 0


def test_roster_actions_report_missing_people(plugin):
    for action in ("get_person", "update_person", "forget"):
        result = plugin.dispatch("face_recognition", {
            "action": action, "person_id": "p-404", "profile": "x"})
        assert result["ok"] is False
        assert result["reason"] == face_plugin.REASON_BAD_INPUT

    for action in ("get_person", "update_person"):
        with pytest.raises(ValueError):
            plugin.dispatch("face_recognition", {"action": action})


def test_forget_all_unknowns(plugin):
    engine = plugin._require_engine()
    engine.db.add("Mona", [_unit(110)])
    engine.db.enroll_unknown(_unit(111))
    engine.db.enroll_unknown(_unit(112))

    result = plugin.dispatch("face_recognition", {
        "action": "forget", "named": "unknown"})
    assert result["forgotten"] == 2
    stats = plugin._require_engine().db.stats()
    assert (stats["named"], stats["unknown"]) == (1, 0)


def test_update_person_names_an_unknown_keeping_its_id(plugin):
    engine = plugin._require_engine()
    unknown_id = engine.db.enroll_unknown(_unit(120))["id"]
    result = plugin.dispatch("face_recognition", {
        "action": "update_person", "person_id": unknown_id, "profile": "Nina"})
    assert result["person"]["id"] == unknown_id
    assert result["person"]["named"] is True


# ── config ────────────────────────────────────────────────────────────────────

def test_capacity_change_from_the_card_evicts_without_a_reload(plugin):
    engine = plugin._require_engine()
    engine.db.add("Owen", [_unit(130)])
    for seed in range(5):
        engine.db.touch(engine.db.enroll_unknown(_unit(200 + seed))["id"],
                        when=1_000.0 + seed)
    builds = plugin._probe.calls

    result = plugin.dispatch("face_recognition", {
        "action": "config", "unknown_capacity": 2})
    assert result["unknown_evicted"] == 3
    assert result["unknown_capacity"] == 2
    assert plugin._probe.calls == builds, "capacity change must not rebuild the engine"

    stats = plugin._require_engine().db.stats()
    assert (stats["named"], stats["unknown"]) == (1, 2)


def test_threshold_change_does_not_rebuild_but_takes_effect(plugin):
    plugin._require_engine()
    builds = plugin._probe.calls
    plugin.dispatch("face_recognition", {
        "action": "config", "match_threshold": 0.9})
    assert plugin._probe.calls == builds
    assert plugin._plugin_cfg["match_threshold"] == 0.9


def test_model_affecting_change_rebuilds_and_retires_nodes(plugin, tmp_path):
    plugin.dispatch("face_recognition", {"action": "start", "input_topic": "/cam"})
    assert _wait_until(lambda: plugin._nodes, timeout=5.0)
    builds = plugin._probe.calls

    plugin.dispatch("face_recognition", {"action": "config", "device": "gpu"})
    assert plugin._nodes == {}
    assert plugin._engine is None

    plugin.dispatch("face_recognition", {"action": "start", "input_topic": "/cam"})
    assert _wait_until(lambda: plugin._probe.calls > builds, timeout=5.0)


def test_instance_config_rejects_shared_settings(plugin):
    plugin.dispatch("face_recognition", {"action": "start", "input_topic": "/cam"})
    assert _wait_until(lambda: plugin._nodes, timeout=5.0)

    ok = plugin.dispatch("face_recognition", {
        "action": "config", "instance_id": "/cam", "min_interval_ms": 500})
    assert ok["status"] == "configured"
    assert plugin._nodes["/cam"]._min_interval == pytest.approx(0.5)

    with pytest.raises(ValueError):
        plugin.dispatch("face_recognition", {
            "action": "config", "instance_id": "/cam", "match_threshold": 0.4})
