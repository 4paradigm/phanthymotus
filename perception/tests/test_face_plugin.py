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
    """Stands in for `FaceServiceProxy`: one call, results already embedded.

    The plugin no longer calls decode/detect/prepare/embed — that pipeline runs in the
    ORT worker child (`plugins/face_service.py`), because a standalone ONNX Runtime
    session cannot share a process with sherpa-onnx's. So the fake implements the same
    one method the real proxy does, and applies the quality gate itself, which is where
    the gate now lives.

    Identity used to travel through the aligned crop so that `embed` could read it back.
    With one call there is nothing to smuggle it through, so it is read straight from
    the frame spec — simpler, and it removes the int32-vs-uint8 trap that cost a
    debugging round when a test used identity 300.
    """

    def __init__(self, delay: float = 0.0, frame_shape=(480, 640)):
        self.delay = delay
        self.closed = False
        self.frame_shape = frame_shape
        self.seen: list[bytes] = []
        self.embed_calls = 0
        self.calls: list[dict] = []

    device = "cpu"
    providers = ["CPUExecutionProvider"]

    def recognise(self, image_bytes: bytes, max_faces: int = 0,
                  det_thresh: float = 0.5, min_face_px: int = 64,
                  blur_min: float = 60.0, want_aligned: bool = False,
                  embed_all: bool = False):
        self.seen.append(image_bytes)
        self.calls.append({"max_faces": max_faces, "det_thresh": det_thresh,
                           "min_face_px": min_face_px, "blur_min": blur_min,
                           "want_aligned": want_aligned, "embed_all": embed_all})
        if self.delay:
            time.sleep(self.delay)
        if image_bytes == b"corrupt":
            return None, []

        spec = image_bytes.decode()
        height, width = self.frame_shape
        if not spec:
            return (height, width), []

        faces = []
        segments = spec.split("|")
        # Lay the boxes out left to right; the first one is centred, so it wins the
        # centre-weighted dominance comparison when sizes are equal.
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
        if max_faces:
            faces = faces[:max_faces]

        # The gate, where the real service applies it: an unusable face gets no
        # embedding, and `embedding is None` is how the plugin reads that verdict.
        for face in faces:
            usable = (face.det_score >= det_thresh
                      and face.min_side >= min_face_px
                      and face.blur >= blur_min)
            if usable or embed_all:
                self.embed_calls += 1
                face.embedding = _unit(face.identity)
            if want_aligned:
                face.aligned = np.zeros((112, 112, 3), dtype=np.uint8)
        return (height, width), faces

    def close(self):
        self.closed = True


# Kept as an alias: the identity-through-the-crop trick it existed for is gone, but
# several tests name it and the indirection is free.
_SpecAnalyzer = _FakeAnalyzer


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
        "detect_fps": 0,
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
    # embed_all, because `select_subject` is what these tests exercise and it has to see
    # every candidate — including ones the per-frame gate would skip.
    analyzer = _FakeAnalyzer()
    shape, faces = analyzer.recognise(_FakeFrame.build(*specs), embed_all=True)
    return faces, shape


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
                "action": "config", "detect_fps": 100})
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


def test_a_registered_person_is_reported_with_their_name(plugin):
    engine = plugin._require_engine()
    engine.db.add("运营部小王", [_unit(3)], profile={"team": "ops"})

    _node, payloads = _run_one_frame(plugin, _FakeFrame.one(3))
    face = payloads[0]["faces"][0]
    assert face["person_id"] == "p-1"
    assert face["name"] == "运营部小王"
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


# ── register_by_photo ───────────────────────────────────────────────────────

def _write_photo(tmp_path, name: str, frame: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(frame)
    return str(path)


def test_register_from_a_photo_path(plugin, tmp_path):
    path = _write_photo(tmp_path, "alice.jpg", _FakeFrame.one(11))
    result = plugin.dispatch("face_recognition", {
        "action": "register_by_photo", "image_path": path,
        "name": "Alice", "profile": {"team": "ops"},
    })
    assert result["ok"] is True
    assert result["person_id"] == "p-1"
    assert result["name"] == "Alice"
    assert result["profile"] == {"team": "ops"}

    engine = plugin._require_engine()
    assert engine.db.match(_unit(11), 0.35)[0] == "p-1"


def test_base64_input_is_refused_with_a_pointer_to_what_works(plugin):
    """Removed because an LLM failed on it twice in production: a 43 800-char
    string does not survive being carried through a model's context, and what
    arrived was truncated. Refusing loudly beats decoding garbage."""
    import base64
    encoded = base64.b64encode(_FakeFrame.one(12)).decode()
    result = plugin.dispatch("face_recognition", {
        "action": "register_by_photo", "image_b64": encoded, "name": "Bob"})
    assert result["ok"] is False
    assert result["reason"] == face_plugin.REASON_BAD_INPUT
    assert "file/upload" in result["detail"]
    assert "register_by_url" in result["detail"]
    assert plugin._require_engine().db.stats()["persons"] == 0


def test_image_path_is_a_file_picker_routed_through_the_upload_proxy():
    """`format: file` makes the canvas render a picker; `uploadTo: mcp` sends it
    to /api/mcp/<id>/file/upload, which streams the bytes to *this* service and
    returns a path this container can open. Without uploadTo the upload would
    land in agent-core, which perception cannot see — the original bug."""
    spec = face_plugin.TOOLS[0]["inputSchema"]["properties"]["image_path"]
    assert spec["format"] == "file"
    assert spec["accept"] == "image/*"
    assert spec["uploadTo"] == "mcp"
    # No shared mount is involved, so the roots stay as they were.
    assert "/uploads" not in face_plugin.DEFAULT_IMAGE_ROOTS


def test_no_base64_anywhere_on_the_card():
    schema = face_plugin.TOOLS[0]["inputSchema"]
    assert "image_b64" not in schema["properties"]
    for action, spec in schema["x-action-params"].items():
        assert "image_b64" not in spec["params"], action


def test_a_path_outside_the_roots_says_how_to_hand_the_file_over(plugin):
    """The first real failure was image_path=/work/daiwen.jpg — a real file in
    agent-core, invisible here. Told only "cannot read", the LLM retried with
    another invisible path, so the error has to name the mechanism that works."""
    result = plugin.dispatch("face_recognition", {
        "action": "register_by_photo", "image_path": "/etc/hostname"})
    assert result["ok"] is False
    assert result["reason"] == face_plugin.REASON_BAD_INPUT
    assert "file/upload" in result["detail"]
    assert "register_by_url" in result["detail"]


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
            "action": "register_by_photo", "image_path": path, "profile": name})
        assert result["ok"] is False, name
        assert result["reason"] == expected, name
        assert result["detail"], name
    assert plugin._require_engine().db.stats()["persons"] == 0


def test_register_rejects_missing_and_out_of_root_paths(plugin, tmp_path):
    missing = plugin.dispatch("face_recognition", {
        "action": "register_by_photo", "image_path": str(tmp_path / "nope.jpg")})
    assert missing["reason"] == face_plugin.REASON_BAD_INPUT

    escaped = plugin.dispatch("face_recognition", {
        "action": "register_by_photo", "image_path": "/etc/hostname"})
    assert escaped["reason"] == face_plugin.REASON_BAD_INPUT
    assert "must be under" in escaped["detail"]

    nothing = plugin.dispatch("face_recognition", {
        "action": "register_by_photo", "name": "x"})
    assert nothing["reason"] == face_plugin.REASON_BAD_INPUT


def test_registering_a_known_face_merges_instead_of_duplicating(plugin, tmp_path):
    path = _write_photo(tmp_path, "a.jpg", _FakeFrame.one(20))
    first = plugin.dispatch("face_recognition", {
        "action": "register_by_photo", "image_path": path, "name": "Alice"})
    second = plugin.dispatch("face_recognition", {
        "action": "register_by_photo", "image_path": path, "name": "Alice"})

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
        "action": "register_by_photo", "image_path": path, "name": "Dave"})

    assert result["person_id"] == unknown_id
    assert result["promoted"] is True
    record = plugin._require_engine().db.get_person(unknown_id)
    assert record["named"] is True and record["name"] == "Dave"


# ── register_by_stream ───────────────────────────────────────────────────

def test_register_from_the_stream_uses_the_whole_window(plugin):
    node, _payloads = _run_one_frame(plugin, _FakeFrame.one(30))
    for _ in range(4):
        node._image_cb(_FakeCompressedImage(_FakeFrame.one(30)))

    result = plugin.dispatch("face_recognition", {
        "action": "register_by_stream", "name": "Erin"})
    assert result["ok"] is True
    assert result["frames_examined"] >= 2
    assert result["frames_used"] >= 2
    assert result["window_s"] == pytest.approx(3.0)


def test_register_from_the_stream_without_an_instance(plugin):
    result = plugin.dispatch("face_recognition", {
        "action": "register_by_stream", "name": "Nobody"})
    assert result["ok"] is False
    assert result["reason"] == face_plugin.REASON_NO_FRAMES


def test_register_from_the_stream_reports_a_crowd(plugin):
    node, _ = _run_one_frame(
        plugin, _FakeFrame.build((1, 120, 500.0), (2, 119, 500.0)))
    for _ in range(3):
        node._image_cb(_FakeCompressedImage(
            _FakeFrame.build((1, 120, 500.0), (2, 119, 500.0))))

    result = plugin.dispatch("face_recognition", {
        "action": "register_by_stream", "name": "Someone"})
    assert result["ok"] is False
    assert result["reason"] == face_plugin.REASON_AMBIGUOUS
    assert result["frame_reasons"][face_plugin.REASON_AMBIGUOUS] >= 1


def test_register_from_the_stream_refuses_an_unstable_subject(plugin):
    """Two people taking turns must not be enrolled as one identity."""
    node, _ = _run_one_frame(plugin, _FakeFrame.one(40))
    for identity in (41, 42, 43, 44):
        node._image_cb(_FakeCompressedImage(_FakeFrame.one(identity)))

    result = plugin.dispatch("face_recognition", {
        "action": "register_by_stream", "name": "Whoever"})
    assert result["ok"] is False
    assert result["reason"] == face_plugin.REASON_AMBIGUOUS
    assert "stable subject" in result["detail"]


def test_register_stream_window_is_clamped_to_the_config(plugin):
    node, _ = _run_one_frame(plugin, _FakeFrame.one(45))
    node._image_cb(_FakeCompressedImage(_FakeFrame.one(45)))
    result = plugin.dispatch("face_recognition", {
        "action": "register_by_stream", "name": "Frank", "window_s": 999})
    assert result["window_s"] == pytest.approx(3.0)


def test_register_stream_needs_an_instance_id_when_several_run(plugin):
    for index in (1, 2):
        plugin.dispatch("face_recognition", {
            "action": "start", "input_topic": f"/cam{index}",
            "instance_id": f"i{index}"})
    assert _wait_until(lambda: len(plugin._nodes) == 2, timeout=5.0)
    result = plugin.dispatch("face_recognition", {
        "action": "register_by_stream", "name": "x"})
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


# ── register_by_corpus (batch) ──────────────────────────────────────────────

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
        "action": "register_by_corpus", "package": package})

    assert result["ok"] is True
    assert result["total"] == 4
    assert result["registered"] == 1
    assert result["failed"] == 3

    by_file = {item["file"]: item for item in result["results"]}
    assert by_file["alice.jpg"]["ok"] is True
    assert by_file["alice.jpg"]["name"] == "Alice from ops"
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
             "profile": {"badge": "G1"}},
            {"file": "a2.jpg", "profile": "Grace", "person": "grace"},
        ],
    )
    result = plugin.dispatch("face_recognition", {
        "action": "register_by_corpus", "package": package})
    assert result["registered"] == 2
    ids = {item["person_id"] for item in result["results"]}
    assert len(ids) == 1, "photos sharing a person key must become one identity"

    engine = plugin._require_engine()
    person = engine.db.get_person(ids.pop())
    assert person["samples"] == 2
    assert person["profile"] == {"badge": "G1"}


def test_batch_falls_back_to_sidecars_then_the_filename(plugin, tmp_path):
    directory = tmp_path / "pack2"
    directory.mkdir()
    (directory / "heidi.jpg").write_bytes(_FakeFrame.one(80))
    (directory / "heidi.json").write_text(json.dumps(
        {"name": "Heidi from QA", "profile": {"floor": 2}}))
    (directory / "ivan.jpg").write_bytes(_FakeFrame.one(81))
    (directory / "ivan.txt").write_text("Ivan the intern\n")
    (directory / "judy.jpg").write_bytes(_FakeFrame.one(82))

    result = plugin.dispatch("face_recognition", {
        "action": "register_by_corpus", "package": str(directory)})
    names = {item["file"]: item["name"] for item in result["results"]}
    assert names["heidi.jpg"] == "Heidi from QA"
    assert names["ivan.jpg"] == "Ivan the intern"
    assert names["judy.jpg"] == "judy"          # filename stem


def test_batch_accepts_a_zip_and_refuses_traversal(plugin, tmp_path):
    archive = tmp_path / "people.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("kate.jpg", _FakeFrame.one(90))
        handle.writestr("../escape.jpg", _FakeFrame.one(91))
        handle.writestr("/abs.jpg", _FakeFrame.one(92))
        handle.writestr("manifest.json", json.dumps({"kate.jpg": "Kate"}))

    result = plugin.dispatch("face_recognition", {
        "action": "register_by_corpus", "package": str(archive)})
    assert result["registered"] == 1
    assert [item["file"] for item in result["results"]] == ["kate.jpg"]
    assert not (tmp_path / "escape.jpg").exists()


def test_batch_rejects_an_empty_or_oversized_package(plugin, tmp_path, monkeypatch):
    empty = tmp_path / "empty-pack"
    empty.mkdir()
    result = plugin.dispatch("face_recognition", {
        "action": "register_by_corpus", "package": str(empty)})
    assert result["reason"] == face_plugin.REASON_BAD_INPUT
    assert "no images" in result["detail"]

    plugin._plugin_cfg["max_batch"] = 1
    package = _make_package(
        tmp_path, {"a.jpg": _FakeFrame.one(1), "b.jpg": _FakeFrame.one(2)})
    result = plugin.dispatch("face_recognition", {
        "action": "register_by_corpus", "package": package})
    assert result["reason"] == face_plugin.REASON_BAD_INPUT
    assert "max_batch" in result["detail"]


def test_batch_requires_a_package(plugin):
    result = plugin.dispatch("face_recognition", {"action": "register_by_corpus"})
    assert result["reason"] == face_plugin.REASON_BAD_INPUT


# ── roster CRUD ───────────────────────────────────────────────────────────────

def test_roster_crud_through_dispatch(plugin, tmp_path):
    path = _write_photo(tmp_path, "l.jpg", _FakeFrame.one(100))
    created = plugin.dispatch("face_recognition", {
        "action": "register_by_photo", "image_path": path,
        "name": "Leo", "profile": {"team": "ops", "floor": 3}})
    person_id = created["person_id"]

    listed = plugin.dispatch("face_recognition", {"action": "list_persons"})
    assert listed["ok"] is True and listed["total"] == 1

    filtered = plugin.dispatch("face_recognition", {
        "action": "list_persons", "query": "ops"})
    assert filtered["total"] == 1

    updated = plugin.dispatch("face_recognition", {
        "action": "update_person", "person_id": person_id,
        "name": "Leo (facilities)", "profile": {"floor": 4},
        "profile_delete": ["team"]})
    assert updated["person"]["name"] == "Leo (facilities)"
    assert updated["person"]["profile"] == {"floor": 4}

    fetched = plugin.dispatch("face_recognition", {
        "action": "get_person", "person_id": person_id})
    assert fetched["person"]["profile"] == {"floor": 4}

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
        "action": "update_person", "person_id": unknown_id, "name": "Nina"})
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
        "action": "config", "instance_id": "/cam", "detect_fps": 2})
    assert ok["status"] == "configured"
    assert plugin._nodes["/cam"]._detect_interval == pytest.approx(0.5)

    with pytest.raises(ValueError):
        plugin.dispatch("face_recognition", {
            "action": "config", "instance_id": "/cam", "match_threshold": 0.4})


# ── detect_fps (检测频率) ──────────────────────────────────────────────────────

def test_detect_fps_maps_to_an_interval_and_accepts_decimals():
    assert face_plugin.detect_interval({}) == pytest.approx(1.0)        # default
    assert face_plugin.detect_interval({"detect_fps": 2}) == pytest.approx(0.5)
    assert face_plugin.detect_interval({"detect_fps": 0.5}) == pytest.approx(2.0)
    assert face_plugin.detect_interval({"detect_fps": 0.2}) == pytest.approx(5.0)
    assert face_plugin.detect_interval({"detect_fps": 10}) == pytest.approx(0.1)


def test_detect_fps_zero_means_every_frame():
    assert face_plugin.detect_interval({"detect_fps": 0}) == 0.0
    assert face_plugin.detect_interval({"detect_fps": -3}) == 0.0


def test_legacy_min_interval_ms_still_honoured():
    """A canvas saved by the pre-detect_fps build must keep working."""
    assert face_plugin.detect_interval({"min_interval_ms": 500}) == pytest.approx(0.5)
    # detect_fps wins when both are present.
    assert face_plugin.detect_interval(
        {"detect_fps": 4, "min_interval_ms": 500}) == pytest.approx(0.25)


def test_default_config_detects_once_per_second(monkeypatch, tmp_path):
    """The shipped default, end to end through the node."""
    probe = _EngineProbe(tmp_path)
    monkeypatch.setattr(face_plugin, "_build_engine", probe)
    cfg = _base_cfg(tmp_path)
    cfg.pop("detect_fps")                     # fall back to the default
    plugin = face_plugin.FaceRecognitionPlugin(cfg, _FakeExecutor())
    try:
        plugin.dispatch("face_recognition", {"action": "start", "input_topic": "/cam"})
        assert _wait_until(lambda: plugin._nodes.get("/cam"), timeout=5.0)
        assert plugin._nodes["/cam"]._detect_interval == pytest.approx(1.0)
    finally:
        plugin.dispatch("face_recognition", {"action": "stop"})


def test_detect_fps_is_declared_on_the_card_as_a_decimal_number():
    schema = face_plugin.TOOLS[0]["configSchema"]["properties"]["detect_fps"]
    assert schema["type"] == "number"          # not integer — decimals required
    assert schema["default"] == 1.0
    assert schema["minimum"] == 0
    assert schema.get("scope") == "instance"


def test_device_is_declared_on_the_card_with_auto_default():
    schema = face_plugin.TOOLS[0]["configSchema"]["properties"]["device"]
    assert schema["default"] == "auto"
    assert set(schema["enum"]) == {"auto", "cpu", "gpu"}


# ── card surface invariants ──────────────────────────────────────────────────

def test_no_image_url_input():
    """A URL fetch would give an unauthenticated LAN caller a request-forging
    primitive inside the robot's network, for no benefit over path/base64."""
    schema = face_plugin.TOOLS[0]["inputSchema"]
    assert "image_url" not in schema["properties"]
    for spec in schema["x-action-params"].values():
        assert "image_url" not in spec["params"]


def test_register_actions_take_no_person_id():
    """Which identity a photo belongs to is decided by matching, not by the
    caller — otherwise there are two ways to say it and they can disagree."""
    params = face_plugin.TOOLS[0]["inputSchema"]["x-action-params"]
    assert "person_id" not in params["register_by_photo"]["params"]
    assert "person_id" not in params["register_by_stream"]["params"]
    # It remains an input where it identifies an existing record.
    for action in ("get_person", "update_person", "forget"):
        assert "person_id" in params[action]["params"]


def test_register_ignores_a_person_id_argument(plugin, tmp_path):
    """Passing it anyway must not bypass matching."""
    engine = plugin._require_engine()
    engine.db.add("Someone else", [_unit(200)])
    path = _write_photo(tmp_path, "x.jpg", _FakeFrame.one(201))

    result = plugin.dispatch("face_recognition", {
        "action": "register_by_photo", "image_path": path,
        "name": "New person", "person_id": "p-1"})
    assert result["ok"] is True
    assert result["person_id"] == "p-2", "person_id must not force the identity"
    assert engine.db.get_person("p-1")["name"] == "Someone else"


def test_published_payload_carries_no_topic_field(plugin):
    """A subscriber already knows the topic it read from, and OCR's payload
    does not carry one either."""
    _node, payloads = _run_one_frame(plugin, _FakeFrame.one(210))
    assert "topic" not in payloads[0]
    assert set(payloads[0]) == {"ts", "count", "faces", "latency_ms"}


# ── 访问记录表 through dispatch ────────────────────────────────────────────────

def test_list_visits_action_is_on_the_card():
    schema = face_plugin.TOOLS[0]["inputSchema"]
    assert "list_visits" in schema["properties"]["action"]["enum"]
    assert set(schema["x-action-params"]["list_visits"]["params"]) == {
        "person_id", "since", "until", "limit", "offset",
    }


def test_recognition_records_visits_not_one_row_per_frame(plugin):
    node, _ = _run_one_frame(plugin, _FakeFrame.one(300))
    # One frame at a time: LatestFrame overwrites, so pushing several at once
    # publishes fewer results than frames — deliberately, that is its job.
    for expected in range(2, 5):
        node._image_cb(_FakeCompressedImage(_FakeFrame.one(300)))
        assert _wait_until(
            lambda: len(node.publishers[0].messages) >= expected, timeout=5.0)

    visits = plugin.dispatch("face_recognition", {"action": "list_visits"})
    assert visits["ok"] is True
    assert visits["total"] == 1, "one visit for one continuous presence"
    entry = visits["visits"][0]
    assert entry["open"] is True
    assert entry["sightings"] >= 4
    assert entry["topic"] == "/cam/rgb"


def test_list_visits_rejects_an_unparseable_time(plugin):
    plugin._require_engine()
    result = plugin.dispatch("face_recognition", {
        "action": "list_visits", "since": "next thursday"})
    assert result["ok"] is False
    assert result["reason"] == face_plugin.REASON_BAD_INPUT


def test_stopping_an_instance_closes_the_open_visit(plugin):
    """Otherwise a visit in progress is lost when the card is stopped."""
    node, _ = _run_one_frame(plugin, _FakeFrame.one(310))
    engine = plugin._require_engine()
    assert engine.db.stats()["open_visits"] == 1

    plugin.dispatch("face_recognition", {"action": "stop"})
    assert engine.db.stats()["open_visits"] == 0
    closed = engine.db.list_visits()
    assert closed["total"] == 1
    assert not closed["visits"][0].get("open")


def test_payload_reports_name_and_profile_but_no_timestamps(plugin):
    engine = plugin._require_engine()
    engine.db.add("小王", [_unit(320)], profile={"gender": "male"})
    _node, payloads = _run_one_frame(plugin, _FakeFrame.one(320))
    face = payloads[0]["faces"][0]
    assert face["name"] == "小王"
    assert face["profile"] == {"gender": "male"}
    # "when was this person around" is a list_visits question.
    assert "registered_at" not in face and "last_seen_at" not in face


# ── recognize_by_photo / recognize_by_stream (read-only) ─────────────────────

def test_recognize_actions_are_on_the_card():
    schema = face_plugin.TOOLS[0]["inputSchema"]
    actions = set(schema["properties"]["action"]["enum"])
    assert {"recognize_by_photo", "recognize_by_url",
            "recognize_by_stream"} <= actions
    assert actions == set(schema["x-action-params"])
    params = schema["x-action-params"]
    assert set(params["recognize_by_photo"]["params"]) == {"image_path"}
    assert set(params["recognize_by_url"]["params"]) == {"url"}
    assert set(params["recognize_by_stream"]["params"]) == {"window_s"}
    # register/recognize are symmetric by suffix.
    assert {"register_by_photo", "register_by_url",
            "register_by_stream", "register_by_corpus"} <= actions


def test_recognize_by_photo_identifies_a_registered_person(plugin, tmp_path):
    engine = plugin._require_engine()
    engine.db.add("小王", [_unit(400)], profile={"gender": "male"})
    path = _write_photo(tmp_path, "who.jpg", _FakeFrame.one(400))

    result = plugin.dispatch("face_recognition", {
        "action": "recognize_by_photo", "image_path": path})
    assert result["ok"] is True and result["count"] == 1
    face = result["faces"][0]
    assert face["person_id"] == "p-1"
    assert face["name"] == "小王"
    assert face["profile"] == {"gender": "male"}
    assert face["score"] > 0.9


def test_recognize_is_read_only(plugin, tmp_path):
    """A query must not enrol the stranger it failed to recognise, nor log a
    visit — otherwise asking "who is this" quietly changes the answer."""
    engine = plugin._require_engine()
    before = engine.db.stats()
    path = _write_photo(tmp_path, "stranger.jpg", _FakeFrame.one(401))

    result = plugin.dispatch("face_recognition", {
        "action": "recognize_by_photo", "image_path": path})
    assert result["ok"] is True
    face = result["faces"][0]
    assert face["person_id"] is None
    assert face["known"] is False
    # The near-miss score is reported so match_threshold can be tuned.
    assert "best_score" in face

    after = engine.db.stats()
    assert after["persons"] == before["persons"] == 0
    assert after["open_visits"] == 0
    assert engine.db.list_visits()["total"] == 0


def test_recognize_reports_every_face_without_the_ambiguity_gate(plugin, tmp_path):
    """subject_dominance exists for enrolment, which must pick one person; a
    query can just report everyone."""
    engine = plugin._require_engine()
    engine.db.add("A", [_unit(410)])
    engine.db.add("B", [_unit(411)])
    path = _write_photo(tmp_path, "two.jpg",
                        _FakeFrame.build((410, 120, 500.0), (411, 118, 500.0)))

    result = plugin.dispatch("face_recognition", {
        "action": "recognize_by_photo", "image_path": path})
    assert result["ok"] is True
    assert result["count"] == 2
    assert {f["name"] for f in result["faces"]} == {"A", "B"}

    # The same photo is refused for registration, for the opposite reason.
    refused = plugin.dispatch("face_recognition", {
        "action": "register_by_photo", "image_path": path, "name": "C"})
    assert refused["reason"] == face_plugin.REASON_AMBIGUOUS


def test_recognize_by_photo_flags_a_low_quality_face(plugin, tmp_path):
    path = _write_photo(tmp_path, "far.jpg", _FakeFrame.one(420, size=30, blur=4.0))
    result = plugin.dispatch("face_recognition", {
        "action": "recognize_by_photo", "image_path": path})
    face = result["faces"][0]
    assert face["quality"] == "low"
    assert face["reason"] == face_plugin.REASON_LOW_QUALITY


def test_recognize_by_photo_on_an_empty_frame(plugin, tmp_path):
    path = _write_photo(tmp_path, "wall.jpg", b"")
    result = plugin.dispatch("face_recognition", {
        "action": "recognize_by_photo", "image_path": path})
    assert result["ok"] is True and result["count"] == 0 and result["faces"] == []


def test_recognize_by_photo_rejects_bad_input(plugin, tmp_path):
    path = _write_photo(tmp_path, "corrupt.jpg", b"corrupt")
    result = plugin.dispatch("face_recognition", {
        "action": "recognize_by_photo", "image_path": path})
    assert result["ok"] is False
    assert result["reason"] == face_plugin.REASON_BAD_INPUT


def test_recognize_by_stream_answers_who_is_there_now(plugin):
    engine = plugin._require_engine()
    engine.db.add("小李", [_unit(430)])
    node, _ = _run_one_frame(plugin, _FakeFrame.one(430))
    node._image_cb(_FakeCompressedImage(_FakeFrame.one(430)))

    result = plugin.dispatch("face_recognition", {
        "action": "recognize_by_stream"})
    assert result["ok"] is True
    assert result["count"] == 1
    assert result["faces"][0]["name"] == "小李"
    assert result["window_s"] == pytest.approx(1.0)
    assert result["frames_examined"] >= 1


def test_recognize_by_stream_reports_each_person_once(plugin):
    """Several frames of the same two people must not become four entries."""
    engine = plugin._require_engine()
    engine.db.add("A", [_unit(440)])
    engine.db.add("B", [_unit(441)])
    spec = _FakeFrame.build((440, 120, 500.0), (441, 118, 500.0))
    node, _ = _run_one_frame(plugin, spec)
    for _ in range(3):
        node._image_cb(_FakeCompressedImage(spec))

    result = plugin.dispatch("face_recognition", {"action": "recognize_by_stream"})
    assert result["count"] == 2
    assert {f["name"] for f in result["faces"]} == {"A", "B"}


def test_recognize_by_stream_without_a_running_instance(plugin):
    plugin._require_engine()
    result = plugin.dispatch("face_recognition", {"action": "recognize_by_stream"})
    assert result["ok"] is False
    assert result["reason"] == face_plugin.REASON_NO_FRAMES


def test_recognize_by_stream_needs_an_instance_id_when_several_run(plugin):
    for index in (1, 2):
        plugin.dispatch("face_recognition", {
            "action": "start", "input_topic": f"/cam{index}",
            "instance_id": f"i{index}"})
    assert _wait_until(lambda: len(plugin._nodes) == 2, timeout=5.0)
    result = plugin.dispatch("face_recognition", {"action": "recognize_by_stream"})
    assert result["reason"] == face_plugin.REASON_BAD_INPUT
    assert "instance_id" in result["detail"]


# ── batch forget through dispatch ────────────────────────────────────────────

def test_forget_accepts_a_list_of_ids(plugin):
    engine = plugin._require_engine()
    ids = [engine.db.add(f"p{s}", [_unit(500 + s)])["id"] for s in range(4)]

    result = plugin.dispatch("face_recognition", {
        "action": "forget", "person_ids": ids[:3]})
    assert result["ok"] is True
    assert result["forgotten"] == 3
    assert result["person_ids"] == ids[:3]
    assert "missing" not in result
    assert engine.db.stats()["persons"] == 1


def test_forget_accepts_a_comma_separated_string(plugin):
    """An LLM (or a text field) sends "p-1, p-2" rather than a JSON array."""
    engine = plugin._require_engine()
    engine.db.add("A", [_unit(510)])
    engine.db.add("B", [_unit(511)])

    result = plugin.dispatch("face_recognition", {
        "action": "forget", "person_ids": "p-1, p-2"})
    assert result["forgotten"] == 2
    assert engine.db.stats()["persons"] == 0


def test_forget_reports_partial_success(plugin):
    engine = plugin._require_engine()
    engine.db.add("A", [_unit(520)])

    result = plugin.dispatch("face_recognition", {
        "action": "forget", "person_ids": ["p-1", "p-404"]})
    assert result["ok"] is True, 'something was deleted, so the call succeeded'
    assert result["forgotten"] == 1
    assert result["missing"] == ["p-404"]
    assert "p-404" in result["detail"]


def test_forget_with_only_unknown_ids_is_an_error(plugin):
    plugin._require_engine()
    result = plugin.dispatch("face_recognition", {
        "action": "forget", "person_ids": ["p-404", "p-405"]})
    assert result["ok"] is False
    assert result["reason"] == face_plugin.REASON_BAD_INPUT
    assert result["missing"] == ["p-404", "p-405"]


def test_forget_still_takes_a_single_id_and_the_unknown_scope(plugin):
    engine = plugin._require_engine()
    engine.db.add("A", [_unit(530)])
    engine.db.enroll_unknown(_unit(531))
    engine.db.enroll_unknown(_unit(532))

    single = plugin.dispatch("face_recognition", {
        "action": "forget", "person_id": "p-1"})
    assert single["forgotten"] == 1

    scoped = plugin.dispatch("face_recognition", {
        "action": "forget", "named": "unknown"})
    assert scoped["forgotten"] == 2 and scoped["scope"] == "unknown"


def test_forget_requires_something_to_delete(plugin):
    plugin._require_engine()
    with pytest.raises(ValueError):
        plugin.dispatch("face_recognition", {"action": "forget"})


def test_person_ids_is_declared_with_an_example_for_the_llm():
    """The accepted shapes have to be readable off the schema alone."""
    schema = face_plugin.TOOLS[0]["inputSchema"]
    spec = schema["properties"]["person_ids"]
    assert spec["type"] == "array"
    assert 'p-1' in spec["description"]          # a concrete example
    assert 'missing' in spec["description"]      # and the return shape
    assert "person_ids" in schema["x-action-params"]["forget"]["params"]


# ── register_by_url ──────────────────────────────────────────────────────────

def test_register_by_url_fetches_and_registers(plugin, monkeypatch):
    fetched = {}

    def fake_fetch(url, max_bytes):
        fetched['url'] = url
        return _FakeFrame.one(600)

    monkeypatch.setattr(face_plugin, '_fetch_url', fake_fetch)
    result = plugin.dispatch("face_recognition", {
        "action": "register_by_url",
        "url": "https://example.com/alice.jpg", "name": "Alice"})

    assert result["ok"] is True
    assert result["name"] == "Alice"
    assert fetched['url'] == "https://example.com/alice.jpg"
    assert result["source"] == "https://example.com/alice.jpg"


def test_register_by_url_surfaces_a_fetch_failure(plugin, monkeypatch):
    def fake_fetch(url, max_bytes):
        raise face_plugin._BadInput(f"cannot fetch {url!r}: timed out", url)

    monkeypatch.setattr(face_plugin, '_fetch_url', fake_fetch)
    result = plugin.dispatch("face_recognition", {
        "action": "register_by_url", "url": "https://example.com/x.jpg"})
    assert result["ok"] is False
    assert result["reason"] == face_plugin.REASON_BAD_INPUT
    assert "timed out" in result["detail"]


def test_recognize_by_url_is_its_own_action(plugin, monkeypatch):
    engine = plugin._require_engine()
    engine.db.add("Bob", [_unit(610)])
    monkeypatch.setattr(face_plugin, '_fetch_url',
                        lambda url, max_bytes: _FakeFrame.one(610))

    result = plugin.dispatch("face_recognition", {
        "action": "recognize_by_url", "url": "https://example.com/b.jpg"})
    assert result["ok"] is True
    assert result["faces"][0]["name"] == "Bob"


def test_register_by_url_is_on_the_card_as_its_own_action():
    """A URL fetch is a named capability, not a parameter smuggled into the
    photo path — so it is visible on the card."""
    schema = face_plugin.TOOLS[0]["inputSchema"]
    assert "register_by_url" in schema["properties"]["action"]["enum"]
    assert set(schema["x-action-params"]["register_by_url"]["params"]) == {
        "url", "name", "profile"}


# ── size / format handling is local, not a rejection ─────────────────────────

def test_decode_options_come_from_config():
    cfg = {"max_image_side": 1024, "max_image_pixels": 1_000_000}
    assert face_plugin._decode_options(cfg) == {
        "max_side": 1024, "max_pixels": 1_000_000}
    defaults = face_plugin._decode_options({})
    assert defaults["max_side"] == 2048 and defaults["max_pixels"] == 60_000_000


def test_every_input_path_goes_through_the_one_recognise_call(plugin, tmp_path):
    """Resize/convert must apply to register and recognize alike, by photo, url,
    corpus and stream.

    This used to spy on `decode_image` and assert every path passed the same options.
    It cannot any more, and that is the improvement: decoding happens in the ORT worker
    child, which is built **once** with the configured limits
    (`plugins/face_proxy.py`), so no path can pass its own. What is left to check is
    that every path reaches the single call rather than growing its own pipeline.
    """
    engine = plugin._require_engine()
    before = len(engine.analyzer.calls)

    path = _write_photo(tmp_path, "a.jpg", _FakeFrame.one(700))
    plugin.dispatch("face_recognition", {
        "action": "register_by_photo", "image_path": path, "name": "A"})
    plugin.dispatch("face_recognition", {
        "action": "recognize_by_photo", "image_path": path})
    package = _make_package(tmp_path, {"b.jpg": _FakeFrame.one(701)})
    plugin.dispatch("face_recognition", {
        "action": "register_by_corpus", "package": package})

    calls = engine.analyzer.calls[before:]
    assert len(calls) >= 3, calls
    # Registration needs an embedding for whichever face select_subject picks, even a
    # marginal one; the continuous path must not pay for that.
    assert any(c["embed_all"] for c in calls), calls
    assert not all(c["embed_all"] for c in calls), calls


def test_the_decode_limits_reach_the_service(monkeypatch, tmp_path):
    """The limits are constructor state on the proxy now, so assert they get there.

    max_side bounds the letterbox downscale and max_pixels is the decompression-bomb
    guard; a path that silently defaulted them would be a real hole.
    """
    seen = {}

    class _Probe(_FakeAnalyzer):
        def __init__(self, **kwargs):
            seen.update(kwargs)
            super().__init__()

    monkeypatch.setattr(face_plugin, "FaceServiceProxy", _Probe)
    face_plugin._build_engine({"model_dir": str(tmp_path),
                               "db_dir": str(tmp_path / "db"),
                               "max_image_side": 1234,
                               "max_image_pixels": 5_000_000})
    assert seen["max_image_side"] == 1234
    assert seen["max_image_pixels"] == 5_000_000


def test_the_byte_cap_is_a_transfer_guard_not_a_photo_limit():
    """16 MB used to reject real phone photos; oversized images are now
    downscaled locally instead."""
    assert face_plugin.DEFAULT_MAX_IMAGE_BYTES == 64 * 1024 * 1024


def test_corpus_finds_the_formats_both_decoders_read():
    for suffix in (".jpg", ".png", ".webp", ".tif", ".tiff", ".gif", ".bmp"):
        assert suffix in face_plugin._IMAGE_SUFFIXES


# ── the wiring nothing else checks ────────────────────────────────────────────

def test_analyzer_options_are_all_accepted_by_the_proxy():
    """`_analyzer_options` builds the kwargs; the proxy has to accept every one.

    This shipped broken: adding `model` to the options without adding it to
    `FaceServiceProxy.__init__` gave `TypeError: __init__() got an unexpected keyword
    argument 'model'` at card start, on a robot. Nothing caught it, because every test
    either replaces `_build_engine` wholesale or fakes the analyzer with a signature
    that swallows anything — so the one place the real kwargs meet the real signature
    was never exercised.

    Checked by signature rather than by calling it, because constructing the real proxy
    spawns a child and loads two models.
    """
    import inspect

    from plugins.face_proxy import FaceServiceProxy

    produced = set(face_plugin._analyzer_options({}))
    produced |= {"max_image_side", "max_image_pixels"}      # added by _build_engine
    accepted = set(inspect.signature(FaceServiceProxy.__init__).parameters) - {"self"}
    assert produced <= accepted, (
        f"_build_engine would pass {sorted(produced - accepted)}, which "
        f"FaceServiceProxy.__init__ does not take")


def test_the_proxy_forwards_everything_it_takes_to_the_service():
    """And the service has to accept what the proxy sends, for the same reason.

    One hop further along the same chain: proxy -> ort_worker.service ->
    plugins.face_service.build -> FaceService.__init__ -> FaceAnalyzer. A parameter
    that stops halfway is a card that will not start.
    """
    import inspect

    from plugins.face_proxy import FaceServiceProxy
    from plugins.face_service import FaceService

    proxy_takes = set(inspect.signature(FaceServiceProxy.__init__).parameters) - {"self"}
    service_takes = set(inspect.signature(FaceService.__init__).parameters) - {"self"}
    # `device` is resolved to `providers` in the proxy and does not travel as-is.
    forwarded = proxy_takes - {"device"}
    assert forwarded <= service_takes, (
        f"the proxy would forward {sorted(forwarded - service_takes)}, which "
        f"FaceService.__init__ does not take")


def test_the_service_passes_the_model_on_to_the_analyzer():
    """The last hop. `model` selects the weights; silently dropping it would run
    buffalo_sc while the card said something else."""
    import inspect

    from plugins.face_runtime import FaceAnalyzer
    from plugins.face_service import FaceService

    assert "model" in inspect.signature(FaceService.__init__).parameters
    assert "model" in inspect.signature(FaceAnalyzer.__init__).parameters
