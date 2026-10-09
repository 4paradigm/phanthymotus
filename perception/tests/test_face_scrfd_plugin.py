"""Plugin-level contracts for face_scrfd that need no model, no cv2 pixels
and no network: local-path confinement, the honest config lifecycle, and the
one detector fallback. (Reviewer suggestions, #284.)

Run: cd phanthymotus && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
         python3 -m pytest perception/tests/test_face_scrfd_plugin.py -q
"""

from __future__ import annotations

import pathlib
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import plugins.face_scrfd as face_scrfd  # noqa: E402


class _LoadedModel:
    """Presence check only — these tests never run inference through it."""


def _stub_plugin(**fields):
    """A plugin without __init__: exactly the state each branch reads."""
    plugin = object.__new__(face_scrfd.FaceRecognitionPlugin)
    plugin._model = None
    plugin._model_loading = False
    plugin._model_load_error = None
    plugin._model_lock = threading.Lock()
    plugin._model_fetch_status = None
    plugin._input_cfg = {}
    plugin._model_name = "edgeface_base.int8"
    plugin._detector = "scrfd_2.5g"
    plugin._device = "cpu"
    plugin._detector_device = None
    plugin._recognizer_device = None
    plugin._model_dir = "/models/face"
    plugin._face_db_dir = "/models/face_db"
    plugin._similarity_threshold = 0.39
    plugin._confidence = 0.5
    plugin._fps = 3
    plugin._pending_starts = []
    for key, value in fields.items():
        setattr(plugin, key, value)
    return plugin


# ── local-path confinement (plugins/image_input.py rules) ────────────────────

def test_photo_and_corpus_paths_stay_inside_configured_roots():
    """The MCP server is unauthenticated and runs as root: a caller-supplied
    local path is confined to the roots BEFORE anything is opened or decoded,
    so it cannot probe the filesystem through the image decoder."""
    plugin = _stub_plugin(_model=_LoadedModel())

    photo = plugin.dispatch("recognize_by_photo", {"image_path": "/etc/passwd"})
    assert photo["ok"] is False and photo["reason"] == "bad_input"
    assert "path must be under" in photo["detail"]

    corpus = plugin.dispatch("register_by_corpus", {"package": "/etc"})
    assert corpus["ok"] is False and corpus["reason"] == "bad_input"
    assert "path must be under" in corpus["detail"]


def test_roots_come_from_the_plugin_config(tmp_path):
    """`image_roots` in the plugin config replaces the defaults; a path under
    a configured root gets past confinement and fails (if at all) later, at
    decoding — proving the rejection above was the boundary, not a ban."""
    (tmp_path / "ok.jpg").write_bytes(b"not-really-an-image")
    plugin = _stub_plugin(_model=_LoadedModel(),
                          _input_cfg={"image_roots": [str(tmp_path)]})

    result = plugin.dispatch("recognize_by_photo",
                             {"image_path": str(tmp_path / "ok.jpg")})

    assert "path must be under" not in result.get("detail", "")


# ── config lifecycle ─────────────────────────────────────────────────────────

def test_model_affecting_config_is_rejected_once_loaded():
    """After the adapter loaded, model/detector/device/dir keys cannot take
    effect — the reply must say restart_required, not acknowledge a
    configuration that was not applied, and it must not touch anything."""
    plugin = _stub_plugin(_model=_LoadedModel())

    result = plugin.dispatch("config", {"model": "edgeface_s_gamma_05",
                                        "similarity_threshold": "0.5"})

    assert result["status"] == "error" and result["reason"] == "restart_required"
    assert "model" in result["detail"] and "similarity" not in result["detail"]
    # Nothing moved at all — the reply describes the whole state honestly.
    assert plugin._model_name == "edgeface_base.int8"
    assert plugin._similarity_threshold == 0.39


def test_live_tunable_keys_still_apply_after_load():
    plugin = _stub_plugin(_model=_LoadedModel())

    result = plugin.dispatch("config", {"similarity_threshold": "0.45", "fps": "5"})

    assert result["status"] == "configured"
    assert plugin._similarity_threshold == 0.45 and plugin._fps == 5


def test_unchanged_model_value_is_not_rejected():
    """Naming the model that is already loaded is a no-op, not a restart
    demand — the gate is about *changes*."""
    plugin = _stub_plugin(_model=_LoadedModel())

    result = plugin.dispatch("config", {"model": "edgeface_base.int8"})

    assert result["status"] == "configured"


def test_model_change_retries_a_failed_load():
    """After a failed load the model is not resident, so naming a new one IS
    honoured — and must actually restart the load, not just store it."""
    plugin = _stub_plugin(_model_load_error="failed to download edgeface_base.int8.onnx")
    retried = []
    plugin._start_model_loading = lambda: retried.append(True)

    result = plugin.dispatch("config", {"model": "edgeface_s_gamma_05"})

    assert result["status"] == "configured"
    assert plugin._model_name == "edgeface_s_gamma_05"
    assert retried == [True]


# ── detector normalisation ───────────────────────────────────────────────────

def test_unknown_detector_falls_back_to_yunet_everywhere():
    """One fallback, shared: the weights fetched and the session built cannot
    disagree about which detector was asked for."""
    assert face_scrfd._detector_key("bogus") == "yunet"
    assert face_scrfd._detector_key(None) == "yunet"
    assert face_scrfd._detector_key("SCRFD_2.5G") == "scrfd_2.5g"
    assert face_scrfd._detector_key("yunet") == "yunet"

    seen = {}

    def fake_bundle(name, model_dir, base_url, files, progress_cb=None):
        seen["files"] = files
        return {filename: f"{model_dir}/{filename}" for filename in files}

    original = face_scrfd.ensure_verified_bundle
    face_scrfd.ensure_verified_bundle = fake_bundle
    try:
        face_scrfd._ensure_weights("edgeface_base.int8", "/tmp/face-test",
                                   detector="bogus")
    finally:
        face_scrfd.ensure_verified_bundle = original

    assert "face_detection_yunet_2023mar.onnx" in seen["files"]
