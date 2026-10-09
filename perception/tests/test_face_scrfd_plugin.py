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

import pytest

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
    plugin._max_batch = 1000
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


def test_http_corpus_package_is_not_confined(monkeypatch):
    """The corpus action accepts the documented HTTP(S) package URL —
    corpus_entries downloads it. Confining that form would realpath the URL
    string and reject it, which is what broke the advertised capability."""
    plugin = _stub_plugin(_model=_LoadedModel())
    seen = {}

    class _EmptyCorpus:
        def __enter__(self):
            return iter(())

        def __exit__(self, *exc):
            return False

    def fake_corpus(package, max_batch=1000):
        seen["package"] = package
        return _EmptyCorpus()

    monkeypatch.setattr(face_scrfd, "corpus_entries", fake_corpus)

    result = plugin.dispatch("register_by_corpus",
                             {"package": "https://example.invalid/corpus.zip"})

    assert seen["package"] == "https://example.invalid/corpus.zip"
    assert "path must be under" not in str(result)
    assert result["ok"] is True and result["total"] == 0  # reached corpus_entries


def test_local_corpus_package_outside_roots_is_still_rejected():
    plugin = _stub_plugin(_model=_LoadedModel())

    result = plugin.dispatch("register_by_corpus", {"package": "/etc"})

    assert result["ok"] is False and result["reason"] == "bad_input"
    assert "path must be under" in result["detail"]


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


def test_model_change_while_loading_is_rejected():
    """The background loader reads the load keys as it goes, so a change made
    mid-flight can be half-applied — the fingerprint step looking for a file
    that was never fetched, or the old model landing under the new config's
    name. Refused while loading, with the reason naming the race."""
    plugin = _stub_plugin(_model_loading=True)

    result = plugin.dispatch("config", {"model": "edgeface_s_gamma_05",
                                        "detector": "scrfd"})

    assert result["status"] == "error" and result["reason"] == "loading_in_progress"
    assert "model" in result["detail"] and "detector" in result["detail"]
    assert plugin._model_name == "edgeface_base.int8"
    assert plugin._detector == "scrfd_2.5g"


def test_live_keys_apply_even_while_loading():
    """Only the keys the loader reads are gated by it."""
    plugin = _stub_plugin(_model_loading=True)

    result = plugin.dispatch("config", {"similarity_threshold": "0.45"})

    assert result["status"] == "configured"
    assert plugin._similarity_threshold == 0.45


# ── weight sources ───────────────────────────────────────────────────────────

def test_several_weight_sources_ride_one_pinned_bundle(monkeypatch):
    """FACE_MODEL_BASE_URL takes a comma-separated list: the downloader probes
    fastest-first, and the pins make a mirror just a faster route to the very
    same bytes. One source stays a plain string (no probing cost)."""
    assert face_scrfd._model_sources("https://a") == "https://a"
    assert face_scrfd._model_sources("https://a, https://b") == ["https://a", "https://b"]
    # An exported-but-empty override means "default", not "no source".
    assert face_scrfd._model_sources("") == face_scrfd.FACE_SCRFD_EDGEFACE_BASE

    seen = {}

    def fake_bundle(name, model_dir, base_url, files, progress_cb=None):
        seen["base_url"] = base_url
        return {filename: f"{model_dir}/{filename}" for filename in files}

    monkeypatch.setattr(face_scrfd, "ensure_verified_bundle", fake_bundle)
    monkeypatch.setattr(face_scrfd, "_MODEL_BASE_URL", "https://cos.invalid, https://mirror.invalid")

    face_scrfd._ensure_weights("edgeface_base.int8", "/tmp/face-test")

    assert seen["base_url"] == ["https://cos.invalid", "https://mirror.invalid"]


# ── detector normalisation ───────────────────────────────────────────────────

def test_unknown_detector_is_rejected_not_silently_replaced():
    """A typo must not quietly run a different detector: unknown values are
    refused with the valid set, while known ones normalise (case/space)."""
    assert face_scrfd._detector_key(None) == "yunet"
    assert face_scrfd._detector_key("SCRFD_2.5G") == "scrfd_2.5g"
    assert face_scrfd._detector_key("yunet") == "yunet"

    with pytest.raises(ValueError, match="unknown detector"):
        face_scrfd._detector_key("bogus")
    with pytest.raises(ValueError, match="unknown detector"):
        face_scrfd._ensure_weights("edgeface_base.int8", "/tmp/face-test",
                                   detector="bogus")


def test_unknown_model_and_device_are_rejected_synchronously():
    """`model` and `device` are validated against the pinned registry and the
    provider table, not stored to fail later on a background thread."""
    assert face_scrfd._validate_model("edgeface_base.int8") == "edgeface_base.int8"
    with pytest.raises(ValueError, match="unknown face model"):
        face_scrfd._validate_model("nope")

    assert face_scrfd._providers_for_device("gpu") == ["CUDAExecutionProvider",
                                                       "CPUExecutionProvider"]
    with pytest.raises(ValueError, match="unknown device"):
        face_scrfd._providers_for_device("tpu")


def test_config_rejects_invalid_values_without_applying_anything():
    plugin = _stub_plugin()          # nothing loaded yet

    result = plugin.dispatch("config", {"model": "nope", "similarity_threshold": "0.5"})

    assert result["status"] == "error" and result["reason"] == "invalid_config"
    assert "unknown face model" in result["detail"]
    # Validated-first, applied-after: the good key did not go through either.
    assert plugin._model_name == "edgeface_base.int8"
    assert plugin._similarity_threshold == 0.39

    for field, value in (("device", "tpu"), ("detector", "mtcnn")):
        result = plugin.dispatch("config", {field: value})
        assert result["reason"] == "invalid_config", result
        assert getattr(plugin, "_" + field) == ("cpu" if field == "device"
                                                else "scrfd_2.5g")


# ── the extraction namespace must keep up with the plugin ────────────────────

def test_extracted_plugin_code_only_uses_names_the_stub_namespace_provides():
    """The pixel-content suites don't import face_scrfd — they AST-extract its
    functions into a stub namespace, so every module-level name those functions
    read has to be injected by load_face(). Twice now a new dependency
    (check_under_roots, then urlsplit) was missing there and the failure only
    appeared in the image, where those suites actually run. This check is
    static and runs on the host too: it fails the moment a global dependency is
    added without its namespace entry.
    """
    import ast
    import builtins

    from test_face_candidates import load_face

    source = (ROOT / "plugins" / "face_scrfd.py").read_text()
    tree = ast.parse(source)
    extracted = [node for node in tree.body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                 and getattr(node, "name", "") != "_FaceNode"]

    def bound_names(node):
        """Every name the node binds anywhere inside itself."""
        names = set()
        for child in ast.walk(node):
            if isinstance(child, ast.arg):
                names.add(child.arg)
            elif isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
                names.add(child.id)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                names.update(alias.asname or alias.name.split(".")[0]
                             for alias in child.names)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(child.name)
            elif isinstance(child, ast.ExceptHandler) and child.name:
                names.add(child.name)
        return names | set(dir(builtins))

    # Knowingly not provided, and why — both are on paths these suites never
    # take, so the gap is inert rather than a name missing at runtime:
    #   _FaceNode          the ROS node class, excluded from the extraction on
    #                      purpose (see load_face) and only touched by
    #                      _start_node, which needs a live rclpy executor.
    #   _ONNX_ARCFACE_REF  reference embedding of stashed w600k/sface work;
    #                      _similarity_transform is exercised with that work.
    inert = {'FaceRecognitionPlugin': {'_FaceNode'},
             '_similarity_transform': {'_ONNX_ARCFACE_REF'}}

    namespace = load_face()
    missing = {}
    for node in extracted:
        bound = bound_names(node)
        used = {child.id for child in ast.walk(node)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)}
        gap = sorted(used - bound - set(namespace) - inert.get(node.name, set()))
        if gap:
            missing[node.name] = gap

    assert not missing, (
        "load_face() must inject these names for the extracted code to run in "
        f"the image: {missing}")
