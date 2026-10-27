"""Download progress reaching the card, for every plugin that downloads weights.

ASR was the only plugin whose card said how far along a cold fetch was; the
other six showed one static line for the whole transfer, which is what an
operator reads as a hang. These tests hold up the two halves of fixing that:

- the downloader accepts and forwards the callbacks (no silent dropping — a
  `progress_cb=` that lands in a function which ignores it looks wired and is
  not), and
- each plugin turns them into the string its card actually shows.

No network: the fetch is stubbed everywhere, which also pins the rule that an
already-verified model reports nothing at all.

Run: cd phanthymotus && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
         python3 -m pytest perception/tests/test_model_download_progress.py -q
"""

from __future__ import annotations

import inspect
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import model_downloader as md  # noqa: E402
from utils.model_progress import fetch_status  # noqa: E402


# ── the status line itself ───────────────────────────────────────────────────

def test_progress_renders_percent_and_megabytes():
    lines = []
    progress_cb, _ = fetch_status(lines.append, "sensevoice-small")

    progress_cb(40, 180.4, 449.0)

    assert lines == ["正在下载模型 'sensevoice-small' … 40% (180/449 MB)"]


def test_the_extract_stage_replaces_the_frozen_percentage():
    """A 515 MB tarball still has to unpack after 100%; saying so is the point."""
    lines = []
    _, stage_cb = fetch_status(lines.append, "kokoro-multi")

    stage_cb("extract")

    assert lines == ["正在解压模型 'kokoro-multi' …"]


def test_an_unknown_stage_says_nothing():
    """Better a stale percentage than a line naming a stage no one designed."""
    lines = []
    _, stage_cb = fetch_status(lines.append, "m")

    stage_cb("verify")

    assert lines == []


def test_no_sink_means_no_callbacks_at_all():
    """A caller that reports nothing should hand the downloader nothing, so the
    per-chunk bookkeeping is skipped rather than computed and discarded."""
    assert fetch_status(None, "m") == (None, None)


def test_a_failing_status_sink_cannot_abort_a_download():
    def explode(_text):
        raise RuntimeError("the card went away")

    progress_cb, stage_cb = fetch_status(explode, "m")

    progress_cb(10, 1.0, 10.0)      # must not raise
    stage_cb("extract")             # must not raise


# ── every front door forwards what it is given ───────────────────────────────

# (function, kwargs that reach a bundle/archive fetch). The point is coverage:
# a new ensure_* that forgets progress_cb should fail here, not on a rig.
_BUNDLE_FRONT_DOORS = [
    ("ensure_ocr_model", {"model_dir": "/models/ocr"}),
    ("ensure_face_model", {"model_dir": "/models/face/buffalo_sc"}),
    ("ensure_vop_model", {"model_dir": "/models/vop"}),
    ("ensure_depth_model", {"model_dir": "/models/depth"}),
    ("ensure_soundevent_model", {}),
]


@pytest.mark.parametrize("func_name, kwargs", _BUNDLE_FRONT_DOORS)
def test_bundle_front_doors_forward_progress(func_name, kwargs, monkeypatch):
    seen = {}

    def fake_bundle(name, model_dir, base_url, files, progress_cb=None):
        seen["progress_cb"] = progress_cb
        return {filename: f"{model_dir}/{filename}" for filename in files}

    monkeypatch.setattr(md, "ensure_verified_bundle", fake_bundle)
    monkeypatch.setattr(md, "require_models_subpath", lambda path: path)
    monkeypatch.setattr(md, "select_bundle_family", lambda bundles, family=None:
                        sorted(bundles)[0])

    sentinel = lambda *a: None
    getattr(md, func_name)(progress_cb=sentinel, **kwargs)

    assert seen["progress_cb"] is sentinel


_ARCHIVE_FRONT_DOORS = [
    ("ensure_vits2_model", {"model_dir": "/models/vits2"}),
    ("ensure_thai_tts_model", {"model_dir": "/models/mms-th"}),
    ("ensure_kokoro_model", {"model_dir": "/models/kokoro-multi"}),
]


@pytest.mark.parametrize("func_name, kwargs", _ARCHIVE_FRONT_DOORS)
def test_archive_front_doors_forward_both_callbacks(func_name, kwargs, monkeypatch):
    """Archives need the stage too — their second wait is the unpack."""
    seen = {}

    def fake_archive(name, model_dir, url, entry, progress_cb=None, stage_cb=None):
        seen.update(progress_cb=progress_cb, stage_cb=stage_cb)

    monkeypatch.setattr(md, "ensure_verified_archive", fake_archive)
    monkeypatch.setattr(md, "require_models_subpath", lambda path: path)
    monkeypatch.setattr(md, "select_bundle_family", lambda bundles, family=None:
                        sorted(bundles)[0])

    progress, stage = lambda *a: None, lambda *a: None
    getattr(md, func_name)(progress_cb=progress, stage_cb=stage, **kwargs)

    assert seen["progress_cb"] is progress
    assert seen["stage_cb"] is stage


def test_the_archive_path_announces_the_unpack(tmp_path, monkeypatch):
    """Between the last byte and a usable model there is a whole second wait."""
    stages = []
    payload = b"tarball"
    entry = {"size": len(payload), "sha256": "unchecked"}

    monkeypatch.setattr(md, "_fetch_pinned_file",
                        lambda name, url, dest, meta, **kw:
                            pathlib.Path(dest).write_bytes(payload))
    monkeypatch.setattr(md, "_extract_verified_tar", lambda archive, dest: None)
    monkeypatch.setattr(md, "_merge_tree", lambda src, dst: None)

    md.ensure_verified_archive("m", str(tmp_path), "https://h/m.tar.gz", entry,
                               stage_cb=stages.append)

    assert stages == ["extract"]


def test_a_warm_cache_reports_no_progress_at_all(tmp_path, monkeypatch):
    """A verified model returns without a call, so a warm start never flashes a
    percentage — which is why a caller must still set its own "preparing" line."""
    calls = []
    monkeypatch.setattr(md, "_bundle_matches", lambda model_dir, files: True)

    md.ensure_verified_bundle("m", str(tmp_path), "https://h", {"a.bin": {}},
                              progress_cb=lambda *a: calls.append(a))

    assert calls == []


# ── each plugin turns the callbacks into its own card line ───────────────────

def test_every_downloading_plugin_takes_a_status_sink():
    """The wiring, checked by signature: a plugin builder that cannot be handed
    a sink cannot report progress, and that is the state six of these were in.

    By signature rather than by calling, because constructing any of these loads
    a model — the same reason test_face_plugin checks the proxy this way.
    """
    import plugins.face as face_plugin
    import plugins.ocr as ocr_plugin
    import plugins.soundevent as soundevent_plugin
    import plugins.tts as tts_plugin
    from plugins.face_proxy import FaceServiceProxy

    builders = {
        "face._build_engine": face_plugin._build_engine,
        "ocr._build_ocr_adapter": ocr_plugin._build_ocr_adapter,
        "soundevent._build_model": soundevent_plugin._build_model,
        "tts._build_tts_adapter": tts_plugin._build_tts_adapter,
        "face_proxy.FaceServiceProxy": FaceServiceProxy.__init__,
        "tts.MatchaTTSAdapter": tts_plugin.MatchaTTSAdapter.__init__,
        "tts.MmsThaiTTSAdapter": tts_plugin.MmsThaiTTSAdapter.__init__,
        "tts.KokoroTTSAdapter": tts_plugin.KokoroTTSAdapter.__init__,
    }
    missing = [name for name, func in builders.items()
               if "on_status" not in inspect.signature(func).parameters]

    assert not missing, f"these cannot report download progress: {missing}"


def test_soundevent_card_shows_the_progress_line(monkeypatch):
    """The end-to-end shape for one plugin: downloader line -> card message."""
    import plugins.soundevent as soundevent_plugin

    captured = {}

    def fake_ensure(progress_cb=None):
        captured["progress_cb"] = progress_cb
        progress_cb(60, 2.4, 4.0)
        raise RuntimeError("stop before the tflite interpreter")

    monkeypatch.setattr(md, "ensure_soundevent_model", fake_ensure)
    plugin = soundevent_plugin.SoundEventPlugin({}, _NullExecutor())

    def _on_status(text):
        captured["line"] = text

    with pytest.raises(RuntimeError):
        soundevent_plugin._build_model(on_status=_on_status)

    assert captured["line"] == "正在下载模型 'yamnet' … 60% (2/4 MB)"
    # And the card falls back to its static sentence when nothing is downloading.
    assert plugin._desc_locked("loading") == "Loading SoundEvent model..."


class _NullExecutor:
    def add_node(self, node):
        pass

    def remove_node(self, node):
        pass
