"""
tests/test_kokoro_tts_engine.py — the kokoro-multi engine's wiring and guards.

Kokoro is the first engine here whose sample rate is not the pipeline's (24 kHz vs
16 kHz) and the first with a *runtime* language selector. Both of those are easy to
wire up in a way that looks fine and is wrong in a way only audible on a robot, so
what is pinned here is the part that can be checked on a host with no model:

  - `tts_language` reaches the resident adapter without rebuilding the session.
    Kokoro reads `lang` out of GenerationConfig.extra on every generate() call, so
    putting it in _session_keys would cost a 310 MB fp32 session rebuild for a
    dropdown change — the same class of bug as "every speak takes 5 s" (PR that
    added _session_keys), and the reason `speed` is excluded there too.
  - The release validation that has to happen *before* OfflineTts is constructed.
    A version >= 2 Kokoro model with both `lexicon` and `lang` empty makes
    sherpa-onnx call SHERPA_ONNX_EXIT(-1) — a process exit that takes ASR, VOP and
    OCR down with it, which main.py's try/except cannot catch.

The adapter's actual synthesis is not covered here; it needs the model. See
tests/test_resample.py for the 24 kHz -> 16 kHz conversion, which is the part of
that path that *is* host-testable.

Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import json

import pytest

from vision_stubs import _FakeExecutor  # noqa: F401

import plugins.tts as tts  # noqa: E402

_REAL_SHERPA_PLUGIN = tts.SherpaOnnxTTSPlugin


class _LangCountingAdapter:
    """Records rebuilds and per-call setter hits, like _CountingAdapter next door."""

    builds = 0

    def __init__(self, cfg):
        type(self).builds += 1
        self.cfg = dict(cfg)
        self.speeds = []
        self.languages = []

    def synthesize(self, text):
        return b"\x00" * 3200

    def synthesize_stream(self, text):
        yield b"\x00" * 3200

    def warmup(self):
        return 3200

    def set_speed(self, speed):
        self.speeds.append(speed)

    def set_language(self, language):
        self.languages.append(language)


@pytest.fixture
def _kokoro(monkeypatch):
    """A real SherpaOnnxTTSPlugin configured as kokoro-multi, over a fake adapter."""
    _LangCountingAdapter.builds = 0
    holder = {}

    def build(cfg):
        holder["adapter"] = _LangCountingAdapter(cfg)
        return holder["adapter"]

    monkeypatch.setattr(tts, "_build_tts_adapter", build)
    plugin = _REAL_SHERPA_PLUGIN(
        {"engine": "kokoro-multi", "device": "gpu", "speed": 1.0,
         "speaker_id": 3, "tts_language": "en-us"}, _FakeExecutor())
    return plugin, holder


# ── the language selector must not rebuild ────────────────────────────────────

def test_language_is_not_a_session_key():
    """The whole design in one assertion.

    `lang` rides in GenerationConfig.extra per generate() call, so it is a scale on
    the resident model. In _session_keys it would tear down and reload a 310 MB
    fp32 CUDA session every time someone changed the dropdown.
    """
    keys = tts._session_keys({
        "speaker_id": 3, "device": "gpu", "model_dir": "/models/kokoro-multi",
        "tts_language": "zh", "speed": 1.0,
    })
    assert "tts_language" not in keys
    assert "speed" not in keys, "speed is per-call for the same reason"
    assert set(keys) == {"speaker_id", "device", "model_dir"}


def test_language_is_still_carried_across_an_engine_switch():
    """Absent from _session_keys, present in SHARED_CONFIG_KEYS — both are needed.

    SHARED_CONFIG_KEYS is derived from configSchema precisely so a new shared field
    cannot be forgotten; forgetting this one would build the new engine with the
    default language and then rebuild on the next config.
    """
    assert "tts_language" in tts.SHARED_CONFIG_KEYS
    assert "tts_engine" not in tts.SHARED_CONFIG_KEYS


def test_changing_language_alone_updates_the_resident_model(_kokoro):
    plugin, holder = _kokoro
    result = plugin.dispatch("tts", {"action": "config", "tts_language": "en-gb"})
    assert result["rebuilt"] is False
    assert _LangCountingAdapter.builds == 1, "a language change rebuilt the session"
    assert holder["adapter"].languages == ["en-gb"], "set_language was not applied"


def test_repeating_the_same_language_still_does_not_rebuild(_kokoro):
    """The dashboard re-applies a card's whole config around every speak."""
    plugin, _ = _kokoro
    same = {"action": "config", "device": "gpu", "speed": 1,
            "speaker_id": 3, "tts_language": "en-us"}
    for _ in range(3):
        assert plugin.dispatch("tts", same)["rebuilt"] is False
    assert _LangCountingAdapter.builds == 1


def test_speed_and_language_can_change_in_one_config(_kokoro):
    """Both are per-call, so neither may shadow the other."""
    plugin, holder = _kokoro
    result = plugin.dispatch("tts", {"action": "config", "speed": 1.4,
                                     "tts_language": "zh"})
    assert result["rebuilt"] is False
    assert holder["adapter"].speeds == [1.4]
    assert holder["adapter"].languages == ["zh"]


def test_a_language_set_earlier_survives_a_later_rebuild(_kokoro):
    """The reason `tts_language` is written back to self._cfg explicitly.

    It is absent from `incoming` by design, and _build_tts_adapter reads it out of
    self._cfg — so without that write-back, a rebuild triggered by some *other* key
    would silently construct the adapter with the default language.
    """
    plugin, holder = _kokoro
    plugin.dispatch("tts", {"action": "config", "tts_language": "zh"})
    assert _LangCountingAdapter.builds == 1

    # speaker_id IS a session key, so this one really does rebuild.
    result = plugin.dispatch("tts", {"action": "config", "speaker_id": 5})
    assert result["rebuilt"] is True
    assert _LangCountingAdapter.builds == 2
    assert holder["adapter"].cfg["tts_language"] == "zh", \
        "the rebuild lost the selected language"
    assert holder["adapter"].cfg["speaker_id"] == 5


def test_every_adapter_accepts_set_language(_kokoro):
    """_config calls it unconditionally, so the ABC must define a no-op.

    Only Kokoro has a language; Matcha, MMS Thai and VITS2 bake theirs into the
    checkpoint. A missing default would make a language field left over in a
    deployment's config.yaml crash the *other* engines on load.
    """
    assert hasattr(tts.TTSAdapter, "set_language")
    assert tts.MatchaTTSAdapter.set_language is tts.TTSAdapter.set_language
    assert tts.MmsThaiTTSAdapter.set_language is tts.TTSAdapter.set_language
    assert tts.KokoroTTSAdapter.set_language is not tts.TTSAdapter.set_language


# ── configSchema ──────────────────────────────────────────────────────────────

def test_the_language_field_is_exposed_and_scoped_to_kokoro():
    props = tts.TOOLS[0]["configSchema"]["properties"]
    field = props["tts_language"]
    assert field["enum"] == list(tts.KokoroTTSAdapter.LANGUAGES)
    assert field["default"] == tts.KokoroTTSAdapter.DEFAULT_LANGUAGE
    assert field["scope"] == "shared"
    # Hidden for every other engine, or the form offers a control that does nothing.
    assert field["x-show-when"] == {"tts_engine": ["kokoro-multi"]}


def test_the_device_field_is_revealed_for_kokoro():
    """Kokoro is an ONNX Runtime engine, so `device` is meaningful for it.

    Unlike Matcha it also selects *different weight files*, which is why the two
    are not interchangeable in the form's help text.
    """
    device = tts.TOOLS[0]["configSchema"]["properties"]["device"]
    assert "kokoro-multi" in device["x-show-when"]["tts_engine"]
    assert "vits2-zh-en" not in device["x-show-when"]["tts_engine"]


def test_kokoro_defaults_to_gpu_and_the_others_to_cpu(monkeypatch):
    """provider_for_device refuses int8 on CUDA, so gpu implies the fp32 archive."""
    seen = {}

    class _Recorder:
        LANGUAGES = tts.KokoroTTSAdapter.LANGUAGES
        DEFAULT_LANGUAGE = tts.KokoroTTSAdapter.DEFAULT_LANGUAGE

        # **kwargs so a new adapter option does not break this test: it is asserting
        # the device default, not the constructor's full signature.
        def __init__(self, model_dir, speaker_id, speed, device, language=None,
                     **kwargs):
            seen.update(model_dir=model_dir, device=device, language=language,
                        **kwargs)

    monkeypatch.setattr(tts, "KokoroTTSAdapter", _Recorder)
    monkeypatch.setattr(tts, "MatchaTTSAdapter",
                        lambda md, sid, sp, dev: seen.update(device=dev) or object())

    tts._build_tts_adapter({"engine": "kokoro-multi"})
    assert seen["device"] == "gpu", "kokoro must default to gpu"
    assert seen["model_dir"] == tts.ENGINE_MODEL_DIRS["kokoro-multi"]

    seen.clear()
    tts._build_tts_adapter({"engine": "matcha-zh-en"})
    assert seen["device"] == "cpu", "only kokoro's default changed"

    # An explicit device still wins over the per-engine default.
    seen.clear()
    tts._build_tts_adapter({"engine": "kokoro-multi", "device": "cpu"})
    assert seen["device"] == "cpu"


def test_config_yaml_and_dashboard_language_keys_are_both_accepted(monkeypatch):
    """config.yaml says `language`; the dashboard sends `tts_language`.

    Same split as `engine` vs `tts_engine`, which the facade already handles both
    ways. A baked config.yaml must not be silently ignored.
    """
    seen = {}

    class _Recorder:
        DEFAULT_LANGUAGE = tts.KokoroTTSAdapter.DEFAULT_LANGUAGE

        def __init__(self, model_dir, speaker_id, speed, device, language=None,
                     **kwargs):
            seen["language"] = language

    monkeypatch.setattr(tts, "KokoroTTSAdapter", _Recorder)

    tts._build_tts_adapter({"engine": "kokoro-multi", "language": "zh"})
    assert seen["language"] == "zh", "config.yaml's `language` was ignored"

    tts._build_tts_adapter({"engine": "kokoro-multi", "tts_language": "en-gb"})
    assert seen["language"] == "en-gb"

    # The dashboard's key wins when both are present: it is the live value.
    tts._build_tts_adapter({"engine": "kokoro-multi",
                            "language": "zh", "tts_language": "en-gb"})
    assert seen["language"] == "en-gb"


# ── language normalisation ────────────────────────────────────────────────────

@pytest.mark.parametrize("given,expected", [
    ("en-us", "en-us"),
    ("EN-US", "en-us"),
    ("en_gb", "en-gb"),
    ("zh", "zh"),
    ("ja", "ja"),
    ("pt-br", "pt-br"),
    # aliases someone would plausibly write
    ("en", "en-us"),
    ("pt", "pt-br"),
    ("cmn", "zh"),
    ("cn", "zh"),
    ("jp", "ja"),
    ("es-419", "es"),
    ("fr_FR", "fr"),
])
def test_known_languages_normalise(given, expected):
    assert tts.KokoroTTSAdapter._normalize_language(given) == expected


@pytest.mark.parametrize("given", ["", None, "klingon", "th", "de"])
def test_an_unusable_language_falls_back_rather_than_raising(given):
    """A stale value in a baked config.yaml should degrade to a working voice.

    Same policy as provider_for_device: the dashboard's enum is what constrains
    input, and refusing here would stop the card loading over a typo. `de` and `th`
    are espeak voices but Kokoro has no voices for them, so they belong here.
    """
    assert (tts.KokoroTTSAdapter._normalize_language(given)
            == tts.KokoroTTSAdapter.DEFAULT_LANGUAGE)


def test_every_language_maps_to_an_espeak_voice_verified_on_device():
    """The map is a closed set of *measured* espeak voice names, not passthrough.

    Two failure modes made this necessary, both silent: `en-gb` is not a voice
    espeak-ng ships (it has en-GB-x-rp) and produced no audio at all, and `fr-FR`
    produced 18280 samples where `fr` produced 63277 — a truncated utterance with
    nothing raised. Passing the configured label straight through would reintroduce
    both.
    """
    adapter = tts.KokoroTTSAdapter
    assert set(adapter.LANGUAGES) == set(adapter.LANGUAGE_VOICES)
    assert adapter.DEFAULT_LANGUAGE in adapter.LANGUAGE_VOICES
    # The two names that are NOT the obvious guess, pinned so a "tidy-up" cannot
    # quietly restore the broken ones.
    assert adapter.LANGUAGE_VOICES["en-gb"] == "en-gb-x-rp"
    assert adapter.LANGUAGE_VOICES["pt-br"] == "pt-BR"
    assert "en-gb" not in adapter.LANGUAGE_VOICES.values()
    assert "fr-FR" not in adapter.LANGUAGE_VOICES.values()


def test_japanese_uses_a_latin_script_voice_on_purpose():
    """`ja` must NOT map to espeak's `ja`, and that looks wrong until you measure it.

    Kokoro's token table is the misaki inventory and lacks `ʑ`, which espeak-ja emits
    for じ. sherpa drops what it cannot look up — 12 phonemes from one sentence,
    audible as holes. plugins/ja_text_norm.py emits romaji instead, so the voice is
    picked for its phoneme inventory rather than its language:

        kana   + ja     12 dropped,  4.81 s
        romaji + it      0 dropped,  4.60 s
    """
    voice = tts.KokoroTTSAdapter.LANGUAGE_VOICES["ja"]
    assert voice != "ja", (
        "espeak-ja emits phonemes Kokoro cannot represent; ja must use a "
        "Latin-script voice against romanised text")
    assert voice in ("it", "es", "en-us"), voice


def test_chinese_maps_to_an_english_espeak_voice_on_purpose():
    """Not a bug and not redundant with en-us.

    Chinese is phonemised from lexicon-zh.txt, a branch that never consults `lang`,
    so this field only decides how *embedded Latin* in a Chinese sentence is read —
    and English is the right answer. A Chinese espeak voice here would produce no
    audio for those runs, dropping every Latin word out of mixed text.
    """
    assert tts.KokoroTTSAdapter.LANGUAGE_VOICES["zh"] == "en-us"
    assert "zh" in tts.KokoroTTSAdapter.LANGUAGES


def test_the_schema_enum_matches_the_supported_languages():
    field = tts.TOOLS[0]["configSchema"]["properties"]["tts_language"]
    assert set(field["enum"]) == set(tts.KokoroTTSAdapter.LANGUAGES)
    assert len(field["enum"]) == len(tts.KokoroTTSAdapter.LANGUAGES)


# ── release validation, before sherpa-onnx can exit(-1) ───────────────────────

def _write_manifest(tmp_path, **overrides):
    manifest = {
        "model_version": 2,
        "sample_rate": 24000,
        "n_speakers": 54,
        "id2speaker": {"0": "af_alloy", "3": "af_heart", "47": "zf_xiaoxiao"},
        "languages": {"en-us": list(range(20)), "en-gb": list(range(20, 28)),
                      "zh": list(range(45, 53))},
    }
    manifest.update(overrides)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_a_valid_manifest_is_returned(tmp_path):
    expected = _write_manifest(tmp_path)
    assert tts._validate_kokoro_manifest(str(tmp_path)) == expected


def test_a_missing_manifest_names_the_repack_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="repack_kokoro_v1_0"):
        tts._validate_kokoro_manifest(str(tmp_path))


def test_a_v0_19_release_is_refused(tmp_path):
    """version 1 takes the PiperPhonemizeLexicon path and ignores `lang` entirely.

    Loading one would leave the language dropdown visibly present and silently
    inert, which is worse than refusing.
    """
    _write_manifest(tmp_path, model_version=1)
    with pytest.raises(RuntimeError, match="model_version=1"):
        tts._validate_kokoro_manifest(str(tmp_path))


@pytest.mark.parametrize("rate", [16000, 22050, 48000, 0])
def test_a_release_at_another_sample_rate_is_refused(tmp_path, rate):
    """utils/resample.py's filter is built for 24000 -> 16000 (2:3) only.

    Accepting another rate would resample by the wrong ratio: audio at the wrong
    pitch that also drifts against the 100 ms frame clock. Note 16000 is refused
    too — it would need no resampling at all, so the adapter's assumptions would
    not hold either way.
    """
    _write_manifest(tmp_path, sample_rate=rate)
    with pytest.raises(RuntimeError, match="Hz"):
        tts._validate_kokoro_manifest(str(tmp_path))


def test_a_manifest_without_the_speaker_map_is_refused(tmp_path):
    """The map is what the voice/language coherence warning checks against."""
    _write_manifest(tmp_path, id2speaker={})
    with pytest.raises(RuntimeError, match="id2speaker"):
        tts._validate_kokoro_manifest(str(tmp_path))


def test_the_face_plugin_imports_onnxruntime_at_module_scope():
    """Guards a one-line placement that keeps Kokoro working at all.

    perception ends up with two builds of ONNX Runtime in one process: sherpa-onnx
    bundles its own libonnxruntime.so, and the face plugin uses the standalone
    `onnxruntime` package. They export the same symbols, so whichever arrives first
    wins resolution for both. Measured on Orin 6: Kokoro synthesizes fine, the face
    card starts, and every later utterance dies with

        SequenceInsert ... TensorSeq::Add ... IsSameDataType(tensor) was false

    with the real cause one line above it — espeak losing its voice
    ("Unknown phoneme table: ''"), producing no phonemes, so the graph's Loop gets
    an empty sequence.

    Importing at module scope is enough on its own (no session needed, +27 MB,
    0.19 s) because main.py imports plugins.face during startup, before any plugin
    builds a model. Moving it back inside FaceAnalyzer.__init__ — which is where it
    was, and which looks tidier — reintroduces the bug, and nothing else would
    catch that.
    """
    import ast
    import inspect

    from plugins import face_runtime

    tree = ast.parse(inspect.getsource(face_runtime))
    module_level = set()
    nested = set()
    for node in tree.body:                      # module scope only
        for sub in ast.walk(node):
            if isinstance(sub, ast.Import):
                names = {a.name for a in sub.names}
                if isinstance(node, (ast.Import, ast.Try)):
                    module_level |= names
                else:
                    nested |= names

    assert "onnxruntime" in module_level, (
        "plugins/face_runtime.py must import onnxruntime at module scope, before "
        "sherpa-onnx creates any session — see the comment on that import")
    assert "onnxruntime" not in nested, (
        "onnxruntime is imported lazily somewhere in face_runtime; that defers the "
        "library load past sherpa's and breaks kokoro-multi")


def test_the_pipeline_rate_and_kokoro_rate_are_both_stated():
    """Two named constants, so neither is a magic number in the adapter."""
    assert tts.SAMPLE_RATE == 16000
    assert tts.KOKORO_SAMPLE_RATE == 24000
    from utils import resample
    assert (resample.UP, resample.DOWN) == (2, 3)
    assert tts.KOKORO_SAMPLE_RATE * resample.UP // resample.DOWN == tts.SAMPLE_RATE


# ── speaker_id is an index within the language, not a global id ───────────────

# The real manifest's shape, abbreviated. Note es is non-contiguous (28, 29, 53)
# and fr has exactly one voice — both are why a global id is unguessable and why
# the per-language range differs.
_LANGS = {
    "en-us": list(range(0, 20)),
    "en-gb": list(range(20, 28)),
    "es": [28, 29, 53],
    "fr": [30],
    "hi": [31, 32, 33, 34],
    "it": [35, 36],
    "ja": [37, 38, 39, 40, 41],
    "pt-br": [42, 43, 44],
    "zh": list(range(45, 53)),
}
_NAMES = {"0": "af_alloy", "3": "af_heart", "20": "bf_alice", "30": "ff_siwis",
          "37": "jf_alpha", "45": "zf_xiaobei", "53": "em_santa"}


def _bare_adapter(voice_index=0, language="en-us"):
    """A KokoroTTSAdapter with only the fields the mapping logic touches.

    Constructed without __init__ on purpose: resolving a voice index is pure
    manifest arithmetic, and requiring a 310 MB model to test it would mean it never
    got tested at all.

    `_ja_frontend` is pre-filled so switching to `ja` does not drag janome in — this
    fixture is about the speaker mapping, and the Japanese frontend has its own
    tests in test_ja_text_norm.py.
    """
    a = object.__new__(tts.KokoroTTSAdapter)
    a._manifest = {"languages": _LANGS, "id2speaker": _NAMES}
    a._language = language
    a._voice_index = voice_index
    a._ja_frontend = _StubJapaneseFrontend()
    a._sid = a._resolve_sid(voice_index, language, strict=True)
    return a


class _StubJapaneseFrontend:
    def normalize(self, text):
        return text


@pytest.mark.parametrize("language,index,expected", [
    ("en-us", 0, 0),
    ("en-us", 3, 3),
    ("en-gb", 0, 20),      # not 0 — the global ids are unguessable
    ("ja", 0, 37),
    ("ja", 4, 41),
    ("zh", 0, 45),
    ("fr", 0, 30),
    ("es", 2, 53),         # non-contiguous: es is 28, 29 and 53
])
def test_speaker_id_is_relative_to_the_language(language, index, expected):
    assert _bare_adapter(index, language)._sid == expected


def test_voice_zero_of_every_language_is_a_voice_of_that_language():
    """The property that makes the mismatch warning unnecessary."""
    for language, ids in _LANGS.items():
        a = _bare_adapter(0, language)
        assert a._sid == ids[0]
        assert a._sid in ids


@pytest.mark.parametrize("language,bad", [("fr", 1), ("it", 2), ("en-us", 20)])
def test_out_of_range_at_construction_is_an_error(language, bad):
    """A configuration mistake must be reported, naming the language's count."""
    with pytest.raises(ValueError, match=f"language '{language}'"):
        _bare_adapter(bad, language)


def test_switching_language_moves_the_voice_with_it():
    a = _bare_adapter(0, "en-us")
    assert a._sid == 0
    a.set_language("ja")
    assert a._language == "ja"
    assert a._sid == 37, "the voice stayed English while the phonemes went Japanese"
    a.set_language("zh")
    assert a._sid == 45


def test_selecting_japanese_builds_the_frontend_eagerly():
    """A missing janome must fail the config call, not the first utterance.

    Kanji reach sherpa's Chinese branch without it, so the card would come up
    `running` and speak Mandarin — the failure the Thai adapter refuses to ship for
    the same reason.
    """
    import inspect

    for src in (inspect.getsource(tts.KokoroTTSAdapter.set_language),
                inspect.getsource(tts.KokoroTTSAdapter.__init__)):
        assert "self._ja()" in src, (
            "the Japanese frontend must be built when ja is selected, not lazily "
            "on the first speak")


def test_switching_into_a_language_with_fewer_voices_clamps(caplog):
    """Language is per-utterance and free, so it must not be able to fail.

    French has exactly one voice. Raising here would turn the language dropdown
    into a trap for anyone who had picked voice 5 of English first.
    """
    a = _bare_adapter(5, "en-us")
    assert a._sid == 5
    with caplog.at_level("WARNING"):
        a.set_language("fr")
    assert a._sid == 30, "did not fall back to the language's first voice"
    assert a._voice_index == 5, "the requested index is remembered, not overwritten"
    assert "out of range" in caplog.text

    # ...and coming back restores the original voice rather than the clamp.
    a.set_language("en-us")
    assert a._sid == 5


def test_the_resolved_voice_is_named_for_logs_and_info():
    """An operator reads `af_heart`, not `3`."""
    assert _bare_adapter(3, "en-us").voice_name == "af_heart"
    assert _bare_adapter(0, "ja").voice_name == "jf_alpha"
    # Unknown ids degrade rather than raising — the map is only for display.
    a = _bare_adapter(1, "en-us")
    assert a.voice_name == "?"


def test_a_language_absent_from_the_manifest_is_refused():
    a = _bare_adapter(0, "en-us")
    a._manifest = {"languages": {"en-us": [0]}, "id2speaker": {}}
    with pytest.raises(RuntimeError, match="no voices for 'ja'"):
        a._resolve_sid(0, "ja", strict=True)


# ── the download: one archive per device, each in its own subdirectory ─────────

@pytest.fixture
def _downloader(monkeypatch):
    """ensure_kokoro_model with the network stubbed and both variants pinned."""
    from utils import model_downloader

    calls = []
    monkeypatch.setattr(
        model_downloader, "ensure_verified_archive",
        lambda name, model_dir, url, entry: calls.append(
            {"name": name, "model_dir": model_dir, "url": url, "entry": entry}),
    )
    monkeypatch.setitem(model_downloader.KOKORO_MODEL_ARCHIVES, "gpu", {
        "archive": "kokoro-multi-v1_0-24k-fp32.tar.gz", "size": 1, "sha256": "aa"})
    monkeypatch.setitem(model_downloader.KOKORO_MODEL_ARCHIVES, "cpu", {
        "archive": "kokoro-multi-v1_0-24k-int8.tar.gz", "size": 2, "sha256": "bb"})
    return model_downloader, calls


def test_each_device_gets_its_own_subdirectory(_downloader):
    """gpu holds fp32 and cpu holds int8, so they cannot share one directory.

    ensure_verified_archive stages then publishes into the directory it is given;
    two installs in one tree would let the second take the first's weights with it.
    Separate subdirectories also mean flipping `device` back finds its files intact
    instead of re-downloading 330 MB.
    """
    downloader, calls = _downloader
    root = tts.ENGINE_MODEL_DIRS["kokoro-multi"]

    assert downloader.ensure_kokoro_model(root, "gpu") == f"{root}/gpu"
    assert downloader.ensure_kokoro_model(root, "cpu") == f"{root}/cpu"

    assert [c["model_dir"] for c in calls] == [f"{root}/gpu", f"{root}/cpu"]
    # Distinct marker names, or the second install would see the first's
    # `.installed` file and skip its own download.
    assert [c["name"] for c in calls] == ["kokoro/gpu", "kokoro/cpu"]
    assert "fp32" in calls[0]["url"] and "int8" in calls[1]["url"]


@pytest.mark.parametrize("device,expected", [
    ("gpu", "gpu"), ("GPU", "gpu"), ("cuda", "cpu"), ("cpu", "cpu"), ("", "cpu"),
])
def test_only_an_explicit_gpu_selects_the_fp32_archive(_downloader, device, expected):
    """Deliberately literal, not normalize_device.

    The caller has already run normalize_device (the adapter does, before calling
    here), so anything else arriving is a config this function should not guess
    about — and guessing wrong towards gpu would download 330 MB onto a host with
    no CUDA wheel, where provider_for_device then falls back to cpu anyway.
    """
    downloader, _ = _downloader
    root = tts.ENGINE_MODEL_DIRS["kokoro-multi"]
    assert downloader.ensure_kokoro_model(root, device) == f"{root}/{expected}"


def test_an_unpinned_archive_is_refused_rather_than_downloaded(monkeypatch):
    """Every model here is size+SHA256 verified; a 330 MB hole would be the one.

    Same rule ensure_thai_tts_model states. This also means the engine fails loudly
    with a message naming the repack script until the artefact is actually published,
    rather than half-working against an unverified blob.
    """
    from utils import model_downloader

    monkeypatch.setitem(model_downloader.KOKORO_MODEL_ARCHIVES, "gpu", {
        "archive": "kokoro-multi-v1_0-24k-fp32.tar.gz", "size": 0, "sha256": ""})
    with pytest.raises(RuntimeError, match="repack_kokoro_v1_0"):
        model_downloader.ensure_kokoro_model(
            tts.ENGINE_MODEL_DIRS["kokoro-multi"], "gpu")


def test_the_model_dir_cannot_escape_the_models_tree(_downloader):
    """model_dir arrives over MCP config and the downloader runs as root."""
    downloader, _ = _downloader
    with pytest.raises(ValueError, match="must resolve under"):
        downloader.ensure_kokoro_model("/etc/kokoro", "gpu")
