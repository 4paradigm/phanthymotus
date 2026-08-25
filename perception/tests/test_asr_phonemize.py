"""
Phonemizer plumbing for asr_kws — no espeak or models required.

These pin the two things that made a misconfiguration cost 5.2 s per utterance in
production, measured from the plugin's own `kws_phonemize` span:

- phonemizer could not find libespeak-ng, because it looks it up with
  ctypes.util.find_library, which shells out to `ldconfig -p`, and the jetson image
  stubs ldconfig out during apt installs to survive qemu. So the library is
  installed and the cache does not know about it.
- every failure was retried, and `_text_to_ipa` splits an utterance into CJK /
  non-CJK segments — four for '小范小范，你好。', since fullwidth punctuation is
  outside the CJK range. Eight fork+exec of `ldconfig` per utterance from a 3.6 GB
  multi-threaded process.

Out of process the same call took 320 ms, so neither showed up until the spans were
read. That is why these are unit tests now.

Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

sys.modules.setdefault("sherpa_onnx", types.ModuleType("sherpa_onnx"))

from plugins import asr  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_espeak(monkeypatch):
    """Module-level caches, so every test starts from a cold probe."""
    monkeypatch.setattr(asr, "_ESPEAK_BACKENDS", {})
    monkeypatch.setattr(asr, "_ESPEAK_SEP", None)
    monkeypatch.setattr(asr, "_ESPEAK_UNAVAILABLE", None)
    monkeypatch.delenv("PHONEMIZER_ESPEAK_LIBRARY", raising=False)
    yield


# ── locating the library ─────────────────────────────────────────────────────

def test_probe_points_phonemizer_at_the_first_existing_candidate(monkeypatch, tmp_path):
    lib = tmp_path / "libespeak-ng.so.1"
    lib.write_bytes(b"")
    monkeypatch.setattr(asr, "_ESPEAK_LIB_CANDIDATES",
                        ("/nonexistent/libespeak-ng.so.1", str(lib)))
    asr._point_phonemizer_at_espeak()
    assert asr.os.environ["PHONEMIZER_ESPEAK_LIBRARY"] == str(lib)


def test_probe_respects_a_preset_library(monkeypatch, tmp_path):
    """A deployment that already points somewhere must not be overridden."""
    lib = tmp_path / "libespeak-ng.so.1"
    lib.write_bytes(b"")
    monkeypatch.setenv("PHONEMIZER_ESPEAK_LIBRARY", "/operator/choice.so")
    monkeypatch.setattr(asr, "_ESPEAK_LIB_CANDIDATES", (str(lib),))
    asr._point_phonemizer_at_espeak()
    assert asr.os.environ["PHONEMIZER_ESPEAK_LIBRARY"] == "/operator/choice.so"


def test_probe_leaves_the_env_alone_when_nothing_exists(monkeypatch):
    monkeypatch.setattr(asr, "_ESPEAK_LIB_CANDIDATES",
                        ("/nonexistent/a.so", "/nonexistent/b.so"))
    asr._point_phonemizer_at_espeak()
    assert "PHONEMIZER_ESPEAK_LIBRARY" not in asr.os.environ


# ── not retrying a failure ───────────────────────────────────────────────────

def test_a_construction_failure_is_probed_once_and_then_remembered(monkeypatch):
    """The whole 5.2 s was retrying this. espeak either resolves or it does not."""
    attempts = []

    def _boom(lang):
        attempts.append(lang)
        raise RuntimeError("espeak not installed on your system")

    monkeypatch.setattr(asr, "_get_espeak_backend", _boom)

    for _ in range(5):
        with pytest.raises(Exception):
            asr._phonemize_safe("小范小范", "cmn")
    assert len(attempts) == 1, f"probed {len(attempts)} times, must be 1"


def test_text_to_ipa_falls_back_to_characters_without_espeak(monkeypatch):
    monkeypatch.setattr(asr, "_get_espeak_backend",
                        lambda lang: (_ for _ in ()).throw(RuntimeError("nope")))
    ipa = asr._text_to_ipa("小范小范，你好。")
    # Degraded but usable: characters, not an exception and not an empty list.
    assert ipa[:4] == ["小", "范", "小", "范"]


def test_text_to_ipa_probes_once_across_all_segments(monkeypatch):
    """'小范小范，你好。' is four CJK/non-CJK segments; a per-segment retry is what
    multiplied one failure into eight subprocess spawns."""
    attempts = []

    def _boom(lang):
        attempts.append(lang)
        raise RuntimeError("nope")

    monkeypatch.setattr(asr, "_get_espeak_backend", _boom)
    asr._text_to_ipa("小范小范，你好。")
    assert len(attempts) == 1


# ── the happy path keeps its persistent backend ──────────────────────────────

def test_a_working_backend_is_built_once_and_reused(monkeypatch):
    """The cache was always empty before, because construction never succeeded, so
    every segment rebuilt. Reuse is the difference between 0.3 ms and 500 ms."""
    builds = []

    class _Backend:
        def __init__(self, lang, with_stress=False):
            builds.append(lang)

        def phonemize(self, texts, separator=None, strip=False):
            return ["ɕ j ɑu  f a n"]

    # Stub the two phonemizer modules _get_espeak_backend imports lazily.
    backend_mod = types.ModuleType("phonemizer.backend")
    backend_mod.EspeakBackend = _Backend
    sep_mod = types.ModuleType("phonemizer.separator")
    sep_mod.Separator = lambda **kw: object()
    pkg = types.ModuleType("phonemizer")
    monkeypatch.setitem(sys.modules, "phonemizer", pkg)
    monkeypatch.setitem(sys.modules, "phonemizer.backend", backend_mod)
    monkeypatch.setitem(sys.modules, "phonemizer.separator", sep_mod)

    for _ in range(4):
        assert asr._phonemize_safe("小范小范", "cmn") == "ɕ j ɑu  f a n"
    assert builds == ["cmn"], f"built the backend {len(builds)} times, must be once"


def test_mixed_language_text_builds_one_backend_per_language(monkeypatch):
    """'小范小范，jump forward.' needs cmn and en-us — two backends, not two per
    segment."""
    builds = []

    class _Backend:
        def __init__(self, lang, with_stress=False):
            builds.append(lang)

        def phonemize(self, texts, separator=None, strip=False):
            return ["p h"]

    backend_mod = types.ModuleType("phonemizer.backend")
    backend_mod.EspeakBackend = _Backend
    sep_mod = types.ModuleType("phonemizer.separator")
    sep_mod.Separator = lambda **kw: object()
    monkeypatch.setitem(sys.modules, "phonemizer", types.ModuleType("phonemizer"))
    monkeypatch.setitem(sys.modules, "phonemizer.backend", backend_mod)
    monkeypatch.setitem(sys.modules, "phonemizer.separator", sep_mod)

    asr._text_to_ipa("小范小范，jump forward.")
    asr._text_to_ipa("小范小范，jump forward.")
    assert sorted(set(builds)) == ["cmn", "en-us"]
    assert len(builds) == 2, f"built {len(builds)} backends for two languages"


# ── mapping a phoneme index back to a character offset ───────────────────────
#
# The reported symptom: 小潘小潘，现在发生什么了？ triggered on the wake word and the
# remainder came back as 在发生什么了？ — 现 was eaten. end_pos was right; the mapping
# was a phonemes-per-character extrapolation, round(12 * 11 / 29) = 5, measured over
# a segmentation that skipped punctuation where _text_to_ipa splits on it.

def _fake_cmn_phonemizer(monkeypatch, per_char=3):
    """espeak stand-in: `per_char` phonemes per CJK character, one per latin char.

    Deliberately non-uniform across languages, which is what broke the ratio
    estimate; a uniform stub would pass either implementation.
    """
    class _Backend:
        def __init__(self, lang, with_stress=False):
            self.lang = lang

        def phonemize(self, texts, separator=None, strip=False):
            t = texts[0]
            if self.lang == "cmn":
                return [" ".join("p%d" % i for i in range(len(t) * per_char))]
            return [" ".join(list(t))]

    backend_mod = types.ModuleType("phonemizer.backend")
    backend_mod.EspeakBackend = _Backend
    sep_mod = types.ModuleType("phonemizer.separator")
    sep_mod.Separator = lambda **kw: object()
    monkeypatch.setitem(sys.modules, "phonemizer", types.ModuleType("phonemizer"))
    monkeypatch.setitem(sys.modules, "phonemizer.backend", backend_mod)
    monkeypatch.setitem(sys.modules, "phonemizer.separator", sep_mod)


def test_segments_are_offsets_into_the_original_text():
    text = "小范小范，你好。"
    spans = asr._text_segments(text)
    assert [text[s:e] for s, e, _ in spans] == ["小范小范", "，", "你好", "。"]
    assert [cjk for _, _, cjk in spans] == [True, False, True, False]


def test_the_reported_case_keeps_every_character_after_the_wake_word(monkeypatch):
    _fake_cmn_phonemizer(monkeypatch)
    text = "小潘小潘，现在发生什么了？"
    phones, char_ends = asr._text_to_ipa(text, with_positions=True)
    assert len(char_ends) == len(phones)
    # 4 CJK chars x 3 phonemes = the wake word ends at phoneme 12
    assert asr._text_after_phoneme(text, char_ends, 12) == "现在发生什么了？"


@pytest.mark.parametrize("end_pos, expected", [
    (3,  "潘小潘，现在发生什么了？"),   # after the first character
    (6,  "小潘，现在发生什么了？"),
    (12, "现在发生什么了？"),          # after the wake word
    (15, "在发生什么了？"),
])
def test_cut_points_are_exact_not_estimated(monkeypatch, end_pos, expected):
    _fake_cmn_phonemizer(monkeypatch)
    text = "小潘小潘，现在发生什么了？"
    _, char_ends = asr._text_to_ipa(text, with_positions=True)
    assert asr._text_after_phoneme(text, char_ends, end_pos) == expected


def test_mixed_script_cut(monkeypatch):
    """The old estimate was worst here: CJK and latin have very different
    phonemes-per-character, and it applied one ratio to a merged segment."""
    _fake_cmn_phonemizer(monkeypatch)
    text = "小范小范，jump forward."
    _, char_ends = asr._text_to_ipa(text, with_positions=True)
    assert asr._text_after_phoneme(text, char_ends, 12) == "jump forward."


def test_char_ends_are_monotonic(monkeypatch):
    _fake_cmn_phonemizer(monkeypatch)
    _, char_ends = asr._text_to_ipa("小范小范，你好啊。", with_positions=True)
    assert char_ends == sorted(char_ends)


def test_positions_work_on_the_character_fallback(monkeypatch):
    """Without espeak each character is its own phoneme, so the cut is direct."""
    monkeypatch.setattr(asr, "_get_espeak_backend",
                        lambda lang: (_ for _ in ()).throw(RuntimeError("nope")))
    text = "小范小范，你好。"
    phones, char_ends = asr._text_to_ipa(text, with_positions=True)
    assert len(char_ends) == len(phones)
    assert asr._text_after_phoneme(text, char_ends, 4) == "你好。"


def test_out_of_range_end_pos_is_safe(monkeypatch):
    _fake_cmn_phonemizer(monkeypatch)
    text = "小范小范"
    _, char_ends = asr._text_to_ipa(text, with_positions=True)
    assert asr._text_after_phoneme(text, char_ends, 0) == ""
    assert asr._text_after_phoneme(text, char_ends, 9999) == ""
    assert asr._text_after_phoneme(text, [], 5) == ""


def test_positions_are_opt_in(monkeypatch):
    """Callers that only match still get a plain list, and pay no prefix passes."""
    _fake_cmn_phonemizer(monkeypatch)
    out = asr._text_to_ipa("小范小范")
    assert isinstance(out, list) and out and isinstance(out[0], str)
