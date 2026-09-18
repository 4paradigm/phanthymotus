#!/usr/bin/env python3
"""
tests/test_thai_frontend.py — the Thai TTS text frontend.

Pure Python: no model, no ROS, no sherpa-onnx. What it guards is one property,
and the whole engine depends on it — **every character the frontend emits must be
one the MMS Thai tokenizer can actually pronounce.** Anything else is skipped
during synthesis with only a C++ stderr line to show for it, so the audio just
comes back short. Same failure the ZH/EN frontend records in
test_vits2_frontend.py::test_digits_are_not_stripped_by_the_chinese_path.

The optional transliteration deps (pythainlp, wunsen/khanaa, pypinyin) are not
installed on every dev host. Tests that need one skip; the vocabulary-containment
tests do not, because that guarantee has to hold on any host.
"""

from __future__ import annotations

import logging

import pytest

from plugins.thai_frontend import MMS_THAI_VOCAB, ThaiFrontend


def _have(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


needs_pythainlp = pytest.mark.skipif(
    not _have("pythainlp"), reason="pythainlp is not installed on this host"
)
needs_khanaa = pytest.mark.skipif(
    not _have("khanaa"), reason="khanaa is not installed on this host"
)
needs_pypinyin = pytest.mark.skipif(
    not (_have("pypinyin") and _have("wunsen")),
    reason="pypinyin/wunsen are not installed on this host",
)


@pytest.fixture
def frontend():
    return ThaiFrontend()


def _assert_speakable(text: str) -> None:
    """The single invariant: nothing outside the model's alphabet survives."""
    stray = sorted({ch for ch in text if ch not in MMS_THAI_VOCAB})
    assert not stray, f"{stray} cannot be pronounced by the model but reached it"


# ── the vocabulary itself ────────────────────────────────────────────────────

def test_the_alphabet_matches_what_the_model_ships():
    # 71 tokens, the count the exported tokens.txt has. A silent change here
    # would move the goalposts for every other test in this file.
    assert len(MMS_THAI_VOCAB) == 71


def test_sara_am_really_is_absent_which_is_why_it_is_rewritten():
    """The premise of _rewrite_sara_am. If this ever fails, drop the rewrite."""
    assert "ำ" not in MMS_THAI_VOCAB
    # …and both halves of the replacement are present, or the rewrite would only
    # trade one unpronounceable character for two.
    assert "ํ" in MMS_THAI_VOCAB
    assert "า" in MMS_THAI_VOCAB


def test_most_arabic_digits_are_absent_which_is_why_numbers_are_spelled_out():
    """Only 0 1 2 4 are in the table — 3 and 5-9 are not."""
    assert {"0", "1", "2", "4"} <= set(MMS_THAI_VOCAB)
    for digit in "356789":
        assert digit not in MMS_THAI_VOCAB


# ── the invariant, across the shapes real text takes ─────────────────────────

@needs_pythainlp
@pytest.mark.parametrize("text", [
    "สวัสดีครับ ห้อง 305 พร้อมแล้ว",
    "น้ำ ทำ คำ สำหรับ น้ำหนัก",
    "ต่างๆ นานา และมากๆ",
    "ราคา 1,250.50 บาท",
    "เบอร์โทร 0812345678",
    "ปี 2026 เดือน 12",
    "ฯลฯ ๗๘๙ ฿100",
    "ผลลัพธ์ ✅ เรียบร้อย",
    "อุณหภูมิ 25° และ 80%",
    "",
    "   ",
])
def test_output_only_contains_characters_the_model_can_say(frontend, text):
    _assert_speakable(frontend.normalize(text))


@needs_pythainlp
@needs_khanaa
@pytest.mark.parametrize("text", [
    "หุ่นยนต์ Bumi พร้อม",
    "เชื่อมต่อ WiFi แล้ว",
    "ระบบ AI, USB และ G1",
    "Bangkok Sukhumvit station",
])
def test_latin_is_transliterated_not_passed_through(frontend, text):
    out = frontend.normalize(text)
    _assert_speakable(out)
    assert not any(ch.isascii() and ch.isalpha() for ch in out), \
        "a Latin letter reached the model, which would drop it"


# ── the specific rewrites ────────────────────────────────────────────────────

@needs_pythainlp
def test_sara_am_is_rewritten_so_the_vowel_survives(frontend):
    out = frontend.normalize("น้ำ")
    assert "ำ" not in out
    assert "ํา" in out, "the vowel was dropped instead of respelled"
    _assert_speakable(out)


def test_a_vocabulary_that_has_sara_am_is_left_alone():
    """A future voice whose table contains ำ must not be rewritten."""
    richer = ThaiFrontend(vocab=set(MMS_THAI_VOCAB) | {"ำ"})
    assert richer._rewrite_sara_am("น้ำ") == "น้ำ"


@needs_pythainlp
def test_digits_become_words_rather_than_being_dropped(frontend):
    out = frontend.normalize("ห้อง 305")
    # 3 and 5 are not in the table at all, so surviving as digits means silence.
    assert "3" not in out and "5" not in out
    assert "สาม" in out, "the digit was dropped instead of pronounced"
    _assert_speakable(out)


@needs_pythainlp
def test_a_decimal_is_not_cut_apart_by_the_punctuation_sweep(frontend):
    """The sweep turns "." and "," into spaces, so numbers must be read first.

    Running it the other way round read 1,250.50 as three unrelated numbers —
    "one", "two hundred fifty", "fifty".
    """
    out = frontend.normalize("ระยะ 1,250.50 เมตร")
    assert "จุด" in out, "the decimal point was lost"
    assert "หนึ่งพัน" in out, "the thousands group was split off"
    _assert_speakable(out)


@needs_pythainlp
def test_currency_reads_with_thai_units_in_the_right_order(frontend):
    out = frontend.normalize("฿100")
    # Replacing ฿ with บาท on its own put the unit *before* the amount.
    assert out.index("ร้อย") < out.index("บาท")
    _assert_speakable(out)


@needs_pythainlp
def test_a_long_digit_group_is_read_digit_by_digit(frontend):
    """An 10-digit phone number is an identifier, not a quantity."""
    out = frontend.normalize("0812345678")
    assert "ศูนย์" in out and "แปด" in out
    # Cardinal reading would have produced a "hundred million"-scale word.
    assert "ล้าน" not in out
    _assert_speakable(out)


@needs_pythainlp
def test_maiyamok_is_expanded_because_the_marker_is_not_in_the_table(frontend):
    assert "ๆ" not in MMS_THAI_VOCAB
    out = frontend.normalize("ต่างๆ")
    assert "ๆ" not in out
    assert out.count("ต่าง") == 2, "the repeat marker was dropped, not expanded"
    _assert_speakable(out)


@needs_pythainlp
def test_a_leading_maiyamok_has_nothing_to_repeat_and_does_not_crash(frontend, caplog):
    """pythainlp 5.0.4's own `maiyamok` raises IndexError on this.

    Which is one of the two reasons the expansion is done here instead: its
    signature also differs between the versions cp38 and cp310 resolve to (token
    list vs string). Dropping the marker is the only option, but it is logged.
    """
    with caplog.at_level(logging.WARNING, logger="plugins.thai_frontend"):
        out = frontend.normalize("ๆ นำหน้า")
    assert "ๆ" not in out
    assert "นํา" in out, "the rest of the sentence was lost with the marker"
    assert any("ๆ" in record.getMessage() for record in caplog.records)
    _assert_speakable(out)


@needs_pythainlp
def test_maiyamok_glued_to_its_word_by_the_tokeniser_still_expands(frontend):
    """newmm can return "ต่างๆ" as a single token rather than two."""
    out = frontend.normalize("สินค้าต่างๆ พร้อม")
    assert "ๆ" not in out
    assert out.count("ต่าง") == 2
    _assert_speakable(out)


# ── the dry-run probe (device regression) ────────────────────────────────────


@needs_pythainlp
def test_the_thai_dry_run_probe_survives_normalisation(frontend):
    """_TTSNode.start() refuses `running` unless its probe produces audio.

    The shared probe was ".", and this frontend normalises punctuation to a space
    and then strips it — so the probe reached the model as an empty string and
    every Thai start on the robot answered "TTS dry-run produced no audio" while
    the model itself was fine. The probe is now per-adapter; this pins the property
    that made the old one wrong.
    """
    from plugins.tts import MmsThaiTTSAdapter, TTSAdapter

    assert frontend.normalize(TTSAdapter.dry_run_text) == "", \
        "the shared '.' probe is expected to normalise away — that was the bug"
    probe = MmsThaiTTSAdapter.dry_run_text
    assert probe != TTSAdapter.dry_run_text, "the Thai adapter must override it"
    assert frontend.normalize(probe), f"the Thai probe {probe!r} normalises to nothing"
    _assert_speakable(frontend.normalize(probe))


def test_every_adapter_declares_a_dry_run_probe():
    """A new engine that forgets this inherits '.', which its frontend may drop."""
    from plugins.tts import MatchaTTSAdapter, MmsThaiTTSAdapter, TTSAdapter

    for adapter in (TTSAdapter, MatchaTTSAdapter, MmsThaiTTSAdapter):
        probe = getattr(adapter, "dry_run_text", None)
        assert isinstance(probe, str) and probe, f"{adapter.__name__} has no probe"



@needs_pythainlp
def test_the_build_time_self_check_passes():
    """`python3 -m plugins.thai_frontend`, the command the Dockerfile runs."""
    from plugins.thai_frontend import self_check

    self_check()


@needs_pythainlp
def test_the_self_check_actually_fails_when_the_invariant_breaks(monkeypatch):
    """A check that cannot fail is worse than none — it reads as coverage."""
    from plugins import thai_frontend as module

    monkeypatch.setattr(module.ThaiFrontend, "_rewrite_sara_am",
                        lambda self, text: text)
    with pytest.raises(AssertionError):
        module.self_check()


def test_the_dockerfile_can_parse_its_own_self_check_step():
    """A RUN cannot contain a bare newline — the parser reads it as an instruction.

    The first version of the build-time check was a multi-line `python3 -c "..."`
    and failed with `dockerfile parse error: unknown instruction: import` before
    any layer built. Cheap to guard, and it covers the whole file rather than just
    the step this change added.
    """
    import pathlib

    instructions = {
        "FROM", "RUN", "CMD", "LABEL", "EXPOSE", "ENV", "ADD", "COPY",
        "ENTRYPOINT", "VOLUME", "USER", "WORKDIR", "ARG", "ONBUILD",
        "STOPSIGNAL", "HEALTHCHECK", "SHELL",
    }
    dockerfile = pathlib.Path(__file__).resolve().parents[1] / "Dockerfile.jetson"
    offenders = []
    continuing = False
    for number, raw in enumerate(dockerfile.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw.strip()
        if not continuing and stripped and not stripped.startswith("#"):
            head = stripped.split()[0].upper()
            if head not in instructions:
                offenders.append((number, stripped[:60]))
        continuing = raw.rstrip().endswith("\\")
    assert not offenders, f"lines continuing an instruction without a backslash: {offenders}"
    assert not continuing, "the file ends inside a line continuation"


def test_the_dockerfile_names_no_version_specific_thai_apis():
    """Build steps must not reference an API that exists on only one JetPack line.

    This mistake was made twice in a single Dockerfile command. `expand_maiyamok`
    exists in pythainlp 5.3.7 but not 5.0.4; `Kham` exists in khanaa 0.1.1 but not
    0.0.6 — and cp38 (jp5.11) gets the older of each. Naming either in the build
    check turns a working image into a failed build on that one line, which is only
    discovered by building it.

    The versions are pinned per Python version and adapted over in
    plugins/thai_frontend.py; the Dockerfile's job is to prove the packages import.
    """
    import pathlib

    dockerfile = (pathlib.Path(__file__).resolve().parents[1] / "Dockerfile.jetson")
    text = dockerfile.read_text(encoding="utf-8")
    # Only the RUN steps matter — a comment may name these while explaining why.
    code = "\n".join(
        line.split("#", 1)[0] for line in text.splitlines()
        if not line.lstrip().startswith("#")
    )
    for name in ("Kham", "SpellWord", "expand_maiyamok", "spell_out"):
        assert name not in code, (
            f"{name} exists in only one of the pinned versions; assert it in "
            "plugins/thai_frontend.py:self_check() instead"
        )


def test_the_khanaa_adapter_is_resolved_once_and_survives_a_broken_import():
    """khanaa 0.1.1 raises TypeError, not ImportError, when imported on py3.8.

    An `except ImportError` did not catch that and the TypeError escaped
    normalize(), killing the utterance. The loader must swallow any import-time
    failure and let the Latin path fall back to spelling words out.
    """
    from plugins import thai_frontend as module

    # Whatever this host has, the loader answered with a callable or None — never
    # by raising.
    assert module._KHANAA_SPELL is None or callable(module._KHANAA_SPELL)
    if module._KHANAA_SPELL is not None:
        # Both khanaa APIs must produce the same syllable for the same input;
        # verified on cp38 (SpellWord.spell_out) and cp310+ (Kham.form).
        assert module._KHANAA_SPELL("ก", "อา") == "กา"
        assert module._KHANAA_SPELL("สต", "เอะ", "ก", 3) == "เสต๊ก"


@needs_pythainlp
def test_thai_digits_are_read_too(frontend):
    out = frontend.normalize("๗ ชิ้น")
    assert "๗" not in out
    assert "เจ็ด" in out
    _assert_speakable(out)


@needs_pypinyin
@needs_pythainlp
def test_chinese_is_transliterated_rather_than_dropped(frontend):
    out = frontend.normalize("ยินดีต้อนรับ 你好 ครับ")
    assert "你" not in out and "好" not in out
    # The Thai around it must survive intact — the point of the vits2 frontend's
    # test_unvoiceable_characters_do_not_silence_the_rest.
    assert "ยินดีต้อนรับ" in out and "ครับ" in out
    _assert_speakable(out)


# ── the drop is announced ────────────────────────────────────────────────────

@needs_pythainlp
def test_unspeakable_characters_are_named_in_a_warning(frontend, caplog):
    """A silent drop reads downstream as a model bug, so it must be logged."""
    with caplog.at_level(logging.WARNING, logger="plugins.thai_frontend"):
        out = frontend.normalize("ผลลัพธ์ ✅ เรียบร้อย")
    assert any("✅" in record.getMessage() for record in caplog.records), \
        "the dropped character was not named in any warning"
    # …and the surrounding Thai still made it through.
    assert "ผลลัพธ์" in out and "เรียบร้อย" in out
    _assert_speakable(out)


# ── chunking ─────────────────────────────────────────────────────────────────

def test_short_text_is_one_chunk(frontend):
    assert list(frontend.iter_chunks("สวัสดีครับ")) == ["สวัสดีครับ"]


def test_empty_text_yields_nothing(frontend):
    assert list(frontend.iter_chunks("   ")) == []


def test_long_text_is_split_at_spaces_within_the_budget(frontend):
    text = " ".join(["สวัสดีครับ"] * 40)
    chunks = list(frontend.iter_chunks(text, max_chars=90))
    assert len(chunks) > 1
    assert all(len(chunk) <= 90 for chunk in chunks)
    # Nothing may be lost or duplicated by the split.
    assert " ".join(chunks) == text


def test_a_phrase_longer_than_the_budget_is_not_cut_mid_word(frontend):
    """Cutting inside a Thai syllable changes how it is pronounced."""
    text = "ก" * 300
    chunks = list(frontend.iter_chunks(text, max_chars=90))
    assert chunks == [text], "a space-less phrase was split anyway"
