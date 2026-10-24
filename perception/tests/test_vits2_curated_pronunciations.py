"""Real-release phones for reviewed names; no acoustic engine is constructed."""

import pytest

# Reuse the existing release/dependency guards and CPU-only production adapter.
from test_vits2_frontend import (
    g2p_normalized_text_mix,
    normalize_text_mix,
    production_text_adapter,
)
from plugins.vits2_tts_trt.frontend import english


def prepared(adapter, text, mode):
    from plugins.vits2_tts_language.plugin import _direct_request, _prepare_chunks
    from plugins.vits2_tts_language.policy import LanguagePolicy

    chunks = _prepare_chunks(adapter, _direct_request(LanguagePolicy(mode), text)).chunks
    phones, tones, languages = [], [], []
    for normalized, _ in chunks:
        chunk_phones, chunk_tones, chunk_languages, word2ph = g2p_normalized_text_mix(normalized)
        assert sum(word2ph) == len(chunk_phones)
        phones.extend(chunk_phones)
        tones.extend(chunk_tones)
        languages.extend(chunk_languages)
    return " ".join(text for text, _ in chunks), phones, tones, languages


def matching_positions(sequence, expected):
    return [index for index in range(len(sequence) - len(expected) + 1)
            if sequence[index:index + len(expected)] == expected]


@pytest.mark.parametrize("mode", ["zh", "en", "auto"])
@pytest.mark.parametrize("text", [
    "Hello", "hello", "HELLO", "Hello.", "hello!", "HELLO?",
    "你好Hello", "你好hello。", "你好HELLO！", "Hello，世界。",
    "Hello, I am Xiao Fang.",
])
def test_hello_keeps_published_cmu_phones_and_stress(production_text_adapter, mode, text):
    assert english.eng_dict["HELLO"] == [["HH", "AH0"], ["L", "OW1"]]
    _, phones, tones, languages = prepared(production_text_adapter, text, mode)
    positions = matching_positions(phones, ["hh", "ah", "l", "ow"])
    assert len(positions) == 1
    start = positions[0]
    assert tones[start:start + 4] == [3, 1, 3, 2]
    assert languages[start:start + 4] == ["EN"] * 4


@pytest.mark.parametrize("mode", ["zh", "en", "auto"])
@pytest.mark.parametrize("name", ["OpenAI", "openai", "OPENAI"])
@pytest.mark.parametrize("template", ["{}", "We use {}.", "请检查{}。"])
def test_openai_retains_both_letter_names(production_text_adapter, mode, name, template):
    normalized, phones, _, _ = prepared(production_text_adapter, template.format(name), mode)
    assert "open A I" in normalized
    assert matching_positions(phones, ["ow", "p", "ah", "n", "ey", "ay"])


@pytest.mark.parametrize("mode", ["zh", "en", "auto"])
@pytest.mark.parametrize("text,prefix", [("SkyAI", ["s", "k", "ay"]),
                                         ("MBody AI", ["eh", "m", "b", "aa", "d", "iy"])])
def test_protected_generated_ai_is_spelled_at_word_stage(production_text_adapter, mode, text, prefix):
    _, phones, _, _ = prepared(production_text_adapter, text, mode)
    assert matching_positions(phones, prefix + ["ey", "ay"])


def test_ai_exception_does_not_force_known_uppercase_words_into_letters():
    normalized = normalize_text_mix("NASA and AI are ready.")
    assert normalized == "NASA and A I are ready."
    phones, _, _, _ = g2p_normalized_text_mix(normalized)
    assert phones[1:5] == ["n", "ae", "s", "ah"]
    assert matching_positions(phones, ["ey", "ay"])


@pytest.mark.parametrize("mode", ["zh", "en", "auto"])
@pytest.mark.parametrize("title", ["Dr.", "dr.", "DR."])
@pytest.mark.parametrize("prefix", ["", "请联系", "请联系 "])
def test_doctor_title_accepts_case_and_chinese_adjacency(production_text_adapter, mode, title, prefix):
    normalized, phones, _, _ = prepared(production_text_adapter, prefix + title + " Smith。", mode)
    assert "doctor Smith" in normalized
    assert matching_positions(phones, ["d", "aa", "k", "t", "er"])
    assert not matching_positions(phones, ["d", "r", "ay", "V"])


@pytest.mark.parametrize("text", ["The address is Oak Dr.", "Drake is here.",
                                  "caféDr. Smith", "Andr. Smith", "my_Dr. Smith"])
def test_doctor_rule_does_not_rewrite_roads_or_inside_names(text):
    normalized = normalize_text_mix(text)
    assert "doctor" not in normalized.lower()


@pytest.mark.parametrize("mode", ["zh", "en", "auto"])
@pytest.mark.parametrize("name", ["StyleTTS2", "styletts2", "STYLETTS2"])
def test_model_version_language_is_explicit(production_text_adapter, mode, name):
    normalized, phones, _, _ = prepared(production_text_adapter, name, mode)
    # Auto model-only inference follows its existing language policy; fixed
    # modes must use their selected reading without a Chinese FST quantity.
    if mode == "zh":
        assert "二" in normalized and "两" not in normalized
    elif mode == "en":
        assert "two" in normalized and "二" not in normalized
        assert matching_positions(phones, ["t", "uw"])
    assert "T T S" in normalized


@pytest.mark.parametrize("text", ["我们使用StyleTTS2合成语音。", "我们使用StyleTTS2。"])
def test_chinese_model_version_is_not_quantity_liang(production_text_adapter, text):
    normalized, phones, _, _ = prepared(production_text_adapter, text, "zh")
    assert "二" in normalized
    assert "两" not in normalized
    assert matching_positions(phones, ["EE", "er"])


@pytest.mark.parametrize("text", ["caféAI", "AIé", "éAIé", "cafe\u0301AI", "AI\u0301", "my_AI", "AI_test"])
def test_explicit_ai_spelling_respects_complete_unicode_words(text):
    assert english._spell_explicit_uppercase_acronyms(text) == text
    if "_" not in text:
        normalized = normalize_text_mix(text)
        assert "A I" not in normalized


def test_cjk_adjacency_keeps_explicit_ai_letters():
    normalized = normalize_text_mix("请检查AI服务。")
    assert "A I" in normalized
    phones, _, _, _ = g2p_normalized_text_mix(normalized)
    assert matching_positions(phones, ["ey", "ay"])


@pytest.mark.parametrize("text", ["éphanthymotusé", "éRise vGPUé", "foo_phanthymotus_bar",
                                  "prefix_Rise vGPU_suffix", "e\u0301phanthymotus", "phanthymotus\u0301"])
def test_late_technical_layer_cannot_rewrite_protected_word_boundaries(text):
    from plugins.vits2_tts_trt.frontend.technical_text import normalize_technical_lexemes

    assert normalize_technical_lexemes(text) == text
    normalized = normalize_text_mix(text)
    assert "Phanthy Motus" not in normalized
    assert "V G P U" not in normalized


@pytest.mark.parametrize("separator", ["\n", "\r\n", "\r", "\v", "\f", "\x85", "\u2028", "\u2029"])
def test_late_technical_phrase_never_consumes_line_boundaries(separator):
    from plugins.vits2_tts_trt.frontend.technical_text import normalize_technical_lexemes

    text = "rise" + separator + "vgpu"
    assert normalize_technical_lexemes(text) == text
    normalized = normalize_text_mix(text)
    phones, _, _, _ = g2p_normalized_text_mix(normalized)
    assert normalized.count(".") == 1
    assert phones.count(".") == 1


@pytest.mark.parametrize("text", ["Rise vGPU", "Rise\tvgpu", "Rise\u00a0vgpu", "请检查Rise vGPU。", "请检查phanthymotus。"])
def test_technical_boundary_guard_keeps_normal_products(text):
    normalized = normalize_text_mix(text)
    if "phanthymotus" in text:
        assert "Phanthy Motus" in normalized
    else:
        assert "V G P U" in normalized


@pytest.mark.parametrize("text", ["foo_phanthymotus_bar", "prefix_Rise vGPU_suffix", "foo_AI_bar", "my_AI", "AI_test", "id_123"])
def test_identifier_boundary_protection_keeps_all_spoken_pieces(text):
    normalized = normalize_text_mix(text)
    phones, tones, languages, word2ph = g2p_normalized_text_mix(normalized)
    assert "_" not in normalized
    assert "UNK" not in phones
    assert len(phones) == len(tones) == len(languages) == sum(word2ph)
    if "AI" in text:
        assert "A I" in normalized
        assert matching_positions(phones, ["ey", "ay"])
    if text == "foo_phanthymotus_bar":
        assert normalized == "foo,phanthymotus,bar"
        assert matching_positions(phones, ["f", "uw"])
        assert matching_positions(phones, ["b", "aa", "r"])
    elif text == "prefix_Rise vGPU_suffix":
        assert normalized == "prefix,Rise vGPU,suffix"
    elif text == "id_123":
        # Numeric identifiers retain the existing number path, never send raw
        # digits into English G2P or drop their spoken content.
        assert not any(character.isdigit() for character in normalized)
        assert "三" in normalized or "three" in normalized
