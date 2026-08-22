from . import chinese, english, cleaned_text_to_sequence, cleaned_text_to_sequence_mix
language_module_map = {"ZH": chinese, "EN": english}


def clean_text(text, language):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph


def normalize_text_mix(text):
    """Normalize the complete MIX input once, before chunking."""
    has_zh = any("\u4e00" <= c <= "\u9fff" for c in text)
    has_en = any(("a" <= c.lower() <= "z") for c in text)
    if has_zh and not has_en:
        return chinese.text_normalize(text)
    if has_en and not has_zh:
        return chinese.mix_normalize(text)
    return chinese.mix_normalize(text)


def g2p_normalized_text_mix(norm_text):
    """Convert already-normalized MIX text to phones without running TN again."""
    has_zh = any("\u4e00" <= c <= "\u9fff" for c in norm_text)
    has_en = any(("a" <= c.lower() <= "z") for c in norm_text)
    if has_zh and not has_en:
        phones, tones, word2ph = chinese.g2p(norm_text)
        return phones, tones, ["ZH"] * len(phones), word2ph
    if has_en and not has_zh:
        if any(char.isdigit() for char in norm_text):
            raise ValueError("Arabic digits remained in an English-only MIX chunk")
        phones, tones, word2ph = english.g2p(norm_text)
        return phones, tones, ["EN"] * len(phones), word2ph
    from .unified_g2p import unified_g2p

    return unified_g2p(norm_text)


def clean_text_mix(text):
    """Unified ZH/EN text entry without BERT.

    This is the single public entry for `MIX`, but internally it dispatches:

    - pure Chinese  -> chinese.text_normalize() + chinese.g2p()
    - English chunk -> number-free English normalization + english.g2p()
    - true ZH/EN mix -> chinese.mix_normalize() + unified_g2p()

    This keeps pure Chinese equivalent to the native `ZH` path while still
    allowing one external `MIX` label/entry for ZH, EN, and ZH/EN sentences.
    """
    norm_text = normalize_text_mix(text)
    phones, tones, langs, word2ph = g2p_normalized_text_mix(norm_text)
    return norm_text, phones, tones, langs, word2ph


def clean_text_bert(text, language):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    bert = language_module.get_bert_feature(norm_text, word2ph)
    return phones, tones, bert


def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)


def text_to_sequence_mix(text):
    """Convert ZH/EN mixed text to model input sequences."""
    norm_text, phones, tones, langs, word2ph = clean_text_mix(text)
    phone_ids, tone_ids, lang_ids = cleaned_text_to_sequence_mix(phones, tones, langs)
    return norm_text, phone_ids, tone_ids, lang_ids, word2ph


if __name__ == "__main__":
    pass
