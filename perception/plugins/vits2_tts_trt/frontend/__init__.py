"""Text frontend entry — sequence conversion only (no BERT)."""
from .symbols import (
    symbols, num_tones, num_languages, punctuation,
    language_id_map, language_tone_start_map,
)

_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def cleaned_text_to_sequence(cleaned_text, tones, language):
    if language not in language_tone_start_map or language not in language_id_map:
        raise ValueError(f"Unsupported language: {language}")
    phones = [_symbol_to_id[symbol] for symbol in cleaned_text]
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    lang_id = language_id_map[language]
    lang_ids = [lang_id for i in phones]
    return phones, tones, lang_ids


def cleaned_text_to_sequence_mix(phones, tones, langs):
    """Per-phone language tag sequence conversion (MIX pipeline)."""
    if not (len(phones) == len(tones) == len(langs)):
        raise ValueError(
            f"MIX cleaned-text length mismatch: "
            f"phones={len(phones)} tones={len(tones)} langs={len(langs)}"
        )
    phone_ids = [_symbol_to_id[symbol] for symbol in phones]
    tone_ids = [t + language_tone_start_map[lang] for t, lang in zip(tones, langs)]
    lang_ids = [language_id_map[lang] for lang in langs]
    return phone_ids, tone_ids, lang_ids
