"""Load Chinese pronunciation annotations from the shared external lexicon.

Keep legacy numbered pinyin and reviewed tone-mark pinyin separate: their
pypinyin loading styles and jieba registration policies are intentionally
unchanged. This module contains no maintained word lists.
"""
import json
import re

from .external_lexicon import _unique_object
from .release_paths import frontend_lexicon_path

_MAX_BYTES = 4 * 1024 * 1024
_PHRASE = re.compile(r"[\u4e00-\u9fff]{1,64}\Z")
_NUMBERED = re.compile(r"[a-züv]{1,12}[1-5]?\Z")
_TONE_MARKED = re.compile(r"[a-züāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜńňǹḿ]{1,12}\Z")


def _phrase(value):
    return isinstance(value, str) and _PHRASE.fullmatch(value) is not None


def _pronunciations(value, syllable):
    if not isinstance(value, dict) or len(value) > 50000:
        raise ValueError("Chinese pronunciations require a bounded phrase mapping")
    for phrase, reading in value.items():
        if (not _phrase(phrase) or not isinstance(reading, list)
                or len(reading) != len(phrase)):
            raise ValueError("Chinese pinyin must align one syllable list per character")
        for alternatives in reading:
            if (not isinstance(alternatives, list) or not 1 <= len(alternatives) <= 8
                    or any(not isinstance(item, str) or not syllable.fullmatch(item)
                           for item in alternatives)):
                raise ValueError("Chinese pinyin has an invalid syllable/tone format")
    return value


def load_chinese_lexicon(path=None):
    """Read and validate annotations before Chinese frontend initialization.

    Called at startup, never per word. A missing section is a release mismatch,
    not permission to silently drop pronunciation fixes. Restart the frontend
    after changing data; pypinyin/jieba registrations are process-global.
    """
    with open(frontend_lexicon_path() if path is None else path, "rb") as source:
        raw = source.read(_MAX_BYTES + 1)
    if len(raw) > _MAX_BYTES:
        raise ValueError("External Chinese lexicon exceeds 4 MiB")
    resource = json.loads(raw, object_pairs_hook=_unique_object)
    if (not isinstance(resource, dict) or resource.get("version") != 1
            or resource.get("format") != "vits2-external-pronunciations"):
        raise ValueError("Chinese lexicon requires the shared VITS2 resource format")
    data = resource.get("chinese")
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("Chinese lexicon section missing; install the matching bilingual resource")
    legacy = _pronunciations(data.get("custom_dict"), _NUMBERED)
    reviewed = _pronunciations(data.get("robot_phrase_pinyin"), _TONE_MARKED)
    phrases = data.get("jieba_phrases")
    if (not isinstance(phrases, list) or len(phrases) > 50000
            or any(not _phrase(phrase) for phrase in phrases)
            or len(set(phrases)) != len(phrases)):
        raise ValueError("Chinese segmentation phrases require unique complete phrases")
    return legacy, tuple(phrases), reviewed
