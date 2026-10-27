"""Reviewed Chinese phrase readings through the release FST/G2P/token path.

Requires an installed VITS2 frontend release; no acoustic engine is created.
"""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import vision_stubs  # noqa: F401


def _chinese_resource():
    return {
        "format": "vits2-external-pronunciations", "version": 1,
        "chinese": {"version": 1,
                    "custom_dict": {"目的地": [["mu4"], ["di4"], ["di4"]]},
                    "jieba_phrases": ["慢慢地"],
                    "robot_phrase_pinyin": {"重试": [["chóng"], ["shì"]]}},
    }


def test_chinese_annotations_use_shared_resource_and_keep_registration_groups(tmp_path, monkeypatch):
    from plugins.vits2_tts_trt.frontend import heteronym

    path = tmp_path / "external_lexicon.json"
    data = _chinese_resource()
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(heteronym, "frontend_lexicon_path", lambda: path)
    # An older explicit English-only dictionary must not replace Chinese data.
    monkeypatch.setenv("VITS2_LEXICON_PATH", str(tmp_path / "english-only.json"))
    legacy, phrases, reviewed = heteronym.load_chinese_lexicon()
    assert legacy == data["chinese"]["custom_dict"]
    assert phrases == ("慢慢地",)
    assert reviewed == data["chinese"]["robot_phrase_pinyin"]
    legacy["目的地"][0][0] = "bad"
    assert heteronym.load_chinese_lexicon()[0] == data["chinese"]["custom_dict"]


@pytest.mark.parametrize("invalid", [
    "missing", "version", "alignment", "latin_phrase", "bad_pinyin", "flat_pinyin",
    "wrong_style", "duplicate_segmentation", "missing_segmentation", "duplicate_json",
])
def test_invalid_chinese_resource_fails_before_dictionary_registration(tmp_path, invalid):
    from plugins.vits2_tts_trt.frontend.heteronym import load_chinese_lexicon

    data = _chinese_resource()
    section = data["chinese"]
    if invalid == "missing":
        del data["chinese"]
    elif invalid == "version":
        section["version"] = 2
    elif invalid == "alignment":
        section["custom_dict"]["目的地"].pop()
    elif invalid == "latin_phrase":
        section["custom_dict"] = {"A": [["a1"]]}
    elif invalid == "bad_pinyin":
        section["custom_dict"]["目的地"][0] = ["mu9"]
    elif invalid == "flat_pinyin":
        section["custom_dict"]["目的地"] = ["mu4", "di4", "di4"]
    elif invalid == "wrong_style":
        section["robot_phrase_pinyin"]["重试"] = [["chong2"], ["shi4"]]
    elif invalid == "duplicate_segmentation":
        section["jieba_phrases"] *= 2
    elif invalid == "missing_segmentation":
        del section["jieba_phrases"]
    path = tmp_path / "external_lexicon.json"
    raw = json.dumps(data)
    if invalid == "duplicate_json":
        raw = raw[:-1] + ', "chinese": {}}'
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError):
        load_chinese_lexicon(path)


@pytest.fixture(scope="module")
def frontend():
    root = Path(os.getenv("VITS2_MODEL_DIR", "/models/vits2"))
    deps = ("jieba", "pypinyin", "wetext", "kaldifst", "g2p_en", "inflect", "nltk")
    if not (root / "tn_cache").is_dir() or any(importlib.util.find_spec(n) is None for n in deps):
        pytest.skip("requires installed VITS2 frontend resources and dependencies")
    from plugins.vits2_tts_trt.frontend.release_paths import configure_release_paths

    configure_release_paths(root)
    from plugins.vits2_tts_trt.frontend import chinese
    from plugins.vits2_tts_trt.frontend.cleaner import normalize_text_mix, g2p_normalized_text_mix
    from plugins.vits2_tts_trt.frontend.symbols import symbols, num_tones, num_languages
    from plugins.vits2_tts_trt.adapter import Vits2TensorRTAdapter
    from plugins.vits2_tts_trt.runtime.backends.trt_numpy_tts_engine import TensorRTNumpyTTSEngine

    engine = TensorRTNumpyTTSEngine.__new__(TensorRTNumpyTTSEngine)
    engine.add_blank = bool(json.loads((root / "config.json").read_text())["data"].get("add_blank", True))
    adapter = Vits2TensorRTAdapter.__new__(Vits2TensorRTAdapter)
    adapter._engine, adapter.max_chunk_tokens = engine, 256
    return SimpleNamespace(chinese=chinese, normalize=normalize_text_mix,
                           g2p=g2p_normalized_text_mix, adapter=adapter,
                           limits=(len(symbols), num_tones, num_languages))


def _check_character(frontend, text, character, expected_phones, tone):
    normalized = frontend.normalize(text)
    phones, tones, languages, word2ph = frontend.g2p(normalized)
    assert len(phones) == len(tones) == len(languages) == sum(word2ph)
    assert len(word2ph) == len(normalized) + 2
    index = normalized.index(character)
    start = 1 + sum(word2ph[1:index + 1])
    end = start + word2ph[index + 1]
    assert phones[start:end] == expected_phones
    assert tones[start:end] == [tone] * len(expected_phones)
    chunks = list(frontend.adapter.iter_text_chunks(text))
    assert chunks
    for _, ids in chunks:
        assert len(ids[0]) == len(ids[1]) == len(ids[2])
        assert all(0 <= value < limit for values, limit in zip(ids, frontend.limits) for value in values)


@pytest.mark.parametrize("phrase,sentence,character,phones,tone", [
    ("长时间", "请长时间保持运行。", "长", ["ch", "ang"], 2),
    ("长按", "请长按按钮。", "长", ["ch", "ang"], 2),
    ("重试", "连接失败后请重试。", "重", ["ch", "ong"], 2),
    ("重连", "网络断开后正在重连。", "重", ["ch", "ong"], 2),
    ("重发", "请重发消息。", "重", ["ch", "ong"], 2),
    ("调高", "请调高音量。", "调", ["t", "iao"], 2),
    ("调低", "请调低音量。", "调", ["t", "iao"], 2),
    ("调参", "我们正在调参。", "调", ["t", "iao"], 2),
    ("这行代码", "这行代码需要检查。", "行", ["h", "ang"], 2),
])
@pytest.mark.parametrize("context", [False, True])
def test_robot_phrase_reading_reaches_actual_model_ids(frontend, phrase, sentence, character, phones, tone, context):
    _check_character(frontend, sentence if context else phrase, character, phones, tone)


@pytest.mark.parametrize("text,character,phones,tone", [
    ("长大", "长", ["zh", "ang"], 3),
    ("增长", "长", ["zh", "ang"], 3),
    ("队长", "长", ["zh", "ang"], 3),
    ("重量", "重", ["zh", "ong"], 4),
    ("重要", "重", ["zh", "ong"], 4),
    ("尊重", "重", ["zh", "ong"], 4),
    ("重复", "重", ["ch", "ong"], 2),
    ("重庆", "重", ["ch", "ong"], 2),
    ("调度任务", "调", ["d", "iao"], 4),
    ("声调", "调", ["d", "iao"], 4),
    ("音调", "调", ["d", "iao"], 4),
    ("银行", "行", ["h", "ang"], 2),
    ("行走", "行", ["x", "ing"], 2),
    ("执行代码", "行", ["x", "ing"], 2),
    ("这行不行", "行", ["x", "ing"], 2),
    ("加重发热", "重", ["zh", "ong"], 4),
    ("体重试验", "重", ["zh", "ong"], 4),
    ("曲调高昂", "调", ["d", "iao"], 4),
    ("声调低沉", "调", ["d", "iao"], 4),
])
def test_reviewed_phrases_do_not_override_other_word_contexts(frontend, text, character, phones, tone):
    _check_character(frontend, text, character, phones, tone)


@pytest.mark.parametrize("text", [
    "稳定地行走", "安全地行走", "请慢慢地走", "请轻轻地放下", "用力地敲门",
    "满怀信心地行走", "安静地走", "小心地行走", "轻轻地",
    "慢慢地", "悄悄地走", "静静地等待", "默默地工作", "偷偷地看",
    "渐渐地靠近", "缓缓地升起", "狠狠地敲", "好好地学习",
    "高高兴兴地回家", "认认真真地检查", "小心翼翼地走", "不紧不慢地走",
    "非常认真地检查", "十分轻柔地放下", "迅速地回答",
])
def test_adverbial_particle_de_uses_schwa_and_neutral_tone(frontend, text):
    _check_character(frontend, text, "地", ["d", "e"], 5)


@pytest.mark.parametrize("text", [
    "地", "土地", "大地", "天地", "目的地", "地方", "地道", "地铁", "扫地", "遍地",
    "地面", "当地", "地名", "这片地", "一块地", "那块地",
    "原地", "本地", "各地", "满地", "就地", "落地", "擦地",
    "平地", "草地", "所在地", "根据地", "不毛之地", "一席之地",
    "惊天动地", "欢天喜地", "脚踏实地", "设身处地", "开天辟地",
    "占领高地", "肥沃的土地", "地地道道",
    "木樨地", "低洼地", "平整土地", "撂荒地", "苍茫大地", "盐硷地",
])
def test_lexical_di_does_not_become_particle_de_or_neutral_tone(frontend, text):
    _check_character(frontend, text, "地", ["d", "i"], 4)


def test_de_particle_does_not_change_other_de_characters_or_preserved_aa(frontend):
    _check_character(frontend, "吃得好", "得", ["d", "e"], 5)
    _check_character(frontend, "机器人的电量", "的", ["d", "e"], 5)
    _check_character(frontend, "你好啊", "啊", ["AA", "a"], 5)


@pytest.mark.parametrize("phrase", [
    "长时间", "重发", "调高", "这行代码", "稳定地行走", "慢慢地走",
    "轻轻地放下", "静静地等待", "目的地", "脚踏实地",
])
def test_mixed_chinese_run_uses_same_phrase_reading(frontend, phrase):
    expected, expected_tones, _, _ = frontend.g2p(frontend.normalize(phrase))
    expected, expected_tones = expected[1:-1], expected_tones[1:-1]
    mixed = frontend.normalize("Hello," + phrase + ",please.")
    phones, tones, languages, _ = frontend.g2p(mixed)
    assert any(phones[i:i + len(expected)] == expected
               and tones[i:i + len(expected)] == expected_tones
               and set(languages[i:i + len(expected)]) == {"ZH"}
               for i in range(len(phones) - len(expected) + 1))
