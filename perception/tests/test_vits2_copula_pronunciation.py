"""Contextual wéi readings through released FST/G2P; no acoustic inference."""
import importlib.util
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import pytest
import vision_stubs  # noqa: F401

from plugins.vits2_tts_trt.frontend.copula_context import find_copula_spans, wei2_positions
from plugins.vits2_tts_trt.frontend.chunking import iter_text_chunks


@pytest.fixture(scope="module")
def frontend():
    root = Path(os.getenv("VITS2_MODEL_DIR", "/models/vits2"))
    deps = ("jieba", "pypinyin", "wetext", "kaldifst", "g2p_en", "inflect", "nltk")
    if not (root / "tn_cache").is_dir() or any(importlib.util.find_spec(n) is None for n in deps):
        pytest.skip("requires installed VITS2 frontend resources and dependencies")
    from plugins.vits2_tts_trt.frontend.release_paths import configure_release_paths
    configure_release_paths(root)
    from plugins.vits2_tts_trt.frontend.cleaner import normalize_text_mix, g2p_normalized_text_mix
    from plugins.vits2_tts_trt.adapter import Vits2TensorRTAdapter
    from plugins.vits2_tts_trt.runtime.backends.trt_numpy_tts_engine import TensorRTNumpyTTSEngine
    engine = TensorRTNumpyTTSEngine.__new__(TensorRTNumpyTTSEngine)
    engine.add_blank = json.loads((root / "config.json").read_text())["data"].get("add_blank", True)
    adapter = Vits2TensorRTAdapter.__new__(Vits2TensorRTAdapter)
    adapter._engine, adapter.max_chunk_tokens = engine, 256
    return SimpleNamespace(normalize=normalize_text_mix, g2p=g2p_normalized_text_mix, adapter=adapter)


def _check_wei(frontend, text, expected_tones, *, already_normalized=False):
    normalized = text if already_normalized else frontend.normalize(text)
    all_phones, all_tones, languages, w2p = frontend.g2p(normalized)
    assert len(all_phones) == len(all_tones) == len(languages) == sum(w2p)
    # Inspect final complete model phones: a Chinese-only rerun would lose
    # the neighboring English quantity whose context this regression tests.
    if re.search(r"[A-Za-z]", normalized):
        from plugins.vits2_tts_trt.frontend.unified_g2p import _mixed_tokens
        positions = []
        for kind, token in _mixed_tokens(normalized):
            positions.extend(list(token) if kind == "ZH" else [None])
    else:
        positions = [char for char in normalized if not char.isspace()]
    assert len(w2p) == len(positions) + 2
    observed = []
    for index, char in enumerate(positions):
        if char != "为":
            continue
        start = sum(w2p[:index + 1])
        end = start + w2p[index + 1]
        assert all_phones[start:end] == ["w", "ei"]
        assert len(set(all_tones[start:end])) == 1
        observed.append(all_tones[start])
    assert observed == expected_tones, (normalized, observed)


@pytest.mark.parametrize("text,tones", [
    ("本次实验的学习率设为 lr=1e‑4，先保持其他参数不变。", [2]),
    ("电流为 1.2A，内存阈值为 8.5GB，信噪比要求 SNR≥28dB。", [2, 2]),
    ("允许误差为 ±0.5℃，超过这个范围就提醒我。", [2]),
    ("合格长度范围为 3–5 mm，小于下限的零件先放到一边。", [2]),
    ("请设为英文。", [2]),
    ("我们没有把参数设为零。", [2]),
    ("电流为零点五安培。", [2]),
    ("电流为负五安培。", [2]),
    ("误差为负零点一。", [2]),
    ("允许误差为万分之一。", [2]),
    ("长度范围为两到三毫米。", [2]),
    ("Hello,请设为 English, please.", [2]),
    ("商品售价为 ¥1,299.50，请核对。", [2]),
    ("本次到账金额为 135.80 元。", [2]),
    ("调整金额为 88.25 元。", [2]),
    ("计算结果为 17。", [2]),
    ("当前温度为 23℃。", [2]),
    ("本次到账金额为正零点五零元。", [2]),
    ("项目编号为 0917，而不是 0927。", [2]),
    ("这台设备的额定电流为 1.2A。", [2]),
    ("Please confirm,内存阈值为 eight point five gigabytes, then continue.", [2]),
    ("电流不为零，先检查连接。", [2]),
    ("输出电压为 24V，请核对。", [2]),
])
def test_numeric_and_assignment_copula_reaches_real_model_tones(frontend, text, tones):
    _check_wei(frontend, text, tones)


@pytest.mark.parametrize("text,tones", [
    ("为了安全，请先停止设备。", [4]),
    ("因为网络不稳定，请稍后重试。", [4]),
    ("这是为你准备的。", [4]),
    ("作为机器人助手，我会继续检查。", [2]),
    ("希望成为你的助手。", [2]),
    ("请记录设备的行为。", [2]),
    ("误差为你带来了麻烦。", [4]),
    ("电流为设备提供能量。", [4]),
    ("这个人设为你量身定制。", [4]),
    ("这项设置为了方便使用。", [4]),
    ("这个结果为你带来便利。", [4]),
    ("温度为设备运行提供保障。", [4]),
    ("售价为你保留到明天。", [4]),
    ("金额为你单独列出。", [4]),
    ("这个结果为负责任的人带来奖励。", [4]),
    ("这个结果为正负样本提供标签。", [4]),
    ("这个结果为 One Piece 提供参考。", [4]),
    ("这个结果为 four students 提供参考。", [4]),
    ("我不为你关闭设备。", [4]),
    ("我为五元而争论。", [4]),
    ("他为三分钟而努力。", [4]),
    ("为一千元以上的损失负责。", [4]),
])
def test_prepositions_and_other_lexical_readings_are_preserved(frontend, text, tones):
    _check_wei(frontend, text, tones)


@pytest.mark.parametrize("text,character,expected_tone", [
    ("不是", "不", 2),
    ("不对", "不", 2),
    ("不用设为零", "不", 2),
    ("一个", "一", 2),
    ("一起", "一", 4),
    ("很好", "很", 2),
    ("可以", "可", 2),
])
def test_existing_connected_speech_tone_sandhi_is_unchanged(frontend, text, character, expected_tone):
    normalized = frontend.normalize(text)
    phones, tones, _, word2ph = frontend.g2p(normalized)
    index = normalized.index(character)
    start = sum(word2ph[:index + 1])
    assert tones[start:start + word2ph[index + 1]] == [expected_tone] * word2ph[index + 1]


def test_negation_is_resolved_before_tone_sandhi(frontend):
    phones, tones, _, _ = frontend.g2p(frontend.normalize("电流不为零。"))
    local = ["b", "u", "w", "ei", "l", "ing"]
    index = next(i for i in range(len(phones) - len(local) + 1)
                 if phones[i:i + len(local)] == local)
    assert tones[index:index + len(local)] == [4, 4, 2, 2, 2, 2]
    phones, tones, _, _ = frontend.g2p(frontend.normalize("我不为你关闭设备。"))
    local = ["b", "u", "w", "ei", "n", "i"]
    index = next(i for i in range(len(phones) - len(local) + 1)
                 if phones[i:i + len(local)] == local)
    assert tones[index:index + len(local)] == [2, 2, 4, 4, 3, 3]


@pytest.mark.parametrize("predicate,value,tone", [
    ("内存阈值为", "eight point five gigabytes", 2),
    ("电流为", "one point two amperes", 2),
    ("我会为", "four students 提供帮助", 4),
    ("这个结果为", "One Piece 提供参考", 4),
])
def test_actual_256_token_chunks_keep_reading_context_and_all_phones(frontend, predicate, value, tone):
    text = ("Please inspect the power supply and carefully record the current operating conditions "
            "before we confirm that " + predicate + " " + value + " and continue checking the remaining devices "
            "without changing any of the settings that were used for the earlier measurements.")
    normalized = frontend.normalize(text)
    full_phones, full_tones, _, _ = frontend.g2p(normalized)
    chunks = list(frontend.adapter.iter_text_chunks(text))
    assert len(chunks) >= 2
    assert ''.join(chunk for chunk, _ in chunks) == normalized
    assert all(len(ids[0]) <= 256 for _, ids in chunks)
    target_chunks = [chunk for chunk, _ in chunks if "为" in chunk]
    assert len(target_chunks) == 1
    assert value in target_chunks[0]
    _check_wei(frontend, target_chunks[0], [tone], already_normalized=True)
    phonemes, tones = [], []
    for chunk, ids in chunks:
        ph, tn, langs, w2p = frontend.g2p(chunk)
        assert len(ph) == len(tn) == len(langs) == sum(w2p)
        assert len(ids[0]) == len(ids[1]) == len(ids[2])
        phonemes.extend(ph[1:-1])
        tones.extend(tn[1:-1])
    assert phonemes == full_phones[1:-1]
    assert tones == full_tones[1:-1]


def test_actual_model_budget_rejects_an_oversized_protected_context(frontend, monkeypatch):
    monkeypatch.setattr(frontend.adapter, 'max_chunk_tokens', 8)
    with pytest.raises(ValueError, match='Protected copula expression'):
        list(frontend.adapter.iter_text_chunks('内存阈值为 eight point five gigabytes.'))


@pytest.mark.parametrize('text', [
    '请核对金额。', '余额不足。', '请确认名额和总额。', '本次到账属于巨额转账。',
    '请核对超额部分和定额。', '金额为正零点五零元，请核对。',
    '这笔调整金额为 -¥12.3456，请保留四位小数。',
    '请读出金额￥１２.３.４，保留原文。',
])
def test_lexical_currency_e_reaches_model_with_dictionary_tone(frontend, text):
    normalized = frontend.normalize(text)
    phones, tones, languages, word2ph = frontend.g2p(normalized)
    assert len(phones) == len(tones) == len(languages) == sum(word2ph)
    positions = [i for i in range(len(phones) - 1) if phones[i:i + 2] == ['EE', 'e']]
    assert positions, normalized
    assert all(tones[i:i + 2] == [2, 2] for i in positions), (normalized, tones)


@pytest.mark.parametrize('word,pos,finals,expected', [
    ('金额', 'n', ['in1', 'e2'], ['in1', 'e2']),
    ('巨额', 'b', ['v4', 'e2'], ['v4', 'e2']),
    ('超额', 'v', ['ao1', 'e2'], ['ao1', 'e2']),
    ('额', 'e', ['e2'], ['e5']),
    ('额', 'n', ['e2'], ['e5']),
    ('好额', 'e', ['ao3', 'e2'], ['ao3', 'e5']),
    ('呃', 'e', ['e5'], ['e5']),
])
def test_lexical_e_guard_preserves_existing_interjections(word, pos, finals, expected):
    from plugins.vits2_tts_trt.frontend.tone_sandhi import ToneSandhi
    assert ToneSandhi()._neural_sandhi(word, pos, finals) == expected


# Pure context and chunk-budget checks do not require the frontend fixture.
@pytest.mark.parametrize('text', [
    '到账金额为正零点五零元,请核对.', '项目编号为零九一七,而不是零九二七.',
    '额定电流为一点二安培.', '内存阈值为 eight point five gigabytes, then continue.',
    '电流不为零,请检查.', '电流不为 one point two amperes, then continue.',
    '残差为负十二点三四五六伏特.', '允许误差为正负零点五摄氏度.',
    '长度范围为三到五毫米.', '合格比例为三分之二.',
    '结果为 two thirds.', '数量为 one thousand two hundred and ninety nine.',
    '预算为 eight dollars and twenty five cents.', '请设为英文.',
    '请将默认语言设为 English.', '本次实验的学习率设为 L R 等于一乘以十的负四次方.',
])
def test_complete_scalar_or_explicit_assignment_selects_wei2(text):
    assert wei2_positions(text) == frozenset(i for i,c in enumerate(text) if c == '为')


@pytest.mark.parametrize('text', [
    '为了安全,请停止.', '因为网络不稳定.', '我会为你保存记录.', '我不为你关闭设备.',
    '这个人设为你量身定制.', '这项设置为了方便使用.',
    '这个结果为负责任的人带来奖励.', '这个结果为正负样本提供标签.',
    '这个结果为 One Piece 提供参考.', '这个结果为 four students 提供参考.',
    '我为 one dollar 购买记录.', '因为 one,请继续.',
    '我为五元而争论.', '他为三分钟而努力.', '为一千元以上的损失负责.',
])
def test_prepositions_names_and_incomplete_quantity_prefixes_do_not_select_wei2(text):
    assert wei2_positions(text) == frozenset()


def test_context_is_immutable_and_independent_between_requests():
    first = wei2_positions('阈值为 eight gigabytes.')
    assert isinstance(first, frozenset)
    assert wei2_positions('我会为你保存记录.') == frozenset()
    assert first == frozenset({2})


def test_guard_span_does_not_turn_one_piece_into_a_number_after_chunking():
    text = '请认真检查相关记录' * 8 + '这个结果为 One Piece 提供参考.'
    guards = find_copula_spans(text)
    assert guards and not any(span.tone2 for span in guards)
    chunks = list(iter_text_chunks(text, len, 40))
    assert ''.join(chunks) == text
    assert all(not wei2_positions(chunk) for chunk in chunks)
    assert any('为 One Piece' in chunk for chunk in chunks)


def test_short_quantity_remains_with_copula_at_language_boundary():
    text = 'Please check all the previous records and keep their order ' * 4
    text += '内存阈值为 eight point five gigabytes and continue checking the remaining devices.'
    chunks = list(iter_text_chunks(text, len, 64))
    assert ''.join(chunks) == text
    assert all(len(chunk) <= 64 for chunk in chunks)
    assert any('为 eight point five gigabytes' in chunk for chunk in chunks)
    assert sum(len(wei2_positions(chunk)) for chunk in chunks) == 1


def test_a_protected_quantity_too_large_for_the_budget_is_explicitly_rejected():
    text = '内存阈值为 ' + 'one ' * 100 + 'gigabytes.'
    with pytest.raises(ValueError, match='Protected copula expression'):
        list(iter_text_chunks(text, len, 64))
