"""
tests/test_speaker_runtime.py — 输入量化与长段换人检测，都不碰模型。

The two things this file covers are the two things Phase 0's measurements forced
into the design, so they are the two things a future change is most likely to undo
by accident:

* **Shape quantisation.** Unquantised input lengths make ONNX Runtime's CPU arena
  grow without bound (measured: fixed shape 95 calls → +0.0 MB; real VAD segments
  100 calls → still +60 MB and climbing). `reachable_shapes()` is asserted here so
  that widening GRID_S/MAX_S cannot silently reintroduce it.
* **The speaker-change check.** A VAD segment is not guaranteed to hold one voice,
  and a blend published as one person looks correct, which is worse than no answer.

`SpeakerEmbedder` itself needs sherpa-onnx and 28 MB of weights, neither of which a
dev host has, so the embedder is faked down to its one real boundary: `embed`.
Everything above that — conditioning, windowing, the coherence arithmetic — is the
code under test and runs for real.

Pure host-side: numpy only.
Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import numpy as np
import pytest

from vision_stubs import PERCEPTION_ROOT  # noqa: F401  (puts perception on sys.path)

from plugins.speaker_runtime import (  # noqa: E402
    GRID_S,
    MAX_S,
    SAMPLE_RATE,
    SPEAKER_MODELS,
    SpeakerEmbedder,
    grid_samples,
    reachable_shapes,
    speaker_bundle_for,
)

GRID = int(SAMPLE_RATE * GRID_S)
CAP = int(SAMPLE_RATE * MAX_S)


# ── 输入量化 ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("seconds,expected_s", [
    (0.1, 0.5), (0.5, 0.5), (0.51, 1.0), (1.0, 1.0), (2.3, 2.5),
    (5.9, 6.0), (6.0, 6.0), (30.0, 6.0), (75.0, 6.0),
])
def test_grid_rounds_up_and_caps(seconds, expected_s):
    assert grid_samples(int(SAMPLE_RATE * seconds)) == int(SAMPLE_RATE * expected_s)


def test_every_length_lands_on_one_of_a_small_fixed_set():
    """这是 arena 不无限增长的全部理由：可达 shape 必须是个小的有限集。

    原始段长 102 个不同值时实测 jp5.11 上内存还在涨；量化后塌缩到 4 个。这里用
    一万个随机长度（含超长）确认不会漏出第 13 个 shape。
    """
    rng = np.random.default_rng(0)
    lengths = rng.integers(1, SAMPLE_RATE * 90, size=10_000)
    shapes = {grid_samples(int(n)) for n in lengths}
    assert len(shapes) <= reachable_shapes()
    assert reachable_shapes() == 12, (
        "GRID_S/MAX_S 变了：可达 shape 数就是 ORT arena 的上界，改之前先读 "
        "speaker_runtime 的模块文档"
    )
    assert max(shapes) == CAP
    assert all(shape % GRID == 0 for shape in shapes)


def test_grid_never_returns_zero_or_negative():
    for n in (-5, 0, 1):
        assert grid_samples(n) == GRID


def test_condition_pads_with_silence_and_keeps_the_leading_audio():
    samples = np.full(int(SAMPLE_RATE * 1.2), 0.5, dtype=np.float32)
    out = SpeakerEmbedder.condition(samples)
    assert len(out) == int(SAMPLE_RATE * 1.5)
    assert np.allclose(out[: len(samples)], 0.5)
    assert np.allclose(out[len(samples):], 0.0)        # 补的是零，不是重复
    assert out.dtype == np.float32


def test_condition_truncates_rather_than_downsamples():
    samples = np.arange(SAMPLE_RATE * 10, dtype=np.float32)
    out = SpeakerEmbedder.condition(samples)
    assert len(out) == CAP
    # 截断保留的是开头，不是重采样后的整段 —— 后者会改变说话人特征
    assert np.array_equal(out, samples[:CAP])


def test_pcm16_roundtrip_and_odd_byte_tolerance():
    pcm = np.array([0, 16384, -16384, 32767, -32768], dtype="<i2").tobytes()
    out = SpeakerEmbedder.pcm16_to_float(pcm)
    assert out.dtype == np.float32
    assert out[0] == 0.0
    assert out[1] == pytest.approx(0.5, abs=1e-4)
    assert out[2] == pytest.approx(-0.5, abs=1e-4)
    # 奇数字节不能抛 —— DDS 上来的块偶尔会被截断，而丢一个样本比丢一句话好
    assert len(SpeakerEmbedder.pcm16_to_float(pcm + b"\x01")) == 5


# ── 注册表 ───────────────────────────────────────────────────────────────────

def test_default_model_is_in_the_registry_and_declares_its_bundle():
    from plugins.speaker_runtime import DEFAULT_SPEAKER_MODEL
    spec = SPEAKER_MODELS[DEFAULT_SPEAKER_MODEL]
    assert spec["weights"].endswith(".onnx")
    bundle, directory = speaker_bundle_for(DEFAULT_SPEAKER_MODEL, "")
    assert bundle == spec["bundle"]
    assert directory == spec["dir"]


def test_every_registry_entry_has_pinned_bytes_in_the_downloader():
    """注册表里有而下载器里没 pin 的模型，会在机器人上下载到一半才失败。"""
    from utils.model_downloader import SPEAKER_MODEL_BUNDLES
    for name, spec in SPEAKER_MODELS.items():
        assert spec["bundle"] in SPEAKER_MODEL_BUNDLES, name
        _base, files = SPEAKER_MODEL_BUNDLES[spec["bundle"]]
        assert spec["weights"] in files, f"{name} 的权重文件没有 pin"
        entry = files[spec["weights"]]
        assert entry["size"] > 0 and len(entry["sha256"]) == 64


def test_unknown_model_names_the_alternatives():
    with pytest.raises(ValueError, match="unknown speaker model"):
        speaker_bundle_for("no-such-model", "")


def test_model_dir_override_wins_over_the_registry_default():
    _bundle, directory = speaker_bundle_for("campplus_zh_en", "/models/elsewhere")
    assert directory == "/models/elsewhere"


# ── 长段换人检测 ─────────────────────────────────────────────────────────────

class _FakeEmbedder(SpeakerEmbedder):
    """Only `embed` is faked; `condition`/`embed_windowed` are the real code.

    Speaker A is "positive samples", speaker B is "negative samples", and a clip
    containing both embeds to a **third, orthogonal** direction — not to a point
    between A and B.

    That is deliberate and it is what the measurements say. If a blend sat
    between its two speakers, cosine(window, whole clip) would stay high
    (linearly, a clip that is 2/3 speaker A gives 0.89) and no threshold could
    separate "two voices" from "one voice, slightly varying". The real numbers
    are not like that: on 73 long VAD segments from Orin 5, cosine(full, leading
    3 s) had p05 **0.197** and min **0.157** — a window and its own parent clip
    came out essentially unrelated. A blend is its own direction in embedding
    space, so that is what the fake does.
    """

    _A = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _B = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    _MIXED = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    def __init__(self):            # noqa: D107 - no model, no super().__init__
        self._dim = 4
        self.calls: list[int] = []

    def embed(self, samples, sample_rate=SAMPLE_RATE):
        conditioned = self.condition(samples)
        self.calls.append(len(conditioned))
        # 只看非静音部分：condition() 补的零既不属于 A 也不属于 B，把它算进去
        # 会让「补零比例」而不是「有几个说话人」决定结果。
        voiced = conditioned[np.abs(conditioned) > 1e-6]
        if voiced.size == 0:
            return self._MIXED.copy()
        negative = float(np.count_nonzero(voiced < 0)) / voiced.size
        if negative < 0.05:
            return self._A.copy()
        if negative > 0.95:
            return self._B.copy()
        return self._MIXED.copy()


def clip(seconds: float, flip_after: float | None = None) -> np.ndarray:
    n = int(SAMPLE_RATE * seconds)
    out = np.full(n, 0.5, dtype=np.float32)
    if flip_after is not None:
        out[int(SAMPLE_RATE * flip_after):] = -0.5
    return out


def test_short_clip_is_trusted_without_a_second_embedding():
    embedder = _FakeEmbedder()
    _vector, coherence = embedder.embed_windowed(clip(2.0), split_above_s=4.0)
    assert coherence is None, "短段不该付第二次 embedding 的代价"
    assert len(embedder.calls) == 1


def test_long_coherent_clip_reports_high_coherence():
    embedder = _FakeEmbedder()
    _vector, coherence = embedder.embed_windowed(clip(5.0), split_above_s=4.0)
    assert coherence is not None
    assert coherence > 0.9
    assert len(embedder.calls) == 2, "恰好两次 embedding，不是滑窗 N 次"


def test_long_clip_with_a_speaker_change_reports_low_coherence():
    embedder = _FakeEmbedder()
    _vector, coherence = embedder.embed_windowed(
        clip(6.0, flip_after=4.0), split_above_s=4.0)
    assert coherence is not None
    assert coherence < 0.8, (
        "前 4 秒是一个人、之后换人的段必须被判为低一致性 —— 否则两个人的混合会"
        "被当成一个人发布出去，而那看起来是对的"
    )


def test_windowed_returns_the_window_not_the_blend():
    """返回的是窗口的 embedding：一致时两者等价，不一致时窗口是更好的那个猜测。"""
    embedder = _FakeEmbedder()
    vector, _ = embedder.embed_windowed(clip(6.0, flip_after=4.0),
                                        split_above_s=4.0)
    lead_only = embedder.embed(clip(4.0))
    assert np.allclose(vector, lead_only)


def test_window_width_is_also_quantised():
    """窗口本身也要落在网格上，否则换人检测把 shape 数量翻倍。"""
    embedder = _FakeEmbedder()
    embedder.embed_windowed(clip(5.3), split_above_s=4.0)
    assert all(width % GRID == 0 for width in embedder.calls)
    assert all(width <= CAP for width in embedder.calls)
