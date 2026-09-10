"""
tests/test_kokoro_direct.py — tokenisation and chunking for the direct ONNX runtime.

The parts that need a 310 MB model are not covered here; the parts that silently
produce *wrong* audio rather than no audio are. Two conventions are copied from
sherpa's own `OfflineTtsKokoroModel::Run` and are easy to get subtly wrong:

  - tokens are wrapped in a leading and trailing 0;
  - the style row is indexed by the **inner** token count, `styles[sid][len(ids)]`,
    so the 510 axis in voices.bin is a length table and not padding.

Getting either wrong yields audio that plays and sounds off, which is exactly the
class of bug this whole Japanese effort has been chasing.

Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from plugins import kokoro_direct as kd  # noqa: E402


# ── tokens.txt parsing ────────────────────────────────────────────────────────

def test_the_space_token_is_not_mistaken_for_a_malformed_line(tmp_path):
    """`tokens.txt` has a line whose token IS a space, so split from the right.

    Splitting from the left drops the space token, and a dropped space collapses
    every word boundary in the utterance.
    """
    path = tmp_path / "tokens.txt"
    path.write_text("; 1\n: 2\n  3\na 4\n", encoding="utf-8")
    table = kd._read_tokens(str(path))
    assert table[";"] == 1
    assert table[" "] == 3, table
    assert table["a"] == 4


def test_blank_and_malformed_lines_are_skipped(tmp_path):
    path = tmp_path / "tokens.txt"
    path.write_text("a 1\n\nnot-an-id\nb 2\n", encoding="utf-8")
    assert kd._read_tokens(str(path)) == {"a": 1, "b": 2}


def test_an_empty_token_file_is_an_error(tmp_path):
    path = tmp_path / "tokens.txt"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no tokens"):
        kd._read_tokens(str(path))


# ── chunking ──────────────────────────────────────────────────────────────────

def test_short_input_is_one_chunk():
    assert kd.chunk_ids([1, 2, 3], 10) == [[1, 2, 3]]
    assert kd.chunk_ids([], 10) == [[]]


def test_long_input_is_split_rather_than_truncated():
    """sherpa hard-exits past the style table; losing the sentence is worse."""
    ids = list(range(1, 1201))
    chunks = kd.chunk_ids(ids, 509)
    assert sum(len(c) for c in chunks) == len(ids), "audio was lost"
    assert [t for c in chunks for t in c] == ids, "order changed"
    assert all(len(c) <= 509 for c in chunks)


def test_a_split_prefers_punctuation_over_mid_word():
    """A seam at a comma is inaudible; a seam mid-word is not."""
    period = 99
    # 60 tokens, with a period at index 40 — the only sensible place to cut.
    ids = [1] * 40 + [period] + [2] * 19
    chunks = kd.chunk_ids(ids, 50, break_ids={period})
    assert chunks[0][-1] == period, chunks[0][-5:]
    assert len(chunks[0]) == 41


def test_a_clause_longer_than_the_limit_still_splits():
    """No punctuation anywhere: fall back to a hard cut rather than looping."""
    ids = [7] * 1000
    chunks = kd.chunk_ids(ids, 100, break_ids={99})
    assert sum(len(c) for c in chunks) == 1000
    assert all(len(c) <= 100 for c in chunks)


def test_a_break_too_early_in_the_window_is_ignored():
    """Cutting at token 2 of 500 would make a chunk of two tokens and stall."""
    period = 99
    ids = [1, 1, period] + [2] * 200
    chunks = kd.chunk_ids(ids, 100, break_ids={period})
    assert len(chunks[0]) > 3, chunks[0][:6]


# ── the constants that describe the release ───────────────────────────────────

def test_the_style_table_shape_matches_the_shipped_voices_bin():
    """28 200 960 bytes = 54 speakers x 510 lengths x 256 floats, exactly."""
    assert kd.STYLE_LENGTHS == 510
    assert kd.STYLE_DIM == 256
    assert 28_200_960 == 54 * kd.STYLE_LENGTHS * kd.STYLE_DIM * 4


def test_the_model_rate_is_the_one_the_resampler_expects():
    from plugins import tts
    assert kd.SAMPLE_RATE == tts.KOKORO_SAMPLE_RATE == 24000


# ── encode ────────────────────────────────────────────────────────────────────

class _FakeDirect(kd.KokoroDirect):
    """Only the token map, so encode() is testable without a 310 MB model."""

    def __init__(self, table):
        self._token_to_id = table


def test_encode_reports_unknown_phonemes_instead_of_dropping_them():
    """sherpa drops them silently; that is the bug this module exists for.

    The Japanese table is asserted to be fully in-vocabulary elsewhere, so in
    practice `unknown` is always empty — but if it ever is not, the caller finds out.
    """
    direct = _FakeDirect({"a": 1, "b": 2})
    ids, unknown = direct.encode("aab")
    assert ids == [1, 1, 2] and unknown == []

    ids, unknown = direct.encode("axb")
    assert ids == [1, 2], "known phonemes must still be encoded"
    assert unknown == ["x"], "the miss must be reported, not swallowed"


def test_encode_of_empty_input():
    assert _FakeDirect({"a": 1}).encode("") == ([], [])


def test_the_japanese_table_encodes_with_no_unknowns():
    """End-to-end over the vocabulary, with no model: every phoneme maps."""
    from plugins import ja_phonemes
    from test_ja_phonemes import KOKORO_VOCAB

    direct = _FakeDirect({c: i + 1 for i, c in enumerate(sorted(KOKORO_VOCAB))})
    for text in ["コンニチハ", "キョウハニセンニジュウロクネンジュウガツツイタチデス",
                 "ワタシハシャオ・ファントモウシマス", "ナニカオテツダイシマショウカ"]:
        _, unknown = direct.encode(ja_phonemes.kana_to_phonemes(text))
        assert unknown == [], (text, unknown)


# ── the session must never ask for CUDA ───────────────────────────────────────

def test_the_session_is_built_on_cpu_and_cannot_be_asked_for_cuda(tmp_path,
                                                                  monkeypatch):
    """CUDA here is safe only in a process with no other ONNX Runtime in it.

    Two runtimes share one provider bridge holding a single `ProviderHost` pointer, so
    whichever builds the *second* CUDA session on this graph dies with "Could not find
    OrtValue with name '/Squeeze_2_output_0'" — an exception on jp6.1, a SIGSEGV that
    kills all of perception on jp5.11. Measured both ways round on both rigs.

    This shipped once, because the code requested CUDA while a stale docstring asserted
    the request was inert. `provider` exists again now that `plugins/kokoro_worker.py`
    provides a process with nothing else in it — so the invariant has moved rather than
    disappeared, and this asserts where it moved to:

      - the **default** is still CPU, so anything constructing this without thinking
        gets the safe thing;
      - the only caller that passes `provider=` is the worker child.

    A CPU-only session loads no CUDA provider at all, so it never touches the bridge —
    which is why `japanese_worker: false` and `japanese_worker_device: cpu` are both
    safe configurations rather than merely slower ones.
    """
    import ast
    import inspect
    from pathlib import Path

    signature = inspect.signature(kd.KokoroDirect.__init__)
    assert signature.parameters["provider"].default == "cpu", (
        "the default must stay CPU: a caller that does not think about it is in "
        "perception's process, where CUDA here corrupts sherpa's sessions")

    perception_root = Path(__file__).resolve().parents[1]
    passing = set()
    for path in (perception_root / "plugins").rglob("*.py"):
        tree = ast.parse(path.read_text("utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "KokoroDirect"
                    and any(kw.arg == "provider" for kw in node.keywords)):
                passing.add(path.name)
    assert passing <= {"kokoro_worker.py"}, (
        f"only the worker child may ask for a device; also seen in {sorted(passing)}")

    (tmp_path / "tokens.txt").write_text("a 1\n", encoding="utf-8")
    (tmp_path / "voices.bin").write_bytes(
        b"\0" * (kd.STYLE_LENGTHS * kd.STYLE_DIM * 4))
    (tmp_path / "model.onnx").write_bytes(b"not a real model")

    seen = {}

    class _Session:
        def __init__(self, path, opts, providers):
            seen["providers"] = providers

        def get_providers(self):
            return seen["providers"]

    fake_ort = type("ort", (), {"SessionOptions": lambda: type(
        "o", (), {"intra_op_num_threads": 0})(), "InferenceSession": _Session})
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    kd.KokoroDirect(str(tmp_path), "model.onnx")
    assert seen["providers"] == ["CPUExecutionProvider"], seen
