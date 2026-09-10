#!/usr/bin/env python3
"""
plugins/kokoro_direct.py — run Kokoro's ONNX graph ourselves, with our own phonemes.

sherpa-onnx offers no way to hand a model phonemes. Its Kokoro frontend phonemises
with espeak-ng and looks the result up in Kokoro's vocabulary, which was built for
**misaki**; the two alphabets disagree and everything that misses is silently dropped.
For Japanese that was 12 phonemes in one sentence, audible as holes. The only
phoneme-level entry point is the lexicon, and it is unreachable — `lang` falls back to
`meta_data.voice`, which the metadata macro forbids from being empty.

So this module bypasses sherpa for the one case that needs it, exactly as
`plugins/vits2_tts_trt/` already bypasses it for VITS2: own runtime, own frontend, own
tokenisation. Everything else keeps going through `KokoroTTSAdapter`.

The graph is small and fully specified:

    tokens  int64 [1, seq]     style  float [1, 256]     speed  float [1]
        -> audio  float [audio_length]                   at 24 kHz

Two conventions are copied from sherpa's own `OfflineTtsKokoroModel::Run`, because
getting either wrong produces audio that is merely *wrong* rather than absent:

  - tokens are wrapped in a leading and trailing 0;
  - the style vector is indexed by **speaker and by inner token count** —
    `styles[sid][len(ids)]` — so the 510 axis in voices.bin is not padding.
"""

from __future__ import annotations

import logging
import os

import numpy as np

log = logging.getLogger(__name__)

# voices.bin is (n_speakers, 510, 1, 256) float32. 510 is the maximum inner token
# count the style table covers; sherpa hard-exits above it, so we refuse instead.
STYLE_LENGTHS = 510
STYLE_DIM = 256
SAMPLE_RATE = 24000


class KokoroDirect:
    """A Kokoro session driven by phonemes rather than text.

    Construct once per adapter — the ONNX session and the 27 MB style table are both
    held for the lifetime of the object.
    """

    def __init__(self, model_dir: str, weights: str, num_threads: int = 0,
                 provider: str = "cpu", session_key: str = "tts.kokoro",
                 in_process: bool = False):
        """`num_threads=0` means one per core, which is the difference between usable
        and not on the CPU path.

        The session goes into `plugins/ort_worker.py`'s child unless `in_process` is
        set, because the standalone ONNX Runtime and sherpa-onnx's bundled one corrupt
        each other's sessions through a shared provider bridge whenever both are in one
        process. That module's docstring has the mechanism and the measurements; the
        short version is an exception on jp6.1 and a SIGSEGV that kills all of
        perception on jp5.11.

        `in_process=True` is the degraded fallback the callers use when the child cannot
        be reached. It is only safe on CPU — a CPU-only session loads no CUDA provider,
        so it never touches the bridge — which is why the fallback paths pair it with
        the default `provider="cpu"`.

        Speed, measured on Orin 6 (6 cores) on one 9.35 s Japanese utterance:

            cpu, 2 threads  RTF 1.171   <- slower than real time
            cpu, 4 threads  RTF 0.646
            cpu, 6 threads  RTF 0.521
            cpu, 8 threads  RTF 0.644   <- oversubscribed
            cuda, in the worker          RTF 0.069 warm, 1.65 on the first utterance

        The CPU default was 2 and made Japanese unusable for streaming.
        """
        import onnxruntime as ort

        model_path = os.path.join(model_dir, weights)
        tokens_path = os.path.join(model_dir, "tokens.txt")
        voices_path = os.path.join(model_dir, "voices.bin")
        for path in (model_path, tokens_path, voices_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Kokoro release is incomplete: {path}")

        self._token_to_id = _read_tokens(tokens_path)

        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if provider in ("cuda", "gpu") else ["CPUExecutionProvider"])
        # 0 lets ORT pick one thread per core, which measured fastest; an explicit
        # value is honoured so a busy robot can be told to use fewer.
        threads = int(num_threads or 0)

        self._session_key = None
        if in_process:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            self._session = ort.InferenceSession(model_path, opts,
                                                 providers=providers)
        else:
            from plugins import ort_worker
            self._session = ort_worker.get_worker().load(
                session_key, model_path, providers,
                {"intra_op_num_threads": threads})
            self._session_key = session_key
        actual = self._session.get_providers()

        styles = np.fromfile(voices_path, dtype=np.float32)
        if styles.size % (STYLE_LENGTHS * STYLE_DIM):
            raise RuntimeError(
                f"{voices_path} is {styles.size} floats, not a multiple of "
                f"{STYLE_LENGTHS}x{STYLE_DIM}; it does not match this model")
        self._n_speakers = styles.size // (STYLE_LENGTHS * STYLE_DIM)
        self._styles = styles.reshape(self._n_speakers, STYLE_LENGTHS, STYLE_DIM)
        # Ids of sentence punctuation, so a long utterance can be split at a comma
        # rather than mid-word. Read from the release rather than hardcoded.
        self._break_ids = {self._token_to_id[c] for c in ".,;:!?"
                           if c in self._token_to_id}

        log.info("[tts] kokoro_direct ready: %s, %d tokens, %d speakers, providers=%s",
                 os.path.basename(model_path), len(self._token_to_id),
                 self._n_speakers, actual)

    def close(self) -> None:
        """Release the session.

        Dropping the reference is enough in-process, but not for a session living in
        the ORT worker: a child does not notice its parent's garbage collector, so it
        would stay resident — 310 MB of weights plus its share of the GPU pool — for
        the life of the process. Unload it explicitly and leave the child alive for
        whatever else it holds, face's sessions included.
        """
        key, self._session_key = self._session_key, None
        self._session = None
        if key is not None:
            from plugins import ort_worker
            ort_worker.get_worker().unload(key)

    @property
    def num_speakers(self) -> int:
        return self._n_speakers

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def encode(self, phonemes: str):
        """Phonemes -> token ids, reporting anything the vocabulary cannot hold.

        Returns `(ids, unknown)`. Unknown characters are **not** silently dropped the
        way sherpa drops them — the caller decides, and the tests assert the list is
        empty for everything the Japanese table can produce.
        """
        ids, unknown = [], []
        for ch in phonemes:
            token_id = self._token_to_id.get(ch)
            if token_id is None:
                unknown.append(ch)
                continue
            ids.append(token_id)
        return ids, unknown

    def synthesize(self, phonemes: str, speaker_id: int = 0, speed: float = 1.0):
        """Phoneme string -> float32 waveform at 24 kHz.

        Splits on the style table's capacity rather than truncating: an utterance
        longer than 510 tokens is synthesized in chunks and concatenated, because
        sherpa's equivalent calls SHERPA_ONNX_EXIT(-1) at that boundary and losing a
        long sentence is worse than a seam in it.
        """
        ids, unknown = self.encode(phonemes)
        if unknown:
            log.warning("[tts] kokoro_direct: %d phoneme(s) not in the vocabulary "
                        "and skipped: %s", len(unknown), "".join(sorted(set(unknown))))
        if not ids:
            return np.zeros(0, dtype=np.float32)
        if not 0 <= speaker_id < self._n_speakers:
            raise ValueError(
                f"speaker_id must be 0..{self._n_speakers - 1}, got {speaker_id}")

        pieces = [self._run(chunk, speaker_id, speed)
                  for chunk in chunk_ids(ids, STYLE_LENGTHS - 1, self._break_ids)]
        return np.concatenate(pieces) if len(pieces) > 1 else pieces[0]

    def _run(self, ids, speaker_id: int, speed: float):
        # A leading and trailing 0, and the style row is chosen by the *inner* count —
        # both copied from sherpa's OfflineTtsKokoroModel::Run.
        tokens = np.asarray([[0, *ids, 0]], dtype=np.int64)
        style = self._styles[speaker_id, len(ids)].reshape(1, STYLE_DIM)
        outputs = self._session.run(
            None,
            {"tokens": tokens,
             "style": style.astype(np.float32),
             "speed": np.asarray([speed], dtype=np.float32)},
        )
        return np.asarray(outputs[0], dtype=np.float32).reshape(-1)


def _read_tokens(path: str) -> dict:
    """Parse `tokens.txt`: one `<token> <id>` per line.

    Split from the right, because the token itself may be a space — `" 16"` is the
    space token, not a malformed line.
    """
    table = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            token, _, token_id = line.rpartition(" ")
            if not token_id.lstrip("-").isdigit():
                continue
            table[token] = int(token_id)
    if not table:
        raise RuntimeError(f"{path} yielded no tokens")
    return table


def chunk_ids(ids, limit: int, break_ids=frozenset()):
    """Split into runs of at most `limit`, preferring a punctuation boundary.

    A seam mid-word is audible; a seam at a comma is not. Falls back to a hard split
    only when a single clause exceeds the limit.
    """
    if len(ids) <= limit:
        return [ids]
    chunks, start = [], 0
    while start < len(ids):
        end = min(start + limit, len(ids))
        if end < len(ids):
            window = ids[start:end]
            cut = max((i for i, t in enumerate(window) if t in break_ids), default=-1)
            if cut > limit // 4:
                end = start + cut + 1
        chunks.append(ids[start:end])
        start = end
    return chunks

