# Vendored FireRedVAD runtime (ONNX, no torch dependency).
#
# Source: https://github.com/FireRedTeam/FireRedVAD (Apache License 2.0)
# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu, Wenpeng Li, Kai Huang, Kun Liu)
#
# The DFSMN model is distributed as ONNX. This module replicates
# fireredvad.core.{audio_feat,vad_postprocessor} using only numpy +
# kaldi_native_fbank + onnxruntime so each VAD worker avoids importing torch.

from __future__ import annotations

import logging
import os
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_LENGTH_S = 0.025
FRAME_SHIFT_S = 0.010


class _CMVN:
    """Apply kaldi CMVN stats dumped to npz (means / inverse std)."""

    def __init__(self, npz_path: str):
        stats = np.load(npz_path)
        self.means = stats["means"]
        self.istd = stats["istd"]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.means) * self.istd).astype(np.float32)


class _KaldifeatFbank:
    def __init__(self, num_mel_bins: int = 80):
        import kaldi_native_fbank as knf

        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = SAMPLE_RATE
        opts.frame_opts.frame_length_ms = 25
        opts.frame_opts.frame_shift_ms = 10
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = True
        opts.mel_opts.num_bins = num_mel_bins
        opts.mel_opts.debug_mel = False
        self._opts = opts
        self._knf = knf

    def __call__(self, wav_np: np.ndarray) -> np.ndarray:
        assert wav_np.ndim == 1
        fbank = self._knf.OnlineFbank(self._opts)
        fbank.accept_waveform(SAMPLE_RATE, wav_np.tolist())
        fbank.input_finished()
        feat = []
        for i in range(fbank.num_frames_ready):
            feat.append(fbank.get_frame(i))
        if not feat:
            return np.zeros((0, self._opts.mel_opts.num_bins), dtype=np.float32)
        return np.vstack(feat).astype(np.float32)


class _VadPostprocessor:
    """Vendored from fireredvad.core.vad_postprocessor (numpy only)."""

    SILENCE, POSSIBLE_SPEECH, SPEECH, POSSIBLE_SILENCE = 0, 1, 2, 3

    def __init__(self, smooth_window_size, prob_threshold, min_speech_frame,
                 max_speech_frame, min_silence_frame, merge_silence_frame,
                 extend_speech_frame):
        self.smooth_window_size = max(1, smooth_window_size)
        self.prob_threshold = prob_threshold
        self.min_speech_frame = min_speech_frame
        self.max_speech_frame = max_speech_frame
        self.min_silence_frame = min_silence_frame
        self.merge_silence_frame = merge_silence_frame
        self.extend_speech_frame = extend_speech_frame

    def process(self, raw_probs):
        if len(raw_probs) == 0:
            return []
        smoothed_probs = self._smooth_prob(raw_probs)
        binary_preds = (np.asarray(smoothed_probs) >= self.prob_threshold).astype(int).tolist()
        decisions = self._smooth_preds_with_state_machine(binary_preds)
        fixed = self._fix_smooth_window_start(decisions)
        merged = self._merge_short_silence_segments(fixed)
        extended = self._extend_speech_segments(merged)
        return self._split_long_speech_segments(extended, raw_probs)

    def decision_to_segment(self, decisions, wav_dur=None):
        segments = []
        speech_start = None
        for t, decision in enumerate(decisions):
            if decision == 1 and speech_start is None:
                speech_start = t
            elif decision == 0 and speech_start is not None:
                segments.append((speech_start * FRAME_SHIFT_S, t * FRAME_SHIFT_S))
                speech_start = None
        if speech_start is not None:
            end_time = len(decisions) * FRAME_SHIFT_S + FRAME_LENGTH_S
            if wav_dur is not None:
                end_time = min(end_time, wav_dur)
            segments.append((speech_start * FRAME_SHIFT_S, end_time))
        return [(round(s, 3), round(e, 3)) for s, e in segments]

    def _smooth_prob(self, probs):
        if self.smooth_window_size <= 1:
            return np.asarray(probs)
        probs_np = np.array(probs)
        kernel = np.ones(self.smooth_window_size) / self.smooth_window_size
        smoothed = np.convolve(probs_np, kernel, mode="full")[: len(probs)]
        for i in range(min(self.smooth_window_size - 1, len(probs))):
            smoothed[i] = np.mean(probs_np[: i + 1])
        return smoothed

    def _smooth_preds_with_state_machine(self, binary_preds):
        if self.min_speech_frame <= 0 and self.min_silence_frame <= 0:
            return binary_preds
        decisions = [0] * len(binary_preds)
        state = self.SILENCE
        speech_start = -1
        silence_start = -1
        for t, is_speech in enumerate(binary_preds):
            if state == self.SILENCE:
                if is_speech:
                    state = self.POSSIBLE_SPEECH
                    speech_start = t
            elif state == self.POSSIBLE_SPEECH:
                if is_speech:
                    if t - speech_start >= self.min_speech_frame:
                        state = self.SPEECH
                        decisions[speech_start:t] = [1] * (t - speech_start)
                else:
                    state = self.SILENCE
                    speech_start = -1
            elif state == self.SPEECH:
                if not is_speech:
                    state = self.POSSIBLE_SILENCE
                    silence_start = t
            elif state == self.POSSIBLE_SILENCE:
                if not is_speech:
                    if t - silence_start >= self.min_silence_frame:
                        state = self.SILENCE
                        speech_start = -1
                else:
                    state = self.SPEECH
                    silence_start = -1
            decisions[t] = 1 if state in (self.SPEECH, self.POSSIBLE_SILENCE) else 0
        return decisions

    def _fix_smooth_window_start(self, decisions):
        new_decisions = decisions.copy()
        for t, decision in enumerate(decisions):
            if t > 0 and decisions[t - 1] == 0 and decision == 1:
                start = max(0, t - self.smooth_window_size)
                new_decisions[start:t] = [1] * (t - start)
        return new_decisions

    def _merge_short_silence_segments(self, decisions):
        if self.merge_silence_frame <= 0:
            return decisions
        new_decisions = decisions.copy()
        silence_start = None
        for t, decision in enumerate(decisions):
            if t > 0 and decisions[t - 1] == 1 and decision == 0 and silence_start is None:
                silence_start = t
            elif t > 0 and decisions[t - 1] == 0 and decision == 1 and silence_start is not None:
                if t - silence_start < self.merge_silence_frame:
                    new_decisions[silence_start:t] = [1] * (t - silence_start)
                silence_start = None
        return new_decisions

    def _extend_speech_segments(self, decisions):
        if self.extend_speech_frame <= 0:
            return decisions
        kernel = np.ones(2 * self.extend_speech_frame + 1)
        extended = np.convolve(np.array(decisions), kernel, mode="same")
        return (extended > 0).astype(int).tolist()

    def _split_long_speech_segments(self, decisions, probs):
        new_decisions = decisions.copy()
        for start_s, end_s in self.decision_to_segment(decisions):
            start_frame = int(start_s / FRAME_SHIFT_S)
            end_frame = int(end_s / FRAME_SHIFT_S)
            if end_frame - start_frame > self.max_speech_frame:
                segment_probs = probs[start_frame:end_frame]
                for split_point in self._find_split_points(segment_probs):
                    new_decisions[start_frame + split_point] = 0
        return new_decisions

    def _find_split_points(self, probs):
        split_points = []
        length = len(probs)
        start = 0
        while start < length:
            if (length - start) <= self.max_speech_frame:
                break
            window_start = int(start + self.max_speech_frame / 2)
            window_end = int(start + self.max_speech_frame)
            min_index = window_start + int(np.argmin(probs[window_start:window_end]))
            split_points.append(min_index)
            start = min_index + 1
        return split_points


class FireRedVadOnnx:
    """ONNX-only FireRedVAD detector for the runtime VAD pipeline."""

    def __init__(
        self,
        model_dir: str,
        speech_threshold: float = 0.4,
        smooth_window_size: int = 5,
        min_speech_frame: int = 20,
        max_speech_frame: int = 2000,
        min_silence_frame: int = 20,
        merge_silence_frame: int = 0,
        extend_speech_frame: int = 0,
        num_threads: int = 1,
    ):
        import onnxruntime as ort

        onnx_path = os.path.join(model_dir, "firered_vad.onnx")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(
            onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._cmvn = _CMVN(os.path.join(model_dir, "cmvn.npz"))
        self._fbank = _KaldifeatFbank()
        self._post = _VadPostprocessor(
            smooth_window_size, speech_threshold, min_speech_frame,
            max_speech_frame, min_silence_frame, merge_silence_frame,
            extend_speech_frame,
        )

    def detect(self, wav_int16_float: np.ndarray) -> list[tuple[float, float]]:
        """wav_int16_float: 1-D samples @16kHz in raw int16 amplitude range
        (matches fireredvad's AudioFeat, which feeds int16 values to
        kaldi_native_fbank — do NOT normalize to [-1, 1]).
        Returns [(start_s, end_s)]."""
        assert wav_int16_float.ndim == 1
        dur = wav_int16_float.shape[0] / SAMPLE_RATE
        feats = self._fbank(wav_int16_float)
        if feats.shape[0] == 0:
            return []
        feats = self._cmvn(feats)
        probs = self._sess.run(["probs"], {"feat": feats[None]})[0]
        probs = np.asarray(probs).squeeze()
        if probs.ndim == 2:
            probs = probs[:, 0]
        decisions = self._post.process(probs.tolist())
        return self._post.decision_to_segment(decisions, dur)
