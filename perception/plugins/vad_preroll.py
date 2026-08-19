"""Bounded PCM history used to restore audio before a VAD segment."""

from __future__ import annotations

from typing import Optional


class PcmHistory:
    """PCM16 history addressable by the VAD's absolute sample offsets."""

    def __init__(self, max_samples: int):
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self._max_samples = int(max_samples)
        self._pcm = bytearray()
        self.start_sample = 0
        self.total_samples = 0

    def clear(self) -> None:
        self._pcm.clear()
        self.start_sample = 0
        self.total_samples = 0

    def append(self, pcm: bytes) -> None:
        """Append aligned PCM16 bytes and evict history older than the limit."""
        pcm = pcm[: len(pcm) // 2 * 2]
        if not pcm:
            return
        sample_count = len(pcm) // 2
        self._pcm.extend(pcm)
        self.total_samples += sample_count

        excess_samples = len(self._pcm) // 2 - self._max_samples
        if excess_samples > 0:
            del self._pcm[: excess_samples * 2]
            self.start_sample += excess_samples

    def slice(self, start_sample: int, end_sample: int) -> bytes:
        """Return an absolute half-open sample range still present in history."""
        start = max(int(start_sample), self.start_sample)
        end = min(int(end_sample), self.total_samples)
        if start >= end:
            return b""
        local_start = start - self.start_sample
        local_end = end - self.start_sample
        return bytes(self._pcm[local_start * 2 : local_end * 2])

    def pre_roll(
        self,
        segment_start: Optional[int],
        segment_samples: int,
        pre_roll_samples: int,
        silence_samples: int,
    ) -> bytes:
        """Return PCM immediately before a completed sherpa VAD segment.

        Recent sherpa-onnx releases expose ``segment.start`` as an absolute
        sample offset. The fallback matches the upstream buffering contract
        when that attribute is unavailable.
        """
        if segment_start is None:
            segment_start = max(
                0,
                self.total_samples - int(segment_samples) - int(silence_samples),
            )
        segment_start = int(segment_start)
        return self.slice(
            segment_start - max(0, int(pre_roll_samples)),
            segment_start,
        )
