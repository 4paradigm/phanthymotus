"""Non-neural native Kokoro phase/noise DSP, with no Torch or model execution.

The prefix scan keeps native CUDA FP32 rounding. FP64 only emulates the
correctly rounded FP32 coordinate/output fused multiply-add operations; every
externally visible tensor remains FP32. Six captured-native phases were exact.
"""
import numpy as np


def source_phase(f0, initial_phase, *, block_samples=32768):
    f0 = np.asarray(f0)
    initial_phase = np.asarray(initial_phase)
    if f0.dtype != np.float32 or f0.ndim != 2 or f0.shape[0] != 1 or not np.isfinite(f0).all():
        raise ValueError('Phase DSP requires finite FP32 F0[1,F]')
    if initial_phase.shape != (1, 9) or initial_phase.dtype != np.float32 or not np.isfinite(initial_phase).all() or initial_phase[0, 0] != 0:
        raise ValueError('Native initial phase must be finite FP32[1,9] with fundamental zero')
    if int(block_samples) <= 0:
        raise ValueError('Phase DSP block_samples must be positive')
    count = f0.shape[1]
    if count < 2:
        raise ValueError('Native phase DSP requires at least two F0 frames')
    # Native nearest x300 -> rad -> linear /300 samples positions 149.5,
    # 449.5, ...: both taps belong to the same repeated F0 frame. The initial
    # phase addition affects sample0 only, which neither downsample tap reads.
    rad = np.remainder(f0[:, :, None] * np.arange(1, 10, dtype=np.float32)[None, None, :]
                       * np.float32(1 / 24000), np.float32(1))
    # Native CUDA outer-dimension scan uses one FP32 add per frame.
    phase = np.cumsum(rad, axis=1, dtype=np.float32)
    phase = phase * np.float32(2) * np.float32(np.pi) * np.float32(300)
    samples = count * 300
    result = np.empty((1, samples, 9), dtype=np.float32)
    for start in range(0, samples, int(block_samples)):
        stop = min(samples, start + int(block_samples))
        dst_half = np.arange(start, stop, dtype=np.float32) + np.float32(.5)
        # CUDA uses one FMA for scale * (dst + .5) - .5, not two roundings.
        at = (dst_half.astype(np.float64) * np.float64(np.float32(1 / 300)) - .5).astype(np.float32)
        at = np.maximum(at, np.float32(0))
        left = np.floor(at).astype(np.int64)
        right = np.minimum(left + 1, count - 1)
        fraction = (at - left).astype(np.float32)[None, :, None]
        # Match the original FP32 right product followed by FP32 fma(left).
        right_product = phase[:, right, :] * fraction
        result[:, start:stop, :] = (phase[:, left, :].astype(np.float64)
            * (np.float32(1) - fraction).astype(np.float64) + right_product.astype(np.float64)).astype(np.float32)
    return result


def excitation_inputs(f0, *, seed=42):
    """Retain uniform initial phase and independent standard-normal source noise."""
    rng = np.random.default_rng(int(seed))
    initial = rng.random((1, 9), dtype=np.float32)
    initial[:, 0] = 0
    phase = source_phase(f0, initial)
    noise = rng.standard_normal(phase.shape, dtype=np.float32)
    return np.ascontiguousarray(phase), np.ascontiguousarray(noise)


class ExcitationCache:
    """Immutable seed-42 noise and coordinates for the verified F6..1460 range.

    The sentence-dependent F0 scan and phase interpolation run on every call.
    Coordinate arithmetic retains the reference's FP64 simulation of FP32 FMA;
    right indices are clamped for this call's frame count, never for the cache.
    The standalone functions above remain the uncached numerical reference.
    """
    MIN_FRAMES = 6
    MAX_FRAMES = 1460
    SEED = 42

    def __init__(self):
        samples = self.MAX_FRAMES * 300
        half_sample = np.arange(samples, dtype=np.float32) + np.float32(.5)
        positions = (half_sample.astype(np.float64)
                     * np.float64(np.float32(1 / 300)) - .5).astype(np.float32)
        self._positions = np.maximum(positions, np.float32(0))
        self._left = np.floor(self._positions).astype(np.int64)
        self._fraction = (self._positions - self._left).astype(np.float32)
        rng = np.random.default_rng(self.SEED)
        self._initial_phase = rng.random((1, 9), dtype=np.float32)
        self._initial_phase[:, 0] = 0
        # Consume the original nine uniform values before drawing any noise.
        self._noise = rng.standard_normal((1, samples, 9), dtype=np.float32)
        for array in (self._positions, self._left, self._fraction,
                      self._initial_phase, self._noise):
            array.flags.writeable = False

    @property
    def persistent_array_bytes(self):
        return sum(array.nbytes for array in (
            self._positions, self._left, self._fraction,
            self._initial_phase, self._noise))

    def source_phase(self, f0, initial_phase=None, *, block_samples=32768):
        """Return a new phase array; cached values never include speech F0."""
        f0 = np.asarray(f0)
        initial_phase = np.asarray(self._initial_phase if initial_phase is None else initial_phase)
        if f0.dtype != np.float32 or f0.ndim != 2 or f0.shape[0] != 1 or not np.isfinite(f0).all():
            raise ValueError('Phase DSP requires finite FP32 F0[1,F]')
        if (initial_phase.shape != (1, 9) or initial_phase.dtype != np.float32
                or not np.isfinite(initial_phase).all() or initial_phase[0, 0] != 0):
            raise ValueError('Native initial phase must be finite FP32[1,9] with fundamental zero')
        if int(block_samples) <= 0:
            raise ValueError('Phase DSP block_samples must be positive')
        count = f0.shape[1]
        if not self.MIN_FRAMES <= count <= self.MAX_FRAMES:
            raise ValueError('Cached phase DSP requires F in the verified range 6..1460')
        rad = np.remainder(f0[:, :, None] * np.arange(1, 10, dtype=np.float32)[None, None, :]
                           * np.float32(1 / 24000), np.float32(1))
        phase = np.cumsum(rad, axis=1, dtype=np.float32)
        phase = phase * np.float32(2) * np.float32(np.pi) * np.float32(300)
        samples = count * 300
        result = np.empty((1, samples, 9), dtype=np.float32)
        # These two bounded work arrays are per call, never persistent cache.
        # The original rad/scan/multiply sequence above is unchanged.
        rows = min(samples, int(block_samples))
        phase64 = phase[0].astype(np.float64)
        left_work = np.empty((rows, 9), dtype=np.float64)
        right_work = np.empty((rows, 9), dtype=np.float32)
        for start in range(0, samples, int(block_samples)):
            stop = min(samples, start + int(block_samples))
            left = self._left[start:stop]
            right = np.minimum(left + 1, count - 1)
            fraction = self._fraction[start:stop, None]
            left_product = left_work[:stop-start]
            right_product = right_work[:stop-start]
            # Cached coordinates are unchanged, monotonic and proven in range
            # for every F6..1460. clip avoids take's mode=raise output buffer.
            np.take(phase64, left, axis=0, out=left_product, mode='clip')
            np.multiply(left_product, (np.float32(1) - fraction).astype(np.float64), out=left_product)
            np.take(phase[0], right, axis=0, out=right_product, mode='clip')
            np.multiply(right_product, fraction, out=right_product)
            # Preserve FP32 right multiplication, FP64 left multiply/add,
            # then the original single FP32 output cast; no reassociation/FMA.
            np.add(left_product, right_product, out=left_product)
            result[0, start:stop, :] = left_product
        return result

    def excitation_inputs(self, f0, *, seed=42):
        """Match the reference excitation pair without repeated RNG/grid work."""
        if int(seed) != self.SEED:
            raise ValueError('Excitation cache only supports the fixed production seed 42')
        phase = self.source_phase(f0)
        noise = self._noise[:, :phase.shape[1], :]
        # The prefix is contiguous and read-only; retaining it retains its cache.
        return np.ascontiguousarray(phase), np.ascontiguousarray(noise)
