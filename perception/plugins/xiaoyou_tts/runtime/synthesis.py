"""Independent V6 TensorRT FP32 backend. No CPU neural fallback.

Emit a safe first punctuation clause, then its intact tail, retaining full frontend
context. Short/ambiguous openings retain the complete-sentence policy. Only excess zero PCM at their real sentence join is shortened. Actual
token/frame overflow keeps the original lossless semantic fallback.
Named numerical checks do not imply exhaustive shape or latency acceptance.
"""
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import threading
import time
from concurrent.futures import CancelledError

import numpy as np

from .cuda_engine import GPUEnginePool
from ..frontend.phonemizer import FrontendWorker, require_runtime
from .model_bundle import (
    CONFIG_SHA, EXPECTED_PROFILES, FRONTEND_SHA, GLOBAL_STEP, GRAPH_SHAS,
    JP5_EXPECTED_PROFILES, JP5_GRAPH_SHAS, JP5_PLAN_SHAS,
    NATIVE_WEIGHTS_SHA, NORMALIZATION_MODULE_SHAS, OPERATING_F_RANGE, PHASE_DSP_SHA,
    PLAN_SHAS, SOURCE_SHA, STREAM_POLICY, STYLE_FILE_SHA, STYLE_RAW_SHA, SYNTHESIS_SOURCE_SHA, TERMS_FILE,
    TERMS_SHA, TRT_VERSION_PREFIX, VOICE_ID, ensure_runtime_bundle, resolve_runtime_manifest_pin, sha,
    validate_frontend_files,
)


PREFIX_SAMPLES = 8000
NATIVE_PREFIX_SAMPLES = 12023
JOIN_ZERO_KEEP_SAMPLES = 2000


def safe_prefix(prefix, *, head=False, tail=False):
    """Return an already join-trimmed prefix, or None when proof needs the tail."""
    if head and tail:
        raise ValueError('One nonempty span cannot be on both sides of the join')
    if not isinstance(prefix, bytes) or len(prefix) != 2 * PREFIX_SAMPLES:
        return None
    nonzero = np.flatnonzero(np.frombuffer(prefix, dtype='<i2'))
    if tail and (not nonzero.size or nonzero[-1] < PREFIX_SAMPLES - JOIN_ZERO_KEEP_SAMPLES - 1):
        return None
    if head and not nonzero.size:
        return None
    removed = max(0, int(nonzero[0]) - JOIN_ZERO_KEEP_SAMPLES) if head else 0
    return prefix[2 * removed:]


def trim_join_zeros(pcm, *, head=False, tail=False):
    """Keep up to 125 ms of zero PCM on each side of a sentence join."""
    if head and tail:
        raise ValueError('One nonempty span cannot be on both sides of the join')
    if not (head or tail) or not pcm:
        return pcm
    samples = np.frombuffer(pcm, dtype='<i2')
    nonzero = np.flatnonzero(samples)
    head_zeros = int(nonzero[0]) if nonzero.size else len(samples)
    tail_zeros = len(samples) - int(nonzero[-1]) - 1 if nonzero.size else len(samples)
    head_cut = max(0, head_zeros - JOIN_ZERO_KEEP_SAMPLES) if head else 0
    tail_cut = max(0, tail_zeros - JOIN_ZERO_KEEP_SAMPLES) if tail else 0
    return pcm[2 * head_cut:len(pcm) - 2 * tail_cut]


def iter_exact_pcm(wave, *, resample, pcm16, head=False, tail=False, cancel=None):
    """Emit a safe prefix, followed by the rest of the whole-wave resample."""
    def cancelled():
        return cancel is not None and cancel.is_set()

    if cancelled():
        return
    emitted = b''
    if len(wave) >= NATIVE_PREFIX_SAMPLES:
        prefix = resample(wave[:NATIVE_PREFIX_SAMPLES])[:PREFIX_SAMPLES]
        admitted = safe_prefix(pcm16(prefix), head=head, tail=tail)
        if admitted is not None:
            if cancelled():
                return
            emitted = admitted
            yield emitted
    if cancelled():
        return
    # Preserve the original FIR phase and EOF padding.
    whole = trim_join_zeros(pcm16(resample(wave)), head=head, tail=tail)
    if not whole.startswith(emitted):
        raise RuntimeError('Prefix differs from complete reference PCM')
    rest = whole[len(emitted):]
    if rest and not cancelled():
        yield rest


def _join_producers(producers, timeout=30.0):
    """Drain only these owned producers; never wait while holding their locks."""
    deadline = time.monotonic() + timeout
    for producer in producers:
        if producer is threading.current_thread():
            raise RuntimeError('V6 producer cannot dispose its own active backend')
        producer.join(timeout=max(0.0, deadline-time.monotonic()))
        if producer.is_alive():
            raise RuntimeError('V6 producer did not stop within30s; backend resources retained')


class _CancelledGeneration(RuntimeError):
    pass


class ProfileLimitError(ValueError):
    """Text is intact, but the verified model profile cannot contain this clause."""


def validate_options(language='auto', speed=1.0, speaker_id=0):
    if language not in ('auto', 'zh', 'en') or float(speed) != 1.0 or int(speaker_id) != 0:
        raise ValueError('V6 permits auto/zh/en, fixed01yuewen speaker0 and speed1 only')




def safe_split(text):
    from ..frontend import numbers
    try:
        left,right=numbers.safe_split(text)
    except numbers.ProfileLimitError as error:
        raise ProfileLimitError(str(error)) from error
    if left+right!=text:raise ValueError('Semantic split changed normalized text')
    return left,right


def _concrete(shape, symbols):
    try:
        result = tuple(symbols[x] if isinstance(x, str) else int(x) for x in shape)
    except KeyError as error:
        raise ValueError('Unknown V6 shape symbol: ' + str(error)) from error
    if any(x <= 0 for x in result):
        raise ValueError('V6 dynamic output is empty')
    return result


def verified_limits(stages, policy, jetpack_major=6):
    """Bind the plans, input profiles and I/O types against the fixed release."""
    if jetpack_major == 5:
        plan_shas, graph_shas, profiles = JP5_PLAN_SHAS, JP5_GRAPH_SHAS, JP5_EXPECTED_PROFILES
        duration_dtype = 'int32'
    else:
        plan_shas, graph_shas, profiles = PLAN_SHAS, GRAPH_SHAS, EXPECTED_PROFILES
        duration_dtype = 'int64'
    if policy != STREAM_POLICY or set(stages) != set(plan_shas):
        raise ValueError('01yuewen requires the exact stages and inference policy')
    for name,stage in stages.items():
        if (stage.get('source_graph_sha256') != graph_shas[name]
                or stage.get('sha256') != plan_shas[name]
                or stage.get('profiles') != profiles[name]):
            raise ValueError('01yuewen actual graph/plan/profile binding differs: '+name)
        if (set(stage.get('io_dtypes',{})) != set(stage['inputs'])|set(stage['outputs'])
                or set(stage['profiles']) != set(stage['inputs'])):
            raise ValueError('01yuewen typed I/O/profile declaration incomplete: '+name)
        for key,dtype in stage['io_dtypes'].items():
            expected='int32' if key in ('tokens','indices') else duration_dtype if key=='duration' else 'float32'
            if dtype != expected:raise ValueError('01yuewen actual integer/FP32 ABI differs: '+key)
    min_frames, max_frames = OPERATING_F_RANGE
    return {'max_tokens': 510, 'min_F': min_frames, 'max_F': max_frames, 'max_T': max_frames // 2}


# Native FP32 excitation for the acoustic and vocoder stages.

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


class TRTBackend:
    dry_run_text = 'Hi.'

    def __init__(self, model_dir=None, *, cpu_threads=2, backend='tensorrt_fp32', expected_manifest_sha256=None,
                 progress_cb=None, stage_cb=None):
        major = require_runtime()
        self.first_output_policy = os.environ.get('XIAOYOU01_FIRST_OUTPUT_POLICY', STREAM_POLICY['default_first_output_policy'])
        if self.first_output_policy not in STREAM_POLICY['allowed_first_output_policies']:
            raise ValueError('First output policy must be clause or sentence')
        prefix_mode = os.environ.get('XIAOYOU01_PCM_PREFIX', '1')
        if prefix_mode not in ('0', '1'):
            raise ValueError('PCM prefix setting must be 0 or 1')
        self.pcm_prefix_enabled = prefix_mode == '1'
        cache_mode = os.environ.get('XIAOYOU01_EXCITATION_CACHE', '1')
        if cache_mode not in ('0', '1'):
            raise ValueError('Excitation cache setting must be 0 or 1')
        gpu_noise_mode = os.environ.get('XIAOYOU01_GPU_NOISE_CACHE', '0')
        if gpu_noise_mode not in ('0', '1'):
            raise ValueError('GPU noise cache setting must be 0 or 1')
        if backend != 'tensorrt_fp32' or int(cpu_threads) not in (1, 2):
            raise ValueError('V6 production backend requires TensorRT FP32 and one/two CPU threads')
        if sha(Path(__file__)) != SYNTHESIS_SOURCE_SHA:
            raise ValueError('Xiaoyou synthesis source differs from the verified runtime')
        pin = resolve_runtime_manifest_pin(expected_manifest_sha256, jetpack_major=major)
        asset_started = time.perf_counter()
        self.root = ensure_runtime_bundle(model_dir, pin, jetpack_major=major,
                                          progress_cb=progress_cb, stage_cb=stage_cb)
        backend_started = time.perf_counter()
        self.asset_prepare_seconds = backend_started - asset_started
        print('XIAOYOU_STARTUP ' + json.dumps({
            'event': 'bundle_verified', 'asset_prepare_seconds': self.asset_prepare_seconds,
        }), flush=True)
        manifest_bytes = (self.root / 'runtime_manifest.json').read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != pin:
            raise ValueError('V6 runtime manifest SHA differs')
        m = json.loads(manifest_bytes)
        if (m.get('artifact_kind') != 'xiaoyou_v6_gpu_runtime'
                or m.get('source_checkpoint_sha256') != SOURCE_SHA
                or m.get('native_weights_sha256') != NATIVE_WEIGHTS_SHA
                or m.get('runtime_code') != {'excitation.py': PHASE_DSP_SHA}):
            raise ValueError('Wrong V6 GPU model lineage or runtime code')
        frontend_source_files = validate_frontend_files(
            m.get('frontend', {}), Path(__file__).resolve().parents[1])
        pack = m.get('platform', {})
        trt_prefix = TRT_VERSION_PREFIX[major]
        if pack.get('jetpack_major') != major or pack.get('tensorrt_version_prefix') != trt_prefix:
            raise ValueError(f'01yuewen requires the verified JetPack {major}/TensorRT {trt_prefix} engine pack')
        files = m['files']
        for name, expected in (('config.json', CONFIG_SHA), ('style.npy', STYLE_FILE_SHA)):
            if files.get(name, {}).get('sha256') != expected:
                raise ValueError('V6 fixed configuration/style asset differs: ' + name)
        stages = m['stages']
        self.limits = verified_limits(stages, m.get('inference_policy'), major)
        for stage in stages.values():
            name = stage['file']
            if files.get(name, {}).get('sha256') != stage['sha256']:
                raise ValueError('V6 actual plan asset differs: ' + name)
        if os.environ.get('NVIDIA_TF32_OVERRIDE') is not None:
            raise ValueError('NVIDIA_TF32_OVERRIDE must be unset for this FP32 pack')
        self.excitation_cache = ExcitationCache() if cache_mode == '1' else None
        if self.excitation_cache is not None and self.excitation_cache.persistent_array_bytes > 24 * 1024**2:
            raise RuntimeError('Exact excitation cache exceeds its bounded CPU storage')
        style = np.load(self.root / 'style.npy', allow_pickle=False)
        if style.shape != (1, 256) or style.dtype != np.float32 or not np.isfinite(style).all() or hashlib.sha256(np.ascontiguousarray(style).tobytes()).hexdigest() != STYLE_RAW_SHA:
            raise ValueError('V6 style is not the exact01yuewen vector')
        self._frontend_lock = threading.Lock()
        self._call_lock = threading.RLock()
        self._warmup_lock = threading.Lock()
        self._producer_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._producers = {}
        self._closed = False
        self._warmed = False
        self.frontend = None
        self.pool = None
        self.style = None
        self.gpu_noise = None
        try:
            self.frontend = FrontendWorker(self.root)
            if (self.frontend.metadata['frontend_sha256'] != FRONTEND_SHA
                    or self.frontend.metadata['terms_sha256'] != TERMS_SHA
                    or self.frontend.metadata['normalization_module_sha256'] != NORMALIZATION_MODULE_SHAS):
                raise ValueError('V6 imported frontend metadata differs from the fixed source contract')
            self.legal_ids = self.frontend.legal_ids
            self.pool = GPUEnginePool(self.root, stages, expected_trt_prefix=trt_prefix)
            self.style = self.pool.upload(np.ascontiguousarray(style))
            if gpu_noise_mode == '1' and self.excitation_cache is not None:
                noise = self.excitation_cache._noise
                if (noise.dtype != np.float32 or noise.shape != (1, ExcitationCache.MAX_FRAMES * 300, 9)
                        or not noise.flags.c_contiguous or noise.flags.writeable):
                    raise ValueError('GPU noise cache requires the immutable contiguous FP32 seed42 source')
                self.gpu_noise = self.pool.upload(noise)
        except BaseException:
            self.close()
            raise
        self.stages = stages
        self.metadata = {
            'name': 'xiaoyou01', 'model_id': 'xiaoyou01', 'voice_id': VOICE_ID,
            'backend': 'tensorrt_fp32', 'accelerated': True, 'device': 'cuda:0',
            'precision': 'FP32', 'cpu_neural_fallback': False,
            'runtime_manifest_sha256': pin, 'source_checkpoint_sha256': SOURCE_SHA,
            'global_step': GLOBAL_STEP, 'jetpack_major': major,
            'tensorrt_version': self.pool.version, 'limits': self.limits,
            'frontend_source_files': frontend_source_files,
            'frontend_dictionary_initialization_seconds': self.frontend.dictionary_seconds,
            'streaming': 'safe first punctuation clause or configured complete sentence, then intact tail; join-only excess zero PCM shortened; bounded prefetch and lossless profile fallback',
            'first_output_policy': self.first_output_policy,
        }
        self.backend_load_seconds = time.perf_counter() - backend_started

    validate_options = staticmethod(validate_options)

    @property
    def frontend_error(self):
        frontend = self.frontend
        return frontend.failure_reason if frontend is not None else None

    def resolve_language(self, text, language):
        validate_options(language)
        return language if language != 'auto' else ('zh' if re.search(r'[\u3400-\u9fff]', text) or not re.search('[A-Za-z]', text) else 'en')

    def _outputs(self, stage, symbols):
        return {name: _concrete(shape, symbols) for name, shape in self.stages[stage]['outputs'].items()}

    def synthesize_wave(self, text, *, language='auto', speed=1.0, _cancel=None,
                        _normalization_language=None, _normalized_context=None, auto_fallback=None):
        validate_options(language, speed)
        requested_language = language if _normalization_language is None else _normalization_language
        validate_options(requested_language, speed)
        if _cancel is not None and _cancel.is_set():
            raise _CancelledGeneration('Request cancelled before generation')
        with self._frontend_lock:
            if self._closed:
                raise RuntimeError('V6 backend is closed')
            if _cancel is not None and _cancel.is_set():
                raise _CancelledGeneration('Request cancelled while waiting for frontend')
            def cancelled():
                return self._closed or (_cancel is not None and _cancel.is_set())

            try:
                if _normalized_context is None:
                    _, ids = self.frontend(text, language, normalization_language=requested_language,
                                           auto_fallback=auto_fallback, cancelled=cancelled)
                else:
                    full, start, end = _normalized_context
                    if full[start:end] != text:
                        raise ValueError('Normalized semantic span differs from full request')
                    _, ids = self.frontend.phonemize_normalized(
                        full, language, start=start, end=end, normalization_language=requested_language,
                        auto_fallback=auto_fallback, cancelled=cancelled)
            except CancelledError as error:
                raise _CancelledGeneration('Request cancelled during frontend startup') from error
        if _cancel is not None and _cancel.is_set():
            raise _CancelledGeneration('Request cancelled after frontend')
        return self._synthesize_ids(ids, _cancel=_cancel)

    def synthesize_tokens(self, tokens, style=None, speed=1.0, *, _cancel=None):
        """Pure GPU harness API: only complete tokens/style/speed, no CPU NN oracle."""
        token_array = np.asarray(tokens)
        value = np.asarray(speed)
        if value.size != 1 or float(value.reshape(-1)[0]) != 1.0:
            raise ValueError('xiaoyou01 speed is fixed to1')
        if token_array.ndim != 2 or token_array.shape[0] != 1 or not 3 <= token_array.shape[1] <= 512 or token_array.dtype.kind not in 'iu' or token_array[0,0] != 0 or token_array[0,-1] != 0:
            raise ValueError('xiaoyou01 requires complete integer tokens[1,N] with both zero boundaries')
        if style is not None:
            supplied = np.asarray(style)
            if supplied.shape != (1,256) or supplied.dtype != np.float32 or hashlib.sha256(np.ascontiguousarray(supplied).tobytes()).hexdigest() != STYLE_RAW_SHA:
                raise ValueError('xiaoyou01 only accepts its fixed style vector')
        return self._synthesize_ids(token_array[0,1:-1].tolist(), _cancel=_cancel)

    def _synthesize_ids(self, ids, *, _cancel=None):
        if not ids or len(ids) > 510:
            raise ProfileLimitError('V6 complete clause exceeds510 phoneme IDs; no truncation')
        if not set(ids).issubset(self.legal_ids):
            raise ValueError('V6 frontend emitted a token outside the pinned ABI')
        with self._call_lock:
            if self._closed:
                raise RuntimeError('V6 GPU backend is closed')
            live = []
            def execute(stage, inputs, symbols):
                if _cancel is not None and _cancel.is_set():
                    raise _CancelledGeneration('Request cancelled between GPU stages')
                out = self.pool.infer(stage, inputs, self._outputs(stage, symbols))
                live.extend(out.values())
                if _cancel is not None and _cancel.is_set():
                    raise _CancelledGeneration('Request cancelled after GPU stage')
                return out
            try:
                n = len(ids) + 2
                token_type = np.dtype(self.stages['encoder']['io_dtypes']['tokens'])
                enc = execute('encoder', {'tokens': np.array([[0, *ids, 0]], dtype=token_type),
                    'style': self.style, 'speed': np.ones((1,), np.float32)}, {'N': n})
                durations = enc['duration'].numpy()
                enc['duration'].release()
                if durations.shape != (n,) or durations.dtype.kind not in 'iu' or (np.any(durations < 1) or np.any(durations > 50)):
                    raise RuntimeError('V6 GPU predicted invalid integer durations')
                t = int(durations.astype(np.int64).sum())
                if t > self.limits['max_T'] or t < 3:
                    raise ProfileLimitError('V6 predicted duration exceeds the verified GPU range; no waveform was truncated')
                f = 2 * t
                symbols = {'N': n, 'T': t, 'F': f, 'samples': 300*f, 'padded_samples': 300*f+20,
                           'H': 60*f+1, 'Q': 10*f, 'R': 60*f}
                indices = np.repeat(np.arange(n, dtype=np.int32), durations.astype(np.int64))
                ac = execute('acoustic', {'d': enc['d'], 't_en': enc['t_en'], 'indices': indices, 'style': self.style}, symbols)
                enc['d'].release(); enc['t_en'].release()
                f0 = ac['f0'].numpy()
                if f0.shape != (1, f) or f0.dtype != np.float32:
                    raise RuntimeError('V6 GPU F0 frame rate differs')
                cache = getattr(self, 'excitation_cache', None)
                phase, noise = (cache.excitation_inputs(f0, seed=42) if cache is not None
                                else excitation_inputs(f0, seed=42))
                gpu_noise = getattr(self, 'gpu_noise', None)
                if gpu_noise is not None:
                    noise = gpu_noise.prefix_view(phase.shape)
                    live.append(noise)
                vocoder = execute('vocoder', {'features': ac['features'], 'style': self.style,
                    'f0': ac['f0'], 'source_phase': phase, 'excitation_noise': noise}, symbols)
                del phase, noise
                ac['f0'].release(); ac['features'].release()
                wave = vocoder['audio'].numpy()
                if wave.shape != (600*t,) or wave.dtype != np.float32 or not np.isfinite(wave).all():
                    raise RuntimeError('V6 GPU waveform is not complete finite FP32 at24kHz')
                return wave, durations, ids
            finally:
                for value in live:
                    value.release()

    def _audio_clauses(self, text, language, cancel, *, normalization_language=None, auto_fallback=None):
        from ..frontend.numbers import first_clause_spans, first_complete_sentence_spans
        policy = getattr(self, 'first_output_policy', STREAM_POLICY['default_first_output_policy'])
        if policy not in STREAM_POLICY['allowed_first_output_policies']:
            raise ValueError('First output policy must be clause or sentence')
        spans = first_clause_spans(text) if policy == 'clause' else first_complete_sentence_spans(text)
        if not 1 <= len(spans) <= 2 or ''.join(text[a:b] for a,b in spans) != text:
            raise ValueError('First-output policy changed normalized text')
        join = spans[0][1] if len(spans) == 2 else None
        for start, end in spans:
            if cancel is not None and cancel.is_set():
                return
            # Yield the first complete piece before starting the tail. Recursive
            # profile fallback receives the same absolute join and full context.
            yield from self._audio_clause(text[start:end], language, cancel,
                normalization_language=normalization_language, auto_fallback=auto_fallback,
                normalized_context=(text, start, end), sentence_join=join)

    def _audio_clause(self, text, language, cancel, *, normalized_context,
                      normalization_language=None, sentence_join=None, auto_fallback=None):
        if cancel is not None and cancel.is_set():
            return
        full_text, start, end = normalized_context
        try:
            wave, _, _ = self.synthesize_wave(
                text, language=language, _cancel=cancel,
                _normalization_language=normalization_language,
                _normalized_context=normalized_context,
                auto_fallback=auto_fallback,
            )
        except _CancelledGeneration:
            return
        except ProfileLimitError:
            cursor = 0
            for piece in safe_split(text):
                offset = text.find(piece, cursor)
                if offset < 0 or text[cursor:offset].strip():
                    raise ValueError('Profile fallback omitted normalized text')
                cursor = offset + len(piece)
                yield from self._audio_clause(piece, language, cancel,
                    normalization_language=normalization_language, auto_fallback=auto_fallback,
                    normalized_context=(full_text, start + offset, start + cursor), sentence_join=sentence_join)
            if text[cursor:].strip():
                raise ValueError('Profile fallback omitted normalized tail')
            return
        from utils.resample import resample_poly, float_to_pcm16
        if sentence_join is not None and not 0 < sentence_join < len(full_text):
            raise ValueError('Sentence join is not an interior absolute text boundary')
        head = start == sentence_join
        tail = end == sentence_join
        if getattr(self, 'pcm_prefix_enabled', False):
            yield from iter_exact_pcm(wave, resample=resample_poly, pcm16=float_to_pcm16,
                head=head, tail=tail, cancel=cancel)
        else:
            yield trim_join_zeros(float_to_pcm16(resample_poly(wave)), head=head, tail=tail)

    def iter_pcm(self, text, *, language='auto', speed=1.0, cancel=None, auto_fallback=None):
        validate_options(language, speed)
        if not isinstance(text, str) or not text.strip():
            return
        if not any(c.isalnum() for c in text):
            raise ValueError('xiaoyou01 text contains no spoken content')
        requested_language = language
        # Keep the original G2P route, but normalize the COMPLETE request before
        # splitting: commas in tensors/operators/groups must not be cut first.
        # Unsupported expressions fail before any producer/PCM is started.
        with self._frontend_lock:
            if self._closed:
                raise RuntimeError('V6 backend is closed')
            if cancel is not None and cancel.is_set():
                return
            try:
                text = self.frontend.normalize_text(
                    text, requested_language, auto_fallback=auto_fallback,
                    cancelled=lambda: self._closed or (cancel is not None and cancel.is_set()))
            except CancelledError:
                return
        if not isinstance(text, str) or not text.strip():
            raise ValueError('V6 normalization produced no spoken text')
        pending = queue.Queue(maxsize=2)
        stop = threading.Event()
        def offer(item):
            while not stop.is_set() and not (cancel is not None and cancel.is_set()):
                try:
                    pending.put(item, timeout=.05)
                    return True
                except queue.Full:
                    pass
            return False
        class CombinedCancel:
            def is_set(inner):
                return stop.is_set() or self._closed or (cancel is not None and cancel.is_set())
        combined = CombinedCancel()
        def produce():
            try:
                for pcm in self._audio_clauses(text, language, combined,
                        normalization_language=requested_language, auto_fallback=auto_fallback):
                    if not offer(('pcm', pcm)):
                        return
                offer(('end', None))
            except BaseException as error:
                offer(('error', error))
            finally:
                with self._producer_lock:
                    self._producers.pop(threading.current_thread(), None)
        producer = threading.Thread(target=produce, name='xiaoyou-v6-gpu-prefetch', daemon=True)
        with self._producer_lock:
            if self._closed:
                raise RuntimeError('V6 backend is closed')
            self._producers[producer] = stop
            try:
                producer.start()
            except BaseException:
                self._producers.pop(producer, None)
                raise
        try:
            while not combined.is_set():
                try:
                    kind, value = pending.get(timeout=.05)
                except queue.Empty:
                    if self._closed or not producer.is_alive():
                        return
                    continue
                if kind == 'error':
                    raise value
                if kind == 'end':
                    return
                yield value
        finally:
            stop.set()
            _join_producers([producer])

    def synthesize(self, text, **options):
        return b''.join(self.iter_pcm(text, **options))

    def warmup(self):
        if self._closed:
            raise RuntimeError('V6 backend is closed')
        if self._warmed:
            return 0
        with self._warmup_lock:
            if self._warmed:
                return 0
            # Startup-only coverage; consume and discard every complete output.
            # These independent sentences do not change production segmentation.
            count = 0
            for text, language in (
                (self.dry_run_text, 'en'),
                ('今天天气晴朗，欢迎来到我们的工作空间，大家可以先坐下来稍作休息，喝杯热茶，再一起交流新的想法。', 'zh'),
                ('Welcome to our workspace, where the team shares useful ideas and helps visitors explore new projects in a friendly atmosphere.', 'en'),
                ('欢迎来到小优的工作空间，今天一起体验语音服务，Please take a seat.', 'auto'),
            ):
                case_bytes = len(self.synthesize(text, language=language))
                if not case_bytes:
                    raise RuntimeError('V6 GPU warmup produced no PCM')
                count += case_bytes
            self._warmed = True
            return count

    def close(self):
        # Closed blocks new registrations; it does not mean resources are gone.
        # A failed drain/close remains reachable and can be retried safely.
        with self._close_lock:
            with self._producer_lock:
                self._closed = True
                producers = list(self._producers)
                for stop in self._producers.values():
                    stop.set()
            _join_producers(producers)
            with self._frontend_lock:
                if self.frontend is not None:
                    self.frontend.close()
                    self.frontend = None
            with self._call_lock:
                if self.style is not None:
                    self.style.release()
                    self.style = None
                if getattr(self, 'gpu_noise', None) is not None:
                    self.gpu_noise.release()
                    self.gpu_noise = None
                if self.pool is not None:
                    self.pool.close()
                    self.pool = None
                self.excitation_cache = None


_SHARED = {}
_SHARED_LOCK = threading.Lock()


class _Lease:
    def __init__(self, key, backend):
        self._key, self._backend, self._released = key, backend, False

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def close(self):
        if self._released:
            return
        with _SHARED_LOCK:
            if self._released:
                return
            entry = _SHARED[self._key]
            if entry['references'] == 1:
                # Do not commit the last release before the backend has drained.
                entry['backend'].close()
                del _SHARED[self._key]
            entry['references'] -= 1
            self._released = True


def acquire_gpu_backend(model_dir=None, *, cpu_threads=2, expected_manifest_sha256=None,
                        progress_cb=None, stage_cb=None):
    root = str(Path(model_dir or os.environ.get('XIAOYOU_V6_MODEL_DIR') or '/models/01yuewen').expanduser().resolve())
    pin = resolve_runtime_manifest_pin(expected_manifest_sha256)
    key = (root, pin, int(cpu_threads))
    with _SHARED_LOCK:
        entry = _SHARED.get(key)
        if entry is None:
            backend = TRTBackend(model_dir, cpu_threads=cpu_threads,
                                 expected_manifest_sha256=expected_manifest_sha256,
                                 progress_cb=progress_cb, stage_cb=stage_cb)
            entry = {'backend': backend, 'references': 0}
            _SHARED[key] = entry
        entry['references'] += 1
        return _Lease(key, entry['backend'])
