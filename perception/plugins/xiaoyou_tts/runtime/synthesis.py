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

import numpy as np

from .cuda_engine import GPUEnginePool
from .excitation import ExcitationCache, excitation_inputs
from .frontend_worker import FrontendWorker, require_runtime
from .model_bundle import (
    CONFIG_SHA, EXPECTED_PROFILES, FRONTEND_SHA, GLOBAL_STEP, GRAPH_SHAS,
    JP5_EXPECTED_PROFILES, JP5_GRAPH_SHAS, JP5_PLAN_SHAS,
    NATIVE_WEIGHTS_SHA, NORMALIZATION_MODULE_SHAS, OPERATING_F_RANGE, PHASE_DSP_SHA,
    PLAN_SHAS, SOURCE_SHA, STREAM_POLICY, STYLE_FILE_SHA, STYLE_RAW_SHA, TERMS_FILE,
    TERMS_SHA, TRT_VERSION_PREFIX, VOICE_ID, ensure_runtime_bundle, resolve_runtime_manifest_pin, sha,
    validate_frontend_files,
)
from .speech_session import iter_exact_pcm, trim_join_zeros


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
    from ..frontend import segmentation
    try:
        left,right=segmentation.safe_split(text)
    except segmentation.ProfileLimitError as error:
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


class TRTBackend:
    dry_run_text = 'Hi.'

    def __init__(self, model_dir=None, *, cpu_threads=2, backend='tensorrt_fp32', expected_manifest_sha256=None):
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
        if sha(Path(__file__).with_name('excitation.py')) != PHASE_DSP_SHA:
            raise ValueError('01yuewen production phase DSP source differs from the release binding')
        pin = resolve_runtime_manifest_pin(expected_manifest_sha256, jetpack_major=major)
        asset_started = time.perf_counter()
        self.root = ensure_runtime_bundle(model_dir, pin, jetpack_major=major)
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
            if _normalized_context is None:
                _, ids = self.frontend(text, language, normalization_language=requested_language,
                                       auto_fallback=auto_fallback)
            else:
                full, start, end = _normalized_context
                if full[start:end] != text:
                    raise ValueError('Normalized semantic span differs from full request')
                _, ids = self.frontend.phonemize_normalized(
                    full, language, start=start, end=end,
                    normalization_language=requested_language, auto_fallback=auto_fallback)
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
        from ..frontend.segmentation import first_clause_spans, first_complete_sentence_spans
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
            text = self.frontend.normalize_text(text, requested_language, auto_fallback=auto_fallback)
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


def acquire_gpu_backend(model_dir=None, *, cpu_threads=2, expected_manifest_sha256=None):
    root = str(Path(model_dir or os.environ.get('XIAOYOU_V6_MODEL_DIR') or '/models/01yuewen').expanduser().resolve())
    pin = resolve_runtime_manifest_pin(expected_manifest_sha256)
    key = (root, pin, int(cpu_threads))
    with _SHARED_LOCK:
        entry = _SHARED.get(key)
        if entry is None:
            entry = {'backend': TRTBackend(model_dir, cpu_threads=cpu_threads, expected_manifest_sha256=expected_manifest_sha256), 'references': 0}
            _SHARED[key] = entry
        entry['references'] += 1
        return _Lease(key, entry['backend'])
