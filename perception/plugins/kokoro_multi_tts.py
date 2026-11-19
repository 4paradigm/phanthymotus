"""Kokoro Multi FP32 TensorRT adapter and JetPack launcher."""

import json
import logging
import multiprocessing
import os
from pathlib import Path
import re
import runpy
import sys
import tempfile
import threading
from time import monotonic

import numpy as np

from plugins.tts import PCM_FRAME_S, TTSAdapter
from utils.resample import downsample_24k_to_16k
from .kokoro_multi_frontend import (
    Frontend, LANGUAGES, ReadingSession, sentence_phoneme_chunks,
    sentence_parts, phoneme_chunks)

log = logging.getLogger(__name__)


class SequenceTooLong(ValueError):
    """Predicted audio exceeds the deployed decoder profile; re-split text."""


class KokoroTRT:
    def __init__(self, encoder_plan, acoustic_plan, *, vocoder_plan, device_id=0):
        import tensorrt as trt
        from utils.tensorrt_runtime import CudaRuntime, trt_dtype_to_numpy

        self._lock = threading.RLock()
        self._trt, self._numpy_dtype = trt, trt_dtype_to_numpy
        self._buffers, self._engines, self._contexts = {}, [], []
        paths = (encoder_plan, acoustic_plan, vocoder_plan)
        self._profiles = [None] * len(paths)
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._cuda = CudaRuntime(device_id)
        try:
            self._dynamic_activation = hasattr(trt, 'ExecutionContextAllocationStrategy')
            if not self._dynamic_activation:
                trt.init_libnvinfer_plugins(self._logger, '')
            self._runtime = trt.Runtime(self._logger)
            for path in paths:
                engine = self._runtime.deserialize_cuda_engine(Path(path).read_bytes())
                if engine is None:
                    raise RuntimeError(f"Cannot deserialize TensorRT plan: {path}")
                self._engines.append(engine)
                context = (engine.create_execution_context(
                    strategy=trt.ExecutionContextAllocationStrategy.USER_MANAGED)
                    if self._dynamic_activation else engine.create_execution_context_without_device_memory())
                if context is None:
                    raise RuntimeError(f"Cannot create TensorRT context: {path}")
                self._contexts.append(context)
            # TRT8 uses one shared engine-sized arena. TRT10 sizes it on demand.
            self._activation_bytes = 0
            if not self._dynamic_activation:
                self._activation_bytes = max(engine.device_memory_size for engine in self._engines)
                if self._activation_bytes:
                    pointer = self._buffer('activation', self._activation_bytes)
                    for context in self._contexts:
                        context.device_memory = pointer
            contracts = [({'tokens', 'style', 'speed'}, {'d', 't_en', 'duration'}),
                         ({'d', 't_en', 'indices', 'style'}, {'features', 'f0'}),
                         ({'features', 'f0', 'style'}, {'audio'})]
            for engine, (inputs, outputs) in zip(self._engines, contracts):
                names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
                actual = {n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT}
                if actual != inputs or names - actual != outputs:
                    raise ValueError(f"Unexpected TensorRT I/O: {actual}, {names - actual}")
                if any(engine.get_tensor_location(n) != trt.TensorLocation.DEVICE for n in names):
                    raise ValueError("Kokoro plans must expose device I/O")
            for name in ('d', 't_en', 'style'):
                if self._dtype(0, name) != self._dtype(1, name):
                    raise ValueError(f"A/B dtype mismatch for shared GPU tensor {name}")
            for name in ('features', 'f0', 'style'):
                if self._dtype(1, name) != self._dtype(2, name):
                    raise ValueError(f"Acoustic/vocoder dtype mismatch for {name}")
            limits = []
            duration_inputs = [(self._engines[1], 'indices', 1),
                               (self._engines[2], 'f0', 2)]
            for engine, name, ratio in duration_inputs:
                limits.append(max(engine.get_tensor_profile_shape(name, i)[2][-1]
                                  for i in range(engine.num_optimization_profiles)) // ratio)
            self.max_duration = min(limits)
        except BaseException:
            self.close()
            raise

    def _dtype(self, stage, name):
        return self._numpy_dtype(self._trt, self._engines[stage].get_tensor_dtype(name))

    def _shapes(self, stage, shapes):
        engine, context = self._engines[stage], self._contexts[stage]
        for profile in range(engine.num_optimization_profiles):
            covered = True
            for name, shape in shapes.items():
                declared = tuple(engine.get_tensor_shape(name))
                if all(d >= 0 for d in declared):
                    covered &= tuple(shape) == declared
                else:
                    low, _, high = engine.get_tensor_profile_shape(name, profile)
                    covered &= len(shape) == len(low) and all(a <= d <= b for d, a, b in zip(shape, low, high))
            if covered:
                break
        else:
            raise ValueError(f"Stage {stage} input shapes outside engine profiles: {shapes}")
        if self._profiles[stage] != profile:
            if not context.set_optimization_profile_async(profile, self._cuda.stream_handle):
                raise RuntimeError(f"Cannot select stage {stage} profile {profile}")
            self._profiles[stage] = profile
        for name, shape in shapes.items():
            if not context.set_input_shape(name, tuple(shape)):
                raise ValueError(f"Stage {stage} rejected {name} shape {shape}")
        if self._dynamic_activation:
            required = context.update_device_memory_size_for_shapes()
            if required <= 0:
                raise RuntimeError(f"Stage {stage} could not size activation memory for {shapes}")
            pointer = self._buffer('activation', required)
            self._activation_bytes = self._buffers['activation'][1]
            context.set_device_memory(pointer, self._activation_bytes)

    def _buffer(self, name, size):
        size = max(1, int(size))
        pointer, capacity = self._buffers.get(name, (0, 0))
        if size > capacity:
            if pointer:
                self._cuda.synchronize()
                self._cuda.free(pointer)
                self._buffers.pop(name)
            pointer = self._cuda.malloc(size)
            self._buffers[name] = (pointer, size)
        return pointer

    def _bind(self, stage, name, pointer):
        if not self._contexts[stage].set_tensor_address(name, pointer):
            raise RuntimeError(f"Cannot bind stage {stage} tensor {name}")

    def _upload(self, stage, name, array):
        pointer = self._buffer(f'{stage}:{name}', array.nbytes)
        self._bind(stage, name, pointer)
        self._cuda.copy_host_to_device(pointer, array)
        return pointer

    def _output(self, stage, name):
        shape = tuple(self._contexts[stage].get_tensor_shape(name))
        if any(d < 0 for d in shape):
            raise ValueError(f"Unresolved stage {stage} output {name}: {shape}")
        pointer = self._buffer(f'{stage}:{name}', int(np.prod(shape)) * self._dtype(stage, name).itemsize)
        self._bind(stage, name, pointer)
        return shape, pointer

    def infer(self, tokens, style, speed):
        """Return (audio, duration) after the synchronized GPU stages."""
        with self._lock:
            if self._cuda is None:
                raise RuntimeError("KokoroTRT is closed")
            try:
                inputs = {n: np.ascontiguousarray(v, dtype=self._dtype(0, n))
                          for n, v in (('tokens', tokens), ('style', style), ('speed', speed))}
                if inputs['tokens'].ndim != 2 or inputs['tokens'].shape[0] != 1:
                    raise ValueError("tokens must have shape [1,N]")
                n = inputs['tokens'].shape[1]
                self._shapes(0, {k: v.shape for k, v in inputs.items()})
                input_ptrs = {k: self._upload(0, k, v) for k, v in inputs.items()}
                outputs = {name: self._output(0, name) for name in ('d', 't_en', 'duration')}
                duration = np.empty(outputs['duration'][0], dtype=self._dtype(0, 'duration'))
                if not self._contexts[0].execute_async_v3(self._cuda.stream_handle):
                    raise RuntimeError("Kokoro encoder GPU execution failed")
                self._cuda.copy_device_to_host(duration, outputs['duration'][1])
                self._cuda.synchronize()
                if duration.shape != (n,) or duration.dtype.kind not in 'iu' or np.any(duration < 1):
                    raise ValueError(f"Invalid predicted duration: {duration.shape}, {duration.dtype}")
                if int(duration.sum()) > self.max_duration:
                    raise SequenceTooLong(
                        f"Predicted duration {duration.sum()} exceeds {self.max_duration}")
                indices = np.repeat(np.arange(n, dtype=np.int32), duration)
                indices = np.ascontiguousarray(indices, dtype=self._dtype(1, 'indices'))
                self._shapes(1, {'d': outputs['d'][0], 't_en': outputs['t_en'][0],
                                 'indices': indices.shape, 'style': inputs['style'].shape})
                for name in ('d', 't_en'):
                    self._bind(1, name, outputs[name][1])
                self._bind(1, 'style', input_ptrs['style'])
                self._upload(1, 'indices', indices)
                names = ('features', 'f0')
                decoder_outputs = {name: self._output(1, name) for name in names}
                if not self._contexts[1].execute_async_v3(self._cuda.stream_handle):
                    raise RuntimeError("Kokoro decoder GPU execution failed")
                self._cuda.synchronize()  # Release the shared arena before the vocoder.
                self._shapes(2, {**{name: decoder_outputs[name][0] for name in names},
                                 'style': inputs['style'].shape})
                for name in names:
                    self._bind(2, name, decoder_outputs[name][1])
                self._bind(2, 'style', input_ptrs['style'])
                shape, pointer = self._output(2, 'audio')
                if not self._contexts[2].execute_async_v3(self._cuda.stream_handle):
                    raise RuntimeError("Kokoro vocoder GPU execution failed")
                audio = np.empty(shape, dtype=self._dtype(2, 'audio'))
                self._cuda.copy_device_to_host(audio, pointer)
                self._cuda.synchronize()
                if not np.isfinite(audio).all():
                    raise ValueError("Kokoro audio contains non-finite values")
                return audio, duration
            except BaseException:
                self._cuda.synchronize()
                raise

    def close(self):
        with self._lock:
            if self._cuda is None:
                return
            self._cuda.synchronize()
            self._contexts.clear()
            self._engines.clear()
            self._runtime = None
            for pointer, _ in self._buffers.values():
                self._cuda.free(pointer)
            self._buffers.clear()
            self._cuda.close()
            self._cuda = None


def _run_jp5_trt(connection, model_dir, warmup):
    from utils import logsafe

    logsafe.install(check_fd=False)
    runtime = None
    try:
        try:
            root = Path(model_dir)
            runtime = KokoroTRT(root / "encoder.plan", root / "acoustic.plan",
                                vocoder_plan=root / "vocoder.plan")
            runtime.infer(*warmup)
        except Exception as error:
            connection.send(("error", str(error)))
            return
        connection.send(("ready", None))
        while True:
            try:
                inputs = connection.recv()
            except EOFError:
                return
            try:
                result = ("audio", runtime.infer(*inputs))
            except SequenceTooLong as error:
                result = ("duration", str(error))
            except Exception as error:
                result = ("error", str(error))
            connection.send(result)
    finally:
        if runtime is not None:
            runtime.close()
        connection.close()


class _JP5TRTWorker:
    def __init__(self, model_dir, warmup):
        context = multiprocessing.get_context("spawn")
        self._connection, child = context.Pipe()
        self._lock = threading.RLock()
        self._process = context.Process(target=_run_jp5_trt, args=(child, model_dir, warmup),
                                        name="kokoro-jp5-trt", daemon=True)
        try:
            try:
                self._process.start()
            finally:
                child.close()
            status, value = self._receive(120)
            if status != "ready":
                raise RuntimeError(f"Kokoro JP5 GPU startup failed: {value}")
        except BaseException:
            self.close()
            raise

    def _receive(self, seconds):
        if not self._connection.poll(seconds):
            raise TimeoutError(f"Kokoro JP5 GPU worker timed out after {seconds}s")
        return self._connection.recv()

    def infer(self, tokens, style, speed):
        with self._lock:
            if self._connection.closed:
                raise RuntimeError("Kokoro JP5 GPU worker is closed")
            try:
                self._connection.send((tokens, style, speed))
                status, value = self._receive(60)
            except (EOFError, OSError) as error:
                self.close()
                raise RuntimeError("Kokoro JP5 GPU worker stopped responding") from error
        if status == "duration":
            raise SequenceTooLong(value)
        if status != "audio":
            raise RuntimeError(f"Kokoro JP5 GPU inference failed: {value}")
        return value

    def close(self):
        with self._lock:
            if self._connection.closed:
                return
            self._connection.close()
            if self._process.pid is not None:
                self._process.join(2)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(2)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(2)
            if not self._process.is_alive():
                self._process.close()


class KokoroMultiTTSAdapter(TTSAdapter):
    LANGUAGES = LANGUAGES
    dry_run_text = "Hi."
    reuse_ros_node = True
    handles_text_chunking = True
    wait_for_audio_subscriber = True

    def __init__(self, model_dir, speaker_id=0, speed=1.0, device="gpu",
                 language="auto", on_status=None, benchmark=False):
        if device != "gpu":
            raise ValueError("Kokoro Multi TensorRT requires GPU")
        if speaker_id != 0:
            raise ValueError("Kokoro Multi selects its release voice per language; speaker_id must be 0")
        self.frame_interval_s = 0.0 if benchmark else PCM_FRAME_S
        self._lock = threading.RLock()
        self._runtime = None
        self.set_speed(speed)
        self.set_language(language)
        root = Path(model_dir)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["sample_rate"] != 24000 or manifest["precision"] != "float32":
            raise ValueError("Kokoro Multi requires its 24kHz FP32 release")
        self._vocab = json.loads((root / "vocab.json").read_text())
        self._voices = {lang: np.load(root / manifest["voices"][lang], allow_pickle=False)
                        for lang in LANGUAGES}
        if any(v.shape != (510, 1, 256) or v.dtype != np.float32
               or not np.isfinite(v).all() for v in self._voices.values()):
            raise ValueError("Kokoro Multi voice packs must be finite float32 [510,1,256]")
        if manifest.get("jetpack") == "5.1.1":
            if on_status:
                on_status("Loading TensorRT in the JP5 GPU worker")
            ids = [self._vocab[phone] for phone in "hˈI."]
            warmup = (np.array([[0, *ids, 0]], dtype=np.int64),
                      self._voices["en"][len(ids) - 1], np.array([self._speed], dtype=np.float32))
            self._runtime = _JP5TRTWorker(root, warmup)
            log.info("[kokoro-multi] JP5 GPU worker warmup complete")
            try:
                if on_status:
                    on_status("Loading eight-language frontend")
                self._frontend = Frontend(root)
            except BaseException:
                self.close()
                raise
            return
        if on_status:
            on_status("Loading eight-language frontend")
        self._frontend = Frontend(root)
        if on_status:
            on_status("Loading FP32 TensorRT")
        self._runtime = KokoroTRT(root / "encoder.plan", root / "acoustic.plan",
                                 vocoder_plan=root / "vocoder.plan")

    def set_speed(self, speed):
        speed = float(speed)
        if not np.isfinite(speed) or not 0.5 <= speed <= 2.0:
            raise ValueError("Kokoro Multi speed must be between 0.5 and 2.0")
        with self._lock:
            self._speed = speed

    def set_language(self, language):
        language = str(language).lower().replace("_", "-").split("-")[0]
        if language not in ("auto", *LANGUAGES):
            raise ValueError(f"Unsupported Kokoro Multi language: {language}")
        with self._lock:
            self._language = language

    def synthesize(self, text):
        return b"".join(self.synthesize_stream(text))

    def new_reading_session(self):
        return ReadingSession()

    def synthesize_stream_for_session(self, text, session, reading_language=None):
        yield from self.synthesize_stream(
            text, reading_session=session, reading_language=reading_language)

    def commit_reading_session(self, session):
        session.commit()

    def warmup(self):
        # Exercise each native frontend and realistic GPU shapes before readiness.
        phrases = (
            ("zh", "欢迎来到第四范式，请跟随工作人员参观展厅。"),
            ("en", "Welcome to the exhibition. Please follow our guide to the next room."),
            ("ja", "展示会へようこそ。係員の案内に従ってお進みください。"),
            ("ru", "Добро пожаловать на выставку. Пожалуйста, следуйте за нашим гидом."),
            ("de", "Willkommen in unserer Ausstellung. Bitte folgen Sie unserem Mitarbeiter."),
            ("fr", "Bienvenue dans notre exposition. Veuillez suivre notre guide."),
            ("es", "Bienvenidos a nuestra exposición. Por favor, sigan a nuestro guía."),
            ("th", "ยินดีต้อนรับเข้าสู่นิทรรศการของเรา กรุณาเดินตามเจ้าหน้าที่ไปยังห้องถัดไป"),
        )
        return sum(len(pcm) for language, text in phrases
                   for pcm in self.synthesize_stream(text, reading_language=language))

    def synthesize_stream(self, text, *, reading_session=None, reading_language=None):
        started = monotonic()
        with self._lock:
            preference = reading_language or self._language
            language = (reading_session.resolve(self._frontend, text, preference)
                        if reading_session is not None else
                        self._frontend.detect_language(text) if preference == "auto" else preference)
            speed = np.array([self._speed], dtype=np.float32)
        groups = []
        if preference == "auto" and language == "zh":
            inherited = reading_session.previous_language if reading_session is not None else None
            groups = self._frontend.bilingual_sentence_groups(text, inherited or language)
        samples, count = 0, 0
        for source, reading in groups or ((text, language),):
            for phones in self._frontend.iter_phonemes(source, reading):
                frontend_done = monotonic()
                for audio, phone_count in self._synthesize_phones(phones, reading, speed, source):
                    inference_done = monotonic()
                    pcm = downsample_24k_to_16k(audio)
                    if count == 0:
                        now = monotonic()
                        log.info("[kokoro-multi] first_pcm language=%s seconds=%.3f "
                                 "frontend_s=%.3f inference_s=%.3f resample_s=%.3f phones=%d",
                                 reading, now - started, frontend_done - started,
                                 inference_done - frontend_done, now - inference_done, phone_count)
                    samples += audio.size
                    count += 1
                    yield pcm
        if groups and reading_session is not None:
            reading_session.pending_language = groups[-1][1]
        log.info("[kokoro-multi] complete language=%s chunks=%d audio_s=%.3f wall_s=%.3f",
                 language, count, samples / 24000, monotonic() - started)

    def _synthesize_phones(self, phones, language, speed, source_text):
        phones = re.sub(r"\s+", " ", phones).strip()
        unknown = set(phones) - self._vocab.keys()
        decorations = {char for char in unknown if char in source_text and (
            0x1F000 <= ord(char) <= 0x1FAFF or char in "\u200d\ufe0e\ufe0f")}
        if decorations:
            log.info("[kokoro-multi] ignoring unsupported decorations: %s", sorted(decorations))
            phones = "".join(char for char in phones if char not in decorations)
            unknown -= decorations
        if unknown:
            raise ValueError(f"Kokoro Multi unsupported phonemes ({language}): {sorted(unknown)}")
        chunks = sentence_phoneme_chunks(phones, pack_tail=language in ("en", "fr", "es"))
        while chunks:
            part = chunks.pop(0)
            ids = [self._vocab[c] for c in part]
            tokens = np.array([[0, *ids, 0]], dtype=np.int64)
            style = self._voices[language][len(ids) - 1]
            try:
                audio, duration = self._runtime.infer(tokens, style, speed)
            except SequenceTooLong:
                # Nothing from this chunk has been published. Keep complete
                # sentences before reducing to smaller word groups.
                smaller = sentence_parts(part)
                if len(smaller) == 1:
                    limit = len(part) // 2
                    if limit < 1:
                        raise
                    smaller = phoneme_chunks(part, limit=limit,
                                             preferred=limit, first_limit=0)
                chunks[:0] = smaller
                continue
            audio = audio.reshape(-1)
            if audio.size != 600 * int(duration.sum()):
                raise RuntimeError("Kokoro Multi output length differs from predicted duration")
            yield audio, len(ids)

    def close(self):
        if self._runtime is not None:
            self._runtime.close()


WORK = Path(__file__).resolve().parents[1]
ASSETS = {
    "511": (
        "https://www.modelscope.cn/api/v1/models/lanxijiang/Kokoro-Multi-TRT-JP61/repo?Revision=3a2c495b3710257ef4a855747c59edcf7763bf55&FilePath=kokoro-multi-jp511-fp32-v18-pronunciation.tar.gz",
        "2f0c0246b4bd8790b41ded690b9ba7e022c5dd09bfc6ba81c9c630bcbf0125c0",
        332864006,
        "/models/kokoro-multi-trt-jp511",
    ),
    "61": (
        "https://www.modelscope.cn/api/v1/models/lanxijiang/Kokoro-Multi-TRT-JP61/repo?Revision=91be48e84e7652d898197ac308a8f4c066811ac0&FilePath=kokoro-multi-jp61-fp32-v17-pronunciation.tar.gz",
        "7b550b1966bfc7151eb211529d3df64169f6ed5c5d117ef41023a09c33b839f6",
        326831088,
        "/models/kokoro-multi-trt",
    ),
}


def enabled():
    jp = os.environ.get("KOKORO_MULTI_BUILT_FOR_JP", "")
    value = os.environ.get("KOKORO_MULTI_ENABLE", "0")
    if value not in ("0", "1"):
        raise ValueError("KOKORO_MULTI_ENABLE must be 0 or 1")
    if value == "1" and jp not in ASSETS:
        raise RuntimeError("Kokoro Multi requires a prepared JetPack 5.1.1 or 6.1 image")
    return value == "1"


def prepare_model(model_dir, on_status=None):
    from utils.model_downloader import ensure_verified_archive, require_models_subpath
    from utils.model_progress import fetch_status

    jp = os.environ.get("KOKORO_MULTI_BUILT_FOR_JP", "")
    if jp not in ASSETS:
        raise RuntimeError("Kokoro Multi requires a prepared JetPack 5.1.1 or 6.1 image")
    asset_url, asset_sha256, asset_size, cache = ASSETS[jp]
    url = os.environ.get("KOKORO_MULTI_ASSET_URL", asset_url)
    sha256 = os.environ.get("KOKORO_MULTI_ASSET_SHA256", asset_sha256)
    size = int(os.environ.get("KOKORO_MULTI_ASSET_SIZE", str(asset_size)))
    if not url or not re.fullmatch(r"[0-9a-f]{64}", sha256) or size <= 0:
        raise RuntimeError(
            "Kokoro Multi release is not pinned: set KOKORO_MULTI_ASSET_URL, "
            "KOKORO_MULTI_ASSET_SHA256 and KOKORO_MULTI_ASSET_SIZE"
        )
    model_dir = require_models_subpath(os.environ.get("KOKORO_MULTI_MODEL_DIR", model_dir or cache))
    progress_cb, stage_cb = fetch_status(on_status, "kokoro-multi")
    ensure_verified_archive(
        f"kokoro-multi-jp{jp}", model_dir, url, {"sha256": sha256, "size": size},
        progress_cb=progress_cb, stage_cb=stage_cb)
    return model_dir


def prepare_config():
    import yaml

    source = Path(os.environ.get("CONFIG_PATH", str(WORK / "config.yaml")))
    with source.open() as handle:
        cfg = yaml.safe_load(handle)
    benchmark = os.environ.get("KOKORO_MULTI_BENCHMARK", "0") == "1"
    tts = {
        "enabled": True,
        "engine": "kokoro-multi",
        "backend": "trt",
        "benchmark": benchmark,
        "device": "gpu",
        "language": "auto",
        "warmup": True,
        "speaker_id": 0,
        "speed": 1.0,
    }
    if benchmark:
        # Darvin drains PCM without live playback.
        cfg["plugins"] = {"tts": tts, "echo_cancellation": {"enabled": False},
                          "audio_mixer": {"enabled": False}}
    else:
        cfg.setdefault("plugins", {})["tts"] = tts
    for key, default in (("mcp_port", 15720), ("ws_port", 15721)):
        port = int(os.environ.get(key.upper(), cfg.get(key, default)))
        if not 1 <= port <= 65535:
            raise ValueError(f"Invalid {key}: {port}")
        cfg[key] = port
    return cfg


def main():
    sys.path.insert(0, str(WORK))
    if not enabled():
        # No model loading, downloads, or configuration changes.
        runpy.run_path(str(WORK / "main.py"), run_name="__main__")
        return

    import yaml
    logging.basicConfig(level=logging.INFO)
    cfg = prepare_config()
    with tempfile.TemporaryDirectory(prefix="kokoro-multi-config-") as directory:
        config = Path(directory) / "config.yaml"
        config.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        os.environ["CONFIG_PATH"] = str(config)
        # Default Fast DDS announcements can delay a new reader by 3 seconds.
        # Scope the shorter announcement period to this Kokoro process only.
        if cfg["plugins"]["tts"]["benchmark"] and "FASTRTPS_DEFAULT_PROFILES_FILE" not in os.environ:
            discovery = Path(directory) / "fastdds.xml"
            discovery.write_text(
                '<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">'
                '<participant profile_name="kokoro_tts_discovery" is_default_profile="true">'
                '<rtps><builtin><discovery_config><leaseAnnouncement>'
                '<sec>0</sec><nanosec>100000000</nanosec>'
                '</leaseAnnouncement></discovery_config></builtin></rtps>'
                '</participant></profiles>')
            os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = str(discovery)
        logging.info("Kokoro Multi jp%s: MCP port %s, WS port %s",
                     os.environ["KOKORO_MULTI_BUILT_FOR_JP"],
                     cfg["mcp_port"], cfg["ws_port"])
        runpy.run_path(str(WORK / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
