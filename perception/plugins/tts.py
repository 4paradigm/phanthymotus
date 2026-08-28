#!/usr/bin/env python3
"""
plugins/tts.py — TTSPlugin: sherpa-onnx VITS TTS.

On-device text-to-speech using sherpa-onnx MeloTTS (Chinese + English).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_BYTES = 3200  # 100ms @ 16kHz 16-bit mono
PCM_FRAME_S = CHUNK_BYTES / (SAMPLE_RATE * 2)  # 0.1s of audio per frame

# Frames held back before pacing starts, then published in one burst, so the
# consumer begins with a real cushion. 5 frames = 500ms, matching the
# vits2_tts_trt engine's MIX_VITS_PREBUFFER_FRAMES default. This — not the pacing
# interval — is where the downstream margin comes from.
PREBUF_FRAMES = 5
# Pace at exactly the audio each frame carries. Anything shorter over-delivers
# forever, and over-delivery has no safe landing on a live consumer: it either
# buffers without bound or has to discard audio. Briefly setting this to 0.07s
# (matching what the vits2 engine then did) accrued 30ms of surplus per frame
# until the browser player hit its lead cap, rewound its own schedule into audio
# it had already queued, and played back overlapped and 1.43x too fast.
FRAME_INTERVAL_S = PCM_FRAME_S
if FRAME_INTERVAL_S > PCM_FRAME_S:
    log.warning(
        "[tts] FRAME_INTERVAL_S=%.3fs is slower than the %.3fs of audio each "
        "frame carries; downstream will underrun on every utterance",
        FRAME_INTERVAL_S, PCM_FRAME_S,
    )
# Depth of the synthesis→publish handoff queue, in frames (20s of audio).
SYNTH_QUEUE_FRAMES = 200

# Sentinel closing the synthesis→publish queue. A dedicated object rather than
# None so a genuinely empty frame could never be mistaken for end-of-stream.
_SYNTH_DONE = object()

# EOF magic: 8 bytes (4 samples [1, -1, 1, -1])，标记 utterance 结束
# 正常 chunk 始终 3200 bytes，8 bytes 短 chunk 不会被误判
# 即使被不识别 EOF 的旧 Speaker 播放，也只是 0.25ms 极微弱交流声
AUDIO_EOF_MAGIC = b'\x01\x00\xff\xff\x01\x00\xff\xff'

_LOW_LAT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=200,
    durability=DurabilityPolicy.VOLATILE,
)


def _agent_core_url() -> str:
    import os
    return os.environ.get("AGENT_CORE_URL", "https://localhost:15678")


def _unverified_ssl_context():
    """Agent Core serves HTTPS with a self-signed certificate."""
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _post_json(path: str, payload: dict, timeout: float) -> None:
    import urllib.request
    request = urllib.request.Request(
        f"{_agent_core_url()}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(request, timeout=timeout, context=_unverified_ssl_context())


def fire_hook(name: str) -> None:
    """Fire an agent-core hook without blocking the caller.

    on_speaking used to be a synchronous urlopen(timeout=2) inside the publish
    loop, fired after the pacing clock had already been latched — so the request
    latency was charged to the utterance's pacing budget and every frame after
    the prebuffer went out late. Hooks are advisory (LED state); they must never
    sit between two audio frames.
    """
    def _send():
        try:
            _post_json("/api/hooks/fire", {"hook": name}, timeout=2)
        except Exception as exc:
            log.debug("[tts] hook %s failed: %s", name, exc)

    threading.Thread(target=_send, name=f"tts-hook-{name}", daemon=True).start()


def _complete_action(action_id: str, text: str, frames_sent: int,
                     interrupted: bool) -> None:
    """Notify Agent Core that a speak action has terminated.

    Module level, not a node method: an utterance can also die before the worker
    ever sees it (interrupt/stop drains the queue), and the ACP barrier in
    agent-core waits out its full timeout for every action it registered but
    never heard back about.
    """
    if not action_id:
        return
    try:
        _post_json("/api/acp/complete", {
            "action_id": action_id,
            "status": "cancelled" if interrupted else "completed",
            "result": {"text": text[:100], "frames": frames_sent},
        }, timeout=3)
        log.info("[tts] ACP complete: %s (%s)", action_id,
                 "cancelled" if interrupted else "completed")
    except Exception as exc:
        log.warning("[tts] ACP callback failed: %s", exc)


TOOLS = [
    {
        "name": "tts",
        "type": "processor",
        "multiInstance": True,
        "description": "TTS — start/stop speech synthesis, speak text, or get status",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "speak", "info", "config", "interrupt"],
                    "description": "Action to perform"
                },
                "input_topic": {
                    "type": "string",
                    "description": "ROS2 topic for text input (data/json, required for action=start)"
                },
                "text": {
                    "type": "string",
                    "description": "Text to synthesize (required for action=speak)"
                },
            },
            "required": ["action"],
            "x-completion": {
                "actions": ["speak"],
                "timeout": 60
            },
            "x-hooks": {
                "on_interrupt_speak": {"action": "interrupt"},
            }
        },
        "configSchema": {
            "type": "object",
            "properties": {
                # Engine belongs here, not only in config.yaml: the dashboard
                # builds the config form from configSchema, so an engine that
                # exists solely as a baked YAML key cannot be seen or switched
                # without rebuilding the image. Mirrors asr_model in asr.py.
                "tts_engine": {"type": "string", "enum": ["vits2_trt", "sherpa_onnx"],
                               "description": "TTS engine (vits2_trt = VITS2 TensorRT on Jetson, "
                                              "sherpa_onnx = sherpa-onnx Matcha)",
                               "default": "vits2_trt", "scope": "shared"},
                # sherpa_onnx only — vits2_trt is a TensorRT engine and never
                # touches ONNX Runtime, so this field does nothing for it. Matcha's
                # weights are fp32, so both devices load the same files and only
                # the provider changes; measured 4.3x faster on gpu.
                "device":      {"type": "string", "enum": ["cpu", "gpu"],
                                "description": "Inference device for the sherpa_onnx engine "
                                               "(gpu needs the CUDA sherpa-onnx wheel; ~4.3x faster)",
                                "default": "cpu", "scope": "shared",
                                "x-show-when": {"tts_engine": "sherpa_onnx"}},
                "speaker_id": {"type": "integer", "description": "Speaker ID (VITS2 supports 0 only)", "default": 0, "scope": "shared"},
                "speed":      {"type": "number", "description": "Speech speed (1.0 = normal)", "default": 1.0, "scope": "shared"},
            },
            "required": []
        },
        "topic_in":  [{"format": "data/json",     "desc": "text to synthesize"}],
        "topic_out": [{"format": "audio/pcm-16k", "desc": "synthesized PCM audio"}],
    }
]


# ── TTS Adapter ──────────────────────────────────────────────────────────────

class TTSAdapter(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> bytes: ...

    def synthesize_stream(self, text: str):
        """Yield raw PCM bytes as they arrive. Default: collect all."""
        yield self.synthesize(text)


class SherpaOnnxTTSAdapter(TTSAdapter):
    """On-device TTS using sherpa-onnx Matcha (flow-matching, fast non-autoregressive)."""

    def __init__(self, model_dir: str, speaker_id: int = 0, speed: float = 1.0,
                 device: str = "cpu"):
        import os
        from utils.model_downloader import ensure_model
        from utils.onnx_provider import provider_for_device
        ensure_model("tts", model_dir)
        ensure_model("tts_vocoder", model_dir)

        import sherpa_onnx
        # Matcha model files
        acoustic_model = os.path.join(model_dir, "model-steps-3.onnx")
        vocoder = os.path.join(model_dir, "vocos-16khz-univ.onnx")
        lexicon_path = os.path.join(model_dir, "lexicon.txt")
        tokens_path = os.path.join(model_dir, "tokens.txt")
        data_dir = os.path.join(model_dir, "espeak-ng-data")
        if not os.path.isdir(data_dir):
            data_dir = ""
        # Both weights are fp32, so there is only one file set and device just
        # picks the provider — measured 4.3x faster on gpu at num_threads=2.
        provider = provider_for_device(device, (acoustic_model, vocoder))

        # Gather rule FSTs
        rule_fsts = []
        for name in ("date-zh.fst", "number-zh.fst", "phone-zh.fst"):
            p = os.path.join(model_dir, name)
            if os.path.exists(p):
                rule_fsts.append(p)

        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                    acoustic_model=acoustic_model,
                    vocoder=vocoder,
                    lexicon=lexicon_path if os.path.exists(lexicon_path) else "",
                    tokens=tokens_path,
                    data_dir=data_dir,
                    length_scale=1.0 / speed if speed else 1.0,
                ),
                num_threads=2,
                provider=provider,
            ),
            rule_fsts=",".join(rule_fsts) if rule_fsts else "",
        )
        self._tts = sherpa_onnx.OfflineTts(tts_config)
        self._sid = speaker_id
        self._speed = speed
        log.info(f"[tts] sherpa-onnx Matcha loaded: model_dir={model_dir}, "
                 f"speaker_id={speaker_id}, speed={speed}, "
                 f"device={device}, provider={provider}")

    def synthesize(self, text: str) -> bytes:
        return b''.join(self.synthesize_stream(text))

    def synthesize_stream(self, text: str):
        import struct
        audio = self._tts.generate(text, sid=self._sid, speed=self._speed)
        float_samples = audio.samples
        # Matcha + vocos-16khz outputs 16kHz directly, no resampling needed
        pcm = struct.pack(f'<{len(float_samples)}h',
                         *[int(max(-32768, min(32767, s * 32767))) for s in float_samples])
        for i in range(0, len(pcm), CHUNK_BYTES):
            yield pcm[i:i + CHUNK_BYTES]




def _build_tts_adapter(cfg: dict) -> TTSAdapter:
    import os
    from utils.onnx_provider import normalize_device
    model_dir = cfg.get('model_dir', '/models/sherpa-onnx/tts')
    speaker_id = int(cfg.get('speaker_id', 0))
    speed = float(cfg.get('speed', 1.0))
    device = normalize_device(cfg.get('device'), cfg.get('hw_provider'))
    return SherpaOnnxTTSAdapter(model_dir, speaker_id, speed, device)


# ── ROS2 Node ─────────────────────────────────────────────────────────────────

class _TTSNode(Node):
    def __init__(self, input_topic: Optional[str], adapter: Optional[TTSAdapter], node_suffix: str = ''):
        node_name = f"tts_{node_suffix}" if node_suffix else "tts"
        super().__init__(node_name)
        self._input_topic  = input_topic or ''
        self._output_topic = f"{input_topic}/tts" if input_topic else '/perception/tts'
        self._adapter      = adapter
        self.state         = "idle"
        self._text_queue   = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event   = threading.Event()
        self._interrupt_flag = threading.Event()  # 打断标志：设置后立即停止当前 utterance
        from audio_msgs.msg import AudioChunk
        self._pub = self.create_publisher(AudioChunk, self._output_topic, _LOW_LAT_QOS)
        self._perf_pub = self.create_publisher(String, '/perception/perf_spans', _LOW_LAT_QOS)
        if input_topic:
            self._sub = self.create_subscription(String, self._input_topic, self._text_cb, _LOW_LAT_QOS)
        else:
            self._sub = None
        log.info(f"[tts] node created: subscribing={self._input_topic or '(none)'}, publishing={self._output_topic}")

    def start(self) -> dict:
        while not self._text_queue.empty():
            try: self._text_queue.get_nowait()
            except Exception: break
        if self.state == "running":
            return self._status_dict()
        if not self._adapter:
            raise RuntimeError("TTS adapter not configured")
        # Dry-run: verify model can synthesize before declaring running
        try:
            test_chunks = list(self._adapter.synthesize_stream("."))
            if not test_chunks:
                return {"state": "error", "message": "TTS dry-run produced no audio"}
        except Exception as e:
            return {"state": "error", "message": f"TTS dry-run failed: {e}"}
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        self.state = "running"
        return self._status_dict()

    def stop(self) -> dict:
        self._stop_event.set()
        self._complete_discarded_actions()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3)
        self.state = "idle"
        return {"state": "idle"}

    def _complete_discarded_actions(self) -> int:
        """Cancel queued ACP actions that will never reach the worker."""
        discarded = []
        while True:
            try:
                item = self._text_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, tuple):
                text = str(item[0])
                action_id = item[2] if len(item) >= 3 else ''
            else:
                text, action_id = str(item), ''
            discarded.append((text, action_id))
        for text, action_id in discarded:
            _complete_action(action_id, text, 0, interrupted=True)
        return len(discarded)

    def interrupt(self) -> dict:
        """立即中止当前播放：清空队列 + 设置 interrupt flag 让 worker 停止当前 utterance。"""
        # 清空待播放队列。丢掉的 item 必须逐个回 ACP cancelled，否则 agent-core
        # 的 barrier 会为每个注册过的 action 干等到超时。
        cleared = self._complete_discarded_actions()
        # 设置 interrupt flag（worker 在每个 frame 前检查）
        self._interrupt_flag.set()
        log.info(f"[tts] interrupted: cleared {cleared} queued item(s)")
        return {"status": "interrupted", "cleared": cleared}

    def enqueue(self, text: str, trace_id: str = '', action_id: str = ''):
        if self.state != "running":
            raise RuntimeError("TTS not running; call start first")
        # One queue item = one utterance = one EOF = one ACP action. The 280-char
        # split happens inside the worker instead: splitting here put each
        # segment on the queue as its own utterance, so a long text emitted an
        # EOF and reset the pacing clock every 280 characters, and the gap
        # between segments was a full synthesis with no audio flowing at all.
        self._text_queue.put((text, trace_id, action_id))

    @staticmethod
    def _split_text(text: str, max_chars: int = 280) -> list:
        """按标点分段，每段不超过 max_chars 字。"""
        import re as _re
        sentences = _re.split(r'(?<=[。！？；\n])', text)
        segments = []
        current = ""
        for sent in sentences:
            if not sent:
                continue
            if len(current) + len(sent) > max_chars and current:
                segments.append(current)
                current = sent
            else:
                current += sent
        if current:
            segments.append(current)
        return segments if segments else [text]

    def _text_cb(self, msg: String):
        if self.state != "running": return
        try:
            text = json.loads(msg.data).get("text","")
        except Exception:
            text = msg.data.strip()
        if text:
            log.info(f"[tts] received text from topic: {text[:50]}...")
            self._text_queue.put((text, ''))

    def _worker(self):
        from audio_msgs.msg import AudioChunk
        import time as _time

        while not self._stop_event.is_set():
            try:
                item = self._text_queue.get(timeout=1)
            except queue.Empty:
                continue
            # Unpack queue item: (text, trace_id, action_id) or legacy formats
            if isinstance(item, tuple):
                if len(item) == 3:
                    text, _trace_id, _action_id = item
                elif len(item) == 2:
                    text, _trace_id = item
                    _action_id = ''
                else:
                    text, _trace_id, _action_id = str(item[0]), '', ''
            else:
                text, _trace_id, _action_id = item, '', ''
            synth_thread = None
            # Set on every exit path. The stop/interrupt flags are not enough to
            # release the synth thread: if the consumer dies on an exception,
            # nothing is cancelled and nothing is draining, so a blocking put
            # would wedge that thread forever. Bound before the try so the
            # finally can always reach it.
            utterance_abort = threading.Event()
            try:
                t_start = _time.monotonic()
                t_start_wall = _time.time()  # wall-clock for perf span
                t0_wall = None  # wall-clock when playback starts (prebuf complete)
                total = 0
                buf = b''
                t0 = None  # monotonic start of the pacing schedule
                frames_sent = 0
                prebuf = []   # pre-buffer queue

                def publish(frame: bytes) -> None:
                    nonlocal frames_sent
                    msg = AudioChunk()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.format = "audio/pcm-16k"
                    msg.data = list(frame)
                    self._pub.publish(msg)
                    frames_sent += 1

                def emit(frame: bytes) -> None:
                    """Pace and publish one frame; latches the clock on the first."""
                    nonlocal t0, t0_wall
                    if t0 is None:
                        # Backdate by the frames already in the prebuffer so all
                        # of them are due in the past and go out in one burst —
                        # the consumer then starts holding PREBUF_FRAMES of audio.
                        now = _time.monotonic()
                        t0 = now - max(0, len(prebuf) - 1) * FRAME_INTERVAL_S
                        t0_wall = _time.time()
                        fire_hook("on_speaking")
                    target = t0 + frames_sent * FRAME_INTERVAL_S
                    now = _time.monotonic()
                    if now < target:
                        _time.sleep(target - now)
                    # No rebase when behind: publishing immediately is the
                    # catch-up, since the schedule is already ahead of realtime.
                    publish(frame)

                def flush_prebuf() -> None:
                    while prebuf:
                        emit(prebuf[0])
                        prebuf.pop(0)

                # 分段：超过 280 字按标点切分，避免超长合成导致延迟或失败。分段是
                # 合成的实现细节 —— pacing/prebuffer/EOF 都跨段延续，下游看到的
                # 仍然是一句完整的话。
                segments = self._split_text(text, max_chars=280)
                if len(segments) > 1:
                    log.info(f"[tts] split {len(text)} chars into {len(segments)} segments")

                # Synthesis runs on its own thread. This adapter's
                # synthesize_stream calls generate() once per segment and blocks
                # until the whole segment's audio exists, so doing it on the
                # publishing thread meant no frame at all went out for the
                # duration of every segment after the first — a guaranteed gap
                # every 280 characters, as long as the synthesis took. Here
                # segment N+1 is synthesized while segment N is still playing.
                frame_queue: queue.Queue = queue.Queue(maxsize=SYNTH_QUEUE_FRAMES)
                synth_state: dict = {"total": 0}

                def enqueue_frame(frame) -> bool:
                    """Blocking put that still honours interrupt/stop/abort."""
                    while True:
                        if (self._stop_event.is_set() or self._interrupt_flag.is_set()
                                or utterance_abort.is_set()):
                            return False
                        try:
                            frame_queue.put(frame, timeout=0.1)
                            return True
                        except queue.Full:
                            continue

                def synthesize_into_queue() -> None:
                    pending = b''
                    try:
                        for seg in segments:
                            for raw_chunk in self._adapter.synthesize_stream(seg):
                                pending += raw_chunk
                                synth_state["total"] += len(raw_chunk)
                                while len(pending) >= CHUNK_BYTES:
                                    frame, pending = pending[:CHUNK_BYTES], pending[CHUNK_BYTES:]
                                    if not enqueue_frame(frame):
                                        return
                        if pending:
                            enqueue_frame(pending)
                    except BaseException as exc:  # surfaced on the worker thread
                        synth_state["error"] = exc
                    finally:
                        # Unblock the consumer on every exit path.
                        enqueue_frame(_SYNTH_DONE)

                synth_thread = threading.Thread(
                    target=synthesize_into_queue, name="tts-synth", daemon=True)
                synth_thread.start()

                interrupted = False
                while True:
                    if self._stop_event.is_set() or self._interrupt_flag.is_set():
                        interrupted = True
                        break
                    try:
                        frame = frame_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if frame is _SYNTH_DONE:
                        break
                    if len(frame) < CHUNK_BYTES:
                        # Trailing partial frame — paced, then done.
                        buf = frame
                        break
                    if t0 is None:
                        prebuf.append(frame)
                        if len(prebuf) >= PREBUF_FRAMES:
                            flush_prebuf()
                        continue
                    emit(frame)

                cancelled = self._stop_event.is_set() or self._interrupt_flag.is_set()
                synth_thread.join(timeout=5)
                if "error" in synth_state and not cancelled:
                    raise synth_state["error"]
                total = synth_state["total"]  # read after join, not concurrently
                # Flush any remaining pre-buffer (utterances < PREBUF_FRAMES)
                if prebuf and not cancelled:
                    flush_prebuf()

                # flush remainder
                if buf and not cancelled:
                    if t0 is not None:
                        target = t0 + frames_sent * FRAME_INTERVAL_S
                        now = _time.monotonic()
                        if now < target:
                            _time.sleep(target - now)
                    publish(buf)

                # Capture before clearing: reading the flag after the clear below
                # made this always False, so an interrupted utterance reported ACP
                # "completed" instead of "cancelled".
                was_interrupted = self._interrupt_flag.is_set()
                # Clear interrupt flag after utterance is done (interrupted or complete)
                if was_interrupted:
                    self._interrupt_flag.clear()
                    log.info(f"[tts] utterance interrupted after {frames_sent} frames")
                else:
                    log.info(f"[tts] spoke {len(text)} chars → {total} bytes ({frames_sent} frames) in {_time.monotonic() - t_start:.2f}s")

                # 发布 EOF 标记：告知下游 Speaker 当前 utterance 已结束
                self._publish_eof()
                # 上报 TTS perf spans（生成 + 播放）
                try:
                    import json as _json
                    t_end_wall = _time.time()
                    spans = []
                    _span_base = {"type": "perf_span", "component": "perception"}
                    if _trace_id:
                        _span_base["trace_id"] = _trace_id
                    if t0_wall:
                        spans.append({**_span_base, "span": "tts_generate",
                                      "start_ts": t_start_wall, "end_ts": t0_wall,
                                      "meta": {"chars": len(text)}})
                        spans.append({**_span_base, "span": "tts_playback",
                                      "start_ts": t0_wall, "end_ts": t_end_wall,
                                      "meta": {"frames": frames_sent}})
                    else:
                        # 没有 prebuf（极短文本），合并为一个 span
                        spans.append({**_span_base, "span": "tts_generate",
                                      "start_ts": t_start_wall, "end_ts": t_end_wall,
                                      "meta": {"chars": len(text), "frames": frames_sent}})
                    for sp in spans:
                        perf_msg = String()
                        perf_msg.data = _json.dumps(sp)
                        self._perf_pub.publish(perf_msg)
                except Exception:
                    pass

                # ACP: 推送动作完成回调到 Agent Core
                # Also fire on_idle hook (LED off immediately after playback)
                if _action_id:
                    fire_hook("on_idle")
                    _complete_action(_action_id, text, frames_sent, was_interrupted)
            except Exception as e:
                log.error(f"[tts] synthesis error: {e}", exc_info=True)
                _complete_action(_action_id, text, 0, interrupted=True)
                self._publish_eof()
            finally:
                # Release the synth thread on every path, including the one where
                # the consumer above died: it can otherwise sit on a blocking put
                # forever, holding a reference to the adapter.
                utterance_abort.set()
                if synth_thread is not None and synth_thread.is_alive():
                    synth_thread.join(timeout=5)
                    if synth_thread.is_alive():
                        log.error("[tts] synth thread did not exit")

    def _status_dict(self) -> dict:
        return {
            "state":     self.state,
            "topic_in":  [{"topic": self._input_topic,  "format": "data/json",     "desc": "text to synthesize"}],
            "topic_out": [{"topic": self._output_topic, "format": "audio/pcm-16k", "desc": "synthesized PCM audio"}],
        }

    def _publish_eof(self):
        """发布 EOF magic chunk，标记当前 utterance 结束。"""
        from audio_msgs.msg import AudioChunk
        msg = AudioChunk()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "audio/pcm-16k"
        msg.data = list(AUDIO_EOF_MAGIC)
        self._pub.publish(msg)


# ── Plugin ────────────────────────────────────────────────────────────────────

class SherpaOnnxTTSPlugin:
    PREFIX = "tts"

    def __init__(self, plugin_cfg: dict, executor):
        self._cfg      = plugin_cfg
        self._loading  = False
        self._load_error = None
        try:
            self._adapter  = _build_tts_adapter(plugin_cfg)
        except Exception as e:
            log.error(f"[tts] failed to load model: {e}", exc_info=True)
            self._adapter = None
            self._load_error = str(e)
        self._nodes: dict[str, _TTSNode] = {}
        # main.py serves MCP over ThreadingHTTPServer, so start/stop/speak/config
        # can run concurrently. Every read-modify-write of _nodes must hold this:
        # otherwise two threads both pass a "key not in _nodes" check, both build
        # a node, and the dict keeps only the last — leaving the other running but
        # unreachable, with a duplicate publisher on the same topic that nothing
        # can stop. See perception/README.md § Plugin Concurrency.
        # RLock: dispatch paths nest (start → _dispose_node).
        self._nodes_lock = threading.RLock()
        self._executor = executor
        log.info(f"[tts] plugin init: sherpa-onnx VITS, "
                 f"speaker_id={plugin_cfg.get('speaker_id', 0)}, speed={plugin_cfg.get('speed', 1.0)}")

    def _dispose_node(self, node: _TTSNode, key: str = "") -> dict:
        """Stop a node and release its ROS endpoints. Caller holds _nodes_lock.

        destroy_node() matters: without it the publisher and the ROS node name
        outlive the node object, so a later start on the same key collides with a
        still-registered ghost.
        """
        result = {"state": "idle"}
        try:
            result = node.stop()
        except Exception:
            log.error(f"[tts] node.stop() failed while disposing '{key}'", exc_info=True)
        try:
            self._executor.remove_node(node)
        except Exception as error:
            log.warning(f"[tts] failed to remove ROS node '{key}': {error}")
        try:
            node.destroy_node()
        except Exception as error:
            log.warning(f"[tts] failed to destroy ROS node '{key}': {error}")
        return result

    def get_tools(self) -> list:
        return TOOLS

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action") if name == "tts" else name
        instance_id = args.get("instance_id", "")

        if action == "info":
            if self._loading:
                return {
                    "name": "TTS", "manufacture": "Embodied", "model": "tts",
                    "state": "loading",
                    "desc": "Downloading TTS model...",
                }
            if self._load_error:
                return {
                    "name": "TTS", "manufacture": "Embodied", "model": "tts",
                    "state": "error",
                    "desc": f"Model load failed: {self._load_error}",
                }
            input_topic = args.get("input_topic", "")
            # Snapshot under the lock: info is a heartbeat probe and iterating the
            # live dict can raise "dictionary changed size" mid-start.
            with self._nodes_lock:
                node = self._nodes.get(instance_id) if instance_id else None
                nodes_snapshot = list(self._nodes.values())
            if instance_id and node is not None:
                return {
                    "name": "TTS", "manufacture": "Embodied", "model": "tts",
                    "state": node.state,
                    "topic_in":  [{"topic": node._input_topic,  "format": "data/json",     "desc": ""}],
                    "topic_out": [{"topic": node._output_topic, "format": "audio/pcm-16k", "desc": ""}],
                    "desc": "TTS service — converts text to audio/pcm-16k",
                }
            if instance_id:
                # Instance requested but not running — return inferred topics for this instance only.
                inferred_out = f"{input_topic}/tts" if input_topic else "/perception/tts"
                return {
                    "name": "TTS", "manufacture": "Embodied", "model": "tts",
                    "state": "idle",
                    "topic_in":  [{"topic": input_topic,  "format": "data/json",     "desc": ""}] if input_topic else [],
                    "topic_out": [{"topic": inferred_out, "format": "audio/pcm-16k", "desc": ""}],
                    "desc": "TTS service — converts text to audio/pcm-16k",
                }
            # Aggregate info (no instance_id = ping/overview only)
            if nodes_snapshot:
                topics_in = [{"topic": n._input_topic, "format": "data/json", "desc": ""} for n in nodes_snapshot]
                topics_out = [{"topic": n._output_topic, "format": "audio/pcm-16k", "desc": ""} for n in nodes_snapshot]
                states = list(set(n.state for n in nodes_snapshot))
                state = "running" if "running" in states else states[0] if states else "idle"
            else:
                inferred_out = f"{input_topic}/tts" if input_topic else "/perception/tts"
                topics_in = [{"topic": input_topic, "format": "data/json", "desc": ""}]
                topics_out = [{"topic": inferred_out, "format": "audio/pcm-16k", "desc": ""}]
                state = "idle"
            return {
                "name": "TTS", "manufacture": "Embodied", "model": "tts",
                "state": state,
                "topic_in": topics_in,
                "topic_out": topics_out,
                "desc": "TTS service — converts text to audio/pcm-16k",
            }

        elif action == "start":
            if self._loading:
                return {"state": "loading", "message": "TTS model is being downloaded, please wait..."}
            if self._load_error:
                return {"state": "error", "message": f"TTS model failed to load: {self._load_error}"}
            if not self._adapter:
                return {"state": "error", "message": "TTS model not loaded"}
            input_topic = args.get("input_topic") or ''
            node_key = instance_id or input_topic or '_default'
            with self._nodes_lock:
                # Clean up _default node if it would conflict with this instance
                if '_default' in self._nodes and node_key != '_default':
                    default_node = self._nodes['_default']
                    if default_node._input_topic == input_topic or default_node._output_topic == (f"{input_topic}/tts" if input_topic else '/perception/tts'):
                        del self._nodes['_default']
                        self._dispose_node(default_node, '_default')
                node = self._nodes.get(node_key)
                if node is None:
                    node = _TTSNode(input_topic or None, self._adapter,
                                    node_suffix=node_key.replace('/', '_').replace('-', '_'))
                    self._executor.add_node(node)
                    self._nodes[node_key] = node
                elif input_topic and node._input_topic != input_topic:
                    # Input topic changed for existing instance — recreate
                    del self._nodes[node_key]
                    self._dispose_node(node, node_key)
                    node = _TTSNode(input_topic, self._adapter,
                                    node_suffix=node_key.replace('/', '_').replace('-', '_'))
                    self._executor.add_node(node)
                    self._nodes[node_key] = node
                return node.start()

        elif action == "stop":
            with self._nodes_lock:
                if instance_id:
                    node = self._nodes.pop(instance_id, None)
                    if node is None:
                        return {"state": "idle"}
                    return self._dispose_node(node, instance_id)
                for key in list(self._nodes.keys()):
                    self._dispose_node(self._nodes.pop(key), key)
                return {"state": "idle"}

        elif action == "speak":
            if self._loading:
                return {"state": "loading", "message": "TTS model is being downloaded, please wait..."}
            if self._load_error or not self._adapter:
                return {"state": "error", "message": f"TTS model not available: {self._load_error or 'not loaded'}"}
            text = args.get("text", "")
            if not text:
                raise ValueError("text is required")
            # Find any existing running node to reuse
            with self._nodes_lock:
                node = None
                for n in self._nodes.values():
                    if n.state == "running":
                        node = n
                        break
                if node is None:
                    # No running node — use instance key or fallback
                    node_key = instance_id or '_default'
                    node = self._nodes.get(node_key)
                    if node is None:
                        input_topic = args.get("input_topic") or None
                        # No per-instance adapter: `config` writes into self._cfg
                        # globally (it strips instance_id), so there has never
                        # been anywhere to read a per-instance config from. This
                        # used to consult a self._instance_configs that is never
                        # assigned anywhere, i.e. `speak` with an instance_id and
                        # no running node raised AttributeError instead of
                        # synthesizing.
                        node = _TTSNode(input_topic, self._adapter,
                                        node_suffix=node_key.replace('/', '_').replace('-', '_'))
                        self._executor.add_node(node)
                        self._nodes[node_key] = node
                    if node.state != "running":
                        node.start()
            # ACP: 生成 action_id
            import uuid as _uuid
            action_id = f"speak-{_uuid.uuid4().hex[:8]}"
            node.enqueue(text, trace_id=args.get('_trace_id', ''), action_id=action_id)
            return {"status": "queued", "action_id": action_id, "text": text}

        elif action == "config":
            cfg = {k: v for k, v in args.items() if k not in ('action', 'instance_id') and v}
            # Update config and rebuild adapter
            if 'speaker_id' in cfg:
                self._cfg['speaker_id'] = int(cfg['speaker_id'])
            if 'speed' in cfg:
                self._cfg['speed'] = float(cfg['speed'])
            if 'device' in cfg:
                self._cfg['device'] = cfg['device']
            self._adapter = _build_tts_adapter(self._cfg)
            # Stop all nodes (they'll use new adapter on next start)
            with self._nodes_lock:
                for key in list(self._nodes.keys()):
                    self._dispose_node(self._nodes.pop(key), key)
            return {"status": "configured"}

        elif action == "interrupt":
            # 立即中止所有 TTS 播放（清空队列 + 停止当前 utterance）
            total_cleared = 0
            interrupted_count = 0
            with self._nodes_lock:
                if instance_id:
                    targets = [self._nodes[instance_id]] if instance_id in self._nodes else []
                else:
                    targets = [n for n in self._nodes.values() if n.state == "running"]
            for node in targets:
                result = node.interrupt()
                total_cleared += result.get('cleared', 0)
                interrupted_count += 1
            return {"status": "interrupted", "nodes": interrupted_count, "cleared": total_cleared}

        return None

    def synthesize_raw(self, text: str) -> bytes:
        """Synthesize text and return raw PCM bytes (16kHz 16-bit mono)."""
        if not self._adapter:
            raise RuntimeError("TTS adapter not configured")
        return self._adapter.synthesize(text)


DEFAULT_TTS_ENGINE = "vits2_trt"
TTS_ENGINES = ("vits2_trt", "sherpa_onnx")
# Where each engine keeps its own model files. Used for any engine other than
# the one config.yaml was written for; see TTSPlugin._model_dir_for.
ENGINE_MODEL_DIRS = {
    "vits2_trt": "/models/vits2",
    "sherpa_onnx": "/models/sherpa-onnx/tts",
}
# How long an `action=config` engine switch waits for the new engine before
# answering `loading`. Sized so the bounded part of a build finishes inside it
# (constructing the sherpa session measured ~2 s on cpu, ~5 s on gpu) while a cold
# model download does not — that one genuinely has to go async. Also under the 30 s
# the LLM path allows a tools/call (agent-core/src/mcp_client.py), in case a config
# ever arrives from there rather than from the dashboard.
ENGINE_SWITCH_WAIT_S = 20


class TTSPlugin:
    """The single public TTS plugin, delegating to a config-selected engine.

    A facade rather than a `__new__` switch: the engine is a configSchema field,
    so it can change at runtime (`action=config`, `tts_engine=...`) and not only
    at process start. Switching disposes the previous engine's nodes and builds
    the new one on a background thread, because sherpa-onnx downloads its Matcha
    model in its constructor and that is open-ended.

    `action=config` then waits up to ENGINE_SWITCH_WAIT_S for that build and only
    answers `loading` if it is still going, so the bounded part — constructing the
    session, ~2 s on cpu and ~5 s on gpu — is not something callers have to poll
    for. It can afford to wait: the dashboard's start-project path
    (agent-core/src/api/mcp_manage.py mcp_call_tool) sets no client timeout at all,
    and its loading watcher polls for up to 900 s. The 60 s figure that used to be
    cited here belongs to the *LLM* tool path (agent-core/src/mcp_client.py), which
    does not send config.
    """

    PREFIX = "tts"

    def __init__(self, plugin_cfg: dict, executor):
        self._cfg = dict(plugin_cfg)
        self._executor = executor
        self._lock = threading.Lock()
        self._impl = None
        self._impl_engine = ""
        self._building = ""          # engine name while a build is in flight
        self._build_error = None
        # `start` calls that arrived while a build was in flight, keyed by
        # instance_id, replayed against the new engine once it is resident. Agent
        # Core's contract for a start that answers `state: loading` is that the
        # tool will reach `running` on its own — it polls `info` and reports
        # "启动已取消" if it ever sees `idle` (see agent-core api/config.py
        # _settle_loading_item). Dropping the start satisfied the letter of
        # "never block" and broke that contract: switching engine and starting in
        # the same batch, which is exactly what the dashboard does, left the card
        # cancelled even though the engine had loaded fine.
        self._pending_starts: dict[str, dict] = {}
        engine = self._select_engine(self._cfg.get("engine")
                                     or self._cfg.get("tts_engine"))
        self._engine = engine
        # model_dir is per engine: sherpa-onnx wants its Matcha/vocoder pair,
        # VITS2 wants its TensorRT release. config.yaml carries one model_dir,
        # written for the engine it also declares — handing that same path to the
        # other engine made sherpa download its models into /models/vits2 and
        # then try to load them from there. So the configured path applies only
        # to the configured engine; every other engine gets its own default.
        self._configured_engine = engine
        self._configured_model_dir = self._cfg.get("model_dir") or ""
        # Built inline at startup so info/start are immediately truthful, and so
        # a misconfigured engine shows up in the boot log rather than on the
        # first utterance. Runtime switches take the background path below.
        try:
            self._impl = self._build(engine)
            self._impl_engine = engine
        except Exception as error:  # noqa: BLE001 - surfaced via info/start
            log.error("[tts] failed to build engine %r: %s", engine, error, exc_info=True)
            self._build_error = str(error)

    # ── engine plumbing ─────────────────────────────────────────────────

    @staticmethod
    def _select_engine(value) -> str:
        engine = str(value or DEFAULT_TTS_ENGINE).strip().lower()
        if engine not in TTS_ENGINES:
            raise ValueError(f"Unsupported TTS engine: {engine}")
        return engine

    def _model_dir_for(self, engine: str) -> str:
        if engine == self._configured_engine and self._configured_model_dir:
            return self._configured_model_dir
        return ENGINE_MODEL_DIRS[engine]

    def _build(self, engine: str):
        cfg = dict(self._cfg)
        cfg["engine"] = engine
        cfg["model_dir"] = self._model_dir_for(engine)
        impl = (self._build_vits2(cfg) if engine == "vits2_trt"
                else SherpaOnnxTTSPlugin(cfg, self._executor))
        # An implementation may swallow its own model-load failure and come back
        # as an object that reports error through info (sherpa does exactly
        # that). Installing it would make the facade claim ready and let a start
        # or a speak "succeed" against a model that never loaded, so ask it.
        state = impl.dispatch("tts", {"action": "info"}) or {}
        if state.get("state") == "error":
            raise RuntimeError(
                state.get("error") or state.get("desc")
                or f"engine {engine} reported an error after construction"
            )
        return impl

    def _build_vits2(self, cfg: dict):
        from plugins.vits2_tts import Vits2TTSPlugin

        return Vits2TTSPlugin(cfg, self._executor)

    def _build_async(self, engine: str) -> None:
        """Build an engine off the request thread; the old one is already gone."""
        def _run():
            try:
                impl = self._build(engine)
            except Exception as error:  # noqa: BLE001
                log.error("[tts] failed to build engine %r: %s", engine, error,
                          exc_info=True)
                with self._lock:
                    if self._building == engine:
                        self._building = ""
                        self._build_error = str(error)
                return
            stale = None
            replay = []
            with self._lock:
                if self._building != engine:
                    stale = impl          # another switch superseded this one
                else:
                    self._impl = impl
                    self._impl_engine = engine
                    self._building = ""
                    self._build_error = None
                    replay = list(self._pending_starts.values())
                    self._pending_starts.clear()
            if stale is not None:
                _dispose_impl(stale)
                return

            # Honour the starts that arrived mid-build. Without this the engine
            # comes up idle, and Agent Core — which polls `info` after a start
            # answered `loading` — reports the card as "启动已取消".
            for start_args in replay:
                try:
                    log.info("[tts] replaying start deferred during the %s build: %s",
                             engine, start_args.get("instance_id") or
                             start_args.get("input_topic"))
                    impl.dispatch("tts", start_args)
                except Exception:
                    log.error("[tts] deferred start failed after the %s build",
                              engine, exc_info=True)

        threading.Thread(target=_run, name=f"tts-engine-{engine}", daemon=True).start()

    def get_tools(self) -> list:
        return TOOLS

    # ── dispatch ────────────────────────────────────────────────────────

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action") if name == "tts" else name
        if action == "config":
            return self._config(name, args)

        with self._lock:
            impl = self._impl
            building = self._building
            error = self._build_error
            engine = self._engine
        if impl is not None:
            result = impl.dispatch(name, args)
            if isinstance(result, dict) and action == "info":
                result.setdefault("engine", engine)
            return result

        # No engine resident: only happens while a switch is building, or after
        # a build failure. Never block the caller waiting for it.
        if action == "info":
            state = "loading" if building else "error"
            result = {
                "name": "TTS", "manufacture": "Embodied", "model": engine,
                "engine": engine, "state": state,
                "desc": (f"Switching to the {building} engine..." if building
                         else f"Engine {engine} failed to load: {error}"),
            }
            if state == "error" and error:
                result["error"] = error
            return result
        if building:
            if action == "start":
                # Defer, do not drop. The dashboard sends config (which triggers
                # the switch) and start back to back, so this is the normal path,
                # not a rare race — and _build_async replays it.
                with self._lock:
                    if self._building:
                        key = args.get("instance_id") or args.get("input_topic") or ""
                        self._pending_starts[key] = dict(args)
                        deferred = True
                    else:
                        deferred = False   # build finished while we waited on the lock
                if not deferred:
                    return self.dispatch(name, args)
            elif action == "stop":
                # Cancel a deferred start rather than letting it revive the node
                # after the operator asked for it to stop.
                with self._lock:
                    key = args.get("instance_id") or args.get("input_topic") or ""
                    if key in self._pending_starts:
                        self._pending_starts.pop(key, None)
                    elif not key:
                        self._pending_starts.clear()
                return {"state": "idle"}
            return {"state": "loading",
                    "message": f"TTS engine {building} is initializing, retry shortly"}
        return {"state": "error", "message": f"TTS engine {engine} failed: {error}"}

    def _await_build(self, engine: str, timeout_s: float) -> tuple[bool, str | None]:
        """Wait for an in-flight build. Returns (ready, error).

        `ready` false with no error means it is still going, which is the caller's
        cue to answer `loading` and let the deferred-start machinery finish the job.
        Another switch superseding this one also reads as not-ready: this build's
        result is about to be discarded, so claiming success would be wrong.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            with self._lock:
                if self._engine != engine:
                    return False, None          # superseded by a newer switch
                if self._impl is not None and self._impl_engine == engine:
                    return True, None
                if self._build_error:
                    return False, self._build_error
                if not self._building:
                    # Neither resident nor building nor failed: nothing to wait on.
                    return False, None
            if time.monotonic() >= deadline:
                return False, None
            time.sleep(0.2)

    def _config(self, name: str, args: dict) -> dict:
        """Apply shared config, switching engines when tts_engine changes."""
        requested = args.get("tts_engine") or args.get("engine")
        forwarded = {k: v for k, v in args.items()
                     if k not in ("tts_engine", "engine")}
        for key in ("speaker_id", "speed", "device"):
            if key in args:
                self._cfg[key] = args[key]

        if requested:
            engine = self._select_engine(requested)
            with self._lock:
                switching = engine != self._impl_engine or self._impl is None
                if switching:
                    outgoing, self._impl = self._impl, None
                    self._impl_engine = ""
                    self._engine = engine
                    self._cfg["engine"] = engine
                    self._building = engine
                    self._build_error = None
            if switching:
                # Stop the old engine's nodes before the new one publishes on
                # the same topics — two live TTS publishers on one topic is the
                # duplicate-audio failure mode in README § Plugin Concurrency.
                if outgoing is not None:
                    _dispose_impl(outgoing)
                self._build_async(engine)
                log.info("[tts] switching engine to %s", engine)
                # Wait for it, up to a bound. Only part of a build is open-ended
                # (downloading a model); constructing the session afterwards took
                # ~2 s on cpu and ~5 s on gpu, and answering `loading` for that is
                # what made the dashboard send a start the engine could not honour.
                # A time bound rather than "does it need to download" because the
                # download is not the only slow phase — the gpu paraformer encoder
                # is 636 MB and reading it cold takes seconds on its own.
                ready, error = self._await_build(engine, ENGINE_SWITCH_WAIT_S)
                if error:
                    return {"status": "error", "engine": engine,
                            "state": "error", "message": error}
                if ready:
                    log.info("[tts] engine %s ready", engine)
                    return {"status": "configured", "engine": engine}
                return {"status": "configured", "state": "loading",
                        "engine": engine,
                        "message": f"loading the {engine} engine"}

        with self._lock:
            impl = self._impl
            engine = self._engine
        if impl is None:
            return {"status": "configured", "state": "loading", "engine": engine}
        result = impl.dispatch(name, {**forwarded, "action": "config"})
        if isinstance(result, dict):
            result.setdefault("engine", engine)
        return result

    def synthesize_raw(self, text: str) -> bytes:
        """Synthesize text and return raw PCM bytes (16kHz 16-bit mono)."""
        with self._lock:
            impl = self._impl
            engine = self._engine
            error = self._build_error
        if impl is None:
            raise RuntimeError(f"TTS engine {engine} not ready: {error or 'loading'}")
        return impl.synthesize_raw(text)


def _dispose_impl(impl) -> None:
    """Stop every node an engine implementation owns before dropping it."""
    try:
        impl.dispatch("tts", {"action": "stop"})
    except Exception:
        log.error("[tts] failed to stop the outgoing engine", exc_info=True)
