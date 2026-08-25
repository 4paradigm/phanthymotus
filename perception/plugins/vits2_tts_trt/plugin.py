"""ROS2/MCP plugin for in-process VITS2 TensorRT synthesis."""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from typing import Optional

from audio_msgs.msg import AudioChunk
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .adapter import (
    CHUNK_BYTES,
    PCM_FRAME_MS,
    SAMPLE_RATE,
    TTSAdapter,
    Vits2TensorRTAdapter,
    build_adapter,
)


log = logging.getLogger(__name__)


# End-of-utterance marker used by the public TTS audio protocol.
AUDIO_EOF_MAGIC = b"\x01\x00\xff\xff\x01\x00\xff\xff"


def _env_int(name: str, default: int, low: int, high: int) -> int:
    """Read a bounded int from the environment, falling back on bad input.

    These are tuning knobs, and this module is imported while main.py builds
    the plugin list. Raising here would take the whole perception process down
    — ASR, VOP and OCR included — over one mistyped TTS variable, so an
    unusable value is logged and the default is used instead.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        log.warning("[vits2_tts_trt] %s=%r is not an integer; using %d", name, raw, default)
        return default
    if not low <= value <= high:
        log.warning(
            "[vits2_tts_trt] %s=%d is outside [%d, %d]; using %d",
            name, value, low, high, default,
        )
        return default
    return value


FRAME_INTERVAL_MS = _env_int("MIX_VITS_FRAME_INTERVAL_MS", 70, 0, 1000)
FIRST_FRAME_DELAY_MS = _env_int("MIX_VITS_FIRST_FRAME_DELAY_MS", 0, 0, 1000)
SUBSCRIBER_WAIT_MS = _env_int("MIX_VITS_SUBSCRIBER_WAIT_MS", 5000, 0, 60000)
SUBSCRIBER_POLL_MS = _env_int("MIX_VITS_SUBSCRIBER_POLL_MS", 10, 1, 1000)
SUBSCRIBER_SETTLE_MS = _env_int("MIX_VITS_SUBSCRIBER_SETTLE_MS", 500, 0, 5000)
ALLOW_FAST_DELIVERY = os.getenv("MIX_VITS_ALLOW_FAST_DELIVERY", "1") == "1"
if FRAME_INTERVAL_MS < PCM_FRAME_MS and not ALLOW_FAST_DELIVERY:
    log.warning(
        "[vits2_tts_trt] MIX_VITS_FRAME_INTERVAL_MS=%d sends %.0fms PCM frames "
        "faster than realtime; pacing at %.0fms instead (set "
        "MIX_VITS_ALLOW_FAST_DELIVERY=1 for an offline benchmark)",
        FRAME_INTERVAL_MS, PCM_FRAME_MS, PCM_FRAME_MS,
    )
    FRAME_INTERVAL_MS = int(PCM_FRAME_MS)

_LOW_LAT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=200,
    durability=DurabilityPolicy.VOLATILE,
)

# The tool contract (actions, x-completion, x-hooks, configSchema) is owned by
# plugins/tts.py — this package is one implementation behind it. Importing it
# rather than restating it is what keeps the two engines from drifting into
# different config forms for the same tool.
from plugins.tts import TOOLS  # noqa: E402
def _error_chain(error: BaseException) -> str:
    """Flatten an exception chain into one line for info/error reporting.

    The interesting part is usually the cause, not the wrapper: "TensorRT is not
    available in this runtime" says nothing, while
    "... : libnvdla_compiler.so: file too short" points straight at a container
    started without the nvidia runtime.
    """
    parts = []
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = str(current).strip() or type(current).__name__
        if text not in parts:
            parts.append(text)
        current = current.__cause__ or current.__context__
    return ": ".join(parts[:4])


def _release_actions(items) -> None:
    """Cancel ACP actions for utterances that will never be played."""
    for text, action_id in items:
        _complete_action(action_id, text, 0, interrupted=True)


def _node_suffix(key: str) -> str:
    """Turn an instance key into a ROS-node-name-safe suffix."""
    return key.replace("/", "_").replace("-", "_")


def _complete_action(
    action_id: str, text: str, frames_sent: int, interrupted: bool
) -> None:
    """Notify Agent Core that an MCP speak action has terminated.

    Module level, not a node method: an utterance can also die before any node
    exists (load failure, stop while the model is still downloading), and the
    ACP barrier in agent-core waits for every action it registered.
    """
    if not action_id:
        return
    try:
        import ssl
        import urllib.request
        from urllib.parse import urlparse

        agent_core_url = os.getenv("AGENT_CORE_URL", "https://localhost:15678")
        if urlparse(agent_core_url).scheme != "https":
            raise ValueError("AGENT_CORE_URL must use HTTPS")
        # Agent Core serves HTTPS with a self-signed certificate, so the default
        # context rejects it and every completion POST fails — which leaves the
        # ACP barrier waiting out its full timeout on each utterance. Same
        # unverified context every other component uses for this endpoint
        # (perception/main.py, the drivers' _acp_notify).
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        payload = json.dumps(
            {
                "action_id": action_id,
                "status": "cancelled" if interrupted else "completed",
                "result": {"text": text[:100], "frames": frames_sent},
            }
        ).encode()
        request = urllib.request.Request(
            f"{agent_core_url}/api/acp/complete",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(request, timeout=3, context=ctx)
    except Exception as exc:
        log.warning("[vits2_tts_trt] ACP completion callback failed: %s", exc)


class _Vits2TTSNode(Node):
    def __init__(
        self,
        input_topic: Optional[str],
        adapter: TTSAdapter,
        node_suffix: str = "",
    ):
        super().__init__(f"vits2_trt_{node_suffix}" if node_suffix else "vits2_trt")
        self._input_topic = input_topic or ""
        self._output_topic = (
            f"{input_topic}/tts" if input_topic else "/perception/tts"
        )
        self._adapter = adapter
        self.state = "idle"
        self._text_queue = queue.Queue()
        self._worker_thread = None
        self._stop_event = threading.Event()
        self._interrupt_event = threading.Event()
        self._pub = self.create_publisher(AudioChunk, self._output_topic, _LOW_LAT_QOS)
        self._sub = (
            self.create_subscription(
                String, self._input_topic, self._text_callback, _LOW_LAT_QOS
            )
            if input_topic
            else None
        )

    def start(self):
        if self.state == "running":
            return self.status()
        self._stop_event.clear()
        self._interrupt_event.clear()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        self.state = "running"
        return self.status()

    def stop(self):
        self.request_stop()
        self._complete_discarded_actions()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3)
        self.state = "idle"
        return {"state": "idle"}

    def request_stop(self):
        """Request worker shutdown without waiting for thread termination."""
        self._stop_event.set()

    def interrupt(self) -> dict:
        """Cancel the active utterance and discard queued utterances."""
        self._interrupt_event.set()
        cleared = self._complete_discarded_actions()
        return {"status": "interrupted", "cleared": cleared}

    def _complete_discarded_actions(self) -> int:
        """Cancel queued MCP actions that will not reach the worker."""
        discarded = []
        while True:
            try:
                item = self._text_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, tuple):
                text, action_id = item
            else:
                text, action_id = str(item), ""
            discarded.append((text, action_id))
        for text, action_id in discarded:
            _complete_action(action_id, text, 0, interrupted=True)
        return len(discarded)

    def enqueue(self, text: str, action_id: str = ""):
        if self.state != "running":
            raise RuntimeError("TTS not running; call start first")
        self._text_queue.put((text, action_id))

    def _text_callback(self, message: String):
        if self.state != "running":
            return
        try:
            text = json.loads(message.data).get("text", "")
        except Exception:
            text = message.data.strip()
        if text:
            self._text_queue.put((text, ""))

    def _publish(self, pcm: bytes):
        message = AudioChunk()
        message.header.stamp = self.get_clock().now().to_msg()
        message.format = "audio/pcm-16k"
        message.data = list(pcm)
        self._pub.publish(message)

    def _publish_eof(self):
        """Publish the protocol end-of-utterance marker."""
        self._publish(AUDIO_EOF_MAGIC)

    def _utterance_cancelled(self) -> bool:
        return self._stop_event.is_set() or self._interrupt_event.is_set()

    def _wait_for_audio_subscriber(
        self, cancel_event: Optional[threading.Event] = None
    ) -> tuple[float, float, int]:
        """Wait until an audio subscriber remains DDS-matched long enough."""
        started = time.monotonic()
        deadline = started + (SUBSCRIBER_WAIT_MS + SUBSCRIBER_SETTLE_MS) / 1000.0
        matched_at = None
        while not self._utterance_cancelled() and not (
            cancel_event and cancel_event.is_set()
        ):
            now = time.monotonic()
            count = self._pub.get_subscription_count()
            if count > 0:
                if matched_at is None:
                    matched_at = now
                settled = now - matched_at
                if settled >= SUBSCRIBER_SETTLE_MS / 1000.0:
                    return matched_at - started, settled, count
            else:
                # Require a continuous stable match. A transient graph match is
                # not sufficient for a BEST_EFFORT reader to receive frame 0.
                matched_at = None
            if now >= deadline:
                raise RuntimeError(
                    "no stable matched TTS audio subscriber within "
                    f"{SUBSCRIBER_WAIT_MS + SUBSCRIBER_SETTLE_MS}ms "
                    f"on {self._output_topic}"
                )
            time.sleep(SUBSCRIBER_POLL_MS / 1000.0)
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("subscriber wait cancelled")
        if self._interrupt_event.is_set():
            raise RuntimeError("TTS interrupted while waiting for an audio subscriber")
        raise RuntimeError("TTS stopped while waiting for an audio subscriber")

    def _worker(self):
        frame_interval = FRAME_INTERVAL_MS / 1000.0
        while not self._stop_event.is_set():
            try:
                item = self._text_queue.get(timeout=1)
            except queue.Empty:
                continue
            if isinstance(item, tuple):
                text, action_id = item
            else:
                text, action_id = str(item), ""
            if self._interrupt_event.is_set():
                self._interrupt_event.clear()
                self._publish_eof()
                _complete_action(action_id, text, 0, interrupted=True)
                continue
            subscriber_gate_cancel = threading.Event()
            subscriber_gate_done = threading.Event()
            subscriber_gate_result = {}

            def wait_for_subscriber() -> None:
                try:
                    subscriber_gate_result["value"] = (
                        self._wait_for_audio_subscriber(subscriber_gate_cancel)
                    )
                except BaseException as exc:
                    subscriber_gate_result["error"] = exc
                finally:
                    subscriber_gate_done.set()

            subscriber_gate_thread = threading.Thread(
                target=wait_for_subscriber,
                name="vits2-trt-subscriber-gate",
                daemon=True,
            )
            # DDS discovery/settling runs in parallel with frontend + first
            # TensorRT synthesis so the stronger BEST_EFFORT guard does not
            # become pure TTFT overhead.
            subscriber_gate_thread.start()
            eof_published = False
            try:
                task_started = time.monotonic()
                first_published_at = None
                total_bytes = 0
                started = None
                frames_sent = 0
                buffer = bytearray()
                subscriber_wait_seconds = None
                subscriber_settle_seconds = None
                subscriber_count = 0

                def publish_frame(frame: bytes) -> bool:
                    nonlocal started, frames_sent, first_published_at, total_bytes
                    nonlocal subscriber_wait_seconds, subscriber_settle_seconds
                    nonlocal subscriber_count
                    if self._utterance_cancelled():
                        return False
                    now = time.monotonic()
                    if started is None:
                        while not subscriber_gate_done.wait(timeout=0.05):
                            if self._utterance_cancelled():
                                return False
                        if "error" in subscriber_gate_result:
                            if self._utterance_cancelled():
                                return False
                            raise subscriber_gate_result["error"]
                        (
                            subscriber_wait_seconds,
                            subscriber_settle_seconds,
                            subscriber_count,
                        ) = subscriber_gate_result["value"]
                        if FIRST_FRAME_DELAY_MS:
                            time.sleep(FIRST_FRAME_DELAY_MS / 1000.0)
                        started = time.monotonic()
                        now = started
                    if frame_interval:
                        target = started + frames_sent * frame_interval
                        if target < now - frame_interval:
                            started = now - frames_sent * frame_interval
                            target = now
                        delay = target - now
                        if delay > 0:
                            time.sleep(delay)
                    if self._utterance_cancelled():
                        return False
                    self._publish(frame)
                    if first_published_at is None:
                        first_published_at = time.monotonic()
                    total_bytes += len(frame)
                    frames_sent += 1
                    return True

                interrupted = False
                for pcm in self._adapter.synthesize_stream(text):
                    if self._utterance_cancelled():
                        interrupted = True
                        break
                    buffer.extend(pcm)
                    while len(buffer) >= CHUNK_BYTES:
                        frame = bytes(buffer[:CHUNK_BYTES])
                        del buffer[:CHUNK_BYTES]
                        if not publish_frame(frame):
                            interrupted = True
                            break
                    if interrupted:
                        break

                if buffer and not self._utterance_cancelled():
                    if not publish_frame(bytes(buffer)):
                        interrupted = True
                interrupted = interrupted or self._interrupt_event.is_set()
                if total_bytes:
                    finished_at = time.monotonic()
                    audio_seconds = total_bytes / (SAMPLE_RATE * 2)
                    elapsed = finished_at - task_started
                    log.info(
                        "[vits2_tts_trt] server delivery: bytes=%d frames=%d "
                        "ttft=%.3fs elapsed=%.3fs audio=%.3fs rtf=%.4f "
                        "chunk_bytes=%d frame_interval_ms=%d "
                        "first_frame_delay_ms=%d subscriber_wait_ms=%.1f "
                        "subscriber_settle_ms=%.1f subscriber_count=%d",
                        total_bytes,
                        frames_sent,
                        first_published_at - task_started,
                        elapsed,
                        audio_seconds,
                        elapsed / audio_seconds,
                        CHUNK_BYTES,
                        FRAME_INTERVAL_MS,
                        FIRST_FRAME_DELAY_MS,
                        (subscriber_wait_seconds or 0.0) * 1000.0,
                        (subscriber_settle_seconds or 0.0) * 1000.0,
                        subscriber_count,
                    )
                self._publish_eof()
                eof_published = True
                _complete_action(action_id, text, frames_sent, interrupted)
                if interrupted:
                    log.info(
                        "[vits2_tts_trt] utterance interrupted after %d frames",
                        frames_sent,
                    )
                self._interrupt_event.clear()
            except Exception:
                log.exception("[vits2_tts_trt] synthesis failed")
                _complete_action(action_id, text, 0, interrupted=True)
            finally:
                if not eof_published:
                    self._publish_eof()
                subscriber_gate_cancel.set()
                subscriber_gate_thread.join(timeout=0.1)

    def status(self):
        return {
            "state": self.state,
            "topic_in": [
                {"topic": self._input_topic, "format": "data/json", "desc": ""}
            ],
            "topic_out": [
                {"topic": self._output_topic, "format": "audio/pcm-16k", "desc": ""}
            ],
        }

class TTSPlugin:
    """VITS2 TensorRT implementation behind the standard ``tts`` tool.

    Lifecycle mirrors plugins/ocr.py, for the same reason: the slow work (a
    60 MB release download, three TensorRT engines, a warmup pass) is open-ended,
    so it must not run on the request thread. The bound it would blow through is
    the 60 s the *LLM* tool path allows a processor tools/call
    (agent-core/src/mcp_client.py); the dashboard's start-project path sets no
    client timeout, but a download still has no upper bound worth blocking on.

        idle --start--> loading --ok--> ready/running
                          |               ^
                          +----fail--> error (next start retries)

    * first start records the instance as pending, spawns the single loader and
      immediately returns {"state": "loading"};
    * concurrent starts only add pending instances — one download, N instances;
    * info never blocks and never triggers a load;
    * stop during loading cancels the pending instance;
    * config bumps a generation token so a stale loader cannot install an
      adapter built from an outdated configuration.
    """

    PREFIX = "tts"
    NAME = "VITS2 TTS"
    DESC = "VITS2 TensorRT text-to-speech"

    def __init__(self, plugin_cfg: dict, executor):
        self._cfg = dict(plugin_cfg)
        self._executor = executor

        # Everything below is guarded by _state_lock, which is only ever held
        # for dict/flag updates — never across a download, an engine build or a
        # node start/stop. See perception/README.md § Plugin Concurrency.
        self._state_lock = threading.RLock()
        self._nodes = {}
        self._pending_starts = {}        # node_key -> input_topic
        self._pending_speech = {}        # node_key -> [(text, action_id), ...]
        self._adapter = None
        self._adapter_state = "idle"     # idle|loading|ready|error
        self._load_error = None
        self._load_generation = 0
        self._model_name = "vits2"

        backend = str(self._cfg.get("backend", "trt")).lower()
        if backend != "trt":
            raise ValueError("The VITS2 TTS plugin supports backend=trt only")
        if int(self._cfg.get("speaker_id", 0)) != 0:
            raise ValueError("The VITS2 model supports only speaker_id=0")
        log.info(
            "[vits2_tts_trt] plugin init: model_dir=%s, speed=%s, warmup=%s",
            self._cfg.get("model_dir", "/models/vits2"),
            self._cfg.get("speed", 1.0),
            self._cfg.get("warmup", True),
        )

    def get_tools(self):
        return TOOLS

    # ── background loader (single-flight) ────────────────────────────────

    def _spawn_loader_locked(self) -> None:
        """Start the one background adapter loader. Caller holds the lock."""
        self._adapter_state = "loading"
        self._load_error = None
        threading.Thread(
            target=self._loader, args=(self._load_generation, dict(self._cfg)),
            name="vits2-adapter-loader", daemon=True,
        ).start()

    def _loader(self, generation: int, cfg: dict) -> None:
        try:
            from utils.model_downloader import ensure_vits2_model

            # Returns the engine directory of the family that matches the
            # TensorRT actually importable here, so the adapter never has to
            # guess between engines/jp61 and engines/jp511.
            cfg["engine_dir"] = ensure_vits2_model(
                cfg.get("model_dir", "/models/vits2")
            )
            adapter = build_adapter(cfg)
            if not isinstance(adapter, Vits2TensorRTAdapter):
                raise RuntimeError("Unexpected non-TensorRT VITS2 adapter")
            if cfg.get("warmup", True):
                started = time.monotonic()
                warmup_bytes = adapter.warmup()
                log.info(
                    "[vits2_tts_trt] engine ready: bytes=%d elapsed=%.3fs",
                    warmup_bytes, time.monotonic() - started,
                )
        except Exception as error:  # noqa: BLE001 - surfaced via state/info
            log.exception("[vits2_tts_trt] adapter load failed")
            with self._state_lock:
                if generation != self._load_generation:
                    return
                self._adapter_state = "error"
                self._load_error = _error_chain(error)
                # Abandon the queued work rather than leave the caller's ACP
                # action pending forever; the reason is on the tool state.
                self._pending_starts.clear()
                abandoned = self._take_pending_speech_locked(list(self._pending_speech))
            _release_actions(abandoned)
            return

        with self._state_lock:
            superseded = generation != self._load_generation
            if not superseded:
                self._adapter = adapter
                self._adapter_state = "ready"
                self._load_error = None
                self._model_name = "vits2-tensorrt"
        if superseded:
            # A config change happened mid-load; never install the result.
            return

        self._bring_up_pending(generation, adapter)

    def _bring_up_pending(self, generation: int, adapter) -> None:
        """Start every instance whose start arrived while the model loaded."""
        while True:
            with self._state_lock:
                if generation != self._load_generation or not self._pending_starts:
                    return
                node_key, input_topic = next(iter(self._pending_starts.items()))
            try:
                node = _Vits2TTSNode(
                    input_topic or None, adapter, _node_suffix(node_key)
                )
            except Exception:  # noqa: BLE001 - keep serving the other instances
                log.exception("[vits2_tts_trt] failed to build instance %r", node_key)
                with self._state_lock:
                    self._pending_starts.pop(node_key, None)
                    abandoned = self._take_pending_speech_locked([node_key])
                _release_actions(abandoned)
                continue
            # Register before starting (README lifecycle rule): a started but
            # unregistered node is unreachable, and its publisher would keep the
            # ROS node name until the process exits.
            committed = False
            with self._state_lock:
                still_wanted = (
                    generation == self._load_generation
                    and self._pending_starts.get(node_key) == input_topic
                )
                if still_wanted:
                    try:
                        self._executor.add_node(node)
                    except Exception:  # noqa: BLE001
                        log.exception(
                            "[vits2_tts_trt] failed to register instance %r", node_key
                        )
                    else:
                        self._nodes[node_key] = node
                        committed = True
                self._pending_starts.pop(node_key, None)
            if not committed:
                self._dispose_node(node, node_key)
                with self._state_lock:
                    abandoned = self._take_pending_speech_locked([node_key])
                _release_actions(abandoned)
                continue
            try:
                node.start()
            except Exception:  # noqa: BLE001
                log.exception("[vits2_tts_trt] failed to start instance %r", node_key)
                with self._state_lock:
                    self._nodes.pop(node_key, None)
                    abandoned = self._take_pending_speech_locked([node_key])
                self._dispose_node(node, node_key)
                _release_actions(abandoned)
                continue
            self._flush_pending_speech(node_key, node)

    def _flush_pending_speech(self, node_key: str, node) -> None:
        """Hand a freshly started node the utterances queued while it loaded."""
        with self._state_lock:
            queued = self._pending_speech.pop(node_key, [])
        for text, action_id in queued:
            try:
                node.enqueue(text, action_id=action_id)
            except Exception:  # noqa: BLE001
                log.exception("[vits2_tts_trt] failed to queue deferred utterance")
                _complete_action(action_id, text, 0, interrupted=True)

    def _take_pending_speech_locked(self, node_keys: list) -> list:
        """Detach queued utterances so the caller can release them off-lock.

        The completion callback is an HTTPS POST with a 3 s timeout; running it
        under _state_lock would stall info/start for every instance.
        """
        taken = []
        for node_key in node_keys:
            taken.extend(self._pending_speech.pop(node_key, []))
        return taken

    # ── helpers ─────────────────────────────────────────────────────────

    def _dispose_node(self, node, key):
        """Release a node after its caller has removed it from _nodes."""
        try:
            node.request_stop()
            node.stop()
        except Exception:
            log.exception("[vits2_tts_trt] node stop failed: %s", key)
        try:
            self._executor.remove_node(node)
        except Exception:
            log.exception("[vits2_tts_trt] node removal failed: %s", key)
        try:
            node.destroy_node()
        except Exception:
            log.exception("[vits2_tts_trt] node destroy failed: %s", key)

    def _identity(self, **extra) -> dict:
        base = {
            "name": self.NAME,
            "manufacture": "Embodied",
            "model": self._model_name,
            "engine": "vits2_trt",
            "desc": self.DESC,
        }
        base.update(extra)
        return base

    def _describe(self, state: str) -> str:
        if state == "loading":
            return "Downloading and initializing the VITS2 TensorRT model..."
        if state == "error" and self._load_error:
            return f"Model load failed: {self._load_error}"
        return self.DESC

    def _request_load_locked(self) -> None:
        """Ensure a loader is running (or has already produced an adapter)."""
        if self._adapter_state in ("loading", "ready"):
            return
        self._spawn_loader_locked()

    # ── dispatch ────────────────────────────────────────────────────────

    def dispatch(self, name: str, args: dict):
        action = args.get("action") if name == "tts" else name
        instance_id = args.get("instance_id", "")

        if action == "info":
            return self._info(args, instance_id)

        if action == "start":
            return self._start(args, instance_id)

        if action == "stop":
            return self._stop(instance_id)

        if action == "speak":
            return self._speak(args, instance_id)

        if action == "config":
            return self._config(args)

        if action == "interrupt":
            return self._interrupt(instance_id)

        return None

    def _info(self, args: dict, instance_id: str) -> dict:
        input_topic = args.get("input_topic", "")
        with self._state_lock:
            node = self._nodes.get(instance_id) if instance_id else None
            nodes = list(self._nodes.values())
            adapter_state = self._adapter_state
            pending = bool(self._pending_starts)
            error = self._load_error
            # An instance that started while the model loads has no node yet, and
            # info is routinely called without input_topic (Agent Core's start
            # sequencer does). Remembering the topic the pending start asked for
            # is what keeps the reported output topic real instead of the
            # /perception/tts fallback — and that reported topic is what the
            # dashboard subscribes to for its waveform. Same reason ocr.py keeps
            # its _pending_starts lookup.
            if instance_id and node is None and not input_topic:
                input_topic = self._pending_starts.get(instance_id, "")

        if instance_id and node is not None:
            state = node.state
            if adapter_state == "loading" and state != "running":
                state = "loading"
            return self._identity(**{**node.status(), "state": state,
                                     "desc": self._describe(state)})

        if instance_id:
            # Known instance that is not up yet: report the shared model state
            # so the dashboard shows "loading" instead of a misleading idle.
            state = ("loading" if adapter_state == "loading"
                     else "error" if adapter_state == "error" else "idle")
            output_topic = f"{input_topic}/tts" if input_topic else "/perception/tts"
            result = self._identity(
                state=state,
                topic_in=([{"topic": input_topic, "format": "data/json", "desc": ""}]
                          if input_topic else []),
                topic_out=[{"topic": output_topic, "format": "audio/pcm-16k",
                            "desc": ""}],
                desc=self._describe(state),
            )
            if state == "error" and error:
                result["error"] = error
            return result

        if any(n.state == "running" for n in nodes):
            state = "running"
        elif adapter_state == "loading" or pending:
            state = "loading"
        elif adapter_state == "error":
            state = "error"
        else:
            state = "idle"
        topics_in = [{"topic": n._input_topic, "format": "data/json", "desc": ""}
                     for n in nodes]
        topics_out = [{"topic": n._output_topic, "format": "audio/pcm-16k", "desc": ""}
                      for n in nodes]
        if not topics_out:
            output_topic = f"{input_topic}/tts" if input_topic else "/perception/tts"
            topics_in = ([{"topic": input_topic, "format": "data/json", "desc": ""}]
                         if input_topic else [])
            topics_out = [{"topic": output_topic, "format": "audio/pcm-16k",
                           "desc": ""}]
        result = self._identity(state=state, topic_in=topics_in,
                                topic_out=topics_out, desc=self._describe(state))
        if state == "error" and error:
            result["error"] = error
        return result

    def _start(self, args: dict, instance_id: str) -> dict:
        input_topic = args.get("input_topic") or ""
        key = instance_id or input_topic or "_default"
        with self._state_lock:
            node = self._nodes.get(key)
            stale = None
            if node is not None and input_topic and node._input_topic != input_topic:
                stale = self._nodes.pop(key)
                node = None
            if node is not None:
                if self._adapter is not None:
                    return node.start()
                # Adapter was discarded by a config change; rebuild from scratch.
                stale = self._nodes.pop(key)
                node = None
            self._pending_starts[key] = input_topic
            adapter = self._adapter
            state = self._adapter_state
            error = self._load_error
            if adapter is None:
                self._request_load_locked()
        if stale is not None:
            self._dispose_node(stale, key)

        if adapter is None:
            if state == "error" and error and self._adapter_state == "error":
                with self._state_lock:
                    self._pending_starts.pop(key, None)
                return {"state": "error", "message": error}
            return {"state": "loading",
                    "message": "VITS2 model is initializing; it will start "
                               "automatically when ready"}

        # Model already resident: bring this instance up inline, which is fast.
        self._bring_up_pending(self._load_generation, adapter)
        with self._state_lock:
            node = self._nodes.get(key)
            error = self._load_error
        if node is None:
            return {"state": "error",
                    "message": error or "failed to start VITS2 instance"}
        return node.status()

    def _stop(self, instance_id: str) -> dict:
        with self._state_lock:
            keys = [instance_id] if instance_id else list(self._nodes)
            nodes = [(key, self._nodes.pop(key)) for key in keys if key in self._nodes]
            cancel = [instance_id] if instance_id else list(self._pending_starts)
            for key in cancel:
                self._pending_starts.pop(key, None)
            abandoned = self._take_pending_speech_locked(
                [instance_id] if instance_id else list(self._pending_speech)
            )
        for key, node in nodes:
            self._dispose_node(node, key)
        _release_actions(abandoned)
        return {"state": "idle"}

    def _speak(self, args: dict, instance_id: str) -> dict:
        text = args.get("text", "").strip()
        if not text:
            raise ValueError("text is required")
        import uuid

        action_id = f"speak-{uuid.uuid4().hex[:8]}"
        with self._state_lock:
            node = next((n for n in self._nodes.values() if n.state == "running"), None)
            if node is None:
                # Queue against the instance that will serve it and make sure a
                # load is in flight. The ACP action stays pending until the
                # utterance actually plays (or the load fails), which is what
                # the barrier in agent-core expects.
                key = instance_id or "_default"
                if self._adapter_state == "error" and self._load_error:
                    return {"state": "error", "message": self._load_error}
                self._pending_starts.setdefault(key, args.get("input_topic") or "")
                self._pending_speech.setdefault(key, []).append((text, action_id))
                self._request_load_locked()
                adapter = self._adapter
        if node is not None:
            node.enqueue(text, action_id=action_id)
            return {"status": "queued", "action_id": action_id, "text": text}
        if adapter is not None:
            # Model is resident (a stopped instance, say) — start it now.
            self._bring_up_pending(self._load_generation, adapter)
        return {"status": "queued", "action_id": action_id, "text": text,
                "state": "loading"}

    def _config(self, args: dict) -> dict:
        if "speaker_id" in args:
            self._cfg["speaker_id"] = int(args["speaker_id"])
        if "speed" in args:
            self._cfg["speed"] = float(args["speed"])
        if int(self._cfg.get("speaker_id", 0)) != 0:
            raise ValueError("The VITS2 model supports only speaker_id=0")
        speed = float(self._cfg.get("speed", 1.0))
        with self._state_lock:
            # Speed is a scale on the loaded engine, so an in-place update
            # avoids tearing down a resident model for a slider change.
            adapter = self._adapter
            if adapter is None:
                # No model yet: invalidate the loader in flight so it cannot
                # install an adapter built from the previous configuration, then
                # start a fresh one if anybody is still waiting to come up. The
                # canvas does config→start within seconds, so dropping the
                # pending start here would leave the instance stuck at idle.
                self._load_generation += 1
                self._adapter_state = "idle"
                self._load_error = None
                if self._pending_starts or self._pending_speech:
                    self._spawn_loader_locked()
        if adapter is not None:
            adapter.set_speed(speed)
        return {"status": "configured"}

    def _interrupt(self, instance_id: str) -> dict:
        with self._state_lock:
            if instance_id:
                targets = ([self._nodes[instance_id]]
                           if instance_id in self._nodes else [])
                queues = [instance_id]
            else:
                targets = [n for n in self._nodes.values() if n.state == "running"]
                queues = list(self._pending_speech)
            abandoned = self._take_pending_speech_locked(queues)
        cleared = len(abandoned)
        _release_actions(abandoned)
        for node in targets:
            cleared += node.interrupt().get("cleared", 0)
        return {"status": "interrupted", "nodes": len(targets), "cleared": cleared}

    def synthesize_raw(self, text: str) -> bytes:
        """Synthesize to PCM for the HTTP test endpoint (blocking by design)."""
        with self._state_lock:
            adapter = self._adapter
            if adapter is None:
                self._request_load_locked()
                state = self._adapter_state
                error = self._load_error
        if adapter is None:
            raise RuntimeError(
                f"VITS2 model not ready ({state}): {error or 'still loading'}"
            )
        return adapter.synthesize(text)
