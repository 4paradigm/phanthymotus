from __future__ import annotations

import logging
import queue
import re
import threading
from typing import Any

from storage.atomic_writer import AudioSegmentWriter


log = logging.getLogger(__name__)


def source_stamp_ns(message: Any) -> int:
    try:
        stamp = message.header.stamp
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError):
        return 0


class AudioWritePump:
    """Bounded handoff between the ROS callback and the durable WAV writer."""

    def __init__(self, writer: AudioSegmentWriter, *, queue_chunks: int) -> None:
        self.writer = writer
        self._queue: queue.Queue[tuple[bytes, int]] = queue.Queue(maxsize=max(1, int(queue_chunks)))
        self._writer_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._accepting = False
        self.received = 0
        self.dropped = 0
        self.invalid_format = 0
        self.last_error = ""

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._accepting = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="audio-inspector-writer")
        self._thread.start()

    def submit_message(self, message: Any) -> bool:
        if not self._accepting:
            return False
        if str(getattr(message, "format", "")) != "audio/pcm-16k":
            self.invalid_format += 1
            self.last_error = f"unsupported AudioChunk format: {getattr(message, 'format', '')!r}"
            return False
        pcm = bytes(getattr(message, "data", b""))
        if not pcm:
            return False
        self.received += 1
        try:
            self._queue.put_nowait((pcm, source_stamp_ns(message)))
            return True
        except queue.Full:
            self.dropped += 1
            self.last_error = "writer queue full; audio chunk dropped"
            return False

    def _run(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                pcm, stamp_ns = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                with self._writer_lock:
                    self.writer.write_chunk(
                        pcm,
                        source_stamp_ns=stamp_ns,
                        dropped_before_writer=self.dropped,
                    )
            except Exception as exc:
                self.last_error = str(exc)
                log.exception("audio segment write failed")
            finally:
                self._queue.task_done()

    def flush(self) -> dict[str, Any] | None:
        with self._writer_lock:
            return self.writer.finalize()

    def stop(self, *, timeout: float) -> dict[str, Any] | None:
        self._accepting = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=max(0.1, timeout))
            if self._thread.is_alive():
                raise TimeoutError(f"audio writer queue did not drain within {timeout:.1f}s")
        with self._writer_lock:
            return self.writer.finalize()

    def stats(self) -> dict[str, Any]:
        return {
            "received": self.received,
            "dropped": self.dropped,
            "invalid_format": self.invalid_format,
            "queue_depth": self._queue.qsize(),
            "last_error": self.last_error,
        }


class AudioRecorderRuntime:
    def __init__(
        self,
        *,
        executor: Any,
        writer: AudioSegmentWriter,
        instance_id: str,
        input_topic: str,
        queue_seconds: float,
        shutdown_timeout_seconds: float,
    ) -> None:
        self.executor = executor
        self.instance_id = instance_id
        self.input_topic = input_topic
        self.shutdown_timeout_seconds = float(shutdown_timeout_seconds)
        # The verified G1 microphone publishes about 31 chunks/s. Keep a small
        # margin while preserving a hard upper bound on memory.
        self.pump = AudioWritePump(writer, queue_chunks=max(1, round(float(queue_seconds) * 40)))
        self._node = None
        self._subscription = None

    def start(self) -> None:
        if self.executor is None:
            raise RuntimeError("ROS2 executor is required for audioinspector runtime_mode=ros2")
        from audio_msgs.msg import AudioChunk
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

        pump = self.pump
        node_name = "audioinspector_" + re.sub(r"[^a-zA-Z0-9_]", "_", self.instance_id)

        class AudioInspectorNode(Node):
            def __init__(self) -> None:
                super().__init__(node_name)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pump.start()
        try:
            self._node = AudioInspectorNode()
            self._subscription = self._node.create_subscription(
                AudioChunk,
                self.input_topic,
                pump.submit_message,
                qos,
            )
            self.executor.add_node(self._node)
        except Exception:
            self.pump.stop(timeout=self.shutdown_timeout_seconds)
            if self._node is not None:
                self._node.destroy_node()
                self._node = None
            raise

    def flush(self) -> dict[str, Any] | None:
        return self.pump.flush()

    def stop(self) -> dict[str, Any] | None:
        if self._node is not None:
            if self._subscription is not None:
                self._node.destroy_subscription(self._subscription)
                self._subscription = None
            self.executor.remove_node(self._node)
            self._node.destroy_node()
            self._node = None
        return self.pump.stop(timeout=self.shutdown_timeout_seconds)

    def stats(self) -> dict[str, Any]:
        return self.pump.stats()
