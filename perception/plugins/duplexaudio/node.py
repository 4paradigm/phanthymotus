from __future__ import annotations

import logging
import re
import time
from typing import Optional

from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from .aec import AECProcessor


log = logging.getLogger(__name__)

_AUDIO_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
    durability=DurabilityPolicy.VOLATILE,
)
_MIN_CLEAN_CHUNK_BYTES = 1024


class DuplexAudioNode(Node):
    """External-mic bridge: AEC then stable clean-PCM re-chunking."""

    def __init__(
        self,
        input_topic: str,
        clean_topic: str,
        instance_id: str,
        aec: Optional[AECProcessor],
        failure_policy: str = "fail_closed",
    ):
        safe_id = re.sub(r"[^a-zA-Z0-9_]", "_", instance_id)
        super().__init__(f"duplexaudio_bridge_{safe_id}")
        self.input_topic = input_topic
        self.clean_topic = clean_topic
        self.state = "running"
        self.last_error: Optional[str] = None
        self._aec = aec
        self._failure_policy = failure_policy
        self._emit_buffer = bytearray()
        from audio_msgs.msg import AudioChunk

        self._publisher = self.create_publisher(
            AudioChunk, clean_topic, _AUDIO_QOS
        )
        self._subscription = self.create_subscription(
            AudioChunk, input_topic, self._on_audio, _AUDIO_QOS
        )
        log.info(
            "[duplexaudio] bridge started: %s -> %s (aec=%s)",
            input_topic,
            clean_topic,
            aec.backend_name if aec is not None else "disabled",
        )

    def _on_audio(self, msg) -> None:
        if self.state != "running":
            return
        if msg.format != "audio/pcm-16k":
            self.last_error = (
                f"unsupported audio format {msg.format!r}; expected audio/pcm-16k"
            )
            self.state = "error"
            log.error("[duplexaudio] %s", self.last_error)
            return

        pcm = bytes(msg.data)
        if len(pcm) % 2:
            self.last_error = "microphone PCM byte length must be even"
            self.state = "error"
            log.error("[duplexaudio] %s", self.last_error)
            return
        capture_ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if capture_ts < 1e9:
            capture_ts = time.time()

        clean = pcm
        if self._aec is not None:
            try:
                clean = self._aec.process_pcm(pcm, capture_ts)
            except Exception as exc:
                self.last_error = f"AEC processing failed: {exc}"
                log.exception("[duplexaudio] %s", self.last_error)
                if self._failure_policy == "passthrough":
                    clean = pcm
                else:
                    self.state = "error"
                    return

        self._emit_buffer.extend(clean)
        from audio_msgs.msg import AudioChunk

        while len(self._emit_buffer) >= _MIN_CLEAN_CHUNK_BYTES:
            chunk = bytes(self._emit_buffer[:_MIN_CLEAN_CHUNK_BYTES])
            del self._emit_buffer[:_MIN_CLEAN_CHUNK_BYTES]
            output = AudioChunk()
            output.header = msg.header
            if output.header.stamp.sec == 0:
                output.header.stamp = self.get_clock().now().to_msg()
            output.format = "audio/pcm-16k"
            output.data = list(chunk)
            self._publisher.publish(output)

    def stop(self) -> dict:
        if self._subscription is not None:
            try:
                self.destroy_subscription(self._subscription)
            except Exception:
                pass
            self._subscription = None
        self._emit_buffer.clear()
        self.state = "idle"
        return {"state": "idle"}

    def stats(self) -> dict:
        result = {
            "enabled": self._aec is not None,
            "failure_policy": self._failure_policy,
            "bridge_state": self.state,
            "last_error": self.last_error,
            "pending_output_bytes": len(self._emit_buffer),
        }
        if self._aec is not None:
            result.update(self._aec.stats())
        return result
