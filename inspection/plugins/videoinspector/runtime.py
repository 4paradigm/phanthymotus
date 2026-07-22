from __future__ import annotations

import logging
import queue
import re
import threading
import time
from typing import Any, Callable

from plugins.audioinspector.runtime import source_stamp_ns
from storage.video_writer import VideoFragmentStore


log = logging.getLogger(__name__)


class VideoFramePump:
    """Bounded JPEG queue with deterministic max-fps admission."""

    def __init__(
        self,
        consumer: Callable[[bytes, int, int, int], None],
        *,
        queue_frames: int,
        max_fps: float,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._consumer = consumer
        self._queue: queue.Queue[tuple[bytes, int, int]] = queue.Queue(maxsize=max(1, int(queue_frames)))
        self._max_fps = float(max_fps)
        self._minimum_interval_ns = int(1_000_000_000 / self._max_fps)
        self._clock = monotonic_ns
        self._last_accepted_ns = 0
        self._admission_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._accepting = False
        self.received = 0
        self.accepted = 0
        self.dropped = 0
        self.rate_limited = 0
        self.invalid_format = 0
        self.invalid_payload = 0
        self.consecutive_invalid = 0
        self.last_received_ns = 0
        self.last_error = ""

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._accepting = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="video-inspector-writer")
        self._thread.start()

    def submit_message(self, message: Any) -> bool:
        if not self._accepting:
            return False
        message_format = str(getattr(message, "format", "")).lower()
        if "jpeg" not in message_format and "jpg" not in message_format:
            self.invalid_format += 1
            self.consecutive_invalid += 1
            self.last_error = f"unsupported CompressedImage format: {message_format!r}"
            return False
        jpeg = bytes(getattr(message, "data", b""))
        if len(jpeg) < 4 or not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
            self.invalid_payload += 1
            self.consecutive_invalid += 1
            self.last_error = "invalid JPEG payload: missing SOI/EOI markers"
            return False
        self.received += 1
        received_ns = self._clock()
        self.last_received_ns = received_ns
        self.consecutive_invalid = 0
        if self.last_error.startswith(("unsupported CompressedImage", "invalid JPEG payload")):
            self.last_error = ""
        with self._admission_lock:
            if self._last_accepted_ns and received_ns - self._last_accepted_ns < self._minimum_interval_ns:
                self.dropped += 1
                self.rate_limited += 1
                return False
            self._last_accepted_ns = received_ns
        try:
            self._queue.put_nowait((jpeg, source_stamp_ns(message), received_ns))
            self.accepted += 1
            return True
        except queue.Full:
            self.dropped += 1
            self.last_error = "writer queue full; video frame dropped"
            return False

    def _run(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                jpeg, stamp_ns, received_ns = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._consumer(jpeg, stamp_ns, received_ns, self.dropped)
            except Exception as exc:
                self.last_error = str(exc)
                log.exception("video frame encode failed")
            finally:
                self._queue.task_done()

    def stop(self, *, timeout: float, drain: bool = True) -> None:
        self._accepting = False
        self._stop_event.set()
        if not drain:
            discarded = 0
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                self._queue.task_done()
                discarded += 1
            if discarded:
                self.dropped += discarded
                if not self.last_error:
                    self.last_error = f"discarded {discarded} queued frames after pipeline failure"
        if self._thread:
            self._thread.join(timeout=max(0.1, timeout))
            if self._thread.is_alive():
                raise TimeoutError(f"video writer queue did not drain within {timeout:.1f}s")

    def stats(self) -> dict[str, Any]:
        return {
            "received": self.received,
            "accepted": self.accepted,
            "dropped": self.dropped,
            "rate_limited": self.rate_limited,
            "invalid_format": self.invalid_format,
            "invalid_payload": self.invalid_payload,
            "consecutive_invalid": self.consecutive_invalid,
            "last_received_ns": self.last_received_ns,
            "queue_depth": self._queue.qsize(),
            "last_error": self.last_error,
        }


class VideoRecorderRuntime:
    def __init__(
        self,
        *,
        executor: Any,
        store: VideoFragmentStore,
        instance_id: str,
        input_topic: str,
        encoder: str,
        target_bitrate_kbps: int,
        segment_seconds: int,
        max_segment_bytes: int,
        max_fps: float,
        queue_frames: int,
        shutdown_timeout_seconds: float,
        input_start_timeout_seconds: float = 10.0,
        input_stall_timeout_seconds: float = 5.0,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.executor = executor
        self.store = store
        self.instance_id = instance_id
        self.input_topic = input_topic
        self.encoder = encoder
        self.target_bitrate_kbps = int(target_bitrate_kbps)
        self.segment_seconds = int(segment_seconds)
        self.max_segment_bytes = max(1, int(max_segment_bytes))
        self.max_fps = float(max_fps)
        self.shutdown_timeout_seconds = float(shutdown_timeout_seconds)
        self.input_start_timeout_seconds = float(input_start_timeout_seconds)
        self.input_stall_timeout_seconds = float(input_stall_timeout_seconds)
        self._clock = monotonic_ns
        self._started_ns = 0
        self._node = None
        self._subscription = None
        self._pipeline = None
        self._appsrc = None
        self._splitmux = None
        self._bus_thread: threading.Thread | None = None
        self._bus_stop = threading.Event()
        self._eos_seen = threading.Event()
        self._first_receive_ns = 0
        self._last_source_stamp_ns = 0
        self._gst = None
        self.last_error = ""
        self.pump = VideoFramePump(
            self._push_frame,
            queue_frames=queue_frames,
            max_fps=max_fps,
            monotonic_ns=monotonic_ns,
        )

    def _pipeline_description(self) -> str:
        fps = max(1, round(self.max_fps))
        common_head = (
            f"appsrc name=source is-live=true block=true format=time do-timestamp=false "
            f"caps=image/jpeg,framerate={fps}/1 ! "
            f"queue max-size-buffers=4 leaky=no ! jpegparse ! "
        )
        if self.encoder == "nvv4l2h264enc":
            encode = (
                "nvjpegdec ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
                f"nvv4l2h264enc bitrate={self.target_bitrate_kbps * 1000} control-rate=1 "
                f"iframeinterval={fps} insert-sps-pps=true ! "
            )
        elif self.encoder == "libx264":
            encode = (
                "jpegdec ! videoconvert ! video/x-raw,format=I420 ! "
                f"x264enc bitrate={self.target_bitrate_kbps} speed-preset=ultrafast "
                f"tune=zerolatency key-int-max={fps} ! "
            )
        else:
            raise ValueError(f"unsupported video encoder: {self.encoder}")
        return (
            common_head + encode +
            "h264parse config-interval=-1 ! "
            f"splitmuxsink name=segments location=/tmp/inspection-unused-%05d.mp4.part "
            f"max-size-time={self.segment_seconds * 1_000_000_000} "
            f"max-size-bytes={self.max_segment_bytes} "
            "send-keyframe-requests=true async-finalize=false use-robust-muxing=true"
        )

    def start(self) -> None:
        if self.executor is None:
            raise RuntimeError("ROS2 executor is required for videoinspector runtime_mode=ros2-gstreamer")
        self._started_ns = self._clock()
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CompressedImage

        Gst.init(None)
        self._gst = Gst
        self._pipeline = Gst.parse_launch(self._pipeline_description())
        self._appsrc = self._pipeline.get_by_name("source")
        self._splitmux = self._pipeline.get_by_name("segments")
        if self._appsrc is None or self._splitmux is None:
            raise RuntimeError("failed to create GStreamer appsrc/splitmuxsink")
        self._splitmux.connect("format-location", self._format_location)
        self._splitmux.connect("muxer-added", self._configure_muxer)
        self._bus_stop.clear()
        self._eos_seen.clear()
        self._bus_thread = threading.Thread(target=self._bus_loop, daemon=True, name="video-inspector-gst-bus")
        self._bus_thread.start()
        state_result = self._pipeline.set_state(Gst.State.PLAYING)
        if state_result == Gst.StateChangeReturn.FAILURE:
            self._abort_pipeline()
            raise RuntimeError(f"GStreamer pipeline could not start for encoder {self.encoder}")

        self.pump.start()
        node_name = "videoinspector_" + re.sub(r"[^a-zA-Z0-9_]", "_", self.instance_id)

        class VideoInspectorNode(Node):
            def __init__(self) -> None:
                super().__init__(node_name)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            durability=DurabilityPolicy.VOLATILE,
        )
        try:
            self._node = VideoInspectorNode()
            self._subscription = self._node.create_subscription(
                CompressedImage,
                self.input_topic,
                self.pump.submit_message,
                qos,
            )
            self.executor.add_node(self._node)
        except Exception:
            self.pump.stop(timeout=self.shutdown_timeout_seconds)
            self._abort_pipeline()
            raise

    def _configure_muxer(self, _splitmux, muxer) -> None:
        for name, value in (
            ("reserved-moov-update-period", 1_000_000_000),
            ("reserved-max-duration", self.segment_seconds * 2 * 1_000_000_000),
        ):
            if muxer.find_property(name) is not None:
                muxer.set_property(name, value)

    def _format_location(self, _splitmux, fragment_id: int) -> str:
        return self.store.create_location(fragment_id, first_source_stamp_ns=self._last_source_stamp_ns)

    def _push_frame(self, jpeg: bytes, stamp_ns: int, received_ns: int, dropped: int) -> None:
        Gst = self._gst
        if Gst is None or self._appsrc is None:
            raise RuntimeError("GStreamer pipeline is not running")
        if self._first_receive_ns == 0:
            self._first_receive_ns = received_ns
        self._last_source_stamp_ns = stamp_ns
        buffer = Gst.Buffer.new_allocate(None, len(jpeg), None)
        buffer.fill(0, jpeg)
        buffer.pts = max(0, received_ns - self._first_receive_ns)
        buffer.dts = buffer.pts
        buffer.duration = int(1_000_000_000 / max(1.0, self.max_fps))
        result = self._appsrc.emit("push-buffer", buffer)
        if result != Gst.FlowReturn.OK:
            raise RuntimeError(f"GStreamer push-buffer failed: {result.value_nick}")
        self.store.note_frame(
            source_stamp_ns=stamp_ns,
            receive_monotonic_ns=received_ns,
            dropped=dropped,
        )

    def _bus_loop(self) -> None:
        Gst = self._gst
        if Gst is None or self._pipeline is None:
            return
        bus = self._pipeline.get_bus()
        mask = Gst.MessageType.ERROR | Gst.MessageType.EOS | Gst.MessageType.ELEMENT
        while not self._bus_stop.is_set():
            message = bus.timed_pop_filtered(100_000_000, mask)
            if message is None:
                continue
            if message.type == Gst.MessageType.ERROR:
                error, debug = message.parse_error()
                self.last_error = f"{error}: {debug or ''}".strip()
                self._eos_seen.set()
                log.error("GStreamer video pipeline error: %s", self.last_error)
            elif message.type == Gst.MessageType.EOS:
                self._eos_seen.set()
            elif message.type == Gst.MessageType.ELEMENT:
                structure = message.get_structure()
                if structure and structure.get_name() == "splitmuxsink-fragment-closed":
                    location = structure.get_string("location")
                    if location:
                        try:
                            self.store.finalize_location(location)
                        except Exception as exc:
                            self.last_error = str(exc)
                            log.exception("failed to finalize video fragment %s", location)

    def flush(self) -> dict[str, Any] | None:
        if self._splitmux is None:
            raise RuntimeError("video pipeline is not running")
        self._splitmux.emit("split-now")
        return {"split_requested": True, "last_finalized": self.store.last_finalized}

    def stop(self) -> dict[str, Any] | None:
        deadline = time.monotonic() + self.shutdown_timeout_seconds
        cleanup_errors: list[str] = []
        if self._node is not None:
            if self._subscription is not None:
                self._node.destroy_subscription(self._subscription)
                self._subscription = None
            self.executor.remove_node(self._node)
            self._node.destroy_node()
            self._node = None

        pipeline_error = self.last_error
        if pipeline_error:
            # appsrc uses block=true. Once the native decoder/encoder has failed,
            # draining before tearing down the pipeline can block the writer
            # thread forever inside push-buffer.
            self._abort_pipeline()

        try:
            self.pump.stop(
                timeout=max(0.1, deadline - time.monotonic()),
                drain=not bool(pipeline_error),
            )
        except Exception as exc:
            cleanup_errors.append(str(exc))
            self._abort_pipeline()
            try:
                self.pump.stop(timeout=2.0, drain=False)
            except Exception as retry_exc:
                cleanup_errors.append(str(retry_exc))

        if self._pipeline is not None:
            try:
                self._shutdown_pipeline(timeout=max(0.1, deadline - time.monotonic()))
            except Exception as exc:
                cleanup_errors.append(str(exc))
                self._abort_pipeline()

        if pipeline_error or cleanup_errors:
            reasons = [item for item in [pipeline_error, *cleanup_errors] if item]
            reason = "; ".join(dict.fromkeys(reasons))
            self.store.preserve_open_fragments_as_corrupt(reason)
            self.last_error = reason
            raise RuntimeError(reason)
        return self.store.last_finalized

    def _shutdown_pipeline(self, *, timeout: float) -> None:
        Gst = self._gst
        if Gst is None or self._pipeline is None:
            return
        eos_complete = True
        if self._appsrc is not None:
            self._appsrc.emit("end-of-stream")
            eos_complete = self._eos_seen.wait(timeout=max(0.1, timeout))
        self._pipeline.set_state(Gst.State.NULL)
        self._bus_stop.set()
        if self._bus_thread:
            self._bus_thread.join(timeout=2)
        self._pipeline = None
        self._appsrc = None
        self._splitmux = None
        if not eos_complete:
            raise TimeoutError("GStreamer pipeline did not finalize the current MP4 before shutdown timeout")

    def _abort_pipeline(self) -> None:
        Gst = self._gst
        self._bus_stop.set()
        if Gst is not None and self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
        if self._bus_thread:
            self._bus_thread.join(timeout=2)
        self._pipeline = None
        self._appsrc = None
        self._splitmux = None

    def stats(self) -> dict[str, Any]:
        stats = self.pump.stats()
        now_ns = self._clock()
        last_received_ns = int(stats.get("last_received_ns", 0))
        if last_received_ns:
            stats["last_frame_age_seconds"] = max(0.0, (now_ns - last_received_ns) / 1_000_000_000)
        else:
            stats["last_frame_age_seconds"] = None

        input_error = ""
        error_kind = ""
        if int(stats.get("consecutive_invalid", 0)) >= 3:
            input_error = str(stats.get("last_error") or "received invalid JPEG frames")
            error_kind = (
                "invalid_input_format" if int(stats.get("invalid_format", 0)) > 0
                else "invalid_jpeg"
            )
            stats["input_state"] = "invalid"
        elif self._started_ns and not last_received_ns:
            wait_seconds = max(0.0, (now_ns - self._started_ns) / 1_000_000_000)
            stats["input_state"] = "waiting_first_frame"
            if wait_seconds >= self.input_start_timeout_seconds:
                input_error = (
                    f"no valid JPEG frame received from {self.input_topic} "
                    f"within {self.input_start_timeout_seconds:.1f}s"
                )
                error_kind = "input_start_timeout"
                stats["input_state"] = "stalled"
        elif last_received_ns:
            stats["input_state"] = "healthy"
            if stats["last_frame_age_seconds"] >= self.input_stall_timeout_seconds:
                input_error = (
                    f"JPEG input {self.input_topic} stalled for "
                    f"{stats['last_frame_age_seconds']:.1f}s"
                )
                error_kind = "input_stalled"
                stats["input_state"] = "stalled"
        else:
            stats["input_state"] = "not_started"

        if input_error:
            stats["input_error"] = input_error
            stats["error_kind"] = error_kind
            stats["last_error"] = input_error
            stats["input_failed"] = True
        if self.last_error:
            stats["last_error"] = self.last_error
            stats["error_kind"] = "encoder_pipeline"
            stats["pipeline_failed"] = True
        return stats
