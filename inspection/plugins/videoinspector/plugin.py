from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

from plugins.base import InspectorPlugin, RecordingInstance
from storage.ledger import SegmentLedger
from storage.video_writer import VideoFragmentStore, reconcile_video_store


log = logging.getLogger(__name__)


class VideoInspectorPlugin(InspectorPlugin):
    def __init__(self, plugin_config: dict[str, Any] | None = None, executor: Any = None) -> None:
        plugin_config = plugin_config or {}
        runtime_mode = str(plugin_config.get("runtime_mode", "gate1-contract-only"))
        if runtime_mode not in {"gate1-contract-only", "ros2-gstreamer"}:
            raise ValueError(f"unsupported videoinspector runtime_mode: {runtime_mode}")
        super().__init__(
            card_id="videoinspector",
            display_name="Video Inspector",
            input_format="image/jpeg",
            input_description="sensor_msgs/CompressedImage JPEG frames",
            instance_properties={
                "segment_seconds": {"type": "integer", "minimum": 5, "maximum": 600, "default": 60, "scope": "instance"},
                "local_retention_hours": {"type": "number", "minimum": 1, "maximum": 168, "default": 6, "scope": "instance"},
                "local_max_gb": {"type": "number", "minimum": 0.1, "default": 20, "scope": "instance"},
                "encoder": {"type": "string", "enum": ["nvv4l2h264enc", "libx264"], "default": "nvv4l2h264enc", "scope": "instance"},
                "target_bitrate_kbps": {"type": "integer", "minimum": 256, "maximum": 20000, "default": 4000, "scope": "instance"},
                "max_fps": {"type": "number", "minimum": 1, "default": 30, "scope": "instance"},
                "queue_frames": {"type": "integer", "minimum": 1, "default": 8, "scope": "instance"},
                "auto_resume_after_reboot": {"type": "boolean", "default": False, "scope": "instance"},
            },
            runtime_mode=runtime_mode,
            storage_ready=runtime_mode == "ros2-gstreamer",
        )
        self._executor = executor
        self._data_root = Path(plugin_config.get("data_root", "/opt/phanthy-motus/inspection-data"))
        self._state_root = Path(plugin_config.get("state_root", "/opt/phanthy-motus/inspection-state"))
        self._runtimes: dict[str, Any] = {}
        self._ledger: SegmentLedger | None = None
        self._recovery_stats: dict[str, int] = {}
        self._closed = False
        if runtime_mode == "ros2-gstreamer":
            if executor is None:
                raise RuntimeError("ROS2 executor is required for videoinspector runtime_mode=ros2-gstreamer")
            self._ledger = SegmentLedger(self._state_root / "ledger.sqlite3")
            self._ledger.reset_uploading_for_recovery()
            self._recovery_stats = reconcile_video_store(self._data_root, self._ledger)
            log.info("video store recovery complete: %s", self._recovery_stats)
            self._restore_desired_instances()

    def _start_runtime(self, instance: RecordingInstance, config: dict[str, Any]) -> None:
        if self._ledger is None:
            return super()._start_runtime(instance, config)
        from plugins.videoinspector.runtime import VideoRecorderRuntime

        encoder = str(config.get("encoder", "nvv4l2h264enc"))
        bitrate = int(config.get("target_bitrate_kbps", 4000))
        store = VideoFragmentStore(
            data_root=self._data_root,
            ledger=self._ledger,
            card_id=self.card_id,
            instance_id=instance.instance_id,
            input_topic=instance.input_topic,
            session_id=instance.session_id,
            device_id=str(config.get("device_id", "unknown")),
            encoder=encoder,
            target_bitrate_kbps=bitrate,
        )
        runtime = VideoRecorderRuntime(
            executor=self._executor,
            store=store,
            instance_id=instance.instance_id,
            input_topic=instance.input_topic,
            encoder=encoder,
            target_bitrate_kbps=bitrate,
            segment_seconds=int(config.get("segment_seconds", 60)),
            max_fps=float(config.get("max_fps", 30)),
            queue_frames=int(config.get("queue_frames", 8)),
            shutdown_timeout_seconds=float(config.get("shutdown_finalize_timeout_seconds", 15)),
        )
        runtime.start()
        try:
            self._ledger.set_instance_state(
                card_id=self.card_id,
                instance_id=instance.instance_id,
                input_topic=instance.input_topic,
                desired_state="recording",
                auto_resume=bool(config.get("auto_resume_after_reboot", False)),
                session_id=instance.session_id,
                config=config,
            )
        except Exception:
            runtime.stop()
            raise
        self._runtimes[instance.instance_id] = runtime

    def _flush_runtime(self, instance: RecordingInstance) -> dict[str, Any] | None:
        runtime = self._runtimes.get(instance.instance_id)
        if runtime is None:
            return super()._flush_runtime(instance)
        return runtime.flush()

    def _stop_runtime(self, instance: RecordingInstance, *, for_shutdown: bool) -> None:
        runtime = self._runtimes.get(instance.instance_id)
        if runtime is not None:
            runtime.stop()
            self._runtimes.pop(instance.instance_id, None)
        if self._ledger is not None and not for_shutdown:
            self._ledger.set_instance_state(
                card_id=self.card_id,
                instance_id=instance.instance_id,
                input_topic=instance.input_topic,
                desired_state="idle",
                auto_resume=False,
                session_id=instance.session_id,
                config=self._effective_config(instance.instance_id),
            )

    def _runtime_stats(self, instance: RecordingInstance) -> dict[str, Any]:
        stats = self._ledger.summary(instance_id=instance.instance_id) if self._ledger is not None else {}
        runtime = self._runtimes.get(instance.instance_id)
        if runtime is not None:
            stats.update(runtime.stats())
        return stats

    def _restore_desired_instances(self) -> None:
        assert self._ledger is not None
        for saved in self._ledger.list_desired_recording(card_id=self.card_id):
            instance = RecordingInstance(
                instance_id=str(saved["instance_id"]),
                input_topic=str(saved["input_topic"]),
                session_id=f"session-recovered-{uuid.uuid4().hex}",
                state="idle",
                started_monotonic=time.monotonic(),
                resume_required=not bool(saved["auto_resume"]),
            )
            if saved["auto_resume"]:
                try:
                    instance.state = "recording"
                    self._start_runtime(instance, dict(saved.get("config") or {}))
                except Exception as exc:
                    instance.state = "idle"
                    instance.resume_required = True
                    instance.last_error = f"automatic resume failed: {exc}"
                    log.exception("failed to auto-resume video inspector %s", instance.instance_id)
            self._instances[instance.instance_id] = instance

    def shutdown(self) -> None:
        if self._closed:
            return
        super().shutdown()
        if self._ledger is not None and not self._runtimes:
            self._ledger.close()
        elif self._runtimes:
            log.error("leaving ledger open because video writer threads did not stop: %s", sorted(self._runtimes))
        self._closed = True
