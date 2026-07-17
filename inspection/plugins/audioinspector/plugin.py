from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

from plugins.base import InspectorPlugin, RecordingInstance
from storage.atomic_writer import AudioSegmentWriter
from storage.ledger import SegmentLedger
from storage.recovery import reconcile_audio_store
from storage.services import DurableServices


log = logging.getLogger(__name__)


class AudioInspectorPlugin(InspectorPlugin):
    def __init__(self, plugin_config: dict[str, Any] | None = None, executor: Any = None) -> None:
        plugin_config = plugin_config or {}
        runtime_mode = str(plugin_config.get("runtime_mode", "gate1-contract-only"))
        if runtime_mode not in {"gate1-contract-only", "ros2"}:
            raise ValueError(f"unsupported audioinspector runtime_mode: {runtime_mode}")
        super().__init__(
            card_id="audioinspector",
            display_name="Audio Inspector",
            input_format="audio/pcm-16k",
            input_description="PCM_S16_LE, 16000 Hz, mono",
            instance_properties={
                "segment_seconds": {"type": "integer", "minimum": 5, "maximum": 600, "default": 60, "scope": "instance"},
                "local_retention_hours": {"type": "number", "minimum": 1, "maximum": 720, "default": 24, "scope": "instance"},
                "local_max_gb": {"type": "number", "minimum": 0.1, "default": 4, "scope": "instance"},
                "container": {"type": "string", "enum": ["wav"], "default": "wav", "scope": "instance"},
                "queue_seconds": {"type": "number", "minimum": 1, "default": 5, "scope": "instance"},
                "record_mode": {"type": "string", "enum": ["continuous"], "default": "continuous", "scope": "instance"},
                "auto_resume_after_reboot": {"type": "boolean", "default": False, "scope": "instance"},
            },
            runtime_mode="ros2-durable-audio" if runtime_mode == "ros2" else runtime_mode,
            storage_ready=runtime_mode == "ros2",
        )
        self._executor = executor
        self._data_root = Path(plugin_config.get("data_root", "/opt/phanthy-motus/inspection-data"))
        self._state_root = Path(plugin_config.get("state_root", "/opt/phanthy-motus/inspection-state"))
        self._runtimes: dict[str, Any] = {}
        self._ledger: SegmentLedger | None = None
        self._services: DurableServices | None = None
        self._recovery_stats: dict[str, int] = {}
        self._closed = False
        if runtime_mode == "ros2":
            if executor is None:
                raise RuntimeError("ROS2 executor is required for audioinspector runtime_mode=ros2")
            self._ledger = SegmentLedger(self._state_root / "ledger.sqlite3")
            self._recovery_stats = reconcile_audio_store(self._data_root, self._ledger)
            log.info("audio store recovery complete: %s", self._recovery_stats)
            self._services = DurableServices(
                ledger=self._ledger,
                data_root=self._data_root,
                card_id=self.card_id,
                on_critical=self._handle_critical,
            )
            self._services.restore_latest()
            self._restore_desired_instances()

    def _apply_config(self, args: dict[str, Any]) -> dict[str, Any]:
        result = super()._apply_config(args)
        if self._services is not None:
            instance_id = str(args.get("instance_id", ""))
            ready = self._services.configure(self._effective_config(instance_id))
            result["adapter_ok"] = ready
            result["upload_ready"] = ready
            if not ready:
                result["message"] = self._services.last_error
        return result

    def _validate_start_config(self, instance_id: str) -> None:
        super()._validate_start_config(instance_id)
        config = self._effective_config(instance_id)
        estimated = 16000 * 2 * 3600 * float(config["local_retention_hours"])
        budget = float(config["local_max_gb"]) * 1024 * 1024 * 1024
        if estimated > budget * 0.8:
            raise ValueError(
                f"audio retention estimate {estimated / 1024**3:.2f} GiB exceeds 80% of local_max_gb"
            )
        if self._services is not None:
            self._services.retention.sweep_once()
            if bool(config.get("upload_enabled", True)) and not self._services.configure(config):
                raise ValueError(self._services.last_error or "COS uploader is not ready")
        if self._ledger is not None:
            local_bytes = self._ledger.summary(card_id=self.card_id, instance_id=instance_id)["local_bytes"]
            if local_bytes > budget * 0.95:
                raise ValueError("local spool is above the 95% critical watermark")

    def _start_runtime(self, instance: RecordingInstance, config: dict[str, Any]) -> None:
        if self._ledger is None:
            return super()._start_runtime(instance, config)
        from plugins.audioinspector.runtime import AudioRecorderRuntime

        writer = AudioSegmentWriter(
            data_root=self._data_root,
            ledger=self._ledger,
            card_id=self.card_id,
            instance_id=instance.instance_id,
            input_topic=instance.input_topic,
            session_id=instance.session_id,
            device_id=str(config.get("device_id", "unknown")),
            segment_seconds=int(config.get("segment_seconds", 60)),
        )
        runtime = AudioRecorderRuntime(
            executor=self._executor,
            writer=writer,
            instance_id=instance.instance_id,
            input_topic=instance.input_topic,
            queue_seconds=float(config.get("queue_seconds", 5)),
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
        stats = self._ledger.summary(card_id=self.card_id, instance_id=instance.instance_id) if self._ledger is not None else {}
        runtime = self._runtimes.get(instance.instance_id)
        if runtime is not None:
            stats.update(runtime.stats())
        if self._services is not None:
            stats.update(self._services.stats())
            if not stats.get("last_error"):
                stats["last_error"] = stats.get("upload_last_error") or stats.get("upload_service_error", "")
        return stats

    def _test_upload(self) -> dict[str, Any]:
        if self._services is None:
            return super()._test_upload()
        return self._services.test_upload()

    def _handle_critical(self, instance_id: str, local_bytes: int, max_bytes: int) -> None:
        with self._lock:
            instance = self._instances.get(instance_id)
            if instance is None or instance.state != "recording":
                return
            try:
                self._stop_runtime(instance, for_shutdown=False)
                instance.state = "paused_disk_full"
                instance.resume_required = True
                instance.last_error = f"local spool critical: {local_bytes} > {max_bytes} bytes"
            except Exception as exc:
                instance.last_error = f"failed to pause at disk critical watermark: {exc}"

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
                    saved_config = dict(saved.get("config") or {})
                    self._apply_config({"action": "config", "instance_id": instance.instance_id, **saved_config})
                    self._validate_start_config(instance.instance_id)
                    instance.state = "recording"
                    self._start_runtime(instance, self._effective_config(instance.instance_id))
                except Exception as exc:
                    instance.state = "idle"
                    instance.resume_required = True
                    instance.last_error = f"automatic resume failed: {exc}"
                    log.exception("failed to auto-resume audio inspector %s", instance.instance_id)
            self._instances[instance.instance_id] = instance

    def shutdown(self) -> None:
        if self._closed:
            return
        super().shutdown()
        services_stopped = True
        if self._services is not None:
            services_stopped = self._services.stop()
        if self._ledger is not None and not self._runtimes and services_stopped:
            self._ledger.close()
        elif self._runtimes or not services_stopped:
            log.error("leaving ledger open because storage workers did not stop cleanly")
        self._closed = True
