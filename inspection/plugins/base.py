from __future__ import annotations

import copy
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any


_SHARED_PROPERTIES = {
    "storage_backend": {
        "type": "string", "enum": ["cos"], "default": "cos", "scope": "shared",
        "description": "Storage backend. Local durability is independent from asynchronous upload.",
    },
    "credential_profile": {
        "type": "string", "default": "default", "scope": "shared",
        "description": "Deployment secret profile name; never contains credentials.",
    },
    "cos_region": {
        "type": "string", "default": "ap-beijing", "scope": "shared",
        "description": "Tencent COS region.",
    },
    "cos_bucket": {
        "type": "string", "scope": "shared",
        "description": "Full Tencent COS bucket name.",
    },
    "cos_prefix": {
        "type": "string", "default": "inspection-data", "scope": "shared",
        "description": "Object key prefix without leading or trailing slash.",
    },
    "device_id": {
        "type": "string", "default": socket.gethostname(), "scope": "shared",
        "description": "Stable device identifier used in local and COS paths.",
    },
    "upload_enabled": {
        "type": "boolean", "default": True, "scope": "shared",
        "description": "Enable asynchronous COS upload when the durable backend lands.",
    },
    "upload_concurrency": {
        "type": "integer", "minimum": 1, "maximum": 8, "default": 2, "scope": "shared",
    },
    "multipart_threshold_mb": {
        "type": "integer", "minimum": 1, "default": 64, "scope": "shared",
    },
    "multipart_stale_hours": {
        "type": "integer", "minimum": 1, "maximum": 168, "default": 24, "scope": "shared",
    },
    "retry_max_seconds": {
        "type": "integer", "minimum": 1, "default": 300, "scope": "shared",
    },
    "shutdown_finalize_timeout_seconds": {
        "type": "integer", "minimum": 5, "maximum": 60, "default": 15, "scope": "shared",
    },
}


@dataclass
class RecordingInstance:
    instance_id: str
    input_topic: str
    session_id: str
    state: str = "recording"
    finalized_segments: int = 0
    started_monotonic: float = 0.0
    resume_required: bool = False
    last_error: str = ""


class InspectorPlugin:
    """Shared Inspector card contract with overridable lifecycle hooks."""

    def __init__(
        self,
        *,
        card_id: str,
        display_name: str,
        input_format: str,
        input_description: str,
        instance_properties: dict[str, dict[str, Any]],
        runtime_mode: str = "gate1-contract-only",
        storage_ready: bool = False,
    ) -> None:
        self.PREFIX = card_id
        self.card_id = card_id
        self.display_name = display_name
        self.input_format = input_format
        self.input_description = input_description
        self._lock = threading.RLock()
        self._instances: dict[str, RecordingInstance] = {}
        self._instance_config: dict[str, dict[str, Any]] = {}
        self._shared_config = self._defaults(_SHARED_PROPERTIES)
        self._instance_properties = copy.deepcopy(instance_properties)
        self._instance_defaults = self._defaults(self._instance_properties)
        self._runtime_mode = runtime_mode
        self._storage_ready = storage_ready
        self._tool = self._build_tool()

    @staticmethod
    def _defaults(properties: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return {key: spec["default"] for key, spec in properties.items() if "default" in spec}

    def _build_tool(self) -> dict[str, Any]:
        config_properties = copy.deepcopy(_SHARED_PROPERTIES)
        config_properties.update(copy.deepcopy(self._instance_properties))
        return {
            "name": self.card_id,
            "type": "inspector",
            "multiInstance": True,
            "agentCallable": False,
            "description": f"{self.display_name} — record a connected topic to durable storage",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["info", "config", "start", "stop", "flush", "testupload"],
                        "description": "Lifecycle or diagnostic action.",
                    },
                    "instance_id": {"type": "string", "description": "Canvas card instance id."},
                    "input_topic": {"type": "string", "description": "Connected ROS2 input topic."},
                },
                "required": ["action"],
            },
            "configSchema": {
                "type": "object",
                "properties": config_properties,
                "required": ["cos_bucket"],
            },
            "topic_in": [{"format": self.input_format, "desc": self.input_description}],
            "topic_out": [],
        }

    def get_tools(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(self._tool)]

    def _effective_config(self, instance_id: str) -> dict[str, Any]:
        return {
            **self._shared_config,
            **self._instance_defaults,
            **self._instance_config.get(instance_id, {}),
        }

    @staticmethod
    def _validate_config_value(key: str, value: Any, spec: dict[str, Any]) -> None:
        expected = spec.get("type")
        valid_type = {
            "string": isinstance(value, str),
            "boolean": isinstance(value, bool),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        }.get(str(expected), True)
        if not valid_type:
            raise ValueError(f"config field {key} must be {expected}")
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"config field {key} must be one of {spec['enum']}")
        if "minimum" in spec and value < spec["minimum"]:
            raise ValueError(f"config field {key} must be >= {spec['minimum']}")
        if "maximum" in spec and value > spec["maximum"]:
            raise ValueError(f"config field {key} must be <= {spec['maximum']}")

    def _apply_config(self, args: dict[str, Any]) -> dict[str, Any]:
        instance_id = str(args.get("instance_id", ""))
        staged: list[tuple[str, Any, dict[str, Any]]] = []
        properties = self._tool["configSchema"]["properties"]
        for key, value in args.items():
            if key in {"action", "instance_id"}:
                continue
            spec = properties.get(key)
            if not spec:
                raise ValueError(f"unknown config field: {key}")
            if spec.get("scope") == "instance" and not instance_id:
                raise ValueError(f"instance_id is required for instance config field: {key}")
            self._validate_config_value(key, value, spec)
            staged.append((key, value, spec))

        with self._lock:
            changed_shared = any(
                spec.get("scope") != "instance" and self._shared_config.get(key) != value
                for key, value, spec in staged
            )
            changed_instance = any(
                spec.get("scope") == "instance"
                and self._effective_config(instance_id).get(key) != value
                for key, value, spec in staged
            )
            if changed_shared and any(item.state == "recording" for item in self._instances.values()):
                raise ValueError("stop all recording instances before changing shared config")
            active = self._instances.get(instance_id)
            if changed_instance and active is not None and active.state == "recording":
                raise ValueError(f"stop instance {instance_id} before changing instance config")

            applied: list[str] = []
            for key, value, spec in staged:
                target = (
                    self._instance_config.setdefault(instance_id, {})
                    if spec.get("scope") == "instance"
                    else self._shared_config
                )
                target[key] = value
                applied.append(key)
        return {
            "state": "configured",
            "instance_id": instance_id or None,
            "applied": sorted(applied),
            "runtime_mode": self._runtime_mode,
        }

    def _validate_start_config(self, instance_id: str) -> None:
        cfg = self._effective_config(instance_id)
        if cfg.get("upload_enabled", True) and not str(cfg.get("cos_bucket", "")).strip():
            raise ValueError("cos_bucket is required when upload_enabled=true")
        prefix = str(cfg.get("cos_prefix", ""))
        if prefix.startswith("/") or prefix.endswith("/"):
            raise ValueError("cos_prefix must not start or end with '/'")

    def _instance_info(self, instance_id: str, input_topic: str = "") -> dict[str, Any]:
        instance = self._instances.get(instance_id)
        topic = instance.input_topic if instance else input_topic
        state = instance.state if instance else "idle"
        runtime = self._runtime_stats(instance) if instance else {}
        return {
            "name": self.display_name,
            "card_id": self.card_id,
            "type": "inspector",
            "state": state,
            "instance_id": instance_id,
            "session_id": instance.session_id if instance else None,
            "topic_in": [{"topic": topic, "format": self.input_format, "desc": self.input_description}] if topic else [],
            "topic_out": [],
            "recording": state == "recording",
            "local_bytes": int(runtime.get("local_bytes", 0)),
            "upload_backlog": int(runtime.get("upload_backlog", 0)),
            "uploaded_verified": int(runtime.get("uploaded_verified", 0)),
            "dropped": int(runtime.get("dropped", 0)),
            "finalized_segments": int(runtime.get("finalized_segments", instance.finalized_segments if instance else 0)),
            "resume_required": bool(instance.resume_required) if instance else False,
            "last_error": runtime.get("last_error", instance.last_error if instance else ""),
            "runtime_mode": self._runtime_mode,
            "storage_ready": self._storage_ready,
            "desc": (
                "Durable local segment writer is active; COS upload may still be pending."
                if self._storage_ready else
                "Contract-only fake writer; no local files or COS objects are produced in Gate 1."
            ),
        }

    def _runtime_stats(self, instance: RecordingInstance) -> dict[str, Any]:
        return {}

    def _start_runtime(self, instance: RecordingInstance, config: dict[str, Any]) -> None:
        return None

    def _flush_runtime(self, instance: RecordingInstance) -> dict[str, Any] | None:
        instance.finalized_segments += 1
        return None

    def _stop_runtime(self, instance: RecordingInstance, *, for_shutdown: bool) -> None:
        return None

    def _test_upload(self) -> dict[str, Any]:
        return {
            "state": "unsupported",
            "verified": False,
            "runtime_mode": self._runtime_mode,
            "message": "COS backend is not available in this runtime mode.",
        }

    def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        if name != self.card_id:
            return None
        action = str(args.get("action", "info"))
        instance_id = str(args.get("instance_id", ""))

        if action == "config":
            return self._apply_config(args)

        if action == "info":
            with self._lock:
                if instance_id:
                    return self._instance_info(instance_id, str(args.get("input_topic", "")))
                running = sum(1 for item in self._instances.values() if item.state == "recording")
                return {
                    "name": self.display_name,
                    "card_id": self.card_id,
                    "type": "inspector",
                    "state": "recording" if running else "idle",
                    "instances": len(self._instances),
                    "recording_instances": running,
                    "topic_in": [],
                    "topic_out": [],
                    "runtime_mode": self._runtime_mode,
                    "storage_ready": self._storage_ready,
                }

        if action == "start":
            if not instance_id:
                raise ValueError("instance_id is required")
            input_topic = str(args.get("input_topic", ""))
            if not input_topic:
                input_topics = args.get("input_topics") or []
                input_topic = str(input_topics[0]) if input_topics else ""
            if not input_topic:
                raise ValueError("input_topic is required")
            self._validate_start_config(instance_id)
            with self._lock:
                existing = self._instances.get(instance_id)
                if existing and existing.state == "recording":
                    if existing.input_topic != input_topic:
                        raise ValueError(
                            f"instance {instance_id} already records {existing.input_topic}; stop it before changing input_topic"
                        )
                    return self._instance_info(instance_id)
                instance = RecordingInstance(
                    instance_id=instance_id,
                    input_topic=input_topic,
                    session_id=f"session-{uuid.uuid4().hex}",
                    started_monotonic=time.monotonic(),
                )
                self._start_runtime(instance, self._effective_config(instance_id))
                self._instances[instance_id] = instance
                return self._instance_info(instance_id)

        if action == "flush":
            if not instance_id:
                raise ValueError("instance_id is required")
            with self._lock:
                instance = self._instances.get(instance_id)
                if not instance or instance.state != "recording":
                    raise ValueError(f"instance {instance_id} is not recording")
                finalized = self._flush_runtime(instance)
                return {
                    "state": "recording",
                    "instance_id": instance_id,
                    "finalized_segment": finalized,
                    "finalized_segments": self._instance_info(instance_id)["finalized_segments"],
                    "runtime_mode": self._runtime_mode,
                }

        if action == "stop":
            if not instance_id:
                raise ValueError("instance_id is required")
            with self._lock:
                instance = self._instances.get(instance_id)
                if not instance:
                    return self._instance_info(instance_id, str(args.get("input_topic", "")))
                self._stop_runtime(instance, for_shutdown=False)
                instance.state = "idle"
                instance.resume_required = False
                return self._instance_info(instance_id)

        if action == "testupload":
            return self._test_upload()

        raise ValueError(f"unsupported action: {action}")

    def shutdown(self) -> None:
        with self._lock:
            for instance in self._instances.values():
                if instance.state == "recording":
                    try:
                        self._stop_runtime(instance, for_shutdown=True)
                    except Exception as exc:
                        instance.last_error = str(exc)
                instance.state = "idle"
