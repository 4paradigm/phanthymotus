from __future__ import annotations

import copy
import threading
import time
from typing import Any

from .policy import (
    CardError,
    DEFAULT_TASK,
    PolicyClient,
    derive_health_url,
    validate_http_url,
    validate_seed,
    validate_task,
)


TOOL = {
    "name": "vlapi05g1",
    "type": "processor",
    "multiInstance": True,
    "description": "π0.5 G1 VLA inference — combine image and robot state topics into action proposals",
    "inputSchema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["info", "config", "start", "stop", "predict"],
                "description": "Card lifecycle or prediction action",
            },
            "instance_id": {
                "type": "string",
                "description": "Stable instance identifier",
            },
            "image_topic": {
                "type": "string",
                "description": "sensor_msgs/msg/CompressedImage input topic required by start",
            },
            "state_topic": {
                "type": "string",
                "description": "std_msgs/msg/String joints JSON input topic required by start",
            },
            "input_topics": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 2,
                "description": "Dashboard start compatibility: [image_topic, state_topic] in topic_in port order",
            },
            "output_topic": {
                "type": "string",
                "description": "Optional action proposal topic; defaults to {image_topic}/vlapi05g1",
            },
            "task": {
                "type": "string",
                "description": "Instance task or one-shot predict override",
            },
            "seed": {
                "type": "integer",
                "minimum": 0,
                "maximum": 2147483647,
                "description": "Instance seed or one-shot predict override",
            },
            "policy_url": {"type": "string"},
            "health_url": {"type": "string"},
            "request_timeout_s": {"type": "number", "minimum": 0.1, "maximum": 300.0},
            "max_image_bytes": {"type": "integer", "minimum": 1024, "maximum": 16777216},
            "max_image_age_s": {"type": "number", "minimum": 0.05, "maximum": 10.0},
            "max_state_age_s": {"type": "number", "minimum": 0.05, "maximum": 10.0},
        },
        "required": ["action"],
    },
    "configSchema": {
        "type": "object",
        "properties": {
            "policy_url": {
                "type": "string",
                "default": "http://127.0.0.1:18080/predict",
                "scope": "shared",
            },
            "health_url": {
                "type": "string",
                "default": "",
                "scope": "shared",
            },
            "request_timeout_s": {
                "type": "number",
                "default": 120.0,
                "minimum": 0.1,
                "maximum": 300.0,
                "scope": "shared",
            },
            "max_image_bytes": {
                "type": "integer",
                "default": 8388608,
                "minimum": 1024,
                "maximum": 16777216,
                "scope": "shared",
            },
            "task": {
                "type": "string",
                "default": DEFAULT_TASK,
                "scope": "instance",
            },
            "seed": {
                "type": "integer",
                "default": 0,
                "minimum": 0,
                "maximum": 2147483647,
                "scope": "instance",
            },
            "max_image_age_s": {
                "type": "number",
                "default": 1.0,
                "minimum": 0.05,
                "maximum": 10.0,
                "scope": "instance",
            },
            "max_state_age_s": {
                "type": "number",
                "default": 0.5,
                "minimum": 0.05,
                "maximum": 10.0,
                "scope": "instance",
            },
        },
        "required": [],
    },
    "topic_in": [
        {"format": "image/jpeg", "desc": "sensor_msgs/msg/CompressedImage RGB observation"},
        {"format": "data/json", "desc": "std_msgs/msg/String G1 joints state observation"},
    ],
    "topic_out": [
        {"format": "data/json", "desc": "pi05.g1.action_chunk.v1 action proposal"},
    ],
}


SHARED_DEFAULTS = {
    "policy_url": "http://127.0.0.1:18080/predict",
    "health_url": "",
    "request_timeout_s": 120.0,
    "max_image_bytes": 8 * 1024 * 1024,
}

INSTANCE_DEFAULTS = {
    "task": DEFAULT_TASK,
    "seed": 0,
    "max_image_age_s": 1.0,
    "max_state_age_s": 0.5,
}

SHARED_FIELDS = frozenset(SHARED_DEFAULTS)
INSTANCE_FIELDS = frozenset(INSTANCE_DEFAULTS)
CONFIG_FIELDS = SHARED_FIELDS | INSTANCE_FIELDS


def _bounded_number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be within {minimum}..{maximum}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be within {minimum}..{maximum}") from exc
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} must be within {minimum}..{maximum}")
    return number


def _bounded_integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer within {minimum}..{maximum}")
    return value


def _validate_topic(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty ROS topic")
    topic = value.strip()
    if any(character.isspace() for character in topic):
        raise ValueError(f"{name} must not contain whitespace")
    return topic


def _validate_shared(config: dict[str, Any]) -> dict[str, Any]:
    policy_url = validate_http_url(config["policy_url"], "policy_url")
    health_value = config.get("health_url")
    health_url = (
        validate_http_url(health_value, "health_url")
        if isinstance(health_value, str) and health_value.strip()
        else derive_health_url(policy_url)
    )
    return {
        "policy_url": policy_url,
        "health_url": health_url,
        "request_timeout_s": _bounded_number(config["request_timeout_s"], "request_timeout_s", 0.1, 300.0),
        "max_image_bytes": _bounded_integer(config["max_image_bytes"], "max_image_bytes", 1024, 16 * 1024 * 1024),
    }


def _validate_instance(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": validate_task(config["task"]),
        "seed": validate_seed(config["seed"]),
        "max_image_age_s": _bounded_number(config["max_image_age_s"], "max_image_age_s", 0.05, 10.0),
        "max_state_age_s": _bounded_number(config["max_state_age_s"], "max_state_age_s", 0.05, 10.0),
    }


class VLAPi05G1Plugin:
    PREFIX = "vlapi05g1"
    HEALTH_TIMEOUT_CAP_S = 5.0

    def __init__(self, plugin_cfg: dict, executor):
        self._lifecycle_lock = threading.RLock()
        self._health_lock = threading.Lock()
        shared = dict(SHARED_DEFAULTS)
        instance = dict(INSTANCE_DEFAULTS)
        for key in SHARED_FIELDS:
            if key in plugin_cfg:
                shared[key] = plugin_cfg[key]
        for key in INSTANCE_FIELDS:
            if key in plugin_cfg:
                instance[key] = plugin_cfg[key]
        health_value = plugin_cfg.get("health_url")
        self._health_url_is_explicit = isinstance(health_value, str) and bool(health_value.strip())
        self._shared_config = _validate_shared(shared)
        self._default_instance_config = _validate_instance(instance)
        self._instance_configs: dict[str, dict[str, Any]] = {}
        self._instances: dict[str, Any] = {}
        self._last_health: dict[str, Any] | None = None
        self._executor = executor

    def get_tools(self) -> list[dict[str, Any]]:
        return [TOOL]

    def dispatch(self, name: str, args: dict) -> dict[str, Any]:
        action = args.get("action") if name == self.PREFIX else name
        if action == "info":
            self._refresh_health()
            with self._lifecycle_lock:
                return self._info(args.get("instance_id"))
        if action == "config":
            with self._lifecycle_lock:
                return self._config(args)
        if action == "start":
            with self._lifecycle_lock:
                return self._start(args)
        if action == "stop":
            with self._lifecycle_lock:
                return self._stop(args)
        if action == "predict":
            return self._predict(args)
        raise ValueError(f"unsupported vlapi05g1 action: {action}")

    def _instance_config(self, instance_id: str) -> dict[str, Any]:
        return dict(self._instance_configs.get(instance_id, self._default_instance_config))

    def _refresh_health(self) -> None:
        with self._health_lock:
            with self._lifecycle_lock:
                shared = dict(self._shared_config)
            timeout_s = min(
                float(shared["request_timeout_s"]),
                self.HEALTH_TIMEOUT_CAP_S,
            )
            checked_at = time.time()
            started = time.monotonic()
            try:
                client = PolicyClient(
                    shared["policy_url"],
                    timeout_s,
                    shared["health_url"],
                )
                response = client.health()
                result = {
                    "status": "ok" if response.get("ok") is True else "error",
                    "checked_at": checked_at,
                    "health_url": shared["health_url"],
                    "latency_s": time.monotonic() - started,
                    "response": response,
                }
            except CardError as exc:
                result = {
                    "status": "error",
                    "checked_at": checked_at,
                    "health_url": shared["health_url"],
                    "latency_s": time.monotonic() - started,
                    "error_code": exc.code,
                    "message": exc.message,
                }
            with self._lifecycle_lock:
                if self._shared_config["health_url"] == shared["health_url"]:
                    self._last_health = result

    def _info(self, instance_id: Any) -> dict[str, Any]:
        if instance_id is not None:
            instance_id = str(instance_id).strip()
            node = self._instances.get(instance_id)
            if node is not None:
                status = node.status()
                status["shared_config"] = dict(self._shared_config)
                status["last_health"] = copy.deepcopy(self._last_health)
                return status
            return {
                "name": "π0.5 G1 VLA 推理",
                "state": "idle",
                "instance_id": instance_id,
                "topic_in": [],
                "topic_out": [],
                "config": self._instance_config(instance_id),
                "shared_config": dict(self._shared_config),
                "in_flight": False,
                "last_health": copy.deepcopy(self._last_health),
            }

        instances = {key: node.status() for key, node in self._instances.items()}
        topics_in = []
        topics_out = []
        for status in instances.values():
            status["last_health"] = copy.deepcopy(self._last_health)
            topics_in.extend(status["topic_in"])
            topics_out.extend(status["topic_out"])
        return {
            "name": "π0.5 G1 VLA 推理",
            "state": "running" if instances else "idle",
            "topic_in": topics_in,
            "topic_out": topics_out,
            "shared_config": dict(self._shared_config),
            "default_instance_config": dict(self._default_instance_config),
            "instances": instances,
            "last_health": copy.deepcopy(self._last_health),
        }

    def _config(self, args: dict[str, Any]) -> dict[str, Any]:
        updates = {
            key: value
            for key, value in args.items()
            if key not in {"action", "instance_id"} and value is not None
        }
        unknown = sorted(set(updates) - CONFIG_FIELDS)
        if unknown:
            raise ValueError(f"unknown vlapi05g1 config fields: {unknown}")
        shared_updates = {key: value for key, value in updates.items() if key in SHARED_FIELDS}
        instance_updates = {key: value for key, value in updates.items() if key in INSTANCE_FIELDS}
        if shared_updates and instance_updates:
            raise ValueError("shared and instance config must be updated in separate calls")

        if shared_updates:
            candidate = dict(self._shared_config)
            candidate.update(shared_updates)
            health_url_is_explicit = self._health_url_is_explicit
            if "policy_url" in shared_updates and "health_url" not in shared_updates:
                if not self._health_url_is_explicit:
                    candidate["health_url"] = ""
            if "health_url" in shared_updates:
                health_value = shared_updates["health_url"]
                health_url_is_explicit = isinstance(health_value, str) and bool(health_value.strip())
            validated = _validate_shared(candidate)
            changed = validated != self._shared_config
            if changed and self._instances:
                return {
                    "status": "error",
                    "error_code": "restart_required",
                    "message": "shared config changes require all vlapi05g1 instances to be stopped",
                    "applied": "none",
                    "restart_required": True,
                }
            self._health_url_is_explicit = health_url_is_explicit
            if changed:
                self._shared_config = validated
                self._last_health = None
            return {
                "status": "configured",
                "scope": "shared",
                "config": dict(self._shared_config),
                "applied": "immediate" if changed else "none",
                "restart_required": False,
            }

        if instance_updates:
            instance_id = args.get("instance_id")
            if not isinstance(instance_id, str) or not instance_id.strip():
                raise ValueError("instance_id is required for instance config")
            instance_id = instance_id.strip()
            candidate = self._instance_config(instance_id)
            candidate.update(instance_updates)
            validated = _validate_instance(candidate)
            self._instance_configs[instance_id] = validated
            node = self._instances.get(instance_id)
            if node is not None:
                node.update_instance_config(validated)
            return {
                "status": "configured",
                "scope": "instance",
                "instance_id": instance_id,
                "config": dict(validated),
                "applied": "immediate",
                "restart_required": False,
            }

        return {
            "status": "configured",
            "scope": "none",
            "shared_config": dict(self._shared_config),
            "applied": "none",
            "restart_required": False,
        }

    def _start(self, args: dict[str, Any]) -> dict[str, Any]:
        instance_id = args.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ValueError("instance_id is required for start")
        instance_id = instance_id.strip()
        input_topics = args.get("input_topics")
        if input_topics is not None:
            if args.get("image_topic") is not None or args.get("state_topic") is not None:
                raise ValueError("input_topics cannot be combined with image_topic or state_topic")
            if not isinstance(input_topics, list) or len(input_topics) != 2:
                raise ValueError("input_topics must be [image_topic, state_topic]")
            image_topic = _validate_topic(input_topics[0], "input_topics[0]")
            state_topic = _validate_topic(input_topics[1], "input_topics[1]")
        else:
            image_topic = _validate_topic(args.get("image_topic"), "image_topic")
            state_topic = _validate_topic(args.get("state_topic"), "state_topic")
        output_value = args.get("output_topic")
        output_topic = (
            _validate_topic(output_value, "output_topic")
            if output_value is not None and str(output_value).strip()
            else f"{image_topic.rstrip('/')}/{self.PREFIX}"
        )
        if len({image_topic, state_topic, output_topic}) != 3:
            raise ValueError("image_topic, state_topic and output_topic must be distinct")

        existing = self._instances.get(instance_id)
        if existing is not None:
            if (
                existing.image_topic != image_topic
                or existing.state_topic != state_topic
                or existing.output_topic != output_topic
            ):
                raise ValueError("instance is already running with different topics; stop it first")
            return existing.status()

        from .node import VLAPi05G1Node

        node = VLAPi05G1Node(
            image_topic=image_topic,
            state_topic=state_topic,
            output_topic=output_topic,
            instance_id=instance_id,
            shared_config=self._shared_config,
            instance_config=self._instance_config(instance_id),
        )
        try:
            added = self._executor.add_node(node)
            if added is False:
                raise RuntimeError("executor rejected vlapi05g1 node")
        except Exception:
            node.shutdown_card()
            node.destroy_node()
            raise
        self._instances[instance_id] = node
        return node.status()

    def _stop(self, args: dict[str, Any]) -> dict[str, Any]:
        instance_id = args.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ValueError("instance_id is required for stop")
        instance_id = instance_id.strip()
        node = self._instances.pop(instance_id, None)
        if node is not None:
            try:
                node.shutdown_card()
            finally:
                try:
                    self._executor.remove_node(node)
                finally:
                    node.destroy_node()
        return {
            "name": "π0.5 G1 VLA 推理",
            "state": "idle",
            "instance_id": instance_id,
            "topic_in": [],
            "topic_out": [],
        }

    def _predict(self, args: dict[str, Any]) -> dict[str, Any]:
        instance_id = args.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ValueError("instance_id is required for predict")
        instance_id = instance_id.strip()
        with self._lifecycle_lock:
            node = self._instances.get(instance_id)
        if node is None:
            return CardError("not_running", f"instance {instance_id} is not running").as_result()
        try:
            return node.predict(task=args.get("task"), seed=args.get("seed"))
        except CardError as exc:
            return exc.as_result()
