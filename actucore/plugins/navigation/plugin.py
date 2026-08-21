"""Single public Navigation card owning mapping, planning and semantic runtime."""

from __future__ import annotations

import logging
import threading
import time

from .contract import (
    CONTROLLED_SEMANTIC_SPATIAL_TOOL_NAME,
    NAVIGATION_ACTIONS,
    NAVIGATION_CONFIG_SCHEMA,
    navigation_tool_definition,
)
from .mapping.contract import FAST_LIVO2_ACTIONS
from .mapping.plugin import FastLivo2Plugin
from .planning.contract import NAV2_ACTIONS
from .planning.plugin import Nav2Plugin
from .runtime import NavigationRuntime
from .semantic.plugin import VisionAndLanguageNavigationPlugin


log = logging.getLogger(__name__)


class NavigationPlugin:
    """Expose one Canvas card and one lifecycle for the whole navigation stack."""

    PREFIX = CONTROLLED_SEMANTIC_SPATIAL_TOOL_NAME

    def __init__(
        self,
        plugin_cfg: dict,
        namespace: str,
        executor,
        *,
        runtime=None,
        mapping_plugin=None,
        planning_plugin=None,
        semantic_plugin=None,
    ):
        raw_cfg = dict(plugin_cfg or {})
        raw_cfg.pop("enabled", None)
        self._namespace = str(raw_cfg.pop("namespace", namespace) or namespace).strip(
            "/"
        )
        if not self._namespace:
            self._namespace = "ubuntu"
        mapping_cfg = {
            "namespace": self._namespace,
            **dict(raw_cfg.pop("mapping", {}) or {}),
        }
        planning_cfg = {
            "namespace": self._namespace,
            **dict(raw_cfg.pop("planning", {}) or {}),
        }
        semantic_cfg = {
            "namespace": self._namespace,
            **dict(raw_cfg.pop("semantic", {}) or {}),
        }
        self._config_error = (
            "unsupported navigation config sections: "
            + ",".join(sorted(raw_cfg))
            if raw_cfg
            else None
        )
        self._runtime = runtime or NavigationRuntime()
        self._mapping = mapping_plugin or FastLivo2Plugin(mapping_cfg, executor)
        self._planning = planning_plugin or Nav2Plugin(planning_cfg, executor)
        self._semantic = semantic_plugin or VisionAndLanguageNavigationPlugin(
            semantic_cfg,
            self._namespace,
            executor,
            goal_handler=self._handle_semantic_goal,
        )
        self._lock = threading.RLock()
        self._transition_lock = threading.Lock()
        self._started = False
        self._instance_id = ""
        self._external_wiring: dict[str, str] = {}

    def get_tools(self) -> list:
        return [navigation_tool_definition(self._namespace)]

    def dispatch(self, name: str, args: dict) -> dict | None:
        if name != self.PREFIX:
            return None
        if not isinstance(args, dict):
            return self._error("invalid_argument", "arguments must be an object")
        action = args.get("action")
        if action == "info":
            return self._info()
        if action == "config":
            return self._configure(args)
        if action == "start":
            return self._start(args)
        if action == "stop":
            return self._stop()
        if action not in NAVIGATION_ACTIONS:
            return self._error(
                "unsupported_action", f"unsupported navigation action: {action!r}"
            )
        with self._lock:
            if not self._started:
                return self._error(
                    "canvas_not_started",
                    "connect LiDAR, IMU and RGB, then start the "
                    "ControlledSemanticSpatial card",
                )
        if not self._runtime.info().get("running", False):
            return self._error(
                "navigation_runtime_unavailable",
                "an owned ControlledSemanticSpatial child process is not running; "
                "stop and restart the card",
            )
        if action in FAST_LIVO2_ACTIONS:
            return self._mapping.dispatch("fast_livo2", args)
        if action in NAV2_ACTIONS:
            return self._planning.dispatch("nav2", args)
        return self._semantic.dispatch("vln", args)

    def stop(self) -> dict:
        return self._stop()

    def close(self) -> None:
        self._stop()

    def _start(self, args: dict) -> dict:
        with self._transition_lock:
            return self._start_serialized(args)

    def _start_serialized(self, args: dict) -> dict:
        with self._lock:
            if self._started:
                result = self._info()
                result["already_started"] = True
                return result
            if self._config_error:
                return self._error("invalid_config", self._config_error)
        wiring = self._validate_external_wiring(args)
        if wiring.get("status") == "error":
            return wiring

        started: list[tuple[str, object]] = []
        try:
            self._set_mapping_runtime_active(True)
            runtime_result = self._runtime.start()
            if runtime_result.get("state") != "running":
                raise RuntimeError("navigation child processes did not stay running")
            started.append(("runtime", self._runtime))

            mapping_result = self._mapping.dispatch(
                "fast_livo2",
                {
                    "action": "start",
                    "instance_id": args.get("instance_id"),
                    "input_bindings": [
                        {"port": port, "topic": wiring["wired_topics"][port]}
                        for port in ("lidar", "imu")
                    ],
                },
            )
            self._require_started("mapping", mapping_result)
            started.append(("mapping", self._mapping))

            planner_bindings = [
                {
                    "port": "livo_odom",
                    "topic": f"/{self._namespace}/navigation/odom",
                },
                {
                    "port": "registered_cloud",
                    "topic": f"/{self._namespace}/navigation/cloud_registered",
                },
                {
                    "port": "static_map",
                    "topic": f"/{self._namespace}/navigation/static_map",
                },
            ]
            planning_start = {
                "action": "start",
                "instance_id": args.get("instance_id"),
                "input_bindings": planner_bindings,
            }
            planning_result = self._planning.dispatch("nav2", planning_start)
            if self._is_transient_planning_discovery_failure(planning_result):
                if not self._runtime.info().get("running", False):
                    raise RuntimeError(
                        "planning start failed after a navigation child process exited"
                    )
                log.warning(
                    "[ControlledSemanticSpatial] Nav2 command subscriber was not "
                    "discovered on the first attempt; rebuilding only the planning "
                    "bridge before rollback"
                )
                time.sleep(0.25)
                planning_result = self._planning.dispatch("nav2", planning_start)
            self._require_started("planning", planning_result)
            started.append(("planning", self._planning))

            semantic_result = self._semantic.dispatch(
                "vln",
                {
                    "action": "start",
                    "input_bindings": [
                        {
                            "port": "rgb",
                            "topic": wiring["wired_topics"]["rgb"],
                        },
                        {
                            "port": "livo_odom",
                            "topic": f"/{self._namespace}/navigation/odom",
                        },
                        {
                            "port": "livo_status",
                            "topic": (
                                f"/{self._namespace}/navigation/fast_livo2/status"
                            ),
                        },
                    ],
                },
            )
            self._require_started("semantic", semantic_result)
            started.append(("semantic", self._semantic))
        except Exception as exc:
            log.error(
                "[ControlledSemanticSpatial] start failed: %s",
                exc,
                exc_info=True,
            )
            cleanup = self._stop_started(started)
            self._set_mapping_runtime_active(False)
            return self._error(
                "navigation_start_failed",
                str(exc),
                cleanup=cleanup,
            )

        with self._lock:
            self._started = True
            self._instance_id = str(args.get("instance_id") or "default").strip()
            self._external_wiring = dict(wiring["wired_topics"])
        result = self._info()
        result.update({"state": "ready", "status": "ready"})
        return result

    def _stop(self) -> dict:
        with self._transition_lock:
            return self._stop_serialized()

    def _stop_serialized(self) -> dict:
        results = {}
        # Mapping owns the only operation that can legitimately require a
        # retryable stop: it must wait for FAST-LIVO2 to exit and persist the
        # map transaction.  Do it first so a retry leaves all owned processes
        # and the already-started Canvas lifecycle intact.
        try:
            results["mapping"] = self._mapping.dispatch(
                "fast_livo2", {"action": "stop"}
            )
        except Exception as exc:
            results["mapping"] = self._error("component_stop_failed", str(exc))
        mapping_result = results["mapping"]
        if (
            isinstance(mapping_result, dict)
            and mapping_result.get("retryable") is True
            and (
                mapping_result.get("status") == "error"
                or mapping_result.get("state") == "error"
            )
        ):
            return self._error(
                "navigation_stop_pending",
                "mapping finalization is retryable; retry stop without restarting "
                "ControlledSemanticSpatial",
                component_results=results,
                retryable=True,
                canvas_wired=True,
            )

        for name, plugin, prefix in (
            ("semantic", self._semantic, "vln"),
            ("planning", self._planning, "nav2"),
        ):
            try:
                results[name] = plugin.dispatch(prefix, {"action": "stop"})
            except Exception as exc:
                results[name] = self._error("component_stop_failed", str(exc))
        try:
            results["runtime"] = self._runtime.stop()
        except Exception as exc:
            results["runtime"] = self._error("runtime_stop_failed", str(exc))
        self._set_mapping_runtime_active(False)
        with self._lock:
            self._started = False
            self._instance_id = ""
            self._external_wiring = {}
        failures = [
            name
            for name, result in results.items()
            if isinstance(result, dict)
            and (result.get("status") == "error" or result.get("state") == "error")
        ]
        if failures:
            return self._error(
                "navigation_stop_failed",
                "failed to stop: " + ",".join(failures),
                component_results=results,
            )
        return {
            "state": "idle",
            "status": "idle",
            "canvas_wired": False,
            "component_results": results,
            "shadow_only": True,
            "physical_execution": False,
        }

    def _set_mapping_runtime_active(self, active: bool) -> None:
        setter = getattr(self._mapping, "set_runtime_active", None)
        if callable(setter):
            setter(active)

    def _configure(self, args: dict) -> dict:
        with self._transition_lock:
            return self._configure_serialized(args)

    def _configure_serialized(self, args: dict) -> dict:
        with self._lock:
            if self._started:
                return self._error(
                    "config_while_running",
                    "stop ControlledSemanticSpatial before changing config",
                )
        updates = {
            key: value
            for key, value in args.items()
            if key not in {"action", "instance_id"}
        }
        unknown = sorted(
            set(updates) - set(NAVIGATION_CONFIG_SCHEMA["properties"])
        )
        if unknown:
            return self._error(
                "invalid_config",
                "unsupported navigation config fields: " + ",".join(unknown),
            )
        vlm_required = {"vlm_base_url", "vlm_api_key", "vlm_model"}
        vlm_supplied = set(updates) & {
            *vlm_required,
            "vlm_timeout_sec",
        }
        if vlm_supplied and not vlm_required <= set(updates):
            missing = sorted(vlm_required - set(updates))
            return self._error(
                "invalid_config",
                "VLM config must include vlm_base_url, vlm_api_key and vlm_model; "
                "missing=" + ",".join(missing),
            )
        mapping_updates = {
            "action": "config",
            **{
                target: updates[source]
                for source, target in (
                    ("mapping_request_timeout_sec", "request_timeout_sec"),
                    ("mapping_discovery_timeout_sec", "discovery_timeout_sec"),
                    ("map_voxel_size_m", "map_voxel_size_m"),
                    ("obstacle_min_height_m", "obstacle_min_height_m"),
                    ("obstacle_max_height_m", "obstacle_max_height_m"),
                    ("collection_enabled", "collection_enabled"),
                    ("collection_directory", "collection_directory"),
                )
                if source in updates
            },
        }
        planning_updates = {
            "action": "config",
            **{
                target: updates[source]
                for source, target in (
                    ("planning_request_timeout_sec", "request_timeout_sec"),
                    ("planning_discovery_timeout_sec", "discovery_timeout_sec"),
                    ("min_x_mps", "min_x_mps"),
                    ("max_x_mps", "max_x_mps"),
                    ("min_y_mps", "min_y_mps"),
                    ("max_y_mps", "max_y_mps"),
                    ("min_yaw_rps", "min_yaw_rps"),
                    ("max_yaw_rps", "max_yaw_rps"),
                )
                if source in updates
            },
        }
        results = {
            "mapping": self._mapping.dispatch("fast_livo2", mapping_updates),
            "planning": self._planning.dispatch("nav2", planning_updates),
        }
        semantic_keys = {
            "vlm_base_url": "base_url",
            "vlm_api_key": "api_key",
            "vlm_model": "model",
            "vlm_timeout_sec": "timeout_sec",
        }
        if any(key in updates for key in semantic_keys):
            semantic_updates = {
                "action": "config",
                **{
                    target: updates[source]
                    for source, target in semantic_keys.items()
                    if source in updates
                },
            }
            results["semantic"] = self._semantic.dispatch("vln", semantic_updates)
        failures = [
            name
            for name, result in results.items()
            if not isinstance(result, dict)
            or result.get("status") in {"error", "invalid_config"}
            or result.get("state") == "error"
        ]
        if failures:
            return self._error(
                "invalid_config",
                "invalid component config: " + ",".join(failures),
                component_results=results,
            )
        return {
            "state": "configured",
            "status": "configured",
            "component_results": results,
            "takes_effect": "next_start",
            "shadow_only": True,
            "physical_execution": False,
        }

    def _validate_external_wiring(self, args: dict) -> dict:
        tool = self.get_tools()[0]
        expected = {item["port"]: item["topic"] for item in tool["topic_in"]}
        required = {
            item["port"]
            for item in tool["topic_in"]
            if item.get("required", True)
        }
        mapping_info = self._mapping.dispatch("fast_livo2", {"action": "info"})
        collection_enabled = (
            isinstance(mapping_info, dict)
            and isinstance(mapping_info.get("config"), dict)
            and mapping_info["config"].get("collection_enabled") is True
        )
        if collection_enabled:
            required.update({"rgb_v2", "depth_v2"})
        bindings = args.get("input_bindings") or []
        raw_topics = args.get("input_topics") or []
        if isinstance(raw_topics, str):
            raw_topics = [raw_topics]
        if args.get("input_topic"):
            raw_topics = [*raw_topics, args["input_topic"]]
        if bindings:
            if not isinstance(bindings, list) or any(
                not isinstance(item, dict) for item in bindings
            ):
                return self._error(
                    "invalid_canvas_wiring", "input_bindings must be an array"
                )
            ports = [str(item.get("port", "")) for item in bindings]
            if len(ports) != len(set(ports)):
                return self._error(
                    "invalid_canvas_wiring", "input_bindings contains duplicate ports"
                )
            wired = {
                str(item.get("port", "")): str(item.get("topic", "")).strip()
                for item in bindings
            }
        else:
            if not isinstance(raw_topics, list) or any(
                not isinstance(topic, str) for topic in raw_topics
            ):
                return self._error(
                    "invalid_canvas_wiring", "input_topics must be an array"
                )
            selected = {topic.strip() for topic in raw_topics}
            wired = {
                port: topic for port, topic in expected.items() if topic in selected
            }
        unknown = sorted(set(wired) - set(expected))
        missing = sorted(port for port in required if wired.get(port) != expected[port])
        wrong = sorted(
            port
            for port, topic in wired.items()
            if port in expected and topic != expected[port]
        )
        if unknown or missing or wrong:
            details = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if wrong:
                details.append("wrong_topic=" + ",".join(wrong))
            if unknown:
                details.append("unexpected=" + ",".join(unknown))
            return self._error(
                "invalid_canvas_wiring",
                "ControlledSemanticSpatial requires exact external bindings ("
                + "; ".join(details)
                + ")",
            )
        return {"wired_topics": wired}

    def _handle_semantic_goal(self, goal: dict, *, control_nav_id=None) -> dict:
        request = {
            "action": "navigate_to_pose",
            "x": goal["x"],
            "y": goal["y"],
            "yaw": goal["yaw"],
            "speed": goal["speed"],
        }
        if control_nav_id is not None:
            request["_control_nav_id"] = control_nav_id
        return self._planning.dispatch(
            "nav2",
            request,
        )

    def _info(self) -> dict:
        tool = self.get_tools()[0]
        with self._lock:
            started = self._started
            instance_id = self._instance_id
            wired = dict(self._external_wiring)
            config_error = self._config_error
        runtime_info = self._runtime.info()
        runtime_failed = started and not runtime_info.get("running", False)
        state = (
            "error"
            if config_error or runtime_failed
            else "ready"
            if started
            else "idle"
        )
        return {
            "name": self.PREFIX,
            "type": "processor",
            "state": state,
            "status": state,
            "canvas_wired": started,
            "instance_id": instance_id or None,
            "runtime": runtime_info,
            "mapping": self._mapping.dispatch("fast_livo2", {"action": "info"}),
            "planning": self._planning.dispatch("nav2", {"action": "info"}),
            "semantic": self._semantic.dispatch("vln", {"action": "info"}),
            "topic_in": [
                {**item, "connected": wired.get(item["port"]) == item["topic"]}
                for item in tool["topic_in"]
            ],
            "topic_out": tool["topic_out"],
            "container_model": "single_actucore_container",
            "docker_runtime_dependency": False,
            "task_identity": "actucore_generated_nav_id",
            "error_code": (
                "invalid_config"
                if config_error
                else "navigation_runtime_unavailable"
                if runtime_failed
                else None
            ),
            "error": config_error or (
                "an owned ControlledSemanticSpatial child process exited"
                if runtime_failed
                else None
            ),
            "shadow_only": True,
            "physical_execution": False,
        }

    def _stop_started(self, started: list[tuple[str, object]]) -> dict:
        results = {}
        for name, component in reversed(started):
            try:
                if name == "runtime":
                    results[name] = component.stop()
                else:
                    prefix = {
                        "mapping": "fast_livo2",
                        "planning": "nav2",
                        "semantic": "vln",
                    }[name]
                    results[name] = component.dispatch(prefix, {"action": "stop"})
            except Exception as exc:
                results[name] = self._error("cleanup_failed", str(exc))
        return results

    @staticmethod
    def _require_started(name: str, result: dict | None) -> None:
        if not isinstance(result, dict):
            raise RuntimeError(f"{name} returned an invalid start response")
        if result.get("state") == "error" or result.get("status") == "error":
            raise RuntimeError(
                f"{name} start failed: "
                + str(result.get("error") or result.get("error_code") or result)
            )

    @staticmethod
    def _is_transient_planning_discovery_failure(result: dict | None) -> bool:
        return (
            isinstance(result, dict)
            and result.get("error_code") == "nav2_runtime_unavailable"
            and "not subscribed to the command topic"
            in str(result.get("error") or result.get("message") or "")
        )

    @staticmethod
    def _error(code: str, message: str, **extra) -> dict:
        return {
            **extra,
            "state": "error",
            "status": "error",
            "error_code": code,
            "error": message,
            "message": message,
            "shadow_only": True,
            "physical_execution": False,
        }


__all__ = ["NavigationPlugin"]
