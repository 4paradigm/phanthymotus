"""The controlled_semantic_spatial card owns mapping, planning and semantic runtime."""

from __future__ import annotations

import logging
import re
import threading
from copy import deepcopy

from ..control_stream import negotiate
from .contract import (
    CONTROLLED_SEMANTIC_SPATIAL_TOOL_NAME,
    CONTROLLED_SEMANTIC_SPATIAL_ACTIONS,
    CONTROLLED_SEMANTIC_SPATIAL_CONFIG_SCHEMA,
    SEMANTIC_ACTIONS,
    controlled_semantic_spatial_tool_definition,
)
from .mapping.contract import FAST_LIVO2_ACTIONS
from .mapping.plugin import (
    ConfigError as MappingConfigError,
    FastLivo2Plugin,
    _validated_config as validated_mapping_config,
)
from .planning.contract import NAV2_ACTIONS
from .planning.plugin import (
    ConfigError as PlanningConfigError,
    Nav2Plugin,
    _validated_config as validated_planning_config,
)
from .runtime import ControlledSemanticSpatialRuntime
from .semantic.plugin import VisionAndLanguageNavigationPlugin


log = logging.getLogger(__name__)
_ABSOLUTE_ROS_TOPIC_RE = re.compile(
    r"^/(?:[A-Za-z_][A-Za-z0-9_]*)(?:/[A-Za-z_][A-Za-z0-9_]*)*$"
)


class ControlledSemanticSpatialPlugin:
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
        completion_callback=None,
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
        legacy_semantic_options = sorted(
            {"goal_topic", "subscriber_timeout_sec"}.intersection(semantic_cfg)
        )
        if (
            str(semantic_cfg.get("sensor_sync_mode", "receive_time")).strip()
            != "receive_time"
        ):
            legacy_semantic_options.append("sensor_sync_mode")
            # Keep plugin construction available so start can return the config error.
            semantic_cfg["sensor_sync_mode"] = "receive_time"
        if legacy_semantic_options:
            legacy_error = (
                "unsupported semantic config options: "
                + ",".join(legacy_semantic_options)
            )
            self._config_error = (
                self._config_error + "; " + legacy_error
                if self._config_error else legacy_error
            )
        self._runtime = runtime or ControlledSemanticSpatialRuntime()
        self._mapping = mapping_plugin or FastLivo2Plugin(mapping_cfg, executor)
        self._planning = planning_plugin or Nav2Plugin(planning_cfg, executor)
        set_completion_callback = getattr(
            self._planning, "set_completion_callback", None
        )
        if callable(set_completion_callback):
            set_completion_callback(completion_callback)
        self._semantic = semantic_plugin or VisionAndLanguageNavigationPlugin(
            semantic_cfg,
            self._namespace,
            executor,
            goal_handler=self._handle_semantic_goal,
        )
        self._lock = threading.RLock()
        self._transition_lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._pending_stops = 0
        self._started = False
        self._generation = None
        self._inflight_planning = 0
        self._inflight_semantic = 0
        self._inflight_mapping = 0
        self._last_terminal_stop_error = None
        self._pending_terminal_stop_error = None
        self._instance_id = ""
        self._external_wiring: dict[str, str] = {}
        self._control_interface: dict = {}
        # A start may leave partially acquired resources until rollback succeeds.
        self._wiring_transition_unknown = False

    def get_tools(self) -> list:
        return [controlled_semantic_spatial_tool_definition(self._namespace)]

    def dispatch(self, name: str, args: dict) -> dict | None:
        if name != self.PREFIX:
            return None
        if not isinstance(args, dict):
            return self._error("invalid_argument", "arguments must be an object")
        action = args.get("action")
        if action == "info":
            return self._info(args.get("instance_id"))
        if action == "config":
            return self._configure(args)
        if action == "start":
            return self._start(args)
        if action == "stop":
            return self._stop()
        if action not in CONTROLLED_SEMANTIC_SPATIAL_ACTIONS:
            return self._error(
                "unsupported_action", f"unsupported navigation action: {action!r}"
            )
        with self._lock:
            if self._stop_requested.is_set():
                return self._error("navigation_stopping", "wait for stop to finish")
            if not self._started:
                return self._error(
                    "canvas_not_started",
                    "connect LiDAR and IMU, then start the "
                    "controlled_semantic_spatial card",
                )
            if action in SEMANTIC_ACTIONS and not self._external_wiring.get("rgb"):
                return self._error(
                    "rgb_not_connected",
                    "stop the card, connect the RGB input, then restart before "
                    "using capture or navigate",
                )
            ticket = self._generation
            semantic_operation = action in SEMANTIC_ACTIONS
            mapping_operation = action in FAST_LIVO2_ACTIONS
            if semantic_operation:
                self._inflight_semantic += 1
            if mapping_operation:
                self._inflight_mapping += 1
        try:
            if not self._runtime.info().get("running", False):
                return self._error(
                    "navigation_runtime_unavailable",
                    "an owned controlled_semantic_spatial child process is not running; "
                    "stop and restart the card",
                )
            if mapping_operation:
                with self._lock:
                    if (ticket is not self._generation or self._stop_requested.is_set()
                            or not self._started):
                        return self._error("navigation_stopping", "navigation admission expired")
                return self._mapping.dispatch("fast_livo2", args)
            if action in NAV2_ACTIONS:
                return self._dispatch_planning(ticket, args)
            if action == "navigate":
                navigate = getattr(self._semantic, "navigate_with_ticket", None)
                if not callable(navigate):
                    return self._error(
                        "semantic_admission_unavailable",
                        "semantic navigation does not support lifecycle tickets",
                    )
                return navigate(args, ticket)
            return self._semantic.dispatch("vln", args)
        finally:
            if semantic_operation or mapping_operation:
                with self._lock:
                    if semantic_operation:
                        self._inflight_semantic -= 1
                    if mapping_operation:
                        self._inflight_mapping -= 1

    def _dispatch_planning(self, ticket, args: dict) -> dict:
        with self._lock:
            if (ticket is None or ticket is not self._generation
                    or self._stop_requested.is_set() or not self._started):
                return self._error("navigation_stopping", "navigation admission expired")
            self._inflight_planning += 1
        try:
            return self._planning.dispatch("nav2", args)
        finally:
            with self._lock:
                self._inflight_planning -= 1

    def stop(self) -> dict:
        return self._stop()

    def close(self) -> None:
        self._stop()

    def _start(self, args: dict) -> dict:
        if self._stop_requested.is_set():
            return self._error("navigation_start_cancelled", "stop was requested")
        with self._transition_lock:
            return self._start_serialized(args)

    def _start_serialized(self, args: dict) -> dict:
        # Legacy calls without an ID own the reserved default instance. An
        # explicit default ID refers to that same owner, but omission never
        # aliases a different card. A retry must repeat its actual wiring.
        requested_id = str(args.get("instance_id") or "default").strip() or "default"
        descriptor = args.get("control_interface", {})
        if not isinstance(descriptor, dict):
            return self._error("invalid_control_interface", "control_interface must be an object")
        descriptor = deepcopy(descriptor)
        if descriptor:
            capabilities = {
                "control_mode": "twist", "action_dim": 6,
                "control_hz": 10.0, "chunk_size": 1,
            }
            try:
                problems = negotiate.check(capabilities, descriptor, label="导航")
            except (TypeError, ValueError, OverflowError, AttributeError) as error:
                return self._error("invalid_control_interface", str(error))
            if descriptor.get("mode") != "twist":
                problems.append("下游 mode 必须是 twist")
            if problems:
                return self._error("incompatible_control_interface", "；".join(problems))
        with self._lock:
            if self._stop_requested.is_set():
                return self._error("navigation_start_cancelled", "stop was requested")
            already_started = self._started
            if already_started and requested_id != self._instance_id:
                return self._error(
                    "navigation_instance_conflict",
                    "navigation is owned by another instance; stop it before starting "
                    "a different instance",
                    instance_id=self._instance_id,
                )
            if self._config_error and not already_started:
                return self._error("invalid_config", self._config_error)
        wiring = self._validate_external_wiring(args)
        with self._lock:
            # Stop signals before acquiring the transition lock, including
            # while validation is inspecting the mapping configuration.
            if self._stop_requested.is_set():
                return self._error("navigation_start_cancelled", "stop was requested")
            if wiring.get("status") == "error":
                return wiring
            if already_started:
                if descriptor != self._control_interface:
                    return self._error(
                        "navigation_control_interface_conflict",
                        "navigation is already started with a different control "
                        "interface; stop it before reconnecting the output",
                        instance_id=self._instance_id,
                    )
                if wiring["wired_topics"] != self._external_wiring:
                    return self._error(
                        "navigation_wiring_conflict",
                        "navigation is already started with different wiring; "
                        "stop it before reconnecting inputs",
                        instance_id=self._instance_id,
                    )
                result = self._info(self._instance_id)
                result["already_started"] = True
                return result
            # Mark the acquisition before letting stop/info observe it.
            self._wiring_transition_unknown = True

        started: list[tuple[str, object]] = []
        generation = object()
        try:
            started.append(("runtime", self._runtime))
            runtime_result = self._runtime.start(
                namespace=self._namespace,
                input_topics=wiring["wired_topics"],
                control_interface=descriptor,
            )
            if runtime_result.get("state") != "running":
                raise RuntimeError("navigation child processes did not stay running")
            self._check_start_cancelled()

            started.append(("mapping", self._mapping))
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
            self._check_start_cancelled()

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
            started.append(("planning", self._planning))
            planning_result = self._planning.dispatch("nav2", planning_start)
            if self._is_transient_planning_discovery_failure(planning_result):
                self._check_start_cancelled()
                if not self._runtime.info().get("running", False):
                    raise RuntimeError(
                        "planning start failed after a navigation child process exited"
                    )
                log.warning(
                    "[controlled_semantic_spatial] Nav2 command subscriber was not "
                    "discovered on the first attempt; rebuilding only the planning "
                    "bridge before rollback"
                )
                # Wake promptly for stop and recheck immediately before retry.
                # A stop racing after this check still requires normal rollback.
                self._stop_requested.wait(0.25)
                self._check_start_cancelled()
                planning_result = self._planning.dispatch("nav2", planning_start)
            self._require_started("planning", planning_result)
            self._check_start_cancelled()

            rgb_topic = wiring["wired_topics"].get("rgb")
            if rgb_topic:
                started.append(("semantic", self._semantic))
                semantic_result = self._semantic.dispatch(
                    "vln",
                    {
                        "action": "start",
                        "input_bindings": [
                            {"port": "rgb", "topic": rgb_topic},
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
                self._check_start_cancelled()
            with self._lock:
                self._check_start_cancelled()
                self._started = True
                self._generation = generation
                self._instance_id = requested_id
                self._external_wiring = dict(wiring["wired_topics"])
                self._control_interface = descriptor
                self._wiring_transition_unknown = False
        except Exception as exc:
            log.error(
                "[controlled_semantic_spatial] start failed: %s",
                exc,
                exc_info=True,
            )
            cancelled = self._stop_requested.is_set()
            cleanup, cleanup_confirmed = self._stop_started(started)
            if cleanup_confirmed:
                with self._lock:
                    self._wiring_transition_unknown = False
            else:
                with self._lock:
                    self._stop_requested.set()
            return self._error(
                "navigation_start_cancelled" if cancelled else "navigation_start_failed",
                str(exc), cleanup=cleanup, cleanup_confirmed=cleanup_confirmed,
            )

        result = self._info(self._instance_id)
        result.update({"state": "ready", "status": "ready"})
        return result

    def _check_start_cancelled(self) -> None:
        if self._stop_requested.is_set():
            raise RuntimeError("navigation start cancelled by stop")

    def _stop(self) -> dict:
        # Signal before taking the transition lock; never tear down a live request.
        with self._lock:
            self._pending_stops += 1
            self._stop_requested.set()
        with self._transition_lock:
            confirmed = False
            try:
                result = self._stop_serialized()
                confirmed = self._stop_confirmed(result) or (
                    isinstance(result, dict)
                    and result.get("error_code") == "navigation_stop_failed"
                    and result.get("terminal_confirmed") is True
                    and result.get("canvas_wired") is False
                )
                return result
            finally:
                # Keep result accounting inside the transition: a later stop's
                # pending result must not be cleared by an earlier success.
                with self._lock:
                    self._pending_stops -= 1
                    if confirmed and self._pending_stops == 0:
                        self._stop_requested.clear()

    def _stop_serialized(self) -> dict:
        results = {}
        # A request admitted before stop may not yet have entered the planner.
        # Stop it promptly, but never tear down until every such call has
        # returned and a *subsequent* planner stop is confirmed.
        with self._lock:
            planning_was_inflight = self._inflight_planning > 0
            semantic_was_inflight = self._inflight_semantic > 0
            mapping_was_inflight = self._inflight_mapping > 0
        try:
            if planning_was_inflight or semantic_was_inflight or mapping_was_inflight:
                cancel = getattr(self._planning, "stop_active_only", None)
                if callable(cancel):
                    results["planning"] = cancel()
                else:
                    results["planning"] = self._error(
                        "navigation_stop_pending",
                        "planner does not support non-releasing cancellation",
                    )
            else:
                results["planning"] = self._planning.dispatch("nav2", {"action": "stop"})
        except Exception as exc:
            results["planning"] = self._error("component_stop_failed", str(exc))
        with self._lock:
            planning_inflight = self._inflight_planning > 0
            semantic_inflight = self._inflight_semantic > 0
            mapping_inflight = self._inflight_mapping > 0
        if (planning_was_inflight or semantic_was_inflight or mapping_was_inflight
                or planning_inflight or semantic_inflight or mapping_inflight):
            return self._error(
                "navigation_stop_pending",
                "navigation requests are still settling; retry stop",
                component_results=results, retryable=True, canvas_wired=True,
            )

        planning_result = results["planning"]
        if not self._stop_confirmed(planning_result):
            return self._error(
                "navigation_stop_pending",
                "active navigation stop is unconfirmed; retry stop without "
                "restarting controlled_semantic_spatial",
                component_results=results,
                retryable=True,
                canvas_wired=True,
            )

        try:
            results["semantic"] = self._semantic.dispatch("vln", {"action": "stop"})
        except Exception as exc:
            results["semantic"] = self._error("component_stop_failed", str(exc))
        semantic_result = results["semantic"]
        if not self._stop_confirmed(semantic_result):
            return self._error(
                "semantic_stop_pending",
                "semantic navigation stop is unconfirmed; retry stop without "
                "restarting controlled_semantic_spatial",
                component_results=results,
                retryable=True,
                canvas_wired=True,
            )

        # Mapping must persist its map transaction before its runtime and
        # Canvas lifecycle can be released. Any non-terminal response is
        # retryable here, regardless of how the child classified the error.
        try:
            results["mapping"] = self._mapping.dispatch(
                "fast_livo2", {"action": "stop"}
            )
        except Exception as exc:
            results["mapping"] = self._error("component_stop_failed", str(exc))
        mapping_result = results["mapping"]
        if not self._mapping_stop_confirmed(mapping_result):
            return self._error(
                "navigation_stop_pending",
                "mapping finalization is unconfirmed; retry stop without restarting "
                "controlled_semantic_spatial",
                component_results=results,
                retryable=True,
                canvas_wired=True,
            )

        if (isinstance(mapping_result, dict)
                and mapping_result.get("status") == "error"):
            with self._lock:
                self._last_terminal_stop_error = dict(mapping_result)
                self._pending_terminal_stop_error = dict(mapping_result)
        try:
            results["runtime"] = self._runtime.stop()
        except Exception as exc:
            results["runtime"] = self._error("runtime_stop_failed", str(exc))
            return self._error(
                "navigation_stop_pending",
                "runtime shutdown is unconfirmed; retry stop without restarting "
                "controlled_semantic_spatial",
                component_results=results,
                retryable=True,
                canvas_wired=True,
            )
        if not self._runtime_stop_confirmed(results["runtime"]):
            return self._error(
                "navigation_stop_pending", "runtime shutdown is unconfirmed; retry stop",
                component_results=results, retryable=True, canvas_wired=True,
            )
        with self._lock:
            self._started = False
            self._generation = None
            self._instance_id = ""
            self._external_wiring = {}
            self._control_interface = {}
            self._wiring_transition_unknown = False
            terminal_mapping_error = self._pending_terminal_stop_error
            self._pending_terminal_stop_error = None
        if terminal_mapping_error is not None:
            results["mapping"] = terminal_mapping_error
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
                terminal_confirmed=True,
                canvas_wired=False,
            )
        return {
            "state": "idle",
            "status": "idle",
            "canvas_wired": False,
            "component_results": results,
            "shadow_only": False,
            "physical_execution": False,
            "control_output_can_drive_hardware": True,
        }

    @classmethod
    def _runtime_stop_confirmed(cls, result) -> bool:
        if not cls._stop_confirmed(result):
            return False
        children = result.get("children")
        if children is None:
            # Compatibility with injected runtimes which expose only a summary.
            return result.get("running") is False
        return (isinstance(children, list)
                and all(isinstance(child, dict) and child.get("running") is False
                        for child in children))

    @classmethod
    def _mapping_stop_confirmed(cls, result) -> bool:
        # Mapping's protocol separately reports a failed map transaction whose
        # resources have nevertheless reached a confirmed terminal state.
        return cls._stop_confirmed(result) or (
            isinstance(result, dict)
            and result.get("state") == "error"
            and result.get("status") == "error"
            and result.get("terminal_confirmed") is True
            and result.get("canvas_wired") is not True
            and result.get("retryable") is not True
        )

    @staticmethod
    def _stop_confirmed(result) -> bool:
        if not isinstance(result, dict) or result.get("ok") is False:
            return False
        if result.get("error") or result.get("error_code"):
            return False
        statuses = [result[key] for key in ("status", "state") if result.get(key) is not None]
        if not statuses or any(status not in {"idle", "stopped", "disabled"} for status in statuses):
            return False
        return result.get("terminal_confirmed") is not False

    def _configure(self, args: dict) -> dict:
        with self._transition_lock:
            return self._configure_serialized(args)

    @staticmethod
    def _component_config(component, prefix: str, validator) -> dict:
        info = component.dispatch(prefix, {"action": "info"})
        current = info.get("config") if isinstance(info, dict) else None
        return validator(dict(current) if isinstance(current, dict) else {}, {})

    @staticmethod
    def _config_failed(result) -> bool:
        return (
            not isinstance(result, dict)
            or result.get("status") in {"error", "invalid_config"}
            or result.get("state") == "error"
        )

    def _configure_serialized(self, args: dict) -> dict:
        with self._lock:
            if self._started:
                return self._error(
                    "config_while_running",
                    "stop controlled_semantic_spatial before changing config",
                )
        updates = {
            key: value
            for key, value in args.items()
            if key not in {"action", "instance_id"}
        }
        unknown = sorted(
            set(updates) - set(CONTROLLED_SEMANTIC_SPATIAL_CONFIG_SCHEMA["properties"])
        )
        if unknown:
            return self._error(
                "invalid_config",
                "unsupported navigation config fields: " + ",".join(unknown),
            )
        mapping_updates = {
            "action": "config",
            **{
                target: updates[source]
                for source, target in (
                    ("mapping_request_timeout_sec", "request_timeout_sec"),
                    ("mapping_discovery_timeout_sec", "discovery_timeout_sec"),
                    ("lidar_qos_depth", "lidar_qos_depth"),
                    ("imu_qos_depth", "imu_qos_depth"),
                    ("map_voxel_size_m", "map_voxel_size_m"),
                    ("obstacle_min_height_m", "obstacle_min_height_m"),
                    ("obstacle_max_height_m", "obstacle_max_height_m"),
                    ("map_view_enabled", "map_view_enabled"),
                    ("fault_capture_enabled", "fault_capture_enabled"),
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
                )
                if source in updates
            },
        }
        if "rotate_speed_rps" in updates:
            planning_updates.update(
                min_yaw_rps=updates["rotate_speed_rps"],
                max_yaw_rps=updates["rotate_speed_rps"],
            )
        semantic_keys = {
            "vlm_base_url": "base_url",
            "vlm_api_key": "api_key",
            "vlm_model": "model",
            "vlm_timeout_sec": "timeout_sec",
        }
        semantic_updates = None
        if any(key in updates for key in semantic_keys):
            semantic_updates = {
                "action": "config",
                **{
                    target: updates[source]
                    for source, target in semantic_keys.items()
                    if source in updates
                },
            }

        try:
            mapping_previous = self._component_config(
                self._mapping, "fast_livo2", validated_mapping_config
            )
            planning_previous = self._component_config(
                self._planning, "nav2", validated_planning_config
            )
            validated_mapping_config(
                mapping_previous,
                {key: value for key, value in mapping_updates.items() if key != "action"},
            )
            validated_planning_config(
                planning_previous,
                {key: value for key, value in planning_updates.items() if key != "action"},
            )
        except (MappingConfigError, PlanningConfigError, TypeError, ValueError) as exc:
            return self._error("invalid_config", str(exc))

        components = [
            (
                "mapping",
                self._mapping,
                "fast_livo2",
                mapping_updates,
                mapping_previous,
            ),
            (
                "planning",
                self._planning,
                "nav2",
                planning_updates,
                planning_previous,
            ),
        ]
        if semantic_updates is not None:
            components.append(
                ("semantic", self._semantic, "vln", semantic_updates, None)
            )

        results = {}
        applied = []
        for name, component, prefix, request, previous in components:
            try:
                result = component.dispatch(prefix, request)
            except Exception as exc:
                result = self._error("component_config_failed", str(exc))
            results[name] = result
            if self._config_failed(result):
                rollback_results = {}
                rollback_failures = []
                for (
                    applied_name,
                    applied_component,
                    applied_prefix,
                    _applied_request,
                    applied_previous,
                ) in reversed(applied):
                    try:
                        rollback = applied_component.dispatch(
                            applied_prefix,
                            {"action": "config", **applied_previous},
                        )
                    except Exception as exc:
                        rollback = self._error("config_rollback_failed", str(exc))
                    rollback_results[applied_name] = rollback
                    if self._config_failed(rollback):
                        rollback_failures.append(applied_name)
                if rollback_failures:
                    return self._error(
                        "config_rollback_failed",
                        "component config failed and rollback failed: "
                        + ",".join(rollback_failures),
                        component_results=results,
                        rollback_component_results=rollback_results,
                    )
                return self._error(
                    "invalid_config",
                    "invalid component config: " + name,
                    component_results=results,
                    rollback_component_results=rollback_results,
                )
            applied.append((name, component, prefix, request, previous))
        return {
            "state": "configured",
            "status": "configured",
            "component_results": results,
            "takes_effect": "next_start",
            "shadow_only": False,
            "physical_execution": False,
            "control_output_can_drive_hardware": True,
        }

    def _validate_external_wiring(self, args: dict) -> dict:
        tool = self.get_tools()[0]
        expected = {item["port"]: item["topic"] for item in tool["topic_in"]}
        required = {
            item["port"]
            for item in tool["topic_in"]
            if item["required"]
        }
        descriptors = args.get("input_descriptors")
        if not isinstance(descriptors, list):
            return self._error("invalid_canvas_wiring", "live input_descriptors are required; reconnect inputs")
        wired = {}
        for binding in descriptors:
            if not isinstance(binding, dict):
                return self._error("invalid_canvas_wiring", "invalid input descriptor")
            port, topic, descriptor = binding.get("port"), binding.get("topic"), binding.get("descriptor")
            if (not isinstance(port, str) or port not in expected or port in wired
                    or not isinstance(topic, str) or not isinstance(descriptor, dict)
                    or descriptor.get("topic") != topic
                    or descriptor.get("format") != next(item["format"] for item in tool["topic_in"] if item["port"] == port)):
                return self._error("invalid_canvas_wiring", "conflicting input descriptor")
            wired[port] = topic
            if port == "lidar":
                imu = descriptor.get("associated_imu")
                if (descriptor.get("ros_type") != "sensor_msgs/msg/PointCloud2"
                        or not isinstance(imu, dict)
                        or imu.get("format") != "sensor/imu"
                        or imu.get("ros_type") != "sensor_msgs/msg/Imu"
                        or not isinstance(imu.get("topic"), str)
                        or imu.get("topic") == topic):
                    return self._error("invalid_canvas_wiring", "lidar descriptor requires associated MID360 IMU")
                wired["imu"] = imu["topic"]
        missing = sorted(port for port in required if not wired.get(port))
        invalid = sorted(port for port, topic in wired.items()
                         if not isinstance(topic, str) or len(topic) > 255
                         or _ABSOLUTE_ROS_TOPIC_RE.fullmatch(topic) is None)
        supplied_topics = args.get("input_topics", [])
        if (not isinstance(supplied_topics, list)
                or any(not isinstance(t, str) for t in supplied_topics)
                or len(set(supplied_topics)) != len(supplied_topics)):
            return self._error("invalid_canvas_wiring", "input_topics must be a unique array of strings")
        if "input_topic" in args:
            if not isinstance(args["input_topic"], str):
                return self._error("invalid_canvas_wiring", "input_topic must be a string")
            if args["input_topic"]:
                supplied_topics = [*supplied_topics, args["input_topic"]]
        external_topics = {topic for port, topic in wired.items() if port != "imu"}
        if (missing or invalid or len(set(wired.values())) != len(wired)
                or (supplied_topics and set(supplied_topics) != external_topics)):
            return self._error("invalid_canvas_wiring", "missing, invalid or conflicting input topics")
        if "input_bindings" in args:
            bindings = args["input_bindings"]
            if (not isinstance(bindings, list)
                    or any(not isinstance(b, dict) or not isinstance(b.get("port"), str)
                           or not isinstance(b.get("topic"), str) for b in bindings)
                    or len(bindings) != len(external_topics)
                    or {b["port"]: b["topic"] for b in bindings}
                    != {port: topic for port, topic in wired.items() if port != "imu"}):
                return self._error("invalid_canvas_wiring", "input bindings conflict with descriptors")
        return {"wired_topics": wired}

    def _handle_semantic_goal(
        self, goal: dict, *, control_nav_id=None, admission_ticket=None
    ) -> dict:
        request = {
            "action": "navigate_to_pose",
            "x": goal["x"],
            "y": goal["y"],
            "yaw": goal["yaw"],
            "speed": goal["speed"],
        }
        if control_nav_id is not None:
            request["_control_nav_id"] = control_nav_id
        return self._dispatch_planning(admission_ticket, request)

    def _info(self, requested_instance_id=None) -> dict:
        tool = self.get_tools()[0]
        with self._lock:
            started = self._started
            instance_id = self._instance_id
            wired = dict(self._external_wiring)
            wiring_unknown = self._wiring_transition_unknown
            config_error = self._config_error
            last_terminal_stop_error = self._last_terminal_stop_error
        discovery = requested_instance_id is None or requested_instance_id == ""
        instance_match = bool(started and instance_id and requested_instance_id == instance_id)
        if discovery:
            # Agent Core heartbeat back-fills tool definitions from this view.
            # Keep every declared topic even if the corresponding port is idle.
            topic_in = [
                {
                    **item,
                    "connected": (
                        None if wiring_unknown else bool(wired.get(item["port"]))
                    ),
                }
                for item in tool["topic_in"]
            ]
        else:
            # Uncommitted startup/rollback cannot prove a port is disconnected.
            # Do not trust bindings supplied by the info request itself.
            topic_in = [
                {
                    **item,
                    "topic": wired.get(item["port"], "") if instance_match else "",
                    "connected": (
                        bool(wired.get(item["port"])) if instance_match
                        else None if wiring_unknown else False
                    ),
                }
                for item in tool["topic_in"]
            ]
        runtime_info = self._runtime.info()
        runtime_failed = started and not runtime_info.get("running", False)
        state = (
            "stopping"
            if self._stop_requested.is_set()
            else "error"
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
            "instance_match": None if discovery else instance_match,
            "topic_in_scope": "declaration" if discovery else "instance",
            "topic_in_binding_state": (
                "unknown" if wiring_unknown
                else "known" if instance_match or (discovery and started)
                else "none"
            ),
            "last_terminal_stop_error": last_terminal_stop_error,
            "runtime": runtime_info,
            "mapping": self._mapping.dispatch("fast_livo2", {"action": "info"}),
            "planning": self._planning.dispatch("nav2", {"action": "info"}),
            "semantic": self._semantic.dispatch("vln", {"action": "info"}),
            "topic_in": topic_in,
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
                "an owned controlled_semantic_spatial child process exited"
                if runtime_failed
                else None
            ),
            "shadow_only": False,
            "physical_execution": False,
            "control_output_can_drive_hardware": True,
        }

    def _stop_started(self, started: list[tuple[str, object]]) -> tuple[dict, bool]:
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
            if name == "mapping":
                confirmed = self._mapping_stop_confirmed(results[name])
                if (confirmed and isinstance(results[name], dict)
                        and results[name].get("status") == "error"):
                    with self._lock:
                        self._last_terminal_stop_error = dict(results[name])
            elif name == "runtime":
                confirmed = self._runtime_stop_confirmed(results[name])
            else:
                confirmed = self._stop_confirmed(results[name])
            if not confirmed:
                return results, False
        return results, True

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
            "shadow_only": False,
            "physical_execution": False,
            "control_output_can_drive_hardware": True,
        }


__all__ = ["ControlledSemanticSpatialPlugin"]
