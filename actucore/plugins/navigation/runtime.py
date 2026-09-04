"""Own the ROS navigation runtime inside the ActuCore container.

The public ControlledSemanticSpatial card is the only lifecycle owner.
FAST-LIVO2 adapters and Nav2 are regular child process groups in the same
container; Docker is not used or required at runtime.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable


log = logging.getLogger(__name__)


_DESCENDANT_SIGINT_GRACE_SEC = 0.25
_DESCENDANT_SIGTERM_TIMEOUT_SEC = 2.0
_DESCENDANT_SIGKILL_TIMEOUT_SEC = 2.0
_DESCENDANT_MONITOR_INTERVAL_SEC = 0.05


@dataclass
class _Child:
    name: str
    command: tuple[str, ...]
    process: subprocess.Popen | None = None
    last_return_code: int | None = None
    independent_groups: dict[int, _OwnedProcessGroup] = field(
        default_factory=dict, repr=False
    )
    independent_groups_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False
    )
    monitor_stop: threading.Event | None = field(default=None, repr=False)
    monitor_thread: threading.Thread | None = field(default=None, repr=False)


@dataclass(frozen=True)
class _OwnedProcessGroup:
    group_id: int
    leader_start_time: str


class NavigationRuntime:
    """Start and stop the in-container FAST-LIVO2 and Nav2 ROS launch files."""

    def __init__(
        self,
        *,
        popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
        stop_timeout_sec: float = 10.0,
        startup_grace_sec: float = 0.25,
    ):
        self._popen_factory = popen_factory
        self._stop_timeout_sec = float(stop_timeout_sec)
        self._startup_grace_sec = float(startup_grace_sec)
        self._lock = threading.RLock()
        self._children = [
            _Child(
                "fast_livo2",
                (
                    "ros2",
                    "launch",
                    "fast_livo2",
                    "fast_livo2.launch.py",
                ),
            ),
            _Child(
                "nav2",
                ("ros2", "launch", "nav2", "nav2.launch.py"),
            ),
        ]

    def start(
        self,
        *,
        namespace: str = "ubuntu",
        input_topics: dict[str, str] | None = None,
    ) -> dict:
        with self._lock:
            running = [child for child in self._children if self._is_running(child)]
            if len(running) == len(self._children):
                return self.info(already_started=True)
            if any(child.process is not None for child in self._children):
                self._stop_locked()

            started: list[_Child] = []
            try:
                launch_arguments = self._launch_arguments(
                    namespace, input_topics or {}
                )
                for child in self._children:
                    command = [
                        *child.command,
                        *launch_arguments.get(child.name, ()),
                    ]
                    child.process = self._popen_factory(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=None,
                        stderr=None,
                        env=os.environ.copy(),
                        start_new_session=True,
                    )
                    child.last_return_code = None
                    with child.independent_groups_lock:
                        child.independent_groups.clear()
                    self._start_descendant_monitor(child)
                    started.append(child)
                    if self._startup_grace_sec > 0:
                        time.sleep(self._startup_grace_sec)
                    return_code = child.process.poll()
                    if return_code is not None:
                        child.last_return_code = int(return_code)
                        raise RuntimeError(
                            f"{child.name} runtime exited during startup "
                            f"with code {return_code}"
                        )
            except Exception:
                self._terminate(started)
                raise
            return self.info()

    @staticmethod
    def _launch_arguments(
        namespace: str, input_topics: dict[str, str]
    ) -> dict[str, tuple[str, ...]]:
        root = "/" + str(namespace or "ubuntu").strip("/")
        navigation = f"{root}/navigation"
        fast_livo2 = f"{navigation}/fast_livo2"
        nav2 = f"{navigation}/nav2"

        def topic(port: str, default: str) -> str:
            return str(input_topics.get(port) or default).strip()

        fast_values = {
            "lidar_topic": topic("lidar", f"{navigation}/lidar"),
            "imu_topic": topic("imu", f"{navigation}/imu"),
            "rgb_topic": topic("rgb", f"{root}/camera/rgb_frame"),
            "depth_topic": topic(
                "depth_frame", f"{root}/camera/depth_frame"
            ),
            "raw_odom_topic": f"{fast_livo2}/raw/odom",
            "raw_cloud_topic": f"{fast_livo2}/raw/cloud_registered",
            "odom_topic": f"{navigation}/odom",
            "cloud_topic": f"{navigation}/cloud_registered",
            "obstacle_map_topic": f"{navigation}/obstacle_map",
            "static_map_topic": f"{navigation}/static_map",
            "map_view_topic": f"{fast_livo2}/map_view",
            "diagnostics_topic": f"{fast_livo2}/diagnostics",
            "reset_topic": f"{fast_livo2}/reset_map",
            "map_control_topic": f"{fast_livo2}/map_control",
            "map_control_status_topic": f"{fast_livo2}/map_control_status",
            "command_topic": f"{fast_livo2}/command",
            "status_topic": f"{fast_livo2}/status",
            "collection_status_topic": f"{fast_livo2}/collection_status_raw",
        }
        nav2_values = {
            "odom_topic": f"{navigation}/odom",
            "obstacle_cloud_topic": f"{navigation}/cloud_registered",
            "cmd_vel_raw_topic": f"{nav2}/cmd_vel_raw",
            "velocity_proposal_topic": f"{nav2}/velocity_proposal",
            "command_topic": f"{nav2}/command",
            "status_topic": f"{nav2}/status",
            "segment_status_topic": f"{nav2}/segment_status",
            "speed_limit_topic": f"{nav2}/speed_limit",
        }
        return {
            "fast_livo2": tuple(
                f"{key}:={value}" for key, value in fast_values.items()
            ),
            "nav2": tuple(
                f"{key}:={value}" for key, value in nav2_values.items()
            ),
        }

    def stop(self) -> dict:
        with self._lock:
            return self._stop_locked()

    def _stop_locked(self) -> dict:
        self._terminate(list(reversed(self._children)))
        return self.info()

    def _terminate(self, children: list[_Child]) -> None:
        for child in children:
            process = child.process
            if process is None:
                continue
            self._remember_descendant_process_groups(child)
            self._stop_descendant_monitor(child)
            independent_groups = self._remaining_process_groups(
                self._remembered_process_groups(child)
            )
            root_running = process.poll() is None
            try:
                self._signal_process_groups(
                    independent_groups, signal.SIGINT
                )
                if root_running:
                    try:
                        os.killpg(process.pid, signal.SIGINT)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=self._stop_timeout_sec)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            process.wait(timeout=2.0)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            process.wait(timeout=2.0)
            finally:
                self._terminate_independent_process_groups(
                    independent_groups
                )
            return_code = process.poll()
            child.last_return_code = (
                int(return_code) if return_code is not None else None
            )
            child.process = None
            with child.independent_groups_lock:
                child.independent_groups.clear()

    def _start_descendant_monitor(self, child: _Child) -> None:
        stop_event = threading.Event()
        monitor = threading.Thread(
            target=self._monitor_descendant_process_groups,
            args=(child, stop_event),
            name=f"navigation-{child.name}-descendants",
            daemon=True,
        )
        child.monitor_stop = stop_event
        child.monitor_thread = monitor
        monitor.start()

    def _stop_descendant_monitor(self, child: _Child) -> None:
        stop_event = child.monitor_stop
        monitor = child.monitor_thread
        if stop_event is not None:
            stop_event.set()
        if monitor is not None and monitor is not threading.current_thread():
            monitor.join(timeout=max(0.2, _DESCENDANT_MONITOR_INTERVAL_SEC * 4))
            if monitor.is_alive():
                log.warning(
                    "descendant monitor did not stop promptly for %s", child.name
                )
        child.monitor_stop = None
        child.monitor_thread = None

    def _monitor_descendant_process_groups(
        self, child: _Child, stop_event: threading.Event
    ) -> None:
        while not stop_event.is_set():
            self._remember_descendant_process_groups(child)
            process = child.process
            if process is None or process.poll() is not None:
                return
            stop_event.wait(_DESCENDANT_MONITOR_INTERVAL_SEC)

    def _remember_descendant_process_groups(self, child: _Child) -> None:
        process = child.process
        if process is None:
            return
        groups = self._independent_descendant_process_groups(process.pid)
        if not groups:
            return
        with child.independent_groups_lock:
            for group in groups:
                child.independent_groups[group.group_id] = group

    @staticmethod
    def _remembered_process_groups(
        child: _Child,
    ) -> tuple[_OwnedProcessGroup, ...]:
        with child.independent_groups_lock:
            return tuple(
                child.independent_groups[group_id]
                for group_id in sorted(child.independent_groups)
            )

    @staticmethod
    def _independent_descendant_process_groups(
        root_pid: int,
    ) -> tuple[_OwnedProcessGroup, ...]:
        """Return descendant-led process groups that escape the launch group.

        FAST-LIVO2 currently creates its algorithm with ``start_new_session``.
        A signal to the ROS launch process group does not reach that nested
        session.  On Linux, ``/proc/.../children`` lets us find only descendants
        of our owned launch process before terminating it.  We signal only
        groups whose leader is itself such a descendant, never an arbitrary
        inherited process group.  Systems without procfs simply fall back to
        the normal launch-group cleanup.
        """
        descendants: set[int] = set()
        pending = [int(root_pid)]
        while pending:
            parent_pid = pending.pop()
            children_path = (
                f"/proc/{parent_pid}/task/{parent_pid}/children"
            )
            try:
                with open(children_path, encoding="ascii") as stream:
                    raw_children = stream.read().split()
            except OSError:
                continue
            for raw_pid in raw_children:
                try:
                    child_pid = int(raw_pid)
                except ValueError:
                    continue
                if child_pid > 0 and child_pid not in descendants:
                    descendants.add(child_pid)
                    pending.append(child_pid)
        groups: dict[int, _OwnedProcessGroup] = {}
        for pid in descendants:
            try:
                group_id = os.getpgid(pid)
            except OSError:
                continue
            if group_id in descendants and group_id != root_pid:
                start_time = NavigationRuntime._process_start_time(group_id)
                if start_time is not None:
                    groups[group_id] = _OwnedProcessGroup(
                        group_id=group_id,
                        leader_start_time=start_time,
                    )
        return tuple(groups[group_id] for group_id in sorted(groups))

    @staticmethod
    def _process_start_time(pid: int) -> str | None:
        """Read Linux's immutable process start tick for PID reuse checks."""
        try:
            with open(f"/proc/{pid}/stat", encoding="ascii") as stream:
                raw_stat = stream.read()
        except OSError:
            return None
        command_end = raw_stat.rfind(")")
        if command_end < 0:
            return None
        fields_after_command = raw_stat[command_end + 1 :].split()
        if len(fields_after_command) <= 19:
            return None
        if fields_after_command[0] == "Z":
            return None
        return fields_after_command[19]

    @classmethod
    def _remaining_process_groups(
        cls, groups: tuple[_OwnedProcessGroup, ...]
    ) -> tuple[_OwnedProcessGroup, ...]:
        remaining = []
        for group in groups:
            try:
                current_group_id = os.getpgid(group.group_id)
            except OSError:
                continue
            if current_group_id != group.group_id:
                continue
            if cls._process_start_time(group.group_id) != group.leader_start_time:
                continue
            remaining.append(group)
        return tuple(remaining)

    @classmethod
    def _wait_for_process_groups(
        cls,
        groups: tuple[_OwnedProcessGroup, ...],
        timeout_sec: float,
    ) -> tuple[_OwnedProcessGroup, ...]:
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        remaining = cls._remaining_process_groups(groups)
        while remaining:
            delay = deadline - time.monotonic()
            if delay <= 0:
                break
            time.sleep(min(0.05, delay))
            remaining = cls._remaining_process_groups(remaining)
        return remaining

    @classmethod
    def _terminate_independent_process_groups(
        cls, groups: tuple[_OwnedProcessGroup, ...]
    ) -> None:
        remaining = cls._wait_for_process_groups(
            groups, _DESCENDANT_SIGINT_GRACE_SEC
        )
        if not remaining:
            return
        cls._signal_process_groups(remaining, signal.SIGTERM)
        remaining = cls._wait_for_process_groups(
            remaining, _DESCENDANT_SIGTERM_TIMEOUT_SEC
        )
        if not remaining:
            return
        cls._signal_process_groups(remaining, signal.SIGKILL)
        remaining = cls._wait_for_process_groups(
            remaining, _DESCENDANT_SIGKILL_TIMEOUT_SEC
        )
        if remaining:
            log.warning(
                "owned descendant process groups did not exit after SIGKILL: %s",
                [group.group_id for group in remaining],
            )

    @staticmethod
    def _signal_process_groups(
        groups: tuple[_OwnedProcessGroup, ...], sig: int
    ) -> None:
        for group in groups:
            try:
                os.killpg(group.group_id, sig)
            except ProcessLookupError:
                continue

    def info(self, *, already_started: bool = False) -> dict:
        with self._lock:
            children = []
            all_running = True
            for child in self._children:
                running = self._is_running(child)
                all_running = all_running and running
                process = child.process
                children.append(
                    {
                        "name": child.name,
                        "running": running,
                        "pid": process.pid if running and process is not None else None,
                        "last_return_code": child.last_return_code,
                        "command": list(child.command),
                    }
                )
            return {
                "state": "running" if all_running else "idle",
                "running": all_running,
                "already_started": already_started,
                "container_model": "single_actucore_container",
                "docker_runtime_dependency": False,
                "children": children,
            }

    @staticmethod
    def _is_running(child: _Child) -> bool:
        process = child.process
        if process is None:
            return False
        return_code = process.poll()
        if return_code is None:
            return True
        child.last_return_code = int(return_code)
        return False


__all__ = ["NavigationRuntime"]
