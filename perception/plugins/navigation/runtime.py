"""Own the ROS navigation runtime inside the Perception container.

The public controlled_semantic_spatial card is the only lifecycle owner.
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
from dataclasses import dataclass
from typing import Callable


log = logging.getLogger(__name__)


@dataclass
class _Child:
    name: str
    command: tuple[str, ...]
    process: subprocess.Popen | None = None
    last_return_code: int | None = None


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
                    "g1_fast_livo2",
                    "g1_fast_livo2.launch.py",
                ),
            ),
            _Child(
                "nav2",
                ("ros2", "launch", "g1_nav2", "g1_nav2.launch.py"),
            ),
        ]

    def start(self) -> dict:
        with self._lock:
            running = [child for child in self._children if self._is_running(child)]
            if len(running) == len(self._children):
                return self.info(already_started=True)
            if running:
                self._stop_locked()

            started: list[_Child] = []
            try:
                for child in self._children:
                    child.process = self._popen_factory(
                        list(child.command),
                        stdin=subprocess.DEVNULL,
                        stdout=None,
                        stderr=None,
                        env=os.environ.copy(),
                        start_new_session=True,
                    )
                    child.last_return_code = None
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
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                    process.wait(timeout=self._stop_timeout_sec)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=2.0)
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait(timeout=2.0)
            return_code = process.poll()
            child.last_return_code = (
                int(return_code) if return_code is not None else None
            )
            child.process = None

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
                "container_model": "single_perception_container",
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
        child.process = None
        return False


__all__ = ["NavigationRuntime"]
