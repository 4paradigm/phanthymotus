"""MCP card adapter for a single ROS 2 bag recording session."""

from __future__ import annotations

import threading

from .recorder import Recorder, validate_config
from .store import BagStore


class RosbagRecorderPlugin:
    PREFIX = "rosbag_recorder"

    def __init__(self, cfg: dict, executor=None):
        self._lock = threading.RLock()
        self._config_error = None
        self._topics = cfg.get("topics", [])
        self._max_duration_sec = cfg.get("max_duration_sec", 3600)
        self._min_free_bytes = cfg.get("min_free_bytes", 1073741824)
        root = cfg.get("output_root", "/bags")
        self._store = BagStore(root)
        self._recorder = Recorder(
            root,
            ros2_executable=cfg.get("ros2_executable", "ros2"),
        )

    def get_tools(self) -> list:
        return [{
            "name": self.PREFIX,
            "type": "processor",
            "multiInstance": False,
            "description": "录制指定 ROS 2 话题。启动画布即开始录制，停止画布结束录制；不提供回放。",
            "inputSchema": {
                "type": "object",
                "properties": {"action": {"type": "string", "enum": ["start", "stop", "info", "config"]}},
                "required": ["action"],
                "x-rosbag-manager": {"version": 1},
                # A recorder that already ended is not ready to record.
                "x-start": {"failed_states": ["completed"]},
                # Graceful SIGINT (15s) plus a bounded kill/reap (5s) can
                # legitimately outlast the platform's normal stop deadline.
                "x-stop": {"timeout_s": 25},
                "x-action-params": {
                    "info": {"params": [], "description": "查看录制状态与最近一次录制结果"},
                    "config": {"params": [], "description": "更新录制配置"},
                },
            },
            "configSchema": {
                "type": "object",
                "properties": {
                    "topics": {"type": "array", "items": {"type": "string"},
                               "description": "要录制的完整 ROS 2 话题名称", "scope": "shared"},
                    "max_duration_sec": {"type": "number", "minimum": 0, "default": 3600,
                                         "description": "录制时长上限（秒）；0 表示不限制", "scope": "shared"},
                    "min_free_bytes": {"type": "integer", "minimum": 0, "default": 1073741824,
                                       "description": "停止录制的剩余磁盘空间阈值（字节）；0 表示不限制", "scope": "shared"},
                },
                "required": ["topics"],
            },
        }]

    @staticmethod
    def _error(message: str) -> dict:
        return {"state": "error", "message": message}

    def dispatch(self, name: str, args: dict | None) -> dict | None:
        args = args or {}
        action = args.get("action") or name
        with self._lock:
            if action == "start":
                if self._config_error:
                    return self._error(self._config_error)
                return self._recorder.start(
                    self._topics, max_duration_sec=self._max_duration_sec,
                    min_free_bytes=self._min_free_bytes,
                )
            if action == "stop":
                return self._with_result(self._recorder.stop())
            if action == "info":
                snapshot = self._recorder.info()
                view = args.get("view") or "status"
                active_id = self._active_id(snapshot)
                try:
                    if view == "list":
                        return self._store.list_bags(active_id=active_id,
                                                     offset=args.get("offset", 0))
                    if view == "inspect":
                        bag_id = args.get("bag_id")
                        expected = snapshot.get("topics") if self._snapshot_id(snapshot) == bag_id else None
                        return self._store.inspect(bag_id, expected_topics=expected,
                                                   active=(bag_id == active_id))
                    if view == "status":
                        return self._with_result(snapshot)
                    return self._error("不支持的查询")
                except (OSError, ValueError) as exc:
                    return self._error(str(exc))
            if action == "config":
                return self._config(args)
        return None

    @staticmethod
    def _snapshot_id(snapshot):
        path = snapshot.get("path")
        if not path:
            return None
        from pathlib import Path
        return Path(path).name

    def _active_id(self, snapshot):
        if snapshot.get("process_active", snapshot.get("state") in ("loading", "recording", "stopping")):
            return self._snapshot_id(snapshot)
        return None

    def _with_result(self, snapshot):
        bag_id = self._snapshot_id(snapshot)
        if bag_id and not self._active_id(snapshot):
            try:
                snapshot["result_confirmation"] = self._store.inspect(
                    bag_id, expected_topics=snapshot.get("topics"))
            except (OSError, ValueError) as exc:
                snapshot["result_confirmation"] = {
                    "metadata_status": "unavailable", "warnings": [str(exc)]}
        return snapshot

    def open_recording_file(self, bag_id, filename):
        """Open an existing bag file for the local Agent Core download proxy."""
        with self._lock:
            snapshot = self._recorder.info()
            return self._store.open_file(bag_id, filename,
                                         active=(bag_id == self._active_id(snapshot)))

    def _config(self, args: dict) -> dict:
        snapshot = self._recorder.info()
        if snapshot.get("process_active", snapshot.get("state") in ("loading", "recording", "stopping")):
            return self._error("录制进行中不能更改配置")
        topics = args.get("topics", self._topics)
        duration = args.get("max_duration_sec", self._max_duration_sec)
        min_free = args.get("min_free_bytes", self._min_free_bytes)
        try:
            validate_config(topics, duration, min_free)
        except (ValueError, TypeError) as exc:
            self._config_error = str(exc)
            return self._error(self._config_error)
        self._topics = list(topics)
        self._max_duration_sec = duration
        self._min_free_bytes = min_free
        self._config_error = None
        return {"state": "ok", "topics": self._topics,
                "max_duration_sec": duration, "min_free_bytes": min_free}

    def shutdown(self) -> None:
        with self._lock:
            self._recorder.shutdown()
