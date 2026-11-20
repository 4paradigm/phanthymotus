"""MCP card adapter for a single ROS 2 bag recording session."""

from __future__ import annotations

import math
import re
import threading

from .recorder import Recorder


class RosbagRecorderPlugin:
    PREFIX = "rosbag_recorder"

    def __init__(self, cfg: dict, executor=None):
        self._lock = threading.RLock()
        self._config_error = None
        self._topics = cfg.get("topics", [])
        self._max_duration_sec = cfg.get("max_duration_sec", 3600)
        self._min_free_bytes = cfg.get("min_free_bytes", 1073741824)
        self._recorder = Recorder(
            cfg.get("output_root", "/bags"),
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
                try:
                    self._validate(self._topics, self._max_duration_sec, self._min_free_bytes)
                    return self._recorder.start(
                        list(self._topics), max_duration_sec=self._max_duration_sec,
                        min_free_bytes=self._min_free_bytes,
                    )
                except (ValueError, TypeError, OSError) as exc:
                    return self._error(str(exc))
            if action == "stop":
                return self._recorder.stop()
            if action == "info":
                return self._recorder.info()
            if action == "config":
                return self._config(args)
        return None

    @staticmethod
    def _validate(topics, duration, min_free):
        if not isinstance(topics, list) or not topics or any(
            not isinstance(t, str) or not re.fullmatch(r"/[A-Za-z0-9_~/]+", t)
            for t in topics
        ):
            raise ValueError("topics 必须是非空的绝对 ROS 2 话题名称列表")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            raise ValueError("max_duration_sec 必须是非负有限数字")
        if isinstance(min_free, bool) or not isinstance(min_free, int) or min_free < 0:
            raise ValueError("min_free_bytes 必须是非负整数")

    def _config(self, args: dict) -> dict:
        if self._recorder.info().get("state") in ("recording", "stopping"):
            return self._error("录制进行中不能更改配置")
        topics = args.get("topics", self._topics)
        duration = args.get("max_duration_sec", self._max_duration_sec)
        min_free = args.get("min_free_bytes", self._min_free_bytes)
        try:
            self._validate(topics, duration, min_free)
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
