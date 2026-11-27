"""Read the recorder's own bag directories without loading bag payloads."""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path

import yaml


_BAG_ID = re.compile(r"bag_[0-9]{8}_[0-9]{6}_[0-9a-f]{12}\Z")
_MAX_METADATA_BYTES = 2 * 1024 * 1024
_REQUEST_FILE = "rosbag_card.json"


class BagStore:
    """Read-only access to bags created by this recorder under one root."""

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()

    def _bag_dir(self, bag_id):
        if not isinstance(bag_id, str) or not _BAG_ID.fullmatch(bag_id):
            raise ValueError("无效的录制编号")
        directory = self.root / bag_id
        if directory.is_symlink() or not directory.is_dir():
            raise FileNotFoundError("录制目录不存在")
        if directory.resolve().parent != self.root:
            raise ValueError("录制目录不在存储目录中")
        return directory

    def list_bags(self, active_id=None, offset=0, limit=100):
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("无效的分页位置")
        if not self.root.exists():
            return {"recordings": [], "offset": offset, "total": 0, "has_more": False}
        entries = []
        for entry in self.root.iterdir():
            if not _BAG_ID.fullmatch(entry.name) or entry.is_symlink() or not entry.is_dir():
                continue
            try:
                modified = entry.stat().st_mtime
            except OSError:
                continue
            entries.append({"id": entry.name, "modified_at": modified,
                            "recording": entry.name == active_id})
        entries.sort(key=lambda item: (item["modified_at"], item["id"]), reverse=True)
        return {"recordings": entries[offset:offset + limit], "offset": offset,
                "total": len(entries), "has_more": offset + limit < len(entries)}

    @staticmethod
    def _files(directory, bag_id):
        files = []
        for entry in directory.iterdir():
            name = entry.name
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
                continue
            if not (name in ("metadata.yaml", _REQUEST_FILE) or
                    (name.startswith(bag_id + "_") and name.endswith((".db3", ".mcap")))):
                continue
            try:
                if entry.is_symlink() or not entry.is_file():
                    continue
                size = entry.stat().st_size
            except OSError:
                continue
            files.append({"name": name, "size_bytes": size})
        return sorted(files, key=lambda item: item["name"])

    def remember_expected(self, bag_id, topics):
        """Persist the selection once so result checks survive service restarts."""
        directory = self._bag_dir(bag_id)
        path = directory / _REQUEST_FILE
        data = json.dumps({"version": 1, "requested_topics": list(topics)},
                          ensure_ascii=False).encode("utf-8")
        try:
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                         getattr(os, "O_NOFOLLOW", 0), 0o600)
        except FileExistsError:
            return
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)

    @staticmethod
    def _remembered_topics(directory):
        path = directory / _REQUEST_FILE
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
                return None
            with path.open("r", encoding="utf-8") as stream:
                value = json.load(stream)
            topics = value.get("requested_topics") if isinstance(value, dict) and value.get("version") == 1 else None
            if not isinstance(topics, list) or not topics or not all(isinstance(t, str) for t in topics):
                return None
            return topics
        except (OSError, UnicodeError, ValueError):
            return None

    def inspect(self, bag_id, expected_topics=None, active=False):
        directory = self._bag_dir(bag_id)
        files = self._files(directory, bag_id)
        result = {"id": bag_id, "path": str(directory), "recording": active, "files": files,
                  "size_bytes": sum(item["size_bytes"] for item in files),
                  "metadata_status": "unavailable", "message_count": None,
                  "duration_sec": None, "start_time_sec": None, "topics": [], "warnings": []}
        if active:
            result["warnings"].append("录制正在进行，停止后才能确认结果或下载文件")
            return result
        metadata = directory / "metadata.yaml"
        try:
            if metadata.is_symlink() or not metadata.is_file():
                result["warnings"].append("没有可读取的 metadata.yaml，无法确认录制结果")
                return result
            if metadata.stat().st_size > _MAX_METADATA_BYTES:
                raise ValueError("metadata.yaml 过大")
            with metadata.open("r", encoding="utf-8") as stream:
                document = yaml.safe_load(stream)
            info = document.get("rosbag2_bagfile_information") if isinstance(document, dict) else None
            if not isinstance(info, dict):
                raise ValueError("metadata.yaml 缺少录制信息")
            count = info.get("message_count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("metadata.yaml 的消息数量无效")
            raw_topics = info.get("topics_with_message_count")
            if raw_topics is None and count == 0:
                raw_topics = []
            if not isinstance(raw_topics, list):
                raise ValueError("metadata.yaml 的话题信息无效")
            topics = []
            for item in raw_topics:
                if not isinstance(item, dict) or not isinstance(item.get("topic_metadata"), dict):
                    raise ValueError("metadata.yaml 的话题信息无效")
                topic = item["topic_metadata"]
                number = item.get("message_count")
                if (not isinstance(topic.get("name"), str) or isinstance(number, bool)
                        or not isinstance(number, int) or number < 0):
                    raise ValueError("metadata.yaml 的话题消息数量无效")
                type_name = topic.get("type", "")
                if not isinstance(type_name, str):
                    raise ValueError("metadata.yaml 的话题类型无效")
                topics.append({"name": topic["name"], "type": type_name,
                               "message_count": number})
            duration = info.get("duration") or {}
            nanos = duration.get("nanoseconds") if isinstance(duration, dict) else None
            if isinstance(nanos, int) and not isinstance(nanos, bool) and nanos >= 0:
                result["duration_sec"] = nanos / 1_000_000_000
            start = info.get("starting_time") or {}
            started_ns = start.get("nanoseconds_since_epoch") if isinstance(start, dict) else None
            if isinstance(started_ns, int) and not isinstance(started_ns, bool) and started_ns > 0:
                result["start_time_sec"] = started_ns / 1_000_000_000
            result.update(metadata_status="available", message_count=count, topics=topics,
                          storage_id=info.get("storage_identifier"))
            if count == 0:
                result["warnings"].append("此 bag 没有记录到消息")
            if expected_topics is None:
                expected_topics = self._remembered_topics(directory)
            if expected_topics is not None:
                counts = {item["name"]: item["message_count"] for item in topics}
                missing = [name for name in expected_topics if counts.get(name, 0) == 0]
                result["missing_topics"] = missing
                if missing:
                    result["warnings"].append("以下目标话题没有记录到消息：" + ", ".join(missing))
            else:
                result["warnings"].append("未保存原始目标话题列表，无法确认是否遗漏目标话题")
            available = {item["name"] for item in files}
            if count > 0 and not any(name.endswith((".db3", ".mcap")) for name in available):
                result["warnings"].append("没有可读取的数据文件")
            relative_paths = info.get("relative_file_paths") or []
            if not isinstance(relative_paths, list):
                raise ValueError("metadata.yaml 的文件列表无效")
            for name in relative_paths:
                if not isinstance(name, str) or name not in available:
                    result["warnings"].append("元信息引用的存储文件缺失或不可读取")
                    break
            result["warnings"].append("结果依据 metadata.yaml；尚未逐条读取或验证消息")
        except (OSError, UnicodeError, ValueError, TypeError, yaml.YAMLError) as exc:
            result.update(metadata_status="invalid", message_count=None, topics=[],
                          duration_sec=None, start_time_sec=None)
            result.pop("missing_topics", None)
            result.pop("storage_id", None)
            result["warnings"] = ["无法解析录制元信息：" + str(exc)]
        return result

    def open_file(self, bag_id, filename, active=False):
        if active:
            raise ValueError("录制进行中，停止后才能下载")
        directory = self._bag_dir(bag_id)
        if not isinstance(filename, str) or filename not in {f["name"] for f in self._files(directory, bag_id)}:
            raise FileNotFoundError("录制文件不存在")
        path = directory / filename
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(str(path), flags)
        try:
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError("不能下载该文件")
            return os.fdopen(fd, "rb"), file_stat.st_size
        except BaseException:
            os.close(fd)
            raise
