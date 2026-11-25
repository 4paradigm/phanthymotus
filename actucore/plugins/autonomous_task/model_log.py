"""One directory per task and immutable HTTP JSON bodies per API call."""
import json
import os
from pathlib import Path
import re
import tempfile
import time


_JSON = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
_SESSION = "0000-session.json"
_END = "0000-end.json"


def _directory(root, action_id):
    if not isinstance(action_id, str) or not re.fullmatch(r"autonomous_task_[0-9a-f]{32}", action_id):
        raise ValueError("无效的 action_id")
    path = Path(root) / action_id
    if path.is_symlink():
        raise ValueError("模型日志路径不能是符号链接")
    return path


def _publish(path, value, raw_body=None):
    # Publish only complete files. A duplicate event must never replace training data.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as temp:
        pending = Path(temp.name)
        try:
            if raw_body is not None:
                temp.write(raw_body)
            else:
                for chunk in _JSON.iterencode(value):
                    temp.write(chunk.encode("utf-8"))
            temp.flush()
            os.fsync(temp.fileno())
            os.link(pending, path)
        finally:
            pending.unlink(missing_ok=True)


class ModelLog:
    """Owned by one worker executing successive model calls within a submit."""

    def __init__(self, root: Path, run: dict):
        self.path = _directory(root, run["action_id"])
        metadata = {k: run[k] for k in ("action_id", "instruction", "arm_mode", "started_at")}
        metadata.update(schema_version=3, request_id=run.get("request_id"))
        self.path.mkdir(mode=0o700, parents=True, exist_ok=False)
        try:
            _publish(self.path / _SESSION, metadata)
        except Exception:
            self.path.rmdir()
            raise
        self._round = 0
        self._step = None

    def append(self, kind, *, step=None, phase=None, raw_body=None, **fields):
        if kind == "request":
            # Number HTTP calls, independently of any step labels the caller uses.
            number = self._round + 1
            _publish(self.path / f"{number:04d}-request.json", fields["payload"], raw_body)
            self._round, self._step = number, step
            return
        if kind == "task_end":
            _publish(self.path / _END, {"type": kind, "ts": time.time(), **fields})
            return
        if not self._round or (step is not None and self._step is not None and self._step != step):
            raise ValueError("事件没有对应的模型请求")
        if kind == "response":
            _publish(self.path / f"{self._round:04d}-response.json", fields["payload"], raw_body)
            return
        if kind not in ("tool_result", "error"):
            raise ValueError(f"未知日志事件 {kind}")
        event = {"type": kind, "ts": time.time(), **fields}
        if step is not None:
            event["step"] = step
        if phase is not None:
            event["phase"] = phase
        suffix = "tool-result" if kind == "tool_result" else "error"
        _publish(self.path / f"{self._round:04d}-{suffix}.json", event)
