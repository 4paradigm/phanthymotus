"""Exercise the card's local session files and atomic publication without hardware."""
import base64
import json
import os
from pathlib import Path
import stat
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.autonomous_task import model_log


def task(digit):
    return {"action_id": "autonomous_task_" + digit * 32, "request_id": None, "started_at": 42.5,
            "instruction": "观察胶带", "arm_mode": "left"}


def test_http_files_match_wire_and_sessions_restart_numbering(tmp_path):
    first = task("a")
    log = model_log.ModelLog(tmp_path, first)
    assert log.path == tmp_path / first["action_id"]
    info = log.path / "0000-session.json"
    assert json.loads(info.read_bytes()) == {"schema_version": 3, **first}
    assert stat.S_IMODE(info.stat().st_mode) == 0o600
    assert stat.S_IMODE(log.path.stat().st_mode) == 0o700

    image = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
    request = {"model": "vision", "instructions": "看一眼", "input": [{"image_url": image}]}
    wire_request = json.dumps(request, ensure_ascii=True, indent=2).encode() + b"\n"
    wire_reply = b'{ "\\u0069d": "r", "number": 1.00, "output": [] }\n'
    log.append("request", step=17, phase="decide", payload=request, raw_body=wire_request)
    assert (log.path / "0001-request.json").read_bytes() == wire_request
    assert not (log.path / "0001-response.json").exists()
    log.append("response", step=17, payload=json.loads(wire_reply), raw_body=wire_reply)
    assert (log.path / "0001-response.json").read_bytes() == wire_reply
    log.append("request", step=41, payload={"input": ["new"]})
    log.append("response", step=41, payload={"output": []})
    log.append("task_end", status="completed", reason="ok")
    assert json.loads((log.path / "0002-request.json").read_bytes()) == {"input": ["new"]}
    assert json.loads((log.path / "0002-response.json").read_bytes()) == {"output": []}
    assert json.loads((log.path / "0000-end.json").read_bytes())["status"] == "completed"

    second = model_log.ModelLog(tmp_path, task("b"))
    second.append("request", payload={"x": "另一个任务"})
    assert sorted(p.name for p in second.path.iterdir()) == ["0000-session.json", "0001-request.json"]
    assert json.loads((second.path / "0001-request.json").read_bytes()) == {"x": "另一个任务"}
    with pytest.raises(FileExistsError):
        model_log.ModelLog(tmp_path, first)
    assert (log.path / "0001-response.json").read_bytes() == wire_reply
    assert not (log.path / "model_io.json").exists()


def test_errors_and_tool_outcomes_stay_separate_from_raw_bodies(tmp_path):
    run = task("a")
    log = model_log.ModelLog(tmp_path, run)
    error = {"type": "RuntimeError", "message": "HTTP 422"}
    for step in (1, 2, 3):
        log.append("request", step=step, payload={"input": []})
        if step == 1:
            log.append("error", step=step, error={"message": "network unavailable"})
        elif step == 2:
            log.append("response", step=step, payload={"error": {"code": "bad_request"}})
            log.append("error", step=step, phase="review", error=error, provider_response={"http_status": 422})
        else:
            log.append("response", step=step, payload={"output": [{"type": "function_call", "call_id": "c"}]})
            log.append("tool_result", step=step, phase="decide", call_id="c", name="left_move_by",
                       arguments={"delta": [0, 0, 0]}, payload={"status": "rejected"})
    assert not (log.path / "0001-response.json").exists()
    assert json.loads((log.path / "0002-response.json").read_bytes()) == {"error": {"code": "bad_request"}}
    err = json.loads((log.path / "0002-error.json").read_bytes())
    assert (err["step"], err["phase"], err["error"], err["provider_response"]) == (2, "review", error, {"http_status": 422})
    result = json.loads((log.path / "0003-tool-result.json").read_bytes())
    assert (result["call_id"], result["payload"]) == ("c", {"status": "rejected"})
    with pytest.raises(ValueError, match="对应"):
        log.append("response", step=2, payload={})
    with pytest.raises(FileExistsError):
        log.append("response", step=3, payload={"overwrite": True})


@pytest.mark.parametrize("operation", ["link", "fsync"])
def test_failed_publication_leaves_previous_bodies_untouched(tmp_path, monkeypatch, operation):
    log = model_log.ModelLog(tmp_path, task("b"))
    info = (log.path / "0000-session.json").read_bytes()

    def fail(*args):
        raise OSError("disk failure")

    with monkeypatch.context() as failing:
        failing.setattr(model_log.os, operation, fail)
        with pytest.raises(OSError, match="disk failure"):
            log.append("request", step=1, payload={"input": []})
        with pytest.raises(OSError, match="disk failure"):
            model_log.ModelLog(tmp_path, task("c"))
    assert not (tmp_path / task("c")["action_id"]).exists()
    assert [f.name for f in log.path.iterdir()] == ["0000-session.json"]
    assert (log.path / "0000-session.json").read_bytes() == info
    log.append("request", step=1, payload={"input": []})
    with pytest.raises(ValueError):
        log.append("response", step=1, payload={"bad": float("nan")})
    assert not (log.path / "0001-response.json").exists()
    log.append("response", step=1, payload={"ok": True})
    assert json.loads((log.path / "0001-response.json").read_bytes()) == {"ok": True}


def test_request_file_is_published_complete_and_later_rounds_preserve_it(tmp_path, monkeypatch):
    log = model_log.ModelLog(tmp_path, task("f"))
    request = log.path / "0001-request.json"
    link = os.link

    def inspect_then_link(src, dst):
        assert dst == request and not request.exists()
        assert Path(src).read_bytes() == body
        return link(src, dst)

    image = "data:image/png;base64," + base64.b64encode(bytes(range(256)) * 600).decode()
    payload = {"img": image}
    body = json.dumps(payload).encode()
    with monkeypatch.context() as writing:
        writing.setattr(model_log.os, "link", inspect_then_link)
        log.append("request", payload=payload, raw_body=body)
    log.append("response", payload={"unicode": "好"})
    log.append("request", payload={"index": 2})
    assert request.read_bytes() == body
    assert sorted(path.name for path in log.path.iterdir()) == [
        "0000-session.json", "0001-request.json", "0001-response.json", "0002-request.json"]


def test_existing_historical_session_is_preserved(tmp_path):
    run = task("e")
    directory = tmp_path / run["action_id"]
    directory.mkdir()
    path = directory / "model_io.json"
    body = b'{"schema_version": 2, "events": []}'
    path.write_bytes(body)

    with pytest.raises(FileExistsError):
        model_log.ModelLog(tmp_path, run)
    assert list(directory.iterdir()) == [path]
    assert path.read_bytes() == body
