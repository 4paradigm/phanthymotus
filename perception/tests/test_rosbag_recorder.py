"""ROS bag recording contracts, using fake processes and no ROS runtime."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from plugins.rosbag import plugin as card_module  # noqa: E402
from plugins.rosbag import recorder as recorder_module  # noqa: E402


class FakeProcess:
    def __init__(self):
        self.returncode = None
        self.signals = []
        self.killed = False
        self.timeout_once = False

    def poll(self):
        return self.returncode

    def send_signal(self, sig):
        self.signals.append(sig)

    def wait(self, timeout):
        if self.timeout_once:
            self.timeout_once = False
            raise subprocess.TimeoutExpired("ros2", timeout)
        self.returncode = -9 if self.killed else 0
        return self.returncode

    def kill(self):
        self.killed = True


@pytest.fixture
def harness(tmp_path, monkeypatch):
    process = FakeProcess()
    popen = Mock(return_value=process)
    thread = Mock()
    clock = SimpleNamespace(now=100.0)
    disk = SimpleNamespace(free=10_000)
    fake_time = SimpleNamespace(monotonic=lambda: clock.now, strftime=lambda _fmt: "bag_20260101_000000_")
    monkeypatch.setattr(recorder_module, "time", fake_time)
    monkeypatch.setattr(recorder_module.subprocess, "Popen", popen)
    monkeypatch.setattr(recorder_module.shutil, "disk_usage", lambda _path: disk)
    monkeypatch.setattr(recorder_module.threading, "Thread", thread)
    return SimpleNamespace(
        recorder=recorder_module.Recorder(tmp_path), process=process,
        popen=popen, clock=clock, disk=disk, thread=thread,
    )


def _metadata(result):
    path = Path(result["path"])
    path.mkdir(parents=True)
    (path / (path.name + "_0.db3")).touch()
    (path / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n", encoding="utf-8")


def test_card_declares_bounded_stop_deadline(tmp_path):
    card = card_module.RosbagRecorderPlugin({"output_root": str(tmp_path)})
    tool = card.get_tools()[0]
    assert tool["inputSchema"]["x-stop"]["timeout_s"] == 25
    assert tool["inputSchema"]["x-start"]["failed_states"] == ["completed"]


def test_records_requested_topics_and_closes_on_sigint(harness):
    started = harness.recorder.start(["/camera/image", "/imu"])
    assert started["state"] == "loading"
    args, kwargs = harness.popen.call_args
    assert args[0][:7] == ["ros2", "bag", "record", "--include-hidden-topics", "-s", "sqlite3", "-o"]
    assert args[0][7:] == [started["path"], "/camera/image", "/imu"]
    assert kwargs.get("shell", False) is False
    _metadata(started)
    assert harness.recorder.info()["state"] == "recording"
    harness.clock.now += 3

    stopped = harness.recorder.stop()
    assert harness.process.signals == [signal.SIGINT]
    assert stopped["state"] == "completed"
    assert stopped["stop_reason"] == "requested"
    assert stopped["duration_sec"] == 3
    request = json.loads((Path(started["path"]) / "rosbag_card.json").read_text(encoding="utf-8"))
    assert request["requested_topics"] == ["/camera/image", "/imu"]


def test_missing_ros2_executable_is_reported(harness):
    harness.popen.side_effect = FileNotFoundError("ros2 not found")
    result = harness.recorder.start(["/imu"])
    assert result["state"] == "error"
    assert "ros2 not found" in result["message"]


def test_start_error_does_not_report_a_previous_bag(harness):
    started = harness.recorder.start(["/old"])
    _metadata(started)
    harness.clock.now += 3
    assert harness.recorder.stop()["state"] == "completed"
    harness.popen.side_effect = FileNotFoundError("ros2 not found")

    result = harness.recorder.start(["/new"])

    assert result["state"] == "error"
    assert "ros2 not found" in result["message"]
    assert result["path"] is None
    assert result["topics"] == ["/new"]
    assert result["duration_sec"] == 0
    assert result["stop_reason"] is None
    assert harness.recorder.info() == result


def test_unexpected_exit_is_reported_by_info(harness):
    harness.recorder.start(["/imu"])
    harness.process.returncode = 2
    result = harness.recorder.info()
    assert result["state"] == "error"
    assert result["stop_reason"] == "unexpected_exit"
    assert "code 2" in result["message"]
    assert "incomplete" in result["message"]


def test_missing_metadata_after_stop_is_incomplete(harness):
    harness.recorder.start(["/imu"])
    result = harness.recorder.stop()
    assert result["state"] == "error"
    assert "incomplete" in result["message"]


def test_stop_timeout_kills_process_and_reports_incomplete(harness):
    harness.recorder.start(["/imu"])
    harness.process.timeout_once = True
    result = harness.recorder.stop()
    assert harness.process.signals == [signal.SIGINT]
    assert harness.process.killed
    assert result["state"] == "error"
    assert "incomplete" in result["message"]


class OneTick:
    def __init__(self):
        self.calls = 0

    def wait(self, _timeout):
        self.calls += 1
        return self.calls > 1


@pytest.mark.parametrize("limit", ["duration", "disk"])
def test_watch_stops_at_duration_or_free_space_limit(harness, limit):
    started = harness.recorder.start(["/imu"], max_duration_sec=30, min_free_bytes=100)
    _metadata(started)
    if limit == "duration":
        harness.clock.now += 31
        reason = "duration_limit"
    else:
        harness.disk.free = 99
        reason = "disk_space_limit"

    harness.recorder._watch(harness.process, OneTick())
    result = harness.recorder.info()
    assert harness.process.signals == [signal.SIGINT]
    assert result["state"] == "completed"
    assert result["stop_reason"] == reason


def test_disk_space_preflight_refuses_to_start(harness):
    harness.disk.free = 99
    result = harness.recorder.start(["/imu"], min_free_bytes=100)
    assert result["state"] == "error"
    assert "free space" in result["message"]
    assert harness.recorder.info() == result
    harness.popen.assert_not_called()


def test_disk_preflight_error_does_not_report_a_previous_bag(harness):
    started = harness.recorder.start(["/old"])
    _metadata(started)
    assert harness.recorder.stop()["state"] == "completed"
    harness.disk.free = 99

    result = harness.recorder.start(["/new"], min_free_bytes=100)

    assert result["state"] == "error"
    assert result["path"] is None
    assert result["topics"] == ["/new"]
    assert result["stop_reason"] is None
    assert harness.recorder.info() == result


def test_duplicate_start_does_not_replace_running_state(harness):
    started = harness.recorder.start(["/imu"])
    refused = harness.recorder.start(["/other"])

    assert refused["state"] == "error"
    info = harness.recorder.info()
    assert info["state"] == "loading"
    assert info["path"] == started["path"]
    assert info["topics"] == ["/imu"]
    harness.popen.assert_called_once()


def test_recorder_that_never_opens_storage_is_stopped(harness):
    started = harness.recorder.start(["/imu"], max_duration_sec=60)
    assert started["state"] == "loading"
    harness.clock.now += 31
    harness.recorder._watch(harness.process, OneTick())
    result = harness.recorder.info()
    assert result["state"] == "error"
    assert result["stop_reason"] == "startup_timeout"
    assert not result["process_active"]


def test_bad_configuration_does_not_start_with_previous_topics(tmp_path, monkeypatch):
    recorder = Mock()
    recorder.info.return_value = {"state": "idle"}
    recorder.start.return_value = {"state": "recording"}
    monkeypatch.setattr(card_module, "Recorder", Mock(return_value=recorder))
    card = card_module.RosbagRecorderPlugin({"output_root": str(tmp_path)})

    assert card.dispatch(card.PREFIX, {"action": "config", "topics": ["/old"]})["state"] == "ok"
    assert card.dispatch(card.PREFIX, {"action": "config", "topics": ["invalid"]})["state"] == "error"
    assert card.dispatch(card.PREFIX, {"action": "start"})["state"] == "error"
    recorder.start.assert_not_called()

    assert card.dispatch(card.PREFIX, {"action": "config", "topics": ["/new"]})["state"] == "ok"
    assert card.dispatch(card.PREFIX, {"action": "start"})["state"] == "recording"
    assert recorder.start.call_args.args[0] == ["/new"]


def test_config_is_rejected_while_process_is_loading_or_failed_but_active(tmp_path, monkeypatch):
    recorder = Mock()
    monkeypatch.setattr(card_module, "Recorder", Mock(return_value=recorder))
    card = card_module.RosbagRecorderPlugin({"output_root": str(tmp_path)})
    for state in ("loading", "error"):
        recorder.info.return_value = {"state": state, "process_active": True}
        assert card.dispatch(card.PREFIX, {"action": "config", "topics": ["/imu"]})["state"] == "error"


@pytest.mark.parametrize('topic', ['/', '/camera/', '/camera//image', '/camera/1', '/camera/~private'])
def test_invalid_absolute_topic_names_are_refused(topic):
    with pytest.raises(ValueError):
        recorder_module.validate_config([topic], 0, 0)


def test_hidden_topic_is_valid():
    recorder_module.validate_config(['/_internal/camera_1'], 0, 0)


def test_topic_length_respects_ros_limit():
    recorder_module.validate_config(['/' + 'a' * 246], 0, 0)
    with pytest.raises(ValueError):
        recorder_module.validate_config(['/' + 'a' * 247], 0, 0)
