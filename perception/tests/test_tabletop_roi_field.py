"""Isolated field safety, evidence and diagnostics checks; no camera or lab access."""

import importlib.util
import json
import os
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest
from plugins.tabletop_roi_goal_verifier.contracts import Refusal
from plugins.tabletop_roi_goal_verifier.evidence import save_session
from plugins.tabletop_roi_goal_verifier.server import handler
from test_tabletop_roi_goal_verifier import (
    configuration,
    feed,
    invoke,
    picture,
    task,
    verify,
)
from test_tabletop_roi_transport import running

FIELD = Path(__file__).parents[1] / "plugins/tabletop_roi_goal_verifier/deploy/field"


def module(name):
    spec = importlib.util.spec_from_file_location(
        "roi_field_" + name, FIELD / (name + ".py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_evidence_saves_images_coordinates_parameters_times_and_hashes(tmp_path):
    import hashlib

    from plugins.tabletop_roi_goal_verifier.plugin import TabletopROIGoalVerifierPlugin

    card = TabletopROIGoalVerifierPlugin(configuration())
    try:
        identity = task(card)
        feed(card, picture())
        invoke(card, "capture_before", request_id="b", **identity)
        verify(card, identity, picture(True))
        exported = invoke(card, "export_session", **identity)
        saved = save_session(exported, tmp_path, "PASS", "SYNTHETIC only")
        directory = Path(saved["directory"])
        session = json.loads((directory / "session.json").read_text())
        assert session["before"]["provenance"] == "synthetic"
        assert session["target_roi"] and session["roi_options"]
        assert (
            session["result"]["evidence"]["before"]["captured_at"]
            < session["marker"]["not_before"]
            < session["result"]["evidence"]["after"][0]["captured_at"]
        )
        assert (directory / "before-roi.jpg").stat().st_size > 0 and (
            directory / "after-roi.jpg"
        ).stat().st_size > 0
        assert "session_token" not in session
        assert session["test_record"]["hardware_acceptance"] is False
        for line in (directory / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split("  ")
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
        with pytest.raises(Refusal):
            save_session(exported, tmp_path, "../escape")
    finally:
        card.stop()


def test_doctor_reports_software_only_without_claiming_hardware():
    doctor = module("doctor")
    with running() as (card, base):
        invoke(card, "start")
        checks = doctor.web_checks(base)
        assert all(x["ok"] for x in checks.values())
        assert checks["card"]["data"]["sources"][0]["provenance"] == "synthetic"


def test_runner_chooses_another_port_and_stops_only_its_process(tmp_path):
    runner = module("runner")
    root = tmp_path
    for name in ("logs", "evidence", ".tmp", "deps"):
        (root / name).mkdir()
    # Point source at the new Card implementation; all writes remain in tmp_path.
    source = Path(__file__).parents[1]
    (root / "src").mkdir()
    (root / "src/perception").symlink_to(source, target_is_directory=True)
    (root / "config.json").write_text(json.dumps(configuration()))
    (root / "runtime.json").write_text(
        json.dumps({"python": sys.executable, "method": "test"})
    )
    with socket.socket() as occupant:
        occupant.bind(("127.0.0.1", 0))
        occupant.listen(1)
        port = occupant.getsockname()[1]
        if port > 65436:
            pytest.skip("OS selected port too close to limit")
        state = runner.start(root, port, offline=True)
        try:
            assert state["port"] != port and state["offline_test"] is True
            assert runner.start(root, port, offline=True)["already_running"]
            assert occupant.getsockname()[1] == port
        finally:
            assert runner.stop(root)["state"] == "stopped"
        assert runner.stop(root)["state"] == "already_stopped"
        assert occupant.getsockname()[1] == port
    # A live, different PID must never receive a signal.
    other = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(15)"])
    try:
        (root / "service.json").write_text(
            json.dumps({"pid": other.pid, "start_ticks": "invalid"})
        )
        with pytest.raises(RuntimeError, match="identity changed"):
            runner.stop(root)
        assert other.poll() is None
    finally:
        other.terminate()
        other.wait(timeout=3)


def test_evidence_endpoint_refuses_unverified_session():
    import tempfile
    import threading
    from http.server import ThreadingHTTPServer

    from plugins.tabletop_roi_goal_verifier.plugin import TabletopROIGoalVerifierPlugin

    card = TabletopROIGoalVerifierPlugin(configuration())
    with tempfile.TemporaryDirectory() as directory:
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler(card, Path(directory)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            identity = task(card)
            req = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/evidence",
                data=json.dumps(dict(**identity, expected="FAIL")).encode(),
                headers={"Content-Type": "application/json"},
            )
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(req) as response:
                out = json.load(response)
            assert out["error"]["data"]["reason_code"] == "VERIFICATION_REQUIRED"
            assert not list(Path(directory).iterdir())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)
            card.stop()


def test_stop_uses_a_pinned_process_handle_instead_of_a_reusable_pid(
    tmp_path, monkeypatch
):
    runner = module("runner")
    state = {"pid": 12345, "start_ticks": "test"}
    (tmp_path / "service.json").write_text(json.dumps(state))
    states = iter([{"process_state": "S"}, None])
    monkeypatch.setattr(runner, "identity", lambda pid: next(states))
    monkeypatch.setattr(runner, "owned", lambda *args: True)
    opened = []
    sent = []
    closed = []
    monkeypatch.setattr(
        runner, "_open_pid_handle", lambda pid: opened.append(pid) or 77
    )
    monkeypatch.setattr(os, "close", closed.append)
    monkeypatch.setattr(
        os, "kill", lambda *args: pytest.fail("numeric PID must not be signalled")
    )
    monkeypatch.setattr(runner, "_signal_pid_handle", lambda fd, sig: sent.append(fd))
    assert runner.stop(tmp_path)["state"] == "stopped"
    assert opened == [12345] and sent == [77] and closed == [77]


def test_doctor_retains_initialization_stage_and_traceback(monkeypatch):
    from types import SimpleNamespace

    import rclpy

    doctor = module("doctor")
    monkeypatch.setitem(
        sys.modules,
        "rclpy.qos",
        SimpleNamespace(
            QoSProfile=object,
            ReliabilityPolicy=object,
            QoSCompatibility=object,
            qos_check_compatible=object,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "sensor_msgs.msg",
        SimpleNamespace(CompressedImage=object, Image=object),
    )
    # Inject the Humble API surface missing from the shared minimal ROS stubs.
    monkeypatch.setitem(
        sys.modules, "rclpy.context", SimpleNamespace(Context=lambda: object())
    )
    monkeypatch.setitem(
        sys.modules,
        "rclpy.executors",
        SimpleNamespace(SingleThreadedExecutor=lambda **kw: None),
    )
    monkeypatch.setattr(rclpy, "__file__", "/SYNTHETIC/rclpy.py", raising=False)
    monkeypatch.setattr(
        rclpy,
        "init",
        lambda **kw: (_ for _ in ()).throw(
            RuntimeError("DDS initialization unavailable")
        ),
        raising=False,
    )
    result = doctor.ros_checks({"domain_id": 42}, 0.01)
    assert result["ros"]["code"] == "ROS_INIT_FAILED"
    assert result["ros"]["error_type"] == "RuntimeError"
    assert "DDS initialization unavailable" in result["ros"]["traceback"]


def test_domain_contract_rejects_222_before_ros_or_process_creation(
    tmp_path, monkeypatch
):
    from plugins.tabletop_roi_goal_verifier.configuration import validate

    doctor = module("doctor")
    assert (
        doctor.ros_checks({"domain_id": 222}, 0.01)["ros"]["code"]
        == "ROS_DOMAIN_42_REQUIRED"
    )
    cfg = configuration()
    cfg["sources"]["camera"].update(
        kind="ros_image",
        domain_id=222,
        topic="/SYNTHETIC_ONLY",
        message_type="compressed",
        stamp_semantics="capture",
        capture_time_record="SYNTHETIC ONLY",
    )
    with pytest.raises(Refusal) as refused:
        validate(cfg)
    assert refused.value.code == "ROS_DOMAIN_42_REQUIRED"
    (tmp_path / "runtime.json").write_text(json.dumps({"python": sys.executable}))
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    runner = module("runner")
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *a, **kw: pytest.fail("must reject before creating a process"),
    )
    with pytest.raises(RuntimeError, match="ROS_DOMAIN_42_REQUIRED"):
        runner.start(tmp_path, 18084, offline=True)


def test_ros_cleanup_failure_still_destroys_all_owned_nodes_and_retires_sessions():
    from types import SimpleNamespace

    from plugins.tabletop_roi_goal_verifier.plugin import TabletopROIGoalVerifierPlugin

    card = TabletopROIGoalVerifierPlugin(configuration())
    task(card)
    cleaned = []
    context = SimpleNamespace(
        ok=lambda: True, shutdown=lambda: cleaned.append("context")
    )

    def fail_shutdown(**kwargs):
        raise RuntimeError("executor shutdown failed")

    executor = SimpleNamespace(shutdown=fail_shutdown, remove_node=lambda n: None)
    card.sources.ros = (context, executor)
    card.sources.nodes = [
        (SimpleNamespace(destroy_node=lambda: cleaned.append("first")), executor),
        (SimpleNamespace(destroy_node=lambda: cleaned.append("second")), executor),
    ]
    with pytest.raises(RuntimeError, match="ROS_CLEANUP_FAILED"):
        card.stop()
    assert cleaned == ["first", "second", "context"]
    assert not card.engine.running and not card.engine.sessions and not card.started
    card.stop()  # cleanup failure must not make a repeated stop unsafe
