from __future__ import annotations

import json
import sys
import threading
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.navigation.planning.goal_input import GoalInputCleanupError, GoalTopicInput, _goal


def payload(**changes):
    return {"schema": "phanthy.navigation.goal.v1", "goal_id": "goal-1", "x": 1, "y": 2, "yaw": 0, **changes}


@pytest.mark.parametrize("changes", [
    {"goal_id": " bad"}, {"x": True}, {"x": "1"}, {"x": float("nan")},
    {"speed": 0.2}, {"mode": True}, {"mode": 1}, {"extra": 1},
    {"schema": "other"},
])
def test_validation_rejects_invalid(changes):
    with pytest.raises(ValueError):
        _goal(json.dumps(payload(**changes)))


def test_validation_rejects_duplicate_and_missing():
    with pytest.raises(ValueError):
        _goal('{"schema":"phanthy.navigation.goal.v1","goal_id":"x","x":1,"x":2,"y":0,"yaw":0}')
    with pytest.raises(ValueError):
        _goal(json.dumps({"schema": "phanthy.navigation.goal.v1"}))
    assert _goal(json.dumps(payload()))["_control_nav_id"] == "goal-1"


@pytest.fixture
def ros(monkeypatch):
    class Node:
        instances = []
        def __init__(self, name):
            self.destroyed = False
            self.destroy_calls = 0
            self.destroy_failures = 0
            self.instances.append(self)
        def create_subscription(self, msg, topic, callback, qos):
            self.topic, self.callback, self.qos = topic, callback, qos
            return object()
        def destroy_node(self):
            self.destroy_calls += 1
            if self.destroy_failures:
                self.destroy_failures -= 1
                raise RuntimeError("destroy failed")
            self.destroyed = True
    class QoS:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    rclpy = types.ModuleType("rclpy")
    node = types.ModuleType("rclpy.node")
    node.Node = Node
    qos = types.ModuleType("rclpy.qos")
    qos.QoSProfile = QoS
    qos.DurabilityPolicy = types.SimpleNamespace(VOLATILE="volatile")
    qos.HistoryPolicy = types.SimpleNamespace(KEEP_LAST="last")
    qos.ReliabilityPolicy = types.SimpleNamespace(BEST_EFFORT="best")
    msgs = types.ModuleType("std_msgs.msg")
    msgs.String = type("String", (), {})
    for name, module in [("rclpy", rclpy), ("rclpy.node", node), ("rclpy.qos", qos), ("std_msgs", types.ModuleType("std_msgs")), ("std_msgs.msg", msgs)]:
        monkeypatch.setitem(sys.modules, name, module)
    class Executor:
        Node = node.Node
        def __init__(self):
            self.nodes = []
            self.remove_calls = 0
            self.remove_failures = 0
        def add_node(self, node):
            self.nodes.append(node)
        def remove_node(self, node):
            self.remove_calls += 1
            if self.remove_failures:
                self.remove_failures -= 1
                raise RuntimeError("remove failed")
            if node in self.nodes:
                self.nodes.remove(node)
    return Executor()


def test_subscribe_deduplicate_latest_and_close(ros):
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    calls = []
    def dispatch(goal):
        calls.append(goal["_control_nav_id"])
        if len(calls) == 1:
            entered.set()
            assert release.wait(2)
        else:
            done.set()
        return {"status": "navigating"}
    sub = GoalTopicInput("/custom/goal", ros, dispatch)
    node = ros.nodes[0]
    assert node.topic == "/custom/goal"
    assert (node.qos.reliability, node.qos.depth, node.qos.durability) == ("best", 20, "volatile")
    def emit(goal_id):
        node.callback(types.SimpleNamespace(data=json.dumps(payload(goal_id=goal_id))))
    emit("one")
    assert entered.wait(2)
    emit("one")
    emit("two")
    emit("three")
    release.set()
    assert done.wait(2)
    assert calls == ["one", "three"]
    sub.close()
    emit("four")
    sub._worker.join(2)
    assert not sub._worker.is_alive() and calls == ["one", "three"]
    assert node.destroyed and not ros.nodes


@pytest.mark.parametrize("failed_stage", ["remove", "destroy"])
def test_close_failure_retains_pending_stage_for_retry(ros, failed_stage):
    calls = []
    sub = GoalTopicInput("/goal", ros, lambda goal: calls.append(goal))
    node = ros.nodes[0]
    if failed_stage == "remove":
        ros.remove_failures = 1
    else:
        node.destroy_failures = 1
    with pytest.raises(RuntimeError, match=f"{failed_stage} failed"):
        sub.close()
    node.callback(types.SimpleNamespace(data=json.dumps(payload())))
    if failed_stage == "remove":
        assert not node.destroyed and node in ros.nodes
    else:
        assert not node.destroyed and not ros.nodes
    sub.close()
    sub.close()
    sub._worker.join(2)
    assert not sub._worker.is_alive()
    assert node.destroyed and not ros.nodes and not calls
    assert ros.remove_calls == (2 if failed_stage == "remove" else 1)
    assert node.destroy_calls == (2 if failed_stage == "destroy" else 1)


@pytest.mark.parametrize("stage", ["subscription", "add", "worker_create", "worker_start"])
def test_construction_failure_cleans_node(ros, monkeypatch, stage):
    if stage == "subscription":
        def fail_subscription(*args):
            raise RuntimeError("construct failed")
        monkeypatch.setattr(ros.Node, "create_subscription", fail_subscription)
    elif stage == "add":
        def fail_add(node):
            ros.nodes.append(node)
            raise RuntimeError("construct failed")
        monkeypatch.setattr(ros, "add_node", fail_add)
    elif stage == "worker_create":
        def fail_create(*args, **kwargs):
            raise RuntimeError("construct failed")
        monkeypatch.setattr(threading, "Thread", fail_create)
    else:
        def fail_start(self):
            raise RuntimeError("construct failed")
        monkeypatch.setattr(threading.Thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="construct failed"):
        GoalTopicInput("/goal", ros, lambda goal: None)
    assert ros.Node.instances[-1].destroyed
    assert not ros.nodes
    assert ros.remove_calls == (0 if stage == "subscription" else 1)


@pytest.mark.parametrize("failed_stage", ["remove", "destroy"])
def test_construction_cleanup_failure_exposes_retry_handle(ros, monkeypatch, failed_stage):
    def fail_start(self):
        raise RuntimeError("worker failed")
    monkeypatch.setattr(threading.Thread, "start", fail_start)
    if failed_stage == "remove":
        ros.remove_failures = 1
    else:
        original = ros.Node.destroy_node
        def fail_once(node):
            node.destroy_failures = 1
            monkeypatch.setattr(ros.Node, "destroy_node", original)
            return original(node)
        monkeypatch.setattr(ros.Node, "destroy_node", fail_once)
    with pytest.raises(GoalInputCleanupError) as caught:
        GoalTopicInput("/goal", ros, lambda goal: None)
    exc = caught.value
    assert str(exc.construction_error) == "worker failed"
    assert str(exc.__cause__) == f"{failed_stage} failed"
    node = ros.Node.instances[-1]
    assert not node.destroyed
    exc.goal_input.close()
    exc.goal_input.close()
    assert node.destroyed and not ros.nodes


def test_conflicting_seen_and_pending_goals_are_logged(ros, caplog):
    entered, release = threading.Event(), threading.Event()
    calls = []
    def dispatch(goal):
        calls.append(goal)
        entered.set()
        assert release.wait(2)
        return {"status": "navigating"}
    sub = GoalTopicInput("/goal", ros, dispatch)
    node = ros.nodes[0]
    def emit(**kwargs):
        node.callback(types.SimpleNamespace(data=json.dumps(payload(**kwargs))))
    try:
        emit(goal_id="seen")
        assert entered.wait(2)
        emit(goal_id="seen", x=8)
        emit(goal_id="pending")
        emit(goal_id="pending", x=9)
        assert caplog.text.count("conflicting goal topic payload") == 2
        assert sub._pending["x"] == 1
    finally:
        sub.close()
        release.set()
        sub._worker.join(2)
    assert len(calls) == 1


@pytest.mark.parametrize("result", [{"ok": False}, {"error": "failed"}, {"state": "error"}, {"status": "error"}, None])
def test_dispatch_failures_are_logged_and_remain_deduplicated(ros, caplog, result):
    called = threading.Event()
    calls = []
    def dispatch(goal):
        calls.append(goal)
        called.set()
        return result
    sub = GoalTopicInput("/goal", ros, dispatch)
    node = ros.nodes[0]
    message = types.SimpleNamespace(data=json.dumps(payload()))
    try:
        node.callback(message)
        assert called.wait(2)
        node.callback(message)
        sub.close()
        sub._worker.join(2)
        assert len(calls) == 1
        assert "goal topic dispatch not confirmed" in caplog.text
    finally:
        sub.close()
