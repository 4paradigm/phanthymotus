"""Tests for the generic ROS 2 node-graph card.

No rclpy: the graph builder takes a tiny source object, and the HTTP page is
exercised against a fake source on an ephemeral loopback port.
"""

import json
import types
import urllib.request

import pytest

from plugins.ros2_node_graph import Ros2NodeGraphPlugin
from plugins.ros2_node_graph_page import PAGE_HTML
from plugins.ros2_node_graph_runtime import (
    Config,
    build_snapshot,
    compact_payload,
    full_payload,
    url_info,
)


def _endpoint(name, namespace="/bumi", topic_type="std_msgs/msg/String"):
    return types.SimpleNamespace(node_name=name, node_namespace=namespace,
                                 topic_type=topic_type)


class FakeSource:
    def __init__(self):
        self.nodes = [
            ("state", "/bumi"),
            ("loco", "/bumi"),
            ("orphan", "/perception"),
            ("ros2_node_graph_card", "/"),
        ]
        self.topics = [
            ("/bumi/joints", ["sensor_msgs/msg/JointState"]),
            ("/perception/orphan", ["std_msgs/msg/String"]),
            ("/rosout", ["rcl_interfaces/msg/Log"]),
        ]
        self.publishers = {
            "/bumi/joints": [_endpoint("state", topic_type="sensor_msgs/msg/JointState")],
            "/perception/orphan": [_endpoint("orphan", "/perception")],
            "/rosout": [_endpoint("state")],
        }
        self.subscriptions = {
            "/bumi/joints": [_endpoint("loco", topic_type="sensor_msgs/msg/JointState")],
            "/perception/orphan": [],
            "/rosout": [],
        }

    def get_nodes(self):
        return self.nodes

    def get_topics(self):
        return self.topics

    def get_publishers(self, topic):
        return self.publishers.get(topic, [])

    def get_subscriptions(self, topic):
        return self.subscriptions.get(topic, [])


# ── graph model ──────────────────────────────────────────────────────────────

def test_snapshot_contains_nodes_topics_and_flows():
    snapshot = build_snapshot(FakeSource(), Config())
    assert snapshot["stats"]["nodes"] == 4
    assert snapshot["stats"]["topics"] == 2          # /rosout is hidden by default
    assert snapshot["stats"]["links"] == 3
    names = {topic["name"] for topic in snapshot["topics"]}
    assert names == {"/bumi/joints", "/perception/orphan"}
    joints = next(topic for topic in snapshot["topics"]
                  if topic["name"] == "/bumi/joints")
    assert joints["publishers"] == ["/bumi/state"]
    assert joints["subscribers"] == ["/bumi/loco"]
    assert "-> /bumi/joints [sensor_msgs/msg/JointState] -> /bumi/loco" in snapshot["summary"]


def test_system_topics_can_be_included():
    config = Config.from_dict({"include_system_topics": True})
    snapshot = build_snapshot(FakeSource(), config)
    assert "/rosout" in {topic["name"] for topic in snapshot["topics"]}


def test_topic_and_namespace_exclusions():
    config = Config.from_dict({
        "exclude_topics": ["/perception/"],
        "exclude_namespaces": ["/bumi"],
    })
    snapshot = build_snapshot(FakeSource(), config)
    assert snapshot["topics"] == []
    names = {node["name"] for node in snapshot["nodes"]}
    assert "/bumi/state" not in names and "/bumi/loco" not in names
    assert "/perception/orphan" in names


def test_topic_without_subscribers_is_still_reported():
    snapshot = build_snapshot(FakeSource(), Config())
    orphan = next(topic for topic in snapshot["topics"]
                  if topic["name"] == "/perception/orphan")
    assert orphan["subscribers"] == []
    assert "(no subscribers)" in snapshot["summary"]


def test_caps_mark_truncated():
    source = FakeSource()
    source.topics = [(f"/t/{index}", ["std_msgs/msg/String"]) for index in range(20)]
    source.publishers = {topic: [_endpoint("state")] for topic, _ in source.topics}
    source.subscriptions = {topic: [] for topic, _ in source.topics}
    config = Config.from_dict({"max_topics": 10, "max_flows": 5})
    snapshot = build_snapshot(source, config)
    assert snapshot["truncated"] is True
    assert len(snapshot["topics"]) == 10
    assert len(snapshot["flows"]) == 5
    assert "Truncated" in snapshot["summary"]


def test_summary_respects_character_budget():
    source = FakeSource()
    source.topics = [(f"/very/long/topic/name/{index}", ["std_msgs/msg/String"])
                     for index in range(40)]
    source.publishers = {topic: [_endpoint("state")] for topic, _ in source.topics}
    source.subscriptions = {topic: [_endpoint("loco", "/bumi")] for topic, _ in source.topics}
    config = Config.from_dict({"summary_max_chars": 500, "max_flows": 40})
    snapshot = build_snapshot(source, config)
    assert len(snapshot["summary"]) <= 500


def test_payloads_keep_the_full_graph_out_of_the_agent_topic():
    snapshot = build_snapshot(FakeSource(), Config())
    compact = compact_payload(snapshot, Config(), "http://127.0.0.1:15730/", "")
    assert "nodes" not in compact and "topics" not in compact
    assert compact["summary"] == snapshot["summary"]
    full = full_payload(snapshot, Config(), "http://127.0.0.1:15730/", "")
    assert full["nodes"] and full["topics"]


# ── URL / tunnel ─────────────────────────────────────────────────────────────

def test_loopback_url_needs_an_ssh_tunnel():
    config = Config.from_dict({"port": 15730, "ssh_host": "192.168.3.22",
                              "ssh_port": 2222, "ssh_user": "noetix"})
    info = url_info(config, 15730)
    assert info["mode"] == "tunnel"
    assert info["url"] == "http://127.0.0.1:15730/"
    assert info["tunnel"] == "ssh -p 2222 noetix@192.168.3.22 -L 15730:127.0.0.1:15730"


def test_public_url_skips_the_tunnel():
    config = Config.from_dict({"public_base_url": "https://robot.example/graph"})
    info = url_info(config, 15730)
    assert info["mode"] == "public"
    assert info["url"] == "https://robot.example/graph/"
    assert info["tunnel"] == ""


def test_direct_bind_uses_the_lan_address():
    config = Config.from_dict({"bind_host": "0.0.0.0", "ssh_host": "192.168.3.22",
                              "port": 15730})
    info = url_info(config, 15730)
    assert info["mode"] == "direct"
    assert info["url"] == "http://192.168.3.22:15730/"
    assert info["tunnel"] == ""


def test_invalid_public_url_is_rejected():
    with pytest.raises(ValueError):
        Config.from_dict({"public_base_url": "javascript:alert(1)"})


# ── page / HTTP ─────────────────────────────────────────────────────────────

def test_page_is_self_contained():
    assert "ROS 2 Node Graph" in PAGE_HTML
    assert "api/graph" in PAGE_HTML
    assert "textContent" in PAGE_HTML
    assert "https://" not in PAGE_HTML  # no CDN / external asset


def test_card_has_one_output_and_no_input():
    plugin = Ros2NodeGraphPlugin({"port": 0}, executor=None,
                                 graph_source=FakeSource())
    try:
        tool = plugin.get_tools()[0]
        assert tool["type"] == "sensor"
        assert "topic_in" not in tool
        assert tool["topic_out"][0]["format"] == "data/json"
        assert "x-completion" not in tool["inputSchema"]
        assert "port" in tool["configSchema"]["properties"]
        assert "public_base_url" in tool["configSchema"]["properties"]
    finally:
        plugin.shutdown()


def test_http_page_and_graph_endpoints():
    plugin = Ros2NodeGraphPlugin(
        {"port": 0, "bind_host": "127.0.0.1", "ssh_host": "192.168.3.22"},
        executor=None, graph_source=FakeSource())
    try:
        assert plugin._http_port, plugin._http_error
        base = f"http://127.0.0.1:{plugin._http_port}"
        with urllib.request.urlopen(base + "/", timeout=3) as response:
            assert response.status == 200
            assert "ROS 2 Node Graph" in response.read().decode()
        with urllib.request.urlopen(base + "/api/graph", timeout=3) as response:
            payload = json.load(response)
        assert payload["stats"]["topics"] == 2
        assert any(topic["name"] == "/bumi/joints" for topic in payload["topics"])
        with urllib.request.urlopen(base + "/api/summary", timeout=3) as response:
            summary = response.read().decode()
        assert "/bumi/joints" in summary
        info = plugin.dispatch("info", {})
        assert info["url"] == base + "/"
        assert "-L" in info["tunnel"]
        tools = plugin.get_tools()
        assert tools[0]["name"] == "ros2_node_graph"
        assert tools[0]["topic_out"][0]["format"] == "data/json"
        assert base in tools[0]["description"]
    finally:
        plugin.shutdown()


def test_config_action_changes_the_public_url():
    plugin = Ros2NodeGraphPlugin({"port": 0}, executor=None,
                                 graph_source=FakeSource())
    try:
        result = plugin.dispatch("config", {
            "public_base_url": "https://robot.example/graph"})
        assert result["status"] == "configured"
        assert result["url"] == "https://robot.example/graph/"
        assert plugin.dispatch("info", {})["url"] == "https://robot.example/graph/"
    finally:
        plugin.shutdown()


def test_start_refresh_stop_with_an_injected_source():
    plugin = Ros2NodeGraphPlugin({"port": 0}, executor=None,
                                 graph_source=FakeSource())
    try:
        started = plugin.dispatch("start", {})
        assert started["state"] == "running"
        assert started["stats"]["topics"] == 2
        refreshed = plugin.dispatch("refresh", {})
        assert refreshed["state"] == "running"
        assert "/bumi/joints" in refreshed["summary"]
        assert plugin.dispatch("stop", {})["state"] == "idle"
        assert plugin.dispatch("info", {})["state"] == "idle"
    finally:
        plugin.shutdown()

