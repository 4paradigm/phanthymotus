#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String


INPUT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)
OUTPUT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)


class CoreClient:
    def __init__(self, core_url: str, server_name: str):
        self.core_url = core_url.rstrip("/")
        self.server_name = server_name
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPSHandler(context=context),
        )
        registrations = self.request("GET", "/api/mcp")["data"]
        matches = [item for item in registrations if item.get("server_name") == server_name]
        if len(matches) != 1:
            raise RuntimeError(f"expected one Core registration for {server_name}, got {len(matches)}")
        self.mcp = matches[0]
        self.mcp_id = self.mcp["id"]

    def request(self, method: str, path: str, body=None):
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.core_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        with self.opener.open(request, timeout=30.0) as response:
            return json.loads(response.read())

    def save_shared_config(self, config: dict) -> None:
        response = self.request(
            "PUT",
            f"/api/canvas/tool-config/{self.mcp_id}/vlapi05g1",
            config,
        )
        if response.get("code") != 200:
            raise RuntimeError(f"Core tool-config save failed: {response}")

    def call(self, arguments: dict) -> dict:
        response = self.request(
            "POST",
            f"/api/mcp/{self.mcp_id}/call",
            {"tool": "vlapi05g1", "arguments": arguments},
        )
        if response.get("code") != 200:
            raise RuntimeError(f"Core tool call failed: {response}")
        content = response.get("data") or []
        if not isinstance(content, list) or not content:
            raise RuntimeError(f"Core tool call returned no content: {response}")
        return json.loads(content[0]["text"])


class GateDNode(Node):
    def __init__(self, image_topic: str, state_topic: str, output_topic: str):
        super().__init__("vlapi05g1_gate_d_replay")
        self.outputs: list[dict] = []
        self.image_publisher = self.create_publisher(CompressedImage, image_topic, INPUT_QOS)
        self.state_publisher = self.create_publisher(String, state_topic, INPUT_QOS)
        self.output_subscription = self.create_subscription(
            String,
            output_topic,
            self._on_output,
            OUTPUT_QOS,
        )

    def _on_output(self, message: String) -> None:
        self.outputs.append(json.loads(message.data))


def wait_until(node: Node, predicate, timeout_s: float, description: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if predicate():
            return
    raise TimeoutError(f"timed out waiting for {description}")


def action_hash(action_chunk: list[list[float]]) -> str:
    encoded = json.dumps(action_chunk, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run vlapi05g1 Gate D over real ROS2 topics")
    parser.add_argument("--core-url", default="https://127.0.0.1:15678")
    parser.add_argument("--server-name", required=True)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-action-sha256", required=True)
    parser.add_argument("--instance-id", default="stage6-gate-d")
    args = parser.parse_args()

    image_bytes = args.image.read_bytes()
    state_values = json.loads(args.state.read_text(encoding="utf-8"))
    if not isinstance(state_values, list) or len(state_values) != 29:
        raise ValueError("recorded state must contain 29 values")
    state_payload = json.dumps(
        {"joints": [{"idx": index, "q": value} for index, value in enumerate(state_values)]},
        separators=(",", ":"),
    )

    image_topic = "/stage6/gate_d/camera/compressed"
    state_topic = "/stage6/gate_d/g1/joints"
    output_topic = "/stage6/gate_d/vla/action_proposal"
    frame_id = "stage6_g1_front_camera"
    core = CoreClient(args.core_url, args.server_name)
    core.save_shared_config(
        {
            "policy_url": "http://127.0.0.1:18080/predict",
            "health_url": "",
            "request_timeout_s": 120.0,
            "max_image_bytes": 8388608,
        }
    )

    start_arguments = {
        "action": "start",
        "instance_id": args.instance_id,
        "image_topic": image_topic,
        "state_topic": state_topic,
        "output_topic": output_topic,
    }
    core.call({"action": "stop", "instance_id": args.instance_id})
    started = False
    rclpy.init()
    node = GateDNode(image_topic, state_topic, output_topic)
    try:
        started_result = core.call(start_arguments)
        started = True
        if started_result.get("state") != "running":
            raise AssertionError(f"card did not start: {started_result}")
        wait_until(
            node,
            lambda: node.image_publisher.get_subscription_count() >= 1
            and node.state_publisher.get_subscription_count() >= 1
            and node.count_publishers(output_topic) >= 1,
            8.0,
            "ROS2 publisher/subscriber discovery",
        )

        image_message = CompressedImage()
        image_message.header.frame_id = frame_id
        image_message.header.stamp = node.get_clock().now().to_msg()
        image_message.format = "jpeg"
        image_message.data = list(image_bytes)
        published_image_stamp = (
            float(image_message.header.stamp.sec)
            + float(image_message.header.stamp.nanosec) / 1_000_000_000.0
        )
        state_message = String()
        state_message.data = state_payload

        for _ in range(5):
            node.image_publisher.publish(image_message)
            node.state_publisher.publish(state_message)
            rclpy.spin_once(node, timeout_sec=0.08)

        predict_result = core.call({"action": "predict", "instance_id": args.instance_id})
        if predict_result.get("status") != "published":
            raise AssertionError(f"predict did not publish: {predict_result}")
        wait_until(node, lambda: len(node.outputs) >= 1, 8.0, "action proposal output")
        if len(node.outputs) != 1:
            raise AssertionError(f"expected one action proposal, got {len(node.outputs)}")
        proposal = node.outputs[0]

        if proposal.get("request_id") != predict_result.get("request_id"):
            raise AssertionError("MCP result and ROS output request_id differ")
        if proposal.get("schema") != "pi05.g1.action_chunk.v1":
            raise AssertionError(f"unexpected proposal schema: {proposal.get('schema')}")
        observation = proposal.get("observation") or {}
        expected_observation = {
            "image_topic": image_topic,
            "image_frame_id": frame_id,
            "image_stamp_source": "header",
            "state_topic": state_topic,
            "state": state_values,
        }
        for key, expected in expected_observation.items():
            if observation.get(key) != expected:
                raise AssertionError(f"observation.{key} mismatch")
        if abs(float(observation["image_stamp"]) - published_image_stamp) > 1e-9:
            raise AssertionError("image header stamp was not preserved")
        image_received_at = float(observation["image_received_at"])
        state_received_at = float(observation["state_received_at"])
        created_at = float(proposal["created_at"])
        if image_received_at <= 0.0 or state_received_at <= 0.0 or created_at <= 0.0:
            raise AssertionError("proposal timestamps must be positive")
        if created_at < max(image_received_at, state_received_at):
            raise AssertionError("proposal created_at predates the cached observation")
        if not 0.0 <= float(observation["image_age_at_request_s"]) <= 1.0:
            raise AssertionError("image age is outside configured freshness window")
        if not 0.0 <= float(observation["state_age_at_request_s"]) <= 0.5:
            raise AssertionError("state age is outside configured freshness window")
        if proposal.get("action_shape") != [1, 50, 18]:
            raise AssertionError(f"unexpected action shape: {proposal.get('action_shape')}")
        if proposal.get("action_space") != "physical_quantile_unnormalized":
            raise AssertionError("unexpected action space")
        if proposal.get("execution_authorized") is not False:
            raise AssertionError("proposal unexpectedly authorizes execution")
        actual_action_hash = action_hash(proposal["action_chunk"])
        if actual_action_hash != args.expected_action_sha256:
            raise AssertionError(
                f"action hash mismatch: expected {args.expected_action_sha256}, got {actual_action_hash}"
            )
        if "confidence" in proposal:
            raise AssertionError("policy does not define confidence; card must not invent one")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(proposal, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        output_count_before_stale = len(node.outputs)
        stale_deadline = time.monotonic() + 1.2
        while time.monotonic() < stale_deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        stale_result = core.call({"action": "predict", "instance_id": args.instance_id})
        if stale_result.get("error_code") != "observation_stale":
            raise AssertionError(f"stale observation was not rejected: {stale_result}")
        stale_output_deadline = time.monotonic() + 0.8
        while time.monotonic() < stale_output_deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        if len(node.outputs) != output_count_before_stale:
            raise AssertionError("stale predict unexpectedly published another proposal")

        invalid_state = String()
        invalid_state.data = "{invalid-json"
        node.state_publisher.publish(invalid_state)
        time.sleep(0.2)
        invalid_state_info = core.call({"action": "info", "instance_id": args.instance_id})
        if (invalid_state_info.get("last_error") or {}).get("error_code") != "invalid_state":
            raise AssertionError(f"invalid state did not produce invalid_state: {invalid_state_info}")

        empty_image = CompressedImage()
        empty_image.header.frame_id = frame_id
        empty_image.header.stamp = node.get_clock().now().to_msg()
        empty_image.format = "jpeg"
        empty_image.data = []
        node.image_publisher.publish(empty_image)
        time.sleep(0.2)
        invalid_image_info = core.call({"action": "info", "instance_id": args.instance_id})
        if (invalid_image_info.get("last_error") or {}).get("error_code") != "invalid_image":
            raise AssertionError(f"empty image did not produce invalid_image: {invalid_image_info}")

        wrong_type_script = """
import sys
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

rclpy.init()
node = Node("vlapi05g1_gate_d_wrong_type")
publisher = node.create_publisher(Int32, sys.argv[1], 1)
message = Int32()
message.data = 1
deadline = time.monotonic() + 0.6
while time.monotonic() < deadline:
    publisher.publish(message)
    rclpy.spin_once(node, timeout_sec=0.05)
node.destroy_node()
rclpy.shutdown()
"""
        wrong_type_result = subprocess.run(
            [sys.executable, "-c", wrong_type_script, state_topic],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if wrong_type_result.returncode != 0:
            raise AssertionError(
                f"wrong-type ROS publisher failed: {wrong_type_result.stderr.strip()}"
            )
        time.sleep(0.2)
        alive_info = core.call({"action": "info", "instance_id": args.instance_id})
        if alive_info.get("state") != "running":
            raise AssertionError("wrong ROS message type disrupted the card")

        summary = {
            "status": "PASS",
            "mcp_id": core.mcp_id,
            "instance_id": args.instance_id,
            "request_id": proposal["request_id"],
            "output_path": str(args.output),
            "action_sha256": actual_action_hash,
            "image_frame_id": observation["image_frame_id"],
            "image_stamp": observation["image_stamp"],
            "image_stamp_source": observation["image_stamp_source"],
            "confidence": "not_applicable_policy_does_not_emit_confidence",
            "stale_predict_error": stale_result["error_code"],
            "output_count_after_stale_predict": len(node.outputs),
            "invalid_state_error": invalid_state_info["last_error"]["error_code"],
            "invalid_image_error": invalid_image_info["last_error"]["error_code"],
            "wrong_message_type_card_state": alive_info["state"],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if started:
            first_stop = core.call({"action": "stop", "instance_id": args.instance_id})
            second_stop = core.call({"action": "stop", "instance_id": args.instance_id})
            if first_stop != second_stop or first_stop.get("state") != "idle":
                raise AssertionError("Gate D cleanup stop is not idempotent")
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
