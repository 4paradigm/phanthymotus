#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from ros2_gate_d import CoreClient, INPUT_QOS, wait_until


class GateFNode(Node):
    def __init__(self, image_topic: str, state_topic: str):
        super().__init__("vlapi05g1_gate_f_policy_error")
        self.image_publisher = self.create_publisher(CompressedImage, image_topic, INPUT_QOS)
        self.state_publisher = self.create_publisher(String, state_topic, INPUT_QOS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify policy failures do not stop vlapi05g1 MCP")
    parser.add_argument("--core-url", default="https://127.0.0.1:15678")
    parser.add_argument("--server-name", required=True)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--instance-id", default="stage6-gate-f-policy-error")
    args = parser.parse_args()

    image_bytes = args.image.read_bytes()
    state_values = json.loads(args.state.read_text(encoding="utf-8"))
    state_payload = json.dumps(
        {"joints": [{"idx": index, "q": value} for index, value in enumerate(state_values)]},
        separators=(",", ":"),
    )
    image_topic = "/stage6/gate_f/camera/compressed"
    state_topic = "/stage6/gate_f/g1/joints"
    output_topic = "/stage6/gate_f/vla/action_proposal"
    core = CoreClient(args.core_url, args.server_name)
    core.call({"action": "stop", "instance_id": args.instance_id})
    core.save_shared_config(
        {
            "policy_url": "http://127.0.0.1:1/predict",
            "health_url": "",
            "request_timeout_s": 0.2,
            "max_image_bytes": 8388608,
        }
    )

    started = False
    rclpy.init()
    node = GateFNode(image_topic, state_topic)
    try:
        start = core.call(
            {
                "action": "start",
                "instance_id": args.instance_id,
                "input_topics": [image_topic, state_topic],
                "output_topic": output_topic,
            }
        )
        started = True
        if start.get("state") != "running":
            raise AssertionError(f"card did not start: {start}")
        wait_until(
            node,
            lambda: node.image_publisher.get_subscription_count() >= 1
            and node.state_publisher.get_subscription_count() >= 1,
            8.0,
            "Gate F ROS2 subscriptions",
        )

        image_message = CompressedImage()
        image_message.header.frame_id = "stage6_gate_f_camera"
        image_message.header.stamp = node.get_clock().now().to_msg()
        image_message.format = "jpeg"
        image_message.data = list(image_bytes)
        state_message = String()
        state_message.data = state_payload
        for _ in range(5):
            node.image_publisher.publish(image_message)
            node.state_publisher.publish(state_message)
            rclpy.spin_once(node, timeout_sec=0.08)

        predict = core.call({"action": "predict", "instance_id": args.instance_id})
        if predict.get("error_code") != "policy_unreachable":
            raise AssertionError(f"policy failure was not isolated: {predict}")
        alive = core.call({"action": "info", "instance_id": args.instance_id})
        if alive.get("state") != "running":
            raise AssertionError(f"MCP/card stopped after policy failure: {alive}")
        if (alive.get("last_error") or {}).get("error_code") != "policy_unreachable":
            raise AssertionError(f"policy failure was not retained as structured status: {alive}")
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "mcp_id": core.mcp_id,
                    "instance_id": args.instance_id,
                    "predict_error": predict["error_code"],
                    "card_state_after_error": alive["state"],
                    "last_error": alive["last_error"]["error_code"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        if started:
            core.call({"action": "stop", "instance_id": args.instance_id})
        core.save_shared_config(
            {
                "policy_url": "http://127.0.0.1:18080/predict",
                "health_url": "",
                "request_timeout_s": 120.0,
                "max_image_bytes": 8388608,
            }
        )
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
