#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from ros2_gate_d import CoreClient, INPUT_QOS, OUTPUT_QOS, action_hash, wait_until


class GateENode(Node):
    def __init__(self, topics: dict[str, dict[str, str]]):
        super().__init__("vlapi05g1_gate_e_multiinstance")
        self.outputs: dict[str, list[dict]] = {key: [] for key in topics}
        self.image_publishers = {}
        self.state_publishers = {}
        self.output_subscriptions = []
        for key, item in topics.items():
            self.image_publishers[key] = self.create_publisher(
                CompressedImage, item["image"], INPUT_QOS
            )
            self.state_publishers[key] = self.create_publisher(
                String, item["state"], INPUT_QOS
            )
            self.output_subscriptions.append(
                self.create_subscription(
                    String,
                    item["output"],
                    lambda message, instance_key=key: self.outputs[instance_key].append(
                        json.loads(message.data)
                    ),
                    OUTPUT_QOS,
                )
            )


def expected_node_name(instance_id: str) -> str:
    normalized = "".join(character if character.isalnum() or character == "_" else "_" for character in instance_id)
    suffix = hashlib.sha256(instance_id.encode("utf-8")).hexdigest()[:8]
    return f"vlapi05g1_{normalized}_{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run vlapi05g1 Gate E multi-instance isolation")
    parser.add_argument("--core-url", default="https://127.0.0.1:15678")
    parser.add_argument("--server-name", required=True)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-action-sha256", required=True)
    args = parser.parse_args()

    image_bytes = args.image.read_bytes()
    state_values = json.loads(args.state.read_text(encoding="utf-8"))
    if not isinstance(state_values, list) or len(state_values) != 29:
        raise ValueError("recorded state must contain 29 values")
    state_payload = json.dumps(
        {"joints": [{"idx": index, "q": value} for index, value in enumerate(state_values)]},
        separators=(",", ":"),
    )

    instance_ids = {"a": "stage6-gate-e-a", "b": "stage6-gate-e-b"}
    topics = {
        key: {
            "image": f"/stage6/gate_e/{key}/camera/compressed",
            "state": f"/stage6/gate_e/{key}/g1/joints",
            "output": f"/stage6/gate_e/{key}/vla/action_proposal",
        }
        for key in instance_ids
    }
    core = CoreClient(args.core_url, args.server_name)
    core.save_shared_config(
        {
            "policy_url": "http://127.0.0.1:18080/predict",
            "health_url": "",
            "request_timeout_s": 120.0,
            "max_image_bytes": 8388608,
        }
    )
    for instance_id in instance_ids.values():
        core.call({"action": "stop", "instance_id": instance_id})

    started: set[str] = set()
    rclpy.init()
    node = GateENode(topics)
    try:
        for key, instance_id in instance_ids.items():
            result = core.call(
                {
                    "action": "start",
                    "instance_id": instance_id,
                    "image_topic": topics[key]["image"],
                    "state_topic": topics[key]["state"],
                    "output_topic": topics[key]["output"],
                }
            )
            started.add(instance_id)
            if result.get("state") != "running":
                raise AssertionError(f"instance {key} did not start: {result}")

        expected_nodes = {expected_node_name(instance_id) for instance_id in instance_ids.values()}
        wait_until(
            node,
            lambda: expected_nodes.issubset(set(node.get_node_names())),
            8.0,
            "two distinct vlapi05g1 ROS nodes",
        )
        wait_until(
            node,
            lambda: all(
                node.image_publishers[key].get_subscription_count() >= 1
                and node.state_publishers[key].get_subscription_count() >= 1
                and node.count_publishers(topics[key]["output"]) >= 1
                for key in instance_ids
            ),
            8.0,
            "multi-instance ROS2 topic discovery",
        )

        all_info = core.call({"action": "info"})
        if set((all_info.get("instances") or {}).keys()) != set(instance_ids.values()):
            raise AssertionError(f"Core does not report exactly two instances: {all_info}")

        missing_b = core.call({"action": "predict", "instance_id": instance_ids["b"]})
        if missing_b.get("error_code") != "observation_missing":
            raise AssertionError(f"instance B unexpectedly received instance A data: {missing_b}")

        proposals = {}
        for key in ("a", "b"):
            image_message = CompressedImage()
            image_message.header.frame_id = f"stage6_gate_e_{key}_camera"
            image_message.header.stamp = node.get_clock().now().to_msg()
            image_message.format = "jpeg"
            image_message.data = list(image_bytes)
            state_message = String()
            state_message.data = state_payload
            for _ in range(5):
                node.image_publishers[key].publish(image_message)
                node.state_publishers[key].publish(state_message)
                rclpy.spin_once(node, timeout_sec=0.08)

            other_key = "b" if key == "a" else "a"
            other_count_before = len(node.outputs[other_key])
            predict = core.call({"action": "predict", "instance_id": instance_ids[key]})
            if predict.get("status") != "published":
                raise AssertionError(f"instance {key} did not publish: {predict}")
            wait_until(
                node,
                lambda instance_key=key: len(node.outputs[instance_key]) == 1,
                8.0,
                f"instance {key} output",
            )
            if len(node.outputs[other_key]) != other_count_before:
                raise AssertionError(f"instance {key} prediction leaked to instance {other_key}")
            proposal = node.outputs[key][0]
            expected_request_prefix = f"vlapi05g1-{instance_ids[key]}-"
            if not str(proposal.get("request_id", "")).startswith(expected_request_prefix):
                raise AssertionError(f"instance {key} output has wrong request_id prefix")
            observation = proposal.get("observation") or {}
            if observation.get("image_topic") != topics[key]["image"]:
                raise AssertionError(f"instance {key} output has wrong image topic")
            if observation.get("state_topic") != topics[key]["state"]:
                raise AssertionError(f"instance {key} output has wrong state topic")
            if action_hash(proposal["action_chunk"]) != args.expected_action_sha256:
                raise AssertionError(f"instance {key} action hash differs from Gate C")
            proposals[key] = proposal

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"proposals": proposals}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "mcp_id": core.mcp_id,
                    "instances": instance_ids,
                    "ros_nodes": sorted(expected_nodes),
                    "topics": topics,
                    "request_ids": {
                        key: proposals[key]["request_id"] for key in instance_ids
                    },
                    "action_sha256": args.expected_action_sha256,
                    "instance_b_before_input": missing_b["error_code"],
                    "output_counts": {
                        key: len(node.outputs[key]) for key in instance_ids
                    },
                    "output_path": str(args.output),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        for instance_id in instance_ids.values():
            if instance_id in started:
                first_stop = core.call({"action": "stop", "instance_id": instance_id})
                second_stop = core.call({"action": "stop", "instance_id": instance_id})
                if first_stop != second_stop or first_stop.get("state") != "idle":
                    raise AssertionError(f"cleanup stop is not idempotent for {instance_id}")
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
