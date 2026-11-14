#!/usr/bin/env python3
"""Read-only ROS graph, image and loopback Card diagnostics; never starts a driver."""

import argparse
import itertools
import json
import os
import time
import urllib.request
from pathlib import Path
from uuid import uuid4

NAME = "tabletop_roi_goal_verifier"


def web_checks(url):
    checks = {}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def fetch(path):
        with opener.open(url.rstrip("/") + path, timeout=4) as response:
            return response.read()

    def rpc(method, params=None):
        body = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
        ).encode()
        req = urllib.request.Request(
            url.rstrip("/") + "/mcp",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with opener.open(req, timeout=5) as response:
            result = json.load(response)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["result"]

    for key, fn in (
        ("service", lambda: json.loads(fetch("/health"))),
        ("browser", lambda: {"accessible": b"<canvas" in fetch("/")}),
        ("tools", lambda: rpc("tools/list")),
        (
            "card",
            lambda: json.loads(
                rpc("tools/call", {"name": NAME, "arguments": {"action": "info"}})[
                    "content"
                ][0]["text"]
            ),
        ),
    ):
        try:
            checks[key] = {"ok": True, "data": fn()}
        except Exception as exc:  # noqa: BLE001 — report each independent diagnostic failure
            checks[key] = {"ok": False, "error": str(exc)}
    if checks.get("tools", {}).get("ok"):
        checks["tools"]["ok"] = [
            t["name"] for t in checks["tools"]["data"]["tools"]
        ] == [NAME]
    if checks.get("browser", {}).get("ok"):
        checks["browser"]["ok"] = checks["browser"]["data"]["accessible"]
    if checks.get("card", {}).get("ok"):
        card = checks["card"]["data"]
        checks["card"]["ok"] = card.get("state") == "running" and not card.get(
            "source_errors"
        )
    return checks


def ros_checks(binding, timeout):
    import cv2
    import numpy as np
    import rclpy
    from rclpy.context import Context
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CompressedImage, Image

    context = Context()
    rclpy.init(context=context, domain_id=binding["domain_id"])
    node = Node("tabletop_roi_readonly_doctor_" + uuid4().hex[:8], context=context)
    samples = []
    errors = []

    def receive(msg):
        try:
            if binding["message_type"] == "compressed":
                image = cv2.imdecode(
                    np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR
                )
            else:
                if msg.encoding not in ("bgr8", "rgb8"):
                    raise ValueError("unsupported encoding " + msg.encoding)
                image = (
                    np.frombuffer(bytes(msg.data), np.uint8)
                    .reshape(msg.height, msg.step)[:, : msg.width * 3]
                    .reshape(msg.height, msg.width, 3)
                )
            if image is None:
                raise ValueError("image decode failed")
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            samples.append(
                {
                    "captured_at": stamp,
                    "host_time": time.time(),
                    "age_s": time.time() - stamp,
                    "frame_id": msg.header.frame_id,
                    "image_size": [image.shape[1], image.shape[0]],
                    "decoded": True,
                }
            )
        except Exception as exc:  # noqa: BLE001 — report each independent diagnostic failure
            errors.append(str(exc))

    typ = CompressedImage if binding["message_type"] == "compressed" else Image
    node.create_subscription(
        typ,
        binding["topic"],
        receive,
        QoSProfile(depth=3, reliability=ReliabilityPolicy.BEST_EFFORT),
    )
    try:
        deadline = time.monotonic() + timeout
        topics = {}
        while time.monotonic() < deadline and len(samples) < 3:
            rclpy.spin_once(node, timeout_sec=0.15)
            topics = dict(node.get_topic_names_and_types())
        expected = (
            "sensor_msgs/msg/CompressedImage"
            if binding["message_type"] == "compressed"
            else "sensor_msgs/msg/Image"
        )
        publisher_nodes = [
            {"node_name": x.node_name, "node_namespace": x.node_namespace}
            for x in node.get_publishers_info_by_topic(binding["topic"])
        ]
        checks = {
            "domain": {
                "ok": context.get_domain_id() == binding["domain_id"],
                "actual": context.get_domain_id(),
                "configured": binding["domain_id"],
                "environment": os.environ.get("ROS_DOMAIN_ID"),
            },
            "topic": {
                "ok": binding["topic"] in topics,
                "topic": binding["topic"],
                "types": topics.get(binding["topic"], []),
                "publishers": publisher_nodes,
            },
            "message_type": {
                "ok": expected in topics.get(binding["topic"], []),
                "expected": expected,
            },
            "decode": {"ok": bool(samples), "samples": samples, "errors": errors},
            "frame_id": {
                "ok": bool(samples)
                and all(x["frame_id"] == binding["frame_id"] for x in samples),
                "expected": binding["frame_id"],
            },
            "capture_clock": {
                "ok": len(samples) >= 2
                and all(0 <= x["age_s"] <= 2 for x in samples)
                and all(
                    b["captured_at"] > a["captured_at"]
                    for a, b in itertools.pairwise(samples)
                ),
                "note": "Only checks increasing header timestamps and host clock proximity; capture semantics still require driver/operator evidence.",
            },
            "camera_identity": {
                "ok": False,
                "status": "MANUAL_VERIFICATION_REQUIRED",
                "configured_camera_id": binding.get("camera_id"),
                "note": "frame_id and image topic cannot prove D435 model/serial or fixed camera pose.",
            },
        }
        return checks
    finally:
        node.destroy_node()
        context.shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:18084")
    p.add_argument("--timeout", type=float, default=8)
    p.add_argument(
        "--no-ros",
        action="store_true",
        help="Software checks only; cannot pass field camera acceptance",
    )
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    binding = next(iter(config["sources"].values()))
    out = {
        "read_only": True,
        "hardware_acceptance": False,
        "url": args.url,
        "checks": web_checks(args.url),
    }
    if not args.no_ros:
        try:
            out["checks"].update(ros_checks(binding, args.timeout))
        except Exception as exc:  # noqa: BLE001 — report each independent diagnostic failure
            out["checks"]["ros"] = {"ok": False, "error": str(exc)}
    else:
        out["checks"]["ros"] = {"ok": False, "reason": "NOT_RUN_SOFTWARE_ONLY"}
    if os.environ.get("ROS_DOMAIN_ID") not in (None, str(binding["domain_id"])):
        out["checks"]["domain_environment"] = {
            "ok": False,
            "error": "ROS_DOMAIN_ID differs from config",
        }
    operational = [v["ok"] for k, v in out["checks"].items() if k != "camera_identity"]
    out["operational_ready"] = all(operational)
    out["manual_camera_identity_and_fixed_view_required"] = True
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["operational_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
