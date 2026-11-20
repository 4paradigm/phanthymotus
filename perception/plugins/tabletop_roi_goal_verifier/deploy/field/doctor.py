#!/usr/bin/env python3
"""Read-only ROS graph, image and loopback Card diagnostics; never starts a driver."""

import argparse
import json
import math
import os
import sys
import time
import traceback
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


def failure(exc, stage):
    return {
        "ok": False,
        "code": "ROS_" + stage.upper() + "_FAILED",
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def ros_checks(binding, timeout):
    # Explicit ownership: never initialize or spin the process-global context.
    if type(binding.get("domain_id")) is not int or binding["domain_id"] != 42:
        return {
            "ros": {
                "ok": False,
                "code": "ROS_DOMAIN_42_REQUIRED",
                "error": "Domain 222 camera is outside the platform domain; no DDS participant created",
            }
        }
    context = executor = node = None
    checks = {}
    stage = "import"
    try:
        import cv2
        import numpy as np
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import (
            QoSCompatibility,
            QoSProfile,
            ReliabilityPolicy,
            qos_check_compatible,
        )
        from sensor_msgs.msg import CompressedImage, Image

        checks["ros_runtime"] = {
            "ok": True,
            "python": sys.executable,
            "version": sys.version,
            "rclpy_file": rclpy.__file__,
            "numpy_file": np.__file__,
            "dds_profile": os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE"),
            "rmw_implementation": os.environ.get("RMW_IMPLEMENTATION"),
        }
        stage = "init"
        context = Context()
        rclpy.init(context=context, domain_id=42)
        stage = "executor"
        executor = SingleThreadedExecutor(context=context)
        stage = "node"
        node = Node(
            "tabletop_roi_readonly_doctor_" + uuid4().hex[:8],
            context=context,
            enable_rosout=False,
            start_parameter_services=False,
        )
        executor.add_node(node)
        samples, errors = [], []
        received = [0]

        def receive(msg):
            received[0] += 1
            try:
                if binding["message_type"] == "compressed":
                    if (
                        "jpeg" not in msg.format.lower()
                        or len(msg.data) > 8 * 1024 * 1024
                    ):
                        raise ValueError("Expected JPEG CompressedImage (max 8 MiB)")
                    image = cv2.imdecode(
                        np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR
                    )
                else:
                    if msg.encoding not in ("bgr8", "rgb8"):
                        raise ValueError("Unsupported encoding " + msg.encoding)
                    image = (
                        np.frombuffer(bytes(msg.data), np.uint8)
                        .reshape(msg.height, msg.step)[:, : msg.width * 3]
                        .reshape(msg.height, msg.width, 3)
                    )
                if image is None or not image.size:
                    raise ValueError("Image decode failed")
            except Exception as exc:  # noqa: BLE001 — diagnostic callback retains cause
                errors.append({"code": "IMAGE_DECODE_FAILED", "error": str(exc)})
                return
            sec, nano = msg.header.stamp.sec, msg.header.stamp.nanosec
            valid = (
                type(sec) is int
                and type(nano) is int
                and sec >= 0
                and 0 <= nano < 10**9
            )
            stamp = sec + nano / 1e9 if valid else None
            host = time.time()
            valid = valid and stamp > 0 and math.isfinite(stamp)
            samples.append(
                {
                    "captured_at": stamp,
                    "host_time": host,
                    "age_s": host - stamp if valid else None,
                    "stamp_valid": valid,
                    "frame_id": msg.header.frame_id,
                    "image_size": [image.shape[1], image.shape[0]],
                    "decoded": True,
                }
            )

        stage = "subscribe"
        typ = CompressedImage if binding["message_type"] == "compressed" else Image
        subscription_qos = QoSProfile(
            depth=3, reliability=ReliabilityPolicy.BEST_EFFORT
        )
        node.create_subscription(
            typ,
            binding["topic"],
            receive,
            subscription_qos,
        )
        stage = "spin"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and len(samples) < 3:
            executor.spin_once(
                timeout_sec=min(0.15, max(0, deadline - time.monotonic()))
            )
        stage = "graph"
        topics = dict(node.get_topic_names_and_types())
        expected = "sensor_msgs/msg/" + (
            "CompressedImage" if binding["message_type"] == "compressed" else "Image"
        )
        endpoints = node.get_publishers_info_by_topic(binding["topic"])
        publisher_types = {x.topic_type for x in endpoints}
        compatible = [
            qos_check_compatible(x.qos_profile, subscription_qos)
            for x in endpoints
            if x.topic_type == expected
        ]
        qos_ok = any(status != QoSCompatibility.ERROR for status, _ in compatible)
        publishers = [
            {
                "message_type": x.topic_type,
                "node_name": x.node_name,
                "node_namespace": x.node_namespace,
                "reliability": str(x.qos_profile.reliability),
                "durability": str(x.qos_profile.durability),
            }
            for x in endpoints
        ]
        increasing = all(
            b["captured_at"] > a["captured_at"]
            for a, b in zip(samples, samples[1:])
            if a["stamp_valid"] and b["stamp_valid"]
        )
        clock_ok = (
            len(samples) >= 2
            and all(x["stamp_valid"] and 0 <= x["age_s"] <= 2 for x in samples)
            and increasing
        )
        if not publishers:
            code = "TOPIC_NO_PUBLISHER_DISCOVERED"
        elif expected not in publisher_types:
            code = "MESSAGE_TYPE_MISMATCH"
        elif not qos_ok:
            code = "QOS_INCOMPATIBLE"
        elif not received[0]:
            code = "TOPIC_NO_NEW_FRAMES"
        elif not samples:
            code = "IMAGE_DECODE_FAILED"
        elif not clock_ok:
            code = "INVALID_OR_INSUFFICIENT_CAPTURE_TIMESTAMPS"
        elif not all(x["frame_id"] == binding["frame_id"] for x in samples):
            code = "CAMERA_FRAME_MISMATCH"
        else:
            code = "CAMERA_STREAM_OK"
        checks.update(
            {
                "ros": {
                    "ok": code == "CAMERA_STREAM_OK",
                    "code": code,
                    "received_messages": received[0],
                    "note": "No frames can mean publisher inactivity, DDS profile/domain or QoS mismatch; graph presence alone does not prove streaming.",
                },
                "domain": {
                    "ok": context.get_domain_id() == 42,
                    "actual": context.get_domain_id(),
                    "configured": 42,
                    "environment": os.environ.get("ROS_DOMAIN_ID"),
                },
                "topic": {
                    "ok": bool(publishers),
                    "topic": binding["topic"],
                    "types": topics.get(binding["topic"], []),
                    "publishers": publishers,
                },
                "message_type": {
                    "ok": expected in publisher_types,
                    "expected": expected,
                    "publisher_types": sorted(publisher_types),
                },
                "qos": {
                    "ok": qos_ok,
                    "checks": [
                        {"compatibility": str(status), "reason": reason}
                        for status, reason in compatible
                    ],
                },
                "decode": {
                    "ok": bool(samples) and not errors,
                    "samples": samples,
                    "errors": errors,
                },
                "frame_id": {
                    "ok": bool(samples)
                    and all(x["frame_id"] == binding["frame_id"] for x in samples),
                    "expected": binding["frame_id"],
                },
                "capture_clock": {
                    "ok": clock_ok,
                    "note": "Increasing header stamps and host proximity only; capture semantics require driver/operator evidence.",
                },
                "camera_identity": {
                    "ok": False,
                    "status": "MANUAL_VERIFICATION_REQUIRED",
                    "configured_camera_id": binding.get("camera_id"),
                    "note": "Topic/frame_id cannot prove camera model/serial or fixed pose.",
                },
            }
        )
    except Exception as exc:  # noqa: BLE001 — retain stage and full traceback
        checks["ros"] = failure(exc, stage)
    finally:
        # Try every owned cleanup even if construction or a previous cleanup failed.
        for label, cleanup in (
            (
                "executor_shutdown",
                lambda: executor.shutdown(timeout_sec=2) if executor else None,
            ),
            ("node_destroy", lambda: node.destroy_node() if node else None),
            (
                "context_shutdown",
                lambda: context.shutdown() if context and context.ok() else None,
            ),
        ):
            try:
                cleanup()
            except Exception as exc:  # noqa: BLE001 — do not mask the primary error
                checks[label] = failure(exc, label)
    return checks


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
            out["checks"]["ros"] = failure(exc, "diagnostic")
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
