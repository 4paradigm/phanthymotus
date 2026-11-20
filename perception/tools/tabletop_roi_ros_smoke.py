"""Bounded real-rclpy integration with synthetic images on unique loopback topics.

Run separately from pytest: perception's shared conftest installs ROS stubs.
Never opens a camera or publishes robot commands. Requires sourced ROS2 Humble.
"""

import argparse
import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perception"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    profile = Path(os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE", "/missing"))
    # Refuse unconfined DDS. Use the repository's unchanged official profile.
    if (
        not profile.is_file()
        or profile.read_bytes()
        != (ROOT / "agent-core/deploy/dds-local.xml").read_bytes()
    ):
        p.error("source ROS and select the unchanged official dds-local.xml")
    if os.environ.get("ROS_DOMAIN_ID") != "42":
        p.error("ROS_DOMAIN_ID=42 required")

    import cv2
    import numpy as np
    import rclpy
    from plugins.tabletop_roi_goal_verifier.plugin import TabletopROIGoalVerifierPlugin
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CompressedImage, Image

    spec = importlib.util.spec_from_file_location(
        "field_doctor",
        ROOT / "perception/plugins/tabletop_roi_goal_verifier/deploy/field/doctor.py",
    )
    doctor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(doctor)
    topic = "/roi_pr393_synthetic_" + uuid4().hex
    binding = {
        "source_id": "camera",
        "kind": "ros_image",
        "camera_id": "SYNTHETIC",
        "domain_id": 42,
        "topic": topic,
        "message_type": "compressed",
        "frame_id": "synthetic_optical",
        "view_id": "synthetic_fixed",
        "distortion": "raw",
        "view_mode": "fixed",
        "fixed_view_record": "SYNTHETIC ONLY",
        "stamp_semantics": "capture",
        "capture_time_record": "SYNTHETIC host wall clock",
        "provenance": "synthetic",
        "allow_test_data": True,
    }
    cfg = {"verifier_id": "SYNTHETIC-ROS-SMOKE", "sources": {"camera": binding}}
    context = Context()
    rclpy.init(context=context, domain_id=42)
    executor = SingleThreadedExecutor(context=context)
    node = Node(
        "roi_pr393_synthetic_" + uuid4().hex[:8],
        context=context,
        enable_rosout=False,
        start_parameter_services=False,
    )
    executor.add_node(node)
    publisher = node.create_publisher(
        CompressedImage,
        topic,
        QoSProfile(depth=3, reliability=ReliabilityPolicy.BEST_EFFORT),
    )
    pixels = np.full((240, 320, 3), 150, np.uint8)
    pixels[100:150, 200:240] = (20, 40, 220)
    ok, jpeg = cv2.imencode(".jpg", pixels)
    assert ok
    mode = ["valid"]
    halt = threading.Event()
    errors = []

    def stream():
        try:
            while not halt.wait(0.04):
                if mode[0] != "silent":
                    msg = CompressedImage()
                    stamp = time.time_ns() if mode[0] != "invalid_stamp" else 0
                    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(
                        stamp, 10**9
                    )
                    msg.header.frame_id = binding["frame_id"]
                    msg.format = "jpeg"
                    msg.data = list(
                        jpeg.tobytes() if mode[0] != "corrupt" else b"bad JPEG"
                    )
                    publisher.publish(msg)
                executor.spin_once(timeout_sec=0.01)
        except Exception as exc:  # noqa: BLE001 — propagate worker failures to assertions
            errors.append(repr(exc))

    thread = threading.Thread(target=stream)
    card = TabletopROIGoalVerifierPlugin(cfg)
    results = {}
    thread.start()
    try:
        assert not rclpy.ok(), "private contexts must not initialize the global context"
        # Doctor must work with a private context even when the global one is unset.
        checks = doctor.ros_checks(binding, 3)
        assert checks["ros"]["code"] == "CAMERA_STREAM_OK", checks
        assert checks["decode"]["samples"][0]["image_size"] == [320, 240]
        results["compressed_private_context"] = checks
        assert card.dispatch(card.PREFIX, {"action": "start"})["state"] == "running"
        packet = card.sources.read_image("camera", timeout=3)
        assert packet["frame"]["provenance"] == "synthetic"
        assert packet["frame"]["frame_id"] == binding["frame_id"]
        results["sources_real_subscription"] = True
        card.stop()
        assert card.dispatch(card.PREFIX, {"action": "start"})["state"] == "running"
        card.sources.read_image("camera", timeout=3)
        results["stop_restart"] = True
        mode[0] = "silent"
        checks = doctor.ros_checks(binding, 0.6)
        assert checks["ros"]["code"] == "TOPIC_NO_NEW_FRAMES", checks
        results["topic_without_frames"] = checks
        mode[0] = "corrupt"
        checks = doctor.ros_checks(binding, 0.6)
        assert checks["ros"]["code"] == "IMAGE_DECODE_FAILED", checks
        results["jpeg_decode_failure"] = checks
        mode[0] = "invalid_stamp"
        checks = doctor.ros_checks(binding, 2)
        assert checks["ros"]["code"] == "INVALID_OR_INSUFFICIENT_CAPTURE_TIMESTAMPS", (
            checks
        )
        results["invalid_capture_timestamp"] = checks
        mode[0] = "valid"
        checks = doctor.ros_checks({**binding, "topic": topic + "_no_publisher"}, 0.6)
        assert checks["ros"]["code"] == "TOPIC_NO_PUBLISHER_DISCOVERED", checks
        results["no_publisher"] = checks
        wrong_topic = topic + "_wrong_type"
        wrong = node.create_publisher(Image, wrong_topic, 3)
        checks = doctor.ros_checks({**binding, "topic": wrong_topic}, 0.6)
        assert checks["ros"]["code"] == "MESSAGE_TYPE_MISMATCH", checks
        node.destroy_publisher(wrong)
        results["wrong_publisher_type"] = checks
        # Exercise cleanup after real Context init but failed executor creation.
        from unittest.mock import patch

        import rclpy.context as contexts
        from rclpy import executors

        created = []
        real_context = contexts.Context

        def new_context():
            value = real_context()
            created.append(value)
            return value

        failed = TabletopROIGoalVerifierPlugin(cfg)
        with patch.object(contexts, "Context", new_context), patch.object(
            executors,
            "MultiThreadedExecutor",
            side_effect=RuntimeError("SYNTHETIC executor construction failure"),
        ):
            out = failed.dispatch(failed.PREFIX, {"action": "start"})
        assert out["result"] == "INCONCLUSIVE", out
        assert created and not created[0].ok()
        assert failed.sources.ros is None and not failed.engine.running
        failed.stop()
        results["partial_initialization_cleanup"] = True
        assert not rclpy.ok() and not errors, errors
        results["global_context_untouched"] = True
    finally:
        card.stop()
        halt.set()
        thread.join(3)
        executor.shutdown(timeout_sec=2)
        node.destroy_node()
        context.shutdown()
    results["hardware_acceptance"] = False
    results["rclpy_file"] = rclpy.__file__
    results["python"] = sys.version
    args.output.write_text(json.dumps(results, indent=2))
    print("10 real-rclpy synthetic checks passed; no camera/hardware acceptance")


if __name__ == "__main__":
    main()
