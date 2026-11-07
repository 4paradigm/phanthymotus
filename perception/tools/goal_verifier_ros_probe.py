"""Actual ROS2 loopback probe with explicitly SYNTHETIC pixels; no camera/owner."""

import argparse
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(domain):
    import cv2
    import numpy as np
    import rclpy
    from rclpy.context import Context
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image, CompressedImage
    from plugins.goal_verifier.plugin import GoalVerifierPlugin
    from plugins.goal_verifier.contracts import NAME

    key = uuid4().hex[:10]
    topic = "/piper_goal_probe_" + key
    base = dict(
        camera_id="SYNTHETIC-PROBE",
        frame_id="synthetic_optical",
        distortion="raw",
        kind="ros_image",
        domain_id=domain,
        stamp_semantics="capture",
        capture_time_record="SYNTHETIC generation timestamp, not physical camera",
        view_id="synthetic-view",
        view_mode="fixed",
        fixed_view_record="SYNTHETIC ONLY",
        provenance="synthetic",
        allow_test_data=True,
        classes=[
            dict(
                class_name="red_piece",
                color="red",
                hsv_min=[0, 150, 150],
                hsv_max=[10, 255, 255],
                min_area_px=20,
            )
        ],
    )
    cfg = dict(
        piper_id="SYNTHETIC-PIPER-PROBE",
        sources={
            k: dict(base, source_id=k, topic=topic + "/" + k, message_type=k)
            for k in ("raw", "compressed")
        },
    )
    card = GoalVerifierPlugin(cfg)
    ctx = Context()
    node = None
    try:
        rclpy.init(context=ctx, domain_id=domain)
        node = Node("goal_probe_" + key, context=ctx)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        pubs = {
            k: node.create_publisher(t, topic + "/" + k, qos)
            for k, t in [("raw", Image), ("compressed", CompressedImage)]
        }
        result = card.dispatch(NAME, dict(action="start"))
        if result.get("state") != "running":
            raise RuntimeError(result)
        pixels = np.zeros((80, 120, 3), np.uint8)
        pixels[20:60, 40:80] = (0, 0, 255)
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            for k, pub in pubs.items():
                msg = Image() if k == "raw" else CompressedImage()
                stamp = time.time()
                msg.header.frame_id = base["frame_id"]
                msg.header.stamp.sec = int(stamp)
                msg.header.stamp.nanosec = int(stamp % 1 * 1e9)
                if k == "raw":
                    msg.height = 80
                    msg.width = 120
                    msg.step = 360
                    msg.encoding = "bgr8"
                    msg.data = pixels.tobytes()
                else:
                    msg.format = "jpeg"
                    msg.data = cv2.imencode(".jpg", pixels)[1].tobytes()
                pub.publish(msg)
            if len(card.sources.frames) == 2:
                break
            time.sleep(0.05)
        observations = {
            k: card.dispatch(NAME, dict(action="observe", source_id=k)) for k in pubs
        }
        for out in observations.values():
            if (
                out.get("state") != "observed"
                or out["observation"]["provenance"] != "synthetic"
                or not out["observation"]["objects"]
            ):
                raise RuntimeError(out)
        # Future stamp crosses the actual DDS callback and must poison the cache.
        bad = Image()
        bad.header.frame_id = base["frame_id"]
        bad.header.stamp.sec = int(time.time()) + 100
        bad.height = 80
        bad.width = 120
        bad.step = 360
        bad.encoding = "bgr8"
        bad.data = pixels.tobytes()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and "raw" not in card.sources.errors:
            pubs["raw"].publish(bad)
            time.sleep(0.03)
        refused = card.dispatch(NAME, dict(action="observe", source_id="raw"))
        if (
            refused.get("result") != "INCONCLUSIVE"
            or refused.get("code") != "SOURCE_INPUT_ERROR"
        ):
            raise RuntimeError(refused)
        return dict(
            state="DONE",
            real_ros2_dds=True,
            domain=domain,
            input="SYNTHETIC_PIXELS",
            real_camera=False,
            observations=observations,
            future_stamp_refused=refused,
            motion_executed=False,
        )
    finally:
        card.stop()
        if node:
            node.destroy_node()
        if ctx.ok():
            ctx.shutdown()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = run(a.domain)
    with a.output.open("x") as out:
        json.dump(result, out, indent=2)
    print(
        json.dumps(dict(state=result["state"], real_ros2_dds=True, real_camera=False))
    )
