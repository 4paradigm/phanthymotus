#!/usr/bin/env python3
"""Publish a real local image/video and capture VOP JSON + annotated JPEG over DDS."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



def setup_monitor(core_url, mcp_url, topic, instance_id):
    """Opt-in: start a real replay instance and append its card without replacing a canvas."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def request(url, data=None):
        req = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=30) as response:
            body = json.load(response)
        if body.get("error") or body.get("code", 200) != 200:
            raise RuntimeError(body)
        return body
    started = request(mcp_url, {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
        "name": "vop", "arguments": {"action": "start", "instance_id": instance_id, "input_topic": topic}}})
    result = json.loads(started["result"]["content"][0]["text"])
    if result.get("error") or result.get("state") in ("error", "idle", "stopping"):
        raise RuntimeError(result)
    base = core_url.rstrip("/")
    request(base + "/api/mcp", {"name": "VOP replay perception", "transport": "http",
                                 "url": mcp_url, "category": "perception"})
    mcp = next(m for m in request(base + "/api/mcp")["data"] if m.get("url") == mcp_url)
    layout = request(base + "/api/canvas/layout").get("data") or {}
    cards = layout.setdefault("cards", [])
    existing = next((c for c in cards if c.get("id") == instance_id), None)
    if existing and (existing.get("mcpId") != mcp["id"] or existing.get("toolName") != "vop"):
        raise RuntimeError("Instance ID belongs to a different canvas card")
    if not existing:
        cards.append({"id": instance_id, "mcpId": mcp["id"], "toolName": "vop", "type": "processor",
                      "x": 100, "y": 100, "topicIn": [{"topic": topic, "format": "image/jpeg"}],
                      "topicOut": [{"topic": topic + "/objects", "format": "data/json"},
                                   {"topic": topic + "/objects/preview", "format": "image/jpeg"}]})
        owner = "vop-replay-setup"
        request(base + "/api/canvas/claim-edit", {"session_id": owner})
        try:
            request(base + "/api/canvas/layout", {**layout, "session_id": owner})
        finally:
            request(base + "/api/canvas/release-edit", {"session_id": owner})
    print(json.dumps({"monitor": base, "instance": instance_id, "topic": topic}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Local image/video, or 'bus' for the Ultralytics example image")
    parser.add_argument("--topic", default="/vop_replay/camera")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=60, help="Seconds; 0 means until Ctrl-C")
    parser.add_argument("--fps", type=float, default=2)
    parser.add_argument("--core-url", help="Opt in to registering the MCP and adding a monitor card")
    parser.add_argument("--mcp-url", help="Perception endpoint reachable from both replay and Core")
    parser.add_argument("--instance-id", default="vop-replay")
    args = parser.parse_args()
    if bool(args.core_url) != bool(args.mcp_url):
        parser.error("Supply both --core-url and --mcp-url, or neither")
    if args.fps <= 0 or args.duration < 0:
        parser.error("fps must be positive and duration nonnegative")
    from utils.cv2_compat import load_cv2
    import rclpy
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import String
    from rclpy.qos import qos_profile_sensor_data
    cv2 = load_cv2()
    if args.source == "bus":
        from ultralytics.utils import ASSETS
        source = ASSETS / "bus.jpg"
    else:
        source = Path(args.source)
    if not source.is_file():
        parser.error(f"Not a local file: {source}")
    frame = cv2.imread(str(source))
    video = None if frame is not None else cv2.VideoCapture(str(source))
    if frame is None and not video.isOpened():
        parser.error("Cannot decode source")
    if args.core_url:
        setup_monitor(args.core_url, args.mcp_url, args.topic, args.instance_id)
    args.output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = rclpy.create_node("vop_replay")
    pub = node.create_publisher(CompressedImage, args.topic, qos_profile_sensor_data)
    records = (args.output / "results.ndjson").open("a")
    successful = 0
    def result(msg):
        nonlocal successful
        data = json.loads(msg.data)
        records.write(json.dumps(data, ensure_ascii=False) + "\n"); records.flush()
        if data.get("status") == "ok":
            successful += 1
            (args.output / "latest.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
    def preview(msg):
        (args.output / "latest.jpg").write_bytes(bytes(msg.data))
        key = f"{msg.header.stamp.sec}-{msg.header.stamp.nanosec}"
        # Bounded capture: latest only plus the first 20 source frames, including failure banners.
        existing = list(args.output.glob("frame-*.jpg"))
        path = args.output / f"frame-{key}.jpg"
        if path.exists() or len(existing) < 20:
            path.write_bytes(bytes(msg.data))
    node.create_subscription(String, args.topic + "/objects", result, qos_profile_sensor_data)
    node.create_subscription(CompressedImage, args.topic + "/objects/preview", preview, qos_profile_sensor_data)
    start, next_frame = time.monotonic(), 0.0
    try:
        while rclpy.ok() and (args.duration == 0 or time.monotonic() - start < args.duration):
            if time.monotonic() >= next_frame:
                if video is not None:
                    ok, frame = video.read()
                    if not ok:
                        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = video.read()
                        if not ok: raise RuntimeError("Video contains no decodable frames")
                ok, encoded = cv2.imencode(".jpg", frame)
                if not ok: raise RuntimeError("Input JPEG encoding failed")
                msg = CompressedImage()
                msg.header.stamp = node.get_clock().now().to_msg()
                msg.header.frame_id = "vop_replay_camera_optical_frame"
                msg.format, msg.data = "jpeg", encoded.tobytes()
                pub.publish(msg)
                next_frame = time.monotonic() + 1 / args.fps
            rclpy.spin_once(node, timeout_sec=0.02)
    except KeyboardInterrupt:
        pass
    finally:
        if video is not None: video.release()
        records.close(); node.destroy_node(); rclpy.shutdown()
    print(json.dumps({"successful_frames": successful, "evidence": str(args.output)}))
    if successful == 0:
        raise SystemExit("No successful VOP results received; this replay has not passed")


if __name__ == "__main__":
    main()
