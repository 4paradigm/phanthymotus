"""Local Dashboard replay with a test head or an explicitly supplied local engine."""
import argparse
import json
import ssl
import threading
import time
import urllib.error
import urllib.request
import zlib
from itertools import cycle
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import CompressedImage

import main
from plugins.segment import SegmentationPlugin
from plugins.vision_runtime import LetterboxMeta
from utils.qos import CAMERA_QOS


class FixtureEngine:
    input_size = (64, 64)

    def infer(self, image):
        assert image.shape[:2] == (64, 64)
        rows = np.array([[[0, 0, 64, 64, .99, 0, 1, 0]]], np.float32)
        prototypes = np.ones((1, 2, 8, 8), np.float32)
        return (rows, prototypes), LetterboxMeta(1, 0, 0, 64, 64)


def local_segmentation(executor, engine_path):
    from plugins.vop import VideoObjectPerceptionPlugin

    class LocalVop(VideoObjectPerceptionPlugin):
        def _prefetch_vocabulary(self):
            self._vocabulary = self._read_vocab(engine_path.with_name("vocab.json"))

        def _resolve_engine(self):
            return str(engine_path), self._read_vocab(engine_path.with_name("vocab.json"))

    # Keep production TensorRT session creation and vocabulary precedence.
    # The lab replay only replaces the bundle resolver and its prefetch.
    provider = LocalVop({}, "fusion_test", executor)
    return SegmentationPlugin({}, executor, provider.segmentation_model, provider.model_load_message)


class Replay:
    PREFIX = "fusion_test_replay"
    topics = [
        {"topic": "/fusion_test/camera/color", "format": "image/jpeg"},
        {"topic": "/fusion_test/camera/depth", "format": "image/depth-zlib"},
    ]

    def __init__(self, executor, recording=None):
        self.node = rclpy.create_node("fusion_test_replay")
        executor.add_node(self.node)
        self.pubs = [self.node.create_publisher(CompressedImage, p["topic"], CAMERA_QOS)
                     for p in self.topics]
        self.recording = recording
        self.model = dict(schema="motus.camera/1", id="fusion_test_camera", width=64, height=64,
                          distortion_model="pinhole", K=[60, 0, 32, 0, 60, 32, 0, 0, 1],
                          vendor={"depth_aligned_to": "color"})
        if recording is None:
            rgb = np.full((64, 64, 3), (20, 100, 170), np.uint8)
            self.packets = (cv2.imencode(".jpg", rgb)[1].tobytes(),
                            zlib.compress(np.full((64, 64), 2000, "<u2").tobytes()))
        else:
            manifest = json.loads((recording / "manifest.json").read_text(encoding="utf-8"))
            self.model.update({key: manifest[key] for key in ("id", "width", "height", "K")})
            if not manifest["frames"]:
                raise ValueError("recording must contain at least one paired frame")
            self.frames = cycle(manifest["frames"])
        self.running = False
        self.paused = False
        self.node.create_timer(.15, self.publish)

    def publish(self):
        if self.running:
            if self.recording is None:
                packets = self.packets
            else:
                frame = next(self.frames)
                jpg = (self.recording / frame["rgb"]).read_bytes()
                depth = np.load(self.recording / frame["depth"], allow_pickle=False)
                if (depth.shape != (self.model["height"], self.model["width"])
                        or depth.dtype.kind != "u" or depth.dtype.itemsize != 2):
                    raise ValueError("recorded depth must match the declared Z16 color grid")
                packets = jpg, zlib.compress(depth.astype("<u2", copy=False).tobytes())
            stamp = self.node.get_clock().now().to_msg()
            encodings = ("jpeg", "16UC1; compressedDepth zlib")
            for index, (pub, encoding, packet) in enumerate(zip(self.pubs, encodings, packets)):
                if index and self.paused:
                    continue
                msg = CompressedImage(format=encoding, data=packet)
                msg.header.stamp = stamp
                msg.header.frame_id = "fusion_test_color_optical_frame"
                pub.publish(msg)

    def get_tools(self):
        description = "Synthetic RGB-D test replay" if self.recording is None else "Recorded RGB-D replay"
        return [dict(name=self.PREFIX, type="sensor", description=description,
                     inputSchema=dict(type="object", required=["action"], properties={
                         "action": {"type": "string", "enum": ["start", "stop", "info"]}}),
                     topic_out=self.topics)]

    def dispatch(self, name, args):
        action = args.get("action")
        if action == "start":
            self.running = True
        elif action == "stop":
            self.running = False
        elif action == "pause_depth":
            self.paused = True
        elif action == "resume_depth":
            self.paused = False
        return dict(state="running" if self.running else "idle", topic_out=self.topics,
                    camera_info=[dict(self.model, **port) for port in self.topics])


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", type=Path)
    parser.add_argument("--engine", type=Path)
    args = parser.parse_args()
    if bool(args.recording) != bool(args.engine):
        parser.error("--recording and --engine must be supplied together")
    if args.engine and (args.engine.suffix != ".engine" or not args.engine.is_file()
                        or not args.engine.with_name("vocab.json").is_file()):
        parser.error("an existing .engine and its adjacent vocab.json are required")
    rclpy.init()
    executor = MultiThreadedExecutor(num_threads=3)
    threading.Thread(target=executor.spin, daemon=True).start()
    plugins = {
        "semantic_surface_fusion": {"enabled": True},
        "echo_cancellation": {"enabled": False},
        "audio_mixer": {"enabled": False},
    }
    bundle = main.PerceptionBundle({"plugins": plugins}, executor)
    bundle._plugins.append(Replay(executor, args.recording))
    if args.engine:
        bundle._plugins.append(local_segmentation(executor, args.engine.resolve()))
    else:
        engine = FixtureEngine()
        bundle._plugins.append(SegmentationPlugin(
            {"cards": {"fixture_chair": ["chair"]}, "publish_overlay": False},
            executor, lambda: (engine, ["chair"]), lambda: ""))
    main._bundle = bundle
    server = main.ThreadingHTTPServer(("127.0.0.1", 15784), main.make_handler())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    request = urllib.request.Request(
        "https://127.0.0.1:15678/api/mcp",
        data=json.dumps({"name": "Fusion Browser Fixture", "transport": "http",
                         "url": "http://127.0.0.1:15784/mcp", "category": "perception"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    for _ in range(40):
        try:
            with urllib.request.urlopen(request, context=ssl._create_unverified_context(), timeout=4) as response:
                print("FUSION_FIXTURE_REGISTERED", response.status, flush=True)
            break
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    else:
        raise RuntimeError("Could not register fusion fixture in local Dashboard")
    threading.Event().wait()


if __name__ == "__main__":
    run()
