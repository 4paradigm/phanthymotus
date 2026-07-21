#!/usr/bin/env python3
"""
Local verification script for OCR integration.

This script mocks the ROS2 runtime so we can verify that:
1. main.py loads without errors
2. config.yaml parses correctly
3. OCR plugin is loaded when enabled
4. MCP tools/list includes the ocr tool
5. tools/call dispatch works for ocr/info
6. If local RapidOCR models are present, a sample image can be recognized.

Full runtime verification still requires the ROS2 Docker environment.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path


# ── ROS2 mocks ───────────────────────────────────────────────────────────────

def _make_qos_profile(**kwargs):
    return types.SimpleNamespace(**kwargs)


def _build_ros_mocks():
    """Inject minimal ROS2 stub modules into sys.modules."""

    # rclpy
    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda *args, **kwargs: None
    rclpy.shutdown = lambda *args, **kwargs: None

    def _spin(*args, **kwargs):
        pass

    rclpy.executors = types.ModuleType("rclpy.executors")
    rclpy.executors.MultiThreadedExecutor = lambda: types.SimpleNamespace(spin=_spin, shutdown=lambda: None)

    rclpy.qos = types.ModuleType("rclpy.qos")
    rclpy.qos.QoSProfile = _make_qos_profile
    rclpy.qos.ReliabilityPolicy = types.SimpleNamespace(RELIABLE=1)
    rclpy.qos.HistoryPolicy = types.SimpleNamespace(KEEP_LAST=1)
    rclpy.qos.DurabilityPolicy = types.SimpleNamespace(VOLATILE=1)

    rclpy.node = types.ModuleType("rclpy.node")

    class _Node:
        def __init__(self, name: str):
            self.name = name
            self._subs = []
            self._pubs = []

        def create_subscription(self, msg_type, topic, callback, qos):
            self._subs.append((msg_type, topic, callback, qos))
            return types.SimpleNamespace(destroy=lambda: None)

        def create_publisher(self, msg_type, topic, qos):
            self._pubs.append((msg_type, topic, qos))
            pub = types.SimpleNamespace()
            pub.publish = lambda msg: None
            return pub

        def destroy_subscription(self, sub):
            pass

    rclpy.node.Node = _Node
    sys.modules["rclpy"] = rclpy
    sys.modules["rclpy.node"] = rclpy.node
    sys.modules["rclpy.qos"] = rclpy.qos
    sys.modules["rclpy.executors"] = rclpy.executors

    # sensor_msgs
    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs.msg = types.ModuleType("sensor_msgs.msg")

    class _CompressedImage:
        def __init__(self):
            self.data = b""
            self.format = "jpeg"

    sensor_msgs.msg.CompressedImage = _CompressedImage
    sys.modules["sensor_msgs"] = sensor_msgs
    sys.modules["sensor_msgs.msg"] = sensor_msgs.msg

    # std_msgs
    std_msgs = types.ModuleType("std_msgs")
    std_msgs.msg = types.ModuleType("std_msgs.msg")

    class _String:
        def __init__(self):
            self.data = ""

    std_msgs.msg.String = _String
    sys.modules["std_msgs"] = std_msgs
    sys.modules["std_msgs.msg"] = std_msgs.msg


# ── Verification ─────────────────────────────────────────────────────────────

def main() -> int:
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    print("[verify] Building ROS2 mocks...")
    _build_ros_mocks()

    print("[verify] Importing main.py...")
    import importlib.util

    spec = importlib.util.spec_from_file_location("main", script_dir / "main.py")
    main_mod = importlib.util.module_from_spec(spec)
    sys.modules["main"] = main_mod
    spec.loader.exec_module(main_mod)

    print("[verify] Loading config.yaml...")
    cfg = main_mod._load_config()
    assert "plugins" in cfg, "plugins section missing"
    ocr_cfg = cfg["plugins"]["ocr"]
    assert ocr_cfg.get("enabled") is True, "ocr not enabled"

    # Use local model bundle for verification if it exists.
    local_model_dir = script_dir / "models" / "ocr" / "ppocrv6-small-int8"
    if local_model_dir.is_dir():
        ocr_cfg["model_dir"] = str(local_model_dir.resolve())
    print(f"[verify] OCR config: provider={ocr_cfg.get('provider')}, model_dir={ocr_cfg.get('model_dir')}")

    # Disable other plugins for this structural verification (their deps are not mocked)
    for plugin_name in ("asr", "tts", "htmsg", "vop"):
        cfg["plugins"].setdefault(plugin_name, {})["enabled"] = False

    print("[verify] Loading PerceptionBundle with OCR enabled...")
    executor = types.SimpleNamespace(spin=lambda: None, shutdown=lambda: None)
    bundle = main_mod.PerceptionBundle(cfg, executor)

    loaded = [type(p).__name__ for p in bundle._plugins]
    assert "OCRPlugin" in loaded, f"OCRPlugin not loaded, got {loaded}"
    print(f"[verify] Loaded plugins: {loaded}")

    print("[verify] Checking MCP tools/list...")
    tools = bundle.get_all_tools()
    tool_names = [t["name"] for t in tools]
    assert "ocr" in tool_names, f"ocr tool missing, got {tool_names}"
    print(f"[verify] Tools: {tool_names}")

    print("[verify] Calling ocr/info dispatch...")
    result = bundle.dispatch("ocr", {"action": "info"})
    assert result is not None, "ocr/info returned None"
    assert result.get("state") == "idle", f"unexpected state: {result.get('state')}"
    print(f"[verify] ocr/info result: {json.dumps(result, ensure_ascii=False, indent=2)}")

    # Optional: if local models exist, verify RapidOCRAdapter can load and run.
    model_dir = Path(ocr_cfg.get("model_dir", "/models/ocr/ppocrv6-small-int8"))
    required = ("det.onnx", "rec.onnx", "cls.onnx", "keys.txt")
    if all((model_dir / name).is_file() for name in required):
        print(f"[verify] Local models found in {model_dir}, testing RapidOCRAdapter...")
        try:
            from plugins.ocr_runtime import RapidOCRAdapter
        except ImportError as e:
            print(f"[verify] rapidocr not installed in this environment ({e}), skipping inference")
            print("\n[verify] All structural checks passed.")
            return 0
        try:
            adapter = RapidOCRAdapter(
                str(model_dir),
                use_angle_cls=ocr_cfg.get("use_angle_cls", True),
                num_threads=int(ocr_cfg.get("num_threads", 1)),
                max_side_len=int(ocr_cfg.get("max_side_len", 1600)),
            )
            sample = script_dir.parent / "docs" / "images" / "dashboard.png"
            if sample.is_file():
                results = adapter.recognize(sample.read_bytes())
                print(f"[verify] Sample OCR result: {len(results)} boxes")
                for item in results[:3]:
                    print(f"  {item}")
            else:
                print(f"[verify] No sample image at {sample}, skipping inference")
        except Exception as e:
            print(f"[verify] WARNING: RapidOCRAdapter test failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        missing = [name for name in required if not (model_dir / name).is_file()]
        print(f"[verify] Local models not complete in {model_dir} (missing {missing}), skipping inference")

    print("\n[verify] All structural checks passed.")
    print("Note: full runtime verification requires ROS2 Docker image.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
