import json
import pathlib
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import ros2_bridge


def test_pointcloud2_has_distinct_type_mapping(monkeypatch):
    marker = type("PointCloud2", (), {})
    messages = types.ModuleType("sensor_msgs.msg")
    messages.PointCloud2 = marker
    monkeypatch.setitem(sys.modules, "sensor_msgs", types.ModuleType("sensor_msgs"))
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", messages)
    assert ros2_bridge._resolve_msg_type("sensor/pointcloud2") is marker


def test_canvas_preview_keeps_header_metadata_without_exposing_raw_cloud():
    msg = types.SimpleNamespace(
        header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=123, nanosec=456),
                                     frame_id="optical"),
        width=2, height=1, point_step=12, row_step=24, is_bigendian=False,
        data=b"untrusted bytes")
    summary = json.loads(ros2_bridge._pointcloud2_metadata(msg))
    assert summary["stamp"] == {"sec": 123, "nanosec": 456}
    assert summary["frame_id"] == "optical"
    assert summary["width"] == 2 and summary["height"] == 1
    assert "data" not in summary


def test_metric_depth_mapping_and_preview(monkeypatch):
    marker = type("Image", (), {})
    messages = types.ModuleType("sensor_msgs.msg")
    messages.Image = marker
    monkeypatch.setitem(sys.modules, "sensor_msgs", types.ModuleType("sensor_msgs"))
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", messages)
    assert ros2_bridge._resolve_msg_type("sensor/depth-f32") is marker
    msg = types.SimpleNamespace(header=types.SimpleNamespace(
        stamp=types.SimpleNamespace(sec=123, nanosec=456), frame_id="optical"),
        width=640, height=480, encoding="32FC1", step=2560, data=b"pixels")
    summary = json.loads(ros2_bridge._depth_f32_metadata(msg))
    assert summary["stamp"] == {"sec": 123, "nanosec": 456}
    assert summary["frame_id"] == "optical" and summary["encoding"] == "32FC1"
    assert "data" not in summary
