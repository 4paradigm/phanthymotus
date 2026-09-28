"""
tests/test_pointcloud_math.py — 纯数学层的单元测试（只依赖 numpy，主机可跑）。

覆盖 pointcloud_math.py 全部函数，重点是三条 load-bearing 契约：

    1. encode_packet 的二进制协议（pointcloud.js / ros2_bridge 按它解包）；
    2. Q_from_baseline 的符号约定（SGBM 视差 d ≥ 0 → Tx 必须为负，否则
       W = Q[3,2]·d ≤ 0，深度全为负数或除零——review issue 1 的根因）；
    3. 标定 blob 的分档与文件名校验（review issue 3）。

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest perception/tests -q
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

import plugins.pointcloud_math as pcm


# ── decimation_stride ─────────────────────────────────────────────────────────

def test_decimation_stride_reference_values():
    # q5 参照：ceil(sqrt(w*h/max_points))
    assert pcm.decimation_stride(640, 480, 20000) == 4
    assert pcm.decimation_stride(640, 480, 40000) == 3  # ceil(sqrt(307200/40000))
    assert pcm.decimation_stride(640, 480, 307200) == 1  # 不抽稀
    assert pcm.decimation_stride(100, 100, 20000) == 1


def test_decimation_stride_zero_max_points_does_not_divide_by_zero():
    assert pcm.decimation_stride(100, 100, 0) == 100


# ── backproject_depth ────────────────────────────────────────────────────────

def test_backproject_depth_center_pixel_is_straight_ahead():
    depth = np.full((3, 3), 2.0, dtype=np.float32)
    fx = fy = 100.0
    cx = cy = 1.0  # 3x3 的画幅中心
    xyz = pcm.backproject_depth(depth, fx, fy, cx, cy, 1, 0.1, 10.0)
    center = xyz[(xyz[:, 0] == 0) & (xyz[:, 1] == 0)]
    assert len(center) == 1
    assert center[0, 2] == pytest.approx(2.0)


def test_backproject_depth_applies_intrinsics_and_range_mask():
    depth = np.array([[1.0, 2.0, 50.0]], dtype=np.float32)
    fx = fy = 100.0
    cx = cy = 0.0
    # max 10m → 第三列被范围掩码剔除
    xyz = pcm.backproject_depth(depth, fx, fy, cx, cy, 1, 0.1, 10.0)
    assert xyz.shape[0] == 2
    # x = (col - cx) * z / fx：col=1, z=2 → x = 1*2/100 = 0.02
    assert xyz[0, 0] == pytest.approx(0.0)
    assert xyz[1, 0] == pytest.approx(0.02)
    assert xyz[:, 2].tolist() == [1.0, 2.0]


def test_backproject_depth_nonfinite_dropped():
    depth = np.array([[np.nan, np.inf, 1.0]], dtype=np.float32)
    xyz = pcm.backproject_depth(depth, 100, 100, 0, 0, 1, 0.1, 10.0)
    assert xyz.shape[0] == 1
    assert xyz[0, 2] == pytest.approx(1.0)


def test_backproject_depth_empty_when_all_invalid():
    depth = np.full((2, 2), 99.0, dtype=np.float32)
    xyz = pcm.backproject_depth(depth, 100, 100, 1, 1, 1, 0.1, 10.0)
    assert xyz.shape == (0, 3)


def test_backproject_depth_stride_samples_a_quarter():
    depth = np.arange(64, dtype=np.float32).reshape(8, 8) + 1.0
    xyz = pcm.backproject_depth(depth, 100, 100, 0, 0, 2, 0.1, 100.0)
    # stride=2 → 4x4 采样 = 16 点
    assert xyz.shape[0] == 16
    # z 取的是 [::2, ::2] 的值：depth[0,0]=1
    assert xyz[:, 2].min() == pytest.approx(1.0)


# ── make_Q / Q_from_baseline ─────────────────────────────────────────────────

def test_make_Q_reference_form():
    Q = pcm.make_Q(100.0, 320.0, 240.0, -0.02443)
    assert Q[0, 0] == 1 and Q[1, 1] == 1
    assert Q[0, 3] == -320.0 and Q[1, 3] == -240.0
    assert Q[2, 3] == 100.0
    assert Q[3, 2] == pytest.approx(-1.0 / -0.02443)


def test_q_from_baseline_normalizes_sign_both_ways():
    # OpenCV stereoCalibrate 的 T[0] 是负；Unitree 出厂标定是正。两者都
    # 必须得到 Q[3,2] > 0（SGBM d ≥ 0 → W > 0 → 正深度）。
    for baseline in (0.02443, -0.02443):
        Q = pcm.Q_from_baseline(100.0, 320.0, 240.0, baseline)
        assert Q[3, 2] > 0
        assert Q[3, 2] == pytest.approx(1.0 / 0.02443)


def test_q_from_baseline_reprojects_disparity_to_positive_depth():
    # 端到端符号验证：d=10 视差，fx=100，基线 0.02443 → Z = fx·B/d
    Q = pcm.Q_from_baseline(100.0, 0.0, 0.0, 0.02443)
    d = 10.0
    W = Q[3, 2] * d + Q[3, 3]
    Z = Q[2, 3] / W  # W 分母里的 Q[2,3]=fx
    assert Z == pytest.approx(100.0 * 0.02443 / d)


# ── filter_cloud / decimate_cloud ─────────────────────────────────────────────

def test_filter_cloud_drops_nonfinite_and_out_of_range():
    xyz = np.array([
        [0, 0, 1.0],
        [np.nan, 0, 1.0],
        [0, 0, 0.05],   # 近于 min
        [0, 0, 20.0],   # 远于 max
        [0, 0, 5.0],
    ], dtype=np.float32)
    out = pcm.filter_cloud(xyz, 0.1, 10.0)
    assert out.shape[0] == 2
    assert 5.0 in out[:, 2] and 1.0 in out[:, 2]


def test_filter_cloud_empty_and_all_invalid_pass_through_shape():
    empty = np.zeros((0, 3), dtype=np.float32)
    assert pcm.filter_cloud(empty, 0.1, 10.0).shape == (0, 3)
    bad = np.full((3, 3), np.nan, dtype=np.float32)
    assert pcm.filter_cloud(bad, 0.1, 10.0).shape == (0, 3)


def test_decimate_cloud_takes_every_stride_th_point():
    xyz = np.arange(30, dtype=np.float32).reshape(10, 3)
    out = pcm.decimate_cloud(xyz, 3)
    assert out.shape[0] == 4  # indices 0, 3, 6, 9
    assert out[0, 0] == 0.0 and out[1, 0] == 9.0
    # stride ≤ 1 原样返回（仍保证 contiguous float32）
    same = pcm.decimate_cloud(xyz, 1)
    assert same.shape == xyz.shape
    assert np.array_equal(same, xyz)


# ── encode_packet ─────────────────────────────────────────────────────────────

def test_encode_packet_header_matches_js_protocol():
    xyz = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    packet = pcm.encode_packet(xyz)
    magic, n = struct.unpack("<II", packet[:8])
    assert magic == 12
    assert n == 1
    assert len(packet) == 8 + 12


def test_encode_packet_axis_order_is_display_ready():
    """出包 (p_x,p_y,p_z) = (−z, x, y_down)：渲染器默认映射后不镜像。"""
    xyz = np.array([[2.0, -1.0, 3.0]], dtype=np.float32)  # x右, y下, z前
    vals = np.frombuffer(pcm.encode_packet(xyz)[8:], dtype="<f4")
    assert vals[0] == pytest.approx(-3.0)  # p_x = -z
    assert vals[1] == pytest.approx(2.0)   # p_y = x
    assert vals[2] == pytest.approx(-1.0)  # p_z = y_down


def test_encode_packet_rejects_wrong_shape():
    with pytest.raises(ValueError):
        pcm.encode_packet(np.zeros((4,), dtype=np.float32))


def test_encode_packet_round_trips_many_points():
    rng = np.random.default_rng(42)
    xyz = rng.normal(size=(500, 3)).astype(np.float32)
    packet = pcm.encode_packet(xyz)
    _, n = struct.unpack("<II", packet[:8])
    out = np.frombuffer(packet[8:], dtype="<f4").reshape(n, 3)
    expected = np.stack((-xyz[:, 2], xyz[:, 0], xyz[:, 1]), axis=-1)
    assert np.allclose(out, expected)


# ── summarize_cloud ──────────────────────────────────────────────────────────

def _region_cloud():
    # x ∈ [0, 3)：左区 z=1.05，中区 z=2.55，右区 z=4.05（各含一个 0.01
    # 离群点，验证 5th percentile 而非 min）
    pts = []
    for x, z in ((0.0, 1.05), (1.0, 2.55), (2.0, 4.05)):
        for _ in range(20):
            pts.append([x, 0.0, z])
        pts.append([x, 0.0, 0.01])
    return np.asarray(pts, dtype=np.float32)


def test_summarize_cloud_regions_use_fifth_percentile():
    summary = pcm.summarize_cloud(_region_cloud(), 0.1, 10.0)
    assert summary["points"] == 63
    assert summary["nearest_by_region"]["left"] == pytest.approx(1.05)
    assert summary["nearest_by_region"]["center"] == pytest.approx(2.55)
    assert summary["nearest_by_region"]["right"] == pytest.approx(4.05)


def test_summarize_cloud_outliers_do_not_invent_obstacles():
    summary = pcm.summarize_cloud(_region_cloud(), 0.1, 10.0)
    # 0.01 离群点只在每区 21 点里占 1 个，5th percentile 仍是主体值
    assert summary["nearest"] == pytest.approx(0.01)  # 全局 min 仍如实报告
    assert summary["nearest_by_region"]["left"] != pytest.approx(0.01)


def test_summarize_cloud_empty_has_nulls():
    summary = pcm.summarize_cloud(np.zeros((0, 3), dtype=np.float32), 0.1, 10.0)
    assert summary["points"] == 0
    assert summary["nearest"] is None and summary["farthest"] is None
    assert summary["nearest_by_region"] == {"left": None, "center": None, "right": None}
    assert summary["z_range_m"] == [0.1, 10.0]


# ── intrinsics / blob loading ────────────────────────────────────────────────

def test_intrinsics_from_hfov_90_degrees_is_half_width():
    fx, fy, cx, cy = pcm.intrinsics_from_hfov(640, 480, 90.0)
    assert np.allclose([fx, fy, cx, cy], [320.0, 320.0, 320.0, 240.0])


def test_pinhole_of_prefers_explicit_fx():
    blob = {"fx": 500.0, "fy": 510.0, "cx": 300.0, "cy": 200.0}
    assert pcm.pinhole_of(blob, 640, 480) == (500.0, 510.0, 300.0, 200.0)


def test_pinhole_of_hfov_fallback():
    fx, fy, cx, cy = pcm.pinhole_of({"hfov": 90.0}, 640, 480)
    assert np.allclose([fx, fy], [320.0, 320.0])
    assert (cx, cy) == (320.0, 240.0)


def test_pinhole_of_defaults_to_90_degrees():
    fx, fy, cx, cy = pcm.pinhole_of({}, 100, 50)
    assert np.allclose([fx, fy], [50.0, 50.0])  # w/2


def test_load_calibration_blob_accepts_dict_json_and_file(tmp_path):
    blob = {"fx": 100.0, "cx": 5, "cy": 5, "Tx": 0.02}
    assert pcm.load_calibration_blob(blob) is blob
    assert pcm.load_calibration_blob('{"fx": 1}') == {"fx": 1}
    path = tmp_path / "calib.json"
    path.write_text('{"fx": 2}', encoding="utf-8")
    assert pcm.load_calibration_blob(str(path)) == {"fx": 2}


def test_load_calibration_blob_rejects_junk():
    assert pcm.load_calibration_blob(None) is None
    assert pcm.load_calibration_blob("") is None
    assert pcm.load_calibration_blob("not json and not a file") is None
    assert pcm.load_calibration_blob("/nonexistent/path/xyz.json") is None


def test_calibration_tier_classification():
    assert pcm.calibration_tier({"K1": [], "D1": [], "K2": [], "D2": []}) == "raw"
    assert pcm.calibration_tier({"fx": 1, "cx": 1, "cy": 1, "Tx": 1}) == "rectified"
    assert pcm.calibration_tier({"hfov": 90}) == "pinhole"
    assert pcm.calibration_tier({"fx": 1, "fy": 1}) == "pinhole"
    assert pcm.calibration_tier({}) == "unknown"
    assert pcm.calibration_tier({"rms": 0.1}) == "unknown"


# ── sanitize_calib_name（review issue 3：路径穿越）────────────────────────────

def test_sanitize_calib_name_accepts_plain_names():
    ok, err = pcm.sanitize_calib_name("stereo_20240101_120000")
    assert ok == "stereo_20240101_120000" and err is None
    ok, err = pcm.sanitize_calib_name("my-calib.v2")
    assert ok == "my-calib.v2" and err is None


def test_sanitize_calib_name_blocks_path_traversal():
    for bad in ("../../etc/passwd", "a/b", "a\\b", "..", ".", "", None, "  ", "a;b"):
        ok, err = pcm.sanitize_calib_name(bad)
        assert ok is None, f"{bad!r} should be rejected"
        assert err


def test_default_calib_name_is_sanitizable():
    assert pcm.sanitize_calib_name(pcm.default_calib_name())[0] is not None
