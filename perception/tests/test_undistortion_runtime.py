"""Geometry and inference contract checks, without ROS or GPU hardware."""
import copy

import cv2
import numpy as np
import pytest

from plugins import undistortion_runtime as runtime
from plugins.undistortion_runtime import Rectifier, declaration, validate, _valid_rectangle


def calibration(model="plumb_bob"):
    return {"id": "test/front", "width": 64, "height": 48,
            "K": [35, 0, 32, 0, 35, 24, 0, 0, 1],
            "D": [0, 0, 0, 0], "distortion_model": model}


@pytest.mark.parametrize("model", ["plumb_bob", "equidistant", "rational_polynomial"])
def test_remap_matches_opencv_and_crops_unobserved_border(model):
    cal = calibration(model)
    if model == "rational_polynomial":
        cal["D"] = [-0.2, 0.02, 0, 0, 0, 0, 0, 0]
    elif model == "plumb_bob":
        cal["D"] = [0.3, 0.02, 0, 0]

    wrapper = Rectifier(cal)
    image = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    output = wrapper.apply(image)
    k = np.asarray(cal["K"], np.float64).reshape(3, 3)
    d = np.asarray(cal["D"], np.float64)
    init = (cv2.fisheye.initUndistortRectifyMap if model == "equidistant"
            else cv2.initUndistortRectifyMap)
    mx, my = init(k, d, np.eye(3), k, (64, 48), cv2.CV_32FC1)
    expected = cv2.remap(image, mx, my, cv2.INTER_LINEAR)
    x, y, width, height = wrapper.roi
    np.testing.assert_array_equal(output, expected[y:y + height, x:x + width])
    assert wrapper._maps[2][y:y + height, x:x + width].all()
    info = wrapper.camera_info({})
    assert info["width"] == output.shape[1]
    assert info["height"] == output.shape[0]
    assert info["K"][2] == cal["K"][2] - x
    assert info["K"][5] == cal["K"][5] - y
    cache = wrapper._maps
    wrapper.apply(image)
    assert wrapper._maps is cache
    with pytest.raises(ValueError, match="input size"):
        wrapper.apply(image[:20])


def test_output_declaration_describes_rectified_geometry_without_mutating_input():
    cal = calibration()
    upstream = {"schema": "motus.camera/1", "id": "test/front", "width": 64,
                "height": 48, "D": [1] * 5, "half_fov_rad": 0.423,
                "vendor": {"other": True}}
    before = copy.deepcopy(upstream)
    out = declaration(upstream, cal)
    assert upstream == before
    assert out["D"] == [0] * 5
    assert out["K"] == cal["K"]
    assert out["half_fov_rad"] != 0.423
    assert out["pipeline"] == ["perception/undistort"]
    with pytest.raises(ValueError, match="different camera"):
        declaration({"id": "test/rear"}, cal)
    with pytest.raises(ValueError, match="resolution"):
        declaration({"width": 128}, cal)


@pytest.mark.parametrize("change", [
    {"D": [10]}, {"D": [float("nan")] * 4}, {"width": 0},
    {"distortion_model": "unknown"}, {"K": [600, 600]}, {"id": ""},
])
def test_rejects_incomplete_or_invalid_calibration(change):
    cal = calibration()
    cal.update(change)
    with pytest.raises(ValueError):
        validate(cal)


def test_disabled_is_identity_declaration():
    assert validate({}) is None
    assert declaration({"id": "front"}, {}) == {"id": "front"}


def test_valid_crop_does_not_include_interior_holes():
    mask = np.ones((6, 8), dtype=bool)
    mask[2:4, 3:5] = False
    x, y, w, h = _valid_rectangle(mask)
    assert mask[y:y + h, x:x + w].all()
    assert w * h == 18
    with pytest.raises(ValueError, match="no valid"):
        _valid_rectangle(np.zeros((4, 4), dtype=bool))


# ── built-in calibration table ────────────────────────────────────────────────

PROVENANCE = ("camera", "measured_on", "method", "source_resolution",
              "frames_used", "rms_px", "note")


@pytest.mark.parametrize("camera_id", sorted(runtime.CALIBRATIONS))
def test_every_table_row_is_a_valid_calibration_that_says_where_it_came_from(camera_id):
    row = runtime.CALIBRATIONS[camera_id]
    assert validate(row)["id"] == camera_id
    for field in PROVENANCE:
        assert row.get(field), f"{camera_id} 缺少出处字段 {field}"
    # A row that fits worse than half a pixel was not a usable chessboard run.
    assert row["rms_px"] < 0.5
    # And it must rectify to a usable picture at the size it declares.
    rectifier = Rectifier(row)
    x, y, w, h = rectifier.roi
    assert w * h >= 0.8 * row["width"] * row["height"]


def test_the_s10_front_row_matches_the_measured_field_of_view():
    """The chessboard run measured 90.8 degrees across after rectification; the
    driver now declares the same angle (phanthymotus-driver#501). A row whose
    K drifted from that would disagree with the camera's own declaration."""
    row = runtime.lookup("deeprobotics/s10_pro/camera_front")
    info = Rectifier(row).camera_info({"id": row["id"], "width": 640, "height": 360})
    assert info["width"] == 640 and info["height"] == 360
    assert info["half_fov_rad"] == pytest.approx(0.7926, abs=1e-3)
    assert info["D"] == [0.0] * 5


S10_PORTS = ("camera_front", "camera_rear", "camera_left", "camera_right")


def test_all_four_s10_cameras_are_calibrated_and_agree_as_one_lens_model():
    """Four units of the same module: a typo in one row stands out against the
    other three. Measured spread is 312.8-315.0 px focal and 90.8-91.3 deg."""
    infos = []
    for port in S10_PORTS:
        row = runtime.lookup(f"deeprobotics/s10_pro/{port}")
        assert row is not None, port
        assert (row["width"], row["height"], row["distortion_model"]) == (640, 360, "plumb_bob")
        infos.append(Rectifier(row).camera_info({"id": row["id"], "width": 640, "height": 360}))
        assert infos[-1]["width"] == 640 and infos[-1]["height"] == 360, port
    fx = [info["K"][0] for info in infos]
    fov = [info["half_fov_rad"] for info in infos]
    assert max(fx) - min(fx) < 5
    assert max(fov) - min(fov) < 0.01
    k1 = [runtime.CALIBRATIONS[f"deeprobotics/s10_pro/{p}"]["D"][0] for p in S10_PORTS]
    assert max(k1) - min(k1) < 0.02


def test_lookup_returns_a_copy_and_none_for_an_unknown_camera():
    row = runtime.lookup("deeprobotics/s10_pro/camera_front")
    row["K"][0] = 1.0
    assert runtime.CALIBRATIONS["deeprobotics/s10_pro/camera_front"]["K"][0] != 1.0
    assert runtime.lookup("acme/cam/1") is None
    assert runtime.lookup("") is None


def test_resolve_prefers_typed_in_then_upstream_then_table(monkeypatch):
    monkeypatch.setitem(runtime.CALIBRATIONS, "test/front",
                        dict(calibration(), K=[30, 0, 32, 0, 30, 24, 0, 0, 1]))
    upstream_k = [40, 0, 32, 0, 40, 24, 0, 0, 1]
    upstream = dict(calibration(), K=upstream_k)

    typed = dict(calibration(), K=[50, 0, 32, 0, 50, 24, 0, 0, 1])
    cal, source = runtime.resolve(typed, upstream)
    assert (source, cal["K"][0]) == ("manual", 50)

    cal, source = runtime.resolve({}, upstream)
    assert (source, cal["K"][0]) == ("upstream", 40)

    # A driver declaration with no usable model says nothing about distortion.
    for model in ("unknown", "pinhole", None):
        bare = {"id": "test/front", "width": 64, "height": 48,
                "distortion_model": model, "K": None, "D": None}
        cal, source = runtime.resolve({}, bare)
        assert (source, cal["K"][0]) == ("table:test/front", 30)


def test_an_upstream_that_claims_a_model_owns_its_numbers():
    """Not silently replaced by the table: a broken declaration is an error."""
    broken = dict(calibration(), D=[10])
    with pytest.raises(ValueError, match="D length"):
        runtime.resolve({}, broken)


@pytest.mark.parametrize("upstream", [
    {"id": "acme/cam/1", "width": 64, "height": 48, "distortion_model": "unknown"},
    {}, None,
])
def test_an_unknown_camera_says_which_one_and_how_to_fix_it(upstream):
    with pytest.raises(ValueError) as error:
        runtime.resolve({}, upstream)
    message = str(error.value)
    assert "calibrate_chessboard.py" in message and "CALIBRATIONS" in message
    if upstream and upstream.get("id"):
        assert "acme/cam/1" in message
