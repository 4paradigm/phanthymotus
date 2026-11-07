"""Geometry and inference contract checks, without ROS or GPU hardware."""
import copy

import cv2
import numpy as np
import pytest

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
