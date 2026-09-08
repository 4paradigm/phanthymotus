"""Image-to-distance processing with a fake engine, not model artifacts."""

import time

import cv2
import numpy as np
import pytest

from plugins.obstacle_distance_core import metric_tensorrt_backend as backend_module
from plugins.obstacle_distance_core.contracts import ErrorCode, ObstacleDistanceError


class _Engine:
    input_shape = (1, 3, 384, 512)
    is_static = True
    output_names = ["depth_m"]

    def __init__(self):
        self.depth = np.full((1, 1, 384, 512), 1.5, dtype=np.float32)
        self.closed = False
        self.input = None

    def infer(self, tensor):
        self.input = tensor
        return [self.depth]

    def close(self):
        self.closed = True


@pytest.fixture
def engine(monkeypatch):
    engine = _Engine()
    monkeypatch.setattr(backend_module, "TensorRTEngine", lambda path: engine)
    return engine


def _jpeg(height=480, width=640):
    image = np.full((height, width, 3), [30, 60, 90], dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _predict(config=None, image=None):
    backend = backend_module.MetricDepthBackend("unused.engine", config or {})
    try:
        return backend.predict_indoor_distance(
            _jpeg() if image is None else image, time.monotonic() + 10,
        )
    finally:
        backend.close()


def test_jpeg_to_rgb_float_nchw(engine):
    jpeg = _jpeg()
    assert _predict(image=jpeg) == 1.5
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert engine.input.shape == (1, 3, 384, 512)
    assert engine.input.dtype == np.float32 and engine.input.flags.c_contiguous
    np.testing.assert_allclose(engine.input[0, :, 0, 0], decoded[0, 0, ::-1] / 255.0)


def test_bad_jpeg_does_not_load_engine(monkeypatch):
    def unexpected(path):
        pytest.fail("invalid image loaded an engine")
    monkeypatch.setattr(backend_module, "TensorRTEngine", unexpected)
    with pytest.raises(ObstacleDistanceError) as error:
        _predict(image=b"not a jpeg")
    assert error.value.code is ErrorCode.INVALID_IMAGE


@pytest.mark.parametrize("attribute,value", [
    ("input_shape", (1, 3, 640, 640)), ("is_static", False), ("output_names", ["a", "b"]),
])
def test_incompatible_engine_is_closed(engine, attribute, value):
    setattr(engine, attribute, value)
    with pytest.raises(ObstacleDistanceError) as error:
        _predict()
    assert error.value.code is ErrorCode.MODEL_ERROR and engine.closed


def test_wrong_output_shape_is_rejected(engine):
    engine.depth = np.ones((1, 1, 384, 511), dtype=np.float32)
    with pytest.raises(ObstacleDistanceError) as error:
        _predict()
    assert error.value.code is ErrorCode.INVALID_DEPTH


@pytest.mark.parametrize("height,width,percentile", [(480, 640, 1), (720, 1280, 50)])
def test_scaled_roi_and_percentile_on_depth_plane(engine, height, width, percentile):
    y, x = np.mgrid[:384, :512]
    engine.depth = (1 + x / 512 + 2 * y / 384).astype(np.float32)[None, None]
    y, x = np.mgrid[:height, :width]
    expected = 1 + (x * 511 / (width - 1)) / 512 + 2 * (y * 383 / (height - 1)) / 384
    roi = expected[:round(height * 300 / 480), round(width * 213 / 640):round(width * 426 / 640)]
    # OpenCV remap uses a quantized interpolation table.
    assert _predict({"depth_percentile": percentile}, _jpeg(height, width)) == pytest.approx(
        np.percentile(roi, percentile), abs=3e-4,
    )


@pytest.mark.parametrize("value,expected", [(0.1, 0.3), (1.5, 1.5), (20.0, 10.0)])
def test_metric_output_clamps_without_score_conversion(engine, value, expected):
    engine.depth.fill(value)
    assert _predict() == pytest.approx(expected)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -1.0])
def test_invalid_depth_pixels_are_not_clear_space(engine, value):
    engine.depth.fill(value)
    with pytest.raises(ObstacleDistanceError) as error:
        _predict()
    assert error.value.code is ErrorCode.NO_VALID_DEPTH
