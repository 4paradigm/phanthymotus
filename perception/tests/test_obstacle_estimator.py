"""Metric-distance validation and structured failure outputs."""

from types import SimpleNamespace

import pytest

from plugins.obstacle_distance_core.contracts import ErrorCode, ObstacleDistanceError
from plugins.obstacle_distance_core.estimator import ObstacleDistanceEstimator


def _estimator(value=1.5, **config):
    backend = SimpleNamespace(predict_indoor_distance=lambda data, deadline: value)
    return ObstacleDistanceEstimator(backend, config)


@pytest.mark.parametrize("distance,near", [(1.5, True), (2.0, False), (2.5, False)])
def test_metric_distance_and_strict_threshold(distance, near):
    result = _estimator(distance, decision_threshold_m=2.0).estimate(b"image")
    assert result.distance_m == distance and result.near_obstacle is near
    assert result.scene == "indoor" and result.status == "ok"
    assert not result.fallback and result.error_code is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_invalid_distance_is_reported_as_failure(value):
    result = _estimator(value).estimate(b"image")
    assert result.fallback and result.error_code == "model_error"
    assert result.status == "error" and result.distance_m == 3.0


@pytest.mark.parametrize("image", [b"", None, "not bytes"])
def test_invalid_image_is_rejected_before_backend(image):
    def predict(data, deadline):
        pytest.fail("invalid image reached model")
    estimator = ObstacleDistanceEstimator(SimpleNamespace(predict_indoor_distance=predict), {})
    result = estimator.estimate(image)
    assert result.fallback and result.error_code == "invalid_image"


@pytest.mark.parametrize("code", [ErrorCode.TIMEOUT, ErrorCode.NO_VALID_DEPTH])
def test_backend_failure_reason_is_preserved(code):
    def predict(data, deadline):
        raise ObstacleDistanceError(code, "inference failed")
    estimator = ObstacleDistanceEstimator(SimpleNamespace(predict_indoor_distance=predict), {})
    result = estimator.estimate(b"image")
    assert result.fallback and result.error_code == code.value


@pytest.mark.parametrize("config", [{"decision_threshold_m": 0}, {"soft_timeout_s": -1}, {"fallback_distance_m": float("nan")}])
def test_invalid_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        _estimator(**config)
