from __future__ import annotations

import math
import warnings
from numbers import Real
from typing import Iterable, Iterator

import numpy as np

from .contracts import (
    CameraCalibration,
    ErrorCode,
    InstanceMask,
    ObstacleDistanceError,
)


_GEOMETRY_CHUNK_ROWS = 64


def _invalid_depth(message: str) -> ObstacleDistanceError:
    return ObstacleDistanceError(ErrorCode.INVALID_DEPTH, message)


def _invalid_calibration(message: str) -> ObstacleDistanceError:
    return ObstacleDistanceError(ErrorCode.INVALID_CALIBRATION, message)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise _invalid_depth(f"{name} must be a finite number")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        raise _invalid_depth(f"{name} must be a finite number") from None
    if not math.isfinite(converted):
        raise _invalid_depth(f"{name} must be a finite number")
    return converted


def _validate_depth_map(depth: object) -> np.ndarray:
    try:
        source_array = np.asarray(depth)
    except Exception:
        raise _invalid_depth(
            "depth map must be convertible to a numeric array"
        ) from None
    if source_array.dtype.kind == "O" or np.iscomplexobj(source_array):
        raise _invalid_depth("depth map must use a real numeric dtype")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                depth_array = np.asarray(source_array, dtype=np.float32)
    except Exception:
        raise _invalid_depth(
            "depth map must be convertible to a numeric array"
        ) from None
    if depth_array.ndim != 2 or depth_array.size == 0:
        raise _invalid_depth("depth map must be a nonempty two-dimensional array")
    if not np.isfinite(depth_array).any():
        raise _invalid_depth("depth map must contain at least one finite value")
    return depth_array


def _configuration(
    *,
    allowed_classes: object,
    min_confidence: object,
    percentile: object,
    min_depth_m: object,
    max_depth_m: object,
) -> tuple[set[str] | frozenset[str], float, float, float, float]:
    if (
        not isinstance(allowed_classes, (set, frozenset))
        or not allowed_classes
        or any(
            not isinstance(class_name, str) or not class_name.strip()
            for class_name in allowed_classes
        )
    ):
        raise _invalid_depth(
            "allowed_classes must be a nonempty set of class names"
        )

    confidence = _finite_number(min_confidence, "min_confidence")
    percentile_value = _finite_number(percentile, "percentile")
    minimum = _finite_number(min_depth_m, "min_depth_m")
    maximum = _finite_number(max_depth_m, "max_depth_m")
    if not 0 <= confidence <= 1:
        raise _invalid_depth("min_confidence must be between 0 and 1")
    if not 0 <= percentile_value <= 100:
        raise _invalid_depth("percentile must be between 0 and 100")
    if minimum < 0 or maximum <= minimum:
        raise _invalid_depth(
            "depth range must satisfy 0 <= min_depth_m < max_depth_m"
        )
    return allowed_classes, confidence, percentile_value, minimum, maximum


def _valid_target_depth(
    depth_m: object,
    *,
    instances: Iterable[InstanceMask],
    allowed_classes: object,
    min_confidence: object,
    percentile: object,
    min_depth_m: object,
    max_depth_m: object,
) -> tuple[np.ndarray, np.ndarray, float]:
    (
        allowed,
        confidence,
        percentile_value,
        minimum,
        maximum,
    ) = _configuration(
        allowed_classes=allowed_classes,
        min_confidence=min_confidence,
        percentile=percentile,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )
    depth_array = _validate_depth_map(depth_m)

    try:
        instance_values = tuple(instances)
    except Exception:
        raise _invalid_depth("instances must contain instance masks") from None
    if any(not isinstance(value, InstanceMask) for value in instance_values):
        raise _invalid_depth("instances must contain instance masks")

    selected = [
        value
        for value in instance_values
        if value.class_name in allowed and value.confidence >= confidence
    ]
    if not selected:
        raise ObstacleDistanceError(
            ErrorCode.NO_TARGET_INSTANCE,
            "no target instance passed class and confidence filters",
        )

    merged_mask = np.zeros(depth_array.shape, dtype=bool)
    for value in selected:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mask = np.asarray(value.mask)
        except Exception:
            raise _invalid_depth(
                "instance mask must be a two-dimensional boolean array"
            ) from None
        if (
            mask.dtype.kind in {"O", "c"}
            or mask.dtype != np.dtype(bool)
            or mask.ndim != 2
            or mask.shape != depth_array.shape
        ):
            raise _invalid_depth(
                "instance mask must match depth shape and use boolean dtype"
            )
        np.logical_or(merged_mask, mask, out=merged_mask)

    valid_mask = (
        merged_mask
        & np.isfinite(depth_array)
        & (depth_array >= minimum)
        & (depth_array <= maximum)
    )
    if not valid_mask.any():
        raise ObstacleDistanceError(
            ErrorCode.NO_VALID_DEPTH,
            "target masks contain no valid depth pixels",
        )
    return depth_array, valid_mask, percentile_value


def _valid_pixel_chunks(
    depth_array: np.ndarray,
    valid_mask: np.ndarray,
) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    for row_start in range(0, valid_mask.shape[0], _GEOMETRY_CHUNK_ROWS):
        row_end = min(row_start + _GEOMETRY_CHUNK_ROWS, valid_mask.shape[0])
        block = valid_mask[row_start:row_end]
        local_rows, columns = np.nonzero(block)
        if columns.size:
            yield (
                row_start,
                local_rows,
                columns,
                depth_array[row_start + local_rows, columns],
            )


def _rigid_camera_to_ego(calibration: object) -> tuple[np.ndarray, tuple[float, float]]:
    if calibration is None:
        raise ObstacleDistanceError(
            ErrorCode.MISSING_CALIBRATION,
            "camera calibration is required for vehicle geometry",
        )
    if not isinstance(calibration, CameraCalibration):
        raise _invalid_calibration("camera calibration is invalid")

    matrix = np.asarray(calibration.camera_to_ego, dtype=np.float64).reshape(4, 4)
    rotation = matrix[:3, :3]
    if not np.allclose(
        matrix[3],
        np.array([0.0, 0.0, 0.0, 1.0]),
        rtol=0.0,
        atol=1e-5,
    ):
        raise _invalid_calibration(
            "camera_to_ego must have a homogeneous final row"
        )
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        rtol=1e-5,
        atol=1e-4,
    ):
        raise _invalid_calibration(
            "camera_to_ego rotation must be orthogonal"
        )
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(determinant, 1.0, rel_tol=1e-5, abs_tol=1e-4):
        raise _invalid_calibration(
            "camera_to_ego rotation must be a proper rotation"
        )
    return matrix, calibration.bumper_xy


def _finite_percentile(values: np.ndarray, percentile: float) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = float(np.percentile(values, percentile))
    if not math.isfinite(result):
        raise ObstacleDistanceError(
            ErrorCode.NO_VALID_DEPTH,
            "distance percentile is not finite",
        )
    return result


def vehicle_distance_m(
    depth_m: object,
    *,
    instances: Iterable[InstanceMask],
    calibration: CameraCalibration | None,
    allowed_classes: object,
    min_confidence: object,
    percentile: object,
    min_depth_m: object,
    max_depth_m: object,
) -> float:
    matrix, bumper_xy = _rigid_camera_to_ego(calibration)
    depth_array, valid_mask, percentile_value = _valid_target_depth(
        depth_m,
        instances=instances,
        allowed_classes=allowed_classes,
        min_confidence=min_confidence,
        percentile=percentile,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )

    horizontal = np.empty(
        int(np.count_nonzero(valid_mask)),
        dtype=np.float64,
    )
    cursor = 0
    for row_start, local_rows, columns, block_depth in _valid_pixel_chunks(
        depth_array,
        valid_mask,
    ):
        z_camera = block_depth.astype(np.float64, copy=False)
        x_camera = (
            (columns.astype(np.float64) - calibration.cx) / calibration.fx
        ) * z_camera
        rows = local_rows.astype(np.float64)
        rows += row_start
        y_camera = ((rows - calibration.cy) / calibration.fy) * z_camera
        with np.errstate(over="ignore", invalid="ignore"):
            ego_x = (
                matrix[0, 0] * x_camera
                + matrix[0, 1] * y_camera
                + matrix[0, 2] * z_camera
                + matrix[0, 3]
            )
            ego_y = (
                matrix[1, 0] * x_camera
                + matrix[1, 1] * y_camera
                + matrix[1, 2] * z_camera
                + matrix[1, 3]
            )
            block_horizontal = np.hypot(
                ego_x - bumper_xy[0],
                ego_y - bumper_xy[1],
            )
        finite = np.isfinite(block_horizontal)
        finite_count = int(np.count_nonzero(finite))
        if finite_count:
            horizontal[cursor : cursor + finite_count] = block_horizontal[
                finite
            ]
            cursor += finite_count

    horizontal = horizontal[:cursor]
    if horizontal.size == 0:
        raise ObstacleDistanceError(
            ErrorCode.NO_VALID_DEPTH,
            "target geometry contains no finite horizontal distances",
        )
    return _finite_percentile(horizontal, percentile_value)


def approximate_vehicle_distance_m(
    depth_m: object,
    *,
    instances: Iterable[InstanceMask],
    allowed_classes: object,
    min_confidence: object,
    percentile: object,
    min_depth_m: object,
    max_depth_m: object,
    camera_to_bumper_offset_m: object,
) -> float:
    offset = _finite_number(
        camera_to_bumper_offset_m,
        "camera_to_bumper_offset_m",
    )
    if offset < 0:
        raise _invalid_depth(
            "camera_to_bumper_offset_m must be nonnegative"
        )
    depth_array, valid_mask, percentile_value = _valid_target_depth(
        depth_m,
        instances=instances,
        allowed_classes=allowed_classes,
        min_confidence=min_confidence,
        percentile=percentile,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )
    approximate = np.empty(
        int(np.count_nonzero(valid_mask)),
        dtype=np.float64,
    )
    cursor = 0
    for _, _, _, block_depth in _valid_pixel_chunks(depth_array, valid_mask):
        z_camera = block_depth.astype(np.float64, copy=False)
        with np.errstate(over="ignore", invalid="ignore"):
            block_approximate = np.maximum(z_camera - offset, 0.0)
        finite = np.isfinite(block_approximate)
        finite_count = int(np.count_nonzero(finite))
        if finite_count:
            approximate[cursor : cursor + finite_count] = block_approximate[
                finite
            ]
            cursor += finite_count

    approximate = approximate[:cursor]
    if approximate.size == 0:
        raise ObstacleDistanceError(
            ErrorCode.NO_VALID_DEPTH,
            "target geometry contains no finite approximate distances",
        )
    return _finite_percentile(approximate, percentile_value)
