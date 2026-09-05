from __future__ import annotations

import inspect

from .contracts import (
    DepthBackend,
    IndoorDistanceBackend,
    InstanceSegmentationBackend,
    SceneDomain,
)


def _accepts_protocol_call(
    backend: object,
    protocol: type,
    method_name: str,
    arguments: tuple[object, ...],
) -> bool:
    try:
        if not isinstance(backend, protocol):
            return False
        method = getattr(backend, method_name)
        if not callable(method):
            return False
        inspect.signature(method).bind(*arguments)
    except Exception:
        return False
    return True


def is_valid_depth_backend(backend: object) -> bool:
    return _accepts_protocol_call(
        backend,
        DepthBackend,
        "predict_depth",
        (b"", SceneDomain.INDOOR, 0.0),
    )


def is_valid_indoor_distance_backend(backend: object) -> bool:
    return _accepts_protocol_call(
        backend,
        IndoorDistanceBackend,
        "predict_indoor_distance",
        (b"", 0.0),
    )


def is_valid_segmentation_backend(backend: object) -> bool:
    return _accepts_protocol_call(
        backend,
        InstanceSegmentationBackend,
        "predict_instances",
        (b"", 0.0),
    )
