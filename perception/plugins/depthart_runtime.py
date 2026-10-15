"""K-aware metric depth using a prebuilt TensorRT engine and SelectiveScan plugin."""
from __future__ import annotations

import ctypes
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def camera_matrix(camera):
    """Read a motus.camera/1 declaration; K is expressed at width × height."""
    if not isinstance(camera, dict):
        raise ValueError("DepthART requires camera_info with K, width and height")
    try:
        matrix = np.asarray(camera["K"], dtype=np.float32)
        width, height = int(camera["width"]), int(camera["height"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("DepthART requires camera_info with a nine-value K, width and height") from exc
    if matrix.shape != (9,) or width <= 0 or height <= 0:
        raise ValueError("camera_info must contain a nine-value K and positive width/height")
    matrix = matrix.reshape(3, 3).copy()
    if (not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1], atol=1e-6)
            or abs(float(matrix[1, 0])) > 1e-6):
        raise ValueError("camera_info K must be finite pinhole intrinsics with positive focal lengths")
    return matrix, width, height


# A centre crop is free on the axis it does not touch, but it is still a loss
# of field of view, and past a point it stops being a crop and becomes a slit.
# A 4:3 source into this canvas gives up 26% of its height, a 16:10 source 7%,
# a square 44%; a portrait phone would give up 69%. Refuse beyond half, where
# what is left no longer describes the scene the camera was pointed at.
MAX_CROP_FRACTION = 0.5


def _rounded(value, minimum):
    size = int(np.round(value / 32) * 32)
    return size if size >= minimum else int(np.ceil(value / 32) * 32)


def canvas_resize(width, height, input_size):
    """The engine-canvas size this frame resizes to — the official rule.

    Lower-bound cover, rounded to a multiple of 32. Equal to `input_size` when
    the frame can enter the canvas as-is.
    """
    target_w, target_h = input_size
    scale = max(target_w / width, target_h / height)
    return _rounded(width * scale, target_w), _rounded(height * scale, target_h)


def fit_to_canvas(width, height, input_size):
    """The centre crop that lets this frame enter a fixed engine canvas.

    Returns `(x0, y0, w, h)` in frame pixels — the whole frame whenever the
    resize already lands on the canvas, which the rounding to multiples of 32
    makes true for a band of ratios, not just the canvas's exact one. 16:9 into
    an 864x480 canvas needs no crop even though 864/480 is 1.8; cropping it to
    1.8 anyway would throw away 9 rows for nothing.

    **Why a crop and not a resize to fit.** The engine's input binding is
    static (`DepthARTSession.input_size` is read off the engine), so every
    frame has to become exactly that tensor. Stretching a 4:3 frame into a 1.8
    canvas is one line and silently wrong: it squashes the world horizontally
    by 1.35x while `K` still says the pixels are square, and DepthART derives
    its metric scale *from* K. The output would be confidently wrong metres on
    a topic whose contract is metres — the failure mode this card exists to
    avoid. Cropping keeps every remaining pixel at its true angle; only the
    principal point moves, which `prepare_image` folds into K.

    Letterboxing would keep the field of view instead, at the cost of feeding
    the network grey bars it never saw in training, whose depth would then have
    to be masked. Rebuilding the engine for another canvas is the only thing
    that removes the limit outright, and that is an artifact problem rather
    than a code one.
    """
    target_w, target_h = input_size
    if canvas_resize(width, height, input_size) == (target_w, target_h):
        return 0, 0, width, height

    want = target_w / target_h
    if width / height > want:             # too wide: trim the sides
        crop_w, crop_h = int(round(height * want)), height
    else:                                 # too tall: trim top and bottom
        crop_w, crop_h = width, int(round(width / want))
    crop_w, crop_h = max(1, min(width, crop_w)), max(1, min(height, crop_h))
    lost = 1.0 - (crop_w * crop_h) / float(width * height)
    if lost > MAX_CROP_FRACTION:
        raise ValueError(
            f"DepthART engine canvas {target_w}x{target_h} cannot take a "
            f"{width}x{height} image: fitting it would crop away "
            f"{lost * 100:.0f}% of the frame. Feed it a source closer to "
            f"{want:.2f}:1.")
    return (width - crop_w) // 2, (height - crop_h) // 2, crop_w, crop_h


def prepare_image(frame, camera, input_size):
    """Official lower-bound/multiple-of-32 resize, with matching K and RGB normalization.

    Returns `(blob, K, crop)` where `crop` is `(x0, y0, w, h)` in frame pixels.
    """
    import cv2

    matrix, camera_w, camera_h = camera_matrix(camera)
    if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) <= 0:
        raise ValueError("DepthART requires a nonempty BGR image")
    height, width = frame.shape[:2]
    target_w, target_h = input_size

    # K arrives expressed at the declaration's resolution, which may be a
    # higher-resolution version of this same frame. Bring it to frame pixels
    # first, because the crop offsets below are in frame pixels and mixing the
    # two coordinate systems is the kind of error that still produces a
    # plausible number.
    matrix[0] *= width / camera_w
    matrix[1] *= height / camera_h

    x0, y0, crop_w, crop_h = fit_to_canvas(width, height, input_size)
    if (crop_w, crop_h) != (width, height):
        frame = frame[y0:y0 + crop_h, x0:x0 + crop_w]
        # A crop moves the principal point and leaves the focal lengths alone.
        # So a vertical crop costs nothing on the horizontal axis — the half
        # field of view a camera declares stays exactly true for the width.
        matrix[0, 2] -= x0
        matrix[1, 2] -= y0
        height, width = crop_h, crop_w

    new_w, new_h = canvas_resize(width, height, input_size)
    if (new_w, new_h) != (target_w, target_h):
        raise ValueError(
            f"DepthART engine canvas {target_w}x{target_h} does not support this image aspect ratio "
            f"({width}x{height}); its aspect-preserving resize is {new_w}x{new_h}"
        )
    matrix[0] *= new_w / width
    matrix[1] *= new_h / height
    image = frame[:, :, ::-1].astype(np.float32) / 255.0
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    image = (image - np.asarray([.485, .456, .406], dtype=np.float32)) / np.asarray([.229, .224, .225], dtype=np.float32)
    blob = np.ascontiguousarray(image.transpose(2, 0, 1)[None])
    return blob, matrix[None], (x0, y0, crop_w, crop_h)


class _CameraSession:
    def __init__(self, session, camera):
        self._session, self._camera = session, camera

    def infer(self, frame):
        camera = self._camera() if callable(self._camera) else self._camera
        return self._session.infer(frame, camera)


class DepthARTSession:
    def __init__(self, engine_path, plugin_path):
        from utils.tensorrt_runtime import TensorRTEngine

        if not engine_path or not plugin_path:
            raise ValueError("DepthART requires provisioned depthart_engine_path and depthart_plugin_path")
        engine_path, plugin_path = Path(engine_path), Path(plugin_path)
        if not engine_path.is_file() or not plugin_path.is_file():
            raise FileNotFoundError("DepthART engine/plugin missing; provision artifacts matching this JetPack/TensorRT")
        self._plugin = ctypes.CDLL(str(plugin_path.resolve()), mode=ctypes.RTLD_GLOBAL)
        try:
            version = self._plugin.depthart_selective_scan_trt_version
        except AttributeError as exc:
            raise ValueError("DepthART plugin is missing its version symbol; provision a matching SelectiveScan plugin") from exc
        version.restype = ctypes.c_char_p
        if version() != b"SelectiveScan-1":
            raise ValueError("unsupported DepthART SelectiveScan plugin version")
        self._engine = TensorRTEngine(engine_path, primary_input="image")
        shape = self._engine.input_shape
        if (shape is None or len(shape) != 4 or shape[:2] != (1, 3)
                or self._engine.auxiliary_shapes != {"camera_K": (1, 3, 3)}):
            self._engine.close()
            raise ValueError("DepthART requires static image [1,3,H,W] and camera_K [1,3,3] inputs")
        self.input_size = (shape[3], shape[2])
        # The crop in force, for whoever reports what this card can see.
        self.last_crop = None
        self._crop_logged = False

    def for_camera(self, camera):
        return _CameraSession(self, camera)

    def infer(self, frame, camera):
        blob, matrix, crop = prepare_image(frame, camera, self.input_size)
        self.last_crop = crop
        height, width = frame.shape[:2]
        if not self._crop_logged and (crop[2], crop[3]) != (width, height):
            # Once per engine, not per frame: at 2 fps a per-frame line would
            # bury the log, and a crop is a property of the wiring, not of this
            # frame. Said at all because the depth map then covers less than
            # the camera does, and nothing else downstream would reveal it.
            self._crop_logged = True
            log.warning(
                "[depthart] %dx%d source centre-cropped to %dx%d to fit the "
                "%dx%d engine canvas — the depth map covers %.0f%% of the "
                "frame's height. Horizontal field of view is unaffected.",
                width, height, crop[2], crop[3], *self.input_size,
                crop[3] / height * 100)
        return self._engine.infer(blob, auxiliary_inputs={"camera_K": matrix}), None

    def close(self):
        self._engine.close()
