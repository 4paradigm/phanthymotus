"""Calibrated ArUco detection and rendering with the image's OpenCV."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DICTIONARIES = ("DICT_4X4_50", "DICT_5X5_50", "DICT_6X6_250", "DICT_7X7_50")
DEFAULT_DICTIONARY = DICTIONARIES[0]
MAX_REPROJECTION_PX = 3.0


@dataclass(frozen=True)
class Calibration:
    width: int
    height: int
    matrix: np.ndarray
    distortion: np.ndarray | None

    @classmethod
    def from_declaration(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Provide calibration for the input image (width, height, K, distortion_model, D).")
        width, height = value.get("width"), value.get("height")
        if any(type(n) is not int or n <= 0 for n in (width, height)):
            raise ValueError("Calibration needs positive integer width and height.")
        try:
            matrix = np.asarray(value.get("K"), dtype=np.float64).reshape(3, 3)
            coeffs = value.get("D")
            distortion = np.asarray([] if coeffs is None else coeffs, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Calibration requires numeric K (nine values) and D.") from exc
        if (not np.isfinite(matrix).all() or min(matrix[0, 0], matrix[1, 1]) <= 0
                or matrix[0, 1] != 0 or matrix[1, 0] != 0
                or not np.array_equal(matrix[2], [0, 0, 1])):
            raise ValueError("Calibration K must be a finite pinhole matrix with positive focal lengths.")
        if distortion.ndim != 1 or not np.isfinite(distortion).all():
            raise ValueError("Calibration D must be a finite list of distortion coefficients.")
        model = value.get("distortion_model")
        if model in ("pinhole", "none") and not np.any(distortion):
            return cls(width, height, matrix, None)
        if ((model == "plumb_bob" and distortion.size in (4, 5))
                or (model == "rational_polynomial" and distortion.size == 8)):
            return cls(width, height, matrix, distortion)
        raise ValueError("Calibration needs a pinhole, plumb_bob, or rational_polynomial model with matching D. "
                         "For an unsupported camera model, supply calibration for a rectified stream.")


class MarkerDetector:
    def __init__(self, dictionary):
        try:
            import cv2

            aruco = cv2.aruco
            get_dictionary = getattr(aruco, "getPredefinedDictionary", None) or aruco.Dictionary_get
            marker_dictionary = get_dictionary(getattr(aruco, dictionary))
            if hasattr(aruco, "ArucoDetector"):
                self._detect = aruco.ArucoDetector(marker_dictionary, aruco.DetectorParameters()).detectMarkers
            else:
                parameters = aruco.DetectorParameters_create()
                self._detect = lambda gray: aruco.detectMarkers(gray, marker_dictionary, parameters=parameters)
            # Check the rest of the native API before accepting a camera stream.
            if not all(callable(getattr(cv2, name, None)) for name in (
                "solvePnP", "solvePnPGeneric", "projectPoints", "Rodrigues", "drawFrameAxes",
            )) or not hasattr(cv2, "SOLVEPNP_IPPE_SQUARE"):
                raise AttributeError("calibrated pose API is unavailable")
        except (AttributeError, ImportError, OSError) as exc:
            raise RuntimeError("This Perception image lacks OpenCV ArUco or the calibrated pose API. "
                               "Use a compatible Perception image.") from exc
        self.cv = cv2

    def estimate(self, image, calibration, marker_size_m):
        """Draw accepted poses in-place and return marker→camera measurements."""
        cv = self.cv
        if image.shape[:2] != (calibration.height, calibration.width):
            raise ValueError("Decoded frame dimensions differ from camera calibration.")
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        corners, ids, _ = self._detect(gray)
        if ids is None:
            return []
        half = marker_size_m / 2
        # OpenCV IPPE_SQUARE order matches the detected, marker-oriented corners.
        square = np.array([[-half, half, 0], [half, half, 0],
                           [half, -half, 0], [-half, -half, 0]], dtype=np.float64)
        markers = []
        for marker_id, quad in zip(ids.reshape(-1), corners):
            pixels = np.asarray(quad, dtype=np.float64).reshape(4, 2)
            pose = self._solve(square, pixels, calibration)
            if pose is None:
                continue
            rms, rotation, translation = pose
            cv.drawFrameAxes(image, calibration.matrix, calibration.distortion,
                             rotation, translation, half)
            cv.polylines(image, [np.rint(pixels).astype(np.int32)], True, (0, 200, 255), 1)
            origin = tuple(np.rint(pixels[0]).astype(int))
            cv.putText(image, str(int(marker_id)), origin, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            markers.append({"id": int(marker_id), "translation_m": translation.reshape(3).tolist(),
                            "rotation_rvec_rad": rotation.reshape(3).tolist(), "reprojection_rms_px": rms})
        return sorted(markers, key=lambda marker: marker["id"])

    def _solve(self, square, pixels, calibration):
        cv = self.cv
        matrix, distortion = calibration.matrix, calibration.distortion
        candidates = []
        try:
            _, rotations, translations, *_ = cv.solvePnPGeneric(
                square, pixels, matrix, distortion, flags=cv.SOLVEPNP_IPPE_SQUARE)
            candidates.extend(zip(rotations, translations))
        except cv.error:
            pass
        # IPPE can return degenerate candidates for a square viewed head-on.
        try:
            ok, rotation, translation = cv.solvePnP(
                square, pixels, matrix, distortion, flags=cv.SOLVEPNP_ITERATIVE)
            if ok:
                candidates.append((rotation, translation))
        except cv.error:
            pass
        best = None
        for rotation, translation in candidates:
            if not (np.isfinite(rotation).all() and np.isfinite(translation).all()):
                continue
            axes, _ = cv.Rodrigues(rotation)
            if np.any((square @ axes.T + translation.reshape(3))[:, 2] <= 0):
                continue
            projected, _ = cv.projectPoints(square, rotation, translation, matrix, distortion)
            residual = projected.reshape(4, 2) - pixels
            rms = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
            if rms <= MAX_REPROJECTION_PX and (best is None or rms < best[0]):
                best = rms, rotation, translation
        return best
