"""ArUco detection and pose estimation with OpenCV."""

from __future__ import annotations

import functools

import numpy as np

from plugins.camera_info import camera_matrix


DICTIONARIES = ("DICT_4X4_50", "DICT_5X5_50", "DICT_6X6_250", "DICT_7X7_50")
DEFAULT_DICTIONARY = DICTIONARIES[0]
# OpenCV's distortion vector lengths.
_DISTORTION_SIZES = (4, 5, 8, 12, 14)


@functools.lru_cache(maxsize=1)
def unavailable_reason() -> str:
    """Empty when this image's OpenCV has ArUco, otherwise why not."""
    try:
        import cv2
        cv2.aruco.getPredefinedDictionary
    except (ImportError, AttributeError, OSError) as exc:
        return f"This Perception image's OpenCV has no ArUco module: {exc}"
    return ""


def intrinsics(declaration, width, height):
    """`(K, D, pose_status)` for a `width` x `height` frame.

    `calibrated` when the camera declares K, `approximate` when K is derived
    from its declared field of view (pinhole, no distortion), and
    `uncalibrated` with no K when it declares neither — markers are still
    detected then, only without a metric pose. A K declared at another
    resolution is rescaled to the frame.
    """
    if not declaration:
        return None, None, "uncalibrated"
    try:
        matrix, declared_w, declared_h = camera_matrix(declaration)
    except ValueError:
        return None, None, "uncalibrated"
    matrix = matrix.astype(np.float64)
    matrix[0] *= width / declared_w
    matrix[1] *= height / declared_h
    if declaration.get("K") in (None, "", []):
        return matrix, None, "approximate"
    coeffs = declaration.get("D")
    distortion = None
    if isinstance(coeffs, (list, tuple)) and len(coeffs) in _DISTORTION_SIZES:
        distortion = np.asarray(coeffs, dtype=np.float64)
        if not np.isfinite(distortion).all() or not distortion.any():
            distortion = None
    return matrix, distortion, "calibrated"


class MarkerDetector:
    def __init__(self, dictionary):
        import cv2

        aruco = cv2.aruco
        marker_dictionary = aruco.getPredefinedDictionary(getattr(aruco, dictionary))
        if hasattr(aruco, "ArucoDetector"):  # OpenCV >= 4.7
            self._detect = aruco.ArucoDetector(marker_dictionary, aruco.DetectorParameters()).detectMarkers
        else:  # older OpenCV (JetPack 5)
            parameters = aruco.DetectorParameters_create()
            self._detect = lambda gray: aruco.detectMarkers(gray, marker_dictionary, parameters=parameters)
        self.cv = cv2

    def detect(self, image, marker_size_m, matrix, distortion, *, draw=False):
        """Markers in `image` sorted by ID, with a pose when `matrix` is given."""
        cv = self.cv
        corners, ids, _ = self._detect(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
        if ids is None:
            return []
        half = marker_size_m / 2
        # The corner order SOLVEPNP_IPPE_SQUARE expects, which is the detector's.
        square = np.array([[-half, half, 0], [half, half, 0],
                           [half, -half, 0], [-half, -half, 0]], dtype=np.float64)
        markers = []
        for marker_id, quad in zip(ids.reshape(-1), corners):
            pixels = quad.reshape(4, 2).astype(np.float64)
            marker = {"id": int(marker_id), "center_px": [round(float(v), 1) for v in pixels.mean(axis=0)]}
            pose = self._pose(square, pixels, matrix, distortion) if matrix is not None else None
            if pose is not None:
                rvec, tvec = pose
                marker.update(translation_m=[round(float(v), 4) for v in tvec.reshape(3)],
                              rotation_rvec_rad=[round(float(v), 4) for v in rvec.reshape(3)],
                              distance_m=round(float(np.linalg.norm(tvec)), 4))
                if draw:
                    cv.drawFrameAxes(image, matrix, distortion, rvec, tvec, half)
            if draw:
                outline = np.rint(pixels).astype(np.int32)
                cv.polylines(image, [outline], True, (0, 200, 255), 1)
                cv.putText(image, str(marker["id"]), tuple(int(v) for v in outline[0]),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            markers.append(marker)
        return sorted(markers, key=lambda m: m["id"])

    def _pose(self, square, pixels, matrix, distortion):
        """`(rvec, tvec)` with the marker in front of the camera and facing it, or None.

        IPPE first. OpenCV 4.6's IPPE fails on a marker seen exactly head-on
        (both candidates are the identity, 141 px off), so fall back to the
        iterative solver, which handles that case.
        """
        cv = self.cv
        candidates = []
        try:
            _, rvecs, tvecs, _ = cv.solvePnPGeneric(square, pixels, matrix, distortion,
                                                    flags=cv.SOLVEPNP_IPPE_SQUARE)
            candidates += zip(rvecs, tvecs)
        except cv.error:
            pass
        try:
            ok, rvec, tvec = cv.solvePnP(square, pixels, matrix, distortion, flags=cv.SOLVEPNP_ITERATIVE)
            if ok:
                candidates.append((rvec, tvec))
        except cv.error:
            pass
        for rvec, tvec in candidates:
            if np.isfinite(tvec).all() and tvec[2, 0] > 0 and cv.Rodrigues(rvec)[0][2, 2] < 0:
                return rvec, tvec
        return None
