"""Optional measured PiPER plane mapping, independent fit/check evidence."""

from copy import deepcopy
import cv2
import numpy as np
from .contracts import need, number, text, vector


def fit(record):
    need(
        record.get("provenance") == "operator_measured", "MEASURED_CALIBRATION_REQUIRED"
    )
    keys = (
        "piper_id",
        "camera_id",
        "frame_id",
        "view_id",
        "image_size",
        "distortion",
        "created_at",
        "expires_at",
        "measurement_record",
        "object_plane_record",
        "class_name",
        "valid_pixel_bounds",
    )
    out = {k: deepcopy(record[k]) for k in keys}
    for k in (
        "piper_id",
        "camera_id",
        "frame_id",
        "view_id",
        "measurement_record",
        "object_plane_record",
        "class_name",
    ):
        text(out[k])
    fit_pixels = np.array([vector(p, 2) for p in record["fit_pixels"]], float)
    fit_mm = np.array([vector(p, 2) for p in record["fit_mm"]], float)
    check_pixels = np.array([vector(p, 2) for p in record["check_pixels"]], float)
    check_mm = np.array([vector(p, 2) for p in record["check_mm"]], float)
    need(
        len(fit_pixels) >= 4
        and fit_pixels.shape == fit_mm.shape
        and len(check_pixels) >= 3
        and check_pixels.shape == check_mm.shape,
        "INDEPENDENT_POINTS_REQUIRED",
    )
    need(
        np.linalg.matrix_rank(np.c_[fit_pixels, np.ones(len(fit_pixels))]) == 3,
        "COLLINEAR_CALIBRATION",
    )
    need(
        len(np.unique(check_pixels, axis=0)) == len(check_pixels)
        and all(
            np.min(np.linalg.norm(fit_pixels - p, axis=1)) > 1e-6 for p in check_pixels
        ),
        "INDEPENDENT_POINTS_REQUIRED",
    )
    matrix, _ = cv2.findHomography(fit_pixels, fit_mm, 0)
    need(
        matrix is not None
        and np.isfinite(matrix).all()
        and np.linalg.cond(matrix) < 1e9,
        "INVALID_CALIBRATION",
    )

    def project(h, points):
        p = np.c_[points, np.ones(len(points))] @ h.T
        need(np.min(np.abs(p[:, 2])) > 1e-9, "SINGULAR_PROJECTION")
        return p[:, :2] / p[:, 2, None]

    fit_error = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (project(np.linalg.inv(matrix), fit_mm) - fit_pixels) ** 2, axis=1
                )
            )
        )
    )
    check_error = float(
        np.max(np.linalg.norm(project(matrix, check_pixels) - check_mm, axis=1))
    )
    need(
        0 < number(record["max_reprojection_px"]) <= 5
        and 0 < number(record["max_check_mm"]) <= 10,
        "EXPLICIT_CALIBRATION_ERROR_LIMITS_REQUIRED",
    )
    uncertainty = number(record["measurement_uncertainty_mm"])
    need(0 < uncertainty <= 10, "MEASUREMENT_UNCERTAINTY_REQUIRED")
    need(
        fit_error <= record["max_reprojection_px"]
        and check_error <= record["max_check_mm"],
        "CALIBRATION_RESIDUAL_EXCEEDED",
    )
    bounds = vector(out["valid_pixel_bounds"], 4)
    need(bounds[0] < bounds[2] and bounds[1] < bounds[3], "INVALID_CALIBRATION_BOUNDS")
    hull = cv2.convexHull(fit_pixels.astype(np.float32))
    need(
        all(
            cv2.pointPolygonTest(hull, (x, y), False) >= 0
            for x in (bounds[0], bounds[2])
            for y in (bounds[1], bounds[3])
        ),
        "CALIBRATION_EXTRAPOLATION",
    )
    out.update(
        provenance="operator_measured",
        pixel_to_plane_mm=matrix.tolist(),
        max_error_mm=max(check_error, uncertainty),
        checks=dict(
            reprojection_rmse_px=fit_error,
            independent_max_mm=check_error,
            fit_pixels=fit_pixels.tolist(),
            fit_mm=fit_mm.tolist(),
            check_pixels=check_pixels.tolist(),
            check_mm=check_mm.tolist(),
        ),
    )
    return out
