import math
import cv2
import numpy as np


def estimate_pose(face, size, camera_matrix=None):
    eyes = face.points[:2]
    delta = eyes[1]-eyes[0]
    eye_distance = float(np.linalg.norm(delta))
    midpoint = eyes.mean(0)
    axis = delta/max(eye_distance, 1.)
    yaw_proxy = float(np.dot(face.points[2]-midpoint, axis)/max(eye_distance, 1.))
    result = {"roll_deg": math.degrees(math.atan2(delta[1], delta[0])),
              "yaw_proxy": yaw_proxy, "approximate": True,
              "camera_calibrated": camera_matrix is not None,
              "pnp_valid": False, "reprojection_px": None}
    if face.landmarks is None:
        return result
    # Generic, non-person-specific 3D template; output is not metric head pose.
    object_points = np.array([(0, 0, 0), (0, 330, 65), (-225, -170, 135),
                              (225, -170, 135), (-150, 150, 125), (150, 150, 125)], np.float64)
    indices = [1, 152, 33, 263, 61, 291] if face.topology == "mediapipe478" else [30, 8, 36, 45, 58, 62]
    image_points = face.landmarks[indices].astype(np.float64)
    w, h = size
    k = (np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], np.float64)
         if camera_matrix is None else np.asarray(camera_matrix, np.float64).reshape(3, 3))
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("invalid camera intrinsics")
    try:
        ok, rvec, tvec = cv2.solvePnP(object_points, image_points, k, None, flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error:
        return result
    if not ok or not np.isfinite(rvec).all() or not np.isfinite(tvec).all() or tvec[2, 0] <= 0:
        return result
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, k, None)
    error = float(np.sqrt(np.mean((projected[:, 0]-image_points)**2)))
    result["reprojection_px"] = error
    if error > max(4., eye_distance*.15):
        return result
    rotation, _ = cv2.Rodrigues(rvec)
    angles = cv2.RQDecomp3x3(rotation)[0]
    result.update(pnp_valid=True, pitch_deg=float(angles[0]), yaw_deg=float(angles[1]))
    return result
