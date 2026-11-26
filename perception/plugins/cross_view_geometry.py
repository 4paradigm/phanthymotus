"""Metric association of visible target observations from two optical frames."""
import numpy as np


def calibration(record, camera_id, *, require_registered=True):
    """Validate a surveyed camera-to-world transform and rectified intrinsics."""
    if record["camera_id"] != camera_id or record["direction"] != "camera_to_world":
        raise ValueError("calibration_identity_or_direction")
    if record["rectified"] is not True or (require_registered and record.get("depth_registered") is not True):
        raise ValueError("unregistered_depth_or_distorted_image")
    k = np.asarray(record["K"], dtype=float).reshape(3, 3)
    pose = np.asarray(record["T_world_camera"], dtype=float).reshape(4, 4)
    width, height = record["size"]
    error = float(record["error_m"])
    r = pose[:3, :3]
    if (not np.isfinite(k).all() or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1])
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(r), 1)
            or not np.allclose(k[2], [0, 0, 1])
            or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.isclose(k[1, 0], 0)
            or not all(type(v) is int and 0 < v <= 8192 for v in (width, height))
            or not 0 <= k[0, 2] < width or not 0 <= k[1, 2] < height
            or not np.isfinite(error) or error < 0 or not record["world_frame"]):
        raise ValueError("invalid_calibration")
    return k, pose, width, height, error


def observations(objects, depth, camera, minimum):
    k, pose, width, height, _ = camera
    dh, dw = depth.shape
    targets = []
    for index, obj in enumerate(objects):
        target = {"id": index, "label": obj.get("name"), "reason": "invalid_detection_geometry"}
        targets.append(target)
        try:
            score = float(obj["confidence"])
            x1, y1, x2, y2 = map(float, obj["bbox"])
            if not (isinstance(target["label"], str) and minimum <= score <= 1
                    and 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                continue
            u, v = (x1 + x2) / 2, (y1 + y2) / 2
            col, row = int(u * dw / width), int(v * dh / height)
            patch = depth[max(0, row-2):row+3, max(0, col-2):col+3]
            valid = patch[np.isfinite(patch) & (patch > 0)]
            if len(valid) < max(3, patch.size * 0.6):
                target["reason"] = "missing_target_depth"
                continue
            z = float(np.median(valid))
            spread = float(np.percentile(valid, 90) - np.percentile(valid, 10))
            if spread > 0.08:
                target["reason"] = "mixed_target_depth_possible_partial_occlusion"
                continue
            point = pose[:3, :3] @ (np.linalg.solve(k, [u, v, 1]) * z) + pose[:3, 3]
            target.update(point=point, score=score * len(valid) / patch.size,
                          spread=spread, pixel=[u, v], reason=None)
        except (KeyError, TypeError, ValueError, OverflowError, np.linalg.LinAlgError):
            continue
    return targets


def visibility(point, camera, depth, margin):
    k, pose, width, height, error = camera
    xyz = pose[:3, :3].T @ (point - pose[:3, 3])
    if xyz[2] <= 0:
        return "outside_other_view", None
    uv = (k @ xyz)[:2] / xyz[2]
    u, v = uv.tolist()
    if not (0 <= u < width and 0 <= v < height):
        return "outside_other_view", [u, v]
    dh, dw = depth.shape
    col, row = int(u * dw / width), int(v * dh / height)
    patch = depth[max(0, row-1):row+2, max(0, col-1):col+2]
    if np.isfinite(patch).sum() < patch.size * 0.6:
        return "other_view_depth_missing", [u, v]
    if np.nanmedian(patch) + margin + error < xyz[2]:
        return "possible_occlusion", [u, v]
    return "not_detected_in_other_view", [u, v]


def compare(objects, depths, cameras, tolerance, gate, ambiguity, minimum):
    a, b = [observations(o, d, c, minimum) for o, d, c in zip(objects, depths, cameras)]
    uncertainty = sum(camera[4] for camera in cameras)
    edges = {}
    for left in a:
        if left["reason"]:
            continue
        for right in b:
            if right["reason"] or left["label"] != right["label"]:
                continue
            distance = float(np.linalg.norm(left["point"] - right["point"]))
            if distance <= gate:
                edges[left["id"], right["id"]] = distance

    ranked = [{}, {}]
    for pair, distance in edges.items():
        for side, index in enumerate(pair):
            ranked[side].setdefault(index, []).append((distance, pair))
    competing = set()
    for groups in ranked:
        for choices in groups.values():
            choices.sort()
            margin = ambiguity + uncertainty
            if len(choices) > 1 and choices[1][0] - choices[0][0] <= margin:
                competing.update(pair for distance, pair in choices if distance - choices[0][0] <= margin)

    matched = []
    used = [set(), set()]
    for (i, j), distance in sorted(edges.items()):
        pair = (i, j)
        if pair in competing or any(ranked[s][n][0][1] != pair for s, n in enumerate(pair)):
            continue
        spread = uncertainty + a[i]["spread"] + b[j]["spread"]
        threshold = tolerance + spread
        state = "inconsistent" if distance > threshold else "consistent"
        if spread >= tolerance:
            state = "insufficient_evidence"
        confidence = min(a[i]["score"], b[j]["score"]) * max(0, 1 - spread / tolerance)
        matched.append(dict(a_id=i, b_id=j, label=a[i]["label"], state=state,
                            position_residual_m=distance, confidence=confidence,
                            a_pixel=a[i]["pixel"], b_pixel=b[j]["pixel"]))
        used[0].add(i)
        used[1].add(j)

    unmatched = []
    for side, targets in enumerate((a, b)):
        for target in targets:
            index = target["id"]
            if index in used[side]:
                continue
            reason, pixel = target["reason"], None
            if ranked[side].get(index):
                reason = "ambiguous_assignment"
            elif reason is None:
                reason, pixel = visibility(target["point"], cameras[1-side],
                                           depths[1-side], tolerance + uncertainty)
            unmatched.append(dict(view="ab"[side], id=index, label=target["label"],
                                  reason=reason, other_pixel=pixel))

    states = {item["state"] for item in matched}
    if any(item["reason"] == "ambiguous_assignment" for item in unmatched):
        state, reason = "ambiguous", "competing_geometric_matches"
    elif "inconsistent" in states:
        state, reason = "inconsistent", "residual_exceeds_uncertainty"
    elif matched and not unmatched and states == {"consistent"}:
        state, reason = "consistent", "geometry_agrees"
    else:
        state, reason = "insufficient_evidence", "incomplete_geometric_evidence"
    return dict(state=state, reason=reason, matched_targets=matched, unmatched_targets=unmatched,
                position_residual_m=max((m["position_residual_m"] for m in matched), default=None),
                confidence=min((m["confidence"] for m in matched), default=0) if not unmatched else 0)



def epipolar_candidates(objects, cameras, tolerance_px):
    """Count compatible box centers in rectified images without inferring identity or range."""
    ka, a, *_ = cameras[0]
    kb, b, *_ = cameras[1]
    r = b[:3, :3].T @ a[:3, :3]
    x, y, z = b[:3, :3].T @ (a[:3, 3] - b[:3, 3])
    if np.linalg.norm([x, y, z]) < 1e-6:
        return None
    essential = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]]) @ r
    f = np.linalg.inv(kb).T @ essential @ np.linalg.inv(ka)

    def centers(records, camera):
        _, _, width, height, _ = camera
        for obj in records:
            try:
                x1, y1, x2, y2 = map(float, obj["bbox"])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height:
                yield obj.get("name"), np.array([(x1 + x2) / 2, (y1 + y2) / 2, 1])

    right = list(centers(objects[1], cameras[1]))
    compatible = 0
    for label, point in centers(objects[0], cameras[0]):
        line_b = f @ point
        for other_label, other_point in right:
            if not isinstance(label, str) or label != other_label:
                continue
            line_a = f.T @ other_point
            norm = min(np.linalg.norm(line_a[:2]), np.linalg.norm(line_b[:2]))
            if norm > 0 and abs(other_point @ line_b) <= tolerance_px * norm:
                compatible += 1
    return compatible
