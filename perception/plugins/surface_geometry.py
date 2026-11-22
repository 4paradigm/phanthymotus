"""Bounded decoding and metric geometry for visible semantic surfaces."""
import struct
import zlib

import numpy as np

MAX_PIXELS = 1920 * 1080
MAX_PACKET = 2 * MAX_PIXELS + 65536


def decode_mask(payload, width, height):
    import cv2

    expected = (width, height, 8, 0, 0, 0, 0)
    if (len(payload) < 33 or len(payload) > MAX_PACKET
            or bytes(payload[:16]) != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            or struct.unpack(">IIBBBBB", payload[16:29]) != expected):
        raise ValueError("invalid_mask_png")
    mask = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.shape != (height, width) or mask.dtype != np.uint8:
        raise ValueError("invalid_mask_png")
    return mask


def decode_depth(payload, width, height, scale):
    expected = width * height * 2
    if expected > 2 * MAX_PIXELS or len(payload) > MAX_PACKET:
        raise ValueError("depth_size_limit")
    decoder = zlib.decompressobj()
    raw = decoder.decompress(payload, expected + 1)
    if len(raw) != expected or not decoder.eof or decoder.unused_data:
        raise ValueError("invalid_depth_size")
    depth = np.frombuffer(raw, "<u2").reshape(height, width).astype(np.float32)
    depth *= scale
    return depth


def unavailable(reason, state="insufficient_evidence"):
    return dict(state=state, class_name=None, camera_id=None, frame_id=None,
                timestamp=None, visible_center_m=None, visible_bounds_m=None,
                geometry_scope="class_visible_surface", instance_attribution="unknown",
                multiple_regions_detected=None,
                valid_depth_fraction=0.0, point_count=0, quality="unavailable",
                reason=reason)


def estimate(mask, depth, label, k, max_points):
    if mask.shape != depth.shape or mask.ndim != 2 or mask.size > MAX_PIXELS:
        raise ValueError("image_size_mismatch")
    selected = mask == label
    indices = np.flatnonzero(selected).astype(np.int32)
    total = len(indices)
    if not total:
        return unavailable("empty_mask"), None

    # Count only substantial disconnected regions. A disconnected mask can
    # also come from occlusion, so connectivity never establishes identity.
    multiple_regions = False
    if 16 <= total < selected.size:
        import cv2

        _, _, stats, _ = cv2.connectedComponentsWithStats(
            selected.view(np.uint8), connectivity=8, ltype=cv2.CV_32S)
        areas = stats[1:, cv2.CC_STAT_AREA]
        multiple_regions = np.count_nonzero(areas >= max(8, total * .05)) > 1

    z = depth.ravel()[indices]
    valid = np.isfinite(z) & (z >= 0.15) & (z <= 10.0)
    if not valid.all():
        indices, z = indices[valid], z[valid]
    valid_fraction = len(z) / total
    if len(z) and z.min() != z.max():
        median = np.median(z)
        deviation = np.abs(z - median)
        mad = np.median(deviation)
        inlier = deviation <= max(0.05, 4.4478 * mad)
        if not inlier.all():
            indices, z = indices[inlier], z[inlier]
    count = len(z)
    if count < 8 or count / total < 0.1:
        return dict(unavailable("too_few_inliers"), point_count=count,
                    valid_depth_fraction=valid_fraction,
                    multiple_regions_detected=bool(multiple_regions)), None

    fx, _, cx, _, fy, cy, *_ = k
    rows, cols = np.divmod(indices, mask.shape[1])
    xyz = np.empty((3, count), dtype=np.float32)
    xyz[0] = (cols.astype(np.float32) - cx) * z / fx
    xyz[1] = (rows.astype(np.float32) - cy) * z / fy
    xyz[2] = z

    cloud = None
    if max_points:
        sample = np.linspace(0, count - 1, min(max_points, count), dtype=np.intp)
        cloud = np.ascontiguousarray(xyz[:, sample].T, dtype="<f4")
    # Each axis is contiguous. Percentile partitions it in place once the
    # cloud has been sampled in the original raster order.
    low, center, high = np.percentile(xyz, [2, 50, 98], axis=1, overwrite_input=True)
    good = count >= 32 and count / total >= 0.6 and not multiple_regions
    metrics = dict(
        state="valid" if good else "degraded",
        geometry_scope="class_visible_surface",
        instance_attribution="unknown",
        multiple_regions_detected=bool(multiple_regions),
        visible_center_m=None if multiple_regions else center.tolist(),
        visible_bounds_m={"min": low.tolist(), "max": high.tolist(),
                          "extent": (high - low).tolist()},
        valid_depth_fraction=valid_fraction, point_count=count,
        quality="good" if good else "limited",
        reason=("multiple_class_regions" if multiple_regions
                else None if good else "limited_inlier_support"),
    )
    return metrics, cloud
