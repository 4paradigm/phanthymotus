"""CPU region-state comparison. Scores describe pixels, never object identity."""

import base64
from copy import deepcopy

import cv2
import numpy as np

from .contracts import fields, need, number, vector

FRAME_SCHEMA = "motus.roi-image-observation/1"
DEFAULTS = {
    "change_threshold": 25,
    "min_change_fraction": 0.04,
    "stable_frames": 2,
    "stability_threshold": 8,
}


def settings(region, options=None):
    x1, y1, x2, y2 = vector(region, 4)
    need(0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1, "INVALID_TARGET_ROI")
    options = options if options is not None else {}
    fields(options, (), DEFAULTS)
    out = {**DEFAULTS, **deepcopy(options)}
    need(8 <= number(out["change_threshold"]) <= 80, "INVALID_ROI_THRESHOLD")
    need(0.005 <= number(out["min_change_fraction"]) <= 0.5, "INVALID_ROI_AREA")
    need(
        type(out["stable_frames"]) is int and 2 <= out["stable_frames"] <= 5,
        "INVALID_STABLE_FRAMES",
    )
    need(2 <= number(out["stability_threshold"]) <= 40, "INVALID_STABILITY_THRESHOLD")
    return [x1, y1, x2, y2], out


def prepare(packet, binding, now, max_age):
    """Only locally captured packets enter here; no public pixel-submission API."""
    meta, image = deepcopy(packet["frame"]), packet["pixels"]
    need(meta["schema"] == FRAME_SCHEMA, "ROI_IMAGE_REQUIRED")
    for key in ("source_id", "camera_id", "frame_id", "distortion", "view_id"):
        need(meta[key] == binding[key], "SOURCE_BINDING_MISMATCH")
    need(meta["timestamp_basis"] == "capture", "CAPTURE_TIME_UNPROVEN")
    need(
        meta["provenance"] == "real"
        or (
            meta["provenance"] in ("synthetic", "replay")
            and binding.get("allow_test_data") is True
        ),
        "TEST_DATA_REFUSED",
    )
    need(0 <= now - number(meta["captured_at"]) <= max_age, "STALE_OR_FUTURE")
    need(
        isinstance(image, np.ndarray)
        and image.dtype == np.uint8
        and image.ndim == 3
        and image.shape[2] == 3
        and meta["image_size"] == [image.shape[1], image.shape[0]],
        "INVALID_IMAGE",
    )
    need(
        1 <= image.shape[0] <= 8192 and 1 <= image.shape[1] <= 8192,
        "INVALID_IMAGE_SIZE",
    )
    need(2 < float(image.mean()) < 253, "IMAGE_SATURATED_OR_EMPTY")
    scale = min(1.0, 640 / image.shape[1], 480 / image.shape[0])
    image = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return meta, image


def crop(image, region):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = region
    bounds = [
        int(x1 * w),
        int(y1 * h),
        min(w, int(np.ceil(x2 * w))),
        min(h, int(np.ceil(y2 * h))),
    ]
    a, b, c, d = bounds
    need(c - a >= 8 and d - b >= 8, "ROI_TOO_SMALL")
    return image[b:d, a:c], bounds


def jpeg(image):
    scale = min(1.0, 240 / max(image.shape[:2]))
    thumbnail = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
    )
    ok, data = cv2.imencode(".jpg", thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 80])
    need(ok, "IMAGE_ENCODING_FAILED")
    return base64.b64encode(data).decode("ascii")


def statistics(image, region):
    patch, _ = crop(image, region)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return {
        "mean_brightness": float(gray.mean()),
        "edge_energy": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
    }


def compare(before, afters, region, options):
    """Simple difference + connected area + fresh-frame stability, with guards."""
    need(len(afters) == options["stable_frames"], "ROI_FRAMES_REQUIRED")
    base = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    _, bounds = crop(before, region)
    x1, y1, x2, y2 = bounds
    context = np.ones(base.shape, bool)
    context[max(0, y1 - 4) : y2 + 4, max(0, x1 - 4) : x2 + 4] = False
    need(int(context.sum()) >= 64, "ROI_CONTEXT_TOO_SMALL")
    scores, normalized = [], []
    for after in afters:
        need(after.shape == before.shape, "VIEW_OR_RESOLUTION_CHANGED")
        gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
        offset = float(np.median(gray[context].astype(float) - base[context]))
        need(abs(offset) <= 35, "ROI_LIGHTING_CHANGED")
        corrected = np.clip(after.astype(np.float32) - offset, 0, 255).astype(np.uint8)
        corrected_gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
        context_diff = cv2.absdiff(base, corrected_gray)[context]
        need(
            float(np.mean(context_diff > options["change_threshold"])) < 0.35,
            "ROI_CAMERA_OR_OCCLUSION_CHANGED",
        )
        # Use background structure when available; do not require texture on a
        # plain tabletop or pretend to have recovered camera pose.
        if float(base[context].std()) > 10:
            a = np.where(context, base, 0).astype(np.float32)
            b = np.where(context, corrected_gray, 0).astype(np.float32)
            shift, response = cv2.phaseCorrelate(a, b)
            need(not (response > 0.2 and np.hypot(*shift) > 3), "ROI_CAMERA_MOVED")
        patch0 = before[y1:y2, x1:x2]
        patch1 = corrected[y1:y2, x1:x2]
        difference = cv2.absdiff(
            cv2.GaussianBlur(patch0, (3, 3), 0), cv2.GaussianBlur(patch1, (3, 3), 0)
        ).max(axis=2)
        mask = (difference >= options["change_threshold"]).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, _, components, _ = cv2.connectedComponentsWithStats(mask, 8)
        fraction = float(mask.mean())
        largest = (
            float(components[1:, cv2.CC_STAT_AREA].max() / mask.size)
            if count > 1
            else 0.0
        )
        need(fraction < 0.8, "ROI_OCCLUDED_OR_REPLACED")
        if fraction >= options["min_change_fraction"]:
            # Neutral local illumination/shadow/reflection can imitate an object.
            # Refuse achromatic-only changes; color change is still not identity proof.
            delta = patch1.astype(np.int16) - patch0.astype(np.int16)
            neutral = np.ptp(delta, axis=2)[mask.astype(bool)] <= 4
            need(float(neutral.mean()) < 0.98, "ROI_PHOTOMETRIC_AMBIGUITY")

        sharp0 = float(
            cv2.Laplacian(cv2.cvtColor(patch0, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()
        )
        sharp1 = float(
            cv2.Laplacian(cv2.cvtColor(patch1, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()
        )
        need(not (sharp0 > 30 and sharp1 < sharp0 * 0.15), "ROI_BLURRED")
        scores.append(
            {
                "changed_fraction": fraction,
                "largest_component_fraction": largest,
                "mean_difference": float(difference.mean()),
                "brightness_offset": offset,
            }
        )
        normalized.append(patch1)
    instability = 0.0
    # JetPack 5 / the official Perception image still ships Python 3.8.
    for a, b in zip(normalized, normalized[1:]):
        instability = max(instability, float(cv2.absdiff(a, b).mean()))
    need(instability <= options["stability_threshold"], "ROI_NOT_STABLE")
    required = options["min_change_fraction"]
    changed = [
        s["changed_fraction"] >= required
        and s["largest_component_fraction"] >= required * 0.3
        for s in scores
    ]
    unchanged = [s["changed_fraction"] < required * 0.5 for s in scores]
    result = "PASS" if all(changed) else "FAIL" if all(unchanged) else "INCONCLUSIVE"
    return {
        "result": result,
        "code": {
            "PASS": "ROI_CHANGE_MET",
            "FAIL": "ROI_NO_CHANGE",
            "INCONCLUSIVE": "ROI_CHANGE_UNCERTAIN",
        }[result],
        "data_valid": True,
        "roi_statistics": scores,
        "stability_difference": instability,
        "processing_size": [before.shape[1], before.shape[0]],
    }
