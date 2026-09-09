"""Optional, uncalibrated mask-min depth and same-frame VOP preview."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

import numpy as np

from utils.cv2_compat import load_cv2

ULTRALYTICS_VERSION = "8.4.144"
WEIGHTS = {
    "yolo26n-depth.pt": "befed1b8561d8b2eaa66274b070cdfc9c44853bda6d6d62a04759f9383af74e9",
    "sam2.1_t.pt": "3c1e81ca9b037dd39d70a014ddb9a813d6c4c4e12555420db7eaff31689bd4e3",
}


def ensure_weight(directory: Path, name: str) -> Path:
    """Verify cached bytes before unpickling; install complete downloads atomically."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name

    def digest(path):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    if target.exists():
        if digest(target) != WEIGHTS[name]:
            raise ValueError(f"Weight checksum mismatch: {name}; replace the cached file")
        return target
    url = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{name}"
    fd, temporary = tempfile.mkstemp(dir=directory, suffix=".partial")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=30) as src:
            while chunk := src.read(1024 * 1024):
                out.write(chunk)
        if digest(Path(temporary)) != WEIGHTS[name]:
            raise ValueError(f"Downloaded weight checksum mismatch: {name}")
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def unavailable(status: str) -> dict:
    return {"value": None, "unit": "relative", "method": "mask_min",
            "status": status, "pixel": None}


def mask_min(depth: np.ndarray, mask: np.ndarray) -> dict:
    """Minimum positive finite prediction inside the *visible predicted mask*."""
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("Depth and mask must match the original image dimensions")
    mask = mask.astype(bool)
    if not mask.any():
        return unavailable("empty_mask")
    valid = mask & np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return unavailable("invalid_depth")
    # Exact minimum, not a quantile: single-pixel noise remains a known model limit.
    y, x = np.unravel_index(np.argmin(np.where(valid, depth, np.inf)), depth.shape)
    return {"value": float(depth[y, x]), "unit": "relative", "method": "mask_min",
            "status": "ok", "pixel": [int(x), int(y)]}


class DepthEstimator:
    """Caller serializes model access with the VOP inference lock."""
    def __init__(self, directory: str, device: str):
        self.directory = Path(directory)
        self.device = device
        self.depth_model = self.segment_model = None
        self.error = None

    def estimate(self, frame, boxes):
        if self.error:
            raise RuntimeError(self.error)
        if self.depth_model is None:
            try:
                import ultralytics
                from ultralytics import YOLO, SAM
                if ultralytics.__version__ != ULTRALYTICS_VERSION:
                    raise RuntimeError(f"Depth requires ultralytics=={ULTRALYTICS_VERSION}")
                self.depth_model = YOLO(str(ensure_weight(self.directory, "yolo26n-depth.pt")))
                self.segment_model = SAM(str(ensure_weight(self.directory, "sam2.1_t.pt")))
            except Exception as exc:
                self.depth_model = self.segment_model = None
                self.error = str(exc)
                raise
        depth_result = self.depth_model(frame, device=self.device, verbose=False)[0]
        depth = depth_result.depth.data.detach().cpu().numpy()
        masks = self.segment_model(frame, bboxes=boxes, device=self.device,
                                   verbose=False)[0].masks
        if masks is None:
            masks = np.zeros((len(boxes), *frame.shape[:2]), dtype=bool)
        else:
            masks = masks.data.detach().cpu().numpy().astype(bool)
        if depth.shape != frame.shape[:2] or masks.shape != (len(boxes), *frame.shape[:2]):
            raise ValueError("Model output is not aligned to the source image")
        return [mask_min(depth, mask) for mask in masks], masks


def extract_objects(result, shape):
    height, width = shape[:2]
    objects = []
    for index, box in enumerate(result.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        objects.append({
            "id": index + 1,
            "name": str(result.names[int(box.cls[0])]),
            "position": [round((x1 + x2) / width - 1, 3),
                         round((y1 + y2) / height - 1, 3)],
            "confidence": round(float(box.conf[0]), 2),
            "bbox_xyxy": [x1, y1, x2, y2],
        })
    return objects


def render_preview(frame, objects, masks=None, *, status="ok", depth_enabled=False,
                   source_stamp=None, sequence=0, frame_age_s=None):
    """Render metadata onto the actual inference frame, never the latest camera frame."""
    cv2 = load_cv2()
    height, width = frame.shape[:2]
    image = np.zeros((max(height, 70 + len(objects) * 60), width + 340, 3), dtype=np.uint8)
    image[:height, :width] = frame
    scene = image[:height, :width]
    colors = [(80, 230, 80), (255, 180, 50), (180, 80, 255), (40, 220, 255)]
    for i, obj in enumerate(objects):
        color = colors[i % len(colors)]
        if masks is not None:
            mask = masks[i]
            scene[mask] = (scene[mask] * 0.75 + np.array(color) * 0.25).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, color, 2)
        x1, y1, x2, y2 = np.rint(obj["bbox_xyxy"]).astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        depth = obj.get("obstacle_depth", unavailable("disabled"))
        if depth["pixel"] is not None:
            cv2.drawMarker(image, tuple(depth["pixel"]), color, cv2.MARKER_CROSS, 16, 2)
        # Short frame-local IDs stay by the boxes; a legend avoids overlapping labels.
        cv2.putText(image, f'#{obj["id"]}', (max(0, min(x1, width - 35)), max(50, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        label = f'#{obj["id"]} {obj["name"][:30]} ({obj["confidence"]:.2f})'
        detail = (f'nearest: {depth["value"]:.4f} rel' if depth["status"] == "ok"
                  else f'depth: N/A ({depth["status"]})')
        for row, text in enumerate((label, detail)):
            cv2.putText(image, text, (width + 12, 65 + i * 60 + row * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    banner = f'VOP #{sequence} | {status} | ' + (
        'RELATIVE DEPTH / UNCALIBRATED' if depth_enabled else 'depth disabled')
    cv2.rectangle(image, (0, 0), (image.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(image, banner, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255) if status == "ok" else (80, 80, 255), 1, cv2.LINE_AA)
    age = f'{frame_age_s:.2f}s' if frame_age_s is not None else 'N/A'
    cv2.putText(image, f'source stamp: {source_stamp} | frame age at publish: {age}', (6, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Preview JPEG encoding failed")
    return encoded.tobytes()
