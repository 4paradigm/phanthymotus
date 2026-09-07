#!/usr/bin/env python3
"""
plugins/face_runtime.py — 人脸检测 + 特征提取推理封装（无 ROS 依赖）。

Mirrors `plugins/ocr_runtime.py`'s role: everything that touches a model lives
here, so `plugins/face.py` only deals with ROS, MCP and lifecycle.

Two ONNX models, both from the InsightFace **buffalo_sc** pack — the smallest
pack that still ships keypoints, which the alignment step needs:

    det_500m.onnx    SCRFD-500M-BNKPS   ~2.5 MB   detection + 5 landmarks
    w600k_mbf.onnx   ArcFace MobileFaceNet ~13 MB  512-d embedding

The `insightface` package is deliberately **not** a dependency: it pulls in
onnx, scikit-image, scikit-learn, Cython and a build toolchain, none of which
are in the perception image, to wrap ~200 lines of pre/post-processing. Those
200 lines are below instead. The decode follows upstream's `scrfd.py` exactly
(strides 8/16/32, 2 anchors per location, distance-coded boxes and keypoints) —
this is a wire format, not a design choice, and getting it wrong produces
plausible-looking boxes in the wrong places.

Both models run on `onnxruntime`, so `device: cpu | gpu` is a provider choice
(`utils/onnx_provider.ort_providers_for_device`) rather than two code paths.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np

from utils.cv2_compat import load_cv2
from utils.model_downloader import ensure_face_model
from utils.onnx_provider import ort_providers_for_device, warn_on_parked_cores

log = logging.getLogger(__name__)

DEFAULT_FACE_MODEL_DIR = "/models/face/buffalo_sc"

DET_MODEL_FILE = "det_500m.onnx"
REC_MODEL_FILE = "w600k_mbf.onnx"

DEFAULT_DET_SIZE = (640, 640)
DEFAULT_DET_THRESH = 0.5
DEFAULT_NMS_THRESH = 0.4
DEFAULT_MIN_FACE_PX = 64
DEFAULT_BLUR_MIN = 60.0
# Longest side an incoming photo is downscaled to before detection. 2048 keeps
# a small distant face well above min_face_px while bounding the decoded array.
DEFAULT_MAX_IMAGE_SIDE = 2048
# Decompression-bomb guard, in pixels (~60 MP). A 24 MP phone photo passes.
DEFAULT_MAX_IMAGE_PIXELS = 60_000_000

EMBEDDING_DIM = 512
_REC_INPUT_SIZE = 112

# SCRFD wire format for a 3-level, 2-anchor, keypoint-carrying model. Upstream
# derives these from the output count; det_500m always has 9 outputs.
_FEAT_STRIDES = (8, 16, 32)
_NUM_ANCHORS = 2
_NUM_KPS = 5

# ArcFace's canonical 112x112 landmark template. Aligning every face onto these
# five points is what makes two embeddings comparable at all.
_ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass
class DetectedFace:
    """One detection, with everything the quality gate and payload need."""

    bbox: tuple[float, float, float, float]      # x1, y1, x2, y2 in source px
    det_score: float
    kps: np.ndarray                              # (5, 2) float32, source px
    blur: float = 0.0                            # variance of Laplacian
    aligned: np.ndarray | None = field(default=None, repr=False)
    embedding: np.ndarray | None = field(default=None, repr=False)

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def min_side(self) -> float:
        return min(self.width, self.height)

    @property
    def area(self) -> float:
        return self.width * self.height

    def bbox_xywh(self) -> list[int]:
        x1, y1, x2, y2 = self.bbox
        return [int(round(x1)), int(round(y1)),
                int(round(x2 - x1)), int(round(y2 - y1))]


def _umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (rotation + uniform scale + shift).

    This is what `skimage.transform.SimilarityTransform` computes, written out
    so scikit-image is not a dependency. Not `cv2.estimateAffinePartial2D`:
    that runs RANSAC/LMEDS, and on exactly five correspondences a robust
    estimator can discard a point and return a different transform run to run —
    which would make the same photo produce different embeddings.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    num, dim = src.shape

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    covariance = dst_demean.T @ src_demean / num
    reflect = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(covariance) < 0:
        reflect[dim - 1] = -1.0

    u_matrix, singular, vt_matrix = np.linalg.svd(covariance)
    rank = np.linalg.matrix_rank(covariance)
    transform = np.eye(dim + 1, dtype=np.float64)
    if rank == 0:
        raise ValueError("degenerate landmark set; cannot align")
    if rank == dim - 1:
        if np.linalg.det(u_matrix) * np.linalg.det(vt_matrix) > 0:
            transform[:dim, :dim] = u_matrix @ vt_matrix
        else:
            saved = reflect[dim - 1]
            reflect[dim - 1] = -1.0
            transform[:dim, :dim] = u_matrix @ np.diag(reflect) @ vt_matrix
            reflect[dim - 1] = saved
    else:
        transform[:dim, :dim] = u_matrix @ np.diag(reflect) @ vt_matrix

    variance = src_demean.var(axis=0).sum()
    if variance <= 1e-9:
        raise ValueError("degenerate landmark set; cannot align")
    scale = float(singular @ reflect) / variance
    transform[:dim, dim] = dst_mean - scale * (transform[:dim, :dim] @ src_mean)
    transform[:dim, :dim] *= scale
    return transform[:dim].astype(np.float32)


def _distance2bbox(centers: np.ndarray, distances: np.ndarray) -> np.ndarray:
    x1 = centers[:, 0] - distances[:, 0]
    y1 = centers[:, 1] - distances[:, 1]
    x2 = centers[:, 0] + distances[:, 2]
    y2 = centers[:, 1] + distances[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(centers: np.ndarray, distances: np.ndarray) -> np.ndarray:
    points = []
    for index in range(0, distances.shape[1], 2):
        points.append(centers[:, 0] + distances[:, index])
        points.append(centers[:, 1] + distances[:, index + 1])
    return np.stack(points, axis=-1)


def _nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> list[int]:
    """Plain greedy IoU suppression — no cv2.dnn, no torchvision."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1 + 1) * np.maximum(0.0, y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[current] + areas[rest] - inter)
        order = rest[iou <= thresh]
    return keep


class FaceAnalyzer:
    """SCRFD detection + ArcFace embedding over onnxruntime.

    Stateless per call and safe to share across threads: `InferenceSession.run`
    is thread-safe, and nothing here mutates instance state after __init__. The
    plugin holds exactly one analyzer for every instance, the same
    single-flight arrangement `plugins/ocr.py` uses for its adapter.
    """

    def __init__(
        self,
        model_dir: str = DEFAULT_FACE_MODEL_DIR,
        device: str = "auto",
        det_size: tuple[int, int] = DEFAULT_DET_SIZE,
        det_thresh: float = DEFAULT_DET_THRESH,
        nms_thresh: float = DEFAULT_NMS_THRESH,
        num_threads: int = 2,
        warmup: bool = True,
    ):
        import onnxruntime as ort

        self._cv2 = load_cv2()
        self._requested_device = (device or "auto").strip().lower()
        self._det_thresh = float(det_thresh)
        self._nms_thresh = float(nms_thresh)
        width, height = (int(det_size[0]), int(det_size[1]))
        # SCRFD's largest stride is 32; a size that is not a multiple of it
        # would make the anchor grid and the feature map disagree in shape and
        # the reshape below would fail with an opaque numpy error.
        self._det_size = (width - width % 32 or 32, height - height % 32 or 32)

        paths = ensure_face_model(model_dir)
        det_path = paths.get(DET_MODEL_FILE) or os.path.join(model_dir, DET_MODEL_FILE)
        rec_path = paths.get(REC_MODEL_FILE) or os.path.join(model_dir, REC_MODEL_FILE)

        providers = ort_providers_for_device(self._requested_device)
        # `device` resolves here, not in config: `auto` means "gpu when the
        # installed wheel has one". Recorded so info/logs report what is
        # actually running rather than what was asked for.
        self._device = "gpu" if providers[0] != "CPUExecutionProvider" else "cpu"
        # Session creation is where a parked-core Jetson abort()s under a bad
        # ORT version, and an abort prints no Python traceback. Say the
        # precondition first, so the last line before a silent death names it.
        warn_on_parked_cores("face")

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(num_threads))
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # The models ship pre-optimised and the container filesystem may be
        # read-only for /models; let ORT keep its optimised graph in memory.
        options.log_severity_level = 3

        self._det = ort.InferenceSession(det_path, options, providers=providers)
        self._rec = ort.InferenceSession(rec_path, options, providers=providers)
        self._det_input = self._det.get_inputs()[0].name
        self._rec_input = self._rec.get_inputs()[0].name
        self._det_outputs = [output.name for output in self._det.get_outputs()]

        if len(self._det_outputs) != len(_FEAT_STRIDES) * 3:
            raise RuntimeError(
                f"{DET_MODEL_FILE} has {len(self._det_outputs)} outputs; this "
                f"decoder expects {len(_FEAT_STRIDES) * 3} (3 strides x "
                "score/bbox/kps). A detector without keypoints cannot be "
                "aligned for recognition."
            )

        log.info(
            "[face] analyzer ready: onnxruntime=%s device=%s (requested %s) "
            "providers=%s det=%s rec=%s size=%s",
            ort.__version__, self._device, self._requested_device,
            self._det.get_providers(),
            os.path.basename(det_path), os.path.basename(rec_path),
            self._det_size,
        )

        if warmup:
            self._warmup()

    # ── properties ────────────────────────────────────────────────────────

    @property
    def device(self) -> str:
        """The device actually in use — `auto` is already resolved."""
        return self._device

    @property
    def providers(self) -> list[str]:
        return list(self._det.get_providers())

    @property
    def det_thresh(self) -> float:
        return self._det_thresh

    def _warmup(self) -> None:
        """Pay the first-inference cost here, not on an operator's first face.

        Same reasoning as `plugins/asr.py`: on CUDA the first run costs ~1.7 s
        (lazy kernels, cuDNN autotune, memory pool) against a steady state
        measured in tens of milliseconds.
        """
        try:
            blank = np.zeros(
                (self._det_size[1], self._det_size[0], 3), dtype=np.uint8
            )
            self.detect(blank)
            self.embed(np.zeros((_REC_INPUT_SIZE, _REC_INPUT_SIZE, 3), dtype=np.uint8))
            log.info("[face] warmup done")
        except Exception:  # noqa: BLE001 - warmup must never block startup
            log.warning("[face] warmup failed; continuing", exc_info=True)

    # ── image helpers ─────────────────────────────────────────────────────

    def decode_image(
        self,
        data: bytes,
        max_side: int = DEFAULT_MAX_IMAGE_SIDE,
        max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    ) -> np.ndarray | None:
        """Decode arbitrary image bytes to a BGR array, downscaled if huge.

        Converting and resizing **here** rather than rejecting at the API is
        deliberate: an operator registering a face has whatever their phone or
        camera produced, and "your photo is 24 MB" or "we only take JPEG" is a
        limitation of ours, not a property of their photo.

        Two decoders, because each covers formats the other does not: OpenCV
        reads JPEG/PNG/BMP/WEBP/TIFF/PPM, and Pillow (already in the image)
        adds GIF, ICO and some TIFF variants. Anything neither can read returns
        None and the caller reports `bad_input`. HEIC/HEIF — what an iPhone
        shoots by default — needs `pillow-heif`, which is not installed, so it
        is the one common format still unsupported.

        `max_side` exists for accuracy as much as memory: detection letterboxes
        to `det_size` (640) anyway, so a 6000 px photo is downscaled by the
        detector regardless. Doing it once here with INTER_AREA is both cheaper
        and better than letting the letterbox do it — and it bounds the
        intermediate array, which for a 60 MP image is 180 MB of uint8.

        `max_pixels` is a decompression-bomb guard: a few hundred KB of PNG can
        declare a 30000x30000 canvas, which would be 2.7 GB decoded and take the
        whole perception process down with the OOM killer.
        """
        if not data:
            return None
        image = self._decode_with_cv2(data)
        if image is None:
            image = self._decode_with_pillow(data)
        if image is None or image.size == 0:
            return None
        if max_pixels and image.shape[0] * image.shape[1] > max_pixels:
            log.warning("[face] refusing a %dx%d image (%.1f MP > %.1f MP cap)",
                        image.shape[1], image.shape[0],
                        image.shape[0] * image.shape[1] / 1e6, max_pixels / 1e6)
            return None
        return self._downscale(image, max_side)

    def _decode_with_cv2(self, data: bytes) -> np.ndarray | None:
        try:
            buffer = np.frombuffer(data, dtype=np.uint8)
            if buffer.size == 0:
                return None
            image = self._cv2.imdecode(buffer, self._cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001 - fall through to Pillow
            return None
        return image if image is not None and image.size else None

    def _decode_with_pillow(self, data: bytes) -> np.ndarray | None:
        """Fallback decoder, converting whatever it reads to 3-channel BGR."""
        try:
            import io

            from PIL import Image

            with Image.open(io.BytesIO(data)) as handle:
                # convert() also flattens palettes, drops alpha and collapses
                # 16-bit channels, so the result is always plain 8-bit RGB.
                rgb = handle.convert("RGB")
                array = np.asarray(rgb)
        except Exception:  # noqa: BLE001 - genuinely undecodable
            return None
        if array.size == 0:
            return None
        return np.ascontiguousarray(array[:, :, ::-1])      # RGB -> BGR

    def _downscale(self, image: np.ndarray, max_side: int) -> np.ndarray:
        if not max_side:
            return image
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= max_side:
            return image
        scale = max_side / float(longest)
        resized = self._cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=self._cv2.INTER_AREA,   # the right filter for shrinking
        )
        log.debug("[face] downscaled %dx%d -> %dx%d", width, height,
                  resized.shape[1], resized.shape[0])
        return resized

    # Kept as an alias: the name predates supporting anything but JPEG.
    decode_jpeg = decode_image

    # ── detection ─────────────────────────────────────────────────────────

    def detect(self, bgr: np.ndarray, max_faces: int = 0) -> list[DetectedFace]:
        """Detect faces in a BGR image, largest first."""
        if bgr is None or bgr.size == 0:
            return []
        height, width = bgr.shape[:2]
        input_w, input_h = self._det_size

        # Letterbox: preserve aspect ratio, paste top-left, remember the scale.
        # Distorting the aspect ratio moves the landmarks off the face and the
        # alignment silently degrades every embedding.
        image_ratio = height / max(1, width)
        model_ratio = input_h / input_w
        if image_ratio > model_ratio:
            new_h = input_h
            new_w = int(new_h / image_ratio)
        else:
            new_w = input_w
            new_h = int(new_w * image_ratio)
        scale = new_h / max(1, height)
        resized = self._cv2.resize(bgr, (max(1, new_w), max(1, new_h)))
        padded = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        padded[: resized.shape[0], : resized.shape[1]] = resized

        blob = padded[:, :, ::-1].astype(np.float32)          # BGR → RGB
        blob = (blob - 127.5) / 128.0
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])

        outputs = self._det.run(self._det_outputs, {self._det_input: blob})
        levels = len(_FEAT_STRIDES)

        boxes_all: list[np.ndarray] = []
        kps_all: list[np.ndarray] = []
        scores_all: list[np.ndarray] = []
        for index, stride in enumerate(_FEAT_STRIDES):
            scores = outputs[index].reshape(-1)
            bbox_preds = outputs[index + levels].reshape(-1, 4) * stride
            kps_preds = outputs[index + levels * 2].reshape(-1, _NUM_KPS * 2) * stride

            grid_h, grid_w = input_h // stride, input_w // stride
            centers = np.stack(
                np.mgrid[:grid_h, :grid_w][::-1], axis=-1
            ).astype(np.float32).reshape(-1, 2) * stride
            if _NUM_ANCHORS > 1:
                centers = np.stack([centers] * _NUM_ANCHORS, axis=1).reshape(-1, 2)

            positive = np.where(scores >= self._det_thresh)[0]
            if positive.size == 0:
                continue
            boxes_all.append(_distance2bbox(centers, bbox_preds)[positive])
            kps_all.append(
                _distance2kps(centers, kps_preds)[positive].reshape(-1, _NUM_KPS, 2)
            )
            scores_all.append(scores[positive])

        if not scores_all:
            return []

        boxes = np.concatenate(boxes_all) / scale
        keypoints = np.concatenate(kps_all) / scale
        scores = np.concatenate(scores_all)
        keep = _nms(boxes, scores, self._nms_thresh)

        faces = [
            DetectedFace(
                bbox=(
                    float(max(0.0, boxes[i][0])),
                    float(max(0.0, boxes[i][1])),
                    float(min(width, boxes[i][2])),
                    float(min(height, boxes[i][3])),
                ),
                det_score=float(scores[i]),
                kps=keypoints[i].astype(np.float32),
            )
            for i in keep
        ]
        faces.sort(key=lambda face: face.area, reverse=True)
        if max_faces and len(faces) > max_faces:
            faces = faces[:max_faces]
        return faces

    # ── alignment, quality, embedding ─────────────────────────────────────

    def align(self, bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        """Warp a face onto the 112x112 ArcFace template."""
        matrix = _umeyama_similarity(np.asarray(kps, dtype=np.float32), _ARCFACE_DST)
        return self._cv2.warpAffine(
            bgr, matrix, (_REC_INPUT_SIZE, _REC_INPUT_SIZE), borderValue=0.0
        )

    def sharpness(self, aligned: np.ndarray) -> float:
        """Variance of the Laplacian — the blur metric behind `low_quality`.

        Measured on the *aligned* crop, not the source frame, so the number
        means the same thing whether the person was 1 m or 4 m away: a distant
        face is upscaled to 112x112 and its softness shows up here.
        """
        gray = self._cv2.cvtColor(aligned, self._cv2.COLOR_BGR2GRAY)
        return float(self._cv2.Laplacian(gray, self._cv2.CV_64F).var())

    def prepare(self, bgr: np.ndarray, face: DetectedFace) -> DetectedFace:
        """Fill in `aligned` and `blur` for a detection."""
        face.aligned = self.align(bgr, face.kps)
        face.blur = self.sharpness(face.aligned)
        return face

    def embed(self, aligned: np.ndarray) -> np.ndarray:
        """512-d L2-normalised embedding of an aligned 112x112 crop."""
        if aligned.shape[0] != _REC_INPUT_SIZE or aligned.shape[1] != _REC_INPUT_SIZE:
            aligned = self._cv2.resize(aligned, (_REC_INPUT_SIZE, _REC_INPUT_SIZE))
        blob = aligned[:, :, ::-1].astype(np.float32)          # BGR → RGB
        blob = (blob - 127.5) / 127.5
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])
        vector = self._rec.run(None, {self._rec_input: blob})[0].reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            raise RuntimeError("recognition model returned a zero embedding")
        return (vector / norm).astype(np.float32)

    def analyze(
        self, bgr: np.ndarray, max_faces: int = 0, with_embeddings: bool = True
    ) -> list[DetectedFace]:
        """Detect, align and (optionally) embed every face in one frame."""
        faces = self.detect(bgr, max_faces=max_faces)
        for face in faces:
            self.prepare(bgr, face)
            if with_embeddings:
                face.embedding = self.embed(face.aligned)
        return faces

    def close(self) -> None:
        """Drop the sessions. Called via `_close_quietly` on config changes."""
        self._det = None
        self._rec = None


__all__ = [
    "DEFAULT_BLUR_MIN",
    "DEFAULT_MAX_IMAGE_PIXELS",
    "DEFAULT_MAX_IMAGE_SIDE",
    "DEFAULT_DET_SIZE",
    "DEFAULT_DET_THRESH",
    "DEFAULT_FACE_MODEL_DIR",
    "DEFAULT_MIN_FACE_PX",
    "DEFAULT_NMS_THRESH",
    "DET_MODEL_FILE",
    "EMBEDDING_DIM",
    "REC_MODEL_FILE",
    "DetectedFace",
    "FaceAnalyzer",
]
