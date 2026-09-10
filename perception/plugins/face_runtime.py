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

from plugins import scrfd_decode
from utils.cv2_compat import load_cv2
from utils.model_downloader import ensure_face_model
from utils.onnx_provider import ort_providers_for_device, warn_on_parked_cores

# Imported at module scope, not lazily inside FaceAnalyzer, and that placement is
# load-bearing.
#
# This process ends up with **two builds of ONNX Runtime**: sherpa-onnx bundles its
# own libonnxruntime.so in its wheel, and this plugin uses the standalone
# `onnxruntime` package. They export the same symbols, so whichever one is brought
# in first wins symbol resolution for both. If sherpa gets there first and the
# standalone library is loaded afterwards, sherpa's later inferences execute
# against the wrong implementation and fail an internal type check:
#
#   Non-zero status code returned while running SequenceInsert node ...
#   TensorSeq::Add ... IsSameDataType(tensor) was false
#
# Measured on Orin 6: Kokoro TTS synthesizes fine, the face card is then started,
# and every subsequent Kokoro utterance fails — with the real cause visible one
# line earlier as espeak losing its voice ("Unknown phoneme table: ''"). Importing
# here instead is enough on its own; no session has to be created. `main.py`
# imports plugins.face (and so this module) during startup, before any plugin
# builds a model, so this lands first.
#
# Only ASR/TTS graphs that use the sequence ops are affected, which is why the
# conflict lay dormant until Kokoro — the first model here whose graph has a
# Loop/SequenceInsert. sherpa's sensevoice ASR was measured unaffected.
#
# Guarded because the host test suite imports this module without onnxruntime
# installed; FaceAnalyzer raises a clear error if it is genuinely missing.
try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - exercised on dev hosts, not on device
    ort = None

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
# Single source of truth is plugins/scrfd_decode.py, which the worker child imports
# too — two copies of "this detector has three strides" is exactly the kind of drift
# that produces an opaque reshape error.
_FEAT_STRIDES = scrfd_decode.FEAT_STRIDES
_NUM_ANCHORS = scrfd_decode.NUM_ANCHORS
_NUM_KPS = scrfd_decode.NUM_KPS

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


# The decode moved to plugins/scrfd_decode.py so the worker child can import the same
# implementation instead of a copy. Re-exported under the old private names: everything
# below reads them, and a rename would be churn for its own sake.
_distance2bbox = scrfd_decode.distance2bbox
_distance2kps = scrfd_decode.distance2kps
_nms = scrfd_decode.nms


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
        if ort is None:
            raise RuntimeError(
                "the face plugin needs the standalone onnxruntime package; it is "
                "imported at the top of plugins/face_runtime.py, and that import "
                "must keep happening there — see the comment on it"
            )
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

        # Both sessions go into plugins/ort_worker.py's child process, because the
        # standalone ONNX Runtime and sherpa-onnx's bundled one corrupt each other's
        # sessions through a shared provider bridge whenever both are in one process —
        # an exception on jp6.1, a SIGSEGV that kills all of perception on jp5.11, and
        # with face's session built first, sherpa's Kokoro engine cannot be constructed
        # at all. That module's docstring has the mechanism and the measurements.
        #
        # `_open_session` keeps this to one branch rather than two code paths: the
        # proxy answers get_inputs/get_outputs/get_providers from the load reply, so
        # everything below is unchanged either way.
        self._session_keys = []
        self._det = self._open_session("face.det", det_path, providers, options,
                                       num_threads)
        self._rec = self._open_session("face.rec", rec_path, providers, options,
                                       num_threads)
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

    def _open_session(self, key: str, model_path: str, providers, options,
                      num_threads: int):
        """One session, in the ORT worker child unless it has been turned off.

        Falls back to an in-process session if the child cannot be reached, and says so
        loudly: that is the configuration where sherpa's runtime and this one collide,
        so it is degraded rather than equivalent. Face itself survives the collision —
        its graph has none of the fused squeeze outputs Kokoro's has — so the fallback
        keeps face working and puts the *Kokoro engine* at risk instead. That is the
        lesser harm, and worth a warning either way.
        """
        from plugins import ort_worker

        if ort_worker.worker_enabled():
            try:
                # Only the detector gets a post-processor: its head is 0.96 MiB of
                # candidates before thresholding and a few hundred bytes after. The
                # recogniser already returns a 2 kB embedding, so there is nothing to
                # reduce and nothing to couple.
                postprocess = ({"module": "plugins.scrfd_decode", "func": "decode"}
                               if key == "face.det" else None)
                session = ort_worker.get_worker().load(
                    key, model_path, providers,
                    {"intra_op_num_threads": max(1, int(num_threads)),
                     "graph_optimization_level": "ORT_ENABLE_ALL",
                     "log_severity_level": 3},
                    postprocess=postprocess,
                )
                self._session_keys.append(key)
                return session
            except Exception as exc:                              # noqa: BLE001
                log.warning(
                    "[face] the ORT worker could not load %s (%s); falling back to an "
                    "in-process session. sherpa-onnx and this runtime then share one "
                    "provider bridge, which breaks whichever builds a CUDA session "
                    "second — see plugins/ort_worker.py", key, exc)
        return ort.InferenceSession(model_path, options, providers=providers)

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

        if self._session_keys:
            # The session is in the worker; decode there, so the 16 800 candidates
            # never cross. `post_kwargs` carry what changes per frame.
            boxes, keypoints, scores = self._det.run(
                self._det_outputs, {self._det_input: blob},
                post_kwargs={"input_h": input_h, "input_w": input_w, "scale": scale,
                             "det_thresh": self._det_thresh,
                             "nms_thresh": self._nms_thresh})
        else:
            outputs = self._det.run(self._det_outputs, {self._det_input: blob})
            boxes, keypoints, scores = scrfd_decode.decode(
                outputs, input_h, input_w, scale,
                self._det_thresh, self._nms_thresh)

        if scores.size == 0:
            return []
        keep = range(len(scores))

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
        """Drop the sessions. Called via `_close_quietly` on config changes.

        Dropping the reference is enough for an in-process session, but not for one
        living in the ORT worker: a child does not notice its parent's garbage
        collector, so the session would stay resident — 13 MB of weights and its share
        of the GPU pool — for the life of the process. Unload it explicitly, and keep
        the child alive for whatever else it holds.
        """
        for key in getattr(self, "_session_keys", ()):
            try:
                from plugins import ort_worker
                ort_worker.get_worker().unload(key)
            except Exception as exc:                              # noqa: BLE001
                log.warning("[face] could not unload %s from the ORT worker: %s",
                            key, exc)
        self._session_keys = []
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
