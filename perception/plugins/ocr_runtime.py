from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from plugins.ocr_preprocess import preprocess_for_ocr
from utils.tensorrt_runtime import TensorRTEngine

_log = logging.getLogger(__name__)


TENSORRT_MODEL_FILES = ("det.engine", "rec.engine", "keys.txt")
TENSORRT_CLASSIFIER_MODEL_FILE = "cls.engine"
MAX_RECOGNITION_WIDTH = 2048
_OCR_MEAN = (127.5, 127.5, 127.5)
_OCR_NORMAL = (1 / 127.5, 1 / 127.5, 1 / 127.5)
DEFAULT_MAX_SIDE_LEN = 1600
DEFAULT_REC_MIN_SCORE = 0.9
DEFAULT_DET_THRESH = 0.3
DEFAULT_DET_BOX_THRESH = 0.5
DEFAULT_DET_UNCLIP_RATIO = 0.7
DEFAULT_CLS_THRESH = 0.9
DEFAULT_CROP_REFINEMENT_ENABLED = True
DEFAULT_CROP_REFINEMENT_MIN_SCORE = 0.9
DEFAULT_CROP_REFINEMENT_MIN_GAIN = 0.12
DEFAULT_CROP_REFINEMENT_PROFILES = (
    "prefix_65",
    "upper_center",
    "upper_tight",
)
DEFAULT_EMPTY_RESULT_RETRY_ENABLED = True
DEFAULT_EMPTY_RESULT_RETRY_DET_THRESH = 0.1
DEFAULT_EMPTY_RESULT_RETRY_DET_BOX_THRESH = 0.1

_CROP_REFINEMENT_PROFILE_BOXES = {
    "prefix_65": (0.05, 0.00, 0.70, 1.00),
    "upper_center": (0.20, 0.05, 0.80, 0.68),
    "upper_tight": (0.28, 0.10, 0.72, 0.64),
}


@dataclass(frozen=True)
class CropRefinementConfig:
    enabled: bool = DEFAULT_CROP_REFINEMENT_ENABLED
    min_score: float = DEFAULT_CROP_REFINEMENT_MIN_SCORE
    min_gain: float = DEFAULT_CROP_REFINEMENT_MIN_GAIN
    min_text_length: int = 2
    profiles: tuple[str, ...] = DEFAULT_CROP_REFINEMENT_PROFILES

    @classmethod
    def from_mapping(cls, value: dict | None) -> "CropRefinementConfig":
        mapping = dict(value or {})
        profiles = tuple(
            str(profile).strip()
            for profile in mapping.get(
                "profiles", DEFAULT_CROP_REFINEMENT_PROFILES
            )
            if str(profile).strip()
        )
        unknown = set(profiles) - set(_CROP_REFINEMENT_PROFILE_BOXES)
        if unknown:
            raise ValueError(
                "unknown OCR crop refinement profiles: "
                + ", ".join(sorted(unknown))
            )
        min_score = float(
            mapping.get("min_score", DEFAULT_CROP_REFINEMENT_MIN_SCORE)
        )
        min_gain = float(
            mapping.get("min_gain", DEFAULT_CROP_REFINEMENT_MIN_GAIN)
        )
        min_text_length = int(mapping.get("min_text_length", 2))
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("OCR crop refinement min_score must be in [0, 1]")
        if min_gain < 0.0:
            raise ValueError("OCR crop refinement min_gain must be non-negative")
        if min_text_length < 1:
            raise ValueError(
                "OCR crop refinement min_text_length must be positive"
            )
        return cls(
            enabled=bool(
                mapping.get("enabled", DEFAULT_CROP_REFINEMENT_ENABLED)
            ),
            min_score=min_score,
            min_gain=min_gain,
            min_text_length=min_text_length,
            profiles=profiles,
        )


@dataclass(frozen=True)
class EmptyResultRetryConfig:
    enabled: bool = DEFAULT_EMPTY_RESULT_RETRY_ENABLED
    det_thresh: float = DEFAULT_EMPTY_RESULT_RETRY_DET_THRESH
    det_box_thresh: float = DEFAULT_EMPTY_RESULT_RETRY_DET_BOX_THRESH

    @classmethod
    def from_mapping(cls, value: dict | None) -> "EmptyResultRetryConfig":
        mapping = dict(value or {})
        det_thresh = float(
            mapping.get("det_thresh", DEFAULT_EMPTY_RESULT_RETRY_DET_THRESH)
        )
        det_box_thresh = float(
            mapping.get(
                "det_box_thresh", DEFAULT_EMPTY_RESULT_RETRY_DET_BOX_THRESH
            )
        )
        if not 0.0 <= det_thresh <= 1.0:
            raise ValueError("OCR empty-result retry det_thresh must be in [0, 1]")
        if not 0.0 <= det_box_thresh <= 1.0:
            raise ValueError(
                "OCR empty-result retry det_box_thresh must be in [0, 1]"
            )
        return cls(
            enabled=bool(
                mapping.get("enabled", DEFAULT_EMPTY_RESULT_RETRY_ENABLED)
            ),
            det_thresh=det_thresh,
            det_box_thresh=det_box_thresh,
        )


def decode_vips_overview(image_bytes: bytes, max_side: int):
    import numpy as np
    import pyvips

    if max_side <= 0:
        raise ValueError("max_side must be positive")
    # no_rotate=True: keep the stored pixel orientation. Boxes are scaled back
    # with the JPEG header size (probe_image_header), so applying the EXIF
    # orientation here would swap width/height and misplace every bbox on
    # rotated frames. Robot cameras publish upright frames; consumers that
    # need EXIF orientation must transform coordinates themselves.
    image = pyvips.Image.thumbnail_buffer(
        image_bytes,
        max_side,
        size="down",
        no_rotate=True,
    )
    if image.hasalpha():
        image = image.flatten(background=[255, 255, 255])
    image = image.colourspace("srgb")
    try:
        rgb = image.numpy()
    except ValueError:
        memory = image.write_to_memory()
        rgb = np.frombuffer(memory, dtype=np.uint8).reshape(
            image.height, image.width, min(image.bands, 4)
        )
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("decoded image must have three color channels")
    return np.ascontiguousarray(rgb[:, :, ::-1], dtype=np.uint8)


def probe_image_header(image_bytes: bytes) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size
    except Exception as exc:
        raise ValueError("invalid or unsupported compressed image header") from exc

    if width <= 0 or height <= 0:
        raise ValueError("compressed image has invalid dimensions")
    return width, height


def scale_ocr_items(
    items: list[dict], scale_x: float, scale_y: float
) -> list[dict]:
    scaled_items = []
    for item in items:
        x1, y1, x2, y2 = item["bbox"]
        scaled_items.append(
            {
                **item,
                "bbox": [
                    math.floor(x1 * scale_x),
                    math.floor(y1 * scale_y),
                    math.ceil(x2 * scale_x),
                    math.ceil(y2 * scale_y),
                ],
            }
        )
    return scaled_items


def annotate_ocr_items(items: list[dict]) -> list[dict]:
    """Attach the size/hierarchy fields an LLM consumer needs to rank text.

    Detection returns items in reading order (top-to-bottom, then left) with
    only bbox+score, so a downstream LLM cannot tell a headline from fine
    print. Each item gains:

    - ``height``: bbox height in source pixels — a font-size proxy.
    - ``prominence``: height relative to the largest text on the frame,
      rounded to 2 decimals (1.0 = the most prominent text).
    - ``line``: visual line index; an item joins the current line when its
      vertical center falls inside that line's band.
    """
    annotated: list[dict] = []
    band: tuple[float, float] | None = None
    line = -1
    for item in items:
        x1, y1, x2, y2 = item["bbox"]
        height = max(1, round(y2 - y1))
        center = (y1 + y2) / 2
        if band is None or not band[0] <= center <= band[1]:
            line += 1
            band = (y1, y2)
        else:
            band = (min(band[0], y1), max(band[1], y2))
        annotated.append({**item, "height": height, "line": line})
    if annotated:
        max_height = max(item["height"] for item in annotated)
        for item in annotated:
            item["prominence"] = round(item["height"] / max_height, 2)
    return annotated


def build_ocr_payload(
    results, timestamp, language, error=None, image_size=None
) -> dict:
    """Assemble the published OCR result.

    ``text`` joins recognized fragments per visual line (left-to-right)
    with newlines between lines; ``items`` carry bbox (source-pixel
    [x1,y1,x2,y2]), score, height, prominence and line so consumers can
    reason about size and importance; ``image_size`` ([width, height]) is
    the coordinate reference frame for every bbox.
    """
    items = annotate_ocr_items(results)
    lines: dict[int, list[dict]] = {}
    for item in items:
        if item.get("text"):
            lines.setdefault(item["line"], []).append(item)
    text = "\n".join(
        " ".join(
            item["text"]
            for item in sorted(members, key=lambda entry: entry["bbox"][0])
        )
        for _line, members in sorted(lines.items())
    )
    payload = {
        "text": text,
        "items": items,
        "timestamp": timestamp,
        "language": language,
    }
    if image_size is not None:
        payload["image_size"] = [int(image_size[0]), int(image_size[1])]
    if error is not None:
        payload["error"] = str(error)
    return payload


def recognize_to_payload(
    adapter, image_bytes: bytes, language: str, timestamp: float
) -> dict:
    try:
        items = adapter.recognize(image_bytes, language)
        try:
            image_size = probe_image_header(image_bytes)
        except ValueError:
            image_size = None
        return build_ocr_payload(
            items, timestamp, language, image_size=image_size
        )
    except Exception as exc:
        return build_ocr_payload([], timestamp, language, error=exc)


class _TensorRTModelSession:
    """OCR view over one shared TensorRTEngine: uint8 HWC in, normalized NCHW.

    Everything CUDA/TensorRT related (deserialize, profiles, buffers, stream,
    execution, cleanup) lives in utils.tensorrt_runtime.TensorRTEngine; this
    class only adds PP-OCR's mean/normal preprocessing and the image-shape
    helpers the detector/recognizer/classifier need.
    """

    def __init__(
        self,
        engine_path: Path,
        *,
        device_id: int,
        mean: tuple[float, float, float],
        normal: tuple[float, float, float],
    ):
        import numpy as np

        self._np = np
        self._engine = TensorRTEngine(engine_path, device_id=device_id)
        if len(self._engine.output_names) != 1:
            self._engine.close()
            raise RuntimeError(
                "OCR TensorRT engine must have exactly one output; got "
                f"{self._engine.output_names}"
            )
        self._mean = np.asarray(mean, dtype=np.float32).reshape(1, 3, 1, 1)
        self._normal = np.asarray(normal, dtype=np.float32).reshape(1, 3, 1, 1)

    @property
    def _profiles(self):
        return self._engine.profiles

    @property
    def optimization_shape(self) -> tuple[int, ...]:
        return self._engine.optimization_shape

    def fit_input_image_shape(
        self, height: int, width: int
    ) -> tuple[int, int]:
        """Return the smallest profile-compatible HWC canvas for an image."""
        height = int(height)
        width = int(width)
        if height <= 0 or width <= 0:
            raise ValueError("OCR TensorRT image dimensions must be positive")

        candidates = []
        for minimum, _optimum, maximum in self._profiles:
            if (
                len(minimum) == 4
                and minimum[0] <= 1 <= maximum[0]
                and minimum[1] <= 3 <= maximum[1]
            ):
                candidate = (
                    max(height, minimum[2]), max(width, minimum[3])
                )
                if candidate[0] <= maximum[2] and candidate[1] <= maximum[3]:
                    candidates.append(candidate)

        if candidates:
            return min(candidates, key=lambda shape: shape[0] * shape[1])

        # Reuse the standard diagnostic, including all supported ranges.
        self._engine.select_profile((1, 3, height, width))
        raise AssertionError("unreachable")

    def max_batch_size(self, height: int, width: int) -> int:
        compatible = [
            maximum[0]
            for minimum, _optimum, maximum in self._profiles
            if len(minimum) == 4
            and minimum[1] <= 3 <= maximum[1]
            and minimum[2] <= height <= maximum[2]
            and minimum[3] <= width <= maximum[3]
            and minimum[0] <= 1 <= maximum[0]
        ]
        if not compatible:
            self._engine.select_profile((1, 3, int(height), int(width)))
        return max(compatible)

    def run_uint8(self, image, shape: tuple[int, ...]):
        image = self._np.ascontiguousarray(image, dtype=self._np.uint8)
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
            raise ValueError(f"invalid OCR TensorRT NCHW shape: {shape}")
        if image.shape[:2] != (shape[2], shape[3]):
            raise ValueError(
                f"OCR TensorRT image/shape mismatch: {image.shape} vs {shape}"
            )
        return self.run_uint8_batch(image[None], shape)

    def run_uint8_batch(self, images, shape: tuple[int, ...]):
        np = self._np
        images = np.ascontiguousarray(images, dtype=np.uint8)
        if len(shape) != 4 or shape[1] != 3:
            raise ValueError(f"invalid OCR TensorRT NCHW shape: {shape}")
        if images.ndim != 4 or images.shape != (
            shape[0], shape[2], shape[3], shape[1]
        ):
            raise ValueError(
                f"OCR TensorRT image batch/shape mismatch: {images.shape} vs "
                f"{shape}"
            )
        nchw = images.transpose(0, 3, 1, 2)
        if self._engine.input_dtype == np.dtype(np.float32):
            array = np.empty(nchw.shape, dtype=np.float32)
            np.subtract(nchw, self._mean, out=array)
            np.multiply(array, self._normal, out=array)
        else:
            value = (nchw.astype(np.float32) - self._mean) * self._normal
            array = np.ascontiguousarray(value, dtype=self._engine.input_dtype)
        return self._engine.infer(array)[0]

    def close(self) -> None:
        engine = getattr(self, "_engine", None)
        if engine is not None:
            engine.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Interpreter shutdown can unload CUDA before Python finalizers run.
            pass


class _TensorRTPipeline:
    @staticmethod
    def _multiple_of_32(value: float) -> int:
        return max(32, int(round(value / 32)) * 32)

    def _detector_input(self, image):
        import cv2
        import numpy as np

        height, width = image.shape[:2]
        scale = min(1.0, self._max_side_len / max(height, width))
        target_height = self._multiple_of_32(height * scale)
        target_width = self._multiple_of_32(width * scale)
        canvas_height, canvas_width = self._det.fit_input_image_shape(
            target_height, target_width
        )

        # Uniformly enlarge small images until one canvas edge is filled, then
        # pad the remaining edge. This preserves aspect ratio while satisfying
        # TensorRT profiles whose minimum dimensions exceed the camera frame.
        fit_scale = min(
            canvas_height / target_height,
            canvas_width / target_width,
        )
        content_height = min(
            canvas_height,
            max(target_height, int(round(target_height * fit_scale))),
        )
        content_width = min(
            canvas_width,
            max(target_width, int(round(target_width * fit_scale))),
        )

        if (content_width, content_height) == (width, height):
            resized = image
        else:
            shrinking = content_height <= height and content_width <= width
            resized = cv2.resize(
                image,
                (content_width, content_height),
                interpolation=(
                    cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
                ),
            )

        if (content_height, content_width) == (
            canvas_height,
            canvas_width,
        ):
            return resized, (content_height, content_width)

        padded = np.full(
            (canvas_height, canvas_width, 3), 128, dtype=np.uint8
        )
        padded[:content_height, :content_width] = resized
        return padded, (content_height, content_width)

    @staticmethod
    def _crop_detector_prediction(
        prediction, input_shape, content_shape
    ):
        if input_shape == content_shape:
            return prediction

        import numpy as np

        input_height, input_width = input_shape
        content_height, content_width = content_shape
        output_height, output_width = prediction.shape[-2:]
        crop_height = min(
            output_height,
            max(1, int(round(output_height * content_height / input_height))),
        )
        crop_width = min(
            output_width,
            max(1, int(round(output_width * content_width / input_width))),
        )
        return np.ascontiguousarray(
            prediction[..., :crop_height, :crop_width]
        )

    def _run_detector(self, image):
        detector_input, content_shape = self._detector_input(image)
        height, width = detector_input.shape[:2]
        prediction = self._det.run_uint8(
            detector_input, (1, 3, height, width)
        )
        prediction = self._crop_detector_prediction(
            prediction, (height, width), content_shape
        )
        return prediction, image.shape[:2]

    @staticmethod
    def _postprocess_detection(prediction, image_shape, postprocess):
        import numpy as np

        boxes, scores = postprocess(prediction, image_shape)
        if len(boxes) == 0:
            return np.empty((0, 4, 2), dtype=np.float32), []
        order = sorted(
            range(len(boxes)),
            key=lambda index: (boxes[index][0][1], boxes[index][0][0]),
        )
        return boxes[order], [scores[index] for index in order]

    @staticmethod
    def _prepare_recognition_crop(crop):
        import cv2
        import numpy as np

        height, width = crop.shape[:2]
        if height <= 0 or width <= 0:
            return None
        ratio = width / float(height)
        target_height = 48
        target_width = min(
            MAX_RECOGNITION_WIDTH,
            max(320, int(math.ceil(target_height * ratio))),
        )
        target_width = max(32, int(math.ceil(target_width / 8)) * 8)
        resized_width = min(
            target_width, max(1, int(math.ceil(target_height * ratio)))
        )
        resized = cv2.resize(
            crop,
            (resized_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )
        padded = np.full(
            (target_height, target_width, 3), 128, dtype=np.uint8
        )
        padded[:, :resized_width] = resized
        return padded, ratio

    def __init__(
        self,
        root: Path,
        *,
        device_id: int,
        max_side_len: int,
        rec_min_score: float = DEFAULT_REC_MIN_SCORE,
        enable_preprocess: bool = True,
        det_thresh: float = DEFAULT_DET_THRESH,
        det_box_thresh: float = DEFAULT_DET_BOX_THRESH,
        det_unclip_ratio: float = DEFAULT_DET_UNCLIP_RATIO,
        crop_refinement: CropRefinementConfig | None = None,
        empty_result_retry: EmptyResultRetryConfig | None = None,
        use_angle_cls: bool = False,
        cls_thresh: float = DEFAULT_CLS_THRESH,
    ):
        from rapidocr.ch_ppocr_det.utils import DBPostProcess
        from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode
        from rapidocr.utils.process_img import get_rotate_crop_image

        self._det = _TensorRTModelSession(
            root / "det.engine",
            device_id=device_id,
            mean=_OCR_MEAN,
            normal=_OCR_NORMAL,
        )
        try:
            self._rec = _TensorRTModelSession(
                root / "rec.engine",
                device_id=device_id,
                mean=_OCR_MEAN,
                normal=_OCR_NORMAL,
            )
        except Exception:
            self._det.close()
            raise
        self._cls = None
        if use_angle_cls:
            try:
                self._cls = _TensorRTModelSession(
                    root / TENSORRT_CLASSIFIER_MODEL_FILE,
                    device_id=device_id,
                    mean=_OCR_MEAN,
                    normal=_OCR_NORMAL,
                )
            except Exception:
                self._rec.close()
                self._det.close()
                raise
        self._det_postprocess = DBPostProcess(
            thresh=float(det_thresh),
            box_thresh=float(det_box_thresh),
            max_candidates=1000,
            unclip_ratio=float(det_unclip_ratio),
            use_dilation=True,
            score_mode="fast",
        )
        self._rec_decode = CTCLabelDecode(character_path=root / "keys.txt")
        self._crop = get_rotate_crop_image
        self._rec_min_score = float(rec_min_score)
        self._enable_preprocess = bool(enable_preprocess)
        self._max_side_len = max(32, int(max_side_len))
        self._cls_thresh = float(cls_thresh)
        self._crop_refinement = (
            crop_refinement
            if crop_refinement is not None
            else CropRefinementConfig()
        )
        retry = (
            empty_result_retry
            if empty_result_retry is not None
            else EmptyResultRetryConfig()
        )
        self._empty_result_retry_postprocess = None
        if retry.enabled:
            self._empty_result_retry_postprocess = DBPostProcess(
                thresh=retry.det_thresh,
                box_thresh=retry.det_box_thresh,
                max_candidates=1000,
                unclip_ratio=float(det_unclip_ratio),
                use_dilation=True,
                score_mode="fast",
            )

    def warm_up(self) -> None:
        import numpy as np

        det_shape = self._det.optimization_shape
        rec_shape = self._rec.optimization_shape
        self._det.run_uint8_batch(
            np.zeros(
                (det_shape[0], det_shape[2], det_shape[3], 3),
                dtype=np.uint8,
            ),
            det_shape,
        )
        self._rec.run_uint8_batch(
            np.zeros(
                (rec_shape[0], rec_shape[2], rec_shape[3], 3),
                dtype=np.uint8,
            ),
            rec_shape,
        )
        if self._cls is not None:
            cls_shape = self._cls.optimization_shape
            self._cls.run_uint8_batch(
                np.zeros(
                    (cls_shape[0], cls_shape[2], cls_shape[3], 3),
                    dtype=np.uint8,
                ),
                cls_shape,
            )

    @staticmethod
    def _prepare_classifier_crop(crop):
        import cv2
        import numpy as np

        height, width = crop.shape[:2]
        if height <= 0 or width <= 0:
            return None
        target_height = 48
        target_width = 192
        resized_width = min(
            target_width,
            max(1, int(math.ceil(target_height * width / float(height)))),
        )
        resized = cv2.resize(
            crop,
            (resized_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )
        padded = np.zeros(
            (target_height, target_width, 3), dtype=np.uint8
        )
        padded[:, :resized_width] = resized
        return padded

    def _orient_crops(self, crops):
        import cv2
        import numpy as np

        classifier = getattr(self, "_cls", None)
        if classifier is None:
            return crops
        oriented = list(crops)
        prepared = []
        for index, crop in enumerate(crops):
            value = self._prepare_classifier_crop(crop)
            if value is not None:
                prepared.append((index, value))
        max_batch = classifier.max_batch_size(48, 192)
        for offset in range(0, len(prepared), max_batch):
            chunk = prepared[offset:offset + max_batch]
            images = np.stack([entry[1] for entry in chunk])
            prediction = classifier.run_uint8_batch(
                images,
                (len(chunk), 3, 48, 192),
            )
            for entry, probabilities in zip(chunk, prediction):
                label = int(np.argmax(probabilities))
                score = float(probabilities[label])
                if label == 1 and score > self._cls_thresh:
                    oriented[entry[0]] = cv2.rotate(
                        crops[entry[0]], cv2.ROTATE_180
                    )
        return oriented

    def _recognize_boxes(self, image, boxes):
        import copy
        from collections import defaultdict

        import numpy as np

        prepared_by_width = defaultdict(list)
        recognized = [("", 0.0)] * len(boxes)
        crops = [
            self._crop(image, copy.deepcopy(box)) for box in boxes
        ]
        for index, crop in enumerate(self._orient_crops(crops)):
            prepared = self._prepare_recognition_crop(crop)
            if prepared is None:
                continue
            padded, ratio = prepared
            prepared_by_width[padded.shape[1]].append((index, padded, ratio))

        for target_width, group in prepared_by_width.items():
            max_batch = self._rec.max_batch_size(48, target_width)
            for offset in range(0, len(group), max_batch):
                chunk = group[offset:offset + max_batch]
                images = np.stack([entry[1] for entry in chunk])
                ratios = tuple(entry[2] for entry in chunk)
                prediction = self._rec.run_uint8_batch(
                    images,
                    (len(chunk), 3, 48, target_width),
                )
                line_results, _ = self._rec_decode(
                    prediction,
                    False,
                    wh_ratio_list=ratios,
                    max_wh_ratio=target_width / 48,
                )
                for entry, result in zip(chunk, line_results):
                    text, score = result
                    recognized[entry[0]] = str(text), float(score)
        return recognized

    def close(self) -> None:
        classifier = getattr(self, "_cls", None)
        if classifier is not None:
            classifier.close()
            self._cls = None
        self._rec.close()
        self._det.close()

    @staticmethod
    def _sub_quad(box, x0: float, y0: float, x1: float, y1: float):
        import numpy as np

        def point(u: float, v: float):
            top = box[0] * (1.0 - u) + box[1] * u
            bottom = box[3] * (1.0 - u) + box[2] * u
            return top * (1.0 - v) + bottom * v

        return np.asarray(
            [
                point(x0, y0),
                point(x1, y0),
                point(x1, y1),
                point(x0, y1),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _item(box, text: str, score: float) -> dict:
        import numpy as np

        xs = box[:, 0]
        ys = box[:, 1]
        return {
            "text": text,
            "bbox": [
                float(np.min(xs)),
                float(np.min(ys)),
                float(np.max(xs)),
                float(np.max(ys)),
            ],
            "score": score,
        }

    def _recognize_with_refinement(self, image, boxes) -> list[dict]:
        recognized = self._recognize_boxes(image, boxes)
        refinement = getattr(
            self, "_crop_refinement", CropRefinementConfig(enabled=False)
        )
        if not refinement.enabled:
            return [
                self._item(box, text, score)
                for box, (text, score) in zip(boxes, recognized)
                if text.strip() and score >= self._rec_min_score
            ]

        refined_boxes = []
        refined_sources = []
        for box_index, (box, (_, score)) in enumerate(
            zip(boxes, recognized)
        ):
            if score >= self._rec_min_score:
                continue
            for profile in refinement.profiles:
                refined_boxes.append(
                    self._sub_quad(
                        box, *_CROP_REFINEMENT_PROFILE_BOXES[profile]
                    )
                )
                refined_sources.append(box_index)

        refined_results = self._recognize_boxes(image, refined_boxes)
        candidates_by_box = {}
        for box, source, result in zip(
            refined_boxes, refined_sources, refined_results
        ):
            text, score = result
            if len(text.strip()) < refinement.min_text_length:
                continue
            candidate = candidates_by_box.get(source)
            if candidate is None or score > candidate[2]:
                candidates_by_box[source] = (box, text, score)

        items = []
        for box_index, (box, (text, score)) in enumerate(
            zip(boxes, recognized)
        ):
            selected_box = box
            selected_text = text
            selected_score = score
            candidate = candidates_by_box.get(box_index)
            if (
                candidate is not None
                and candidate[2] >= refinement.min_score
                and candidate[2] >= score + refinement.min_gain
            ):
                selected_box, selected_text, selected_score = candidate

            score_floor = (
                refinement.min_score
                if selected_box is not box
                else self._rec_min_score
            )
            if not selected_text.strip() or selected_score < score_floor:
                continue
            items.append(
                self._item(selected_box, selected_text, selected_score)
            )
        return items

    def infer(self, image) -> list[dict]:
        det_image = image
        if self._enable_preprocess:
            try:
                det_image = preprocess_for_ocr(image)
            except Exception:
                _log.debug("preprocess failed, using original image", exc_info=True)

        prediction, image_shape = self._run_detector(det_image)
        boxes, _ = self._postprocess_detection(
            prediction, image_shape, self._det_postprocess
        )
        items = self._recognize_with_refinement(image, boxes)
        retry_postprocess = getattr(
            self, "_empty_result_retry_postprocess", None
        )
        if items or retry_postprocess is None:
            return items

        retry_boxes, _ = self._postprocess_detection(
            prediction, image_shape, retry_postprocess
        )
        return self._recognize_with_refinement(image, retry_boxes)


class RapidOCRAdapter:
    def __init__(
        self,
        model_dir: str,
        device_id: int = 0,
        use_angle_cls: bool = True,
        max_side_len: int = DEFAULT_MAX_SIDE_LEN,
        rec_min_score: float = DEFAULT_REC_MIN_SCORE,
        enable_preprocess: bool = True,
        det_thresh: float = DEFAULT_DET_THRESH,
        det_box_thresh: float = DEFAULT_DET_BOX_THRESH,
        det_unclip_ratio: float = DEFAULT_DET_UNCLIP_RATIO,
        crop_refinement: dict | None = None,
        empty_result_retry: dict | None = None,
    ):
        root = Path(model_dir)
        self._max_side_len = max_side_len
        self._request_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        crop_refinement_config = CropRefinementConfig.from_mapping(
            crop_refinement
        )
        empty_result_retry_config = EmptyResultRetryConfig.from_mapping(
            empty_result_retry
        )
        required_files = TENSORRT_MODEL_FILES + (
            (TENSORRT_CLASSIFIER_MODEL_FILE,) if use_angle_cls else ()
        )
        missing = [name for name in required_files if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"OCR TensorRT model files missing: {', '.join(missing)}"
            )
        pipeline = _TensorRTPipeline(
            root,
            device_id=device_id,
            crop_refinement=crop_refinement_config,
            empty_result_retry=empty_result_retry_config,
            use_angle_cls=use_angle_cls,
            max_side_len=max_side_len,
            rec_min_score=rec_min_score,
            enable_preprocess=enable_preprocess,
            det_thresh=det_thresh,
            det_box_thresh=det_box_thresh,
            det_unclip_ratio=det_unclip_ratio,
        )
        try:
            pipeline.warm_up()
        except Exception:
            pipeline.close()
            raise
        self._pipeline = pipeline

    @staticmethod
    def _probe_image_header(image_bytes: bytes) -> tuple[int, int]:
        return probe_image_header(image_bytes)

    def _infer_image(self, image) -> list[dict]:
        with self._inference_lock:
            return self._pipeline.infer(image)

    def _recognize_single_pass(self, image_bytes: bytes) -> list[dict]:
        source_size = self._probe_image_header(image_bytes)
        image = decode_vips_overview(image_bytes, self._max_side_len)
        decoded_height, decoded_width = image.shape[:2]
        items = self._infer_image(image)
        source_width, source_height = source_size
        return scale_ocr_items(
            items,
            scale_x=source_width / decoded_width,
            scale_y=source_height / decoded_height,
        )

    def recognize(self, image_bytes: bytes, language: str = "zh") -> list:
        request_lock = getattr(self, "_request_lock", None)
        if request_lock is None:
            return self._recognize_single_pass(image_bytes)
        with request_lock:
            return self._recognize_single_pass(image_bytes)

    def close(self) -> None:
        request_lock = getattr(self, "_request_lock", None)
        if request_lock is None:
            self._close_resources()
            return
        with request_lock:
            self._close_resources()

    def _close_resources(self) -> None:
        pipeline = getattr(self, "_pipeline", None)
        self._pipeline = None
        if pipeline is not None:
            pipeline.close()
