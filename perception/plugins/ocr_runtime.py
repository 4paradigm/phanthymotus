from __future__ import annotations

import logging
import math
import tempfile
import threading
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from plugins.ocr_tiled_strategy import (
    AdaptiveTiledOCRStrategy,
    LargeImageStrategyConfig,
    decode_vips_overview,
    jpeg_dimensions,
)

from plugins.ocr_preprocess import preprocess_for_ocr

_log = logging.getLogger(__name__)


ORT_MODEL_FILES = ("det.onnx", "rec.onnx", "cls.onnx", "keys.txt")
MNN_MODEL_FILES = ("det.mnn", "rec.mnn", "keys.txt")
_OCR_MEAN = (127.5, 127.5, 127.5)
_OCR_NORMAL = (1 / 127.5, 1 / 127.5, 1 / 127.5)


@dataclass(frozen=True)
class ImageHeader:
    format: str
    width: int
    height: int

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


def probe_image_header(image_bytes: bytes) -> ImageHeader:
    jpeg_size = jpeg_dimensions(image_bytes)
    if jpeg_size is not None:
        return ImageHeader("JPEG", *jpeg_size)

    try:
        from PIL import Image

        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size
            image_format = str(image.format or "unknown").upper()
    except Exception as exc:
        raise ValueError("invalid or unsupported compressed image header") from exc

    if width <= 0 or height <= 0:
        raise ValueError("compressed image has invalid dimensions")
    return ImageHeader(image_format, width, height)


def normalize_rapidocr_output(
    output,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    round_bbox: bool = True,
) -> list[dict]:
    if output is None or output.boxes is None:
        return []

    items = []
    for polygon, text, score in zip(output.boxes, output.txts, output.scores):
        xs = [float(point[0]) * scale_x for point in polygon]
        ys = [float(point[1]) * scale_y for point in polygon]
        bbox = [min(xs), min(ys), max(xs), max(ys)]
        if round_bbox:
            bbox = [
                math.floor(min(xs)),
                math.floor(min(ys)),
                math.ceil(max(xs)),
                math.ceil(max(ys)),
            ]
        items.append(
            {
                "text": str(text),
                "bbox": bbox,
                "score": float(score),
            }
        )
    return items


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


def build_ocr_payload(results, timestamp, language, error=None) -> dict:
    payload = {
        "text": " ".join(item["text"] for item in results if item.get("text")),
        "items": results,
        "timestamp": timestamp,
        "language": language,
    }
    if error is not None:
        payload["error"] = str(error)
    return payload


def recognize_to_payload(
    adapter, image_bytes: bytes, language: str, timestamp: float
) -> dict:
    try:
        return build_ocr_payload(
            adapter.recognize(image_bytes, language), timestamp, language
        )
    except Exception as exc:
        return build_ocr_payload([], timestamp, language, error=exc)


class _MNNModelSession:
    def __init__(
        self,
        model_path: Path,
        *,
        num_threads: int,
        mean: tuple[float, float, float],
        normal: tuple[float, float, float],
    ):
        import MNN

        self._net = MNN.Interpreter(str(model_path))
        self._session = self._net.createSession(
            {
                "backend": "CPU",
                "numThread": int(num_threads),
                "precision": "low",
                "memory": 2,
                "power": 0,
            }
        )
        self._input = self._net.getSessionInput(self._session)
        self._process = MNN.CVImageProcess(
            {
                "sourceFormat": MNN.CV_ImageFormat_BGR,
                "destFormat": MNN.CV_ImageFormat_BGR,
                "filterType": MNN.CV_Filter_BILINEAL,
                "mean": (*mean, 0.0),
                "normal": (*normal, 1.0),
            }
        )

    def run_uint8(self, image, shape: tuple[int, ...]):
        import numpy as np

        image = np.ascontiguousarray(image, dtype=np.uint8)
        self._net.resizeTensor(self._input, shape)
        self._net.resizeSession(self._session)
        pointer = image.__array_interface__["data"][0]
        self._process.convert(
            pointer,
            image.shape[1],
            image.shape[0],
            image.strides[0],
            self._input,
        )
        self._net.runSession(self._session)
        output = self._net.getSessionOutput(self._session)
        import numpy as _np, MNN as _MNN
        out_shape = output.getShape()
        host = _MNN.Tensor(
            out_shape, _MNN.Halide_Type_Float, _MNN.Tensor_DimensionType_Caffe
        )
        output.copyToHostTensor(host)
        return _np.array(host.getData(), dtype=_np.float32).reshape(out_shape).copy()

    def close(self) -> None:
        release = getattr(self._net, "releaseSession", None)
        if callable(release):
            release(self._session)


class _MNNPipeline:
    def __init__(self, root: Path, *, num_threads: int, max_side_len: int,
                 rec_min_score: float = 0.3,
                 enable_preprocess: bool = True,
                 det_thresh: float = 0.3,
                 det_box_thresh: float = 0.5,
                 det_unclip_ratio: float = 1.6):
        from rapidocr.ch_ppocr_det.utils import DBPostProcess
        from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode
        from rapidocr.utils.process_img import get_rotate_crop_image

        self._det = _MNNModelSession(
            root / "det.mnn",
            num_threads=num_threads,
            mean=_OCR_MEAN,
            normal=_OCR_NORMAL,
        )
        try:
            self._rec = _MNNModelSession(
                root / "rec.mnn",
                num_threads=num_threads,
                mean=_OCR_MEAN,
                normal=_OCR_NORMAL,
            )
        except Exception:
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

    @staticmethod
    def _multiple_of_32(value: float) -> int:
        return max(32, int(round(value / 32)) * 32)

    def _detector_input(self, image):
        import cv2

        height, width = image.shape[:2]
        scale = min(1.0, self._max_side_len / max(height, width))
        target_height = self._multiple_of_32(height * scale)
        target_width = self._multiple_of_32(width * scale)
        if (target_width, target_height) == (width, height):
            return image
        return cv2.resize(
            image,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )

    def _detect(self, image):
        import numpy as np

        detector_input = self._detector_input(image)
        height, width = detector_input.shape[:2]
        prediction = self._det.run_uint8(
            detector_input, (1, 3, height, width)
        )
        boxes, scores = self._det_postprocess(prediction, image.shape[:2])
        if len(boxes) == 0:
            return np.empty((0, 4, 2), dtype=np.float32), []
        order = sorted(
            range(len(boxes)),
            key=lambda index: (boxes[index][0][1], boxes[index][0][0]),
        )
        return boxes[order], [scores[index] for index in order]

    def _recognize_crop(self, crop):
        import cv2
        import numpy as np

        height, width = crop.shape[:2]
        if height <= 0 or width <= 0:
            return "", 0.0
        ratio = width / float(height)
        target_height = 48
        target_width = max(320, int(math.ceil(target_height * ratio)))
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
        prediction = self._rec.run_uint8(
            padded, (1, 3, target_height, target_width)
        )
        line_results, _ = self._rec_decode(
            prediction,
            False,
            wh_ratio_list=(ratio,),
            max_wh_ratio=target_width / target_height,
        )
        if not line_results:
            return "", 0.0
        text, score = line_results[0]
        return str(text), float(score)

    def infer(self, image) -> list[dict]:
        import copy
        import numpy as np

        det_image = image
        if self._enable_preprocess:
            try:
                det_image = preprocess_for_ocr(image)
            except Exception:
                _log.debug("preprocess failed, using original image", exc_info=True)

        boxes, _ = self._detect(det_image)
        items = []
        for box in boxes:
            # rec uses the ORIGINAL image crop (not the preprocessed one),
            # so bbox coordinates map back to the original without adjustment
            crop = self._crop(image, copy.deepcopy(box))
            text, score = self._recognize_crop(crop)
            if not text.strip() or score < self._rec_min_score:
                continue
            xs = box[:, 0]
            ys = box[:, 1]
            items.append(
                {
                    "text": text,
                    "bbox": [
                        float(np.min(xs)),
                        float(np.min(ys)),
                        float(np.max(xs)),
                        float(np.max(ys)),
                    ],
                    "score": score,
                }
            )
        return items

    def warm_up(self) -> None:
        import numpy as np

        self._det.run_uint8(
            np.zeros((64, 64, 3), dtype=np.uint8), (1, 3, 64, 64)
        )
        self._rec.run_uint8(
            np.zeros((48, 320, 3), dtype=np.uint8), (1, 3, 48, 320)
        )

    def close(self) -> None:
        self._det.close()
        self._rec.close()


class RapidOCRAdapter:
    def __init__(
        self,
        model_dir: str,
        backend: str = "onnxruntime",
        fallback_backend: str = "",
        fallback_model_dir: str = "",
        device: str = "cpu",
        device_id: int = 0,
        gpu_mem_mb: int = 512,
        use_angle_cls: bool = True,
        num_threads: int = 2,
        max_side_len: int = 1600,
        rec_min_score: float = 0.3,
        enable_preprocess: bool = True,
        det_thresh: float = 0.3,
        det_box_thresh: float = 0.5,
        det_unclip_ratio: float = 1.6,
        large_image_strategy: dict | None = None,
    ):
        root = Path(model_dir)
        backend = str(backend).strip().lower()
        fallback_backend = str(fallback_backend).strip().lower()
        if backend not in ("mnn", "onnxruntime"):
            raise ValueError("OCR backend must be 'mnn' or 'onnxruntime'")
        if fallback_backend not in ("", "onnxruntime"):
            raise ValueError("OCR fallback_backend must be empty or 'onnxruntime'")

        device = str(device).strip().lower()
        if device not in ("cpu", "cuda"):
            raise ValueError("OCR device must be 'cpu' or 'cuda'")
        use_cuda = device == "cuda"
        if backend == "onnxruntime" and use_cuda:
            import onnxruntime as ort

            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" not in providers:
                raise RuntimeError(
                    "OCR device=cuda requires CUDAExecutionProvider; "
                    f"available providers: {providers}"
                )

        self._use_angle_cls = use_angle_cls
        self._max_side_len = max_side_len
        self._request_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        strategy_config = LargeImageStrategyConfig.from_mapping(
            large_image_strategy
        )
        self._large_image_strategy = (
            AdaptiveTiledOCRStrategy(strategy_config, max_side_len)
            if strategy_config.enabled
            else None
        )

        self._pipeline = None
        self._backend_name = backend
        if backend == "mnn":
            try:
                missing = [
                    name for name in MNN_MODEL_FILES
                    if not (root / name).is_file()
                ]
                if missing:
                    raise FileNotFoundError(
                        f"OCR MNN model files missing: {', '.join(missing)}"
                    )
                if use_angle_cls:
                    raise ValueError("OCR MNN backend does not load angle classifier")
                self._pipeline = _MNNPipeline(
                    root,
                    num_threads=num_threads,
                    max_side_len=max_side_len,
                    rec_min_score=rec_min_score,
                    enable_preprocess=enable_preprocess,
                    det_thresh=det_thresh,
                    det_box_thresh=det_box_thresh,
                    det_unclip_ratio=det_unclip_ratio,
                )
                self._pipeline.warm_up()
                self._engine = None
                return
            except Exception:
                if self._pipeline is not None:
                    self._pipeline.close()
                    self._pipeline = None
                if fallback_backend != "onnxruntime" or not fallback_model_dir:
                    raise
                root = Path(fallback_model_dir)
                self._backend_name = "onnxruntime"

        missing = [
            name for name in ORT_MODEL_FILES if not (root / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"OCR ONNX model files missing: {', '.join(missing)}"
            )

        # Load rapidocr's own default config, override model paths
        from rapidocr import RapidOCR
        from rapidocr.main import DEFAULT_CFG_PATH

        import yaml
        with open(DEFAULT_CFG_PATH) as f:
            cfg = yaml.safe_load(f)

        cfg["Det"].update(
            {
                "engine_type": "onnxruntime",
                "lang_type": "ch",
                "model_type": "tiny",
                "ocr_version": "PP-OCRv6",
                "model_path": str(root / "det.onnx"),
            }
        )
        cfg["Cls"].update(
            {
                "engine_type": "onnxruntime",
                "lang_type": "ch",
                "model_type": "mobile",
                "ocr_version": "PP-OCRv4",
                "model_path": str(root / "cls.onnx"),
            }
        )
        cfg["Rec"].update(
            {
                "engine_type": "onnxruntime",
                "lang_type": "ch",
                "model_type": "tiny",
                "ocr_version": "PP-OCRv6",
                "model_path": str(root / "rec.onnx"),
                "rec_keys_path": str(root / "keys.txt"),
            }
        )
        cfg["Global"]["use_cls"] = use_angle_cls
        # Engine-level cap must NOT shrink tiles: the single-pass path already
        # resizes to max_side_len before reaching the engine, while the tiled
        # strategy feeds full tile_size crops through the same engine.
        # (rapidocr crashes on max_side_len <= 0, so fall back to its default.)
        engine_global_cap = max_side_len if max_side_len > 0 else 2000
        if strategy_config.enabled:
            engine_global_cap = max(engine_global_cap, strategy_config.tile_size)
        cfg["Global"]["max_side_len"] = engine_global_cap

        engine_cfg = cfg.setdefault("EngineConfig", {}).setdefault(
            "onnxruntime", {}
        )
        engine_cfg["intra_op_num_threads"] = num_threads
        engine_cfg["inter_op_num_threads"] = 1
        engine_cfg["use_cuda"] = use_cuda
        if use_cuda:
            cuda_ep_cfg = engine_cfg.setdefault("cuda_ep_cfg", {})
            cuda_ep_cfg.update(
                {
                    "device_id": int(device_id),
                    "gpu_mem_limit": int(gpu_mem_mb) * 1024 * 1024,  # ORT CUDA EP: bytes
                }
            )

        with tempfile.TemporaryDirectory(prefix="rapidocr-config-") as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            with config_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, default_flow_style=False)
            self._engine = RapidOCR(config_path=str(config_path))

    @staticmethod
    def _jpeg_dimensions(image_bytes: bytes) -> tuple[int, int] | None:
        return jpeg_dimensions(image_bytes)

    @staticmethod
    def _probe_image_header(image_bytes: bytes) -> ImageHeader:
        return probe_image_header(image_bytes)

    def _infer_image(self, image) -> list[dict]:
        with self._inference_lock:
            pipeline = getattr(self, "_pipeline", None)
            if pipeline is not None:
                return pipeline.infer(image)
            output = self._engine(
                image,
                use_det=True,
                use_cls=self._use_angle_cls,
                use_rec=True,
            )
        return normalize_rapidocr_output(output, round_bbox=False)

    def _recognize_single_pass(self, image_bytes: bytes) -> list[dict]:
        header = self._probe_image_header(image_bytes)
        source_size = header.size
        image = decode_vips_overview(image_bytes, self._max_side_len)
        decoded_height, decoded_width = image.shape[:2]
        items = self._infer_image(image)
        source_width, source_height = source_size
        return scale_ocr_items(
            items,
            scale_x=source_width / decoded_width,
            scale_y=source_height / decoded_height,
        )

    def _recognize_request(self, image_bytes: bytes) -> list:
        strategy = getattr(self, "_large_image_strategy", None)
        if strategy is not None:
            header = self._probe_image_header(image_bytes)
            source_size = header.size
            if strategy.should_handle(source_size):
                return strategy.recognize(image_bytes, self._infer_image)
        return self._recognize_single_pass(image_bytes)

    def recognize(self, image_bytes: bytes, language: str = "zh") -> list:
        request_lock = getattr(self, "_request_lock", None)
        if request_lock is None:
            return self._recognize_request(image_bytes)
        with request_lock:
            return self._recognize_request(image_bytes)
