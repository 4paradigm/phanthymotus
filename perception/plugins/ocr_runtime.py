from __future__ import annotations

import math
import tempfile
import threading
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from plugins.ocr_tiled_strategy import (
    AdaptiveTiledOCRStrategy,
    LargeImageStrategyConfig,
    jpeg_dimensions,
)


REQUIRED_MODEL_FILES = ("det.onnx", "rec.onnx", "cls.onnx", "keys.txt")
_MIB = 1024 * 1024
_DECODE_CHANNELS = 3


class ImageTooLargeError(ValueError):
    """Raised before inference when an image would exceed a configured limit."""


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


class RapidOCRAdapter:
    def __init__(
        self,
        model_dir: str,
        device: str = "cpu",
        device_id: int = 0,
        gpu_mem_mb: int = 512,
        use_angle_cls: bool = True,
        num_threads: int = 2,
        max_side_len: int = 1600,
        max_input_mb: int = 16,
        max_decode_mb: int = 64,
        large_image_strategy: dict | None = None,
    ):
        root = Path(model_dir)
        missing = [
            name for name in REQUIRED_MODEL_FILES if not (root / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"OCR model files missing: {', '.join(missing)}"
            )

        device = str(device).strip().lower()
        if device not in ("cpu", "cuda"):
            raise ValueError("OCR device must be 'cpu' or 'cuda'")
        use_cuda = device == "cuda"
        if use_cuda:
            import onnxruntime as ort

            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" not in providers:
                raise RuntimeError(
                    "OCR device=cuda requires CUDAExecutionProvider; "
                    f"available providers: {providers}"
                )

        from rapidocr import RapidOCR
        from rapidocr.main import DEFAULT_CFG_PATH

        self._use_angle_cls = use_angle_cls
        self._max_side_len = max_side_len
        if max_input_mb <= 0:
            raise ValueError("max_input_mb must be positive")
        if max_decode_mb <= 0:
            raise ValueError("max_decode_mb must be positive")
        self._max_input_bytes = int(max_input_mb) * _MIB
        self._max_decode_bytes = int(max_decode_mb) * _MIB
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

        # Load rapidocr's own default config, override model paths
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

    def _jpeg_decode_factor(self, source_size: tuple[int, int]) -> int:
        max_side_len = getattr(self, "_max_side_len", 1600)
        if max_side_len <= 0:
            return 1

        longest_side = max(source_size)
        if longest_side <= max_side_len:
            return 1
        for factor in (2, 4, 8):
            if math.ceil(longest_side / factor) <= max_side_len:
                return factor
        return 8

    def _preflight_image(self, image_bytes: bytes) -> ImageHeader:
        max_input_bytes = getattr(self, "_max_input_bytes", 16 * _MIB)
        if len(image_bytes) > max_input_bytes:
            raise ImageTooLargeError(
                f"compressed image is {len(image_bytes)} bytes; "
                f"limit is {max_input_bytes} bytes"
            )

        header = self._probe_image_header(image_bytes)
        factor = (
            self._jpeg_decode_factor(header.size)
            if header.format == "JPEG"
            else 1
        )
        decoded_width = math.ceil(header.width / factor)
        decoded_height = math.ceil(header.height / factor)
        estimated_bytes = decoded_width * decoded_height * _DECODE_CHANNELS
        max_decode_bytes = getattr(self, "_max_decode_bytes", 64 * _MIB)
        if estimated_bytes > max_decode_bytes:
            raise ImageTooLargeError(
                f"{header.format} image {header.width}x{header.height} would decode "
                f"to about {estimated_bytes} bytes after {factor}x reduction; "
                f"limit is {max_decode_bytes} bytes"
            )
        return header

    def _decode_flag(self, cv2, source_size: tuple[int, int] | None) -> int:
        if not source_size:
            return cv2.IMREAD_COLOR
        return {
            1: cv2.IMREAD_COLOR,
            2: cv2.IMREAD_REDUCED_COLOR_2,
            4: cv2.IMREAD_REDUCED_COLOR_4,
            8: cv2.IMREAD_REDUCED_COLOR_8,
        }[self._jpeg_decode_factor(source_size)]

    def _infer_image(self, image) -> list[dict]:
        with self._inference_lock:
            output = self._engine(
                image,
                use_det=True,
                use_cls=self._use_angle_cls,
                use_rec=True,
            )
        return normalize_rapidocr_output(output, round_bbox=False)

    def _recognize_single_pass(self, image_bytes: bytes) -> list[dict]:
        import cv2
        import numpy as np

        header = self._preflight_image(image_bytes)
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        source_size = header.size
        jpeg_size = source_size if header.format == "JPEG" else None
        image = cv2.imdecode(encoded, self._decode_flag(cv2, jpeg_size))
        if image is None:
            raise ValueError("invalid compressed image")

        decoded_height, decoded_width = image.shape[:2]
        decoded_bytes = getattr(
            image,
            "nbytes",
            decoded_width * decoded_height * _DECODE_CHANNELS,
        )
        max_decode_bytes = getattr(self, "_max_decode_bytes", 64 * _MIB)
        if decoded_bytes > max_decode_bytes:
            raise ImageTooLargeError(
                f"decoded image uses {decoded_bytes} bytes; "
                f"limit is {max_decode_bytes} bytes"
            )

        if (
            self._max_side_len > 0
            and max(decoded_width, decoded_height) > self._max_side_len
        ):
            resize_scale = self._max_side_len / max(decoded_width, decoded_height)
            target_width = max(1, round(decoded_width * resize_scale))
            target_height = max(1, round(decoded_height * resize_scale))
            image = cv2.resize(
                image,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )
            decoded_width, decoded_height = target_width, target_height

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
            header = self._preflight_image(image_bytes)
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
