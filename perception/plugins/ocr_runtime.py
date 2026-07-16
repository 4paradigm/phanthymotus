from __future__ import annotations

import math
import tempfile
import threading
from pathlib import Path

from plugins.ocr_tiled_strategy import (
    AdaptiveTiledOCRStrategy,
    LargeImageStrategyConfig,
    jpeg_dimensions,
)


REQUIRED_MODEL_FILES = ("det.onnx", "rec.onnx", "cls.onnx", "keys.txt")


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
        use_angle_cls: bool = True,
        num_threads: int = 2,
        max_side_len: int = 1600,
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

        from rapidocr import RapidOCR
        from rapidocr.main import DEFAULT_CFG_PATH

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
        cfg["Global"]["max_side_len"] = max_side_len

        engine_cfg = cfg.setdefault("EngineConfig", {}).setdefault(
            "onnxruntime", {}
        )
        engine_cfg["intra_op_num_threads"] = num_threads
        engine_cfg["inter_op_num_threads"] = 1
        engine_cfg["use_cuda"] = False

        with tempfile.TemporaryDirectory(prefix="rapidocr-config-") as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            with config_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, default_flow_style=False)
            self._engine = RapidOCR(config_path=str(config_path))

    @staticmethod
    def _jpeg_dimensions(image_bytes: bytes) -> tuple[int, int] | None:
        return jpeg_dimensions(image_bytes)

    def _decode_flag(self, cv2, source_size: tuple[int, int] | None) -> int:
        if not source_size or self._max_side_len <= 0:
            return cv2.IMREAD_COLOR

        longest_side = max(source_size)
        if longest_side <= self._max_side_len:
            return cv2.IMREAD_COLOR
        for factor, flag in (
            (2, cv2.IMREAD_REDUCED_COLOR_2),
            (4, cv2.IMREAD_REDUCED_COLOR_4),
            (8, cv2.IMREAD_REDUCED_COLOR_8),
        ):
            if math.ceil(longest_side / factor) <= self._max_side_len:
                return flag
        return cv2.IMREAD_REDUCED_COLOR_8

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

        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        source_size = self._jpeg_dimensions(image_bytes)
        image = cv2.imdecode(encoded, self._decode_flag(cv2, source_size))
        if image is None:
            raise ValueError("invalid compressed image")

        decoded_height, decoded_width = image.shape[:2]
        if source_size is None:
            source_size = (decoded_width, decoded_height)

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
            source_size = self._jpeg_dimensions(image_bytes)
            if strategy.should_handle(source_size):
                return strategy.recognize(image_bytes, self._infer_image)
        return self._recognize_single_pass(image_bytes)

    def recognize(self, image_bytes: bytes, language: str = "zh") -> list:
        request_lock = getattr(self, "_request_lock", None)
        if request_lock is None:
            return self._recognize_request(image_bytes)
        with request_lock:
            return self._recognize_request(image_bytes)
