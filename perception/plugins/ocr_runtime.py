from __future__ import annotations

import math
import tempfile
import threading
from pathlib import Path


REQUIRED_MODEL_FILES = ("det.onnx", "rec.onnx", "cls.onnx", "keys.txt")


def normalize_rapidocr_output(
    output, scale_x: float = 1.0, scale_y: float = 1.0
) -> list[dict]:
    if output is None or output.boxes is None:
        return []

    items = []
    for polygon, text, score in zip(output.boxes, output.txts, output.scores):
        xs = [float(point[0]) * scale_x for point in polygon]
        ys = [float(point[1]) * scale_y for point in polygon]
        items.append(
            {
                "text": str(text),
                "bbox": [
                    math.floor(min(xs)),
                    math.floor(min(ys)),
                    math.ceil(max(xs)),
                    math.ceil(max(ys)),
                ],
                "score": float(score),
            }
        )
    return items


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
        self._inference_lock = threading.Lock()

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
        if len(image_bytes) < 4 or image_bytes[:2] != b"\xff\xd8":
            return None

        sof_markers = {
            0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6,
            0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
        }
        offset = 2
        while offset + 3 < len(image_bytes):
            if image_bytes[offset] != 0xFF:
                offset += 1
                continue
            while offset < len(image_bytes) and image_bytes[offset] == 0xFF:
                offset += 1
            if offset >= len(image_bytes):
                break

            marker = image_bytes[offset]
            offset += 1
            if marker in (0x01, 0xD8, 0xD9):
                continue
            if offset + 2 > len(image_bytes):
                break

            segment_len = int.from_bytes(image_bytes[offset:offset + 2], "big")
            if segment_len < 2 or offset + segment_len > len(image_bytes):
                break
            if marker in sof_markers and segment_len >= 7:
                height = int.from_bytes(
                    image_bytes[offset + 3:offset + 5], "big"
                )
                width = int.from_bytes(
                    image_bytes[offset + 5:offset + 7], "big"
                )
                if width > 0 and height > 0:
                    return width, height
            offset += segment_len
        return None

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

    def recognize(self, image_bytes: bytes, language: str = "zh") -> list:
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

        with self._inference_lock:
            output = self._engine(
                image,
                use_det=True,
                use_cls=self._use_angle_cls,
                use_rec=True,
            )
        source_width, source_height = source_size
        return normalize_rapidocr_output(
            output,
            scale_x=source_width / decoded_width,
            scale_y=source_height / decoded_height,
        )
