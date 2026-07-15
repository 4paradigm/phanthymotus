from __future__ import annotations

import math
import tempfile
from pathlib import Path


REQUIRED_MODEL_FILES = ("det.onnx", "rec.onnx", "cls.onnx", "keys.txt")


def normalize_rapidocr_output(output) -> list[dict]:
    if output is None or output.boxes is None:
        return []

    items = []
    for polygon, text, score in zip(output.boxes, output.txts, output.scores):
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
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
        self, model_dir: str, use_angle_cls: bool = True, num_threads: int = 2
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

    def recognize(self, image_bytes: bytes, language: str = "zh") -> list:
        import cv2
        import numpy as np

        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("invalid compressed image")
        output = self._engine(
            image,
            use_det=True,
            use_cls=self._use_angle_cls,
            use_rec=True,
        )
        return normalize_rapidocr_output(output)
