"""PP-OCRv6 Tiny ONNX 离线中英文 OCR。"""

from __future__ import annotations

import math
import os
from typing import Iterable


def resolve_model_base_url(cfg: dict) -> str:
    """环境变量优先，其次使用 YAML 配置。"""
    return os.environ.get("OCR_MODEL_BASE_URL") or cfg.get("model_base_url", "")


def sort_ocr_results(results: list[dict]) -> list[dict]:
    """按从上到下、从左到右排序 OCR 文本行。"""
    return sorted(
        results,
        key=lambda item: (
            int(item["bbox"][1]) // 10,
            int(item["bbox"][0]),
            int(item["bbox"][1]),
        ),
    )


def sanitize_bbox(bbox: Iterable[float], width: int, height: int) -> list[int]:
    """将 bbox 排序并裁剪到图片范围。"""
    values = list(bbox)
    if len(values) != 4:
        raise ValueError("bbox must contain four coordinates")
    x1, y1, x2, y2 = (float(v) for v in values)
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    return [
        max(0, min(width - 1, int(round(left)))),
        max(0, min(height - 1, int(round(top)))),
        max(0, min(width - 1, int(round(right)))),
        max(0, min(height - 1, int(round(bottom)))),
    ]


class PPOCRv6ONNXAdapter:
    """使用 ONNX Runtime CPU 执行 PP-OCRv6 Tiny 检测和识别。"""

    def __init__(self, cfg: dict):
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
            import onnxruntime  # noqa: F401
            import pyclipper  # noqa: F401
            import yaml  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "local_ppocr requires numpy, opencv-python, onnxruntime, "
                "pyclipper and pyyaml"
            ) from exc

        model_dir = cfg.get("model_dir", "/models/ppocr-v6-tiny")
        self.det_model_path = cfg.get(
            "det_model_path", os.path.join(model_dir, "det.onnx")
        )
        self.rec_model_path = cfg.get(
            "rec_model_path", os.path.join(model_dir, "rec.onnx")
        )
        self.rec_config_path = cfg.get(
            "rec_config_path", os.path.join(model_dir, "inference.yml")
        )

        self.det_limit_side_len = int(cfg.get("det_limit_side_len", 736))
        self.det_thresh = float(cfg.get("det_thresh", 0.2))
        self.det_box_thresh = float(cfg.get("det_box_thresh", 0.4))
        self.det_unclip_ratio = float(cfg.get("det_unclip_ratio", 1.4))
        self.rec_score_thresh = float(cfg.get("rec_score_thresh", 0.3))
        self.num_threads = max(1, int(cfg.get("num_threads", 2)))
        self.max_candidates = max(1, int(cfg.get("max_candidates", 1000)))

        self._ensure_models(cfg)
        self.characters = self._load_characters(self.rec_config_path)
        self.det_session = self._create_session(self.det_model_path)
        self.rec_session = self._create_session(self.rec_model_path)

    def _ensure_models(self, cfg: dict) -> None:
        if all(
            os.path.isfile(path)
            for path in (
                self.det_model_path,
                self.rec_model_path,
                self.rec_config_path,
            )
        ):
            return

        if not cfg.get("auto_download", True):
            raise FileNotFoundError(
                "PP-OCRv6 Tiny model files are missing and auto_download is disabled"
            )

        from utils.model_downloader import ensure_model

        base_url = resolve_model_base_url(cfg)
        model_dir = os.path.dirname(self.det_model_path)
        ensure_model("ocr_ppocrv6_tiny", model_dir, base_url=base_url)

    def _create_session(self, model_path: str):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = self.num_threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    @staticmethod
    def _load_characters(config_path: str) -> list[str]:
        import yaml

        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        characters = config.get("PostProcess", {}).get("character_dict")
        if not isinstance(characters, list) or not characters:
            raise RuntimeError(f"character_dict not found in {config_path}")
        return ["blank"] + [str(char) for char in characters]

    def recognize(self, image_bytes: bytes, language: str = "zh") -> list[dict]:
        del language  # PP-OCRv6 使用统一多语言字典，不需要切换语言模型
        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("unsupported or invalid image")

        polygons = self._detect(image)
        results = []
        image_height, image_width = image.shape[:2]
        for polygon, det_score in polygons:
            crop = self._crop_polygon(image, polygon)
            if crop.size == 0:
                continue
            text, rec_score = self._recognize_crop(crop)
            if not text or rec_score < self.rec_score_thresh:
                continue

            xs = polygon[:, 0]
            ys = polygon[:, 1]
            bbox = sanitize_bbox(
                [xs.min(), ys.min(), xs.max(), ys.max()],
                image_width,
                image_height,
            )
            results.append(
                {
                    "text": text,
                    "bbox": bbox,
                    "confidence": round(float(rec_score), 6),
                    "det_confidence": round(float(det_score), 6),
                }
            )
        return sort_ocr_results(results)

    def _detect(self, image):
        import cv2
        import numpy as np

        original_height, original_width = image.shape[:2]
        ratio = min(
            self.det_limit_side_len / max(original_height, original_width),
            1.0,
        )
        resized_width = max(32, int(round(original_width * ratio / 32) * 32))
        resized_height = max(32, int(round(original_height * ratio / 32) * 32))
        resized = cv2.resize(image, (resized_width, resized_height))
        normalized = resized.astype("float32") / 255.0
        normalized = (
            normalized - np.array([0.485, 0.456, 0.406], dtype="float32")
        ) / np.array([0.229, 0.224, 0.225], dtype="float32")
        tensor = normalized.transpose(2, 0, 1)[None, ...]

        input_name = self.det_session.get_inputs()[0].name
        prediction = self.det_session.run(None, {input_name: tensor})[0]
        probability_map = np.squeeze(prediction)
        if probability_map.ndim != 2:
            raise RuntimeError(
                f"unexpected detection output shape: {prediction.shape}"
            )

        bitmap = (probability_map > self.det_thresh).astype("uint8") * 255
        contours, _ = cv2.findContours(
            bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        scale_x = original_width / probability_map.shape[1]
        scale_y = original_height / probability_map.shape[0]
        boxes = []
        for contour in contours[: self.max_candidates]:
            box, short_side = self._mini_box(contour)
            if short_side < 3:
                continue
            score = self._box_score(probability_map, box)
            if score < self.det_box_thresh:
                continue
            expanded = self._unclip(box)
            if expanded is None:
                continue
            box, short_side = self._mini_box(expanded)
            if short_side < 5:
                continue
            box[:, 0] = np.clip(box[:, 0] * scale_x, 0, original_width - 1)
            box[:, 1] = np.clip(box[:, 1] * scale_y, 0, original_height - 1)
            boxes.append((box.astype("float32"), score))
        return boxes

    @staticmethod
    def _mini_box(contour):
        import cv2
        import numpy as np

        rectangle = cv2.minAreaRect(contour)
        points = cv2.boxPoints(rectangle)
        points = sorted(points.tolist(), key=lambda point: point[0])
        left = sorted(points[:2], key=lambda point: point[1])
        right = sorted(points[2:], key=lambda point: point[1])
        ordered = np.array([left[0], right[0], right[1], left[1]], dtype="float32")
        return ordered, min(rectangle[1])

    @staticmethod
    def _box_score(probability_map, box) -> float:
        import cv2
        import numpy as np

        height, width = probability_map.shape
        x_min = max(0, int(np.floor(box[:, 0].min())))
        x_max = min(width - 1, int(np.ceil(box[:, 0].max())))
        y_min = max(0, int(np.floor(box[:, 1].min())))
        y_max = min(height - 1, int(np.ceil(box[:, 1].max())))
        if x_max <= x_min or y_max <= y_min:
            return 0.0

        mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype="uint8")
        shifted = box.copy()
        shifted[:, 0] -= x_min
        shifted[:, 1] -= y_min
        cv2.fillPoly(mask, [shifted.astype("int32")], 1)
        return float(
            cv2.mean(
                probability_map[y_min : y_max + 1, x_min : x_max + 1],
                mask,
            )[0]
        )

    def _unclip(self, box):
        import cv2
        import numpy as np
        import pyclipper

        area = abs(cv2.contourArea(box.astype("float32")))
        perimeter = cv2.arcLength(box.astype("float32"), True)
        if area <= 0 or perimeter <= 0:
            return None
        distance = area * self.det_unclip_ratio / perimeter
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(
            box.astype("int32").tolist(),
            pyclipper.JT_ROUND,
            pyclipper.ET_CLOSEDPOLYGON,
        )
        expanded = offset.Execute(distance)
        if not expanded:
            return None
        largest = max(expanded, key=lambda path: abs(cv2.contourArea(np.array(path))))
        return np.array(largest, dtype="float32").reshape(-1, 1, 2)

    @staticmethod
    def _crop_polygon(image, polygon):
        import cv2
        import numpy as np

        width = int(
            max(
                np.linalg.norm(polygon[0] - polygon[1]),
                np.linalg.norm(polygon[2] - polygon[3]),
            )
        )
        height = int(
            max(
                np.linalg.norm(polygon[0] - polygon[3]),
                np.linalg.norm(polygon[1] - polygon[2]),
            )
        )
        if width < 2 or height < 2:
            return np.empty((0, 0, 3), dtype=image.dtype)
        target = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(polygon.astype("float32"), target)
        crop = cv2.warpPerspective(
            image,
            matrix,
            (width, height),
            borderMode=cv2.BORDER_REPLICATE,
        )
        if crop.shape[0] / max(crop.shape[1], 1) >= 1.5:
            crop = np.rot90(crop)
        return crop

    def _recognize_crop(self, crop) -> tuple[str, float]:
        import cv2
        import numpy as np

        target_height = 48
        max_width = 320
        ratio = crop.shape[1] / max(crop.shape[0], 1)
        resized_width = max(1, min(max_width, int(math.ceil(target_height * ratio))))
        resized = cv2.resize(crop, (resized_width, target_height))
        normalized = resized.astype("float32") / 255.0
        normalized = (normalized - 0.5) / 0.5
        canvas = np.zeros((target_height, max_width, 3), dtype="float32")
        canvas[:, :resized_width, :] = normalized
        tensor = canvas.transpose(2, 0, 1)[None, ...]

        input_name = self.rec_session.get_inputs()[0].name
        probabilities = self.rec_session.run(None, {input_name: tensor})[0][0]
        indices = probabilities.argmax(axis=1)
        scores = probabilities.max(axis=1)

        text_parts = []
        text_scores = []
        previous = -1
        for index, score in zip(indices.tolist(), scores.tolist()):
            if index != 0 and index != previous and index < len(self.characters):
                text_parts.append(self.characters[index])
                text_scores.append(float(score))
            previous = index
        confidence = sum(text_scores) / len(text_scores) if text_scores else 0.0
        return "".join(text_parts), confidence
