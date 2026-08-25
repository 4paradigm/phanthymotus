"""Shape-preserving image enhancement used before text detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    import cv2
    import numpy as np

    if image is None or image.size == 0:
        raise ValueError("invalid image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = image

    if float(np.mean(gray)) < 90 and np.count_nonzero(gray < 90) / gray.size > 0.6:
        result = 255 - result
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    if float(np.std(gray)) > 60:
        kernel = max(result.shape[:2]) // 30
        kernel = kernel if kernel % 2 else kernel + 1
        value = result.astype(np.float32)
        background = np.maximum(
            cv2.GaussianBlur(value, (kernel, kernel), 0), 1.0
        )
        value /= background
        value /= np.percentile(value, 99.5)
        result = np.clip(value * 255, 0, 255).astype(np.uint8)

    if float(np.mean(gray)) < 100:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        lightness = cv2.createCLAHE(
            clipLimit=2.0, tileGridSize=(8, 8)
        ).apply(lightness)
        result = cv2.cvtColor(
            cv2.merge([lightness, channel_a, channel_b]),
            cv2.COLOR_LAB2BGR,
        )

    return result
