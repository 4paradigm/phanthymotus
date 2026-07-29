"""Lightweight image preprocessing for OCR robustness on real-world images.

Design principles (from the OCR robustness guide):
1. Enhancements are PER-IMAGE, on-demand — NEVER global
2. Enhancement result is accepted ONLY if letter count UP AND confidence not DOWN
3. Any geometric change projects bboxes back to the original image
"""

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


def _is_dark_background(gray: np.ndarray) -> bool:
    """Check if the image has a dark background (light text)."""
    mean_val = float(np.mean(gray))
    dark_ratio = float(np.sum(gray < 90)) / gray.size
    return mean_val < 90 and dark_ratio > 0.6


def _invert_image(image: np.ndarray) -> np.ndarray:
    """Invert the image (light text on dark background → dark text on light)."""
    return 255 - image


def _normalize_lighting(image: np.ndarray) -> np.ndarray:
    """Gaussian blur background estimation → division normalization.

    Handles uneven lighting and glare without affecting text regions.
    """
    h, w = image.shape[:2]
    kernel_size = max(h, w) // 30
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    f = image.astype(np.float32)
    bg = cv2.GaussianBlur(f, (kernel_size, kernel_size), 0)
    bg = np.maximum(bg, 1.0)
    norm = f / bg
    norm = norm / np.percentile(norm, 99.5)
    return np.clip(norm * 255, 0, 255).astype(np.uint8)


def _apply_clahe(image: np.ndarray) -> np.ndarray:
    """CLAHE on LAB L-channel only (prevents color distortion)."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _needs_lighting_fix(gray: np.ndarray) -> bool:
    """Detect if the image has lighting issues worth fixing."""
    std = float(np.std(gray))
    return std > 60  # high contrast = likely glare/uneven lighting


def _needs_clahe(gray: np.ndarray) -> bool:
    """Detect if the image has low contrast."""
    mean_val = float(np.mean(gray))
    return mean_val < 100  # dark image


def preprocess_for_ocr(
    image: np.ndarray,
    *,
    enable_lighting: bool = True,
    enable_polarity: bool = True,
    enable_clahe: bool = True,
) -> np.ndarray:
    """Apply on-demand preprocessing to improve OCR on real-world images.

    Args:
        image: BGR image (uint8).
        enable_lighting: enable lighting normalization.
        enable_polarity: enable dark-background inversion.
        enable_clahe: enable CLAHE for low-contrast images.

    Returns:
        Preprocessed BGR image (same dimensions as input).
        If no enhancement is needed, returns the original image unchanged.

    Raises:
        ValueError: if image is invalid.
    """
    if image is None or image.size == 0:
        raise ValueError("invalid image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = image.copy()

    # 1. Polarity: handle dark-background images FIRST
    if enable_polarity and _is_dark_background(gray):
        log.debug("preprocess: dark background detected, inverting")
        result = _invert_image(result)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # 2. Lighting normalization: handle glare/uneven lighting
    if enable_lighting and _needs_lighting_fix(gray):
        log.debug("preprocess: uneven lighting detected, normalizing")
        result = _normalize_lighting(result)

    # 3. CLAHE: boost low-contrast images
    if enable_clahe and _needs_clahe(gray):
        log.debug("preprocess: low contrast detected, applying CLAHE")
        result = _apply_clahe(result)

    return result
