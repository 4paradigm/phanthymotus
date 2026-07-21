#!/usr/bin/env python3
"""
Prepare PP-OCRv6 ONNX model bundles for RapidOCR.

Downloads detection/recognition ONNX models from RapidAI/RapidOCR (hosted on
ModelScope), extracts the embedded character dictionary, and optionally
quantizes the models to shrink size.

The output directory is laid out exactly as the runtime expects:

    <output>/<bundle-name>/
        det.onnx
        rec.onnx
        cls.onnx
        keys.txt

For int8 bundles, det.onnx/rec.onnx are the quantized int8 models, so the
runtime can load them with the standard file names.

Usage:
    conda activate ocr-test
    python3 prepare_ocr_models.py --model-set ppocrv6_small --quant int8 --output ./models/ocr

After preparation, copy the bundle folder to the model server / mount path, e.g.:
    rsync -av ./models/ocr/ppocrv6-small-int8  /mnt/data/lizhuoju/ocr-models/
    # then upload as http://172.28.4.81:34567/lizhuoju/embodied-ai/ppocrv6-small-int8
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# RapidAI/RapidOCR ONNX models on ModelScope
RAPIDAI_BASE = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.1/onnx/PP-OCRv6"

MODEL_SETS = {
    "ppocrv6_tiny": {"det": "det/PP-OCRv6_det_tiny.onnx", "rec": "rec/PP-OCRv6_rec_tiny.onnx"},
    "ppocrv6_small": {"det": "det/PP-OCRv6_det_small.onnx", "rec": "rec/PP-OCRv6_rec_small.onnx"},
    "ppocrv6_medium": {"det": "det/PP-OCRv6_det_medium.onnx", "rec": "rec/PP-OCRv6_rec_medium.onnx"},
}

# Reuse the PP-OCRv4 mobile angle-classifier (PP-OCRv6 has no dedicated cls).
CLS_URL = (
    "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.1/"
    "onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx"
)


def _download(url: str, save_path: Path):
    import urllib.request

    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        print(f"[skip] {save_path.name} already exists")
        return
    print(f"[download] {url} -> {save_path}")
    urllib.request.urlretrieve(url, save_path)
    print(f"[done] {save_path.name}")


def _extract_dict_from_onnx_metadata(rec_onnx: Path, save_path: Path):
    if save_path.exists():
        print(f"[skip] dict already exists: {save_path}")
        return
    import onnxruntime as ort

    sess = ort.InferenceSession(str(rec_onnx))
    meta = sess.get_modelmeta().custom_metadata_map
    chars = meta.get("character", "")
    if not chars:
        print("[warn] no 'character' metadata in rec ONNX; skipping external dict")
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(chars if chars.endswith("\n") else chars + "\n")
    print(f"[done] dict saved ({chars.count(chr(10))} chars): {save_path}")


def _download_cls(model_dir: Path):
    cls_path = model_dir / "cls.onnx"
    _download(CLS_URL, cls_path)


def _norm(arr: np.ndarray) -> np.ndarray:
    """PP-OCRv6 default normalization: mean=0.5, std=0.5."""
    return (arr.astype(np.float32) / 255.0 - 0.5) / 0.5


def _get_font(size: int):
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _synthetic_text(font, max_len: int = 12) -> str:
    """Generate a short mixed Chinese/English/digit string."""
    import random

    digits = "0123456789"
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    # Common CJK characters for calibration (simplified)
    cjk = "的一是不了在人有我他这个们中来上大为和国地到以说时要就出会可也你对生能而子那得于着下自之年过发后作里如进看道还所然事前面后新方两些又现么经么点起都已其里把意气第从多小公无入情很当"
    pool = digits + letters + cjk
    return "".join(random.choices(pool, k=random.randint(4, max_len)))


def _generate_det_calibration(out_dir: Path, num: int = 20):
    """Generate scene-text-like images for detection calibration.

    Detection models require input dimensions to be multiples of 32.
    """
    font = _get_font(20)
    paths = []
    sizes = [(320, 160), (384, 192), (448, 224), (512, 256), (576, 288)]
    for i in range(num):
        w, h = sizes[i % len(sizes)]
        img = Image.new("RGB", (w, h), color=(250, 250, 250))
        draw = ImageDraw.Draw(img)
        for row in range(0, h - 30, 30):
            draw.text((10, row), _synthetic_text(font, 20), fill=(0, 0, 0), font=font)
        path = out_dir / f"det_calib_{i}.png"
        img.save(path)
        paths.append(path)
    return paths


def _generate_rec_calibration(out_dir: Path, num: int = 50):
    """Generate text-line crops for recognition calibration."""
    font = _get_font(24)
    paths = []
    for i in range(num):
        w = 160 + (i % 10) * 24
        img = Image.new("RGB", (w, 48), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((5, 5), _synthetic_text(font, 16), fill=(0, 0, 0), font=font)
        path = out_dir / f"rec_calib_{i}.png"
        img.save(path)
        paths.append(path)
    return paths


def _image_to_chw(path: Path, height: int | None = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if height is not None and img.height != height:
        ratio = height / img.height
        img = img.resize((int(img.width * ratio), height), Image.Resampling.BILINEAR)
    arr = np.array(img)
    arr = _norm(arr)
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, 0)


class _DataReader:
    def __init__(self, paths: Iterable[Path], height: int | None = None):
        self._iter = iter(paths)
        self._height = height

    def get_next(self):
        try:
            path = next(self._iter)
        except StopIteration:
            return None
        return {"x": _image_to_chw(path, self._height)}


def _quantize_int8(src: Path, dst: Path, is_rec: bool):
    if dst.exists():
        print(f"[skip] int8 model already exists: {dst}")
        return
    from onnxruntime.quantization import quantize_static, QuantFormat, QuantType

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        calib_imgs = tmp_path / "calib"
        calib_imgs.mkdir()
        if is_rec:
            paths = _generate_rec_calibration(calib_imgs)
            height = 48
        else:
            paths = _generate_det_calibration(calib_imgs)
            height = None

        print(f"[quantize] int8 static {'rec' if is_rec else 'det'}: {src} -> {dst}")
        quantize_static(
            str(src),
            str(dst),
            _DataReader(paths, height=height),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
        )
    print(f"[done] int8 model size={dst.stat().st_size / 1e6:.2f}MB")


def main():
    parser = argparse.ArgumentParser(description="Prepare PP-OCRv6 ONNX model bundles for RapidOCR")
    parser.add_argument(
        "--model-set",
        choices=list(MODEL_SETS.keys()),
        default="ppocrv6_tiny",
        help="Which PP-OCRv6 model set to prepare",
    )
    parser.add_argument(
        "--output",
        default="./models/ocr",
        help="Output directory for prepared model bundles",
    )
    parser.add_argument(
        "--quant",
        choices=["none", "int8"],
        default="none",
        help="Quantization mode: none | int8 (det+rec static QDQ)",
    )
    parser.add_argument(
        "--bundle-name",
        default="",
        help="Override output bundle folder name (default: <model_set> with '_int8' suffix)",
    )
    args = parser.parse_args()

    bundle_name = args.bundle_name or (
        f"{args.model_set}_int8" if args.quant == "int8" else args.model_set
    )
    model_dir = Path(args.output) / bundle_name
    model_dir.mkdir(parents=True, exist_ok=True)
    urls = MODEL_SETS[args.model_set]

    det_url = f"{RAPIDAI_BASE}/{urls['det']}"
    rec_url = f"{RAPIDAI_BASE}/{urls['rec']}"

    # Always use canonical names so the runtime can find them.
    det_onnx = model_dir / "det.onnx"
    rec_onnx = model_dir / "rec.onnx"

    _download(det_url, det_onnx)
    _download(rec_url, rec_onnx)

    # Dictionary is embedded in RapidAI ONNX rec models.
    _extract_dict_from_onnx_metadata(rec_onnx, model_dir / "keys.txt")

    # Optional cls model (kept for compatibility; not used when use_cls=false).
    _download_cls(model_dir)

    if args.quant == "int8":
        # Replace fp32 models with int8 quantized ones, keeping canonical names.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_det = Path(tmp) / "det_int8.onnx"
            tmp_rec = Path(tmp) / "rec_int8.onnx"
            _quantize_int8(det_onnx, tmp_det, is_rec=False)
            _quantize_int8(rec_onnx, tmp_rec, is_rec=True)
            shutil.move(str(tmp_det), str(det_onnx))
            shutil.move(str(tmp_rec), str(rec_onnx))

    print(f"\nPrepared model bundle: {model_dir.resolve()}")
    total = 0
    for p in sorted(model_dir.iterdir()):
        size = p.stat().st_size
        total += size
        print(f"  {p.name:30s} {size / 1e6:6.2f} MB")
    print(f"  {'TOTAL':30s} {total / 1e6:6.2f} MB")

    if total > 15 * 1024 * 1024:
        print("WARNING: bundle exceeds 15 MiB leaderboard limit!")


if __name__ == "__main__":
    main()
