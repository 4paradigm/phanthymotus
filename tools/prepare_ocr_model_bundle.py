from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


BASE_URL = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master"
SOURCES = {
    "det.onnx": "onnx/PP-OCRv6/det/PP-OCRv6_det_tiny.onnx",
    "rec.onnx": "onnx/PP-OCRv6/rec/PP-OCRv6_rec_tiny.onnx",
    "cls.onnx": "onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    "keys.txt": (
        "paddle/PP-OCRv6/rec/PP-OCRv6_rec_tiny/ppocrv6_tiny_dict.txt"
    ),
}
MAX_BUNDLE_BYTES = 15 * 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for destination, source in SOURCES.items():
        path = output / destination
        urlretrieve(f"{BASE_URL}/{source}", path)
        if path.stat().st_size == 0:
            raise ValueError(f"Downloaded file is empty: {destination}")

    total = sum((output / name).stat().st_size for name in SOURCES)
    if total > MAX_BUNDLE_BYTES:
        raise ValueError(f"OCR model bundle exceeds 15 MiB: {total} bytes")
    print(f"prepared {len(SOURCES)} files, total={total} bytes")


if __name__ == "__main__":
    main()
