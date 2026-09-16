"""Real CPU inference smoke test; run with PYTHONPATH=perception.

Downloads pinned weights under /models, but needs no ROS, camera or GPU.
"""
import argparse
import json
import time

import cv2
import numpy as np

from plugins.ocr_runtime import RapidOCRAdapter, recognize_to_payload
from utils.model_downloader import ensure_ocr_onnx_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="/models/ocr/ppocrv6-small-onnx")
    args = parser.parse_args()
    ensure_ocr_onnx_model(args.model_dir)
    adapter = RapidOCRAdapter(args.model_dir, backend="onnx-cpu")
    try:
        canvas = np.full((240, 800, 3), 255, dtype=np.uint8)
        cv2.putText(canvas, "HELLO 1234", (40, 135), cv2.FONT_HERSHEY_SIMPLEX,
                    2.5, (0, 0, 0), 4, cv2.LINE_AA)
        start = time.monotonic()
        payload = recognize_to_payload(adapter, cv2.imencode(".png", canvas)[1].tobytes(), "en", 1)
        assert "HELLO1234" in payload["text"].replace(" ", ""), payload
        assert payload["items"] and payload["image_size"] == [800, 240], payload
        print(json.dumps({"text": payload["text"], "seconds": time.monotonic() - start}))
        canvas[:] = 255
        empty = recognize_to_payload(adapter, cv2.imencode(".png", canvas)[1].tobytes(), "en", 2)
        assert empty["text"] == "" and not empty["items"] and "error" not in empty, empty
        bad = recognize_to_payload(adapter, b"not an image", "en", 3)
        assert bad.get("error"), bad
        print("OCR_CPU_REAL_INFERENCE_PASS text blank corrupt-input")
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
