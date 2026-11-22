"""Compare YuNet thresholds on a few labeled still-image controls."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from plugins.yunet_lk_mosaic.config import MosaicConfig
from plugins.yunet_lk_mosaic.runtime import MosaicEngine, _iou


def read(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"could not read {path}")
    return image


def score(boxes, references):
    matches = 0
    pairs = sorted(((_iou(box, ref), b, r) for b, box in enumerate(boxes)
                    for r, ref in enumerate(references)), reverse=True)
    seen_boxes, seen_refs = set(), set()
    for overlap, box, ref in pairs:
        if overlap >= .25 and box not in seen_boxes and ref not in seen_refs:
            seen_boxes.add(box)
            seen_refs.add(ref)
            matches += 1
    return {"matched": matches, "missed": len(references) - matches,
            "unmatched_detections": len(boxes) - matches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--portrait", type=Path, required=True, help="scikit-image astronaut.png")
    parser.add_argument("--profiles", type=Path, required=True, help="1280×720 side-face still")
    parser.add_argument("--slide", type=Path, required=True, help="matching empty slide still")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    portrait = read(args.portrait)
    profiles = read(args.profiles)
    slide = read(args.slide)
    if portrait.shape[:2] != (512, 512) or profiles.shape[:2] != (720, 1280):
        raise ValueError("portrait or profile fixture has unexpected dimensions")

    # Visible facial area marked by inspection. Uses the same source geometry
    # for all derived scenes, without consuming detector output.
    reference = (184, 70, 260, 161)
    face_roi = tuple(value * 1.25 for value in reference)
    scenes = {"frontal": (portrait, [face_roi])}
    for brightness in (.30, .12, .07):
        scenes[f"brightness_{brightness:.2f}"] = (
            (portrait * brightness).astype(np.uint8), [face_roi],
        )
    for size in (192, 144, 112, 80):
        canvas = np.full((360, 640, 3), 114, dtype=np.uint8)
        canvas[40:40 + size, 80:80 + size] = cv2.resize(
            portrait, (size, size), interpolation=cv2.INTER_AREA,
        )
        a, b, c, d = reference
        small_ref = (80 + a * size / 512, 40 + b * size / 512,
                     80 + c * size / 512, 40 + d * size / 512)
        scenes[f"small_{size}"] = (canvas, [small_ref])
    scenes.update({
        "profiles": (profiles, [(91, 62, 144, 132), (501, 45, 555, 123)]),
        "slide": (slide, []),
        "noise": (np.random.default_rng(153).integers(0, 256, (360, 640, 3), np.uint8), []),
        "blank": (np.zeros((360, 640, 3), dtype=np.uint8), []),
    })

    report = {
        "inputs_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                          for key, path in (("portrait", args.portrait), ("profiles", args.profiles),
                                            ("slide", args.slide), ("model", args.model))},
        "references": {name: refs for name, (_, refs) in scenes.items()},
        "thresholds": {},
    }
    for threshold in (.10, .22, .28, .45, .70, .90):
        engine = MosaicEngine(str(args.model), MosaicConfig().updated({
            "confidence_threshold": threshold,
        }))
        per_scene = {}
        try:
            for name, (image, refs) in scenes.items():
                ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not ok:
                    raise RuntimeError(f"encode failed: {name}")
                boxes = engine._boxes(engine.decode(buf.tobytes()))
                per_scene[name] = score(boxes, refs)
        finally:
            engine.close()
        report["thresholds"][str(threshold)] = per_scene
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
