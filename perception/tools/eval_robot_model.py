"""Calibrate and score recorded robot-model card results on disjoint units.

Usage: python3 tools/eval_robot_model.py manifest.json
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.robot_model_catalog import CANDIDATES


def _iou(left: list, right: list) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    a = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    b = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    return intersection / (a + b - intersection) if a + b > intersection else 0.0


def _observations(entries: list[dict]) -> list[dict]:
    observations = []
    for entry in entries:
        predictions = entry["result"]["objects"]
        used = set()
        for truth in entry["truth"]:
            ranked = sorted(
                ((index, _iou(truth["bbox"], item["bbox"]))
                 for index, item in enumerate(predictions) if index not in used),
                key=lambda pair: pair[1], reverse=True)
            prediction = None
            if ranked and ranked[0][1] >= 0.5:
                index = ranked[0][0]
                used.add(index)
                prediction = predictions[index]
            candidate = (prediction or {}).get("candidates") or []
            top_id = candidate[0]["model_id"] if candidate else None
            score = (prediction or {}).get("score")
            margin = (prediction or {}).get("margin")
            observations.append({"truth": truth["model_id"], "top_id": top_id,
                                 "score": score, "margin": margin})
    return observations


def _accepted(observation: dict, score: float | None,
              margin: float | None) -> bool:
    return (score is not None and margin is not None and
            observation["top_id"] is not None and
            observation["score"] is not None and
            observation["margin"] is not None and
            math.isfinite(observation["score"]) and
            math.isfinite(observation["margin"]) and
            observation["score"] >= score and observation["margin"] >= margin)


def _calibrate(observations: list[dict]) -> dict:
    scores = sorted({item["score"] for item in observations
                     if item["score"] is not None and math.isfinite(item["score"])})
    margins = sorted({item["margin"] for item in observations
                      if item["margin"] is not None and math.isfinite(item["margin"])})
    best = (0, None, None)
    for score, margin in itertools.product(scores, margins):
        accepted = [item for item in observations if _accepted(item, score, margin)]
        if not accepted or any(item["truth"] != item["top_id"] for item in accepted):
            continue
        candidate = (len(accepted), score, margin)
        if candidate > best:
            best = candidate
    if best[0] == 0:
        return {"min_score": None, "min_margin": None}
    return {"min_score": best[1], "min_margin": best[2]}


def evaluate(entries: list[dict]) -> dict:
    """Fit gates on calibration units, report only disjoint test outcomes."""
    if not entries or {item["split"] for item in entries} != {"calibration", "test"}:
        raise ValueError("both calibration and test splits are required")
    units = {split: {str(item["unit_id"]) for item in entries if item["split"] == split}
             for split in ("calibration", "test")}
    if units["calibration"] & units["test"]:
        raise ValueError("unit_id overlaps between calibration and test")
    images = {split: {str(item["image"]) for item in entries if item["split"] == split}
              for split in ("calibration", "test")}
    if images["calibration"] & images["test"]:
        raise ValueError("image overlaps between calibration and test")
    ids = {model_id for model_id, _ in CANDIDATES} | {"unknown"}
    for entry in entries:
        if not entry["truth"] or any(item["model_id"] not in ids for item in entry["truth"]):
            raise ValueError("every truth box needs a supported model_id or unknown")
    calibration = _observations([item for item in entries if item["split"] == "calibration"])
    test = _observations([item for item in entries if item["split"] == "test"])
    thresholds = _calibrate(calibration)
    decided = [item["top_id"] if _accepted(item, thresholds["min_score"],
                                             thresholds["min_margin"]) else "unknown"
               for item in test]
    recall = {}
    for model_id, _ in CANDIDATES:
        indices = [index for index, item in enumerate(test) if item["truth"] == model_id]
        recall[model_id] = (sum(decided[index] == model_id for index in indices) / len(indices)
                            if indices else None)
    unknown = [index for index, item in enumerate(test) if item["truth"] == "unknown"]
    latencies = [float(item["result"]["latency_ms"]) for item in entries
                 if item["split"] == "test" and "latency_ms" in item["result"]]
    return {
        "thresholds": thresholds,
        "calibration_images": len(images["calibration"]),
        "test_images": len(images["test"]),
        "test_objects": len(test),
        "per_model_recall": recall,
        "unknown_false_identification_rate": (
            sum(decided[index] != "unknown" for index in unknown) / len(unknown)
            if unknown else None),
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
    }


if __name__ == "__main__":
    manifest = json.loads(Path(sys.argv[1]).read_text())
    print(json.dumps(evaluate(manifest), indent=2))
