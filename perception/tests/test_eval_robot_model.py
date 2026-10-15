"""Held-out evaluation must not leak physical units or claim absent classes."""

import pytest

from tools.eval_robot_model import evaluate


def row(split, unit, truth, score, margin, prediction, image=None):
    return {"image": image or f"{split}-{unit}.jpg", "split": split,
            "unit_id": unit, "truth": [{"bbox": [0, 0, 20, 20], "model_id": truth}],
            "result": {"objects": [{"bbox": [0, 0, 20, 20],
                                    "model_id": prediction, "score": score,
                                    "margin": margin,
                                    "candidates": [{"model_id": prediction,
                                                    "score": score}]}]}}


def test_rejects_unit_and_image_leakage():
    rows = [row("calibration", "robot-a", "unitree/g1", .8, .3, "unitree/g1"),
            row("test", "robot-a", "unitree/g1", .8, .3, "unitree/g1")]
    with pytest.raises(ValueError, match="unit_id"):
        evaluate(rows)
    rows[1]["unit_id"] = "robot-b"
    rows[1]["image"] = rows[0]["image"]
    with pytest.raises(ValueError, match="image"):
        evaluate(rows)


def test_unknown_false_identification_and_absent_model_recall():
    rows = [row("calibration", "known-a", "unitree/g1", .8, .3, "unitree/g1"),
            row("calibration", "other-a", "unknown", .2, .1, "unitree/g1"),
            row("test", "known-b", "unitree/g1", .9, .4, "unitree/g1"),
            row("test", "other-b", "unknown", .95, .5, "unitree/g1")]
    report = evaluate(rows)
    assert report["thresholds"] == {"min_score": .8, "min_margin": .3}
    assert report["per_model_recall"]["unitree/g1"] == 1.0
    assert report["per_model_recall"]["unitree/go1"] is None
    assert report["unknown_false_identification_rate"] == 1.0
    assert report["test_images"] == 2


def test_uncalibrated_when_no_clean_separation():
    rows = [row("calibration", "a", "unitree/g1", .6, .2, "unitree/g1"),
            row("calibration", "b", "unknown", .9, .3, "unitree/g1"),
            row("test", "c", "unitree/g1", .6, .2, "unitree/g1")]
    report = evaluate(rows)
    assert report["thresholds"] == {"min_score": None, "min_margin": None}
    assert report["per_model_recall"]["unitree/g1"] == 0.0


def test_crossed_wrong_scores_do_not_crash_calibration():
    rows = [row("calibration", "a", "unknown", .8, .1, "unitree/g1"),
            row("calibration", "b", "unknown", .7, .2, "unitree/g1"),
            row("test", "c", "unknown", .9, .3, "unitree/g1")]
    assert evaluate(rows)["thresholds"] == {"min_score": None, "min_margin": None}
