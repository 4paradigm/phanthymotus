"""Robot model candidates and the refusal to guess from weak evidence."""

import math

from plugins.robot_model_catalog import CANDIDATES, decide_model


DRIVER_IDS = {
    "unitree/g1", "unitree/go1", "unitree/go2", "unitree/r1",
    "dji/M300", "dji/mavic3e", "dji/mavic4e", "engineai/t800",
    "noetix/bumi", "x-humanoid/tianyi2.0", "deep_robotics/lynx_m20",
    "chasing/qianjiao_p200_pro", "pnpbotics/adam",
}


def test_catalogue_covers_current_driver_models():
    assert {model_id for model_id, _ in CANDIDATES} == DRIVER_IDS
    assert all(prompt.strip() for _, prompt in CANDIDATES)


def test_no_scores_cannot_produce_a_model():
    result = decide_model({}, 0.3, 0.05)
    assert result["model_id"] == "unknown"
    assert result["reason"] == "no_candidates"


def test_uncalibrated_gates_withhold_even_the_top_candidate():
    result = decide_model({"unitree/go2": 0.9, "unitree/go1": 0.2}, None, None)
    assert result["model_id"] == "unknown"
    assert result["reason"] == "uncalibrated"
    assert result["score"] == 0.9
    assert result["margin"] == 0.7


def test_unknown_or_nonfinite_candidates_cannot_win():
    result = decide_model({
        "unlisted/model": 0.99,
        "unitree/g1": math.nan,
        "unitree/go2": 0.8,
    }, 0.3, 0.05)
    assert result["model_id"] == "unknown"
    assert result["reason"] == "insufficient_candidates"
    assert result["candidates"] == [{"model_id": "unitree/go2", "score": 0.8}]


def test_low_score_withholds_a_specific_model():
    result = decide_model({"unitree/go2": 0.29}, 0.3, 0.05)
    assert result["model_id"] == "unknown"
    assert result["reason"] == "low_score"


def test_close_runner_up_withholds_a_specific_model():
    result = decide_model({"unitree/go2": 0.8, "unitree/go1": 0.77}, 0.3, 0.05)
    assert result["model_id"] == "unknown"
    assert result["reason"] == "ambiguous"
    assert result["margin"] == 0.03
