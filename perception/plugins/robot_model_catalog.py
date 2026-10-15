"""Visual candidates from phanthymotus-driver README at 44a74e5e80597cb3682be60b6a4ebe9dbac1a43f."""

from __future__ import annotations

import math


CANDIDATES = (
    ("unitree/g1", "a photo of a Unitree G1 humanoid robot"),
    ("unitree/go1", "a photo of a Unitree Go1 quadruped robot"),
    ("unitree/go2", "a photo of a Unitree Go2 quadruped robot"),
    ("unitree/r1", "a photo of a Unitree R1 humanoid robot"),
    ("dji/M300", "a photo of a DJI Matrice 300 RTK drone"),
    ("dji/mavic3e", "a photo of a DJI Mavic 3E drone"),
    ("dji/mavic4e", "a photo of a DJI Mavic 4E drone"),
    ("engineai/t800", "a photo of an EngineAI T800 humanoid robot"),
    ("noetix/bumi", "a photo of a Noetix Bumi-EDU humanoid robot"),
    ("x-humanoid/tianyi2.0", "a photo of an X-Humanoid Tianyi 2.0 robot"),
    ("deep_robotics/lynx_m20", "a photo of a Deep Robotics Lynx M20 quadruped robot"),
    ("chasing/qianjiao_p200_pro", "a photo of a Chasing Qianjiao P200 Pro underwater robot"),
    ("pnpbotics/adam", "a photo of a PNPbotics Adam humanoid robot"),
)

_IDS = frozenset(model_id for model_id, _ in CANDIDATES)


def decide_model(scores: dict[str, float], min_score: float | None,
                 min_margin: float | None) -> dict:
    """Return a model only when calibrated score and runner-up gates pass."""
    ranked = sorted(
        ((model_id, float(score)) for model_id, score in scores.items()
         if model_id in _IDS and math.isfinite(float(score))),
        key=lambda item: item[1], reverse=True,
    )
    candidates = [{"model_id": model_id, "score": score}
                  for model_id, score in ranked]
    if not ranked:
        return {"model_id": "unknown", "score": None, "margin": None,
                "reason": "no_candidates", "candidates": []}

    model_id, score = ranked[0]
    margin = round(score - ranked[1][1], 6) if len(ranked) > 1 else None
    reason = None
    if min_score is None or min_margin is None:
        reason = "uncalibrated"
    elif score < min_score:
        reason = "low_score"
    elif margin is None:
        reason = "insufficient_candidates"
    elif margin < min_margin:
        reason = "ambiguous"
    return {"model_id": model_id if reason is None else "unknown",
            "score": score, "margin": margin, "reason": reason,
            "candidates": candidates}
