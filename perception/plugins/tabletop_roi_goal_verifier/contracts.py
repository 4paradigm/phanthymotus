"""Independent tabletop ROI evidence contract; normalized image coordinates."""

import hashlib
import json
import math

NAME = "tabletop_roi_goal_verifier"
OBSERVATION = "motus.roi-image-observation/1"
RESULT = "motus.tabletop-roi-goal-result/1"


class Refusal(ValueError):
    def __init__(self, code, message=None):
        self.code = code
        super().__init__(message or code.replace("_", " ").lower())


def need(condition, code, message=None):
    if not condition:
        raise Refusal(code, message)


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise Refusal("INVALID_JSON", "输入必须是有限数值的 JSON 数据") from exc


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def number(value):
    need(type(value) in (int, float) and math.isfinite(value), "INVALID_NUMBER")
    return float(value)


def text(value):
    need(
        isinstance(value, str) and 0 < len(value) <= 256 and value.strip() == value,
        "INVALID_IDENTIFIER",
    )
    return value


def vector(value, size):
    need(isinstance(value, list) and len(value) == size, "INVALID_VECTOR")
    return [number(v) for v in value]


def fields(value, required, optional=()):
    need(isinstance(value, dict), "INVALID_OBJECT")
    need(
        not set(required) - value.keys(),
        "MISSING_FIELD",
        "缺少字段: " + str(sorted(set(required) - value.keys())),
    )
    need(not value.keys() - set(required) - set(optional), "UNKNOWN_FIELD")


def error_result(exc):
    return {
        "schema": RESULT,
        "state": "error",
        "result": "INCONCLUSIVE",
        "data_valid": False,
        "code": getattr(exc, "code", "INPUT_ERROR"),
        "message": str(exc),
    }
