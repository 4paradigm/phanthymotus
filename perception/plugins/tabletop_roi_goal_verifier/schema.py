"""Independent MCP schema. No classes, detections, VOP or object-mode arguments."""

from .contracts import RESULT

ACTION_FIELDS = {
    "info": (),
    "ui": (),
    "start": (),
    "stop": (),
    "observe": ("source_id",),
    "get_preview": ("source_id", "observation_id"),
    "start_session": ("request_id", "task_id", "source_id", "target_roi"),
    "capture_before": ("request_id", "session_id", "session_token"),
    "mark_operation_complete": ("request_id", "session_id", "session_token"),
    "verify_after": ("request_id", "session_id", "session_token", "marker_id"),
    "get_result": ("session_id", "session_token"),
    "export_session": ("session_id", "session_token"),
    "reset_session": ("request_id", "session_id", "session_token"),
    "close_session": ("request_id", "session_id", "session_token"),
}
OPTIONAL = {
    "start_session": ("roi_options",),
    "start": ("input_topics", "output_topics", "bindings", "camera_info"),
}


def input_schema():
    props = {
        k: {"type": "string", "minLength": 1, "maxLength": 256}
        for k in (
            "request_id",
            "task_id",
            "source_id",
            "session_id",
            "session_token",
            "marker_id",
            "observation_id",
        )
    }
    props.update(
        action={"type": "string", "enum": list(ACTION_FIELDS), "default": "ui"},
        instance_id={"type": "string", "maxLength": 256},
        target_roi={
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {"type": "number", "minimum": 0, "maximum": 1},
        },
        roi_options={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "change_threshold": {"type": "number", "minimum": 8, "maximum": 80},
                "min_change_fraction": {
                    "type": "number",
                    "minimum": 0.005,
                    "maximum": 0.5,
                },
                "stable_frames": {"type": "integer", "minimum": 2, "maximum": 5},
                "stability_threshold": {"type": "number", "minimum": 2, "maximum": 40},
            },
        },
        input_topics={"type": "array", "items": {"type": "string"}},
        output_topics={"type": "array", "maxItems": 0},
        bindings={"type": "object"},
        camera_info={"type": ["object", "array"]},
    )
    return dict(
        type="object",
        required=["action"],
        additionalProperties=False,
        properties=props,
        **{
            "x-action-params": {
                a: {"params": list(f) + list(OPTIONAL.get(a, ())), "description": a}
                for a, f in ACTION_FIELDS.items()
                if a not in ("start", "stop")
            },
            "x-workbench": {
                "protocol": "motus.workbench/1",
                "entry": "ui",
                "title": "桌面 ROI 目标区域验证",
                "readOnly": True,
                "actions": list(ACTION_FIELDS),
            },
        },
    )


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "schema": {"const": RESULT},
        "result": {"enum": ["PASS", "FAIL", "INCONCLUSIVE"]},
        "code": {"type": "string"},
        "data_valid": {"type": "boolean"},
    },
}
