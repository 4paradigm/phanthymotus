"""MCP input schema and action-specific required/allowed fields."""

from .contracts import RESULT, OBSERVATION

STRING = dict(type="string", minLength=1, maxLength=256)


def vec(n):
    return dict(type="array", minItems=n, maxItems=n, items=dict(type="number"))


GOAL = dict(
    type="object",
    required=["units", "frame_id", "shape", "tolerance"],
    additionalProperties=False,
    properties=dict(
        units=dict(enum=["pixels", "normalized", "plane_mm"]),
        frame_id=STRING,
        shape=dict(enum=["circle", "rectangle"]),
        tolerance=dict(type="number", minimum=0),
        center=vec(2),
        radius=dict(type="number", exclusiveMinimum=0),
        bounds=vec(4),
    ),
)
GOAL["allOf"] = [
    {
        "if": {"properties": {"shape": {"const": "circle"}}},
        "then": {"required": ["center", "radius"]},
    },
    {
        "if": {"properties": {"shape": {"const": "rectangle"}}},
        "then": {"required": ["bounds"]},
    },
]
OBS_SCHEMA = dict(
    type="object",
    additionalProperties=False,
    required=[
        "schema",
        "observation_id",
        "source_id",
        "source_epoch",
        "sequence",
        "camera_id",
        "frame_id",
        "view_id",
        "image_size",
        "distortion",
        "captured_at",
        "timestamp_basis",
        "provenance",
        "objects",
    ],
    properties={
        **{
            k: STRING
            for k in (
                "observation_id",
                "source_id",
                "source_epoch",
                "camera_id",
                "frame_id",
                "view_id",
                "calibration_id",
                "track_epoch",
            )
        },
        "schema": dict(const=OBSERVATION),
        "image_sha256": dict(type="string", pattern="^[0-9a-f]{64}$"),
        "sequence": dict(type="integer", minimum=0),
        "image_size": dict(
            type="array",
            minItems=2,
            maxItems=2,
            items=dict(type="integer", minimum=1, maximum=8192),
        ),
        "distortion": dict(enum=["raw", "rectified"]),
        "captured_at": dict(type="number"),
        "timestamp_basis": dict(const="capture"),
        "provenance": dict(enum=["real", "synthetic", "replay"]),
        "objects": dict(
            type="array",
            maxItems=100,
            items=dict(
                type="object",
                additionalProperties=False,
                required=["detection_id", "class_name", "confidence", "bbox"],
                properties=dict(
                    detection_id=STRING,
                    class_name=STRING,
                    color=STRING,
                    track_id=STRING,
                    confidence_kind=STRING,
                    confidence=dict(type="number", minimum=0, maximum=1),
                    bbox=vec(4),
                ),
            ),
        ),
    },
)

ACTION_FIELDS = {
    "info": (),
    "ui": (),
    "start": (),
    "stop": (),
    "observe": ("source_id",),
    "get_preview": ("source_id", "observation_id"),
    "submit_observation": ("observation",),
    "start_session": ("request_id", "task_id", "source_id", "selector", "goal"),
    "set_goal": ("request_id", "session_id", "session_token", "goal"),
    "capture_before": ("request_id", "session_id", "session_token"),
    "mark_operation_complete": ("request_id", "session_id", "session_token"),
    "verify_after": ("request_id", "session_id", "session_token", "marker_id"),
    "get_result": ("session_id", "session_token"),
    "export_session": ("session_id", "session_token"),
    "reset_session": ("request_id", "session_id", "session_token"),
    "close_session": ("request_id", "session_id", "session_token"),
}
OPTIONAL = {
    "start_session": ("min_confidence", "max_displacement_norm"),
    "capture_before": ("observation",),
    "verify_after": ("observation",),
    "start": ("input_topics", "output_topics", "camera_info", "bindings"),
}


def input_schema():
    strings = (
        "request_id",
        "task_id",
        "source_id",
        "session_id",
        "session_token",
        "marker_id",
        "observation_id",
    )
    props = {name: dict(type="string", minLength=1, maxLength=256) for name in strings}
    props["session_token"].update({"format": "password", "x-sensitive": True})
    props.update(
        action=dict(
            type="string",
            enum=["ui"] + [a for a in ACTION_FIELDS if a != "ui"],
            default="ui",
        ),
        instance_id=dict(type="string", maxLength=256),
        selector=dict(
            type="object",
            required=["class_name"],
            additionalProperties=False,
            properties={
                k: dict(type="string", minLength=1, maxLength=256)
                for k in (
                    "class_name",
                    "color",
                    "detection_id",
                    "observation_id",
                    "track_id",
                )
            },
        ),
        goal=GOAL,
        observation=OBS_SCHEMA,
        min_confidence=dict(type="number", minimum=0.1, maximum=1),
        max_displacement_norm=dict(type="number", exclusiveMinimum=0, maximum=1.4143),
        input_topics=dict(type="array", items=dict(type="string")),
        output_topics=dict(type="array", items=dict(type="string")),
        camera_info=dict(type=["object", "array"]),
        bindings=dict(type="object"),
    )
    schema = dict(
        type="object", required=["action"], additionalProperties=False, properties=props
    )
    schema["x-action-params"] = {
        a: dict(params=list(keys), description=a)
        for a, keys in ACTION_FIELDS.items()
        if a not in ("start", "stop")
    }
    # Core preserves inputSchema but filters tool.annotations on discovery.
    # This is UI metadata only, never an execution permission.
    schema["x-workbench"] = dict(
        protocol="motus.workbench/1",
        entry="ui",
        title="PiPER 物体目标达成验证",
        readOnly=True,
        actions=list(ACTION_FIELDS),
    )
    return schema


OUTPUT_SCHEMA = dict(
    type="object",
    properties=dict(
        schema=dict(const=RESULT),
        result=dict(enum=["PASS", "FAIL", "INCONCLUSIVE"]),
        code=dict(type="string"),
        data_valid=dict(type="boolean"),
    ),
)
