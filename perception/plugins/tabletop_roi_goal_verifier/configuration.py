"""Explicit real camera binding; offline buffers must opt into labelled test data."""

from copy import deepcopy

from .contracts import canonical, need, number, text


def validate(config):
    canonical(config)
    text(config["verifier_id"])
    need(
        isinstance(config.get("sources"), dict) and 1 <= len(config["sources"]) <= 8,
        "INVALID_SOURCE_CONFIG",
    )
    need(0.05 <= number(config.get("max_age_s", 2)) <= 5, "INVALID_MAX_AGE")
    need(1 <= number(config.get("session_ttl_s", 600)) <= 3600, "INVALID_SESSION_TTL")
    for key, b in config["sources"].items():
        text(key)
        need(b["source_id"] == key, "SOURCE_BINDING_MISMATCH")
        for field in ("camera_id", "frame_id", "view_id", "fixed_view_record"):
            need(
                isinstance(b.get(field), str) and bool(b[field].strip()),
                "CAMERA_BINDING_REQUIRED",
                f"请填写已核实的 sources.{key}.{field}",
            )
            text(b[field])
        need(b.get("view_mode") == "fixed", "VIEW_EVIDENCE_REQUIRED")
        need(b["distortion"] in ("raw", "rectified"), "DISTORTION_REQUIRED")
        need(
            b.get("kind") in ("ros_image", "buffered_image"),
            "ROI_IMAGE_SOURCE_REQUIRED",
        )
        provenance = b.get("provenance", "real")
        need(provenance in ("real", "synthetic", "replay"), "INVALID_PROVENANCE")
        need(type(b.get("allow_test_data", False)) is bool, "INVALID_SOURCE_CONFIG")
        need(
            provenance == "real" or b.get("allow_test_data") is True,
            "TEST_DATA_REFUSED",
        )
        if b["kind"] == "buffered_image":
            need(
                provenance in ("synthetic", "replay")
                and b.get("allow_test_data") is True,
                "OFFLINE_TEST_DATA_REQUIRED",
            )
        else:
            need(b.get("stamp_semantics") == "capture", "CAPTURE_TIME_UNPROVEN")
            need(
                isinstance(b.get("capture_time_record"), str)
                and bool(b["capture_time_record"].strip()),
                "CAPTURE_TIME_UNPROVEN",
                f"请填写 sources.{key}.capture_time_record 的真实核实记录",
            )
            text(b["capture_time_record"])
            text(b["topic"])
            need(b["topic"].startswith("/"), "INVALID_TOPIC")
            need(
                type(b["domain_id"]) is int and b["domain_id"] == 42,
                "ROS_DOMAIN_42_REQUIRED",
            )
            need(b["message_type"] in ("raw", "compressed"), "INVALID_IMAGE_TYPE")
    return deepcopy(config)
