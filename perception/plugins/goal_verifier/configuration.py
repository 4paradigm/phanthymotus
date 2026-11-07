"""Explicit PiPER vision bindings; no laboratory camera/topic defaults."""

from copy import deepcopy
from .contracts import need, text, canonical


def validate(config):
    canonical(config)
    text(config["piper_id"])
    need(
        isinstance(config.get("sources"), dict) and len(config["sources"]) <= 8,
        "INVALID_SOURCE_CONFIG",
    )
    for key, b in config["sources"].items():
        text(key)
        need(b["source_id"] == key, "SOURCE_BINDING_MISMATCH")
        for field in ("camera_id", "frame_id"):
            text(b[field])
        need(b["distortion"] in ("raw", "rectified"), "DISTORTION_REQUIRED")
        kind = b.get("kind", "normalized")
        need(
            b.get("provenance", "real") in ("real", "synthetic", "replay"),
            "INVALID_PROVENANCE",
        )
        need(
            b.get("provenance", "real") == "real" or b.get("allow_test_data") is True,
            "TEST_DATA_REFUSED",
        )
        need(
            kind in ("normalized", "buffered_image", "ros_normalized", "ros_image"),
            "UNKNOWN_SOURCE_KIND",
        )
        for field in (
            "allow_test_data",
            "verified_persistent_tracks",
            "accept_external_observations",
        ):
            need(type(b.get(field, False)) is bool, "INVALID_SOURCE_CONFIG")
        if b.get("verified_persistent_tracks"):
            text(b["identity_record"])
        if b.get("accept_external_observations"):
            text(b["external_observation_record"])
        if kind in ("ros_image", "buffered_image"):
            text(b["view_id"])
            need(b.get("view_mode") == "fixed", "VIEW_EVIDENCE_REQUIRED")
            text(b["fixed_view_record"])
            need(b.get("detector", "color") in ("color", "vop"), "UNKNOWN_DETECTOR")
            if b.get("detector", "color") == "color":
                need(
                    isinstance(b.get("classes"), list) and bool(b["classes"]),
                    "COLOR_CLASSES_REQUIRED",
                )
            else:
                text(b["vop_mcp_url"])
        if kind.startswith("ros_"):
            need(b.get("stamp_semantics") == "capture", "CAPTURE_TIME_UNPROVEN")
            text(b["capture_time_record"])
            text(b["topic"])
            need(b["topic"].startswith("/"), "INVALID_TOPIC")
            need(
                type(b["domain_id"]) is int and 0 <= b["domain_id"] <= 232,
                "INVALID_ROS_DOMAIN",
            )
        if kind == "ros_image":
            need(b["message_type"] in ("compressed", "raw"), "INVALID_IMAGE_TYPE")
    return deepcopy(config)


def template(mode="color"):
    binding = dict(
        source_id="camera_input",
        camera_id=None,
        frame_id=None,
        distortion=None,
        kind="ros_normalized" if mode == "normalized" else "ros_image",
        topic=None,
        domain_id=None,
        allow_test_data=False,
        verified_persistent_tracks=False,
        accept_external_observations=False,
    )
    binding.update(stamp_semantics=None, capture_time_record=None)
    if mode == "normalized":
        binding.update(producer_record=None, external_observation_record=None)
    if mode != "normalized":
        binding.update(
            message_type=None,
            view_id=None,
            view_mode="fixed",
            fixed_view_record=None,
            detector="vop" if mode == "vop" else "color",
        )
        if mode == "color":
            binding["classes"] = [
                dict(
                    class_name=None,
                    color=None,
                    hsv_min=None,
                    hsv_max=None,
                    min_area_px=100,
                )
            ]
        else:
            binding.update(vop_mcp_url=None, vop_tool="vop")
    return dict(
        piper_id=None,
        max_age_s=2,
        session_ttl_s=600,
        sources={"camera_input": binding},
        calibrations={},
    )
