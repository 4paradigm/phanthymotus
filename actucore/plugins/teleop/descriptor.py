"""Pure, DDS-free descriptors for the ActuCore teleoperation tools."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

PROFILE_ID = "tianyi2_dual_arm_hands_v1"
SHADOW_PROTOCOL = "motus.teleop.shadow.v1"
LIVE_PROTOCOL = "motus.teleop.live.v1"
RTC_FRAME_PROTOCOL = "motus.teleop.rtc-frame.v1"
RTC_CONTROL_PROTOCOL = "motus.teleop.rtc-control.v1"
SIGNALING_PROTOCOL = "motus.teleop.webrtc-offer-answer.v1"
SIGNALING_AUDIENCE = "motus-teleop-rtc"
CAPTURE_PROTOCOL = "motus.teleop.capture.v1"
PREFLIGHT_SCHEMA = "motus.teleop.g1-preflight.v1"
RECORDING_DISPATCH = "motus.teleop.dispatch.recording.v1"
HARDWARE_DISPATCH = "motus.teleop.dispatch.hardware.v1"
MODES = frozenset({"shadow", "live"})

CAPABILITIES = {
    "profile_id": PROFILE_ID,
    "input_bindings": {
        "head": {"required": True, "role": "reference"},
        "left_controller": {"required": True, "role": "left_end_effector"},
        "right_controller": {"required": True, "role": "right_end_effector"},
    },
    "outputs": {
        "dual_arm": {"enabled": True, "joint_count": 14},
        "base": {"enabled": False},
        "hands": {"enabled": True},
    },
    "effectors": ["dual_arm", "hands"],
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def protocol_for_mode(mode: str) -> str:
    _require_mode(mode)
    return LIVE_PROTOCOL if mode == "live" else SHADOW_PROTOCOL


def dispatch_for_mode(mode: str) -> str:
    _require_mode(mode)
    return HARDWARE_DISPATCH if mode == "live" else RECORDING_DISPATCH


def capability_binding(mode: str, capabilities=None) -> dict:
    _require_mode(mode)
    return {
        "protocol": protocol_for_mode(mode),
        "mode": mode,
        "profile_id": (capabilities or CAPABILITIES)["profile_id"],
        "capabilities": copy.deepcopy(capabilities or CAPABILITIES),
        "dispatch_contract": dispatch_for_mode(mode),
        "signaling": {
            "protocol": SIGNALING_PROTOCOL,
            "capture_protocol": CAPTURE_PROTOCOL,
            "path": "/ws/teleop-capture",
            "access": "paired-capture-credential-only",
            "audience": SIGNALING_AUDIENCE,
        },
    }


def capability_digest(mode: str, capabilities=None) -> str:
    return hashlib.sha256(canonical_json(capability_binding(mode, capabilities))).hexdigest()



def _require_mode(mode):
    if mode not in MODES:
        raise ValueError("mode must be shadow or live")
