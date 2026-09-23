"""Pure Frame v1 validation and HMAC ticket helpers for G1 teleoperation.

This module deliberately has no aiohttp or aiortc dependency.  It is shared by
the HTTP/RTC adapters and the offline protocol tests.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import re
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .descriptor import SIGNALING_AUDIENCE

FRAME_SCHEMA_VERSION = 1
MAX_FRAME_BYTES = 64 * 1024
MAX_SEQUENCE = (1 << 63) - 1
SUPPORTED_MODES = frozenset({"shadow", "live"})
_TOKEN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
_FENCE_RE = re.compile(r"^[A-Za-z0-9_-]{24,128}$")


class ProtocolError(ValueError):
    """A stable, user-visible protocol validation failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class TicketError(ProtocolError):
    """A ticket authentication or replay failure."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


FRAME_REQUIRED_FIELDS = frozenset({
    "schema_version", "boot_id", "session_id", "epoch", "fence",
    "sequence", "client_monotonic_ns", "mode", "deadman",
    "clutch_sequence", "tracking", "head", "left_controller",
    "right_controller", "controllers",
})
FRAME_OPTIONAL_FIELDS = frozenset({"base_twist"})
RTC_PRIVATE_FIELDS = frozenset({"boot_id", "session_id", "epoch", "fence"})
RTC_FRAME_REQUIRED_FIELDS = FRAME_REQUIRED_FIELDS - RTC_PRIVATE_FIELDS


def _strict_object(value: Any, name: str, required: set[str], optional: set[str] | None = None) -> dict:
    if not isinstance(value, dict):
        raise ProtocolError("invalid_type", f"{name} must be an object")
    optional = optional or set()
    keys = set(value)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise ProtocolError("missing_field", f"{name} missing fields: {sorted(missing)}")
    if unknown:
        raise ProtocolError("unknown_field", f"{name} contains unknown fields: {sorted(unknown)}")
    return value


def _strict_int(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError("invalid_type", f"{name} must be an integer")
    if value < minimum:
        raise ProtocolError("out_of_range", f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ProtocolError("out_of_range", f"{name} must be <= {maximum}")
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ProtocolError("invalid_type", f"{name} must be a boolean")
    return value


def _finite_number(value: Any, name: str, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError("invalid_type", f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ProtocolError("non_finite", f"{name} must be finite")
    if number < lower or number > upper:
        raise ProtocolError("out_of_range", f"{name} must be in [{lower}, {upper}]")
    return number


def _float_array(value: Any, name: str, *, size: int | None, maximum_size: int | None,
                 lower: float, upper: float) -> list[float]:
    if not isinstance(value, list):
        raise ProtocolError("invalid_type", f"{name} must be an array")
    if size is not None and len(value) != size:
        raise ProtocolError("invalid_length", f"{name} must contain exactly {size} values")
    if maximum_size is not None and len(value) > maximum_size:
        raise ProtocolError("invalid_length", f"{name} may contain at most {maximum_size} values")
    return [
        _finite_number(item, f"{name}[{index}]", lower=lower, upper=upper)
        for index, item in enumerate(value)
    ]


def _validate_uuid(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ProtocolError("invalid_type", f"{name} must be a UUID string")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ProtocolError("invalid_value", f"{name} must be a valid UUID") from exc
    if str(parsed) != value.lower():
        raise ProtocolError("invalid_value", f"{name} must use canonical UUID form")
    return str(parsed)


def _validate_fence(value: Any) -> str:
    if not isinstance(value, str) or not _FENCE_RE.fullmatch(value):
        raise ProtocolError("invalid_value", "fence must be a URL-safe token of 24-128 characters")
    return value


def _validate_pose(value: Any, name: str, tracking_valid: bool) -> dict | None:
    if value is None:
        if tracking_valid:
            raise ProtocolError("tracking_mismatch", f"{name} is required while tracking is valid")
        return None
    if not tracking_valid:
        raise ProtocolError("tracking_mismatch", f"{name} must be null while tracking is invalid")
    pose = _strict_object(value, name, {"position", "orientation"})
    position = _float_array(
        pose["position"], f"{name}.position", size=3, maximum_size=None, lower=-100.0, upper=100.0
    )
    orientation = _float_array(
        pose["orientation"], f"{name}.orientation", size=4, maximum_size=None, lower=-1.0, upper=1.0
    )
    norm = math.sqrt(sum(item * item for item in orientation))
    if abs(norm - 1.0) > 0.02:
        raise ProtocolError("invalid_quaternion", f"{name}.orientation must be normalized")
    return {"position": position, "orientation": orientation}


def validate_tracked_pose_v1(value: Any, name: str = "pose") -> dict:
    """Apply the exact Frame v1 bounds to one required tracked pose."""

    result = _validate_pose(value, name, True)
    assert result is not None
    return result


def _validate_controller(value: Any, name: str) -> dict:
    controller = _strict_object(value, name, {"axes", "buttons"})
    return {
        "axes": _float_array(
            controller["axes"], f"{name}.axes", size=None, maximum_size=8, lower=-1.0, upper=1.0
        ),
        "buttons": _float_array(
            controller["buttons"], f"{name}.buttons", size=None, maximum_size=16, lower=0.0, upper=1.0
        ),
    }


def _normalize_frame_v1(value: Any, *, expected_mode: str) -> dict:
    if expected_mode not in SUPPORTED_MODES:
        raise ValueError("expected_mode must be 'shadow' or 'live'")
    frame = _strict_object(
        value,
        "frame",
        set(FRAME_REQUIRED_FIELDS),
        set(FRAME_OPTIONAL_FIELDS),
    )
    version = _strict_int(frame["schema_version"], "schema_version", minimum=1)
    if version != FRAME_SCHEMA_VERSION:
        raise ProtocolError("unsupported_version", f"schema_version must be {FRAME_SCHEMA_VERSION}")
    if frame["mode"] != expected_mode:
        raise ProtocolError("unsupported_mode", f"mode must be {expected_mode!r}")

    tracking = _strict_object(
        frame["tracking"], "tracking", {"head", "left_controller", "right_controller"}
    )
    normalized_tracking = {
        key: _strict_bool(tracking[key], f"tracking.{key}")
        for key in ("head", "left_controller", "right_controller")
    }
    controllers = _strict_object(frame["controllers"], "controllers", {"left", "right"})

    normalized = {
        "schema_version": version,
        "boot_id": _validate_uuid(frame["boot_id"], "boot_id"),
        "session_id": _validate_uuid(frame["session_id"], "session_id"),
        "epoch": _strict_int(frame["epoch"], "epoch", minimum=1),
        "fence": _validate_fence(frame["fence"]),
        "sequence": _strict_int(
            frame["sequence"], "sequence", minimum=0, maximum=MAX_SEQUENCE
        ),
        "client_monotonic_ns": _strict_int(
            frame["client_monotonic_ns"], "client_monotonic_ns", minimum=0
        ),
        "mode": expected_mode,
        "deadman": _strict_bool(frame["deadman"], "deadman"),
        "clutch_sequence": _strict_int(
            frame["clutch_sequence"],
            "clutch_sequence",
            minimum=0,
            maximum=MAX_SEQUENCE,
        ),
        "tracking": normalized_tracking,
        "head": _validate_pose(frame["head"], "head", normalized_tracking["head"]),
        "left_controller": _validate_pose(
            frame["left_controller"], "left_controller", normalized_tracking["left_controller"]
        ),
        "right_controller": _validate_pose(
            frame["right_controller"], "right_controller", normalized_tracking["right_controller"]
        ),
        "controllers": {
            "left": _validate_controller(controllers["left"], "controllers.left"),
            "right": _validate_controller(controllers["right"], "controllers.right"),
        },
    }
    if "base_twist" in frame:
        twist = _strict_object(frame["base_twist"], "base_twist", {"linear", "angular"})
        normalized["base_twist"] = {
            "linear": _float_array(
                twist["linear"], "base_twist.linear", size=3, maximum_size=None, lower=-20.0, upper=20.0
            ),
            "angular": _float_array(
                twist["angular"], "base_twist.angular", size=3, maximum_size=None,
                lower=-20.0, upper=20.0
            ),
        }
    return normalized


def validate_frame_v1(
    value: Any,
    *,
    max_bytes: int = MAX_FRAME_BYTES,
    expected_mode: str,
) -> dict:
    """Validate and normalize one authority-bearing Teleop Frame v1 object."""

    try:
        encoded_size = len(canonical_json(value))
    except (TypeError, ValueError, RecursionError) as exc:
        raise ProtocolError("invalid_json", "frame must contain JSON-compatible finite values") from exc
    if encoded_size > max_bytes:
        raise ProtocolError("frame_too_large", f"frame exceeds {max_bytes} bytes")
    return _normalize_frame_v1(value, expected_mode=expected_mode)


def bind_rtc_frame_v1(
    value: Any,
    *,
    authority: Mapping[str, Any],
    max_bytes: int = MAX_FRAME_BYTES,
    expected_mode: str,
) -> dict:
    """Bind a public RTC wire Frame to its ticket-authorized private identity.

    The browser is deliberately forbidden from supplying any authority field.
    The returned internal Frame can therefore use the same runtime and final
    dispatch validation as authenticated MCP diagnostics without disclosing the
    session fence to JavaScript.
    """

    try:
        encoded_size = len(canonical_json(value))
    except (TypeError, ValueError, RecursionError) as exc:
        raise ProtocolError("invalid_json", "RTC frame must contain JSON-compatible finite values") from exc
    if encoded_size > max_bytes:
        raise ProtocolError("frame_too_large", f"RTC frame exceeds {max_bytes} bytes")
    wire = _strict_object(
        value,
        "RTC frame",
        set(RTC_FRAME_REQUIRED_FIELDS),
        set(FRAME_OPTIONAL_FIELDS),
    )
    if not isinstance(authority, Mapping):
        raise ProtocolError("invalid_rtc_binding", "RTC authority binding is invalid")
    try:
        internal = {
            "boot_id": authority["boot_id"],
            "session_id": authority["session_id"],
            "epoch": authority["epoch"],
            "fence": authority["fence"],
            **wire,
        }
    except KeyError as exc:
        raise ProtocolError("invalid_rtc_binding", "RTC authority binding is incomplete") from exc
    return _normalize_frame_v1(internal, expected_mode=expected_mode)


def sdp_digest(sdp: str) -> str:
    if not isinstance(sdp, str) or not sdp:
        raise TicketError("invalid_sdp", "sdp must be a non-empty string")
    return hashlib.sha256(sdp.encode("utf-8")).hexdigest()


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise TicketError("malformed_ticket", "ticket segment is empty")
    try:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except Exception as exc:
        raise TicketError("malformed_ticket", "ticket is not valid base64url") from exc


@dataclass(frozen=True)
class TicketClaims:
    boot_id: str
    session_id: str
    epoch: int
    fence: str
    capability_digest: str
    sdp_sha256: str
    iat: int
    exp: int
    jti: str
    audience: str = SIGNALING_AUDIENCE

    def as_dict(self) -> dict:
        return {
            "v": 1,
            "aud": self.audience,
            "boot_id": self.boot_id,
            "session_id": self.session_id,
            "epoch": self.epoch,
            "fence": self.fence,
            "capability_digest": self.capability_digest,
            "sdp_sha256": self.sdp_sha256,
            "iat": self.iat,
            "exp": self.exp,
            "jti": self.jti,
        }


class TicketCodec:
    """Small HMAC-SHA256 codec for Driver-internal Capture authorization."""

    def __init__(self, secret: str | bytes):
        if isinstance(secret, str):
            try:
                secret_bytes = secret.encode("utf-8")
            except UnicodeEncodeError:
                raise ValueError("ticket secret must be valid UTF-8") from None
        else:
            secret_bytes = bytes(secret)
        if len(secret_bytes) < 32:
            raise ValueError("ticket secret must contain at least 32 bytes")
        self._secret = secret_bytes

    def sign(self, claims: TicketClaims) -> str:
        payload = _b64encode(canonical_json(claims.as_dict()))
        signature = _b64encode(hmac.new(self._secret, payload.encode("ascii"), hashlib.sha256).digest())
        return f"{payload}.{signature}"

    def decode_and_verify_signature(self, token: str) -> dict:
        if not isinstance(token, str) or token.count(".") != 1:
            raise TicketError("malformed_ticket", "ticket must contain payload and signature")
        payload, signature = token.split(".", 1)
        expected = _b64encode(hmac.new(self._secret, payload.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            raise TicketError("invalid_signature", "ticket signature is invalid")
        try:
            decoded = json.loads(_b64decode(payload))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TicketError("malformed_ticket", "ticket payload is not valid JSON") from exc
        return _strict_object(
            decoded,
            "ticket",
            {
                "v", "aud", "boot_id", "session_id", "epoch", "fence",
                "capability_digest", "sdp_sha256", "iat", "exp", "jti",
            },
        )


class TicketVerifier:
    """Verify binding, expiry and one-time use for RTC offers."""

    def __init__(
        self,
        codec: TicketCodec,
        *,
        max_ttl_seconds: int = 30,
        max_replay_entries: int = 4096,
        wall_clock: Callable[[], float] = time.time,
        audience: str = SIGNALING_AUDIENCE,
    ):
        self._codec = codec
        self._max_ttl = int(max_ttl_seconds)
        self._max_replay_entries = int(max_replay_entries)
        self._wall_clock = wall_clock
        self._audience = audience
        self._used: dict[str, int] = {}
        self._lock = threading.Lock()

    def verify_and_consume(self, token: str, *, expected: Mapping[str, Any], sdp: str) -> dict:
        claims = self._codec.decode_and_verify_signature(token)
        now = int(self._wall_clock())
        if claims["v"] != 1 or claims["aud"] != self._audience:
            raise TicketError("invalid_audience", "ticket version or audience is invalid")
        iat = _strict_int(claims["iat"], "ticket.iat", minimum=0)
        exp = _strict_int(claims["exp"], "ticket.exp", minimum=1)
        if iat > now + 5:
            raise TicketError("ticket_not_yet_valid", "ticket issue time is in the future")
        if exp <= now:
            raise TicketError("ticket_expired", "ticket has expired")
        if exp - iat <= 0 or exp - iat > self._max_ttl:
            raise TicketError("invalid_ticket_ttl", f"ticket TTL must be <= {self._max_ttl} seconds")
        jti = claims["jti"]
        if not isinstance(jti, str) or not _TOKEN_ID_RE.fullmatch(jti):
            raise TicketError("invalid_jti", "ticket jti is invalid")
        if claims["sdp_sha256"] != sdp_digest(sdp):
            raise TicketError("sdp_mismatch", "ticket is not bound to this SDP offer")
        for key, value in expected.items():
            if claims.get(key) != value:
                raise TicketError("binding_mismatch", f"ticket {key} does not match the active session")

        with self._lock:
            self._used = {key: expiry for key, expiry in self._used.items() if expiry > now}
            if jti in self._used:
                raise TicketError("ticket_replayed", "ticket has already been used")
            if len(self._used) >= self._max_replay_entries:
                raise TicketError("replay_cache_full", "ticket replay cache is full")
            self._used[jti] = exp
        return claims


def make_ticket_claims(
    *,
    session: Mapping[str, Any],
    sdp: str,
    ttl_seconds: int = 20,
    wall_clock: Callable[[], float] = time.time,
    jti: str | None = None,
    audience: str = SIGNALING_AUDIENCE,
) -> TicketClaims:
    """Reference claim builder for Core integration and local tests."""

    now = int(wall_clock())
    return TicketClaims(
        boot_id=str(session["boot_id"]),
        session_id=str(session["session_id"]),
        epoch=int(session["epoch"]),
        fence=str(session["fence"]),
        capability_digest=str(session["capability_digest"]),
        sdp_sha256=sdp_digest(sdp),
        iat=now,
        exp=now + int(ttl_seconds),
        jti=jti or _b64encode(uuid.uuid4().bytes),
        audience=audience,
    )
