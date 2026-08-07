from __future__ import annotations

import base64
import json
import math
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


STATE_DIM = 29
ACTION_DIM = 18
ACTION_STEPS = 50
ACTION_SHAPE = [1, ACTION_STEPS, ACTION_DIM]
ACTION_SPACE = "physical_quantile_unnormalized"
NUM_INFERENCE_STEPS = 10
FREQUENCY_HZ = 30
DEFAULT_TASK = "move blue box back and forth between tables"


class CardError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def as_result(self) -> dict[str, Any]:
        return {"status": "error", "error_code": self.code, "message": self.message}


@dataclass(frozen=True)
class ObservationSnapshot:
    image_bytes: bytes
    image_topic: str
    image_frame_id: str
    image_stamp: float
    image_stamp_source: str
    image_received_at: float
    image_age_at_request_s: float
    state: tuple[float, ...]
    state_topic: str
    state_received_at: float
    state_age_at_request_s: float


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CardError("policy_contract_error", f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise CardError("policy_contract_error", f"{name} must be finite")
    return number


def validate_http_url(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    url = value.strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{name} must be an absolute http:// or https:// URL")
    return url


def derive_health_url(policy_url: str) -> str:
    parsed = urlparse(policy_url)
    return parsed._replace(path="/health", params="", query="", fragment="").geturl()


def validate_task(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("task must be a non-empty string")
    return value.strip()


def validate_seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**31 - 1:
        raise ValueError("seed must be an integer within 0..2147483647")
    return value


def validate_jpeg(image_bytes: bytes, max_image_bytes: int) -> bytes:
    if not isinstance(image_bytes, bytes):
        image_bytes = bytes(image_bytes)
    if not image_bytes:
        raise CardError("invalid_image", "JPEG payload is empty")
    if len(image_bytes) > max_image_bytes:
        raise CardError(
            "invalid_image",
            f"JPEG payload exceeds max_image_bytes={max_image_bytes}",
        )
    if len(image_bytes) < 4 or not image_bytes.startswith(b"\xff\xd8") or not image_bytes.endswith(b"\xff\xd9"):
        raise CardError("invalid_image", "payload is not a complete JPEG byte stream")
    return image_bytes


def parse_state_message(text: str) -> list[float]:
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise CardError("invalid_state", "state message must be valid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("joints"), list):
        raise CardError("invalid_state", "state message must contain a joints array")

    by_index: dict[int, float] = {}
    for position, item in enumerate(payload["joints"]):
        if not isinstance(item, dict) or "idx" not in item or "q" not in item:
            raise CardError("invalid_state", f"joints[{position}] must contain idx and q")
        idx = item["idx"]
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise CardError("invalid_state", f"joints[{position}].idx must be an integer")
        if idx in by_index:
            raise CardError("invalid_state", f"duplicate joint index: {idx}")
        try:
            q = _finite_number(item["q"], f"joints[{position}].q")
        except CardError as exc:
            raise CardError("invalid_state", exc.message) from exc
        by_index[idx] = q

    missing = [idx for idx in range(STATE_DIM) if idx not in by_index]
    if missing:
        raise CardError("invalid_state", f"state is missing joint indices: {missing}")
    return [by_index[idx] for idx in range(STATE_DIM)]


def build_policy_payload(snapshot: ObservationSnapshot, task: str, seed: int) -> dict[str, Any]:
    return {
        "image_base64": base64.b64encode(snapshot.image_bytes).decode("ascii"),
        "state": list(snapshot.state),
        "task": validate_task(task),
        "seed": validate_seed(seed),
    }


def validate_policy_response(payload: Any, expected_seed: int) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CardError("policy_contract_error", "policy response must be a JSON object")
    if payload.get("ok") is not True:
        message = payload.get("message") or payload.get("error") or "policy returned ok != true"
        raise CardError("policy_rejected", str(message))
    if payload.get("action_shape") != ACTION_SHAPE:
        raise CardError("policy_contract_error", f"action_shape must be {ACTION_SHAPE}")
    if payload.get("action_space") != ACTION_SPACE:
        raise CardError("policy_contract_error", f"action_space must be {ACTION_SPACE}")
    if payload.get("fresh_inference_per_request") is not True:
        raise CardError("policy_contract_error", "fresh_inference_per_request must be true")
    if payload.get("num_inference_steps") != NUM_INFERENCE_STEPS:
        raise CardError(
            "policy_contract_error",
            f"num_inference_steps must be {NUM_INFERENCE_STEPS}",
        )
    if payload.get("seed") != expected_seed:
        raise CardError("policy_contract_error", "response seed does not match request seed")

    raw_chunk = payload.get("action_chunk")
    if not isinstance(raw_chunk, list) or len(raw_chunk) != ACTION_STEPS:
        raise CardError("policy_contract_error", f"action_chunk must contain {ACTION_STEPS} rows")
    chunk: list[list[float]] = []
    for row_index, raw_row in enumerate(raw_chunk):
        if not isinstance(raw_row, list) or len(raw_row) != ACTION_DIM:
            raise CardError(
                "policy_contract_error",
                f"action_chunk[{row_index}] must contain {ACTION_DIM} values",
            )
        chunk.append(
            [
                _finite_number(value, f"action_chunk[{row_index}][{column_index}]")
                for column_index, value in enumerate(raw_row)
            ]
        )

    infer_seconds = _finite_number(payload.get("infer_seconds"), "infer_seconds")
    if infer_seconds < 0:
        raise CardError("policy_contract_error", "infer_seconds must be non-negative")
    return {"action_chunk": chunk, "policy_infer_seconds": infer_seconds}


def build_action_proposal(
    *,
    request_id: str,
    created_at: float,
    snapshot: ObservationSnapshot,
    task: str,
    seed: int,
    validated_response: dict[str, Any],
    card_elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "schema": "pi05.g1.action_chunk.v1",
        "request_id": request_id,
        "created_at": created_at,
        "observation": {
            "image_topic": snapshot.image_topic,
            "image_frame_id": snapshot.image_frame_id,
            "image_stamp": snapshot.image_stamp,
            "image_stamp_source": snapshot.image_stamp_source,
            "image_received_at": snapshot.image_received_at,
            "image_age_at_request_s": snapshot.image_age_at_request_s,
            "state_topic": snapshot.state_topic,
            "state_received_at": snapshot.state_received_at,
            "state_age_at_request_s": snapshot.state_age_at_request_s,
            "state": list(snapshot.state),
        },
        "task": task,
        "action_space": ACTION_SPACE,
        "frequency_hz": FREQUENCY_HZ,
        "action_shape": list(ACTION_SHAPE),
        "action_chunk": validated_response["action_chunk"],
        "fresh_inference_per_request": True,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "seed": seed,
        "policy_infer_seconds": validated_response["policy_infer_seconds"],
        "card_elapsed_seconds": card_elapsed_seconds,
        "execution_authorized": False,
    }


class PolicyClient:
    def __init__(self, policy_url: str, timeout_s: float, health_url: str | None = None):
        self.policy_url = validate_http_url(policy_url, "policy_url")
        self.health_url = validate_http_url(
            health_url or derive_health_url(self.policy_url),
            "health_url",
        )
        self.timeout_s = float(timeout_s)
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _request_json(self, request: urllib.request.Request) -> Any:
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise CardError("policy_rejected", f"policy HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            reason = getattr(exc, "reason", exc)
            raise CardError("policy_unreachable", f"policy request failed: {reason}") from exc
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CardError("policy_contract_error", "policy response is not valid JSON") from exc
        return decoded

    def health(self) -> dict[str, Any]:
        decoded = self._request_json(
            urllib.request.Request(self.health_url, method="GET")
        )
        if not isinstance(decoded, dict):
            raise CardError("policy_contract_error", "policy health response must be a JSON object")
        return decoded

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.policy_url,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._request_json(request)
