from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import threading
import time
from typing import Any

from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from .policy import (
    CardError,
    ObservationSnapshot,
    PolicyClient,
    build_action_proposal,
    build_policy_payload,
    parse_state_message,
    validate_jpeg,
    validate_policy_response,
    validate_seed,
    validate_task,
)


_INPUT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)

_OUTPUT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)


class VLAPi05G1Node(Node):
    def __init__(
        self,
        *,
        image_topic: str,
        state_topic: str,
        output_topic: str,
        instance_id: str,
        shared_config: dict[str, Any],
        instance_config: dict[str, Any],
    ):
        normalized_id = re.sub(r"[^a-zA-Z0-9_]", "_", instance_id)
        instance_hash = hashlib.sha256(instance_id.encode("utf-8")).hexdigest()[:8]
        safe_id = f"{normalized_id}_{instance_hash}"
        super().__init__(f"vlapi05g1_{safe_id}")
        self.instance_id = instance_id
        self.image_topic = image_topic
        self.state_topic = state_topic
        self.output_topic = output_topic
        self._shared_config = dict(shared_config)
        self._instance_config = dict(instance_config)

        self._data_lock = threading.Lock()
        self._predict_lock = threading.Lock()
        self._accepting_predictions = True
        self._request_sequence = 0
        self._latest_image: dict[str, Any] | None = None
        self._latest_state: dict[str, Any] | None = None
        self._last_request_id: str | None = None
        self._last_policy_infer_seconds: float | None = None
        self._last_card_elapsed_seconds: float | None = None
        self._last_error: dict[str, str] | None = None

        self._policy = PolicyClient(
            self._shared_config["policy_url"],
            self._shared_config["request_timeout_s"],
        )
        self._worker = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"vlapi05g1_{safe_id}_predict",
        )
        self._publisher = self.create_publisher(String, output_topic, _OUTPUT_QOS)
        self._image_subscription = self.create_subscription(
            CompressedImage,
            image_topic,
            self._on_image,
            _INPUT_QOS,
        )
        self._state_subscription = self.create_subscription(
            String,
            state_topic,
            self._on_state,
            _INPUT_QOS,
        )

    def _record_error(self, error: CardError) -> None:
        with self._data_lock:
            self._last_error = {"error_code": error.code, "message": error.message}
        self.get_logger().warning(f"[{error.code}] {error.message}")

    def _on_image(self, msg: CompressedImage) -> None:
        received_at = time.time()
        received_monotonic = time.monotonic()
        try:
            image_bytes = validate_jpeg(
                bytes(msg.data),
                int(self._shared_config["max_image_bytes"]),
            )
        except CardError as exc:
            self._record_error(exc)
            return

        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1_000_000_000.0
        if stamp > 0:
            stamp_source = "header"
        else:
            stamp = received_at
            stamp_source = "receive_time"
        with self._data_lock:
            self._latest_image = {
                "bytes": image_bytes,
                "frame_id": str(msg.header.frame_id),
                "stamp": stamp,
                "stamp_source": stamp_source,
                "received_at": received_at,
                "received_monotonic": received_monotonic,
            }

    def _on_state(self, msg: String) -> None:
        received_at = time.time()
        received_monotonic = time.monotonic()
        try:
            state = parse_state_message(msg.data)
        except CardError as exc:
            self._record_error(exc)
            return
        with self._data_lock:
            self._latest_state = {
                "state": tuple(state),
                "received_at": received_at,
                "received_monotonic": received_monotonic,
            }

    def update_instance_config(self, config: dict[str, Any]) -> None:
        with self._data_lock:
            self._instance_config = dict(config)

    def _freeze_observation(self, config: dict[str, Any]) -> ObservationSnapshot:
        now_monotonic = time.monotonic()
        with self._data_lock:
            image = dict(self._latest_image) if self._latest_image is not None else None
            state = dict(self._latest_state) if self._latest_state is not None else None
        if image is None or state is None:
            missing = []
            if image is None:
                missing.append("image")
            if state is None:
                missing.append("state")
            raise CardError("observation_missing", f"missing observation inputs: {missing}")

        image_age = max(0.0, now_monotonic - image["received_monotonic"])
        state_age = max(0.0, now_monotonic - state["received_monotonic"])
        if image_age > float(config["max_image_age_s"]):
            raise CardError(
                "observation_stale",
                f"image age {image_age:.6f}s exceeds max_image_age_s={config['max_image_age_s']}",
            )
        if state_age > float(config["max_state_age_s"]):
            raise CardError(
                "observation_stale",
                f"state age {state_age:.6f}s exceeds max_state_age_s={config['max_state_age_s']}",
            )
        return ObservationSnapshot(
            image_bytes=image["bytes"],
            image_topic=self.image_topic,
            image_frame_id=image["frame_id"],
            image_stamp=image["stamp"],
            image_stamp_source=image["stamp_source"],
            image_received_at=image["received_at"],
            image_age_at_request_s=image_age,
            state=state["state"],
            state_topic=self.state_topic,
            state_received_at=state["received_at"],
            state_age_at_request_s=state_age,
        )

    def _next_request_id(self) -> str:
        with self._data_lock:
            self._request_sequence += 1
            return f"vlapi05g1-{self.instance_id}-{self._request_sequence:06d}"

    def predict(self, *, task: Any = None, seed: Any = None) -> dict[str, Any]:
        if not self._accepting_predictions:
            raise CardError("not_running", f"instance {self.instance_id} is stopping")
        if not self._predict_lock.acquire(blocking=False):
            raise CardError("busy", f"instance {self.instance_id} already has an in-flight prediction")
        try:
            if not self._accepting_predictions:
                raise CardError("not_running", f"instance {self.instance_id} is stopping")
            with self._data_lock:
                config = dict(self._instance_config)
            try:
                effective_task = validate_task(config["task"] if task is None else task)
                effective_seed = validate_seed(config["seed"] if seed is None else seed)
            except ValueError as exc:
                raise CardError("invalid_request", str(exc)) from exc
            snapshot = self._freeze_observation(config)
            request_id = self._next_request_id()
            future = self._worker.submit(
                self._run_prediction,
                request_id,
                snapshot,
                effective_task,
                effective_seed,
            )
            return future.result()
        finally:
            self._predict_lock.release()

    def _run_prediction(
        self,
        request_id: str,
        snapshot: ObservationSnapshot,
        task: str,
        seed: int,
    ) -> dict[str, Any]:
        started = time.monotonic()
        try:
            payload = build_policy_payload(snapshot, task, seed)
            response = self._policy.predict(payload)
            validated = validate_policy_response(response, seed)
            elapsed = time.monotonic() - started
            proposal = build_action_proposal(
                request_id=request_id,
                created_at=time.time(),
                snapshot=snapshot,
                task=task,
                seed=seed,
                validated_response=validated,
                card_elapsed_seconds=elapsed,
            )
            output = String()
            output.data = json.dumps(proposal, ensure_ascii=False, separators=(",", ":"))
            self._publisher.publish(output)
            with self._data_lock:
                self._last_request_id = request_id
                self._last_policy_infer_seconds = validated["policy_infer_seconds"]
                self._last_card_elapsed_seconds = elapsed
                self._last_error = None
            return {
                "status": "published",
                "request_id": request_id,
                "output_topic": self.output_topic,
                "policy_infer_seconds": validated["policy_infer_seconds"],
                "card_elapsed_seconds": elapsed,
            }
        except CardError as exc:
            self._record_error(exc)
            raise
        except Exception as exc:
            wrapped = CardError("policy_contract_error", f"unexpected prediction failure: {exc}")
            self._record_error(wrapped)
            raise wrapped from exc

    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._data_lock:
            image = dict(self._latest_image) if self._latest_image is not None else None
            state = dict(self._latest_state) if self._latest_state is not None else None
            config = dict(self._instance_config)
            last_error = dict(self._last_error) if self._last_error is not None else None
            last_request_id = self._last_request_id
            last_policy_infer_seconds = self._last_policy_infer_seconds
            last_card_elapsed_seconds = self._last_card_elapsed_seconds
        return {
            "name": "π0.5 G1 VLA 推理",
            "state": "running" if self._accepting_predictions else "idle",
            "instance_id": self.instance_id,
            "topic_in": [
                {"topic": self.image_topic, "format": "image/jpeg"},
                {"topic": self.state_topic, "format": "data/json"},
            ],
            "topic_out": [{"topic": self.output_topic, "format": "data/json"}],
            "config": config,
            "observation": {
                "image_ready": image is not None,
                "image_age_s": (max(0.0, now - image["received_monotonic"]) if image else None),
                "state_ready": state is not None,
                "state_age_s": (max(0.0, now - state["received_monotonic"]) if state else None),
            },
            "in_flight": self._predict_lock.locked(),
            "last_request_id": last_request_id,
            "last_policy_infer_seconds": last_policy_infer_seconds,
            "last_card_elapsed_seconds": last_card_elapsed_seconds,
            "last_error": last_error,
        }

    def shutdown_card(self) -> None:
        self._accepting_predictions = False
        with self._predict_lock:
            pass
        if self._image_subscription is not None:
            self.destroy_subscription(self._image_subscription)
            self._image_subscription = None
        if self._state_subscription is not None:
            self.destroy_subscription(self._state_subscription)
            self._state_subscription = None
        if self._publisher is not None:
            self.destroy_publisher(self._publisher)
            self._publisher = None
        self._worker.shutdown(wait=True, cancel_futures=True)
        with self._data_lock:
            self._latest_image = None
            self._latest_state = None
