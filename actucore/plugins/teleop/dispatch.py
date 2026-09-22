"""Final-dispatch arbitration shared by G1 shadow and live modes.

The adapter in this package records what a live adapter *would* receive.  It
never imports a robot SDK or publishes a hardware command.  The arbiter is the
executable reference for the safety boundary that later live adapters must
implement: a one-slot motion mailbox, a non-droppable stop path, generation
fencing, final deadline checks, and acknowledged startup/shutdown safety.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import re
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .protocol import MAX_SEQUENCE

_TRACKING_KEYS = frozenset({"head", "left_controller", "right_controller"})
_ACK_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
IK_HOLD_CODES = frozenset({
    "ik_target_unreachable", "ik_not_converged", "ik_timeout", "ik_motion_deadline", "ik_jump",
    "torso_collision", "arm_collision", "workspace_limit", "g1_body_collision",
})
IK_RETRY_CODES = frozenset({"ik_target_unreachable", "ik_not_converged", "ik_timeout"})
COLLISION_HOLD_CODES = frozenset({"torso_collision", "arm_collision", "g1_body_collision"})
RECOVERABLE_HOLD_CODES = IK_HOLD_CODES | {"arms_not_stationary", "base_not_stationary", "feedback_unavailable", "driver_execution_ack_timeout", "command_timeout", "triggers_not_neutral"}


def _tracking_ready(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == _TRACKING_KEYS
        and all(item is True for item in value.values())
    )


def _normalized_ack(value: Any, invalid_code: str) -> AdapterAck:
    if not isinstance(value, AdapterAck) or not isinstance(value.ok, bool):
        return AdapterAck(False, invalid_code)
    if not isinstance(value.code, str) or not _ACK_CODE_RE.fullmatch(value.code):
        return AdapterAck(False, invalid_code)
    return value


@dataclass(frozen=True)
class AdapterAck:
    """Bounded adapter acknowledgement with a stable, non-secret code."""

    ok: bool
    code: str = "ok"


@dataclass(frozen=True)
class MotionIntent:
    """Sanitized motion intent delivered to an output adapter.

    ``frame`` deliberately omits the fence.  Session and dispatch generations
    are internal monotonic fences and are safe to expose to a test adapter.
    """

    session_generation: int
    dispatch_generation: int
    sequence: int
    clutch_sequence: int
    admitted_monotonic: float
    expires_monotonic: float
    frame: Mapping[str, Any]
    allow_reclutch: bool = False
    received_monotonic: float | None = None


@dataclass(frozen=True)
class StopRequest:
    dispatch_generation: int
    reason: str
    requested_monotonic: float
    deadline_monotonic: float


class OutputAdapter(Protocol):
    """Serial adapter contract used by the final-dispatch owner thread.

    Implementations must enforce their deadline in the same critical section as
    the real SDK write. ``apply`` may not leave an unbounded background output;
    any asynchronous downstream work must carry the supplied generation and a
    hardware-enforced TTL. A successful ``safe_stop`` ack means all older
    generations are cancelled, drained, or made incapable of later output.
    Adapter methods must also impose their own downstream I/O timeout: Python
    can detect a stalled call and revoke authority, but cannot interrupt a
    blocking vendor SDK call safely. Adapter methods must not re-enter the
    arbiter; a defensive self-close rejection exists only to avoid deadlock.
    """

    hardware_output: bool

    def startup_safe(self, deadline_monotonic: float) -> AdapterAck: ...

    def apply(self, intent: MotionIntent) -> AdapterAck: ...

    def safe_stop(self, request: StopRequest) -> AdapterAck: ...

    def snapshot(self) -> dict: ...

    def close(self) -> AdapterAck | None: ...


class StopHandle:
    """Waitable result for one non-droppable safety request."""

    def __init__(self, request: StopRequest, *, safety_deadline: float):
        self.request = request
        self.safety_deadline = safety_deadline
        self._event = threading.Event()
        self._ack: AdapterAck | None = None

    def _finish(self, ack: AdapterAck) -> None:
        self._ack = ack
        self._event.set()

    def wait(self, timeout: float) -> AdapterAck:
        if not self._event.wait(timeout):
            return AdapterAck(False, "stop_ack_timeout")
        return self._ack or AdapterAck(False, "stop_ack_missing")

    @property
    def done(self) -> bool:
        return self._event.is_set()


@dataclass(frozen=True)
class _PendingMotion:
    intent: MotionIntent
    authority_digest: str


def _binding_digest(value: Mapping[str, Any]) -> str:
    required = ("boot_id", "session_id", "epoch", "fence")
    if any(key not in value for key in required):
        raise ValueError("authority binding is incomplete")
    encoded = json.dumps(
        {key: value[key] for key in required},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RecordingAdapter:
    """Bounded, thread-safe zero-actuation adapter for visible verification."""

    hardware_output = False

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_records: int = 64,
    ):
        if max_records < 8 or max_records > 4096:
            raise ValueError("max_records must be in [8, 4096]")
        self._clock = clock
        self._lock = threading.Lock()
        self._records: deque[dict] = deque(maxlen=max_records)
        self._current = {"kind": "safe", "reason": "not_started"}
        self._closed = False

    def _record(self, value: dict) -> None:
        record = {"recorded_monotonic": self._clock(), **copy.deepcopy(value)}
        self._records.append(record)
        self._current = copy.deepcopy(value)

    def startup_safe(self, deadline_monotonic: float) -> AdapterAck:
        with self._lock:
            if self._closed:
                return AdapterAck(False, "adapter_closed")
            self._record({
                "kind": "would_stop",
                "reason": "startup_safe",
                "dispatch_generation": 0,
                "deadline_monotonic": deadline_monotonic,
            })
            if self._clock() >= deadline_monotonic:
                return AdapterAck(False, "startup_safe_deadline_missed")
            return AdapterAck(True)

    def apply(self, intent: MotionIntent) -> AdapterAck:
        with self._lock:
            if self._closed:
                return AdapterAck(False, "adapter_closed")
            if self._clock() >= intent.expires_monotonic:
                return AdapterAck(False, "intent_expired_at_adapter")
            # The adapter contract itself refuses unsafe content even if a
            # caller accidentally bypasses the runtime admission layer.
            if intent.frame.get("deadman") is not True:
                return AdapterAck(False, "unsafe_deadman")
            if not _tracking_ready(intent.frame.get("tracking")):
                return AdapterAck(False, "unsafe_tracking")
            self._record({
                "kind": "would_apply",
                "sequence": intent.sequence,
                "clutch_sequence": intent.clutch_sequence,
                "session_generation": intent.session_generation,
                "dispatch_generation": intent.dispatch_generation,
                "base_twist": copy.deepcopy(intent.frame.get("base_twist")),
            })
            return AdapterAck(True)

    def safe_stop(self, request: StopRequest) -> AdapterAck:
        with self._lock:
            if self._closed:
                return AdapterAck(False, "adapter_closed")
            self._record({
                "kind": "would_stop",
                "reason": request.reason,
                "dispatch_generation": request.dispatch_generation,
                "deadline_monotonic": request.deadline_monotonic,
            })
            if self._clock() >= request.deadline_monotonic:
                return AdapterAck(False, "adapter_stop_deadline_missed")
            return AdapterAck(True)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "kind": "recording",
                "closed": self._closed,
                "current": copy.deepcopy(self._current),
                "records": copy.deepcopy(list(self._records)),
            }

    def close(self) -> None:
        with self._lock:
            self._closed = True


class FinalDispatchArbiter:
    """Own all adapter I/O and enforce the last command authority check."""

    _MOTION_STATES = frozenset({"safe_waiting_frame", "motion_eligible"})

    def __init__(
        self,
        adapter: OutputAdapter,
        *,
        clock: Callable[[], float] = time.monotonic,
        safety_clock: Callable[[], float] = time.monotonic,
        io_timeout_ms: int = 250,
        stop_timeout_ms: int | None = None,
        motion_interval_ms: int = 0,
    ):
        if io_timeout_ms < 20 or io_timeout_ms > 5000:
            raise ValueError("io_timeout_ms must be in [20, 5000]")
        self._motion_interval = motion_interval_ms / 1000.0
        self._next_motion = 0.0
        self._adapter = adapter
        self._hardware_output = getattr(adapter, "hardware_output", None)
        self._clock = clock
        self._safety_clock = safety_clock
        self._io_timeout = io_timeout_ms / 1000.0
        stop_timeout_ms=io_timeout_ms if stop_timeout_ms is None else stop_timeout_ms
        if type(stop_timeout_ms) is not int or not 20<=stop_timeout_ms<=5000:
            raise ValueError('stop_timeout_ms must be in [20, 5000]')
        self._stop_timeout=stop_timeout_ms/1000.0
        self._condition = threading.Condition(threading.RLock())
        self._stop_queue: deque[StopHandle] = deque()
        self._mailbox: _PendingMotion | None = None
        self._generation = 0
        self._authority_digest: str | None = None
        self._session_generation: int | None = None
        self._state = "starting"
        self._fault_code: str | None = None
        self._stop_acknowledged = False
        self._last_admitted_sequence: int | None = None
        self._last_applied_sequence: int | None = None
        self._last_decision = "starting"
        self._external_release_generation = 0
        self._external_release_reason: str | None = None
        self._counters: Counter[str] = Counter()
        self._latency_samples: dict[str, deque[float]] = {
            "receive_to_admit": deque(maxlen=256),
            "mailbox_wait": deque(maxlen=256),
        }
        self._shutdown = False
        self._closed = False
        self._close_lock = threading.Lock()
        self._adapter_snapshot_lock = threading.Lock()
        self._close_result: AdapterAck | None = None
        self._adapter_close_ack: AdapterAck | None = None
        self._adapter_snapshot: dict = {
            "kind": "unavailable",
            "reason": "startup_in_progress",
        }
        self._io_inflight_kind: str | None = None
        self._io_deadline_safety: float | None = None
        self._io_phase: str | None = None
        self._io_started_safety: float | None = None
        self._last_io_stall: dict | None = None
        self._startup_done = threading.Event()
        self._startup_decision = threading.Event()
        self._startup_owner_done = threading.Event()
        # Fail-safe default: unless the constructor explicitly hands the
        # adapter to the final-dispatch worker, the startup owner closes it
        # after startup_safe eventually returns.
        self._startup_should_close = True

        started_at = self._clock()
        started_safety = self._safety_clock()
        startup_result: list[AdapterAck] = []

        def run_startup_safe() -> None:
            try:
                startup_result.append(_normalized_ack(
                    adapter.startup_safe(started_at + self._io_timeout),
                    "invalid_startup_ack",
                ))
            except Exception:  # noqa: BLE001 -- normalize adapter faults
                startup_result.append(AdapterAck(False, "startup_safe_exception"))
            finally:
                self._capture_adapter_snapshot_owned()
                self._startup_done.set()
            # The startup call may outlive the constructor timeout.  It keeps
            # exclusive adapter ownership until the constructor chooses either
            # a worker hand-off or cleanup; close/snapshot never race it.
            self._startup_decision.wait()
            try:
                if self._startup_should_close:
                    self._close_adapter_owned()
            finally:
                self._startup_owner_done.set()

        self._startup_thread = threading.Thread(
            target=run_startup_safe,
            daemon=True,
            name="g1-teleop-startup-safe",
        )
        self._startup_thread.start()
        completed = self._startup_done.wait(self._io_timeout)
        elapsed_safety = self._safety_clock() - started_safety
        ack = (
            startup_result[0]
            if completed and startup_result
            else AdapterAck(False, "startup_safe_timeout")
        )
        if ack.ok and elapsed_safety <= self._io_timeout:
            self._state = "safe_unarmed"
            self._stop_acknowledged = True
            self._last_decision = "startup_safe_ack"
            self._counters["startup_safe_acks"] += 1
        else:
            self._state = "fault_latched"
            self._fault_code = "startup_safe_timeout" if ack.ok else ack.code
            self._last_decision = "startup_safe_failed"
            self._counters["adapter_faults"] += 1

        self._worker: threading.Thread | None = None
        if self._fault_code is None:
            self._startup_should_close = False
        self._startup_decision.set()
        if self._fault_code is None:
            # The startup owner performs no adapter calls after hand-off.  This
            # short wait also prevents successful starts from leaving a
            # transient daemon behind in short-lived tests.
            self._startup_owner_done.wait(self._io_timeout)
            self._worker = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="g1-teleop-final-dispatch",
            )
            self._worker.start()

    @property
    def hardware_output(self) -> bool | None:
        """Return the adapter's immutable output declaration for runtime gating."""

        return self._hardware_output

    @property
    def ready(self) -> bool:
        with self._condition:
            self._refresh_io_stall_locked()
            self._poll_adapter_signals_locked()
            return self._fault_code is None and not self._closed

    def begin_prepare(self, reason: str = "prepare") -> StopHandle:
        return self.trip(
            reason,
            target_state="transitioning",
            retain_authority=False,
        )

    def complete_prepare(
        self,
        handle: StopHandle,
        binding: Mapping[str, Any],
        session_generation: int,
    ) -> AdapterAck:
        digest = _binding_digest(binding)
        with self._condition:
            if self._closed:
                return AdapterAck(False, "dispatch_closed")
            if self._fault_code is not None:
                return AdapterAck(False, self._fault_code)
            if (
                handle.request.dispatch_generation != self._generation
                or not handle.done
                or self._state != "transitioning"
            ):
                return AdapterAck(False, "prepare_superseded")
            ack = handle.wait(0)
            if not ack.ok:
                return ack
            self._authority_digest = digest
            self._session_generation = session_generation
            self._state = "safe_waiting_frame"
            self._stop_acknowledged = True
            self._last_admitted_sequence = None
            self._last_applied_sequence = None
            self._external_release_reason = None
            self._last_decision = "prepared_after_stop_ack"
            self._counters["sessions_armed"] += 1
            return AdapterAck(True)

    def publish_latest(
        self,
        frame: Mapping[str, Any],
        *,
        session_generation: int,
        expires_monotonic: float,
        allow_reclutch: bool = False,
        received_monotonic: float | None = None,
    ) -> AdapterAck:
        if frame.get("deadman") is not True:
            return AdapterAck(False, "unsafe_deadman")
        if not _tracking_ready(frame.get("tracking")):
            return AdapterAck(False, "unsafe_tracking")
        admitted = self._clock()
        try:
            digest = _binding_digest(frame)
            sequence = frame["sequence"]
            clutch_sequence = frame["clutch_sequence"]
            if (
                isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence < 0
                or sequence > MAX_SEQUENCE
            ):
                raise ValueError("invalid sequence")
            if (
                isinstance(clutch_sequence, bool)
                or not isinstance(clutch_sequence, int)
                or clutch_sequence < 0
                or clutch_sequence > MAX_SEQUENCE
            ):
                raise ValueError("invalid clutch sequence")
            expiry = float(expires_monotonic)
            if not math.isfinite(expiry):
                raise ValueError("invalid expiry")
            received = admitted if received_monotonic is None else float(received_monotonic)
            if not math.isfinite(received) or received > admitted:
                raise ValueError("invalid receive time")
        except (KeyError, TypeError, ValueError, OverflowError):
            return AdapterAck(False, "invalid_intent")

        sanitized = copy.deepcopy(dict(frame))
        sanitized.pop("fence", None)
        with self._condition:
            self._poll_adapter_signals_locked()
            if self._closed:
                return AdapterAck(False, "dispatch_closed")
            if self._fault_code is not None:
                return AdapterAck(False, self._fault_code)
            if self._session_generation != session_generation:
                return AdapterAck(False, "stale_session_generation")
            if self._authority_digest is None or not hmac.compare_digest(
                digest, self._authority_digest
            ):
                return AdapterAck(False, "authority_mismatch")
            if self._state == "safe_reclutch_required" and allow_reclutch:
                self._state = "motion_eligible"
                self._external_release_reason = None
                self._counters["dispatch_reclutches"] += 1
            elif self._state not in self._MOTION_STATES:
                return AdapterAck(False, "motion_inhibited")
            if expiry <= admitted:
                return AdapterAck(False, "intent_expired")
            if (
                self._last_admitted_sequence is not None
                and sequence <= self._last_admitted_sequence
            ):
                return AdapterAck(False, "sequence_not_increasing")
            if self._mailbox is not None:
                self._counters["mailbox_replacements"] += 1
            intent = MotionIntent(
                session_generation=session_generation,
                dispatch_generation=self._generation,
                sequence=sequence,
                clutch_sequence=clutch_sequence,
                admitted_monotonic=admitted,
                expires_monotonic=expiry,
                frame=sanitized,
                allow_reclutch=bool(allow_reclutch),
                received_monotonic=received,
            )
            self._mailbox = _PendingMotion(intent, digest)
            self._state = "motion_eligible"
            self._last_admitted_sequence = sequence
            self._last_decision = "admitted"
            self._counters["motion_admitted"] += 1
            self._observe_latency_locked("receive_to_admit", admitted - received)
            self._condition.notify_all()
            return AdapterAck(True)

    def trip(
        self,
        reason: str,
        *,
        target_state: str,
        retain_authority: bool,
    ) -> StopHandle:
        if not isinstance(reason, str) or not reason:
            raise ValueError("stop reason is required")
        with self._condition:
            self._generation += 1
            now = self._clock()
            request = StopRequest(
                dispatch_generation=self._generation,
                reason=reason,
                requested_monotonic=now,
                deadline_monotonic=now + self._stop_timeout,
            )
            handle = StopHandle(
                request,
                safety_deadline=self._safety_clock() + self._stop_timeout,
            )
            if self._mailbox is not None:
                self._mailbox = None
                self._counters["mailbox_cleared_by_stop"] += 1
            self._state = target_state
            self._stop_acknowledged = False
            self._last_decision = f"stop_requested:{reason}"
            if not retain_authority:
                self._authority_digest = None
                self._session_generation = None
            if self._closed or self._shutdown:
                handle._finish(AdapterAck(False, "dispatch_closed"))
            elif self._worker is None:
                handle._finish(AdapterAck(False, "dispatch_unavailable"))
            else:
                self._stop_queue.append(handle)
                self._counters["stop_requests"] += 1
                self._condition.notify_all()
            return handle

    @staticmethod
    def wait_safe(handle: StopHandle, timeout: float = 0.5) -> AdapterAck:
        return handle.wait(timeout)

    def wait_dispatched(self, sequence: int, timeout: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._last_applied_sequence != sequence:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._fault_code is not None or self._closed:
                    return False
                self._condition.wait(remaining)
            return True

    def snapshot(self) -> dict:
        with self._condition:
            self._refresh_io_stall_locked()
            self._poll_adapter_signals_locked()
            sequence_evidence = (
                {"last_published_sequence": self._last_applied_sequence}
                if self._hardware_output is True
                else {"last_would_apply_sequence": self._last_applied_sequence}
            )
            state = {
                "kind": "hardware" if self._hardware_output is True else "recording",
                "state": self._state,
                "ready": self._fault_code is None and not self._closed,
                "generation": self._generation,
                "mailbox_depth": int(self._mailbox is not None),
                "stop_queue_depth": len(self._stop_queue),
                "last_admitted_sequence": self._last_admitted_sequence,
                **sequence_evidence,
                "last_decision": self._last_decision,
                "stop_acknowledged": self._stop_acknowledged,
                "fault_code": self._fault_code,
                "io_inflight": self._io_inflight_kind,
                "last_io_stall": None if self._last_io_stall is None else dict(self._last_io_stall),
                "counters": dict(sorted(self._counters.items())),
                "latency_ms": {
                    name: self._latency_summary(samples)
                    for name, samples in self._latency_samples.items()
                },
                "external_release_reason": self._external_release_reason,
            }
        with self._adapter_snapshot_lock:
            cached_adapter = self._adapter_snapshot
        # Cache entries are replaced, never mutated.  Copying outside both
        # arbiter locks prevents high-rate status reads from starving the I/O
        # owner while still returning an isolated public value.
        adapter = copy.deepcopy(cached_adapter)
        if self._fault_code is not None and isinstance(adapter, dict):
            output = adapter.get("output")
            if isinstance(output, dict):
                output["state"] = "fault"
                output["arm_sdk_weight"] = (
                    0.0 if self._hardware_output is True else None
                )
                output["fault_reason"] = self._fault_code
        elif self._external_release_reason is not None and isinstance(adapter, dict):
            output = adapter.get("output")
            if isinstance(output, dict):
                output["state"] = (
                    "stopped" if self._stop_acknowledged else "releasing"
                )
                if self._external_release_reason in {"feedback_unavailable", "base_not_stationary", "operator_pause"}:
                    output["state"] = "hold" if self._stop_acknowledged else "hold_unconfirmed"
                elif self._stop_acknowledged:
                    output["arm_sdk_weight"] = 0.0
                output["fault_reason"] = None
        state["adapter"] = adapter
        return state

    def safety_status(self) -> dict:
        """Live safety checks without copying public diagnostic histories."""
        with self._condition:
            self._refresh_io_stall_locked()
            self._poll_adapter_signals_locked()
            return {'state': self._state, 'fault_code': self._fault_code,
                    'external_release_reason': self._external_release_reason,
                    'stop_acknowledged': self._stop_acknowledged}

    def _poll_adapter_signals_locked(self) -> None:
        """Import asynchronous hardware state without invoking adapter I/O.

        These optional probes are read-only and thread-safe. Polling avoids a
        callback from the high-rate publisher into the dispatch lock and the
        corresponding lock-order cycle.
        """

        fault_probe = getattr(self._adapter, "external_fault_code", None)
        try:
            external_fault = fault_probe() if callable(fault_probe) else None
        except Exception:  # noqa: BLE001 -- diagnostics fail closed
            external_fault = "adapter_fault_probe_failed"
        if external_fault is not None:
            code = str(external_fault)[:96]
            if not _ACK_CODE_RE.fullmatch(code):
                code = "adapter_async_fault"
            self._latch_fault_locked(code, decision=f"async_fault:{code}")
            return

        release_probe = getattr(self._adapter, "external_release_signal", None)
        if not callable(release_probe):
            return
        try:
            signal = release_probe()
            generation = signal.get("generation")
            reason = signal.get("reason")
            acknowledged = signal.get("acknowledged")
            if (
                isinstance(generation, bool)
                or not isinstance(generation, int)
                or generation < 0
                or generation > MAX_SEQUENCE
                or (
                    reason is not None
                    and reason not in {"command_timeout", "intent_expired", "feedback_unavailable", "base_not_stationary", "operator_pause"}
                )
                or not isinstance(acknowledged, bool)
            ):
                raise ValueError("invalid external release signal")
        except Exception:  # noqa: BLE001 -- malformed safety signal is terminal
            self._latch_fault_locked(
                "adapter_release_probe_failed",
                decision="async_fault:adapter_release_probe_failed",
            )
            return
        if generation == 0 or reason is None:
            return
        if generation < self._external_release_generation:
            self._latch_fault_locked(
                "adapter_release_generation_regressed",
                decision="async_fault:adapter_release_generation_regressed",
            )
            return
        if generation > self._external_release_generation:
            self._external_release_generation = generation
            self._external_release_reason = reason
            self._counters["autonomous_releases"] += 1
            if self._state in self._MOTION_STATES or (reason in {"feedback_unavailable", "base_not_stationary"} and self._state == "safe_reclutch_required"):
                self._generation += 1
                self._mailbox = None
                self._state = "safe_reclutch_required"
                self._stop_acknowledged = acknowledged
                self._last_decision = (
                    f"safe_stop_ack:{reason}"
                    if acknowledged
                    else f"stop_requested:{reason}"
                )
                self._condition.notify_all()
            else:
                # An explicit Pause/Release/prepare transition is stronger
                # than a concurrently observed hardware TTL release. Keep its
                # state and acknowledgement evidence intact.
                self._counters["autonomous_releases_merged"] += 1
        elif (
            acknowledged
            and not self._stop_acknowledged
            and self._state == "safe_reclutch_required"
        ):
            self._stop_acknowledged = True
            self._state = "safe_reclutch_required"
            self._last_decision = f"safe_stop_ack:{reason}"
            self._counters["autonomous_release_acks"] += 1
            self._condition.notify_all()

    def _latch_fault_locked(self, code: str, *, decision: str) -> None:
        if self._fault_code is not None:
            return
        self._generation += 1
        self._fault_code = code
        self._state = "fault_latched"
        self._authority_digest = None
        self._session_generation = None
        self._mailbox = None
        while self._stop_queue:
            self._stop_queue.popleft()._finish(AdapterAck(False, code))
        self._stop_acknowledged = False
        self._io_inflight_kind = None
        self._io_deadline_safety = None
        self._last_decision = decision[:96]
        self._counters["adapter_faults"] += 1
        self._counters["adapter_async_faults"] += 1
        self._condition.notify_all()

    def _refresh_io_stall_locked(self) -> None:
        if (
            self._io_inflight_kind is None
            or self._io_deadline_safety is None
            or self._fault_code is not None
            or self._safety_clock() < self._io_deadline_safety
        ):
            return
        observed = self._safety_clock()
        self._last_io_stall = {
            "kind": self._io_inflight_kind,
            "phase": self._io_phase,
            "started_monotonic": self._io_started_safety,
            "deadline_monotonic": self._io_deadline_safety,
            "observed_monotonic": observed,
            "overdue_ms": (observed - self._io_deadline_safety) * 1000,
        }
        self._fault_code = "adapter_io_stalled"
        self._state = "fault_latched"
        self._authority_digest = None
        self._session_generation = None
        self._mailbox = None
        self._stop_acknowledged = False
        self._last_decision = f"io_stalled:{self._io_inflight_kind}"
        self._counters["adapter_faults"] += 1
        self._counters["adapter_io_stalls"] += 1
        # The owner may currently be blocked, but the stop remains queued and
        # will be its first operation if the vendor call ever returns.
        self.trip(
            "adapter_io_stalled",
            target_state="fault_latched",
            retain_authority=False,
        )

    def _observe_latency_locked(self, name: str, seconds: float) -> None:
        milliseconds = max(0.0, min(60_000.0, float(seconds) * 1000.0))
        self._latency_samples[name].append(milliseconds)

    @staticmethod
    def _latency_summary(samples: deque[float]) -> dict:
        if not samples:
            return {"last": None, "p50": None, "p95": None, "p99": None, "count": 0}
        ordered = sorted(samples)

        def percentile(value: float) -> float:
            index = min(len(ordered) - 1, max(0, math.ceil(value * len(ordered)) - 1))
            return round(ordered[index], 3)

        return {
            "last": round(samples[-1], 3),
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "count": len(samples),
        }

    def close(self, timeout: float = 0.5) -> AdapterAck:
        if self._worker is threading.current_thread():
            return AdapterAck(False, "owner_cannot_self_close")
        with self._close_lock:
            unavailable_result: AdapterAck | None = None
            with self._condition:
                if self._close_result is not None:
                    return self._close_result
                worker = self._worker
                if worker is None:
                    code = self._fault_code or "dispatch_unavailable"
                    self._closed = True
                    self._shutdown = True
                    self._state = "fault_latched"
                    self._stop_acknowledged = False
                    self._close_result = AdapterAck(False, code)
                    unavailable_result = self._close_result
            if unavailable_result is not None:
                # A timed-out startup call still owns the adapter.  It will
                # close it serially if it ever returns; wait only within the
                # caller's shutdown budget.
                if self._startup_thread is not threading.current_thread():
                    self._startup_owner_done.wait(timeout)
                return unavailable_result

            handle = self.trip(
                "service_close",
                target_state="closing",
                retain_authority=False,
            )
            stop_ack = self.wait_safe(handle, timeout)
            with self._condition:
                self._closed = True
                self._shutdown = True
                self._mailbox = None
                self._state = "closing" if stop_ack.ok else "fault_latched"
                if not stop_ack.ok:
                    self._fault_code = self._fault_code or stop_ack.code
                self._condition.notify_all()
            worker.join(timeout=max(timeout, self._io_timeout * 2))
            with self._condition:
                if worker.is_alive():
                    result = AdapterAck(False, stop_ack.code if not stop_ack.ok else "owner_close_timeout")
                elif not stop_ack.ok:
                    result = stop_ack
                elif self._adapter_close_ack is None:
                    result = AdapterAck(False, "adapter_close_ack_missing")
                else:
                    result = self._adapter_close_ack
                self._close_result = result
                self._state = "closed" if result.ok else "fault_latched"
                if not result.ok:
                    self._fault_code = self._fault_code or result.code
                    self._stop_acknowledged = False
                return result

    def _worker_loop(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._stop_queue and self._mailbox is None and not self._shutdown:
                        self._condition.wait()
                    if self._shutdown and not self._stop_queue:
                        break
                    if self._stop_queue:
                        item: StopHandle | _PendingMotion = self._stop_queue.popleft()
                        is_stop = True
                    else:
                        assert self._mailbox is not None
                        remaining = self._next_motion - self._safety_clock()
                        if remaining > 0:
                            self._condition.wait(remaining)
                            continue
                        self._next_motion = self._safety_clock() + self._motion_interval
                        item = self._mailbox
                        self._mailbox = None
                        is_stop = False
                if is_stop:
                    assert isinstance(item, StopHandle)
                    self._perform_stop(item)
                else:
                    assert isinstance(item, _PendingMotion)
                    self._perform_motion(item)
        finally:
            self._close_adapter_owned()

    def _close_adapter_owned(self) -> None:
        """Close the adapter from its sole current owner thread."""

        try:
            raw_ack = self._adapter.close()
            close_ack = _normalized_ack(
                AdapterAck(True) if raw_ack is None else raw_ack,
                "invalid_close_ack",
            )
        except Exception:  # noqa: BLE001 -- normalize adapter close faults
            close_ack = AdapterAck(False, "adapter_close_exception")
        self._capture_adapter_snapshot_owned()
        with self._condition:
            self._adapter_close_ack = close_ack
            self._condition.notify_all()

    def _capture_adapter_snapshot_owned(self) -> None:
        """Refresh public adapter evidence without crossing the owner boundary."""

        try:
            snapshot = self._adapter.snapshot()
            if not isinstance(snapshot, dict):
                snapshot = {
                    "kind": "unavailable",
                    "reason": "invalid_adapter_snapshot",
                }
            else:
                snapshot = copy.deepcopy(snapshot)
        except Exception:  # noqa: BLE001 -- status must survive adapter diagnostics
            snapshot = {
                "kind": "unavailable",
                "reason": "adapter_snapshot_exception",
            }
        with self._adapter_snapshot_lock:
            self._adapter_snapshot = snapshot

    def _perform_stop(self, handle: StopHandle) -> None:
        request = handle.request
        started_safety = self._safety_clock()
        with self._condition:
            self._io_inflight_kind = "safe_stop"
            self._io_phase = "adapter_call"
            self._io_started_safety = started_safety
            self._io_deadline_safety = handle.safety_deadline
        try:
            ack = _normalized_ack(
                self._adapter.safe_stop(request),
                "invalid_stop_ack",
            )
            with self._condition:
                self._io_phase = "adapter_snapshot"
            self._capture_adapter_snapshot_owned()
        except Exception:  # noqa: BLE001 -- normalize adapter faults
            ack = AdapterAck(False, "adapter_stop_exception")
        finished_safety = self._safety_clock()
        elapsed_safety = finished_safety - started_safety
        if ack.ok:
            if finished_safety >= handle.safety_deadline:
                ack = AdapterAck(False, "adapter_stop_deadline_missed")
            elif elapsed_safety > self._stop_timeout:
                ack = AdapterAck(False, "adapter_stop_timeout")
        with self._condition:
            self._io_inflight_kind = None
            self._io_deadline_safety = None
            handle._finish(ack)
            if ack.ok:
                self._counters["stop_acks"] += 1
                if (
                    request.dispatch_generation == self._generation
                    and self._state != "fault_latched"
                ):
                    self._stop_acknowledged = True
                    self._last_decision = (
                        f"safe_stop_ack:{request.reason}"
                        if self._hardware_output is True
                        else f"would_stop:{request.reason}"
                    )
            elif ack.code == "feedback_wait" and request.dispatch_generation == self._generation and self._state == "safe_reclutch_required":
                self._stop_acknowledged = False
                self._last_decision = "hold_waiting_feedback"
            else:
                self._fault_code = self._fault_code or ack.code
                self._state = "fault_latched"
                self._authority_digest = None
                self._session_generation = None
                self._mailbox = None
                self._stop_acknowledged = False
                self._last_decision = f"stop_failed:{ack.code}"
                self._counters["adapter_faults"] += 1
            self._condition.notify_all()

    def _perform_motion(self, pending: _PendingMotion) -> None:
        intent = pending.intent
        with self._condition:
            self._poll_adapter_signals_locked()
            now = self._clock()
            valid = (
                self._fault_code is None
                and not self._closed
                and self._state == "motion_eligible"
                and intent.dispatch_generation == self._generation
                and intent.session_generation == self._session_generation
                and self._authority_digest is not None
                and hmac.compare_digest(pending.authority_digest, self._authority_digest)
            )
            if not valid:
                if (
                    self._fault_code is None
                    and self._state in self._MOTION_STATES
                ):
                    self._last_decision = "motion_dropped_stale"
                self._counters["motion_dropped_stale"] += 1
                self._condition.notify_all()
                return
            if now >= intent.expires_monotonic:
                self._last_decision = "motion_dropped_expired"
                self._counters["motion_dropped_expired"] += 1
                self.trip(
                    "pose_timeout",
                    target_state="safe_reclutch_required",
                    retain_authority=True,
                )
                self._condition.notify_all()
                return
            # This is the linearization point.  A concurrent trip after this
            # commit is queued behind the bounded apply and cannot acknowledge
            # until its safe-stop has executed.
            self._last_decision = "motion_committed"
            self._observe_latency_locked("mailbox_wait", now - intent.admitted_monotonic)
            remaining = max(0.0, intent.expires_monotonic - now)
            started_safety = self._safety_clock()
            self._io_inflight_kind = "apply"
            self._io_phase = "adapter_call"
            self._io_started_safety = started_safety
            self._io_deadline_safety = started_safety + min(self._io_timeout, remaining)

        try:
            ack = _normalized_ack(
                self._adapter.apply(intent),
                "invalid_apply_ack",
            )
            with self._condition:
                self._io_phase = "adapter_snapshot"
            self._capture_adapter_snapshot_owned()
        except Exception:  # noqa: BLE001 -- normalize adapter faults
            ack = AdapterAck(False, "adapter_apply_exception")
        finished = self._clock()
        finished_safety = self._safety_clock()
        elapsed_safety = finished_safety - started_safety
        if ack.ok:
            if finished >= intent.expires_monotonic:
                ack = AdapterAck(False, "adapter_apply_deadline_missed")
            elif elapsed_safety > self._io_timeout:
                ack = AdapterAck(False, "adapter_apply_timeout")
        with self._condition:
            self._io_inflight_kind = None
            self._io_deadline_safety = None
            if (
                ack.ok
                and self._fault_code is None
                and not self._closed
                and intent.dispatch_generation == self._generation
            ):
                self._last_applied_sequence = intent.sequence
                self._last_decision = (
                    "published" if self._hardware_output is True else "would_apply"
                )
                self._counters["motion_applied"] += 1
            elif ack.ok:
                self._counters["late_adapter_returns"] += 1
            elif ack.code in RECOVERABLE_HOLD_CODES:
                # No target was written. Still require an acknowledged stop
                # and a fresh physical clutch before accepting more motion.
                if (self._fault_code is None and not self._closed
                        and intent.dispatch_generation == self._generation):
                    self._external_release_reason = ack.code
                    self.trip(ack.code, target_state="safe_reclutch_required",
                              retain_authority=True)
            elif ack.code in {"arm_sdk_reclutch_required", "arm_sdk_releasing"}:
                # A hardware TTL can expire between admission and apply. Import
                # that fail-safe release as a recoverable HOLD, not a terminal
                # adapter failure.
                self._poll_adapter_signals_locked()
                if self._state != "safe_reclutch_required":
                    self._latch_fault_locked(
                        "arm_sdk_release_unconfirmed",
                        decision="async_fault:arm_sdk_release_unconfirmed",
                    )
            else:
                already_faulted = self._fault_code is not None
                self._fault_code = self._fault_code or ack.code
                self._state = "fault_latched"
                self._authority_digest = None
                self._session_generation = None
                self._mailbox = None
                self._stop_acknowledged = False
                self._last_decision = f"apply_failed:{self._fault_code}"
                if not already_faulted:
                    self._counters["adapter_faults"] += 1
                # A failed motion write still schedules the independent safety
                # path.  Its acknowledgement cannot clear the fault latch.
                if not already_faulted:
                    self.trip(
                        "adapter_fault",
                        target_state="fault_latched",
                        retain_authority=False,
                    )
                    self._last_decision = f"apply_failed:{self._fault_code}"[:96]
            self._condition.notify_all()
