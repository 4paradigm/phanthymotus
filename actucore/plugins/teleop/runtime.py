"""Thread-safe ActuCore teleoperation session runtime."""

from __future__ import annotations

import copy
import math
import re
import secrets
import threading
import time
import uuid
from collections import Counter, deque
from collections.abc import Callable, Mapping
from typing import Any

from .descriptor import CAPABILITIES, PROFILE_ID, capability_digest
from .dispatch import FinalDispatchArbiter, OutputAdapter, StopHandle, RECOVERABLE_HOLD_CODES, COLLISION_HOLD_CODES, IK_RETRY_CODES
from .protocol import (
    ProtocolError,
    bind_rtc_frame_v1,
    validate_frame_v1,
)

SESSION_STATES = {
    "idle",
    "prepared_shadow",
    "active_shadow",
    "prepared_live",
    "active_live",
    "hold",
    "paused",
    "released",
    "fault",
}
class TeleopRuntime:
    """Own fencing, watchdogs and the sole ActuCore final-dispatch boundary."""

    def __init__(
        self,
        *,
        lease_timeout_ms: int = 1000,
        pose_timeout_ms: int = 200,
        watchdog_interval_ms: int = 25,
        driver_id: str = "tianyi2-actucore",
        driver_name: str = "Tianyi ActuCore",
        robot_id: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        auto_watchdog: bool = True,
        dispatch_io_timeout_ms: int = 100,
        dispatch_ack_timeout_ms: int = 200,
        mode: str,
        capabilities: dict | None = None,
        motion_interval_ms: int = 0,
        adapter: OutputAdapter,
    ):
        if lease_timeout_ms < 100:
            raise ValueError("lease_timeout_ms must be >= 100")
        if pose_timeout_ms < 20:
            raise ValueError("pose_timeout_ms must be >= 20")
        if watchdog_interval_ms < 5:
            raise ValueError("watchdog_interval_ms must be >= 5")
        if dispatch_io_timeout_ms < 20 or dispatch_io_timeout_ms > 150:
            raise ValueError("dispatch_io_timeout_ms must be in [20, 150]")
        if (
            dispatch_ack_timeout_ms < dispatch_io_timeout_ms
            or dispatch_ack_timeout_ms > 200
        ):
            raise ValueError(
                "dispatch_ack_timeout_ms must be >= dispatch_io_timeout_ms and <= 200"
            )
        if mode not in ("shadow", "live"):
            raise ValueError("mode must be 'shadow' or 'live'")
        expected_hardware = mode == "live"
        if getattr(adapter, "hardware_output", None) is not expected_hardware:
            raise ValueError("adapter hardware_output must match the configured session mode")
        self.boot_id = str(uuid.uuid4())
        self.driver_id = driver_id
        self.driver_name = driver_name
        self.robot_id = robot_id or driver_id
        self.mode = mode
        self.capabilities = copy.deepcopy(capabilities or CAPABILITIES)
        self.profile_id = self.capabilities["profile_id"]
        self.capability_digest = capability_digest(mode, self.capabilities)
        self._auto_collision_recovery = getattr(adapter, "auto_collision_recovery", False) is True
        self._auto_retry_codes = (COLLISION_HOLD_CODES if self._auto_collision_recovery else frozenset())
        if getattr(adapter, "auto_ik_recovery", False) is True:
            self._auto_retry_codes |= IK_RETRY_CODES
        if mode == "shadow" and getattr(adapter, "auto_shadow_feedback_recovery", False) is True:
            self._auto_retry_codes |= {"feedback_unavailable"}
        if mode == "live" and getattr(adapter, "auto_live_transient_recovery", False) is True:
            self._auto_retry_codes |= {"command_timeout", "driver_execution_ack_timeout", "feedback_unavailable", "arms_not_stationary"}
        self.actuation_enabled = expected_hardware
        self._prepared_state = f"prepared_{mode}"
        self._active_state = f"active_{mode}"
        self._prepare_action = f"prepare_{mode}"
        self._lease_timeout = lease_timeout_ms / 1000.0
        self._pose_timeout = pose_timeout_ms / 1000.0
        self._watchdog_interval = watchdog_interval_ms / 1000.0
        self._clock = clock
        self._lock = threading.RLock()
        self._transition_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._watchdog_thread: threading.Thread | None = None
        stop_timeout_ms=getattr(adapter,'stop_confirmation_timeout_ms',dispatch_io_timeout_ms)
        self._dispatch_ack_timeout=dispatch_ack_timeout_ms/1000.
        if stop_timeout_ms>dispatch_io_timeout_ms:
            self._dispatch_ack_timeout=max(self._dispatch_ack_timeout,stop_timeout_ms/1000.+.1)
        self._dispatcher = FinalDispatchArbiter(
            adapter,
            clock=clock,
            io_timeout_ms=dispatch_io_timeout_ms,
            stop_timeout_ms=stop_timeout_ms,
            motion_interval_ms=motion_interval_ms,
        )
        self._closed = False

        self._epoch = 0
        self._generation = 0
        self._state = "idle"
        self._reason: str | None = None
        self._session_id: str | None = None
        self._fence: str | None = None
        self._prepared_at: float | None = None
        self._last_lease_at: float | None = None
        self._lease_armed = False
        self._lease_grace_until: float | None = None
        self._capture_id: str | None = None
        self._authority_valid = False
        self._authority_invalid_reason: str | None = "not_prepared"
        self._last_pose_at: float | None = None
        self._latest_frame: dict | None = None
        self.pose_observer = None  # Non-blocking, diagnostic-only accepted-frame sink.
        self._last_sequence: int | None = None
        self._hold_clutch_sequence = -1
        self._neutral_required = True
        self._rtc_connected = False
        self._channels = {"teleop-control": False, "teleop-pose": False}
        self._counters: Counter[str] = Counter()
        self._frame_times: deque[float] = deque(maxlen=256)
        self._sequence_gaps = 0
        self._rtc_rtt_ms: float | None = None
        if auto_watchdog:
            self.start()

    def start(self) -> None:
        """Start one long-lived watchdog thread; calling twice is harmless."""

        with self._lock:
            if self._closed:
                return
            if self._watchdog_thread and self._watchdog_thread.is_alive():
                return
            self._stop_event.clear()
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop,
                daemon=True,
                name="actucore-teleop-watchdog",
            )
            self._watchdog_thread.start()

    def close(self) -> None:
        self._stop_event.set()
        thread = self._watchdog_thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self._watchdog_interval * 4))
        with self._transition_lock:
            with self._lock:
                if self._closed:
                    return
                self._closed = True
                if self._authority_valid:
                    self._generation += 1
                self._authority_valid = False
                self._authority_invalid_reason = "service_close"
                self._state = "released"
                self._reason = "service_close"
                self._clear_session_locked()
            ack = self._dispatcher.close(self._dispatch_ack_timeout)
            if not ack.ok:
                with self._lock:
                    self._latch_dispatch_fault_locked(ack.code)

    def prepare(
        self,
        requested_mode: str,
        args: dict,
        *,
        lease_armed: bool = True,
    ) -> dict:
        """Install one fenced session at the private runtime boundary.

        Public MCP callers never supply these authority fields.  The low-level
        method remains available to deterministic adapter tests while
        :meth:`prepare_local_session` creates production authority inside the
        Driver.
        """

        if requested_mode != self.mode:
            raise ProtocolError(
                "mode_not_configured",
                f"Driver is configured for {self.mode!r}; {requested_mode!r} prepare is unavailable",
            )
        session_id = self._canonical_uuid(args.get("session_id"), "session_id")
        epoch = args.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
            raise ProtocolError("invalid_epoch", "epoch must be an integer >= 1")
        fence = args.get("fence")
        if not isinstance(fence, str) or not re.fullmatch(r"[A-Za-z0-9_-]{24,128}", fence):
            raise ProtocolError("invalid_fence", "fence must be a URL-safe token of 24-128 characters")
        binding = {
            "boot_id": self.boot_id,
            "session_id": session_id,
            "epoch": epoch,
            "fence": fence,
        }
        with self._transition_lock:
            with self._lock:
                self._require_open_locked()
                if epoch <= self._epoch:
                    raise ProtocolError("stale_epoch", f"epoch must be greater than {self._epoch}")
                # Fence RTC and Frame callbacks before the safe-stop leaves the
                # runtime lock.  The new authority is installed only after the
                # final adapter has acknowledged that stop.
                self._generation += 1
                generation = self._generation
                self._authority_valid = False
                self._authority_invalid_reason = self._prepare_action
                self._state = "hold"
                self._reason = self._prepare_action
                self._clear_session_locked()
                handle = self._dispatcher.begin_prepare(self._prepare_action)
            ack = self._dispatcher.wait_safe(handle, self._dispatch_ack_timeout)
            if not ack.ok:
                with self._lock:
                    self._latch_dispatch_fault_locked(ack.code)
                raise ProtocolError(
                    "dispatch_stop_unconfirmed",
                    "final output adapter did not acknowledge prepare safe-stop",
                )
            with self._lock:
                self._require_open_locked()
                dispatch_ack = self._dispatcher.complete_prepare(handle, binding, generation)
                if not dispatch_ack.ok:
                    self._latch_dispatch_fault_locked(dispatch_ack.code)
                    raise ProtocolError(
                        "dispatch_prepare_failed",
                        "final output adapter could not arm the configured teleoperation session",
                    )
                now = self._clock()
                self._epoch = epoch
                self._state = self._prepared_state
                self._reason = None
                self._session_id = session_id
                self._fence = fence
                self._prepared_at = now
                self._last_lease_at = now if lease_armed else None
                self._lease_armed = bool(lease_armed)
                self._lease_grace_until = None
                self._capture_id = None
                self._authority_valid = True
                self._authority_invalid_reason = None
                self._last_pose_at = None
                self._latest_frame = None
                self._last_sequence = None
                self._hold_clutch_sequence = -1
                self._neutral_required = True
                self._rtc_connected = False
                self._channels = {"teleop-control": False, "teleop-pose": False}
                self._counters["sessions_prepared"] += 1
                return self._public_snapshot_locked(now)

    def prepare_local_session(self) -> dict:
        """Create or return the Driver-owned exclusive teleoperation session."""

        with self._lock:
            self._require_open_locked()
            if self._authority_valid:
                if self._state == "paused":
                    raise ProtocolError(
                        "session_paused",
                        "release the paused session before starting a new one",
                    )
                return self._public_snapshot_locked(self._clock())
            epoch = self._epoch + 1
        return self.prepare(
            self.mode,
            {
                "session_id": str(uuid.uuid4()),
                "epoch": epoch,
                "fence": secrets.token_urlsafe(32),
            },
            # Pairing and putting on the headset may take arbitrarily longer
            # than the motion lease.  Authority starts expiring only when the
            # authenticated, focused Capture binds to this session.
            lease_armed=False,
        )

    def bind_capture(self, capture_id: str) -> tuple[dict, int]:
        """Bind the sole authenticated Capture and arm its movement lease."""

        capture_id = self._canonical_uuid(capture_id, "capture_id")
        now = self._clock()
        with self._lock:
            self._require_open_locked()
            self._require_session_live_locked()
            if self._capture_id is not None and self._capture_id != capture_id:
                raise ProtocolError(
                    "capture_conflict",
                    "another paired Capture already owns the local session",
                )
            if self._lease_armed and self._lease_deadline_passed_locked(now):
                self._invalidate_authority_locked("lease_timeout", target_state="hold")
                raise ProtocolError(
                    "session_expired",
                    "the Capture control lease expired; start a new local session",
                )
            self._capture_id = capture_id
            self._lease_armed = True
            self._last_lease_at = now
            self._lease_grace_until = None
            self._counters["capture_bindings"] += 1
            self._counters["lease_heartbeats"] += 1
            return self._ticket_binding_locked(), self._generation

    def renew_capture_lease(self, capture_id: str, generation: int) -> dict:
        """Renew authority only from the paired Capture control connection."""

        capture_id = self._canonical_uuid(capture_id, "capture_id")
        now = self._clock()
        with self._lock:
            self._require_open_locked()
            self._require_session_live_locked()
            if generation != self._generation:
                raise ProtocolError(
                    "stale_capture_generation",
                    "Capture belongs to a stale session",
                )
            if self._capture_id != capture_id:
                raise ProtocolError(
                    "capture_mismatch",
                    "Capture does not own the local session",
                )
            self._require_authority_valid_locked(now)
            self._last_lease_at = now
            self._lease_grace_until = None
            self._counters["lease_heartbeats"] += 1
            return self._public_snapshot_locked(now)

    def begin_capture_negotiation(
        self,
        capture_id: str,
        generation: int,
        *,
        grace_ms: int = 15_000,
    ) -> dict:
        """Grant ICE/DTLS setup time anchored to the last real presence.

        Repeated offers are not lease heartbeats and therefore cannot slide
        this deadline forward.  A later authenticated Capture presence may
        renew the lease and establish a new bounded negotiation window.
        """

        if grace_ms < 1000 or grace_ms > 30_000:
            raise ProtocolError(
                "invalid_negotiation_grace",
                "negotiation grace must be in [1000, 30000] ms",
            )
        capture_id = self._canonical_uuid(capture_id, "capture_id")
        now = self._clock()
        with self._lock:
            self._require_open_locked()
            self._require_session_live_locked()
            if generation != self._generation or self._capture_id != capture_id:
                raise ProtocolError(
                    "capture_mismatch",
                    "Capture does not own the local session",
                )
            self._require_authority_valid_locked(now)
            if self._last_lease_at is None:
                raise ProtocolError(
                    "capture_lease_missing",
                    "Capture negotiation requires a real presence lease",
                )
            grace_until = self._last_lease_at + grace_ms / 1000.0
            if now > grace_until:
                self._invalidate_authority_locked(
                    "lease_timeout",
                    target_state="hold",
                )
                raise ProtocolError(
                    "session_expired",
                    "Capture control lease expired; start a new local session",
                )
            self._lease_grace_until = grace_until
            self._counters["capture_negotiations"] += 1
            return self._public_snapshot_locked(now)

    def mark_capture_disconnected(self, capture_id: str, generation: int) -> dict:
        """Enter HOLD once without letting a stale socket affect a new session."""

        capture_id = self._canonical_uuid(capture_id, "capture_id")
        with self._transition_lock:
            now = self._clock()
            with self._lock:
                if generation != self._generation or self._capture_id != capture_id:
                    self._counters["stale_capture_callbacks"] += 1
                    return self._public_snapshot_locked(now)
                if not self._authority_valid:
                    return self._public_snapshot_locked(now)
                handle = None
                if self._state in (self._prepared_state, self._active_state):
                    handle = self._enter_hold_locked("capture_disconnected")
                self._counters["capture_disconnects"] += 1
            self._wait_for_transition_stop(handle, "capture disconnect")
            with self._lock:
                return self._public_snapshot_locked(self._clock())

    def capture_hold(self, capture_id: str, generation: int, reason: str) -> dict:
        """Immediately inhibit motion when authenticated Capture loses focus."""

        capture_id = self._canonical_uuid(capture_id, "capture_id")
        if not isinstance(reason, str) or not reason.startswith("capture_"):
            raise ProtocolError("invalid_hold_reason", "Capture HOLD reason is invalid")
        with self._transition_lock:
            now = self._clock()
            with self._lock:
                if generation != self._generation or self._capture_id != capture_id:
                    self._counters["stale_capture_callbacks"] += 1
                    return self._public_snapshot_locked(now)
                if not self._authority_valid:
                    return self._public_snapshot_locked(now)
                handle = None
                if self._state in (self._prepared_state, self._active_state):
                    handle = self._enter_hold_locked(reason)
                    self._counters["capture_focus_losses"] += 1
            self._wait_for_transition_stop(handle, reason)
            with self._lock:
                return self._public_snapshot_locked(self._clock())

    def pause_local(self) -> dict:
        with self._transition_lock:
            now = self._clock()
            with self._lock:
                self._require_open_locked()
                self._require_session_live_locked()
                self._require_authority_valid_locked(now)
                handle = self._enter_hold_locked(
                    "operator_pause",
                    target_state="paused",
                )
            self._wait_for_transition_stop(handle, "pause")
            with self._lock:
                return self._public_snapshot_locked(self._clock())

    def release_local(self, *, reason: str = "operator_release") -> dict:
        if reason not in {
            "operator_release",
            "lifecycle_stop",
            "emergency_stop",
            "capture_revoked",
        }:
            raise ProtocolError("invalid_release_reason", "local release reason is invalid")
        with self._transition_lock:
            with self._lock:
                self._require_open_locked()
                if not self._authority_valid:
                    return self._public_snapshot_locked(self._clock())
                handle = self._invalidate_authority_locked(
                    reason,
                    target_state="released",
                )
                self._counters["sessions_released"] += 1
                if reason == "emergency_stop":
                    self._counters["emergency_stops"] += 1
            self._wait_for_transition_stop(handle, reason)
            with self._lock:
                return self._public_snapshot_locked(self._clock())

    def heartbeat(self, args: dict, *, include_status: bool = True) -> dict:
        now = self._clock()
        with self._lock:
            self._require_open_locked()
            self._require_session_live_locked()
            self._require_identity_locked(args)
            self._require_authority_valid_locked(now)
            self._last_lease_at = now
            self._counters["lease_heartbeats"] += 1
            return self._public_snapshot_locked(now) if include_status else {}

    def pause(self, args: dict) -> dict:
        with self._transition_lock:
            now = self._clock()
            with self._lock:
                self._require_open_locked()
                self._require_session_live_locked()
                self._require_identity_locked(args)
                self._require_authority_valid_locked(now)
                handle = self._enter_hold_locked("operator_pause", target_state="paused")
            self._wait_for_transition_stop(handle, "pause")
            with self._lock:
                return self._public_snapshot_locked(self._clock())

    def soft_stop(self, args: dict) -> dict:
        with self._transition_lock:
            now = self._clock()
            with self._lock:
                self._require_open_locked()
                self._require_session_live_locked()
                self._require_identity_locked(args)
                self._require_authority_valid_locked(now)
                if self._state == "paused":
                    raise ProtocolError(
                        "session_paused",
                        f"a paused session requires a new {self._prepare_action}",
                    )
                handle = self._enter_hold_locked("soft_stop")
                self._counters["soft_stops"] += 1
            self._wait_for_transition_stop(handle, "soft_stop")
            with self._lock:
                return self._public_snapshot_locked(self._clock())

    def release(self, args: dict | None = None, *, lifecycle: bool = False) -> dict:
        with self._transition_lock:
            with self._lock:
                self._require_open_locked()
                if not lifecycle:
                    self._require_session_live_locked()
                    self._require_identity_locked(args or {})
                reason = "lifecycle_stop" if lifecycle else "operator_release"
                handle = self._invalidate_authority_locked(reason, target_state="released")
                self._counters["sessions_released"] += 1
            self._wait_for_transition_stop(handle, "release")
            with self._lock:
                return self._public_snapshot_locked(self._clock())

    def submit_frame(
        self,
        raw_frame: Any,
        *,
        source: str,
        rtc_generation: int | None = None,
        include_status: bool = True,
    ) -> dict:
        received_at = self._clock()
        with self._lock:
            self._counters["frames_received"] += 1
        try:
            frame = validate_frame_v1(raw_frame, expected_mode=self.mode)
            controls=[frame['controllers'][side]['buttons'] for side in ('left','right')]
            if any(len(b)!=2 for b in controls):raise ProtocolError('trigger_binding_missing','trigger/squeeze required')
            if frame['deadman'] and any(b[1]<0.75 for b in controls):
                raise ProtocolError('dual_grip_required','both squeeze controls must be held')
        except ProtocolError as exc:
            with self._lock:
                self._counters["frames_rejected"] += 1
                self._counters[f"reject_{exc.code}"] += 1
            raise

        now = self._clock()
        with self._lock:
            try:
                self._require_open_locked()
                self._require_session_live_locked()
                self._require_identity_locked(frame)
                self._require_authority_valid_locked(now)
                if source == "rtc":
                    if rtc_generation != self._generation:
                        raise ProtocolError(
                            "stale_rtc_generation",
                            "RTC Pose belongs to a stale session generation",
                        )
                    if not self._rtc_connected or not all(self._channels.values()):
                        if self._state in (self._prepared_state, self._active_state):
                            self._enter_hold_locked("rtc_not_ready")
                        raise ProtocolError(
                            "rtc_not_ready",
                            "both teleop-control and teleop-pose must be open for RTC Pose",
                        )
                if self._state == "paused":
                    raise ProtocolError("session_inactive", f"cannot accept frames in state {self._state}")
                sequence = frame["sequence"]
                if self._last_sequence is not None and sequence <= self._last_sequence:
                    raise ProtocolError(
                        "sequence_not_increasing",
                        f"sequence must be greater than {self._last_sequence}",
                    )
            except ProtocolError as exc:
                self._counters["frames_rejected"] += 1
                self._counters[f"reject_{exc.code}"] += 1
                raise

            if self._last_sequence is not None and sequence > self._last_sequence + 1:
                self._sequence_gaps += sequence - self._last_sequence - 1
            self._latest_frame = frame
            self._last_sequence = sequence
            self._last_pose_at = now
            self._counters["frames_accepted"] += 1
            if self.pose_observer is not None:
                try:self.pose_observer(frame, received_at)
                except Exception:self._counters["recording_errors"] += 1
            self._counters[f"frames_from_{source}"] += 1
            self._frame_times.append(now)

            tracking_ok = all(frame["tracking"].values())
            dispatch_frame = False
            allow_reclutch = False
            if self._state == "hold" and self._reason == "soft_stop":
                # Core has no resume/rearm action.  An operator soft-stop is a
                # latch: only release followed by a newer prepare can re-arm.
                self._counters["frames_held_after_soft_stop"] += 1
            elif not tracking_ok:
                self._enter_hold_locked("tracking_lost")
            elif self._neutral_required:
                # An inhibited deadman does not prove a physical grip release.
                if (frame["deadman"] or all(b[1]>=0.75 for b in controls)
                        or any(b[0]>0.05 for b in controls)):
                    self._counters["frames_held_neutral_required"] += 1
                    if self._state == self._active_state:
                        self._enter_hold_locked("neutral_required")
                else:
                    self._neutral_required = False
                    self._hold_clutch_sequence = max(
                        self._hold_clutch_sequence,
                        int(frame["clutch_sequence"]),
                    )
                    self._counters["neutral_observations"] += 1
            elif not frame["deadman"]:
                self._enter_hold_locked("deadman_released")
                # This tracked frame can itself prove release; repeated neutral
                # frames must not relatch the requirement on alternating frames.
                self._neutral_required = (all(b[1]>=0.75 for b in controls)
                                          or any(b[0]>0.05 for b in controls))
            elif self._state == self._prepared_state:
                if frame["clutch_sequence"] > self._hold_clutch_sequence:
                    self._state = self._active_state
                    self._reason = None
                    dispatch_frame = True
                    allow_reclutch = True
                else:
                    self._counters["frames_held_without_reclutch"] += 1
            elif self._state == "hold":
                automatic_retry = (self._reason in self._auto_retry_codes
                                   and self._dispatcher.safety_status().get("stop_acknowledged") is True)
                if automatic_retry or frame["clutch_sequence"] > self._hold_clutch_sequence:
                    counter = ("feedback_rechecks" if self._reason == "feedback_unavailable" else
                               "collision_rechecks" if self._reason in COLLISION_HOLD_CODES else "ik_rechecks")
                    self._state = self._active_state
                    self._reason = None
                    self._counters[counter if automatic_retry else "explicit_reclutches"] += 1
                    dispatch_frame = True
                    allow_reclutch = True
                else:
                    self._counters["frames_held_without_reclutch"] += 1
            elif self._state == self._active_state:
                dispatch_frame = True

            if dispatch_frame:
                lease_started = (
                    self._last_lease_at if self._last_lease_at is not None else now
                )
                dispatch_ack = self._dispatcher.publish_latest(
                    frame,
                    session_generation=self._generation,
                    expires_monotonic=min(
                        now + self._pose_timeout,
                        lease_started + self._lease_timeout,
                    ),
                    allow_reclutch=allow_reclutch,
                    received_monotonic=received_at,
                )
                if not dispatch_ack.ok:
                    self._counters["dispatch_rejections"] += 1
                    self._counters[f"dispatch_reject_{dispatch_ack.code}"] += 1
                    dispatch_state = self._dispatcher.snapshot()["state"]
                    if dispatch_ack.code == "intent_expired" or (
                        dispatch_ack.code == "motion_inhibited"
                        and dispatch_state == "safe_reclutch_required"
                    ):
                        self._enter_hold_locked("pose_timeout")
                    else:
                        self._dispatcher.trip(
                            "dispatch_fault",
                            target_state="fault_latched",
                            retain_authority=False,
                        )
                        self._latch_dispatch_fault_locked(dispatch_ack.code)
                        raise ProtocolError(
                            "dispatch_fault",
                            "final output authority check failed; a new Driver process is required",
                        )

            result = self._public_snapshot_locked(now) if include_status else {}
            result["accepted_sequence"] = sequence
            return result

    def mark_channel(self, generation: int, label: str, opened: bool) -> dict:
        now = self._clock()
        with self._lock:
            if generation != self._generation:
                self._counters["stale_rtc_callbacks"] += 1
                return self._public_snapshot_locked(now)
            if self._state in ("idle", "released"):
                self._counters["terminal_rtc_callbacks_ignored"] += 1
                return self._public_snapshot_locked(now)
            if not self._authority_valid or self._lease_deadline_passed_locked(now):
                if self._authority_valid:
                    self._invalidate_authority_locked("lease_timeout", target_state="hold")
                self._counters["expired_rtc_callbacks_ignored"] += 1
                return self._public_snapshot_locked(now)
            was_connected = self._rtc_connected
            if label in self._channels:
                self._channels[label] = bool(opened)
            self._rtc_connected = all(self._channels.values())
            if self._rtc_connected and not was_connected:
                self._neutral_required = True
                self._counters["rtc_neutral_resets"] += 1
            if not opened and self._state in (self._prepared_state, self._active_state):
                self._enter_hold_locked("rtc_disconnected")
            return self._public_snapshot_locked(now)

    def submit_rtc_frame(
        self,
        raw_frame: Any,
        *,
        authority: Mapping[str, Any],
        rtc_generation: int,
        include_status: bool = True,
    ) -> dict:
        """Bind one public RTC frame before entering the common safety path."""

        try:
            frame = bind_rtc_frame_v1(
                raw_frame,
                authority=authority,
                expected_mode=self.mode,
            )
        except ProtocolError as exc:
            with self._lock:
                self._counters["frames_rejected"] += 1
                self._counters[f"reject_{exc.code}"] += 1
            raise
        return self.submit_frame(
            frame,
            source="rtc",
            rtc_generation=rtc_generation,
            include_status=include_status,
        )

    def mark_rtc_disconnected(self, generation: int, reason: str = "rtc_disconnected") -> dict:
        now = self._clock()
        with self._lock:
            if generation != self._generation:
                self._counters["stale_rtc_callbacks"] += 1
                return self._public_snapshot_locked(now)
            if self._state in ("idle", "released"):
                self._counters["terminal_rtc_callbacks_ignored"] += 1
                return self._public_snapshot_locked(now)
            if not self._authority_valid or self._lease_deadline_passed_locked(now):
                if self._authority_valid:
                    self._invalidate_authority_locked("lease_timeout", target_state="hold")
                self._counters["expired_rtc_callbacks_ignored"] += 1
                return self._public_snapshot_locked(now)
            self._rtc_connected = False
            self._channels = {"teleop-control": False, "teleop-pose": False}
            if self._state in (self._prepared_state, self._active_state):
                self._enter_hold_locked(reason)
            self._counters["rtc_disconnects"] += 1
            return self._public_snapshot_locked(now)

    def record_protocol_error(self, code: str) -> None:
        with self._lock:
            self._counters["protocol_errors"] += 1
            self._counters[f"protocol_{code}"] += 1

    def record_rtc_rtt(self, milliseconds: float | None) -> None:
        with self._lock:
            if milliseconds is None:
                self._rtc_rtt_ms = None
                return
            value = float(milliseconds)
            if value < 0.0 or value > 60_000.0:
                raise ValueError("RTC RTT must be in [0, 60000] milliseconds")
            self._rtc_rtt_ms = round(value, 3)

    def status(self) -> dict:
        with self._lock:
            now = self._clock()
            if self._authority_valid and self._lease_deadline_passed_locked(now):
                self._invalidate_authority_locked("lease_timeout", target_state="hold")
            return self._public_snapshot_locked(now)

    def control_status(self) -> dict:
        """Current safety state without copying visualization/diagnostic histories."""
        with self._lock:
            now=self._clock()
            if self._authority_valid and self._lease_deadline_passed_locked(now):
                self._invalidate_authority_locked('lease_timeout',target_state='hold')
            self._sync_dispatch_fault_locked()
            return {'state':self._state,'reason':self._reason,
                    'dispatch':self._dispatcher.safety_status()}

    def run_ik_diagnostic(self, operation: Callable[[], Any]) -> Any:
        """Run zero-output IK work without racing a session transition.

        Session prepare/release and this guard share ``_transition_lock``.  The
        runtime state is checked before the callback starts, then the callback
        may take the adapter's shared IK lock without retaining ``_lock``.  This
        fixes both the inactive-check TOCTOU window and a runtime-lock/IK-lock
        inversion with the real-time adapter.
        """

        if not callable(operation):
            raise TypeError("IK diagnostic operation must be callable")
        with self._transition_lock:
            with self._lock:
                self._require_open_locked()
                if self._authority_valid or self._state not in ("idle", "released"):
                    raise ProtocolError(
                        "teleop_session_active",
                        "teleop_ik requires an idle or released teleoperation session",
                    )
            return operation()

    def ticket_binding(self) -> dict:
        with self._lock:
            self._require_open_locked()
            self._require_session_live_locked()
            self._require_authority_valid_locked(self._clock())
            return self._ticket_binding_locked()

    def rtc_authority_snapshot(self) -> tuple[dict, int]:
        """Capture one immutable RTC authority tuple under the runtime lock."""

        with self._lock:
            self._require_open_locked()
            self._require_session_live_locked()
            self._require_authority_valid_locked(self._clock())
            return self._ticket_binding_locked(), self._generation

    def session_generation(self) -> int:
        """Return the opaque internal generation used only to fence RTC callbacks."""

        with self._lock:
            return self._generation

    def generation_matches(self, generation: int) -> bool:
        with self._lock:
            self._sync_dispatch_fault_locked()
            if generation != self._generation:
                return False
            if not self._authority_valid:
                return False
            if self._lease_deadline_passed_locked(self._clock()):
                self._invalidate_authority_locked("lease_timeout", target_state="hold")
                return False
            return True

    def watchdog_tick(self, *, include_status: bool = True) -> dict:
        now = self._clock()
        with self._lock:
            self._sync_dispatch_fault_locked()
            if self._authority_valid and self._state in (
                self._prepared_state, self._active_state, "hold", "paused"
            ):
                if self._lease_deadline_passed_locked(now):
                    self._invalidate_authority_locked("lease_timeout", target_state="hold")
                elif (
                    self._state == self._active_state
                    and self._last_pose_at is not None
                    and now - self._last_pose_at > self._pose_timeout
                ):
                    self._enter_hold_locked("pose_timeout")
                    self._counters["pose_timeouts"] += 1
            return self._public_snapshot_locked(now) if include_status else {}

    def _watchdog_loop(self) -> None:
        while not self._stop_event.wait(self._watchdog_interval):
            self.watchdog_tick(include_status=False)

    def _enter_hold_locked(
        self,
        reason: str,
        *,
        target_state: str = "hold",
    ) -> StopHandle | None:
        changed = self._state != target_state or self._reason != reason
        self._state = target_state
        self._reason = reason
        self._neutral_required = True
        if self._latest_frame is not None:
            self._hold_clutch_sequence = max(
                self._hold_clutch_sequence,
                int(self._latest_frame["clutch_sequence"]),
            )
        if not changed:
            return None
        dispatch_state = (
            "safe_latched"
            if target_state == "paused" or reason == "soft_stop"
            else "safe_reclutch_required"
        )
        return self._dispatcher.trip(
            reason,
            target_state=dispatch_state,
            retain_authority=True,
        )

    def _lease_deadline_passed_locked(self, now: float) -> bool:
        if self._lease_grace_until is not None and now <= self._lease_grace_until:
            return False
        return self._lease_armed and (
            self._last_lease_at is None
            or now - self._last_lease_at > self._lease_timeout
        )

    def _invalidate_authority_locked(
        self,
        reason: str,
        *,
        target_state: str,
    ) -> StopHandle:
        was_valid = self._authority_valid
        self._authority_valid = False
        self._authority_invalid_reason = reason
        if was_valid:
            self._generation += 1
        self._state = target_state
        self._reason = reason
        self._clear_session_locked()
        handle = self._dispatcher.trip(
            reason,
            target_state="safe_revoked",
            retain_authority=False,
        )
        if was_valid and reason == "lease_timeout":
            self._counters["lease_timeouts"] += 1
        return handle

    def _clear_session_locked(self) -> None:
        self._session_id = None
        self._fence = None
        self._prepared_at = None
        self._last_lease_at = None
        self._lease_armed = False
        self._lease_grace_until = None
        self._capture_id = None
        self._last_pose_at = None
        self._latest_frame = None
        self._last_sequence = None
        self._hold_clutch_sequence = -1
        self._neutral_required = True
        self._frame_times.clear()
        self._sequence_gaps = 0
        self._rtc_connected = False
        self._channels = {"teleop-control": False, "teleop-pose": False}

    def _wait_for_transition_stop(
        self,
        handle: StopHandle | None,
        operation: str,
    ) -> None:
        if handle is None:
            return
        ack = self._dispatcher.wait_safe(handle, self._dispatch_ack_timeout)
        if ack.ok:
            return
        with self._lock:
            self._latch_dispatch_fault_locked(ack.code)
        raise ProtocolError(
            "dispatch_stop_unconfirmed",
            f"final output adapter did not acknowledge {operation} safe-stop",
        )

    def _latch_dispatch_fault_locked(self, code: str) -> None:
        if self._authority_valid:
            self._generation += 1
        self._authority_valid = False
        self._authority_invalid_reason = "dispatch_fault"
        self._state = "fault"
        self._reason = "dispatch_fault"
        self._clear_session_locked()
        self._counters["dispatch_faults"] += 1
        self._counters[f"dispatch_fault_{code}"] += 1

    def _sync_dispatch_fault_locked(self) -> None:
        dispatch = self._dispatcher.safety_status()
        code = dispatch.get("fault_code")
        if code is not None and self._authority_invalid_reason != "dispatch_fault":
            self._latch_dispatch_fault_locked(str(code))
            return
        release_reason = dispatch.get("external_release_reason")
        dispatch_state = dispatch.get("state")
        if (
            (
                release_reason in {"command_timeout", "intent_expired"}
                or dispatch_state == "safe_reclutch_required"
            )
            and self._authority_valid
            and self._state == self._active_state
        ):
            self._state = "hold"
            self._reason = (
                str(release_reason)
                if release_reason in {"command_timeout", "intent_expired"} | RECOVERABLE_HOLD_CODES
                else "pose_timeout"
            )
            self._neutral_required = self._reason not in self._auto_retry_codes
            if self._latest_frame is not None:
                self._hold_clutch_sequence = max(
                    self._hold_clutch_sequence,
                    int(self._latest_frame["clutch_sequence"]),
                )
            self._counters["hardware_ttl_holds"] += 1
            self._counters[f"dispatch_hold_{self._reason}"] += 1

    def _require_open_locked(self) -> None:
        if self._closed:
            raise ProtocolError("service_closed", "the Driver process is closing")
        self._sync_dispatch_fault_locked()

    def _require_authority_valid_locked(self, now: float) -> None:
        if not self._authority_valid:
            self._raise_invalid_authority_locked()
        if self._lease_deadline_passed_locked(now):
            self._invalidate_authority_locked("lease_timeout", target_state="hold")
            raise ProtocolError(
                "session_expired",
                "Capture control lease expired; start a new local session",
            )

    def _require_session_live_locked(self) -> None:
        if not self._authority_valid:
            self._raise_invalid_authority_locked()

    def _raise_invalid_authority_locked(self) -> None:
        if self._authority_invalid_reason == "lease_timeout":
            raise ProtocolError(
                "session_expired",
                "Capture control lease expired; start a new local session",
            )
        raise ProtocolError(
            "session_inactive",
            f"a new {self._prepare_action} session is required",
        )

    def _require_identity_locked(self, value: dict) -> None:
        required = ("boot_id", "session_id", "epoch", "fence")
        missing = [key for key in required if key not in value]
        if missing:
            raise ProtocolError("missing_identity", f"missing session identity fields: {missing}")
        if value["boot_id"] != self.boot_id:
            raise ProtocolError("boot_mismatch", "boot_id does not match this Driver process")
        if value["session_id"] != self._session_id:
            raise ProtocolError("session_mismatch", "session_id does not match the active session")
        if isinstance(value["epoch"], bool) or value["epoch"] != self._epoch:
            raise ProtocolError("epoch_mismatch", "epoch does not match the active session")
        if not isinstance(value["fence"], str) or not secrets.compare_digest(value["fence"], self._fence or ""):
            raise ProtocolError("fence_mismatch", "fence does not match the active session")

    def _ticket_binding_locked(self) -> dict:
        assert self._session_id is not None and self._fence is not None
        return {
            "boot_id": self.boot_id,
            "session_id": self._session_id,
            "epoch": self._epoch,
            "fence": self._fence,
            "capability_digest": self.capability_digest,
        }

    @staticmethod
    def _canonical_uuid(value: Any, name: str) -> str:
        if not isinstance(value, str):
            raise ProtocolError("invalid_session_id", f"{name} must be a canonical UUID")
        try:
            parsed = uuid.UUID(value)
        except ValueError as exc:
            raise ProtocolError("invalid_session_id", f"{name} must be a canonical UUID") from exc
        if str(parsed) != value.lower():
            raise ProtocolError("invalid_session_id", f"{name} must be a canonical UUID")
        return str(parsed)

    def _public_snapshot_locked(self, now: float) -> dict:
        self._sync_dispatch_fault_locked()
        lease_age = None if self._last_lease_at is None else max(0.0, now - self._last_lease_at)
        pose_age = None if self._last_pose_at is None else max(0.0, now - self._last_pose_at)
        latest = copy.deepcopy(self._latest_frame)
        if latest is not None:
            latest.pop("fence", None)
        dispatch = self._dispatcher.snapshot()
        adapter_snapshot = dispatch.get("adapter", {})
        adapter_diagnostics = (
            adapter_snapshot.get("diagnostics", {})
            if isinstance(adapter_snapshot, dict)
            else {}
        )
        adapter_latency = adapter_diagnostics.get("latency_ms", {})
        dispatch_latency = dispatch.get("latency_ms", {})
        empty_latency = {"last": None, "p50": None, "p95": None, "p99": None, "count": 0}
        latency = {
            "receive_to_admit": copy.deepcopy(
                dispatch_latency.get("receive_to_admit", empty_latency)
            ),
            "mailbox_wait": copy.deepcopy(
                dispatch_latency.get("mailbox_wait", empty_latency)
            ),
            "ik": copy.deepcopy(adapter_latency.get("ik", empty_latency)),
            "adapter_apply": copy.deepcopy(
                adapter_latency.get("adapter_apply", empty_latency)
            ),
            "robot_follow": copy.deepcopy(
                adapter_latency.get("robot_follow", empty_latency)
            ),
        }
        if len(self._frame_times) >= 2:
            span = self._frame_times[-1] - self._frame_times[0]
            frame_rate_hz = 0.0 if span <= 0.0 else (len(self._frame_times) - 1) / span
        else:
            frame_rate_hz = 0.0
        transport = {
            "rtc_rtt_ms": self._rtc_rtt_ms,
            "pose_age_ms": None if pose_age is None else round(pose_age * 1000, 3),
            "frame_rate_hz": round(min(1000.0, max(0.0, frame_rate_hz)), 3),
            "frames_received": int(self._counters.get("frames_received", 0)),
            "frames_rejected": int(self._counters.get("frames_rejected", 0)),
            "sequence_gaps": int(self._sequence_gaps),
            "mailbox_replacements": int(
                dispatch.get("counters", {}).get("mailbox_replacements", 0)
            ),
        }
        output = copy.deepcopy(adapter_snapshot.get("output", {}))
        output_active = output.get("output_active") if self.actuation_enabled else False
        return {
            "driver": self.driver_id,
            "driver_id": self.driver_id,
            "driver_name": self.driver_name,
            "driver_type": "teleop",
            "robot_id": self.robot_id,
            "profile_id": self.profile_id,
            "mode": self.mode,
            "actuation_enabled": self.actuation_enabled,
            "publisher_present": output.get("publisher_present") if self.actuation_enabled else False,
            "output_active": output_active,
            "boot_id": self.boot_id,
            "session_id": self._session_id,
            "epoch": self._epoch,
            "state": self._state,
            "reason": self._reason,
            "authority_valid": self._authority_valid,
            "capability_digest": self.capability_digest,
            "capabilities": copy.deepcopy(self.capabilities),
            "lease": {
                "source": "paired-capture-control-only",
                "timeout_ms": round(self._lease_timeout * 1000),
                "age_ms": None if lease_age is None else round(lease_age * 1000, 3),
                "armed": self._lease_armed,
                "fresh": (
                    self._authority_valid
                    and (
                        not self._lease_armed
                        or (lease_age is not None and lease_age <= self._lease_timeout)
                    )
                ),
                "authority_valid": self._authority_valid,
                "expired_latched": self._authority_invalid_reason == "lease_timeout",
                "negotiation_grace": bool(
                    self._lease_grace_until is not None
                    and now <= self._lease_grace_until
                ),
            },
            "capture": {
                "bound": self._capture_id is not None,
                "capture_id": self._capture_id,
                "renews_lease": True,
            },
            "pose": {
                "timeout_ms": round(self._pose_timeout * 1000),
                "age_ms": None if pose_age is None else round(pose_age * 1000, 3),
                "fresh": pose_age is not None and pose_age <= self._pose_timeout,
                "latest_sequence": self._last_sequence,
                "latest": latest,
            },
            "rtc": {
                "connected": self._rtc_connected,
                "channels": dict(self._channels),
                "renews_lease": False,
            },
            "dispatch": dispatch,
            "diagnostics": {
                "transport": transport,
                "latency_ms": latency,
            },
            "output": output,
            "counters": dict(sorted(self._counters.items())),
        }
