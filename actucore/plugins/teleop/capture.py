"""Driver-owned Capture pairing, presence lease and private RTC signaling.

The JSON wire contract is compatible with the existing Meta/PICO native
Capture client.  Its long-lived credential authenticates only this WSS control
plane; neither stock Core MCP nor Pose frames can use it as authority.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .descriptor import CAPTURE_PROTOCOL, RTC_FRAME_PROTOCOL
from .protocol import ProtocolError, TicketCodec, make_ticket_claims
from .rtc import RtcManager, RtcRequestError
from .runtime import TeleopRuntime

MAX_CAPTURE_MESSAGE_BYTES = 128 * 1024
MAX_SIGNALING_SDP_BYTES = 120 * 1024
MAX_CAPTURE_CA_PEM_BYTES = 32 * 1024
MAX_CAPTURE_CA_BASE64_CHARS = 4 * ((MAX_CAPTURE_CA_PEM_BYTES + 2) // 3)
_UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_APP_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,31}$")


class CaptureError(RuntimeError):
    def __init__(self, code: str, *, status: int = 400):
        super().__init__(code)
        self.code = code
        self.status = status


class _CaptureStatePersistenceError(OSError):
    """Classify whether the atomic state-file commit point was crossed."""

    def __init__(self, *, committed: bool):
        super().__init__("Capture credential state persistence failed")
        self.committed = bool(committed)


@dataclass
class _Pairing:
    code_digest: bytes
    expires_monotonic: float


@dataclass
class _Capture:
    credential_digest: bytes
    client_kind: str
    app_version: str


@dataclass
class CaptureConnection:
    capture_id: str
    connection_id: str
    generation: int = 0
    assignment: dict | None = None
    ready_for_assignment: bool = False
    last_presence_monotonic: float = 0.0
    events: asyncio.Queue[dict] = field(default_factory=asyncio.Queue)


class CaptureManager:
    """Own exactly one paired headset and one active control connection."""

    def __init__(
        self,
        runtime: TeleopRuntime,
        rtc: RtcManager,
        ticket_codec: TicketCodec,
        *,
        pairing_ttl_seconds: int = 60,
        ticket_ttl_seconds: int = 20,
        presence_interval_ms: int = 1000,
        presence_timeout_ms: int = 5000,
        state_file: str | Path | None = None,
        public_wss_url: str | None = None,
        ca_certificate_base64: str | None = None,
        clock=time.monotonic,
        wall_clock=time.time,
    ):
        if not 10 <= pairing_ttl_seconds <= 600:
            raise ValueError("pairing_ttl_seconds must be in [10, 600]")
        if not 1 <= ticket_ttl_seconds <= 30:
            raise ValueError("ticket_ttl_seconds must be in [1, 30]")
        if not 250 <= presence_interval_ms <= 10_000:
            raise ValueError("presence_interval_ms must be in [250, 10000]")
        if not 1000 <= presence_timeout_ms <= 30_000:
            raise ValueError("presence_timeout_ms must be in [1000, 30000]")
        if presence_timeout_ms <= presence_interval_ms:
            raise ValueError("presence_timeout_ms must exceed presence_interval_ms")
        self._runtime = runtime
        self._rtc = rtc
        self._ticket_codec = ticket_codec
        self._pairing_ttl = int(pairing_ttl_seconds)
        self._ticket_ttl = int(ticket_ttl_seconds)
        self.presence_interval_ms = int(presence_interval_ms)
        self.presence_timeout_ms = int(presence_timeout_ms)
        self._state_file = Path(state_file) if state_file else None
        self._public_wss_url = public_wss_url
        self._ca_certificate_base64 = ca_certificate_base64
        self._clock = clock
        self._wall_clock = wall_clock
        self._pairings: dict[str, _Pairing] = {}
        self._captures: dict[str, _Capture] = {}
        self._connection: CaptureConnection | None = None
        self._assignment_generation = 0
        self._assignment_session_id: str | None = None
        self._negotiating = False
        self._lock = asyncio.Lock()
        self._load_state()

    def _load_state(self) -> None:
        path = self._state_file
        if path is None or not path.exists():
            return
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise ValueError("Capture state file must be a regular 0600 file")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Capture state file is unreadable or invalid") from exc
        if not isinstance(payload, dict) or set(payload) != {"schema_version", "capture"}:
            raise ValueError("Capture state file schema is invalid")
        if payload["schema_version"] != 1:
            raise ValueError("Capture state file schema version is unsupported")
        record = payload["capture"]
        if record is None:
            return
        if not isinstance(record, dict) or set(record) != {
            "capture_id", "credential_sha256", "client_kind", "app_version"
        }:
            raise ValueError("Capture state record is invalid")
        capture_id = self._uuid4(record["capture_id"], "capture_id")
        digest_hex = record["credential_sha256"]
        if not isinstance(digest_hex, str) or not re.fullmatch(r"[0-9a-f]{64}", digest_hex):
            raise ValueError("Capture state credential digest is invalid")
        client_kind = record["client_kind"]
        app_version = record["app_version"]
        if client_kind not in {"native_openxr", "webxr"}:
            raise ValueError("Capture state client kind is invalid")
        if not isinstance(app_version, str) or not _APP_VERSION_RE.fullmatch(app_version):
            raise ValueError("Capture state app version is invalid")
        self._captures[capture_id] = _Capture(
            credential_digest=bytes.fromhex(digest_hex),
            client_kind=client_kind,
            app_version=app_version,
        )

    def _persist_state(self) -> None:
        path = self._state_file
        if path is None:
            return
        committed = False
        temporary: Path | None = None
        try:
            if path.exists() and path.is_symlink():
                raise OSError("Capture state file may not be a symlink")
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(path.parent, 0o700)
            record = None
            if self._captures:
                capture_id, capture = next(iter(self._captures.items()))
                record = {
                    "capture_id": capture_id,
                    "credential_sha256": capture.credential_digest.hex(),
                    "client_kind": capture.client_kind,
                    "app_version": capture.app_version,
                }
            encoded = json.dumps(
                {"schema_version": 1, "capture": record},
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(temporary, flags, 0o600)
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            # Atomic replacement is the state transition's commit point.  A
            # later permission or directory-durability failure must not make
            # the in-memory identity roll back behind the new on-disk record.
            committed = True
            os.chmod(path, 0o600)
            directory_flags = os.O_RDONLY
            if hasattr(os, "O_DIRECTORY"):
                directory_flags |= os.O_DIRECTORY
            if hasattr(os, "O_CLOEXEC"):
                directory_flags |= os.O_CLOEXEC
            directory_descriptor = os.open(path.parent, directory_flags)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError as exc:
            raise _CaptureStatePersistenceError(committed=committed) from exc
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except OSError:
                    # Persistence success/failure is determined by the target
                    # commit, never by best-effort cleanup of a 0600 temp file.
                    pass

    @staticmethod
    def _digest(secret: str) -> bytes:
        return hashlib.sha256(secret.encode("utf-8")).digest()

    @staticmethod
    def _uuid4(value: Any, field_name: str) -> str:
        if not isinstance(value, str) or not _UUID4_RE.fullmatch(value):
            raise CaptureError(f"invalid_{field_name}")
        return value

    @staticmethod
    def _exact(value: Any, keys: set[str]) -> dict:
        if not isinstance(value, dict) or set(value) != keys:
            raise CaptureError("capture_message_invalid")
        return value

    @staticmethod
    def _validate_client(message: dict) -> tuple[str, str]:
        if message["capture_protocol"] != CAPTURE_PROTOCOL:
            raise CaptureError("capture_protocol_unsupported")
        if message["frame_protocol"] != RTC_FRAME_PROTOCOL:
            raise CaptureError("frame_protocol_unsupported")
        client_kind = message["client_kind"]
        if client_kind not in {"native_openxr", "webxr"}:
            raise CaptureError("capture_client_unsupported")
        app_version = message["app_version"]
        if not isinstance(app_version, str) or not _APP_VERSION_RE.fullmatch(app_version):
            raise CaptureError("capture_message_invalid")
        return client_kind, app_version

    async def create_pairing(self) -> dict:
        pairing_id = str(uuid.uuid4())
        pairing_code = secrets.token_urlsafe(32)
        now = self._clock()
        async with self._lock:
            if self._captures:
                raise CaptureError('revoke_existing_headset_first', status=409)
            self._pairings.clear()
            self._pairings[pairing_id] = _Pairing(
                code_digest=self._digest(pairing_code),
                expires_monotonic=now + self._pairing_ttl,
            )
        return {
            "pairing_id": pairing_id,
            "pairing_code": pairing_code,
            "expires_in_seconds": self._pairing_ttl,
            "websocket_path": "/ws/teleop-capture",
            "capture_protocol": CAPTURE_PROTOCOL,
            "frame_protocol": RTC_FRAME_PROTOCOL,
            "wss_url": self._public_wss_url,
            "ca_certificate_base64": self._ca_certificate_base64,
            "credential_persistence": (
                "state_file" if self._state_file is not None else "process_memory"
            ),
        }

    async def connect(self, raw: Any) -> tuple[CaptureConnection, dict]:
        if not isinstance(raw, dict):
            raise CaptureError("capture_message_invalid")
        message_type = raw.get("type")
        now = self._clock()
        fresh_credential: str | None = None
        async with self._lock:
            if self._connection is not None:
                raise CaptureError("capture_busy", status=409)
            if message_type == "pair":
                if self._captures:
                    raise CaptureError('revoke_existing_headset_first', status=409)
                message = self._exact(raw, {
                    "type", "pairing_id", "pairing_code", "capture_protocol",
                    "frame_protocol", "client_kind", "app_version",
                })
                try:
                    pairing_id = self._uuid4(message["pairing_id"], "pairing_id")
                except CaptureError as exc:
                    raise CaptureError("capture_pairing_invalid", status=401) from exc
                code = message["pairing_code"]
                if not isinstance(code, str) or not 32 <= len(code) <= 128:
                    raise CaptureError("capture_pairing_invalid", status=401)
                pairing = self._pairings.pop(pairing_id, None)
                if (
                    pairing is None
                    or pairing.expires_monotonic <= now
                    or not hmac.compare_digest(pairing.code_digest, self._digest(code))
                ):
                    raise CaptureError("capture_pairing_invalid", status=401)
                client_kind, app_version = self._validate_client(message)
                capture_id = str(uuid.uuid4())
                fresh_credential = secrets.token_urlsafe(32)
                previous_captures = dict(self._captures)
                self._captures.clear()
                self._captures[capture_id] = _Capture(
                    credential_digest=self._digest(fresh_credential),
                    client_kind=client_kind,
                    app_version=app_version,
                )
                try:
                    self._persist_state()
                except OSError as exc:
                    if not (
                        isinstance(exc, _CaptureStatePersistenceError)
                        and exc.committed
                    ):
                        self._captures = previous_captures
                    raise CaptureError("capture_state_unavailable", status=503) from exc
            elif message_type == "credential":
                message = self._exact(raw, {
                    "type", "capture_id", "capture_credential", "capture_protocol",
                    "frame_protocol", "client_kind", "app_version",
                })
                try:
                    capture_id = self._uuid4(message["capture_id"], "capture_id")
                except CaptureError as exc:
                    raise CaptureError("capture_credential_invalid", status=401) from exc
                credential = message["capture_credential"]
                if not isinstance(credential, str) or not 32 <= len(credential) <= 128:
                    raise CaptureError("capture_credential_invalid", status=401)
                client_kind, app_version = self._validate_client(message)
                capture = self._captures.get(capture_id)
                if (
                    capture is None
                    or capture.client_kind != client_kind
                    or not hmac.compare_digest(
                        capture.credential_digest,
                        self._digest(credential),
                    )
                ):
                    raise CaptureError("capture_credential_invalid", status=401)
                if capture.app_version != app_version:
                    previous_app_version = capture.app_version
                    capture.app_version = app_version
                    try:
                        self._persist_state()
                    except OSError as exc:
                        if not (
                            isinstance(exc, _CaptureStatePersistenceError)
                            and exc.committed
                        ):
                            capture.app_version = previous_app_version
                        raise CaptureError("capture_state_unavailable", status=503) from exc
            else:
                raise CaptureError("capture_message_invalid")

            connection = CaptureConnection(
                capture_id=capture_id,
                connection_id=str(uuid.uuid4()),
                last_presence_monotonic=now,
            )
            # Authentication alone never arms the motion lease.  The native
            # client must report focused `xr_standby` first.
            self._connection = connection
            acknowledgement = {
                "type": "paired" if fresh_credential is not None else "connected",
                "capture_id": capture_id,
                "capture_protocol": CAPTURE_PROTOCOL,
                "frame_protocol": RTC_FRAME_PROTOCOL,
                "presence_interval_ms": self.presence_interval_ms,
                "presence_timeout_ms": self.presence_timeout_ms,
            }
            if fresh_credential is not None:
                acknowledgement["capture_credential"] = fresh_credential
            return connection, acknowledgement

    def _new_assignment(self, binding: dict) -> dict:
        session_id = binding["session_id"]
        if session_id != self._assignment_session_id:
            self._assignment_session_id = session_id
            self._assignment_generation = 0
        self._assignment_generation += 1
        now = float(self._wall_clock())
        capabilities = copy.deepcopy(self._runtime.capabilities)
        effectors = list(capabilities.get("effectors", []))
        return {
            "id": str(uuid.uuid4()),
            "generation": self._assignment_generation,
            "session_id": session_id,
            "mode": self._runtime.mode,
            "profile_id": self._runtime.profile_id,
            "capability_digest": self._runtime.capability_digest,
            "capabilities": capabilities,
            "effectors": effectors,
            "state": "issued",
            "created_at": now,
            "updated_at": now,
            "failure_code": None,
        }

    async def _bind_and_assign_locked(self, connection: CaptureConnection) -> bool:
        try:
            binding, generation = await asyncio.to_thread(
                self._runtime.bind_capture,
                connection.capture_id,
            )
        except ProtocolError as exc:
            # An expired motion lease waits for operator resume just like an
            # inactive session; it must not tear down the paired WSS link.
            if exc.code in {"session_inactive", "session_expired"}:
                return False
            raise CaptureError(exc.code, status=409) from exc
        if self._connection is not connection or not connection.ready_for_assignment:
            self._runtime.capture_hold(connection.capture_id, generation, "capture_not_ready")
            return False
        connection.generation = generation
        connection.assignment = self._new_assignment(binding)
        await connection.events.put({
            "type": "assignment",
            "assignment": connection.assignment,
        })
        return True

    async def issue_assignment_if_connected(self) -> None:
        async with self._lock:
            connection = self._connection
            if (
                connection is None
                or connection.assignment is not None
                or not connection.ready_for_assignment
            ):
                return
            await self._bind_and_assign_locked(connection)

    async def revoke_assignment(self, reason: str) -> None:
        async with self._lock:
            connection = self._connection
            if connection is None or connection.assignment is None:
                return
            assignment_id = connection.assignment["id"]
            connection.assignment = None
            connection.generation = 0
            await connection.events.put({
                "type": "assignment_revoked",
                "assignment_id": assignment_id,
                "reason": reason,
            })

    async def revoke_headset(self) -> dict:
        async with self._lock:
            connection = self._connection
            capture_id = connection.capture_id if connection else next(iter(self._captures), None)
            had_capture = bool(self._captures)
            previous_captures = dict(self._captures)
            self._captures.clear()
            self._pairings.clear()
            if connection is not None:
                self.disconnect_immediate(connection)
            committed_failure: OSError | None = None
            try:
                self._persist_state()
            except OSError as exc:
                if not (
                    isinstance(exc, _CaptureStatePersistenceError)
                    and exc.committed
                ):
                    self._captures = previous_captures
                    raise CaptureError("capture_state_unavailable", status=503) from exc
                committed_failure = exc
            if connection is not None:
                connection.generation = 0
                connection.assignment = None
                await connection.events.put({
                    "type": "capture_revoked",
                    "reason": "operator_revoked",
                })
            if committed_failure is not None:
                raise CaptureError(
                    "capture_state_unavailable",
                    status=503,
                ) from committed_failure
            return {"revoked": had_capture, "capture_id": capture_id}

    async def presence(self, connection: CaptureConnection, raw: Any) -> dict:
        message = self._exact(raw, {"type", "state", "assignment_id"})
        states = {
            "browser_ready", "error", "rtc_connecting", "streaming",
            "xr_ended", "xr_standby",
        }
        if message["type"] != "presence" or message["state"] not in states:
            raise CaptureError("capture_message_invalid")
        state = message["state"]
        assignment_id = message["assignment_id"]
        assignment_bound = state in {"error", "rtc_connecting", "streaming"}
        focus_lost = state in {"browser_ready", "error", "xr_ended"}
        if focus_lost:
            # This task and RTC callbacks share one event loop. Validate and
            # latch HOLD synchronously before acquiring a lock or awaiting a
            # response, close frame, or peer cleanup.
            if self._connection is not connection:
                raise CaptureError("capture_stale", status=409)
            current_id = (
                connection.assignment["id"]
                if connection.assignment is not None
                else None
            )
            if state == "error":
                if current_id is None:
                    if assignment_id is not None:
                        raise CaptureError("capture_message_invalid")
                else:
                    self._uuid4(assignment_id, "assignment_id")
                    if assignment_id != current_id:
                        raise CaptureError("capture_assignment_mismatch", status=409)
            elif assignment_id is not None:
                raise CaptureError("capture_message_invalid")
            connection.ready_for_assignment = False
            hold_generation = connection.generation
            connection.generation = 0
            revoke_event = None
            if connection.assignment is not None:
                revoke_event = {
                    "type": "assignment_revoked",
                    "assignment_id": connection.assignment["id"],
                    "reason": f"capture_{state}",
                }
                connection.assignment = None
            connection.last_presence_monotonic = self._clock()
            if hold_generation:
                self._runtime.capture_hold(
                    connection.capture_id,
                    hold_generation,
                    f"capture_{state}",
                )
            if revoke_event is not None:
                await connection.events.put(revoke_event)
            if hold_generation:
                await self._rtc.close_all()
            return {"type": "presence_ack", "state": state}

        async with self._lock:
            if self._connection is not connection:
                raise CaptureError("capture_stale", status=409)
            current_id = (
                connection.assignment["id"]
                if connection.assignment is not None
                else None
            )
            if assignment_bound:
                self._uuid4(assignment_id, "assignment_id")
                if assignment_id != current_id:
                    raise CaptureError("capture_assignment_mismatch", status=409)
            elif assignment_id is not None:
                raise CaptureError("capture_message_invalid")
            if connection.generation and state in {
                "xr_standby", "rtc_connecting", "streaming"
            }:
                try:
                    await asyncio.to_thread(
                        self._runtime.renew_capture_lease,
                        connection.capture_id,
                        connection.generation,
                    )
                except ProtocolError as exc:
                    raise CaptureError(exc.code, status=409) from exc

            if state == "xr_standby":
                connection.ready_for_assignment = True
                if connection.assignment is None:
                    await self._bind_and_assign_locked(connection)
            connection.last_presence_monotonic = self._clock()
        return {"type": "presence_ack", "state": state}

    async def signaling_offer(self, connection: CaptureConnection, raw: Any) -> dict:
        message = self._exact(raw, {"type", "assignment_id", "offer"})
        if message["type"] != "signaling_offer":
            raise CaptureError("capture_message_invalid")
        assignment_id = self._uuid4(message["assignment_id"], "assignment_id")
        offer = self._exact(message["offer"], {"type", "sdp"})
        sdp = offer["sdp"]
        if (
            offer["type"] != "offer"
            or not isinstance(sdp, str)
            or not sdp
            or len(sdp.encode("utf-8")) > MAX_SIGNALING_SDP_BYTES
            or "\0" in sdp
        ):
            raise CaptureError("invalid_signaling_offer")

        async with self._lock:
            if self._connection is not connection:
                raise CaptureError("capture_stale", status=409)
            if (
                self._clock() - connection.last_presence_monotonic
                > self.presence_timeout_ms / 1000.0
            ):
                raise CaptureError("capture_presence_timeout", status=408)
            if (
                connection.assignment is None
                or connection.assignment["id"] != assignment_id
                or connection.generation == 0
            ):
                raise CaptureError("capture_assignment_mismatch", status=409)
            try:
                binding, generation = await asyncio.to_thread(
                    self._runtime.rtc_authority_snapshot,
                )
                if generation != connection.generation:
                    raise CaptureError("capture_assignment_stale", status=409)
                await asyncio.to_thread(
                    self._runtime.begin_capture_negotiation,
                    connection.capture_id,
                    connection.generation,
                    grace_ms=15_000,
                )
            except ProtocolError as exc:
                raise CaptureError(exc.code, status=409) from exc
            ticket = self._ticket_codec.sign(make_ticket_claims(
                session=binding,
                sdp=sdp,
                ttl_seconds=self._ticket_ttl,
                wall_clock=self._wall_clock,
                jti=secrets.token_urlsafe(18),
            ))
            self._negotiating = True
            try:
                answer = await self._rtc.accept_offer({
                    "type": "offer",
                    "sdp": sdp,
                    "ticket": ticket,
                })
            except RtcRequestError as exc:
                raise CaptureError(exc.code, status=exc.status) from exc
            finally:
                self._negotiating = False
            if (
                self._connection is not connection
                or connection.generation != generation
            ):
                await self._rtc.close_all()
                raise CaptureError("capture_stale", status=409)
            return {
                "type": "signaling_answer",
                "assignment_id": assignment_id,
                "answer": {"type": answer["type"], "sdp": answer["sdp"]},
            }

    def disconnect_immediate(self, connection: CaptureConnection) -> bool:
        """Synchronously fence Capture authority before any socket await."""

        if self._connection is not connection:
            return False
        self._connection = None
        generation = connection.generation
        connection.generation = 0
        if generation:
            self._runtime.mark_capture_disconnected(
                connection.capture_id,
                generation,
            )
        return True

    async def disconnect(self, connection: CaptureConnection) -> None:
        self.disconnect_immediate(connection)
        await self._rtc.close_all()

    def presence_expired(self, connection: CaptureConnection) -> bool:
        return (
            not self._negotiating
            and self._clock() - connection.last_presence_monotonic
            > self.presence_timeout_ms / 1000.0
        )

    async def status(self) -> dict:
        async with self._lock:
            now = self._clock()
            self._pairings = {
                key: value
                for key, value in self._pairings.items()
                if value.expires_monotonic > now
            }
            connection = self._connection
            return {
                "paired_devices": len(self._captures),
                "pending_pairings": len(self._pairings),
                "connected": connection is not None,
                "capture_id": connection.capture_id if connection else None,
                "assignment_id": (
                    connection.assignment["id"]
                    if connection and connection.assignment is not None
                    else None
                ),
                "presence_interval_ms": self.presence_interval_ms,
                "presence_timeout_ms": self.presence_timeout_ms,
                "credential_persistence": (
                    "state_file" if self._state_file is not None else "process_memory"
                ),
                "pairing_survives_restart": self._state_file is not None,
            }


__all__ = [
    "CAPTURE_PROTOCOL",
    "MAX_CAPTURE_MESSAGE_BYTES",
    "MAX_CAPTURE_CA_PEM_BYTES",
    "MAX_CAPTURE_CA_BASE64_CHARS",
    "CaptureConnection",
    "CaptureError",
    "CaptureManager",
]
