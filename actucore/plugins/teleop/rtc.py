"""aiortc data-channel transport for the root G1 Driver."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from aiortc import RTCConfiguration, RTCDataChannel, RTCPeerConnection, RTCSessionDescription

from .protocol import ProtocolError, TicketError, TicketVerifier
from .runtime import TeleopRuntime


class RtcRequestError(RuntimeError):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code


class RtcManager:
    def __init__(self, runtime: TeleopRuntime, verifier: TicketVerifier | None):
        self._runtime = runtime
        self._verifier = verifier
        self._peers: set[RTCPeerConnection] = set()
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self._verifier is not None

    async def accept_offer(self, payload: Any) -> dict:
        if self._verifier is None:
            raise RtcRequestError(
                503,
                "rtc_verifier_unavailable",
                "RTC verifier is unavailable",
            )
        if not isinstance(payload, dict):
            raise RtcRequestError(400, "invalid_offer", "offer body must be an object")
        unknown = set(payload) - {"sdp", "type", "ticket"}
        missing = {"sdp", "type", "ticket"} - set(payload)
        if missing or unknown:
            raise RtcRequestError(
                400,
                "invalid_offer",
                f"offer fields invalid; missing={sorted(missing)}, unknown={sorted(unknown)}",
            )
        if payload["type"] != "offer" or not isinstance(payload["sdp"], str):
            raise RtcRequestError(400, "invalid_offer", "type must be 'offer' and sdp must be a string")

        try:
            binding, generation = self._runtime.rtc_authority_snapshot()
            self._verifier.verify_and_consume(
                payload["ticket"],
                expected=binding,
                sdp=payload["sdp"],
            )
        except TicketError as exc:
            self._runtime.record_protocol_error(exc.code)
            raise RtcRequestError(401, exc.code, str(exc)) from exc
        except ProtocolError as exc:
            raise RtcRequestError(409, exc.code, str(exc)) from exc

        async with self._lock:
            # MCP may prepare a newer session while an offer is being verified.
            # Never let a ticket for the old authority create a peer whose
            # callbacks are accidentally bound to the new runtime generation.
            try:
                current_binding, current_generation = (
                    self._runtime.rtc_authority_snapshot()
                )
                if current_binding != binding or current_generation != generation:
                    raise RtcRequestError(
                        409,
                        "session_changed",
                        "session changed while the RTC offer was being authorized",
                    )
            except ProtocolError as exc:
                raise RtcRequestError(409, exc.code, str(exc)) from exc
            await self._close_all_locked()
            peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
            self._peers.add(peer)
            self._install_peer_handlers(peer, generation, binding)
            try:
                await peer.setRemoteDescription(
                    RTCSessionDescription(sdp=payload["sdp"], type=payload["type"])
                )
                answer = await peer.createAnswer()
                await peer.setLocalDescription(answer)
            except Exception as exc:
                self._peers.discard(peer)
                await peer.close()
                self._runtime.record_protocol_error("invalid_sdp")
                raise RtcRequestError(400, "invalid_sdp", f"failed to negotiate offer: {exc}") from exc

        local = peer.localDescription
        return {
            "sdp": local.sdp,
            "type": local.type,
            "boot_id": binding["boot_id"],
            "session_id": binding["session_id"],
            "epoch": binding["epoch"],
            "capability_digest": binding["capability_digest"],
            "mode": self._runtime.mode,
            "actuation_enabled": self._runtime.actuation_enabled,
        }

    async def close_all(self) -> None:
        async with self._lock:
            await self._close_all_locked()

    async def _close_all_locked(self) -> None:
        peers = list(self._peers)
        self._peers.clear()
        if peers:
            await asyncio.gather(*(peer.close() for peer in peers), return_exceptions=True)

    def _install_peer_handlers(
        self,
        peer: RTCPeerConnection,
        generation: int,
        authority: dict,
    ) -> None:
        accepted_labels: set[str] = set()

        @peer.on("datachannel")
        def on_datachannel(channel: RTCDataChannel) -> None:
            label = channel.label
            if label in accepted_labels or not self._channel_contract_valid(channel):
                self._runtime.record_protocol_error("invalid_data_channel")
                channel.close()
                return
            accepted_labels.add(label)

            @channel.on("open")
            def on_open() -> None:
                self._runtime.mark_channel(generation, label, True)

            @channel.on("close")
            def on_close() -> None:
                self._runtime.mark_channel(generation, label, False)

            @channel.on("message")
            def on_message(message: str | bytes) -> None:
                if label == "teleop-pose":
                    self._handle_pose_message(message, generation, authority)
                else:
                    if not self._runtime.generation_matches(generation):
                        self._runtime.record_protocol_error("stale_rtc_message")
                        return
                    self._handle_control_message(channel, message)

            # aiortc emits the remote `datachannel` event after moving the
            # channel to open, so its earlier `open` event is not observable
            # from this callback.  Record that already-open state explicitly.
            if channel.readyState == "open":
                self._runtime.mark_channel(generation, label, True)

        @peer.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            if peer.connectionState in ("failed", "closed", "disconnected"):
                self._runtime.mark_rtc_disconnected(generation, f"rtc_{peer.connectionState}")
                self._peers.discard(peer)
                if peer.connectionState != "closed":
                    await peer.close()

    @staticmethod
    def _channel_contract_valid(channel: RTCDataChannel) -> bool:
        if channel.label == "teleop-control":
            return (
                channel.ordered is True
                and channel.maxRetransmits is None
                and channel.maxPacketLifeTime is None
            )
        if channel.label == "teleop-pose":
            return (
                channel.ordered is False
                and channel.maxRetransmits == 0
                and channel.maxPacketLifeTime is None
            )
        return False

    def _decode_message(self, message: str | bytes, *, maximum: int) -> Any:
        if isinstance(message, bytes):
            if len(message) > maximum:
                raise ProtocolError("message_too_large", f"RTC message exceeds {maximum} bytes")
            try:
                message = message.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ProtocolError("invalid_encoding", "RTC message must be UTF-8 JSON") from exc
        if not isinstance(message, str):
            raise ProtocolError("invalid_encoding", "RTC message must be UTF-8 JSON")
        try:
            encoded_size = len(message.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ProtocolError("invalid_encoding", "RTC message must be UTF-8 JSON") from exc
        if encoded_size > maximum:
            raise ProtocolError("message_too_large", f"RTC message exceeds {maximum} bytes")
        try:
            return json.loads(message)
        except (json.JSONDecodeError, ValueError, RecursionError) as exc:
            raise ProtocolError("invalid_json", "RTC message must be valid JSON") from exc

    def _handle_pose_message(
        self,
        message: str | bytes,
        generation: int,
        authority: dict,
    ) -> None:
        try:
            self._runtime.submit_rtc_frame(
                self._decode_message(message, maximum=64 * 1024),
                authority=authority,
                rtc_generation=generation,
                include_status=False,
            )
        except ProtocolError as exc:
            self._runtime.record_protocol_error(exc.code)

    def _handle_control_message(
        self,
        channel: RTCDataChannel,
        message: str | bytes,
    ) -> None:
        try:
            payload = self._decode_message(message, maximum=8 * 1024)
            if not isinstance(payload, dict) or set(payload) - {"type", "request_id"}:
                raise ProtocolError(
                    "invalid_control",
                    "control message contains invalid or private authority fields",
                )
            message_type = payload.get("type")
            if message_type in ("peer_ping", "status"):
                result = self._runtime.status()
            elif message_type == "heartbeat":
                raise ProtocolError(
                    "rtc_cannot_renew_lease",
                    "RTC control messages never renew the Capture presence lease",
                )
            elif message_type in ("pause", "soft_stop", "release"):
                raise ProtocolError(
                    "rtc_control_requires_card",
                    "session mutations must use the local teleop_session card",
                )
            else:
                raise ProtocolError("invalid_control", f"unsupported control message type: {message_type}")
            channel.send(json.dumps({
                "type": "response",
                "request_id": payload.get("request_id"),
                "ok": True,
                "state": result["state"],
                "lease_renewed": False,
            }, separators=(",", ":")))
        except ProtocolError as exc:
            self._runtime.record_protocol_error(exc.code)
            if channel.readyState == "open":
                channel.send(json.dumps({
                    "type": "response",
                    "ok": False,
                    "error": {"code": exc.code, "message": str(exc)},
                    "lease_renewed": False,
                }, separators=(",", ":")))
