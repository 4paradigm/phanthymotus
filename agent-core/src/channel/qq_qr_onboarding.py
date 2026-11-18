"""Short-lived, process-local QQ QR onboarding sessions.

The Tencent SDK owns QR task creation, polling and AES-GCM decryption. This
module only exposes a random session handle and an SVG QR to the authenticated
Dashboard; bot credentials are passed directly to the server-side configure
callback and are never included in session state or API responses.
"""
from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
import logging
import secrets
import threading
import time
from typing import Awaitable, Callable

_LOG = logging.getLogger(__name__)
_SESSION_TTL = 300.0
_SESSION_RETENTION = 600.0
_MAX_ACTIVE_SESSIONS = 8
_MAX_STORED_SESSIONS = 32
_SESSION_ID_LENGTH_MIN = 32

ConfigureCallback = Callable[[str, str], Awaitable[dict]]


@dataclass
class _Session:
    session_id: str
    status: str = "starting"
    message: str = "正在生成安全连接二维码…"
    qr_data_url: str = ""
    channel_id: str = ""
    connected: bool = False
    flow_ready: bool = False
    input_subscription_ready: bool = False
    subscription_pending: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    task: asyncio.Task | None = None
    loop: asyncio.AbstractEventLoop | None = None


_LOCK = threading.RLock()
_SESSIONS: dict[str, _Session] = {}


class QRSessionError(ValueError):
    """Safe-to-display QR session error."""


def _qr_data_url(url: str) -> str:
    """Render the one-time Tencent connect URL as a self-contained SVG QR."""
    import qrcode
    from qrcode.image.svg import SvgPathImage

    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=8,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    svg = qr.make_image(image_factory=SvgPathImage).to_string()
    encoded = base64.b64encode(svg).decode("ascii")
    return "data:image/svg+xml;base64," + encoded


async def _start_sdk_onboard(on_qr_ready):
    # Import lazily so non-QQ deployments and unit tests do not need to import
    # the SDK until a user explicitly starts QQ onboarding.
    # The SDK logs its one-time task_id at INFO. Keep that bearer-like value
    # out of production logs; AppSecret is never logged by this integration.
    logging.getLogger("qqbot_agent_sdk.onboard").setLevel(logging.WARNING)
    # Fail before creating a Tencent bind task if the runtime was mounted
    # from local source but its Python environment lacks the QR dependency.
    try:
        from qrcode.image.svg import SvgPathImage  # noqa: F401
    except ImportError:
        raise QRSessionError(
            "扫码服务缺少二维码组件，请先同步后端依赖。"
        ) from None
    from qqbot_agent_sdk import start_onboard

    return await start_onboard(
        on_qr_ready=on_qr_ready,
        poll_interval=2.0,
        poll_timeout=_SESSION_TTL,
    )


def _cancel_task(session: _Session) -> None:
    task = session.task
    if task is None or task.done():
        return
    loop = session.loop
    if loop is not None and loop.is_running():
        loop.call_soon_threadsafe(task.cancel)
    else:
        task.cancel()


def _prune_locked(now: float) -> None:
    for key, session in list(_SESSIONS.items()):
        age = now - session.updated_at
        if (session.task is not None and not session.task.done()
                and session.status in ("starting", "waiting_scan")):
            if now - session.created_at > _SESSION_TTL + 30:
                _cancel_task(session)
                session.status = "expired"
                session.message = "二维码已过期，请重新扫码。"
                session.qr_data_url = ""
                session.updated_at = now
            continue
        if age > _SESSION_RETENTION:
            _SESSIONS.pop(key, None)


def _public(session: _Session) -> dict:
    return {
        "session_id": session.session_id,
        "status": session.status,
        "message": session.message,
        "qr_data_url": session.qr_data_url if session.status == "waiting_scan" else "",
        "channel_id": session.channel_id,
        "connected": session.connected,
        "flow_ready": session.flow_ready,
        "input_subscription_ready": session.input_subscription_ready,
        "subscription_pending": session.subscription_pending,
        "expires_in": max(0, int(_SESSION_TTL - (time.time() - session.created_at)))
        if session.status in ("starting", "waiting_scan") else 0,
    }


async def _run(session: _Session, configure: ConfigureCallback) -> None:
    def on_qr_ready(url: str) -> None:
        try:
            qr_data_url = _qr_data_url(url)
        except Exception:
            session.status = "failed"
            session.message = "二维码生成失败，请刷新后重试。"
            session.updated_at = time.time()
            _LOG.warning("QQ QR image rendering failed")
            return
        session.qr_data_url = qr_data_url
        session.status = "waiting_scan"
        session.message = "请使用手机 QQ 扫描二维码并确认授权。"
        session.updated_at = time.time()

    try:
        result = await _start_sdk_onboard(on_qr_ready)
        session.qr_data_url = ""
        session.status = "configuring"
        session.message = "扫码成功，正在自动配置 QQ 通道…"
        session.updated_at = time.time()

        # Do not retain result, client_secret or scanner OpenID in session state.
        outcome = await configure(str(result.app_id), str(result.client_secret))
        session.channel_id = str(outcome.get("channel_id", ""))
        session.connected = bool(outcome.get("connected"))
        session.flow_ready = bool(outcome.get("flow_ready"))
        session.input_subscription_ready = bool(outcome.get("input_subscription_ready"))
        session.subscription_pending = bool(outcome.get("subscription_pending"))
        if (session.connected and session.flow_ready
                and session.input_subscription_ready):
            session.status = "connected"
            session.message = str(outcome.get(
                "message", "QQ 已连接，消息输入订阅已就绪，可以正常收发消息。",
            ))
        elif not session.flow_ready:
            session.status = "action_required"
            session.message = str(outcome.get(
                "message",
                "QQ 授权和通道配置已完成，但当前画布无法自动选择唯一助手。",
            ))
        else:
            session.status = "configured"
            session.message = str(outcome.get(
                "message",
                "QQ 已授权并保存配置，消息输入订阅尚未就绪；无需重新扫码。",
            ))
    except asyncio.CancelledError:
        session.status = "cancelled"
        session.message = "已取消 QQ 连接。"
        raise
    except Exception as exc:
        # Never expose raw SDK exceptions, request URLs, bind keys or credentials.
        try:
            from qqbot_agent_sdk import OnboardAPIError, OnboardExpiredError
        except ImportError:
            OnboardAPIError = ()  # type: ignore[assignment]
            OnboardExpiredError = ()  # type: ignore[assignment]

        if isinstance(exc, TimeoutError) or (
                OnboardExpiredError and isinstance(exc, OnboardExpiredError)):
            session.status = "expired"
            session.message = "二维码已过期，请重新扫码。"
        elif OnboardAPIError and isinstance(exc, OnboardAPIError):
            session.status = "failed"
            session.message = f"QQ 授权服务暂时无法完成请求（错误码 {exc.retcode}），请重试。"
        elif isinstance(exc, QRSessionError):
            session.status = "failed"
            session.message = str(exc)
        elif type(exc).__name__ in ("ConnectError", "ConnectTimeout", "ReadTimeout"):
            session.status = "failed"
            session.message = "连接 QQ 授权服务失败，请检查网络后重新扫码。"
            _LOG.warning("QQ QR onboarding network unavailable (%s)", type(exc).__name__)
        else:
            session.status = "failed"
            session.message = "QQ 自动配置失败，请重新扫码；如重复失败，请检查 Dashboard 连接状态。"
            _LOG.warning("QQ QR onboarding failed (%s)", type(exc).__name__)
    finally:
        session.updated_at = time.time()


async def create_session(configure: ConfigureCallback) -> dict:
    """Start an isolated QR flow and return its session state."""
    now = time.time()
    with _LOCK:
        _prune_locked(now)
        active = sum(
            1 for item in _SESSIONS.values()
            if item.task is not None and not item.task.done()
        )
        if active >= _MAX_ACTIVE_SESSIONS:
            raise QRSessionError("当前扫码请求较多，请稍后再试。")
        if len(_SESSIONS) >= _MAX_STORED_SESSIONS:
            completed = sorted(
                (item for item in _SESSIONS.values()
                 if item.task is None or item.task.done()),
                key=lambda item: item.updated_at,
            )
            while len(_SESSIONS) >= _MAX_STORED_SESSIONS and completed:
                _SESSIONS.pop(completed.pop(0).session_id, None)
            if len(_SESSIONS) >= _MAX_STORED_SESSIONS:
                raise QRSessionError("当前扫码请求较多，请稍后再试。")
        session_id = secrets.token_urlsafe(32)
        session = _Session(
            session_id=session_id,
            loop=asyncio.get_running_loop(),
        )
        _SESSIONS[session_id] = session
        session.task = asyncio.create_task(
            _run(session, configure), name="qq-qr-onboard-" + session_id[:8],
        )
    # QR task creation requires one Tencent API round-trip. Give it a short
    # head start so the first Dashboard response usually contains the QR.
    await asyncio.sleep(0)
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        if session.qr_data_url or session.status not in ("starting",):
            break
        await asyncio.sleep(0.05)
    return _public(session)


def get_session(session_id: str) -> dict:
    if not isinstance(session_id, str) or len(session_id) < _SESSION_ID_LENGTH_MIN:
        raise QRSessionError("扫码会话不存在或已过期，请重新开始。")
    with _LOCK:
        session = _SESSIONS.get(session_id)
        if session is None:
            raise QRSessionError("扫码会话不存在或已过期，请重新开始。")
        if (time.time() - session.created_at > _SESSION_TTL + 30
                and session.status in ("starting", "waiting_scan")):
            _cancel_task(session)
            session.status = "expired"
            session.message = "二维码已过期，请重新扫码。"
            session.qr_data_url = ""
            session.updated_at = time.time()
        return _public(session)


def cancel_session(session_id: str) -> dict:
    if not isinstance(session_id, str) or len(session_id) < _SESSION_ID_LENGTH_MIN:
        raise QRSessionError("扫码会话不存在或已过期。")
    with _LOCK:
        session = _SESSIONS.get(session_id)
        if session is None:
            raise QRSessionError("扫码会话不存在或已过期。")
        if session.status == "configuring":
            return _public(session)
        _cancel_task(session)
        session.status = "cancelled"
        session.message = "已取消 QQ 连接。"
        session.qr_data_url = ""
        session.updated_at = time.time()
        return _public(session)
