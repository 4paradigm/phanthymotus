"""QR onboarding session lifecycle and server-side credential handling."""
import asyncio
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

from channel import qq_qr_onboarding as qr
from api import channel_setup
from channel.manager import manager


def test_qr_session_configures_server_side_and_never_returns_credential(monkeypatch):
    async def scenario():
        seen = []
        credential_value = "x" * 32

        async def fake_onboard(on_qr_ready):
            on_qr_ready("https://q.qq.com/qqbot/openclaw/connect.html?task_id=one-time")
            result = SimpleNamespace(app_id="12345678", user_openid="scanner-openid")
            setattr(result, "client_" + "secret", credential_value)
            return result

        async def configure(app_id, credential):
            seen.append((app_id, credential))
            return {
                "channel_id": "qq_official_12345678",
                "connected": True,
                "flow_ready": True,
                "input_subscription_ready": True,
                "subscription_pending": False,
            }

        monkeypatch.setattr(qr, "_start_sdk_onboard", fake_onboard)
        monkeypatch.setattr(qr, "_qr_data_url",
                            lambda url: "data:image/svg+xml;base64,PHN2Zy8+")
        state = await qr.create_session(configure)
        session_id = state["session_id"]
        await qr._SESSIONS[session_id].task

        result = qr.get_session(session_id)
        assert result["status"] == "connected"
        assert result["channel_id"] == "qq_official_12345678"
        assert result["qr_data_url"] == ""
        assert seen == [("12345678", credential_value)]
        assert credential_value not in repr(result)
        assert "scanner-openid" not in repr(result)
        qr._SESSIONS.pop(session_id, None)

    asyncio.run(scenario())


def test_qr_session_is_not_marked_connected_before_input_subscription_is_live(monkeypatch):
    async def scenario():
        async def fake_onboard(on_qr_ready):
            on_qr_ready("https://q.qq.com/one-time")
            return SimpleNamespace(app_id="12345678", client_secret="z" * 32)

        async def configure(*_):
            return {
                "channel_id": "qq_official_12345678",
                "connected": True,
                "flow_ready": True,
                "input_subscription_ready": False,
                "subscription_pending": True,
                "message": "消息输入订阅将在启动智能控制时激活。",
            }

        monkeypatch.setattr(qr, "_start_sdk_onboard", fake_onboard)
        monkeypatch.setattr(qr, "_qr_data_url",
                            lambda url: "data:image/svg+xml;base64,PHN2Zy8+")
        state = await qr.create_session(configure)
        await qr._SESSIONS[state["session_id"]].task
        result = qr.get_session(state["session_id"])
        assert result["status"] == "configured"
        assert result["connected"] is True
        assert result["input_subscription_ready"] is False
        assert result["subscription_pending"] is True
        assert "启动智能控制" in result["message"]
        qr._SESSIONS.pop(state["session_id"], None)

    asyncio.run(scenario())


def test_qr_session_can_be_cancelled_before_scan(monkeypatch):
    async def scenario():
        started = asyncio.Event()

        async def fake_onboard(on_qr_ready):
            on_qr_ready("https://q.qq.com/one-time")
            started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(qr, "_start_sdk_onboard", fake_onboard)
        monkeypatch.setattr(qr, "_qr_data_url",
                            lambda url: "data:image/svg+xml;base64,PHN2Zy8+")
        state = await qr.create_session(lambda *_: asyncio.sleep(0))
        await started.wait()
        result = await asyncio.to_thread(qr.cancel_session, state["session_id"])
        assert result["status"] == "cancelled"
        try:
            await qr._SESSIONS[state["session_id"]].task
            raise AssertionError("cancelled task should propagate cancellation")
        except asyncio.CancelledError:
            pass
        qr._SESSIONS.pop(state["session_id"], None)

    asyncio.run(scenario())


def test_expired_qr_has_retryable_state(monkeypatch):
    async def scenario():
        async def fake_onboard(on_qr_ready):
            raise TimeoutError("hidden low-level detail")

        monkeypatch.setattr(qr, "_start_sdk_onboard", fake_onboard)
        state = await qr.create_session(lambda *_: asyncio.sleep(0))
        await qr._SESSIONS[state["session_id"]].task
        result = qr.get_session(state["session_id"])
        assert result["status"] == "expired"
        assert "重新扫码" in result["message"]
        assert "hidden low-level detail" not in repr(result)
        qr._SESSIONS.pop(state["session_id"], None)

    asyncio.run(scenario())


def test_scanned_credentials_are_persisted_and_not_written_to_channel_config(monkeypatch):
    app_id = "12345678"
    credential_value = "y" * 32
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp_dir:
        credential_dir = Path(temp_dir) / "qq-credentials"
        monkeypatch.setattr(channel_setup.qq_credential_store, "directory",
                            lambda: credential_dir)

        state = {"entry": None}

        def add(channel_id, platform, cfg, enabled):
            state["entry"] = {
                "id": channel_id, "platform": platform, "config": dict(cfg),
                "enabled": enabled, "status": "disconnected",
            }
            return state["entry"]

        monkeypatch.setattr(channel_setup, "_get_channel_configs",
                            lambda: [state["entry"]] if state["entry"] else [])
        monkeypatch.setattr(channel_setup, "get_channel_config",
                            lambda channel_id: state["entry"]
                            if state["entry"] and state["entry"]["id"] == channel_id else None)
        monkeypatch.setattr(channel_setup, "add_channel_config", add)
        monkeypatch.setattr(channel_setup, "ensure_default_flow",
                            lambda channel_id: {"ready": True, "status": "created"})
        monkeypatch.setattr(manager, "sync_from_canvas", lambda: None)
        monkeypatch.setattr(manager, "_active_input_channels", {f"qq_official_{app_id}"})
        monkeypatch.setattr(manager, "_active_output_channels", {f"qq_official_{app_id}"})
        health = {}
        monkeypatch.setattr(manager, "_health", health)

        async def start_adapter(channel_id, retries=1):
            health[channel_id] = (True, "")
        monkeypatch.setattr(manager, "restart_adapter", start_adapter)

        result = asyncio.run(
            channel_setup._configure_scanned_qq(app_id, credential_value))
        cfg = state["entry"]["config"]
        path = cfg["app_secret_file"]

        assert result["connected"] is True
        assert result["flow_ready"] is True
        assert result["input_subscription_ready"] is False
        assert result["subscription_pending"] is True
        assert "智能控制启动时" in result["message"]
        assert cfg["app_id"] == app_id
        assert cfg["enabled_scopes"] == ["c2c", "group"]
        assert "app_secret" not in cfg
        assert "app_secret_runtime" not in cfg
        assert credential_value not in repr(cfg)
        if os.name != "nt":
            assert os.stat(path).st_mode & 0o777 == 0o600
        with open(path, encoding="utf-8") as saved:
            assert saved.read() == credential_value
        assert credential_value not in repr(result)



def test_missing_qrcode_dependency_is_reported_before_tencent_request(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "qrcode.image.svg", None)
    try:
        asyncio.run(qr._start_sdk_onboard(lambda url: None))
        raise AssertionError("missing QR dependency should fail")
    except qr.QRSessionError as exc:
        assert "二维码组件" in str(exc)


def test_connect_error_reports_network_problem_without_exposing_sdk_details(monkeypatch):
    async def scenario():
        import httpx

        async def fail_network(on_qr_ready):
            raise httpx.ConnectError("confidential service connection detail")

        monkeypatch.setattr(qr, "_start_sdk_onboard", fail_network)
        state = await qr.create_session(lambda *_: asyncio.sleep(0))
        await qr._SESSIONS[state["session_id"]].task
        status = qr.get_session(state["session_id"])
        assert status["status"] == "failed"
        assert "网络" in status["message"]
        assert "confidential" not in repr(status)
        qr._SESSIONS.pop(state["session_id"], None)
    asyncio.run(scenario())

def test_completed_qr_sessions_are_bounded(monkeypatch):
    async def scenario():
        monkeypatch.setattr(qr, "_MAX_STORED_SESSIONS", 1)

        async def fake_onboard(on_qr_ready):
            on_qr_ready("https://q.qq.com/one-time")
            return SimpleNamespace(app_id="12345678", client_secret="z" * 32)

        async def configure(*_):
            return {"channel_id": "qq_official_12345678",
                    "connected": True, "flow_ready": True,
                    "input_subscription_ready": True, "subscription_pending": False}

        monkeypatch.setattr(qr, "_start_sdk_onboard", fake_onboard)
        monkeypatch.setattr(qr, "_qr_data_url",
                            lambda url: "data:image/svg+xml;base64,PHN2Zy8+")
        first = await qr.create_session(configure)
        await qr._SESSIONS[first["session_id"]].task
        second = await qr.create_session(configure)
        await qr._SESSIONS[second["session_id"]].task
        assert first["session_id"] not in qr._SESSIONS
        assert second["session_id"] in qr._SESSIONS
        qr._SESSIONS.pop(second["session_id"], None)

    asyncio.run(scenario())
