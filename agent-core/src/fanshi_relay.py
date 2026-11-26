"""Client for the provisioned Fanshi fleet gateway (v1 protocol)."""
import hashlib
import json
import os
from pathlib import Path
import re
import ssl
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPSHandler, HTTPRedirectHandler, ProxyHandler

DEFAULT_CONFIG = "/work/resource/fanshi-relay/client.json"
ACTION_RE = re.compile(r"fanshi_[0-9a-f]{32}\Z")
FINAL = {"accepted", "partial", "failed", "cancelled", "unknown"}


class RelayError(Exception):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def unknown(action_id, message):
    return {"ok": False, "status": "unknown", "state": "error",
            "message": message, "action_id": action_id}


class RelayClient:
    def __init__(self, config):
        try:
            self.identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            self.url = config["relay_url"].rstrip("/")
            self.device_id = config["device_id"]
            self.token = config["token"]
            parsed = urlsplit(self.url)
            if (parsed.scheme != "https" or not parsed.hostname or parsed.username or
                    parsed.password or parsed.query or parsed.fragment or parsed.path):
                raise ValueError()
            if not re.fullmatch(r"[0-9a-f]{32}", self.device_id):
                raise ValueError()
            if not re.fullmatch(r"[A-Za-z0-9_-]{40,100}", self.token):
                raise ValueError()
            context = ssl.create_default_context(cadata=config["ca_pem"])
            context.minimum_version = ssl.TLSVersion.TLSv1_2
        except (KeyError, TypeError, ValueError, ssl.SSLError, AttributeError):
            raise RelayError("统一发群接入文件无效，请重新下发设备接入文件。") from None
        # Corporate proxy settings must not intercept a LAN bearer credential.
        self.opener = build_opener(ProxyHandler({}), NoRedirect(), HTTPSHandler(context=context))

    @classmethod
    def load(cls):
        path = Path(os.environ.get("FANSHI_RELAY_CONFIG", DEFAULT_CONFIG))
        try:
            with path.open("rb") as stream:
                raw = stream.read(32769)
        except FileNotFoundError:
            return None
        except OSError:
            raise RelayError("无法读取统一发群接入文件，请重新安装连接信息。") from None
        try:
            if len(raw) > 32768:
                raise ValueError()
            return cls(json.loads(raw))
        except (ValueError, TypeError):
            raise RelayError("统一发群接入文件格式无效，请重新安装连接信息。") from None

    def _request(self, method, route, data=None):
        body = None if data is None else json.dumps(data, ensure_ascii=False).encode("utf-8")
        req = Request(self.url + route, data=body, method=method, headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + self.token,
            "X-Robot-ID": self.device_id,
        })
        try:
            try:
                response = self.opener.open(req, timeout=12)
            except HTTPError as exc:
                response = exc
            with response:
                code = response.status
                raw = response.read(65537)
            if len(raw) > 65536:
                raise ValueError()
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError()
            if code == 401:
                raise RelayError("统一服务拒绝设备凭据，请重新下发设备接入文件。")
            if code == 404:
                raise RelayError("统一服务中没有该记录；请核对群聊，不要直接重发。")
            if code not in (200, 202, 400, 409, 429):
                raise RelayError("统一服务响应异常；请查询结果，不要直接重发。")
            return value
        except RelayError:
            raise
        except Exception:
            raise RelayError("无法确认统一服务结果；请检查统一服务、网络和证书后查询结果。") from None

    @staticmethod
    def _validate_result(value, action_id):
        if (value.get("action_id") != action_id or
                value.get("status") not in FINAL | {"queued", "sending"}):
            raise RelayError("统一服务返回的 action_id 或状态无效。")
        # Return only documented result fields; never forward arbitrary server data.
        keys = {"ok", "status", "message", "state", "action_id", "parts", "accepted_parts",
                "total_parts", "request_count", "platform_code", "failed_groups", "has_group_failures"}
        return {key: item for key, item in value.items() if key in keys}

    def result(self, action_id):
        if not isinstance(action_id, str) or not ACTION_RE.fullmatch(action_id):
            raise RelayError("action_id 格式无效。")
        return self._validate_result(self._request("GET", "/v1/result/" + action_id), action_id)

    def run(self, action_id, payload, cancelled, progress, timeout=100, poll_interval=1):
        data = {"action_id": action_id}
        for item in payload["message"]["body"]:
            if item["type"] == "TEXT":
                data["text"] = item["content"]
            elif item["type"] == "IMAGE":
                data["image_base64"] = item["content"]
            elif item["type"] == "AT":
                data["at_userid"] = item["atuserids"][0]
        # There is precisely one POST that can create a send. A lost response
        # is recovered using read-only result queries, never another POST.
        if cancelled():
            return {"ok": False, "status": "cancelled", "action_id": action_id,
                    "message": "提交前卡片已停止，本组消息未发送。"}
        try:
            value = self._validate_result(self._request("POST", "/v1/send", data), action_id)
        except RelayError:
            try:
                value = self.result(action_id)
            except RelayError as exc:
                return unknown(action_id, str(exc))
        deadline = time.monotonic() + timeout
        cancel_requested = False
        while value["status"] not in FINAL:
            progress(value)
            if cancelled() and not cancel_requested:
                try:
                    value = self._validate_result(self._request(
                        "POST", "/v1/cancel", {"action_id": action_id}), action_id)
                except RelayError:
                    pass
                cancel_requested = True
            if value["status"] in FINAL:
                break
            if time.monotonic() >= deadline:
                return {**unknown(action_id, "统一服务仍在处理；稍后用相同 action_id 查询，勿重新发送。"),
                        "parts": value.get("parts", [])}
            time.sleep(poll_interval)
            try:
                value = self.result(action_id)
            except RelayError as exc:
                return {**unknown(action_id, str(exc)), "parts": value.get("parts", [])}
        return value
