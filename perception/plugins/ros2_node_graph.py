"""ROS 2 node-graph query card.

The card answers two questions about the live ROS 2 graph:

* **For the decision centre** — a compact text description of every
  publisher/subscriber flow, published as ``data/json``.  This is the data
  output and the reason the card exists; it is capped so wiring it into an LLM
  does not flood the context.
* **For a human** — a self-contained browser page showing the same graph.
  The page is served by a small HTTP server inside this container.

Ports on the robot's compute board are not exposed to the office network, so the
server binds loopback by default and the card hands out either
``http://127.0.0.1:<port>/`` plus the exact SSH tunnel command, or a configured
``public_base_url`` when the deployment has a reverse proxy / open port.  The
URL and tunnel are configurable from the card, which is why the same code works
on a laptop, a robot reached over SSH, or a robot behind a gateway.

The module imports rclpy only inside ``start`` so the pure graph model and the
HTTP/page layer can be tested without ROS.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from plugins.ros2_node_graph_page import PAGE_HTML
from plugins.ros2_node_graph_runtime import (
    DEFAULT_TOPIC,
    TOPIC_FORMAT,
    Config,
    _as_bool,
    _as_strings,
    build_snapshot,
    compact_payload,
    full_payload,
    json_dumps,
    url_info,
)

log = logging.getLogger(__name__)

PREFIX = "ros2_node_graph"
_GRAPH_CACHE_S = 1.0


class RosGraphNodeSource:
    """Thin wrapper around rclpy's graph API, kept replaceable by tests."""

    def __init__(self, node):
        self._node = node

    def get_nodes(self):
        return self._node.get_node_names_and_namespaces()

    def get_topics(self):
        return self._node.get_topic_names_and_types()

    def get_publishers(self, topic):
        return self._node.get_publishers_info_by_topic(topic)

    def get_subscriptions(self, topic):
        return self._node.get_subscriptions_info_by_topic(topic)


class _ThreadingServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class Ros2NodeGraphPlugin:
    PREFIX = PREFIX

    def __init__(self, plugin_cfg: dict, executor, graph_source=None):
        self._cfg = dict(plugin_cfg or {})
        self._config = Config.from_dict(self._cfg)
        self._executor = executor
        self._topic = str(self._cfg.get("topic") or DEFAULT_TOPIC)
        self._graph_source = graph_source
        self._owns_source = graph_source is None
        self._lock = threading.RLock()

        self._snapshot = None
        self._snapshot_at = 0.0
        self._last_published = ""
        self._last_publish_at = 0.0
        self._last_error = ""
        self._running = False
        self._node = None
        self._publisher = None
        self._timer = None

        self._http = None
        self._http_thread = None
        self._http_port = 0
        self._http_error = ""
        self._start_http()

    # ── Card definition ──────────────────────────────────────────────────────

    def _url_info(self) -> dict:
        return url_info(self._config, self._http_port or self._config.port)

    def get_tools(self) -> list:
        access = self._url_info()
        if access["tunnel"]:
            how_to_open = (
                f"网页：{access['url']}（复制到浏览器；如果打不开，先在本机执行："
                f"{access['tunnel']}）。"
            )
        else:
            how_to_open = f"网页：{access['url']}（直接打开）。"
        if self._http_error:
            how_to_open = f"网页端口启动失败（{self._http_error}）。" + how_to_open
        return [{
            "name": PREFIX,
            "type": "sensor",
            "multiInstance": False,
            "description": (
                how_to_open +
                "ROS 2 节点图查询卡片：读取当前 ROS 2 域内所有节点、话题以及它们之间的"
                "发布/订阅关系，输出一段拓扑文字描述（publisher -> topic -> subscriber），"
                "供决策中心判断消息流向；同时提供一个自包含的网页用于人工调试。"
                " 只统计发布/订阅，不包含 service/action；默认隐藏 /rosout 和 "
                "/parameter_events。网页无鉴权，默认只绑定 127.0.0.1，请用 SSH 隧道访问；"
                "如果部署在反代之后，把 public_base_url 配成浏览器可访问的地址。"
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["start", "stop", "info", "refresh", "config"],
                        "description": "Action to perform",
                    },
                },
                "required": ["action"],
                "additionalProperties": False,
                "x-action-params": {
                    "start": {"params": [],
                              "description": "启动节点图采集和 data/json 输出；网页服务在卡片加载时已启动"},
                    "stop": {"params": [],
                             "description": "停止采集和发布；网页仍可查看最后一次快照"},
                    "info": {"params": [],
                             "description": "返回状态、网页地址、SSH 隧道命令、统计和最近错误"},
                    "refresh": {"params": [],
                                "description": "立即重新扫描节点图并发布，返回最新文字描述"},
                    "config": {"params": [],
                               "description": "更新采集、端口或 URL/隧道配置"},
                },
            },
            "topic_out": [{
                "topic": self._topic,
                "format": TOPIC_FORMAT,
                "desc": ("节点图拓扑文字摘要（summary）+ 统计；完整节点表只在网页里，"
                         "不进入 LLM 上下文"),
            }],
            "configSchema": {
                "type": "object",
                "properties": {
                    "port": {"type": "integer", "default": 15730,
                             "minimum": 0, "maximum": 65535, "scope": "instance",
                             "description": "网页监听端口；0 表示交给系统随机分配"},
                    "bind_host": {"type": "string", "enum": ["127.0.0.1", "0.0.0.0"],
                                  "default": "127.0.0.1", "scope": "instance",
                                  "description": "监听地址。默认 loopback，需 SSH 隧道；0.0.0.0 允许局域网直连（无鉴权）"},
                    "public_base_url": {"type": "string", "default": "",
                                        "scope": "instance",
                                        "description": "可选的浏览器地址（反代/已开放端口），设置后优先使用，例如 https://robot.example/graph/"},
                    "ssh_host": {"type": "string", "default": "", "scope": "instance",
                                 "description": "SSH 隧道里的机器人地址；留空自动探测局域网 IP"},
                    "ssh_port": {"type": "integer", "default": 2222,
                                 "minimum": 1, "maximum": 65535, "scope": "instance",
                                 "description": "SSH 端口"},
                    "ssh_user": {"type": "string", "default": "noetix",
                                 "scope": "instance", "description": "SSH 用户名"},
                    "ssh_local_port": {"type": "integer", "default": 0,
                                       "minimum": 0, "maximum": 65535,
                                       "scope": "instance",
                                       "description": "本机映射端口；0 表示与网页端口相同"},
                    "publish_interval_s": {"type": "number", "default": 5.0,
                                           "minimum": 1, "maximum": 300,
                                           "scope": "instance",
                                           "description": "重新扫描/发布的间隔；内容不变时只发心跳"},
                    "include_system_topics": {"type": "boolean", "default": False,
                                              "scope": "instance",
                                              "description": "是否包含 /rosout 和 /parameter_events"},
                    "max_flows": {"type": "integer", "default": 60,
                                  "minimum": 5, "maximum": 500,
                                  "scope": "instance",
                                  "description": "文字摘要里最多列多少条消息流"},
                    "summary_max_chars": {"type": "integer", "default": 4000,
                                          "minimum": 500, "maximum": 20000,
                                          "scope": "instance",
                                          "description": "data/json 文字摘要的最大字符数"},
                },
                "required": [],
            },
        }]

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> dict:
        with self._lock:
            if self._running:
                return self._info()
            if self._graph_source is None:
                node = None
                try:
                    from rclpy.node import Node
                    from std_msgs.msg import String
                    node = Node("ros2_node_graph_card")
                    if self._executor is not None:
                        self._executor.add_node(node)
                    publisher = node.create_publisher(String, self._topic, 10)
                except Exception as error:  # noqa: BLE001
                    if node is not None:
                        try:
                            if self._executor is not None:
                                self._executor.remove_node(node)
                        except Exception:  # noqa: BLE001
                            pass
                        try:
                            node.destroy_node()
                        except Exception:  # noqa: BLE001
                            pass
                    self._last_error = f"无法创建 ROS 节点：{error}"
                    log.warning("ros2_node_graph: %s", self._last_error)
                    return {"state": "error", "error": self._last_error}
                self._node = node
                self._publisher = publisher
                self._graph_source = RosGraphNodeSource(node)
            self._running = True
            self._restart_timer()
        self._refresh(force_publish=True)
        return self._info()

    def stop(self) -> dict:
        with self._lock:
            self._running = False
            node, timer, publisher = self._node, self._timer, self._publisher
            self._node = self._timer = self._publisher = None
            if self._owns_source:
                self._graph_source = None
        if node is not None:
            try:
                if timer is not None:
                    node.destroy_timer(timer)
            except Exception:  # noqa: BLE001
                pass
            try:
                if publisher is not None:
                    node.destroy_publisher(publisher)
            except Exception:  # noqa: BLE001
                pass
            try:
                if self._executor is not None:
                    self._executor.remove_node(node)
            except Exception:  # noqa: BLE001
                pass
            try:
                node.destroy_node()
            except Exception:  # noqa: BLE001
                pass
        return {"state": "idle", "url": self._url_info()["url"],
                "message": "已停止采集；网页仍可查看最后一次快照"}

    def _restart_timer(self) -> None:
        node = self._node
        if node is None:
            return
        try:
            if self._timer is not None:
                node.destroy_timer(self._timer)
            period = max(1.0, self._config.publish_interval_s)
            self._timer = node.create_timer(period, self._tick)
        except Exception as error:  # noqa: BLE001
            self._last_error = f"创建扫描定时器失败：{error}"

    def _tick(self) -> None:
        if not self._running:
            return
        try:
            self._refresh(force_publish=False)
        except Exception as error:  # noqa: BLE001
            self._last_error = f"{type(error).__name__}: {error}"
            log.warning("ros2_node_graph tick failed: %s", error)

    def _refresh(self, force_publish: bool = False):
        with self._lock:
            source = self._graph_source
        if source is None:
            return None
        try:
            snapshot = build_snapshot(source, self._config)
        except Exception as error:  # noqa: BLE001
            self._last_error = f"扫描 ROS 图失败：{type(error).__name__}: {error}"
            log.warning("ros2_node_graph scan failed: %s", error)
            return None
        access = self._url_info()
        payload = compact_payload(snapshot, self._config,
                                  access["url"], access["tunnel"])
        with self._lock:
            self._snapshot = snapshot
            self._snapshot_at = time.monotonic()
            self._last_error = ""
            publisher = self._publisher
            encoded = json_dumps(payload)
            now = time.monotonic()
            due = (force_publish or encoded != self._last_published
                   or now - self._last_publish_at >= self._config.heartbeat_s)
            if due:
                self._last_published = encoded
                self._last_publish_at = now
        if due and publisher is not None:
            try:
                from std_msgs.msg import String
                publisher.publish(String(data=encoded))
            except Exception as error:  # noqa: BLE001
                self._last_error = f"发布节点图失败：{error}"
                log.warning("ros2_node_graph publish failed: %s", error)
        return payload

    # ── queries / config ─────────────────────────────────────────────────────

    def dispatch(self, action: str, args: dict):
        args = args or {}
        if action == "start":
            return self.start()
        if action == "stop":
            return self.stop()
        if action == "info":
            return self._info()
        if action == "refresh":
            payload = self._refresh(force_publish=True)
            if payload is None:
                return {"state": "error",
                        "error": self._last_error or "节点图尚未启动，请先 start"}
            return payload
        if action == "config":
            return self._config_action(args)
        return None

    def _info(self) -> dict:
        access = self._url_info()
        with self._lock:
            snapshot = self._snapshot
            age = (round(time.monotonic() - self._snapshot_at, 1)
                   if snapshot is not None else None)
        result = {
            "state": "running" if self._running else "idle",
            "url": access["url"],
            "tunnel": access["tunnel"],
            "access_mode": access["mode"],
            "bind_host": self._config.bind_host,
            "port": self._http_port or self._config.port,
            "topic": self._topic,
            "protocol": TOPIC_FORMAT,
            "stats": snapshot["stats"] if snapshot else None,
            "last_scan_age_s": age,
            "config": self._config.as_dict(),
        }
        if snapshot:
            result["summary"] = snapshot["summary"]
            result["truncated"] = snapshot["truncated"]
        if self._http_error:
            result["web_error"] = self._http_error
        if self._last_error:
            result["last_error"] = self._last_error
        return result

    def _config_action(self, args: dict) -> dict:
        allowed = {
            "port": int,
            "bind_host": str,
            "public_base_url": str,
            "ssh_host": str,
            "ssh_port": int,
            "ssh_user": str,
            "ssh_local_port": int,
            "publish_interval_s": float,
            "heartbeat_s": float,
            "include_system_topics": lambda value: _as_bool(value, False),
            "exclude_topics": lambda value: list(_as_strings(value)),
            "exclude_namespaces": lambda value: list(_as_strings(value)),
            "max_nodes": int,
            "max_topics": int,
            "max_flows": int,
            "summary_max_chars": int,
        }
        changed = {}
        merged = dict(self._cfg)
        for key, cast in allowed.items():
            if key not in args or args[key] is None:
                continue
            try:
                value = cast(args[key])
            except (TypeError, ValueError):
                return {"status": "error", "error": f"{key} 类型不正确"}
            changed[key] = value
            merged[key] = value
        if not changed:
            return {"status": "configured", "config": self._config.as_dict(),
                    "url": self._url_info()["url"]}
        try:
            new_config = Config.from_dict(merged)
        except ValueError as error:
            return {"status": "error", "error": str(error)}

        previous = self._config
        needs_http = (new_config.bind_host != previous.bind_host
                      or new_config.port != previous.port)
        needs_timer = (new_config.publish_interval_s
                       != previous.publish_interval_s)
        if needs_http:
            self._stop_http()
        self._config = new_config
        if needs_http:
            self._start_http()
            if self._http_error:
                bind_error = self._http_error
                self._config = previous
                self._start_http()
                return {"status": "error",
                        "error": f"网页端口/地址启动失败：{bind_error}"}
        self._cfg = merged
        if needs_timer and self._running:
            self._restart_timer()
        return {"status": "configured", "changed": changed,
                "config": self._config.as_dict(),
                "url": self._url_info()["url"],
                "tunnel": self._url_info()["tunnel"]}

    # ── HTTP page ────────────────────────────────────────────────────────────

    def _start_http(self) -> None:
        self._stop_http()
        plugin = self
        config = self._config

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                plugin._serve_http(self)

            def log_message(self, fmt, *args):
                return

        try:
            server = _ThreadingServer((config.bind_host, config.port), Handler)
        except OSError as error:
            self._http = None
            self._http_port = 0
            self._http_error = str(error)
            log.warning("ros2_node_graph: cannot bind %s:%s: %s",
                        config.bind_host, config.port, error)
            return
        self._http = server
        self._http_port = server.server_address[1]
        self._http_error = ""
        thread = threading.Thread(target=server.serve_forever,
                                  daemon=True, name="ros2_node_graph_http")
        self._http_thread = thread
        thread.start()
        log.info("ros2_node_graph: web page on %s:%s",
                 config.bind_host, self._http_port)

    def _stop_http(self) -> None:
        server, thread = self._http, self._http_thread
        self._http = None
        self._http_thread = None
        self._http_port = 0
        if server is not None:
            try:
                server.shutdown()
            except Exception:  # noqa: BLE001
                pass
            try:
                server.server_close()
            except Exception:  # noqa: BLE001
                pass
        if (thread is not None and thread.is_alive()
                and thread is not threading.current_thread()):
            thread.join(timeout=2)

    def _snapshot_for_http(self):
        with self._lock:
            snapshot = self._snapshot
            age = time.monotonic() - self._snapshot_at
            source = self._graph_source
        if snapshot is not None and age <= _GRAPH_CACHE_S:
            return snapshot
        if source is None:
            return snapshot
        try:
            snapshot = build_snapshot(source, self._config)
        except Exception as error:  # noqa: BLE001
            self._last_error = f"扫描 ROS 图失败：{type(error).__name__}: {error}"
            return snapshot
        with self._lock:
            self._snapshot = snapshot
            self._snapshot_at = time.monotonic()
        return snapshot

    def _serve_http(self, handler: BaseHTTPRequestHandler) -> None:
        path = urllib.parse.urlsplit(handler.path).path
        if path in ("/", "/index.html"):
            self._respond(handler, 200, PAGE_HTML.encode("utf-8"),
                          "text/html; charset=utf-8")
            return
        if path == "/healthz":
            self._respond(handler, 200, b"ok", "text/plain; charset=utf-8")
            return
        if path == "/favicon.ico":
            self._respond(handler, 204, b"", "image/x-icon")
            return
        snapshot = self._snapshot_for_http()
        if snapshot is None:
            body = json_dumps({
                "schema": "motus.ros_graph/1",
                "state": "idle",
                "error": "节点图尚未启动，请在卡片上执行 start",
            }).encode("utf-8")
            self._respond(handler, 503, body, "application/json; charset=utf-8")
            return
        if path == "/api/graph":
            access = self._url_info()
            payload = full_payload(snapshot, self._config,
                                   access["url"], access["tunnel"])
            payload["state"] = "running" if self._running else "idle"
            body = json_dumps(payload).encode("utf-8")
            self._respond(handler, 200, body, "application/json; charset=utf-8")
            return
        if path == "/api/summary":
            self._respond(handler, 200, snapshot["summary"].encode("utf-8"),
                          "text/plain; charset=utf-8")
            return
        self._respond(handler, 404, b"not found", "text/plain; charset=utf-8")

    @staticmethod
    def _respond(handler: BaseHTTPRequestHandler, status: int, body: bytes,
                 content_type: str) -> None:
        try:
            handler.send_response(status)
            handler.send_header("Content-Type", content_type)
            handler.send_header("Content-Length", str(len(body)))
            handler.send_header("Cache-Control", "no-store")
            handler.send_header("X-Content-Type-Options", "nosniff")
            handler.end_headers()
            if body:
                handler.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # ── lifecycle hooks ──────────────────────────────────────────────────────

    def shutdown(self) -> None:
        try:
            self.stop()
        finally:
            self._stop_http()
