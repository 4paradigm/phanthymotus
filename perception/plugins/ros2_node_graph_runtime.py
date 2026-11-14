"""Pure graph model for the ROS 2 node-graph card.

This module deliberately has no ROS import: the plugin feeds it a small
``source`` object that wraps rclpy's graph API, and the tests feed it a fake
source.  That keeps the interesting part — turning nodes/topics/endpoints into
a stable text description and a browser payload — testable without a robot.
"""

from __future__ import annotations

import json
import math
import socket
from dataclasses import dataclass, field
from datetime import datetime

SCHEMA = "motus.ros_graph/1"
DEFAULT_TOPIC = "/perception/ros2_node_graph/state"
TOPIC_FORMAT = "data/json"
DEFAULT_PORT = 15730
PAGE_REFRESH_MS = 2500
SYSTEM_TOPICS = ("/rosout", "/parameter_events")
_BIND_HOSTS = ("127.0.0.1", "0.0.0.0")


def _as_bool(value, default=False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
    return bool(value)


def _as_int(value, default: int, low: int, high: int) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return min(high, max(low, number))


def _as_float(value, default: float, low: float, high: float) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return min(high, max(low, number))


def _as_strings(value) -> tuple:
    if value is None:
        return ()
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (list, tuple, set)):
        return ()
    return tuple(text.strip() for text in (str(item) for item in value)
                 if text.strip())


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _normalise_node(name: str, namespace: str) -> str:
    name = str(name or "").strip("/")
    namespace = str(namespace or "").strip()
    if not name:
        return ""
    if namespace in ("", "/"):
        return "/" + name
    return "/" + namespace.strip("/") + "/" + name


def _endpoint_name(endpoint) -> str:
    return _normalise_node(getattr(endpoint, "node_name", ""),
                           getattr(endpoint, "node_namespace", ""))


def _endpoint_type(endpoint, fallback: str = "") -> str:
    return str(getattr(endpoint, "topic_type", "") or fallback)


@dataclass
class Config:
    port: int = DEFAULT_PORT
    bind_host: str = "127.0.0.1"
    public_base_url: str = ""
    ssh_host: str = ""
    ssh_port: int = 2222
    ssh_user: str = "noetix"
    ssh_local_port: int = 0
    publish_interval_s: float = 5.0
    heartbeat_s: float = 30.0
    include_system_topics: bool = False
    exclude_topics: tuple = field(default_factory=tuple)
    exclude_namespaces: tuple = field(default_factory=tuple)
    max_nodes: int = 120
    max_topics: int = 200
    max_flows: int = 60
    summary_max_chars: int = 4000

    @classmethod
    def from_dict(cls, values: dict | None) -> "Config":
        values = values or {}
        config = cls()
        config.port = _as_int(values.get("port"), DEFAULT_PORT, 0, 65535)
        bind_host = str(values.get("bind_host") or "127.0.0.1").strip()
        config.bind_host = bind_host if bind_host in _BIND_HOSTS else "127.0.0.1"
        config.public_base_url = str(values.get("public_base_url") or "").strip()
        config.ssh_host = str(values.get("ssh_host") or "").strip()
        config.ssh_port = _as_int(values.get("ssh_port"), 2222, 1, 65535)
        config.ssh_user = str(values.get("ssh_user") or "").strip()
        config.ssh_local_port = _as_int(values.get("ssh_local_port"), 0, 0, 65535)
        config.publish_interval_s = _as_float(
            values.get("publish_interval_s"), 5.0, 1.0, 300.0)
        config.heartbeat_s = _as_float(
            values.get("heartbeat_s"), 30.0, 5.0, 3600.0)
        config.include_system_topics = _as_bool(
            values.get("include_system_topics"), False)
        config.exclude_topics = _as_strings(values.get("exclude_topics"))
        config.exclude_namespaces = _as_strings(values.get("exclude_namespaces"))
        config.max_nodes = _as_int(values.get("max_nodes"), 120, 10, 1000)
        config.max_topics = _as_int(values.get("max_topics"), 200, 10, 2000)
        config.max_flows = _as_int(values.get("max_flows"), 60, 5, 500)
        config.summary_max_chars = _as_int(
            values.get("summary_max_chars"), 4000, 500, 20000)
        config.validate()
        return config

    def validate(self) -> None:
        if self.bind_host not in _BIND_HOSTS:
            raise ValueError("bind_host must be 127.0.0.1 or 0.0.0.0")
        if not 0 <= self.port <= 65535:
            raise ValueError("port must be in [0, 65535]")
        if self.public_base_url:
            if not self.public_base_url.startswith(("http://", "https://")):
                raise ValueError("public_base_url must start with http:// or https://")
            self.public_base_url = self.public_base_url.rstrip("/") + "/"
        if not 1 <= self.ssh_port <= 65535:
            raise ValueError("ssh_port must be in [1, 65535]")
        if not 0 <= self.ssh_local_port <= 65535:
            raise ValueError("ssh_local_port must be in [0, 65535]")
        if not 1.0 <= self.publish_interval_s <= 300.0:
            raise ValueError("publish_interval_s must be in [1, 300]")
        if not 5.0 <= self.heartbeat_s <= 3600.0:
            raise ValueError("heartbeat_s must be in [5, 3600]")
        if not 10 <= self.max_nodes <= 1000:
            raise ValueError("max_nodes must be in [10, 1000]")
        if not 10 <= self.max_topics <= 2000:
            raise ValueError("max_topics must be in [10, 2000]")
        if not 5 <= self.max_flows <= 500:
            raise ValueError("max_flows must be in [5, 500]")
        if not 500 <= self.summary_max_chars <= 20000:
            raise ValueError("summary_max_chars must be in [500, 20000]")

    def as_dict(self) -> dict:
        return {
            "port": self.port,
            "bind_host": self.bind_host,
            "public_base_url": self.public_base_url,
            "ssh_host": self.ssh_host,
            "ssh_port": self.ssh_port,
            "ssh_user": self.ssh_user,
            "ssh_local_port": self.ssh_local_port,
            "publish_interval_s": self.publish_interval_s,
            "heartbeat_s": self.heartbeat_s,
            "include_system_topics": self.include_system_topics,
            "exclude_topics": list(self.exclude_topics),
            "exclude_namespaces": list(self.exclude_namespaces),
            "max_nodes": self.max_nodes,
            "max_topics": self.max_topics,
            "max_flows": self.max_flows,
            "summary_max_chars": self.summary_max_chars,
        }


def _topic_allowed(topic: str, config: Config) -> bool:
    if not config.include_system_topics:
        if topic in SYSTEM_TOPICS:
            return False
        if topic.endswith("/parameter_events"):
            return False
    return not any(pattern in topic for pattern in config.exclude_topics)


def _namespace_allowed(full_name: str, config: Config) -> bool:
    return not any(full_name.startswith(prefix)
                   for prefix in config.exclude_namespaces)


def build_snapshot(source, config: Config, now: str | None = None) -> dict:
    """Turn rclpy's graph API (or a fake source) into one JSON-safe snapshot."""
    node_names = set()
    for name, namespace in source.get_nodes():
        full = _normalise_node(name, namespace)
        if full and _namespace_allowed(full, config):
            node_names.add(full)

    topics = []
    for raw_topic, raw_types in source.get_topics():
        topic = str(raw_topic or "")
        if not topic or not _topic_allowed(topic, config):
            continue
        types = [str(item) for item in (raw_types or []) if str(item)]
        try:
            publishers = list(source.get_publishers(topic) or [])
        except Exception:
            publishers = []
        try:
            subscribers = list(source.get_subscriptions(topic) or [])
        except Exception:
            subscribers = []
        pub_names = sorted({_endpoint_name(item) for item in publishers
                            if _endpoint_name(item)
                            and _namespace_allowed(_endpoint_name(item), config)})
        sub_names = sorted({_endpoint_name(item) for item in subscribers
                            if _endpoint_name(item)
                            and _namespace_allowed(_endpoint_name(item), config)})
        if not pub_names and not sub_names:
            continue
        if not types:
            types = sorted({_endpoint_type(item) for item in publishers + subscribers
                            if _endpoint_type(item)})
        node_names.update(pub_names)
        node_names.update(sub_names)
        topics.append({
            "name": topic,
            "types": types,
            "publishers": pub_names,
            "subscribers": sub_names,
        })

    node_names = sorted(node_names)
    topics = sorted(topics, key=lambda item: item["name"])
    total_nodes, total_topics = len(node_names), len(topics)
    truncated = (total_nodes > config.max_nodes
                 or total_topics > config.max_topics
                 or total_topics > config.max_flows)
    shown_nodes = node_names[:config.max_nodes]
    shown_topics = topics[:config.max_topics]
    shown_flows = topics[:config.max_flows]
    nodes = [{
        "name": name,
        "namespace": name.rsplit("/", 1)[0] or "/",
        "internal": "ros2_node_graph" in name,
    } for name in shown_nodes]
    links = sum(len(item["publishers"]) + len(item["subscribers"])
                for item in topics)
    snapshot = {
        "schema": SCHEMA,
        "generated_at": now or _now_iso(),
        "stats": {
            "nodes": total_nodes,
            "topics": total_topics,
            "links": links,
            "publishers": sum(len(item["publishers"]) for item in topics),
            "subscribers": sum(len(item["subscribers"]) for item in topics),
        },
        "truncated": truncated,
        "nodes": nodes,
        "topics": shown_topics,
        "flows": shown_flows,
    }
    snapshot["summary"] = summarize(snapshot, config)
    return snapshot


def summarize(snapshot: dict, config: Config) -> str:
    """Compact text for the decision centre; the web page gets the full graph."""
    stats = snapshot["stats"]
    lines = [
        f"ROS 2 node graph: {stats['nodes']} nodes, {stats['topics']} topics, "
        f"{stats['links']} links."
    ]
    flows = snapshot.get("flows") or []
    if flows:
        lines.append("Message flows (publishers -> topic [type] -> subscribers):")
        for flow in flows:
            publishers = ", ".join(flow["publishers"]) or "(none)"
            subscribers = ", ".join(flow["subscribers"]) or "(no subscribers)"
            types = ", ".join(flow["types"]) or "unknown type"
            lines.append(f"- {publishers} -> {flow['name']} [{types}] -> {subscribers}")
            if sum(len(line) + 1 for line in lines) > config.summary_max_chars:
                lines.pop()
                lines.append("- ... flow list truncated; open the card web page "
                             "for the complete graph.")
                break
    else:
        lines.append("No publisher/subscriber links found.")

    nodes = [item["name"] for item in snapshot.get("nodes") or []]
    if nodes:
        line = "Nodes: " + ", ".join(nodes)
        if sum(len(item) + 1 for item in lines) + len(line) <= config.summary_max_chars:
            lines.append(line)
        else:
            lines.append("Nodes: ... (list truncated)")

    if snapshot.get("truncated"):
        lines.append("Truncated: the graph is larger than the card's display caps; "
                     "see the web page or raise max_nodes/max_topics/max_flows.")
    text = "\n".join(lines)
    if len(text) > config.summary_max_chars:
        suffix = "\n... summary truncated at summary_max_chars."
        cut = max(0, config.summary_max_chars - len(suffix))
        text = text[:cut].rstrip() + suffix
    return text


def compact_payload(snapshot: dict, config: Config, url: str = "",
                    tunnel: str = "") -> dict:
    """What goes on the data/json topic: text for the agent, not the full graph."""
    return {
        "schema": SCHEMA,
        "state": "running",
        "generated_at": snapshot["generated_at"],
        "stats": snapshot["stats"],
        "truncated": snapshot["truncated"],
        "summary": snapshot["summary"],
        "url": url,
        "tunnel": tunnel,
    }


def full_payload(snapshot: dict, config: Config, url: str = "",
                 tunnel: str = "") -> dict:
    payload = dict(snapshot)
    payload.update({"state": "running", "url": url, "tunnel": tunnel})
    return payload


def detect_lan_ip() -> str:
    """Best-effort robot LAN address (UDP connect does not send packets)."""
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.2)
        sock.connect(("8.8.8.8", 80))
        address = sock.getsockname()[0]
        return "" if address.startswith("127.") else address
    except OSError:
        return ""
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


def url_info(config: Config, actual_port: int) -> dict:
    """Return the browser URL and the one-line SSH tunnel that reaches it.

    ``public_base_url`` wins: a deployment behind a reverse proxy or with an
    open port can hand out a URL that needs no tunnel.  A loopback bind (the
    default) is intentionally unreachable from another machine, so the card
    gives the exact tunnel command instead of pretending the port is open.
    """
    port = int(actual_port or config.port or DEFAULT_PORT)
    if config.public_base_url:
        return {"url": config.public_base_url, "tunnel": "",
                "mode": "public", "bind": config.bind_host, "port": port}
    host = config.ssh_host or detect_lan_ip() or "<robot-ip>"
    if config.bind_host == "0.0.0.0":
        return {"url": f"http://{host}:{port}/", "tunnel": "",
                "mode": "direct", "bind": config.bind_host, "port": port}
    local = config.ssh_local_port or port
    user = f"{config.ssh_user}@" if config.ssh_user else ""
    tunnel = f"ssh -p {config.ssh_port} {user}{host} -L {local}:127.0.0.1:{port}"
    return {"url": f"http://127.0.0.1:{local}/", "tunnel": tunnel,
            "mode": "tunnel", "bind": config.bind_host, "port": port}


def json_dumps(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
