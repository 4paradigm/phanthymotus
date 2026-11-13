"""Read-only local/ROS/MCP/Core checks. Never registers or starts an owner."""

import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path
import ssl
import sys
import time
import urllib.request

from .configuration import validate
from .contracts import NAME, need
from .engine import Engine
from .server import loads, rpc


def http(url, body=None, *, method=None, token_file=None, ca=None, raw=False):
    from urllib.parse import urlparse

    parsed = urlparse(url)
    need(parsed.scheme in ("http", "https"), "INVALID_URL")
    headers = {"Content-Type": "application/json"}
    if token_file:
        need(
            parsed.scheme == "https"
            or parsed.hostname in ("localhost", "127.0.0.1", "::1"),
            "TOKEN_REQUIRES_TLS_OR_LOOPBACK",
        )
        headers["Authorization"] = "Bearer " + Path(token_file).read_text().strip()
    req = urllib.request.Request(
        url,
        data=json.dumps(body, allow_nan=False).encode() if body is not None else None,
        method=method,
        headers=headers,
    )
    context = (
        ssl.create_default_context(cafile=ca) if parsed.scheme == "https" else None
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=context)
    )
    with opener.open(req, timeout=6) as response:
        return response.read().decode() if raw else loads(response.read())


def check(
    config, *, core=None, mcp=None, ros=False, sample=False, token_file=None, ca=None
):
    checks = {}
    details = {}
    failures = {}

    def run(key, fn):
        try:
            details[key] = fn()
            checks[key] = True
        except Exception as exc:
            failures[key] = dict(
                code=getattr(exc, "code", "CHECK_FAILED"), message=str(exc)
            )
            checks[key] = False

    def dependencies():
        result = {}
        for name in ("numpy", "jsonschema", "pyyaml"):
            result[name] = importlib.metadata.version(name)
        import cv2

        result["opencv"] = cv2.__version__
        return result

    run("dependencies", dependencies)
    run(
        "configuration",
        lambda: dict(
            piper_id=validate(config)["piper_id"],
            sources=list(config["sources"]),
            max_age_s=Engine(config).max_age,
        ),
    )
    try:
        details["ros_import_available"] = importlib.util.find_spec("rclpy") is not None
    except ValueError:
        details["ros_import_available"] = (
            False  # unit-test stub is not a ROS installation
        )
    if ros or sample:

        def topics():
            from .plugin import GoalVerifierPlugin

            card = GoalVerifierPlugin(config)
            server = None
            thread = None
            try:
                if sample and any(
                    b.get("detector") == "vop" for b in config["sources"].values()
                ):
                    from http.server import ThreadingHTTPServer
                    import threading
                    from .server import handler
                    from .exchange import FrameExchange

                    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(card))
                    card.image_exchange = FrameExchange(
                        f"http://127.0.0.1:{server.server_port}"
                    )
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                card.sources.start_ros()
                found = {}
                for node, _ in card.sources.nodes:
                    for name, types in node.get_topic_names_and_types():
                        found[name] = types
                for binding in config["sources"].values():
                    if binding.get("kind", "").startswith("ros_"):
                        expected = (
                            "std_msgs/msg/String"
                            if binding["kind"] == "ros_normalized"
                            else "sensor_msgs/msg/CompressedImage"
                            if binding["message_type"] == "compressed"
                            else "sensor_msgs/msg/Image"
                        )
                        need(
                            expected in found.get(binding["topic"], []),
                            "TOPIC_TYPE_NOT_CONFIRMED",
                        )
                if sample:
                    for source in config["sources"]:
                        data = card.sources.read(source, timeout=5)
                        card.engine.ingest(data)
                        found[source] = dict(
                            observation_id=data["observation_id"],
                            camera_id=data["camera_id"],
                            captured_at=data["captured_at"],
                            provenance=data["provenance"],
                            count=len(data["objects"]),
                        )
                return found
            finally:
                card.stop()
                if server:
                    server.shutdown()
                    server.server_close()
                if thread:
                    thread.join(2)

        run("visual_topics_and_sample" if sample else "visual_topics", topics)
    if mcp:

        def tools():
            init = rpc(mcp, "initialize")
            listing = rpc(mcp, "tools/list")["tools"]
            selected = next(t for t in listing if t["name"] == NAME)
            need(
                selected["type"] == "processor"
                and selected["inputSchema"]["x-workbench"]["protocol"]
                == "motus.workbench/1",
                "CARD_CONTRACT_MISMATCH",
            )
            info = rpc(
                mcp, "tools/call", dict(name=NAME, arguments=dict(action="info"))
            )
            info = loads(info["content"][0]["text"])
            need(info["motion_capability"] is False, "UNEXPECTED_CONTROL_CAPABILITY")
            need(info["piper_id"] == config["piper_id"], "PIPER_BINDING_MISMATCH")
            return dict(server=init["serverInfo"], tool=NAME, info=info)

        run("mcp", tools)
    if core and mcp:

        def core_card():
            listing = http(core.rstrip("/") + "/api/mcp", token_file=token_file, ca=ca)[
                "data"
            ]
            matches = [
                m for m in listing if m.get("url", "").rstrip("/") == mcp.rstrip("/")
            ]
            need(len(matches) == 1, "CARD_NOT_REGISTERED_OR_AMBIGUOUS")
            need(
                matches[0].get("category") == "perception",
                "CORE_CATEGORY_NOT_PERCEPTION",
            )
            result = http(
                core.rstrip("/") + "/api/mcp/" + matches[0]["id"] + "/call?timeout_s=5",
                dict(tool=NAME, arguments=dict(action="info")),
                token_file=token_file,
                ca=ca,
            )
            need(result.get("code") == 200, "CORE_CALL_FAILED")
            info = loads(result["data"][0]["text"])
            need(info.get("piper_id") == config["piper_id"], "PIPER_BINDING_MISMATCH")
            return dict(mcp_id=matches[0]["id"], piper_id=info["piper_id"])

        run("core_to_card", core_card)

        def frontend():
            canvas = http(
                core.rstrip("/") + "/js/canvas.js",
                token_file=token_file,
                ca=ca,
                raw=True,
            )
            wb = http(
                core.rstrip("/") + "/js/workbench.js",
                token_file=token_file,
                ca=ca,
                raw=True,
            )
            need(
                "x-workbench" in canvas and "motus.workbench/1" in wb,
                "DASHBOARD_WORKBENCH_UNSUPPORTED",
            )
            return dict(workbench="motus.workbench/1", browser_interaction_tested=False)

        run("dashboard_interfaces", frontend)
    return dict(
        state="READY FOR ONSITE TEST" if checks and all(checks.values()) else "BLOCKED",
        checks=checks,
        details=details,
        failures=failures,
        python=sys.version,
        platform=platform.platform(),
        checked_at=time.time(),
        scope="read_only_no_robot_owner",
        real_visual_sample_checked=sample
        and checks.get("visual_topics_and_sample", False),
        browser_test=False,
        physical_result_verified=False,
    )
