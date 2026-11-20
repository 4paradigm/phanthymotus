"""Real Core FastAPI + SQLite + MCP + Chromium/native canvas workbench.

Default mode replaces the unrelated Core process lifespan. --full-core-lifespan
runs normal startup/shutdown and its ROS bridge with LLM/peer execution disabled
by configuration. HTTP/auth/MCP/SQLite/Dashboard/workbench code is real in both modes.
Visual input is explicitly synthetic BGR frames through the production CPU
image adapter. Never imports or starts a PiPER motion owner.
"""

import argparse
from contextlib import asynccontextmanager
import hashlib
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perception"))


def request(url, body=None):
    from plugins.goal_verifier.contracts import canonical

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(
        url,
        data=canonical(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with opener.open(req, timeout=8) as response:
        return json.load(response)


def run(
    output,
    core_root=None,
    *,
    full_core=False,
    assets_root=None,
    real_image=None,
    ros_input=False,
):
    from plugins.goal_verifier.contracts import NAME, canonical
    from plugins.goal_verifier.plugin import GoalVerifierPlugin
    from plugins.goal_verifier.server import handler

    core_root = Path(core_root or ROOT / "agent-core").resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    assets_root = Path(assets_root or ROOT / ".cache").resolve()
    cfg = dict(
        piper_id="SYNTHETIC-PIPER-BROWSER",
        sources={
            "image": dict(
                source_id="image",
                kind="buffered_image",
                camera_id="SYNTHETIC-CAMERA",
                frame_id="optical",
                view_id="synthetic-view",
                distortion="rectified",
                view_mode="fixed",
                fixed_view_record="SYNTHETIC-TEST-ONLY",
                allow_test_data=True,
                detector="color",
                classes=[
                    dict(
                        class_name="red_piece",
                        color="red",
                        hsv_min=[0, 150, 150],
                        hsv_max=[10, 255, 255],
                        min_area_px=25,
                    )
                ],
            )
        },
    )
    if ros_input:
        cfg["sources"]["image"].update(
            kind="ros_image",
            topic="/piper_goal_e2e_" + str(os.getpid()),
            domain_id=int(os.environ["ROS_DOMAIN_ID"]),
            message_type="raw",
            stamp_semantics="capture",
            capture_time_record="SYNTHETIC frame generation timestamp",
            provenance="synthetic",
        )
    card = GoalVerifierPlugin(cfg)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(card))
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    mcp = f"http://127.0.0.1:{server.server_port}/mcp"
    stop = threading.Event()
    mode = {"x": 0.5, "visible": True}

    def camera():
        import numpy as np

        context = node = publisher = None
        try:
            if ros_input:
                import rclpy
                from rclpy.context import Context
                from rclpy.node import Node
                from rclpy.qos import QoSProfile, ReliabilityPolicy
                from sensor_msgs.msg import Image

                context = Context()
                rclpy.init(context=context, domain_id=int(os.environ["ROS_DOMAIN_ID"]))
                node = Node(
                    "goal_synthetic_image_e2e_" + str(os.getpid()), context=context
                )
                publisher = node.create_publisher(
                    Image,
                    cfg["sources"]["image"]["topic"],
                    QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
                )
            while not stop.wait(0.04):
                image = np.zeros((400, 600, 3), np.uint8)
                if mode["visible"]:
                    x = int(mode["x"] * 600)
                    image[180:220, x - 20 : x + 20] = (0, 0, 255)
                if ros_input:
                    msg = Image()
                    stamp = time.time()
                    msg.header.frame_id = "optical"
                    msg.header.stamp.sec = int(stamp)
                    msg.header.stamp.nanosec = int(stamp % 1 * 1e9)
                    msg.height = 400
                    msg.width = 600
                    msg.step = 1800
                    msg.encoding = "bgr8"
                    msg.data = image.tobytes()
                    publisher.publish(msg)
                else:
                    card.sources.feed_image(
                        "image", image, time.time(), provenance="synthetic"
                    )
        finally:
            if node:
                node.destroy_node()
            if context and context.ok():
                context.shutdown()

    producer = threading.Thread(target=camera, daemon=True)
    producer.start()
    core = None
    core_thread = None
    browser = None
    try:
        # All Core writes go to this NEW private directory, never the old tree.
        work = output / "core-runtime"
        work.mkdir()
        (work / "web").symlink_to(core_root / "web", target_is_directory=True)
        os.environ["DB_PATH"] = str(work / "data.db")
        os.chdir(work)
        sys.path.insert(0, str(core_root / "src"))
        import config

        config.main["core"] = {
            **config.main.get("core", {}),
            "configured": True,
            "main_loop_enable": False,
            "auto_start": False,
        }
        config.main["services"] = {
            **config.main.get("services", {}),
            "resource_center": {"url": "http://127.0.0.1:1"},
        }
        config.main["peer_settings"] = {
            **config.main.get("peer_settings", {}),
            "enabled": False,
        }
        import start
        import auth
        import uvicorn

        @asynccontextmanager
        async def local_http_lifespan(app):
            # No production lifespan, DDS shim, fake routes or AST replacements.
            auth.init()
            config.main["core"] = {
                **config.main.get("core", {}),
                "configured": True,
                "main_loop_enable": False,
                "auto_start": False,
            }
            yield

        if not full_core:
            start.app.router.lifespan_context = local_http_lifespan
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        core = uvicorn.Server(
            uvicorn.Config(start.app, log_level="warning", lifespan="on")
        )
        core_thread = threading.Thread(
            target=lambda: core.run(sockets=[sock]), daemon=True
        )
        core_thread.start()
        deadline = time.monotonic() + 15
        while not core.started:
            if time.monotonic() > deadline:
                raise RuntimeError("Core startup timeout")
            time.sleep(0.05)
        base = f"http://127.0.0.1:{port}"
        mid = request(
            base + "/api/mcp",
            dict(name="Goal verifier local acceptance", url=mcp, category="perception"),
        )["data"]["id"]
        services = config.main.get("services", {})
        services["resource_center"] = {"url": base}
        config.main["services"] = services
        deadline = time.monotonic() + 8
        while True:
            listing = request(base + "/api/mcp")["data"]
            entry = next(m for m in listing if m["id"] == mid)
            if any(t["name"] == NAME for t in entry.get("tools", [])):
                break
            if time.monotonic() > deadline:
                raise RuntimeError("Core MCP discovery timeout")
            time.sleep(0.05)
        # A native, persisted canvas containing ONLY this read-only card.
        config.main["canvas_layout"] = dict(
            cards=[
                dict(
                    id="goal-acceptance-card",
                    mcpId=mid,
                    toolName=NAME,
                    driverName="Goal verifier local acceptance",
                    x=30,
                    y=30,
                )
            ],
            connections=[],
            execConnections=[],
        )
        from playwright.sync_api import sync_playwright, expect

        trace = []
        errors = []
        console_errors = []
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
            context = browser.new_context(
                viewport=dict(width=1600, height=1100),
                accept_downloads=True,
                record_video_dir=str(output / "video"),
            )
            page = context.new_page()
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on(
                "console",
                lambda msg: (
                    console_errors.append(msg.text) if msg.type == "error" else None
                ),
            )
            page.on(
                "request",
                lambda req: (
                    trace.append(
                        dict(method=req.method, url=req.url, body=req.post_data)
                    )
                    if "/api/mcp/" in req.url and "/call" in req.url
                    else None
                ),
            )
            assets = {
                "https://cdn.jsdelivr.net/npm/three@0.164/build/three.module.js": assets_root
                / "dashboard-cdn/three.module.js",
                "https://cdn.jsdelivr.net/npm/three@0.164/examples/jsm/controls/OrbitControls.js": assets_root
                / "dashboard-cdn/OrbitControls.js",
            }

            def local_assets(route):
                url = route.request.url
                if url in assets:
                    if not assets[url].is_file():
                        raise RuntimeError(
                            "Cache Dashboard CDN files before browser test: " + url
                        )
                    route.fulfill(
                        path=assets[url],
                        content_type="text/javascript",
                        headers={"Access-Control-Allow-Origin": "*"},
                    )
                elif url.startswith((base, "data:", "blob:")):
                    route.continue_()
                else:
                    route.abort()  # Cosmetic fonts/logo; no external requests.

            page.route("**/*", local_assets)
            page.goto(base, wait_until="domcontentloaded")
            page.wait_for_timeout(1500)
            page.screenshot(path=str(output / "bootstrap.png"), full_page=True)
            (output / "bootstrap.html").write_text(page.content())
            (output / "bootstrap-errors.json").write_text(json.dumps(errors, indent=2))
            (output / "layout.json").write_text(
                canonical(request(base + "/api/canvas/layout"))
            )
            card_ui = page.locator(".canvas-card").filter(
                has=page.locator(".canvas-card-tool", has_text=NAME)
            )
            expect(card_ui).to_be_visible(timeout=15000)
            # Native sidebar entry works WITHOUT enabling the project/other
            # cards. A seeded canvas proves discovery there as well.
            entry = page.locator(
                f'.sidebar-tool-card[data-tool-name="{NAME}"] .tool-readonly-workbench-btn'
            )
            expect(entry).to_be_visible()
            entry.click()
            frame = page.frame_locator("iframe")
            expect(frame.locator("#binding")).to_contain_text(
                "SYNTHETIC-PIPER", timeout=10000
            )
            if ros_input:
                frame.locator("#start").click()
                expect(frame.locator("#start")).to_be_enabled()
            frame.locator("#observe").click()
            expect(frame.locator("#class")).to_have_value("red_piece")
            expect(frame.locator("#observe")).to_be_enabled()
            expect(frame.locator("#measurement")).to_contain_text("历史关联帧")
            frame.locator("#session").click()
            expect(frame.locator("#state")).to_contain_text("session_token")
            frame.locator("#before").click()
            expect(frame.locator("#state")).to_contain_text("before_captured")
            mode["x"] = 0.75
            frame.locator("#mark").click()
            expect(frame.locator("#state")).to_contain_text("marker_id")
            frame.locator("#after").click()
            expect(frame.locator("#result")).to_contain_text("PASS", timeout=10000)
            expect(frame.locator("#after")).to_be_enabled()
            frame.locator("body").evaluate("() => window.scrollTo(0, 0)")
            page.screenshot(path=str(output / "pass.png"), full_page=True)
            # Repeated requests must return one historical result, not acquire again.
            session = next(iter(card.engine.sessions.values()))
            verification = [
                t for t in trace if t["body"] and "verify_after" in t["body"]
            ][-1]
            repeated = request(verification["url"], json.loads(verification["body"]))
            assert json.loads(repeated["data"][0]["text"])["result"] == "PASS"
            assert len(session["history"]) == 1
            mode["x"] = 0.5
            time.sleep(0.08)
            frame.locator("#after").click()
            expect(frame.locator("#result")).to_contain_text("FAIL", timeout=10000)
            expect(frame.locator("#after")).to_be_enabled()
            frame.locator("body").evaluate("() => window.scrollTo(0, 0)")
            page.screenshot(path=str(output / "fail.png"), full_page=True)
            mode["visible"] = False
            time.sleep(0.08)
            frame.locator("#after").click()
            expect(frame.locator("#result")).to_contain_text(
                "INCONCLUSIVE", timeout=10000
            )
            expect(frame.locator("#after")).to_be_enabled()
            frame.locator("body").evaluate("() => window.scrollTo(0, 0)")
            page.screenshot(path=str(output / "inconclusive.png"), full_page=True)
            frame.locator("#export").click()
            expect(frame.locator("#export-data")).to_contain_text("")
            expect(frame.locator("#state")).to_contain_text("history")
            (output / "session-evidence.json").write_text(
                frame.locator("#export-data").input_value()
            )
            frame.locator("#reset").click()
            expect(frame.locator("#result")).to_have_text("尚未验证")
            # A malformed observation is diagnosed in the actual page.
            frame.locator("summary", has_text="标准观测导入").click()
            frame.locator("#observation").fill('{"schema":"wrong"}')
            frame.locator("#submit").click()
            expect(frame.locator("#result")).to_contain_text("INCONCLUSIVE")
            expect(frame.locator("#submit")).to_be_enabled()
            if real_image:
                from goal_verifier_real_image import inspect_image

                measured = inspect_image(
                    real_image,
                    [
                        dict(
                            class_name="configured_light_region",
                            color="low_saturation_bright",
                            hsv_min=[0, 0, 180],
                            hsv_max=[179, 60, 255],
                            min_area_px=3000,
                        )
                    ],
                )
                (output / "real-image-detection.json").write_text(
                    json.dumps(measured, indent=2)
                )
                frame.locator("#observation").fill(
                    json.dumps(measured["inadmissible_observation"])
                )
                frame.locator("#submit").click()
                expect(frame.locator("#submit")).to_be_enabled()
                expect(frame.locator("#result")).to_contain_text("INCONCLUSIVE")
                expect(frame.locator("#state")).to_contain_text("ARGUMENT_SCHEMA")
                page.screenshot(
                    path=str(output / "real-image-missing-capture-rejected.png"),
                    full_page=True,
                )
            exported = json.loads((output / "session-evidence.json").read_text())
            assert [r["result"] for r in exported["history"]] == [
                "PASS",
                "FAIL",
                "INCONCLUSIVE",
            ]
            assert not errors, errors
            assert request(base + "/api/config/project-running")["running"] is False
            assert request(base + "/api/canvas/edit-status")["editor"] is None
            context.close()
            browser.close()
            browser = None
        external_dds_binds = None
        if full_core:
            import dds_isolation
            import ros2_bridge

            external_dds_binds = dds_isolation._external_dds_binds()
            if external_dds_binds or not ros2_bridge._running:
                raise RuntimeError(
                    "DDS loopback/runtime verification failed: "
                    + str(external_dds_binds)
                )
        result = dict(
            state="PASSED_REAL_BROWSER_E2E",
            core_app="actual start.app FastAPI",
            core_lifespan="normal startup and shutdown, real ROS2 bridge; LLM execution/peer disabled by configuration"
            if full_core
            else "isolated HTTP-only; DDS/LLM/cloud startup omitted",
            native_canvas=True,
            native_workbench=True,
            project_never_started=True,
            canvas_editor_not_claimed=True,
            real_chromium=True,
            real_mcp_http=True,
            external_dds_binds=external_dds_binds,
            sqlite=True,
            outcomes=["PASS", "FAIL", "INCONCLUSIVE"],
            repeat_idempotent=True,
            reset=True,
            error_display=True,
            visual_source="SYNTHETIC_BGR_THROUGH_CPU_COLOR_ADAPTER",
            real_camera=False,
            visual_transport="actual ROS2 sensor_msgs/Image"
            if ros_input
            else "in-process synthetic pixels",
            real_photograph_detection_and_missing_metadata_rejection=bool(real_image),
            robot_owner=False,
            browser_errors=errors,
            console_errors=console_errors,
            requests=trace,
            core_base_sha="990257f881af540a1949237678b71dceaab5ab1c",
            core_source_sha256={
                name: hashlib.sha256((core_root / name).read_bytes()).hexdigest()
                for name in (
                    "src/start.py",
                    "src/api/mcp_manage.py",
                    "web/js/canvas.js",
                    "web/js/workbench.js",
                    "web/js/sidebar.js",
                )
            },
        )
        (output / "report.json").write_text(json.dumps(result, indent=2))
        print(
            canonical({k: v for k, v in result.items() if k != "requests"}), flush=True
        )
        return result
    finally:
        stop.set()
        producer.join(2)
        card.stop()
        server.shutdown()
        server.server_close()
        serving.join(2)
        if browser:
            try:
                browser.close()
            except Exception:
                pass
        if core:
            core.should_exit = True
        if core_thread:
            core_thread.join(4)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--core", type=Path)
    p.add_argument("--full-core-lifespan", action="store_true")
    p.add_argument("--assets-root", type=Path)
    p.add_argument("--real-image", type=Path)
    p.add_argument("--ros-input", action="store_true")
    args = p.parse_args()
    run(
        args.output,
        args.core,
        full_core=args.full_core_lifespan,
        assets_root=args.assets_root,
        real_image=args.real_image,
        ros_input=args.ros_input,
    )
