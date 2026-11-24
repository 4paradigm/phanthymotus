"""Opt-in real ROS2 DDS, MCP HTTP and official SCRFD/recognition inference.

Run separately from perception/tests (which installs ROS doubles):
  PRIVACY_VIEW_LIVE=1 PRIVACY_TEST_MODELS=<isolated-cache> \
    PYTHONPATH=perception python -m pytest perception/integration_tests -v

Use a dedicated ROS_DOMAIN_ID and a loopback-only DDS transport profile.
No robot, camera device, external Core registration or production ports.
The model-cache guard is relocated *only in this harness*, in both parent
and spawned worker. Hash verification, model/session code and IPC are real.
"""

import hashlib
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("PRIVACY_VIEW_LIVE") != "1", reason="opt-in real DDS/model test"
)


def _allow_test_cache():
    from utils import model_downloader as md

    root = str(Path(os.environ["PRIVACY_TEST_MODELS"]).resolve())
    md.require_models_subpath.__defaults__ = (root,)


def _local_worker_main(*args):
    # No ROS imports in this module's top level or in the worker entry point.
    _allow_test_cache()
    from plugins.ort_worker import _ort_worker_main

    _ort_worker_main(*args)


def wait_for(predicate, timeout=12):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.02)
    assert predicate(), "live integration condition timed out"


@pytest.fixture(scope="module")
def models():
    from plugins import ort_worker as ow
    from plugins.face_runtime import DEFAULT_FACE_MODEL
    from utils.model_downloader import ensure_face_model

    _allow_test_cache()
    root = Path(os.environ["PRIVACY_TEST_MODELS"])
    model_dir = root / DEFAULT_FACE_MODEL
    ensure_face_model(str(model_dir), bundle=DEFAULT_FACE_MODEL)
    original = ow._ort_worker_main
    ow._ort_worker_main = _local_worker_main
    yield str(model_dir)
    ow.get_worker().close()
    ow._ort_worker_main = original


def synthetic_jpeg():
    """A drawing, with no real-person pixels or test assets on disk."""
    import cv2
    import numpy as np

    image = np.full((360, 640, 3), (150, 170, 180), np.uint8)
    for x in (180, 436):
        cv2.ellipse(image, (x, 180), (62, 85), 0, 0, 360, (130, 180, 220), -1)
        cv2.ellipse(image, (x, 110), (60, 25), 0, 180, 360, (20, 25, 35), -1)
        for eye in (-22, 22):
            cv2.ellipse(image, (x + eye, 165), (13, 6), 0, 0, 360, (240, 240, 240), -1)
            cv2.circle(image, (x + eye, 165), 5, (20, 20, 20), -1)
            cv2.line(image, (x + eye - 12, 150), (x + eye + 12, 148), (30, 30, 30), 3)
        cv2.line(image, (x, 170), (x - 5, 200), (95, 135, 170), 3)
        cv2.ellipse(image, (x, 220), (23, 7), 0, 0, 180, (60, 60, 140), 3)
    from plugins.privacy_runtime import encode_jpeg

    return encode_jpeg(image)


@pytest.fixture
def live(models):
    import main
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import String
    from utils.qos import CAMERA_QOS

    rclpy.init()
    executor = MultiThreadedExecutor(num_threads=4)
    bundle = main.PerceptionBundle(
        {
            "plugins": {
                "privacy_view": {"enabled": True, "model_dir": models, "device": "cpu"},
                "echo_cancellation": {"enabled": False},
                "audio_mixer": {"enabled": False},
            }
        },
        executor,
    )
    main._bundle = bundle
    server = main.ThreadingHTTPServer(("127.0.0.1", 0), main.make_handler())
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    source = Node("privacy_live_synthetic_source")
    executor.add_node(source)
    spin_errors = []

    def spin():
        try:
            executor.spin()
        except Exception as error:  # noqa: BLE001 - record unexpected executor failure
            spin_errors.append(error)

    spin_thread = threading.Thread(target=spin, daemon=True)
    server_thread.start()
    spin_thread.start()
    publishers = {}

    def rpc(method, params=None):
        body = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
        ).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/mcp",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=15) as response:
            return json.load(response)["result"]

    def call(action, key="live_a", **args):
        result = rpc(
            "tools/call",
            {
                "name": "privacy_view",
                "arguments": {"action": action, "instance_id": key, **args},
            },
        )
        return json.loads(result["content"][0]["text"])

    def send(
        topic="/privacy_test/camera_a",
        data=None,
        stamp=None,
        frame_id="synthetic_optical",
    ):
        if topic not in publishers:
            publishers[topic] = source.create_publisher(
                CompressedImage, topic, CAMERA_QOS
            )
        msg = CompressedImage()
        now = time.time_ns() if stamp is None else stamp
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(now, 10**9)
        msg.header.frame_id = frame_id
        msg.format, msg.data = "jpeg", synthetic_jpeg() if data is None else data
        publishers[topic].publish(msg)
        return msg

    def outputs(key="live_a", topic="/privacy_test/camera_a"):
        info = call("info", key, input_topic=topic)
        images, statuses = [], []
        source.create_subscription(
            CompressedImage, info["topic_out"][0]["topic"], images.append, CAMERA_QOS
        )
        source.create_subscription(
            String,
            info["topic_out"][1]["topic"],
            lambda msg: statuses.append(json.loads(msg.data)),
            CAMERA_QOS,
        )
        return images, statuses

    class Context:
        pass

    ctx = Context()
    ctx.call, ctx.rpc, ctx.send, ctx.outputs = call, rpc, send, outputs
    ctx.bundle, ctx.source, ctx.executor, ctx.models = bundle, source, executor, models
    ctx.mcp_url = f"http://127.0.0.1:{server.server_port}/mcp"
    ctx.plugin = bundle._plugins[0]
    try:
        yield ctx
    finally:
        bundle.shutdown()
        wait_for(lambda: not ctx.plugin._nodes)
        server.shutdown()
        server.server_close()
        server_thread.join(2)
        executor.shutdown()
        spin_thread.join(2)
        source.destroy_node()
        rclpy.shutdown()
        assert not spin_thread.is_alive() and not server_thread.is_alive()
        assert not spin_errors


def test_real_mcp_dds_model_chain(live):
    assert live.rpc("initialize")["serverInfo"]["name"] == "perception-bundle"
    tool = live.rpc("tools/list")["tools"][0]
    assert tool["name"] == "privacy_view" and tool["multiInstance"]
    images, statuses = live.outputs()
    assert live.call("start", input_topic="/privacy_test/camera_a")["state"] in (
        "loading",
        "running",
    )
    wait_for(lambda: live.call("info")["model_ready"])
    for _ in range(20):
        sent = live.send(frame_id="")
        if images:
            break
        time.sleep(0.1)
    wait_for(lambda: images and any(s["image_published"] for s in statuses))
    from plugins.privacy_runtime import decode_jpeg

    assert decode_jpeg(bytes(images[-1].data)).shape == (360, 640, 3)
    assert images[-1].header.frame_id == sent.header.frame_id
    assert any(
        s.get("source", {}).get("stamp_ns")
        == images[-1].header.stamp.sec * 10**9 + images[-1].header.stamp.nanosec
        for s in statuses
    )
    from plugins import ort_worker

    worker = ort_worker.get_worker()
    key = next(iter(worker._service_refs))
    assert worker.call(key, "describe")["recognition_loaded"] is False
    reply = worker.call(
        key,
        "detect",
        {
            "image_bytes": synthetic_jpeg(),
            "frame_token": "pixel_audit",
            "det_thresh": 0.5,
        },
    )
    assert reply["faces"]
    decoded = decode_jpeg(bytes(images[-1].data))
    for face in reply["faces"]:
        x1, y1, x2, y2 = face["bbox"]
        # Check the interior, away from JPEG ringing at rectangle boundaries.
        left, right = round((3 * x1 + x2) / 4), round((x1 + 3 * x2) / 4)
        top, bottom = round((3 * y1 + y2) / 4), round((y1 + 3 * y2) / 4)
        assert decoded[top:bottom, left:right].max() <= 5


def test_real_dds_corrupt_frame_and_timestamp_fail_closed(live):
    images, statuses = live.outputs()
    live.call("start", input_topic="/privacy_test/camera_a")
    wait_for(lambda: live.call("info")["model_ready"])
    for _ in range(8):
        live.send(data=b"invalid jpeg")
        live.send(stamp=0)
        time.sleep(0.1)
    wait_for(lambda: statuses)
    assert images == []
    assert (
        live.call("info")["metrics"]["errors"]
        + live.call("info")["metrics"]["rejected"]
        > 0
    )


def test_real_two_instances_stop_and_resource_reclamation(live):
    images_a, _ = live.outputs()
    images_b, _ = live.outputs("live_b", "/privacy_test/camera_b")
    live.call("start", input_topic="/privacy_test/camera_a")
    live.call("start", "live_b", input_topic="/privacy_test/camera_b")
    wait_for(
        lambda: (
            live.call("info")["model_ready"]
            and live.call("info", "live_b")["model_ready"]
        )
    )
    from plugins import ort_worker

    worker = ort_worker.get_worker()
    assert list(worker._service_refs.values()) == [2]
    for _ in range(20):
        live.send()
        live.send("/privacy_test/camera_b")
        if images_a and images_b:
            break
        time.sleep(0.1)
    wait_for(lambda: images_a and images_b)
    output_a = live.plugin._nodes["live_a"].topics[0]["topic"]
    output_b = live.plugin._nodes["live_b"].topics[0]["topic"]
    live.call("stop")
    wait_for(lambda: "live_a" not in live.plugin._nodes)
    wait_for(lambda: live.source.count_publishers(output_a) == 0)
    time.sleep(0.2)  # allow DDS messages already in transport to drain
    count_a, count_b = len(images_a), len(images_b)
    for _ in range(5):
        live.send()
        live.send("/privacy_test/camera_b")
        time.sleep(0.1)
    assert len(images_a) == count_a and len(images_b) > count_b
    live.call("stop", "live_b")
    wait_for(lambda: not live.plugin._nodes)
    wait_for(lambda: live.source.count_publishers(output_b) == 0
             and live.source.count_subscribers("/privacy_test/camera_a") == 0
             and live.source.count_subscribers("/privacy_test/camera_b") == 0)
    assert worker._service_refs == {} and worker._services == {}
    assert not any(t.name.startswith("privacy_view_") for t in threading.enumerate())


def test_real_model_recognition_and_detection_lease_recovery(models):
    import numpy as np
    from plugins import ort_worker
    from plugins.face_proxy import FaceServiceProxy

    privacy = FaceServiceProxy(model_dir=models, device="cpu")
    recognition = FaceServiceProxy(model_dir=models, device="cpu")
    worker = ort_worker.get_worker()
    data = synthetic_jpeg()
    try:
        assert privacy._service_key == recognition._service_key
        reply = privacy.detect(data, "synthetic", timeout=10)
        assert reply["complete"] and reply["sha256"] == hashlib.sha256(data).hexdigest()
        shape, _faces = recognition.recognise(data)
        assert shape == (360, 640)
        embeddings = recognition.embed_aligned([data])
        assert len(embeddings[0]) == 512 and np.isfinite(embeddings[0]).all()
        with pytest.raises(ort_worker.OrtWorkerError):
            privacy.detect(data, "timed-out-frame", timeout=0.00001)
        assert (
            privacy.detect(data, "after-timeout", timeout=10)["frame_token"]
            == "after-timeout"
        )
        privacy.close()
        assert recognition.recognise(data)[0] == (360, 640)
        worker._proc.kill()
        worker._proc.join(3)
        assert recognition.recognise(data)[0] == (360, 640)
    finally:
        privacy.close()
        recognition.close()
    assert worker._services == {} and worker._service_refs == {}


def test_real_face_card_coexists_and_each_stop_preserves_other(live):
    from plugins import ort_worker
    from plugins.face import FaceRecognitionPlugin, _close_quietly

    face = FaceRecognitionPlugin(
        {
            "model_dir": live.models,
            "device": "cpu",
            "db_dir": str(Path(live.models).parent / "synthetic_roster"),
            "store_sample_crops": False,
            "detect_fps": 10,
        },
        live.executor,
    )
    data = synthetic_jpeg()
    try:
        face.dispatch(
            "face_recognition",
            {
                "action": "start",
                "instance_id": "recognition",
                "input_topic": "/privacy_test/camera_a",
            },
        )
        wait_for(lambda: face._engine_state in ("ready", "error"))
        assert face._engine_state == "ready", face._load_error
        assert face._engine.analyzer.recognise(data)[0] == (360, 640)
        live.call("start", input_topic="/privacy_test/camera_a")
        wait_for(lambda: live.call("info")["model_ready"])
        worker = ort_worker.get_worker()
        assert list(worker._service_refs.values()) == [2]
        images, _ = live.outputs()
        for _ in range(12):
            live.send()
            time.sleep(0.05)
        wait_for(lambda: images)
        face.dispatch(
            "face_recognition", {"action": "stop", "instance_id": "recognition"}
        )
        before = len(images)
        for _ in range(6):
            live.send()
            time.sleep(0.05)
        wait_for(lambda: len(images) > before)
        face.dispatch(
            "face_recognition",
            {
                "action": "start",
                "instance_id": "recognition",
                "input_topic": "/privacy_test/camera_a",
            },
        )
        live.call("stop")
        wait_for(lambda: not live.plugin._nodes)
        assert face._engine.analyzer.recognise(data)[0] == (360, 640)
        assert list(worker._service_refs.values()) == [1]
    finally:
        face.dispatch("face_recognition", {"action": "stop"})
        # The existing recognition Card intentionally caches its engine for
        # photo/roster actions after stream stop. Release that cache at test exit.
        if face._engine is not None:
            _close_quietly(face._engine)
    assert ort_worker.get_worker()._service_refs == {}


def test_large_frame_worker_crashes_reclaim_ipc_threads(models):
    from plugins import ort_worker

    worker = ort_worker.get_worker()
    previous = {id(t) for t in threading.enumerate() if t.name == "QueueFeederThread"}
    for _ in range(4):
        worker._ensure_started()
        # Also cover an idle worker killed outside the request timeout path.
        old_cmd, old_res = worker._cmd_q, worker._res_q
        worker._proc.kill()
        worker._proc.join(5)
        worker._ensure_started()
        assert old_cmd._reader.closed and old_cmd._writer.closed
        assert old_res._reader.closed and old_res._writer.closed
        with worker._lock, pytest.raises(ort_worker.OrtWorkerError):
            worker._request(
                ("call", "absent", "detect", {"image_bytes": b"x" * (8 * 1024 * 1024)}),
                timeout=0.00001,
            )
        wait_for(
            lambda: all(
                id(t) in previous
                for t in threading.enumerate()
                if t.name == "QueueFeederThread"
            )
        )
    assert worker._service_refs == {} and worker._services == {}


def test_measured_video_performance(live, tmp_path):
    import platform

    import numpy as np
    import psutil
    from plugins import ort_worker
    from plugins.face_proxy import FaceServiceProxy

    image = synthetic_jpeg()
    _images_a, states_a = live.outputs()
    _images_b, states_b = live.outputs("live_b", "/privacy_test/camera_b")
    live.call("config", detect_fps=30)
    live.call("start", input_topic="/privacy_test/camera_a")
    wait_for(lambda: live.call("info")["model_ready"])
    worker = ort_worker.get_worker()
    parent, child = psutil.Process(), psutil.Process(worker._proc.pid)
    measurements = []
    recognition_count = [0]

    def sample(name, sources, seconds=3):
        start_count = [len(states_a), len(states_b)]
        cpu_before = [
            p.cpu_times().user + p.cpu_times().system for p in (parent, child)
        ]
        started = time.monotonic()
        recognition_before = recognition_count[0]
        sent = 0
        peak_rss = [0, 0]
        while time.monotonic() - started < seconds:
            tick = time.monotonic()
            for topic in sources:
                live.send(topic, data=image)
                sent += 1
            for i, process in enumerate((parent, child)):
                peak_rss[i] = max(peak_rss[i], process.memory_info().rss)
            time.sleep(max(0, 1 / 30 - (time.monotonic() - tick)))
        duration = time.monotonic() - started
        records = [
            s
            for group, start in zip((states_a, states_b), start_count)
            for s in group[start:]
            if s.get("image_published")
        ]
        latencies = {
            key: {
                "p50": float(
                    np.percentile([s["latency_ms"][key] for s in records], 50)
                ),
                "p95": float(
                    np.percentile([s["latency_ms"][key] for s in records], 95)
                ),
            }
            for key in (
                "decode",
                "detect",
                "redact",
                "encode",
                "processing",
                "receive_to_publish",
            )
        }
        cpu_after = [p.cpu_times().user + p.cpu_times().system for p in (parent, child)]
        measurements.append(
            {
                "scenario": name,
                "duration_s": duration,
                "sent_frames": sent,
                "output_frames": len(records),
                "throughput_fps": len(records) / duration,
                "recognition_calls": recognition_count[0] - recognition_before,
                "recognition_fps": (recognition_count[0] - recognition_before)
                / duration,
                "latency_ms": latencies,
                "cpu_percent_of_one_core": [
                    (b - a) / duration * 100 for a, b in zip(cpu_before, cpu_after)
                ],
                "peak_rss_bytes": peak_rss,
                "faces_detected": sorted({s["faces_detected"] for s in records}),
                "instance_stats": {
                    key: node.snapshot()["metrics"]
                    for key, node in live.plugin._nodes.items()
                },
            }
        )
        assert records

    sample("single_640x360_30hz", ["/privacy_test/camera_a"])
    live.call("config", "live_b", detect_fps=30)
    live.call("start", "live_b", input_topic="/privacy_test/camera_b")
    wait_for(lambda: live.call("info", "live_b")["model_ready"])
    sample(
        "two_640x360_30hz_each", ["/privacy_test/camera_a", "/privacy_test/camera_b"]
    )
    recognition = FaceServiceProxy(model_dir=live.models, device="cpu")
    halt = threading.Event()
    recognition_errors = []

    def recognise():
        try:
            while not halt.wait(0.1):
                recognition.recognise(image, embed_all=True)
                recognition_count[0] += 1
        except Exception as error:  # noqa: BLE001 - assert on background failure below
            recognition_errors.append(type(error).__name__)

    thread = threading.Thread(target=recognise)
    thread.start()
    try:
        sample(
            "two_plus_recognition_interval_0.1s",
            ["/privacy_test/camera_a", "/privacy_test/camera_b"],
        )
    finally:
        halt.set()
        thread.join(5)
        recognition.close()
    assert not thread.is_alive() and not recognition_errors
    target = (
        Path(os.getenv("PRIVACY_TEST_EVIDENCE", str(tmp_path))) / "performance.json"
    )
    target.write_text(
        json.dumps(
            {
                "platform": platform.platform(),
                "architecture": platform.machine(),
                "python": platform.python_version(),
                "model": "SCRFD-2.5G, CPUExecutionProvider, 2 ORT threads",
                "source": "programmatic drawing, no real people",
                "measurements": measurements,
            },
            indent=2,
        )
    )


def test_real_core_canvas_and_browser(live, tmp_path):
    """Real Core subprocess, real MCP HTTP client, real DDS->WS and browser."""
    import asyncio
    import secrets
    import subprocess
    import sys

    import httpx

    class SyntheticCamera:
        PREFIX = "privacy_test_camera"

        def get_tools(self):
            return [
                {
                    "name": self.PREFIX,
                    "type": "sensor",
                    "description": "Synthetic JPEG source",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["start", "stop", "info"],
                            }
                        },
                    },
                    "topic_out": [
                        {"topic": "/privacy_test/camera_a", "format": "image/jpeg"}
                    ],
                }
            ]

        def dispatch(self, name, args):
            return {
                "state": "idle" if args.get("action") == "stop" else "running",
                "topic_out": self.get_tools()[0]["topic_out"],
            }

    live.bundle._plugins.append(SyntheticCamera())
    root = Path(__file__).resolve().parents[2] / "agent-core"
    runtime = tmp_path / "core"
    runtime.mkdir()
    (runtime / "web").symlink_to(root / "web", target_is_directory=True)
    token = secrets.token_urlsafe(24)
    (runtime / ".env").write_text("ACCESS_TOKEN=" + token + "\n")
    log = open(runtime / "core.log", "w")  # noqa: SIM115 - closed after child teardown
    proc = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).with_name("core_host.py")),
            "--core-root",
            str(root),
            "--runtime",
            str(runtime),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    halt = threading.Event()

    def produce():
        while not halt.wait(0.05):
            live.send()

    producer = threading.Thread(target=produce, daemon=True)

    async def exercise():
        from playwright.async_api import async_playwright

        end = time.monotonic() + 20
        while (
            not (runtime / "port.json").exists()
            and proc.poll() is None
            and time.monotonic() < end
        ):
            await asyncio.sleep(0.05)
        assert (runtime / "port.json").exists(), (runtime / "core.log").read_text()[
            -4000:
        ]
        port = json.loads((runtime / "port.json").read_text())["port"]
        base = f"http://127.0.0.1:{port}"
        async with httpx.AsyncClient(
            base_url=base,
            headers={"Authorization": "Bearer " + token},
            trust_env=False,
            timeout=20,
        ) as client:
            added = (
                await client.post(
                    "/api/mcp",
                    json={
                        "name": "Privacy software test",
                        "url": live.mcp_url,
                        "category": "perception",
                    },
                )
            ).json()
            mcp_id = added["data"]["id"]
            for _ in range(100):
                registry = (await client.get("/api/mcp")).json()["data"]
                entry = next((e for e in registry if e["id"] == mcp_id), {})
                if any(t.get("name") == "privacy_view" for t in entry.get("tools", [])):
                    break
                await asyncio.sleep(0.05)
            assert any(t.get("name") == "privacy_view" for t in entry.get("tools", []))
            response = await client.put(
                f"/api/canvas/tool-config/{mcp_id}/privacy_view/core_privacy",
                json={"mode": "opaque"},
            )
            assert response.status_code == 200
            session = "privacy_acceptance"
            assert (
                await client.post(
                    "/api/canvas/claim-edit", json={"session_id": session}
                )
            ).status_code == 200
            camera = {
                "id": "core_camera",
                "mcpId": mcp_id,
                "toolName": "privacy_test_camera",
                "driverName": "Synthetic camera",
                "x": 80,
                "y": 80,
                "topicOut": [
                    {"topic": "/privacy_test/camera_a", "format": "image/jpeg"}
                ],
            }
            card = {
                "id": "core_privacy",
                "mcpId": mcp_id,
                "toolName": "privacy_view",
                "driverName": "Privacy",
                "x": 420,
                "y": 80,
                "topicIn": [{"format": "image/jpeg"}],
                "topicOut": [{"format": "image/jpeg"}, {"format": "data/json"}],
            }
            layout = {
                "cards": [camera, card],
                "connections": [
                    {
                        "fromCardId": "core_camera",
                        "toCardId": "core_privacy",
                        "fromPortIdx": 0,
                        "toPortIdx": 0,
                        "topic": "/privacy_test/camera_a",
                        "format": "image/jpeg",
                    }
                ],
                "session_id": session,
            }
            assert (
                await client.post("/api/canvas/layout", json=layout)
            ).status_code == 200
            producer.start()
            assert (await client.post("/api/config/start-project")).status_code == 200
            for _ in range(150):
                if (
                    "core_privacy" in live.plugin._nodes
                    and live.plugin._nodes["core_privacy"].snapshot()["metrics"][
                        "published"
                    ]
                ):
                    break
                await asyncio.sleep(0.05)
            assert "core_privacy" in live.plugin._nodes, (
                runtime / "core.log"
            ).read_text()[-4000:]
            assert (
                live.plugin._nodes["core_privacy"].snapshot()["metrics"]["published"]
                > 0
            )
            output = live.plugin._nodes["core_privacy"].topics[0]["topic"]
            evidence = Path(os.getenv("PRIVACY_TEST_EVIDENCE", str(tmp_path)))
            evidence.mkdir(parents=True, exist_ok=True)
            (evidence / "core-topics-probe.json").write_text(
                json.dumps((await client.get("/api/topics/status")).json(), indent=2)
            )
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(
                    headless=True, args=["--no-sandbox", "--no-proxy-server"]
                )
                page = await browser.new_page(viewport={"width": 1440, "height": 1000})
                console = []
                page.on("console", lambda message: console.append(message.text[:300]))
                await page.add_init_script(
                    "localStorage.setItem('phanthy_access_token', "
                    + json.dumps(token)
                    + ");"
                )
                await page.goto(base)
                await page.wait_for_selector(".canvas-card", timeout=15000)
                assert await page.locator(".canvas-card").count() >= 2
                # Use the actual generic image detail renderer, fed through
                # production Core DDS->inspection->WebSocket transport.
                await page.evaluate(
                    """async topic => { const m = await import('/js/detail-panel.js'); m.showTopicDetail(topic, 'image/jpeg'); }""",
                    output,
                )
                try:
                    await page.wait_for_function(
                        """() => [...document.images].some(i => i.naturalWidth === 640 && i.naturalHeight === 360)""",
                        timeout=15000,
                    )
                finally:
                    (evidence / "browser-console.json").write_text(
                        json.dumps(console, indent=2)
                    )
                    (evidence / "core-topics-final-probe.json").write_text(
                        json.dumps(
                            (await client.get("/api/topics/status")).json(), indent=2
                        )
                    )
                    (evidence / "publication-probe.json").write_text(
                        json.dumps(
                            {
                                "source_subscribers": live.source.count_subscribers(
                                    "/privacy_test/camera_a"
                                ),
                                "privacy_subscribers": live.plugin._nodes[
                                    "core_privacy"
                                ]._pub.get_subscription_count(),
                                "privacy_metrics": live.plugin._nodes[
                                    "core_privacy"
                                ].snapshot()["metrics"],
                            },
                            indent=2,
                        )
                    )
                    await page.screenshot(path=str(evidence / "canvas-attempt.png"))
                evidence = Path(os.getenv("PRIVACY_TEST_EVIDENCE", str(tmp_path)))
                evidence.mkdir(parents=True, exist_ok=True)
                await page.screenshot(path=str(evidence / "canvas-privacy.png"))
                (evidence / "core-browser-receipt.json").write_text(
                    json.dumps(
                        {
                            "real_core": True,
                            "production_lifespan": False,
                            "mcp_discovery": True,
                            "canvas_saved_connection": True,
                            "project_start": True,
                            "jpeg_browser_size": [640, 360],
                            "privacy_mode": live.plugin._nodes[
                                "core_privacy"
                            ]._cfg.mode,
                        },
                        indent=2,
                    )
                )
                await browser.close()
            assert (await client.post("/api/config/stop-project")).status_code == 200
            for _ in range(100):
                if "core_privacy" not in live.plugin._nodes:
                    break
                await asyncio.sleep(0.05)
            assert "core_privacy" not in live.plugin._nodes

            # Start once more through the real MCP route, then delete the
            # saved card. Core's stop_removed_cards must reclaim it.
            started = await client.post(
                f"/api/mcp/{mcp_id}/call",
                json={
                    "tool": "privacy_view",
                    "arguments": {
                        "action": "start",
                        "instance_id": "core_privacy",
                        "input_topic": "/privacy_test/camera_a",
                    },
                },
            )
            assert started.status_code == 200
            await client.post(
                "/api/canvas/layout",
                json={**layout, "cards": [camera], "connections": []},
            )
            for _ in range(100):
                if "core_privacy" not in live.plugin._nodes:
                    break
                await asyncio.sleep(0.05)
            assert "core_privacy" not in live.plugin._nodes

            # Exercise the ordinary browser editing path too: drag the live
            # discovered Card, save its instance config, connect its input,
            # and start/stop with the existing project button.
            await client.post("/api/canvas/release-edit", json={"session_id": session})
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(
                    headless=True, args=["--no-sandbox", "--no-proxy-server"]
                )
                page = await browser.new_page(viewport={"width": 1440, "height": 1000})
                await page.add_init_script(
                    "localStorage.setItem('phanthy_access_token', "
                    + json.dumps(token)
                    + ");"
                )
                await page.goto(base)
                await page.wait_for_selector("#canvas-claim-btn")
                await page.locator("#canvas-claim-btn").click()
                await page.wait_for_selector("#canvas-release-btn")
                source_card = page.locator(
                    '.sidebar-tool-card[data-tool-name="privacy_view"]'
                )
                await source_card.drag_to(
                    page.locator("#canvas-area"), target_position={"x": 580, "y": 320}
                )
                target = page.locator(".canvas-card").filter(
                    has=page.locator(".canvas-card-instance-cfg-btn")
                )
                await target.wait_for()
                new_id = await target.get_attribute("data-card-id")
                field = page.locator('#tool-config-body [data-key="mode"]')
                if not await field.is_visible():
                    await target.locator(".canvas-card-instance-cfg-btn").click()
                await field.select_option("pixelate")
                await page.locator("#tool-config-save").click()
                await page.wait_for_selector("#tool-config-overlay", state="hidden")
                camera_port = page.locator(
                    '[data-card-id="core_camera"] .canvas-port[data-dir="out"]'
                ).first
                input_port = page.locator(
                    f'[data-card-id="{new_id}"] .canvas-port[data-dir="in"]'
                ).first
                first, second = (
                    await camera_port.bounding_box(),
                    await input_port.bounding_box(),
                )
                await page.mouse.move(
                    first["x"] + first["width"] / 2, first["y"] + first["height"] / 2
                )
                await page.mouse.down()
                await page.mouse.move(
                    second["x"] + second["width"] / 2,
                    second["y"] + second["height"] / 2,
                    steps=12,
                )
                await page.mouse.up()
                for _ in range(100):
                    saved = (await client.get("/api/canvas/layout")).json()["data"]
                    if any(
                        c.get("toCardId") == new_id
                        for c in saved.get("connections", [])
                    ):
                        break
                    await asyncio.sleep(0.05)
                assert any(
                    c.get("toCardId") == new_id for c in saved.get("connections", [])
                )
                await page.locator("#canvas-project-toggle").click()
                for _ in range(150):
                    node = live.plugin._nodes.get(new_id)
                    if node and node.snapshot()["metrics"]["published"]:
                        break
                    await asyncio.sleep(0.05)
                assert (
                    node
                    and node._cfg.mode == "pixelate"
                    and node.snapshot()["metrics"]["published"] > 0
                )
                await page.screenshot(path=str(evidence / "canvas-drag-config.png"))
                await page.locator("#canvas-project-toggle").click()
                for _ in range(100):
                    if new_id not in live.plugin._nodes:
                        break
                    await asyncio.sleep(0.05)
                assert new_id not in live.plugin._nodes
                await browser.close()
            (evidence / "core-browser-receipt.json").write_text(
                json.dumps(
                    {
                        "real_core": True,
                        "production_lifespan": False,
                        "mcp_discovery": True,
                        "canvas_saved_connection": True,
                        "project_start_stop": True,
                        "removed_card_reclaimed": True,
                        "browser_drag": True,
                        "browser_connect": True,
                        "browser_config": "pixelate",
                        "browser_start_stop": True,
                        "jpeg_browser_size": [640, 360],
                    },
                    indent=2,
                )
            )

    try:
        asyncio.run(exercise())
    finally:
        halt.set()
        if producer.is_alive():
            producer.join(2)
        proc.terminate()
        proc.wait(timeout=10)
        log.close()
        (runtime / ".env").unlink(missing_ok=True)
