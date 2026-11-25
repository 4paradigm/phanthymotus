"""Remote card boundaries: isolated UI, protected credentials and asynchronous jobs."""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from motion_ui_helpers import chromium_options

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plugins.motion import MotionPlugin
from plugins.motion.client import MotionClient, ServiceError


@pytest.fixture
def service():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("Authorization"), body))
            if body["action"] == "redirect":
                self.send_response(307)
                self.send_header("Location", "/credential-destination")
                self.end_headers()
                return
            if body["action"] == "failure":
                self.send_response(400)
                result = {"state": "error", "message": "failed token secret-test"}
            elif body["action"] == "large":
                self.send_response(200)
                result = {"text": "x" * 100}
            else:
                self.send_response(200)
                result = {"job": {"id": "job-1", "state": "queued"}}
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        def do_GET(self):
            requests.append((self.path, self.headers.get("Authorization"), None))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"state":"ready"}')

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", requests
    server.shutdown()
    server.server_close()
    thread.join()


def test_training_is_a_real_remote_submission_with_only_declared_arguments(service):
    endpoint, requests = service
    plugin = MotionPlugin({"endpoint": endpoint, "api_key": "secret-test"})
    result = plugin.dispatch("motion", {"action": "train", "profile_id": "g1",
                                        "dataset_id": "dataset-1", "num_envs": 64,
                                        "gpu_ids": [2, 3],
                                        "max_iterations": 1000, "command": "evil", "endpoint": "http://elsewhere"})
    assert result["job"]["state"] == "queued"
    path, auth, body = requests[0]
    assert path == "/v1/call" and auth == "Bearer secret-test"
    assert body == {"action": "train", "profile_id": "g1", "dataset_id": "dataset-1",
                    "gpu_ids": [2, 3], "num_envs": 64, "max_iterations": 1000}
    assert plugin.dispatch("motion", {"action": "info"})["busy"] is False
    plugin.stop()
    assert len(requests) == 1  # closing a page/host must not cancel a durable job


def test_workbench_is_self_contained_and_does_not_require_core_network_access():
    plugin = MotionPlugin({})
    tool = plugin.get_tools()[0]
    spec = tool["inputSchema"]["x-workbench"]
    html = plugin.dispatch("motion", {"action": "ui"})["html"]
    assert len(html) < 250000
    assert "fetch(" not in html and "<script src=" not in html
    assert all(action in spec["actions"] for action in ["info", "stop", "train", "upload_chunk"])
    assert spec["config"] == ["endpoint", "api_key", "environment_job"]
    assert "resources" in spec["actions"]
    assert tool["inputSchema"]["properties"]["gpu_ids"] == {
        "type": "array", "items": {"type": "integer", "minimum": 0},
        "minItems": 1, "uniqueItems": True,
    }
    assert tool["configSchema"]["properties"]["api_key"]["x-sensitive"] is True


def test_unconfigured_card_does_not_claim_connected_or_start_work():
    plugin = MotionPlugin({})
    assert plugin.dispatch("motion", {"action": "start"})["state"] == "unconfigured"
    assert plugin.dispatch("motion", {"action": "train"})["state"] == "error"
    assert plugin.start() is None


def test_invalid_boot_config_does_not_break_other_actucore_cards():
    plugin = MotionPlugin({"endpoint": "http://user:secret@host", "api_key": "secret-test"})
    state = plugin.dispatch("motion", {"action": "info"})
    assert state["state"] == "unconfigured" and state["message"]
    assert "secret" not in state["message"]
    assert plugin.get_tools()[0]["name"] == "motion"
    assert plugin.dispatch("motion", {"action": "config", "endpoint": "http://server:15732"})["state"] == "ready"


def test_secret_never_leaves_info_or_config_and_omission_preserves_it(service):
    endpoint, requests = service
    plugin = MotionPlugin({"endpoint": endpoint, "api_key": "secret-test"})
    result = plugin.dispatch("motion", {"action": "config", "endpoint": endpoint})
    assert "secret-test" not in json.dumps(result)
    assert result["config_secrets"] == {"api_key": True}
    assert plugin.dispatch("motion", {"action": "check"})["state"] == "ready"
    assert requests[-1][1] == "Bearer secret-test"


def test_bad_config_is_rejected_without_replacing_previous_connection(service):
    endpoint, _ = service
    plugin = MotionPlugin({"endpoint": endpoint, "api_key": "secret-test"})
    result = plugin.dispatch("motion", {"action": "config", "endpoint": "http://user:password@example.com"})
    assert result["state"] == "error" and result["error"]  # existing Core invoke must reject config
    assert plugin.dispatch("motion", {"action": "info"})["config_values"]["endpoint"] == endpoint


def test_shared_environment_connection_is_used_once_and_explicit_config_wins(monkeypatch):
    monkeypatch.setenv("MOTION_SERVICE_URL", "http://server:15732")
    monkeypatch.setenv("MOTION_API_TOKEN", "environment-secret")
    plugin = MotionPlugin({"endpoint": "", "api_key": ""})
    assert plugin.dispatch("motion", {"action": "info"})["config_values"]["endpoint"] == "http://server:15732"
    plugin.dispatch("motion", {"action": "config", "endpoint": "http://other:15732"})
    assert plugin._client._api_key == "environment-secret"
    explicit = MotionPlugin({"endpoint": "http://explicit:15732", "api_key": "explicit-secret"})
    assert explicit._client.endpoint == "http://explicit:15732"
    assert explicit._client._api_key == "explicit-secret"


def test_redirect_does_not_forward_credentials(service):
    endpoint, requests = service
    with pytest.raises(ServiceError, match="重定向"):
        MotionClient(endpoint, "secret-test").call("redirect")
    assert len(requests) == 1 and requests[0][0] == "/v1/call"


def test_remote_error_is_redacted(service):
    endpoint, _ = service
    with pytest.raises(ServiceError) as failure:
        MotionClient(endpoint, "secret-test").call("failure")
    assert "secret-test" not in str(failure.value)


def test_response_is_bounded(service):
    endpoint, _ = service
    client = MotionClient(endpoint, "secret-test")
    client.MAX_RESPONSE = 10
    with pytest.raises(ServiceError, match="响应过大"):
        client.call("large")


@pytest.mark.parametrize("endpoint", ["ftp://host", "https://user:secret@host", "http://host?q=1", "http://host/#x", "http://host:bad", "http://host\n/path"])
def test_endpoint_cannot_carry_inline_credentials_or_request_fragments(endpoint):
    with pytest.raises(ServiceError):
        MotionClient(endpoint, "secret-test")


def test_browser_upload_training_evaluation_and_export_under_existing_sandbox(tmp_path):
    """Actual HTML, CSP and MessagePort; only the remote service is a fixture."""
    import base64
    playwright = pytest.importorskip("playwright.sync_api")
    plugin = MotionPlugin({"endpoint": "http://127.0.0.1:15732", "api_key": "secret-test"})
    profile = {"id": "g1", "label": "G1", "joint_names": ["knee"], "control_hz": 50}
    dataset = {"id": "data1", "name": "walk.npz", "profile_id": "g1", "profile_hash": "profile-g1", "sha256": "walk-sha", "frames": 3, "fps": 50}
    model = {"id": "model1", "name": "策略 1", "profile_id": "g1", "profile_hash": "profile-g1", "reference_sha256": "walk-sha", "dataset_id": "data1", "job_id": "train1"}
    frames = [{"time": i * .02, "joint_pos": [.1 * i], "reference_joint_pos": [.11 * i]} for i in range(3)]
    submitted = []
    uploaded = bytearray()
    jobs = []

    def remote(action, **args):
        submitted.append((action, args))
        if action == "catalog":
            return {"profiles": [profile], "datasets": [dataset], "models": [model]}
        if action == "jobs":
            return {"jobs": jobs}
        if action == "resources":
            return {"sampled_at": 1700000000, "gpus": [
                {"id": 2, "name": "A100", "memory_total_mib": 81920,
                 "memory_used_mib": 2000, "utilization_percent": 5,
                 "selectable": True, "shared": True, "reason": ""}],
                "training": {"available": True, "multi_gpu_supported": True,
                             "max_gpus": 8, "max_num_envs": 8192, "max_iterations": 1000000}}
        if action == "plan":
            return {"gpu_mode": args["gpu_mode"], "ready": True, "queueable": True, "reason": "",
                    "gpu_ids": [2], "world_size": 1, "num_envs": args["num_envs"], "envs_per_rank": args["num_envs"]}
        if action == "dataset_render":
            return {"available": False, "reason": "本测试未提供动作画面"}
        if action == "dataset_preview":
            return {"dataset": dataset, "frames": frames}
        if action == "upload_begin":
            return {"upload_id": "upload1"}
        if action == "upload_chunk":
            assert args["offset"] == len(uploaded)
            uploaded.extend(base64.b64decode(args["data"]))
            return {"offset": len(uploaded)}
        if action == "upload_finish":
            return {"dataset": dataset}
        if action in {"train", "evaluate"}:
            job = {"id": action + "1", "kind": action, "profile_id": "g1", "dataset_id": "data1",
                   "state": "queued" if action == "train" else "succeeded"}
            if action == "evaluate":
                job["model_id"] = args["model_id"]
            jobs.append(job)
            return {"job": job}
        if action == "job":
            job = next(j for j in jobs if j["id"] == args["job_id"])
            return {"job": job, "log": "real service fixture log", "evaluation": {
                        "joint_rmse_rad": .03, "early_terminations": 0, "timeouts": 0,
                        "steps": 3, "onnx_checked": True, "onnx_max_abs_error": 1e-7,
                        "qualification": "evaluated"},
                    "rollout": {"frames": frames}}
        if action == "export":
            return {"package_id": "package1", "size": 12345, "manifest": {
                "profile_id": "g1", "qualification": "evaluated",
                "files": {"policy.onnx": "test-hash", "reference.npz": "reference-hash"}}}
        raise AssertionError(action)

    plugin._client.call = remote

    def host(message):
        if message["method"] == "config":
            if "values" in message.get("args", {}):
                jobs.clear()  # Switching services has a different job history.
                return {"saved": True}
            return {"values": {"endpoint": plugin._client.endpoint}, "runtime_secret_set": {"api_key": True}}
        return plugin.dispatch("motion", message["args"])

    html = plugin.dispatch("motion", {"action": "ui"})["html"]
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(**chromium_options(), headless=True)
        page = browser.new_page(viewport={"width": 1000, "height": 780})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.expose_function("hostCall", host)
        page.route('http://localhost/workbench', lambda route: route.fulfill(body='<iframe sandbox="allow-scripts" style="width:960px;height:720px;border:0"></iframe>', content_type='text/html'))
        page.goto('http://localhost/workbench')
        page.evaluate("""html => {
          const f=document.querySelector('iframe');
          f.onload=()=>{const c=new MessageChannel();c.port1.onmessage=async e=>{
            try {c.port1.postMessage({id:e.data.id,result:await window.hostCall(e.data)});}
            catch(error){c.port1.postMessage({id:e.data.id,error:error.message});}
          };f.contentWindow.postMessage({protocol:'motus.workbench/1'},'*',[c.port2]);};
          f.srcdoc=`<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; connect-src 'none'">${html}`;
        }""", html)
        frame = page.frame_locator("iframe")
        playwright.expect(frame.locator("#status")).to_have_text("已连接")
        frame.locator("#datasets .item").click()
        assert frame.locator("#datasets .item").evaluate("e=>typeof e.onclick") == "function"
        playwright.expect(frame.locator("#datasetPreview")).to_be_visible()
        page.screenshot(path=str(tmp_path / "motion-data.png"))
        file_bytes = b"npz-test-content" * 2000
        frame.locator("#importAction").click()
        frame.locator("#file").set_input_files({"name": "walk.npz", "mimeType": "application/octet-stream", "buffer": file_bytes})
        frame.locator("#upload").click()
        playwright.expect(frame.locator("#uploadState")).to_have_text("已导入")
        assert bytes(uploaded) == file_bytes
        frame.locator('#retrain').click()
        frame.locator("#train").evaluate("button => {button.click();button.click();}")
        playwright.expect(frame.locator("#jobStatus")).to_have_text("排队中")
        frame.locator('#breadcrumbActions').click()
        frame.locator("#modelHistory").evaluate("node=>node.open=true")
        frame.locator('#openEvaluation').click()
        frame.locator("#evaluate").click()
        playwright.expect(frame.locator("#evalResult")).to_contain_text("关节 RMSE(rad)")
        playwright.expect(frame.locator("#evalRaw")).not_to_be_visible()
        playwright.expect(frame.locator("#evalReplay")).to_be_visible()
        page.screenshot(path=str(tmp_path / "motion-evaluation.png"))
        frame.locator('#breadcrumbActions').click()
        frame.locator('#openEvaluation').click()
        frame.locator("#export").click()
        playwright.expect(frame.locator("#exportResult")).to_contain_text("package1")
        playwright.expect(frame.locator("#exportResult")).to_contain_text("12.1 KiB")
        playwright.expect(frame.locator("#exportRaw")).not_to_be_visible()
        page.screenshot(path=str(tmp_path / "motion-export.png"))
        assert sum(action == "train" for action, _ in submitted) == 1
        frame.locator("#connection").click()
        frame.locator("#manualConnection > summary").click()
        frame.locator("#endpoint").fill("http://another-server:15732")
        frame.locator("#saveConnection").click()
        playwright.expect(frame.locator("#settings")).not_to_be_visible()
        playwright.expect(frame.locator("#settings")).not_to_be_visible()
        page.screenshot(path=str(tmp_path / "motion-workbench.png"))
        assert not errors
        assert "secret-test" not in frame.locator("body").inner_text()
        browser.close()
