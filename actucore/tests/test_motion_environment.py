"""Host deployment jobs: durable API, strict operations, cancellation and real logs."""
from __future__ import annotations

import hashlib
import base64
from http.server import ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_environment.manager import EnvironmentManager, atomic_json, read_json, secret
from motion_environment.recipe import Cancelled, NoRedirect, Task, unpack_profile
from motion_environment.recipe import OWNER
from motion_environment.server import make_handler
from motion_environment.image_fetch import Fetcher, select_manifest
from motion_environment.wheel_cache import wheel_rank, download


def test_target_wheel_tags_accept_stable_abi_and_reject_free_threaded_future_abi():
    assert wheel_rank("onnx-1.20.1-cp312-abi3-manylinux_2_28_x86_64.whl") is not None
    assert wheel_rank("onnx-1.20.1-cp314-abi3-manylinux_2_28_x86_64.whl") is None
    assert wheel_rank("onnx-1.20.1-cp313-cp313t-manylinux_2_28_x86_64.whl") is None


@pytest.mark.parametrize("change", [{"sha256": "../../outside"}, {"filename": "../outside.whl"},
                                     {"filename": "payload.py"}, {"size": True}, {"size": 0}])
def test_download_rejects_unsafe_metadata_before_creating_cache(tmp_path, change):
    entry = {"sha256": "a" * 64, "filename": "test.whl", "size": 1, **change}
    with pytest.raises(ValueError):
        download(entry, tmp_path / "not-created")
    assert not (tmp_path / "not-created").exists()


def test_verified_raw_wheel_cache_rejects_corruption_and_wrong_package(manager, tmp_path, monkeypatch):
    monkeypatch.setattr("motion_environment.wheel_cache.PACKAGES", {})
    cache = tmp_path / "wheels"
    cache.mkdir()
    filename = "torch-2.14.1+cu126-cp313-cp313-manylinux_2_28_x86_64.whl"
    source = cache / filename
    warp_name = "warp_lang-1.18.0+cu12-py3-none-manylinux_2_28_x86_64.whl"
    warp = cache / warp_name
    with zipfile.ZipFile(warp, "w") as archive:
        archive.writestr("warp_lang.dist-info/METADATA", "Name: warp-lang\nVersion: 1.18.0+cu12\n")
    warp_entry = {"name": "warp-lang", "version": "1.18.0+cu12", "filename": warp_name,
                  "size": warp.stat().st_size, "sha256": hashlib.sha256(warp.read_bytes()).hexdigest(),
                  "official_url": "https://github.com/NVIDIA/warp/releases/download/v1.18.0/" + warp_name}
    def write(package):
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("torch.dist-info/METADATA", f"Name: {package}\nVersion: 2.14.1+cu126\n")
        entry = {"name": "torch", "version": "2.14.1+cu126", "filename": filename,
                 "size": source.stat().st_size, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                 "official_url": "https://download.pytorch.org/whl/" + filename}
        atomic_json(cache / "manifest.json", {"wheels": [entry, warp_entry]})
    job = manager.submit("install", {"target_id": "gpu-server"})["job"]
    task = Task(manager.root, job, manager.targets["gpu-server"], {"wheel_cache": str(cache)})
    write("torch")
    task.prepare_wheels(tmp_path / "valid")
    assert (tmp_path / "valid" / filename).read_bytes() == source.read_bytes()
    original = source.read_bytes()
    source.write_bytes(b"X" + original[1:])
    with pytest.raises(ValueError, match="SHA"):
        task.prepare_wheels(tmp_path / "corrupt")
    write("unrelated-package")
    with pytest.raises(ValueError, match="名称或版本"):
        task.prepare_wheels(tmp_path / "wrong-name")


def test_distributed_check_failure_never_records_success_and_removes_script(manager, monkeypatch):
    job = manager.submit("install", {"target_id": "gpu-server"})["job"]
    task = Task(manager.root, job, manager.targets["gpu-server"], manager.config)
    task.job["gpu_verification"] = {"cuda_compute": True}
    monkeypatch.setattr(task, "run", lambda *a, **kw: "")
    calls = []
    monkeypatch.setattr("motion_environment.recipe.subprocess.run", lambda cmd, **kw: calls.append(cmd))
    def fail(*args, **kwargs):
        assert args[-1] == "0,2"
        raise RuntimeError("actual NCCL error")
    monkeypatch.setattr(task, "probe", fail)
    with pytest.raises(RuntimeError, match="actual NCCL error"):
        task.verify_distributed("owned-runtime")
    assert "distributed" not in task.job["gpu_verification"]
    assert calls[-1][0:6] == ["docker", "exec", "--user", "0", "owned-runtime", "rm"]


def test_pending_cache_wait_is_explicit_bounded_cancellable_and_uses_real_bytes(manager, tmp_path, monkeypatch):
    job = manager.submit("install", {"target_id": "gpu-server"})["job"]
    task = Task(manager.root, job, manager.targets["gpu-server"], {})
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "package.whl.uploading").write_bytes(b"x" * 1048576)
    with pytest.raises(TimeoutError, match="尚未上传完整"):
        task.await_wheel_manifest(cache)
    task.config["cache_wait_seconds"] = 2
    monkeypatch.setattr("motion_environment.recipe.time.monotonic", lambda: 100.)
    monkeypatch.setattr("motion_environment.recipe.time.sleep", lambda _: manager.cancel({"job_id": job["id"]}))
    with pytest.raises(Cancelled):
        task.await_wheel_manifest(cache)
    assert any("已接收 1.0 MiB" in event["message"] for event in manager.status({"job_id": job["id"]})["events"])
    assert task.job["state"] == "running"


@pytest.fixture
def manager(tmp_path):
    token = tmp_path / "token"
    token.write_text("private-test-token-" + "x" * 32)
    token.chmod(0o600)
    config = tmp_path / "config.json"
    atomic_json(config, {"targets": [{"id": "gpu-server", "label": "GPU server", "gpu_ids": [0, 2],
                                      "existing": {"endpoint": "http://127.0.0.1:15732", "token_file": str(token)}}]})
    launched = []
    return EnvironmentManager(tmp_path / "state", config, launcher=launched.append)


def test_targets_hide_secrets_and_reopen_latest_job(manager):
    job = manager.call({"action": "install", "target_id": "gpu-server", "mode": "reuse"})["job"]
    targets = manager.call({"action": "targets"})
    assert targets["targets"][0]["latest_job"]["id"] == job["id"]
    assert "token_file" not in json.dumps(targets)
    assert "private-test-token" not in json.dumps(targets)
    assert manager.call({"action": "status"})["jobs"][0]["id"] == job["id"]
    assert manager.call({"action": "install", "target_id": "gpu-server", "mode": "reuse"})["reused"]


@pytest.mark.parametrize("args", [
    {"action": "shell", "command": "echo bad"},
    {"action": "install", "target_id": "arbitrary-host"},
    {"action": "install", "target_id": "gpu-server", "recipe_id": "user-image"},
    {"action": "install", "target_id": "gpu-server", "mode": "check"},
    {"action": "status", "job_id": "../../secret"},
])
def test_no_arbitrary_commands_images_targets_or_paths(manager, args):
    with pytest.raises(ValueError):
        manager.call(args)


def test_recovery_marks_missing_worker_interrupted_without_losing_history(manager):
    job = manager.submit("check", {"target_id": "gpu-server"})["job"]
    job.update(state="running", pid=99999999, updated_at=time.time() - 100)
    atomic_json(manager._folder(job["id"]) / "job.json", job)
    result = manager.status({"job_id": job["id"]})["job"]
    assert result["state"] == "interrupted"
    assert result["id"] == job["id"]
    # Retry is permitted; old logs and artifact records remain available.
    assert manager.submit("check", {"target_id": "gpu-server"})["job"]["id"] != job["id"]


def test_recovery_preserves_live_durable_worker(manager, monkeypatch):
    job = manager.submit("check", {"target_id": "gpu-server"})["job"]
    job.update(state="running", pid=123, updated_at=time.time() - 100)
    atomic_json(manager._folder(job["id"]) / "job.json", job)
    monkeypatch.setattr("motion_environment.manager.worker_alive", lambda j: True)
    assert manager.status({"job_id": job["id"]})["job"]["state"] == "running"


def test_terminal_worker_result_wins_exit_race(manager, monkeypatch):
    job = manager.submit("check", {"target_id": "gpu-server"})["job"]
    job.update(state="running", pid=123, updated_at=time.time() - 100)
    path = manager._folder(job["id"]) / "job.json"
    atomic_json(path, job)
    def exited(_):
        atomic_json(path, {**job, "state": "succeeded"})
        return False
    monkeypatch.setattr("motion_environment.manager.worker_alive", exited)
    assert manager.status({"job_id": job["id"]})["job"]["state"] == "succeeded"


def test_authenticated_redirect_is_rejected_before_forwarding():
    handler = NoRedirect()
    request = urllib.request.Request("http://private-source.test", headers={"Authorization": "Bearer secret"})
    with pytest.raises(RuntimeError, match="未转发"):
        handler.redirect_request(request, None, 302, "Found", {}, "http://other-host.test")


@pytest.mark.parametrize("health", [{"protocol": "motus.motion/1", "runtime": {}},
                                   {"protocol": "wrong", "runtime": {"mujoco": True}}])
def test_api_alive_is_not_runtime_ready(manager, monkeypatch, health):
    job = manager.submit("install", {"target_id": "gpu-server"})["job"]
    task = Task(manager.root, job, manager.targets["gpu-server"], manager.config)
    monkeypatch.setattr(task, "http", lambda *a, **kw: health)
    with pytest.raises(RuntimeError, match="协议或运行依赖不完整"):
        task.verify(manager.targets["gpu-server"]["existing"])
    assert task.job["state"] != "succeeded"


@pytest.mark.parametrize("fault", [None, "legacy", "observer", "heartbeat"])
def test_workbench_readiness_requires_view_lease_and_closes_owned_probe(manager, monkeypatch, fault):
    job = manager.submit("install", {"target_id": "gpu-server"})["job"]
    task = Task(manager.root, job, manager.targets["gpu-server"], manager.config)
    calls = []
    view = "environment-" + job["id"]
    control = {"epoch": 3, "owner_view_id": view, "is_controller": True}
    def api(connection, action, **args):
        calls.append((action, args))
        if action == "health":
            return {"protocol": "motus.motion/1", "runtime": {
                name: True for name in ("mujoco", "supermimic", "mjlab", "onnxruntime")}}
        if action == "catalog":
            return {"profiles": [{"id": "g1", "robot_type": "unitree_g1"}]}
        if action == "simulation_create":
            assert args["view_id"] == view and args["request_id"] == job["id"]
            return {"session": {"id": "owned-probe"}, **({} if fault == "legacy" else {"control": control})}
        if action == "simulation_inspect":
            assert args["view_id"] != view
            return {"control": {**control, "is_controller": fault == "observer"}}
        if action in {"simulation_heartbeat", "simulation_step", "simulation_close"}:
            assert args["session_id"] == "owned-probe"
            if fault != "legacy":
                assert args["view_id"] == view and args["control_epoch"] == 3
            if action == "simulation_heartbeat" and fault == "heartbeat":
                raise RuntimeError("heartbeat unavailable")
            return {"snapshot": {"time": 0.02}}
        if action == "simulation_render":
            assert args["view_id"] != view
            return {"image": {"mime": "image/jpeg", "data": base64.b64encode(b"\xff\xd8test").decode()}}
        raise AssertionError(action)
    monkeypatch.setattr(task, "http", api)
    if fault:
        with pytest.raises(RuntimeError):
            task.verify(manager.targets["gpu-server"]["existing"])
        assert task.job["state"] != "succeeded"
        assert not (task.folder / "connection.json").exists()
    else:
        task.verify(manager.targets["gpu-server"]["existing"])
        assert task.job["state"] == "succeeded"
        assert task.job["verification"]["view_lease"]["observer_read_only"] is True
    assert calls[-1][0] == "simulation_close"


def test_events_cursor_partial_lines_and_secret_redaction(manager):
    job = manager.submit("check", {"target_id": "gpu-server"})["job"]
    task = Task(manager.root, job, manager.targets["gpu-server"], manager.config)
    task.redactions.append("private-value")
    task.log("download private-value https://user:pass@example.test/file Bearer abc123")
    result = manager.status({"job_id": job["id"]})
    assert "private-value" not in json.dumps(result)
    assert "user:pass" not in json.dumps(result)
    assert "abc123" not in json.dumps(result)
    cursor = result["next_cursor"]
    with (task.folder / "events.jsonl").open("ab") as stream:
        stream.write(b'{"message":"pending"')
    assert manager.status({"job_id": job["id"], "cursor": cursor})["events"] == []
    with (task.folder / "events.jsonl").open("ab") as stream:
        stream.write(b'}\n')
    assert manager.status({"job_id": job["id"], "cursor": cursor})["events"] == [{"message": "pending"}]


def test_running_subprocess_cancel_is_prompt_and_terminates_child(manager):
    job = manager.submit("check", {"target_id": "gpu-server"})["job"]
    task = Task(manager.root, job, manager.targets["gpu-server"], manager.config)
    failures = []
    def execute():
        try:
            task.run([sys.executable, "-c", "import time; print('started',flush=True); time.sleep(60)"])
        except Exception as exc:
            failures.append(exc)
    thread = threading.Thread(target=execute)
    thread.start()
    deadline = time.monotonic() + 3
    while not read_json(task.folder / "job.json").get("command_pid"):
        assert time.monotonic() < deadline
        time.sleep(.01)
    pid = read_json(task.folder / "job.json")["command_pid"]
    manager.cancel({"job_id": job["id"]})
    thread.join(3)
    assert not thread.is_alive()
    assert isinstance(failures[0], Cancelled)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_failure_preserves_real_command_output(manager):
    job = manager.submit("check", {"target_id": "gpu-server"})["job"]
    task = Task(manager.root, job, manager.targets["gpu-server"], manager.config)
    with pytest.raises(RuntimeError, match="退出码 7"):
        task.run([sys.executable, "-c", "print('real install failure'); raise SystemExit(7)"])
    assert any("real install failure" in event["message"] for event in manager.status({"job_id": job["id"]})["events"])


def test_connection_only_after_success_and_never_inside_public_status(manager):
    job = manager.submit("install", {"target_id": "gpu-server"})["job"]
    with pytest.raises(ValueError, match="尚未成功"):
        manager.connection({"job_id": job["id"]})
    job["state"] = "succeeded"
    folder = manager._folder(job["id"])
    atomic_json(folder / "job.json", job)
    atomic_json(folder / "connection.json", manager.targets["gpu-server"]["existing"])
    assert manager.connection({"job_id": job["id"]})["api_key"].startswith("private-test-token")
    assert "private-test-token" not in json.dumps(manager.status({"job_id": job["id"]}))


def test_credentials_require_private_permissions(tmp_path):
    path = tmp_path / "token"
    path.write_text("x" * 32)
    path.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        secret(path)


def test_unowned_container_never_modified(manager, monkeypatch):
    job = manager.submit("check", {"target_id": "gpu-server"})["job"]
    task = Task(manager.root, job, manager.targets["gpu-server"], manager.config)
    replies = iter(["containerid", '{"Config":{"Labels":{},"Image":"other"}}'])
    commands = []
    monkeypatch.setattr(task, "run", lambda argv, **kw: (commands.append(argv), next(replies))[1])
    with pytest.raises(RuntimeError, match="其他环境"):
        task._container("motus-env-test")
    assert all(command[1] in {"ps", "inspect"} for command in commands)
    assert ".Config.Env" not in str(commands)


@pytest.mark.parametrize("already_exists", [True, False])
def test_unowned_volume_including_create_race_is_never_mounted(manager, monkeypatch, already_exists):
    job = manager.submit("check", {"target_id": "gpu-server"})["job"]
    task = Task(manager.root, job, manager.targets["gpu-server"], manager.config)
    replies = iter(["reserved-data", "null"] if already_exists else ["", "reserved-data", "null"])
    commands = []
    monkeypatch.setattr(task, "run", lambda argv, **kw: (commands.append(argv), next(replies))[1])
    with pytest.raises(RuntimeError, match="其他环境"):
        task.ensure_volume("reserved-data")
    assert all(command[1] == "volume" for command in commands)
    if already_exists:
        assert all(command[2] != "create" for command in commands)


@pytest.mark.parametrize("owner,created_here,stopped", [("other", True, False),
                                                      ("gpu-server", False, False),
                                                      ("gpu-server", True, True)])
def test_cancel_stops_only_runtime_started_by_this_task(manager, monkeypatch, owner, created_here, stopped):
    from types import SimpleNamespace
    job = manager.submit("install", {"target_id": "gpu-server"})["job"]
    task = Task(manager.root, job, manager.targets["gpu-server"], manager.config)
    job.update(container="owned-runtime", image="fixed-image")
    labels = {OWNER: owner, OWNER + ".job": job["id"] if created_here else "another-job"}
    commands = []
    def run(argv, **_):
        commands.append(argv)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"image": "fixed-image", "labels": labels, "running": True}))
    monkeypatch.setattr("motion_environment.recipe.subprocess.run", run)
    task.cleanup_started_runtime()
    assert any(command[1] == "stop" for command in commands) is stopped


def test_profile_bundle_rejects_traversal_and_checksum_errors(tmp_path):
    archive = tmp_path / "bad.tgz"
    with tarfile.open(archive, "w:gz") as stream:
        entry = tarfile.TarInfo("../escape")
        entry.size = 1
        stream.addfile(entry, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="清单"):
        unpack_profile(archive, tmp_path / "out")
    assert not (tmp_path / "escape").exists()


def test_http_auth_request_bounds_and_async_return(manager):
    token = "http-private-token" + "x" * 32
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(manager, token))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1/call"
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(url, b'{}'))
        assert error.value.code == 401
        request = urllib.request.Request(url, json.dumps({"action": "check", "target_id": "gpu-server"}).encode(),
                                         {"Authorization": "Bearer " + token})
        result = json.load(urllib.request.urlopen(request))
        assert result["job"]["state"] == "queued"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_official_image_download_checks_digest_before_publishing_cache(tmp_path):
    expected = "sha256:" + hashlib.sha256(b"verified layer").hexdigest()
    class Response(io.BytesIO):
        headers = {"Content-Length": "9"}
    class Opener:
        requests = []
        def open(self, request, **_):
            self.requests.append(request)
            return Response(b"corrupted")
    fetcher = Fetcher(tmp_path)
    fetcher.token = "registry-test-token"
    fetcher.opener = Opener()
    with pytest.raises(ValueError, match="摘要校验失败"):
        fetcher.blob("blobs", expected)
    assert not list(tmp_path.iterdir())
    request = fetcher.opener.requests[0]
    assert "Authorization" not in request.headers  # Redirects cannot inherit this header.
    assert request.unredirected_hdrs["Authorization"] == "Bearer registry-test-token"


def test_official_multiarch_index_requires_unique_linux_amd64():
    manifests = [{"platform": {"os": "linux", "architecture": "arm64"}, "digest": "arm"},
                 {"platform": {"os": "linux", "architecture": "amd64"}, "digest": "x86"}]
    assert select_manifest({"manifests": manifests}) == "x86"
    with pytest.raises(ValueError):
        select_manifest({"manifests": manifests + [manifests[1]]})


def test_wheel_selection_targets_linux_glibc_not_mac_downloader():
    older = wheel_rank("nvidia_cufft_cu12-11.3.0.4-py3-none-manylinux2014_x86_64.whl")
    newer = wheel_rank("nvidia_cufft_cu12-11.3.0.4-py3-none-manylinux_2_27_x86_64.whl")
    assert newer > older
    assert wheel_rank("x-1-cp313-cp313-manylinux_2_38_x86_64.whl") is None
    assert wheel_rank("x-1-cp313-cp313-manylinux_2_28_aarch64.whl") is None
    assert wheel_rank("x-1-cp313t-cp313t-manylinux_2_28_x86_64.whl") is None
    assert wheel_rank("x-1-cp312-cp312-manylinux_2_28_x86_64.whl") is None


def test_corrupt_same_size_chunk_is_discarded_and_next_download_recovers(tmp_path, monkeypatch):
    expected = hashlib.sha256(b"GOOD").hexdigest()
    entry = {"name": "fixture", "filename": "fixture.whl", "size": 4,
             "sha256": expected, "download_url": "https://example.test/fixture.whl"}
    parts = tmp_path / ".parts" / expected
    parts.mkdir(parents=True)
    (parts / "00000").write_bytes(b"BAD!")
    with pytest.raises(ValueError, match="完整 SHA"):
        download(entry, tmp_path, workers=1)
    assert not parts.exists()
    assert not (tmp_path / "fixture.whl").exists()
    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": "bytes 0-3/4"}
    monkeypatch.setattr("motion_environment.wheel_cache.urllib.request.urlopen", lambda *a, **kw: Response(b"GOOD"))
    assert download(entry, tmp_path, workers=1).read_bytes() == b"GOOD"
