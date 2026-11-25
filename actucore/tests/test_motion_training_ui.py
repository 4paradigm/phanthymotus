"""Multi-GPU form and job monitoring through the real plugin and browser sandbox."""

import base64
import io
from pathlib import Path
import sys

import pytest
from motion_ui_helpers import chromium_options

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.motion import MotionPlugin


def test_inspect_modes_are_read_only_and_preview_size_is_host_owned():
    plugin = MotionPlugin({"endpoint": "http://server:15732", "api_key": "test-secret"})
    calls = []
    plugin._client.call = lambda action, **args: calls.append((action, args)) or {"available": False}
    for mode in ("resources", "jobs"):
        plugin.dispatch("motion", {"action": "inspect", "mode": mode, "command": "ignored"})
        assert calls[-1] == (mode, {})
    plugin.dispatch("motion", {"action": "inspect", "mode": "job", "job_id": "job1"})
    assert calls[-1] == ("job", {"job_id": "job1"})
    plugin.dispatch("motion", {"action": "inspect", "mode": "training_preview", "job_id": "job1", "width": 640, "height": 360})
    assert calls[-1] == ("training_preview", {"job_id": "job1", "width": 640, "height": 360})
    plugin.dispatch("motion", {"action": "inspect", "mode": "plan", "gpu_mode": "auto",
                               "num_envs": 64, "preset": "small", "command": "ignored", "job_id": "ignored"})
    assert calls[-1] == ("plan", {"gpu_mode": "auto", "num_envs": 64, "preset": "small"})
    plugin.dispatch("motion", {"action": "train", "gpu_mode": "auto", "num_envs": 64, "preset": "small"})
    assert calls[-1] == ("train", {"gpu_mode": "auto", "num_envs": 64, "preset": "small"})
    assert plugin.get_tools()[0]["inputSchema"]["properties"]["gpu_mode"]["enum"] == ["auto", "manual"]
    count = len(calls)
    for mode in ("train", "cancel", "simulation_run", "shell", None, ["train"]):
        assert plugin.dispatch("motion", {"action": "inspect", "mode": mode})["state"] == "error"
    assert len(calls) == count


@pytest.fixture
def training_browser():
    playwright = pytest.importorskip("playwright.sync_api")
    profile = {"id": "g1", "label": "G1", "joint_names": ["knee"], "control_hz": 50}
    dataset = {"id": "walk", "name": "walk.npz", "profile_id": "g1", "profile_hash": "profile-g1", "sha256": "walk-sha", "frames": 150, "fps": 50}
    prior_job = {"id": "prior-job", "kind": "train", "state": "running", "profile_id": "g1", "dataset_id": "walk"}
    state = {"calls": [], "jobs": [prior_job], "models": [], "preview": {"available": False, "reason": "等待训练采样"}}
    state["resources"] = {"sampled_at": 1700000000, "gpus": [
        {"id": i, "name": "NVIDIA A100", "memory_total_mib": 81920,
         "memory_used_mib": 2048 + i * 100, "utilization_percent": 0 if i == 2 else 10 + i,
         "shared": True, "selectable": True, "reason": ""} for i in range(8)],
        "training": {"available": True, "multi_gpu_supported": True, "max_gpus": 8,
                     "max_num_envs": 8192, "max_iterations": 1000000}}
    plugin = MotionPlugin({"endpoint": "http://server:15732", "api_key": "test-secret"})

    def remote(action, **args):
        state["calls"].append((action, dict(args)))
        if action == "dataset_render":
            return {"available": False, "reason": "本测试未提供动作画面"}
        if action == "catalog":
            return {"profiles": [profile, {**profile, "id": "g2", "label": "G2"}], "datasets": [dataset], "models": state["models"]}
        if action == "resources":
            return state["resources"]
        if action == "plan":
            if state.get("plan_response"):
                return state["plan_response"]
            ids = args["gpu_ids"] if args["gpu_mode"] == "manual" else [2] if args["num_envs"] < 1024 else [0, 1, 2, 3]
            unavailable = next((gpu for gpu in state["resources"]["gpus"] if gpu["id"] in ids and not gpu["selectable"]), None)
            if unavailable:
                from plugins.motion.client import ServiceError
                raise ServiceError(unavailable["reason"])
            return {"gpu_mode": args["gpu_mode"], "ready": True, "queueable": True,
                    "reason": "", "gpu_ids": ids,
                    "world_size": len(ids), "num_envs": args["num_envs"], "envs_per_rank": args["num_envs"] // len(ids)}
        if action == "jobs":
            return {"jobs": state["jobs"]}
        if action == "job":
            job = next(item for item in state["jobs"] if item["id"] == args["job_id"])
            return {"job": job, "log": "worker log", "artifacts": job.get("artifacts", [])}
        if action == "train":
            ids = args["gpu_ids"] if args["gpu_mode"] == "manual" else [2] if args["num_envs"] < 1024 else [0, 1, 2, 3]
            job = {"id": f"new-job-{len(state['jobs'])}", "kind": "train", "state": "queued",
                   "profile_id": args["profile_id"], "dataset_id": args["dataset_id"],
                   "config": {**args, "gpu_ids": ids, "world_size": len(ids),
                              "envs_per_rank": args["num_envs"] // len(ids)}}
            if state.get("plan_response", {}).get("queueable") and not state["plan_response"]["ready"]:
                job["message"] = state["plan_response"]["reason"]
                job["config"].update(gpu_ids=[], world_size=0, envs_per_rank=0)
            state["jobs"].append(job)
            if state.get("lose_train_response"):
                from plugins.motion.client import ServiceError
                raise ServiceError("服务器请求超时")
            return {"job": job}
        if action == "cancel":
            job = next(item for item in state["jobs"] if item["id"] == args["job_id"])
            job["state"] = "cancelled"
            return {"job": job}
        if action == "training_preview":
            assert 320 <= args["width"] <= 960 and 160 <= args["height"] <= 640
            return state["preview"](args) if callable(state["preview"]) else state["preview"]
        raise AssertionError(action)

    plugin._client.call = remote
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(**chromium_options(), headless=True)
        page = browser.new_page(viewport={"width": 1000, "height": 720})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.expose_function("hostCall", lambda message: plugin.dispatch("motion", message["args"]))
        page.route('http://localhost/workbench', lambda route: route.fulfill(body='<iframe sandbox="allow-scripts" style="width:958px;height:664px;border:0"></iframe>', content_type='text/html'))
        page.goto('http://localhost/workbench')
        page.evaluate("""html => {
          const frame=document.querySelector('iframe');frame.onload=()=>{const channel=new MessageChannel();
            channel.port1.onmessage=async event=>{try{channel.port1.postMessage({id:event.data.id,result:await window.hostCall(event.data)});}
              catch(error){channel.port1.postMessage({id:event.data.id,error:error.message});}};
            frame.contentWindow.postMessage({protocol:'motus.workbench/1'},'*',[channel.port2]);};
          frame.srcdoc=`<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; connect-src 'none'; form-action 'none'; base-uri 'none'">${html}`;
        }""", plugin.dispatch("motion", {"action": "ui"})["html"])
        frame = page.frame_locator("iframe")
        playwright.expect(frame.locator("#status")).to_have_text("已连接")
        frame.locator('#motionPrimary').click()
        frame.locator("#trainingSetupBack").click()
        frame.locator("#trainingAdvanced > summary").click()
        playwright.expect(frame.locator("#train")).to_be_enabled()
        yield page, frame, state, playwright
        assert not errors
        browser.close()


def test_shared_eight_gpu_preflight_allocation_progress_and_cancel(training_browser, tmp_path):
    page, frame, state, playwright = training_browser
    frame.locator("#gpuSummary").click()
    frame.locator("#gpuMode").select_option("manual")
    playwright.expect(frame.locator('#gpuList input[value="2"]')).to_be_checked()
    assert frame.locator("#gpuList input:checked").count() == 1
    assert frame.locator("#gpuList .tag").all_text_contents() == ["共享"] * 8
    frame.locator('[data-gpu-count="8"]').click()
    assert frame.locator("#gpuList input:checked").count() == 8
    playwright.expect(frame.locator("#allocation")).to_contain_text("每 GPU 8 个机器人实例")
    assert frame.locator("#gpuList").evaluate("list=>list.scrollHeight===list.clientHeight")
    assert frame.locator("body").evaluate("""()=>{
      const viewport=document.querySelector('main').getBoundingClientRect();
      return [...document.querySelectorAll('#train')].every(node=>{
        const rect=node.getBoundingClientRect();return rect.top>=viewport.top&&rect.bottom<=viewport.bottom;
      });
    }""")
    page.screenshot(path=str(tmp_path / "eight-gpu-layout.png"))
    frame.locator("#numEnvs").fill("65")
    playwright.expect(frame.locator("#train")).to_be_disabled()
    playwright.expect(frame.locator("#trainValidation")).to_contain_text("整除")
    frame.locator("#numEnvs").fill("64")
    playwright.expect(frame.locator("#train")).to_be_enabled()
    state["resources"]["gpus"][7].update(selectable=False, reason="显存余量不足")
    frame.locator("#train").click()
    playwright.expect(frame.locator("#trainValidation")).to_contain_text("显存余量不足")
    assert frame.locator("#gpuList input:checked").count() == 8
    assert not any(action == "train" for action, _ in state["calls"])
    state["resources"]["gpus"][7].update(selectable=True, reason="")
    frame.locator("#refreshResources").click()
    playwright.expect(frame.locator("#train")).to_be_enabled()
    playwright.expect(frame.locator("#trainProfile")).to_have_value("g1")
    playwright.expect(frame.locator("#trainDataset")).to_have_value("walk")
    frame.locator("#train").evaluate("button=>{button.click();button.click();}")
    playwright.expect(frame.locator("#jobStatus")).to_have_text("排队中")
    submitted = [args for action, args in state["calls"] if action == "train"]
    assert len(submitted) == 1 and submitted[0]["gpu_ids"] == list(range(8)) and submitted[0]["num_envs"] == 64
    playwright.expect(frame.locator("#jobProgress")).to_be_hidden()
    job = state["jobs"][-1]
    job["state"] = "running"
    page.wait_for_timeout(3200)
    playwright.expect(frame.locator("#jobPhase")).to_have_text("启动训练进程")
    playwright.expect(frame.locator("#jobProgressBar")).to_be_hidden()
    playwright.expect(frame.locator("#jobProgressText")).to_have_text("")
    job.update(state="running", progress={"iteration": 3, "total_iterations": 1000,
               "elapsed_s": 7, "phase": "world_model", "world_model_updates": 6,
               "policy_updates": 3, "ranks_ready": 8})
    page.wait_for_timeout(3200)
    playwright.expect(frame.locator("#jobProgressText")).to_have_text("迭代 3 / 1000")
    playwright.expect(frame.locator("#jobProgressInfo")).to_contain_text("8 / 8")
    playwright.expect(frame.locator("#jobInfo")).to_contain_text("每 GPU 实例")
    assert any(action == "training_preview" for action, _ in state["calls"])
    page.screenshot(path=str(tmp_path / "multi-gpu-training.png"))
    frame.locator("#cancel").click()
    playwright.expect(frame.locator("#jobStatus")).to_have_text("已取消")
    assert state["jobs"][0]["state"] == "running"
    assert [args for action, args in state["calls"] if action == "cancel"] == [{"job_id": job["id"]}]
    job.update(state="succeeded", model_id="model1", artifacts=["actor.pt", "world_model.pt"])
    job["progress"]["phase"] = "succeeded"
    state["models"] = [{"id": "model1", "profile_id": "g1", "profile_hash": "profile-g1", "reference_sha256": "walk-sha", "dataset_id": "walk", "job_id": job["id"]}]
    page.wait_for_timeout(3200)
    playwright.expect(frame.locator("#jobArtifacts")).to_contain_text("actor.pt")
    playwright.expect(frame.locator("#jobPhase")).to_have_text("已完成")
    frame.locator("#jobModel").click()
    playwright.expect(frame.locator("#modelDetails")).to_be_visible()
    playwright.expect(frame.locator("#publish")).to_be_visible()
    playwright.expect(frame.locator("#exportModel")).to_have_value("model1")


def test_training_monitor_has_only_real_frames_and_stops_when_hidden(training_browser, tmp_path):
    page, frame, state, playwright = training_browser
    image_module = pytest.importorskip("PIL.Image")
    frame.locator("#gpuSummary").click()
    frame.locator("#gpuMode").select_option("manual")
    frame.locator('[data-gpu-count="2"]').click()
    frame.locator("#train").click()
    playwright.expect(frame.locator("#jobStatus")).to_have_text("排队中")
    playwright.expect(frame.locator("#previewStatus")).to_have_text("等待训练采样")
    playwright.expect(frame.locator("#trainingImage")).to_be_hidden()
    payload = io.BytesIO()
    image_module.new("RGB", (896, 320), (35, 42, 50)).save(payload, format="JPEG")
    job = state["jobs"][-1]
    job.update(state="running", progress={"phase": "verifying"})
    playwright.expect(frame.locator("#jobPhase")).to_have_text("校验", timeout=7000)
    state["preview"] = {"available": True, "image": {"mime": "image/jpeg", "width": 896, "height": 320,
        "data": base64.b64encode(payload.getvalue()).decode()}, "samples": [
            {"rank": 0, "env_id": 3, "time": 1.2, "iteration": 5, "captured_at": 1700000000},
            {"rank": 1, "env_id": 7, "time": 1.3, "iteration": 5, "captured_at": 1700000000}],
        "total_envs": 64, "world_size": 2, "stale": False}
    playwright.expect(frame.locator("#trainingImage")).to_be_visible(timeout=10000)
    playwright.expect(frame.locator("#trainingPreviewInfo")).to_contain_text("2 / 64")
    playwright.expect(frame.locator(".train-setup")).to_be_hidden()
    playwright.expect(frame.locator("#cancel")).to_be_visible()
    assert frame.locator("#trainingImage").evaluate("""image=>{
      const rect=image.getBoundingClientRect(),viewport=document.querySelector('main').getBoundingClientRect();
      return rect.width>850&&rect.height>230&&rect.top>=viewport.top&&rect.bottom<=viewport.bottom;
    }""")
    frame.locator("body").evaluate("() => {Object.defineProperty(document,'hidden',{configurable:true,value:true});document.dispatchEvent(new Event('visibilitychange'));}")
    page.wait_for_timeout(150)
    before = sum(action == "training_preview" for action, _ in state["calls"])
    page.wait_for_timeout(2200)
    assert sum(action == "training_preview" for action, _ in state["calls"]) == before
    frame.locator("body").evaluate("() => {Object.defineProperty(document,'hidden',{configurable:true,value:false});document.dispatchEvent(new Event('visibilitychange'));}")
    page.wait_for_timeout(600)
    assert sum(action == "training_preview" for action, _ in state["calls"]) > before
    original_frame = frame.locator("#trainingImage").get_attribute("src")
    # Hold a real image decode while a separate job poll reports completion.
    # The completed image must not restore a stale "实时采样" label.
    frame.locator("body").evaluate("""() => {
      const original=Image.prototype.decode;let hold=true;
      Image.prototype.decode=function(){const result=original.call(this);if(!hold)return result;
        hold=false;return result.then(()=>new Promise(resolve=>{
          document.body.dataset.decodeWaiting='true';window.releaseDecode=resolve;
        }));};
    }""")
    playwright.expect(frame.locator("body")).to_have_attribute("data-decode-waiting", "true", timeout=10000)
    job.update(state="failed", message="GPU 0: CUDA out of memory")
    job["progress"]["phase"] = "failed"
    playwright.expect(frame.locator("#jobMessage")).to_have_text("GPU 0: CUDA out of memory", timeout=7000)
    playwright.expect(frame.locator("#jobPhase")).to_have_text("失败")
    frame.locator("body").evaluate("() => window.releaseDecode()")
    playwright.expect(frame.locator("#previewStatus")).to_have_text("已结束", timeout=10000)
    assert frame.locator("#trainingImage").get_attribute("src") == original_frame
    playwright.expect(frame.locator("#trainingPreviewInfo")).to_contain_text("快照时间")
    page.screenshot(path=str(tmp_path / "training-monitor-fixture.png"))
    frame.locator("#trainingSetupBack").click()
    playwright.expect(frame.locator(".train-setup")).to_be_visible()
    playwright.expect(frame.locator("#train")).to_be_visible()


def test_lost_submission_response_refreshes_jobs_without_retrying(training_browser):
    page, frame, state, playwright = training_browser
    state["lose_train_response"] = True
    frame.locator("#train").click()
    playwright.expect(frame.locator("#status")).to_have_text("服务器请求超时")
    playwright.expect(frame.locator("#jobs .item")).to_have_count(2)
    assert sum(action == "train" for action, _ in state["calls"]) == 1
    assert state["jobs"][0]["state"] == "running"
    page.wait_for_timeout(3200)
    playwright.expect(frame.locator("#jobStatus")).to_have_text("排队中")
    playwright.expect(frame.locator("#cancel")).to_be_enabled()


def test_auto_mode_uses_service_plan_and_never_submits_stale_manual_ids(training_browser, tmp_path):
    page, frame, state, playwright = training_browser
    playwright.expect(frame.locator("#gpuMode")).to_have_value("auto")
    playwright.expect(frame.locator("#gpuQuickSelect")).to_be_hidden()
    playwright.expect(frame.locator("#gpuList")).to_be_hidden()
    playwright.expect(frame.locator("#allocation")).to_contain_text("预计 1 张 GPU")
    page.screenshot(path=str(tmp_path / "automatic-allocation.png"))
    frame.locator("#gpuSummary").click()
    frame.locator("#gpuMode").select_option("manual")
    frame.locator('[data-gpu-count="8"]').click()
    playwright.expect(frame.locator("#allocation")).to_contain_text("预计 8 张 GPU")
    frame.locator("#gpuMode").select_option("auto")
    frame.locator("#numEnvs").fill("65")
    playwright.expect(frame.locator("#train")).to_be_enabled()
    playwright.expect(frame.locator("#allocation")).to_contain_text("每 GPU 65 个机器人实例")
    frame.locator("#preset").select_option("standard")
    playwright.expect(frame.locator("#allocation")).to_contain_text("预计 4 张 GPU")
    frame.locator("#train").click()
    playwright.expect(frame.locator("#jobStatus")).to_have_text("排队中")
    submitted = [args for action, args in state["calls"] if action == "train"]
    assert len(submitted) == 1 and submitted[0]["gpu_mode"] == "auto"
    assert "gpu_ids" not in submitted[0]
    assert submitted[0]["num_envs"] == 1024 and submitted[0]["preset"] == "standard"
    playwright.expect(frame.locator("#jobAllocation")).to_contain_text("预计分配 4 张 GPU")
    state["jobs"][-1].update(state="running", started_at=1700000000)
    state["jobs"][-1]["config"].update(gpu_ids=[4, 5], world_size=2, envs_per_rank=512)
    page.wait_for_timeout(3200)
    playwright.expect(frame.locator("#jobAllocation")).to_have_text("实际分配 2 张 GPU · 每 GPU 512 个实例 · GPU 4、5")
    assert all("gpu_ids" not in args for action, args in state["calls"] if action == "plan" and args["gpu_mode"] == "auto")


def test_auto_resource_wait_is_queueable_and_cancellable(training_browser, tmp_path):
    page, frame, state, playwright = training_browser
    state["resources"]["training"].update(available=False, checking=True, reason="正在检查训练运行环境")
    state["plan_response"] = {"gpu_mode": "auto", "ready": False, "queueable": True,
        "reason": "等待可用 GPU 显存", "gpu_ids": [], "world_size": None, "num_envs": 128, "envs_per_rank": None}
    frame.locator("#numEnvs").fill("128")
    playwright.expect(frame.locator("#allocation")).to_have_text("等待可用 GPU 显存")
    playwright.expect(frame.locator("#train")).to_be_enabled()
    frame.locator("#train").click()
    playwright.expect(frame.locator("#jobStatus")).to_have_text("排队中")
    playwright.expect(frame.locator("#jobMessage")).to_have_text("等待可用 GPU 显存")
    playwright.expect(frame.locator("#jobProgressBar")).to_be_hidden()
    playwright.expect(frame.locator("#jobAllocation")).to_be_hidden()
    playwright.expect(frame.locator("#cancel")).to_be_enabled()
    page.screenshot(path=str(tmp_path / "automatic-resource-wait.png"))
    frame.locator("#cancel").click()
    playwright.expect(frame.locator("#jobStatus")).to_have_text("已取消")
    assert sum(action == "train" for action, _ in state["calls"]) == 1
    assert state["jobs"][0]["state"] == "running"


def test_late_plan_does_not_replace_new_form_allocation(training_browser):
    page, frame, state, playwright = training_browser
    page.evaluate("""()=>{
      const original=window.hostCall;
      window.hostCall=async message=>{
        const result=await original(message);
        if(message.args.mode==='plan'&&message.args.num_envs===128){
          document.body.dataset.planHeld='true';await new Promise(resolve=>window.releasePlan=resolve);
        }
        return result;
      };
    }""")
    frame.locator("#numEnvs").fill("128")
    playwright.expect(page.locator("body")).to_have_attribute("data-plan-held", "true")
    frame.locator("#numEnvs").fill("256")
    playwright.expect(frame.locator("#allocation")).to_have_text("")
    playwright.expect(frame.locator("#train")).to_be_disabled()
    page.evaluate("()=>window.releasePlan()")
    playwright.expect(frame.locator("#allocation")).to_contain_text("每 GPU 256 个机器人实例")
    playwright.expect(frame.locator("#train")).to_be_enabled()
    assert not any(action == "train" for action, _ in state["calls"])
