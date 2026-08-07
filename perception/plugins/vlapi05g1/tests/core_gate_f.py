#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
import ssl
import subprocess
import time
import urllib.error
import urllib.request


class HttpClient:
    def __init__(self, core_url: str, mcp_url: str, server_name: str):
        self.core_url = core_url.rstrip("/")
        self.mcp_url = mcp_url
        self.server_name = server_name
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        self.core_opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPSHandler(context=context),
        )
        self.mcp_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.mcp_id = ""
        self.request_id = 0

    def core_request(self, method: str, path: str, body=None) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.core_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        with self.core_opener.open(request, timeout=30.0) as response:
            return json.loads(response.read())

    def wait_for_unique_registration(self, timeout_s: float = 20.0) -> dict:
        deadline = time.monotonic() + timeout_s
        matches = []
        while time.monotonic() < deadline:
            items = self.core_request("GET", "/api/mcp")["data"]
            matches = [item for item in items if item.get("server_name") == self.server_name]
            if len(matches) == 1 and matches[0].get("tools"):
                self.mcp_id = matches[0]["id"]
                return matches[0]
            time.sleep(0.5)
        raise RuntimeError(f"expected one Core registration, got {len(matches)}")

    def save_config(self, request_timeout_s: float) -> None:
        response = self.core_request(
            "PUT",
            f"/api/canvas/tool-config/{self.mcp_id}/vlapi05g1",
            {
                "policy_url": "http://127.0.0.1:18080/predict",
                "health_url": "",
                "request_timeout_s": request_timeout_s,
                "max_image_bytes": 8388608,
            },
        )
        if response.get("code") != 200:
            raise RuntimeError(f"Core config save failed: {response}")

    def core_call(self, arguments: dict) -> dict:
        response = self.core_request(
            "POST",
            f"/api/mcp/{self.mcp_id}/call",
            {"tool": "vlapi05g1", "arguments": arguments},
        )
        if response.get("code") != 200:
            raise RuntimeError(f"Core tool call failed: {response}")
        return json.loads(response["data"][0]["text"])

    def direct_rpc(self, method: str, params=None) -> dict:
        self.request_id += 1
        request = urllib.request.Request(
            self.mcp_url,
            data=json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": self.request_id,
                    "method": method,
                    "params": params or {},
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.mcp_opener.open(request, timeout=10.0) as response:
            return json.loads(response.read())

    def direct_call(self, arguments: dict) -> dict:
        response = self.direct_rpc(
            "tools/call", {"name": "vlapi05g1", "arguments": arguments}
        )
        if "error" in response:
            raise RuntimeError(f"direct MCP call failed: {response}")
        return json.loads(response["result"]["content"][0]["text"])

    def wait_for_direct_initialize(self, timeout_s: float = 20.0) -> dict:
        deadline = time.monotonic() + timeout_s
        last_error = None
        while time.monotonic() < deadline:
            try:
                return self.direct_rpc("initialize")
            except urllib.error.URLError as exc:
                last_error = exc
                time.sleep(0.5)
        raise RuntimeError(f"direct MCP did not become ready: {last_error}")

    def wait_for_actual_timeout(self, expected: float, timeout_s: float = 10.0) -> dict:
        deadline = time.monotonic() + timeout_s
        last_info = None
        while time.monotonic() < deadline:
            last_info = self.direct_call({"action": "info"})
            if last_info["shared_config"]["request_timeout_s"] == expected:
                return last_info
            time.sleep(0.2)
        raise RuntimeError(
            f"card request_timeout_s did not become {expected}: {last_info}"
        )


def run(*command: str) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


def container_resource_sample(card_container: str, policy_container: str) -> dict:
    card_pid = int(run("docker", "inspect", "-f", "{{.State.Pid}}", card_container))
    status_text = open(f"/proc/{card_pid}/status", encoding="utf-8").read()
    status = {}
    for key in ("Threads", "VmRSS"):
        match = re.search(rf"^{key}:\s+(\d+)", status_text, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"missing {key} in /proc/{card_pid}/status")
        status[key] = int(match.group(1))

    process_lines = run("docker", "top", card_container, "-eo", "pid,comm").splitlines()
    graph_script = """
import time
import rclpy
from rclpy.node import Node
rclpy.init()
node = Node("vlapi05g1_gate_f_graph_probe")
deadline = time.monotonic() + 0.5
while time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.05)
print("\\n".join(sorted(node.get_node_names())))
node.destroy_node()
rclpy.shutdown()
"""
    ros_nodes = run(
        "docker",
        "exec",
        card_container,
        "bash",
        "-lc",
        "source /opt/ros/humble/setup.bash && python3 -c " + shlex.quote(graph_script),
    ).splitlines()
    policy_pid = int(run("docker", "inspect", "-f", "{{.State.Pid}}", policy_container))
    gpu_rows = run(
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ).splitlines()
    policy_gpu_memory_mib = None
    for row in gpu_rows:
        columns = [column.strip() for column in row.split(",")]
        if len(columns) == 2 and columns[0] == str(policy_pid):
            policy_gpu_memory_mib = int(columns[1])
            break
    device_requests = json.loads(
        run("docker", "inspect", "-f", "{{json .HostConfig.DeviceRequests}}", card_container)
    )
    return {
        "card_pid": card_pid,
        "card_threads": status["Threads"],
        "card_rss_kib": status["VmRSS"],
        "card_process_count": max(0, len(process_lines) - 1),
        "vlapi05g1_ros_nodes": sorted(
            name for name in ros_nodes if name.startswith("/vlapi05g1_")
        ),
        "card_gpu_device_requests": device_requests,
        "policy_pid": policy_pid,
        "policy_gpu_memory_mib": policy_gpu_memory_mib,
    }


def assert_logs_redacted(card_container: str) -> dict:
    logs = run("docker", "logs", "--since", "30m", card_container)
    patterns = {
        "credential_value": re.compile(
            r"authorization[ \t]*[:=][ \t]*[\"']?(?:bearer[ \t]+)?[a-z0-9._~+/=-]{8,}"
            r"|bearer[ \t]+[a-z0-9._~+/=-]{8,}"
            r"|api[_-]?key[ \t]*[:=][ \t]*[\"']?[a-z0-9._~+/=-]{8,}"
            r"|x-execution-token[ \t]*[:=][ \t]*[\"']?[a-z0-9._~+/=-]{8,}",
            re.IGNORECASE,
        ),
        "image_base64": re.compile(
            r"image_base64.{0,80}[A-Za-z0-9+/]{80}", re.IGNORECASE | re.DOTALL
        ),
        "action_chunk": re.compile(r'"action_chunk"\s*:'),
    }
    hits = {name: bool(pattern.search(logs)) for name, pattern in patterns.items()}
    if any(hits.values()):
        raise AssertionError(f"sensitive or oversized payload found in logs: {hits}")
    return {"checked_bytes": len(logs.encode("utf-8")), "pattern_hits": hits}


def stop_test_ros_daemon(card_container: str) -> None:
    subprocess.run(
        [
            "docker",
            "exec",
            card_container,
            "bash",
            "-lc",
            "source /opt/ros/humble/setup.bash && ros2 daemon stop",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run vlapi05g1 Core restart/resource Gate F")
    parser.add_argument("--core-url", default="https://127.0.0.1:15678")
    parser.add_argument("--mcp-url", default="http://127.0.0.1:15735/mcp")
    parser.add_argument("--server-name", required=True)
    parser.add_argument("--card-container", required=True)
    parser.add_argument("--policy-container", default="pi05-g1-policy")
    args = parser.parse_args()

    client = HttpClient(args.core_url, args.mcp_url, args.server_name)
    first_registration = client.wait_for_unique_registration()
    original_mcp_id = first_registration["id"]
    client.save_config(121.0)
    run("docker", "restart", args.card_container)
    initialize = client.wait_for_direct_initialize()
    restarted_registration = client.wait_for_unique_registration()
    if initialize["result"]["serverInfo"]["name"] != args.server_name:
        raise AssertionError("MCP identity changed after restart")
    before_lazy = client.direct_call({"action": "info"})
    if before_lazy["shared_config"]["request_timeout_s"] != 120.0:
        raise AssertionError(f"unexpected startup config: {before_lazy['shared_config']}")

    lazy_result = client.core_call(
        {"action": "predict", "instance_id": "stage6-gate-f-lazy-config"}
    )
    if lazy_result.get("error_code") != "not_running":
        raise AssertionError(f"unexpected lazy replay business result: {lazy_result}")
    after_lazy = client.direct_call({"action": "info"})
    if after_lazy["shared_config"]["request_timeout_s"] != 121.0:
        raise AssertionError(f"Core saved config was not replayed: {after_lazy['shared_config']}")
    client.save_config(120.0)
    restored = client.wait_for_actual_timeout(120.0)

    stop_test_ros_daemon(args.card_container)
    instance_id = "stage6-gate-f-cycle"

    def run_cycle(label: str) -> None:
        started = client.core_call(
            {
                "action": "start",
                "instance_id": instance_id,
                "input_topics": [
                    "/stage6/gate_f/cycle/camera/compressed",
                    "/stage6/gate_f/cycle/g1/joints",
                ],
            }
        )
        if started.get("state") != "running":
            raise AssertionError(f"{label} start failed: {started}")
        stopped = client.core_call({"action": "stop", "instance_id": instance_id})
        if stopped.get("state") != "idle":
            raise AssertionError(f"{label} stop failed: {stopped}")
        direct_info = client.direct_call({"action": "info"})
        if direct_info.get("instances"):
            raise AssertionError(f"{label} left registered instances")

    run_cycle("warm-up cycle")
    time.sleep(1.0)
    baseline = container_resource_sample(args.card_container, args.policy_container)
    for index in range(10):
        run_cycle(f"cycle {index + 1}")

    time.sleep(2.0)
    final = container_resource_sample(args.card_container, args.policy_container)
    if final["card_process_count"] != baseline["card_process_count"]:
        raise AssertionError("card process count changed across 10 cycles")
    if final["card_threads"] > baseline["card_threads"] + 1:
        raise AssertionError("card threads leaked across 10 cycles")
    if final["card_rss_kib"] > baseline["card_rss_kib"] + 8192:
        raise AssertionError(
            "card RSS grew by more than 8 MiB across 10 stable-state cycles: "
            f"{baseline['card_rss_kib']} -> {final['card_rss_kib']} KiB"
        )
    if final["vlapi05g1_ros_nodes"]:
        raise AssertionError("vlapi05g1 ROS nodes remain after 10 cycles")
    if final["card_gpu_device_requests"]:
        raise AssertionError("card container unexpectedly requests a GPU")
    if final["policy_pid"] != baseline["policy_pid"]:
        raise AssertionError("external policy process restarted during card lifecycle test")
    if final["policy_gpu_memory_mib"] != baseline["policy_gpu_memory_mib"]:
        raise AssertionError("external policy GPU memory changed during card lifecycle test")
    log_result = assert_logs_redacted(args.card_container)

    print(
        json.dumps(
            {
                "status": "PASS",
                "mcp_id_before_restart": original_mcp_id,
                "mcp_id_after_restart": restarted_registration["id"],
                "registration_count_after_restart": 1,
                "startup_request_timeout_s": before_lazy["shared_config"]["request_timeout_s"],
                "lazy_business_result": lazy_result["error_code"],
                "replayed_request_timeout_s": after_lazy["shared_config"]["request_timeout_s"],
                "restored_request_timeout_s": restored["shared_config"]["request_timeout_s"],
                "lifecycle_warmup_cycles": 1,
                "lifecycle_cycles": 10,
                "resource_baseline": baseline,
                "resource_final": final,
                "logs": log_result,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
