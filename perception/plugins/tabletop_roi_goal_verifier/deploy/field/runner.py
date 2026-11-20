#!/usr/bin/env python3
"""Personal-directory lifecycle. Never kills a port owner or an unverified PID."""

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

MODULE = "plugins.tabletop_roi_goal_verifier.server"


def identity(pid):
    try:
        proc = Path("/proc") / str(pid)
        cmdline = (proc / "cmdline").read_bytes().split(b"\0")
        stat = (proc / "stat").read_text().rsplit(") ", 1)[1].split()
        return {
            "start_ticks": stat[19],
            "process_state": stat[0],
            "uid": proc.stat().st_uid,
            "cmdline": [x.decode() for x in cmdline if x],
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def owned(state, root):
    current = identity(state["pid"])
    if current is None:
        return False
    return (
        current["uid"] == os.getuid()
        and current["start_ticks"] == state["start_ticks"]
        and MODULE in current["cmdline"]
        and str(root / "config.json") in current["cmdline"]
    )


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


@contextmanager
def lock(root):
    with (root / ".service.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def start(root, requested, offline=False):
    statefile = root / "service.json"
    if statefile.exists():
        previous = json.loads(statefile.read_text())
        if (old_identity := identity(previous["pid"])) is not None and old_identity[
            "process_state"
        ] != "Z":
            if owned(previous, root):
                return dict(**previous, already_running=True)
            raise RuntimeError(
                "Recorded PID now belongs to another process; refusing to touch it. Inspect service.json and use a new personal directory."
            )
    runtime = json.loads((root / "runtime.json").read_text())
    config = json.loads((root / "config.json").read_text())
    kinds = {b["kind"] for b in config["sources"].values()}
    if kinds != {"ros_image"} and not offline:
        raise RuntimeError(
            "Field startup requires real ROS image config; synthetic config requires explicit --offline-test"
        )
    if not offline and any(
        b.get("provenance", "real") != "real" or b.get("allow_test_data")
        for b in config["sources"].values()
    ):
        raise RuntimeError("Field startup refuses test provenance")
    domains = {
        b["domain_id"] for b in config["sources"].values() if b["kind"] == "ros_image"
    }
    if domains and domains != {42}:
        raise RuntimeError(
            "ROS_DOMAIN_42_REQUIRED: repository DDS contract requires domain 42; domain 222 camera cannot be read directly"
        )
    env = os.environ.copy()
    if domains:
        domain = str(next(iter(domains)))
        if env.get("ROS_DOMAIN_ID") not in (None, domain):
            raise RuntimeError("ROS_DOMAIN_ID conflicts with config")
        env["ROS_DOMAIN_ID"] = domain
        profile = root / "dds-local.xml"
        if not profile.is_file():
            raise RuntimeError(
                "Missing verified official loopback dds-local.xml in personal directory"
            )
        env["FASTRTPS_DEFAULT_PROFILES_FILE"] = str(profile)
        env["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
        env.pop("FASTDDS_BUILTIN_TRANSPORTS", None)
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        ROS_HOME=str(root / ".ros"),
        ROS_LOG_DIR=str(root / "logs/ros"),
        XDG_CACHE_HOME=str(root / ".cache"),
        TMPDIR=str(root / ".tmp"),
    )
    env["PYTHONPATH"] = (
        str(root / "deps")
        + ":"
        + str(root / "src/perception")
        + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    )
    code = "import json,sys;from plugins.tabletop_roi_goal_verifier.configuration import validate;validate(json.load(open(sys.argv[1])))"
    if domains:
        code += ";import rclpy"
    check = subprocess.run(
        [runtime["python"], "-c", code, str(root / "config.json")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if check.returncode:
        raise RuntimeError(
            "Configuration / ROS Python check failed: " + check.stderr[-3000:]
        )
    chosen = None
    for port in range(requested, min(requested + 100, 65536)):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
        chosen = port
        break
    if chosen is None:
        raise RuntimeError(
            "No free loopback port in requested range; no existing service was stopped"
        )
    with (root / "logs/server.log").open("ab") as logs:
        process = subprocess.Popen(
            [
                runtime["python"],
                "-m",
                MODULE,
                "--config",
                str(root / "config.json"),
                "--port",
                str(chosen),
                "--evidence-dir",
                str(root / "evidence"),
            ],
            cwd=root,
            env=env,
            stdout=logs,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    current = identity(process.pid)
    if current is None:
        raise RuntimeError("Server exited immediately; inspect logs/server.log")
    state = {
        "pid": process.pid,
        "start_ticks": current["start_ticks"],
        "url": f"http://127.0.0.1:{chosen}",
        "port": chosen,
        "root": str(root),
        "python": runtime["python"],
        "started_at": time.time(),
        "offline_test": offline,
    }
    write_json(statefile, state)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Server exited; inspect logs/server.log")
        try:
            with opener.open(state["url"] + "/health", timeout=0.5) as response:
                if json.load(response).get("service") == "tabletop-roi-goal-verifier":
                    body = json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tools/call",
                            "params": {
                                "name": "tabletop_roi_goal_verifier",
                                "arguments": {"action": "info"},
                            },
                        }
                    ).encode()
                    req = urllib.request.Request(
                        state["url"] + "/mcp",
                        data=body,
                        headers={"Content-Type": "application/json"},
                    )
                    with opener.open(req, timeout=2) as reply:
                        card = json.loads(
                            json.load(reply)["result"]["content"][0]["text"]
                        )
                    if card.get("state") != "running":
                        stop(root)
                        raise RuntimeError(
                            "HTTP started but ROS Card startup failed; owned service stopped. Inspect server.log"
                        )
                    return state
        except (OSError, ValueError):
            pass
        time.sleep(0.1)
    raise RuntimeError(
        "Server did not become healthy in 12s; inspect logs and stop only with this directory stop.sh"
    )


def _pidfd_call(number, *arguments):
    # Some portable Python 3.10 builds omit the pidfd bindings. The Linux
    # syscall numbers are identical on our supported x86_64/aarch64 targets.
    import ctypes
    import platform

    if sys.platform != "linux" or platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("Safe stop requires Linux pidfd support on x86_64/aarch64")
    libc = ctypes.CDLL(None, use_errno=True)
    value = libc.syscall(ctypes.c_long(number), *arguments)
    if value < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return value


def _open_pid_handle(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    import ctypes

    return _pidfd_call(434, ctypes.c_int(pid), ctypes.c_uint(0))


def _signal_pid_handle(fd, signum):
    if hasattr(signal, "pidfd_send_signal"):
        return signal.pidfd_send_signal(fd, signum)
    import ctypes

    return _pidfd_call(
        424,
        ctypes.c_int(fd),
        ctypes.c_int(signum),
        ctypes.c_void_p(0),
        ctypes.c_uint(0),
    )


def stop(root):
    statefile = root / "service.json"
    if not statefile.exists():
        return {"state": "not_started"}
    state = json.loads(statefile.read_text())
    if (recorded := identity(state["pid"])) is None or recorded["process_state"] == "Z":
        return {"state": "already_stopped"}
    if not owned(state, root):
        raise RuntimeError("PID identity changed; refusing to signal it")
    # Pin the kernel process object before the final identity check: a numeric
    # PID could be reused between /proc validation and os.kill.
    try:
        process_fd = _open_pid_handle(state["pid"])
    except ProcessLookupError:
        return {"state": "already_stopped"}
    try:
        if not owned(state, root):
            raise RuntimeError("PID identity changed; refusing to signal it")
        try:
            _signal_pid_handle(process_fd, signal.SIGTERM)
        except ProcessLookupError:
            return {"state": "already_stopped"}
    finally:
        os.close(process_fd)
    for _ in range(80):
        current = identity(state["pid"])
        if current is None or current["process_state"] == "Z":
            state.update(stopped_at=time.time())
            write_json(statefile, state)
            return {"state": "stopped", "pid": state["pid"]}
        time.sleep(0.1)
    raise RuntimeError(
        "Owned service did not stop in 8s; no forced kill issued; inspect logs/server.log"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["start", "stop"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--port", type=int, default=18084)
    p.add_argument("--offline-test", action="store_true")
    args = p.parse_args()
    root = args.root.resolve(strict=True)
    if not 1024 <= args.port <= 65436:
        p.error("port must be 1024..65436")
    try:
        with lock(root):
            out = (
                start(root, args.port, args.offline_test)
                if args.action == "start"
                else stop(root)
            )
        print(json.dumps(out, indent=2))
    except Exception as exc:  # noqa: BLE001 — CLI prints an actionable diagnostic
        print("Lifecycle failed: " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
