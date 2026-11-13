"""Run the unchanged Core app for local integration, without host bootstrap.

Its actual HTTP routes, Canvas, MCP client and DDS bridge run. The production
lifespan also manages Docker, networking, peers and agents; those operations
are outside this local software test. No fake API or MCP client is installed.
"""

import argparse
import asyncio
import json
import os
import socket
import sys
from pathlib import Path


async def serve(core_root, runtime):
    os.chdir(runtime)
    sys.path.insert(0, str(core_root / "src"))
    os.environ["DB_PATH"] = str(runtime / "data.db")
    import config

    config.main["core"] = {
        "configured": True,
        "auto_start": False,
        "main_loop_enable": False,
    }
    config.main["services"] = {
        "mcp": [],
        "resource_center": {"url": "http://127.0.0.1:1"},
    }
    config.main["peer_settings"] = {"enabled": False}
    import auth
    import ros2_bridge
    import start
    import uvicorn

    auth._ENV_PATH = runtime / ".env"
    auth._ENV_PATH_DEV = runtime / ".env"
    auth.init()
    start._register_core_mcp(silent=True)
    ros2_bridge.start(asyncio.get_running_loop())

    async def diagnostics():
        while True:
            import threading

            data = {
                "running": ros2_bridge._running,
                "spin_alive": ros2_bridge._thread.is_alive(),
                "last_seen": dict(ros2_bridge._last_seen),
                "rmw": os.getenv("RMW_IMPLEMENTATION"),
                "domain": os.getenv("ROS_DOMAIN_ID"),
                "subscriptions": {
                    k: {
                        "topic": v["topic"],
                        "publishers": ros2_bridge._node_main.count_publishers(
                            v["topic"]
                        ),
                    }
                    for k, v in ros2_bridge._subs.items()
                },
                "threads": [t.name for t in threading.enumerate()],
            }
            (runtime / "dds-diagnostics.json").write_text(json.dumps(data, indent=2))
            await asyncio.sleep(0.5)

    diagnostic_task = asyncio.create_task(diagnostics())
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(start.app, log_level="warning", access_log=False, lifespan="off")
    )
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        while not server.started and not task.done():
            await asyncio.sleep(0.02)
        (runtime / "port.json").write_text(
            json.dumps(
                {
                    "port": port,
                    "production_lifespan": False,
                    "real_http_mcp_dds_canvas": True,
                }
            )
        )
        await task
    finally:
        diagnostic_task.cancel()
        ros2_bridge.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(serve(args.core_root.resolve(), args.runtime.resolve()))
