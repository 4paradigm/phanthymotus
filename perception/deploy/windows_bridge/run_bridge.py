from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from bridge_core import BridgeStore, VisitorBridge
from edge_backend import EdgeVisitorBackend
from http_server import serve


def main():
    parser = argparse.ArgumentParser(description="范式前台访客桥接服务")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    base = config_path.parent

    log_path = base / "data" / "bridge.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() and log_path.stat().st_size > 5 * 1024 * 1024:
        rotated = log_path.with_suffix(".log.1")
        if rotated.exists():
            rotated.unlink()
        os.replace(log_path, rotated)
    log_stream = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = log_stream
    sys.stderr = log_stream

    database_path = Path(config.get("database_path", "data/bridge.db"))
    if not database_path.is_absolute():
        database_path = base / database_path
    edge_config = dict(config["edge"])
    profile_dir = Path(edge_config.get("profile_dir", "data/edge-profile"))
    if not profile_dir.is_absolute():
        edge_config["profile_dir"] = str(base / profile_dir)

    token = str(config.get("shared_token", "")).strip()
    if not token or "请换成" in token:
        raise SystemExit("config.json 中没有有效 shared_token")

    backend = EdgeVisitorBackend(edge_config, base / "data")
    store = BridgeStore(database_path)
    bridge = VisitorBridge(
        backend,
        store,
        ticket_seconds=float(config.get("ticket_seconds", 12 * 3600)),
        printable_statuses=tuple(config.get("printable_statuses", ["未入场"])),
        success_statuses=tuple(config.get("success_statuses", ["已入场"])),
    )
    try:
        serve(bridge, token, str(config.get("bind_host", "0.0.0.0")), int(config.get("port", 8765)))
    finally:
        backend.close()
        store.close()


if __name__ == "__main__":
    main()

