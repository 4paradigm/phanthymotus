"""Run queued visitor-browser jobs on demand; no network service is opened."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

from bridge_core import BridgeFailure, BridgeStore, VisitorBridge
from edge_backend import EdgeVisitorBackend


def _write_result(path: Path, result: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _run_job(bridge: VisitorBridge, backend: EdgeVisitorBackend, job: dict) -> dict:
    action = str(job.get("action") or "")
    body = job.get("body") if isinstance(job.get("body"), dict) else {}
    if action == "health":
        return {"state": "running", "transport": "ssh"}
    if action == "printers":
        result = backend.list_printers()
        pages_path = Path(backend.data_dir) / "edge-pages.json"
        try:
            pages = json.loads(pages_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            pages = []
        result["pages"] = pages if isinstance(pages, list) else []
        return result
    selected_url = str(body.get("visitor_page_url") or "").strip()
    parsed = urlsplit(selected_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise BridgeFailure("visitor_page_required", "请在卡片配置中选择已登录并能看到访客列表的网页")
    backend.list_url = selected_url
    backend.query_url = f"{parsed.scheme}://{parsed.netloc}/smart-admin-api/visitInfo/page/query"
    if action == "list":
        return bridge.list_visits()
    if action == "print":
        return bridge.print_tag(
            str(body.get("ticket") or ""),
            str(body.get("request_id") or ""),
            str(body.get("printer_name") or ""),
        )
    raise BridgeFailure("bad_action", "不支持的 SSH 自动化动作")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    base = config_path.parent
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    jobs = base / "data" / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)

    database_path = Path(config.get("database_path", "data/automation.db"))
    if not database_path.is_absolute():
        database_path = base / database_path
    edge_config = dict(config["edge"])
    profile = Path(edge_config.get("profile_dir", "data/edge-profile"))
    if not profile.is_absolute():
        edge_config["profile_dir"] = str(base / profile)

    backend = EdgeVisitorBackend(edge_config, base / "data")
    store = BridgeStore(database_path)
    bridge = VisitorBridge(
        backend,
        store,
        ticket_seconds=float(config.get("ticket_seconds", 300)),
        printable_statuses=tuple(config.get("printable_statuses", ["未入场"])),
        success_statuses=tuple(config.get("success_statuses", ["已入场"])),
    )
    try:
        # Drain every request visible at startup. Task Scheduler serializes
        # runs, while newly arriving files are picked up during a grace period
        # so a request racing a just-finished Task Scheduler invocation is not
        # stranded until another call happens to wake the task.
        idle_since = None
        while True:
            requests = sorted(jobs.glob("*.request.json"), key=lambda item: item.stat().st_mtime)
            if not requests:
                if idle_since is None:
                    idle_since = time.monotonic()
                if time.monotonic() - idle_since >= 10:
                    break
                time.sleep(0.1)
                continue
            idle_since = None
            request_path = requests[0]
            result_path = request_path.with_name(request_path.name.replace(".request.json", ".result.json"))
            try:
                job = json.loads(request_path.read_text(encoding="utf-8-sig"))
                result = _run_job(bridge, backend, job)
            except BridgeFailure as exc:
                result = {"ok": False, "code": exc.code, "error": str(exc)}
            except Exception as exc:
                result = {"ok": False, "code": "automation_error", "error": str(exc)[:500]}
            finally:
                request_path.unlink(missing_ok=True)
            _write_result(result_path, result)
    finally:
        backend.close()
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
