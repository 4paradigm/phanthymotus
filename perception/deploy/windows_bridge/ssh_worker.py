"""Run queued visitor-browser jobs on demand; no network service is opened.

On startup the worker drains every *.request.json file visible in the jobs
directory, then waits a short grace period for late-arriving files (so a
request that races a just-finished Task Scheduler invocation is not stranded
until the next call wakes the task).

Reliability features
--------------------
* All stdout/stderr is appended to ``data/worker.log`` (rotated at 5 MB).
* If backend / store / bridge construction fails before the job loop starts,
  every pending *.request.json gets a clear ``worker_startup_failed`` result
  instead of being silently abandoned.
* A ``data/jobs/worker.started`` sentinel is created right before the job
  loop, so callers can distinguish "task never started" from "task started
  but is busy initialising Edge".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from urllib.parse import urlsplit


def _setup_logging(base: Path) -> Path:
    """Redirect stdout/stderr to a rotating log file; return the log path."""
    log_path = base / "data" / "worker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() and log_path.stat().st_size > 5 * 1024 * 1024:
        rotated = log_path.with_suffix(".log.1")
        if rotated.exists():
            rotated.unlink()
        os.replace(log_path, rotated)
    # Line-buffered so that crash-time output is flushed.
    log_stream = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = log_stream
    sys.stderr = log_stream
    return log_path


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _write_result(path: Path, result: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _fail_all_pending(jobs_dir: Path, error_code: str, error_message: str) -> None:
    """Write an error result for every pending request and remove the request.

    Used when the worker cannot even start the job loop: without this,
    pending requests sit forever in the directory and the caller times out
    with no information about what went wrong.
    """
    pending = sorted(jobs_dir.glob("*.request.json"), key=lambda p: p.stat().st_mtime)
    if not pending:
        return
    result = {"ok": False, "code": error_code, "error": error_message}
    for request_path in pending:
        result_path = request_path.with_name(
            request_path.name.replace(".request.json", ".result.json")
        )
        try:
            _write_result(result_path, result)
        except OSError:
            pass
        try:
            request_path.unlink(missing_ok=True)
        except OSError:
            pass
    print(f"[{_ts()}] [worker] failed {len(pending)} pending request(s): {error_code}", flush=True)


def _run_job(bridge, backend, job: dict) -> dict:
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
        from bridge_core import BridgeFailure
        raise BridgeFailure("visitor_page_required", "请在卡片配置中选择已登录并能看到访客列表的网页")
    backend.list_url = selected_url
    backend.query_url = f"{parsed.scheme}://{parsed.netloc}/smart-admin-api/visitInfo/page/query"
    if action == "list":
        return bridge.list_visits()
    if action == "print":
        request_id = str(body.get("request_id") or "")
        name = str(body.get("name") or "").strip()
        if name:
            return bridge.print_today_visit(
                name,
                str(body.get("host") or ""),
                request_id,
                str(body.get("printer_name") or ""),
            )
        return bridge.print_tag(
            str(body.get("ticket") or ""),
            request_id,
            str(body.get("printer_name") or ""),
        )
    from bridge_core import BridgeFailure
    raise BridgeFailure("bad_action", "不支持的 SSH 自动化动作")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    base = config_path.parent

    # Redirect output as early as possible so import-time errors are captured.
    try:
        _setup_logging(base)
    except OSError as exc:
        # Last-ditch: print to stderr, which the Task Scheduler may capture.
        print(f"[worker] cannot open log file: {exc}", file=sys.stderr)
        return 2

    print(f"[{_ts()}] [worker] starting (pid={os.getpid()})", flush=True)

    jobs_dir = base / "data" / "jobs"
    try:
        jobs_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[{_ts()}] [worker] cannot create jobs dir: {exc}", flush=True)
        return 2

    # Load config first (it's cheap); if this fails there is nothing to do.
    try:
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        msg = f"无法读取配置文件：{exc}"
        _fail_all_pending(jobs_dir, "config_error", msg)
        print(f"[{_ts()}] [worker] config error: {exc}", flush=True)
        return 1

    database_path = Path(config.get("database_path", "data/automation.db"))
    if not database_path.is_absolute():
        database_path = base / database_path
    edge_config = dict(config["edge"])
    profile = Path(edge_config.get("profile_dir", "data/edge-profile"))
    if not profile.is_absolute():
        edge_config["profile_dir"] = str(base / profile)

    # Build backend / store / bridge inside a single try so that ANY startup
    # failure (Edge profile lock, SQLite corruption, missing dependency, …)
    # produces actionable result files instead of silently leaving requests
    # stranded in the jobs directory.
    backend = None
    store = None
    try:
        from bridge_core import BridgeFailure, BridgeStore, VisitorBridge
        from edge_backend import EdgeVisitorBackend

        backend = EdgeVisitorBackend(edge_config, base / "data")
        store = BridgeStore(database_path)
        bridge = VisitorBridge(
            backend,
            store,
            ticket_seconds=float(config.get("ticket_seconds", 12 * 3600)),
            printable_statuses=tuple(config.get("printable_statuses", ["未入场"])),
            success_statuses=tuple(config.get("success_statuses", ["已入场"])),
        )
    except Exception as exc:
        tb = traceback.format_exc(limit=3)
        msg = f"自动化启动失败：{exc}"
        _fail_all_pending(jobs_dir, "worker_startup_failed", msg)
        print(f"[{_ts()}] [worker] startup error: {exc}", flush=True)
        print(tb, flush=True)
        # Best-effort cleanup of anything that was partially constructed.
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
        return 1

    # Write the started sentinel so callers can distinguish "task didn't
    # start" from "task started but Edge is still warming up".
    started_sentinel = jobs_dir / "worker.started"
    try:
        started_sentinel.write_text(
            json.dumps({"pid": os.getpid(), "started_at": time.time()}),
            encoding="utf-8",
        )
    except OSError:
        pass

    print(f"[{_ts()}] [worker] ready, entering job loop", flush=True)
    processed = 0
    try:
        idle_since = None
        while True:
            requests = sorted(
                jobs_dir.glob("*.request.json"),
                key=lambda item: item.stat().st_mtime,
            )
            if not requests:
                if idle_since is None:
                    idle_since = time.monotonic()
                if time.monotonic() - idle_since >= 10:
                    break
                time.sleep(0.1)
                continue
            idle_since = None
            request_path = requests[0]
            result_path = request_path.with_name(
                request_path.name.replace(".request.json", ".result.json")
            )
            job_id = request_path.stem.replace(".request", "")
            print(f"[{_ts()}] [worker] job {job_id[:8]} start", flush=True)
            try:
                job = json.loads(request_path.read_text(encoding="utf-8-sig"))
                result = _run_job(bridge, backend, job)
            except BridgeFailure as exc:
                result = {"ok": False, "code": exc.code, "error": str(exc)}
            except Exception as exc:
                result = {"ok": False, "code": "automation_error", "error": str(exc)[:500]}
                print(f"[{_ts()}] [worker] job {job_id[:8]} error: {exc}", flush=True)
            finally:
                request_path.unlink(missing_ok=True)
            _write_result(result_path, result)
            processed += 1
            ok = result.get("ok", False)
            code = "" if ok else str(result.get("code", ""))
            print(
                f"[{_ts()}] [worker] job {job_id[:8]} done ok={ok}"
                + (f" code={code}" if code else ""),
                flush=True,
            )
    finally:
        # Remove the sentinel before we tear down Edge, so the next caller
        # that finds it can trust that a worker is actually alive.
        try:
            started_sentinel.unlink(missing_ok=True)
        except OSError:
            pass
        print(f"[{_ts()}] [worker] exiting, processed {processed} job(s)", flush=True)
        try:
            backend.close()
        except Exception as exc:
            print(f"[{_ts()}] [worker] backend close warning: {exc}", flush=True)
        try:
            store.close()
        except Exception as exc:
            print(f"[{_ts()}] [worker] store close warning: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
