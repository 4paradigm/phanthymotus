from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SERVICE_FILE = ROOT / "perception" / "deploy" / "service.yml"
EDGE_BACKEND_FILE = ROOT / "perception" / "deploy" / "windows_bridge" / "edge_backend.py"
SSH_WORKER_FILE = ROOT / "perception" / "deploy" / "windows_bridge" / "ssh_worker.py"


def test_service_template_does_not_require_new_printer_mount():
    service_document = yaml.safe_load(SERVICE_FILE.read_text(encoding="utf-8"))
    perception = service_document["perception"]
    assert not any("ql_printer" in volume for volume in perception["volumes"])


def test_documentation_does_not_tell_operators_to_commit_private_keys():
    text = (ROOT / "docs" / "ql-printer.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "Do not place private keys in this repository" in normalized
    assert "No manual public-key registration" in text


def test_image_retains_offline_windows_dependencies():
    source = (ROOT / "perception" / "Dockerfile.jetson").read_text(encoding="utf-8")
    assert "RUN python3 /deploy/windows_bridge/build_offline.py" in source


def test_print_waits_for_browser_afterprint_before_edge_closes():
    source = EDGE_BACKEND_FILE.read_text(encoding="utf-8")
    assert "window.__paradigmPrintState = 'submitted'" in source
    assert "window.addEventListener('afterprint'" in source
    assert "window.__paradigmPrintState = 'complete'" in source
    assert "setTimeout(nativeClose, 2000)" in source
    assert "candidate not in existing_pages" in source
    assert "candidate.evaluate(\"window.__paradigmPrintState || 'idle'\")" in source


def test_worker_keeps_a_grace_period_for_late_queued_requests():
    source = SSH_WORKER_FILE.read_text(encoding="utf-8")
    assert "time.monotonic() - idle_since >= 10" in source
