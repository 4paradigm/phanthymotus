from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
SERVICE_FILE = ROOT / "perception" / "deploy" / "service.yml"
sys.path.insert(0, str(ROOT / "perception" / "deploy" / "windows_bridge"))

from edge_backend import print_readiness_script


def test_service_template_does_not_require_robot_specific_printer_credentials():
    service_document = yaml.safe_load(SERVICE_FILE.read_text(encoding="utf-8"))
    perception = service_document["perception"]
    assert not any("ql_printer" in volume for volume in perception["volumes"])


def test_documentation_does_not_tell_operators_to_commit_private_keys():
    text = (ROOT / "docs" / "ql-printer.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "Do not place private keys in this repository" in normalized
    assert "/run/ql-printer-auth/frontdesk_bridge_ed25519" in text


def test_print_waits_for_browser_afterprint_before_edge_closes():
    script = print_readiness_script()
    assert "window.__paradigmPrintState = 'submitted'" in script
    assert "window.addEventListener('afterprint'" in script
    assert "window.__paradigmPrintState = 'complete'" in script
