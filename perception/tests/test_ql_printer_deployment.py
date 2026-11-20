from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SERVICE_FILE = ROOT / "perception" / "deploy" / "service.yml"


def test_service_template_does_not_require_robot_specific_printer_credentials():
    service_document = yaml.safe_load(SERVICE_FILE.read_text(encoding="utf-8"))
    perception = service_document["perception"]
    assert not any("ql_printer" in volume for volume in perception["volumes"])


def test_documentation_does_not_tell_operators_to_commit_private_keys():
    text = (ROOT / "docs" / "ql-printer.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "Do not place private keys in this repository" in normalized
    assert "/run/ql-printer-auth/frontdesk_bridge_ed25519" in text
