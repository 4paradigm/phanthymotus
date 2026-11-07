"""Image layout and interpreter contracts, without claiming a Docker build."""

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_goal_sources_parse_with_jetpack_python38_grammar():
    sources = list((ROOT / "perception/plugins/goal_verifier").rglob("*.py"))
    sources += list((ROOT / "perception/tools").glob("goal_verifier*.py"))
    for path in sources:
        ast.parse(path.read_text(), filename=str(path), feature_version=8)


def test_package_paths_remain_confined_on_supported_interpreters(tmp_path):
    from plugins.goal_verifier.packaging import safe
    from plugins.goal_verifier.contracts import Refusal

    root = tmp_path / "payload"
    root.mkdir()
    assert safe(root, "sub/file.py") == root / "sub/file.py"
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    for name in ("../outside", str(outside), "link/file.py", "link"):
        with pytest.raises(Refusal) as refused:
            safe(root, name)
        assert refused.value.code == "UNSAFE_PACKAGE_PATH"


def test_docker_application_copy_loads_card_without_development_checkout(tmp_path):
    dockerfile = (ROOT / "perception/Dockerfile.jetson").read_text()
    for source, target in (("main.py", "/work/main.py"), ("plugins/", "/work/plugins/"),
                           ("utils/", "/work/utils/"), ("config.yaml", "/work/config.yaml")):
        assert any(line.split() == ["COPY", "perception/" + source, target]
                   for line in dockerfile.splitlines())
    work = tmp_path / "image-work"
    work.mkdir()
    for folder in ("plugins", "utils"):
        shutil.copytree(ROOT / "perception" / folder, work / folder,
                        ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("main.py", "config.yaml"):
        shutil.copy2(ROOT / "perception" / name, work / name)
    code = """
import json, sys
from plugins.goal_verifier import GoalVerifierPlugin
card = GoalVerifierPlugin({'piper_id': 'OFFLINE-IMAGE-CONTRACT', 'sources': {}})
tools = card.get_tools()
assert len(tools) == 1 and tools[0]['name'] == 'piper_object_goal_verifier'
assert tools[0]['inputSchema']['properties']['session_token']['x-sensitive'] is True
assert not card.started
assert not any(n.startswith(('piper_sdk', 'agilex', 'can')) for n in sys.modules)
card.stop()
print(json.dumps({'tool': tools[0]['name'], 'motion_executed': False}))
"""
    env = {**os.environ, "PYTHONPATH": str(work), "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONNOUSERSITE": "1"}
    result = subprocess.run([sys.executable, "-c", code], cwd=work, env=env,
                            text=True, capture_output=True, timeout=20, check=True)
    assert json.loads(result.stdout)["tool"] == "piper_object_goal_verifier"
