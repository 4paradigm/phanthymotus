"""Self-contained Card payload and ownership-checked local stage/rollback."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import difflib
import subprocess
from uuid import uuid4

from .contracts import need
from .server import loads

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    payload = output / "payload"
    payload.mkdir()
    files = []
    for folder in (
        "perception/plugins/goal_verifier",
        "perception/plugins/measure",
        "perception/utils",
    ):
        files.extend(
            p
            for p in (ROOT / folder).rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and p.suffix in (".py", ".json", ".md", ".txt", ".sh", ".example")
        )
    files.append(ROOT / "perception/plugins/goal_verifier/workbench.html")
    files.extend(
        ROOT / p
        for p in (
            "LICENSE",
            "agent-core/web/js/sidebar.js",
            "perception/main.py",
            "perception/plugins/vop.py",
            "perception/plugins/vision_runtime.py",
            "perception/plugins/image_input.py",
            "perception/plugins/camera_info.py",
            "perception/tests/vision_stubs.py",
            "perception/tests/conftest.py",
            "perception/tests/test_goal_verifier.py",
            "perception/tests/test_goal_verifier_integration.py",
            "perception/tests/test_goal_verifier_build.py",
            "perception/tools/goal_verifier_cli.py",
            "perception/tools/goal_verifier_e2e.py",
            "perception/tools/goal_verifier_ros_probe.py",
            "perception/tools/goal_verifier_real_image.py",
        )
    )
    manifest = {}
    for path in sorted(set(files)):
        need(path.is_file() and not path.is_symlink(), "PACKAGE_SOURCE_MISSING")
        relative = path.relative_to(ROOT)
        target = payload / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        manifest[str(relative)] = sha(target)
    # Reviewable optional Core patch only. No install/apply/restart action.
    try:
        base = subprocess.check_output(
            ["git", "merge-base", "HEAD", "origin/main"], cwd=ROOT, text=True
        ).strip()
        original = subprocess.check_output(
            ["git", "show", base + ":agent-core/web/js/sidebar.js"], cwd=ROOT, text=True
        )
        updated = (ROOT / "agent-core/web/js/sidebar.js").read_text()
        patch = "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile="a/agent-core/web/js/sidebar.js",
                tofile="b/agent-core/web/js/sidebar.js",
            )
        )
        target = payload / "integration/core-readonly-entry.patch"
        target.parent.mkdir()
        target.write_text(patch)
        manifest["integration/core-readonly-entry.patch"] = sha(target)
    except (subprocess.CalledProcessError, FileNotFoundError):
        base = None
    (payload / "MANIFEST.json").write_text(
        json.dumps(
            dict(
                schema="motus.piper-goal-package/1",
                files=manifest,
                robot_motion=False,
                defaults="unconfigured_no_test_data",
                core_patch_base=base,
            ),
            indent=2,
        )
        + "\n"
    )
    archive = output / "piper-goal-verifier.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(payload, arcname="piper-goal-verifier")
    result = dict(
        archive=str(archive),
        sha256=sha(archive),
        files=len(manifest),
        state="DONE",
        deployment_performed=False,
    )
    (output / "package.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def safe(root, name):
    p = Path(name)
    need(not p.is_absolute() and ".." not in p.parts, "UNSAFE_PACKAGE_PATH")
    path = root / p
    # Python 3.8 is shipped by the official JetPack 5.11 image. relative_to
    # rejects escapes on that interpreter too; is_relative_to needs 3.9.
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        need(False, "UNSAFE_PACKAGE_PATH")
    need(not path.is_symlink(), "UNSAFE_PACKAGE_PATH")
    return path


def stage(payload, target):
    payload, target = Path(payload), Path(target)
    m = loads((payload / "MANIFEST.json").read_text())
    need(m["schema"] == "motus.piper-goal-package/1", "PACKAGE_SCHEMA")
    for name, hashed in m["files"].items():
        need(sha(safe(payload, name)) == hashed, "PACKAGE_HASH_MISMATCH")
    target.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name in m["files"]:
        out = safe(target, name)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(safe(payload, name), out)
    marker = dict(
        schema="motus.piper-goal-owned-stage/1",
        run_id=uuid4().hex,
        uid=os.getuid(),
        root=str(target.resolve()),
        files=m["files"],
    )
    (target / ".goal-verifier-owned.json").write_text(
        json.dumps(marker, indent=2) + "\n"
    )
    return dict(
        state="DONE",
        stage=str(target.resolve()),
        run_id=marker["run_id"],
        services_changed=False,
    )


def rollback(target, run_id, *, remove=False):
    target = Path(target)
    need(not target.is_symlink(), "OWNED_STAGE_REQUIRED")
    marker = loads((target / ".goal-verifier-owned.json").read_text())
    need(
        marker["schema"] == "motus.piper-goal-owned-stage/1"
        and marker["uid"] == os.getuid()
        and marker["run_id"] == run_id
        and marker["root"] == str(target.resolve()),
        "OWNED_STAGE_REQUIRED",
    )
    files = {
        str(p.relative_to(target))
        for p in target.rglob("*")
        if p.is_file() or p.is_symlink()
    }
    need(
        files == set(marker["files"]) | {".goal-verifier-owned.json"},
        "PRESERVE_ADDED_OR_MISSING_FILES",
    )
    for name, hashed in marker["files"].items():
        need(sha(safe(target, name)) == hashed, "PRESERVE_MODIFIED_FILES")
    if remove:
        shutil.rmtree(target)
    return dict(
        state="DONE", removed=remove, services_changed=False, evidence_deleted=False
    )
