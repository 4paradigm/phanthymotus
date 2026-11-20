"""Build an independent ROI runtime + focused self-test source archive."""

import argparse
import hashlib
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = [
        p
        for p in (ROOT / "perception/plugins/tabletop_roi_goal_verifier").rglob("*")
        if p.is_file()
        and not {"__pycache__", ".ruff_cache", ".pytest_cache"}.intersection(p.parts)
        and p.suffix != ".pyc"
    ]
    files += [
        ROOT / p
        for p in (
            "LICENSE",
            "perception/main.py",
            "perception/plugins/camera_info.py",
            "perception/utils/logsafe.py",
            "perception/utils/__init__.py",
            "perception/tests/conftest.py",
            "perception/tests/vision_stubs.py",
            "perception/tests/test_tabletop_roi_goal_verifier.py",
            "perception/tests/test_tabletop_roi_transport.py",
            "perception/tests/test_tabletop_roi_field.py",
            "perception/tests/test_tabletop_roi_review.py",
            "perception/tools/package_tabletop_roi.py",
            "perception/tools/package_tabletop_roi_field.py",
            "perception/tools/tabletop_roi_browser.py",
            "perception/tools/tabletop_roi_ros_smoke.py",
            "agent-core/deploy/dds-local.xml",
            "agent-core/web/js/workbench.js",
        )
    ]
    archive = args.output / "tabletop-roi-goal-verifier.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for p in sorted(files):
            tar.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    (args.output / "SHA256SUMS").write_text(f"{sha}  {archive.name}\n")
    (args.output / "FILES.txt").write_text(
        "\n".join(str(p.relative_to(ROOT)) for p in sorted(files)) + "\n"
    )
    print(sha, archive)


if __name__ == "__main__":
    main()
