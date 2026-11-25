"""Package verified local CPython 3.10 wheels and ROI source for offline field use.

--wheels-dir contains pip-*.whl and wheelhouse/*.whl prepared with pip download.
Never downloads, installs dependencies or touches an existing deployment.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wheels-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    archive = output.parent / "tabletop-roi-d435-field.tar.gz"
    if output.exists() or archive.exists():
        p.error(
            "output directory or field archive already exists; choose a new location"
        )
    pip = list(args.wheels_dir.glob("pip-*.whl"))
    wheels = list((args.wheels_dir / "wheelhouse").glob("*.whl"))
    if len(pip) != 1 or not wheels:
        p.error("requires one pip bootstrap wheel and wheelhouse/*.whl")
    output.mkdir(parents=True)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "perception/tools/package_tabletop_roi.py"),
            "--output",
            str(output),
        ],
        check=True,
    )
    for file in (
        ROOT / "perception/plugins/tabletop_roi_goal_verifier/deploy/field"
    ).iterdir():
        if file.is_file():
            shutil.copy2(file, output / file.name)
    shutil.copy2(pip[0], output / pip[0].name)
    # Use the unchanged official loopback profile, in the personal deployment.
    shutil.copy2(ROOT / "agent-core/deploy/dds-local.xml", output / "dds-local.xml")
    (output / "wheelhouse").mkdir()
    for file in wheels:
        shutil.copy2(file, output / "wheelhouse" / file.name)
    files = sorted(
        x for x in output.rglob("*") if x.is_file() and x.name != "SHA256SUMS"
    )
    (output / "SHA256SUMS").write_text(
        "".join(
            hashlib.sha256(x.read_bytes()).hexdigest()
            + "  "
            + str(x.relative_to(output))
            + "\n"
            for x in files
        )
    )
    with tarfile.open(archive, "w:gz") as tar:
        for file in sorted(x for x in output.rglob("*") if x.is_file()):
            tar.add(file, arcname=str(file.relative_to(output)), recursive=False)
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output.parent / "FIELD-SHA256SUMS").write_text(sha + "  " + archive.name + "\n")
    print(
        json.dumps(
            {
                "archive": str(archive),
                "sha256": sha,
                "kit": str(output),
                "bytes": archive.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
