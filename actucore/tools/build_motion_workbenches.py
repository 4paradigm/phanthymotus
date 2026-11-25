#!/usr/bin/env python3
"""Bundle shared workbench sources into the two offline iframe artifacts.

The Motion repository owns the source. Pass --driver-output explicitly when
releasing the Driver artifact; neither runtime needs the other repository.
"""
import argparse
import hashlib
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "plugins/motion/ui"
INCLUDE = re.compile(r"/\* @include ([a-z0-9.-]+) \*/|<!-- @include ([a-z0-9.-]+) -->")


def bundle(name):
    def expand(match):
        return INCLUDE.sub(expand, (SOURCE / (match[1] or match[2])).read_text())
    result = INCLUDE.sub(expand, (SOURCE / name).read_text())
    digest = hashlib.sha256((SOURCE / "simulation.js").read_bytes()).hexdigest()
    result = result.replace("<!doctype html>", f"<!doctype html><!-- shared-simulation-sha256:{digest} -->", 1)
    if len(result) > 250_000:
        raise ValueError(f"{name} exceeds the host's 250,000 character limit")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver-output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = [(ROOT / "plugins/motion/workbench.html", "motion.html")]
    if args.driver_output:
        outputs.append((args.driver_output, "simulation.html"))
    for path, source in outputs:
        text = bundle(source)
        if args.check:
            if path.read_text() != text:
                raise SystemExit(f"stale generated workbench: {path}")
        else:
            path.write_text(text)
        print(f"{path.name}: {len(text)} characters; {'verified' if args.check else 'built'}")


if __name__ == "__main__":
    main()
