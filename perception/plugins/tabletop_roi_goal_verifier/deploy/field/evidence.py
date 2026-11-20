#!/usr/bin/env python3
"""Save a browser-exported session JSON under this personal deployment's evidence/."""

import argparse
import json
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src/perception"))
    sys.path.insert(0, str(root / "deps"))
    from plugins.tabletop_roi_goal_verifier.evidence import save_session

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument(
        "--expected", choices=["FAIL", "PASS", "INCONCLUSIVE"], required=True
    )
    p.add_argument("--notes", default="")
    args = p.parse_args()
    print(
        json.dumps(
            save_session(
                json.loads(args.input.read_text()),
                root / "evidence",
                args.expected,
                args.notes,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
