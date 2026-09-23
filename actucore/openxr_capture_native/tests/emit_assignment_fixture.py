"""Production Python assignments for the native parser; no network or robot I/O.

Use an ActuCore test Python with its normal scipy/aiortc dependencies. The
runtime descriptor is inert; CaptureManager builds the exact outbound envelope.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plugins.teleop.capture import CaptureManager
from plugins.teleop.descriptor import CAPABILITIES, capability_digest
from plugins.teleop.g1 import CAPABILITIES_G1
from plugins.teleop.motion_control import CAPABILITIES_EEF


def main():
    for capabilities in (CAPABILITIES, CAPABILITIES_G1, CAPABILITIES_EEF):
        for mode in ("shadow", "live"):
            runtime = SimpleNamespace(
                capabilities=capabilities,
                profile_id=capabilities["profile_id"],
                capability_digest=capability_digest(mode, capabilities),
                mode=mode,
            )
            manager = CaptureManager(runtime, None, None, wall_clock=lambda: 1000.0)
            assignment = manager._new_assignment({
                "session_id": "7ad3de66-64f2-4d47-89a2-c8da2940eb97",
            })
            print(json.dumps({"type": "assignment", "assignment": assignment}))


if __name__ == "__main__":
    main()
