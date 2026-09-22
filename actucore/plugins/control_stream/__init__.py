"""What every card that emits a `motus.control/1` stream needs, and nothing else.

Two cards produce command streams now — `vla` (a policy) and `navi` (a visual
servo) — and the number only goes up: a grasp policy, a whole-body controller.
All of them build the same message and answer the same question before starting
("does what I emit fit what is wired downstream"), so both live here rather than
inside whichever card happened to need them first.

They were in `plugins/vla/` until the second consumer appeared. `plugins/vla/`
keeps two re-export shims so nothing that imported them from there has to change
— including the existing tests, which are the record of what these two are
supposed to do.

Deliberately free of ROS and of the cards themselves: `build` is a pure function
over a dict, and `negotiate` is two dicts in and a list of complaints out. That
is what lets the whole wire format be tested without a node, a canvas, or a
robot.
"""

from .message import SCHEMA, build
from .negotiate import check, effective_rate, ttl_ms

__all__ = ["SCHEMA", "build", "check", "effective_rate", "ttl_ms"]
