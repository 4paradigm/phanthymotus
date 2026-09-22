"""Moved to `plugins/control_stream/message.py` when a second card needed it.

Kept as a re-export rather than updated in place: the existing tests import it
from here, and they are the record of what this module is supposed to do. A
move that also rewrites its tests is a move with nothing checking it.
"""

from ..control_stream.message import SCHEMA, build

__all__ = ["SCHEMA", "build"]
