"""Reading `motus.odom/1` — the consumer half of a cross-repo contract.

The producer is `phanthymotus-driver/common/odom.py`, in another repository and
another image. Per the project's rule for cross-repo protocols the spec is a
document and each side implements against it, so **this file is our half of the
contract test**. The samples below are written by hand from the spec rather than
produced by the driver's builder; that is the point, because a test that used
the producer's code would pass even if both sides drifted together.

Nearly every test here is about one thing: `None` must not become `0.0`.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plugins.navi import odom as O  # noqa: E402


def _sample(twist, frame="body", schema=O.SCHEMA):
    return {"schema": schema, "stamp_ms": 1758537600123, "frame": frame,
            "units": {"linear": "m/s", "angular": "rad/s", "length": "m"},
            "twist": list(twist), "pose": None}


# ── the axis order is the contract ───────────────────────────────────────────

def test_the_axes_are_motus_control_twist_axes():
    """The reason this format was given this shape: commanded and measured line
    up index by index, so a comparison is a subtraction. If these diverge, the
    stuck detector starts comparing forward speed against yaw rate."""
    assert O.AXES == ("vx", "vy", "vz", "wx", "wy", "wz")


def test_each_axis_reads_its_own_slot():
    sample = _sample([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    for i, name in enumerate(O.AXES):
        assert O.axis_of(sample, name) == float(i + 1)


# ── null is not zero ─────────────────────────────────────────────────────────

def test_an_unmeasured_axis_reads_as_none():
    sample = _sample([0.31, 0.02, None, None, None, -0.42])
    assert O.axis_of(sample, "vz") is None
    assert O.axis_of(sample, "vx") == 0.31


def test_a_measured_zero_is_not_none():
    """The distinction the whole format exists for, from this side."""
    assert O.axis_of(_sample([0.0] * 6), "vx") == 0.0
    assert O.axis_of(_sample([None] * 6), "vx") is None


def test_it_survives_a_json_round_trip():
    """`null` arrives over the wire, not as a Python None."""
    wire = json.loads(json.dumps(_sample([None, 0.0, None, None, None, 0.1])))
    assert O.axis_of(wire, "vx") is None
    assert O.axis_of(wire, "vy") == 0.0


# ── everything unusable is None, never a number ──────────────────────────────

def test_a_world_frame_sample_is_not_read_as_body():
    """World-frame odometry is real, but comparing it against a body-frame
    command needs the robot's heading and a rotation. Treating it as body gives
    a wrong answer whenever the robot is not facing along world x — silently."""
    assert O.axis_of(_sample([1.0] * 6, frame="world"), "vx") is None


def test_a_foreign_schema_is_not_read():
    assert O.axis_of(_sample([1.0] * 6, schema="motus.control/1"), "vx") is None


def test_a_short_or_absent_twist_is_none():
    assert O.axis_of({"schema": O.SCHEMA, "frame": "body", "twist": [1.0]}, "wz") is None
    assert O.axis_of({"schema": O.SCHEMA, "frame": "body"}, "vx") is None


def test_junk_is_none_rather_than_an_exception():
    """One malformed frame must not take the stream down."""
    for junk in (None, [], "", {"schema": O.SCHEMA}, 42):
        assert O.axis_of(junk, "vx") is None


def test_a_boolean_is_not_accepted_as_a_speed():
    """`True` is 1.0 to Python's float(), which would read as a robot moving at
    one metre per second."""
    assert O.axis_of(_sample([True, None, None, None, None, None]), "vx") is None


def test_an_unknown_axis_name_is_none():
    assert O.axis_of(_sample([1.0] * 6), "yaw_speed") is None


# ── the declaration ──────────────────────────────────────────────────────────

def test_provides_is_read_from_the_interface():
    info = {"odom_interface": {"schema": O.SCHEMA, "frame": "body",
                               "provides": ["vx", "vy", "wz"], "rate_hz": 10}}
    assert O.interface_provides(info, "vx") is True
    assert O.interface_provides(info, "vz") is False


def test_a_missing_interface_provides_nothing():
    """Which is what lets the card say it has no stuck protection, rather than
    appearing to have one."""
    assert O.interface_provides({}, "vx") is False
    assert O.interface_provides(None, "vx") is False
    assert O.interface_provides({"odom_interface": {"provides": ["vx"]}}, "vx") is False
